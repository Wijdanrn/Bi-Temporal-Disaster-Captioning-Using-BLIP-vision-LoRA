"""
Fase 1 -- EN->ID translation of the DisasterM3 disaster-captioning subset.

Ground truth is a fixed 7-section structured report (see docs/dataset_audit.md
S4), not a short sentence. Per project decision, each section is translated
independently (not the whole blob at once) and reassembled with Indonesian
section headers, to keep NLLB's input short/clean per call and preserve the
report structure exactly.

Usage:
    python scripts/translate_captions.py --split train --limit 10 --out data/interim/test_run.jsonl
    python scripts/translate_captions.py --split train --out data/interim/train_id.jsonl
    python scripts/translate_captions.py --split test  --out data/interim/test_id.jsonl
"""
import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPLIT_CONFIG = {
    "train": dict(path=ROOT / "DisasterM3_Instruct" / "train_release.json", task="disaster caption"),
    "test": dict(path=ROOT / "DisasterM3_Bench" / "benchmark_release.json", task="Disaster Report"),
}

# Order matters: matches the fixed section order observed in ground_truth text.
SECTION_HEADERS_EN = ["DISASTER", "BUILDING", "ROAD", "VEGETATION", "WATER_BODY", "AGRICULTURE", "CONCLUSION"]
SECTION_HEADER_ID = {
    "DISASTER": "BENCANA",
    "BUILDING": "BANGUNAN",
    "ROAD": "JALAN",
    "VEGETATION": "VEGETASI",
    "WATER_BODY": "BADAN_AIR",
    "AGRICULTURE": "PERTANIAN",
    "CONCLUSION": "KESIMPULAN",
}

SECTION_SPLIT_RE = re.compile(r"(?:^|\n)(" + "|".join(SECTION_HEADERS_EN) + r"): ")

# Known NLLB false-friend / domain mistranslations observed in manual QC of a
# 10-record test batch (see docs/dataset_audit.md translation notes). Applied
# as a conservative whole-word regex replace post-translation. This domain is
# narrow/technical (structured damage reports), so these are low false-positive-risk.
TERMINOLOGY_FIXES = [
    (re.compile(r"\byayasan\b", re.IGNORECASE), "fondasi"),  # "foundation" (building) mistranslated as "foundation" (nonprofit org)
]


def apply_terminology_fixes(text: str) -> str:
    for pattern, replacement in TERMINOLOGY_FIXES:
        text = pattern.sub(replacement, text)
    return text


def parse_sections(text: str) -> "list[tuple[str, str]]":
    """Split a ground_truth blob into ordered (SECTION, text) pairs.

    Returns [] if the text doesn't match the expected fixed-header format
    (caller should treat that record as unparseable and skip/flag it,
    rather than silently mistranslating garbage).
    """
    matches = list(SECTION_SPLIT_RE.finditer(text))
    if not matches:
        return []
    sections = []
    for i, m in enumerate(matches):
        header = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((header, text[start:end].strip()))
    return sections


def load_records(split: str, limit: int | None = None):
    cfg = SPLIT_CONFIG[split]
    with open(cfg["path"], encoding="utf-8") as f:
        data = json.load(f)
    records = [r for r in data if r.get("task") == cfg["task"]]
    if limit:
        records = records[:limit]
    return records


class NllbTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device=None, batch_size=128, fp16=True):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"[translator] loading {model_name} on {self.device} (fp16={fp16}) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        self.tgt_lang_id = self.tokenizer.convert_tokens_to_ids("ind_Latn")
        print("[translator] ready", flush=True)

    def translate_batch(self, texts: "list[str]", max_new_tokens=200) -> "list[str]":
        if not texts:
            return []
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.device
        )
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [apply_terminology_fixes(t) for t in decoded]

    def translate_many(self, texts: "list[str]") -> "list[str]":
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self.translate_batch(batch))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="only process first N records (for quick testing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    args = ap.parse_args()

    records = load_records(args.split, args.limit)
    print(f"[main] {len(records)} '{SPLIT_CONFIG[args.split]['task']}' records loaded from split={args.split}", flush=True)

    parsed = []
    unparseable = 0
    for r in records:
        sections = parse_sections(r["ground_truth"])
        if not sections:
            unparseable += 1
            continue
        parsed.append({"record": r, "sections": sections})
    if unparseable:
        print(f"[main] WARNING: {unparseable}/{len(records)} records did not match the expected section format and were skipped", flush=True)

    translator = NllbTranslator(model_name=args.model, batch_size=args.batch_size)

    # Flatten every section text across all records into one big list so the
    # translator can batch efficiently, then regroup by record afterward.
    flat_texts = []
    owner_idx = []  # (record_index_in_parsed, section_index)
    for ri, p in enumerate(parsed):
        for si, (header, text) in enumerate(p["sections"]):
            flat_texts.append(text)
            owner_idx.append((ri, si))

    # Translate the fixed prompt once (it repeats verbatim across records).
    unique_prompts = sorted({r["record"]["prompts"][0] if isinstance(r["record"]["prompts"], list) else r["record"]["prompts"] for r in parsed} if parsed else [])
    prompt_translations = dict(zip(unique_prompts, translator.translate_many(unique_prompts))) if unique_prompts else {}

    print(f"[main] translating {len(flat_texts)} section-texts across {len(parsed)} records (batch_size={args.batch_size}) ...", flush=True)

    # Length-bucket before batching: a batch mixing a short and a very long
    # text pads every item (and beam-search memory) up to the longest one,
    # which caused near-VRAM-exhaustion thrashing (100% GPU util, ~20W power
    # draw, throughput collapsing from ~13/s to ~1/s) on real, variable-length
    # section text. Sorting by length keeps each batch's padding minimal and
    # memory usage predictable; original order is restored via `order` below.
    order = sorted(range(len(flat_texts)), key=lambda idx: len(flat_texts[idx]))
    sorted_texts = [flat_texts[idx] for idx in order]

    t0 = time.time()
    translated_sorted = []
    bs = args.batch_size
    for i in range(0, len(sorted_texts), bs):
        batch = sorted_texts[i : i + bs]
        translated_sorted.extend(translator.translate_batch(batch))
        if (i // bs) % 20 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(sorted_texts) - done) / rate if rate > 0 else float("inf")
            print(f"  {done}/{len(sorted_texts)} sections  ({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)

    translated_flat = [None] * len(flat_texts)
    for idx, translated in zip(order, translated_sorted):
        translated_flat[idx] = translated

    # Regroup translated sections back into per-record structured Indonesian text.
    id_sections_by_record = [[None] * len(p["sections"]) for p in parsed]
    for (ri, si), translated in zip(owner_idx, translated_flat):
        id_sections_by_record[ri][si] = translated

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p, id_sections in zip(parsed, id_sections_by_record):
            r = p["record"]
            headers_en = [h for h, _ in p["sections"]]
            texts_en = [t for _, t in p["sections"]]
            ground_truth_id = "\n".join(
                f"{SECTION_HEADER_ID[h]}: {t_id}" for h, t_id in zip(headers_en, id_sections)
            )
            prompt_en = r["prompts"][0] if isinstance(r["prompts"], list) else r["prompts"]
            row = {
                "pre_image_path": r["pre_image_path"],
                "post_image_path": r["post_image_path"],
                "post_image_type": r.get("post_image_type"),
                "prompt_en": prompt_en,
                "prompt_id": prompt_translations.get(prompt_en, ""),
                "ground_truth_en": r["ground_truth"],
                "ground_truth_id": ground_truth_id,
                "sections_en": dict(zip(headers_en, texts_en)),
                "sections_id": dict(zip(headers_en, id_sections)),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"[main] DONE: wrote {len(parsed)} records to {out_path} in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
