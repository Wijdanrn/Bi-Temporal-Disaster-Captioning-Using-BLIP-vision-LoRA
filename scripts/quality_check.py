"""
Fase 1 -- back-translation quality check for the EN->ID captioning translations.

For every section, back-translate ID->EN and compare against the original EN
text with two signals:
  - similarity: difflib.SequenceMatcher ratio (0-1), catches wording drift
  - length_ratio: len(backtranslated) / len(original), catches DROPPED
    CONTENT specifically (the failure mode found in manual QC of the first
    10-record test batch -- a trailing sentence silently missing translates
    to a back-translation that's suspiciously short vs. the original)

A record is flagged "needs_review" if any section's length_ratio < 0.7 or
similarity < 0.45 -- thresholds are heuristic, meant to catch the worst
cases for human review, not a hard pass/fail gate.

Usage:
    python scripts/quality_check.py --in data/interim/train_id.jsonl --out data/interim/train_qc.jsonl
    python scripts/quality_check.py --in data/interim/test_id.jsonl  --out data/interim/test_qc.jsonl
"""
import argparse
import difflib
import json
import time
from pathlib import Path

from translate_captions import NllbTranslator, SECTION_HEADERS_EN

LENGTH_RATIO_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.45


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def length_ratio(original: str, back: str) -> float:
    n_orig = len(original.split())
    n_back = len(back.split())
    if n_orig == 0:
        return 1.0
    return n_back / n_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    records = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[qc] {len(records)} records loaded from {args.inp}", flush=True)

    flat_id_texts = []
    owner_idx = []  # (record_index, section_header)
    for ri, r in enumerate(records):
        for header in SECTION_HEADERS_EN:
            id_text = r["sections_id"].get(header, "")
            if id_text:
                flat_id_texts.append(id_text)
                owner_idx.append((ri, header))

    translator = NllbTranslator(batch_size=args.batch_size)
    translator.tokenizer.src_lang = "ind_Latn"
    translator.tgt_lang_id = translator.tokenizer.convert_tokens_to_ids("eng_Latn")

    order = sorted(range(len(flat_id_texts)), key=lambda idx: len(flat_id_texts[idx]))
    sorted_texts = [flat_id_texts[idx] for idx in order]

    print(f"[qc] back-translating {len(sorted_texts)} section-texts ...", flush=True)
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
            print(f"  {done}/{len(sorted_texts)}  ({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)

    back_translated_flat = [None] * len(flat_id_texts)
    for idx, translated in zip(order, translated_sorted):
        back_translated_flat[idx] = translated

    per_record_sections = [dict() for _ in records]
    for (ri, header), back_en in zip(owner_idx, back_translated_flat):
        orig_en = records[ri]["sections_en"].get(header, "")
        sim = similarity(orig_en, back_en)
        lr = length_ratio(orig_en, back_en)
        per_record_sections[ri][header] = {
            "back_translated_en": back_en,
            "similarity": round(sim, 3),
            "length_ratio": round(lr, 3),
            "suspect": bool(sim < SIMILARITY_THRESHOLD or lr < LENGTH_RATIO_THRESHOLD),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_needs_review = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r, sections_qc in zip(records, per_record_sections):
            sims = [s["similarity"] for s in sections_qc.values()]
            needs_review = any(s["suspect"] for s in sections_qc.values())
            n_needs_review += needs_review
            row = {
                "pre_image_path": r["pre_image_path"],
                "post_image_path": r["post_image_path"],
                "avg_similarity": round(sum(sims) / len(sims), 3) if sims else None,
                "min_similarity": round(min(sims), 3) if sims else None,
                "needs_review": needs_review,
                "suspect_sections": [h for h, s in sections_qc.items() if s["suspect"]],
                "sections_qc": sections_qc,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(
        f"[qc] DONE: {len(records)} records checked in {elapsed/60:.1f} min. "
        f"{n_needs_review}/{len(records)} ({100*n_needs_review/len(records):.1f}%) flagged needs_review. "
        f"Wrote {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
