"""
Targeted fix for the CONCLUSION section specifically: back-translation QC
found it flagged in 34.3% of train records (mean length_ratio 0.82, vs
~1.0 for other sections) -- NLLB-200-distilled-600M's beam search
occasionally drops a trailing sentence on multi-sentence inputs, confirmed
NOT a max_new_tokens truncation issue (max generated length was 74 tokens
against a 200 cap). Splitting the section into individual sentences before
translating -- one level deeper than the existing 7-section split -- avoids
handing the model a multi-sentence input in the first place.

Only CONCLUSION is re-translated; other sections (5-15% flagged, mean
ratio ~1.0) are left as-is since that's consistent with normal
paraphrase-driven length variation, not systematic content loss.

Usage:
    python scripts/fix_conclusion.py --split train
    python scripts/fix_conclusion.py --split test
"""
import argparse
import json
import re
import time
from pathlib import Path

from translate_captions import (
    NllbTranslator,
    SECTION_HEADER_ID,
    SECTION_HEADERS_EN,
    apply_terminology_fixes,
)

ROOT = Path(__file__).resolve().parent.parent

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> "list[str]":
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts or [text]


def rebuild_ground_truth_id(sections_en_order: "list[str]", sections_id: dict) -> str:
    return "\n".join(f"{SECTION_HEADER_ID[h]}: {sections_id[h]}" for h in sections_en_order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    path = ROOT / "data" / "interim" / f"{args.split}_id.jsonl"
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    print(f"[fix] {len(records)} records loaded from {path}", flush=True)

    # Flatten every CONCLUSION sentence across all records for batched translation.
    flat_sentences = []
    owner_idx = []  # (record_index, sentence_index_within_record)
    sentence_counts = []
    for ri, r in enumerate(records):
        en_conclusion = r["sections_en"].get("CONCLUSION", "")
        sentences = split_sentences(en_conclusion)
        sentence_counts.append(len(sentences))
        for si, sent in enumerate(sentences):
            flat_sentences.append(sent)
            owner_idx.append((ri, si))

    print(f"[fix] {len(flat_sentences)} CONCLUSION sentences to translate "
          f"(was {len(records)} whole-section translations before)", flush=True)

    translator = NllbTranslator(batch_size=args.batch_size)

    order = sorted(range(len(flat_sentences)), key=lambda idx: len(flat_sentences[idx]))
    sorted_texts = [flat_sentences[idx] for idx in order]

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

    translated_flat = [None] * len(flat_sentences)
    for idx, translated in zip(order, translated_sorted):
        translated_flat[idx] = translated

    # Regroup sentence translations back into records, joined into one CONCLUSION block.
    sentences_by_record = [[None] * n for n in sentence_counts]
    for (ri, si), translated in zip(owner_idx, translated_flat):
        sentences_by_record[ri][si] = translated

    n_changed = 0
    for ri, r in enumerate(records):
        new_conclusion = apply_terminology_fixes(" ".join(sentences_by_record[ri]))
        if new_conclusion != r["sections_id"].get("CONCLUSION", ""):
            n_changed += 1
        r["sections_id"]["CONCLUSION"] = new_conclusion
        sections_en_order = [h for h in SECTION_HEADERS_EN if h in r["sections_en"]]
        r["ground_truth_id"] = rebuild_ground_truth_id(sections_en_order, r["sections_id"])

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"[fix] DONE: re-translated CONCLUSION for {len(records)} records "
          f"({n_changed} changed) in {elapsed/60:.1f} min. Overwrote {path}", flush=True)


if __name__ == "__main__":
    main()
