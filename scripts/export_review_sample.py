"""
Fase 1 -- export a sample of translated captions to CSV for manual
native-speaker review.

Stratified sample: half random (typical quality), half the worst-scoring
records by QC min_similarity/length_ratio (worst-case quality) -- a
reviewer who only sees random samples won't encounter the tail cases that
matter most for deciding whether the dataset is usable as-is.

Usage:
    python scripts/export_review_sample.py --split train --n 40 --out data/interim/train_review_sample.csv
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    id_path = ROOT / "data" / "interim" / f"{args.split}_id.jsonl"
    qc_path = ROOT / "data" / "interim" / f"{args.split}_qc.jsonl"

    records = [json.loads(l) for l in open(id_path, encoding="utf-8")]
    qc_records = [json.loads(l) for l in open(qc_path, encoding="utf-8")] if qc_path.exists() else None

    n_worst = args.n // 2
    n_random = args.n - n_worst

    rng = random.Random(args.seed)
    all_idx = list(range(len(records)))

    worst_idx = []
    if qc_records:
        scored = sorted(range(len(qc_records)), key=lambda i: qc_records[i]["min_similarity"])
        worst_idx = scored[:n_worst]

    remaining = [i for i in all_idx if i not in set(worst_idx)]
    rng.shuffle(remaining)
    random_idx = remaining[:n_random]

    chosen = [(i, "worst_case") for i in worst_idx] + [(i, "random") for i in random_idx]
    rng.shuffle(chosen)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_type", "pre_image_path", "post_image_path",
            "ground_truth_en", "ground_truth_id",
            "min_similarity", "avg_similarity", "suspect_sections",
            "reviewer_ok_Y_N", "reviewer_notes",
        ])
        for i, sample_type in chosen:
            r = records[i]
            qc = qc_records[i] if qc_records else {}
            writer.writerow([
                sample_type,
                r["pre_image_path"],
                r["post_image_path"],
                r["ground_truth_en"],
                r["ground_truth_id"],
                qc.get("min_similarity", ""),
                qc.get("avg_similarity", ""),
                ", ".join(qc.get("suspect_sections", [])),
                "",
                "",
            ])

    print(f"[export] wrote {len(chosen)} rows ({n_worst} worst_case + {n_random} random) to {out_path}")


if __name__ == "__main__":
    main()
