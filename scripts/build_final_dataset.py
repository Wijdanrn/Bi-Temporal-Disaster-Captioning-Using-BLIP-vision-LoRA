"""
Fase 1 -- final step: carve train/val out of the translated train split
(test split is already a clean held-out set per Fase 0 audit, 0 pre_image
overlap with train) and write the final processed dataset.

Split is grouped by pre_image_path, not by individual record: several
records share the same pre_image_path with different post_image_type
(Optical vs SAR) and IDENTICAL ground_truth text (confirmed in manual
inspection during Fase 1). A naive per-record random split would leak
near-duplicate examples across train/val. Fixed seed for reproducibility.

Usage:
    python scripts/build_final_dataset.py --val-fraction 0.1 --seed 42
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_records = [json.loads(l) for l in open(ROOT / "data/interim/train_id.jsonl", encoding="utf-8")]
    test_records = [json.loads(l) for l in open(ROOT / "data/interim/test_id.jsonl", encoding="utf-8")]

    groups = defaultdict(list)
    for r in train_records:
        groups[r["pre_image_path"]].append(r)

    group_keys = sorted(groups.keys())  # sorted for determinism before shuffling
    rng = random.Random(args.seed)
    rng.shuffle(group_keys)

    n_val_groups = int(len(group_keys) * args.val_fraction)
    val_keys = set(group_keys[:n_val_groups])
    train_keys = set(group_keys[n_val_groups:])

    final_train = [r for k in group_keys if k in train_keys for r in groups[k]]
    final_val = [r for k in group_keys if k in val_keys for r in groups[k]]

    assert not (train_keys & val_keys), "train/val pre_image_path leakage"

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                row = {
                    "pre_image_path": r["pre_image_path"],
                    "post_image_path": r["post_image_path"],
                    "post_image_type": r["post_image_type"],
                    "prompt_id": r["prompt_id"],
                    "ground_truth_id": r["ground_truth_id"],
                    "prompt_en": r["prompt_en"],
                    "ground_truth_en": r["ground_truth_en"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(final_train, out_dir / "captions_train.jsonl")
    write_jsonl(final_val, out_dir / "captions_val.jsonl")
    write_jsonl(test_records, out_dir / "captions_test.jsonl")

    stats = {
        "seed": args.seed,
        "val_fraction_of_groups": args.val_fraction,
        "n_train_records": len(final_train),
        "n_val_records": len(final_val),
        "n_test_records": len(test_records),
        "n_train_groups": len(train_keys),
        "n_val_groups": len(val_keys),
    }
    with open(out_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"[build] wrote {out_dir / 'captions_train.jsonl'}, captions_val.jsonl, captions_test.jsonl")


if __name__ == "__main__":
    main()
