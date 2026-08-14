"""
Ad-hoc investigation (not part of the main pipeline): does the Building/Road Damage
Counting task in DisasterM3 share pre/post image pairs with the disaster-caption task
we actually trained on? If so, count info could in principle be joined into our report
ground truth. Real data only, no assumptions.

Usage:
    python scripts/check_counting_overlap.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(p):
    return p.replace("\\", "/") if p else p


def pair_key(r):
    return (norm(r.get("pre_image_path")), norm(r.get("post_image_path")))


def main():
    train = load(os.path.join(ROOT, "DisasterM3_Instruct", "train_release.json"))
    bench = load(os.path.join(ROOT, "DisasterM3_Bench", "benchmark_release.json"))
    print("train records:", len(train), "bench records:", len(bench))
    print()

    for name, data, cap_task in [("TRAIN", train, "disaster caption"), ("BENCH", bench, "Disaster Report")]:
        print(f"=== {name} ===")
        tasks: dict[str, list] = {}
        for r in data:
            tasks.setdefault(r["task"], []).append(r)
        cap_recs = tasks.get(cap_task, [])
        cap_pairs = set(pair_key(r) for r in cap_recs)
        print(f"{cap_task}: {len(cap_recs)} records, {len(cap_pairs)} unique pairs")

        for ct in ["Building Damage Counting", "Road Damage Counting"]:
            recs = tasks.get(ct, [])
            cpairs = set(pair_key(r) for r in recs)
            overlap = cap_pairs & cpairs
            pct = len(overlap) / max(len(cap_pairs), 1) * 100
            print(f"{ct}: {len(recs)} records, {len(cpairs)} unique pairs, "
                  f"overlap with captioning pairs: {len(overlap)} ({pct:.1f}% of caption pairs)")
            prompts = sorted(set(r["prompts"] for r in recs))
            print(f"  distinct prompt phrasings ({len(prompts)}):")
            for p in prompts[:10]:
                print("   -", p)

            # how many counting records exist PER overlapping pair (could be >1, e.g.
            # "damaged" vs "undamaged" vs "total" phrasing for the same image pair)
            per_pair: dict = {}
            for r in recs:
                k = pair_key(r)
                if k in cap_pairs:
                    per_pair.setdefault(k, []).append(r)
            if per_pair:
                counts_per_pair = [len(v) for v in per_pair.values()]
                print(f"  records per overlapping pair: min={min(counts_per_pair)}, "
                      f"max={max(counts_per_pair)}, "
                      f"mean={sum(counts_per_pair)/len(counts_per_pair):.2f}")
                # show one real worked example
                example_key = next(iter(per_pair))
                example_cap = next(r for r in cap_recs if pair_key(r) == example_key)
                print(f"  --- worked example ({example_key[0]}) ---")
                print("  caption ground_truth (first 300 chars):",
                      example_cap["ground_truth"][:300].replace("\n", " | "))
                for r in per_pair[example_key]:
                    print(f"  [{ct}] prompt={r['prompts']!r} -> ground_truth={r['ground_truth']!r} "
                          f"(training_answer={r.get('training_answer')!r})")
        print()


if __name__ == "__main__":
    main()
