"""
Urgent, submission-deadline task: friend trained Qwen2.5-VL-3B and evaluated it on a
1182-item subset of the test set (test_filenames_first_1182.csv, a "first-N" slice --
NOT random/stratified, covers only 15/30 disaster types per their own observation) because
full-set eval was going to take ~20h. To compare our BLIP model against theirs FAIRLY, we
must score our model on the EXACT SAME 1182 items -- no GPU regeneration needed, our
predictions for the full test set already exist in results/predictions_test_vision.jsonl.

This script:
  1. Loads the friend's csv (pre_image, post_image, post_type -- basenames only).
  2. Matches those basenames against our own data/processed/captions_test.jsonl (by
     basename, since our jsonl stores full relative paths like "test_images\\X.png").
  3. Filters our already-generated predictions to just the matched subset.
  4. Reports match rate, disaster-type coverage, and writes the matched-subset predictions
     + reference file so compute_metrics-style scoring can run on exactly this subset.

Usage:
    python scripts/match_shared_test_subset.py
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRIEND_CSV = os.path.join(ROOT, "test_filenames_first_1182.csv")
OUR_TEST_JSONL = os.path.join(ROOT, "data", "processed", "captions_test.jsonl")
OUR_PRED_VISION = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_MATCHED_INDICES = os.path.join(ROOT, "results", "shared_subset_1182_matched_indices.json")


def norm_basename(p: str) -> str:
    return os.path.basename(p.replace("\\", "/"))


def main():
    with open(FRIEND_CSV, encoding="utf-8") as f:
        friend_rows = list(csv.DictReader(f))
    print(f"[friend csv] {len(friend_rows)} rows")

    wanted = []
    seen = set()
    for r in friend_rows:
        k = (r["pre_image"], r["post_image"])
        if k not in seen:
            seen.add(k)
            wanted.append(k)
    print(f"[friend csv] {len(wanted)} unique (pre,post) basename pairs")

    with open(OUR_TEST_JSONL, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]
    print(f"[our test set] {len(records)} records")

    ours_by_basename = {}
    for i, r in enumerate(records):
        k = (norm_basename(r["pre_image_path"]), norm_basename(r["post_image_path"]))
        ours_by_basename[k] = i

    matched_idx = []
    missing = []
    for k in wanted:
        if k in ours_by_basename:
            matched_idx.append(ours_by_basename[k])
        else:
            missing.append(k)

    print(f"[match] {len(matched_idx)} / {len(wanted)} matched against our test set")
    if missing:
        print(f"[match] {len(missing)} NOT found in our test set, e.g.: {missing[:5]}")

    # disaster-type coverage check (via ground_truth_id's BENCANA line), matching the
    # friend's own observation that this slice only covers ~15/30 disaster types
    import re
    def get_bencana(text):
        m = re.search(r"BENCANA:\s*(.+)", text)
        return m.group(1).strip().lower() if m else None

    types_in_subset = set()
    for i in matched_idx:
        t = get_bencana(records[i]["ground_truth_id"])
        if t:
            types_in_subset.add(t)
    print(f"[coverage] disaster types present in matched subset: {len(types_in_subset)} -> {sorted(types_in_subset)}")

    # also check full test set's disaster-type coverage for contrast
    types_full = set()
    for r in records:
        t = get_bencana(r["ground_truth_id"])
        if t:
            types_full.add(t)
    print(f"[coverage] disaster types present in FULL test set: {len(types_full)} -> {sorted(types_full)}")
    print(f"[coverage] types MISSING from the shared subset: {sorted(types_full - types_in_subset)}")

    # cross-check against our already-generated vision-LoRA predictions
    with open(OUR_PRED_VISION, encoding="utf-8") as f:
        pred_rows = [json.loads(l) for l in f]
    pred_by_idx = {r["index"]: r for r in pred_rows}
    have_pred = [i for i in matched_idx if i in pred_by_idx]
    print(f"[predictions] {len(have_pred)} / {len(matched_idx)} matched indices already have a "
          f"generated prediction in {os.path.basename(OUR_PRED_VISION)} (no GPU regen needed)")

    with open(OUT_MATCHED_INDICES, "w", encoding="utf-8") as f:
        json.dump({
            "n_friend_csv_rows": len(friend_rows),
            "n_friend_unique_pairs": len(wanted),
            "n_matched": len(matched_idx),
            "n_missing": len(missing),
            "missing_pairs": missing,
            "matched_indices": matched_idx,
            "disaster_types_in_subset": sorted(types_in_subset),
            "disaster_types_full_test_set": sorted(types_full),
            "disaster_types_missing_from_subset": sorted(types_full - types_in_subset),
            "n_matched_with_existing_predictions": len(have_pred),
        }, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_MATCHED_INDICES}")


if __name__ == "__main__":
    main()
