"""
Search the FULL test set (2362 records, results/predictions_test_vision.jsonl) for a genuine
category-(b)-"Akurat" example -- ground truth BANGUNAN describes REAL damage AND the model's
prediction ALSO acknowledges damage (not the boilerplate "tidak ada kerusakan" phrase).

Why this script exists: the 12-example xAI curated set (results/xai_examples/manifest.json)
has ZERO such examples (see make_qualitative_figure_rise.py's b_candidates == [] finding). This
script asks a narrower, cheaper question: does ANY such example exist anywhere in the full test
set? No model/GPU/xAI computation needed -- this is pure text filtering over already-generated
predictions, so it can run over all 2362 records in seconds.

METHODOLOGY CAVEAT (read before trusting the output blindly)
--------------------------------------------------------------
`gen_no_damage` reuses `is_no_damage()` from make_qualitative_figure_rise.py, which its own
docstring says is reliable ONLY for `generated_caption` (confirmed boilerplate: the model
always says one of two fixed phrases when it means "no damage"). That part is trustworthy.

`gt_looks_like_damage` is a NEW heuristic (broader no-damage phrase list than the 12-example
manual table, since hand-verifying 2362 free-form ground-truth texts is not feasible). It is
NOT as trustworthy as GT_NO_DAMAGE_MANUAL -- it WILL have false positives (GT phrased in a way
that doesn't match any no-damage phrase but still means no damage) and false negatives (GT uses
an unlisted no-damage paraphrase). It is used ONLY to narrow 2362 records down to a short
candidate list, which is then meant to be READ, not trusted as-is. This script prints full GT +
generated BANGUNAN text for every candidate so each one can be manually verified before being
cited anywhere.

Usage:
    python scripts/find_accurate_damage_examples.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_model import SECTION_HEADERS  # noqa: E402
from scripts.make_qualitative_figure_rise import get_section_bounded, is_no_damage  # noqa: E402

PRED_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")

# Broader than GT_NO_DAMAGE_MANUAL's exact-phrase table (that one was hand-verified for only 12
# examples) -- a best-effort net of common ways this dataset's GT phrases "no damage", covering
# the paraphrases actually seen across the 12-example manual table plus close variants.
GT_NO_DAMAGE_PATTERNS = [
    r"tidak ada (bukti |tanda(-tanda)? )?kerusakan",
    r"tidak (ada |menunjukkan )?(kerusakan|gangguan|dampak)",
    r"tetap utuh",
    r"tampak utuh",
    r"tidak terpengaruh",
    r"tanpa (tanda|kerusakan|dampak|gangguan)",
    r"tidak mengungkapkan kerusakan",
    r"tidak berubah",
]
GT_NO_DAMAGE_RE = re.compile("|".join(GT_NO_DAMAGE_PATTERNS))

# A second false-positive family found on the first pass: "tidak ada bangunan/struktur yang
# terlihat" (no BUILDING present at all) is a completely different case from "no DAMAGE" -- it
# slips past both the no-damage filter above AND is_no_damage() (neither mentions "kerusakan"),
# so it must be excluded explicitly on both the GT and the generated side.
NO_BUILDING_RE = re.compile(
    r"tidak ada (bangunan|struktur)|tidak ada .*(bangunan|struktur) yang (ada|terlihat|hadir)|"
    r"tidak ada .*(bangunan|struktur) yang dibangun terlihat")

# A third false-positive family: generated-text paraphrases of "no damage" that is_no_damage()'s
# narrow two-phrase check misses (e.g. "tampak secara struktural utuh tanpa kerusakan yang
# terlihat" -- found on inspection of idx 2251). Applying the SAME broadened GT regex to the
# generated text as well catches these.
def gt_looks_like_damage(gt_bangunan: str) -> bool:
    t = (gt_bangunan or "").lower()
    return not GT_NO_DAMAGE_RE.search(t) and not NO_BUILDING_RE.search(t)


def gen_looks_like_damage(gen_bangunan: str) -> bool:
    t = (gen_bangunan or "").lower()
    return (not is_no_damage(t) and not GT_NO_DAMAGE_RE.search(t)
            and not NO_BUILDING_RE.search(t))


def main():
    with open(PRED_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"[load] {len(records)} records from {PRED_PATH}")

    candidates = []
    for r in records:
        gt_b = get_section_bounded(r["ground_truth_id"], "BANGUNAN") or ""
        gen_b = get_section_bounded(r["generated_caption"], "BANGUNAN") or ""
        if not gt_b or not gen_b:
            continue
        if gt_looks_like_damage(gt_b) and gen_looks_like_damage(gen_b):
            candidates.append({
                "index": r["index"],
                "gt_bencana": get_section_bounded(r["ground_truth_id"], "BENCANA"),
                "gen_bencana": get_section_bounded(r["generated_caption"], "BENCANA"),
                "gt_bangunan": gt_b,
                "gen_bangunan": gen_b,
                "pre_image_path": r["pre_image_path"],
                "post_image_path": r["post_image_path"],
            })

    print(f"\n[filter] {len(candidates)}/{len(records)} candidates where GT BANGUNAN does NOT "
          f"match any no-damage pattern AND generated BANGUNAN does NOT match the model's "
          f"boilerplate no-damage phrase.")
    print("=" * 100)
    print("EVERY candidate below must be READ, not trusted -- the GT filter is a heuristic, "
          "not a verified label (see module docstring).")
    print("=" * 100)
    for c in candidates:
        print(f"\n[idx {c['index']:>4}] BENCANA gt={c['gt_bencana']!r} gen={c['gen_bencana']!r}")
        print(f"  GT  BANGUNAN : {c['gt_bangunan']}")
        print(f"  GEN BANGUNAN : {c['gen_bangunan']}")
        print(f"  pre : {c['pre_image_path']}")
        print(f"  post: {c['post_image_path']}")

    out_path = os.path.join(ROOT, "results", "accurate_damage_candidates_full_testset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_total_records": len(records),
            "n_candidates": len(candidates),
            "note": "Candidates are UNVERIFIED (heuristic GT filter) -- read gt_bangunan/"
                    "gen_bangunan before citing any of these as a genuine accurate-detection "
                    "example.",
            "candidates": candidates,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
