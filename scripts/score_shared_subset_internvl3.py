"""
Score InternVL3 (epoch 1 and epoch 2) and BLIP vision-LoRA final on the EXACT SAME 1182-item
subset used for the friend's Qwen2.5-VL-3B evaluation (test_filenames_first_1182.csv, matched
in scripts/match_shared_test_subset.py). No new generation needed -- all three systems already
have full-2362-set predictions on disk; this just filters+rescores on the shared indices, so
the numbers are directly comparable to Qwen2.5-VL-3B once that eval lands.

Usage:
    python scripts/score_shared_subset_internvl3.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_metrics import load_jsonl, coco_style_score, bertscore_f1, paired_bootstrap  # noqa: E402
from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402
from scripts.compare_beam_vs_greedy import diagnostics  # noqa: E402

MATCHED_PATH = os.path.join(ROOT, "results", "shared_subset_1182_matched_indices.json")
EP1_PATH = os.path.join(ROOT, "results", "predictions_test_internvl3_epoch1.jsonl")
EP2_PATH = os.path.join(ROOT, "results", "predictions_test_internvl3_epoch2.jsonl")
BLIP_VISION_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "shared_subset_1182_internvl3_vs_blip.json")


def score_system(name, rows_by_idx, common, refs):
    res = {i: [rows_by_idx[i]["generated_caption"]] for i in common}
    print(f"[metrics] {name}")
    scores = coco_style_score(refs, res)
    bert, _ = bertscore_f1([res[i][0] for i in common], [refs[i][0] for i in common])
    diag = diagnostics([{"ground_truth_id": rows_by_idx[i]["ground_truth_id"],
                          "generated_caption": rows_by_idx[i]["generated_caption"]} for i in common])
    return {"cider": scores["cider_avg"], "bleu4": scores["bleu4_avg"], "rougeL": scores["rougeL_avg"],
            "bertscore_f1": bert, **diag}, scores["cider_per_item"]


def main():
    with open(MATCHED_PATH, encoding="utf-8") as f:
        matched = json.load(f)
    idx_1182 = set(matched["matched_indices"])
    print(f"[subset] {len(idx_1182)} indices matched to friend's Qwen2.5-VL-3B eval")

    ep1_rows = {r["index"]: r for r in load_jsonl(EP1_PATH)}
    ep2_rows = {r["index"]: r for r in load_jsonl(EP2_PATH)}
    blip_rows = {r["index"]: r for r in load_jsonl(BLIP_VISION_PATH)}
    common = sorted(idx_1182 & set(ep1_rows) & set(ep2_rows) & set(blip_rows))
    print(f"[data] {len(common)} / {len(idx_1182)} shared-subset indices present in all three prediction files")

    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)
    refs = {i: [codec.canonicalize(ep1_rows[i]["ground_truth_id"])] for i in common}

    ep1_metrics, ep1_cider = score_system("InternVL3 epoch 1 (shared subset)", ep1_rows, common, refs)
    ep2_metrics, ep2_cider = score_system("InternVL3 epoch 2 (shared subset)", ep2_rows, common, refs)
    blip_metrics, blip_cider = score_system("BLIP vision-LoRA final (shared subset)", blip_rows, common, refs)

    print("[significance] epoch2 vs BLIP (shared subset, per-example CIDEr)")
    sig_ep2_vs_blip = paired_bootstrap(ep2_cider, blip_cider)
    print("[significance] epoch2 vs epoch1 (shared subset, per-example CIDEr)")
    sig_ep2_vs_ep1 = paired_bootstrap(ep2_cider, ep1_cider)

    results = {
        "n_shared_subset": len(common),
        "note": "Identical 1182 test items as the friend's Qwen2.5-VL-3B evaluation "
                "(test_filenames_first_1182.csv) -- once Qwen's own eval on this subset lands, "
                "it can be placed directly alongside these numbers for a genuine 3-way comparison. "
                "This subset is a non-random 'first-N' slice covering only 6/10 disaster types "
                "(see results/shared_subset_1182_matched_indices.json for exact coverage) -- "
                "valid for head-to-head comparison on identical items, NOT representative of "
                "full-test-set performance (use results/internvl3_epoch1_vs_epoch2_vs_blip.json "
                "for the full-2362-set numbers instead).",
        "internvl3_epoch1": {"checkpoint": "checkpoints/internvl3_run1/best (val_loss=0.8709)", **ep1_metrics},
        "internvl3_epoch2": {"checkpoint": "checkpoints/internvl3_run1_ep2/best (val_loss=0.8055)", **ep2_metrics},
        "blip_vision_lora_final": {"checkpoint": "checkpoints/full_run_v3_vision/best (30 epochs)", **blip_metrics},
        "significance_epoch2_vs_blip_cider": sig_ep2_vs_blip,
        "significance_epoch2_vs_epoch1_cider": sig_ep2_vs_ep1,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_JSON}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
