"""
3-way comparison on the full 2,362-item test set: InternVL3-1B-hf LoRA after epoch 1,
after epoch 2 (resumed, val_loss 0.8709 -> 0.8055), and our BLIP vision-LoRA final model
(30 epochs across 3 phases). Reuses every scoring function unchanged from compute_metrics.py
/ compare_beam_vs_greedy.py, same pattern as score_internvl3.py.

Usage:
    python scripts/score_internvl3_epoch2.py
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

EP1_PATH = os.path.join(ROOT, "results", "predictions_test_internvl3_epoch1.jsonl")
EP2_PATH = os.path.join(ROOT, "results", "predictions_test_internvl3_epoch2.jsonl")
BLIP_VISION_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "internvl3_epoch1_vs_epoch2_vs_blip.json")


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
    ep1_rows = {r["index"]: r for r in load_jsonl(EP1_PATH)}
    ep2_rows = {r["index"]: r for r in load_jsonl(EP2_PATH)}
    blip_rows = {r["index"]: r for r in load_jsonl(BLIP_VISION_PATH)}
    common = sorted(set(ep1_rows) & set(ep2_rows) & set(blip_rows))
    print(f"[data] epoch1={len(ep1_rows)} epoch2={len(ep2_rows)} blip={len(blip_rows)} paired={len(common)}")

    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)
    refs = {i: [codec.canonicalize(ep1_rows[i]["ground_truth_id"])] for i in common}

    ep1_metrics, ep1_cider = score_system("InternVL3 epoch 1", ep1_rows, common, refs)
    ep2_metrics, ep2_cider = score_system("InternVL3 epoch 2", ep2_rows, common, refs)
    blip_metrics, blip_cider = score_system("BLIP vision-LoRA final (30ep)", blip_rows, common, refs)

    print("[significance] epoch2 vs epoch1 (per-example CIDEr)")
    sig_ep2_vs_ep1 = paired_bootstrap(ep2_cider, ep1_cider)
    print("[significance] epoch2 vs BLIP (per-example CIDEr)")
    sig_ep2_vs_blip = paired_bootstrap(ep2_cider, blip_cider)

    results = {
        "n_paired": len(common),
        "internvl3_epoch1": {"checkpoint": "checkpoints/internvl3_run1/best (val_loss=0.8709)",
                              "params": "938,193,024 (0.938B), LoRA 1,327,104 trainable (0.14%)",
                              **ep1_metrics},
        "internvl3_epoch2": {"checkpoint": "checkpoints/internvl3_run1_ep2/best (val_loss=0.8055, resumed from epoch1)",
                              "params": "938,193,024 (0.938B), LoRA 1,327,104 trainable (0.14%)",
                              **ep2_metrics},
        "blip_vision_lora_final": {"checkpoint": "checkpoints/full_run_v3_vision/best (30 epochs across 3 phases, val_loss=1.4312)",
                                    "params": "223,971,644 (0.224B)", **blip_metrics},
        "significance_epoch2_vs_epoch1_cider": sig_ep2_vs_ep1,
        "significance_epoch2_vs_blip_cider": sig_ep2_vs_blip,
        "caveat": "InternVL3 trained for only 2 epochs (time-boxed decisions after each epoch), vs "
                  "30 epochs across 3 phases for BLIP's final model. Train/val loss for InternVL3 was "
                  "still decreasing at both stop points (0.8709 -> 0.8055 epoch1->epoch2), i.e. NOT "
                  "converged -- these numbers likely understate InternVL3's ceiling. NOT an "
                  "apples-to-apples architecture comparison (different scale, different amount of "
                  "fine-tuning, different pretrained tokenizer/backbone).",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_JSON}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
