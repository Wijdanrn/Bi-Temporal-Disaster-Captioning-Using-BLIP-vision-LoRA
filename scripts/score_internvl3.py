"""
Score the InternVL3-1B-hf (LoRA, epoch 1 only) predictions on the full test set, reusing
every scoring function from compute_metrics.py / compare_beam_vs_greedy.py unchanged, and
compare directly against our own BLIP vision-LoRA final model on the SAME 2,362 items.

Usage:
    python scripts/score_internvl3.py
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

INTERNVL3_PATH = os.path.join(ROOT, "results", "predictions_test_internvl3_epoch1.jsonl")
BLIP_VISION_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "internvl3_epoch1_vs_blip.json")


def main():
    iv_rows = {r["index"]: r for r in load_jsonl(INTERNVL3_PATH)}
    blip_rows = {r["index"]: r for r in load_jsonl(BLIP_VISION_PATH)}
    common = sorted(set(iv_rows) & set(blip_rows))
    print(f"[data] InternVL3={len(iv_rows)} BLIP-vision={len(blip_rows)} paired={len(common)}")

    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    refs = {i: [codec.canonicalize(iv_rows[i]["ground_truth_id"])] for i in common}
    iv_res = {i: [iv_rows[i]["generated_caption"]] for i in common}
    blip_res = {i: [blip_rows[i]["generated_caption"]] for i in common}

    print("[metrics] InternVL3 (epoch 1 LoRA)")
    iv_scores = coco_style_score(refs, iv_res)
    print("[metrics] BLIP vision-LoRA (final, for reference)")
    blip_scores = coco_style_score(refs, blip_res)

    print("[metrics] BERTScore x2")
    iv_bert, _ = bertscore_f1([iv_res[i][0] for i in common], [refs[i][0] for i in common])
    blip_bert, _ = bertscore_f1([blip_res[i][0] for i in common], [refs[i][0] for i in common])

    print("[significance] paired bootstrap, InternVL3 vs BLIP-vision, per-example CIDEr")
    sig = paired_bootstrap(iv_scores["cider_per_item"], blip_scores["cider_per_item"])

    iv_diag = diagnostics([{"ground_truth_id": iv_rows[i]["ground_truth_id"],
                             "generated_caption": iv_rows[i]["generated_caption"]} for i in common])
    blip_diag = diagnostics([{"ground_truth_id": blip_rows[i]["ground_truth_id"],
                               "generated_caption": blip_rows[i]["generated_caption"]} for i in common])

    results = {
        "n_paired": len(common),
        "internvl3_epoch1": {
            "checkpoint": "checkpoints/internvl3_run1/best (epoch 1 only, val_loss=0.8709, training stopped after epoch 1)",
            "params": "938,193,024 (0.938B), LoRA q_proj/v_proj on LLM+vision, 1,327,104 trainable (0.14%)",
            "cider": iv_scores["cider_avg"], "bleu4": iv_scores["bleu4_avg"],
            "rougeL": iv_scores["rougeL_avg"], "bertscore_f1": iv_bert,
            **iv_diag,
        },
        "blip_vision_lora_final": {
            "checkpoint": "checkpoints/full_run_v3_vision/best (final, 10+10 epochs, val_loss=1.4312)",
            "params": "223,971,644 (0.224B)",
            "cider": blip_scores["cider_avg"], "bleu4": blip_scores["bleu4_avg"],
            "rougeL": blip_scores["rougeL_avg"], "bertscore_f1": blip_bert,
            **blip_diag,
        },
        "significance_internvl3_vs_blip_cider": sig,
        "caveat": "InternVL3 trained for 1 epoch only (time-boxed decision, not necessarily converged -- "
                  "see train loss still trending down at stop time). NOT an apples-to-apples architecture "
                  "comparison: different scale (0.94B vs 0.22B), different pretrained tokenizer/backbone "
                  "quality, different amount of fine-tuning (1 epoch vs 30 epochs across 3 phases -- "
                  "full_run_v1 10ep + full_run_v2 10ep + full_run_v3_vision 10ep -- for BLIP's final model).",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_JSON}")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
