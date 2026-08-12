"""
Fase 5 -- score results/predictions_test_vision.jsonl (text+vision LoRA,
checkpoints/full_run_v3_vision/best) against the original text-only greedy baseline
(results/predictions_test.jsonl, checkpoints/full_run_v2/best), same metrics and the
same diagnostic checks (BENCANA accuracy, BANGUNAN no-damage rate) used throughout
Fase 4-5, to test whether adding vision LoRA actually fixes the documented bias.

Usage:
    python scripts/compare_vision_vs_baseline.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_metrics import (  # noqa: E402
    load_jsonl, coco_style_score, bertscore_f1, paired_bootstrap,
)
from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402
from scripts.compare_beam_vs_greedy import diagnostics  # noqa: E402

BASELINE_PATH = os.path.join(ROOT, "results", "predictions_test.jsonl")
VISION_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "vision_vs_baseline.json")


def main():
    print("[load] predictions")
    base_rows = load_jsonl(BASELINE_PATH)
    vis_rows = load_jsonl(VISION_PATH)
    base_by_idx = {r["index"]: r for r in base_rows}
    vis_by_idx = {r["index"]: r for r in vis_rows}
    common_idx = sorted(set(base_by_idx) & set(vis_by_idx))
    print(f"[data] baseline (text-only): {len(base_rows)}, vision-lora: {len(vis_rows)}, paired: {len(common_idx)}")

    print("[codec] building tokenizer/codec")
    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    refs = {i: [base_by_idx[i]["ground_truth_id_canonical"]] for i in common_idx}
    base_res = {i: [base_by_idx[i]["generated_caption"]] for i in common_idx}
    vis_res = {i: [vis_by_idx[i]["generated_caption"]] for i in common_idx}

    print("[metrics] baseline (text-only LoRA): CIDEr/BLEU-4/ROUGE-L")
    base_scores = coco_style_score(refs, base_res)
    print("[metrics] vision-lora: CIDEr/BLEU-4/ROUGE-L")
    vis_scores = coco_style_score(refs, vis_res)

    print("[metrics] BERTScore -- baseline")
    base_bert_avg, _ = bertscore_f1([base_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])
    print("[metrics] BERTScore -- vision-lora")
    vis_bert_avg, _ = bertscore_f1([vis_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])

    print("[significance] paired bootstrap, vision-lora vs text-only baseline, per-example CIDEr")
    sig = paired_bootstrap(vis_scores["cider_per_item"], base_scores["cider_per_item"])

    base_diag = diagnostics([base_by_idx[i] for i in common_idx])
    vis_diag = diagnostics([vis_by_idx[i] for i in common_idx])

    results = {
        "n_paired": len(common_idx),
        "baseline_text_only_lora": {
            "checkpoint": "checkpoints/full_run_v2/best",
            "cider": base_scores["cider_avg"], "bleu4": base_scores["bleu4_avg"],
            "rougeL": base_scores["rougeL_avg"], "bertscore_f1": base_bert_avg,
            **base_diag,
        },
        "vision_lora": {
            "checkpoint": "checkpoints/full_run_v3_vision/best",
            "cider": vis_scores["cider_avg"], "bleu4": vis_scores["bleu4_avg"],
            "rougeL": vis_scores["rougeL_avg"], "bertscore_f1": vis_bert_avg,
            **vis_diag,
        },
        "significance_vision_vs_baseline_cider": sig,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[done] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
