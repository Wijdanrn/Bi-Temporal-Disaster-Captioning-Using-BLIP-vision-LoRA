"""
Fase 5 -- consolidated FINAL results: zero-shot BLIP baseline vs. text-only LoRA
(full_run_v2, kept as an ablation data point) vs. text+vision LoRA
(full_run_v3_vision, the final reported model). Replaces the two-way
results/metrics.json / metrics_table.md from Fase 4 as the official numbers,
now that the vision-LoRA ablation (Fase 5) is confirmed to help.

Reuses every scoring function from compute_metrics.py -- no metric logic
duplicated, so these numbers are computed the identical way the Fase 4 numbers
were (same tokenizer, same PTBTokenizer/pycocoevalcap pipeline, same BERTScore
model/patch, same paired-bootstrap procedure).

Usage:
    python scripts/compute_final_metrics.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_metrics import (  # noqa: E402
    load_jsonl, coco_style_score, bertscore_f1, paired_bootstrap, SEED,
)
from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402
from scripts.compare_beam_vs_greedy import diagnostics  # noqa: E402

BASELINE_PATH = os.path.join(ROOT, "results", "predictions_baseline_id.jsonl")
TEXT_ONLY_PATH = os.path.join(ROOT, "results", "predictions_test.jsonl")
VISION_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "metrics.json")
OUT_MD = os.path.join(ROOT, "results", "metrics_table.md")


def score_system(refs, res_rows_by_idx, common_idx, field="generated_caption"):
    res = {i: [res_rows_by_idx[i][field]] for i in common_idx}
    return coco_style_score(refs, res), res


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("[load] predictions")
    base_rows = load_jsonl(BASELINE_PATH)
    text_rows = load_jsonl(TEXT_ONLY_PATH)
    vis_rows = load_jsonl(VISION_PATH)
    base_by_idx = {r["index"]: r for r in base_rows}
    text_by_idx = {r["index"]: r for r in text_rows}
    vis_by_idx = {r["index"]: r for r in vis_rows}
    common_idx = sorted(set(base_by_idx) & set(text_by_idx) & set(vis_by_idx))
    print(f"[data] baseline={len(base_rows)} text_only={len(text_rows)} vision={len(vis_rows)} "
          f"paired(all three)={len(common_idx)}")

    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    refs = {i: [text_by_idx[i]["ground_truth_id_canonical"]] for i in common_idx}
    base_res = {i: [codec.canonicalize(base_by_idx[i]["caption_id_nllb"]) or "."] for i in common_idx}

    print("[metrics] baseline (zero-shot BLIP -> NLLB)")
    base_scores = coco_style_score(refs, base_res)
    print("[metrics] text-only LoRA (full_run_v2, ablation)")
    text_scores, text_res = score_system(refs, text_by_idx, common_idx)
    print("[metrics] text+vision LoRA (full_run_v3_vision, FINAL)")
    vis_scores, vis_res = score_system(refs, vis_by_idx, common_idx)

    print("[metrics] BERTScore x3")
    base_bert, _ = bertscore_f1([base_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])
    text_bert, _ = bertscore_f1([text_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])
    vis_bert, _ = bertscore_f1([vis_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])

    print("[significance] final vs baseline (per-example CIDEr)")
    sig_vs_baseline = paired_bootstrap(vis_scores["cider_per_item"], base_scores["cider_per_item"])
    print("[significance] final vs text-only ablation (per-example CIDEr)")
    sig_vs_textonly = paired_bootstrap(vis_scores["cider_per_item"], text_scores["cider_per_item"])

    text_diag = diagnostics([text_by_idx[i] for i in common_idx])
    vis_diag = diagnostics([vis_by_idx[i] for i in common_idx])

    results = {
        "final_checkpoint": "checkpoints/full_run_v3_vision/best",
        "ablation_checkpoint_text_only": "checkpoints/full_run_v2/best",
        "n_test_records_total": 2362,
        "n_paired_examples_scored": len(common_idx),
        "generation_settings": {"num_beams": 1, "max_new_tokens": 300, "decoding": "greedy"},
        "baseline_zero_shot_blip_translated": {
            "cider": base_scores["cider_avg"], "bleu4": base_scores["bleu4_avg"],
            "rougeL": base_scores["rougeL_avg"], "bertscore_f1_indolem": base_bert,
        },
        "ablation_text_only_lora": {
            "cider": text_scores["cider_avg"], "bleu4": text_scores["bleu4_avg"],
            "rougeL": text_scores["rougeL_avg"], "bertscore_f1_indolem": text_bert,
            **text_diag,
        },
        "final_text_plus_vision_lora": {
            "cider": vis_scores["cider_avg"], "bleu4": vis_scores["bleu4_avg"],
            "rougeL": vis_scores["rougeL_avg"], "bertscore_f1_indolem": vis_bert,
            **vis_diag,
        },
        "significance_final_vs_baseline_cider": sig_vs_baseline,
        "significance_final_vs_text_only_ablation_cider": sig_vs_textonly,
        "spice_status": "NOT computed (WordNet has no Indonesian coverage); substituted with "
                        "indolem/indobert-base-uncased BERTScore F1 -- see docs/design_decisions.md SS4.1",
        "baseline_methodology": "see docs/design_decisions.md SS4.2 (fixed before running)",
        "vision_lora_ablation_methodology": "see docs/design_decisions.md SS5 -- beam search "
                                            "ruled out as the cause of the bias first (SS5.x), "
                                            "then vision qkv LoRA added and shown to help",
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_JSON}")

    rng = random.Random(SEED)
    sample_idx = rng.sample(common_idx, min(8, len(common_idx)))

    lines = []
    lines.append("# Fase 4-5 -- Final Real Evaluation Results\n")
    lines.append(f"**Final model checkpoint:** `checkpoints/full_run_v3_vision/best` "
                 f"(text LoRA + vision qkv LoRA + trained embeddings, val_loss 1.4312)\n")
    lines.append(f"Test set: {len(common_idx)} / 2,362 usable records scored\n")
    lines.append("## Aggregate metrics\n")
    lines.append("| Metric | Zero-shot BLIP baseline | Text-only LoRA (ablation) | **Text+Vision LoRA (final)** |")
    lines.append("|---|---|---|---|")
    lines.append(f"| CIDEr | {base_scores['cider_avg']:.4f} | {text_scores['cider_avg']:.4f} | **{vis_scores['cider_avg']:.4f}** |")
    lines.append(f"| BLEU-4 | {base_scores['bleu4_avg']:.4f} | {text_scores['bleu4_avg']:.4f} | **{vis_scores['bleu4_avg']:.4f}** |")
    lines.append(f"| ROUGE-L | {base_scores['rougeL_avg']:.4f} | {text_scores['rougeL_avg']:.4f} | **{vis_scores['rougeL_avg']:.4f}** |")
    lines.append(f"| BERTScore F1 (SPICE substitute) | {base_bert:.4f} | {text_bert:.4f} | **{vis_bert:.4f}** |")
    lines.append("")
    lines.append("## Diagnostic checks (Fase 5 -- disaster-type accuracy & no-damage bias)\n")
    lines.append("| Diagnostic | Text-only LoRA | **Text+Vision LoRA (final)** | Real reference rate |")
    lines.append("|---|---|---|---|")
    lines.append(f"| BENCANA (disaster type) exact-match accuracy | {text_diag['bencana_exact_match_rate']:.1%} | **{vis_diag['bencana_exact_match_rate']:.1%}** | -- |")
    lines.append(f"| BANGUNAN \"no damage\" boilerplate rate | {text_diag['building_no_damage_rate_generated']:.1%} | **{vis_diag['building_no_damage_rate_generated']:.1%}** | {vis_diag['building_no_damage_rate_reference']:.1%} |")
    lines.append(f"| Distinct generated captions | {text_diag['distinct_generated_fraction']:.1%} | {vis_diag['distinct_generated_fraction']:.1%} | -- |")
    lines.append("")
    lines.append("SPICE itself was NOT computed (no Indonesian WordNet coverage) -- see "
                 "docs/design_decisions.md SS4.1 for the substitution decision.\n")
    lines.append("## Statistical significance (paired bootstrap, 10,000 resamples, per-example CIDEr)\n")
    lines.append("**Final model vs. zero-shot baseline:**")
    lines.append(f"- Observed mean CIDEr difference: **{sig_vs_baseline['observed_mean_diff']:.4f}**")
    lines.append(f"- 95% CI: [{sig_vs_baseline['ci95_low']:.4f}, {sig_vs_baseline['ci95_high']:.4f}]")
    lines.append(f"- Approx. two-sided p-value: {sig_vs_baseline['two_sided_p_value_approx']:.4g}\n")
    lines.append("**Final model (+ vision LoRA) vs. text-only LoRA ablation:**")
    lines.append(f"- Observed mean CIDEr difference: **{sig_vs_textonly['observed_mean_diff']:.4f}**")
    lines.append(f"- 95% CI: [{sig_vs_textonly['ci95_low']:.4f}, {sig_vs_textonly['ci95_high']:.4f}]")
    lines.append(f"- Approx. two-sided p-value: {sig_vs_textonly['two_sided_p_value_approx']:.4g}")
    lines.append(f"- Paired N: {sig_vs_textonly['n_paired_examples']}\n")
    lines.append("## Qualitative samples (final model)\n")
    for i in sample_idx:
        ref = refs[i][0]
        gen = vis_res[i][0]
        lines.append(f"### index {i}")
        lines.append(f"**Reference (canonicalized):**\n```\n{ref[:600]}\n```")
        lines.append(f"**Generated (text+vision LoRA):**\n```\n{gen[:600]}\n```")
        lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[done] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
