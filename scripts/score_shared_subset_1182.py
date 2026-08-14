"""
Score our final vision-LoRA model on the EXACT SAME 1182-item subset the friend's
Qwen2.5-VL-3B eval used (matched by filename in scripts/match_shared_test_subset.py),
so the two models can be compared on identical test items -- no GPU regeneration needed,
reuses existing results/predictions_test_vision.jsonl.

Also scores baseline + text-only ablation on the same subset for completeness, reusing
every scoring function from compute_metrics.py unchanged.

Usage:
    python scripts/score_shared_subset_1182.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.compute_metrics import load_jsonl, coco_style_score, bertscore_f1  # noqa: E402
from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402
from scripts.compare_beam_vs_greedy import diagnostics  # noqa: E402

MATCHED_PATH = os.path.join(ROOT, "results", "shared_subset_1182_matched_indices.json")
BASELINE_PATH = os.path.join(ROOT, "results", "predictions_baseline_id.jsonl")
TEXT_ONLY_PATH = os.path.join(ROOT, "results", "predictions_test.jsonl")
VISION_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "shared_subset_1182_metrics.json")
OUT_MD = os.path.join(ROOT, "results", "shared_subset_1182_metrics.md")


def main():
    with open(MATCHED_PATH, encoding="utf-8") as f:
        matched = json.load(f)
    idx = matched["matched_indices"]
    print(f"[subset] {len(idx)} matched indices (identical items to friend's Qwen2.5-VL-3B eval)")

    base_rows = {r["index"]: r for r in load_jsonl(BASELINE_PATH)}
    text_rows = {r["index"]: r for r in load_jsonl(TEXT_ONLY_PATH)}
    vis_rows = {r["index"]: r for r in load_jsonl(VISION_PATH)}

    common = [i for i in idx if i in base_rows and i in text_rows and i in vis_rows]
    print(f"[data] {len(common)} / {len(idx)} present in all three of our prediction files")

    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    refs = {i: [text_rows[i]["ground_truth_id_canonical"]] for i in common}
    base_res = {i: [codec.canonicalize(base_rows[i]["caption_id_nllb"]) or "."] for i in common}
    text_res = {i: [text_rows[i]["generated_caption"]] for i in common}
    vis_res = {i: [vis_rows[i]["generated_caption"]] for i in common}

    print("[metrics] baseline (zero-shot)")
    base_scores = coco_style_score(refs, base_res)
    print("[metrics] text-only LoRA")
    text_scores = coco_style_score(refs, text_res)
    print("[metrics] text+vision LoRA (final -- the one to compare against Qwen2.5-VL-3B)")
    vis_scores = coco_style_score(refs, vis_res)

    print("[metrics] BERTScore x3")
    base_bert, _ = bertscore_f1([base_res[i][0] for i in common], [refs[i][0] for i in common])
    text_bert, _ = bertscore_f1([text_res[i][0] for i in common], [refs[i][0] for i in common])
    vis_bert, _ = bertscore_f1([vis_res[i][0] for i in common], [refs[i][0] for i in common])

    vis_diag = diagnostics(
        [{"ground_truth_id": vis_rows[i]["ground_truth_id"], "generated_caption": vis_rows[i]["generated_caption"]}
         for i in common],
        ref_field="ground_truth_id",
    )

    results = {
        "n_matched_with_friend_qwen25vl3b_eval": len(common),
        "note": "Identical test items to the friend's Qwen2.5-VL-3B evaluation subset "
                "(test_filenames_first_1182.csv) -- a non-random 'first-N' slice covering "
                "only 6/10 disaster types in our test set (missing: api, api liar, "
                "kebakaran hutan, tsunami). Valid for a fair head-to-head vs Qwen2.5-VL-3B "
                "on matched items, but NOT representative of full test-set performance -- "
                "see results/metrics.json for the authoritative full 2,362-item numbers.",
        "disaster_types_covered": matched["disaster_types_in_subset"],
        "disaster_types_missing": matched["disaster_types_missing_from_subset"],
        "baseline_zero_shot": {"cider": base_scores["cider_avg"], "bleu4": base_scores["bleu4_avg"],
                                "rougeL": base_scores["rougeL_avg"], "bertscore_f1": base_bert},
        "text_only_lora": {"cider": text_scores["cider_avg"], "bleu4": text_scores["bleu4_avg"],
                            "rougeL": text_scores["rougeL_avg"], "bertscore_f1": text_bert},
        "text_plus_vision_lora_FINAL": {"cider": vis_scores["cider_avg"], "bleu4": vis_scores["bleu4_avg"],
                                         "rougeL": vis_scores["rougeL_avg"], "bertscore_f1": vis_bert,
                                         **vis_diag},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_JSON}")

    lines = [
        "# BLIP (vision-LoRA final) scored on the shared 1182-item subset\n",
        f"Matched against `test_filenames_first_1182.csv` -- {len(common)} identical items used "
        "for the friend's Qwen2.5-VL-3B evaluation.\n",
        f"**Caveat**: this subset is a non-random first-N slice, covers only "
        f"{len(matched['disaster_types_in_subset'])}/10 disaster types "
        f"(missing: {', '.join(matched['disaster_types_missing_from_subset'])}). "
        "Use for a fair head-to-head vs Qwen2.5-VL-3B only -- NOT as our headline result "
        "(that remains the full 2,362-item `results/metrics_table.md`).\n",
        "| Metric | Baseline zero-shot | Text-only LoRA | **Text+Vision LoRA (final)** |",
        "|---|---|---|---|",
        f"| CIDEr | {base_scores['cider_avg']:.4f} | {text_scores['cider_avg']:.4f} | **{vis_scores['cider_avg']:.4f}** |",
        f"| BLEU-4 | {base_scores['bleu4_avg']:.4f} | {text_scores['bleu4_avg']:.4f} | **{vis_scores['bleu4_avg']:.4f}** |",
        f"| ROUGE-L | {base_scores['rougeL_avg']:.4f} | {text_scores['rougeL_avg']:.4f} | **{vis_scores['rougeL_avg']:.4f}** |",
        f"| BERTScore F1 | {base_bert:.4f} | {text_bert:.4f} | **{vis_bert:.4f}** |",
        "",
        f"BENCANA exact-match accuracy (final): {vis_diag.get('bencana_exact_match_rate')}",
        f"BANGUNAN no-damage rate (final, generated): {vis_diag.get('building_no_damage_rate_generated')}",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[done] wrote {OUT_MD}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
