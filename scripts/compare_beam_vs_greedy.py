"""
Fase 4 follow-up -- score results/predictions_test_beam5.jsonl (num_beams=5) the same
way as the greedy baseline (results/predictions_test.jsonl), then compare the two
directly (paired bootstrap, greedy vs beam5) plus re-run the diagnostic checks
(BENCANA disaster-type accuracy, BANGUNAN no-damage boilerplate rate) that flagged
the original bias, to see whether beam search alone fixes it.

Reuses scoring functions from compute_metrics.py -- no metric logic duplicated.

Usage:
    python scripts/compare_beam_vs_greedy.py
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from scripts.compute_metrics import (  # noqa: E402
    load_jsonl, coco_style_score, bertscore_f1, paired_bootstrap,
)
from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402

GREEDY_PATH = os.path.join(ROOT, "results", "predictions_test.jsonl")
BEAM_PATH = os.path.join(ROOT, "results", "predictions_test_beam5.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "beam_vs_greedy.json")


def get_section(text, name):
    m = re.search(rf"{name}:\s*(.+)", text)
    return m.group(1).strip() if m else None


def get_bencana(text):
    v = get_section(text, "BENCANA")
    return v.lower() if v else None


def building_no_damage(text):
    m = re.search(r"BANGUNAN:\s*(.+?)(?:\n[A-Z_]+:|$)", text, re.DOTALL)
    b = (m.group(1) if m else "").lower()
    return ("tidak ada kerusakan" in b) or ("tidak menunjukkan kerusakan" in b)


def diagnostics(rows, ref_field="ground_truth_id", gen_field="generated_caption"):
    n = len(rows)
    matches = sum(1 for r in rows if get_bencana(r[ref_field]) == get_bencana(r[gen_field])
                  and get_bencana(r[ref_field]) is not None)
    total_labeled = sum(1 for r in rows if get_bencana(r[ref_field]) and get_bencana(r[gen_field]))
    gen_no_damage = sum(1 for r in rows if building_no_damage(r[gen_field]))
    ref_no_damage = sum(1 for r in rows if building_no_damage(r[ref_field]))
    distinct_gen = len(set(r[gen_field] for r in rows))
    return {
        "n": n,
        "bencana_exact_match_rate": matches / total_labeled if total_labeled else None,
        "building_no_damage_rate_generated": gen_no_damage / n,
        "building_no_damage_rate_reference": ref_no_damage / n,
        "distinct_generated_captions": distinct_gen,
        "distinct_generated_fraction": distinct_gen / n,
    }


def main():
    print("[load] predictions")
    greedy_rows = load_jsonl(GREEDY_PATH)
    beam_rows = load_jsonl(BEAM_PATH)
    greedy_by_idx = {r["index"]: r for r in greedy_rows}
    beam_by_idx = {r["index"]: r for r in beam_rows}
    common_idx = sorted(set(greedy_by_idx) & set(beam_by_idx))
    print(f"[data] greedy: {len(greedy_rows)}, beam5: {len(beam_rows)}, paired: {len(common_idx)}")

    print("[codec] building tokenizer/codec")
    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    refs = {i: [greedy_by_idx[i]["ground_truth_id_canonical"]] for i in common_idx}
    greedy_res = {i: [greedy_by_idx[i]["generated_caption"]] for i in common_idx}
    beam_res = {i: [beam_by_idx[i]["generated_caption"]] for i in common_idx}

    print("[metrics] greedy: CIDEr/BLEU-4/ROUGE-L")
    greedy_scores = coco_style_score(refs, greedy_res)
    print("[metrics] beam5: CIDEr/BLEU-4/ROUGE-L")
    beam_scores = coco_style_score(refs, beam_res)

    print("[metrics] BERTScore -- greedy")
    greedy_bert_avg, _ = bertscore_f1([greedy_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])
    print("[metrics] BERTScore -- beam5")
    beam_bert_avg, _ = bertscore_f1([beam_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])

    print("[significance] paired bootstrap, beam5 vs greedy, per-example CIDEr")
    sig = paired_bootstrap(beam_scores["cider_per_item"], greedy_scores["cider_per_item"])

    greedy_diag = diagnostics([greedy_by_idx[i] for i in common_idx])
    beam_diag = diagnostics([beam_by_idx[i] for i in common_idx])

    results = {
        "n_paired": len(common_idx),
        "greedy": {
            "cider": greedy_scores["cider_avg"], "bleu4": greedy_scores["bleu4_avg"],
            "rougeL": greedy_scores["rougeL_avg"], "bertscore_f1": greedy_bert_avg,
            **greedy_diag,
        },
        "beam5": {
            "cider": beam_scores["cider_avg"], "bleu4": beam_scores["bleu4_avg"],
            "rougeL": beam_scores["rougeL_avg"], "bertscore_f1": beam_bert_avg,
            **beam_diag,
        },
        "significance_beam5_vs_greedy_cider": sig,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[done] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
