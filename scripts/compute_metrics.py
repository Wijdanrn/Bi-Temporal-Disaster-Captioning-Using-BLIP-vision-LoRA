"""
Fase 4 -- compute CIDEr / BLEU-4 / ROUGE-L (+ BERTScore substitute for SPICE) on real
generated predictions, for our model AND the translated zero-shot baseline, then run paired
bootstrap significance testing. Methodology fixed in docs/design_decisions.md SS4 before any
number below was produced.

Inputs (must already exist -- produced by generate_predictions.py, generate_baseline.py,
translate_baseline.py):
    results/predictions_test.jsonl       -- our model, 2,362 records
    results/predictions_baseline_id.jsonl -- baseline, NLLB-translated to Indonesian

Outputs:
    results/metrics.json
    results/metrics_table.md

Usage:
    python scripts/compute_metrics.py
"""
from __future__ import annotations

import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402

PRED_PATH = os.path.join(ROOT, "results", "predictions_test.jsonl")
BASELINE_PATH = os.path.join(ROOT, "results", "predictions_baseline_id.jsonl")
METRICS_JSON = os.path.join(ROOT, "results", "metrics.json")
METRICS_MD = os.path.join(ROOT, "results", "metrics_table.md")
N_BOOTSTRAP = 10000
SEED = 42


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def coco_style_score(gts: dict, res: dict):
    """Returns dict of {cider, bleu4, rougeL} averages, plus per-item arrays aligned to
    the SAME iteration order (dict insertion order == sorted key order used consistently)."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge

    tok = PTBTokenizer()
    gts_fmt = {k: [{"caption": c} for c in v] for k, v in gts.items()}
    res_fmt = {k: [{"caption": c} for c in v] for k, v in res.items()}
    gts_tok = tok.tokenize(gts_fmt)
    res_tok = tok.tokenize(res_fmt)

    ids = list(gts_tok.keys())
    assert ids == list(res_tok.keys()), "tokenizer must preserve key order for gts/res"

    cider_avg, cider_scores = Cider().compute_score(gts_tok, res_tok)
    bleu_avg, bleu_scores = Bleu(4).compute_score(gts_tok, res_tok, verbose=0)
    rouge_avg, rouge_scores = Rouge().compute_score(gts_tok, res_tok)

    return {
        "ids": ids,
        "cider_avg": float(cider_avg),
        "cider_per_item": [float(x) for x in cider_scores],
        "bleu4_avg": float(bleu_avg[3]),
        "bleu4_per_item": [float(x) for x in bleu_scores[3]],
        "rougeL_avg": float(rouge_avg),
        "rougeL_per_item": [float(x) for x in rouge_scores],
    }


def _patch_bert_score_tokenizer_max_length():
    """Compatibility shim: bert-score==0.3.13 (last released ~2021) predates transformers 5.x's
    Rust tokenizer backend. `indolem/indobert-base-uncased` ships no `model_max_length` in its
    tokenizer_config.json, so transformers defaults it to a sentinel
    ``int(1e30)``-scale value; the new Rust `enable_truncation` cannot represent that and raises
    ``OverflowError: int too big to convert`` (reproduced, see docs/design_decisions.md SS4.1 run
    log). BERT's real position limit is 512, so capping model_max_length there is not a numeric
    hack that changes results -- no real caption in this corpus is anywhere near 512 tokens after
    wordpiece splitting -- it only fixes a version-mismatch bug in the third-party library."""
    # bert_score/__init__.py does `from .score import score`, which SHADOWS the `bert_score.score`
    # submodule attribute with the function of the same name -- `import bert_score.score as m`
    # therefore silently binds `m` to the function, not the module, and patching that object's
    # `get_tokenizer` name is a no-op. The real submodule (where `get_tokenizer` is actually
    # resolved at call time) only exists in `sys.modules`. Verified live before relying on it.
    import sys
    import bert_score  # noqa: F401  (ensures bert_score.score is registered in sys.modules)
    import bert_score.utils as bs_utils

    orig = bs_utils.get_tokenizer

    def patched(model_type, use_fast=False):
        tok = orig(model_type, use_fast=use_fast)
        if tok.model_max_length > 100_000:
            tok.model_max_length = 512
        return tok

    bs_utils.get_tokenizer = patched
    sys.modules["bert_score.score"].get_tokenizer = patched


def bertscore_f1(cands: list[str], refs: list[str]) -> tuple[float, list[float]]:
    """SPICE substitute (SS4.1): Indonesian-aware semantic similarity via
    indolem/indobert-base-uncased -- the same encoder family used as our tokenizer/vocab base."""
    _patch_bert_score_tokenizer_max_length()
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(
        cands, refs, model_type="indolem/indobert-base-uncased", num_layers=12,
        lang=None, verbose=False, batch_size=32,
    )
    f1 = F1.tolist()
    return float(np.mean(f1)), f1


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    """Paired bootstrap resampling: is mean(a) > mean(b) (a = our model, b = baseline)?"""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    assert len(b) == n
    diffs = a - b
    obs_diff = float(diffs.mean())

    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    frac_a_better = float((boot_diffs > 0).mean())  # one-sided: our model's mean CIDEr > baseline's
    p_value_two_sided = float(2 * min((boot_diffs > 0).mean(), (boot_diffs < 0).mean()))
    return {
        "observed_mean_diff": obs_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "frac_bootstrap_resamples_favoring_ours": frac_a_better,
        "two_sided_p_value_approx": p_value_two_sided,
        "n_bootstrap": n_boot,
        "n_paired_examples": n,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("[load] predictions")
    pred_rows = load_jsonl(PRED_PATH)
    base_rows = load_jsonl(BASELINE_PATH)
    pred_by_idx = {r["index"]: r for r in pred_rows}
    base_by_idx = {r["index"]: r for r in base_rows}

    common_idx = sorted(set(pred_by_idx) & set(base_by_idx))
    print(f"[data] our-model predictions: {len(pred_rows)}, baseline: {len(base_rows)}, "
          f"matched (paired) examples: {len(common_idx)}")

    print("[codec] building tokenizer/codec for canonicalization (no model weights needed)")
    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    # references: same canonicalized text used for BOTH systems
    refs = {i: [pred_by_idx[i]["ground_truth_id_canonical"]] for i in common_idx}
    ours_res = {i: [pred_by_idx[i]["generated_caption"]] for i in common_idx}
    base_res = {
        i: [codec.canonicalize(base_by_idx[i]["caption_id_nllb"]) or "."]
        for i in common_idx
    }

    print("[metrics] our model: CIDEr / BLEU-4 / ROUGE-L (pycocoevalcap, PTBTokenizer)")
    ours_scores = coco_style_score(refs, ours_res)
    print("[metrics] baseline: CIDEr / BLEU-4 / ROUGE-L")
    base_scores = coco_style_score(refs, base_res)

    print("[metrics] BERTScore (SPICE substitute) -- our model")
    ours_bert_avg, ours_bert_per = bertscore_f1(
        [ours_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])
    print("[metrics] BERTScore (SPICE substitute) -- baseline")
    base_bert_avg, base_bert_per = bertscore_f1(
        [base_res[i][0] for i in common_idx], [refs[i][0] for i in common_idx])

    print("[significance] paired bootstrap resampling on per-example CIDEr")
    # coco_style_score's ids are the SAME common_idx list in the same order (dict insertion order
    # preserved through PTBTokenizer since it iterates dict items in order and returns a dict
    # keyed by the same string ids) -- verified by asserting ids match common_idx below.
    assert [str(x) for x in common_idx] == ours_scores["ids"] or \
        [int(x) for x in ours_scores["ids"]] == common_idx, \
        "id ordering drifted between refs/res and scorer output"
    sig = paired_bootstrap(ours_scores["cider_per_item"], base_scores["cider_per_item"])

    results = {
        "checkpoint": "checkpoints/full_run_v2/best",
        "n_test_records_total": 2362,
        "n_paired_examples_scored": len(common_idx),
        "generation_settings": {
            "num_beams": 1, "max_new_tokens_ours": 300, "max_new_tokens_baseline": 50,
            "decoding": "greedy",
        },
        "our_model": {
            "cider": ours_scores["cider_avg"],
            "bleu4": ours_scores["bleu4_avg"],
            "rougeL": ours_scores["rougeL_avg"],
            "bertscore_f1_indolem": ours_bert_avg,
        },
        "baseline_zero_shot_blip_translated": {
            "cider": base_scores["cider_avg"],
            "bleu4": base_scores["bleu4_avg"],
            "rougeL": base_scores["rougeL_avg"],
            "bertscore_f1_indolem": base_bert_avg,
        },
        "significance_test_paired_bootstrap_cider": sig,
        "spice_status": "NOT computed (WordNet has no Indonesian coverage); substituted with "
                        "indolem/indobert-base-uncased BERTScore F1 -- see docs/design_decisions.md SS4.1",
        "baseline_methodology": "see docs/design_decisions.md SS4.2 (fixed before running)",
    }

    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {METRICS_JSON}")

    # ---- qualitative sample: 8 examples ----
    rng = random.Random(SEED)
    sample_idx = rng.sample(common_idx, min(8, len(common_idx)))

    lines = []
    lines.append("# Fase 4 -- Real Evaluation Results\n")
    lines.append(f"Checkpoint: `checkpoints/full_run_v2/best` (best val_loss 1.5343)\n")
    lines.append(f"Test set: {len(common_idx)} / 2,362 usable records scored "
                 f"(paired against baseline predictions)\n")
    lines.append("## Aggregate metrics\n")
    lines.append("| Metric | Our model (fine-tuned) | Baseline (zero-shot BLIP -> NLLB Indonesian) |")
    lines.append("|---|---|---|")
    lines.append(f"| CIDEr | {ours_scores['cider_avg']:.4f} | {base_scores['cider_avg']:.4f} |")
    lines.append(f"| BLEU-4 | {ours_scores['bleu4_avg']:.4f} | {base_scores['bleu4_avg']:.4f} |")
    lines.append(f"| ROUGE-L | {ours_scores['rougeL_avg']:.4f} | {base_scores['rougeL_avg']:.4f} |")
    lines.append(f"| BERTScore F1 (SPICE substitute, indolem) | {ours_bert_avg:.4f} | {base_bert_avg:.4f} |")
    lines.append("")
    lines.append("SPICE itself was NOT computed (no Indonesian WordNet coverage) -- see "
                 "docs/design_decisions.md SS4.1 for the substitution decision.\n")
    lines.append("## Statistical significance (paired bootstrap, 10,000 resamples, per-example CIDEr)\n")
    lines.append(f"- Observed mean CIDEr difference (ours - baseline): **{sig['observed_mean_diff']:.4f}**")
    lines.append(f"- 95% CI on the difference: [{sig['ci95_low']:.4f}, {sig['ci95_high']:.4f}]")
    lines.append(f"- Fraction of bootstrap resamples favoring our model: {sig['frac_bootstrap_resamples_favoring_ours']:.4f}")
    lines.append(f"- Approx. two-sided p-value: {sig['two_sided_p_value_approx']:.4g}")
    lines.append(f"- Paired N: {sig['n_paired_examples']}\n")
    lines.append("## Qualitative samples (our model)\n")
    for i in sample_idx:
        ref = refs[i][0]
        gen = ours_res[i][0]
        lines.append(f"### index {i}")
        lines.append(f"**Reference (canonicalized):**\n```\n{ref[:600]}\n```")
        lines.append(f"**Generated:**\n```\n{gen[:600]}\n```")
        lines.append("")

    with open(METRICS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[done] wrote {METRICS_MD}")


if __name__ == "__main__":
    main()
