"""
Fase 6 (xAI) -- faithfulness / sanity checks for the attention maps produced by
`scripts/xai_attention_rollout.py`.

WHY THIS FILE EXISTS
--------------------
A heatmap that renders without error proves nothing. It can be wired to the wrong layer, the
wrong head, or the wrong token and still look like a competent explanation. This project's
standing rule ("never trust an unverified number") applies to pictures too, so no rollout map
is reported anywhere without the two tests below having been run on the SAME maps.

TEST 1 -- DELETION / INSERTION (does the highlighted region actually drive the output?)
    Rank the 576 patches by the caption-level map, then either progressively gray out the
    top-k% (deletion) or progressively reveal the top-k% from an all-gray image (insertion),
    regenerate the caption at each step, and score it AGAINST THE MODEL'S OWN UNPERTURBED
    CAPTION. Scoring against the model's own output, not the ground truth, is deliberate: the
    question is "how much of this specific output did this specific region cause", which is
    independent of whether the output was correct.
      * a faithful map => deletion curve falls FAST (low AUC) and insertion rises FAST (high AUC)
      * the whole thing is meaningless unless it beats a RANDOM ranking, so a seeded random
        ranking is run under identical conditions and reported next to it. `last_layer` (raw
        layer-11 cross attention) is run too, so the choice of rollout over raw attention is
        decided by measurement rather than by assertion.
    Gray fill uses ImageSpec.pad_color -- the uint8 colour that normalizes to ~0 -- so the
    perturbation removes image content rather than injecting a high-contrast black rectangle
    that is itself a strong signal.

    Metrics per perturbed caption (all vs. the unperturbed caption):
      * CIDEr, computed ONCE over the pooled corpus of every (example, ranking, direction, k)
        pair, so a single IDF is shared and the numbers are mutually comparable.
      * token-F1: corpus-independent, so it cannot be distorted by the pooled-IDF choice.
      * bencana_same / caption_identical: does the actual report content survive.

TEST 2 -- HALF OCCLUSION (the domain-specific version of the same question)
    Gray out the ENTIRE post-disaster half, regenerate, and check whether the damage-bearing
    sections change; then the mirror control (gray out the pre half only). If the model's
    damage words are genuinely read off the post image, killing the post half must move
    BENCANA / BANGUNAN much more than killing the pre half. `get_bencana` and
    `building_no_damage` are the exact helpers from compare_beam_vs_greedy.py, so the
    diagnostic is the same one used for the Fase-4/5 bias analysis.

Usage:
    python scripts/xai_faithfulness_check.py --mode occlusion --n 60
    python scripts/xai_faithfulness_check.py --mode delins --n 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.xai_attention_rollout import (  # noqa: E402
    XaiEngine, OUT_DIR, EXAMPLES_DIR, GRID, N_PATCHES, PRE_COLS,
    get_bencana, building_no_damage, get_section,
    mask_patches, select_examples, load_predictions, token_selectivity,
)
from scripts.dataset import DisasterCaptionDataset  # noqa: E402

DEL_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RANKINGS = ["rollout", "last_layer", "random"]
SECTIONS = ["BENCANA", "BANGUNAN", "JALAN", "VEGETASI", "BADAN_AIR", "PERTANIAN", "KESIMPULAN"]


# ======================================================================================
# text comparison helpers
# ======================================================================================

def token_f1(a: str, b: str) -> float:
    """Bag-of-words F1 between two captions. Corpus-independent, so it is immune to the
    pooled-IDF caveat that CIDEr carries."""
    from collections import Counter
    ta, tb = Counter(a.lower().split()), Counter(b.lower().split())
    overlap = sum((ta & tb).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(tb.values()), overlap / sum(ta.values())
    return 2 * p * r / (p + r)


def section_changed(ref: str, hyp: str, name: str) -> bool:
    return (get_section(ref, name) or "") != (get_section(hyp, name) or "")


def auc(xs, ys) -> float:
    return float(np.trapezoid(np.asarray(ys, dtype=float), np.asarray(xs, dtype=float)))


# ======================================================================================
# TEST 2 -- half occlusion
# ======================================================================================

def mode_occlusion(args):
    eng = XaiEngine(device=args.device)
    ds = DisasterCaptionDataset("test", bundle=eng.bundle, drop_unreadable=True)
    rows = load_predictions()
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(rows, min(args.n, len(rows))), key=lambda r: r["index"])

    post_patches = [r * GRID + c for r in range(GRID) for c in range(PRE_COLS, GRID)]
    pre_patches = [r * GRID + c for r in range(GRID) for c in range(0, PRE_COLS)]

    per_example, t0 = [], time.time()
    for k, rec in enumerate(sample):
        i = rec["index"]
        pv = ds[i]["pixel_values"].unsqueeze(0).to(args.device)
        variants = {
            "intact": pv,
            "post_masked": mask_patches(pv, post_patches, eng.spec),
            "pre_masked": mask_patches(pv, pre_patches, eng.spec),
        }
        # one batched generate for the 3 variants: greedy decoding is sequential, so the
        # wall-clock cost is set by the longest caption, not by the batch size
        stacked = torch.cat(list(variants.values()), dim=0)
        outs = eng.generate(stacked)
        caps = {name: eng.codec.decode(outs[j]) for j, name in enumerate(variants)}
        row = {"index": i, "gt_bencana": get_bencana(rec["ground_truth_id"]), "captions": caps}
        base = caps["intact"]
        for name in ("post_masked", "pre_masked"):
            c = caps[name]
            row[name] = {
                "caption_identical": c == base,
                "token_f1_vs_intact": token_f1(base, c),
                "bencana_intact": get_bencana(base), "bencana_masked": get_bencana(c),
                "bencana_changed": get_bencana(base) != get_bencana(c),
                "no_damage_intact": building_no_damage(base),
                "no_damage_masked": building_no_damage(c),
                "no_damage_flipped": building_no_damage(base) != building_no_damage(c),
                "sections_changed": {s: section_changed(base, c, s) for s in SECTIONS},
            }
        per_example.append(row)
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(sample)} ({el/(k+1):.1f}s/ex, "
                  f"ETA {(len(sample)-k-1)*el/(k+1)/60:.1f} min)", flush=True)

    summary = {}
    for name in ("post_masked", "pre_masked"):
        rs = [r[name] for r in per_example]
        summary[name] = {
            "n": len(rs),
            "caption_identical_rate": float(np.mean([r["caption_identical"] for r in rs])),
            "mean_token_f1_vs_intact": float(np.mean([r["token_f1_vs_intact"] for r in rs])),
            "bencana_changed_rate": float(np.mean([r["bencana_changed"] for r in rs])),
            "no_damage_flipped_rate": float(np.mean([r["no_damage_flipped"] for r in rs])),
            "section_changed_rate": {
                s: float(np.mean([r["sections_changed"][s] for r in rs])) for s in SECTIONS},
        }
    summary["post_minus_pre"] = {
        "bencana_changed_rate": summary["post_masked"]["bencana_changed_rate"]
                                - summary["pre_masked"]["bencana_changed_rate"],
        "mean_token_f1": summary["post_masked"]["mean_token_f1_vs_intact"]
                         - summary["pre_masked"]["mean_token_f1_vs_intact"],
        "interpretation": ("negative token_f1 difference == masking the POST half damages the "
                           "output MORE than masking the PRE half, which is what a model that "
                           "actually reads damage off the post image should show"),
    }
    out = {"n": len(per_example), "seed": args.seed, "summary": summary,
           "per_example": per_example}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "occlusion_test.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {p}")


# ======================================================================================
# TEST 1 -- deletion / insertion
# ======================================================================================

def mode_delins(args):
    eng = XaiEngine(device=args.device)
    ds = DisasterCaptionDataset("test", bundle=eng.bundle, drop_unreadable=True)
    picked = select_examples(args.n, seed=args.seed)
    print(f"[select] {[r['index'] for r in picked]}")
    rng = np.random.default_rng(args.seed)

    all_patches = np.arange(N_PATCHES)
    records = []
    base_captions: dict[int, dict] = {}
    pooled_gts, pooled_res = {}, {}
    t0 = time.time()

    for e, rec in enumerate(picked):
        i = rec["index"]
        pv = ds[i]["pixel_values"].unsqueeze(0).to(args.device)
        tm = eng.explain(pv, index=i, variants=("rollout", "last_layer"))
        base_caption = tm.caption
        content = [j for j, c in enumerate(tm.is_content) if c]
        if not content:
            continue

        orders = {}
        for v in ("rollout", "last_layer"):
            m = tm.maps[v][content].mean(0).numpy()          # caption-level map
            orders[v] = np.argsort(-m)                       # most-attended first
        orders["random"] = rng.permutation(all_patches)

        base_captions[i] = {
            "base_caption": base_caption,
            "token_selectivity_rollout": token_selectivity(tm.maps["rollout"][content]),
            "gt_bencana": get_bencana(rec["ground_truth_id"]),
            "gen_bencana": get_bencana(base_caption),
        }
        for ranking in RANKINGS:
            order = orders[ranking]
            for direction in ("deletion", "insertion"):
                batch, keys = [], []
                for frac in DEL_FRACTIONS:
                    n_top = int(round(frac * N_PATCHES))
                    top = order[:n_top]
                    if direction == "deletion":
                        masked = list(top)
                    else:                                     # insertion: keep only the top
                        keep = set(int(x) for x in top)
                        masked = [p for p in all_patches if p not in keep]
                    batch.append(mask_patches(pv, masked, eng.spec) if masked else pv)
                    keys.append(f"{i}|{ranking}|{direction}|{frac:.1f}")
                outs = eng.generate(torch.cat(batch, dim=0))
                for j, (frac, key) in enumerate(zip(DEL_FRACTIONS, keys)):
                    cap = eng.codec.decode(outs[j])
                    pooled_gts[key] = [base_caption]
                    pooled_res[key] = [cap]
                    records.append({
                        "index": i, "ranking": ranking, "direction": direction, "frac": frac,
                        "key": key, "caption": cap,
                        "token_f1": token_f1(base_caption, cap),
                        "caption_identical": cap == base_caption,
                        "bencana_same": get_bencana(base_caption) == get_bencana(cap),
                    })
        el = time.time() - t0
        print(f"  example {e+1}/{len(picked)} (idx {i}) done "
              f"[{el/(e+1)/60:.1f} min/ex, ETA {(len(picked)-e-1)*el/(e+1)/60:.1f} min]",
              flush=True)

    # CIDEr once, over the pooled corpus -> one shared IDF for every point on every curve
    print("[metrics] pooled CIDEr over "
          f"{len(pooled_gts)} (example, ranking, direction, k) pairs")
    from scripts.compute_metrics import coco_style_score
    sc = coco_style_score(pooled_gts, pooled_res)
    cider = dict(zip(sc["ids"], sc["cider_per_item"]))
    for r in records:
        r["cider_vs_intact"] = float(cider.get(r["key"], float("nan")))

    curves, aucs = {}, {}
    for ranking in RANKINGS:
        curves[ranking], aucs[ranking] = {}, {}
        for direction in ("deletion", "insertion"):
            pts = []
            for frac in DEL_FRACTIONS:
                sub = [r for r in records
                       if r["ranking"] == ranking and r["direction"] == direction
                       and abs(r["frac"] - frac) < 1e-9]
                pts.append({
                    "frac": frac, "n": len(sub),
                    "cider": float(np.mean([r["cider_vs_intact"] for r in sub])),
                    "token_f1": float(np.mean([r["token_f1"] for r in sub])),
                    "bencana_same_rate": float(np.mean([r["bencana_same"] for r in sub])),
                    "caption_identical_rate": float(np.mean([r["caption_identical"] for r in sub])),
                })
            curves[ranking][direction] = pts
            xs = [p["frac"] for p in pts]
            aucs[ranking][direction] = {
                "cider_auc": auc(xs, [p["cider"] for p in pts]),
                "token_f1_auc": auc(xs, [p["token_f1"] for p in pts]),
                "bencana_same_auc": auc(xs, [p["bencana_same_rate"] for p in pts]),
            }

    verdict = {}
    for ranking in ("rollout", "last_layer"):
        d = aucs[ranking]["deletion"]["token_f1_auc"] - aucs["random"]["deletion"]["token_f1_auc"]
        ins = aucs[ranking]["insertion"]["token_f1_auc"] - aucs["random"]["insertion"]["token_f1_auc"]
        verdict[ranking] = {
            "deletion_auc_minus_random": d,     # want NEGATIVE (falls faster than random)
            "insertion_auc_minus_random": ins,  # want POSITIVE (rises faster than random)
            "beats_random_on_deletion": d < 0,
            "beats_random_on_insertion": ins > 0,
        }

    out = {
        "n_examples": len(picked), "seed": args.seed,
        "example_indices": [r["index"] for r in picked],
        "fractions": DEL_FRACTIONS, "rankings": RANKINGS,
        "n_generations": len(records),
        "note": ("all scores compare a perturbed caption to the SAME model's caption on the "
                 "UNPERTURBED image; CIDEr shares one pooled IDF across every point"),
        # frac=0 deletion (and frac=1 insertion) re-generate the UNPERTURBED image inside a
        # batch of 11; if batching perturbed decoding these would not be identical, so this
        # is a free control on the whole batched-generation shortcut
        "batched_generation_control_identical_rate": float(np.mean(
            [r["caption_identical"] for r in records
             if (r["direction"] == "deletion" and r["frac"] == 0.0)
             or (r["direction"] == "insertion" and r["frac"] == 1.0)])),
        "base_captions": base_captions,
        "curves": curves, "auc": aucs, "verdict_vs_random": verdict,
        "records": records,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "deletion_insertion_test.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _plot_curves(curves, os.path.join(OUT_DIR, "deletion_insertion_curves.png"))
    print(json.dumps({"auc": aucs, "verdict_vs_random": verdict}, indent=2))
    print(f"[done] {p}")


def per_example_auc(records, metric="token_f1"):
    """AUC of the perturbation curve, computed SEPARATELY per example.

    The pooled curve in `curves` averages first and integrates second, which gives one number
    with no error bar -- useless for deciding whether "rollout beats random" is real or noise
    at n=12. Per-example AUCs are paired across rankings (same image, same base caption), so
    they feed straight into the project's standard paired_bootstrap.
    """
    out: dict[tuple[str, str], dict[int, float]] = {}
    for ranking in RANKINGS:
        for direction in ("deletion", "insertion"):
            per_idx = {}
            idxs = sorted({r["index"] for r in records})
            for i in idxs:
                sub = sorted([r for r in records
                              if r["index"] == i and r["ranking"] == ranking
                              and r["direction"] == direction], key=lambda r: r["frac"])
                per_idx[i] = auc([r["frac"] for r in sub], [r[metric] for r in sub])
            out[(ranking, direction)] = per_idx
    return out


def significance(records, metric="token_f1"):
    from scripts.compute_metrics import paired_bootstrap
    pe = per_example_auc(records, metric)
    idxs = sorted(pe[("random", "deletion")].keys())
    res = {}
    for ranking in ("rollout", "last_layer"):
        res[ranking] = {}
        for direction in ("deletion", "insertion"):
            a = [pe[(ranking, direction)][i] for i in idxs]
            b = [pe[("random", direction)][i] for i in idxs]
            bs = paired_bootstrap(a, b)
            res[ranking][direction] = {
                f"{metric}_auc_mean_{ranking}": float(np.mean(a)),
                f"{metric}_auc_mean_random": float(np.mean(b)),
                "paired_bootstrap_attention_minus_random": bs,
                "direction_of_faithfulness": (
                    "attention is more faithful than random if this diff is NEGATIVE"
                    if direction == "deletion" else
                    "attention is more faithful than random if this diff is POSITIVE"),
            }
    res["_per_example_auc"] = {f"{r}|{d}": {str(k): v for k, v in pe[(r, d)].items()}
                               for (r, d) in pe}
    return res


def mode_reanalyze(args):
    """Recompute curves/AUC/significance from the ALREADY GENERATED captions in
    results/xai/deletion_insertion_test.json -- no model, no new generations, so the numbers
    are provably about the same 792 captions the first run produced."""
    p = os.path.join(OUT_DIR, "deletion_insertion_test.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    recs = d["records"]
    sig = {m: significance(recs, m) for m in ("token_f1", "cider_vs_intact", "bencana_same")}
    d["significance_paired_bootstrap"] = sig
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    slim = {m: {r: {dd: {k: v for k, v in sig[m][r][dd].items() if k != "_per_example_auc"}
                    for dd in ("deletion", "insertion")}
                for r in ("rollout", "last_layer")} for m in sig}
    print(json.dumps(slim, indent=2))
    print(f"[done] significance added to {p}")


def mode_summary(args):
    """Assemble the headline verdicts next to the heatmaps in results/xai_examples/, so a
    reader who opens the picture folder cannot miss whether the pictures were validated."""
    with open(os.path.join(OUT_DIR, "deletion_insertion_test.json"), encoding="utf-8") as f:
        di = json.load(f)
    with open(os.path.join(OUT_DIR, "occlusion_test.json"), encoding="utf-8") as f:
        oc = json.load(f)
    with open(os.path.join(OUT_DIR, "xai_selftest.json"), encoding="utf-8") as f:
        st = json.load(f)
    agg_p = os.path.join(OUT_DIR, "section_attention_aggregate.json")
    agg = json.load(open(agg_p, encoding="utf-8")) if os.path.exists(agg_p) else None

    sig = di["significance_paired_bootstrap"]["token_f1"]
    out = {
        "READ_THIS_FIRST": (
            "These heatmaps are NOT validated as faithful token-level explanations. The "
            "numbers below are the evidence. See docs/design_decisions.md section 6."),
        "plumbing": {
            "attn_implementation": st["attn_implementation"],
            "eager_was_required": False,
            "toplevel_output_attentions_returns_cross_attentions": not st[
                "toplevel_output_attentions"]["cross_attentions_is_none"],
            "teacher_forced_token_alignment_match": st["example_a"][
                "teacher_forced_argmax_match_vs_greedy"],
        },
        "token_selectivity_mean_pairwise_map_correlation": st["token_selectivity"],
        "deletion_insertion": {
            "n_examples": di["n_examples"], "n_generations": di["n_generations"],
            "example_indices": di["example_indices"],
            "batched_generation_control_identical_rate": di[
                "batched_generation_control_identical_rate"],
            "token_f1_auc": {
                "rollout_deletion": sig["rollout"]["deletion"]["token_f1_auc_mean_rollout"],
                "last_layer_deletion": sig["last_layer"]["deletion"]["token_f1_auc_mean_last_layer"],
                "random_deletion": sig["rollout"]["deletion"]["token_f1_auc_mean_random"],
                "rollout_insertion": sig["rollout"]["insertion"]["token_f1_auc_mean_rollout"],
                "last_layer_insertion": sig["last_layer"]["insertion"]["token_f1_auc_mean_last_layer"],
                "random_insertion": sig["rollout"]["insertion"]["token_f1_auc_mean_random"],
            },
            "paired_bootstrap_vs_random": {
                f"{r}_{d}": sig[r][d]["paired_bootstrap_attention_minus_random"]
                for r in ("rollout", "last_layer") for d in ("deletion", "insertion")},
        },
        "half_occlusion": oc["summary"],
    }
    if agg:
        # is the overall post-half preference distinguishable from an even 50/50 split?
        # bootstrap over EXAMPLES (the unit of independence), not over tokens.
        post = np.array([e["sections"]["rollout"]["__ALL__"]["post"] for e in agg["per_example"]])
        rng = np.random.default_rng(0)
        boots = np.array([post[rng.integers(0, len(post), len(post))].mean()
                          for _ in range(10000)])
        out["section_pre_post_attention"] = {
            "n_examples": agg["n_scored"],
            "overall_post_fraction_mean": float(post.mean()),
            "overall_post_fraction_ci95": [float(np.percentile(boots, 2.5)),
                                           float(np.percentile(boots, 97.5))],
            "frac_bootstrap_above_0.50": float((boots > 0.5).mean()),
            "rollout": agg["summary_by_variant"]["rollout"],
        }
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    p = os.path.join(EXAMPLES_DIR, "faithfulness_summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] {p}")


def _plot_curves(curves, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for col, metric in enumerate(("token_f1", "cider")):
        for row, direction in enumerate(("deletion", "insertion")):
            ax = axes[row][col]
            for ranking in RANKINGS:
                pts = curves[ranking][direction]
                ax.plot([p["frac"] for p in pts], [p[metric] for p in pts],
                        marker="o", ms=3, label=ranking)
            ax.set_title(f"{direction} -- {metric} vs unperturbed caption", fontsize=9)
            ax.set_xlabel("fraction of patches "
                          + ("removed (most-attended first)" if direction == "deletion"
                             else "revealed (most-attended first)"), fontsize=8)
            ax.set_ylabel(metric, fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle("faithfulness: attention-ranked vs random patch perturbation\n"
                 "(a faithful map falls faster than random on deletion, rises faster on insertion)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[plot] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["occlusion", "delins", "reanalyze", "summary"],
                    required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    {"occlusion": mode_occlusion, "delins": mode_delins,
     "reanalyze": mode_reanalyze, "summary": mode_summary}[args.mode](args)


if __name__ == "__main__":
    main()
