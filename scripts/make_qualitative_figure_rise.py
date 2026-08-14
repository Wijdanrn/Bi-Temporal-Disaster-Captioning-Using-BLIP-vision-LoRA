"""
Qualitative 4-column figure for the paper: pre/post composite + RISE heatmap (BANGUNAN
section only) + Ground-Truth/Prediction text boxes, for 4 curated examples drawn from the
12-example xAI set.

SOURCES (see module docstrings for why each is authoritative -- nothing here re-derives
logic that already exists elsewhere in the repo):
  - results/xai_examples/manifest.json      -> the ONLY valid source of the 12 candidate
                                                indices (RISE was only computed, and can only
                                                be recomputed, for these 12).
  - results/predictions_test_vision.jsonl   -> ground_truth_id / generated_caption text.
  - results/xai_rise/deletion_insertion_test.json,
    results/xai_rise/faithfulness_summary.json -> the ONLY source for the disclaimer's
                                                significance numbers and RISE run metadata.
  - scripts/xai_attention_rollout.py        -> XaiEngine, compute_rise_map(), GRID,
                                                N_PATCHES, PRE_COLS, get_section(), the
                                                token/section span logic (TokenMaps /
                                                XaiEngine._sections). Reused, not rewritten.

A SILENT BUG THAT WAS CAUGHT WHILE BUILDING THIS SCRIPT (documented here on purpose):
`xai_attention_rollout.get_section()` / `compare_beam_vs_greedy.get_section()` is
`re.search(rf"{name}:\s*(.+)", text)` with NO `re.DOTALL` and no explicit stop condition
other than "rest of the line". That is safe for `generated_caption`, whose section headers
are separated by real "\n" characters (verified below) -- `.+` naturally stops at the first
newline. It is NOT safe for `ground_truth_id`: verified by direct inspection that this field
has NO real newlines between section headers (e.g. index 471's ground_truth_id is
"...BANGUNAN: Tidak ada kerusakan ... perbandingan. JALAN: Tidak ada gangguan..." all on one
line), so the naive regex silently swallows every subsequent section into "BANGUNAN". This
was caught by literally printing `repr()` of the raw field before writing any figure code --
it renders a perfectly plausible figure with the WRONG text if not caught. This script uses
`get_section_bounded()` below (bounded by the NEXT known section header, independent of
whether a real newline is present) for ALL text shown in the figure. The token-level
section-span logic used for the RISE heatmap AGGREGATION (which patches count as "BANGUNAN")
is untouched and reused verbatim from XaiEngine -- only the CAPTION-TEXT-FOR-DISPLAY
extraction needed a bounded replacement.

Usage:
    python scripts/make_qualitative_figure_rise.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_model import SECTION_HEADERS  # noqa: E402
from scripts.xai_attention_rollout import (  # noqa: E402
    XaiEngine, compute_rise_map, composite_pil, _upsample,
    GRID, N_PATCHES, PRE_COLS,
    RISE_N_MASKS_DEFAULT, RISE_MASK_GRID_DEFAULT, RISE_P_KEEP_DEFAULT, RISE_BATCH_DEFAULT,
)

MANIFEST_PATH = os.path.join(ROOT, "results", "xai_examples", "manifest.json")
PRED_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
DELINS_PATH = os.path.join(ROOT, "results", "xai_rise", "deletion_insertion_test.json")
FAITH_PATH = os.path.join(ROOT, "results", "xai_rise", "faithfulness_summary.json")
FIG_DIR = os.path.join(ROOT, "docs", "figures")

OUT_PNG = os.path.join(FIG_DIR, "qualitative_comparison_rise.png")
OUT_PDF = os.path.join(FIG_DIR, "qualitative_comparison_rise.pdf")
OUT_SELECTION_JSON = os.path.join(FIG_DIR, "qualitative_comparison_rise_selection.json")

HIGHLIGHT_SECTION = "BANGUNAN"
RISE_SEED = 0

# Manual override for panel (b) "Akurat": auto-selection among the 12 curated xAI examples
# found NO genuine match (see select_four()'s b_candidates == [] branch) and fell back to the
# closest-but-still-wrong candidate (idx 2028) with an explicit caveat. idx 1061 is a genuinely
# accurate example (GT: real flood damage to structures; generated: model also describes
# flooding) found by scanning the FULL 2362-item test set
# (scripts/find_accurate_damage_examples.py -> results/accurate_damage_candidates/), picked by
# hand from that gallery. It is NOT part of the original 12-example manifest -- RISE is
# recomputed fresh for it in main() below, same as any other panel. Set to None to fall back to
# the automatic (caveated) selection.
MANUAL_OVERRIDE_B_INDEX: int | None = 1061


# ======================================================================================
# text extraction for DISPLAY (bounded by next known header -- see module docstring)
# ======================================================================================

def get_section_bounded(text: str, name: str) -> str | None:
    """Extract section `name`'s body, bounded by the NEXT SECTION_HEADERS entry, whether or
    not a real '\\n' separates them in the source text. Safe for BOTH ground_truth_id
    (no real newlines between sections) and generated_caption (has them)."""
    others = "|".join(h for h in SECTION_HEADERS if h != name)
    pat = rf"{name}:\s*(.+?)\s*(?:(?:{others}):|$)"
    m = re.search(pat, text, re.DOTALL)
    return m.group(1).strip() if m else None


def all_sections_bounded(text: str) -> dict[str, str]:
    return {h: (get_section_bounded(text, h) or "") for h in SECTION_HEADERS}


NO_DAMAGE_PHRASES = ("tidak ada kerusakan", "tidak menunjukkan kerusakan")


def is_no_damage(bangunan_text: str) -> bool:
    """Reliable ONLY for `generated_caption` text, which is confirmed (by inspecting all 12
    curated examples) to consistently use one of these two canonical boilerplate phrases when
    the model predicts no damage. NOT reliable for `ground_truth_id` text -- see
    GT_NO_DAMAGE_MANUAL below for why."""
    b = (bangunan_text or "").lower()
    return any(p in b for p in NO_DAMAGE_PHRASES)


# Ground-truth BANGUNAN text is free-form (translated from the original DisasterM3 English
# captions), not boilerplate, so a keyword/substring heuristic is NOT safe here -- verified by
# actually trying one first: `is_no_damage()` above, run against ground_truth_id, misses
# paraphrases like idx 81's "tidak mengungkapkan kerusakan yang terlihat" and idx 1944's "tanpa
# kerusakan struktural yang terlihat, runtuh, atau puing-puing" (neither contains the two
# canonical phrases), which would have wrongly excluded them from the "no damage" bucket. The
# opposite failure is just as real: idx 2028's GT describes ACTUAL localized roof damage but
# also contains the clause "Bangunan tetap utuh secara struktural" (structurally intact
# overall) -- a naive "tetap utuh" pattern would have wrongly classified a real-damage example
# as no-damage. Given n=12, every value below was verified by direct reading of the exact
# bounded-extraction text that `print_verification_table()` also prints to console (so it is
# independently checkable, not a black box), rather than trusting a keyword heuristic on
# free-form text for a precision-critical figure.
GT_NO_DAMAGE_MANUAL: dict[int, bool] = {
    81:   True,   # "tidak mengungkapkan kerusakan yang terlihat ... tetap utuh dengan pola spasial yang tidak terganggu"
    380:  True,   # "Semua struktur tampak utuh tanpa tanda-tanda kerusakan atap, runtuh, atau tumpukan puing-puing"
    471:  True,   # "Tidak ada kerusakan struktural yang terlihat pada bangunan ..."
    514:  False,  # REAL damage: "kerusakan luas bangunan ... hancur sepenuhnya ... dampak struktural yang signifikan"
    563:  True,   # "Tidak ada kerusakan bangunan yang terlihat ... Struktur di kanan atas tetap utuh"
    673:  True,   # "Tidak ada bukti kerusakan bangunan yang diamati. Struktur ... tetap utuh ..."
    1234: True,   # "Bangunan tetap secara struktural utuh tanpa dampak yang terlihat atau kerusakan yang diamati"
    1482: True,   # "Tidak ada kerusakan struktural yang terlihat pada bangunan ..."
    1514: False,  # REAL damage: "Pengurangan integritas struktural ... hilangnya beberapa bangunan ... akumulasi puing"
    1944: True,   # "Semua bangunan ... tampak utuh tanpa kerusakan struktural yang terlihat, runtuh, atau puing-puing"
    2028: False,  # REAL (localized) damage: "Kerusakan atap dan puing-puing permukaan terlihat ..." despite "tetap utuh secara struktural"
    2344: True,   # "Tidak ada kerusakan struktural yang terlihat di lingkungan yang dibangun. Bangunan tampak utuh ..."
}

# Manually-verified severity ranking of the 3 GT-has-real-damage examples (higher = more
# severe), used to pick (d)'s most dramatic under-detection case and (b)'s "closest available"
# fallback (the mildest miss, i.e. closest to a near-accurate call). String length is NOT a
# safe severity proxy here (1514's text is shorter than 2028's despite describing WORSE
# damage), so this is a direct qualitative read of the same three texts printed above:
#   514  (rank 3, worst):    "kerusakan luas ... hancur SEPENUHNYA ... dampak struktural yang SIGNIFIKAN" -- widespread total destruction
#   1514 (rank 2, moderate): "hilangnya BEBERAPA BANGUNAN ... akumulasi puing" -- actual building loss
#   2028 (rank 1, mildest):  "Kerusakan atap ... TETAP UTUH SECARA STRUKTURAL dengan HANYA DAMPAK LOKAL" -- localized, non-structural
DAMAGE_SEVERITY_RANK: dict[int, int] = {514: 3, 1514: 2, 2028: 1}


# ======================================================================================
# LANGKAH 1 -- load the 12 curated candidates and classify them
# ======================================================================================

def load_manifest_examples() -> list[dict]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        m = json.load(f)
    return m["examples"]


def load_predictions_for(indices: set[int]) -> dict[int, dict]:
    out = {}
    with open(PRED_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("index") in indices:
                out[r["index"]] = r
    return out


def build_manual_row(index: int, gt_no_damage: bool = False) -> dict:
    """Build a row for an index OUTSIDE the 12-example manifest.json set -- e.g. a genuinely
    accurate example found by scanning the full test set (scripts/find_accurate_damage_examples.py
    -> results/accurate_damage_candidates/) and picked by hand. Reads ONLY
    predictions_test_vision.jsonl (no manifest.json dependency), so this works for any test-set
    index. `gt_no_damage` defaults to False (i.e. "GT indicates real damage") because the only
    reason to manually override a panel is to substitute in a damage example -- the caller is
    expected to have already read the GT text (e.g. via results/accurate_damage_candidates/
    README.md) before picking the index, same standard as GT_NO_DAMAGE_MANUAL above.
    """
    with open(PRED_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("index") == index:
                gt_all = all_sections_bounded(r["ground_truth_id"])
                gen_all = all_sections_bounded(r["generated_caption"])
                gt_bencana = gt_all.get("BENCANA")
                gen_bencana = gen_all.get("BENCANA")
                return {
                    "index": index, "tag": f"idx{index:04d}_manual",
                    "pre_image_path": r["pre_image_path"], "post_image_path": r["post_image_path"],
                    "gt_bencana": gt_bencana, "gen_bencana": gen_bencana,
                    "bencana_correct": (gt_bencana or "").lower() == (gen_bencana or "").lower(),
                    "gt_bangunan": gt_all["BANGUNAN"], "gen_bangunan": gen_all["BANGUNAN"],
                    "gt_all": gt_all, "gen_all": gen_all,
                    "gt_no_damage": gt_no_damage,
                    "gen_no_damage": is_no_damage(gen_all["BANGUNAN"]),
                }
    raise RuntimeError(f"index {index} not found in {PRED_PATH}")


def build_rows() -> list[dict]:
    """One row per manifest example: index, tag, gt/gen BANGUNAN text (bounded), no-damage
    flags. This is the single source of truth for LANGKAH 1's categorization."""
    ex = load_manifest_examples()
    idx_set = {e["index"] for e in ex}
    preds = load_predictions_for(idx_set)
    if set(GT_NO_DAMAGE_MANUAL.keys()) != idx_set:
        raise RuntimeError(
            "GT_NO_DAMAGE_MANUAL's index set no longer matches manifest.json's 12 examples "
            f"(manual={sorted(GT_NO_DAMAGE_MANUAL)} vs manifest={sorted(idx_set)}) -- the "
            "manual ground-truth no-damage table must be re-verified by reading the new "
            "examples' actual text before this script can classify them; refusing to guess.")
    rows = []
    for e in ex:
        i = e["index"]
        p = preds.get(i)
        if p is None:
            raise RuntimeError(f"index {i} from manifest.json missing in {PRED_PATH}")
        gt_all = all_sections_bounded(p["ground_truth_id"])
        gen_all = all_sections_bounded(p["generated_caption"])
        rows.append({
            "index": i, "tag": e["tag"],
            "pre_image_path": e["pre_image_path"], "post_image_path": e["post_image_path"],
            "gt_bencana": e["gt_bencana"], "gen_bencana": e["gen_bencana"],
            "bencana_correct": e["bencana_correct"],
            "gt_bangunan": gt_all["BANGUNAN"], "gen_bangunan": gen_all["BANGUNAN"],
            "gt_all": gt_all, "gen_all": gen_all,
            "gt_no_damage": GT_NO_DAMAGE_MANUAL[i],
            "gen_no_damage": is_no_damage(gen_all["BANGUNAN"]),
        })
    return rows


def print_verification_table(rows: list[dict]) -> None:
    print("=" * 100)
    print("LANGKAH 1 -- all 12 curated candidates (results/xai_examples/manifest.json), "
          "BANGUNAN section GT vs Prediksi (bounded extraction, see module docstring):")
    print("=" * 100)
    for r in rows:
        print(f"\n[idx {r['index']:>4}] tag={r['tag']}  "
              f"BENCANA gt={r['gt_bencana']!r} gen={r['gen_bencana']!r} "
              f"(correct={r['bencana_correct']})")
        print(f"  gt_no_damage={r['gt_no_damage']}  gen_no_damage={r['gen_no_damage']}")
        print(f"  GT  BANGUNAN : {r['gt_bangunan']}")
        print(f"  GEN BANGUNAN : {r['gen_bangunan']}")


def classify(rows: list[dict]) -> dict:
    by_idx = {r["index"]: r for r in rows}

    # a) Tidak Ada Kerusakan (Benar): GT says no damage, prediction agrees.
    a_candidates = [r["index"] for r in rows if r["gt_no_damage"] and r["gen_no_damage"]]

    # damage-in-GT pool (the harder cases)
    damage_gt = [r["index"] for r in rows if not r["gt_no_damage"]]

    # b) Akurat: GT has real damage AND prediction also indicates damage (any non-boilerplate
    #    acknowledgement of damage, i.e. NOT the no-damage phrase).
    b_candidates = [i for i in damage_gt if not by_idx[i]["gen_no_damage"]]

    # d) Under-Deteksi (Bias): GT has real damage but prediction defaults to "no damage".
    d_candidates = [i for i in damage_gt if by_idx[i]["gen_no_damage"]]

    # c) Detail Terlewat: both GT and prediction agree on "no damage" (so NOT itself wrong),
    #    but GT names a specific location / spectral-signature detail the prediction's
    #    boilerplate text drops entirely. Heuristic flag: GT mentions a spatial locator or
    #    "spektral" that the generated text does not.
    LOCATORS = ("kiri", "kanan", "atas", "bawah", "tengah", "pusat", "spektral")
    c_candidates = []
    for i in a_candidates:
        r = by_idx[i]
        gt_l, gen_l = r["gt_bangunan"].lower(), r["gen_bangunan"].lower()
        gt_hits = [w for w in LOCATORS if w in gt_l]
        gen_hits = [w for w in LOCATORS if w in gen_l]
        if gt_hits and not gen_hits:
            c_candidates.append((i, gt_hits))

    return {
        "a_candidates": a_candidates,
        "b_candidates": b_candidates,
        "d_candidates": d_candidates,
        "c_candidates": c_candidates,
        "by_idx": by_idx,
    }


def select_four(classification: dict) -> dict:
    by_idx = classification["by_idx"]
    a_candidates = classification["a_candidates"]
    b_candidates = classification["b_candidates"]
    d_candidates = classification["d_candidates"]
    c_candidates = classification["c_candidates"]

    print("\n" + "=" * 100)
    print("LANGKAH 1 -- classification result:")
    print("=" * 100)
    print(f"  a) Tidak Ada Kerusakan (Benar) candidates: {a_candidates}")
    print(f"  b) Akurat (GT damage + prediction acknowledges damage) candidates: {b_candidates}")
    print(f"  c) Detail Terlewat candidates (idx, matched locator words in GT): {c_candidates}")
    print(f"  d) Under-Deteksi (Bias) candidates (GT damage -> predicted no-damage): {d_candidates}")

    selected = {}

    # a) pick the cleanest short match, preferring a case where BENCANA is ALSO correct so
    # the panel isn't distracted by an unrelated disaster-type error.
    a_pref = [i for i in a_candidates if by_idx[i]["bencana_correct"]] or a_candidates
    a_pick = sorted(a_pref, key=lambda i: len(by_idx[i]["gt_bangunan"]))[0]
    selected["a"] = {"index": a_pick, "category": "Tidak Ada Kerusakan (Benar)", "caveat": None}

    # c) pick the strongest locator-detail case (most locator words dropped)
    if c_candidates:
        c_pick = sorted(c_candidates, key=lambda t: -len(t[1]))[0][0]
    else:
        c_pick = None
    selected["c"] = {"index": c_pick, "category": "Detail Terlewat", "caveat": None}

    # d) pick the most severe/dramatic under-detection (see DAMAGE_SEVERITY_RANK)
    if d_candidates:
        d_pick = sorted(d_candidates, key=lambda i: -DAMAGE_SEVERITY_RANK.get(i, 0))[0]
    else:
        d_pick = None
    selected["d"] = {"index": d_pick, "category": "Under-Deteksi (Bias)", "caveat": None}

    # b) Akurat: genuinely absent among the 12 (verified above: b_candidates == []).
    # Per task instructions: report this plainly, then use the closest available candidate
    # WITH an explicit caveat rather than silently mislabeling it as "accurate".
    if b_candidates:
        b_pick = b_candidates[0]
        selected["b"] = {"index": b_pick, "category": "Akurat", "caveat": None}
    else:
        print("\n" + "!" * 100)
        print("[WARNING] Kategori (b) 'Akurat' TIDAK memiliki contoh yang cocok di antara 12 "
              "index yang tersedia.")
        print("  Semua contoh dengan GT BANGUNAN berkerusakan nyata "
              f"({damage_gt_str(by_idx)}) diprediksi model sebagai 'tidak ada kerusakan' "
              "(under-deteksi total) -- TIDAK ADA satupun contoh di antara 12 data kurasi "
              "di mana model benar mendeteksi & mendeskripsikan kerusakan nyata.")
        print("  Ini konsisten dengan bias yang sudah terdokumentasi di "
              "docs/advisor_discussion_summary.md (rate 'tidak ada kerusakan': 78.5% prediksi "
              "vs 34.2% rate asli di data).")
        remaining = [i for i in d_candidates if i != selected["d"]["index"]]
        # pick the MILDEST remaining miss (lowest DAMAGE_SEVERITY_RANK) as "closest to
        # accurate" -- a small localized miss is a smaller error than missing total
        # destruction, even though both are still complete under-detection failures
        b_pick = (sorted(remaining, key=lambda i: DAMAGE_SEVERITY_RANK.get(i, 0))[0]
                 if remaining else None)
        if b_pick is not None:
            caveat = (
                "TIDAK DITEMUKAN contoh 'Akurat' (deteksi kerusakan nyata yang benar) di "
                "antara 12 data kurasi -- ketiga contoh dengan GT berkerusakan nyata "
                f"(idx {damage_gt_str(by_idx)}) semuanya diprediksi model sebagai 'tidak ada "
                "kerusakan' (under-deteksi total). Kandidat terdekat yang ditampilkan di sini "
                "adalah kasus kerusakan GT paling ringan/lokal yang tetap terlewat -- BUKAN "
                "contoh akurat, ditampilkan secara transparan sebagai bukti kegagalan "
                "kategori ini pada 12 data kurasi (n kecil, tidak mewakili seluruh test set)."
            )
            print(f"  -> Kandidat terdekat dipakai untuk panel (b), idx {b_pick}, dengan "
                  f"caveat eksplisit di judul kolom dan di JSON metadata.")
        else:
            caveat = ("TIDAK DITEMUKAN contoh 'Akurat' ATAU kandidat terdekat di antara 12 "
                      "data kurasi.")
            print("  -> TIDAK ADA kandidat terdekat tersisa (semua contoh under-deteksi sudah "
                  "dipakai di panel (d)). Panel (b) TIDAK dapat diisi dari 12 data ini.")
        print("!" * 100)
        selected["b"] = {"index": b_pick, "category": "Akurat", "caveat": caveat}

    print("\n" + "=" * 100)
    print("FINAL SELECTION (4 panels):")
    print("=" * 100)
    for key in ("a", "b", "c", "d"):
        sel = selected[key]
        i = sel["index"]
        if i is None:
            print(f"  ({key}) {sel['category']}: NO EXAMPLE AVAILABLE")
            continue
        r = by_idx[i]
        print(f"  ({key}) {sel['category']}  idx={i}  tag={r['tag']}")
        print(f"       GT  BANGUNAN: {r['gt_bangunan']}")
        print(f"       GEN BANGUNAN: {r['gen_bangunan']}")
        if sel["caveat"]:
            print(f"       CAVEAT: {sel['caveat']}")
    return selected


def damage_gt_str(by_idx: dict) -> str:
    dmg = [i for i, r in by_idx.items() if not r["gt_no_damage"]]
    return ", ".join(str(i) for i in sorted(dmg))


# ======================================================================================
# LANGKAH 2 -- recompute RISE per selected example, aggregate over BANGUNAN tokens only
# ======================================================================================

def compute_bangunan_heatmap(engine: XaiEngine, rec: dict, index: int) -> tuple[np.ndarray, dict]:
    """Recompute RISE for ONE example, aggregate to a single 24x24 map over BANGUNAN
    content tokens only, upsample to 384x384. Returns (heat_384x384_normalized_0_1, meta).

    Token/section spans come from XaiEngine.explain() (TokenMaps.section_of_token /
    .is_content), i.e. the SAME span logic already used everywhere else in this project for
    per-section aggregation -- not recomputed from scratch here.
    """
    img = composite_pil(engine, rec)
    pv = engine.spec.to_tensor(img).unsqueeze(0).to(engine.device)

    t0 = time.time()
    tm = engine.explain(pv, index=index)
    explain_s = time.time() - t0

    target_ids = [engine.dec_id] + tm.emitted_ids
    t0 = time.time()
    rise_map = compute_rise_map(
        engine, rec,
        n_masks=RISE_N_MASKS_DEFAULT, mask_grid=RISE_MASK_GRID_DEFAULT,
        p_keep=RISE_P_KEEP_DEFAULT, batch_size=RISE_BATCH_DEFAULT,
        seed=RISE_SEED, target_ids=target_ids,
    )  # (n_emitted, N_PATCHES) unnormalized signed importance, CPU
    rise_s = time.time() - t0

    if rise_map.shape[0] != len(tm.emitted_ids):
        raise RuntimeError(
            f"idx {index}: RISE map has {rise_map.shape[0]} rows but TokenMaps has "
            f"{len(tm.emitted_ids)} emitted tokens -- token alignment broken, refusing to "
            f"aggregate (this is exactly the failure mode compute_rise_map's docstring warns "
            f"about for the text-round-trip path; target_ids was passed explicitly so this "
            f"should not happen).")

    bangunan_idx = [j for j, (s, c) in enumerate(zip(tm.section_of_token, tm.is_content))
                    if c and s == HIGHLIGHT_SECTION]
    if not bangunan_idx:
        raise RuntimeError(f"idx {index}: no content tokens tagged {HIGHLIGHT_SECTION!r} in "
                           f"this run's greedy regeneration -- cannot build a section heatmap.")
    bangunan_map = rise_map[bangunan_idx].mean(0)  # (N_PATCHES,)

    heat = _upsample(bangunan_map)  # (384, 384) numpy
    heat_norm = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)

    meta = {
        "index": index,
        "n_emitted_tokens": len(tm.emitted_ids),
        "n_bangunan_content_tokens": len(bangunan_idx),
        "teacher_forced_argmax_match_vs_greedy": tm.teacher_forced_match,
        "explain_seconds": round(explain_s, 2),
        "rise_seconds": round(rise_s, 2),
        "live_regenerated_caption": tm.caption,
    }
    print(f"  [rise] idx {index}: {len(bangunan_idx)}/{len(tm.emitted_ids)} tokens are "
          f"BANGUNAN-content, RISE took {rise_s:.1f}s (explain {explain_s:.1f}s)")
    return heat_norm, meta, np.asarray(img)


def flatten_overlay(base_rgb: np.ndarray, heat01: np.ndarray, alpha: float,
                    cmap_name: str = "jet") -> np.ndarray:
    """Pre-blend the heatmap onto the base image in plain numpy RGB space, instead of
    stacking two `imshow()` layers (opaque base + alpha-transparent heatmap).

    Why: with two stacked imshow layers, the PNG (Agg) backend blends the alpha directly into
    solid raster pixels at render time, but the PDF (vector) backend stores the transparent
    heatmap as a SEPARATE image with a soft mask -- the actual alpha compositing then happens
    inside whatever PDF viewer/renderer opens the file, not inside matplotlib. Different
    viewers composite/color-manage that soft mask differently, which is why the same file can
    look visibly less saturated in PDF than the PNG. Blending here, in numpy, before the image
    ever reaches a plotting backend removes that ambiguity: both formats end up embedding the
    exact same already-composited RGB pixels, so there is nothing left for a renderer to
    reinterpret.
    """
    cmap = matplotlib.colormaps[cmap_name]
    heat_rgb = np.asarray(cmap(heat01))[..., :3] * 255.0  # (H, W, 3), 0..255
    blended = alpha * heat_rgb + (1.0 - alpha) * base_rgb.astype(np.float64)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ======================================================================================
# LANGKAH 2 (continued) -- text panel rendering (dashed box + red underline on BANGUNAN)
# ======================================================================================

CONTEXT_ORDER = ["JALAN", "VEGETASI", "BADAN_AIR", "PERTANIAN", "KESIMPULAN"]


def render_text_panel(ax, sections: dict, role_label: str, box_edge_color: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.985
    if role_label:
        ax.text(0.0, y, role_label, fontsize=6.8, fontweight="bold", va="top", ha="left",
                transform=ax.transAxes, color="black")
        y -= 0.075

    bencana_val = sections.get("BENCANA", "") or "-"
    ax.text(0.0, y, f"BENCANA: {bencana_val}", fontsize=5.3, color="0.42", va="top",
            ha="left", transform=ax.transAxes, style="italic")
    # 0.095 (not the tighter 0.075 used elsewhere in this function) -- box_top below is only
    # 0.028 above THIS y, so the gap between the BENCANA line above and the box's top edge is
    # effectively (this offset - 0.028). At small axis heights (compact rows) 0.075 was NOT
    # enough to clear a 5.3pt italic line's actual rendered height (descenders etc.), so the
    # box's dashed top border visibly cut through the BENCANA text -- caught by rendering and
    # cropping the figure at high zoom, not visible at thumbnail scale.
    y -= 0.095

    # -- highlighted BANGUNAN block --
    box_top = y + 0.028
    ax.text(0.015, y, f"{HIGHLIGHT_SECTION}:", fontsize=6.0, fontweight="bold", color="black",
            va="top", ha="left", transform=ax.transAxes)
    y -= 0.062
    body = sections.get(HIGHLIGHT_SECTION) or "(tidak ditemukan)"
    lines = textwrap.wrap(body, width=34) or [""]
    line_h = 0.062
    for ln in lines:
        ax.text(0.03, y, ln, fontsize=5.8, va="top", ha="left",
                transform=ax.transAxes)
        y -= line_h
    box_bottom = y + line_h * 0.35

    rect = FancyBboxPatch((0.0, box_bottom), 0.99, box_top - box_bottom,
                          boxstyle="round,pad=0.006,rounding_size=0.01",
                          linewidth=1.1, linestyle=(0, (3, 2)), edgecolor=box_edge_color,
                          facecolor=box_edge_color, alpha=0.07, transform=ax.transAxes,
                          zorder=1)
    ax.add_patch(rect)
    rect_border = FancyBboxPatch((0.0, box_bottom), 0.99, box_top - box_bottom,
                                 boxstyle="round,pad=0.006,rounding_size=0.01",
                                 linewidth=1.1, linestyle=(0, (3, 2)), edgecolor=box_edge_color,
                                 facecolor="none", transform=ax.transAxes, zorder=3)
    ax.add_patch(rect_border)

    underline_y = box_bottom - 0.012
    ax.add_line(Line2D([0.0, 0.99], [underline_y, underline_y], color="red", linewidth=1.6,
                       transform=ax.transAxes, zorder=4))

    y = underline_y - 0.05

    # -- dimmed context: remaining sections, small & gray, truncated to fit --
    ctx_parts = []
    for h in CONTEXT_ORDER:
        v = sections.get(h)
        if v:
            ctx_parts.append(f"{h}: {v}")
    ctx_text = "  ".join(ctx_parts)
    max_chars = 260
    if len(ctx_text) > max_chars:
        ctx_text = ctx_text[:max_chars].rsplit(" ", 1)[0] + " (...)"
    ctx_lines = textwrap.wrap(ctx_text, width=42)
    for ln in ctx_lines:
        if y < 0.02:
            ax.text(0.0, y, "(...)", fontsize=4.6, color="0.6", va="top", ha="left",
                    transform=ax.transAxes)
            break
        ax.text(0.0, y, ln, fontsize=4.6, color="0.58", va="top", ha="left",
                transform=ax.transAxes)
        y -= 0.045


# ======================================================================================
# main
# ======================================================================================

def load_rise_run_metadata() -> dict:
    with open(DELINS_PATH, encoding="utf-8") as f:
        delins = json.load(f)
    rc = delins["rise_config"]
    n_masks = rc["n_masks"]
    mean_s = rc["mean_seconds_per_example"]
    p_del = delins["significance_paired_bootstrap"]["token_f1"]["rise"]["deletion"][
        "paired_bootstrap_attention_minus_random"]["two_sided_p_value_approx"]
    p_ins = delins["significance_paired_bootstrap"]["token_f1"]["rise"]["insertion"][
        "paired_bootstrap_attention_minus_random"]["two_sided_p_value_approx"]
    with open(FAITH_PATH, encoding="utf-8") as f:
        faith = json.load(f)
    n_examples = faith["table"]["rise"]["n_examples"]
    return {
        "n_masks": n_masks, "mean_seconds_per_example": mean_s,
        "p_deletion": p_del, "p_insertion": p_ins, "n_examples": n_examples,
    }


def main():
    import matplotlib.pyplot as plt

    rows = build_rows()
    print_verification_table(rows)
    classification = classify(rows)
    selected = select_four(classification)

    if MANUAL_OVERRIDE_B_INDEX is not None:
        selected["b"] = {"index": MANUAL_OVERRIDE_B_INDEX, "category": "Akurat", "caveat": None}
        print(f"\n[override] panel (b) manually set to idx {MANUAL_OVERRIDE_B_INDEX} -- "
              f"user-picked genuine match from results/accurate_damage_candidates/, outside "
              f"the 12-example manifest. RISE recomputed fresh for it below.")

    run_meta = load_rise_run_metadata()
    print("\n" + "=" * 100)
    print("RISE run metadata (verified from results/xai_rise/deletion_insertion_test.json / "
          "faithfulness_summary.json):")
    print(f"  n_masks={run_meta['n_masks']}  mean_seconds_per_example="
          f"{run_meta['mean_seconds_per_example']:.4f}")
    print(f"  deletion p (rise vs random, token_f1, paired bootstrap) = "
          f"{run_meta['p_deletion']:.4f}")
    print(f"  insertion p (rise vs random, token_f1, paired bootstrap) = "
          f"{run_meta['p_insertion']:.4f}")
    print(f"  n_examples (curated set, matches manifest.json) = {run_meta['n_examples']}")
    assert abs(run_meta["p_deletion"] - 0.027) < 0.005, (
        f"deletion p-value {run_meta['p_deletion']} does not match the expected ~0.027 cited "
        f"in the task spec -- STOP, do not use a stale/wrong number in the figure disclaimer.")
    print("=" * 100)

    order = ["a", "b", "c", "d"]
    panels = [selected[k] for k in order]
    if any(p["index"] is None for p in panels):
        missing = [k for k in order if selected[k]["index"] is None]
        raise RuntimeError(
            f"Cannot render figure: no example at all (not even a caveated fallback) for "
            f"panel(s) {missing}. Stopping per task instructions instead of fabricating one.")

    # copy (not mutate classification["by_idx"] in place) -- that dict is also read below by
    # "category_matching_notes" (damage_gt_str etc.), which must keep describing ONLY the
    # original 12-example manifest, not the manually-injected override index.
    by_idx = dict(classification["by_idx"])
    if MANUAL_OVERRIDE_B_INDEX is not None:
        by_idx[MANUAL_OVERRIDE_B_INDEX] = build_manual_row(MANUAL_OVERRIDE_B_INDEX)

    print("\n[load] XaiEngine (this loads the fine-tuned checkpoint) ...")
    engine = XaiEngine(device="cuda" if torch.cuda.is_available() else "cpu")

    heat_cache, meta_cache, img_cache = {}, {}, {}
    t_all0 = time.time()
    for p in panels:
        i = p["index"]
        r = by_idx[i]
        rec = {"pre_image_path": r["pre_image_path"], "post_image_path": r["post_image_path"]}
        heat, meta, img = compute_bangunan_heatmap(engine, rec, i)
        heat_cache[i] = heat
        meta_cache[i] = meta
        img_cache[i] = img
    print(f"[rise] total wall-clock for {len(panels)} examples: {time.time() - t_all0:.1f}s")

    os.makedirs(FIG_DIR, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 8.1))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.15, 1.0, 1.0, 0.4],
                          hspace=0.12, wspace=0.10,
                          left=0.02, right=0.99, top=0.92, bottom=0.05)

    CAT_LABELS = {
        "a": "(a) Tidak Ada Kerusakan (Benar)",
        "b": "(b) Akurat",
        "c": "(c) Detail Terlewat",
        "d": "(d) Under-Deteksi (Bias)",
    }
    BOX_COLORS = {"a": "#1a7a3c", "b": "#7a1a1a", "c": "#8a6a00", "d": "#1a3a8a"}

    for col, key in enumerate(order):
        i = selected[key]["index"]
        r = by_idx[i]
        caveat = selected[key]["caveat"]

        # row 0: composite + heatmap overlay
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(flatten_overlay(img_cache[i], heat_cache[i], 0.45))
        ax_img.axvline(192, color="white", lw=1.2, ls="--")
        ax_img.set_xticks([]); ax_img.set_yticks([])
        title = CAT_LABELS[key]
        if caveat:
            title += "\n(TIDAK DITEMUKAN CONTOH ASLI --\nkandidat terdekat, lihat catatan)"
        title += f"\nidx {i}"
        ax_img.set_title(title, fontsize=6.0, linespacing=1.3)

        # row 1: ground truth
        ax_gt = fig.add_subplot(gs[1, col])
        render_text_panel(ax_gt, r["gt_all"], "Ground Truth", BOX_COLORS[key])

        # row 2: prediction
        ax_pred = fig.add_subplot(gs[2, col])
        render_text_panel(ax_pred, r["gen_all"], "Prediksi", BOX_COLORS[key])

    ax_note = fig.add_subplot(gs[3, :])
    ax_note.axis("off")
    # Each Indonesian-decimal-comma value is formatted in ISOLATION, on its own line, and
    # only THEN interpolated into the sentence below. Chaining `.replace(".", ",")` directly
    # onto an f-string that sits inside a larger implicitly-concatenated string literal is a
    # trap: Python concatenates ALL the adjacent string-literal fragments (across the line
    # breaks) into ONE string BEFORE any method call on the last fragment runs, so
    # `.replace()` silently corrupts every earlier period too (caught by literally rendering
    # the figure and reading the disclaimer text: "Petsiuk et al,, 2018" and "BANGUNAN,
    # Signifikan" instead of the intended sentence-ending periods). Isolating each numeric
    # string in its own variable, with no method call touching the surrounding sentence,
    # avoids that trap entirely.
    p_del_str = f"{run_meta['p_deletion']:.3f}".replace(".", ",")
    p_ins_str = f"{run_meta['p_insertion']:.3f}".replace(".", ",")
    disclaimer = (
        f"Heatmap: peta importance RISE (Petsiuk et al., 2018; N={run_meta['n_masks']} mask), "
        f"dirata-rata untuk token section BANGUNAN. Signifikan secara kausal pada uji "
        f"deletion (p={p_del_str} vs kontrol acak) tetapi TIDAK pada uji insertion "
        f"(p={p_ins_str}); ditampilkan sebagai "
        f"indikasi kualitatif, bukan bukti definitif area penentu keputusan model "
        f"(n={run_meta['n_examples']} contoh kurasi, 1 seed). "
        f"Garis putus-putih vertikal = batas PRE (kiri) / POST (kanan)."
    )
    ax_note.text(0.5, 0.5, "\n".join(textwrap.wrap(disclaimer, width=132)),
                fontsize=5.6, ha="center", va="center",
                transform=ax_note.transAxes, color="0.25")

    fig.suptitle("Perbandingan Kualitatif: Ground Truth vs Prediksi, section BANGUNAN "
                 "(heatmap RISE)", fontsize=8.5, y=0.985)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF, dpi=300)
    plt.close(fig)
    print(f"\n[done] figure -> {OUT_PNG}")
    print(f"[done] figure -> {OUT_PDF}")

    selection_out = {
        "selected": {
            key: {
                "index": selected[key]["index"],
                "category": selected[key]["category"],
                "caveat": selected[key]["caveat"],
                "tag": by_idx[selected[key]["index"]]["tag"],
                "gt_bangunan": by_idx[selected[key]["index"]]["gt_bangunan"],
                "gen_bangunan": by_idx[selected[key]["index"]]["gen_bangunan"],
                "rise_meta": meta_cache[selected[key]["index"]],
            } for key in order
        },
        "category_matching_notes": {
            "a_candidates_all_12": classification["a_candidates"],
            "b_candidates_all_12": classification["b_candidates"],
            "c_candidates_all_12": [t[0] for t in classification["c_candidates"]],
            "d_candidates_all_12": classification["d_candidates"],
            "b_akurat_missing": len(classification["b_candidates"]) == 0,
            "b_akurat_missing_note": (
                "Tidak ada contoh di antara 12 data kurasi (results/xai_examples/manifest.json)"
                " di mana ground truth BANGUNAN menunjukkan kerusakan nyata DAN prediksi model "
                "juga mengakui adanya kerusakan (bukan boilerplate 'tidak ada kerusakan'). "
                "Ketiga contoh GT-berkerusakan yang tersedia (idx "
                f"{damage_gt_str(classification['by_idx'])}) semuanya diprediksi 'tidak ada "
                "kerusakan' oleh model -- under-deteksi total, konsisten dengan bias "
                "terdokumentasi di docs/advisor_discussion_summary.md (rate prediksi 'tidak "
                "ada kerusakan' 78.5% vs rate asli di data 34.2%)."
                if len(classification["b_candidates"]) == 0 else None
            ),
        },
        "rise_n_masks": run_meta["n_masks"],
        "rise_wallclock_note": (
            f"{run_meta['mean_seconds_per_example']:.1f} detik/contoh (rata-rata terukur, "
            f"lihat results/xai_rise/deletion_insertion_test.json rise_config."
            f"mean_seconds_per_example)"
        ),
        "rise_significance": {
            "deletion_p_value_vs_random_paired_bootstrap": run_meta["p_deletion"],
            "insertion_p_value_vs_random_paired_bootstrap": run_meta["p_insertion"],
            "source": "results/xai_rise/deletion_insertion_test.json "
                      "significance_paired_bootstrap.token_f1.rise.{deletion,insertion}"
                      ".paired_bootstrap_attention_minus_random.two_sided_p_value_approx",
        },
        "methodological_note": (
            "Set n=12 dipilih stratified (10 tipe bencana, campuran prediksi benar/salah) "
            "untuk cakupan kualitatif yang beragam -- bukan sampel acak, dan bukan "
            "representasi statistik dari 2.362 data test set penuh. Digunakan set yang "
            "identik dengan eksperimen rollout sebelumnya agar hasil RISE dan rollout dapat "
            "dibandingkan langsung tanpa perbedaan kondisi pengujian."
        ),
        "manual_override_note": (
            f"Panel (b) idx {MANUAL_OVERRIDE_B_INDEX} dipilih manual dari luar 12 data kurasi "
            "(hasil scripts/find_accurate_damage_examples.py -> "
            "results/accurate_damage_candidates/), menggantikan fallback otomatis (idx 2028, "
            "yang punya caveat 'bukan contoh akurat sungguhan'). GT & prediksi BANGUNAN-nya "
            "sama-sama menyebut kerusakan/banjir nyata -- genuinely akurat, bukan kandidat "
            "terdekat. RISE dihitung ulang langsung untuk index ini (sama seperti panel lain), "
            "TIDAK diambil dari cache/pre-komputasi manapun."
            if MANUAL_OVERRIDE_B_INDEX is not None else None
        ),
    }
    with open(OUT_SELECTION_JSON, "w", encoding="utf-8") as f:
        json.dump(selection_out, f, indent=2, ensure_ascii=False)
    print(f"[done] selection metadata -> {OUT_SELECTION_JSON}")


if __name__ == "__main__":
    main()
