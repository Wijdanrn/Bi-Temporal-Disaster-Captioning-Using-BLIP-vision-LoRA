"""
Export a browsable gallery of the "genuine Akurat" candidates found by
`find_accurate_damage_examples.py` (results/accurate_damage_candidates_full_testset.json), so a
human can look at the actual pre/post composite images and pick one -- rather than trusting the
text heuristic alone (it has known false positives, see that script's docstring).

For each of the 73 candidates: renders the same pre|post composite the model actually sees
(scripts/dataset.make_composite, BLIP's real preprocessing spec), saves it as a PNG, and writes
a manifest.json + README.md index (sorted by index) with the full GT/generated BANGUNAN text
next to each image filename so scanning the folder and cross-referencing text is easy.

No GPU / model checkpoint needed -- pure image compositing, same as make_qualitative_figure_diff.py.

Usage:
    python scripts/export_accurate_damage_candidates.py
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import BlipImageProcessor  # noqa: E402

from scripts.build_model import BASE_MODEL  # noqa: E402
from scripts.dataset import ImageSpec, load_image, make_composite  # noqa: E402

CANDIDATES_PATH = os.path.join(ROOT, "results", "accurate_damage_candidates_full_testset.json")
OUT_DIR = os.path.join(ROOT, "results", "accurate_damage_candidates")


def slug(s: str | None) -> str:
    s = (s or "na").lower().strip()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "na"


def main():
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    candidates = sorted(data["candidates"], key=lambda c: c["index"])
    print(f"[load] {len(candidates)} candidates from {CANDIDATES_PATH}")

    ip = BlipImageProcessor.from_pretrained(BASE_MODEL)
    spec = ImageSpec.from_processor(ip)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for c in candidates:
        i = c["index"]
        tag = f"idx{i:04d}_{slug(c['gt_bencana'])}"
        fname = f"{tag}.png"
        img = make_composite(load_image(c["pre_image_path"]), load_image(c["post_image_path"]),
                             spec)
        img.save(os.path.join(OUT_DIR, fname))
        manifest.append({**c, "image_file": fname, "tag": tag})
        print(f"  [{i}] -> {fname}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_total_records": data["n_total_records"],
            "n_candidates": len(manifest),
            "note": data["note"],
            "candidates": manifest,
        }, f, indent=2, ensure_ascii=False)

    lines = [
        "# Kandidat Contoh \"Akurat\" (deteksi kerusakan nyata yang benar) -- full test set",
        "",
        f"{len(manifest)} kandidat, disaring dari 2362 data test set lewat "
        "`scripts/find_accurate_damage_examples.py` (murni text-matching, TIDAK butuh model/GPU"
        " -- lihat docstring script itu untuk metodologi & known false-positive).",
        "",
        "**PENTING**: ini hasil filter HEURISTIK, bukan label terverifikasi manual satu-satu "
        "(beda dengan 12 contoh xAI yang GT-nya diverifikasi manual). Baca sendiri kolom GT vs "
        "GEN di bawah / buka gambar komposit-nya sebelum memilih -- beberapa baris ternyata "
        "false positive (mis. GT sebenarnya bilang TIDAK ada kerusakan tapi lolos filter karena "
        "frasanya tidak persis cocok dengan pola yang dicari).",
        "",
        "Komposit gambar (pre|post, garis putih = batas) ada di folder ini, nama file = "
        "`idx{index}_{bencana}.png`.",
        "",
        "| idx | GT bencana | Gen bencana | GT BANGUNAN (potongan) | Gen BANGUNAN (potongan) | File |",
        "|---|---|---|---|---|---|",
    ]
    for c in manifest:
        gt_snip = (c["gt_bangunan"][:80] + "...") if len(c["gt_bangunan"]) > 80 else c["gt_bangunan"]
        gen_snip = (c["gen_bangunan"][:80] + "...") if len(c["gen_bangunan"]) > 80 else c["gen_bangunan"]
        gt_snip = gt_snip.replace("|", "/").replace("\n", " ")
        gen_snip = gen_snip.replace("|", "/").replace("\n", " ")
        lines.append(f"| {c['index']} | {c['gt_bencana']} | {c['gen_bencana']} | {gt_snip} | "
                     f"{gen_snip} | `{c['image_file']}` |")
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[done] {len(manifest)} images + manifest.json + README.md -> {OUT_DIR}")


if __name__ == "__main__":
    main()
