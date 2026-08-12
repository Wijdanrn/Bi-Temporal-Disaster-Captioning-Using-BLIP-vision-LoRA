"""
Fase 4 -- translate the zero-shot baseline's raw English captions to Indonesian.

Methodology fixed in docs/design_decisions.md SS4.2 BEFORE this was run: whole-caption
translation (no section splitting -- zero-shot BLIP produces one free sentence, not our
7-section format) with the SAME NLLB model used in Fase 1
(facebook/nllb-200-distilled-600M, scripts/translate_captions.py's NllbTranslator).

Usage:
    python scripts/translate_baseline.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.translate_captions import NllbTranslator  # noqa: E402

IN_PATH = os.path.join(ROOT, "results", "predictions_baseline_en.jsonl")
OUT_PATH = os.path.join(ROOT, "results", "predictions_baseline_id.jsonl")


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    print(f"[data] {len(rows)} baseline captions to translate")

    translator = NllbTranslator(batch_size=64)
    texts = [r["caption_en"] if r["caption_en"].strip() else "." for r in rows]
    translated = translator.translate_many(texts)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r, t in zip(rows, translated):
            row = {"index": r["index"], "caption_en": r["caption_en"], "caption_id_nllb": t}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(rows)} translated baseline captions to {OUT_PATH}")


if __name__ == "__main__":
    main()
