"""
Fase 4 -- zero-shot base BLIP baseline predictions on the full test set.

Loads Salesforce/blip-image-captioning-base COMPLETELY UNTOUCHED: original BertTokenizer,
no LoRA, no Indonesian vocabulary swap, no fine-tuning. Receives the SAME pre/post composite
image as our model (same make_composite/ImageSpec pipeline, same underlying checkpoint's
BlipImageProcessor constants -- see docs/design_decisions.md SS4.2 for the full methodology,
written BEFORE this script was run).

Writes results/predictions_baseline_en.jsonl (raw English captions, untranslated) --
translation to Indonesian for scoring is a SEPARATE step (scripts/translate_baseline.py) so the
raw zero-shot output is inspectable on its own.

Usage:
    python scripts/generate_baseline.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import BlipForConditionalGeneration, BlipImageProcessor  # noqa: E402
from scripts.dataset import (  # noqa: E402
    PROCESSED_DIR, ImageSpec, load_image, make_composite, _image_is_readable,
)
from torch.utils.data import Dataset  # noqa: E402

BASE_MODEL = "Salesforce/blip-image-captioning-base"
OUT_PATH = os.path.join(ROOT, "results", "predictions_baseline_en.jsonl")
BATCH_SIZE = 64
MAX_NEW_TOKENS = 50


class RawTestImages(Dataset):
    """Same composite images as our model gets, no text codec involved (baseline needs none)."""

    def __init__(self, spec: ImageSpec, drop_unreadable: bool = True):
        path = os.path.join(PROCESSED_DIR, "captions_test.jsonl")
        with open(path, encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f]
        if drop_unreadable:
            keep = []
            for r in self.records:
                if all(_image_is_readable(r[k]) for k in ("pre_image_path", "post_image_path")):
                    keep.append(r)
            dropped = len(self.records) - len(keep)
            if dropped:
                print(f"[RawTestImages] dropped {dropped} record(s) with unreadable images")
            self.records = keep
        self.spec = spec

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        pre = load_image(r["pre_image_path"])
        post = load_image(r["post_image_path"])
        comp = make_composite(pre, post, self.spec)
        return {"pixel_values": self.spec.to_tensor(comp), "index": i}


def collate(items):
    return {
        "pixel_values": torch.stack([it["pixel_values"] for it in items]),
        "index": [it["index"] for it in items],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("No CUDA device detected.")

    print(f"[load] {BASE_MODEL} (untouched: original tokenizer, no LoRA, no fine-tuning)")
    model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device)
    model.eval()
    image_processor = BlipImageProcessor.from_pretrained(BASE_MODEL)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    spec = ImageSpec.from_processor(image_processor)
    ds = RawTestImages(spec, drop_unreadable=True)
    print(f"[data] test usable records: {len(ds)}")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n_written = 0
    t0 = time.time()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for bi, batch in enumerate(dl):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.no_grad():
                out_ids = model.generate(
                    pixel_values=pixel_values, max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1, do_sample=False,
                )
            captions = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for j, idx in enumerate(batch["index"]):
                row = {"index": idx, "caption_en": captions[j].strip()}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            elapsed = time.time() - t0
            rate = n_written / elapsed
            eta = (len(ds) - n_written) / rate if rate > 0 else float("inf")
            print(f"  batch {bi+1}: {n_written}/{len(ds)} written "
                  f"({rate:.2f}/s, ETA {eta/60:.1f} min)", flush=True)

    print(f"[done] wrote {n_written} baseline captions to {OUT_PATH} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
