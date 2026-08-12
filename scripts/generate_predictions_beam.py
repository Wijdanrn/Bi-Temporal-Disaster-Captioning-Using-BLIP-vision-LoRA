"""
Fase 4 follow-up experiment -- same checkpoint (full_run_v2/best), same test set,
but beam search (num_beams=5) instead of greedy decoding, to test whether the
no-damage / majority-class bias documented in docs/design_decisions.md is
partly a decoding-strategy artifact rather than purely a model/vision issue.

Reuses IndoReportCodec.decode()/canonicalize() and DisasterCaptionDataset,
same as scripts/generate_predictions.py -- only the generation call and
output path differ, so results are directly comparable.

Usage:
    python scripts/generate_predictions_beam.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_model import load_trainable_state  # noqa: E402
from scripts.dataset import DisasterCaptionDataset, make_collate_fn  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints", "full_run_v2", "best")
OUT_PATH = os.path.join(ROOT, "results", "predictions_test_beam5.jsonl")
BATCH_SIZE = 32  # batch=16 -> 3.4GB, batch=48 -> 8.35GB (over budget, thrashed) -> 32 is the safe middle
MAX_NEW_TOKENS = 300
NUM_BEAMS = 5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("No CUDA device detected.")

    print(f"[load] {CKPT_DIR}")
    bundle = load_trainable_state(
        CKPT_DIR, device=device,
        lora_r=8, lora_alpha=32, lora_dropout=0.1, max_length=384,
    )
    model = bundle.model
    model.eval()

    ds = DisasterCaptionDataset("test", bundle=bundle, drop_unreadable=True)
    print(f"[data] test usable records: {len(ds)}")
    collate = make_collate_fn(bundle.codec, keep_meta=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n_written = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for bi, batch in enumerate(dl):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_ids = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    early_stopping=True,
                )
            for j in range(pixel_values.shape[0]):
                idx = batch["index"][j]
                rec = ds.records[idx]
                gen_text = bundle.codec.decode(out_ids[j])
                ref_canon = bundle.codec.canonicalize(rec["ground_truth_id"])
                row = {
                    "index": idx,
                    "pre_image_path": rec["pre_image_path"],
                    "post_image_path": rec["post_image_path"],
                    "post_image_type": rec.get("post_image_type"),
                    "ground_truth_id": rec["ground_truth_id"],
                    "ground_truth_id_canonical": ref_canon,
                    "generated_caption": gen_text,
                    "num_generated_tokens": int(out_ids[j].shape[0]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
            elapsed = time.time() - t0
            rate = n_written / elapsed
            eta = (len(ds) - n_written) / rate if rate > 0 else float("inf")
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            print(f"  batch {bi+1}: {n_written}/{len(ds)} written "
                  f"({rate:.2f}/s, ETA {eta/60:.1f} min, peak_mem {peak_mem:.0f}MB)", flush=True)

    print(f"[done] wrote {n_written} predictions to {OUT_PATH} in {(time.time()-t0)/60:.1f} min "
          f"(num_beams={NUM_BEAMS})")


if __name__ == "__main__":
    main()
