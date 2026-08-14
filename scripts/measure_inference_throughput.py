"""
No original timing log was saved for generate_predictions_vision.py's full-set run (only
training runs are logged under logs/). Rather than guess, this measures REAL throughput
right now on the same checkpoint/settings/hardware, on a small real sample, and extrapolates
honestly to the full 2,362-record test set -- clearly labeled as a fresh measurement, not a
recovered historical number.

Usage:
    python scripts/measure_inference_throughput.py
"""
from __future__ import annotations

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

CKPT_DIR = os.path.join(ROOT, "checkpoints", "full_run_v3_vision", "best")
BATCH_SIZE = 48
MAX_NEW_TOKENS = 300
N_SAMPLE = 192  # 4 batches -- enough to get past warmup and see a stable steady-state rate
FULL_N = 2362


def main():
    device = "cuda"
    print(f"[load] {CKPT_DIR}")
    bundle = load_trainable_state(CKPT_DIR, device=device, lora_r=8, lora_alpha=32,
                                   lora_dropout=0.1, max_length=384, vision_lora=True)
    model = bundle.model
    model.eval()

    ds = DisasterCaptionDataset("test", bundle=bundle, drop_unreadable=True, limit=N_SAMPLE)
    print(f"[data] measuring on {len(ds)} real records (same settings as generate_predictions_vision.py)")
    collate = make_collate_fn(bundle.codec, keep_meta=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4)

    n_done = 0
    batch_times = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for bi, batch in enumerate(dl):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        tb0 = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_ids = model.generate(pixel_values=pixel_values, max_new_tokens=MAX_NEW_TOKENS,
                                      num_beams=1, do_sample=False)
        torch.cuda.synchronize()
        tb1 = time.time()
        n_done += pixel_values.shape[0]
        batch_times.append(tb1 - tb0)
        print(f"  batch {bi+1}: {n_done}/{len(ds)}, this batch {tb1-tb0:.1f}s, "
              f"avg_gen_len={out_ids.shape[1]}")

    total_elapsed = time.time() - t0
    # first batch includes CUDA/cuDNN warmup -- report both with and without it
    warm_batches = batch_times[1:] if len(batch_times) > 1 else batch_times
    steady_rate = n_done / total_elapsed if len(batch_times) <= 1 else \
        (n_done - BATCH_SIZE) / sum(warm_batches)
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    print()
    print(f"[measured] {n_done} records in {total_elapsed:.1f}s total "
          f"(includes 1 warmup batch of {batch_times[0]:.1f}s)")
    print(f"[measured] steady-state rate (excl. warmup batch): {steady_rate:.3f} records/s")
    print(f"[measured] peak GPU memory: {peak_mem:.0f} MB")
    print()
    est_full_s = FULL_N / steady_rate
    print(f"[extrapolated] estimated time for full {FULL_N}-record test set: "
          f"{est_full_s/60:.1f} min ({est_full_s/3600:.2f} h), "
          f"based on this fresh {n_done}-record measurement on this exact GPU/checkpoint/settings")


if __name__ == "__main__":
    main()
