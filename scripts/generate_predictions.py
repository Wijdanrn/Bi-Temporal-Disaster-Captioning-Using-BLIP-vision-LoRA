"""
Fase 4 -- generate real predictions from checkpoints/full_run_v2/best on the full test set.

Runs model.generate() (not just forward-pass loss) on all 2,362 usable test records, decodes
through IndoReportCodec.decode() (never re-implemented), and writes one JSONL record per test
item so results are inspectable individually, not just aggregated into a score.

Usage:
    python scripts/generate_predictions.py
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

from scripts.build_model import load_trainable_state  # noqa: E402
from scripts.dataset import DisasterCaptionDataset, make_collate_fn  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints", "full_run_v2", "best")
OUT_PATH = os.path.join(ROOT, "results", "predictions_test.jsonl")
BATCH_SIZE = 64
MAX_NEW_TOKENS = 300


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
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for bi, batch in enumerate(dl):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_ids = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
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
            done = (bi + 1) * BATCH_SIZE
            rate = n_written / elapsed
            eta = (len(ds) - n_written) / rate if rate > 0 else float("inf")
            print(f"  batch {bi+1}: {n_written}/{len(ds)} written "
                  f"({rate:.2f}/s, ETA {eta/60:.1f} min)", flush=True)

    print(f"[done] wrote {n_written} predictions to {OUT_PATH} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
