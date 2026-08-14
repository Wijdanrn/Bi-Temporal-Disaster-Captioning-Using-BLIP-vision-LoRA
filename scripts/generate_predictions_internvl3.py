"""
Generate real predictions from a LoRA-fine-tuned InternVL3-1B-hf checkpoint on the test
set (or a subset), for later scoring with scripts/compute_metrics.py's reusable functions.
Same two-separate-images approach as training (scripts/internvl3_dataset.py) -- no composite.

Usage:
    python scripts/generate_predictions_internvl3.py --ckpt checkpoints/internvl3_run1/best \
        --out results/predictions_test_internvl3_epoch1.jsonl [--limit N] [--indices_file F]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader
from peft import PeftModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402
from scripts.internvl3_dataset import InternVLDisasterDataset, make_collate_fn, PROMPT_ID  # noqa: E402
from scripts.dataset import load_image  # noqa: E402

REPO = "OpenGVLab/InternVL3-1B-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--indices_file", default=None, help="JSON file with a list of dataset indices to restrict to")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda"
    print(f"[load] base model {REPO}")
    processor = AutoProcessor.from_pretrained(REPO)
    processor.tokenizer.padding_side = "left"  # required for correct batched causal-LM generation
    base = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.bfloat16)
    print(f"[load] LoRA adapter from {args.ckpt}")
    model = PeftModel.from_pretrained(base, args.ckpt)
    model.to(device)
    model.eval()

    ds = InternVLDisasterDataset(args.split, processor, limit=args.limit)
    indices = list(range(len(ds)))
    if args.indices_file:
        with open(args.indices_file, encoding="utf-8") as f:
            wanted = set(json.load(f))
        indices = [i for i in indices if i in wanted]
        print(f"[filter] restricted to {len(indices)} indices from {args.indices_file}")

    print(f"[data] generating on {len(indices)} records, batch_size={args.batch_size}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_written = 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    with open(args.out, "w", encoding="utf-8") as f:
        for bstart in range(0, len(indices), args.batch_size):
            batch_idx = indices[bstart: bstart + args.batch_size]
            records = [ds.records[i] for i in batch_idx]
            prompts, images_list = [], []
            for r in records:
                pre = load_image(r["pre_image_path"])
                post = load_image(r["post_image_path"])
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                                           {"type": "text", "text": PROMPT_ID}]}]
                prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True))
                images_list.append([pre, post])

            inputs = processor(images=images_list, text=prompts, return_tensors="pt",
                                padding=True, max_patches=1)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                )
            gen_only = out_ids[:, inputs["input_ids"].shape[1]:]

            for j, idx in enumerate(batch_idx):
                gen_text = processor.tokenizer.decode(gen_only[j], skip_special_tokens=True)
                row = {
                    "index": idx,
                    "pre_image_path": records[j]["pre_image_path"],
                    "post_image_path": records[j]["post_image_path"],
                    "ground_truth_id": records[j]["ground_truth_id"],
                    "generated_caption": gen_text.strip(),
                    "num_generated_tokens": int(gen_only[j].shape[0]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_written += len(batch_idx)
            elapsed = time.time() - t0
            rate = n_written / elapsed
            eta = (len(indices) - n_written) / rate if rate > 0 else float("inf")
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            print(f"  {n_written}/{len(indices)} ({rate:.3f}/s, ETA {eta/60:.1f} min, "
                  f"peak_mem {peak_mem:.0f}MB)", flush=True)

    print(f"[done] wrote {n_written} predictions to {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
