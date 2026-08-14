"""
LoRA fine-tune OpenGVLab/InternVL3-1B-hf on our translated Indonesian disaster-report data.

Unlike BLIP: no tokenizer/vocab surgery needed (InternVL3's Qwen2 backbone already has a
151k-token multilingual BPE tokenizer with real Indonesian coverage -- the whole reason it
was picked, see docs/design_decisions.md SS1-3 and the deep-research comparison). Pre/post
images are fed as two separate native images (multi-image support), not a composite -- see
scripts/internvl3_dataset.py. LoRA targets q_proj/v_proj on BOTH the language model and the
vision tower, matching the final BLIP recipe (text+vision LoRA outperformed text-only, see
SS5) -- applying the same lesson from the start here instead of re-discovering it.

Usage:
    python scripts/train_internvl3.py --run_name internvl3_run1 --epochs 3 --batch_size 4
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
from peft import LoraConfig, get_peft_model, PeftModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402
from scripts.internvl3_dataset import InternVLDisasterDataset, make_collate_fn  # noqa: E402

REPO = "OpenGVLab/InternVL3-1B-hf"
LOG_DIR = os.path.join(ROOT, "logs")
CKPT_ROOT = os.path.join(ROOT, "checkpoints")


def build_model(resume_from: str | None = None):
    processor = AutoProcessor.from_pretrained(REPO)
    base = AutoModelForImageTextToText.from_pretrained(REPO, dtype=torch.bfloat16)
    if resume_from:
        print(f"[resume] loading existing LoRA adapter from {resume_from} (continuing training, "
              f"NOT re-initializing LoRA -- optimizer state is fresh, matching how BLIP full_run_v2 "
              f"resumed from full_run_v1's weights without carrying over Adam momentum)")
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
    return processor, model


def param_breakdown(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--val_limit", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="hard cap for time-boxed runs")
    ap.add_argument("--resume_from", type=str, default=None, help="path to an existing LoRA adapter dir to continue training from")
    args = ap.parse_args()

    device = "cuda"
    print(f"[load] {REPO}")
    processor, model = build_model(resume_from=args.resume_from)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required with PEFT + gradient checkpointing, else no grad flows
    trainable, total = param_breakdown(model)
    print(f"[params] trainable={trainable:,} total={total:,} ({trainable/total*100:.3f}%)")

    train_ds = InternVLDisasterDataset("train", processor, limit=args.train_limit)
    val_ds = InternVLDisasterDataset("val", processor, limit=args.val_limit)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    collate = make_collate_fn(processor.tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = (len(train_dl) // args.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=min(0.3, args.warmup_steps / max(total_steps, 1)), anneal_strategy="cos",
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    ckpt_dir = os.path.join(CKPT_ROOT, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    steps_log = open(os.path.join(LOG_DIR, f"{args.run_name}_steps.jsonl"), "w", encoding="utf-8")
    epochs_log = open(os.path.join(LOG_DIR, f"{args.run_name}_epochs.jsonl"), "w", encoding="utf-8")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
    run_meta = {
        "run_name": args.run_name, "repo": REPO, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
        "lora_targets": ["q_proj", "v_proj"], "lora_scope": "language_model + vision_tower",
        "trainable_params": trainable, "total_params": total,
        "gpu": gpu_name, "gpu_mem_mb": gpu_mem,
    }

    global_step = 0
    best_val = float("inf")
    t_run0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep0 = time.time()
        optim.zero_grad()
        for bi, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            loss.backward()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % 20 == 0:
                    max_mem = torch.cuda.max_memory_allocated() / 1e6
                    elapsed = time.time() - t_run0
                    rec = {"step": global_step, "epoch": epoch, "loss": out.loss.item(),
                           "lr": sched.get_last_lr()[0], "elapsed_s": round(elapsed, 1),
                           "max_mem_mb": round(max_mem, 1)}
                    steps_log.write(json.dumps(rec) + "\n"); steps_log.flush()
                    print(f"  step {global_step} epoch {epoch} loss={out.loss.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} mem={max_mem:.0f}MB elapsed={elapsed:.0f}s", flush=True)
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                val_losses.append(out.loss.item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        ep_time = time.time() - t_ep0
        elapsed_total = time.time() - t_run0
        rec = {"epoch": epoch, "val_loss": val_loss, "epoch_time_s": round(ep_time, 1),
               "global_step": global_step, "elapsed_total_s": round(elapsed_total, 1)}
        epochs_log.write(json.dumps(rec) + "\n"); epochs_log.flush()
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} time={ep_time:.0f}s total_elapsed={elapsed_total:.0f}s", flush=True)

        ep_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
        model.save_pretrained(ep_dir)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(ckpt_dir, "best"))
            print(f"  [best] new best val_loss={val_loss:.4f}, saved to {ckpt_dir}/best")

        if stop:
            break

    total_wall = time.time() - t_run0
    run_meta.update({
        "total_wall_clock_s": round(total_wall, 1), "total_wall_clock_h": round(total_wall / 3600, 3),
        "global_step": global_step, "best_val_loss": best_val,
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    })
    with open(os.path.join(LOG_DIR, f"{args.run_name}_run.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    steps_log.close()
    epochs_log.close()
    print(f"[done] {args.run_name}: best_val_loss={best_val:.4f} total_time={total_wall/60:.1f}min")


if __name__ == "__main__":
    main()
