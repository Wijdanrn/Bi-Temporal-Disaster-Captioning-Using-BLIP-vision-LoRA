"""
IndoBLIP-Post-Disaster -- Fase 3/4 fine-tuning loop.

Executes the training run against the architecture already decided in Fase 2/3
(`scripts/build_model.py`, `scripts/dataset.py`, `docs/design_decisions.md`). This script does
not re-derive the tokenizer, LoRA target modules, image compositing, or label contract --
it only trains.

Reproducibility contract:
    - fixed seed (default 42), set on python/numpy/torch/cuda RNGs before anything else runs.
    - a JSON config snapshot (all hyperparameters, exact argv, library versions, detected GPU,
      seed) is written to configs/ BEFORE the first optimizer step.
    - a JSONL step log (loss, lr, wall-clock) and epoch-level val log are written to logs/ as
      training proceeds, not reconstructed after the fact.
    - checkpoints are saved with `save_trainable_state()` (build_model.py), never
      `model.save_pretrained()` -- that call alone would silently drop the 24.5M-param
      embedding matrix (see docs/design_decisions.md, risk #6).

Usage:
    python scripts/train.py --run_name short_val --max_steps 200
    python scripts/train.py --run_name full_run --epochs 6
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_model import build_model, save_trainable_state, load_trainable_state  # noqa: E402
from scripts.dataset import DisasterCaptionDataset, make_collate_fn  # noqa: E402


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_memory_mb": round(p.total_memory / 1024**2, 1),
        "capability": f"{p.major}.{p.minor}",
        "multi_processor_count": p.multi_processor_count,
    }


def lib_versions() -> dict:
    import transformers
    import peft
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "platform": platform.platform(),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None  # not a git repo -- fine, still logged as None rather than silently omitted


# --------------------------------------------------------------------------------------
# Param groups: LoRA adapters vs. the embedding matrix / lm-head bias.
#
# These are different in nature -- LoRA deltas are a small rank-8 correction on top of
# pretrained attention weights (already scaled down internally by alpha/r = 4), whereas the
# embedding matrix is being relearned close to from-scratch for 74.6% of its rows (Fase 2,
# design_decisions.md SS3/SS6). Standard practice for each, independently: LoRA is commonly
# trained at 1e-4-2e-4; full-parameter embedding fine-tuning on a small (~7k example) corpus
# uses a lower LR (2e-5-5e-5) to avoid the embedding matrix overfitting/drifting before the
# rest of the model can use it. A single shared LR is a reasonable default (documented as
# permitted in the task brief) but a lower embedding LR is the more defensible choice given
# that mismatch in effective step size, so that is what this script does by default.
# --------------------------------------------------------------------------------------

def build_param_groups(model, lr_lora: float, lr_embed: float):
    # By construction (build_model.py's apply_lora), the ONLY trainable parameters are the
    # LoRA A/B matrices and exactly two others: the word-embedding weight (tied to the LM head)
    # and the LM-head bias -- verified empirically (named_parameters() dump) rather than guessed.
    lora_params, embed_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora_params if "lora_" in name else embed_params).append(p)
    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora, "name": "lora"})
    if embed_params:
        groups.append({"params": embed_params, "lr": lr_embed, "name": "embedding+head_bias"})
    n_lora = sum(p.numel() for p in lora_params)
    n_embed = sum(p.numel() for p in embed_params)
    print(f"[param groups] lora: {len(lora_params)} tensors / {n_lora:,} params @ lr={lr_lora}")
    print(f"[param groups] embed+bias: {len(embed_params)} tensors / {n_embed:,} params @ lr={lr_embed}")
    return groups


# --------------------------------------------------------------------------------------
# Evaluation (val loss)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: int | None = None) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"], labels=batch["labels"])
        total_loss += out.loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="if >0, stop after this many OPTIMIZER steps regardless of epoch "
                         "(used for the short validation run)")
    ap.add_argument("--batch_size", type=int, default=4, help="micro (per-forward) batch size")
    ap.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_embed", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--vision_lora", action="store_true",
                    help="ALSO put LoRA on the vision tower's fused self_attn.qkv "
                         "(Fase 5). Off by default => byte-identical to previous runs.")
    ap.add_argument("--vision_lora_targets", type=str, default=None,
                    help="comma-separated override for the vision target suffixes, e.g. "
                         "'fc1,fc2'. Only used with --vision_lora.")
    ap.add_argument("--allow_new_lora_modules", action="store_true",
                    help="permit --resume_from a checkpoint whose adapter set is a SUBSET "
                         "of the current one (e.g. text-only ckpt + new vision LoRA). The "
                         "new modules start at lora_B=0, so step 0 is a no-op. Verified by "
                         "scripts/test_vision_lora.py.")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--log_every", type=int, default=10, help="steps between step-log writes")
    ap.add_argument("--val_every_steps", type=int, default=0,
                    help="if >0, run a partial val check every N optimizer steps "
                         "(in addition to the end-of-epoch full val pass)")
    ap.add_argument("--val_partial_batches", type=int, default=40,
                    help="batches used for the periodic in-epoch val check (not the full val set)")
    ap.add_argument("--early_stopping_patience", type=int, default=3,
                    help="stop if full val loss hasn't improved for this many epochs")
    ap.add_argument("--limit_train", type=int, default=None,
                    help="debug: truncate the train set to this many records")
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--resume_from", type=str, default=None,
                    help="checkpoint dir to resume model+optimizer+scheduler state from")
    ap.add_argument("--configs_dir", type=str, default=os.path.join(ROOT, "configs"))
    ap.add_argument("--logs_dir", type=str, default=os.path.join(ROOT, "logs"))
    ap.add_argument("--ckpt_dir", type=str, default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--verify_first_checkpoint", action="store_true", default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir_ckpt = os.path.join(args.ckpt_dir, args.run_name)
    os.makedirs(run_dir_ckpt, exist_ok=True)
    os.makedirs(args.configs_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("No CUDA device detected -- refusing to silently fall back to CPU "
                           "for a 225M-param model. Check the environment.")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]

    # ---- config snapshot: written BEFORE training starts, per the reproducibility contract ----
    config = {
        "run_name": args.run_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "hyperparameters": vars(args),
        "seed": args.seed,
        "amp": args.amp,
        "gpu": gpu_info(),
        "library_versions": lib_versions(),
        "git_commit": git_commit(),
    }
    config_path = os.path.join(args.configs_dir, f"{args.run_name}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[config] snapshot written to {config_path}")

    step_log_path = os.path.join(args.logs_dir, f"{args.run_name}_steps.jsonl")
    epoch_log_path = os.path.join(args.logs_dir, f"{args.run_name}_epochs.jsonl")
    run_log_path = os.path.join(args.logs_dir, f"{args.run_name}_run.json")
    step_log_f = open(step_log_path, "a", encoding="utf-8")

    def log_step(rec: dict):
        step_log_f.write(json.dumps(rec) + "\n")
        step_log_f.flush()

    def log_epoch(rec: dict):
        with open(epoch_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- model ----
    print("[build] constructing model...")
    vision_targets = ([s.strip() for s in args.vision_lora_targets.split(",") if s.strip()]
                      if args.vision_lora_targets else None)
    lora_kwargs = dict(
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        vision_lora=args.vision_lora, vision_lora_targets=vision_targets,
        max_length=args.max_length,
    )
    bundle = build_model(dtype=torch.float32, device=device, verbose=True, **lora_kwargs)
    model = bundle.model
    model.train()

    if args.resume_from:
        print(f"[resume] loading trainable state from {args.resume_from}")
        bundle = load_trainable_state(
            args.resume_from, device=device,
            allow_new_lora_modules=args.allow_new_lora_modules,
            **lora_kwargs,
        )
        model = bundle.model
        model.train()

    # ---- data ----
    print("[data] building datasets...")
    train_ds = DisasterCaptionDataset("train", bundle=bundle, drop_unreadable=True,
                                      limit=args.limit_train)
    val_ds = DisasterCaptionDataset("val", bundle=bundle, drop_unreadable=True,
                                    limit=args.limit_val)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    collate = make_collate_fn(bundle.codec)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=min(4, args.num_workers),
        persistent_workers=args.num_workers > 0,
    )

    # ---- optimizer / scheduler ----
    param_groups = build_param_groups(model, args.lr_lora, args.lr_embed)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_optim_steps = (steps_per_epoch * args.epochs if args.max_steps <= 0
                         else args.max_steps)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(args.warmup_steps, max(1, total_optim_steps // 10)),
        num_training_steps=max(total_optim_steps, 1),
    )
    print(f"[schedule] steps/epoch={steps_per_epoch} total_optim_steps={total_optim_steps} "
         f"warmup_steps={min(args.warmup_steps, max(1, total_optim_steps // 10))}")

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    # ---- training loop ----
    run_t0 = time.time()
    global_step = 0
    best_val_loss = float("inf")
    epochs_since_improve = 0
    stop = False

    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_n = 0.0, 0

        for micro_step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = out.loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch} micro_step {micro_step} "
                    f"(global_step {global_step}). Stopping rather than continuing silently."
                )

            if args.amp == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += out.loss.item()
            running_n += 1

            if (micro_step + 1) % args.grad_accum == 0:
                if args.amp == "fp16":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in param_groups for p in g["params"]], args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in param_groups for p in g["params"]], args.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    lrs = {g.get("name", i): g["lr"] for i, g in enumerate(optimizer.param_groups)}
                    mem_mb = torch.cuda.max_memory_allocated() / 1024**2
                    rec = {
                        "step": global_step, "epoch": epoch,
                        "loss": running_loss / running_n,
                        "lr": lrs, "elapsed_s": round(time.time() - run_t0, 1),
                        "max_mem_mb": round(mem_mb, 1),
                    }
                    log_step(rec)
                    print(f"step {global_step:5d} epoch {epoch} loss {rec['loss']:.4f} "
                         f"lr_lora {lrs.get('lora'):.2e} elapsed {rec['elapsed_s']:.0f}s "
                         f"max_mem {mem_mb:.0f}MB")
                    running_loss, running_n = 0.0, 0

                if args.val_every_steps and global_step % args.val_every_steps == 0:
                    v = evaluate(model, val_loader, device, amp_dtype,
                                max_batches=args.val_partial_batches)
                    log_step({"step": global_step, "epoch": epoch, "partial_val_loss": v})
                    print(f"  [partial val] step {global_step} loss {v:.4f}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop = True
                    break

        epoch_time = time.time() - epoch_t0
        val_loss = evaluate(model, val_loader, device, amp_dtype)
        log_epoch({
            "epoch": epoch, "val_loss": val_loss, "epoch_time_s": round(epoch_time, 1),
            "global_step": global_step, "elapsed_total_s": round(time.time() - run_t0, 1),
        })
        print(f"== epoch {epoch} done: val_loss={val_loss:.4f} epoch_time={epoch_time:.0f}s ==")

        # per-epoch checkpoint -- always, so a crash/timeout never loses more than one epoch
        epoch_ckpt_dir = os.path.join(run_dir_ckpt, f"epoch_{epoch}")
        save_trainable_state(bundle, epoch_ckpt_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "global_step": global_step, "epoch": epoch, "best_val_loss": best_val_loss},
            os.path.join(epoch_ckpt_dir, "trainer_state.pt"),
        )
        print(f"[checkpoint] saved {epoch_ckpt_dir}")

        if epoch == 1 and args.verify_first_checkpoint:
            verify_checkpoint_roundtrip(epoch_ckpt_dir, bundle, val_loader, device, amp_dtype, args)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_since_improve = 0
            best_dir = os.path.join(run_dir_ckpt, "best")
            save_trainable_state(bundle, best_dir)
            print(f"[checkpoint] new best (val_loss={val_loss:.4f}) -> {best_dir}")
        else:
            epochs_since_improve += 1
            print(f"[early stopping] no improvement for {epochs_since_improve} epoch(s)")
            if epochs_since_improve >= args.early_stopping_patience:
                print("[early stopping] patience exceeded, stopping.")
                stop = True

        if stop:
            break

    total_time = time.time() - run_t0
    summary = {
        "run_name": args.run_name, "total_wall_clock_s": round(total_time, 1),
        "total_wall_clock_h": round(total_time / 3600, 3),
        "global_step": global_step, "best_val_loss": best_val_loss,
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "gpu": gpu_info(),
    }
    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    step_log_f.close()
    print(f"[done] {json.dumps(summary, indent=2)}")


def verify_checkpoint_roundtrip(ckpt_dir, bundle, val_loader, device, amp_dtype, args):
    """Save -> load into a FRESH model -> confirm identical output. Not assumed, checked."""
    print(f"[verify] round-trip check on {ckpt_dir} ...")
    batch = next(iter(val_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    bundle.model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        logits_before = bundle.model(
            pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], labels=batch["labels"],
        ).logits.float().cpu()
    bundle.model.train()

    fresh = load_trainable_state(
        ckpt_dir, device=device,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        vision_lora=args.vision_lora,
        vision_lora_targets=([s.strip() for s in args.vision_lora_targets.split(",") if s.strip()]
                             if args.vision_lora_targets else None),
        max_length=args.max_length,
    )
    fresh.model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        logits_after = fresh.model(
            pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], labels=batch["labels"],
        ).logits.float().cpu()

    max_abs_diff = (logits_before - logits_after).abs().max().item()
    ok = max_abs_diff < 1e-3
    print(f"[verify] max abs logit diff = {max_abs_diff:.3e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(
            f"Checkpoint round-trip FAILED (max abs diff {max_abs_diff:.3e}). "
            "save/load mechanism is not reproducing the trained model -- stopping."
        )
    del fresh
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
