"""
Fase 5 verification: LoRA on BLIP's FROZEN VISION TOWER.

Proves four things with real numbers, not absence-of-error:

  A. Tokenizer/codec round-trip still holds (regression guard -- nothing in this phase
     should touch the text codec, so any change here is a bug).
  B. The expanded LoRA config actually lands on `vision_model` (measured module counts)
     and the new trainable-parameter count is computed, not quoted.
  C. Loading the OLD text-only checkpoint into the EXPANDED model is a true no-op:
     text-side LoRA tensors are bit-identical, and the forward logits / generate() output
     are IDENTICAL to the old model's. This is the correctness proof that "continue
     training from full_run_v2/best + vision LoRA" does not disturb what was learned.
  D. A real forward+backward puts NON-ZERO gradients on every new vision LoRA parameter.

Run:
    HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 python scripts/test_vision_lora.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from scripts.build_model import (  # noqa: E402
    VISION_LORA_MLP_MODULES,
    VISION_LORA_TARGET_MODULES,
    build_model,
    count_lora_targets,
    load_trainable_state,
    lora_module_names,
    param_breakdown,
    read_checkpoint_target_modules,
    resolve_target_modules,
)
from scripts.dataset import DisasterCaptionDataset, make_collate_fn  # noqa: E402

CKPT = os.path.join("checkpoints", "full_run_v2", "best")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BATCH = 2

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    _results.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def get_batch(bundle):
    ds = DisasterCaptionDataset("val", bundle=bundle, drop_unreadable=True, limit=N_BATCH)
    collate = make_collate_fn(bundle.codec)
    batch = collate([ds[i] for i in range(N_BATCH)])
    return {k: v.to(DEVICE) for k, v in batch.items()}


@torch.no_grad()
def forward_logits(model, batch):
    model.eval()
    out = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], labels=batch["labels"])
    return out.logits.detach().float().cpu(), float(out.loss.detach())


@torch.no_grad()
def gen_ids(model, batch):
    model.eval()
    return model.generate(pixel_values=batch["pixel_values"], max_new_tokens=40,
                          num_beams=1, do_sample=False).detach().cpu()


def main() -> int:
    torch.manual_seed(0)
    print(f"device={DEVICE}  checkpoint={CKPT}")
    print(f"checkpoint target_modules (read from adapter_config.json): "
          f"{read_checkpoint_target_modules(CKPT)}")

    # ============================================================== A. codec round-trip
    banner("A. Text codec round-trip (regression guard -- Fase 5 must not touch this)")
    baseline = build_model(device=None, verbose=False)  # CPU, text-only LoRA
    codec = baseline.codec
    samples = [
        "BENCANA: gempa bumi\nBANGUNAN: Tidak ada kerusakan struktural yang terlihat.\n"
        "JALAN: jalan utama tergenang.\nKESIMPULAN: dampak sedang.",
        "BENCANA: badai\nBADAN_AIR: sungai meluap\nPERTANIAN: sawah rusak\n"
        "VEGETASI: pohon tumbang, puing-puing terlihat",
    ]
    exact = 0
    for s in samples:
        canon = codec.canonicalize(s)
        ids = codec.encode(canon)
        back = codec.decode(ids)
        first_ok = ids[0] == codec.dec_id
        last_ok = ids[-1] == codec.sep_id
        exact += int(back == canon and first_ok and last_ok)
    check(exact == len(samples), f"decode(encode(x)) == x on {len(samples)} canonical reports",
          f"{exact}/{len(samples)} exact, [DEC] first + [SEP] last")
    # real corpus sample
    import json
    with open(os.path.join("data", "processed", "captions_val.jsonl"), encoding="utf-8") as f:
        real = [json.loads(next(f))["ground_truth_id"] for _ in range(25)]
    ok_real = sum(codec.decode(codec.encode(codec.canonicalize(t))) == codec.canonicalize(t)
                  for t in real)
    check(ok_real == 25, "decode(encode(x)) == x on 25 real val records", f"{ok_real}/25 exact")
    del baseline

    # ============================================== B. where the vision adapter lands
    banner("B. Expanded LoRA config: module placement + MEASURED parameter counts")
    print(f"resolve_target_modules(vision_lora=True) -> "
          f"{resolve_target_modules(vision_lora=True)}")

    old = build_model(device=DEVICE, verbose=False)                   # text-only (baseline)
    new = build_model(device=DEVICE, vision_lora=True, verbose=False)  # text + vision qkv

    t_old, t_new = count_lora_targets(old.model), count_lora_targets(new.model)
    print(f"  old adapter modules: {t_old}")
    print(f"  new adapter modules: {t_new}")
    n_vis = sum(v for k, v in t_new.items() if k.startswith("vision"))
    check(sum(v for k, v in t_old.items() if k.startswith("vision")) == 0,
          "baseline has 0 LoRA layers on vision_model (Fase 2 contract intact)")
    check(n_vis == 12, "expanded config adapts all 12 vision self_attn.qkv modules", f"n={n_vis}")
    check(sum(t_old.values()) == 48 and sum(t_new.values()) == 60,
          "text-side module count unchanged by the expansion",
          f"{sum(t_old.values())} -> {sum(t_new.values())} (48 text + 12 vision)")

    b_old, b_new = param_breakdown(old.model), param_breakdown(new.model)
    d_lora = b_new["lora_vision"]
    hidden = old.model.base_model.model.config.vision_config.hidden_size
    expected = 12 * (8 * hidden + 3 * hidden * 8)  # A:(r,768) + B:(2304,r) per layer, r=8
    check(d_lora == expected, "vision LoRA param count matches r*(d_in) + (3*d_out)*r per layer",
          f"measured {d_lora:,} == 12 * (8*{hidden} + 3*{hidden}*8) = {expected:,}")
    print(f"\n  {'bucket':<28}{'text-only':>16}{'+vision qkv':>16}{'delta':>14}")
    for k in ("lora_text", "lora_vision", "embeddings", "other", "trainable", "total"):
        print(f"  {k:<28}{b_old[k]:>16,}{b_new[k]:>16,}{b_new[k] - b_old[k]:>+14,}")
    print(f"  {'trainable fraction':<28}"
          f"{100 * b_old['trainable'] / b_old['total']:>15.4f}%"
          f"{100 * b_new['trainable'] / b_new['total']:>15.4f}%")
    check(b_new["lora_text"] == b_old["lora_text"] and b_new["embeddings"] == b_old["embeddings"],
          "no text-side or embedding params added/removed by the vision expansion")

    # the MLP-placement alternative, costed rather than guessed
    mlp = build_model(vision_lora=True, vision_lora_targets=VISION_LORA_MLP_MODULES,
                      verbose=False)
    b_mlp = param_breakdown(mlp.model)
    print(f"\n  [ablation cost] vision targets={VISION_LORA_MLP_MODULES}: "
          f"{b_mlp['lora_vision']:,} vision-LoRA params "
          f"({b_mlp['lora_vision'] / max(d_lora, 1):.2f}x the qkv option)")
    del mlp

    # =========================================== C. old checkpoint -> expanded model
    banner("C. Loading the OLD text-only checkpoint into the EXPANDED model")

    # C0: refuse-by-default guard must actually fire
    try:
        load_trainable_state(CKPT, device=None, verbose=False, vision_lora=True)
        check(False, "loading an old ckpt into an expanded config without the explicit flag raises")
    except RuntimeError as e:
        check("allow_new_lora_modules=True" in str(e),
              "loading an old ckpt into an expanded config without the explicit flag raises",
              str(e).split(". If")[0][:90])

    del old, new
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    ref = load_trainable_state(CKPT, device=DEVICE, verbose=False)  # old arch, old weights
    batch = get_batch(ref)
    logits_ref, loss_ref = forward_logits(ref.model, batch)
    gen_ref = gen_ids(ref.model, batch)
    text_ref = {n: p.detach().cpu().clone() for n, p in ref.model.named_parameters()
                if "lora_" in n}
    emb_ref = ref.model.base_model.model.get_input_embeddings().weight.detach().cpu().clone()
    print(f"  reference (text-only ckpt): loss={loss_ref:.6f}  logits{tuple(logits_ref.shape)}")
    del ref
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    exp = load_trainable_state(CKPT, device=DEVICE, verbose=True, vision_lora=True,
                               allow_new_lora_modules=True)
    logits_exp, loss_exp = forward_logits(exp.model, batch)
    gen_exp = gen_ids(exp.model, batch)

    # C1: every text-side LoRA tensor bit-identical
    live = {n: p.detach().cpu() for n, p in exp.model.named_parameters() if "lora_" in n}
    missing = [n for n in text_ref if n not in live]
    bad = [n for n in text_ref if n in live and not torch.equal(live[n], text_ref[n])]
    check(not missing and not bad,
          f"all {len(text_ref)} text-side LoRA tensors bit-identical after expansion",
          f"missing={len(missing)} differing={len(bad)}")

    # C2: embeddings bit-identical
    emb_exp = exp.model.base_model.model.get_input_embeddings().weight.detach().cpu()
    check(torch.equal(emb_exp, emb_ref),
          f"embedding matrix {tuple(emb_exp.shape)} bit-identical after expansion")

    # C3: new vision modules are exactly zero-init on B (and non-zero on A)
    vb = {n: p for n, p in exp.model.named_parameters()
          if "vision_model" in n and "lora_B" in n}
    va = {n: p for n, p in exp.model.named_parameters()
          if "vision_model" in n and "lora_A" in n}
    maxB = max(p.detach().abs().max().item() for p in vb.values())
    minA = min(p.detach().abs().max().item() for p in va.values())
    check(len(vb) == 12 and maxB == 0.0,
          "all 12 vision lora_B are EXACTLY zero at init (=> delta_W = B@A = 0)",
          f"max|B| = {maxB}")
    check(len(va) == 12 and minA > 0.0,
          "all 12 vision lora_A are non-zero (Kaiming init, so gradients can flow to B)",
          f"min over modules of max|A| = {minA:.4e}")

    # C4: THE proof -- identical output
    d_logit = (logits_exp - logits_ref).abs().max().item()
    check(d_logit == 0.0, "forward logits IDENTICAL to the text-only checkpoint",
          f"max|delta logits| = {d_logit:.3e}  (loss {loss_ref:.8f} vs {loss_exp:.8f})")
    check(loss_ref == loss_exp, "loss bit-identical", f"{loss_ref!r} vs {loss_exp!r}")
    same_gen = gen_ref.shape == gen_exp.shape and torch.equal(gen_ref, gen_exp)
    check(same_gen, "generate() produces IDENTICAL token ids",
          f"{tuple(gen_ref.shape)} vs {tuple(gen_exp.shape)}")
    dec_ref = [exp.codec.decode(r) for r in gen_ref]
    dec_exp = [exp.codec.decode(r) for r in gen_exp]
    check(dec_ref == dec_exp, "decoded text identical",
          repr(dec_exp[0][:70]) + "...")

    # negative control: the comparison has power
    with torch.no_grad():
        for p in vb.values():
            p.add_(torch.randn_like(p) * 0.02)
    logits_perturbed, loss_perturbed = forward_logits(exp.model, batch)
    d_pert = (logits_perturbed - logits_ref).abs().max().item()
    check(d_pert > 1e-3,
          "NEGATIVE CONTROL: perturbing vision lora_B DOES change the logits "
          "(so the identity above is a real result, not a dead code path)",
          f"max|delta| = {d_pert:.4e}, loss {loss_ref:.4f} -> {loss_perturbed:.4f}")
    with torch.no_grad():
        for p in vb.values():
            p.zero_()
    check((forward_logits(exp.model, batch)[0] - logits_ref).abs().max().item() == 0.0,
          "restoring B=0 restores identical logits")

    # =============================================== D. real forward+backward gradients
    banner("D. Real forward + backward: do the NEW vision LoRA params get gradients?")
    exp.model.train()
    exp.model.zero_grad(set_to_none=True)
    out = exp.model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], labels=batch["labels"])
    out.loss.backward()
    print(f"  training-mode loss = {out.loss.detach().item():.6f}")

    trainable = [(n, p) for n, p in exp.model.named_parameters() if p.requires_grad]
    no_grad = [n for n, p in trainable if p.grad is None]
    zero_grad = [n for n, p in trainable if p.grad is not None and p.grad.abs().max().item() == 0]
    check(not no_grad, f"all {len(trainable)} trainable params have a .grad tensor",
          f"{len(no_grad)} missing")

    vis = [(n, p) for n, p in trainable if "vision_model" in n and "lora_" in n]
    vB = [(n, p) for n, p in vis if "lora_B" in n]
    vA = [(n, p) for n, p in vis if "lora_A" in n]
    nzB = sum(1 for _, p in vB if p.grad is not None and p.grad.abs().max().item() > 0)
    nzA = sum(1 for _, p in vA if p.grad is not None and p.grad.abs().max().item() > 0)
    mB = sum(p.grad.abs().mean().item() for _, p in vB) / len(vB)
    check(nzB == 12, "all 12 vision lora_B have NON-ZERO gradients",
          f"{nzB}/12, mean|grad| = {mB:.4e}")
    check(nzA == 0,
          "all 12 vision lora_A have ZERO gradient at step 0 -- EXPECTED, not a bug "
          "(dL/dA = B^T @ dL/dy and B==0; A starts moving once B leaves zero)",
          f"{nzA}/12 non-zero")

    txt = [(n, p) for n, p in trainable if "lora_" in n and "vision_model" not in n]
    nzT = sum(1 for _, p in txt if p.grad is not None and p.grad.abs().max().item() > 0)
    check(nzT >= 48, f"text-side LoRA still receives gradients ({nzT}/{len(txt)} non-zero)")
    emb_p = exp.model.base_model.model.get_input_embeddings().weight
    check(emb_p.grad is not None and emb_p.grad.abs().mean().item() > 0,
          "embedding matrix still receives gradient",
          f"mean|grad| = {emb_p.grad.abs().mean().item():.4e}")

    # one optimizer step must move B off zero (proves the gradient is USABLE, not just present)
    opt = torch.optim.AdamW([p for _, p in vis], lr=1e-4)
    opt.step()
    moved = sum(1 for _, p in vB if p.detach().abs().max().item() > 0)
    check(moved == 12, "one AdamW step moves all 12 vision lora_B off zero",
          f"{moved}/12")
    print(f"  zero-grad trainable tensors (incl. the expected vision lora_A): {len(zero_grad)}")

    # step 2: now that B != 0, lora_A must start receiving gradient too. Without this the
    # "A has zero grad" result above is indistinguishable from A being permanently dead.
    exp.model.zero_grad(set_to_none=True)
    out2 = exp.model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"], labels=batch["labels"])
    out2.loss.backward()
    nzA2 = sum(1 for _, p in vA if p.grad is not None and p.grad.abs().max().item() > 0)
    mA2 = sum(p.grad.abs().mean().item() for _, p in vA) / len(vA)
    check(nzA2 == 12,
          "after ONE step (B != 0), all 12 vision lora_A now receive non-zero gradient "
          "-- A is not permanently dead, the whole vision adapter trains",
          f"{nzA2}/12, mean|grad| = {mA2:.4e}")

    banner("SUMMARY")
    n_pass = sum(1 for ok, _ in _results if ok)
    for ok, label in _results:
        if not ok:
            print(f"  FAILED: {label}")
    print(f"  {n_pass}/{len(_results)} checks passed")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
