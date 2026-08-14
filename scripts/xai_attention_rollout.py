"""
Fase 6 (xAI) -- cross-attention extraction + attention rollout for the FINAL model
(`checkpoints/full_run_v3_vision/best`).

WHAT THIS FILE IS
-----------------
The reusable engine (`XaiEngine`) plus three CLI modes:

    --mode selftest    plumbing + alignment + image-sensitivity checks (run this FIRST)
    --mode examples    per-example heatmap overlays + captions for a deterministic,
                       diversity-stratified sample of REAL test-set records
    --mode aggregate   memory-frugal pass over N test examples that keeps only the
                       per-section pre/post attention-mass numbers

`scripts/xai_faithfulness_check.py` imports XaiEngine from here -- the deletion/insertion
and half-occlusion tests must run against the exact same maps that get plotted, otherwise
the "sanity check" would be checking something other than the picture.

WHY ROLLOUT AND NOT RAW ATTENTION
---------------------------------
A single decoder layer's cross-attention row tells you where THAT layer's query looked, but
the value that actually reaches the LM head has passed through 12 residual blocks, each of
which re-mixes text positions (self-attention) and re-injects image information (cross-
attention). Reading layer 11 alone therefore attributes the whole prediction to one of the
twelve places the image entered. Abnar & Zuidema (2020) rollout fixes the *structural* part
of that: model every attention block as a linear mixing matrix, add the identity to account
for the residual stream, row-normalize back to a distribution, and chain-multiply.

Three specific extensions were needed here, because plain rollout is defined for a single
self-attention stack and BLIP is an encoder + cross-attending decoder:

  1. JOINT TEXT/IMAGE STATE. We track M in R^{T x (T + N_img)}: row t is a distribution over
     "where the representation at text position t came from" -- either an input text position
     or an image token. Init M = [I_T | 0]. Per decoder layer:
        self :  M <- rownorm(mean_h A_self + I) @ M
        cross:  M <- 0.5 * M + 0.5 * [0_T | mean_h A_cross]
     The 0.5/0.5 is the residual assumption applied to the cross-attention sublayer
     (`hidden = LN(hidden + CrossAttnOut)`), i.e. the same "add identity, renormalize" rule
     as the self-attention step, written out for a block whose two inputs are the residual
     stream and the image. This is the honest reading of the residual connection; it does
     mean later layers carry more weight than earlier ones (layer 11 gets 0.5, layer 10 gets
     0.25, ...). That decay is a property of the residual assumption, not a bug, but it IS
     reported in the design log because it means rollout here is *not* an equal-weight
     average over layers.
  2. VISION-SIDE ROLLOUT -- IMPLEMENTED AND THEN REJECTED ON EVIDENCE. A cross-attention
     column j points at ViT *output* token j, not at input patch j; after 12 ViT layers,
     output token j is itself a mixture of all patches. The textbook fix is to also roll out
     the vision encoder's self-attention (R_v = prod rownorm(A_v + I)) and compose
     P = M[:, T:] @ R_v. That variant is `rollout_vit` here. Measured on a real test example
     it collapses to near-uniform: mean per-token entropy 6.329 nats against a uniform 6.356,
     peak/uniform ratio 1.89 (vs. ~38 for the decoder-only rollout), 19.7% of the mass parked
     on the CLS column, and the pre/post split identical to 4 decimal places for EVERY
     section (0.5064 / 0.4935 for BENCANA, BANGUNAN, JALAN, ... alike). It is the textbook
     "looks like a heatmap, explains nothing" artifact -- ViT rollout's identity-plus-
     renormalize assumption smears completely over 12 dense 577x577 layers. It is kept in
     the code, and reported, precisely so the rejection is inspectable rather than a silent
     choice. The DEFAULT variant is `rollout` = the 12-decoder-layer joint rollout only.
  3. CLS COLUMN. Vision token 0 is the ViT CLS token, which is not a spatial location. Its
     mass is measured and reported separately, then dropped, and the remaining 576 patch
     columns are renormalized to a distribution over the 24x24 grid.

Four map variants (rollout / rollout_vit / last-layer-raw / mean-of-layers-raw) are
implemented, because the ONLY defensible way to claim rollout is the right choice is to
score them against each other on a faithfulness test -- see xai_faithfulness_check.py.
Nothing here assumes rollout wins.

HOW THE ATTENTION IS ACTUALLY OBTAINED (transformers 5.9.0 -- verified, not assumed)
------------------------------------------------------------------------------------
`BlipForConditionalGeneration.forward(..., output_attentions=True)` returns an output object
whose `.attentions` field holds the *VISION* encoder's self-attention and whose
`.cross_attentions` / `.decoder_attentions` are **None** -- BLIP's top-level output dataclass
simply has no field for the text decoder's attentions (see modeling_blip.py: the returned
`BlipForConditionalGenerationModelOutput` is populated from `vision_outputs` only). Building
a "cross-attention heatmap" on `out.attentions` would silently plot 577x577 IMAGE-TO-IMAGE
attention -- it has a plausible shape and would render fine. That is the exact failure mode
this file is written to avoid, so `_verify_plumbing()` asserts it.

The working route is to call the text decoder directly:
    base.text_decoder(input_ids=..., encoder_hidden_states=image_embeds,
                      encoder_attention_mask=ones, output_attentions=True, use_cache=False)
-> `.cross_attentions` = 12 tensors of (B, 12 heads, T, 577). This works because
`BlipTextPreTrainedModel._can_record_outputs` registers an OutputRecorder on
`BlipTextSelfAttention` output index 1 filtered by layer name ".crossattention.".

`attn_implementation="eager"` was NOT required: in transformers 5.9.0 both
`BlipAttention.forward` and `BlipTextSelfAttention.forward` are hand-written eager softmax
implementations with no SDPA/flash branch, and the loaded model reports
`config._attn_implementation == "eager"` already. `_verify_plumbing()` asserts that too, so
the day a transformers upgrade adds an SDPA path, this fails loudly instead of returning
None.

TOKEN ALIGNMENT (the other way a heatmap silently lies)
-------------------------------------------------------
The decoder is causal: the hidden state at input position i predicts token i+1. So the
cross-attention row i is the attention used while EMITTING token i+1, not while "being"
token i. Off-by-one here produces maps that look fine and are attributed to the wrong word.
`_verify_plumbing()` re-runs the greedy-generated sequence teacher-forced and asserts
argmax(logits[:, i]) == generated[i+1] for every position -- if the alignment convention
were wrong, that assert would still pass, so we additionally hard-code the shift in ONE
place (`TokenMaps.emitted_ids` / `.maps` are both indexed by emitted-token position).

Usage:
    python scripts/xai_attention_rollout.py --mode selftest
    python scripts/xai_attention_rollout.py --mode examples --n 12
    python scripts/xai_attention_rollout.py --mode aggregate --n 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.build_model import SECTION_HEADERS, NL_TOKEN, DEC_TOKEN  # noqa: E402
from scripts.build_model import load_trainable_state  # noqa: E402
from scripts.dataset import DisasterCaptionDataset, ImageSpec, make_composite, load_image  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints", "full_run_v3_vision", "best")
PRED_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
OUT_DIR = os.path.join(ROOT, "results", "xai")
EXAMPLES_DIR = os.path.join(ROOT, "results", "xai_examples")

MAX_NEW_TOKENS = 300
GRID = 24                 # 384 / 16
N_PATCHES = GRID * GRID   # 576
PRE_COLS = GRID // 2      # patch columns 0..11 = PRE half, 12..23 = POST half

DEFAULT_VARIANT = "rollout"
MAP_VARIANTS = ("rollout", "rollout_vit", "last_layer", "mean_layers")


# ======================================================================================
# section parsing -- mirrors compare_beam_vs_greedy.get_section() on the TEXT side, but the
# aggregation itself has to happen on the TOKEN side, so the boundaries are found by the
# header tokens' ids. The headers are single added tokens (build_model.SECTION_HEADERS are
# registered via tok.add_tokens with normalized=False), which is what makes this exact.
# ======================================================================================

def get_section(text: str, name: str):
    """Verbatim from scripts/compare_beam_vs_greedy.py -- kept identical on purpose."""
    m = re.search(rf"{name}:\s*(.+)", text)
    return m.group(1).strip() if m else None


def get_bencana(text: str):
    v = get_section(text, "BENCANA")
    return v.lower() if v else None


def building_no_damage(text: str) -> bool:
    m = re.search(r"BANGUNAN:\s*(.+?)(?:\n[A-Z_]+:|$)", text, re.DOTALL)
    b = (m.group(1) if m else "").lower()
    return ("tidak ada kerusakan" in b) or ("tidak menunjukkan kerusakan" in b)


# ======================================================================================
# rollout maths
# ======================================================================================

def _rownorm(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(-1, keepdim=True).clamp_min(1e-12)


def vision_rollout(vision_attn: list[torch.Tensor]) -> torch.Tensor:
    """R_v[j, k] = relevance of ViT INPUT token k to ViT OUTPUT token j.  (577, 577)"""
    n = vision_attn[0].shape[-1]
    eye = torch.eye(n, device=vision_attn[0].device, dtype=torch.float32)
    r = eye
    for a in vision_attn:
        a_bar = a[0].float().mean(0)              # (577, 577), heads averaged
        r = _rownorm(a_bar + eye) @ r
    return r


def decoder_joint_rollout(self_attn: list[torch.Tensor],
                          cross_attn: list[torch.Tensor]) -> torch.Tensor:
    """M[t, T:] -> distribution over the 577 ViT OUTPUT tokens for text position t.

    Returns only the image block, (T, 577), already row-normalized.
    """
    t = self_attn[0].shape[-1]
    n_img = cross_attn[0].shape[-1]
    dev = self_attn[0].device
    eye = torch.eye(t, device=dev, dtype=torch.float32)
    m_txt = eye.clone()                                    # (T, T)
    m_img = torch.zeros(t, n_img, device=dev, dtype=torch.float32)

    for s, c in zip(self_attn, cross_attn):
        s_bar = _rownorm(s[0].float().mean(0) + eye)       # (T, T)
        m_txt = s_bar @ m_txt
        m_img = s_bar @ m_img
        c_bar = _rownorm(c[0].float().mean(0))             # (T, 577)
        # residual sublayer: half the outgoing representation is the incoming stream,
        # half is the cross-attention read of the image
        m_txt = 0.5 * m_txt
        m_img = 0.5 * m_img + 0.5 * c_bar
    return _rownorm(m_img)


def build_maps(vision_attn, self_attn, cross_attn, variant: str) -> tuple[torch.Tensor, float]:
    """-> (maps over the 576 patches, (T, 576) row-normalized ; CLS mass fraction)."""
    if variant == "rollout_vit":
        img = decoder_joint_rollout(self_attn, cross_attn) @ vision_rollout(vision_attn)
    elif variant == "rollout":
        img = decoder_joint_rollout(self_attn, cross_attn)
    elif variant == "last_layer":
        img = _rownorm(cross_attn[-1][0].float().mean(0))
    elif variant == "mean_layers":
        img = _rownorm(torch.stack([c[0].float().mean(0) for c in cross_attn]).mean(0))
    else:
        raise ValueError(f"unknown map variant {variant!r}")
    cls_mass = float(img[:, 0].mean().item())
    patches = _rownorm(img[:, 1:])                          # drop CLS, renormalize
    return patches, cls_mass


def token_selectivity(maps: torch.Tensor) -> float:
    """Mean pairwise Pearson r between the per-token maps of ONE example.

    This is the single most important sanity number in this file. If it is ~1.0, every
    generated word is being explained by the SAME picture, and any per-word or per-section
    heatmap is decoration no matter how sharp it looks. Measured, not assumed.
    """
    m = maps.float().numpy() if maps.is_cpu else maps.float().cpu().numpy()
    if len(m) < 2:
        return float("nan")
    m = m - m.mean(1, keepdims=True)
    m = m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)
    c = m @ m.T
    iu = np.triu_indices(len(c), 1)
    return float(c[iu].mean())


def pre_post_split(maps: torch.Tensor) -> tuple[float, float]:
    """maps (K, 576) already row-normalized -> (pre_fraction, post_fraction), averaged over K."""
    g = maps.reshape(-1, GRID, GRID)
    pre = g[:, :, :PRE_COLS].sum(dim=(1, 2))
    post = g[:, :, PRE_COLS:].sum(dim=(1, 2))
    return float(pre.mean().item()), float(post.mean().item())


# ======================================================================================
# engine
# ======================================================================================

@dataclass
class TokenMaps:
    """Everything about ONE example, with maps indexed by EMITTED-token position."""
    index: int
    emitted_ids: list[int]           # generated[1:]  (generated[0] is [DEC])
    emitted_tokens: list[str]
    caption: str
    maps: dict[str, torch.Tensor]    # variant -> (n_emitted, 576) on cpu, row-normalized
    cls_mass: dict[str, float]
    section_of_token: list[str | None]
    is_content: list[bool]
    teacher_forced_match: float
    meta: dict[str, Any] = field(default_factory=dict)


class XaiEngine:
    def __init__(self, device: str = "cuda", verbose: bool = True):
        self.device = device
        if verbose:
            print(f"[load] {CKPT_DIR}")
        self.bundle = load_trainable_state(
            CKPT_DIR, device=device, verbose=False,
            lora_r=8, lora_alpha=32, lora_dropout=0.1, max_length=384,
            vision_lora=True,
        )
        self.model = self.bundle.model
        self.model.eval()
        base = getattr(self.model, "base_model", self.model)
        self.base = getattr(base, "model", base)
        self.codec = self.bundle.codec
        self.tok = self.bundle.tokenizer
        self.spec = ImageSpec.from_processor(self.bundle.image_processor)
        self.header_ids = {self.tok.convert_tokens_to_ids(h): h for h in SECTION_HEADERS}
        self.nl_id = self.tok.convert_tokens_to_ids(NL_TOKEN)
        self.dec_id = self.tok.convert_tokens_to_ids(DEC_TOKEN)
        self.sep_id = self.tok.sep_token_id
        self.colon_id = self.tok.convert_tokens_to_ids(":")
        for tid, name in list(self.header_ids.items()):
            if tid is None or tid == self.tok.unk_token_id:
                raise RuntimeError(f"section header {name!r} is not a single token id -- "
                                   f"token-level section aggregation would be wrong")

    # -- low level ---------------------------------------------------------------------
    @torch.no_grad()
    def image_embeds(self, pixel_values: torch.Tensor, want_attn: bool = False):
        out = self.base.vision_model(pixel_values=pixel_values, output_attentions=want_attn)
        return out.last_hidden_state, (out.attentions if want_attn else None)

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, max_new_tokens: int = MAX_NEW_TOKENS):
        return self.model.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens,
                                   num_beams=1, do_sample=False)

    @torch.no_grad()
    def decoder_attentions(self, image_embeds: torch.Tensor, input_ids: torch.Tensor):
        mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        out = self.base.text_decoder(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=mask,
            output_attentions=True,
            use_cache=False,
        )
        if out.cross_attentions is None:
            raise RuntimeError("text_decoder returned cross_attentions=None -- refusing to "
                               "produce a heatmap from nothing")
        return out.logits, list(out.attentions), list(out.cross_attentions)

    # -- the whole pipeline for one example --------------------------------------------
    @torch.no_grad()
    def explain(self, pixel_values: torch.Tensor, index: int = -1,
                variants: tuple[str, ...] = MAP_VARIANTS,
                forced_ids: torch.Tensor | None = None) -> TokenMaps:
        """pixel_values (1,3,384,384) on device. If forced_ids is given, that sequence is
        teacher-forced instead of generating (used by the image-sensitivity control)."""
        pv = pixel_values
        seq = forced_ids if forced_ids is not None else self.generate(pv)[0]
        seq = seq.to(self.device)
        if seq[0].item() != self.dec_id:
            # BLIP overwrites position 0 with bos; assert the contract rather than trust it
            raise RuntimeError(f"generated seq[0]={seq[0].item()} != [DEC]={self.dec_id}")

        img, vattn = self.image_embeds(pv, want_attn=("rollout_vit" in variants))
        inp = seq[:-1].unsqueeze(0)                       # drop the last token: it predicts nothing
        logits, sattn, cattn = self.decoder_attentions(img, inp)

        # alignment proof: teacher-forced argmax must reproduce the greedy sequence
        pred = logits[0].argmax(-1)                       # pred[i] is the model's token for i+1
        tgt = seq[1:]
        # exact integer count, then a float64 ratio. `(pred == tgt).float().mean()` returns
        # 0.99999994 (= 1 - 2^-24) for a PERFECT match because torch's float32 mean is
        # sum * (1/N) and 1/N is rounded -- which would look like a real 1-token misalignment
        # in the logs and is exactly the kind of near-miss that gets waved away.
        n_mismatch = int((pred != tgt).sum().item())
        match = 1.0 - n_mismatch / len(tgt)

        maps, cls_mass = {}, {}
        for v in variants:
            m, cm = build_maps(vattn, sattn, cattn, v)
            maps[v] = m.cpu()                             # (T-1, 576) rows == emitted positions
            cls_mass[v] = cm

        emitted = tgt.tolist()
        toks = self.tok.convert_ids_to_tokens(emitted)
        sec, content = self._sections(emitted)
        return TokenMaps(
            index=index, emitted_ids=emitted, emitted_tokens=toks,
            caption=self.codec.decode(seq), maps=maps, cls_mass=cls_mass,
            section_of_token=sec, is_content=content, teacher_forced_match=match,
            meta={"teacher_forced_mismatched_tokens": n_mismatch, "n_target_tokens": len(tgt)},
        )

    def _sections(self, emitted: list[int]):
        """Assign each emitted token to a section; flag which are CONTENT (not header/':'/[NL])."""
        sec: list[str | None] = []
        content: list[bool] = []
        cur = None
        just_header = False
        for tid in emitted:
            if tid in self.header_ids:
                cur = self.header_ids[tid]
                sec.append(cur)
                content.append(False)
                just_header = True
                continue
            is_struct = (tid in (self.nl_id, self.sep_id, self.tok.pad_id if hasattr(self.tok, "pad_id") else self.tok.pad_token_id)
                         or (just_header and tid == self.colon_id))
            sec.append(cur)
            content.append(cur is not None and not is_struct)
            just_header = False
        return sec, content

    # -- aggregation --------------------------------------------------------------------
    def section_stats(self, tm: TokenMaps, variant: str = "rollout") -> dict[str, dict]:
        maps = tm.maps[variant]
        out = {}
        for name in SECTION_HEADERS + ["__ALL__"]:
            if name == "__ALL__":
                idx = [i for i, c in enumerate(tm.is_content) if c]
            else:
                idx = [i for i, (s, c) in enumerate(zip(tm.section_of_token, tm.is_content))
                       if c and s == name]
            if not idx:
                continue
            pre, post = pre_post_split(maps[idx])
            out[name] = {"n_tokens": len(idx), "pre": pre, "post": post}
        return out

    def section_mean_map(self, tm: TokenMaps, section: str, variant: str = "rollout"):
        if section == "__ALL__":
            idx = [i for i, c in enumerate(tm.is_content) if c]
        else:
            idx = [i for i, (s, c) in enumerate(zip(tm.section_of_token, tm.is_content))
                   if c and s == section]
        if not idx:
            return None
        return tm.maps[variant][idx].mean(0)


# ======================================================================================
# image helpers
# ======================================================================================

def composite_pil(engine: XaiEngine, rec: dict):
    pre = load_image(rec["pre_image_path"])
    post = load_image(rec["post_image_path"])
    return make_composite(pre, post, engine.spec)


def _pad_fill(pixel_values: torch.Tensor, spec: ImageSpec) -> torch.Tensor:
    """The uint8 pad colour, normalized, as a (1,3,1,1) broadcast tensor.

    pad_color is the uint8 colour that normalizes to ~0 (see dataset.ImageSpec.pad_color) --
    i.e. the least disruptive fill available, which is what a masking/occlusion test needs:
    the signal removed should be the image content, not "a big black rectangle appeared".
    Shared by `mask_patches` (hard fill) and `blend_patches` (continuous RISE fill) so there
    is exactly one place that defines what "neutral" means for this model.
    """
    return torch.tensor([(c / 255.0 - m) / s for c, m, s in
                         zip(spec.pad_color, spec.mean, spec.std)],
                        dtype=pixel_values.dtype, device=pixel_values.device).view(1, 3, 1, 1)


def mask_patches(pixel_values: torch.Tensor, patch_idx, spec: ImageSpec) -> torch.Tensor:
    """Gray-fill the given flat patch indices (0..575) with the processor's mean colour."""
    out = pixel_values.clone()
    fill = _pad_fill(out, spec)
    for p in patch_idx:
        r, c = divmod(int(p), GRID)
        out[:, :, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = fill
    return out


def blend_patches(pixel_values: torch.Tensor, patch_alpha: torch.Tensor, spec: ImageSpec) -> torch.Tensor:
    """Continuous per-patch alpha-blend toward `spec.pad_color` -- RISE's analogue of
    `mask_patches`'s hard all-or-nothing fill.

    pixel_values : (1,3,384,384) ONE source image, broadcast across the B masks.
    patch_alpha  : (B, GRID, GRID) or (B, N_PATCHES) in [0,1]; 1 = keep original content,
                   0 = fully replaced by pad_color. Values are constant within a patch (the
                   RISE mask is defined at 24x24 patch resolution, see `_generate_rise_masks`),
                   so upsampling to pixel space is a plain 16x nearest-neighbour repeat, not
                   an interpolation that would blur patch boundaries.
    -> (B, 3, 384, 384)
    """
    if patch_alpha.dim() == 2 and patch_alpha.shape[-1] == N_PATCHES:
        patch_alpha = patch_alpha.reshape(-1, GRID, GRID)
    b = patch_alpha.shape[0]
    alpha = patch_alpha.to(device=pixel_values.device, dtype=pixel_values.dtype).unsqueeze(1)
    alpha_full = F.interpolate(alpha, size=(GRID * 16, GRID * 16), mode="nearest")  # (B,1,384,384)
    fill = _pad_fill(pixel_values, spec)
    base = pixel_values.expand(b, -1, -1, -1)
    return alpha_full * base + (1.0 - alpha_full) * fill


# ======================================================================================
# RISE (Petsiuk, Das & Saenko 2018) -- perturbation-based explanation, PER TOKEN
# ======================================================================================
# Rollout (above) is an attention-flow ESTIMATE: it never perturbs the image, it only
# re-derives, algebraically, "where did attention mass go". The token_selectivity finding in
# mode_selftest (r=0.9997 between different words' maps) could in principle be a property of
# attention itself rather than of the model's actual sensitivity to the image -- the only way
# to tell those apart is a method that never looks at attention weights at all. RISE instead
# ablates random subsets of the INPUT and measures the effect on the OUTPUT directly: an
# importance map built this way is faithful by construction (it IS a measurement of causal
# effect, via many correlated random probes), so if IT also comes back token-invariant, that
# is much stronger evidence that the model's caption really doesn't read different sections
# off different regions -- not just that rollout is a bad estimator.
#
# RISE_DEFAULTS mirror Petsiuk et al.'s paper defaults (N=1000-8000 masks, s=7-8, p1=0.5); s=8
# and N=1000 are picked here as the starting point given the 8GB VRAM / single-GPU budget --
# see the timing report printed by `compute_rise_map` before committing to a full run.

RISE_MASK_GRID_DEFAULT = 8
RISE_N_MASKS_DEFAULT = 1000
RISE_P_KEEP_DEFAULT = 0.5
RISE_BATCH_DEFAULT = 16


def _generate_rise_masks(n_masks: int, mask_grid: int, target_size: int, p_keep: float,
                          seed: int) -> torch.Tensor:
    """Standard RISE mask generation (Petsiuk et al. 2018, Algorithm 1).

    Sample a COARSE (mask_grid x mask_grid) i.i.d. Bernoulli(p_keep) grid, bilinear-upsample
    past target_size, then crop with a random per-mask (x, y) offset. The offset is what turns
    hard mask_grid x mask_grid blocks into smoothly-varying continuous alpha: without it every
    mask's cell boundaries would land on the exact same 3 columns/rows, and the resulting maps
    would inherit hard edges at those fixed positions instead of averaging them out.

    -> (n_masks, target_size, target_size) float32 in [0, 1], on CPU.
    """
    rng = np.random.default_rng(seed)
    cell = int(np.ceil(target_size / mask_grid))
    up = (mask_grid + 1) * cell
    grid = (rng.random((n_masks, mask_grid, mask_grid)) < p_keep).astype(np.float32)
    grid_t = torch.from_numpy(grid).unsqueeze(1)                          # (N,1,s,s)
    up_t = F.interpolate(grid_t, size=(up, up), mode="bilinear", align_corners=False)
    off_x = rng.integers(0, cell, size=n_masks)
    off_y = rng.integers(0, cell, size=n_masks)
    masks = torch.empty(n_masks, target_size, target_size, dtype=torch.float32)
    for i in range(n_masks):
        x, y = int(off_x[i]), int(off_y[i])
        masks[i] = up_t[i, 0, x:x + target_size, y:y + target_size]
    return masks


@torch.no_grad()
def compute_rise_map(engine: "XaiEngine", example: dict, n_masks: int = RISE_N_MASKS_DEFAULT,
                     mask_grid: int = RISE_MASK_GRID_DEFAULT, p_keep: float = RISE_P_KEEP_DEFAULT,
                     batch_size: int = RISE_BATCH_DEFAULT, seed: int = 0,
                     target_ids: list[int] | None = None) -> torch.Tensor:
    """RISE, extended to produce ONE importance map PER EMITTED TOKEN instead of one map for
    one scalar score.

    importance[t, patch] = E_mask[ mask[patch] * log p(token_t | image (dot) mask) ]

    -- Petsiuk et al.'s eq. (2) (S = E[M * f(I (dot) M)]) applied to the per-token
    teacher-forced log-probability instead of a single class probability. The paper's extra
    1/(N*p_keep) normalization is a single scalar shared by every patch and every token for a
    fixed mask distribution, so it cannot change the argsort ranking `mode_delins` reads off
    this map; it is left out on purpose rather than silently applied, and this docstring is
    the record of that choice.

    Cost design: for EACH of the n_masks perturbed images, ONE teacher-forced decoder forward
    pass (no autoregressive generation, no gradients) yields log p(token_t | masked image) for
    EVERY token t in the caption simultaneously -- so the cost is O(n_masks) forward passes,
    not O(n_masks * n_tokens). The teacher-forcing target is the caption the model ALREADY
    produced on the UNPERTURBED image -- RISE explains that fixed prediction's causal
    dependence on the image, not whatever new guess a perturbed model might make with fresh
    autoregressive decoding (which would also be ~n_tokens times slower for no benefit).

    TOKEN SOURCE -- read this before passing `example["generated_caption"]` and nothing else.
    Two ways to supply the target sequence, in order of preference:
      1. `target_ids`: the FULL [DEC]-prefixed id sequence (i.e. `[engine.dec_id] + emitted_ids`
         from an `XaiEngine.explain()` TokenMaps), used VERBATIM -- no tokenizer round-trip.
         This is the only option that GUARANTEES the returned map's row count matches another
         caller's `tm.emitted_ids` exactly, which matters whenever the caller needs to index
         both by the same `content`/`is_content` mask (e.g. mode_delins).
      2. `example["generated_caption"]` (text), re-tokenized via `engine.codec.encode()` --
         used only if `target_ids` is None. THIS IS NOT GUARANTEED TOKEN-COUNT-STABLE: decode
         -> encode is a TEXT round-trip, and `IndoReportCodec`'s detokenization rules (e.g.
         collapsing repeated newlines) can merge or drop tokens that survived in the original
         id sequence. Measured failure: example index 471 in this project's 12-example set
         produced 149 rows this way against a live `tm.emitted_ids` of length 150. Safe to use
         ONLY when the caller has no id sequence to hand it (e.g. driving RISE straight off
         `results/predictions_test_vision.jsonl` with no accompanying `XaiEngine.explain()`
         call) and does not need row-for-row alignment with anything else.

    example must have: pre_image_path, post_image_path (used to build the composite image via
    `composite_pil`), and generated_caption (only read when target_ids is None).
    Returns (n_emitted_tokens, N_PATCHES) float32 tensor on CPU, UNNORMALIZED (rows are NOT a
    probability distribution -- they are signed importance scores, since log-probabilities are
    always <= 0.  Higher = more evidence this patch was PRESENT when this token got a higher
    log-prob under masking; callers that need a ranking should argsort, not renormalize).
    """
    device = engine.device
    img = composite_pil(engine, example)
    pv = engine.spec.to_tensor(img).unsqueeze(0).to(device)

    if target_ids is not None:
        seq = torch.tensor(target_ids, device=device)
    else:
        seq_ids = engine.codec.encode(example["generated_caption"])
        seq = torch.tensor(seq_ids, device=device)
    if seq[0].item() != engine.dec_id:
        raise RuntimeError(f"seq[0]={seq[0].item()} != [DEC]={engine.dec_id} -- "
                           f"target sequence does not have the expected [DEC] prefix")
    inp = seq[:-1]                                    # (n_emitted,) fed to the decoder
    tgt = seq[1:]                                      # (n_emitted,) teacher-forcing targets
    n_emitted = int(tgt.shape[0])

    masks = _generate_rise_masks(n_masks, mask_grid, GRID, p_keep, seed)   # (n_masks,GRID,GRID) cpu
    importance = torch.zeros(n_emitted, N_PATCHES, dtype=torch.float32, device=device)

    for start in range(0, n_masks, batch_size):
        batch_masks = masks[start:start + batch_size].to(device)          # (B,GRID,GRID)
        b = batch_masks.shape[0]
        blended = blend_patches(pv, batch_masks, engine.spec)              # (B,3,384,384)
        img_embeds, _ = engine.image_embeds(blended, want_attn=False)      # (B,577,768)
        enc_mask = torch.ones(img_embeds.shape[:-1], dtype=torch.long, device=device)
        inp_b = inp.unsqueeze(0).expand(b, -1)
        out = engine.base.text_decoder(
            input_ids=inp_b,
            attention_mask=torch.ones_like(inp_b),
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=enc_mask,
            output_attentions=False,
            use_cache=False,
        )
        logprobs = F.log_softmax(out.logits.float(), dim=-1)               # (B, n_emitted, vocab)
        tgt_b = tgt.unsqueeze(0).expand(b, -1)
        token_logp = logprobs.gather(-1, tgt_b.unsqueeze(-1)).squeeze(-1)  # (B, n_emitted)
        flat_masks = batch_masks.reshape(b, -1)                            # (B, N_PATCHES)
        importance += torch.einsum("bt,bp->tp", token_logp, flat_masks)
        del blended, img_embeds, out, logprobs, token_logp, flat_masks, batch_masks

    return (importance / n_masks).cpu()


# ======================================================================================
# plotting
# ======================================================================================

def _upsample(map576: torch.Tensor) -> np.ndarray:
    g = map576.reshape(1, 1, GRID, GRID).float()
    up = F.interpolate(g, size=(384, 384), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def _overlay(ax, base_rgb: np.ndarray, map576: torch.Tensor, title: str):
    heat = _upsample(map576)
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)
    ax.imshow(base_rgb)
    ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    ax.axvline(192, color="white", lw=1.5, ls="--")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def plot_example(engine, tm, base_rgb, out_png, variant="rollout"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sections = ["__ALL__"] + [s for s in SECTION_HEADERS
                              if engine.section_mean_map(tm, s, variant) is not None]
    n = len(sections) + 1
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.75 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(base_rgb)
    axes[0].axvline(192, color="white", lw=1.5, ls="--")
    axes[0].set_title("composite input\nLEFT = PRE | RIGHT = POST", fontsize=7)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    stats = engine.section_stats(tm, variant)
    for k, s in enumerate(sections):
        m = engine.section_mean_map(tm, s, variant)
        st = stats.get(s, {})
        label = "ALL CONTENT TOKENS" if s == "__ALL__" else s
        _overlay(axes[k + 1], base_rgb, m,
                 f"{label}  (n={st.get('n_tokens', 0)} tok)\n"
                 f"pre {st.get('pre', float('nan')):.1%} | post {st.get('post', float('nan')):.1%}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"attention rollout ({variant}) -- test index {tm.index}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.0)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_tokens(engine, tm, base_rgb, out_png, variant="rollout", max_tokens=8):
    """Per-token maps for the first content tokens of BENCANA and BANGUNAN."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = []
    for sec in ("BENCANA", "BANGUNAN"):
        idx = [i for i, (s, c) in enumerate(zip(tm.section_of_token, tm.is_content))
               if c and s == sec]
        picks += idx[:max_tokens // 2]
    if not picks:
        return False
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(picks):
        m = tm.maps[variant][i]
        pre, post = pre_post_split(m.unsqueeze(0))
        _overlay(axes[k], base_rgb, m,
                 f"[{tm.section_of_token[i]}] token {i}: {tm.emitted_tokens[i]!r}\n"
                 f"pre {pre:.0%} | post {post:.0%}")
    for ax in axes[len(picks):]:
        ax.axis("off")
    content = [i for i, c in enumerate(tm.is_content) if c]
    r = token_selectivity(tm.maps[variant][content])
    fig.suptitle(f"per-token rollout ({variant}) -- test index {tm.index}\n"
                 f"WARNING: mean pairwise correlation between this model's per-token maps "
                 f"= {r:.4f} -- the panels below are near-identical BY MEASUREMENT, not by "
                 f"plotting error (see results/xai/xai_selftest.json)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.0)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


# ======================================================================================
# sample selection -- deterministic, stratified, NOT hand-picked
# ======================================================================================

def load_predictions():
    with open(PRED_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def select_examples(n: int, seed: int = 0) -> list[dict]:
    """Stratify by ground-truth disaster type, then alternate correct/incorrect BENCANA.

    Deterministic given (n, seed) and the prediction file -- no image was ever looked at
    before choosing, which is the whole point: hand-picking pretty heatmaps would make the
    interpretability check meaningless.
    """
    rows = load_predictions()
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        t = get_bencana(r["ground_truth_id"])
        if t:
            by_type.setdefault(t, []).append(r)
    types = sorted(by_type, key=lambda t: (-len(by_type[t]), t))
    rng = random.Random(seed)
    picked, k = [], 0
    want_correct = True
    while len(picked) < n and k < 50:
        for t in types:
            if len(picked) >= n:
                break
            pool = [r for r in by_type[t]
                    if (get_bencana(r["generated_caption"]) == t) == want_correct
                    and r["index"] not in {p["index"] for p in picked}]
            if not pool:
                continue
            picked.append(rng.choice(pool))
            want_correct = not want_correct
        k += 1
    return sorted(picked, key=lambda r: r["index"])


# ======================================================================================
# modes
# ======================================================================================

def mode_selftest(args):
    import transformers
    eng = XaiEngine(device=args.device)
    res: dict[str, Any] = {
        "checkpoint": CKPT_DIR,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attn_implementation": eng.base.config._attn_implementation,
        "text_decoder_layers": eng.base.text_decoder.config.num_hidden_layers,
        "text_decoder_heads": eng.base.text_decoder.config.num_attention_heads,
        "vision_layers": eng.base.vision_model.config.num_hidden_layers,
        "vision_heads": eng.base.vision_model.config.num_attention_heads,
        "vision_image_size": eng.base.vision_model.config.image_size,
        "vision_patch_size": eng.base.vision_model.config.patch_size,
    }
    ds = DisasterCaptionDataset("test", bundle=eng.bundle, drop_unreadable=True)
    a, b = ds[0], ds[7]
    pv_a = a["pixel_values"].unsqueeze(0).to(args.device)
    pv_b = b["pixel_values"].unsqueeze(0).to(args.device)

    # --- 1. the trap: top-level output_attentions gives VISION attention, not cross ---
    with torch.no_grad():
        top = eng.model(pixel_values=pv_a,
                        input_ids=a["input_ids"].unsqueeze(0).to(args.device)[:, :32],
                        attention_mask=torch.ones(1, 32, dtype=torch.long, device=args.device),
                        output_attentions=True)
    res["toplevel_output_attentions"] = {
        "cross_attentions_is_none": getattr(top, "cross_attentions", None) is None,
        "attentions_shape": list(top.attentions[0].shape) if getattr(top, "attentions", None) else None,
        "attentions_is_vision_self_attention": (
            getattr(top, "attentions", None) is not None
            and top.attentions[0].shape[-1] == top.attentions[0].shape[-2] == 577
        ),
    }

    # --- 2. the working route + alignment ---
    t0 = time.time()
    tm_a = eng.explain(pv_a, index=int(a["index"]))
    res["explain_seconds_per_example"] = round(time.time() - t0, 2)
    res["example_a"] = {
        "index": int(a["index"]),
        "n_emitted_tokens": len(tm_a.emitted_ids),
        "teacher_forced_argmax_match_vs_greedy": tm_a.teacher_forced_match,
        "teacher_forced_mismatched_tokens": tm_a.meta["teacher_forced_mismatched_tokens"],
        "cls_mass": tm_a.cls_mass,
    }

    # --- 3. non-uniformity: a uniform map would be a silent failure ---
    uni = 1.0 / N_PATCHES
    res["non_uniformity"] = {}
    for v in MAP_VARIANTS:
        m = tm_a.maps[v]
        ent = -(m.clamp_min(1e-12) * m.clamp_min(1e-12).log()).sum(-1).mean().item()
        res["non_uniformity"][v] = {
            "mean_entropy_nats": ent,
            "uniform_entropy_nats": float(np.log(N_PATCHES)),
            "max_over_uniform_ratio": float((m.max(-1).values.mean() / uni).item()),
        }

    # --- 4. IMAGE SENSITIVITY: same token sequence, different image -> map must change ---
    forced = torch.tensor([eng.dec_id] + tm_a.emitted_ids, device=args.device)
    tm_b_forced = eng.explain(pv_b, index=int(b["index"]), forced_ids=forced)
    sens = {}
    for v in MAP_VARIANTS:
        ma = tm_a.maps[v].flatten().numpy()
        mb = tm_b_forced.maps[v].flatten().numpy()
        l1 = float(np.abs(tm_a.maps[v] - tm_b_forced.maps[v]).sum(-1).mean().item())
        sens[v] = {
            "mean_L1_distance_per_token": l1,     # 0 = identical, 2 = disjoint supports
            "pearson_r": float(np.corrcoef(ma, mb)[0, 1]),
        }
    res["image_sensitivity_same_tokens_different_image"] = sens

    # --- 5. TOKEN SELECTIVITY: is the map actually a function of the WORD, or only of the
    #        image? Per-head / per-layer, because head-averaging could plausibly be what
    #        washes out a token-selective head (the classic "wrong head" failure).
    content = [i for i, c in enumerate(tm_a.is_content) if c]
    res["token_selectivity"] = {v: token_selectivity(tm_a.maps[v][content]) for v in MAP_VARIANTS}
    img_a, _ = eng.image_embeds(pv_a)
    with torch.no_grad():
        _, _, cattn_a = eng.decoder_attentions(
            img_a, torch.tensor([eng.dec_id] + tm_a.emitted_ids, device=args.device)[:-1].unsqueeze(0))
    per_layer_head = {}
    for L, c in enumerate(cattn_a):
        A = c[0].float().cpu()[:, content, 1:]
        rs = [token_selectivity(_rownorm(A[h])) for h in range(A.shape[0])]
        per_layer_head[f"layer_{L}"] = {
            "head_avg": token_selectivity(_rownorm(A.mean(0))),
            "min_over_heads": float(np.min(rs)), "argmin_head": int(np.argmin(rs)),
            "max_over_heads": float(np.max(rs)),
        }
    res["token_selectivity_per_layer_per_head_raw_cross_attention"] = per_layer_head
    res["token_selectivity_note"] = (
        "mean pairwise Pearson r between per-token patch maps within one example; "
        "1.0 => every generated word gets an identical heatmap")

    # --- 6. VARIANT AGREEMENT: how much does rollout actually differ from raw last layer?
    #        The 0.5 residual weighting means layer 11 contributes half the image mass, so
    #        rollout may be little more than a smoothed last layer -- claim it only if measured.
    agree = {}
    for v in MAP_VARIANTS:
        if v == "rollout":
            continue
        a = tm_a.maps["rollout"][content].numpy()
        b = tm_a.maps[v][content].numpy()
        rs = [float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(len(a))]
        agree[f"rollout_vs_{v}"] = {
            "mean_per_token_pearson_r": float(np.mean(rs)),
            "mean_L1_per_token": float(np.abs(a - b).sum(-1).mean()),
        }
    res["variant_agreement"] = agree

    # --- 7. section token counts sane? ---
    res["example_a_sections"] = eng.section_stats(tm_a)
    res["example_a_sections_rollout_vit"] = eng.section_stats(tm_a, "rollout_vit")

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "xai_selftest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[done] {p}")

    # hard failures -> loud
    if not res["toplevel_output_attentions"]["cross_attentions_is_none"]:
        print("[NOTE] top-level cross_attentions is no longer None -- re-read the docstring.")
    if tm_a.teacher_forced_match < 0.999:
        raise RuntimeError(f"teacher-forced argmax matches greedy only "
                           f"{tm_a.teacher_forced_match:.3%} -- token alignment is NOT safe")
    for v, s in sens.items():
        if s["mean_L1_distance_per_token"] < 0.05:
            print(f"[WARN] variant {v!r} barely changes with the image (L1={s['mean_L1_distance_per_token']:.4f})")


def mode_examples(args):
    eng = XaiEngine(device=args.device)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    picked = select_examples(args.n, seed=args.seed)
    print(f"[select] {len(picked)} examples: "
          f"{[(r['index'], get_bencana(r['ground_truth_id'])) for r in picked]}")

    ds = DisasterCaptionDataset("test", bundle=eng.bundle, drop_unreadable=True)
    # predictions_test_vision.jsonl stores the dataset index it was generated from
    manifest = []
    for r in picked:
        i = r["index"]
        item = ds[i]
        assert ds.records[i]["pre_image_path"] == r["pre_image_path"], "index/record mismatch"
        pv = item["pixel_values"].unsqueeze(0).to(args.device)
        tm = eng.explain(pv, index=i)
        base_rgb = np.asarray(composite_pil(eng, ds.records[i]))

        gt_type = get_bencana(r["ground_truth_id"])
        gen_type = get_bencana(tm.caption)
        tag = f"idx{i:04d}_{(gt_type or 'na').replace(' ', '-')}"

        from PIL import Image
        Image.fromarray(base_rgb).save(os.path.join(EXAMPLES_DIR, f"{tag}_original.png"))
        plot_example(eng, tm, base_rgb, os.path.join(EXAMPLES_DIR, f"{tag}_rollout.png"))
        plot_tokens(eng, tm, base_rgb, os.path.join(EXAMPLES_DIR, f"{tag}_tokens.png"))
        # side-by-side rollout vs raw last layer, so a reader can see they are NOT the same
        plot_example(eng, tm, base_rgb, os.path.join(EXAMPLES_DIR, f"{tag}_lastlayer_raw.png"),
                     variant="last_layer")
        extra_files = {}
        if len(manifest) == 0:
            # one worked example of the REJECTED ViT-composed rollout, so the "it goes
            # uniform" claim in the design log is visually checkable, not just asserted
            plot_example(eng, tm, base_rgb,
                         os.path.join(EXAMPLES_DIR, f"{tag}_REJECTED_rollout_vit.png"),
                         variant="rollout_vit")
            extra_files["rejected_rollout_vit"] = f"{tag}_REJECTED_rollout_vit.png"
        with open(os.path.join(EXAMPLES_DIR, f"{tag}_caption.txt"), "w", encoding="utf-8") as f:
            f.write(f"# test index {i}\n# pre : {r['pre_image_path']}\n"
                    f"# post: {r['post_image_path']} ({r.get('post_image_type')})\n"
                    f"# ground-truth BENCANA: {gt_type}\n# generated BENCANA  : {gen_type}\n"
                    f"# BENCANA correct: {gt_type == gen_type}\n\n"
                    f"=== GENERATED CAPTION ===\n{tm.caption}\n\n"
                    f"=== GROUND TRUTH ===\n{r['ground_truth_id']}\n")
        stats = {v: eng.section_stats(tm, v) for v in MAP_VARIANTS}
        manifest.append({
            "index": i, "tag": tag,
            "pre_image_path": r["pre_image_path"], "post_image_path": r["post_image_path"],
            "post_image_type": r.get("post_image_type"),
            "gt_bencana": gt_type, "gen_bencana": gen_type, "bencana_correct": gt_type == gen_type,
            "n_emitted_tokens": len(tm.emitted_ids),
            "teacher_forced_argmax_match_vs_greedy": tm.teacher_forced_match,
            "teacher_forced_mismatched_tokens": tm.meta["teacher_forced_mismatched_tokens"],
            "cls_mass": tm.cls_mass,
            "section_pre_post": stats,
            "files": {
                "original": f"{tag}_original.png",
                "rollout_sections": f"{tag}_rollout.png",
                "rollout_tokens": f"{tag}_tokens.png",
                "lastlayer_raw_sections": f"{tag}_lastlayer_raw.png",
                "caption": f"{tag}_caption.txt",
                **extra_files,
            },
        })
        print(f"  [{i}] {gt_type} -> {gen_type} | {len(tm.emitted_ids)} tok | "
              f"align {tm.teacher_forced_match:.3f}")

    p = os.path.join(EXAMPLES_DIR, "manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": CKPT_DIR, "seed": args.seed, "examples": manifest},
                  f, indent=2, ensure_ascii=False)
    print(f"[done] {len(manifest)} examples -> {EXAMPLES_DIR}")


def mode_aggregate(args):
    eng = XaiEngine(device=args.device)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_predictions()
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(rows, min(args.n, len(rows))), key=lambda r: r["index"])
    ds = DisasterCaptionDataset("test", bundle=eng.bundle, drop_unreadable=True)

    acc: dict[str, dict[str, list]] = {v: {} for v in MAP_VARIANTS}
    sel: dict[str, list] = {}
    btw: dict[str, list] = {}
    per_example = []
    t0 = time.time()
    for k, r in enumerate(sample):
        i = r["index"]
        pv = ds[i]["pixel_values"].unsqueeze(0).to(args.device)
        tm = eng.explain(pv, index=i)
        content = [j for j, c in enumerate(tm.is_content) if c]
        row = {"index": i, "gt_bencana": get_bencana(r["ground_truth_id"]),
               "gen_bencana": get_bencana(tm.caption),
               "n_tokens": len(tm.emitted_ids),
               "align": tm.teacher_forced_match,
               "align_mismatched_tokens": tm.meta["teacher_forced_mismatched_tokens"],
               "token_selectivity": {v: token_selectivity(tm.maps[v][content]) for v in MAP_VARIANTS},
               "sections": {}}
        for v in MAP_VARIANTS:
            st = eng.section_stats(tm, v)
            row["sections"][v] = {s: {"pre": d["pre"], "post": d["post"], "n": d["n_tokens"]}
                                  for s, d in st.items()}
            # how much does post-fraction move BETWEEN sections inside this one example?
            per_sec = [d["post"] for s, d in st.items() if s != "__ALL__"]
            row["sections"][v]["_between_section_post_std"] = (
                float(np.std(per_sec, ddof=1)) if len(per_sec) > 1 else 0.0)
            for s, d in st.items():
                acc[v].setdefault(s, []).append((d["pre"], d["post"], d["n_tokens"]))
            sel.setdefault(v, []).append(row["token_selectivity"][v])
            btw.setdefault(v, []).append(row["sections"][v]["_between_section_post_std"])
        per_example.append(row)
        del tm, pv
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(sample)}  ({el/(k+1):.2f}s/ex, ETA {(len(sample)-k-1)*el/(k+1)/60:.1f} min)",
                  flush=True)

    summary = {}
    for v in MAP_VARIANTS:
        summary[v] = {}
        for s, vals in acc[v].items():
            pre = np.array([x[0] for x in vals]); post = np.array([x[1] for x in vals])
            n = np.array([x[2] for x in vals], dtype=float)
            summary[v][s] = {
                "n_examples": int(len(vals)),
                "total_tokens": int(n.sum()),
                "pre_mean": float(pre.mean()), "post_mean": float(post.mean()),
                "pre_std": float(pre.std(ddof=1)) if len(pre) > 1 else 0.0,
                "post_minus_pre_mean": float((post - pre).mean()),
                "frac_examples_post_dominant": float((post > pre).mean()),
                # token-weighted (a 3-token section shouldn't count as much as a 60-token one)
                "post_mean_token_weighted": float((post * n).sum() / n.sum()),
            }
        allpost = np.array([x[1] for x in acc[v]["__ALL__"]])
        summary[v]["_diagnostics"] = {
            "mean_token_selectivity": float(np.mean(sel[v])),
            "min_token_selectivity": float(np.min(sel[v])),
            "mean_between_section_post_std": float(np.mean(btw[v])),
            "between_example_post_std": float(allpost.std(ddof=1)),
            # >> 1 means the map moves with the IMAGE but not with the SECTION
            "between_example_over_between_section": float(
                allpost.std(ddof=1) / max(np.mean(btw[v]), 1e-12)),
        }
    out = {
        "checkpoint": CKPT_DIR,
        "n_requested": args.n, "n_scored": len(per_example), "seed": args.seed,
        "patch_grid": [GRID, GRID],
        "pre_half_patch_columns": f"0..{PRE_COLS-1}", "post_half_patch_columns": f"{PRE_COLS}..{GRID-1}",
        "note": ("pre/post are fractions of the per-token rollout distribution over the 576 "
                 "patches (CLS excluded and renormalized); 0.5/0.5 would mean 'no preference'"),
        "summary_by_variant": summary,
        "per_example": per_example,
    }
    p = os.path.join(OUT_DIR, "section_attention_aggregate.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({v: {s: {kk: round(vv, 4) for kk, vv in d.items() if isinstance(vv, float)}
                          for s, d in summary[v].items()} for v in ("rollout", "last_layer")},
                     indent=2))
    print(f"[done] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "examples", "aggregate"], required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    {"selftest": mode_selftest, "examples": mode_examples, "aggregate": mode_aggregate}[args.mode](args)


if __name__ == "__main__":
    main()
