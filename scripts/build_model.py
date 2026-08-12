"""
IndoBLIP-Post-Disaster -- Fase 2 model construction.

Builds an Indonesian-adapted BLIP captioning model:
    base BLIP  ->  swap tokenizer (IndoBERT)  ->  resize+warm-start embeddings
                ->  re-align special tokens to BLIP's decoder contract
                ->  apply LoRA to attention q/v projections

Import and call `build_model()`; the training phase should not re-derive any of this.

    from scripts.build_model import build_model
    bundle = build_model()
    model, tok, codec = bundle.model, bundle.tokenizer, bundle.codec

Design rationale for every non-obvious choice here is recorded in docs/design_decisions.md.

NOTE: always run with HF_HUB_DISABLE_XET=1 in the environment (Xet backend hangs on this machine).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import (  # noqa: E402
    AddedToken,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipImageProcessor,
)
from peft import LoraConfig, get_peft_model  # noqa: E402

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

BASE_MODEL = "Salesforce/blip-image-captioning-base"
INDO_TOKENIZER = "indolem/indobert-base-uncased"

DEC_TOKEN = "[DEC]"   # BLIP's decoder-start token; BLIP injects it BY ID, not via tokenizer
NL_TOKEN = "[NL]"     # newline sentinel: WordPiece destroys "\n", which carries the report structure

SECTION_HEADERS = [
    "BENCANA", "BANGUNAN", "JALAN", "VEGETASI",
    "BADAN_AIR", "PERTANIAN", "KESIMPULAN",
]

# LoRA hits every module whose name ends with one of these. On BLIP this resolves to the
# text decoder's self-attention AND cross-attention q/v. See docs/design_decisions.md.
LORA_TARGET_MODULES = ["query", "value"]

# --- vision tower (Fase 5) --------------------------------------------------------------
# BlipAttention keeps a single FUSED qkv projection, Linear(768 -> 2304); there is no
# separate `query`/`value` module to target the way there is on the text side.
# `qkv` is unique to `vision_model` (verified: 12 modules, 0 elsewhere), so this suffix
# cannot accidentally match a text-decoder module.
# Rationale for targeting the fused module rather than splitting it: design_decisions.md Fase 5.
VISION_LORA_TARGET_MODULES = ["qkv"]

# Alternative vision placement kept switchable for the Fase 5 ablation (MLP instead of
# attention). Not the default -- see design_decisions.md Fase 5.2.
VISION_LORA_MLP_MODULES = ["fc1", "fc2"]


def resolve_target_modules(target_modules: list[str] | None = None,
                           vision_lora: bool = False,
                           vision_lora_targets: list[str] | None = None) -> list[str]:
    """Single source of truth for the LoRA target-module set.

    Kept as a function (not a literal at each call site) so that build_model(),
    apply_lora() and load_trainable_state() can never disagree about which modules the
    adapter covers -- a disagreement there loads weights into the wrong shape silently.
    """
    base = list(target_modules) if target_modules is not None else list(LORA_TARGET_MODULES)
    if vision_lora:
        for m in (vision_lora_targets or VISION_LORA_TARGET_MODULES):
            if m not in base:
                base.append(m)
    return base


# --------------------------------------------------------------------------------------
# Text codec -- converts report text <-> BLIP-decoder-shaped id sequences
# --------------------------------------------------------------------------------------

_DETOK_RULES = [
    (re.compile(r"\s*_\s*"), "_"),                      # BADAN _ AIR -> BADAN_AIR
    (re.compile(r"(?<=\w)\s+-\s+(?=\w)"), "-"),         # tanda - tanda -> tanda-tanda (reduplication)
    (re.compile(r"(?<=\w)\s*/\s*(?=\w)"), "/"),         # dan / atau -> dan/atau
    (re.compile(r"\s+([.,;:!?%\)\]\}>])"), r"\1"),
    (re.compile(r"([\(\[\{<])\s+"), r"\1"),
    (re.compile(r"[ \t]+"), " "),
]


class IndoReportCodec:
    """Encode/decode the structured Indonesian damage reports for BLIP's decoder.

    Contract enforced (verified by round_trip_test()):
        ids[0]  == bos/[DEC]   because BlipForConditionalGeneration.generate() does
                               `input_ids[:, 0] = config.text_config.bos_token_id`
        ids[-1] == [SEP]       because generate() passes
                               `eos_token_id=config.text_config.sep_token_id`
    """

    def __init__(self, tokenizer, max_length: int = 384):
        self.tok = tokenizer
        self.max_length = max_length
        self.dec_id = tokenizer.convert_tokens_to_ids(DEC_TOKEN)
        self.nl_id = tokenizer.convert_tokens_to_ids(NL_TOKEN)
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self._strip = {self.pad_id, self.dec_id, self.cls_id, self.sep_id, -100}

    # -- encode ------------------------------------------------------------------
    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        ml = max_length or self.max_length
        t = re.sub(r"\n+", "\n", text.replace("\r\n", "\n").strip())
        t = t.replace("\n", f" {NL_TOKEN} ")
        ids = self.tok(t, add_special_tokens=False, truncation=True,
                       max_length=ml - 2)["input_ids"]
        return [self.dec_id] + ids + [self.sep_id]

    def encode_batch(self, texts, max_length: int | None = None, device=None):
        """-> dict(input_ids, attention_mask, labels) ready for BlipForConditionalGeneration."""
        seqs = [self.encode(t, max_length) for t in texts]
        n = max(len(s) for s in seqs)
        input_ids, attn, labels = [], [], []
        for s in seqs:
            pad = n - len(s)
            input_ids.append(s + [self.pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
            # BLIP shifts internally (labels[:,1:] vs logits[:,:-1]) -> labels == input_ids
            labels.append(s + [-100] * pad)
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return {k: v.to(device) for k, v in out.items()} if device else out

    # -- decode ------------------------------------------------------------------
    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        ids = [i for i in ids if i not in self._strip]
        s = self.tok.decode(ids, skip_special_tokens=False)
        for pat, rep in _DETOK_RULES:
            s = pat.sub(rep, s)
        s = s.replace(f" {NL_TOKEN} ", "\n").replace(NL_TOKEN, "\n")
        return "\n".join(line.strip() for line in s.split("\n")).strip()

    def canonicalize(self, text: str) -> str:
        """Project raw text onto the codec's fixed-point form.

        Preprocess targets with this so the training text is EXACTLY what the codec can
        reproduce -> zero train/eval skew. Verified idempotent on all 6,999 train records.
        """
        return self.decode(self.encode(text, max_length=10**6))


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------

def build_indonesian_tokenizer(name: str = INDO_TOKENIZER, add_headers: bool = True):
    """IndoBERT tokenizer + the extra tokens BLIP's decoder contract and our report format need."""
    tok = AutoTokenizer.from_pretrained(name)

    # [DEC] must be a real special token: BLIP writes bos_token_id into position 0 by id.
    tok.add_special_tokens(
        {"additional_special_tokens": [AddedToken(DEC_TOKEN, normalized=False, special=True)]}
    )

    # [NL] and the section headers are NORMAL added tokens on purpose: they must survive
    # `skip_special_tokens=True` at decode time, because they are report content, not protocol.
    # normalized=False keeps them exact-case despite the uncased tokenizer.
    extra = [AddedToken(NL_TOKEN, normalized=False, special=False)]
    if add_headers:
        extra += [AddedToken(h, normalized=False, special=False) for h in SECTION_HEADERS]
    tok.add_tokens(extra)
    return tok


# --------------------------------------------------------------------------------------
# Embedding resize + warm start
# --------------------------------------------------------------------------------------

@dataclass
class ResizeStats:
    old_vocab_size: int
    new_vocab_size: int
    hidden_size: int
    old_embedding_params: int
    new_embedding_params: int
    old_bias_params: int
    new_bias_params: int
    warm_started_rows: int
    random_rows: int
    weights_tied: bool

    @property
    def net_new_params(self) -> int:
        return ((self.new_embedding_params + self.new_bias_params)
                - (self.old_embedding_params + self.old_bias_params))

    @property
    def reinitialized_params(self) -> int:
        """Params whose MEANING changed and must be relearned (whole embedding matrix + bias)."""
        return self.new_embedding_params + self.new_bias_params

    def report(self) -> str:
        L = [
            "--- embedding resize ---",
            f"  vocab            : {self.old_vocab_size} -> {self.new_vocab_size}",
            f"  hidden_size      : {self.hidden_size}",
            f"  input embedding  : {self.old_embedding_params:,} -> {self.new_embedding_params:,}"
            f"  ({self.new_vocab_size} x {self.hidden_size})",
            f"  lm head weight   : TIED to input embedding (0 additional params)"
            if self.weights_tied else "  lm head weight   : UNTIED",
            f"  lm head bias     : {self.old_bias_params:,} -> {self.new_bias_params:,}",
            f"  NET new params   : {self.net_new_params:+,}",
            f"  REINIT params    : {self.reinitialized_params:,} (entire vocab matrix is remapped)",
            f"  warm-started rows: {self.warm_started_rows:,} / {self.new_vocab_size:,} "
            f"({100 * self.warm_started_rows / self.new_vocab_size:.1f}%)",
            f"  random rows      : {self.random_rows:,}",
        ]
        return "\n".join(L)


def get_lm_head(model) -> nn.Linear:
    """Return BLIP's LM head.

    IMPORTANT: `BlipForConditionalGeneration.get_output_embeddings()` returns **None** -- the
    outer wrapper never forwards it to the text decoder. Calling the outer accessor and
    trusting it is a silent-bug generator. The head really lives here:
        text_decoder.cls.predictions.decoder   (Linear(hidden, vocab, bias=True))
    Its .weight is tied to the input embedding, and its .bias is the same Parameter object as
    text_decoder.cls.predictions.bias.
    """
    return model.text_decoder.get_output_embeddings()


def _old_vocab_strings(model_name: str = BASE_MODEL) -> dict[str, int]:
    """Surface-string -> id for the ORIGINAL BLIP tokenizer (bert-base-uncased + [DEC]/[ENC])."""
    old = AutoTokenizer.from_pretrained(model_name)
    vocab = dict(old.get_vocab())
    # BLIP's checkpoint has 30524 rows: 30522 wordpieces + [DEC](30522) + [ENC](30523).
    vocab.setdefault(DEC_TOKEN, 30522)
    return vocab


def swap_tokenizer_and_resize(model, tokenizer, warm_start: bool = True,
                              verbose: bool = True) -> ResizeStats:
    """Resize BLIP's text embeddings to the new tokenizer and re-point all special-token ids.

    Warm start: rows whose surface string exists in BLIP's original vocab (punctuation, digits,
    latin subwords, shared loanwords, and crucially [DEC]/[PAD]/[CLS]/[SEP]/[UNK]/[MASK]) are
    copied from the pretrained matrix instead of being randomly initialized.
    """
    tcfg = model.config.text_config
    hidden = tcfg.hidden_size
    old_vocab = model.get_input_embeddings().weight.shape[0]
    old_head = get_lm_head(model)
    old_bias_n = old_head.bias.numel() if getattr(old_head, "bias", None) is not None else 0

    old_matrix = model.get_input_embeddings().weight.data.clone()
    old_str2id = _old_vocab_strings()

    new_vocab = len(tokenizer)
    model.resize_token_embeddings(new_vocab)

    emb = model.get_input_embeddings()
    head = get_lm_head(model)
    tied = emb.weight.data_ptr() == head.weight.data_ptr()
    if head.weight.shape[0] != new_vocab or head.bias.shape[0] != new_vocab:
        raise RuntimeError(
            f"LM head was not resized with the embeddings "
            f"(head={tuple(head.weight.shape)}, bias={tuple(head.bias.shape)}, want vocab={new_vocab})."
        )

    warm = 0
    if warm_start:
        with torch.no_grad():
            new_w = emb.weight.data
            for tokstr, new_id in tokenizer.get_vocab().items():
                old_id = old_str2id.get(tokstr)
                if old_id is not None and old_id < old_matrix.shape[0] and new_id < new_vocab:
                    new_w[new_id] = old_matrix[old_id]
                    warm += 1

    # ---- re-point every special-token id BLIP reads at train/generate time ----
    tcfg.vocab_size = new_vocab
    tcfg.bos_token_id = tokenizer.convert_tokens_to_ids(DEC_TOKEN)  # generate(): input_ids[:,0]
    tcfg.sep_token_id = tokenizer.sep_token_id                      # generate(): eos_token_id=
    tcfg.pad_token_id = tokenizer.pad_token_id
    # Base BLIP ships eos_token_id=2 ([unused1]) which is NEVER used for stopping -- it is a
    # footgun. Point it at [SEP] so every code path agrees.
    tcfg.eos_token_id = tokenizer.sep_token_id
    model.decoder_input_ids = tcfg.bos_token_id
    model.decoder_pad_token_id = tcfg.pad_token_id
    for attr, val in (("bos_token_id", tcfg.bos_token_id),
                      ("eos_token_id", tcfg.sep_token_id),
                      ("pad_token_id", tcfg.pad_token_id)):
        setattr(model.generation_config, attr, val)

    stats = ResizeStats(
        old_vocab_size=old_vocab, new_vocab_size=new_vocab, hidden_size=hidden,
        old_embedding_params=old_vocab * hidden, new_embedding_params=new_vocab * hidden,
        old_bias_params=old_bias_n,
        new_bias_params=head.bias.numel() if getattr(head, "bias", None) is not None else 0,
        warm_started_rows=warm, random_rows=new_vocab - warm, weights_tied=tied,
    )
    if verbose:
        print(stats.report())
    return stats


# --------------------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------------------

def inspect_attention_modules(model) -> dict[str, list[str]]:
    """Group nn.Linear leaf names so LoRA targets are chosen from REAL names, not memory."""
    groups: dict[str, list[str]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            groups.setdefault(name.split(".")[-1], []).append(name)
    return groups


def apply_lora(model, r: int = 8, alpha: int = 32, dropout: float = 0.1,
               target_modules: list[str] | None = None,
               vision_lora: bool = False,
               vision_lora_targets: list[str] | None = None,
               train_embeddings: bool = True, verbose: bool = True):
    """Wrap with LoRA on attention q/v, and (by default) unfreeze the new embedding matrix.

    train_embeddings=True is NOT optional in practice: the whole word-embedding matrix was
    remapped to a new language. LoRA on q/v cannot fix randomly-initialized token vectors,
    so with frozen embeddings the model would train "successfully" and emit garbage.

    vision_lora=False by default so every existing script/checkpoint keeps the exact
    text-only adapter set it was built and trained with.
    """
    resolved = resolve_target_modules(target_modules, vision_lora, vision_lora_targets)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolved,
        bias="none",
        task_type=None,  # BLIP is not one of peft's task types; keep the plain wrapper
    )
    peft_model = get_peft_model(model, cfg)

    if train_embeddings:
        base = peft_model.base_model.model
        emb = base.get_input_embeddings()
        head = get_lm_head(base)
        emb.weight.requires_grad_(True)
        if getattr(head, "bias", None) is not None:
            head.bias.requires_grad_(True)
        if emb.weight.data_ptr() != head.weight.data_ptr():
            raise RuntimeError(
                "Embedding/LM-head weight tie was broken by LoRA wrapping. "
                "Training would update two divergent copies -- refusing to continue."
            )

    if verbose:
        b = param_breakdown(peft_model)
        print("--- LoRA ---")
        print(f"  r={r} alpha={alpha} dropout={dropout} targets={cfg.target_modules}")
        print(f"  LoRA params (text) : {b['lora_text']:,}")
        print(f"  LoRA params (vis)  : {b['lora_vision']:,}")
        print(f"  other trainable    : {b['embeddings']:,} (embedding matrix + lm-head bias)")
        print(f"  total trainable    : {b['trainable']:,}")
        print(f"  total params       : {b['total']:,}")
        print(f"  trainable fraction : {100 * b['trainable'] / b['total']:.4f}%")
    return peft_model


def param_breakdown(peft_model) -> dict[str, int]:
    """Measured (never estimated) trainable-parameter split by where the params live."""
    out = {"lora_text": 0, "lora_vision": 0, "embeddings": 0, "other": 0,
           "trainable": 0, "total": 0}
    for n, p in peft_model.named_parameters():
        out["total"] += p.numel()
        if not p.requires_grad:
            continue
        out["trainable"] += p.numel()
        if "lora_" in n:
            out["lora_vision" if "vision_model" in n else "lora_text"] += p.numel()
        elif "word_embeddings" in n or "predictions" in n or "cls." in n:
            out["embeddings"] += p.numel()
        else:
            out["other"] += p.numel()
    return out


def count_lora_targets(peft_model) -> dict[str, int]:
    from peft.tuners.lora import LoraLayer
    hit: dict[str, int] = {}
    for name, mod in peft_model.named_modules():
        if isinstance(mod, LoraLayer):
            if "vision_model" in name:
                key = "vision"
            elif "crossattention" in name:
                key = "crossattention"
            elif "attention" in name:
                key = "self-attention"
            else:
                key = "other"
            key += "." + name.split(".")[-1]
            hit[key] = hit.get(key, 0) + 1
    return hit


def lora_module_names(peft_model) -> list[str]:
    """Fully-qualified names of every LoRA-adapted module, sorted. Used by the checkpoint
    compatibility check to prove which modules are new vs. pre-existing."""
    from peft.tuners.lora import LoraLayer
    return sorted(n for n, m in peft_model.named_modules() if isinstance(m, LoraLayer))


# --------------------------------------------------------------------------------------
# Top-level builder
# --------------------------------------------------------------------------------------

@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    image_processor: Any
    codec: IndoReportCodec
    resize_stats: ResizeStats
    lora_targets: dict[str, int] = field(default_factory=dict)
    trainable_params: int = 0
    total_params: int = 0


def build_model(base_model: str = BASE_MODEL,
                indo_tokenizer: str = INDO_TOKENIZER,
                lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1,
                target_modules: list[str] | None = None,
                vision_lora: bool = False,
                vision_lora_targets: list[str] | None = None,
                max_length: int = 384,
                warm_start: bool = True,
                train_embeddings: bool = True,
                use_lora: bool = True,
                dtype=torch.float32,
                device: str | None = None,
                verbose: bool = True) -> ModelBundle:
    """Build the full Indonesian-adapted, LoRA-wrapped BLIP model."""
    if verbose:
        print(f"[build_model] loading {base_model}")
    model = BlipForConditionalGeneration.from_pretrained(base_model, dtype=dtype)
    image_processor = BlipImageProcessor.from_pretrained(base_model)

    tokenizer = build_indonesian_tokenizer(indo_tokenizer)
    stats = swap_tokenizer_and_resize(model, tokenizer, warm_start=warm_start, verbose=verbose)
    codec = IndoReportCodec(tokenizer, max_length=max_length)

    if use_lora:
        model = apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                           target_modules=target_modules,
                           vision_lora=vision_lora,
                           vision_lora_targets=vision_lora_targets,
                           train_embeddings=train_embeddings, verbose=verbose)
        targets = count_lora_targets(model)
    else:
        targets = {}

    if device:
        model = model.to(device)

    return ModelBundle(
        model=model, tokenizer=tokenizer, image_processor=image_processor,
        codec=codec, resize_stats=stats, lora_targets=targets,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )


def save_trainable_state(bundle: ModelBundle, out_dir: str):
    """Save LoRA adapter AND the retrained embedding matrix.

    peft's save_pretrained persists ONLY adapter weights. The resized embedding matrix is the
    single most important thing we train here, so saving the adapter alone would silently throw
    away the Indonesian vocabulary. Always use this instead of model.save_pretrained().
    """
    os.makedirs(out_dir, exist_ok=True)
    bundle.model.save_pretrained(out_dir)
    bundle.tokenizer.save_pretrained(out_dir)
    base = getattr(bundle.model, "base_model", bundle.model)
    base = getattr(base, "model", base)
    emb = base.get_input_embeddings()
    head = get_lm_head(base)
    payload = {"word_embeddings": emb.weight.detach().cpu()}
    if getattr(head, "bias", None) is not None:
        payload["lm_head_bias"] = head.bias.detach().cpu()
    torch.save(payload, os.path.join(out_dir, "embeddings.pt"))
    return out_dir


def read_checkpoint_target_modules(out_dir: str) -> list[str] | None:
    """The target_modules the checkpoint was actually SAVED with, read off its adapter_config.

    Never infer this from the current defaults -- the defaults are exactly what changes when
    the adapter set is expanded, so inferring would mask the very mismatch we test for.
    """
    import json
    p = os.path.join(out_dir, "adapter_config.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        tm = json.load(f).get("target_modules")
    return sorted(tm) if isinstance(tm, (list, tuple, set)) else tm


def load_trainable_state(out_dir: str, device: str | None = None, verbose: bool = True,
                         allow_new_lora_modules: bool = False,
                         **build_kwargs) -> ModelBundle:
    """Inverse of save_trainable_state(): rebuild the architecture via build_model(), then load
    the LoRA adapter weights AND the retrained embedding matrix / lm-head bias on top of it.

    `build_kwargs` must reproduce the hyperparameters used to build the checkpointed model
    (lora_r, lora_alpha, lora_dropout, target_modules, max_length, ...) so shapes match; the
    caller is expected to pass the same values recorded in the run's config snapshot.

    Reusing build_model() (rather than re-deriving the special-token/resize logic here) means
    this function cannot drift out of sync with swap_tokenizer_and_resize()/apply_lora().

    ADAPTER EXPANSION (Fase 5)
    --------------------------
    `allow_new_lora_modules=True` permits loading an OLD, smaller adapter (e.g. text-only
    q/v) into a NEWER, larger LoRA config (e.g. + vision `qkv`). This is safe only because
    LoRA's `lora_B` is zero-initialised: modules absent from the checkpoint stay at
    B=0 => delta_W = B@A = 0 => the expanded model is numerically IDENTICAL to the
    checkpoint until training moves B.

    That safety is ASSERTED here, not assumed:
      * every key present in the checkpoint must land (nothing silently dropped),
      * `unexpected_keys` is ALWAYS fatal (checkpoint weight with nowhere to go),
      * `missing_keys` is fatal unless explicitly allowed,
      * every newly-added module's `lora_B` is verified to be exactly zero.
    """
    from peft import set_peft_model_state_dict

    bundle = build_model(device=device, verbose=False, **build_kwargs)

    adapter_st = os.path.join(out_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(out_dir, "adapter_model.bin")
    if os.path.exists(adapter_st):
        from safetensors.torch import load_file as load_safetensors
        sd = load_safetensors(adapter_st, device="cpu")
    elif os.path.exists(adapter_bin):
        sd = torch.load(adapter_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"no adapter weights found in {out_dir}")

    load_result = set_peft_model_state_dict(bundle.model, sd)
    missing = [k for k in getattr(load_result, "missing_keys", []) if "lora_" in k]
    unexpected = list(getattr(load_result, "unexpected_keys", []))

    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} adapter tensors in {out_dir} have no matching module in the "
            f"model being loaded into -- trained weights would be silently discarded. "
            f"First few: {unexpected[:5]}"
        )
    if missing and not allow_new_lora_modules:
        ckpt_tm = read_checkpoint_target_modules(out_dir)
        raise RuntimeError(
            f"{len(missing)} LoRA tensors in the model are absent from the checkpoint "
            f"{out_dir} (checkpoint target_modules={ckpt_tm}). If you are deliberately "
            f"EXPANDING the adapter set, pass allow_new_lora_modules=True. "
            f"First few: {missing[:5]}"
        )

    # -- prove the loaded values are bit-identical to the checkpoint --------------------
    # get_peft_model_state_dict() is the exact inverse of the save path, so it emits keys in
    # the same convention as the file -- no ad-hoc key rewriting (which is itself a silent
    # mismatch risk) is needed.
    from peft import get_peft_model_state_dict
    live = get_peft_model_state_dict(bundle.model)
    for k, v in sd.items():
        if k not in live:
            raise RuntimeError(
                f"checkpoint tensor {k} has no counterpart in the rebuilt adapter -- "
                f"a trained weight would be silently dropped."
            )
        if not torch.equal(live[k].detach().cpu(), v.detach().cpu().to(live[k].dtype)):
            raise RuntimeError(f"checkpoint tensor {k} did not land bit-identically")
    n_loaded = len(sd)

    # -- prove every NEW module is a true no-op (lora_B == 0) --------------------------
    new_modules = sorted({k.rsplit(".lora_", 1)[0] for k in missing})
    for name, p in bundle.model.named_parameters():
        if "lora_B" in name and name.rsplit(".lora_", 1)[0] in new_modules:
            if p.detach().abs().max().item() != 0.0:
                raise RuntimeError(
                    f"newly-added LoRA module {name} is NOT zero-initialised; adding it "
                    f"would perturb the checkpoint's behaviour at step 0."
                )

    payload = torch.load(os.path.join(out_dir, "embeddings.pt"), map_location="cpu")
    base = getattr(bundle.model, "base_model", bundle.model)
    base = getattr(base, "model", base)
    emb = base.get_input_embeddings()
    head = get_lm_head(base)
    with torch.no_grad():
        emb.weight.copy_(payload["word_embeddings"].to(emb.weight.device, emb.weight.dtype))
        if "lm_head_bias" in payload and getattr(head, "bias", None) is not None:
            head.bias.copy_(payload["lm_head_bias"].to(head.bias.device, head.bias.dtype))

    if verbose:
        print(f"[load_trainable_state] loaded adapter + embeddings from {out_dir}")
        print(f"  checkpoint adapter tensors loaded : {n_loaded} "
              f"of {len(live)} in the rebuilt adapter (all bit-identical)")
        if missing:
            print(f"  NEW LoRA modules (zero-init, no-op): {len(new_modules)}")
    return bundle


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vision-lora", action="store_true",
                    help="also adapt the vision tower's fused self_attn.qkv (Fase 5)")
    ap.add_argument("--vision-lora-targets", type=str, default=None,
                    help="comma-separated override, e.g. 'fc1,fc2'")
    a = ap.parse_args()
    b = build_model(
        vision_lora=a.vision_lora,
        vision_lora_targets=([s.strip() for s in a.vision_lora_targets.split(",")]
                             if a.vision_lora_targets else None),
    )
    print("\nLoRA-adapted modules:", b.lora_targets)
    print(f"trainable {b.trainable_params:,} / total {b.total_params:,}")
