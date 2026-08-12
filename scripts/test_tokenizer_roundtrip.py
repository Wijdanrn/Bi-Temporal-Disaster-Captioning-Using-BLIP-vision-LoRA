"""
Round-trip / special-token alignment unit tests for the Indonesian BLIP adaptation.

These guard the SILENT failure modes: a tokenizer or config change can leave the model
perfectly trainable while destroying report structure or misplacing BOS/EOS.

Run:
    HF_HUB_DISABLE_XET=1 python scripts/test_tokenizer_roundtrip.py            # tokenizer only (fast)
    HF_HUB_DISABLE_XET=1 python scripts/test_tokenizer_roundtrip.py --model    # + model/config alignment
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_model import (  # noqa: E402
    DEC_TOKEN, NL_TOKEN, SECTION_HEADERS,
    IndoReportCodec, build_indonesian_tokenizer,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "processed", "captions_train.jsonl")

_results: list[tuple[bool, str]] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    _results.append((bool(cond), label))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def load_records(limit: int | None = None):
    recs = []
    with open(TRAIN, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            recs.append(json.loads(line))
    return recs


def test_tokenizer(full: bool = True):
    print("\n=== T1  tokenizer construction & added tokens ===")
    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)

    check(len(tok) == 31932, "len(tokenizer) == 31932", f"got {len(tok)}")
    check(tok.convert_tokens_to_ids(DEC_TOKEN) != tok.unk_token_id, "[DEC] is a real token",
          f"id={tok.convert_tokens_to_ids(DEC_TOKEN)}")
    check(tok.convert_tokens_to_ids(NL_TOKEN) != tok.unk_token_id, "[NL] is a real token",
          f"id={tok.convert_tokens_to_ids(NL_TOKEN)}")
    for h in SECTION_HEADERS:
        check(tok.tokenize(h) == [h], f"header {h} is atomic & case-preserved",
              f"{tok.tokenize(h)}")
    check(tok.tokenize("bencana") == ["bencana"],
          "lowercase 'bencana' does NOT collide with header token")

    # [NL] must survive skip_special_tokens=True (it is content, not protocol)
    ids = tok("a" + f" {NL_TOKEN} " + "b", add_special_tokens=False)["input_ids"]
    kept = tok.decode(ids, skip_special_tokens=True)
    check(NL_TOKEN in kept, "[NL] survives skip_special_tokens=True", repr(kept))

    print("\n=== T2  newline preservation (the structural silent-failure) ===")
    raw = "BENCANA: banjir\nBANGUNAN: rusak berat\nKESIMPULAN: parah"
    back = codec.decode(codec.encode(raw))
    check(back.count("\n") == 2, "newlines preserved through encode/decode",
          repr(back))
    naive = tok.decode(tok(raw, add_special_tokens=False)["input_ids"],
                       skip_special_tokens=True)
    check("\n" not in naive,
          "control: a naive tokenizer round-trip DOES destroy newlines", repr(naive[:60]))

    print("\n=== T3  BLIP decoder contract: [DEC] first, [SEP] last ===")
    recs = load_records(None if full else 200)
    n = len(recs)
    bad_first = bad_last = bad_inner = 0
    for r in recs:
        seq = codec.encode(r["ground_truth_id"])
        bad_first += seq[0] != codec.dec_id
        bad_last += seq[-1] != codec.sep_id
        bad_inner += any(i in (codec.dec_id, codec.sep_id, codec.cls_id, codec.pad_id)
                         for i in seq[1:-1])
    check(bad_first == 0, f"ids[0] == [DEC] for all {n} records", f"{bad_first} bad")
    check(bad_last == 0, f"ids[-1] == [SEP] for all {n} records", f"{bad_last} bad")
    check(bad_inner == 0, "no stray [DEC]/[SEP]/[CLS]/[PAD] inside body", f"{bad_inner} bad")

    print("\n=== T4  exact idempotent round-trip (zero train/eval skew) ===")
    bad = 0
    first_bad = None
    for r in recs:
        canon = codec.canonicalize(r["ground_truth_id"])
        if codec.decode(codec.encode(canon, max_length=10 ** 6)) != canon:
            bad += 1
            first_bad = first_bad or canon[:200]
    check(bad == 0, f"decode(encode(x)) == x exactly, all {n} records",
          f"{bad} failures" + (f" e.g. {first_bad!r}" if first_bad else ""))

    print("\n=== T5  section structure survives ===")
    ok = 0
    for r in recs:
        got = [l.split(":")[0] for l in codec.canonicalize(r["ground_truth_id"]).split("\n")]
        ok += got == [h for h in SECTION_HEADERS if h in got] and len(got) > 0
    check(ok == n, f"header sequence preserved for all {n} records", f"{n - ok} bad")

    print("\n=== T6  batch collation shapes / label masking ===")
    batch = codec.encode_batch([recs[0]["ground_truth_id"], recs[1]["ground_truth_id"]])
    ii, am, lb = batch["input_ids"], batch["attention_mask"], batch["labels"]
    check(ii.shape == am.shape == lb.shape, "input_ids/attention_mask/labels same shape",
          str(tuple(ii.shape)))
    check(bool(((lb == -100) == (am == 0)).all()),
          "labels are -100 exactly where attention_mask == 0")
    check(bool((lb[am == 1] == ii[am == 1]).all()),
          "labels == input_ids on real tokens (BLIP shifts internally)")
    return tok, codec


def test_model_alignment():
    print("\n=== T7  model/config special-token alignment ===")
    import torch
    from scripts.build_model import build_model, get_lm_head

    b = build_model(verbose=False)
    base = b.model.base_model.model
    tc = base.config.text_config
    tok, V = b.tokenizer, len(b.tokenizer)

    check(tc.bos_token_id == tok.convert_tokens_to_ids(DEC_TOKEN),
          "text_config.bos_token_id == [DEC]", str(tc.bos_token_id))
    check(tc.sep_token_id == tok.sep_token_id,
          "text_config.sep_token_id == [SEP] (generate stops on THIS)", str(tc.sep_token_id))
    check(tc.eos_token_id == tok.sep_token_id,
          "text_config.eos_token_id repointed off [unused1]=2", str(tc.eos_token_id))
    check(tc.pad_token_id == tok.pad_token_id, "text_config.pad_token_id == [PAD]")
    check(tc.vocab_size == V, "text_config.vocab_size == len(tokenizer)")
    check(base.get_input_embeddings().weight.shape[0] == V, "input embedding rows == vocab")
    check(get_lm_head(base).weight.shape[0] == V, "lm head rows == vocab")
    check(get_lm_head(base).bias.shape[0] == V, "lm head bias == vocab")
    check(base.get_input_embeddings().weight.data_ptr() == get_lm_head(base).weight.data_ptr(),
          "embedding/head weight tie intact after LoRA wrapping")
    check(base.get_input_embeddings().weight.requires_grad,
          "embedding matrix is TRAINABLE (new vocab cannot be learned by LoRA alone)")

    print("\n=== T8  forward pass with real target ===")
    recs = load_records(2)
    batch = b.codec.encode_batch([r["ground_truth_id"] for r in recs])
    px = torch.randn(2, 3, 384, 384)
    b.model.eval()
    with torch.no_grad():
        out = b.model(pixel_values=px, **batch)
    check(out.logits.shape[-1] == V, "logits vocab dim == len(tokenizer)",
          str(tuple(out.logits.shape)))
    check(torch.isfinite(out.loss).item(), "loss is finite", f"{float(out.loss):.4f}")


if __name__ == "__main__":
    want_model = "--model" in sys.argv
    test_tokenizer(full="--fast" not in sys.argv)
    if want_model:
        test_model_alignment()

    n_fail = sum(1 for ok, _ in _results if not ok)
    print("\n" + "=" * 66)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed")
    if n_fail:
        for ok, label in _results:
            if not ok:
                print("  FAILED:", label)
    print("=" * 66)
    sys.exit(1 if n_fail else 0)
