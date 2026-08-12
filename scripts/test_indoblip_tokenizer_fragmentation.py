"""
Quick, no-GPU proof #1 (user asked "apakah bisa dibuktikan secara sederhana"): does the
external indoblip-lr-5e-06-epoch-30 checkpoint's tokenizer (stock Salesforce/blip-image-
captioning-base BertTokenizer, English WordPiece vocab, unmodified -- confirmed by its
config.json vocab_size=30524 and the absence of any tokenizer files in the checkpoint folder)
fragment real Indonesian disaster-report text worse than our own indolem-adapted tokenizer?

This is the same measurement class as docs/design_decisions.md SS1 (which disqualified the
English tokenizer for OUR pipeline), just re-run here directly against the indoblip
checkpoint's actual tokenizer object and the REAL test-set ground truth text (not synthetic
examples), so the number is a fresh, reproducible measurement rather than a re-quote.

Usage:
    python scripts/test_indoblip_tokenizer_fragmentation.py
"""
from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer  # noqa: E402
from scripts.build_model import build_indonesian_tokenizer  # noqa: E402

BASE_MODEL = "Salesforce/blip-image-captioning-base"
TEST_PATH = os.path.join(ROOT, "data", "processed", "captions_test.jsonl")
N_SAMPLE = 300


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def main():
    print(f"[load] indoblip checkpoint's tokenizer (stock, unmodified): {BASE_MODEL}")
    en_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"  vocab_size={en_tok.vocab_size}")

    print("[load] our tokenizer (indolem-adapted, Fase 2)")
    id_tok = build_indonesian_tokenizer()
    print(f"  vocab_size={id_tok.vocab_size}")

    with open(TEST_PATH, encoding="utf-8") as f:
        records = [json.loads(l) for l in f][:N_SAMPLE]
    texts = [r["ground_truth_id"] for r in records]

    total_words = 0
    en_tokens, id_tokens = 0, 0
    for t in texts:
        total_words += word_count(t)
        en_tokens += len(en_tok.tokenize(t))
        id_tokens += len(id_tok.tokenize(t))

    en_ratio = en_tokens / total_words
    id_ratio = id_tokens / total_words

    print(f"\n[result] {len(texts)} real ground-truth test reports, {total_words} words total")
    print(f"  indoblip's tokenizer (English WordPiece, unmodified): "
          f"{en_tokens} tokens -> {en_ratio:.3f} tokens/word")
    print(f"  our tokenizer (indolem, Fase 2 adapted):              "
          f"{id_tokens} tokens -> {id_ratio:.3f} tokens/word")
    print(f"  indoblip's tokenizer needs {en_ratio / id_ratio:.2f}x as many tokens per word")

    # A few concrete disaster-domain words, shown individually so the fragmentation is visible
    # token-by-token, not just as a ratio.
    sample_words = ["kerusakan", "bencana", "kebakaran", "reruntuhan", "terendam",
                     "longsor", "evakuasi", "infrastruktur"]
    print("\n[examples] domain-specific word -> subword pieces")
    for w in sample_words:
        print(f"  {w!r:20s} indoblip: {en_tok.tokenize(w)}")
        print(f"  {'':20s} ours:     {id_tok.tokenize(w)}")

    out = {
        "n_reports_sampled": len(texts),
        "total_words": total_words,
        "indoblip_tokenizer": {"name": BASE_MODEL, "vocab_size": en_tok.vocab_size,
                                "total_tokens": en_tokens, "tokens_per_word": en_ratio},
        "our_tokenizer": {"name": "indolem/indobert-base-uncased (adapted)",
                          "vocab_size": id_tok.vocab_size,
                          "total_tokens": id_tokens, "tokens_per_word": id_ratio},
        "fragmentation_multiplier": en_ratio / id_ratio,
    }
    out_path = os.path.join(ROOT, "results", "indoblip_tokenizer_fragmentation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
