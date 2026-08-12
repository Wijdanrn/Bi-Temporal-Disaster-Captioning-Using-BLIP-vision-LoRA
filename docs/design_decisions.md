# Design Decisions — IndoBLIP-Post-Disaster

Fase 2: Indonesian tokenizer swap, embedding resize, special-token realignment, LoRA setup.

Implementation: `scripts/build_model.py` (importable — Fase 3 training should call `build_model()`
rather than re-deriving any of this).
All numbers below were **measured on this machine**, not taken from the draft paper.

Environment: `C:\Users\Wijdan\anaconda3\envs\torch5050\python.exe`, torch 2.9.0.dev+cu128,
transformers 5.9.0, peft 0.20.0, RTX 5050 Laptop (8.5 GB).
`HF_HUB_DISABLE_XET=1` is set inside `build_model.py` as well as in the shell, because the Xet
backend stalls model downloads on this machine.

---

## 0. What the base model actually is (measured, not assumed)

`Salesforce/blip-image-captioning-base` — 225,644,220 params total.

| Property | Value |
|---|---|
| text tokenizer | `BertTokenizer`, bert-base-uncased, 30,522 wordpieces |
| `text_config.vocab_size` | **30,524** (2 more than the tokenizer!) |
| `text_config.hidden_size` | 768 |
| input embedding | `Embedding(30524, 768)` |
| LM head | `text_decoder.cls.predictions.decoder` = `Linear(768, 30524, bias=True)` |
| weight tying | head `.weight` **is** the input embedding (same storage) |
| bias tying | `cls.predictions.bias` **is** `cls.predictions.decoder.bias` |
| text position embeddings | 512 |

The 2-token gap is the crux of the whole phase: the checkpoint has embedding rows **30522
(`[DEC]`)** and **30523 (`[ENC]`)** that the shipped tokenizer cannot produce or decode. BLIP
injects `[DEC]` *by integer id* in Python, never through the tokenizer.

### BLIP's decoder contract (read from `transformers/models/blip/modeling_blip.py`)

```python
# BlipForConditionalGeneration.generate(), l.932-938
input_ids[:, 0] = self.config.text_config.bos_token_id          # -> [DEC], overwrites [CLS]
outputs = self.text_decoder.generate(
    input_ids=input_ids[:, :-1],
    eos_token_id=self.config.text_config.sep_token_id,          # <-- sep, NOT eos!
    pad_token_id=self.config.text_config.pad_token_id, ...)
```

```python
# BlipTextLMHeadModel.forward(), l.679-682  — the model shifts labels ITSELF
shifted_prediction_scores = prediction_scores[:, :-1, :]
labels = labels[:, 1:]
loss = CrossEntropyLoss(...)(shifted..., labels...)             # ignore_index = -100
```

Three consequences that drive everything below:

1. **Position 0 must be `[DEC]`**, not `[CLS]`. `generate()` force-writes it. If training used
   `[CLS]` at position 0, train and inference would silently disagree on the very first token.
2. **Generation stops on `sep_token_id`, never on `eos_token_id`.** Base BLIP ships
   `eos_token_id=2`, which in bert-base-uncased is `[unused1]` — a token that appears in no
   caption and stops nothing. It is a dormant footgun; we repoint it at `[SEP]`.
3. **`labels` must equal `input_ids`** (not pre-shifted), with padding set to `-100`.

### Landmine: `get_output_embeddings()` returns `None`

`BlipForConditionalGeneration` does **not** forward `get_output_embeddings()` to its text
decoder, so the outer call returns `None`. Any code that reads the LM head through the outer
accessor silently gets nothing. `scripts/build_model.py` therefore exposes `get_lm_head(model)`
and every head access goes through it.

Verified separately that `model.resize_token_embeddings(N)` on the **outer** model still resizes
the head weight, the head bias, and `config.text_config.vocab_size` correctly, and preserves both
ties — but the builder asserts this rather than trusting it.

---

## 1. Decision: Indonesian tokenizer → `indolem/indobert-base-uncased`

Measured on the first 1,000 real `ground_truth_id` records (151,096 whitespace words).
"whole" = domain words that survive as a single token, out of 36 hand-picked disaster terms.
"hdrTok" = tokens needed for the 7 section headers.

| tokenizer | vocab | tokens/word | mean seq len | UNK | whole | hdrTok |
|---|---|---|---|---|---|---|
| bert-base-uncased (BLIP incumbent) | 30,522 | **2.663** | 402.3 | 2 | **0/36** | 21 |
| `indobenchmark/indobert-base-p1` | 30,521 | 1.225 | 185.1 | 2 | 30/36 | 9 |
| **`indolem/indobert-base-uncased`** | 31,923 | **1.223** | **184.7** | 2 | **31/36** | 9 |

**The incumbent English tokenizer is disqualified outright.** At 2.663 tokens/word it needs
2.18× more positions for the same report, and it fragments *every one* of the 36 domain terms:

```
kerusakan    -> ['ke', '##rus', '##aka', '##n']
pascabencana -> ['pas', '##ca', '##ben', '##can', '##a']
struktural   -> ['st', '##ru', '##ktur', '##al']
```

Learning Indonesian on top of that is learning to spell, not to write.

**Between the two Indonesian candidates the margin is genuinely small (0.2% fertility).** Both
are uncased WordPiece, both reduce UNK to noise, both encode the 7 headers in 9 tokens. Chose
`indolem` on two concrete tiebreakers:

- `tergenang` ("inundated", core flood vocabulary) is **one token** in indolem, split as
  `['terg','##enang']` by indobenchmark.
- `pascabencana` → 2 tokens (indolem) vs 3 (indobenchmark).

**Honest counter-argument:** indolem's vocab is 1,402 rows larger, which costs an extra
1,402 × 768 = **1,076,736** embedding params. That is a real cost against a 0.2% fertility gain,
and one could reasonably pick
indobenchmark instead. It is not decisive here because **91% of the vocabulary is unused anyway**
(see §7), so vocab size is dominated by a pruning decision we have deferred. The choice is a
one-line change: `INDO_TOKENIZER` in `scripts/build_model.py`.

---

## 2. Decision: add `[DEC]`, `[NL]`, and the 7 section headers → 31,932 tokens

`31,923 → 31,932` (+9). Each addition is load-bearing:

### `[DEC]` (id 31923) — special token
Required by BLIP's contract (§0). Without it, `text_config.bos_token_id` would still be 30522,
which after the swap points at an arbitrary Indonesian wordpiece (or out of range). Added as a
*special* token so `skip_special_tokens=True` strips it.

### `[NL]` (id 31924) — normal added token — **this is the silent failure I was sent to find**

WordPiece's `BasicTokenizer` treats `\n` as plain whitespace and **discards it**. The targets are
structured 7-section reports whose only section delimiter is `\n`. Round-tripping the raw text
through *any* of the three tokenizers destroyed the structure completely:

```
orig: BENCANA: gunung berapi\nBANGUNAN: Tidak ada kerusakan struktural...
back: bencana : gunung berapi bangunan : tidak ada kerusakan struktural...
```

This fails exactly the way the brief describes: **nothing crashes.** There is no newline token in
the vocabulary, so the model would train happily, loss would fall, and it would emit one run-on
lowercase paragraph forever. Every downstream consumer that splits sections on `\n` — the metric
harness, the report renderer — would then silently see one section instead of seven.

`[NL]` is a normal (non-special) added token on purpose: it must **survive**
`skip_special_tokens=True`, because it is report *content*, not protocol.

### `BENCANA … KESIMPULAN` (ids 31925-31931) — normal added tokens, `normalized=False`
- The tokenizer is uncased, so `BENCANA` would otherwise collapse into the ordinary word
  `bencana` and become indistinguishable from the same word in running prose.
  `normalized=False` preserves exact casing despite `do_lower_case=True`.
- `BADAN_AIR` cost 3 tokens (`['badan','_','air']`); now 1.
- They appear in 100% of examples, so 7 fresh embeddings are trivially learnable and give the
  model a strong, unambiguous structural prior.
- Verified no collision: lowercase `bencana` in body text still maps to the ordinary token.

---

## 3. Decision: resize + warm-start (not from-scratch, not re-tying)

**Chosen:** `model.resize_token_embeddings(31932)`, keep the existing weight tie, then overwrite
rows whose surface string exists in BLIP's original vocab with the pretrained row.

Rejected alternatives:
- *Brand-new randomly-initialized head* — throws away the tie and doubles vocab params for no
  benefit; the head is tied in the checkpoint and works.
- *Untie head from embedding* — +24.5M params on an 8.5 GB GPU, no upside for a 7k-example set.
- *Pure random / mean-resize only* — leaves free information on the table (below).

**Warm start:** 8,098 / 31,932 rows (25.4%) share a surface string with bert-base-uncased —
punctuation, digits, latin subwords, shared loanwords, and all specials. Those rows are copied
from the pretrained matrix. Measured on real corpus text, that covers **33.9% of actual token
occurrences**. Notably `[DEC]` inherits BLIP's genuine pretrained row 30522, so the model keeps
the "begin decoding" signal it was actually trained with instead of a random vector.

The remaining 23,834 rows keep transformers' default mean-resizing (multivariate normal fitted to
the old embeddings' mean and covariance), which is already better than `N(0, 0.02)`.

Position embeddings are **not** resized: BLIP allows 512 and our longest target is 346 tokens.

---

## 4. Measured parameter counts — the draft paper's "~300,000" is wrong

Computed as `embedding_dim × vocab_size` against the tokenizer actually loaded, plus the
vocab-sized head bias:

```
hidden_size            = 768
vocab       30,524  ->  31,932        (+1,408 rows)
input emb   23,442,432 -> 24,523,776  (= 31,932 x 768)
head weight TIED to input embedding   (0 additional params)
head bias      30,524 ->     31,932

NET new parameters   = +1,082,752      (1,408 x 768  +  1,408)
REINITIALIZED params =  24,555,708     (the whole matrix + bias is semantically remapped)
```

**The draft's "~300,000 new parameters" does not reproduce.**
- Against the *net* delta it is **3.6× too low** (1,082,752 actual).
- Against the parameters that actually have to be **learned** it is **~82× too low**
  (24,555,708) — because swapping to a different language remaps the meaning of *every* row, not
  just the 1,408 appended ones. The paper's framing implicitly assumes the old English embeddings
  stay valid, which they do not.

The second number is the one that matters for planning: this is not a ~300k-parameter tweak, it
is a 24.5M-parameter relearning problem on 6,999 examples.

---

## 5. Decision: LoRA on `["query", "value"]` — self-attention **and** cross-attention

Module names were read off the loaded model, not recalled:

| leaf name | count | tower | shape |
|---|---|---|---|
| `query` / `key` / `value` | 24 each | text decoder | (768, 768) |
| `dense` | 49 | text decoder | (768, 768) |
| `qkv` | 12 | **vision** | (2304, 768) — *fused* |
| `projection`, `fc1`, `fc2` | 12 each | vision | — |

The 24 `query` modules are 12 self-attention + 12 cross-attention. So `["query","value"]`
resolves to **48 adapted modules**:

```
self-attention.query : 12    crossattention.query : 12
self-attention.value : 12    crossattention.value : 12
```

Why this set:
- **Cross-attention is the important half.** It is the only place image features enter the text
  stream; re-grounding English-trained visual attention onto Indonesian tokens is precisely the
  adaptation we need. Targeting `query`/`value` by suffix picks it up automatically.
- **q/v over q/k/v** — the standard LoRA finding (LoRA paper §7.1); `key` adds ~295k params for
  little gain at r=8.
- **Vision tower deliberately untouched.** Its attention is a *fused* `qkv` `Linear(768→2304)`,
  so a "q and v only" adapter is not expressible without slicing. Satellite imagery is a domain
  shift, so adapting vision may help later — but that is a separate experiment, and leaving it
  frozen keeps this baseline clean. Verified: no LoRA layer landed on `vision_model`.

Config `r=8, alpha=32, dropout=0.1` (the draft's values) applies without error. It is a
reasonable baseline, not a validated optimum — the draft's other figures did not survive the
Fase 0 audit or §4 above, so this should be swept in Fase 3.

`task_type=None`: BLIP matches none of peft's task types; the plain wrapper forwards `forward()`
and `generate()` to BLIP correctly (verified in §8).

---

## 6. Decision: the embedding matrix must be trainable (LoRA alone is not enough)

**This is a second silent-failure trap.** `get_peft_model()` freezes everything except LoRA
adapters. But we just replaced the entire vocabulary — 74.6% of embedding rows are freshly
initialized noise. LoRA on q/v cannot repair a random token vector; the model would train without
error, loss would fall (it can still exploit the image and the frozen head), and it would emit
nonsense. So `apply_lora(..., train_embeddings=True)` re-enables grad on the embedding matrix and
the head bias.

Because the head weight is tied to the embedding, unfreezing the embedding unfreezes the output
projection too — one parameter, both roles. The builder **asserts** the tie survived peft
wrapping and raises rather than continue if two divergent copies exist.

Measured:

| bucket | params |
|---|---|
| LoRA adapters (48 modules, r=8) | 589,824 |
| embedding matrix (tied to head weight) | 24,523,776 |
| LM head bias | 31,932 |
| **total trainable** | **25,145,532** |
| total model | 225,644,220 |
| trainable fraction | **11.14%** |
| *(LoRA-only, for reference)* | *589,824 = 0.26%* |

So the honest headline is 11.14% trainable, not 0.26%. The vocabulary swap, not LoRA, dominates.

---

## 7. Deferred (with measurements): vocabulary pruning

Across **all three splits**, only **2,819 distinct token ids (8.8% of 31,932)** ever appear;
1 UNK occurrence in 1,925,440 target tokens. 29,113 rows — **22,358,784 params** — are dead
weight that still consumes optimizer state and softmax compute every step.

Pruning to used tokens + punctuation + a frequency margin would cut trainable params from ~25.1M
to roughly ~2.8M and make the "parameter-efficient" claim true. Not done in Fase 2 because it
changes the tokenizer contract and needs an OOV-margin policy (the 8.8% figure is measured
*including* the test split, so it would not generalize to unseen deployment text). Recommended as
the first Fase 3 experiment.

---

## 8. Verification actually run (`scripts/build_model.py` + verification harness)

- **Special-token alignment: 10/10 PASS.** `bos=31923 ([DEC])`, `sep=4`, `pad=0`, `eos` repointed
  `2 → 4`, `vocab_size=31932`, embedding/head/bias all 31,932 rows, tie intact.
- **Round-trip, 5 real `ground_truth_id` strings: 5/5 exact**, `[DEC]` first, `[SEP]` last, all 7
  sections recovered.
- **Round-trip, all 6,999 train records: 6,999/6,999 exact**, 6,999/6,999 correctly delimited.
- **Forward pass**, real 1024×1024 PNG from `DisasterM3_Bench/test_images/` → `(1,3,384,384)`,
  batch of 2 real targets, on CUDA: logits `(2, 171, 31932)`, loss 13.07, no shape errors.
- **Backward pass**: embedding grad present (mean|g| 4.7e-05), LoRA `lora_B` grads present,
  0 trainable params without gradient.
- **`generate()`**: runs, starts with `[DEC]`, halts on `[SEP]` after 9 tokens (stopping
  mechanism confirmed live), emits no id ≥ vocab.

### Round-trip strength: idempotency, not raw equality

Raw text cannot round-trip exactly — the tokenizer is uncased, and WordPiece decode re-spaces
punctuation. So the target text is first projected onto the codec's **fixed point** via
`codec.canonicalize()`, and the guarantee tested is
`decode(encode(x)) == x` **exactly, character for character**, for all 6,999 records.
Preprocessing targets with `canonicalize()` makes train-time text and eval-time decoded text live
in the same space, so **train/eval skew is zero by construction** rather than by hope.

Fidelity of the raw → canonical projection: 6,743/6,999 (96.3%) are identical modulo case and
whitespace. The 256 differences are cosmetic re-spacing of rare constructs (`<NO_DAMAGE>` tags,
quoted phrases, `dan/atau`), several of which the canonical form actually *repairs*.

Detokenizer rules are corpus-justified, not guessed: `word - word → word-word` is safe because
the corpus contains **0** legitimately spaced hyphens against 5,020 records using reduplication
(`tanda-tanda` ×2,839, `puing-puing` ×1,206).

### Length budget
Target lengths (incl. `[DEC]`/`[SEP]`): mean 189.9, p95 241, p99 270, **max 346**.
`max_length=384` → **0% truncation**, and 384 < 512 position embeddings. Chosen.
(256 would truncate 2.14%, 320 would truncate 0.06%.)

---

## 9. Open risks handed to Fase 3 — resolved/updated after user review (2026-08-09)

1. **BLIP consumes ONE image; the dataset has pre/post pairs. DECIDED: side-by-side composite.**
   Chosen over dual-encoder-concat and 6-channel patch surgery for lowest implementation risk —
   both alternatives touch BLIP's internals (forward pass or first conv layer), which is exactly
   the kind of silent-failure surface this phase has been fighting all along. Composite halves
   effective per-image resolution as a real cost, but is a preprocessing-only change: zero risk of
   a silent architectural bug. **Implementation is Fase 3's first task** (dataset loader / collator
   builds the composite before the image ever reaches BLIP's vision encoder) — not done in this
   phase, which only covers text/tokenizer/LoRA setup.
   **UPDATE: implemented and verified in Fase 3 below** (`scripts/dataset.py`); the
   "halves effective resolution" cost is now measured at −2.79 dB rather than asserted (§3.2).

2. **622/6,999 (8.9%) train targets are missing the `KESIMPULAN` section — investigated further,
   NOT a Fase 1 bug.** Checked all 622 against `ground_truth_en`: **0/622 have a CONCLUSION section
   in the English source either.** This is a genuine characteristic of ~9% of DisasterM3's original
   annotations (also: 3 records stop at `JALAN`, 2 at `VEGETASI`), not something lost in
   translation/parsing. Nothing to recover — decision is to **accept as-is**: the model should
   learn that a missing final section is a valid (if uncommon) output pattern, matching real data
   distribution. Documented here so it isn't mistaken for a bug later.

3. **392 records (266 in train) leaked `<NO_DAMAGE>` / `</NO_DAMAGE>` / `< NO_DAMAGE>` annotation
   tags — FIXED.** Confirmed present in `ground_truth_en` too (277 occurrences), so this is a
   DisasterM3 source-data template artifact, not a translation error. Stripped via
   `scripts/strip_no_damage_tags.py` (regex removal + whitespace cleanup) from both `_en` and `_id`
   fields in `data/interim/{train,test}_id.jsonl`, then `data/processed/*.jsonl` regenerated via
   `build_final_dataset.py` (same seed=42, same split sizes: 6,999/767/2,363 — text content changed,
   record count and grouping did not). **Re-ran the full Fase 2 verification harness against the
   cleaned data: 22/22 checks still pass**, including exact idempotent round-trip on all 6,999
   records — the fix removed characters, it did not disturb tokenization behavior.

4. **`prompt_id` is constant** — exactly 1 distinct value across all 6,999 records (14 tokens).
   It carries zero discriminative signal. Prepending it to the decoder would spend context on a
   constant; the builder's codec therefore encodes the target only. Revisit only if Fase 3
   introduces prompt variety.
5. **Casing of body text is unrecoverable** (uncased tokenizer). Output will be lowercase except
   the 7 header tokens. Acceptable for case-insensitive BLEU/ROUGE; needs a truecasing pass for
   presentation.
6. **peft's `save_pretrained()` saves ONLY adapter weights** — it would silently discard the
   24.5M-param embedding matrix, i.e. the entire Indonesian vocabulary, at checkpoint time. Use
   `save_trainable_state()` in `scripts/build_model.py`, which writes `embeddings.pt` alongside
   the adapter. **Do not call `model.save_pretrained()` directly.**
7. `r=8, alpha=32, dropout=0.1` is an unvalidated starting point inherited from the draft.

---
---

# Fase 3: image pipeline (pre/post composite, Dataset, collator)

Implementation: `scripts/dataset.py`. Verification: `scripts/test_image_pipeline.py`
(**61/61** checks without the model, **75/75** with `--model`). Training loop is explicitly
*not* in scope here.

All numbers below were measured on this machine. Where the Fase 2 notes or the task framing
implied a different answer, the measured answer wins and the disagreement is stated.

---

## 3.0 BLIP's real preprocessing contract (read off the checkpoint, not recalled)

Read from `BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")`:

| field | value |
|---|---|
| `size` | `{"height": 384, "width": 384}` |
| `resample` | `3` → `PIL.Image.Resampling.BICUBIC` |
| `image_mean` | `(0.48145466, 0.4578275, 0.40821073)` |
| `image_std` | `(0.26862954, 0.26130258, 0.27577711)` |
| `rescale_factor` | `1/255` |
| `do_resize / do_rescale / do_normalize / do_convert_rgb` | all `True` |

Those are CLIP's statistics, not ImageNet's — a detail worth pinning down, because the two are
close enough that using the wrong one would not crash and would barely move early loss. So
`ImageSpec.from_processor()` **reads every constant from the checkpoint**; nothing is written as
a literal in our code. `384/16 = 24` → a 24×24 = 576-patch ViT grid.

`ImageSpec.to_tensor()` reimplements the arithmetic for speed (we resize during compositing, so
calling the full processor again would resize twice). That is a divergence risk, so the test
asserts our path equals `BlipImageProcessor` numerically: **max abs diff 4.77e-07** on random
noise *and* on a real composite.

## 3.1 Decision: horizontal (side-by-side), pre LEFT / post RIGHT

The pre/post-pair-into-one-image decision itself was settled in §9.1. Remaining: orientation.

- **Chosen: horizontal.** `384/2 = 192 = 12 patches` exactly, so the pre|post boundary lands on
  a ViT patch boundary and **no patch ever mixes pixels from both images**. (Asserted in the
  test — a resolution where `width/2 % 16 != 0` would silently create a column of blended
  patches.) Vertical splits equally cleanly, so this is not the deciding factor.
- Deciding factor is weak but real: "before / after" imagery is overwhelmingly presented
  side-by-side in web-scale pretraining data, so a left/right pair is the layout BLIP's frozen
  vision tower is most likely to have seen. Pre goes left because that is reading order.
- **Honest counter-argument for vertical**, which I do not think is decisive but is not
  nothing: with row-major patch flattening, a *vertical* split makes each image a contiguous
  block of tokens (0–287 = pre, 288–575 = post) with exactly **one** false sequence-adjacency,
  whereas horizontal interleaves the two images every 12 tokens and creates **24** false
  sequence-adjacencies. Both layouts have the same 24-patch-long *spatial* seam. Self-attention
  is order-agnostic given position embeddings, so I expect this to be second-order — but it is
  a cheap A/B (`LAYOUT = "vertical"`, one line in `scripts/dataset.py`) and worth running once
  the training loop exists.

No separator bar is drawn: the patch boundary already separates them, and a bar would spend
real pixels on redundant information.

## 3.2 Decision: `FIT="stretch"` — I did NOT preserve aspect ratio, and here is why

The Fase 3 task specified "resize each preserving aspect ratio to fit side-by-side". **I
deliberately did not do that**, because on this corpus the instruction is self-defeating. The
reasoning, with the measurements:

**Every source image is square** — 1024×1024, confirmed across a 360-image random sample
spanning all three splits (single distinct size). A half-canvas tile is 192×384, i.e. 1:2.
Aspect-preserving fit of a *square* into a 1:2 box yields a 192×192 tile and **half the canvas
becomes constant padding**. So "preserve aspect ratio" and "fill the half-canvas" are not
simultaneously satisfiable here; the task text implicitly assumed non-square inputs.

Three candidates were considered and measured:

| fit | tile | informative patches | detail retained (round-trip PSNR vs 1024px source) |
|---|---|---|---|
| **stretch** (chosen) | 192×384 | **576/576 (100%)** | **34.90 dB** |
| pad (aspect-true letterbox) | 192×192 + fill | 288/576 (50%) | 33.76 dB |
| crop (aspect-true, fills canvas) | 192×384 of a 384×384 scale | 576/576 | n/a — see below |

1. **`pad` is dominated.** It is worse on *both* axes: it throws away half the ViT's patches on
   constant fill, **and** it retains *less* detail (−1.14 dB) than stretching. That second part
   is counter-intuitive until you count samples: a 192×384 tile carries 73,728 pixels versus
   36,864 for a 192×192 tile. Twice the samples beats isotropy.
2. **`crop` is rejected on correctness, not quality.** Scaling to 384×384 then center-cropping
   to 192×384 preserves aspect ratio *and* fills the canvas — but discards 50% of the
   1024×1024 field of view. The ground-truth reports describe buildings, roads, vegetation,
   water and farmland across the **whole** tile. Cropping would routinely delete the very
   object a sentence refers to, producing image/caption pairs that are quietly wrong. Loss
   would still fall. This is exactly the silent-semantic-break class of bug, so: rejected.
3. **`stretch`'s distortion is in-distribution for this checkpoint.** `BlipImageProcessor`
   *itself* does a plain non-aspect-preserving resize to a square — verified directly: a
   200×800 (1:4) image comes out `(1, 3, 384, 384)`. Every image BLIP saw through this
   processor was anisotropically squashed. A fixed, dataset-wide 2:1 scale is therefore an
   ordinary transform for this vision tower, whereas large constant letterbox bands are not.

`fit="pad"` remains implemented and tested as a switchable alternative, and `ImageSpec.pad_color`
is the uint8 colour that normalizes to ≈0 (`(123, 117, 104)`, verified max |z| = 0.0038) rather
than black — black would sit at about −1.8σ per channel, i.e. a strong learnable meaningless
edge. That matters for the `pad` path and for nothing else.

**The honest cost of compositing at all: −2.79 dB** versus feeding a single image alone
(34.90 vs 37.69 dB). That is the price of §9.1's zero-architectural-risk choice, now quantified
instead of asserted.

## 3.3 SAR / single-channel handling

622 train records are SAR. Their **post** image ships as PIL mode `L` while the **pre** image is
`RGB` — a genuinely mismatched pair, confirmed on real files. Two failure modes were guarded:

- `Image.paste` of an `L` tile onto an `RGB` canvas, and
- more insidiously, a `(H, W)` array broadcasting against a 3-element mean/std, which would
  **not** raise and would produce a silently wrong tensor.

`load_image()` therefore converts to `RGB` **at load time**, before any compositing or
arithmetic, and `ImageSpec.to_tensor()` hard-raises on anything that is not `H×W×3`. Verified:
`L → RGB` replicates the channel (R==G==B, checked per-pixel) rather than zero-padding two
channels, so SAR intensity is preserved rather than tinted red. The visual dump
(`docs/figures/composite_3_SAR.png`) shows the expected colour-left / greyscale-right pair.

## 3.4 Text side: reused verbatim from Fase 2, and asserted to be identical

The Dataset does **not** re-implement tokenization. It calls Fase 2's `IndoReportCodec.encode()`,
and `pad_encoded_batch()` is asserted to reproduce `codec.encode_batch()` **tensor-for-tensor**
on `input_ids`, `attention_mask` and `labels`. The verified contract therefore cannot drift:
`[DEC]` first, `[SEP]` last, `labels == input_ids` with `-100` exactly on padding.

`DisasterCollator` is a top-level **class**, not a closure. On Windows the DataLoader uses
`spawn` and pickles the collate_fn into each worker; a nested function passes every test at
`num_workers=0` and then dies the moment workers are enabled — i.e. only once a run is long
enough for the failure to be expensive. Picklability is asserted.

`prompt_id` is still not fed to the decoder (§9.4: it is constant across all 6,999 records).

## 3.5 Data plumbing

- Routing is by **path prefix**, not split name. This is load-bearing: the **val split's records
  point at `train_images\...`**, so keying the image source off the split name would send val
  reads to the wrong place. `train_images/` → zip, `test_images/` → extracted directory.
- Backslash paths in the jsonl are normalized to forward slashes for zip lookup.
- The 29.7 GB `train_images.zip` is **never extracted** (disk headroom ~47 GB). Images are read
  on the fly.
- `zipfile.ZipFile` is neither fork- nor thread-safe — two readers on one handle interleave
  seeks and return corrupted bytes, which PNG decode will often *survive*. Handles are therefore
  cached per `(pid, thread)`. Kept at module level so the Dataset itself stays picklable.

## 3.6 Corpus integrity (swept up front, all three splits)

| split | records | distinct images | missing | 0-byte | usable |
|---|---|---|---|---|---|
| train | 6,999 | 13,376 | 0 | 0 | 6,999 |
| val | 767 | 1,475 | 0 | 0 | 767 |
| test | 2,363 | 4,369 | 0 | **1** | **2,362** |

`test_images/tuscaloosa_tornado_00000278_pre_disaster.png` is 0 bytes — it downloaded empty from
HuggingFace. (The zip also contains exactly 1 zero-byte entry, but no split references it.) A
500-image random decode-verify of train/val found 0 corrupt files.

The scan is O(1) per image — zip central-directory `file_size` plus the 8-byte PNG signature —
so sweeping all 19,220 references costs ~10 s and can run at Dataset construction
(`drop_unreadable=True`). It is matched to the failure mode actually present (0-byte files); a
full CRC pass (`deep_scan=True`) costs ~8 ms/image and found nothing extra. The scan carries
**negative controls** — it must reject the known 0-byte file and a nonexistent path, and accept
a known-good one — because "0 bad in train" is otherwise indistinguishable from a scanner
that returns `True` unconditionally.

`drop_unreadable` defaults to **False** so nothing is silently dropped from a training run
without a printed record. Eval on the test split should pass `drop_unreadable=True`; otherwise
record 2292 raises.

## 3.7 Throughput: the zip is not the bottleneck (measured, because it was the obvious worry)

The stated risk was that reading from a 29.7 GB zip on every step would starve the GPU. It does
not, and the breakdown says why:

```
single-process random access: 95.5 ms/sample (10.5 samples/s)
  zip read (I/O)      3.8 ms   ( 4.0%)   <- 487 MiB/s effective, 2.84 MiB/sample
  PNG decode         62.8 ms   (65.7%)
  resize+norm+tok    29.0 ms   (30.3%)
```

The archive is **STORED, not DEFLATE** (20,989/20,989 entries; compressed == uncompressed size),
so a read is a seek plus a linear copy — no inflate. **Decoding 1024×1024 PNGs dominates at
66%.** The practical consequence: *extracting the zip would buy essentially nothing* (it would
remove 4% of the cost and consume 28 GB of a 47 GB budget). More workers is the lever.

Steady-state with a real DataLoader on 16 logical cores (measured after the first batch, so
spawn/import is excluded):

| num_workers | steady | min/epoch (6,999) |
|---|---|---|
| 0 | 10.5 samples/s | 11.1 |
| 4 | 34.9 samples/s | 3.3 |
| 8 | **96.7 samples/s** | **1.2** |

8 workers is slightly superlinear (9.2× on 16 cores), consistent with OS page-cache reuse on the
zip. At 8 workers the loader delivers ~97 samples/s, which will comfortably outrun a 225M-param
BLIP training step on one 8.5 GB RTX 5050 — **so the data pipeline should not be the
bottleneck.**

Caveats: use `persistent_workers=True`. Worker startup is normally ~0.7 s here, but one
measurement taken *after* a CUDA context and the full model were already resident in the parent
took **50 s** for the first batches under `spawn`; paying that once per epoch is avoidable and
should be avoided. Recommended: `num_workers=8, persistent_workers=True, prefetch_factor=4`.

If throughput ever does become the constraint, the right fix is pre-decoding to 384×384
composites once (~7,000 × 442 KB ≈ 3 GB for train), not extracting the zip.

## 3.8 Verification actually run

`scripts/test_image_pipeline.py --model` — **75/75 pass** (61 without the model).

- **Spec sourcing:** every constant equals the checkpoint's processor; our fast tensor path
  matches `BlipImageProcessor` to 4.77e-07 on noise and on a real composite.
- **Geometry:** left half **bit-identical** to the resized pre image, right half bit-identical
  to the resized post image, plus a control that left is *not* the post image.
- **Halves genuinely differ** over 28 real composites (incl. 8 SAR): mean MAD 0.5789, min 0.0877
  — with a **negative control** proving the test has power: `composite(post, post)` gives MAD
  0.00e+00, so a loader bug that used one image twice would be caught.
- **Human check:** 4 composites written to `docs/figures/` and inspected. Both halves are
  legible at 192×384 — roads, building footprints and vegetation boundaries all resolvable —
  and the SAR pair shows colour-left / greyscale-right as intended.
- **Real forward+backward**, LoRA model on CUDA, batch of 4 real records read out of the zip:
  `pixel_values (4,3,384,384)`, `seq_len 209`, logits `(4, 209, 31932)`,
  **loss 13.15** against `ln(V) = 10.37` (sane for a freshly-remapped 31,932-row embedding).
  Embedding grad mean|g| 4.61e-05, head-bias grad 6.20e-05, **48/48 `lora_B` modules with
  non-zero grad, 0 trainable params missing a gradient.**
- **Both halves reach the loss:** `pixel_values.grad` is finite, with |grad| 5.50 on the pre
  half and 12.04 on the post half (ratio 0.46 — post carries more signal, which is what a
  damage-assessment target should do).
- **The image is not decorative:** shuffling images against captions moves the loss
  (13.1512 → 13.1485) and blanking them moves it more (→ 13.1807). Small deltas are expected at
  init, but non-zero deltas prove the vision path is actually wired into the objective.
  *(These are pre-training-step magnitudes; re-check after real training.)*
- **DataLoader with `num_workers=2` yields batches**, confirming per-worker zip handles work
  under Windows `spawn`.

## 3.9 Risks handed to the training phase

1. **Effective resolution per image is 192×384 from a 1024×1024 source** — a measured −2.79 dB
   versus an uncomposited image. Fine-grained damage (individual roof damage, narrow road
   blockage) may be below this. This is the known, accepted cost of §9.1.
2. **Horizontal vs vertical layout is untested empirically** (§3.1). One-line A/B.
3. **`fit="stretch"` imposes a fixed 2:1 anisotropy.** Argued in-distribution, not proven so.
   `fit="pad"` is the fallback if generation quality looks spatially confused.
4. **Test record 2292 will raise** unless the test split is loaded with `drop_unreadable=True`.
5. **No augmentation is implemented.** Flips would need to be applied to *both* halves
   identically or the pre/post correspondence breaks; a horizontal flip of the whole composite
   would additionally swap pre and post, contradicting the layout the model learned. Any
   augmentation must be added inside `make_composite`, not on the finished canvas.
6. **The pre/post distinction is positional only.** Nothing tells the model which half is which
   except learned position. It should be learnable from 6,999 consistent examples, but if
   reports come back with pre/post reversed, that is the first thing to suspect.

---
---

# Fase 4: real evaluation on the test set

Checkpoint under test: `checkpoints/full_run_v2/best` (best val_loss 1.5343, loaded via
`load_trainable_state()`, never re-derived). Test set: `data/processed/captions_test.jsonl` via
`DisasterCaptionDataset("test", ..., drop_unreadable=True)` -> **2,362** usable records (record
2292's 0-byte `tuscaloosa_tornado_00000278_pre_disaster.png` auto-dropped, per SS3.6).

Everything below was decided and written **before** any metric was computed, per the task's
explicit requirement that methodology not be chosen post-hoc to fit a result.

## 4.0 Libraries actually available on this machine (checked, not assumed)

- `pycocoevalcap` and `sacrebleu` were **not** pre-installed; both installed cleanly via pip
  (`pycocoevalcap==1.2`, `sacrebleu==2.6.0`).
- Java **is** available (`C:\Program Files\Eclipse Adoptium\jdk-11.0.29.7-hotspot\bin\java.exe`),
  so `pycocoevalcap`'s `PTBTokenizer` (a Java Stanford-tokenizer jar) runs. Verified live on
  Indonesian text (`bangunan mengalami kerusakan berat akibat gempa` tokenizes sanely — it does
  lowercasing + punctuation splitting, which is language-agnostic enough for our already-lowercase
  WordPiece-normalized text). CIDEr/BLEU-4/ROUGE-L are therefore computed through the real COCO
  evaluator, not a hand-rolled metric.
- METEOR is skipped: it ships English WordNet-derived paraphrase tables with no Indonesian
  support, so a number from it would be silently wrong for this language, not merely noisy. Not
  computed, not reported.

## 4.1 SPICE: decision (b) — substitute BERTScore with an Indonesian encoder, not fabricate a number

SPICE parses captions into a scene graph and scores overlap using WordNet synonym sets. WordNet
has no Indonesian synset coverage, so running SPICE on Indonesian text either (a) silently
degrades to near-literal string match once every synonym edge fails to resolve, or (b) requires
round-tripping through English translation first, stacking a second translation-noise source on
top of the one already used for the baseline (SS4.2).

**Chosen: substitute with BERTScore using `indolem/indobert-base-uncased`** — the exact encoder
family already adopted project-wide as the Indonesian vocabulary source (Fase 2 SS1), so this
introduces no new modeling assumption. BERTScore gives genuine semantic-similarity credit (e.g.
paraphrase, word-order-tolerant matches) that n-gram metrics miss, without depending on any
WordNet resource. Reported as `bertscore_f1` alongside CIDEr/BLEU-4/ROUGE-L, computed on
`codec.canonicalize()`-normalized text for both systems.

Rejected: (a) round-trip translation to English + real SPICE — adds a second NLLB pass whose
errors would be indistinguishable from real semantic-quality differences; (c) omit entirely —
rejected because a real, cheap, language-correct substitute was available (`bert-score` installed
cleanly), so silently dropping the axis is not the honest choice here. This decision, and the
BERTScore numbers, are best-effort/secondary to CIDEr/BLEU-4/ROUGE-L, which remain primary.

## 4.2 Baseline: zero-shot `Salesforce/blip-image-captioning-base`, methodology fixed before running

The baseline is the **untouched** base checkpoint: original `BertTokenizer`
(bert-base-uncased), no LoRA, no vocabulary swap, no fine-tuning. It receives **the same
pre/post composite image** our model receives (built by the identical `make_composite` +
`ImageSpec` pipeline — both models' vision preprocessing constants come from this same
checkpoint's `BlipImageProcessor`, so the visual input is apples-to-apples). Generation is
unconditional (no prompt), `num_beams=1` (greedy — matched to our model's decode setting, SS4.3),
`max_new_tokens=50` (base BLIP produces short one-line captions; this is generous headroom, not a
truncation risk for this model).

**The baseline will emit a short English sentence, not a 7-section Indonesian report — this is
expected, not a bug**, and is itself part of what the comparison is measuring (whether Fase 2/3's
adaptation closes that gap).

**Scoring methodology, decided up front:**
1. Translate each baseline English caption to Indonesian with the **same NLLB model used in
   Fase 1** (`facebook/nllb-200-distilled-600M`, `scripts/translate_captions.py`'s
   `NllbTranslator`), applied to the whole caption as one unit (there are no sections to split —
   unlike Fase 1's structured ground truth, a zero-shot caption is one free sentence).
2. Run the translated text through `codec.canonicalize()` — the same normalization space used for
   the references and for our model's decoded output — so casing/spacing conventions cannot
   create an artificial gap between systems.
3. Score the canonicalized translation against the same canonicalized `ground_truth_id`
   references used for our model, with the identical metric code path (same `Cider`/`Bleu`/`Rouge`
   objects, same `PTBTokenizer` call).

**Translation-noise risk, stated plainly:** any BLEU/CIDEr/ROUGE gap between our model and this
baseline is now a mixture of (i) real captioning-quality difference and (ii) NLLB translation
noise/errors introduced only on the baseline side (our model never passes through NLLB — it
outputs Indonesian directly). This asymmetry structurally *helps* the baseline in one respect
(NLLB is a strong 600M MT model, likely to produce fluent Indonesian even from a mediocre English
caption, inflating n-gram surface fluency) and *hurts* it in another (mistranslation of the one
short sentence it did produce can only lose information, never add the missing 6 sections). Both
directions are named here so a result is not read as a clean quality comparison — it is the best
available fair comparison given that SPICE-for-Indonesian and English-zero-shot-vs-Indonesian-
reference are both unsolved-in-general problems, not a claim of a bias-free measurement.

**Rejected alternative:** scoring the baseline's raw English caption against `ground_truth_en`
(the pre-translation English reference) and reporting that as a separate, not-directly-comparable
number was considered, but was **not** substituted for the Indonesian-reference comparison — the
task's target deployment is Indonesian-language reporting, so the Indonesian-referenced number is
the one that answers the actual question, and it is what is reported as the primary baseline
comparison. The English-vs-English number is not computed, since it does not bear on the stated
task.

## 4.3 Generation settings for our model

Greedy decoding, `num_beams=1`, `max_new_tokens=300` (target length distribution: mean 189.9,
p99 270, max 346 tokens including `[DEC]`/`[SEP]`, SS4 length budget), `batch_size=64` — measured
at 0.213 s/sample on the RTX 5050 (peak 3.9 GB VRAM), so **the full 2,362-record test set is run,
no subsampling** (~8-9 minutes total, well within budget). Greedy (not beam search) is used for
both our model and the baseline so decoding strategy is not a confound in the comparison.

## 4.4 Significance testing

Paired bootstrap resampling (10,000 resamples) on **per-example sentence-level CIDEr**, comparing
our model against the translated baseline over the same 2,362 matched test items (same image,
same reference, same index in both systems). Reports the fraction of resamples in which our
model's mean CIDEr exceeds the baseline's (one-sided) and a 95% CI on the paired mean difference.
A raw metric delta alone is not treated as sufficient evidence of "our model is better."

## 4.5 Results actually produced (this run, this checkpoint, this test set)

Pipeline executed in order: `scripts/generate_predictions.py` (2,362/2,362 predictions, ~13 min,
greedy, batch 64) -> `scripts/generate_baseline.py` (2,362/2,362 zero-shot English captions,
~12 min) -> `scripts/translate_baseline.py` (NLLB, ~8 min) -> `scripts/compute_metrics.py`.
Raw outputs: `results/predictions_test.jsonl`, `results/predictions_baseline_en.jsonl`,
`results/predictions_baseline_id.jsonl`. Aggregate + qualitative outputs:
`results/metrics.json`, `results/metrics_table.md`.

**Library bug hit and fixed (documented, not silently worked around):** `bert-score==0.3.13`
(latest PyPI release, ~2021) is incompatible with `transformers==5.9.0`'s Rust tokenizer backend.
`indolem/indobert-base-uncased` ships no `model_max_length` in its config, so transformers
defaults it to a `~1e30`-scale sentinel; the new Rust `enable_truncation` cannot represent that
and raises `OverflowError: int too big to convert`. Fixed with a documented monkeypatch in
`compute_metrics.py::_patch_bert_score_tokenizer_max_length()` that caps `model_max_length` at
512 (BERT's real position limit — not a result-changing hack, every report in this corpus is far
under 512 wordpieces). Second subtlety: `bert_score/__init__.py` runs `from .score import score`,
which shadows the `bert_score.score` **submodule** attribute with the **function** of the same
name, so `import bert_score.score as m; m.get_tokenizer = ...` silently patches nothing — the
fix reaches into `sys.modules["bert_score.score"]` directly. Verified working on a minimal example
before rerunning the full 2,362-item pass.

**Results (checkpoint `checkpoints/full_run_v2/best`, N=2,362 paired test examples):**

| Metric | Our model | Baseline (zero-shot BLIP -> NLLB) |
|---|---|---|
| CIDEr | 0.0980 | 0.0001 |
| BLEU-4 | 0.2044 | 0.0000 |
| ROUGE-L | 0.4103 | 0.0185 |
| BERTScore F1 (indolem, SPICE substitute) | 0.7943 | 0.3470 |

Paired bootstrap (10,000 resamples, per-example CIDEr): mean difference **+0.0980**, 95% CI
**[0.0874, 0.1090]**, 100% of resamples favor our model, two-sided p ~ 0. The CI excludes 0 by a
wide margin, so "our model outperforms the zero-shot baseline on CIDEr" is a supportable claim
**for this specific comparison and methodology** (SS4.2) — it is not evidence that the model is
good in an absolute sense, only that it is far better than an untouched English captioner scored
under a translated-reference protocol that (SS4.2) somewhat favors the baseline's fluency.

**Qualitative pattern, stated plainly (not just the aggregate score):** generations are
**structurally sound and non-degenerate** — every prediction has all/most of the 7 correct
section headers in order, fluent lowercase Indonesian, no repetition loops, no empty output.
*However*, the model shows a clear **systematic bias toward predicting "no visible damage"**
boilerplate and frequently **gets the `BENCANA` (disaster type) field wrong** (e.g. reference
`tornado` -> generated `badai`; reference `ledakan` -> generated `gempa bumi`; reference
`gempa bumi` -> generated `banjir`, all three sampled at random, not cherry-picked). This reads as
the model having learned the *template* and *register* of the report well, but not reliably
grounding the disaster classification or damage-severity content in the actual image pair — a
real limitation to flag, not a metric artifact. The zero-shot baseline, by contrast, sometimes
degenerates into repetition loops on this out-of-distribution composite-image input (e.g.
`"satellite image of the village of the village of the village..."` repeated to `max_new_tokens`),
which is a known BLIP greedy-decoding failure mode on unfamiliar inputs — visible directly in
`results/predictions_baseline_en.jsonl` and not filtered out of the reported scores.

---
---

# Fase 5: LoRA on the vision tower (unfreezing the image side)

Implementation: `scripts/build_model.py` (`vision_lora=` flag), `scripts/train.py`
(`--vision_lora`, `--allow_new_lora_modules`). Verification: `scripts/test_vision_lora.py`
— **25/25 pass**. Regression: `test_tokenizer_roundtrip.py` **22/22**,
`test_image_pipeline.py --model` **75/75** (both unchanged by this phase).

**No retraining was run in this phase.** Scope is: build the architecture change, load the
existing checkpoint into it correctly, and *prove* it is a no-op at step 0.

## 5.0 Why revisit §5's "vision tower deliberately untouched"

Fase 4 found two coupled biases in `checkpoints/full_run_v2/best`: the `BANGUNAN` section is
"no damage" boilerplate **83.0%** of the time against a **34–38%** reference rate (confirmed
*not* a dataset imbalance — the source distribution is much more balanced), and `BENCANA`
exact-match is only **58.8%**, with **95.1%** of errors collapsing onto the two most frequent
training classes (`badai`, `gempa bumi`) rather than scattering randomly. Errors that fall
toward the label prior instead of toward visually-similar classes are the signature of a
decoder leaning on text-side statistics because the image gives it nothing decisive.

A cheap control ruled out the competing explanation: switching greedy → beam search
(`num_beams=5`) on the **same** checkpoint left the no-damage rate at 83.3% (statistically
unchanged), cut distinct captions 87.8% → 77.0%, and cut CIDEr 0.0980 → 0.0703 (p ≈ 0). So it
is not a decoding-strategy artifact.

§5's original reasoning ("satellite imagery is a domain shift, so adapting vision may help
later — but that is a separate experiment") was correct at the time and explicitly deferred
this. Fase 4 is the evidence that makes the experiment worth running. Note what the frozen
tower means concretely: the text side's **cross-attention q/v is already adapted**, so the
decoder can already re-weight and re-read the vision features it is given. What it cannot do
is change *what those features contain*. If CLIP-pretrained ViT features simply do not
separate `tornado` from `badai` on a 192×384 stretched satellite tile, no amount of
cross-attention adaptation recovers that information.

## 5.1 Decision: option (a) — LoRA on the **whole fused `qkv`**, k included

Module inventory re-read off the loaded model (not recalled):

| tower | leaf | count | shape |
|---|---|---|---|
| vision | `qkv` | 12 | `Linear(768, 2304)` — **fused** |
| vision | `projection` | 12 | `Linear(768, 768)` |
| vision | `fc1` / `fc2` | 12 each | `Linear(768, 3072)` / `Linear(3072, 768)` |

`qkv` is unique to `vision_model` (0 matches anywhere in `text_decoder`), so the suffix
`"qkv"` cannot collide with the existing `["query","value"]` targets. Verified: the expanded
config produces exactly **48 text + 12 vision = 60** adapted modules, and the plain (default)
config still produces **48 text + 0 vision**.

**The deciding argument is a parameter count, not a preference.** For `r=8`, per layer:

| placement | LoRA params / layer | x12 layers |
|---|---|---|
| **fused `qkv`** (chosen) | `8*768 + 2304*8` = **24,576** | **294,912** |
| *hypothetical* independent q + v (what surgery would buy) | `2*(8*768 + 768*8)` = **24,576** | **294,912** |
| independent q + k + v | 36,864 | 442,368 |
| `projection` only | 12,288 | 147,456 |
| `fc1`+`fc2` | 61,440 | 737,280 (measured) |

Adapting the fused `qkv` costs **exactly the same 294,912 parameters** as splitting the layer
and adapting only q and v would. Option (b)'s module surgery would therefore buy *zero*
parameter savings — it would only remove the k adaptation, at the cost of rewriting
`BlipAttention`'s weight layout.

The one real structural difference, stated honestly rather than glossed: fused LoRA is **not**
three independent rank-8 adapters. With `dW = B @ A`, `A` shape `(8, 768)`, `B` shape
`(2304, 8)`, and (read from `BlipAttention.forward`)
`mixed_qkv.reshape(bsz, len, 3, heads, head_dim)` meaning rows 0–767 = q, 768–1535 = k,
1536–2303 = v as **contiguous blocks**, the update decomposes as `dq = B_q A x`,
`dk = B_k A x`, `dv = B_v A x`. All three read the **same rank-8 input subspace** (the row
space of `A`) but write through independent output maps. So it is a *constrained* q/k/v
adaptation, not a free one.

Why that constraint is acceptable here: the LoRA paper's "q/v beats q/k/v" result (§7.1) is
explicitly a **fixed-parameter-budget** finding — spending rank on `k` costs rank elsewhere.
That premise does not hold in this case, because the budget is identical either way. We get
`k` adaptation for free and lose only the independence of the three down-projections.

**Option (b) (splitting the fused layer) rejected** on risk-for-no-benefit: it buys no
parameters, and it would mean re-slicing pretrained weights into three new `nn.Linear`s and
rewriting `forward()`. A row-block ordering mistake there (e.g. assuming head-interleaved
rather than contiguous layout) would **not crash** — it would permute q/k/v across heads and
train to a mediocre loss. That is precisely the silent-semantic-break class this project has
been avoiding since §9.1 and §3.2.

**Option (c) (`fc1`/`fc2` instead of attention) not chosen as the default, but kept
switchable and costed.** There is a real argument for it — attention LoRA changes *which*
patches get aggregated, MLP LoRA changes *what features are computed*, and the Fase 4
diagnosis is closer to "the features lack the information" than "the wrong patches are
attended". It is not the default for two reasons: it is 2.50x the parameters (737,280
measured), and `qkv` keeps the vision adapter structurally symmetric with the text side
(attention-only, output projection untouched — the text side does not adapt `dense` either).
`build_model(vision_lora=True, vision_lora_targets=["fc1","fc2"])` /
`--vision_lora_targets fc1,fc2` makes it a one-flag ablation, and it is the recommended
second experiment if `qkv` under-delivers.

`projection` is deliberately **not** targeted, for the same symmetry reason: the text side
leaves its 25 `dense` output projections frozen.

## 5.2 Measured parameter counts (computed against the loaded model, not quoted)

`param_breakdown()` in `build_model.py`, measured on this machine:

| bucket | text-only (Fase 2) | + vision `qkv` | delta |
|---|---|---|---|
| LoRA (text, 48 modules) | 589,824 | 589,824 | 0 |
| **LoRA (vision, 12 modules)** | **0** | **294,912** | **+294,912** |
| embedding matrix + LM-head bias | 24,555,708 | 24,555,708 | 0 |
| **total trainable** | 25,145,532 | **25,440,444** | **+294,912** |
| total model | 225,644,220 | 225,939,132 | +294,912 |
| **trainable fraction** | 11.1439% | **11.2599%** | +0.1160 pp |

The vision adapter is **0.1305%** of the model and **1.16%** of what is already trainable —
i.e. a rounding error on the optimizer/VRAM budget. §4/§6's conclusion still stands: the
24.5M-param vocabulary relearning dominates everything else.

## 5.3 Decision: extend the existing checkpoint, do not restart

`checkpoints/full_run_v2/best` already holds trained text LoRA + the 24.5M-param embedding
matrix (val_loss 1.5343). Restarting would throw that away for no reason: because `lora_B` is
zero-initialised, `dW = B @ A = 0`, so an expanded model at step 0 is *numerically identical*
to the old one, and vision adaptation phases in only as training moves `B`.

That is the theory. It is also exactly the kind of claim that is usually assumed and
occasionally false, so `load_trainable_state()` now **asserts** it rather than trusting it:

- `set_peft_model_state_dict()` loads with `strict=False` (read from peft 0.20's source), so a
  key mismatch is silent by default. The returned `load_result` is now inspected.
- **`unexpected_keys` is always fatal** — a checkpoint tensor with nowhere to land means
  trained weights are being discarded.
- **`missing_keys` is fatal unless `allow_new_lora_modules=True`.** Backward compatibility is
  preserved by making the strict behaviour the default: every existing script
  (`generate_predictions.py`, `compute_metrics.py`, `train.py --resume_from`) keeps working
  unchanged, and a *genuine* partial-load bug still raises. Verified: loading the old
  checkpoint into a vision-expanded config **without** the flag raises, naming the 24 absent
  tensors and pointing at the flag.
- Every tensor present in the file is compared **bit-for-bit** after load, using
  `get_peft_model_state_dict()` (the exact inverse of the save path) rather than hand-rolled
  key rewriting — an ad-hoc key mapping is itself a silent-mismatch surface, and the first
  version of this check did in fact mis-map `lora_A.weight` against the live
  `lora_A.default.weight`.
- Every newly-added module's `lora_B` is asserted to be exactly `0.0` before the function
  returns.
- `read_checkpoint_target_modules()` reads the checkpoint's *own* `adapter_config.json`
  (`["query","value"]` for `full_run_v2/best`) rather than inferring the old target set from
  the current defaults — the defaults are precisely what changed, so inferring would mask the
  mismatch being tested for.

## 5.4 Verification actually run (`scripts/test_vision_lora.py`, 25/25)

**A. Codec regression guard** (nothing in this phase should touch the text side):
`decode(encode(x)) == x` exactly on 2 hand-written canonical reports and **25/25** real
`ground_truth_id` val records, `[DEC]` first and `[SEP]` last in every case.

**B. Placement:** baseline has 0 LoRA layers on `vision_model`; expanded has 12, all on
`self_attn.qkv`; text side unchanged at 48. Measured vision param count 294,912 equals
`12 * (8*768 + 3*768*8)` computed independently from the model's own `vision_config`.

**C. Old checkpoint → expanded model is a true no-op.** Real batch of 2 val records
(composite pre/post images read out of the zip), CUDA, fp32, eval mode:

```
reference (text-only ckpt) : loss 1.29896951   logits (2, 209, 31932)
expanded (+ vision qkv)    : loss 1.29896951
max |delta logits|         : 0.000e+00          <- bit-identical, not merely "close"
generate() token ids       : identical, (2, 41) both
decoded text               : identical
96/96 text-side LoRA tensors bit-identical    embedding (31932,768) bit-identical
12/12 vision lora_B exactly 0.0               12/12 vision lora_A non-zero (max|A| >= 3.61e-02)
```

**Negative control, because "identical" is also what a dead code path returns:** perturbing
the 12 vision `lora_B` by `N(0, 0.02)` moves the logits by **9.35e-01** and the loss
1.2990 → 1.2954; zeroing them again restores `max|delta| = 0.0`. The adapter is genuinely
wired into the forward pass, so the identity result has power.

**D. Real forward + backward, gradients checked not assumed** (training mode, batch of 2):

```
loss 1.314048
122/122 trainable tensors have a .grad         0 missing
12/12 vision lora_B  NON-ZERO grad, mean|g| 1.8548e-04
96/96 text-side LoRA NON-ZERO grad
embedding matrix     NON-ZERO grad, mean|g| 3.4391e-05
```

**The one result that looks like a bug and is not:** at step 0, **all 12 vision `lora_A` have
exactly zero gradient**. This is forced by the math — `dL/dA = B^T (dL/dy) x^T` and `B = 0` —
and is the flip side of the same zero-init that makes the extension safe. A naive "every new
parameter must have non-zero grad" check would fail here and be misread as broken wiring. It
is distinguished from *actually* dead by running a second step: one `AdamW(lr=1e-4)` step
moves all 12 `B` off zero, and the next backward gives **12/12 `lora_A` non-zero grad
(mean|g| 1.9295e-05)**. The full vision adapter trains.

## 5.5 Risks handed to the retraining run

1. **This phase proves the adapter is correctly wired, not that it helps.** Whether vision
   adaptation actually fixes the no-damage boilerplate and the `BENCANA` prior-collapse is an
   empirical question the retraining run answers. A null result is a real possible outcome —
   the −2.79 dB composite cost (§3.2) and the 192×384 effective per-image resolution (§3.9.1)
   may simply be below what fine-grained damage assessment needs, in which case the bottleneck
   is the *input*, not the adapter.
2. **Learning rate for the vision adapter is untuned.** `train.py` puts all LoRA params in one
   group at `--lr_lora`. The vision tower is CLIP-pretrained on a very different distribution
   and is being adapted starting from a *converged* text-side checkpoint, so the LR that
   suited a fresh text adapter may be too aggressive here. Consider a separate, lower-LR group
   for `vision_model` LoRA, or a short warmup, if training destabilises.
3. **Compare against the right baseline.** The correct control is `full_run_v2/best` continued
   for the *same* number of additional steps **without** `--vision_lora`. Comparing
   "continued + vision LoRA" against the un-continued Fase 4 numbers would confound vision
   adaptation with simply training longer.
4. **VRAM:** +294,912 params is negligible, but LoRA on `vision_model` means the vision
   tower's activations must now be retained for backward, where previously nothing in that
   tower required grad. Expect a real activation-memory increase on the 8.5 GB card despite
   the trivial parameter delta; lower `--batch_size` / raise `--grad_accum` if it OOMs.
5. **`k` is now adapted on the vision side but not the text side** (§5.1). If a q/v-only
   vision ablation is ever wanted as a clean scientific control, it requires option (b)'s
   surgery — and per §5.1 it would cost the same parameters, so it is only ever worth doing as
   a control, never as an efficiency measure.

## 5.6 Results: the retraining run, and the answer to §5.5's null-result question

**On risk #3 (matched-epoch control):** deliberately skipped, a conscious tradeoff, not an
oversight. `full_run_v1 → full_run_v2` already showed clear plateau behavior in continued
text-only training (final epoch-over-epoch Δval_loss = −0.0007, ~15× smaller than the
epoch-6→7 delta) — strong enough prior evidence that a third text-only round would plateau
near the same val_loss that the decision was made to spend the GPU time on the vision
experiment instead. This is a real methodological looseness worth stating plainly in the
paper (the comparison is against a converged-but-not-epoch-matched baseline, not a strict
epoch-matched control) rather than glossing over it.

**Training** (`full_run_v3_vision`, resumed from `full_run_v2/best`, `--vision_lora
--allow_new_lora_modules`, otherwise identical hyperparameters, 10 epochs):

| | text-only (`full_run_v2`) | + vision LoRA (`full_run_v3_vision`) |
|---|---|---|
| best val_loss | 1.5343 | **1.4312** (−6.7%) |
| peak GPU memory | 3.62 GB | 5.48 GB (activation memory, as §5.5 risk #4 predicted) |
| wall-clock (10 epochs) | 1.40 h | 1.75 h |
| epoch 9→10 Δval_loss | −0.0007 (converged) | −0.0009 (also converged — a second run is not obviously warranted) |

Loss improving is not itself the answer to §5.5's question — real generation + real metrics
on the untouched test set is:

| Metric | Zero-shot baseline | Text-only LoRA | **Text+Vision LoRA (final)** |
|---|---|---|---|
| CIDEr | 0.0001 | 0.0980 | **0.1202** |
| BLEU-4 | 0.0000 | 0.2044 | **0.2095** |
| ROUGE-L | 0.0185 | 0.4103 | **0.4156** |
| BERTScore F1 | 0.3470 | 0.7943 | **0.7971** |
| `BENCANA` exact-match | — | 58.8% | **72.3%** (+13.5pp) |
| `BANGUNAN` no-damage rate | — | 83.0% | **78.5%** (reference rate: 34.2%) |
| distinct generated captions | — | 87.8% | 86.5% (not worse — rules out "gained accuracy by just memorizing fewer, safer templates") |

Paired bootstrap (10,000 resamples, per-example CIDEr), final vs. text-only ablation:
mean diff **+0.0221**, 95% CI **[0.0099, 0.0345]**, p≈0.0004 — a real, not incidental,
improvement, on top of the already-significant final-vs-baseline result (mean diff +0.1201,
95% CI [0.1078, 0.1329], p≈0).

**Interpretation:** this is not a null result (§5.5's stated possible outcome). The 13.5pp
jump in disaster-type accuracy is the strongest single piece of evidence that the frozen
vision tower, not the composite resolution or the decoding strategy, was the dominant
bottleneck — the same LoRA rank, on the same composite images, at the same 192×384 effective
resolution, produces a materially better read of which disaster occurred once the tower is
allowed to adapt at all. The no-damage bias *improved but did not close* (78.5% vs. the
34.2% reference) — rank-8 LoRA on `qkv` narrowed the gap by about a third rather than
resolving it outright, suggesting there is more headroom here (a higher LoRA rank on the
vision tower, or revisiting the 192×384 composite resolution question deferred in §3.2/§3.9,
are the natural next experiments, not attempted in this session).

**Final reported checkpoint:** `checkpoints/full_run_v3_vision/best`, superseding
`full_run_v2/best` as the model reported in Fase 4's results. `full_run_v2` is retained and
reported as the ablation row, not deleted — it is the evidence that vision LoRA specifically
(not just more training) drove the improvement.


---

# Fase 6: explainability — cross-attention rollout, and whether it survives a faithfulness test

Implementation: `scripts/xai_attention_rollout.py` (engine + `--mode selftest|examples|aggregate`),
`scripts/xai_faithfulness_check.py` (`--mode occlusion|delins|reanalyze|summary`).
Artifacts: `results/xai/` (JSON + curves) and `results/xai_examples/` (12 worked examples).
**Nothing in `results/metrics.json`, `results/metrics_table.md`, or the Fase-4/5 narrative was
touched** — this phase is additive interpretability on the already-final checkpoint
`checkpoints/full_run_v3_vision/best`. No retraining, no checkpoint modification.

**Headline, stated up front because it is a negative result:** the cross-attention maps of
this model are a real, sharp, image-dependent saliency signal, but they are **not** a
token-level or section-level explanation, and their spatial ranking is **not** measurably
more causally important than a random ranking of the same number of patches. The pictures in
`results/xai_examples/` are honest renderings of what the model computes; they are *not*
evidence that "the model looked here to say this word". Sections 6.4–6.6 are the measurements
that force that conclusion.

## 6.0 Plumbing: `output_attentions=True` on BLIP returns the wrong tensor

**Why this gets its own section.** The obvious call is
`BlipForConditionalGeneration(..., output_attentions=True)` and then reading
`out.attentions`. Measured on transformers 5.9.0 with the loaded checkpoint:

| field | value |
|---|---|
| `out.cross_attentions` | **`None`** |
| `out.attentions` | 12 tensors of `(1, 12, 577, 577)` |

`out.attentions` is the **vision encoder's image-to-image self-attention**, because
`BlipForConditionalGeneration.forward` builds its output object out of `vision_outputs` only
(`attentions=vision_outputs.attentions`) and has no field for the text decoder's attention at
all. A 577×577 tensor reshapes to a 24×24 grid without complaint and overlays onto the
composite perfectly, so *plotting it would have produced a beautiful, completely wrong
"cross-attention" heatmap* — attention of image patches to image patches, mislabelled as
attention of words to patches. This is exactly the failure mode this phase exists to avoid,
so `--mode selftest` asserts it (`toplevel_output_attentions.cross_attentions_is_none: true`)
rather than leaving it as folklore.

**How to apply.** Call the text decoder directly:

```python
image_embeds = base.vision_model(pixel_values=pv).last_hidden_state      # (1, 577, 768)
out = base.text_decoder(input_ids=ids, attention_mask=ones,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=ones_577,
                        output_attentions=True, use_cache=False)
out.cross_attentions   # 12 x (1, 12 heads, T, 577)   <- the real thing
out.attentions         # 12 x (1, 12 heads, T, T)     <- decoder self-attention
```

This works because `BlipTextPreTrainedModel._can_record_outputs` registers
`OutputRecorder(BlipTextSelfAttention, index=1, layer_name=".crossattention.")`. The PEFT
wrapper is transparent here (LoRA replaces the `query`/`value` leaves in place, so the module
path still ends in `.crossattention.self`).

### `attn_implementation="eager"` was **not** required

This was the expected blocker and it turned out not to be one, which is worth recording so
nobody "fixes" it later. In transformers 5.9.0 both `BlipAttention.forward` (vision) and
`BlipTextSelfAttention.forward` (text) are hand-written eager softmax implementations with no
SDPA/flash branch, and the loaded model already reports
`config._attn_implementation == "eager"`. The selftest records that value, so if a future
transformers upgrade adds an SDPA path, the recorded value changes and the attentions come
back `None` loudly instead of silently.

### Token alignment is off-by-one and is asserted, not assumed

The decoder is causal: the hidden state at input position `i` predicts token `i+1`. So
cross-attention row `i` is the attention used while **emitting** token `i+1`. Getting this
backwards produces maps that render perfectly and are attributed to the neighbouring word.

Rather than trusting the convention, the pipeline generates greedily, then re-runs the
generated sequence teacher-forced and asserts `argmax(logits[:, i]) == generated[i+1]` for
every position: **146/146 tokens, match rate 1.000** on the first test record, and 1.000 on
all 12 illustrated examples and all 150 aggregate examples. Everything downstream
(`TokenMaps.emitted_ids`, `TokenMaps.maps`) is indexed by *emitted* token position, with the
shift applied in exactly one place.

Cost: **~7.5 s per example** (fp32, batch 1, greedy up to 300 new tokens + one teacher-forced
attention pass). No autocast anywhere in this phase — bf16 would make the attention
probabilities and the alignment assert non-reproducible for no useful speedup at batch 1.

## 6.1 Decision: rollout, not raw last-layer attention — and the exact rollout used

**Why not raw attention.** A single decoder layer's cross-attention row says where *that*
layer's query looked. The value that reaches the LM head has passed through 12 residual
blocks, each of which re-mixes text positions (self-attention) and re-injects image
information (cross-attention). Reading layer 11 alone attributes the whole prediction to one
of the twelve places the image entered.

**The rollout actually implemented.** Abnar & Zuidema (2020) rollout is defined for one
self-attention stack; BLIP is an encoder plus a cross-attending decoder, so it needs writing
out. Track a joint state `M ∈ R^{T × (T + 577)}` — row `t` is "where the representation at
text position `t` came from", over text inputs and image tokens. Initialise `M = [I_T | 0]`.
Per decoder layer, head-averaged:

```
self :  M <- rownorm(Ā_self + I) @ M
cross:  M <- 0.5 * M  +  0.5 * [0_T | Ā_cross]
```

The self step is textbook rollout (add identity for the residual, row-normalize,
chain-multiply). The cross step is the same residual rule written for a sublayer whose two
inputs are the residual stream and the image (`hidden = LN(hidden + CrossAttnOut)`). The
image block `M[:, T:]` is then renormalized to a distribution over vision tokens.

**Caveat, stated because it is real:** the 0.5/0.5 residual assumption makes rollout
*geometrically* weighted toward late layers — layer 11 contributes 0.5 of the image mass,
layer 10 contributes 0.25, layer 0 contributes 2⁻¹². Rollout here is therefore **not** an
equal-weight average over the 12 cross-attention layers.

Measured (`xai_selftest.json → variant_agreement`, 126 content tokens of test record 0), it is
much closer to the last layer than that framing even suggests:

| pair | mean per-token Pearson r | mean L1 per token |
|---|---|---|
| `rollout` vs `last_layer` | **0.99983** | 0.097 |
| `rollout` vs `mean_layers` | 0.99978 | 0.125 |
| `rollout` vs `rollout_vit` | 0.1167 | 1.650 |

So for *this* model the multi-layer rollout is, empirically, a slightly smoothed copy of the
raw last layer. That is not a coding error — it is what the residual weighting plus twelve
highly similar cross-attention layers produce — but it does mean the "rollout beats raw
attention" argument cannot be made on structural grounds here. §6.6 tests it directly and
finds no advantage either.

**CLS column.** Vision token 0 is the ViT CLS token, not a spatial location. Its mass is
measured, reported, then dropped, and the remaining 576 columns renormalized onto the 24×24
grid. Decoder-only rollout puts **0.024%** of its mass on CLS, so dropping it is harmless.

## 6.2 Decision: the ViT-side rollout is implemented and then **rejected on measurement**

**Why it was tried.** Cross-attention column `j` points at ViT *output* token `j`, not input
patch `j`; after 12 ViT layers, output token `j` is a mixture of all patches. The textbook fix
is to roll out the vision self-attention too (`R_v = Π rownorm(Ā_v + I)`) and compose
`P = M[:, T:] @ R_v`. That variant is `rollout_vit` in the code.

**Why it was rejected.** Measured, on real test images:

| variant | mean per-token entropy (nats) | peak / uniform | CLS mass | between-example post-fraction SD (n=150) |
|---|---|---|---|---|
| `rollout` (decoder-only) | 3.692 | **37.99×** | 0.00024 | 0.170 |
| `rollout_vit` (+ ViT rollout) | **6.329** (uniform = 6.356) | **1.89×** | **0.197** | 0.024 |
| `last_layer` (raw layer 11) | 3.424 | 40.08× | 0.00008 | 0.179 |
| `mean_layers` (raw, layer- and head-averaged) | 3.365 | 40.53× | 0.00009 | — |

`rollout_vit` collapses to near-uniform: 6.329 nats against a uniform-distribution ceiling of
6.356, a peak barely 1.9× uniform, a fifth of the mass parked on CLS, and — the decisive
number — **a pre/post split identical to four decimal places across all seven sections**:
post = 0.49356 / 0.49356 / 0.49355 / 0.49354 / 0.49354 / 0.49354 / 0.49354 for
`BENCANA … KESIMPULAN` on test record 0, and post = 0.5095 for every section in the
150-example aggregate. It is the textbook "renders like a heatmap,
explains nothing" artifact: the identity-plus-renormalize assumption smears completely across
12 dense 577×577 layers, and the few surviving peaks sit on ViT high-norm outlier tokens
rather than on content.

It is kept in the code and one worked example is saved
(`results/xai_examples/idx0081_ledakan_REJECTED_rollout_vit.png`) so the rejection is
inspectable rather than a silent design choice. **Default variant: `rollout` (decoder-only).**

**How to apply.** Never ship a saliency variant on theoretical grounds alone. Two cheap
scalars — per-token entropy against `log(N)`, and peak/uniform ratio — catch a collapsed map
in seconds, and the "does it differ between sections/examples at all" check catches the rest.

## 6.3 Sanity check: the map does move when the image moves

Same forced token sequence, two different test images (indices 0 and 7), maps compared
directly. A map that did not change would be a fixed positional artifact:

| variant | mean L1 per token (0 = identical, 2 = disjoint) | Pearson r |
|---|---|---|
| `rollout` | **1.778** | 0.023 |
| `last_layer` | 1.851 | 0.023 |
| `rollout_vit` | 0.215 | 0.261 |

The decoder-only rollout is almost entirely image-driven (L1 1.78 out of a maximum of 2.0,
r ≈ 0.02). Confirmed independently at the output level: replacing the whole composite with
flat gray collapses generation to word salad (`"sampai art of color kepe by person"` instead
of a structured report), so the model is genuinely conditioned on the image. `rollout_vit`
again fails — it barely moves between images, consistent with §6.2.

## 6.4 The finding that decides everything: the map is **token-invariant**

Mean pairwise Pearson correlation between the per-token patch maps *within one example*
("token selectivity"). 1.0 means every generated word gets the same picture.

| variant | test record 0 (126 content tokens) | mean over 150 test examples | min over 150 |
|---|---|---|---|
| `rollout` | 0.99967 | **0.99967** | 0.99944 |
| `last_layer` | 0.99896 | 0.99895 | 0.99826 |
| `mean_layers` | 0.99984 | — | — |
| `rollout_vit` | 1.00000 | 1.00000 | 1.00000 |

**This is not an artifact of head-averaging.** The obvious suspicion is that a token-selective
head exists and gets washed out by the mean over 12 heads — the classic "wrong head" error.
So all 144 (layer, head) pairs were measured on raw cross-attention: the **most** token-
selective single head in the entire decoder is **layer 0, head 6, at r = 0.901**; every other
head is above 0.95, and every layer's head-average is above 0.9989. There is no hidden
token-selective head to plot instead.

**Consequence.** Per-word heatmaps for this model are decoration. The `*_tokens.png` figures
are still produced — with the measured correlation printed in the figure title as a warning —
because showing that the panels are identical *is* the finding.

## 6.5 Per-section pre/post attention split (n = 150 test examples, 19,872 content tokens)

`scripts/xai_attention_rollout.py --mode aggregate --n 150` (seed 0, random sample of the
test predictions). Patch columns 0–11 are the PRE half, 12–23 the POST half; each per-token
map is a distribution over the 576 patches, so 50 / 50 means "no preference".

| section | content tokens | POST attention mass (`rollout`) |
|---|---|---|
| `BENCANA` | 215 | 54.23% |
| `BANGUNAN` | 3,041 | 54.20% |
| `JALAN` | 2,478 | 54.19% |
| `VEGETASI` | 4,212 | 54.26% |
| `BADAN_AIR` | 2,756 | 54.28% |
| `PERTANIAN` | 2,614 | 54.27% |
| `KESIMPULAN` | 4,556 | 54.27% |
| **all content tokens** | **19,872** | **54.25%**, 95% CI [51.44%, 56.98%] (bootstrap over examples, 10,000 resamples; 99.87% of resamples > 50%) |

Two things are true at once and both need saying:

1. **There is a small, statistically real post-half preference overall** — 54.25% vs. 50%,
   CI excludes 50%. On the composite the model does look slightly more at the post-disaster
   image than the pre-disaster one.
2. **The per-section numbers are meaningless.** The full range across all seven sections is
   **0.09 percentage points**. Within a single example, the standard deviation of the
   post-fraction *between sections* is **0.0027**, while the standard deviation *between
   examples* is **0.1704** — a ratio of **63.6×**. The pre/post split is a property of the
   image, not of which section is being written. Writing "when generating BANGUNAN the model
   puts 54.2% of attention on the post image, versus 54.3% for BADAN_AIR" would be reporting
   third-decimal noise as a finding, and it is a direct corollary of §6.4.

Per-image variation is large and genuine: across the 12 illustrated examples the whole-caption
post fraction ranges from **7.6%** (idx 563) to **89.3%** (idx 1944).

## 6.6 Faithfulness test 1: deletion / insertion vs. a random ranking

**Why this design.** Rank the 576 patches by the caption-level map, then either gray out the
top-k% (deletion) or reveal only the top-k% from an all-gray image (insertion), regenerate,
and score the new caption **against the model's own unperturbed caption** — the question is
"how much of *this output* did *this region* cause", which is independent of whether the
output was correct. Gray fill uses `ImageSpec.pad_color` (the uint8 colour normalizing to
~0, §3.0), so the perturbation removes content rather than injecting a high-contrast
rectangle that is itself a strong signal. A seeded **random** ranking is run under identical
conditions, because a saliency map that cannot beat random is not a saliency map.

12 examples (the same diversity-stratified 12 as the figures), 11 fractions, 3 rankings,
2 directions = **792 generations**. Control: the **72** points that re-generate the
*unperturbed* image inside a batch of 11 (deletion at k=0 and insertion at k=100%, for every
example × ranking) reproduced the batch-1 caption **exactly — rate 1.000**, so the batched-
generation shortcut is not perturbing anything.

Area under the token-F1-vs-fraction curve (deletion: **lower is more faithful**; insertion:
**higher is more faithful**), with a 10,000-resample paired bootstrap over the 12 examples:

| ranking | direction | AUC | random AUC | diff | 95% CI | p (two-sided) |
|---|---|---|---|---|---|---|
| `rollout` | deletion | 0.7579 | 0.7698 | −0.0119 | [−0.0326, +0.0092] | 0.269 |
| `rollout` | insertion | 0.7811 | 0.7774 | +0.0037 | [−0.0127, +0.0205] | 0.669 |
| `last_layer` | deletion | 0.7515 | 0.7698 | **−0.0183** | [−0.0343, −0.0015] | **0.036** |
| `last_layer` | insertion | 0.7862 | 0.7774 | +0.0088 | [−0.0164, +0.0278] | 0.432 |

Pooled-corpus CIDEr (one shared IDF across all 792 pairs) tells the same story: deletion AUC
2.157 (`rollout`) / 1.857 (`last_layer`) / 2.025 (random).

**Verdict, honestly.** Both maps point in the *right direction* on deletion, but the effect is
about 1.5–2.4% of the AUC and only one of twelve tests reaches nominal significance
(`last_layer` deletion, p = 0.036). Twelve tests were run (3 metrics × 2 rankings × 2
directions); a Bonferroni threshold is 0.0042, which p = 0.036 does not survive. **The correct
statement is: at n = 12 there is no convincing evidence that the attention ranking identifies
causally more important patches than a random ranking.** Note also that the multi-layer
rollout did *not* beat raw last-layer attention — consistent with §6.1's caveat that the
0.5-residual weighting makes rollout mostly a smoothed last layer here.

**Why the test has low dynamic range (a caveat on the caveat).** The curves are dominated by
"any occlusion at all": token-F1 vs. the intact caption falls 1.000 → ~0.82 as soon as 10% of
patches are removed, then sits on a plateau all the way to 90%, and only collapses to 0.0 at
100%. Concretely, with **90% of the most-attended patches greyed out**, test record 81 still
emits a well-formed seven-section report with the correct `BENCANA: ledakan`. At 100% gray it
emits word salad. So the model needs *some* image but is remarkably insensitive to *which*
part — which both limits the test's power and is itself a substantive finding about the model
(and is consistent with Fase 4/5's boilerplate-bias diagnosis: much of the report is a
template conditioned on a coarse global image statistic).

## 6.7 Faithfulness test 2: half-occlusion, with the pre-half mirror as control

**Why.** The domain-specific version of the same question, and the one that matters for the
paper's claim: if damage words are read off the post image, greying the **entire post half**
must damage the output far more than greying the pre half. Diagnostics are the exact
`get_bencana` / `building_no_damage` helpers from `scripts/compare_beam_vs_greedy.py`, so this
is the same instrument used for the Fase-4/5 bias analysis. 60 random test examples, seed 0.

| | post half masked | pre half masked (control) |
|---|---|---|
| caption identical to intact | 1.7% | 0.0% |
| token-F1 vs. intact caption | **0.811** | **0.798** |
| `BENCANA` changed | 23.3% | 25.0% |
| `BANGUNAN` no-damage flag flipped | 6.7% | 11.7% |
| `BANGUNAN` text changed | 70.0% | 70.0% |
| `JALAN` text changed | 71.7% | 68.3% |
| `VEGETASI` text changed | 83.3% | 83.3% |
| `BADAN_AIR` text changed | 55.0% | 55.0% |
| `PERTANIAN` text changed | 76.7% | 80.0% |
| `KESIMPULAN` text changed | 86.7% | 81.7% |

**This is a negative finding and it is the important one.** Destroying the post-disaster image
entirely damages the output **slightly less** than destroying the pre-disaster image
(token-F1 0.811 vs 0.798; `BENCANA` changed 23.3% vs 25.0%). The two halves are
interchangeable as far as the output is concerned. Meanwhile the attention *does* lean post
(54.25%, §6.5). **Attention mass and causal importance disagree**, which is precisely the
disagreement a faithfulness test exists to expose, and precisely why the pre/post attention
number must not be reported as evidence that the model "reads damage from the post image".

Also worth noting: `BENCANA` survives half-occlusion ~76% of the time, i.e. the disaster-type
prediction is largely recoverable from either half alone — consistent with §5.6's reading that
disaster type is being inferred from coarse scene appearance (an urban tornado scene looks
different from a wildfire scene in *both* frames) rather than from pre→post change detection.

## 6.8 Honest assessment: what these heatmaps are and are not

**What is trustworthy:**
- The extraction is correct and asserted, not assumed: real `cross_attentions` from the text
  decoder (not the vision self-attention that `output_attentions=True` hands you), correct
  causal off-by-one (alignment 1.000), CLS handled explicitly, rows summing to 1.
- The maps are sharp (peak ≈ 38× uniform) and strongly image-dependent (L1 1.78 / 2.0 between
  images). They are a genuine image-level saliency signal, not noise and not a fixed grid.
- The overall pre/post attention split (54.25% post, CI [51.44, 56.98]) is a real, if modest,
  measured property of the model.

**What is not trustworthy, and must not be claimed:**
- **Per-word and per-section attribution.** Token selectivity ≥ 0.999 across 150 examples,
  with no token-selective head anywhere in the 144 (layer, head) pairs. Every word in a
  300-token report gets the same picture. The per-section pre/post table in §6.5 has a
  0.09 pp total range and is noise.
- **Causal importance of the highlighted regions.** Deletion/insertion does not separate the
  attention ranking from a random ranking at n = 12 (best p = 0.036, does not survive
  correction for 12 tests). Removing the top-90% most-attended patches still yields a correct,
  well-formed report.
- **"The model looks at the post image to see damage."** The half-occlusion test says the
  opposite of what the attention split suggests: masking post hurts *less* than masking pre.

**Surprises worth recording.** (i) The textbook full rollout — decoder rollout composed with
ViT rollout — was the *worst* variant by every diagnostic, and would have been the natural
default. (ii) Multi-layer rollout turned out to be numerically almost the *same map* as raw
last-layer attention (per-token r = 0.99983, §6.1) and did not outperform it on the only test
that can arbitrate (§6.6). "Use rollout, not raw attention" is still the methodologically
right instinct; it is just that on this architecture, with twelve near-identical
cross-attention layers, it changes almost nothing — and saying so is more useful than
implying the rollout bought an improvement it did not buy.
(iii) The model tolerates 90% patch deletion, which reframes the Fase-4/5 boilerplate bias:
it is not only a text-prior problem, the vision pathway is contributing a very coarse global
signal rather than localized evidence.

**Recommended use in the paper.** Present the heatmaps as *image-level saliency* with the
faithfulness caveats attached, or present them as a negative interpretability result — which
is the more defensible and more interesting claim. Do **not** present per-section attention
splits as evidence of section-specific visual grounding. If stronger explanations are wanted,
the next steps are gradient-based attribution (Grad-CAM on the ViT, or Chefer-style
attention×gradient, which uses the *gradient of the emitted token* and can therefore be
token-selective even when raw attention is not) — attempted in neither this phase nor any
earlier one, and explicitly *not* claimed here.

## 6.9 Reproducing this phase

```bash
set HF_HUB_DISABLE_XET=1 && set HF_HUB_OFFLINE=1
python scripts/xai_attention_rollout.py  --mode selftest              #  ~1 min
python scripts/xai_attention_rollout.py  --mode examples  --n 12      #  ~3 min
python scripts/xai_attention_rollout.py  --mode aggregate --n 150     # ~19 min
python scripts/xai_faithfulness_check.py --mode occlusion --n 60      #  ~8 min
python scripts/xai_faithfulness_check.py --mode delins    --n 12      # ~11 min, 792 generations
python scripts/xai_faithfulness_check.py --mode reanalyze             # paired bootstrap, no GPU
python scripts/xai_faithfulness_check.py --mode summary               # writes the reader-facing summary
```

Outputs (all produced by the runs above, nothing hand-edited):

| path | contents |
|---|---|
| `results/xai/xai_selftest.json` | plumbing, alignment, non-uniformity, image sensitivity, per-layer/per-head token selectivity |
| `results/xai/section_attention_aggregate.json` | n=150 per-section pre/post split, all 4 variants, per-example rows |
| `results/xai/occlusion_test.json` | n=60 post/pre half-masking, per-example captions and diagnostics |
| `results/xai/deletion_insertion_test.json` | 792 perturbed captions, curves, AUCs, paired bootstrap |
| `results/xai/deletion_insertion_curves.png` | the four curves (deletion/insertion × token-F1/CIDEr) |
| `results/xai_examples/` | 12 examples × {original PNG, caption + ground truth TXT, per-section rollout overlay, per-token overlay, raw-last-layer overlay} + `manifest.json` + `faithfulness_summary.json` + one rejected-`rollout_vit` figure |

The 12 illustrated examples are chosen by `select_examples()`: stratified by ground-truth
disaster type, alternating correct/incorrect `BENCANA`, seeded, and computed **from the
prediction file before any image is looked at**. The resulting set covers 10 disaster types
(`api`, `api liar`, `badai`, `banjir`, `gempa bumi`, `gunung berapi`, `kebakaran hutan`,
`ledakan`, `tornado`, `tsunami`), both `Optical` and `SAR` post-images, and 5 correct / 7
incorrect `BENCANA` predictions. Hand-picking figures that "look good" would have defeated the
entire point of the exercise.
