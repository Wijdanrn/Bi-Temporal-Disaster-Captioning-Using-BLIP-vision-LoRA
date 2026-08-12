# Fase 1 -- EN->ID Translation Report

**Pipeline:** `facebook/nllb-200-distilled-600M`, fp16, on local GPU (RTX 5050 Laptop, 8.5GB VRAM), `torch5050` conda env.

## Design decisions

1. **Section-aware translation, not whole-blob.** Per project decision (confirmed 2026-08-08), ground truth is translated as-is preserving the 7-section structure (`DISASTER/BUILDING/ROAD/VEGETATION/WATER_BODY/AGRICULTURE/CONCLUSION` -> `BENCANA/BANGUNAN/JALAN/VEGETASI/BADAN_AIR/PERTANIAN/KESIMPULAN`). Each section is translated independently and reassembled, rather than translating the full ~150-300 word blob in one call.
2. **Terminology fix pass.** One known NLLB false-friend found in manual QC: "foundation" (building) -> "yayasan" (nonprofit-org sense in Indonesian) instead of "fondasi". Applied as a regex post-processing fix (`scripts/translate_captions.py::TERMINOLOGY_FIXES`).
3. **Length-bucketed batching.** Real section text varies wildly in length (single-word DISASTER fields to multi-sentence CONCLUSION paragraphs). Naive fixed-order batching mixed short and long texts in the same batch, forcing padding for the whole batch up to the longest item -- this drove VRAM usage to within ~350MB of the 8.5GB limit and collapsed throughput from ~13/s to ~1/s (near-OOM thrashing, 100% GPU util but ~20W power draw = memory churn, not compute). Fix: sort all texts by length before batching, restore original order after. Result: stable throughput, no thrashing, ~14x faster overall (fp16 + batch_size=128 + length-bucketing vs the naive first attempt).

## Issue found and fixed: CONCLUSION section content loss

Back-translation QC (ID->EN, compared against original EN) surfaced a systematic issue specific to the CONCLUSION section: beam search on multi-sentence inputs occasionally dropped the trailing sentence entirely. Confirmed **not** a `max_new_tokens` truncation artifact (generated length maxed out at 74 tokens against a 200 cap) -- an inherent weakness of the distilled 600M model on longer multi-sentence inputs.

**Fix:** CONCLUSION is now split into individual sentences before translation (same principle as the top-level 7-section split, one level deeper), each translated independently, then rejoined. Applied via `scripts/fix_conclusion.py`.

**Before -> after (train split, back-translation length_ratio = len(back-translated EN) / len(original EN), 1.0 = no content lost):**

| Metric | Before fix | After fix |
|---|---|---|
| CONCLUSION flagged (length_ratio <0.7 or >1.4) | 34.3% | 22.8% |
| CONCLUSION mean length_ratio | 0.82 | 0.88 |
| Records with CONCLUSION text changed by the fix | -- | 5,787 / 7,766 (74.5%) |

Residual ~23% flagged rate is now closer to (though still somewhat above) the other sections' baseline (5-15%), which is treated as normal round-trip paraphrase noise rather than a defect -- see next section for why.

## A note on the back-translation QC metric itself

Two signals were computed per section: `similarity` (difflib SequenceMatcher ratio on the character level) and `length_ratio` (word-count ratio). Manual inspection of "flagged" records revealed `similarity` produces a large fraction of **false positives**: round-trip translation naturally reorders and substitutes synonyms (e.g. "explosion"<->"blast", "primarily concentrated near the central area along the main road"<->"mainly concentrated near central areas along major roads") -- semantically perfect, but scores as low as 0.12 on character similarity. `length_ratio` is the reliable signal for actual content loss (a dropped sentence shows up as a real, large length deficit).

Consequence: the raw "84.2% needs_review" figure in `train_qc.jsonl` (driven partly by `similarity`) **overstates** the real issue rate. A more honest number, using `length_ratio` alone and excluding the inherently-noisy 1-2 word `DISASTER` field: **~49% of records have at least one section with length_ratio outside [0.7, 1.4]**. This is still a union-across-6-sections effect (if each section independently has a ~10-15% divergence rate, the chance that *none* of 6 sections diverges is already below 50%) rather than a claim that half the dataset is mistranslated -- confirmed by manual spot-checks showing several "flagged" cases are fine, just more heavily paraphrased than the strict metric tolerates.

**Practical takeaway:** `length_ratio` and `similarity` are used to *prioritize* the manual review sample (below), not as an automatic accept/reject filter. No records were dropped from the dataset based on these scores alone.

## Manual review sample

Exported for native-speaker review, stratified half random / half worst-scoring (by `min_similarity`) so both typical and worst-case quality are represented:
- `data/interim/train_review_sample.csv` -- 40 rows (20 random + 20 worst-case)
- `data/interim/test_review_sample.csv` -- 20 rows (10 random + 10 worst-case)

Columns include `reviewer_ok_Y_N` and `reviewer_notes` for manual fill-in. **This review has not yet been done by a human as of this report** -- pending before treating the translated captions as final for training.

## Final dataset

Grouped split by `pre_image_path` (not per-record) with fixed seed=42, val_fraction=0.1 of groups -- several records share a `pre_image_path` with different `post_image_type` (Optical/SAR) and identical ground truth, so a naive per-record split could leak near-duplicates across train/val.

| Split | Records | Groups (unique pre-images) |
|---|---|---|
| `data/processed/captions_train.jsonl` | 6,999 | 6,377 |
| `data/processed/captions_val.jsonl` | 767 | 708 |
| `data/processed/captions_test.jsonl` | 2,363 | (unchanged from Fase 0 bench split, 0 overlap with train) |

Each record: `pre_image_path, post_image_path, post_image_type, prompt_id, ground_truth_id, prompt_en, ground_truth_en` (EN fields kept for debugging/reference, not meant to be training inputs).

## Scripts (in `scripts/`)

- `translate_captions.py` -- main EN->ID translation (section-aware, length-bucketed, fp16)
- `quality_check.py` -- back-translation QC (ID->EN, similarity + length_ratio per section)
- `fix_conclusion.py` -- targeted CONCLUSION re-translation (sentence-split)
- `export_review_sample.py` -- stratified CSV export for manual review
- `build_final_dataset.py` -- grouped train/val split + final processed dataset

## Known limitations / open items

1. Manual native-speaker review of the sample CSVs has not been done yet.
2. `similarity` metric as implemented is not a reliable semantic-equivalence check (see above) -- if tighter QC is needed later, a semantic metric (e.g. embedding cosine similarity) would be less prone to the paraphrase false-positive problem.
3. Environment note: `HF_HUB_DISABLE_XET=1` is required when downloading HF models/datasets in this project -- Xet Storage's per-file token-refresh calls have no retry/backoff and stall badly on this network (confirmed both for the DisasterM3 dataset download earlier and the NLLB model download in Fase 1, ~58 min for a 2.4GB file with Xet vs. normal speed with it disabled).
