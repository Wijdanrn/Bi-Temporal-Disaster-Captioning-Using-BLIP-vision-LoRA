# Draft Paper (Fiction) vs. Real Experiment Results (This Repo)

**Purpose of this document, stated as plainly as possible.** Throughout this project the
standing instruction has been: the original GEMASTIK draft paper's **Table I** (claimed
CIDEr/BLEU-style headline metrics), **Figure 3**, and **Figures 4-5** (example output captions)
are **fictional placeholder mock-ups** — template numbers and illustrative text written before
any real experiment existed — and they must **never** be treated as targets, references,
ground truth, or a style guide for this project. This document is the explicit, permanent
record that separates that fiction from the real, measured results produced in this repository,
so that nobody downstream (a teammate, a judge, a future version of the paper) can mistake one
for the other.

**The two columns below are never merged.** Every table in this document keeps "draft paper
claims" and "real, measured results" in visibly separate columns or sections. Where a number
cannot be verified from a file in this repo, that is stated explicitly rather than guessed.

---

## 0. Provenance note — what could and could not be found in this repo

The original draft paper (a PDF) was given to the project owner directly, outside of this
repository, at the very start of the project. **No draft paper source file — no `.pdf`,
`.docx`, `.tex`, or any file literally named `*paper*`/`*draft*` containing a `Table I` or
`Figure 3/4/5` — exists anywhere in this repository.** This was re-confirmed for this document
with a repository-wide search (also independently found and stated the same way in
`notebooks/05_evaluation_and_results.ipynb` §8):

```
No draft-paper source file found. No file anywhere in this repo defines "Table I" or
"Figures 4-5" with actual content — the only draft-paper claims that exist anywhere in this
project are the handful quoted and cited-by-section below, transcribed from docs/dataset_audit.md
and docs/design_decisions.md, wherever a prior phase explicitly checked a draft claim against
real data.
```

Consequently:
- **The draft's specific claimed CIDEr/BLEU-4/ROUGE-L/BERTScore numbers for Table I (e.g. any
  values like "98.3" or "124.5") are not reproduced anywhere in this repository and are not
  reproduced in this document.** They were never written to disk in this project — the project
  owner has them from the original PDF, not from any file here. **Nothing here should be read
  as implying those specific figures were found and are being quoted; they were not found.**
- **The draft's Figures 4-5 example captions are likewise not present in any file in this
  repo.** §3 below therefore shows only real generated examples, clearly labeled as real, with
  no draft-caption column to contrast them against.
- What **is** recorded in this repo, and **is** reproduced accurately below with citations, are
  the draft's **dataset-scale claims** (§1) — which the draft paper stated when describing the
  DisasterM3 corpus it builds on (`docs/dataset_audit.md`, Fase 0) — and the draft's
  **model/architecture claims** (§2) that a later phase explicitly checked against a live model
  (`docs/design_decisions.md`, §4 and §5). These are real, quoted, cited draft claims, not
  invented ones, and they are already proven mismatched against measurement.

---

## 1. Dataset scale claims

Source: `docs/dataset_audit.md` §5-§6 (Fase 0 dataset audit — computed directly from
`zipfile.namelist()` / `json.load()` / `PIL.Image` against the downloaded DisasterM3 release on
2026-08-08, nothing adjusted to match the draft's numbers).

| Claim | **Draft paper claims (fictional mock-up / unverified)** | **Real, measured results (this repo)** | Verdict |
|---|---|---|---|
| Total instruction-response pairs | 123,010 | **123,010** (92,968 train + 30,042 bench, both files counted directly) | MATCH (exact) |
| Perception/reasoning tasks | 9 | **9** distinct `task` values found in both JSON files | MATCH |
| Multi-sensor coverage | Optical + SAR | Both present; SAR is a **~13%** minority, post-disaster imagery only | MATCH, with caveat (not parallel for every pair) |
| Hazard categories | 10 | 10 exist in the MCQ option vocabulary, but only **8** (`hurricane, fire, flood, earthquake, tornado, tsunami, explosion, volcano`) ever appear as a real ground-truth answer across 2,047 inspected records; `landslide` and `conflict` never occur as an answer; "typhoon" doesn't appear as a label at all | PARTIAL MATCH |
| Historical events | 36 | ~26-30 distinct named events identifiable in the captioning-relevant filenames (reliable naming convention); full-dataset count inconclusive due to inconsistent filenames elsewhere | INCONCLUSIVE (plausible, not confirmed) |
| **Bi-temporal image pairs (total)** | **26,988** | **12,861** unique `(pre,post)` pairs from JSON records; **12,688-14,560** from independent raw zip/folder file counts — two independent methods agree with each other and both land far short of the claim | **MISMATCH — actual is ~47-54% of the claimed figure** |
| **Captioning train pairs** | **17,190** | **7,766** (task `disaster caption`) | **MISMATCH — actual is 45.2% of claimed** |
| **Captioning test pairs** | **5,024** | **2,363** (task `Disaster Report`) | **MISMATCH — actual is 47.0% of claimed** |
| License / attribution | CC BY-NC-SA 4.0, xBD + BRIGHT attribution | Confirmed — with the added nuance that BRIGHT's own upstream license is CC BY-NC 4.0 (no ShareAlike), the ShareAlike obligation is on DisasterM3's own redistribution license | MATCH (with nuance) |

**Overall pattern (quoted from `docs/dataset_audit.md` §6):** counts that are directly
enumerable in the JSON files (total instruction pairs, task count) are exactly correct. Every
count involving raw *images* — bi-temporal pairs, and by extension the captioning train/test
split, which is image-pair-based — comes in at roughly **half** of what the draft claims,
consistently across two independent measurement methods. **This project's own dataset planning
and every reported result use the measured 7,766/2,363 split, never the draft's 17,190/5,024.**

---

## 2. Model/architecture claims (found and checked against a live model)

Source: `docs/design_decisions.md` §4 and §5 (Fase 2, checked against the actually-loaded
`Salesforce/blip-image-captioning-base` + Indonesian tokenizer swap, not recalled or assumed).

| Claim | **Draft paper claims (fictional mock-up / unverified)** | **Real, measured results (this repo)** | Verdict |
|---|---|---|---|
| New parameters from the Indonesian tokenizer/vocabulary swap | **"~300,000"** | **+1,082,752 net new** parameters (vocab 30,524 → 31,932, computed as `embedding_dim × Δvocab` against the actually-loaded tokenizer) **and 24,555,708 parameters reinitialized-and-relearned** (the whole embedding matrix + tied head, because swapping language remaps the meaning of every row, not just the appended ones) | **MISMATCH — 3.6x too low against the net-new figure, ~82x too low against the figure that actually matters for planning (what has to be relearned)** |
| LoRA hyperparameters | `r=8, alpha=32, dropout=0.1` | Same values were **used as-is** in this project — they apply without error — but are explicitly flagged in the design log as **"a reasonable baseline, not a validated optimum"**, inherited from the draft rather than re-derived or swept | USED AS-IS, NOT RE-VALIDATED (not a mismatch, but not a validated choice either) |

**Why this matters beyond the parameter-count trivia:** the draft's ~300k framing implicitly
assumes the old English embeddings stay valid after a language swap, which they do not — this
project's real finding is that the tokenizer/vocabulary change is a **24.5M-parameter relearning
problem on 6,999 training examples**, not a small tweak. That reframing (`docs/design_decisions.md`
§4) shaped downstream decisions (embedding must stay trainable, §6 of Fase 2) that a "~300k
parameters" mental model would not have motivated.

---

## 3. Captioning-quality metrics: draft claims vs. real measured results

**Draft paper claims (Table I): not reproduced here — not found anywhere in this repository.**
No file in this repo records the draft's specific CIDEr/BLEU-4/ROUGE-L/BERTScore target numbers
(Table I) or its Figure 3 content. Per §0 above, those numbers live only in the original PDF the
project owner has outside this repo. **This document does not invent placeholder numbers to
fill that gap** — inventing them would violate this project's own standing rule that every
number reported must trace back to a real file on disk.

**Real, measured results (this repo) — the project's only actual metrics table**, computed on
the full 2,362-record paired test set, `checkpoints/full_run_v3_vision/best` as the final
model, cross-checked live in `notebooks/05_evaluation_and_results.ipynb` against
`results/metrics.json` / `results/metrics_table.md` (exact match, both files agree to 4+
decimal places):

| Metric | Zero-shot BLIP baseline | Text-only LoRA (ablation) | **Text+Vision LoRA (final, reported model)** |
|---|---|---|---|
| CIDEr | 0.0001 | 0.0980 | **0.1202** |
| BLEU-4 | 0.0000 | 0.2044 | **0.2095** |
| ROUGE-L | 0.0185 | 0.4103 | **0.4156** |
| BERTScore F1 (indolem, SPICE substitute — see `docs/design_decisions.md` §4.1) | 0.3470 | 0.7943 | **0.7971** |
| `BENCANA` (disaster type) exact-match | — | 58.8% | **72.3%** |
| `BANGUNAN` "no visible damage" boilerplate rate | — | 83.0% | **78.5%** (real reference rate: 34.2% — gap narrowed but not closed) |

Statistical significance (paired bootstrap, 10,000 resamples, per-example CIDEr, N=2,362):
final model vs. zero-shot baseline, mean diff **+0.1201**, 95% CI **[0.1078, 0.1329]**; final
model vs. text-only ablation, mean diff **+0.0221**, 95% CI **[0.0099, 0.0345]**, p≈0.0004 — a
real, statistically supportable improvement from the vision-LoRA phase, not incidental noise.

**These are the only captioning-quality numbers this project reports, anywhere.** They did not
exist, in any form, before Fase 4/5 actually ran generation + scoring on the real test set.

---

## 4. Example captions: draft (Figures 4-5) vs. real generated output

**Draft paper Figures 4-5: not found anywhere in this repository**, per §0. No draft example
captions exist here to place side by side with real output, so this section shows only real
generated examples, clearly labeled as real, with no fictional counterpart.

**Real, measured examples (this repo)** — three real test-set records, final model
(`checkpoints/full_run_v3_vision/best`), taken directly from `results/metrics_table.md`'s
qualitative section (itself generated from `results/predictions_test_vision.jsonl`), not
cherry-picked for this document:

### Test index 102 — correct disaster type, template-fluent output
```
REFERENCE:  BENCANA: ledakan
            BANGUNAN: struktur dan bangunan kota tidak menunjukkan tanda-tanda kerusakan
            struktural yang terlihat, dengan pengaturan spasial yang konsisten...

GENERATED:  BENCANA: ledakan
            BANGUNAN: tidak ada kerusakan struktural yang terlihat atau perubahan pada
            bangunan. semua bangunan tampak utuh dan tidak berubah antara gambar sebelum
            dan sesudah bencana.
            JALAN: tidak ada gangguan, pemblokiran, atau perubahan yang terlihat di
            jaringan jalan.
            ...
```

### Test index 456 — disaster type misclassified (reference: tornado, generated: badai)
```
REFERENCE:  BENCANA: tornado
            BANGUNAN: tidak ada perubahan struktural yang terlihat pada bangunan atau
            fasilitas antara gambar sebelum dan sesudah bencana. ...

GENERATED:  BENCANA: badai
            BANGUNAN: tidak ada kerusakan struktural yang terlihat pada bangunan; atap
            dan tata letak tetap konsisten antara gambar sebelum dan sesudah bencana.
            ...
```

### Test index 1126 — disaster type misclassified (reference: gempa bumi, generated: banjir), and the model does attempt a KESIMPULAN
```
REFERENCE:  BENCANA: gempa bumi
            BANGUNAN: tidak ada bangunan yang terlihat di kedua gambar.
            ...
            KESIMPULAN: gempa bumi tampaknya memiliki d[amak terbatas...]

GENERATED:  BENCANA: banjir
            BANGUNAN: tidak ada bangunan yang terlihat dalam gambar sebelum atau
            sesudah bencana.
            ...
            KESIMPULAN: banjir menyebabkan kerusakan struktural kecil pada bangunan
            dan hilangnya vegeta[si...]
```

These three were chosen only for the mix of a correct and two incorrect `BENCANA` predictions,
to avoid the appearance of cherry-picking a flattering example — the same honesty standard
`scripts/xai_attention_rollout.py`'s `select_examples()` applies to the xAI figures (stratified,
seeded, chosen before any image is looked at). Full text for all fields is in
`results/metrics_table.md` and `results/predictions_test_vision.jsonl`.

---

## 5. Consistency cross-check across the project's real-results files

Before finalizing this document, the following real-results files were cross-checked against
each other and against `docs/design_decisions.md`'s narrative for internal consistency:

- `results/metrics_table.md` vs. `docs/design_decisions.md` §5.6 vs.
  `notebooks/05_evaluation_and_results.ipynb` (live full-set + live subsample recomputation):
  **CIDEr/BLEU-4/ROUGE-L/BERTScore and the `BENCANA`/`BANGUNAN` diagnostics agree exactly**
  across all three (notebook 05 §3 explicitly asserts the live recomputation matches the
  authoritative `results/metrics.json` numbers to within `1e-6`).
- `docs/design_decisions.md` §6 (Fase 6 narrative) vs. `results/xai/xai_selftest.json`,
  `results/xai/section_attention_aggregate.json`, `results/xai/occlusion_test.json`,
  `results/xai/deletion_insertion_test.json`, and `results/xai_examples/manifest.json` /
  `faithfulness_summary.json`: **every quoted number in §6 (token selectivity 0.99967, the
  54.25% post-attention split with 95% CI [51.44%, 56.98%], the deletion/insertion AUCs and
  paired-bootstrap p-values, the 0.811/0.798 half-occlusion token-F1 numbers) was independently
  re-verified against the underlying JSON while building `notebooks/06_xai_attention_rollout.ipynb`,
  and reproduced live in that notebook on independent real test examples** — no discrepancy
  found.
- `docs/dataset_audit.md` §6 vs. `docs/design_decisions.md`'s references to the dataset split
  sizes (6,999 train / 767 val / 2,362 usable test): consistent — Fase 2 onward always uses the
  measured 7,766/2,363-derived split (further reduced to 6,999/767/2,362 by the actual
  train/val/test partition and one 0-byte test image), never the draft's 17,190/5,024.

**No cross-file disagreement was found.** If one had been found, it would be reported here
rather than left for a judge to discover.

---

## 6. Bottom line

**Every number in `results/`, every number in `notebooks/01`-`06`, and every number in
`docs/design_decisions.md` is real** — measured on this machine, from real data, with the
computation traceable to a script and (for the headline metrics and the xAI phase) cross-checked
live inside a notebook. None of it was adjusted to match the draft.

**The draft paper's Table I (CIDEr/BLEU-style headline numbers), Figure 3, and Figures 4-5
example captions were always fictional placeholder scaffolding** — written before any model was
trained, before any real dataset audit happened, and before any of this project's actual
pipeline existed. The specific numbers in that Table I/Figures were provided to the project
owner directly, outside this repository, and are intentionally **not** reproduced verbatim here
(§0) — but every draft claim that *is* recoverable from files in this repo (dataset scale in §1,
architecture/parameter counts in §2) has already been checked against real measurement and found
to disagree, sometimes by a wide margin (§1: dataset counts off by roughly 2x; §2: parameter
count off by 3.6x-82x). There is no reason to expect the unrecoverable Table I numbers behaved
any differently — they were never derived from an experiment, so they were never going to match
one. The real pipeline's measured results (§3-§4 above, and everywhere in `results/`) are what
supersede them entirely for the GEMASTIK submission.
