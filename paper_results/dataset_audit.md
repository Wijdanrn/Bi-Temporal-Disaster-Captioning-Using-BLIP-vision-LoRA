# DisasterM3 Dataset Audit (Fase 0)

**Purpose:** a factual record of the dataset audit already performed for Fase 0 — pure exploration/reporting. All numbers below were computed directly from the files on disk (`zipfile.namelist()`, `json.load()`, `PIL.Image`) on 2026-08-08. Nothing here was adjusted to match the draft paper's claims; mismatches are reported as found.

**Scope inspected:**
- `DisasterM3_Bench/benchmark_release.json` (21.7 MB, extracted, 30,042 records)
- `DisasterM3_Bench/test_images/` (extracted, 6,260 PNGs) and `DisasterM3_Bench/masks/` (extracted)
- `DisasterM3_Instruct/train_release.json` (87.3 MB, extracted, 92,968 records)
- `DisasterM3_Instruct/train_images.zip` (29.7 GB, **not extracted** — inspected via `zipfile`), `box_train_images.zip` (2.96 GB, not extracted), `masks.zip` (172 MB, not extracted)
- `README.md` in project root (this is the original HuggingFace dataset card, present locally)

---

## 1. File / Folder Structure

```
Data Mining/
├── README.md                              # original HF dataset card (license, citation)
├── DisasterM3_Bench/                       # ~11 GB, already extracted
│   ├── benchmark_release.json              # 30,042 instruction records (the "test" split)
│   ├── masks/                              # segmentation masks, extracted PNGs
│   │   ├── flooding_mask/                  (117 files)
│   │   ├── intact_buildings_near_flooding/ (117 files)
│   │   ├── intact_buildings_near_lava/     (37 files)
│   │   ├── volcano_lava/                   (39 files)
│   │   ├── test_building_damaged_mask/     (2,855 files)
│   │   ├── test_building_destroyed_mask/   (2,855 files)
│   │   ├── test_building_intact_mask/      (2,855 files)
│   │   ├── test_road_debris_covered_mask/  (2,072 files)
│   │   ├── test_road_flooded_mask/         (2,072 files)
│   │   └── test_road_intact_mask/          (2,072 files)
│   └── test_images/                        # 6,260 PNGs (2,942 pre / 2,942 optical post / 376 SAR post)
│       └── relational_reasoning_images/    (1,018 files)
└── DisasterM3_Instruct/                    # 32.7 GB total, JSON extracted, images still zipped
    ├── train_release.json                  # 92,968 instruction records (the "train" split)
    ├── train_images.zip                    # 29.7 GB, 20,988 PNGs (9,746 pre / 10,289 optical post / 953 SAR post)
    ├── box_train_images.zip                # 2.96 GB, 1,882 PNGs (bounding-box crops for relational-reasoning task)
    └── masks.zip                           # 172 MB, 50,279 PNGs (segmentation masks, same categories as bench)
```

No separate validation-split file exists — only a train file and a bench (test) file were found anywhere in the download.

---

## 2. Real Example Records (Actual Field Names)

Field names do **not** include a literal `"caption"` field anywhere. The actual schema (varies by task) uses: `pre_image_path`, `post_image_path`, `post_image_type`, `image_path`, `image_type`, `prompts`, `ground_truth`, `ground_truth_option`, `options_list`, `options_str`, `training_answer` (train only), `cls_description`, `task`, `objects`.

**Example 1 — `DisasterM3_Instruct/train_release.json`, task = `"disaster caption"` (this is the captioning task, see §4):**
```json
{
  "pre_image_path": "train_images\\bata_explosion_pre_0.png",
  "post_image_path": "train_images\\bata_explosion_post_0.png",
  "task": "disaster caption",
  "post_image_type": "Optical",
  "ground_truth": "DISASTER: Explosion\nBUILDING: Visible damage to buildings is observed in the post-disaster image, characterized by changes in surface reflectance and missing roof structures primarily near the central area along the main road. ...\nROAD: The road network shows no visible obstruction...\nVEGETATION: Localized vegetation loss is evident...\nWATER_BODY: No water body is visible...\nAGRICULTURE: No agricultural land or patterns are present...\nCONCLUSION: The explosion caused moderate building damage and localized vegetation loss...\n",
  "prompts": ["Please describe a comprehensive damage situation based on pre- and post-disaster images."],
  "training_answer": "<identical to ground_truth for this task>"
}
```

**Example 2 — `DisasterM3_Bench/benchmark_release.json`, task = `"Disaster Bearing Bodies Recognition"` (multiple-choice, NOT captioning):**
```json
{
  "pre_image_path": "test_images\\bata_explosion_pre_0.png",
  "post_image_path": "test_images\\bata_explosion_post_0.png",
  "post_image_type": "Optical",
  "ground_truth": "Buildings, Roads, Forest, Open-space ground",
  "ground_truth_option": "B, E, F, G.",
  "options_list": ["Stadiums","Buildings","Coastline","Dams","Open-space ground","Forest","Roads","Storage tank"],
  "options_str": "A. Stadiums, B. Buildings, C. Coastline, D. Dams, E. Open-space ground, F. Forest, G. Roads, H. Storage tank.",
  "prompts": "What essential land-cover objects appear damaged in this disaster zone?",
  "task": "Disaster Bearing Bodies Recognition"
}
```

**Example 3 — `train_release.json`, task = `"Disaster Type Recognition"` (multiple-choice, the source of the hazard-category vocabulary):**
```json
{
  "pre_image_path": "train_images\\bata_explosion_pre_1.png",
  "post_image_path": "train_images\\bata_explosion_post_1.png",
  "ground_truth": "explosion",
  "ground_truth_option": "E",
  "options_list": ["fire","volcano","landslide","tornado","explosion"],
  "prompts": "What kind of calamity has this area experienced?",
  "training_answer": "This area has been affected by an explosion.",
  "task": "Disaster Type Recognition"
}
```

**Example 4 — `train_release.json`, task = `"relational reasoning"` (uses `image_path` + bounding boxes, not `pre/post_image_path`):**
```json
{
  "image_path": "train_images\\box_train_images\\bata_explosion_post_1052_0.png",
  "objects": {"1": [275,139,457,380], "2": [424,65,88,60]},
  "ground_truth": "The destroyed buildings in the red box are below the red intact building in the blue box.",
  "prompts": "Explain how object in the red box spatially relates to object in the blue box.",
  "task": "relational reasoning"
}
```

**Example 5 — `train_release.json`, task = `"Building Damage Counting"`:**
```json
{
  "pre_image_path": "train_images\\bata_explosion_pre_0.png",
  "post_image_path": "train_images\\bata_explosion_post_0.png",
  "ground_truth": 231,
  "prompts": "What is the total number of undamaged buildings following the disaster?",
  "options_list": [231, 323, 277, 139, 185],
  "task": "Building Damage Counting",
  "cls_description": "Intact building counting.",
  "training_answer": "The total number of undamaged buildings after the disaster is 231."
}
```

---

## 3. Caption/Instruction/Response Language — VERIFIED ENGLISH

**Finding: all instruction and response text is English.** This was checked directly, not assumed:

- Scanned all 282,271 text fields (`ground_truth`, `training_answer`, `prompts`) across both JSON files for common Indonesian stopwords (`yang`, `dan`, `tidak`, `adalah`, `dengan`, `ini`, `itu`, `untuk`, `pada`, `dari`, `akan`): **0 matches**.
- Real example strings (verbatim, from `disaster caption` / `Disaster Report` task):
  - Prompt: `"Please describe a comprehensive damage situation based on pre- and post-disaster images."`
  - Ground truth excerpt: `"DISASTER: Explosion\nBUILDING: Visible damage to buildings is observed in the post-disaster image, characterized by changes in surface reflectance and missing roof structures..."`
- Only 97 of 282,271 text fields (0.03%) contain non-ASCII bytes, and inspecting them shows they are **mojibake/encoding-corruption artifacts** (e.g. `façade` rendered as `faﾃｧa`, curly quotes rendered as `窶`), not Indonesian or any other language — a minor data-quality defect (double-encoded UTF-8), not a translation.

**Implication for the project:** this confirms the paper's implicit premise — raw DisasterM3 captions are English and *do* need machine/human translation into Indonesian before they can be used to fine-tune an Indonesian-captioning BLIP model. The translation pipeline is a real, necessary step, not a paper artifact.

---

## 4. Which Subset Is "Disaster Captioning"

The paper's "disaster captioning" task corresponds to:
- **Train split (`train_release.json`):** `task == "disaster caption"` (lowercase, 7,766 records)
- **Test/bench split (`benchmark_release.json`):** `task == "Disaster Report"` (2,363 records) — **note the task is renamed between the two files**; it is not literally called "caption" in the benchmark file. Confirmed equivalent by identical prompt text (`"Please describe a comprehensive damage situation based on pre- and post-disaster images."`) and identical structured multi-section output format (`DISASTER:` / `BUILDING:` / `ROAD:` / `VEGETATION:` / `WATER_BODY:` / `AGRICULTURE:` / `CONCLUSION:`).

This is the only free-form, paragraph-length generative task in the dataset. All other 8 tasks are multiple-choice classification, counting, or segmentation-style (see §5 for the full list) — confirmed by manually inspecting one example of every task.

**Caveat for modeling:** the "caption" is not a single sentence — it's a ~150-300 word structured damage report with 7 fixed sections. If IndoBLIP-Post-Disaster is meant to produce short captions (typical BLIP output), this ground truth will need reformatting/summarizing, not just translation.

---

## 5. Real Statistics

### 5.1 Overall record counts
| File | Records |
|---|---|
| `train_release.json` | 92,968 |
| `benchmark_release.json` | 30,042 |
| **Total instruction-response pairs** | **123,010** |

### 5.2 Tasks (9 confirmed in each file)
**Train (`train_release.json`):**
| Count | Task |
|---|---|
| 37,204 | Referring Expression Segmentation |
| 14,531 | Building Damage Counting |
| 7,766 | Disaster Bearing Bodies Recognition |
| 7,766 | disaster caption |
| 7,765 | disaster restoration advice |
| 7,337 | Road Damage Counting |
| 7,090 | Disaster Scene Recognition |
| 1,882 | relational reasoning |
| 1,627 | Disaster Type Recognition |

**Bench (`benchmark_release.json`):**
| Count | Task |
|---|---|
| 12,348 | Referring Expression Segmentation |
| 4,982 | Building Damage Counting |
| 2,363 | Disaster Bearing Bodies Recognition |
| 2,363 | Disaster Report (= captioning) |
| 2,363 | Disaster Restoration Advice |
| 2,178 | Road Damage Counting |
| 2,007 | Disaster Scene Recognition |
| 1,018 | Relational Reasoning |
| 420 | Disaster Type Recognition |

Task-name capitalization is inconsistent between the two files (`relational reasoning` vs `Relational Reasoning`, `disaster caption` vs `Disaster Report`) — worth normalizing in any preprocessing script.

### 5.3 Captioning subset specifically
| Split | Count |
|---|---|
| Train (`disaster caption`) | **7,766** |
| Test (`Disaster Report`) | **2,363** |
| **Total** | **10,129** |
| Unique pre-disaster images (train) | 7,085 |
| Unique pre-disaster images (test) | 2,006 |
| `post_image_type` split (train) | 7,085 Optical / 681 SAR |
| `post_image_type` split (test) | 2,006 Optical / 357 SAR |

No overlap: 0 shared `pre_image_path` values between train and bench captioning records (clean split, verified programmatically).

### 5.4 Hazard/disaster-type categories
Two different ways to count this from the data, both computed:

**(a) Vocabulary of options offered in the `Disaster Type Recognition` MCQ task** (union of every `options_list` seen, train+bench): 10 distinct values — `conflict, earthquake, explosion, fire, flood, hurricane, landslide, tornado, tsunami, volcano`.

**(b) Actual `ground_truth` answers in that same task** (i.e. categories that really occur in the data, not just offered as wrong-answer distractors), across all 2,047 Disaster Type Recognition records (train+bench combined):
| Count | Type |
|---|---|
| 794 | hurricane |
| 536 | fire |
| 316 | flood |
| 176 | earthquake |
| 94 | tornado |
| 53 | tsunami |
| 51 | explosion |
| 27 | volcano |

**`landslide` and `conflict` never once appear as an actual ground-truth answer anywhere in the 2,047 Disaster Type Recognition records we inspected** — they only ever show up as multiple-choice distractors. "Typhoon" (mentioned in the paper's prose) does not appear as a label at all; storm events are all labeled "hurricane."

Separately, the free-text `DISASTER:` line inside the captioning ground truth uses 11 raw string variants (case/synonym differences): `hurricane, earthquake, fire, wildfire, flooding, bushfire, tornado, volcano, explosion, flood, tsunami` — collapsing `fire/wildfire/bushfire` and `flood/flooding` gives the same ~8 core categories as (b).

### 5.5 Historical events
Parsing base event names out of image filenames (e.g. `turkey_earthquake5`, `hurricane_florence`, `bata_explosion`) is only reliable for the pre/post-disaster-path fields (used consistently); the segmentation/relational-reasoning tasks use a different tile-numbering convention that breaks this parsing, so a fully verified event count across **all 9 tasks** was not achievable from filenames alone. Restricting to the captioning-task records (which use the reliable naming convention):
- Train captioning subset: 26 distinct named events
- Bench captioning subset: 30 distinct named events
- Union: still well under 36; this metric is **inconclusive** for the full dataset rather than confirmed or refuted — flagged as a limitation of filename-based auditing, not a disproof of the paper's "36 events" claim.

### 5.6 Bi-temporal image pairs
Computed two independent ways:

**(a) From the JSON records** (unique `(pre_image_path, post_image_path)` tuples, across all 9 tasks, train+bench combined): **12,861 unique pairs** (train: 9,806, bench: 3,055).

**(b) From the raw zip/folder contents directly** (independent of the JSON, counting actual image files):
- `train_images.zip`: 20,988 files = 9,746 pre + 10,289 optical-post + 953 SAR-post
- `test_images/` (extracted): 6,260 files = 2,942 pre + 2,942 optical-post + 376 SAR-post
- Combined raw pre-disaster image count: 9,746 + 2,942 = **12,688**
- Combined total post-disaster images (all sensors): 10,289+953+2,942+376 = **14,560**

Both methods land in the 12,000–15,000 range, **far below the paper's claimed 26,988 bi-temporal pairs**, no matter how "pair" is defined. See §6.

### 5.7 Image properties (sampled)
Sampled 80 random images each from `train_images.zip`, `box_train_images.zip`, and the extracted `test_images/` folder (240 images total), read with PIL:
- **Resolution: uniformly 1024×1024 in every single sample** (min = median = max = 1024×1024 in all three sources). No resolution variance found.
- Format: PNG throughout.
- Mode: mostly `RGB`; some `L` (single-channel grayscale), consistent with SAR imagery.
- **Corrupt/unreadable files in sample: 0 / 240.**
- **Exact byte-for-byte duplicate images:** 0/80 in `train_images.zip`, 0/80 in `test_images/`, **3/80 in `box_train_images.zip`** (plausible — multiple bounding-box "relational reasoning" instructions can legitimately reuse the same underlying crop image).
- Sensor confirmation: `pre_image_path` never contains "sar" (0 occurrences across 123,010 records) — pre-disaster imagery is always optical; SAR only appears for post-disaster imagery (9,366 of 70,558 records with a known `post_image_type`, ≈13%), consistent with the paper's stated rationale that SAR supplements optical for the *post*-disaster scene specifically.

---

## 6. Paper Claims vs. Actual — Explicit Comparison

| Claim (draft paper) | Actual (measured) | Verdict |
|---|---|---|
| 123,010 instruction-response pairs | 92,968 (train) + 30,042 (bench) = **123,010** | ✅ **MATCH** (exact) |
| 9 perception/reasoning tasks | **9** distinct `task` values found in both files | ✅ **MATCH** |
| Multi-sensor: optical + SAR | Both present; SAR is a ~13% minority, post-disaster only | ✅ **MATCH** (with the caveat that SAR is minority coverage, not parallel for every pair) |
| 10 hazard categories | 10 labels exist in the MCQ *option vocabulary*, but only **8** (`hurricane, fire, flood, earthquake, tornado, tsunami, explosion, volcano`) ever appear as an actual ground-truth answer; `landslide` and `conflict` never occur as an answer in the data inspected; "typhoon" doesn't appear as a label at all | ⚠️ **PARTIAL MATCH** — taxonomy size matches, but 2 of 10 categories are unconfirmed/absent as real examples |
| 36 historical events | ~26–30 distinct named events identifiable in the captioning-relevant filenames (reliable naming); full-dataset count inconclusive due to inconsistent filenames in segmentation-derived files | ⚠️ **INCONCLUSIVE**, plausibly in the right ballpark but not confirmed |
| 26,988 bi-temporal image pairs (total) | **12,861** unique `(pre,post)` pairs from JSON; **12,688–14,560** from raw file counts in the zips/folders — two independent methods agree with each other and both land far short of the claim | ❌ **MISMATCH** — actual is roughly **47–54%** of the claimed figure |
| 17,190 train / 5,024 test captioning pairs | **7,766 train / 2,363 test** (task `disaster caption` / `Disaster Report`) | ❌ **MISMATCH** — actual is **45.2%** / **47.0%** of claimed (train/test ratio is similar: claimed 3.42:1 vs actual 3.29:1, but absolute scale is roughly half) |
| License CC BY-NC-SA 4.0, attribution to xBD and BRIGHT | Confirmed (see §7) | ✅ **MATCH** |

**Overall pattern:** the paper's counts for things that are directly enumerable in the JSON files (total instruction pairs, task count) are exactly correct. But every count involving raw *images* (bi-temporal pairs, and by extension the captioning train/test split, which is image-pair-based) comes in at roughly **half** of what the paper claims. This is consistent across two independent measurement methods (JSON-derived and raw-zip-derived), so it is unlikely to be a counting bug on our side — it's more likely that either (a) the publicly released download differs from what's described in the paper (e.g. a reduced/sampled public release vs. the full research dataset), or (b) the paper's "26,988 pairs" / "17,190/5,024" figures count something the release doesn't expose (e.g. additional unreleased events, or double-counting each optical+SAR post image as 2 separate "pairs" against a shared pre-image in a way that still wouldn't reach 26,988 given our raw counts). **Do not use the paper's split-size numbers for capacity planning — use the measured 7,766/2,363.**

---

## 7. License / Attribution

Found in `README.md` (the original HuggingFace dataset card, present at project root):

- **License (YAML frontmatter):** `license: cc-by-nc-sa-4.0`
- **License (prose):** "All images and their associated annotations in DisasterM3 can be used for academic purposes only, but any commercial use is prohibited," with a CC BY-NC-SA 4.0 badge/link.
- **Acknowledgments section explicitly credits two derivative source datasets:**
  - **xBD** by Ritwik Gupta — license listed as **CC BY-NC-SA 4.0**
  - **BRIGHT** by Hongruixuan Chen — license listed as **CC BY-NC 4.0** (note: **no ShareAlike clause** on the upstream BRIGHT license, unlike xBD)
- Citation (BibTeX) for `wang2025disasterm3`, NeurIPS 2025, is present in the README and matches the paper's stated authors (Wang, Xuan, Qi, Liu, Liu, Wu, Chen, Song, Xia, Zheng, Yokoya).

**Confirmation/correction of the paper's claim:** The paper's claim of "CC BY-NC-SA 4.0 with attribution requirements to xBD and BRIGHT" is **confirmed** for the license type and for the fact that both are credited. One nuance worth carrying into our own project's license page: **BRIGHT's original license is CC BY-NC 4.0, not CC BY-NC-SA 4.0** — the "ShareAlike" obligation is on DisasterM3's own redistribution license and on the xBD-derived portion specifically, not inherited from BRIGHT itself. For IndoBLIP-Post-Disaster's own dataset card, all three (DisasterM3, xBD, BRIGHT) should be credited by name with links, and derivative use must remain non-commercial (NC) with share-alike (SA) terms honored for the DisasterM3/xBD-derived portions.

No separate `LICENSE` file or `NOTICE` file was found anywhere else in the downloaded folders (`DisasterM3_Bench/`, `DisasterM3_Instruct/`) — the README.md at project root is the sole source of license/attribution text found on disk.

---

## 8. Summary Table for Quick Reference

| Metric | Paper claim | Measured | Status |
|---|---|---|---|
| Total instruction-response pairs | 123,010 | 123,010 | Match |
| Tasks | 9 | 9 | Match |
| Hazard categories (labels) | 10 | 10 in vocabulary, 8 confirmed with real examples | Partial |
| Historical events | 36 | ~26–30 (captioning subset only; full-dataset count inconclusive) | Inconclusive |
| Bi-temporal pairs (total) | 26,988 | 12,861 (JSON) / 12,688–14,560 (raw files) | Mismatch (~half) |
| Captioning train pairs | 17,190 | 7,766 | Mismatch (45%) |
| Captioning test pairs | 5,024 | 2,363 | Mismatch (47%) |
| Multi-sensor (optical+SAR) | Yes | Yes (SAR = post-disaster only, ~13%) | Match |
| Caption language | (implied English source) | English, verified (0/282,271 fields match Indonesian stopwords) | Confirmed — translation pipeline is necessary |
| License | CC BY-NC-SA 4.0 + xBD/BRIGHT attribution | Confirmed | Match |
| Image resolution | (not specified numerically) | Uniform 1024×1024 PNG | — |
| Corrupt files (sampled) | — | 0/240 | Clean |
| Duplicate files (sampled) | — | 3/240 (all in box_train_images.zip) | Minor |
