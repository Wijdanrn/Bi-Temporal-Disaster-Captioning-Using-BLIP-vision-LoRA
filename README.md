# IndoBLIP-Post-Disaster

**GEMASTIK Data Mining project.** `Salesforce/blip-image-captioning-base`, adapted with LoRA
(text decoder self-/cross-attention **and**, in the final model, the vision tower's fused
`qkv` projection) plus a full Indonesian-vocabulary swap (`indolem/indobert-base-uncased`),
fine-tuned to generate structured **Indonesian-language disaster-damage reports** (7 fixed
sections: `BENCANA`/`BANGUNAN`/`JALAN`/`VEGETASI`/`BADAN_AIR`/`PERTANIAN`/`KESIMPULAN`) from a
**pre-disaster + post-disaster satellite image pair**, built on top of the DisasterM3 dataset
below. The motivation: DisasterM3's real captioning ground truth is English-only, and disaster
response in Indonesia needs Indonesian-language reports — so every phase (dataset audit,
translation, tokenizer adaptation, image-pair compositing, LoRA training, evaluation, a
vision-tower bias-fix, and an honest cross-attention explainability study) was run for real on
this machine and is reproducible from this repo, not simulated or hand-typed.

**Everything below this section documents *our* project.** Everything below the `---`
separator further down is the **original DisasterM3 dataset's own README** (its HuggingFace
dataset card), kept verbatim because its CC-BY-NC-SA-4.0 license and xBD/BRIGHT attribution
must not be deleted. Do not confuse the two.

## 1. What's real vs. what's fiction (read this first)

The original GEMASTIK draft paper's **Table I** (headline metrics) and **Figures 3-5**
(example outputs) are **fictional placeholder mock-ups** written before any real experiment
existed. They were never treated as targets or ground truth in this repo. The explicit,
never-merged side-by-side of "draft paper claims" vs. "real, measured results from this repo"
lives in **`docs/draft_vs_real_comparison.md`**. Every number quoted anywhere else in this
README (see §6 below) is a *real, measured* result, sourced from `results/metrics.json` /
`results/metrics_table.md`, not from the draft.

## 2. Environment setup

All real work (training, evaluation, xAI, the demo app) runs under the conda env
**`torch5050`**, at:

```
C:\Users\Wijdan\anaconda3\envs\torch5050\python.exe
```

Key package versions (verified on this machine via `python -m pip show <pkg>`, 2026-08-11 —
see `requirements.txt` for the full pinned list):

| package | version |
|---|---|
| Python | 3.12.11 |
| torch | 2.9.0.dev20250831+cu128 (CUDA 12.8 nightly build, needed for this machine's RTX 5050 Laptop GPU) |
| transformers | 5.9.0 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| gradio | 6.22.0 |

To reproduce the environment from a clean checkout:

```bash
conda create -n torch5050 python=3.12 -y
conda activate torch5050
pip install -r requirements.txt
```

`requirements.txt` also lists `pycocoevalcap`, `sacrebleu`, and `bert-score` (evaluation, Fase
4-5) and `matplotlib` (xAI plotting, Fase 6). `pycocoevalcap`'s `PTBTokenizer` additionally
needs a local **Java runtime** on PATH (this machine used
`Eclipse Adoptium jdk-11.0.29.7-hotspot`, verified `java -version` working) — install any JDK
11+ if you don't have one.

**Required environment variables** (set before importing `transformers`/`huggingface_hub` —
every script in `scripts/` and `demo/app.py` sets these at the top via `os.environ.setdefault`,
so you normally don't need to set them yourself, but they matter if you import the modules
some other way):

| variable | value | why |
|---|---|---|
| `HF_HUB_DISABLE_XET` | `1` | the Xet download backend hangs indefinitely on this machine |
| `HF_HUB_OFFLINE` | `1` | avoids a flaky SSL/network path; the base model + tokenizers are already fully cached locally |

## 3. Repo layout

| path | what it is |
|---|---|
| `docs/design_decisions.md` | **the single source of truth for "why"** — every non-obvious design decision, with the measurement behind it, phase by phase (tokenizer swap, image compositing, LoRA placement, evaluation methodology, the vision-LoRA bias fix, the xAI faithfulness study) |
| `docs/dataset_audit.md` | Fase 0 — real numbers measured directly from the downloaded DisasterM3 files, including an explicit claim-vs-measured table against the paper |
| `docs/draft_vs_real_comparison.md` | the explicit, never-merged draft-paper-fiction vs. real-results comparison (see §1) |
| `docs/translation_report.md` | the English-to-Indonesian translation pipeline (Fase 1) |
| `notebooks/01_eda_dataset_audit.ipynb` … `06_xai_attention_rollout.ipynb` | executed, real-output walkthrough notebooks, EDA through xAI, mirroring the phases below |
| `scripts/` | the real pipeline — dataset build, tokenizer/model construction, training, generation, metrics, xAI (see §5) |
| `configs/*.json` | exact hyperparameters **and argv** actually used for each real training run (e.g. `configs/full_run_v3_vision.json` is the literal command line + library versions + GPU used to produce the final checkpoint) |
| `results/metrics.json`, `results/metrics_table.md` | final aggregate metrics, real, on the full test set |
| `results/predictions_test_vision.jsonl` | every one of the 2,362 real test-set generations from the final model |
| `results/xai/`, `results/xai_examples/` | Fase 6 explainability artifacts (JSON + 12 worked examples with heatmap overlays) |
| `checkpoints/full_run_v3_vision/best` | **the final reported model** (text LoRA + vision LoRA + retrained Indonesian embeddings) |
| `checkpoints/full_run_v2/best` | the text-only-LoRA ablation checkpoint (retained as evidence, not deleted — see `docs/design_decisions.md` §5.6) |
| `demo/app.py` | the Gradio demo app (see §7) |
| `requirements.txt` | pinned dependencies (see §2) |

## 4. Dataset

Built from `DisasterM3_Bench/` + `DisasterM3_Instruct/` (downloaded via `download_fast.py` /
`download_disasterm3.py`), filtered to the `disaster caption` / `Disaster Report` task and
translated to Indonesian. Real, measured split sizes (`data/processed/split_stats.json`,
seed 42): **6,999 train / 767 val / 2,363 test** records (2,362 usable at eval time — one
0-byte source image, see `docs/dataset_audit.md` §5.7 and `docs/design_decisions.md` §3.6).
Full provenance, license (CC BY-NC-SA 4.0, xBD + BRIGHT attribution), and paper-claim-vs-
measured discrepancies are in `docs/dataset_audit.md`.

## 5. Reproducing each phase

Full command lines, hyperparameters, and the reasoning behind each choice live in
`docs/design_decisions.md` (organized by phase/"Fase") — this is only a pointer list so it
cannot drift out of sync with that source of truth:

- **Fase 0 — dataset audit:** `docs/dataset_audit.md`, `notebooks/01_eda_dataset_audit.ipynb`.
- **Fase 1 — translation:** `scripts/translate_captions.py`, `scripts/strip_no_damage_tags.py`,
  `scripts/build_final_dataset.py`; write-up in `docs/translation_report.md`.
- **Fase 2 — tokenizer swap / embedding resize / LoRA setup:** `scripts/build_model.py`
  (`build_model()`); reasoning in `docs/design_decisions.md` §0-§9,
  `notebooks/02_data_preparation.ipynb`, `notebooks/03_model_architecture.ipynb`.
- **Fase 3 — pre/post image compositing + `Dataset`/collator:** `scripts/dataset.py`,
  verified by `scripts/test_image_pipeline.py`; reasoning in `docs/design_decisions.md`
  "Fase 3" section.
- **Training:** `scripts/train.py`; the exact argv/hyperparameters/GPU/library-versions used
  for every real run are frozen in `configs/*.json` (e.g. `configs/full_run_v3_vision.json`
  for the final model); walkthrough in `notebooks/04_training.ipynb`.
- **Fase 4 — real evaluation on the test set:** `scripts/generate_predictions.py` /
  `scripts/generate_predictions_vision.py` → `scripts/generate_baseline.py` →
  `scripts/translate_baseline.py` → `scripts/compute_metrics.py` /
  `scripts/compute_final_metrics.py`; methodology and results in `docs/design_decisions.md`
  "Fase 4" section, `notebooks/05_evaluation_and_results.ipynb`.
- **Fase 5 — vision-tower LoRA (the final model) and why it was added:**
  `scripts/build_model.py` (`vision_lora=` flag), `scripts/train.py --vision_lora
  --allow_new_lora_modules`, verified by `scripts/test_vision_lora.py`; the bias diagnosis,
  the decision, and the retraining results are in `docs/design_decisions.md` "Fase 5" section.
- **Fase 6 — explainability (cross-attention rollout + faithfulness tests):**
  `scripts/xai_attention_rollout.py` (`--mode selftest|examples|aggregate`),
  `scripts/xai_faithfulness_check.py` (`--mode occlusion|delins|reanalyze|summary`); exact
  reproduce commands in `docs/design_decisions.md` §6.9; walkthrough in
  `notebooks/06_xai_attention_rollout.ipynb`.

## 6. Headline result (real, measured — see `results/metrics_table.md` for the full table)

Final checkpoint `checkpoints/full_run_v3_vision/best`, N = 2,362 real paired test examples,
greedy decoding (`num_beams=1`, `max_new_tokens=300`):

| Metric | Zero-shot BLIP baseline | Text-only LoRA (ablation) | **Text+Vision LoRA (final)** |
|---|---|---|---|
| CIDEr | 0.0001 | 0.0980 | **0.1202** |
| BLEU-4 | 0.0000 | 0.2044 | **0.2095** |
| ROUGE-L | 0.0185 | 0.4103 | **0.4156** |
| BERTScore F1 (SPICE substitute) | 0.3470 | 0.7943 | **0.7971** |
| `BENCANA` (disaster type) exact-match | — | 58.8% | **72.3%** |

Final vs. zero-shot baseline: paired-bootstrap mean CIDEr difference **+0.1201**, 95% CI
**[0.1078, 0.1329]**. Full metrics, qualitative samples, and the vision-LoRA bias-fix diagnosis
are in `results/metrics_table.md` and `notebooks/05_evaluation_and_results.ipynb`. The
cross-attention heatmaps in `results/xai_examples/` are a real, image-dependent saliency
signal but are **not** validated as faithful per-word explanations (measured: token
selectivity r ≥ 0.999, no significant win over a random patch ranking on a causal
deletion/insertion test) — see `docs/design_decisions.md` Fase 6 for the full, honest
assessment before citing these figures as evidence of "what the model looked at".

## 7. Running the demo

A working Gradio app (`demo/app.py`) lets you upload a pre-/post-disaster image pair (or click
a real test-set example) and runs the actual trained pipeline end-to-end (composite → final
checkpoint → greedy Indonesian generation), with an optional cross-attention-rollout heatmap
view (caveat shown in the UI, per §6 above). Run:

```bash
"C:\Users\Wijdan\anaconda3\envs\torch5050\python.exe" demo/app.py
```

then open the printed local URL (default `http://127.0.0.1:7860`).

---

## Below: the original DisasterM3 dataset README (unmodified)

---
license: cc-by-nc-sa-4.0
task_categories:
- visual-question-answering
- image-segmentation
- image-to-text
- image-classification
language:
- en
tags:
- vision-language
- remote-sensing
- disaster
- multi-task
pretty_name: >-
  A multi-hazard, multi-sensor, and multi-task vision-language dataset for
  global-scale disaster assessment and response.
size_categories:
- 1B<n<10B
---


<h2 align="center">
  DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response
</h2>

<h5 align="center"><a href="https://junjue-wang.github.io/homepage/">Junjue Wang*</a>,
<a href="https://weihaoxuan.com">Weihao Xuan*</a>,
Heli Qi, Zhihao Liu, Kunyi Liu, Yuhan Wu, <a href="https://chrx97.com/"> Hongruixuan Chen</a>,
<a href="https://jtrneo.github.io/"> Jian Song</a></h5>
<h5 align="center">
Junshi Xia, <a href="https://zhuozheng.top/">Zhuo Zheng</a>, <a href="https://naotoyokoya.com/">Naoto Yokoya†</a></h5>

<h5 align="center">
* Equal Contributions
† Corresponding Author</h5>

`Paper`: https://arxiv.org/abs/2505.21089

`Code`: https://github.com/Junjue-Wang/DisasterM3


<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/DisasterM3/task_taxonomy.png?raw=true">
</div>

## Highlights
DisasterM3 includes 26,988 bi-temporal satellite images and 123k instruction pairs across 5 continents, with three characteristics:
1. Multi-hazard: 36 historical disaster events with significant impacts, which are categorized into 10 common natural and man-made disasters
2. Multi-sensor: Extreme weather during disasters often hinders optical sensor imaging, making it necessary to combine Synthetic Aperture Radar (SAR) imagery for post-disaster scenes
3. Multi-task: 9 disaster-related visual perception and reasoning tasks, harnessing the full potential of VLM's reasoning ability


## News
- 2025/10/23, We released the DisasterM3 instruct set.
- 2025/10/17, We released the benchmark set of DisasterM3.
- 2025/09/22, We are preparing the dataset and code.
- 2025/09/22, Our paper got accepted by NeurIPS 2025.


## Benchmark

Please run this code for benchmarking the DisasterM3 dataset.
Two examples:
Qwen2.5 VL:
```
python disaster_m3/pyscripts/run_vllm.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --subset bearing_body
```
InternVL3:
```
python disaster_m3/pyscripts/run_vllm.py --model_id OpenGVLab/InternVL3-78B --subset report
```


## Citation
If you use DisasterM3 in your research, please cite our following papers.
```text
  @article{wang2025disasterm3,
  title={DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response},
  author={Wang, Junjue and Xuan, Weihao and Qi, Heli and Liu, Zhihao and Liu, Kunyi and Wu, Yuhan and Chen, Hongruixuan and Song, Jian and Xia, Junshi and Zheng, Zhuo and Yokoya, Naoto},
  booktitle={Proceedings of the Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgments
This dataset builds upon the following excellent open datasets:
- **xBD dataset** by Ritwik Gupta
  - [Paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html)
  - [Dataset](https://xview2.org/dataset)
  - License: [CC BY-NC-SA 4.0]

- **BRIGHT dataset** by Hongruixuan Chen
  - [Repository](https://github.com/ChenHongruixuan/BRIGHT)
  - License: [CC BY-NC 4.0]


## License
All images and their associated annotations in DisasterM3 can be used for academic purposes only,
<font color="red"><b> but any commercial use is prohibited.</b></font>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">
<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Junjue-Wang/DisasterM3&type=Date)](https://www.star-history.com/#Junjue-Wang/DisasterM3&Date)
