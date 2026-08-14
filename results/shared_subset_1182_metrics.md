# BLIP (vision-LoRA final) scored on the shared 1182-item subset

Matched against `test_filenames_first_1182.csv` -- 1182 identical items used for the friend's Qwen2.5-VL-3B evaluation.

**Caveat**: this subset is a non-random first-N slice, covers only 6/10 disaster types (missing: api, api liar, kebakaran hutan, tsunami). Use for a fair head-to-head vs Qwen2.5-VL-3B only -- NOT as our headline result (that remains the full 2,362-item `results/metrics_table.md`).

| Metric | Baseline zero-shot | Text-only LoRA | **Text+Vision LoRA (final)** |
|---|---|---|---|
| CIDEr | 0.0001 | 0.0845 | **0.1071** |
| BLEU-4 | 0.0000 | 0.1989 | **0.2056** |
| ROUGE-L | 0.0192 | 0.4039 | **0.4112** |
| BERTScore F1 | 0.3471 | 0.7913 | **0.7949** |

BENCANA exact-match accuracy (final): 0.5989847715736041
BANGUNAN no-damage rate (final, generated): 0.733502538071066