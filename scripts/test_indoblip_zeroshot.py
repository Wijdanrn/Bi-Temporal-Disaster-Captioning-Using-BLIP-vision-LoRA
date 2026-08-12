"""
Quick proof #2 (real, not fabricated): load the external indoblip-lr-5e-06-epoch-30
checkpoint's ACTUAL weights, generate captions on a real sample of our disaster test
images (same composite pre|post pipeline every other system in this project uses), and
score it with the exact same CIDEr/BLEU-4/ROUGE-L + diagnostic functions used everywhere
else in Fase 4-5 -- so the three numbers below (indoblip zero-shot / our zero-shot BLIP
baseline / our final vision-LoRA model) are directly comparable on the identical 300
examples.

No training happens here. This is a cheap sanity check of what the checkpoint can
already do on our task as-is, before considering any transfer-learning use of it.

Usage:
    python scripts/test_indoblip_zeroshot.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import BlipForConditionalGeneration, BlipImageProcessor, AutoTokenizer  # noqa: E402
from scripts.dataset import (  # noqa: E402
    PROCESSED_DIR, ImageSpec, load_image, make_composite, _image_is_readable,
)
from scripts.build_model import build_indonesian_tokenizer, IndoReportCodec  # noqa: E402
from scripts.compute_metrics import load_jsonl, coco_style_score, bertscore_f1  # noqa: E402
from scripts.compare_beam_vs_greedy import diagnostics  # noqa: E402

BASE_MODEL_FOR_PROCESSOR = "Salesforce/blip-image-captioning-base"  # indoblip ships no processor of its own
INDOBLIP_CKPT = os.path.join(ROOT, "indoblip-lr-5e-06-epoch-30")
VISION_PRED_PATH = os.path.join(ROOT, "results", "predictions_test_vision.jsonl")
BASELINE_PRED_PATH = os.path.join(ROOT, "results", "predictions_baseline_id.jsonl")
OUT_PATH = os.path.join(ROOT, "results", "indoblip_zeroshot_comparison.json")
N_SAMPLE = 300
SEED = 42
MAX_NEW_TOKENS = 300
BATCH_SIZE = 32


class SampledTestImages(Dataset):
    def __init__(self, spec: ImageSpec, indices: list[int]):
        path = os.path.join(PROCESSED_DIR, "captions_test.jsonl")
        with open(path, encoding="utf-8") as f:
            all_records = [json.loads(l) for l in f]
        self.records = [all_records[i] for i in indices]
        self.indices = indices
        self.spec = spec

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        pre = load_image(r["pre_image_path"])
        post = load_image(r["post_image_path"])
        comp = make_composite(pre, post, self.spec)
        return {"pixel_values": self.spec.to_tensor(comp), "index": self.indices[i]}


def collate(items):
    return {
        "pixel_values": torch.stack([it["pixel_values"] for it in items]),
        "index": [it["index"] for it in items],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("No CUDA device detected.")

    # Same readable-image filtering every other predictions file uses, so indices line up.
    with open(os.path.join(PROCESSED_DIR, "captions_test.jsonl"), encoding="utf-8") as f:
        all_records = [json.loads(l) for l in f]
    readable = [i for i, r in enumerate(all_records)
                if _image_is_readable(r["pre_image_path"]) and _image_is_readable(r["post_image_path"])]
    rng = random.Random(SEED)
    sample_indices = sorted(rng.sample(readable, min(N_SAMPLE, len(readable))))
    print(f"[sample] {len(sample_indices)} test records (seed={SEED}) out of {len(readable)} readable")

    print(f"[load] indoblip checkpoint weights: {INDOBLIP_CKPT}")
    model = BlipForConditionalGeneration.from_pretrained(INDOBLIP_CKPT, dtype=torch.float32).to(device)
    model.eval()
    print(f"[load] processor/tokenizer from {BASE_MODEL_FOR_PROCESSOR} (checkpoint ships none of its own)")
    image_processor = BlipImageProcessor.from_pretrained(BASE_MODEL_FOR_PROCESSOR)
    en_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_FOR_PROCESSOR)
    print(f"  indoblip generation_config: bos={model.generation_config.bos_token_id} "
          f"eos={model.generation_config.eos_token_id} "
          f"(eos_token_id=2 is [unused1] in this vocab, not [SEP]={en_tokenizer.sep_token_id} -- "
          f"the same dormant footgun we found+fixed in our own Fase 2 build)")

    spec = ImageSpec.from_processor(image_processor)
    ds = SampledTestImages(spec, sample_indices)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=2)

    indoblip_captions: dict[int, str] = {}
    n_hit_max_len = 0
    t0 = time.time()
    with open(os.path.join(ROOT, "results", "predictions_test_indoblip.jsonl"), "w", encoding="utf-8") as fout:
        for bi, batch in enumerate(dl):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.no_grad():
                out_ids = model.generate(
                    pixel_values=pixel_values, max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1, do_sample=False,
                )
            captions = en_tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for j, idx in enumerate(batch["index"]):
                gen_len = int((out_ids[j] != en_tokenizer.pad_token_id).sum())
                if gen_len >= MAX_NEW_TOKENS - 1:
                    n_hit_max_len += 1
                indoblip_captions[idx] = captions[j].strip()
                fout.write(json.dumps({"index": idx, "generated_caption": captions[j].strip(),
                                        "num_generated_tokens": gen_len}, ensure_ascii=False) + "\n")
            print(f"  batch {bi+1}: {len(indoblip_captions)}/{len(ds)} "
                  f"({(time.time()-t0):.1f}s elapsed)", flush=True)

    print(f"[done generating] {len(indoblip_captions)} captions, "
          f"{n_hit_max_len} hit the {MAX_NEW_TOKENS}-token cap without stopping "
          f"(eos not reached -- consistent with the eos_token_id=2 bug)")

    print("[load] our existing baseline + final-model predictions for the SAME sample")
    base_rows = {r["index"]: r for r in load_jsonl(BASELINE_PRED_PATH)}
    vis_rows = {r["index"]: r for r in load_jsonl(VISION_PRED_PATH)}
    common = [i for i in sample_indices if i in base_rows and i in vis_rows]
    print(f"[data] {len(common)}/{len(sample_indices)} sampled indices present in all systems")

    print("[codec] building our tokenizer/codec for canonicalizing references")
    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)
    refs = {i: [codec.canonicalize(all_records[i]["ground_truth_id"])] for i in common}

    indoblip_res = {i: [indoblip_captions[i] or "."] for i in common}
    base_res = {i: [codec.canonicalize(base_rows[i]["caption_id_nllb"]) or "."] for i in common}
    vis_res = {i: [vis_rows[i]["generated_caption"]] for i in common}

    print("[metrics] indoblip zero-shot")
    indoblip_scores = coco_style_score(refs, indoblip_res)
    print("[metrics] our zero-shot BLIP baseline (for reference, same sample)")
    base_scores = coco_style_score(refs, base_res)
    print("[metrics] our final vision-LoRA model (for reference, same sample)")
    vis_scores = coco_style_score(refs, vis_res)

    print("[metrics] BERTScore x3")
    indoblip_bert, _ = bertscore_f1([indoblip_res[i][0] for i in common], [refs[i][0] for i in common])
    base_bert, _ = bertscore_f1([base_res[i][0] for i in common], [refs[i][0] for i in common])
    vis_bert, _ = bertscore_f1([vis_res[i][0] for i in common], [refs[i][0] for i in common])

    indoblip_diag = diagnostics([{"ground_truth_id": all_records[i]["ground_truth_id"],
                                   "generated_caption": indoblip_captions[i]} for i in common])
    vis_diag = diagnostics([{"ground_truth_id": all_records[i]["ground_truth_id"],
                              "generated_caption": vis_rows[i]["generated_caption"]} for i in common])

    results = {
        "n_sample": len(common),
        "seed": SEED,
        "indoblip_checkpoint": "indoblip-lr-5e-06-epoch-30 (external, general-domain, zero-shot on our task)",
        "indoblip_generation_config": {"bos_token_id": model.generation_config.bos_token_id,
                                        "eos_token_id": model.generation_config.eos_token_id},
        "n_generations_hit_max_new_tokens_without_eos": n_hit_max_len,
        "indoblip_zero_shot": {"cider": indoblip_scores["cider_avg"], "bleu4": indoblip_scores["bleu4_avg"],
                                "rougeL": indoblip_scores["rougeL_avg"], "bertscore_f1_indolem": indoblip_bert,
                                **indoblip_diag},
        "our_zero_shot_blip_baseline_same_sample": {"cider": base_scores["cider_avg"], "bleu4": base_scores["bleu4_avg"],
                                                      "rougeL": base_scores["rougeL_avg"], "bertscore_f1_indolem": base_bert},
        "our_final_vision_lora_same_sample": {"cider": vis_scores["cider_avg"], "bleu4": vis_scores["bleu4_avg"],
                                               "rougeL": vis_scores["rougeL_avg"], "bertscore_f1_indolem": vis_bert,
                                               **vis_diag},
        "sample_generations": [
            {"index": i, "reference": refs[i][0][:400], "indoblip_generated": indoblip_res[i][0][:400]}
            for i in common[:6]
        ],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {OUT_PATH}")
    print(json.dumps({k: v for k, v in results.items() if k != "sample_generations"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
