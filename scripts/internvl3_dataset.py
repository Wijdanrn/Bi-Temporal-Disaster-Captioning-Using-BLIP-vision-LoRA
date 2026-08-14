"""
Dataset + collator for LoRA fine-tuning OpenGVLab/InternVL3-1B-hf on our translated
Indonesian disaster-report data. Unlike BLIP, InternVL3 takes pre/post as TWO SEPARATE
images (native multi-image support) instead of the side-by-side composite -- no image
surgery needed, the model's own processor handles dynamic tiling per image.

Reuses scripts/dataset.py's load_image() (zip-aware) unchanged -- image *loading* logic
doesn't change, only how the two images are fed to the model.

Prompt: Indonesian instruction (this is an Indonesian-language task; InternVL3's Qwen2
backbone has real Indonesian coverage, unlike BLIP's English tokenizer, so no reason to
prompt in English). Label masking: loss computed ONLY on the assistant's response tokens
(the translated ground-truth report), not on the prompt/user turn.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.dataset import load_image, _image_is_readable  # noqa: E402

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
PROMPT_ID = ("Jelaskan situasi kerusakan secara komprehensif berdasarkan gambar "
             "sebelum dan sesudah bencana, mencakup bencana, bangunan, jalan, vegetasi, "
             "badan air, dan pertanian.")


class InternVLDisasterDataset(Dataset):
    def __init__(self, split: str, processor, max_length: int = 1280,
                 drop_unreadable: bool = True, limit: int | None = None):
        path = os.path.join(PROCESSED_DIR, f"captions_{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f]
        if drop_unreadable:
            self.records = [r for r in self.records
                             if _image_is_readable(r["pre_image_path"]) and _image_is_readable(r["post_image_path"])]
        if limit is not None:
            self.records = self.records[:limit]
        self.processor = processor
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        pre = load_image(r["pre_image_path"])
        post = load_image(r["post_image_path"])
        target = r["ground_truth_id"]

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": PROMPT_ID}]},
        ]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        full_text = prompt_text + target + self.processor.tokenizer.eos_token

        # max_patches=1 disables InternVL's dynamic multi-tile cropping (5,186 tokens for a
        # 2-image prompt otherwise -- far too long for 8GB VRAM) -- single global tile per
        # image instead, same "resize whole image, no tiling" principle used for BLIP's
        # composite (see docs/design_decisions.md Fase 3). Brings the 2-image prompt to 578
        # tokens.
        prompt_inputs = self.processor(images=[pre, post], text=prompt_text, return_tensors="pt", max_patches=1)
        full_inputs = self.processor(images=[pre, post], text=full_text, return_tensors="pt", max_patches=1)

        prompt_len = prompt_inputs["input_ids"].shape[1]
        input_ids = full_inputs["input_ids"][0][: self.max_length]
        attention_mask = full_inputs["attention_mask"][0][: self.max_length]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt/user turn -- loss only on assistant response

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": full_inputs["pixel_values"],
            "image_num_patches": full_inputs.get("image_num_patches"),
            "index": i,
            "target_text": target,
            "ground_truth_id": r["ground_truth_id"],
            "pre_image_path": r["pre_image_path"],
            "post_image_path": r["post_image_path"],
        }


class InternVLCollator:
    """Top-level CLASS, not a closure -- Windows DataLoader spawn pickles the collate_fn
    into each worker; a nested function fails with num_workers > 0 (same fix as
    scripts/dataset.py's DisasterCollator)."""

    def __init__(self, pad_id: int, keep_meta: bool = False):
        self.pad_id = int(pad_id)
        self.keep_meta = bool(keep_meta)

    def __call__(self, items):
        n = max(x["input_ids"].shape[0] for x in items)
        input_ids, attn, labels = [], [], []
        for x in items:
            L = x["input_ids"].shape[0]
            pad = n - L
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            attn.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
            "pixel_values": torch.cat([x["pixel_values"] for x in items], dim=0),
        }
        if items[0].get("image_num_patches") is not None:
            batch["image_num_patches"] = torch.cat([x["image_num_patches"] for x in items], dim=0)
        if self.keep_meta:
            batch["index"] = [x["index"] for x in items]
            batch["target_text"] = [x["target_text"] for x in items]
            batch["ground_truth_id"] = [x["ground_truth_id"] for x in items]
            batch["pre_image_path"] = [x["pre_image_path"] for x in items]
            batch["post_image_path"] = [x["post_image_path"] for x in items]
        return batch


def make_collate_fn(pad_id: int, keep_meta: bool = False) -> InternVLCollator:
    return InternVLCollator(pad_id, keep_meta=keep_meta)
