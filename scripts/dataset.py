"""
IndoBLIP-Post-Disaster -- Fase 3 image pipeline: pre/post composite + Dataset + collator.

BLIP's vision tower accepts ONE image, but every record is a pre/post disaster PAIR.
Decision (docs/design_decisions.md Fase 3): fuse them into a single side-by-side composite
at BLIP's native 384x384, as a PREPROCESSING step. Nothing in BLIP is modified -- no
patch-embed surgery, no dual encoder, no position-embedding interpolation.

    from scripts.build_model import build_model
    from scripts.dataset import DisasterCaptionDataset, make_collate_fn

    bundle = build_model()
    ds = DisasterCaptionDataset("train", bundle)
    dl = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(bundle.codec))

Every normalization constant is READ FROM the checkpoint's own BlipImageProcessor -- none
are hardcoded here. `scripts/test_image_pipeline.py` asserts our fast tensor path is
numerically identical to calling BlipImageProcessor directly.

NOTE: always run with HF_HUB_DISABLE_XET=1 in the environment.
"""

from __future__ import annotations

import io
import json
import os
import threading
import zipfile
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
TRAIN_ZIP = os.path.join(ROOT, "DisasterM3_Instruct", "train_images.zip")
BENCH_DIR = os.path.join(ROOT, "DisasterM3_Bench")

# Where each image prefix physically lives. Routing is by PATH PREFIX, not by split name:
# the val split's records point at `train_images\...` too, so keying on the split would be
# wrong for val. Data-driven routing cannot drift out of sync with the jsonl.
ZIP_PREFIX = "train_images/"
DIR_PREFIX = "test_images/"

# Composite geometry. See docs/design_decisions.md Fase 3 for the measurement behind these.
LAYOUT = "horizontal"   # "horizontal" -> pre|post left/right ; "vertical" -> pre/post top/bottom
FIT = "stretch"         # "stretch" -> fill the half-canvas ; "pad" -> aspect-preserving letterbox


# --------------------------------------------------------------------------------------
# Zip handles: one per (process, thread)
# --------------------------------------------------------------------------------------

_ZIP_CACHE: dict[tuple, zipfile.ZipFile] = {}
_ZIP_LOCK = threading.Lock()


def _get_zip(path: str) -> zipfile.ZipFile:
    """A ZipFile handle private to this (pid, thread).

    zipfile.ZipFile is neither fork-safe nor thread-safe: two readers sharing one handle
    interleave seeks on the same file object and silently return corrupted bytes. That is a
    textbook silent failure -- PNG decode would usually still succeed on garbage. DataLoader
    workers are separate processes, so keying on pid is sufficient in practice; the thread id
    is included so a threaded loader is also safe.
    """
    key = (os.getpid(), threading.get_ident(), path)
    z = _ZIP_CACHE.get(key)
    if z is None:
        with _ZIP_LOCK:
            z = _ZIP_CACHE.get(key)
            if z is None:
                z = zipfile.ZipFile(path)
                _ZIP_CACHE[key] = z
    return z


def normalize_rel_path(p: str) -> str:
    """`train_images\\x.png` (as written in the jsonl) -> `train_images/x.png` (zip entry name)."""
    return p.replace("\\", "/").lstrip("./")


def load_image(rel_path: str) -> Image.Image:
    """Open one dataset image as RGB, from the zip or from disk depending on its prefix.

    Always returns mode RGB. SAR post-disaster images ship as single-channel `L`; 1,038 of
    the 10,129 records have a mode-mismatched pair (pre=RGB, post=L). Converting BEFORE any
    compositing is what stops `Image.paste` from throwing -- and, more importantly, stops a
    1-channel array from silently broadcasting against 3-channel mean/std later.
    """
    rel = normalize_rel_path(rel_path)
    if rel.startswith(ZIP_PREFIX):
        data = _get_zip(TRAIN_ZIP).read(rel)
        if not data:
            raise RuntimeError(f"zero-byte image in zip: {rel}")
        img = Image.open(io.BytesIO(data))
    elif rel.startswith(DIR_PREFIX):
        full = os.path.join(BENCH_DIR, rel)
        if os.path.getsize(full) == 0:
            raise RuntimeError(f"zero-byte image on disk: {full}")
        img = Image.open(full)
    else:
        raise ValueError(
            f"cannot route image path {rel!r}: expected prefix {ZIP_PREFIX!r} or {DIR_PREFIX!r}"
        )
    img.load()
    return img if img.mode == "RGB" else img.convert("RGB")


# --------------------------------------------------------------------------------------
# Normalization -- every constant sourced from the checkpoint's BlipImageProcessor
# --------------------------------------------------------------------------------------

@dataclass
class ImageSpec:
    """BLIP's real preprocessing contract, read off the checkpoint (never hardcoded)."""
    height: int
    width: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    rescale_factor: float
    resample: int
    do_normalize: bool
    do_rescale: bool

    @classmethod
    def from_processor(cls, ip) -> "ImageSpec":
        size = ip.size
        return cls(
            height=int(size["height"]),
            width=int(size["width"]),
            mean=tuple(float(x) for x in ip.image_mean),
            std=tuple(float(x) for x in ip.image_std),
            rescale_factor=float(ip.rescale_factor),
            resample=int(ip.resample),
            do_normalize=bool(ip.do_normalize),
            do_rescale=bool(ip.do_rescale),
        )

    @property
    def pil_resample(self):
        return Image.Resampling(self.resample)

    @property
    def pad_color(self) -> tuple[int, int, int]:
        """The uint8 colour that normalizes to exactly ~0 -- the least disruptive padding.

        Only used when FIT == "pad". Filling with black (0,0,0) instead would land at about
        -1.8 sigma per channel, i.e. a strong, learnable, meaningless edge signal.
        """
        return tuple(int(round(m * 255.0)) for m in self.mean)

    def to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL RGB -> normalized CHW float32, replicating BlipImageProcessor's arithmetic.

        Asserted numerically identical to BlipImageProcessor in scripts/test_image_pipeline.py.
        """
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f"expected HxWx3 RGB, got shape {arr.shape} (mode={img.mode})")
        if self.do_rescale:
            arr = arr * self.rescale_factor
        if self.do_normalize:
            arr = (arr - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


# --------------------------------------------------------------------------------------
# The composite
# --------------------------------------------------------------------------------------

def _fit_into(img: Image.Image, box_w: int, box_h: int, spec: ImageSpec, fit: str) -> Image.Image:
    """Render `img` into a box_w x box_h tile."""
    if fit == "stretch":
        return img.resize((box_w, box_h), spec.pil_resample)
    if fit == "pad":
        w, h = img.size
        s = min(box_w / w, box_h / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        tile = Image.new("RGB", (box_w, box_h), spec.pad_color)
        tile.paste(img.resize((nw, nh), spec.pil_resample), ((box_w - nw) // 2, (box_h - nh) // 2))
        return tile
    raise ValueError(f"unknown fit mode {fit!r}")


def make_composite(pre: Image.Image, post: Image.Image, spec: ImageSpec,
                   layout: str = LAYOUT, fit: str = FIT) -> Image.Image:
    """Fuse a pre/post pair into ONE image at BLIP's native resolution.

    Layout is pre-first in reading order: left|right for horizontal, top/bottom for vertical.
    No separator bar is drawn -- the half-boundary already lands exactly on a ViT patch
    boundary (192 = 12 x 16px patches), so no patch ever straddles both images, and a
    separator would only cost real pixels.
    """
    pre = pre if pre.mode == "RGB" else pre.convert("RGB")
    post = post if post.mode == "RGB" else post.convert("RGB")

    W, H = spec.width, spec.height
    canvas = Image.new("RGB", (W, H), spec.pad_color)
    if layout == "horizontal":
        bw, bh = W // 2, H
        canvas.paste(_fit_into(pre, bw, bh, spec, fit), (0, 0))
        canvas.paste(_fit_into(post, W - bw, bh, spec, fit), (bw, 0))
    elif layout == "vertical":
        bw, bh = W, H // 2
        canvas.paste(_fit_into(pre, bw, bh, spec, fit), (0, 0))
        canvas.paste(_fit_into(post, bw, H - bh, spec, fit), (0, bh))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return canvas


def composite_halves(t: torch.Tensor, layout: str = LAYOUT) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a composite tensor (C,H,W) back into its (pre, post) halves -- for verification."""
    c, h, w = t.shape
    if layout == "horizontal":
        return t[:, :, : w // 2], t[:, :, w // 2:]
    return t[:, : h // 2, :], t[:, h // 2:, :]


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

class DisasterCaptionDataset(Dataset):
    """Pre/post composite + tokenized Indonesian report, ready for BlipForConditionalGeneration.

    Tokenization is delegated entirely to Fase 2's `IndoReportCodec` (bundle.codec) so the
    verified [DEC]-first / [SEP]-last contract cannot drift.

    Returns per item:
        pixel_values   (3, H, W) float32, normalized with the checkpoint's own mean/std
        input_ids      (L,)      long, unpadded -- the collator pads to the batch max
        target_text    str       canonical reference text (eval only; see canonicalize=)
        post_image_type / index / paths  -- metadata, dropped by the collator
    """

    SPLITS = ("train", "val", "test")

    def __init__(self, split: str, bundle=None, *,
                 image_spec: ImageSpec | None = None,
                 codec=None,
                 layout: str = LAYOUT,
                 fit: str = FIT,
                 max_length: int | None = None,
                 canonicalize: bool = False,
                 drop_unreadable: bool = False,
                 deep_scan: bool = False,
                 limit: int | None = None):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        if bundle is None and (image_spec is None or codec is None):
            raise ValueError("pass either bundle=ModelBundle, or both image_spec= and codec=")

        self.split = split
        self.codec = codec if codec is not None else bundle.codec
        self.spec = image_spec if image_spec is not None else ImageSpec.from_processor(bundle.image_processor)
        self.layout, self.fit = layout, fit
        self.max_length = max_length or self.codec.max_length
        self.canonicalize = canonicalize

        path = os.path.join(PROCESSED_DIR, f"captions_{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        if limit is not None:
            self.records = self.records[:limit]

        self.dropped: list[dict] = []
        if drop_unreadable:
            keep, self.dropped = [], []
            for r in self.records:
                bad = [k for k in ("pre_image_path", "post_image_path")
                       if not _image_is_readable(r[k], deep=deep_scan)]
                (self.dropped if bad else keep).append({"record": r, "bad": bad} if bad else r)
            self.records = keep
            if self.dropped:
                print(f"[DisasterCaptionDataset:{split}] DROPPED {len(self.dropped)} record(s) "
                      f"with unreadable images:")
                for d in self.dropped[:10]:
                    print("   ", [d['record'][k] for k in d['bad']])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.records[i]
        pre = load_image(r["pre_image_path"])
        post = load_image(r["post_image_path"])
        comp = make_composite(pre, post, self.spec, layout=self.layout, fit=self.fit)
        pixel_values = self.spec.to_tensor(comp)

        text = r["ground_truth_id"]
        if self.canonicalize:
            text = self.codec.canonicalize(text)
        ids = self.codec.encode(text, max_length=self.max_length)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "target_text": text,
            "post_image_type": r.get("post_image_type"),
            "pre_image_path": r["pre_image_path"],
            "post_image_path": r["post_image_path"],
            "index": i,
        }


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _image_is_readable(rel_path: str, deep: bool = False) -> bool:
    """Cheap existence/soundness check for one image reference.

    Default (`deep=False`) reads only the zip CENTRAL DIRECTORY size and the 8-byte PNG
    signature -- O(1) per image, so scanning all 13,376 train references costs well under a
    second. That is deliberately matched to the failure mode actually observed in this corpus:
    a 0-byte file that downloaded empty from HuggingFace (`test_images/
    tuscaloosa_tornado_00000278_pre_disaster.png`). Measured across all three splits:
    train 0 bad / val 0 bad / test exactly 1 bad, and 0 corrupt in a 500-image decode-verified
    random sample -- so the expensive CRC pass buys nothing here.

    `deep=True` additionally CRC-verifies the PNG chunks (~8 ms/image); use it only if a
    future data refresh is suspected of shipping truncated files.
    """
    try:
        rel = normalize_rel_path(rel_path)
        if rel.startswith(ZIP_PREFIX):
            z = _get_zip(TRAIN_ZIP)
            try:
                info = z.getinfo(rel)
            except KeyError:
                return False                      # referenced path not in the archive at all
            if info.file_size == 0:
                return False
            if deep:
                with z.open(rel) as fh:
                    Image.open(fh).verify()
            else:
                with z.open(rel) as fh:
                    if fh.read(8) != _PNG_MAGIC:
                        return False
        else:
            full = os.path.join(BENCH_DIR, rel)
            if not os.path.exists(full) or os.path.getsize(full) == 0:
                return False
            with open(full, "rb") as fh:
                if deep:
                    Image.open(fh).verify()
                elif fh.read(8) != _PNG_MAGIC:
                    return False
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Collator
# --------------------------------------------------------------------------------------

def pad_encoded_batch(seqs: Sequence[Sequence[int]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad id sequences into a batch, following Fase 2's label contract exactly.

    labels == input_ids (BLIP shifts internally: logits[:, :-1] vs labels[:, 1:]), with -100
    on padding. test_image_pipeline.py asserts this reproduces codec.encode_batch() bit-for-bit.
    """
    n = max(len(s) for s in seqs)
    input_ids, attn, labels = [], [], []
    for s in seqs:
        s = list(s)
        pad = n - len(s)
        input_ids.append(s + [pad_id] * pad)
        attn.append([1] * len(s) + [0] * pad)
        labels.append(s + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class DisasterCollator:
    """Pad a list of dataset items into a batch BLIP can consume.

    Deliberately a top-level CLASS, not a closure. On Windows the DataLoader uses spawn, which
    pickles the collate_fn into each worker; a nested function raises
    `AttributeError: Can't get local object 'make_collate_fn.<locals>.collate'` the moment
    num_workers > 0. That bug only appears once you turn workers on, which is exactly when a
    training run is already long enough for the failure to be expensive.

    Only the pad id is stored, so the collator pickles cheaply (no tokenizer copy per worker).
    """

    def __init__(self, pad_id: int, keep_meta: bool = False):
        self.pad_id = int(pad_id)
        self.keep_meta = bool(keep_meta)

    def __call__(self, items: Iterable[dict]) -> dict[str, Any]:
        items = list(items)
        batch = pad_encoded_batch([it["input_ids"].tolist() for it in items], self.pad_id)
        batch["pixel_values"] = torch.stack([it["pixel_values"] for it in items])
        if self.keep_meta:
            batch["target_text"] = [it["target_text"] for it in items]
            batch["post_image_type"] = [it["post_image_type"] for it in items]
            batch["index"] = [it["index"] for it in items]
        return batch


def make_collate_fn(codec, keep_meta: bool = False) -> DisasterCollator:
    """Build a picklable collate_fn bound to Fase 2's codec (for its pad id)."""
    return DisasterCollator(codec.pad_id, keep_meta=keep_meta)
