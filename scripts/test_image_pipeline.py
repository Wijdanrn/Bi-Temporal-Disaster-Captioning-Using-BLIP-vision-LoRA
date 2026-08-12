"""
Fase 3 verification: pre/post composite, Dataset, collator -- end to end into the real model.

Guards the SILENT failures specific to this phase:
  * normalization constants drifting from the checkpoint's own BlipImageProcessor
  * a composite that looks fine but contains the same image twice (pre never loaded)
  * pre/post pasted in the wrong slot
  * single-channel SAR images broadcasting wrongly against 3-channel mean/std
  * a collator that disagrees with Fase 2's verified encode_batch label contract
  * pixel_values silently not participating in the loss at all

Run:
    HF_HUB_DISABLE_XET=1 python scripts/test_image_pipeline.py           # no model (fast)
    HF_HUB_DISABLE_XET=1 python scripts/test_image_pipeline.py --model   # + real fwd/bwd
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from scripts.build_model import BASE_MODEL, IndoReportCodec, build_indonesian_tokenizer  # noqa: E402
from scripts.dataset import (  # noqa: E402
    DisasterCaptionDataset, ImageSpec, PROCESSED_DIR, composite_halves,
    load_image, make_collate_fn, make_composite, normalize_rel_path, pad_encoded_batch,
)

_results: list[tuple[bool, str]] = []


def check(cond, label: str, detail: str = "") -> bool:
    _results.append((bool(cond), label))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    return bool(cond)


def load_records(split="train", limit=None):
    p = os.path.join(PROCESSED_DIR, f"captions_{split}.jsonl")
    with open(p, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    return recs[:limit] if limit else recs


# ======================================================================================

def test_spec():
    print("\n=== I1  image spec is READ FROM the checkpoint, not hardcoded ===")
    from transformers import BlipImageProcessor
    ip = BlipImageProcessor.from_pretrained(BASE_MODEL)
    spec = ImageSpec.from_processor(ip)

    print(f"       size={spec.width}x{spec.height} resample={spec.resample} "
          f"(PIL {spec.pil_resample.name})")
    print(f"       mean={spec.mean}\n       std ={spec.std}")
    print(f"       rescale={spec.rescale_factor}  pad_color={spec.pad_color}")

    check(spec.height == ip.size["height"] and spec.width == ip.size["width"],
          "spec size == processor size", f"{spec.width}x{spec.height}")
    check(tuple(spec.mean) == tuple(ip.image_mean), "spec mean == processor image_mean")
    check(tuple(spec.std) == tuple(ip.image_std), "spec std == processor image_std")
    check(spec.rescale_factor == ip.rescale_factor, "spec rescale == processor rescale_factor")
    check(spec.height % 16 == 0 and spec.width % 16 == 0,
          "resolution is divisible by ViT patch size 16",
          f"{spec.width // 16}x{spec.height // 16} patch grid")
    check((spec.width // 2) % 16 == 0,
          "HALF-width lands on a patch boundary (no patch straddles pre|post)",
          f"half={spec.width // 2} = {spec.width // 32} patches")
    return spec


def test_normalization_matches_hf(spec):
    print("\n=== I2  our fast tensor path == BlipImageProcessor, numerically ===")
    from transformers import BlipImageProcessor
    ip = BlipImageProcessor.from_pretrained(BASE_MODEL)

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (spec.height, spec.width, 3), dtype=np.uint8))

    ours = spec.to_tensor(img)
    theirs = ip(images=img, do_resize=False, return_tensors="pt")["pixel_values"][0]

    check(ours.shape == theirs.shape, "shape matches", f"{tuple(ours.shape)} vs {tuple(theirs.shape)}")
    d = (ours - theirs).abs().max().item()
    check(d < 1e-5, "max abs difference < 1e-5", f"{d:.3e}")

    # and on a REAL composite, which is what actually gets fed to the model
    r = load_records("train", 1)[0]
    comp = make_composite(load_image(r["pre_image_path"]), load_image(r["post_image_path"]), spec)
    d2 = (spec.to_tensor(comp)
          - ip(images=comp, do_resize=False, return_tensors="pt")["pixel_values"][0]).abs().max().item()
    check(d2 < 1e-5, "identical on a real composite too", f"{d2:.3e}")

    # padding colour really is the neutral one
    flat = Image.new("RGB", (32, 32), spec.pad_color)
    check(spec.to_tensor(flat).abs().max().item() < 0.02,
          "pad_color normalizes to ~0 (neutral padding)",
          f"max |z| = {spec.to_tensor(flat).abs().max().item():.4f}")


def test_sar_channels(spec):
    print("\n=== I3  SAR single-channel handling ===")
    recs = load_records("train")
    sar = [r for r in recs if r["post_image_type"] == "SAR"][:6]
    check(len(sar) > 0, "found SAR records in train", f"{sum(1 for r in recs if r['post_image_type']=='SAR')} total")

    raw_modes = set()
    for r in sar:
        rel = normalize_rel_path(r["post_image_path"])
        import zipfile
        from scripts.dataset import TRAIN_ZIP
        with zipfile.ZipFile(TRAIN_ZIP).open(rel) as fh:
            raw_modes.add(Image.open(fh).mode)
    check("L" in raw_modes, "SAR post images really are single-channel on disk", f"modes={raw_modes}")

    n_ok = n_gray = 0
    for r in sar:
        pre, post = load_image(r["pre_image_path"]), load_image(r["post_image_path"])
        n_ok += (pre.mode == "RGB" and post.mode == "RGB")
        a = np.asarray(post)
        n_gray += bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
        t = spec.to_tensor(make_composite(pre, post, spec))
        assert t.shape == (3, spec.height, spec.width)
    check(n_ok == len(sar), "load_image returns RGB for BOTH halves of every SAR pair",
          f"{n_ok}/{len(sar)}")
    check(n_gray == len(sar), "L->RGB replicated the channel (R==G==B) rather than padding zeros",
          f"{n_gray}/{len(sar)}")

    # the mismatch is real: this pair would crash a naive paste
    r = sar[0]
    import zipfile
    from scripts.dataset import TRAIN_ZIP
    z = zipfile.ZipFile(TRAIN_ZIP)
    with z.open(normalize_rel_path(r["pre_image_path"])) as fh:
        m_pre = Image.open(fh).mode
    with z.open(normalize_rel_path(r["post_image_path"])) as fh:
        m_post = Image.open(fh).mode
    check(m_pre != m_post, "control: this pair genuinely has mismatched raw modes",
          f"pre={m_pre} post={m_post}")

    t = spec.to_tensor(make_composite(load_image(r["pre_image_path"]),
                                      load_image(r["post_image_path"]), spec))
    check(torch.isfinite(t).all().item() and t.shape[0] == 3,
          "mismatched pair still yields a finite (3,H,W) tensor", str(tuple(t.shape)))


def test_composite_geometry(spec):
    print("\n=== I4  composite geometry: pre goes LEFT, post goes RIGHT ===")
    r = load_records("train", 1)[0]
    pre, post = load_image(r["pre_image_path"]), load_image(r["post_image_path"])
    comp = make_composite(pre, post, spec, layout="horizontal", fit="stretch")

    check(comp.size == (spec.width, spec.height), "composite size == BLIP input size", str(comp.size))

    hw, hh = spec.width // 2, spec.height
    left = np.asarray(comp.crop((0, 0, hw, hh)))
    right = np.asarray(comp.crop((hw, 0, spec.width, hh)))
    exp_pre = np.asarray(pre.resize((hw, hh), spec.pil_resample))
    exp_post = np.asarray(post.resize((spec.width - hw, hh), spec.pil_resample))

    check(np.array_equal(left, exp_pre), "LEFT half is bit-identical to resized PRE image")
    check(np.array_equal(right, exp_post), "RIGHT half is bit-identical to resized POST image")
    check(not np.array_equal(left, exp_post), "control: left is NOT the post image")

    # vertical layout still works, for the configurable path
    cv = make_composite(pre, post, spec, layout="vertical", fit="stretch")
    top = np.asarray(cv.crop((0, 0, spec.width, spec.height // 2)))
    check(np.array_equal(top, np.asarray(pre.resize((spec.width, spec.height // 2), spec.pil_resample))),
          "vertical layout puts PRE on top")
    cp = make_composite(pre, post, spec, fit="pad")
    check(cp.size == (spec.width, spec.height), "fit='pad' path also produces a valid canvas")


def test_fit_choice(spec):
    """Guard the reasoning behind FIT='stretch' (docs/design_decisions.md Fase 3.2).

    The choice rests on three measurable premises. If any of them stops holding -- e.g. a
    future data refresh introduces non-square imagery -- the decision has to be revisited,
    and this test is what will say so.
    """
    print("\n=== I4b  premises behind FIT='stretch' still hold ===")
    from transformers import BlipImageProcessor
    ip = BlipImageProcessor.from_pretrained(BASE_MODEL)

    # premise 1: BLIP's OWN processor resizes to a square without preserving aspect ratio,
    # so anisotropic scaling is in-distribution for this checkpoint, not an abuse of it.
    tall = Image.fromarray(np.random.default_rng(0).integers(0, 256, (800, 200, 3), dtype=np.uint8))
    shp = tuple(ip(images=tall, return_tensors="pt")["pixel_values"].shape[-2:])
    check(shp == (spec.height, spec.width),
          "BlipImageProcessor itself STRETCHES a 1:4 image to square (aspect is not preserved)",
          f"200x800 -> {shp[1]}x{shp[0]}")

    # premise 2: sources are square, so aspect-preserving fit CANNOT fill a half-canvas --
    # it would leave 50% of the ViT's patches constant padding.
    recs = load_records("train", 12) + load_records("test", 6)
    sizes = set()
    for r in recs:
        sizes.add(load_image(r["pre_image_path"]).size)
        sizes.add(load_image(r["post_image_path"]).size)
    check(all(w == h for w, h in sizes), "source images are square (this is what forces the choice)",
          f"sizes={sorted(sizes)}")

    half_patches = (spec.width // 2 // 16) * (spec.height // 16)
    pad_side = min(spec.width // 2, spec.height)
    pad_patches = (pad_side // 16) ** 2
    grid = (spec.width // 16) * (spec.height // 16)
    print(f"       informative patches: stretch {2*half_patches}/{grid} (100%)  "
          f"vs pad {2*pad_patches}/{grid} ({200*pad_patches//grid}%)")
    check(2 * half_patches == grid and 2 * pad_patches < grid,
          "stretch uses every ViT patch; pad wastes half of them on constant fill",
          f"{2*half_patches} vs {2*pad_patches}")

    # premise 3: despite the 2:1 distortion, stretch retains MORE detail than the
    # aspect-true tile, because a 192x384 tile simply has 2x the samples of a 192x192 one.
    def psnr(a, b):
        m = float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))
        return 10 * math.log10(255.0 ** 2 / max(m, 1e-9))

    R = spec.pil_resample
    p_stretch, p_pad, p_solo = [], [], []
    for r in load_records("train", 6):
        for k in ("pre_image_path", "post_image_path"):
            im = load_image(r[k])
            W, H = im.size
            p_stretch.append(psnr(im, im.resize((spec.width // 2, spec.height), R).resize((W, H), R)))
            p_pad.append(psnr(im, im.resize((pad_side, pad_side), R).resize((W, H), R)))
            p_solo.append(psnr(im, im.resize((spec.width, spec.height), R).resize((W, H), R)))
    ms, mp_, mo = np.mean(p_stretch), np.mean(p_pad), np.mean(p_solo)
    print(f"       round-trip PSNR vs 1024px source ({len(p_stretch)} imgs): "
          f"stretch {ms:.2f} dB | aspect-true tile {mp_:.2f} dB | uncomposited 384 {mo:.2f} dB")
    check(ms > mp_, "stretch retains MORE detail than the aspect-preserving tile",
          f"+{ms - mp_:.2f} dB")
    print(f"       (cost of compositing at all: -{mo - ms:.2f} dB vs feeding one image alone)")


def test_halves_differ(spec):
    print("\n=== I5  the composite really contains TWO different images ===")
    recs = load_records("train", 400)
    sample = recs[:20] + [r for r in recs if r["post_image_type"] == "SAR"][:8]

    rows, n_diff = [], 0
    for r in sample:
        t = spec.to_tensor(make_composite(load_image(r["pre_image_path"]),
                                          load_image(r["post_image_path"]), spec))
        L, R = composite_halves(t)
        dmean = (L.mean() - R.mean()).abs().item()
        dstd = (L.std() - R.std()).abs().item()
        mad = (L - R).abs().mean().item()
        rows.append((dmean, dstd, mad))
        n_diff += mad > 1e-3
    arr = np.array(rows)
    print(f"       left-vs-right over {len(sample)} composites: "
          f"|dmean| {arr[:,0].mean():.4f}, |dstd| {arr[:,1].mean():.4f}, MAD {arr[:,2].mean():.4f}")
    print(f"       min MAD over the sample = {arr[:,2].min():.4f}")
    check(n_diff == len(sample), "every composite's halves differ (pre != post content)",
          f"{n_diff}/{len(sample)}")

    # NEGATIVE CONTROL -- proves the test above has power. If the loader silently used the
    # post image for both halves, make_composite(post, post) is exactly what we'd get, and
    # the check must be able to see it.
    r = sample[0]
    post = load_image(r["post_image_path"])
    Ld, Rd = composite_halves(spec.to_tensor(make_composite(post, post, spec)))
    same_mad = (Ld - Rd).abs().mean().item()
    check(same_mad < 1e-6,
          "negative control: composite(post,post) halves ARE identical -> test has power",
          f"MAD {same_mad:.2e} vs real-pair mean {arr[:,2].mean():.4f}")


def test_collator_matches_fase2():
    print("\n=== I6  collator reproduces Fase 2's encode_batch exactly ===")
    tok = build_indonesian_tokenizer()
    codec = IndoReportCodec(tok)
    recs = load_records("train", 5)
    texts = [r["ground_truth_id"] for r in recs]

    ref = codec.encode_batch(texts)
    mine = pad_encoded_batch([codec.encode(t) for t in texts], codec.pad_id)
    for k in ("input_ids", "attention_mask", "labels"):
        check(torch.equal(ref[k], mine[k]), f"pad_encoded_batch matches encode_batch: {k}",
              str(tuple(mine[k].shape)))

    ii, am, lb = mine["input_ids"], mine["attention_mask"], mine["labels"]
    check(bool(((lb == -100) == (am == 0)).all()), "labels == -100 exactly where mask == 0")
    check(bool((lb[am == 1] == ii[am == 1]).all()),
          "labels == input_ids on real tokens (BLIP shifts internally)")
    check(bool((ii[:, 0] == codec.dec_id).all()), "every row starts with [DEC]")
    check(all(int(ii[i][am[i] == 1][-1]) == codec.sep_id for i in range(len(texts))),
          "every row's last REAL token is [SEP]")
    return codec


def test_dataset_and_routing(spec, codec):
    print("\n=== I7  Dataset: source routing + text round-trip ===")
    ds_tr = DisasterCaptionDataset("train", image_spec=spec, codec=codec, limit=6)
    ds_va = DisasterCaptionDataset("val", image_spec=spec, codec=codec, limit=4)
    ds_te = DisasterCaptionDataset("test", image_spec=spec, codec=codec, limit=4)

    check(normalize_rel_path(ds_va.records[0]["pre_image_path"]).startswith("train_images/"),
          "VAL images route to the ZIP (val paths are train_images/*, not test_images/*)",
          ds_va.records[0]["pre_image_path"])
    check(normalize_rel_path(ds_te.records[0]["pre_image_path"]).startswith("test_images/"),
          "TEST images route to the extracted directory", ds_te.records[0]["pre_image_path"])

    for name, ds in (("train", ds_tr), ("val", ds_va), ("test", ds_te)):
        it = ds[0]
        ok = (it["pixel_values"].shape == (3, spec.height, spec.width)
              and it["pixel_values"].dtype == torch.float32
              and torch.isfinite(it["pixel_values"]).all())
        check(ok, f"{name}[0] pixel_values is finite {tuple(it['pixel_values'].shape)} float32")

    print("\n=== I8  target text survives the Dataset unchanged ===")
    n_ok = n_dec = 0
    for i in range(len(ds_tr)):
        it = ds_tr[i]
        ids = it["input_ids"].tolist()
        n_dec += ids[0] == codec.dec_id and ids[-1] == codec.sep_id
        canon = codec.canonicalize(it["target_text"])
        n_ok += codec.decode(ids) == canon
    check(n_dec == len(ds_tr), "[DEC] first / [SEP] last on every dataset item", f"{n_dec}/{len(ds_tr)}")
    check(n_ok == len(ds_tr), "decode(dataset ids) == canonicalize(target) for every item",
          f"{n_ok}/{len(ds_tr)}")

    # canonicalize=True must not change the TRAINING ids, only the reference string
    ds_c = DisasterCaptionDataset("train", image_spec=spec, codec=codec, limit=6, canonicalize=True)
    same = sum(torch.equal(ds_tr[i]["input_ids"], ds_c[i]["input_ids"]) for i in range(6))
    check(same == 6, "canonicalize=True leaves input_ids identical (affects reference text only)",
          f"{same}/6")
    return ds_tr


def test_collate_batch(ds, codec, spec):
    print("\n=== I9  collate_fn batching ===")
    collate = make_collate_fn(codec, keep_meta=True)
    items = [ds[i] for i in range(4)]
    b = collate(items)

    check(b["pixel_values"].shape == (4, 3, spec.height, spec.width),
          "pixel_values stacked", str(tuple(b["pixel_values"].shape)))
    L = max(len(it["input_ids"]) for it in items)
    check(b["input_ids"].shape == (4, L), "input_ids padded to batch max", f"{tuple(b['input_ids'].shape)} (max len {L})")
    check(b["input_ids"].shape == b["attention_mask"].shape == b["labels"].shape,
          "text tensors share a shape")
    check(bool(((b["labels"] == -100) == (b["attention_mask"] == 0)).all()),
          "batch labels masked exactly on padding")
    lens = [int(x) for x in b["attention_mask"].sum(1)]
    check(lens == [len(it["input_ids"]) for it in items],
          "attention_mask lengths match unpadded lengths", str(lens))

    # Regression guard: Windows DataLoader uses spawn and PICKLES the collate_fn and the
    # dataset into every worker. A closure-based collator passes every test above and then
    # dies the instant num_workers > 0.
    import pickle
    try:
        rt = pickle.loads(pickle.dumps(collate))
        ok = torch.equal(rt(items)["input_ids"], b["input_ids"])
    except Exception as e:  # noqa: BLE001
        ok, e_ = False, e
        print("       pickle error:", e_)
    check(ok, "collate_fn is picklable and behaves identically after a round trip "
              "(required for num_workers>0 on Windows spawn)")
    try:
        ds_rt = pickle.loads(pickle.dumps(ds))
        ok2 = len(ds_rt) == len(ds)
    except Exception as e:  # noqa: BLE001
        ok2 = False
        print("       pickle error:", e)
    check(ok2, "Dataset is picklable (zip handles are module-level, not instance state)")
    return b


def test_integrity_scan():
    """Scan ALL splits, not just test.

    A dead image discovered at epoch 3 hour 2 is the expensive version of this bug, so the
    whole corpus is swept up front. The scan is O(1) per image (zip central-directory size +
    PNG signature), which is why it is affordable to run over all 19,220 references.
    """
    print("\n=== I10  full-corpus integrity scan (all three splits) ===")
    expect = {"train": 0, "val": 0, "test": 1}   # measured; test's 1 is a 0-byte upstream file
    for split, n_bad in expect.items():
        t0 = time.time()
        ds = DisasterCaptionDataset(split, image_spec=SPEC, codec=CODEC, drop_unreadable=True)
        check(len(ds.dropped) == n_bad,
              f"{split}: {n_bad} unreadable record(s)",
              f"{len(ds.dropped)} dropped, {len(ds)} kept, scan {time.time()-t0:.1f}s")
        for d in ds.dropped:
            print("       dropped:", d["bad"], d["record"]["pre_image_path"])
    check(len(DisasterCaptionDataset("test", image_spec=SPEC, codec=CODEC,
                                     drop_unreadable=True)) == 2362,
          "test split usable size = 2362")

    # NEGATIVE CONTROL: the scan must actually be able to reject something. Without this,
    # "0 bad in train" is indistinguishable from a scanner that returns True unconditionally.
    from scripts.dataset import _image_is_readable
    check(not _image_is_readable("test_images/tuscaloosa_tornado_00000278_pre_disaster.png"),
          "control: scanner rejects the known 0-byte file")
    check(not _image_is_readable("train_images/this_file_does_not_exist.png"),
          "control: scanner rejects a path absent from the zip")
    check(_image_is_readable("train_images/hurricane_florence_00000352_pre_disaster.png"),
          "control: scanner accepts a known-good zip entry")


def dump_sample_composites(spec, n=4):
    """Write a few real composites to disk so a human can eyeball them."""
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures")
    os.makedirs(out, exist_ok=True)
    recs = load_records("train", 300)
    pick = recs[:n - 1] + [r for r in recs if r["post_image_type"] == "SAR"][:1]
    paths = []
    for i, r in enumerate(pick):
        comp = make_composite(load_image(r["pre_image_path"]), load_image(r["post_image_path"]), spec)
        p = os.path.join(out, f"composite_{i}_{r['post_image_type']}.png")
        comp.save(p)
        paths.append(p)
    print("\n=== dumped sample composites (left=PRE, right=POST) ===")
    for p in paths:
        print("      ", p)


def test_throughput(spec, codec):
    """Is reading 29.7 GB of zip on every step fast enough to train against?

    Answered with a stage breakdown rather than a single number, because the answer changes
    what you would do about it: if I/O dominated, extracting the zip would be worth the disk;
    if decode dominates, extraction buys nothing and only worker count helps.
    """
    print("\n=== I11  data throughput (training-time feasibility) ===")
    import zipfile as _zf
    from scripts.dataset import TRAIN_ZIP, load_image, normalize_rel_path

    infos = _zf.ZipFile(TRAIN_ZIP).infolist()
    stored = sum(1 for i in infos if i.compress_type == 0)
    check(stored == len(infos),
          "zip is STORED (uncompressed) -> reads are seeks, not inflate",
          f"{stored}/{len(infos)} entries")

    ds = DisasterCaptionDataset("train", image_spec=spec, codec=codec)
    idx = np.random.default_rng(0).integers(0, len(ds), 40)

    # stage breakdown on the same random sample
    t_io = t_dec = t_rest = 0.0
    for i in idx:
        r = ds.records[int(i)]
        for k in ("pre_image_path", "post_image_path"):
            from scripts.dataset import _get_zip
            t0 = time.perf_counter(); blob = _get_zip(TRAIN_ZIP).read(normalize_rel_path(r[k]))
            t_io += time.perf_counter() - t0
            t0 = time.perf_counter()
            im = Image.open(__import__("io").BytesIO(blob)); im.load()
            t_dec += time.perf_counter() - t0

    t0 = time.time()
    for i in idx:
        ds[int(i)]
    per = (time.time() - t0) / len(idx)
    t_rest = max(per * len(idx) - t_io - t_dec, 0.0)
    tot = t_io + t_dec + t_rest
    print(f"       single-process random access: {per*1000:.1f} ms/sample ({1/per:.1f} samples/s)")
    print(f"         zip read (I/O)   {t_io/len(idx)*1000:6.1f} ms  ({100*t_io/tot:4.1f}%)")
    print(f"         PNG decode       {t_dec/len(idx)*1000:6.1f} ms  ({100*t_dec/tot:4.1f}%)")
    print(f"         resize+norm+tok  {t_rest/len(idx)*1000:6.1f} ms  ({100*t_rest/tot:4.1f}%)")
    check(t_dec > t_io, "PNG DECODE dominates, not zip I/O -> extracting the 29.7 GB zip "
                        "would not help; more workers would",
          f"decode {t_dec/len(idx)*1000:.0f} ms vs io {t_io/len(idx)*1000:.0f} ms")
    print(f"       => 1 epoch ({len(ds)} samples) = {per*len(ds)/60:.1f} min at num_workers=0")
    check(per < 1.0, "per-sample cost is under 1 s", f"{per*1000:.0f} ms")

    # STEADY-STATE with workers, measured after the first batch so Windows spawn + import
    # cost is excluded (that cost is real but paid once per epoch, and only if you forget
    # persistent_workers=True).
    from torch.utils.data import DataLoader
    print(f"       ({os.cpu_count()} logical cores available)")
    best = per and 1 / per
    for w in (4, 8):
        dl = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(codec), shuffle=True,
                        num_workers=w, persistent_workers=True, prefetch_factor=4)
        it = iter(dl)
        t0 = time.perf_counter(); next(it); startup = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(12):
            next(it)
        sps = 48 / (time.perf_counter() - t0)
        best = max(best, sps)
        print(f"       num_workers={w}: first batch {startup:5.1f}s, steady {sps:5.1f} samples/s "
              f"-> {len(ds)/sps/60:.1f} min/epoch")
        del it, dl
    check(best > 3 / per, "worker parallelism recovers >3x over single-process",
          f"{best:.0f} vs {1/per:.0f} samples/s")


def test_model_forward(spec, codec):
    print("\n=== I12  REAL forward + backward through the LoRA model ===")
    from scripts.build_model import build_model, get_lm_head

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = build_model(verbose=False, device=dev)
    ds = DisasterCaptionDataset("train", bundle=b, limit=64)
    collate = make_collate_fn(b.codec)

    B = 4
    batch = collate([ds[i] for i in range(B)])
    batch = {k: v.to(dev) for k, v in batch.items()}
    print(f"       device={dev} batch={B} pixel_values={tuple(batch['pixel_values'].shape)} "
          f"seq_len={batch['input_ids'].shape[1]}")

    b.model.train()
    out = b.model(**batch)
    loss = out.loss
    check(out.logits.shape[:2] == batch["input_ids"].shape,
          "logits align with input_ids", str(tuple(out.logits.shape)))
    check(out.logits.shape[-1] == len(b.tokenizer), "logits vocab dim == len(tokenizer)",
          str(out.logits.shape[-1]))
    check(torch.isfinite(loss).item(), "loss is finite", f"{loss.item():.4f}")
    # a fresh Indonesian embedding matrix should start near ln(V); wildly off = something broken
    lnV = math.log(len(b.tokenizer))
    check(0.2 * lnV < loss.item() < 2.0 * lnV,
          f"loss is in a sane range around ln(V)={lnV:.2f}", f"loss={loss.item():.4f}")

    loss.backward()
    base = b.model.base_model.model
    emb = base.get_input_embeddings()
    head = get_lm_head(base)
    check(emb.weight.grad is not None and emb.weight.grad.abs().mean().item() > 0,
          "embedding matrix has gradient",
          f"mean|g| = {emb.weight.grad.abs().mean().item():.3e}")
    check(head.bias.grad is not None and head.bias.grad.abs().mean().item() > 0,
          "lm head bias has gradient", f"mean|g| = {head.bias.grad.abs().mean().item():.3e}")
    lb = [(n, p) for n, p in b.model.named_parameters() if "lora_B" in n and p.requires_grad]
    n_lb = sum(1 for _, p in lb if p.grad is not None and p.grad.abs().sum().item() > 0)
    check(n_lb > 0, "LoRA lora_B matrices have non-zero gradient", f"{n_lb}/{len(lb)} modules")
    missing = [n for n, p in b.model.named_parameters() if p.requires_grad and p.grad is None]
    check(len(missing) == 0, "no trainable parameter is missing a gradient",
          f"{len(missing)} missing")

    print("\n=== I13  BOTH halves of the composite actually reach the loss ===")
    b.model.zero_grad(set_to_none=True)
    px = batch["pixel_values"].detach().clone().requires_grad_(True)
    out2 = b.model(pixel_values=px, input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"], labels=batch["labels"])
    out2.loss.backward()
    g = px.grad
    check(g is not None and torch.isfinite(g).all().item(), "pixel_values receives a gradient")
    gl, gr = composite_halves(g[0])
    nl, nr = gl.abs().sum().item(), gr.abs().sum().item()
    print(f"       |grad| left(pre) = {nl:.4e}   right(post) = {nr:.4e}   ratio {nl/max(nr,1e-30):.3f}")
    check(nl > 0, "PRE half influences the loss (image is not silently ignored)")
    check(nr > 0, "POST half influences the loss")

    # changing the image must change the loss -- otherwise the vision path is decorative
    b.model.eval()
    with torch.no_grad():
        l_real = b.model(**batch).loss.item()
        shuf = dict(batch)
        shuf["pixel_values"] = batch["pixel_values"].flip(0)
        l_shuf = b.model(**shuf).loss.item()
        blank = dict(batch)
        blank["pixel_values"] = torch.zeros_like(batch["pixel_values"])
        l_blank = b.model(**blank).loss.item()
    print(f"       loss  real={l_real:.4f}  images-shuffled={l_shuf:.4f}  blank={l_blank:.4f}")
    check(abs(l_real - l_shuf) > 1e-4, "mismatching images to captions changes the loss",
          f"delta {abs(l_real - l_shuf):.4e}")
    check(abs(l_real - l_blank) > 1e-4, "blanking the image changes the loss",
          f"delta {abs(l_real - l_blank):.4e}")

    print("\n=== I14  DataLoader with worker processes ===")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate, shuffle=False)
    t0 = time.time()
    nb = 0
    for bb in dl:
        nb += 1
        if nb >= 3:
            break
    check(nb == 3, "DataLoader(num_workers=2) yields batches (per-worker zip handles work)",
          f"{nb} batches in {time.time()-t0:.1f}s")
    del b
    if dev == "cuda":
        torch.cuda.empty_cache()


# ======================================================================================

if __name__ == "__main__":
    want_model = "--model" in sys.argv

    SPEC = test_spec()
    test_normalization_matches_hf(SPEC)
    test_sar_channels(SPEC)
    test_composite_geometry(SPEC)
    test_fit_choice(SPEC)
    test_halves_differ(SPEC)
    CODEC = test_collator_matches_fase2()
    DS = test_dataset_and_routing(SPEC, CODEC)
    test_collate_batch(DS, CODEC, SPEC)
    test_integrity_scan()
    test_throughput(SPEC, CODEC)
    if want_model:
        test_model_forward(SPEC, CODEC)
    if "--dump" in sys.argv:
        dump_sample_composites(SPEC)

    n_fail = sum(1 for ok, _ in _results if not ok)
    print("\n" + "=" * 70)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed")
    for ok, label in _results:
        if not ok:
            print("  FAILED:", label)
    print("=" * 70)
    sys.exit(1 if n_fail else 0)
