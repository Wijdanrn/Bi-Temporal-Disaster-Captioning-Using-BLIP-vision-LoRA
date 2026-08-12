"""
IndoBLIP-Post-Disaster -- Gradio demo.

Upload a pre-disaster and a post-disaster satellite image (or click one of the real
test-set example pairs), and this app runs the ACTUAL trained pipeline:

    pre + post image
        -> scripts.dataset.make_composite() / ImageSpec   (same LAYOUT="horizontal",
           FIT="stretch" composite used for training -- see docs/design_decisions.md SS3.1-3.2)
        -> checkpoints/full_run_v3_vision/best             (final model: text LoRA +
           vision qkv LoRA + retrained Indonesian embeddings -- docs/design_decisions.md
           Fase 5)
        -> greedy decoding (num_beams=1, max_new_tokens=300 -- identical settings used to
           produce results/predictions_test_vision.jsonl, so what you see here is
           reproducible against the reported numbers, not a different decode config)
        -> Indonesian 7-section structured damage report

Optionally (checkbox), it also renders the cross-attention rollout heatmap using the
already-verified engine in scripts/xai_attention_rollout.py (XaiEngine.explain +
_upsample). Per docs/design_decisions.md Fase 6 SS6.4/6.6/6.7/6.8, this heatmap is a real,
image-dependent saliency signal, but it is NOT a per-word explanation (token selectivity
r >= 0.999 across 150 test examples -- every generated word gets almost the same picture)
and it did NOT beat a random patch ranking on a causal deletion/insertion faithfulness
test. The UI shows this caveat next to the heatmap; it must not be read as "what the model
looked at to write this word".

Run with the torch5050 conda env:

    "C:\\Users\\Wijdan\\anaconda3\\envs\\torch5050\\python.exe" demo\\app.py

Then open the printed local URL (default http://127.0.0.1:7860).
"""
from __future__ import annotations

import io
import os
import sys

# Must be set before any HuggingFace/transformers import -- Xet backend hangs on this
# machine, and the base model + tokenizer are already fully cached locally (offline mode
# avoids a network/SSL round-trip that has also been flaky here).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# This machine hangs on outbound HTTPS calls it doesn't strictly need (same class of issue
# as the Xet/SSL hang HF_HUB_OFFLINE works around above) -- disable Gradio's telemetry and
# update check so `demo.launch()` cannot stall waiting on a network call.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

import gradio as gr

from scripts.build_model import SECTION_HEADERS  # noqa: E402
from scripts.dataset import make_composite  # noqa: E402
from scripts.xai_attention_rollout import XaiEngine, _upsample  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints", "full_run_v3_vision", "best")
EXAMPLES_DIR = os.path.join(ROOT, "demo", "examples")
MAX_NEW_TOKENS = 300  # same as results/predictions_test_vision.jsonl (SS4.3 / SS5.6)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEATMAP_CAVEAT = (
    "**Caveat (measured, docs/design_decisions.md Fase 6):** this heatmap is a real, "
    "image-dependent saliency signal (cross-attention rollout), but it is **not** a "
    "per-word or per-section explanation -- every generated word gets almost the same "
    "picture (token-selectivity r >= 0.999 across 150 test examples), and this attention "
    "ranking did **not** beat a random patch ranking on a causal deletion/insertion "
    "faithfulness test. Read it as \"where the model's cross-attention mass sits, "
    "on average, for this whole report\" -- not as evidence of what the model looked at "
    "to produce any specific word."
)

print(f"[demo] loading final checkpoint on {DEVICE}: {CKPT_DIR}")
ENGINE = XaiEngine(device=DEVICE, verbose=True)
print("[demo] model loaded, ready.")


def _to_pil_rgb(img) -> Image.Image:
    if img is None:
        raise gr.Error("Please provide both a pre-disaster and a post-disaster image.")
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def format_report_markdown(text: str) -> str:
    """Render the decoded report with bold section headers, one blank line per section."""
    out = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        for h in SECTION_HEADERS:
            prefix = f"{h}:"
            if line.startswith(prefix):
                body = line[len(prefix):].strip()
                out.append(f"### {h}\n{body}")
                matched = True
                break
        if not matched:
            # continuation of the previous section (report wrapped onto a new line)
            if out:
                out[-1] = out[-1] + " " + line
            else:
                out.append(line)
    return "\n\n".join(out) if out else "_(empty generation)_"


def render_heatmap(composite_pil: Image.Image, map576: torch.Tensor) -> Image.Image:
    """Overlay a (576,) rollout map (mean over all generated content tokens) on the composite."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_rgb = np.asarray(composite_pil.resize((384, 384)))
    heat = _upsample(map576)
    heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=130)
    ax.imshow(base_rgb)
    ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    ax.axvline(192, color="white", lw=1.5, ls="--")  # pre|post patch boundary
    ax.set_title("Cross-attention rollout, averaged over the whole generated report "
                 "(pre | post boundary shown dashed)", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def run_pipeline(pre_img, post_img, show_heatmap: bool):
    pre = _to_pil_rgb(pre_img)
    post = _to_pil_rgb(post_img)

    composite = make_composite(pre, post, ENGINE.spec)  # LAYOUT="horizontal", FIT="stretch"
    pixel_values = ENGINE.spec.to_tensor(composite).unsqueeze(0).to(ENGINE.device)

    if show_heatmap:
        tm = ENGINE.explain(pixel_values, variants=("rollout",))
        caption = tm.caption
        content_idx = [i for i, c in enumerate(tm.is_content) if c]
        if content_idx:
            mean_map = tm.maps["rollout"][content_idx].mean(0)
            heatmap_img = render_heatmap(composite, mean_map)
        else:
            heatmap_img = None
    else:
        out_ids = ENGINE.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
        caption = ENGINE.codec.decode(out_ids[0])
        heatmap_img = None

    report_md = format_report_markdown(caption)
    return report_md, composite, heatmap_img


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="IndoBLIP-Post-Disaster") as demo:
        gr.Markdown(
            "# IndoBLIP-Post-Disaster\n"
            "Upload a **pre-disaster** and a **post-disaster** satellite image of the same "
            "location (or click a real test-set example below) to generate a structured "
            "Indonesian damage report with the fine-tuned model "
            "(`checkpoints/full_run_v3_vision/best`: BLIP + text/vision LoRA + retrained "
            "Indonesian vocabulary, greedy decoding). See `docs/design_decisions.md` for the "
            "full methodology and `results/metrics_table.md` for real evaluation numbers."
        )
        with gr.Row():
            pre_input = gr.Image(label="Pre-disaster image", type="pil")
            post_input = gr.Image(label="Post-disaster image", type="pil")

        heatmap_checkbox = gr.Checkbox(
            label="Also show cross-attention rollout heatmap (slower, ~7-10s extra; see caveat below)",
            value=False,
        )
        run_btn = gr.Button("Generate report", variant="primary")

        with gr.Row():
            composite_out = gr.Image(label="Composite image fed to the model (pre | post)")
            report_out = gr.Markdown(label="Generated report (Indonesian)")
        heatmap_out = gr.Image(label="Cross-attention rollout heatmap (optional)", visible=True)
        gr.Markdown(HEATMAP_CAVEAT)

        run_btn.click(
            fn=run_pipeline,
            inputs=[pre_input, post_input, heatmap_checkbox],
            outputs=[report_out, composite_out, heatmap_out],
        )

        example_rows = []
        manifest_path = os.path.join(EXAMPLES_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            import json
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            for rec in manifest:
                pre_p = os.path.join(ROOT, rec["pre"])
                post_p = os.path.join(ROOT, rec["post"])
                if os.path.exists(pre_p) and os.path.exists(post_p):
                    example_rows.append([pre_p, post_p, False])
        if example_rows:
            gr.Markdown(
                "### Real test-set example pairs (click to load, then press Generate report)"
            )
            gr.Examples(
                examples=example_rows,
                inputs=[pre_input, post_input, heatmap_checkbox],
                label="Examples from data/processed/captions_test.jsonl",
            )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
