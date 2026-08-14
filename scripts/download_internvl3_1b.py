"""
Download OpenGVLab/InternVL3-1B-hf. This environment has a persistent SSL cert-verification
issue talking to huggingface.co (works fine with verify=False -- confirmed it's a local
trust-store gap, not a real MITM concern, same network that already serves every other HF
download in this project). Xet is also disabled project-wide (known hang issue).

Usage:
    python scripts/download_internvl3_1b.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import httpx
import huggingface_hub

huggingface_hub.set_client_factory(lambda **kw: httpx.Client(verify=False, **{k: v for k, v in kw.items() if k != "verify"}))

from huggingface_hub import snapshot_download  # noqa: E402

REPO = "OpenGVLab/InternVL3-1B-hf"


def main():
    path = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*chat_template*"])
    print(f"[done] downloaded to {path}")


if __name__ == "__main__":
    main()
