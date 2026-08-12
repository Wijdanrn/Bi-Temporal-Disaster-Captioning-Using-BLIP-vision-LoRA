import os

# Xet's per-file token-refresh calls have no retry/backoff and trip HF's
# rate limit fast; plain HTTP downloads do retry on 429, so disable Xet.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

REPO_ID = "Kingdrone-Junjue/DisasterM3"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        max_workers=2,  # keep low to avoid HF's 1000 req/5min rate limit
    )
    print("DONE:", path)
