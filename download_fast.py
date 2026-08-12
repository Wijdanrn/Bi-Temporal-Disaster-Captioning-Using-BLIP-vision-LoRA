import os
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import get_token, snapshot_download
from tqdm import tqdm

REPO_ID = "Kingdrone-Junjue/DisasterM3"
REVISION = "main"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Large single-file archives: a lone connection here caps out around
# ~1 MB/s (HF's CDN throttles per-connection), so these get split into
# many parallel byte-range requests instead of one straight-line stream.
#
# Each range is downloaded into its own small part file rather than
# seeking into one pre-sized destination file -- pre-truncating a
# multi-GB file forces Windows to synchronously zero-fill the new
# region (no elevated privilege here to skip that), which blocks for
# minutes before any real download work starts. Per-chunk files also
# give free resume tracking: a part's size on disk *is* its completion
# state, no separate bookkeeping needed.
BIG_FILES = [
    "DisasterM3_Instruct/train_images.zip",
    "DisasterM3_Instruct/box_train_images.zip",
    "DisasterM3_Instruct/masks.zip",
]

CHUNKS_PER_FILE = 20  # empirically ~16 MB/s aggregate here; diminishing returns past ~24
MIN_CHUNK_SIZE = 16 * 1024 * 1024
CHUNK_RETRIES = 4


def log(msg):
    print(msg, flush=True)


def headers():
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def resolve_url(rel_path):
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{rel_path}"


def download_chunk(url, start, end, part_path, pbar, lock, session):
    expected = end - start + 1
    last_err = None
    for attempt in range(1, CHUNK_RETRIES + 1):
        written = 0
        try:
            h = {**headers(), "Range": f"bytes={start}-{end}"}
            with session.get(url, headers=h, stream=True, timeout=(15, 60)) as r:
                r.raise_for_status()
                tmp_path = part_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        written += len(chunk)
                        with lock:
                            pbar.update(len(chunk))
            if written != expected:
                raise IOError(f"got {written} bytes, expected {expected}")
            os.replace(tmp_path, part_path)
            return
        except (requests.RequestException, OSError) as e:
            last_err = e
            with lock:
                pbar.update(-written)
            log(f"  [retry {attempt}/{CHUNK_RETRIES}] range {start}-{end} failed: {e}")
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"range {start}-{end} failed after {CHUNK_RETRIES} attempts: {last_err}")


def download_big_file(rel_path):
    url = resolve_url(rel_path)
    dest = os.path.join(LOCAL_DIR, rel_path)
    parts_dir = dest + ".parts"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    resp = requests.head(url, headers=headers(), allow_redirects=True, timeout=30)
    resp.raise_for_status()
    total_size = int(resp.headers.get("Content-Length", 0))
    accepts_ranges = resp.headers.get("Accept-Ranges") == "bytes"

    if os.path.exists(dest) and os.path.getsize(dest) == total_size and total_size > 0 \
            and not os.path.isdir(parts_dir):
        log(f"[skip] {rel_path} already complete ({total_size / 1e9:.2f} GB)")
        return

    if not accepts_ranges or total_size < MIN_CHUNK_SIZE:
        log(f"[single-stream] {rel_path} ({total_size / 1e9:.2f} GB)")
        with requests.get(url, headers=headers(), stream=True, timeout=(15, 60)) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        return

    os.makedirs(parts_dir, exist_ok=True)

    n_chunks = min(CHUNKS_PER_FILE, max(1, total_size // MIN_CHUNK_SIZE))
    chunk_size = -(-total_size // n_chunks)  # ceil div
    ranges = [
        (i, start, min(start + chunk_size - 1, total_size - 1))
        for i, start in enumerate(range(0, total_size, chunk_size))
    ]

    def part_path(i):
        return os.path.join(parts_dir, f"{i:04d}.part")

    already_done_bytes = 0
    todo = []
    for i, s, e in ranges:
        p = part_path(i)
        expected = e - s + 1
        if os.path.exists(p) and os.path.getsize(p) == expected:
            already_done_bytes += expected
        else:
            todo.append((i, s, e))

    log(f"[start] {rel_path}: {total_size / 1e9:.2f} GB, {len(todo)}/{len(ranges)} "
        f"chunks remaining ({len(ranges)} total connections)")

    lock = threading.Lock()
    t0 = time.time()
    session = requests.Session()

    with tqdm(
        total=total_size,
        initial=already_done_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=os.path.basename(rel_path),
    ) as pbar:
        if todo:
            with ThreadPoolExecutor(max_workers=len(todo)) as ex:
                futures = [
                    ex.submit(download_chunk, url, s, e, part_path(i), pbar, lock, session)
                    for i, s, e in todo
                ]
                for fut in as_completed(futures):
                    fut.result()  # re-raise immediately on chunk failure

        pbar.set_description(f"{os.path.basename(rel_path)} (assembling)")
        tmp_dest = dest + ".assembling"
        with open(tmp_dest, "wb") as out:
            for i, _s, _e in ranges:
                p = part_path(i)
                with open(p, "rb") as pf:
                    shutil.copyfileobj(pf, out, length=8 * 1024 * 1024)
                os.remove(p)
        os.replace(tmp_dest, dest)
        os.rmdir(parts_dir)

    elapsed = time.time() - t0
    avg = (total_size - already_done_bytes) / elapsed / 1e6 if elapsed > 0 else 0
    log(f"[done] {rel_path} in {elapsed:.0f}s ({avg:.1f} MB/s avg this run)")


if __name__ == "__main__":
    log("== Step 1: remaining small/medium files via snapshot_download ==")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        max_workers=8,
        ignore_patterns=BIG_FILES,
    )

    log("== Step 2: big archives via parallel range downloads ==")
    for rel in BIG_FILES:
        download_big_file(rel)

    log("DONE")
