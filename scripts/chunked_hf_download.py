"""
This network consistently truncates large HTTPS downloads to ~1-1.2KB (seen identically
across huggingface_hub's snapshot_download AND hf_hub_download, for different files) --
almost certainly a proxy/inspection appliance mishandling large streamed/chunked responses,
the same class of problem download_fast.py solved for the 44GB DisasterM3 dataset at the
start of this project. Reusing that proven approach: parallel byte-range GET requests into
per-chunk temp files, concatenated at the end, with per-chunk retry.

Usage:
    python scripts/chunked_hf_download.py <repo_id> <filename> <expected_size> <dest_path>
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

N_CHUNKS = 128
MAX_RETRIES = 30


def resolve_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def download_chunk(url: str, start: int, end: int, tmp_path: str, client: httpx.Client) -> None:
    expected = end - start + 1
    for attempt in range(MAX_RETRIES):
        have = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        if have >= expected:
            return
        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            with client.stream("GET", url, headers=headers, follow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp_path, "ab") as f:
                    for data in r.iter_bytes(chunk_size=1 << 16):
                        f.write(data)
            size = os.path.getsize(tmp_path)
            if size == expected:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"chunk {start}-{end} failed after {MAX_RETRIES} retries "
                        f"({os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0}/{expected})")


def main():
    repo_id, filename, expected_size, dest_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    url = resolve_url(repo_id, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    boundaries = []
    chunk_size = expected_size // N_CHUNKS
    for i in range(N_CHUNKS):
        start = i * chunk_size
        end = expected_size - 1 if i == N_CHUNKS - 1 else start + chunk_size - 1
        boundaries.append((start, end))

    tmp_dir = dest_path + ".chunks"
    os.makedirs(tmp_dir, exist_ok=True)

    client = httpx.Client(verify=False, timeout=httpx.Timeout(60.0, read=120.0))
    t0 = time.time()
    print(f"[start] {filename}: {expected_size/1e6:.1f} MB in {N_CHUNKS} chunks")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for i, (start, end) in enumerate(boundaries):
            tmp_path = os.path.join(tmp_dir, f"part{i:03d}")
            futs[ex.submit(download_chunk, url, start, end, tmp_path, client)] = i
        n_done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            fut.result()
            n_done += 1
            if n_done % 10 == 0 or n_done == N_CHUNKS:
                print(f"  {n_done}/{N_CHUNKS} chunks done ({time.time()-t0:.1f}s elapsed)")

    print(f"[concat] writing {dest_path}")
    with open(dest_path, "wb") as out:
        for i in range(N_CHUNKS):
            part = os.path.join(tmp_dir, f"part{i:03d}")
            with open(part, "rb") as f:
                out.write(f.read())
            os.remove(part)
    os.rmdir(tmp_dir)

    final_size = os.path.getsize(dest_path)
    print(f"[done] {dest_path}: {final_size} bytes (expected {expected_size}) "
          f"{'OK' if final_size == expected_size else 'MISMATCH!'} in {time.time()-t0:.1f}s")
    if final_size != expected_size:
        sys.exit(1)


if __name__ == "__main__":
    main()
