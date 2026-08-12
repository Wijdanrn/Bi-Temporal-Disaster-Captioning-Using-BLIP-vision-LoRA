import os
import subprocess
import sys
import time

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_SCRIPT = os.path.join(LOCAL_DIR, "download_disasterm3.py")

SPEED_THRESHOLD_MB_S = 10.0
CHECK_INTERVAL_SEC = 90
CONSECUTIVE_SLOW_CHECKS = 2  # require sustained slowness before restarting


def dir_size_bytes(path):
    # Must include .cache/huggingface/download so in-progress .incomplete
    # files count toward growth -- otherwise a big file's speed reads 0
    # until it finishes, then spikes, which would false-trigger restarts.
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def run_once():
    proc = subprocess.Popen([sys.executable, DOWNLOAD_SCRIPT])
    try:
        time.sleep(CHECK_INTERVAL_SEC)  # let the initial speed burst ramp up

        last_size = dir_size_bytes(LOCAL_DIR)
        last_time = time.time()
        slow_count = 0

        while proc.poll() is None:
            time.sleep(CHECK_INTERVAL_SEC)

            now = time.time()
            size = dir_size_bytes(LOCAL_DIR)
            speed_mb_s = (size - last_size) / (now - last_time) / (1024 * 1024)
            last_size, last_time = size, now

            slow_count = slow_count + 1 if speed_mb_s < SPEED_THRESHOLD_MB_S else 0
            print(f"[watchdog] speed: {speed_mb_s:.2f} MB/s (slow_count={slow_count})")

            if slow_count >= CONSECUTIVE_SLOW_CHECKS:
                print(f"[watchdog] below {SPEED_THRESHOLD_MB_S} MB/s for "
                      f"{CONSECUTIVE_SLOW_CHECKS} checks -> restarting download")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False

        return proc.returncode == 0
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


if __name__ == "__main__":
    while True:
        if run_once():
            print("[watchdog] download finished")
            break
        print("[watchdog] restarting download...")
        time.sleep(3)
