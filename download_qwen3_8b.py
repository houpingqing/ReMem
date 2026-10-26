import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"D:\ReMem\models\Qwen3-8B")
BASE = "https://hf-mirror.com/Qwen/Qwen3-8B/resolve/main"
FILES = [f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)]
if len(sys.argv) > 1:
    FILES = [sys.argv[1]]
CHUNK = 8 * 1024 * 1024

for name in FILES:
    target = ROOT / name
    size = target.stat().st_size if target.exists() else 0
    url = f"{BASE}/{name}"
    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("content-length", "0"))
    if size > total:
        target.unlink()
        size = 0
    if size == total and total:
        print(f"{name}: complete {total}", flush=True)
        continue
    print(f"{name}: {size}/{total}", flush=True)
    with target.open("ab") as out:
        while size < total:
            end = min(size + CHUNK - 1, total - 1)
            attempt = 0
            while True:
                try:
                    # A reset can happen after a partial body was written.
                    # Truncate back to the last verified byte before retrying.
                    out.seek(size)
                    out.truncate()
                    response = requests.get(
                        url,
                        headers={"Range": f"bytes={size}-{end}"},
                        stream=True,
                        timeout=(30, 180),
                    )
                    response.raise_for_status()
                    written = 0
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            out.write(block)
                            written += len(block)
                    expected = end - size + 1
                    if written != expected:
                        raise IOError(f"short range: {written} != {expected}")
                    size += written
                    out.flush()
                    os.fsync(out.fileno())
                    print(f"{name}: {size}/{total} ({size/total:.1%})", flush=True)
                    break
                except Exception as exc:
                    attempt += 1
                    delay = min(60, 2 * attempt)
                    print(f"retry {name} at {size} (attempt {attempt}): {exc}; sleeping {delay}s", flush=True)
                    time.sleep(delay)
    if target.stat().st_size != total:
        raise RuntimeError(f"{name} incomplete")
print("Qwen3-8B download complete", flush=True)
