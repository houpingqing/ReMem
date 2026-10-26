import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"D:\ReMem\models\Qwen3-8B-AWQ")
BASE = "https://hf-mirror.com/Qwen/Qwen3-8B-AWQ/resolve/main"
FILES = ["config.json", "generation_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
CHUNK = 8 * 1024 * 1024
ROOT.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 1:
    FILES = [sys.argv[1]]

for name in FILES:
    target = ROOT / name
    url = f"{BASE}/{name}"
    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("content-length", "0"))
    size = target.stat().st_size if target.exists() else 0
    if total and size == total:
        print(f"{name}: complete {total}", flush=True)
        continue
    if total and size > total:
        target.unlink()
        size = 0
    if not total:
        response = requests.get(url, timeout=(30, 180))
        response.raise_for_status()
        target.write_bytes(response.content)
        print(f"{name}: complete {target.stat().st_size}", flush=True)
        continue
    print(f"{name}: {size}/{total}", flush=True)
    with target.open("ab") as out:
        while size < total:
            end = min(size + CHUNK - 1, total - 1)
            for attempt in range(5):
                try:
                    response = requests.get(url, headers={"Range": f"bytes={size}-{end}"}, stream=True, timeout=(30, 180))
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
                    if attempt == 4:
                        raise
                    print(f"retry {name} at {size}: {exc}", flush=True)
                    time.sleep(2 * (attempt + 1))
    if target.stat().st_size != total:
        raise RuntimeError(f"{name} incomplete")
print("Qwen3-8B-AWQ download complete", flush=True)
