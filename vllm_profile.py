"""Switch the shared WSL vLLM launcher between Sentrix and ReMem profiles."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path("/home/hpq/vllm-runtime")
ACTIVE = ROOT / "run-qwen3-vllm.sh"
SENTRIX = ROOT / "run-qwen3-vllm.sh.pre-remem-32f"
REMEM = ROOT / "run-qwen3-vllm.sh.remem-32f"
REMEM8 = ROOT / "run-qwen3-vllm.sh.remem-8b"
MODEL8 = ROOT / "models/Qwen3-VL-8B-Instruct"
MODEL8_SHARD_SIZES = (4902275944, 4915962496, 4999831048, 2716270024)
SERVER_EXE = str(ROOT / ".venv-vllm312/bin/vllm")


def prepare():
    original = SENTRIX.read_text(encoding="utf-8")
    old = '{"image":2}'
    if (
        original.count(old) != 1
        or original.count("--max-model-len 4096") != 1
        or original.count("--max-num-seqs 8") != 1
        or original.count("--gpu-memory-utilization 0.57") != 1
    ):
        raise RuntimeError("The saved Sentrix profile is different than expected")
    modified = (
        original.replace(old, '{"image":32}')
        .replace("--max-model-len 4096", "--max-model-len 32768")
        .replace("--max-num-seqs 8", "--max-num-seqs 1")
        .replace("--gpu-memory-utilization 0.57", "--gpu-memory-utilization 0.75")
    )
    marker = "  --enforce-eager \\"
    if modified.count(marker) != 1:
        raise RuntimeError("Cannot place ReMem's image processor settings")
    modified = modified.replace(
        marker,
        "  --mm-processor-kwargs '{\"min_pixels\":262144,\"max_pixels\":2097152}' \\\n" + marker,
    )
    modified32 = modified
    modified32 = modified32.replace(marker, "  --mm-processor-cache-gb 0 \\\n" + marker)
    modified32 = modified32.replace(
        "export PYTHONUNBUFFERED=1",
        "export PYTHONUNBUFFERED=1\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPYCACHEPREFIX=/tmp/remem-vllm-pycache\nrm -rf \"$PYTHONPYCACHEPREFIX\"\nmkdir -p \"$PYTHONPYCACHEPREFIX\"",
    )
    current = REMEM.read_text(encoding="utf-8") if REMEM.exists() else None
    if current != modified32:
        shutil.copy2(SENTRIX, REMEM)
        REMEM.write_text(modified32, encoding="utf-8")
    model4_path = str(ROOT / "models/Qwen3-VL-4B-Instruct")
    model8_path = str(MODEL8)
    if modified.count(model4_path) != 1 or modified.count("--load-format pt") != 1:
        raise RuntimeError("Cannot prepare the 8B profile from the saved Sentrix launcher")
    modified8 = (
        modified.replace(model4_path, model8_path)
        .replace("qwen3-vl-4b-instruct qwen3-vl-4b", "qwen3-vl-8b-instruct qwen3-vl-8b")
        .replace("--gpu-memory-utilization 0.75", "--gpu-memory-utilization 0.92")
        .replace("--load-format pt", "--load-format auto \\\n  --safetensors-load-strategy eager")
        # Keep the KV cache in the model's native dtype for the baseline.
        # FP8 KV cache and a forced attention backend were local performance
        # experiments, not part of the paper/lmms-eval configuration.
        .replace("  --enforce-eager \\", "  --enforce-eager \\")
    )
    modified8 = modified8.replace(
        "export PYTHONUNBUFFERED=1",
        "export PYTHONUNBUFFERED=1\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONPYCACHEPREFIX=/tmp/remem-vllm-pycache\nrm -rf \"$PYTHONPYCACHEPREFIX\"\nmkdir -p \"$PYTHONPYCACHEPREFIX\"",
    )
    current8 = REMEM8.read_text(encoding="utf-8") if REMEM8.exists() else None
    if current8 != modified8:
        REMEM8.write_text(modified8, encoding="utf-8")
    return original, modified32, modified8


def server_pids():
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode("utf-8").split("\0")
        except (OSError, UnicodeDecodeError):
            continue
        if len(args) > 2 and args[1] == SERVER_EXE and args[2] == "serve":
            pids.append(int(proc.name))
    return pids


def has_profile(pid, limit, model_path):
    try:
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode("utf-8")
    except OSError:
        return False
    return f'{{"image":{limit}}}' in cmdline and model_path in cmdline


def wait_for_server(expected_limit, expected_context, expected_model, model_path, launcher):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if launcher.poll() is not None:
            raise RuntimeError(f"vLLM launcher exited with code {launcher.returncode}; check {ROOT / 'logs/qwen3-vl-vllm.log'}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2) as response:
                models = json.load(response)["data"]
            pids = server_pids()
            if models and models[0]["max_model_len"] == expected_context and models[0]["id"] == expected_model and pids:
                if has_profile(pids[0], expected_limit, model_path):
                    print(f"vLLM ready: image limit {expected_limit}, context {expected_context}, PID {pids[0]}", flush=True)
                    return
        except (OSError, ValueError, KeyError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy; check {ROOT / 'logs/qwen3-vl-vllm.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("ReMem32", "ReMem8B", "Sentrix"))
    parser.add_argument("--prepare-only", action="store_true", help="Write profile scripts without touching the running service")
    args = parser.parse_args()
    original, modified, modified8 = prepare()
    if args.prepare_only:
        print(f"Prepared {args.profile} launcher; running vLLM was not changed", flush=True)
        return
    if args.profile == "ReMem8B":
        index = MODEL8 / "model.safetensors.index.json"
        if not index.is_file():
            raise RuntimeError(f"8B model is incomplete: missing {index}")
        for i, size in enumerate(MODEL8_SHARD_SIZES, 1):
            shard = MODEL8 / f"model-{i:05d}-of-00004.safetensors"
            if not shard.is_file() or shard.stat().st_size != size:
                raise RuntimeError(f"8B model is incomplete: {shard} should be {size} bytes")
    desired = {"Sentrix": original, "ReMem32": modified, "ReMem8B": modified8}[args.profile]
    # The archived Sentrix launcher predates the runtime pyc workaround.  A
    # corrupted bytecode cache can make the vLLM Python 3.12 process fail
    # before it binds port 8000, so apply the same clean-cache environment to
    # the Sentrix profile before launching it.
    if args.profile == "Sentrix":
        desired = desired.replace(
            "export PYTHONUNBUFFERED=1",
            "export PYTHONUNBUFFERED=1\n"
            "export PYTHONDONTWRITEBYTECODE=1\n"
            "export PYTHONPYCACHEPREFIX=/tmp/remem-vllm-pycache\n"
            "rm -rf \"$PYTHONPYCACHEPREFIX\"\n"
            "mkdir -p \"$PYTHONPYCACHEPREFIX\"",
        )
    expected_limit = 2 if args.profile == "Sentrix" else 32
    expected_context = 4096 if args.profile == "Sentrix" else 32768
    expected_model = "qwen3-vl-8b-instruct" if args.profile == "ReMem8B" else "qwen3-vl-4b-instruct"
    model_path = str(MODEL8 if args.profile == "ReMem8B" else ROOT / "models/Qwen3-VL-4B-Instruct")
    existing = server_pids()
    if ACTIVE.read_text(encoding="utf-8") == desired and existing and all(has_profile(pid, expected_limit, model_path) for pid in existing):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
                models = json.load(response)["data"]
            if any(model["id"] == expected_model and model["max_model_len"] == expected_context for model in models):
                print(f"{args.profile} profile already active; context {expected_context}", flush=True)
                return
        except (OSError, ValueError, KeyError, TimeoutError):
            pass

    # Write the selected profile text itself.  In particular, the Sentrix
    # profile may have the clean-cache workaround injected above; copying the
    # archived source here would silently discard that fix.
    ACTIVE.write_text(desired, encoding="utf-8")
    for pid in existing:
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not any(Path("/proc", str(pid)).exists() for pid in existing):
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Old vLLM process did not stop: {existing}")
    with open(os.devnull, "wb") as sink:
        launcher = subprocess.Popen(
            ["/bin/bash", str(ACTIVE)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=sink,
            start_new_session=True,
        )
    print(f"Starting {args.profile} profile with context {expected_context}...", flush=True)
    wait_for_server(expected_limit, expected_context, expected_model, model_path, launcher)


if __name__ == "__main__":
    main()
