#!/usr/bin/env bash
set -euo pipefail

# One-command server pipeline. The final answering stage is the paper
# launcher (accelerate, 8 processes); parser and frame selection run first and
# never overlap with the answering stage.
REPO_ROOT="${REMEM_REPO_ROOT:-/mnt/d/hpq/ReMem}"
MODEL_ROOT="${REMEM_MODEL_ROOT:-/mnt/d/hpq/models}"
DATA_ROOT="${REMEM_DATA_ROOT:-/mnt/d/hpq/Dataset/VideoMME}"
LMMS_ROOT="${LMMS_EVAL_ROOT:-/mnt/d/hpq/lmms-eval}"
PYTHON="${REMEM_PYTHON:-${CONDA_PREFIX:-}/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python)"
fi

RAW_JSON="${REPO_ROOT}/VideoMME/val_qa_raw_full.json"
PARSED_JSON="${REPO_ROOT}/VideoMME/val_qa_full_qwen3_8b_hf.json"
SELECTED_JSON="${REPO_ROOT}/VideoMME/val_qa_selected_full_32f_qwen3_8b_hf.json"
TASK_JSONL="${REPO_ROOT}/lmms_tasks/remem_videomme_qwen3_8b_qwen3parser.jsonl"
PARQUET="${DATA_ROOT}/test-00000-of-00001.parquet"
VIDEO_ROOT="${DATA_ROOT}/videos"
CLIP_ROOT="${MODEL_ROOT}/clip-vit-large-patch14-336"
TEXT_MODEL="${MODEL_ROOT}/Qwen3-8B"
VL_MODEL="${MODEL_ROOT}/Qwen3-VL-8B-Instruct"
LOCK_FILE="${REPO_ROOT}/.server_eval.lock"

exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "Another ReMem server pipeline is already running."; exit 2; }

for pattern in 'prepare_annotations_hf.py' 'frame_select.py' 'lmms_eval eval'; do
  if pgrep -af "${pattern}" | grep -vE 'pgrep|run_server_qwen3_8b_eval' >/dev/null; then
    echo "A ReMem stage is already running (${pattern}); refusing to start a duplicate." >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/remem-pycache}"
export PYTHONDONTWRITEBYTECODE=1
export REMEM_VIDEO_DECODER=ffmpeg
export PYTHONPATH="${LMMS_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
mkdir -p VideoMME

for path in "${RAW_JSON}" "${PARQUET}" "${VIDEO_ROOT}" "${TEXT_MODEL}" "${VL_MODEL}" "${CLIP_ROOT}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done

count_json() {
  "${PYTHON}" - "$1" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
except Exception:
    print(0)
PY
}

if [[ "$(count_json "${PARSED_JSON}")" -lt 2700 ]]; then
  echo "[1/4] Qwen3-8B question parsing"
  bash eval/remem_qwen3_text_parser.sh "${RAW_JSON}" "${PARSED_JSON}"
else
  echo "[1/4] parser output already complete"
fi

if [[ "$(count_json "${SELECTED_JSON}")" -lt 2700 ]]; then
  echo "[2/4] CLIP + FFmpeg 32-frame selection"
  "${PYTHON}" frame_select.py \
    --labels "${PARSED_JSON}" \
    --video-root "${VIDEO_ROOT}" \
    --output "${SELECTED_JSON}" \
    --clip-model "${CLIP_ROOT}" \
    --num-frames 32 \
    --device cuda
else
  echo "[2/4] selected-frame output already complete"
fi

if [[ ! -f "${TASK_JSONL}" ]] || [[ "$(wc -l < "${TASK_JSONL}")" -ne 2700 ]]; then
  echo "[3/4] preparing lmms_eval JSONL"
  "${PYTHON}" prepare_lmms_videomme.py \
    --selected "${SELECTED_JSON}" \
    --parquet "${PARQUET}" \
    --output "${TASK_JSONL}"
fi
sed -i 's|remem_videomme_qwen3_8b.jsonl|remem_videomme_qwen3_8b_qwen3parser.jsonl|' \
  lmms_tasks/remem_videomme_qwen3_8b_server.yaml

echo "[4/4] paper-style 8-GPU Qwen3-VL evaluation"
export REMEM_REPO_ROOT="${REPO_ROOT}"
export LMMS_EVAL_ROOT="${LMMS_ROOT}"
export REMEM_VL_MODEL="${VL_MODEL}"
export REMEM_VIDEO_ROOT="${VIDEO_ROOT}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export REMEM_NUM_PROCESSES=4
exec bash eval/remem_qwen3_vl_8gpu.sh
