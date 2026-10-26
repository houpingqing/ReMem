#!/usr/bin/env bash
set -euo pipefail

# Linux/server counterpart of eval/videomme_qwen3_vl.sh. The Qwen3-VL
# lmms_eval adapter uses Accelerate data parallelism: one model copy per GPU.
REPO_ROOT="${REMEM_REPO_ROOT:-/mnt/d/hpq/ReMem}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/mnt/d/hpq/lmms-eval}"
MODEL_PATH="${REMEM_VL_MODEL:-/mnt/d/hpq/models/Qwen3-VL-8B-Instruct}"
VIDEO_ROOT="${REMEM_VIDEO_ROOT:-/mnt/d/hpq/Dataset/VideoMME/videos}"
OUTPUT_PATH="${REMEM_OUTPUT_PATH:-${REPO_ROOT}/VideoMME/paper_lmms_qwen3vl8b_qwen3parser}"
MAIN_PORT="${LMMS_MAIN_PROCESS_PORT:-12345}"

export PYTHONPATH="${LMMS_EVAL_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export LMMS_EVAL_ROOT
export REMEM_VIDEO_ROOT="${VIDEO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

cd "${LMMS_EVAL_ROOT}"
exec accelerate launch \
  --num_processes 8 \
  --main_process_port "${MAIN_PORT}" \
  -m lmms_eval eval \
  --include_path "${REPO_ROOT}/lmms_tasks" \
  --tasks remem_videomme_qwen3_8b_server \
  --model qwen3_vl \
  --model_args "pretrained=${MODEL_PATH},device_map=auto,enable_thinking=False,max_num_frames=32,min_pixels=200704,max_pixels=200704,attn_implementation=sdpa" \
  --batch_size 1 \
  --verbosity INFO \
  --output_path "${OUTPUT_PATH}" \
  --log_samples
