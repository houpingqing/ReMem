#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REMEM_REPO_ROOT:-/mnt/d/hpq/ReMem}"
MODEL_PATH="${REMEM_TEXT_MODEL:-/mnt/d/hpq/models/Qwen3-8B}"
INPUT_JSON="${1:?usage: remem_qwen3_text_parser.sh INPUT_JSON OUTPUT_JSON}"
OUTPUT_JSON="${2:?usage: remem_qwen3_text_parser.sh INPUT_JSON OUTPUT_JSON}"
PARSER_GPU="${PARSER_GPU:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/remem-pycache}"
export PYTHONDONTWRITEBYTECODE=1

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${PARSER_GPU}" python prepare_annotations_hf.py \
  "${INPUT_JSON}" \
  "${OUTPUT_JSON}" \
  --model "${MODEL_PATH}"
