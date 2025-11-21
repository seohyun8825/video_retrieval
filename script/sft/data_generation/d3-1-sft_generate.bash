#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_DIR="${ROOT_DIR}/SFT/sft_data_generation"
PREPROCESS_DIR="${SFT_DIR}/preprocess/sft_data_generate"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "video-colbert" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate video-colbert
fi
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
RERANK_DIR="${DATA_DIR}/d2_reranked"

if [[ ! -d "${RERANK_DIR}" ]]; then
  echo "❌ d2 reranked directory not found: ${RERANK_DIR}" >&2
  exit 1
fi

if [[ -z "${RERANK_JSON:-}" ]]; then
  mapfile -t __rerank_jsons < <(find "${RERANK_DIR}" -maxdepth 2 -type f -name "reranking_train_hard.json" | sort)
  if [[ ${#__rerank_jsons[@]} -eq 0 ]]; then
    echo "❌ No reranking_train_hard.json found under ${RERANK_DIR}. Set RERANK_JSON explicitly." >&2
    exit 1
  fi
  RERANK_JSON="${__rerank_jsons[0]}"
fi

if [[ -z "${BASE_NAME:-}" ]]; then
  BASE_NAME="$(basename "$(dirname "${RERANK_JSON}")")"
  BASE_NAME="${BASE_NAME%_global}"
  BASE_NAME="${BASE_NAME%_reranked}"
fi
VIDEO_BASE="${VIDEO_BASE:-/hub_data1/seohyun/hub_data}"
PROMPT_FILE="${SFT_DIR}/prompt/sft_ranking_generate.txt"
API_KEY_FILE="${ROOT_DIR}/openai"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
NUM_FRAMES="${NUM_FRAMES:-12}"
MODEL_NAME="${MODEL_NAME:-gpt-4o}"
SAVE_FRAMES="${SAVE_FRAMES:-false}"
VERBOSE="${VERBOSE:-true}"
DEBUG_DETAIL="${DEBUG_DETAIL:-true}"
MAX_WORKERS="${MAX_WORKERS:-10}"
USE_BATCH_API="${USE_BATCH_API:-true}"

mkdir -p "${DATA_DIR}/d3"
OUTPUT_DIR="${DATA_DIR}/d3"

CMD=(
  python "${PREPROCESS_DIR}/generate_sft_ranking_data.py"
  --reranking_json "${RERANK_JSON}"
  --video_base "${VIDEO_BASE}"
  --output_dir "${OUTPUT_DIR}"
  --prompt_file "${PROMPT_FILE}"
  --api_key_path "${API_KEY_FILE}"
  --model "${MODEL_NAME}"
  --num_samples "${NUM_SAMPLES}"
  --num_frames "${NUM_FRAMES}"
  --base_name "${BASE_NAME}"
  --max_workers "${MAX_WORKERS}"
)

if [[ "${SAVE_FRAMES,,}" == "true" ]]; then
  CMD+=(--save_frames)
fi

if [[ "${VERBOSE,,}" == "true" ]]; then
  CMD+=(--verbose)
fi

if [[ "${DEBUG_DETAIL,,}" == "true" ]]; then
  CMD+=(--debug)
fi

if [[ "${USE_BATCH_API,,}" == "true" ]]; then
  CMD+=(--use_batch_api)
fi

"${CMD[@]}"
