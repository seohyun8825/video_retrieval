#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "video-colbert" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate video-colbert
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo " DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 1
fi

if [[ -z "${INPUT_JSON:-}" ]]; then
  mapfile -t __json_files < <(find "${DATA_DIR}" -maxdepth 1 -type f -name "*.json" | sort)
  if [[ ${#__json_files[@]} -eq 0 ]]; then
    echo "No .json files found in ${DATA_DIR}" >&2
    exit 1
  elif [[ ${#__json_files[@]} -gt 1 ]]; then
    echo "Multiple .json files found in ${DATA_DIR}. Set INPUT_JSON explicitly." >&2
    printf '  %s\n' "${__json_files[@]}"
    exit 1
  fi
  INPUT_JSON="${__json_files[0]}"
fi

INPUT_BASENAME="$(basename "${INPUT_JSON}" .json)"
mkdir -p "${DATA_DIR}/d1"
DEFAULT_OUTPUT_JSON="${DATA_DIR}/d1/${INPUT_BASENAME}_global.json"
OUTPUT_JSON="${OUTPUT_JSON:-${DEFAULT_OUTPUT_JSON}}"
PROMPT_FILE="${REPO_ROOT}/sft_data_generation/prompt/global_caption_generate.txt"
API_KEY_PATH="${REPO_ROOT}/openai"
MODEL_NAME="gpt-4.1"
CHUNK_SIZE=1

# empty or 0 = all entries
LIMIT_COUNT=0

# Debugging용
#RAW_OUTPUT_DIR="${REPO_ROOT}/data/batch_outputs"
RAW_OUTPUT_DIR=


BATCH_API="false"

# 병렬처리
MAX_WORKERS=50

CMD=(
  python3 "${REPO_ROOT}/sft_data_generation/preprocess/generate_global_captions_batch.py"
  --input_json "${INPUT_JSON}"
  --output_json "${OUTPUT_JSON}"
  --prompt_file "${PROMPT_FILE}"
  --api_key_path "${API_KEY_PATH}"
  --model "${MODEL_NAME}"
  --chunk_size "${CHUNK_SIZE}"
  --max_workers "${MAX_WORKERS}"
)

if [[ -n "${LIMIT_COUNT}" && "${LIMIT_COUNT}" -gt 0 ]]; then
  CMD+=(--limit "${LIMIT_COUNT}")
fi

if [[ -n "${RAW_OUTPUT_DIR}" ]]; then
  CMD+=(--raw_output_dir "${RAW_OUTPUT_DIR}")
fi

if [[ "${BATCH_API,,}" == "false" ]]; then
  CMD+=(--no-batch-api)
fi

CMD+=("$@")

"${CMD[@]}"
