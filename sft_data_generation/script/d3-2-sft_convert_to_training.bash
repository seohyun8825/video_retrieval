#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_DIR="${ROOT_DIR}/sft_data_generation"
CONVERT_SCRIPT="${SFT_DIR}/preprocess/sft_data_generate/convert_sft_to_llamafactory.py"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
if [[ -z "${SFT_JSON:-}" ]]; then
  mapfile -t __kept_files < <(find "${DATA_DIR}/d3" -maxdepth 1 -type f -name "*_sft_training_data_kept.json" | sort)
  if [[ ${#__kept_files[@]} -eq 0 ]]; then
    echo "❌ No *_sft_training_data_kept.json found in ${DATA_DIR}/d3. Set SFT_JSON explicitly." >&2
    exit 1
  fi
  SFT_JSON="${__kept_files[0]}"
fi

if [[ -z "${BASE_NAME:-}" ]]; then
  basefile="$(basename "${SFT_JSON}" .json)"
  BASE_NAME="${basefile%_sft_training_data_kept}"
fi

VIDEO_PREFIX="${VIDEO_PREFIX:-activitynet/videos}"
OUTPUT_PATH="${OUTPUT_PATH:-${DATA_DIR}/d3/${BASE_NAME}_sft_llamafactory.json}"

if [[ ! -f "${SFT_JSON}" ]]; then
  echo "❌ SFT JSON not found: ${SFT_JSON}" >&2
  exit 1
fi

python "${CONVERT_SCRIPT}" \
  --sft_json "${SFT_JSON}" \
  --video_prefix "${VIDEO_PREFIX}" \
  --output_path "${OUTPUT_PATH}"
