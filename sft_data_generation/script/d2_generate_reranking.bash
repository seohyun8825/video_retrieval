#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_DIR="${ROOT_DIR}/sft_data_generation"
PREPROCESS_DIR="${SFT_DIR}/preprocess/extract_similarity_matrix"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "video-colbert" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate video-colbert
fi
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
D1_DIR="${DATA_DIR}/d1"

if [[ ! -d "${D1_DIR}" ]]; then
  echo "❌ d1 directory not found: ${D1_DIR}" >&2
  exit 1
fi

if [[ -z "${INPUT_JSON:-}" ]]; then
  mapfile -t __global_files < <(find "${D1_DIR}" -maxdepth 1 -type f -name "*.json" | sort)
  if [[ ${#__global_files[@]} -eq 0 ]]; then
    echo "❌ No global JSON files found in ${D1_DIR}" >&2
    exit 1
  elif [[ ${#__global_files[@]} -gt 1 ]]; then
    echo "❌ Multiple global JSON files found in ${D1_DIR}. Set INPUT_JSON explicitly." >&2
    printf '  %s\n' "${__global_files[@]}"
    exit 1
  fi
  INPUT_JSON="${__global_files[0]}"
fi

VIDEO_BASE="${VIDEO_BASE:-/hub_data1/seohyun/hub_data}"
INPUT_NAME="$(basename "${INPUT_JSON}" .json)"
mkdir -p "${DATA_DIR}/d2_reranked"
DEFAULT_OUTPUT_DIR="${DATA_DIR}/d2_reranked/${INPUT_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

echo "Input JSON     : ${INPUT_JSON}"
echo "Video base path: ${VIDEO_BASE}"
echo "Output dir     : ${OUTPUT_DIR}"

python "${PREPROCESS_DIR}/video_colbert_generate_reranking.py" \
  --input_json "${INPUT_JSON}" \
  --video_base "${VIDEO_BASE}" \
  --output_dir "${OUTPUT_DIR}"

echo "Generated files:"
ls -1 "${OUTPUT_DIR}"/reranking_train_*.json
