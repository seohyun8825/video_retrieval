#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_DIR="${ROOT_DIR}/SFT/sft_data_generation"
PREPROCESS_DIR="${SFT_DIR}/preprocess/extract_similarity_matrix"

# Use current env; if you keep RzenEmbed in a specific env, activate it before running

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
D1_DIR="${D1_DIR:-${DATA_DIR}/d1}"

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

VIDEO_BASE="${VIDEO_BASE:-/hub_data2/dohwan/data/retrieval/activitynet/videos}"
INPUT_NAME="$(basename "${INPUT_JSON}" .json)"
mkdir -p "${DATA_DIR}/d2_reranked"
DEFAULT_OUTPUT_DIR="${DATA_DIR}/d2_reranked/${INPUT_NAME}_rzen"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

echo "Input JSON     : ${INPUT_JSON}"
echo "Video base path: ${VIDEO_BASE}"
echo "Output dir     : ${OUTPUT_DIR}"

# Add local RzenEmbed folder to PYTHONPATH if present
RZEN_LOCAL_DIR="${PREPROCESS_DIR}/RzenEmbed"
if [[ -d "${RZEN_LOCAL_DIR}" ]]; then
  export PYTHONPATH="${RZEN_LOCAL_DIR}:${PYTHONPATH:-}"
fi

# Controls
NUM_FRAMES="${NUM_FRAMES:-8}"
QUERY_INSTR="${QUERY_INSTR:-Find the video snippet that corresponds to the given caption:}"
CAND_INSTR="${CAND_INSTR:-Understand the content of the provided video.}"
LIMIT_COUNT="${LIMIT_COUNT:-0}"
SEED="${SEED:-42}"

CMD=(
  python "${PREPROCESS_DIR}/video_rzenembed_generate_reranking.py"
  --input_json "${INPUT_JSON}"
  --video_base "${VIDEO_BASE}"
  --output_dir "${OUTPUT_DIR}"
  --num_frames "${NUM_FRAMES}"
  --query_instruction "${QUERY_INSTR}"
  --candidate_instruction "${CAND_INSTR}"
  --seed "${SEED}"
)

if [[ -n "${LIMIT_COUNT}" && "${LIMIT_COUNT}" -gt 0 ]]; then
  CMD+=(--limit "${LIMIT_COUNT}")
fi

"${CMD[@]}"

echo "Generated files:"
ls -1 "${OUTPUT_DIR}"/reranking_train_*.json || true
[[ -f "${OUTPUT_DIR}/similarity_matrix.pkl" ]] && echo "- ${OUTPUT_DIR}/similarity_matrix.pkl" || true
