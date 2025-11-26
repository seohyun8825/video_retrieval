#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_DIR="${ROOT_DIR}/SFT/sft_data_generation"
PREPROCESS_DIR="${SFT_DIR}/preprocess/extract_similarity_matrix"

if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "video-colbert" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate video-colbert
fi
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/train_data}"
if [[ -z "${SIMILARITY_PKL:-}" ]]; then
  mapfile -t __pkl_files < <(find "${DATA_DIR}/d2_reranked" -maxdepth 2 -type f -name "similarity_matrix.pkl" | sort)
  if [[ ${#__pkl_files[@]} -eq 0 ]]; then
    echo "❌ No similarity_matrix.pkl found under ${DATA_DIR}/d2_reranked. Set SIMILARITY_PKL explicitly." >&2
    exit 1
  fi
  SIMILARITY_PKL="${__pkl_files[0]}"
fi

DEFAULT_ANALYSIS_DIR="${DATA_DIR}/d2_reranked/analysis"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_ANALYSIS_DIR}}"
TOPKS="${TOPKS:-1 5 10}"
HEATMAP_SIZE="${HEATMAP_SIZE:-4917}"

echo "Similarity PKL : ${SIMILARITY_PKL}"
echo "Output Dir      : ${OUTPUT_DIR}"
echo "Top-Ks          : ${TOPKS}"
echo "Heatmap Size    : ${HEATMAP_SIZE}"

python "${PREPROCESS_DIR}/analyze_similarity_matrix.py" \
  --similarity_pkl "${SIMILARITY_PKL}" \
  --output_dir "${OUTPUT_DIR}" \
  --topks ${TOPKS} \
  --heatmap_size "${HEATMAP_SIZE}"
