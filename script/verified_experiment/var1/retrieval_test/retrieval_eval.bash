#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RETRIEVER_DIR="${ROOT_DIR}/var1_exp/retriever"

# Paths (edit here)
INPUT_JSONL="/home/seohyun/vid_understanding/activitynet_fig_val_2.jsonl"
VIDEO_BASE="/hub_data3/seohyun/activitynet/videos"
OUTPUT_DIR="${ROOT_DIR}/var1_exp/retriever_outputs/activitynet_fig_val_2_rzen"

# Hyperparameters (edit here)
NUM_FRAMES=8
TOPK=100
TOPKS="1 5 10 100"
LIMIT_COUNT=0
LIMIT_VIDEOS=0
BATCH_SIZE=32
SEED=42
QUERY_INSTR="Find the video snippet that contains the moment corresponding to the given caption:"
CAND_INSTR="Understand the content of the provided video."

PYTHON_BIN="python"

# Add local RzenEmbed folder to PYTHONPATH if present
RZEN_LOCAL_DIR="${ROOT_DIR}/SFT/sft_data_generation/preprocess/extract_similarity_matrix/RzenEmbed"
if [[ -d "${RZEN_LOCAL_DIR}" ]]; then
  export PYTHONPATH="${RZEN_LOCAL_DIR}:${PYTHONPATH:-}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "Input JSONL     : ${INPUT_JSONL}"
echo "Video base path : ${VIDEO_BASE}"
echo "Output dir      : ${OUTPUT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${RETRIEVER_DIR}/rzen_retrieval_eval.py"
  --input_jsonl "${INPUT_JSONL}"
  --video_base "${VIDEO_BASE}"
  --output_dir "${OUTPUT_DIR}"
  --num_frames "${NUM_FRAMES}"
  --topk "${TOPK}"
  --topks ${TOPKS}
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --query_field "fig_desc"
  --video_field "video"
  --id_field "desc_id"
  --query_instruction "${QUERY_INSTR}"
  --candidate_instruction "${CAND_INSTR}"
)

if [[ "${LIMIT_COUNT}" -gt 0 ]]; then
  CMD+=(--limit "${LIMIT_COUNT}")
fi

if [[ "${LIMIT_VIDEOS}" -gt 0 ]]; then
  CMD+=(--limit_videos "${LIMIT_VIDEOS}")
fi

"${CMD[@]}"
