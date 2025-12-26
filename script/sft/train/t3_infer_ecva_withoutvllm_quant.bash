#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Conda env activation (optional)
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "llama_factory" ]]; then
  if command -v conda >/dev/null 2>&1; then
    nounset_was_set=0
    case $- in *u*) nounset_was_set=1 ;; esac
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llama_factory || true
    if [[ "${nounset_was_set}" -eq 1 ]]; then set -u; fi
  fi
fi

# Ensure local LLaMA-Factory sources are on PYTHONPATH (for 'llamafactory' imports)
export PYTHONPATH="${ROOT_DIR}/SFT/sft_trainer/LLaMA-Factory/src:${ROOT_DIR}/SFT/sft_trainer:${PYTHONPATH:-}"

# Writable tempdir
if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR="/hub_data1/seohyun/tmp"
fi
mkdir -p "${TMPDIR}"
export TMPDIR

# Defaults (override via env)
: "${MODEL_REPO:=/hub_data4/seohyun/saves/ecva_instruct_1223/full/sft/checkpoint-350/}"
: "${DATASET_REPO:=happy8825/valid_ecva_clean}"
: "${DATASET_FILE:=}"
: "${OUTPUT_JSON:=${ROOT_DIR}/video_retrieval/output_qwen/1223_350_q4.json}"
OUTPUT_DIR="$(dirname "${OUTPUT_JSON}")"

: "${MEDIA_BASE:=/hub_data3/seohyun}"
: "${TEMPLATE:=qwen3_vl}"
: "${VIDEO_FPS:=2.0}"
: "${VIDEO_MAXLEN:=36}"
: "${VIDEO_MAX_PIXELS:=402144}"
: "${IMAGE_MAX_PIXELS:=8000000}"
: "${MAX_SAMPLES:=5000}"

# BitsAndBytes quantization: 4 or 8 (bnb)
: "${QUANTIZATION_BIT:=4}"

# Prompt to prepend before the user Query.
: "${PROMPT:=Are any anomalies directly occurring in this clip? If yes, identify them briefly. }"

# GPUs / concurrency (single GPU default: 3)
: "${GPU_DEVICES:=3}"
: "${MAX_CONCURRENT:=2}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
export MAX_CONCURRENT

# Streaming JSONL controls
: "${STREAM_JSONL:=true}"
: "${TRUNCATE_JSONL:=true}"
: "${JSONL_PATH:=${OUTPUT_JSON%.json}.jsonl}"
JSONL_DIR="$(dirname "${JSONL_PATH}")"

# Push to HF
: "${PUSH_HF:=true}"
__out_base="$(basename "${OUTPUT_JSON}")"
: "${HF_OUT_FILE:=${__out_base}}"
: "${HF_OUT_REPO:=${__out_base%.*}}"

# Resume/skip controls
: "${SKIP_IF_EXISTS:=true}"

echo "Model repo   : ${MODEL_REPO}"
echo "Dataset repo : ${DATASET_REPO} (${DATASET_FILE:-auto})"
echo "Output json  : ${OUTPUT_JSON}"
echo "Media base   : ${MEDIA_BASE}"
echo "Template     : ${TEMPLATE} | fps=${VIDEO_FPS} | maxlen=${VIDEO_MAXLEN} | video_max_pixels=${VIDEO_MAX_PIXELS} | image_max_pixels=${IMAGE_MAX_PIXELS}"
echo "Quantization : ${QUANTIZATION_BIT}-bit (bnb)"
echo "Max samples  : ${MAX_SAMPLES} | Concurrency=${MAX_CONCURRENT}"
echo "Stream jsonl : ${STREAM_JSONL} -> ${JSONL_PATH} (truncate=${TRUNCATE_JSONL})"
echo "Push HF      : ${PUSH_HF} -> ${HF_OUT_REPO}/${HF_OUT_FILE}"
echo "GPU devices  : ${GPU_DEVICES}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${JSONL_DIR}"

dataset_args=()
if [[ -n "${DATASET_FILE}" ]]; then
  dataset_args+=(--dataset_file "${DATASET_FILE}")
fi

script_start_ns=$(date +%s%N)

if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${OUTPUT_JSON}" ]]; then
  echo "[skip] found existing ${OUTPUT_JSON}"
else
  python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_valid_all.py" \
    --model_repo "${MODEL_REPO}" \
    --dataset_repo "${DATASET_REPO}" \
    --output_json "${OUTPUT_JSON}" \
    --media_base "${MEDIA_BASE}" \
    --template "${TEMPLATE}" \
    --video_fps "${VIDEO_FPS}" \
    --video_maxlen "${VIDEO_MAXLEN}" \
    --video_max_pixels "${VIDEO_MAX_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --max_samples "${MAX_SAMPLES}" \
    --prepend_prompt "${PROMPT}" \
    --concurrency "${MAX_CONCURRENT}" \
    --gt_is_label \
    --evqa \
    --echo_ic \
    --ic_prefix "[local|hf|q${QUANTIZATION_BIT}] ic| " \
    $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
    --jsonl_path "${JSONL_PATH}" \
    $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
    $( [[ "${PUSH_HF}" == "true" ]] && echo --push_hf ) \
    --hf_out_repo "${HF_OUT_REPO}" \
    --hf_out_file "${HF_OUT_FILE}" \
    --debug_time \
    --debug_memory \
    --quantization_bit "${QUANTIZATION_BIT}" \
    "${dataset_args[@]}"
fi

script_end_ns=$(date +%s%N)
total_ms=$(( (script_end_ns - script_start_ns) / 1000000 ))
echo "Total wall clock (ms): ${total_ms}"
echo "✅ Inference via HF engine (bnb ${QUANTIZATION_BIT}-bit) finished: ${OUTPUT_JSON}"
