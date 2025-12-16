#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Conda env activation (optional, same as original script)
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "llama_factory" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llama_factory || true
  fi
fi

# Ensure temp dir
if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR="/hub_data1/seohyun/tmp"
fi
mkdir -p "${TMPDIR}"
export TMPDIR

# Reuse defaults from the original script
: "${MODEL_REPO:=Qwen/Qwen3-VL-2B-Instruct}"
: "${DATASET_REPO:=happy8825/activitynet_validset}"
: "${DATASET_FILE:=}"

# Base outputs from the original run
: "${OUTPUT_JSON:=${ROOT_DIR}/video_retrieval/output_vllm/qweninstruct2b_infer_result.json}"
JSONL_DEFAULT="${OUTPUT_JSON%.json}.jsonl"
: "${JSONL_IN:=${JSONL_DEFAULT}}"

# Target cleaned JSON (without error)
DEFAULT_NOERR="${OUTPUT_JSON%.json}_withouterror.json"
: "${OUTPUT_JSON_NOERR:=${DEFAULT_NOERR}}"

# Optional streaming of the rerun results to a new jsonl (depends on OUTPUT_JSON_NOERR)
: "${STREAM_RERUN:=true}"
DEFAULT_NOERR_JSONL="${OUTPUT_JSON_NOERR%.json}.jsonl"
: "${JSONL_OUT_NOERR:=${DEFAULT_NOERR_JSONL}}"
: "${TRUNCATE_RERUN_JSONL:=true}"
: "${STREAM_EXISTING:=true}"

# Media + prompting
: "${MEDIA_BASE:=/hub_data3/seohyun}"
: "${SYSTEM_PROMPT:=}"
: "${PROMPT:=You will receive a Query and N candidates. Each candidate is a video. Your task is to identify the most relevant video to the Query. Output only the index of the most relevant video among the candidates. Your answer should be strictly inside <answer></answer> tags. }"

# Video/token controls (first attempt) + fallbacks to fix length errors
: "${VIDEO_NUM_FRAMES:=48}"
: "${VIDEO_TOTAL_PIXELS:=$((224*224))}"
: "${VIDEO_MIN_PIXELS:=0}"
: "${FALLBACK_FRAMES:=24,16,8}"
: "${FALLBACK_PIXELS:=16384,12544,9216}"
: "${PREFER_MP4:=true}"

# API
: "${API_BASE:=http://localhost:8019/v1}"
: "${API_KEY:=}"
: "${REQUEST_TIMEOUT:=240}"
: "${MAX_RETRIES:=4}"
: "${API_TEMPERATURE:=0.2}"
: "${API_TOP_P:=0.9}"
: "${API_MAX_NEW_TOKENS:=2048}"
: "${API_PRESENCE_PENALTY:=1.0}"
: "${API_TOP_K:=-1}"

# Concurrency
: "${MAX_CONCURRENT:=4}"
export MAX_CONCURRENT

# Push to HF (match naming style from original script)
: "${PUSH_HF:=true}"
__out_base_noerr="$(basename "${OUTPUT_JSON_NOERR}")"
__base_out_orig="$(basename "${OUTPUT_JSON}")"
: "${HF_OUT_FILE:=${__out_base_noerr}}"
# Default repo name is derived from the original output (to keep same repo)
: "${HF_OUT_REPO:=${__base_out_orig%.*}}"
: "${HF_COMMIT_MESSAGE:=Upload without-error results}"

mkdir -p "$(dirname "${OUTPUT_JSON_NOERR}")"

echo "[noerror] Model repo   : ${MODEL_REPO}"
echo "[noerror] Dataset repo : ${DATASET_REPO} (${DATASET_FILE:-auto})"
echo "[noerror] Input jsonl  : ${JSONL_IN}"
echo "[noerror] Base json    : ${OUTPUT_JSON}"
echo "[noerror] Output json  : ${OUTPUT_JSON_NOERR}"
echo "[noerror] Media base   : ${MEDIA_BASE}"
echo "[noerror] API base     : ${API_BASE} | timeout=${REQUEST_TIMEOUT}s | retries=${MAX_RETRIES}"
echo "[noerror] Video cfg    : nframes=${VIDEO_NUM_FRAMES} total_px=${VIDEO_TOTAL_PIXELS} min_px=${VIDEO_MIN_PIXELS}"
echo "[noerror] Fallbacks    : frames=[${FALLBACK_FRAMES}] pixels=[${FALLBACK_PIXELS}]"
echo "[noerror] Concurrency  : ${MAX_CONCURRENT}"
echo "[noerror] Push HF      : ${PUSH_HF} -> ${HF_OUT_REPO}/${HF_OUT_FILE}"
echo "[noerror] Stream rerun : ${STREAM_RERUN} -> ${JSONL_OUT_NOERR} (truncate=${TRUNCATE_RERUN_JSONL}, existing=${STREAM_EXISTING})"

dataset_args=()
if [[ -n "${DATASET_FILE}" ]]; then
  dataset_args+=(--dataset_file "${DATASET_FILE}")
fi

api_args=(
  --api_base "${API_BASE}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --temperature "${API_TEMPERATURE}"
  --top_p "${API_TOP_P}"
  --top_k "${API_TOP_K}"
  --max_new_tokens "${API_MAX_NEW_TOKENS}"
  --presence_penalty "${API_PRESENCE_PENALTY}"
)
if [[ -n "${API_KEY}" ]]; then
  api_args+=(--api_key "${API_KEY}")
fi

python3 "${ROOT_DIR}/SFT/sft_trainer/infer/rerun_errors_from_jsonl.py" \
  --jsonl_in "${JSONL_IN}" \
  --base_json_in "${OUTPUT_JSON}" \
  --output_json "${OUTPUT_JSON_NOERR}" \
  --model_repo "${MODEL_REPO}" \
  --dataset_repo "${DATASET_REPO}" \
  --media_base "${MEDIA_BASE}" \
  --system_prompt "${SYSTEM_PROMPT}" \
  --prepend_prompt "${PROMPT}" \
  --video_num_frames "${VIDEO_NUM_FRAMES}" \
  --video_total_pixels "${VIDEO_TOTAL_PIXELS}" \
  --video_min_pixels "${VIDEO_MIN_PIXELS}" \
  --fallback_frames "${FALLBACK_FRAMES}" \
  --fallback_pixels "${FALLBACK_PIXELS}" \
  $( [[ "${PREFER_MP4}" == "true" ]] && echo --prefer_mp4 ) \
  --concurrency "${MAX_CONCURRENT}" \
  $( [[ "${STREAM_RERUN}" == "true" ]] && echo --stream_jsonl_out "${JSONL_OUT_NOERR}" ) \
  $( [[ "${TRUNCATE_RERUN_JSONL}" == "true" ]] && echo --truncate_stream_out ) \
  $( [[ "${STREAM_EXISTING}" == "true" ]] && echo --stream_existing ) \
  --echo_ic --ic_prefix "[noerror] " \
  "${dataset_args[@]}" \
  "${api_args[@]}"

echo "✅ Rerun for error entries finished: ${OUTPUT_JSON_NOERR}"

if [[ "${PUSH_HF}" == "true" && -s "${OUTPUT_JSON_NOERR}" ]]; then
  echo "[noerror][push] Uploading ${OUTPUT_JSON_NOERR} to HF repo ${HF_OUT_REPO}/${HF_OUT_FILE}"
  python3 "${ROOT_DIR}/SFT/sft_trainer/infer/push_results_to_hf.py" \
    --input_json "${OUTPUT_JSON_NOERR}" \
    --upload_path "${OUTPUT_JSON_NOERR}" \
    --hf_out_repo "${HF_OUT_REPO}" \
    --hf_out_file "${HF_OUT_FILE}" \
    --commit_message "${HF_COMMIT_MESSAGE}"
fi
