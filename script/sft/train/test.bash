#!/usr/bin/env bash

set -euo pipefail

# Minimal test runner to verify that 5 videos are interleaved at <video> anchors
# via the vLLM OpenAI-compatible API path. Other logic mirrors t3_infer_with_vllm_sft.bash.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Optional conda activation
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "llama_factory" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llama_factory || true
  fi
fi

if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR="/hub_data2/seohyun/tmp"
fi
mkdir -p "${TMPDIR}"
export TMPDIR

# Defaults (override via env)
: "${MODEL_REPO:=happy8825/sft-20251126}"
: "${DATASET_REPO:=happy8825/activitynet_validset}"
: "${DATASET_FILE:=}"
: "${OUTPUT_JSON:=${ROOT_DIR}/video_retrieval/output_sft/test_interleave.json}"
OUTPUT_DIR="$(dirname "${OUTPUT_JSON}")"

: "${MEDIA_BASE:=/hub_data2/dohwan/data/retrieval}"
: "${TEMPLATE:=qwen3_vl}"
: "${VIDEO_FPS:=2.0}"
: "${VIDEO_MAXLEN:=48}"
: "${MAX_SAMPLES:=10}"
: "${VIDEO_NUM_FRAMES:=48}"
: "${VIDEO_TOTAL_PIXELS:=$((224*224))}"
: "${VIDEO_MIN_PIXELS:=0}"
: "${LOG_VIDEO_FRAMES:=false}"

# Prompt: do not include dataset query; just interleave 5 placeholders.
# NOTE: run_infer_with_vllm_api.py will still append the dataset's original user content.
# Use a subshell + heredoc (exit code 0) to avoid set -e exiting early.
__PROMPT_DEFAULT="$(cat <<'PROMPT'
describethis each video with video index number

Candidates:
[1] video:<video>
[2] video:<video>
[3] video:<video>
[4] video:<video>
[5] video:<video>
PROMPT
)"
: "${PROMPT:=${__PROMPT_DEFAULT}}"

# API settings
: "${API_BASE:=http://localhost:8011/v1}"
: "${API_KEY:=}"
: "${REQUEST_TIMEOUT:=240}"
: "${MAX_RETRIES:=4}"
: "${API_TEMPERATURE:=0.2}"
: "${API_TOP_P:=0.9}"
: "${API_MAX_NEW_TOKENS:=1024}"
: "${API_PRESENCE_PENALTY:=1.0}"

# Concurrency / Streaming
: "${GPU_DEVICES:=0}"
: "${MAX_CONCURRENT:=2}"
export MAX_CONCURRENT

: "${STREAM_JSONL:=true}"
: "${TRUNCATE_JSONL:=true}"
: "${JSONL_PATH:=${OUTPUT_JSON%.json}.jsonl}"
JSONL_DIR="$(dirname "${JSONL_PATH}")"

# Push to HF disabled for this test
: "${PUSH_HF:=false}"

# Resume/skip controls
: "${SKIP_IF_EXISTS:=false}"

echo "[test] Model repo   : ${MODEL_REPO}"
echo "[test] Dataset repo : ${DATASET_REPO} (${DATASET_FILE:-auto})"
echo "[test] Output json  : ${OUTPUT_JSON}"
echo "[test] Media base   : ${MEDIA_BASE}"
echo "[test] Template     : ${TEMPLATE} | fps=${VIDEO_FPS} | maxlen=${VIDEO_MAXLEN}"
echo "[test] Max samples  : ${MAX_SAMPLES} | Concurrency=${MAX_CONCURRENT}"
echo "[test] API base     : ${API_BASE} | timeout=${REQUEST_TIMEOUT}s | retries=${MAX_RETRIES}"
echo "[test] Video frames : request nframes=${VIDEO_NUM_FRAMES}"
echo "[test] Video pixels : total=${VIDEO_TOTAL_PIXELS} | min=${VIDEO_MIN_PIXELS}"
echo "[test] Prompt (first 80): ${PROMPT:0:80}"
echo "[test] Stream jsonl : ${STREAM_JSONL} -> ${JSONL_PATH} (truncate=${TRUNCATE_JSONL})"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${JSONL_DIR}"

dataset_args=()
if [[ -n "${DATASET_FILE}" ]]; then
  dataset_args+=(--dataset_file "${DATASET_FILE}")
fi

video_meta_args=(
  --video_num_frames "${VIDEO_NUM_FRAMES}"
  --video_total_pixels "${VIDEO_TOTAL_PIXELS}"
  --video_min_pixels "${VIDEO_MIN_PIXELS}"
)

api_common_args=(
  --api_base "${API_BASE}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --temperature "${API_TEMPERATURE}"
  --top_p "${API_TOP_P}"
  --max_new_tokens "${API_MAX_NEW_TOKENS}"
  --presence_penalty "${API_PRESENCE_PENALTY}"
)
if [[ -n "${API_KEY}" ]]; then
  api_common_args+=(--api_key "${API_KEY}")
fi

if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${OUTPUT_JSON}" ]]; then
  echo "[test][skip] Found ${OUTPUT_JSON}"
else
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
  python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_with_vllm_api.py" \
    --model_repo "${MODEL_REPO}" \
    --dataset_repo "${DATASET_REPO}" \
    --output_json "${OUTPUT_JSON}" \
    --media_base "${MEDIA_BASE}" \
    --template "${TEMPLATE}" \
    --video_fps "${VIDEO_FPS}" \
    --video_maxlen "${VIDEO_MAXLEN}" \
    --max_samples "${MAX_SAMPLES}" \
    --prepend_prompt "${PROMPT}" \
    --prepend_only \
    --concurrency "${MAX_CONCURRENT}" \
    --echo_ic \
    --ic_prefix "[test|api] ic| " \
    $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
    --jsonl_path "${JSONL_PATH}" \
    $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
    "${dataset_args[@]}" \
    "${video_meta_args[@]}" \
    "${api_common_args[@]}"
fi

echo "✅ Test run finished: ${OUTPUT_JSON}"
