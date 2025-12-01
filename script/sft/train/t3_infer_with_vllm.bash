#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Conda env activation (optional)
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "llama_factory" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llama_factory || true
  fi
fi

# Ensure we have a writable temp directory (avoid /tmp quota issues)
if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR="/hub_data1/seohyun/tmp"
fi
mkdir -p "${TMPDIR}"
export TMPDIR

# Defaults (override via env)
: "${MODEL_REPO:=happy8825/sft-20251126}"
: "${DATASET_REPO:=happy8825/activitynet_validset}"
: "${DATASET_FILE:=}"
: "${OUTPUT_JSON:=${ROOT_DIR}/video_retrieval/output_vllm/output_valid_all_vllm.json}"
OUTPUT_DIR="$(dirname "${OUTPUT_JSON}")"

: "${MEDIA_BASE:=/hub_data1/seohyun/clean}"
: "${TEMPLATE:=qwen3_vl}"
: "${VIDEO_FPS:=2.0}"
: "${VIDEO_MAXLEN:=48}"
: "${MAX_SAMPLES:=5000}"
: "${VIDEO_NUM_FRAMES:=48}"
: "${VIDEO_TOTAL_PIXELS:=$((224*224))}"
: "${VIDEO_MIN_PIXELS:=0}"
: "${LOG_VIDEO_FRAMES:=false}"
: "${SYSTEM_PROMPT:=}"
: "${TMP_VIDEO_DIR:=}"

# Prompt to prepend before the user Query.
: "${PROMPT:=You will receive a Query and N candidates. Each candidate is a video. Your task is to identify the most relevant video to the Query. Output only the index of the most relevant video among the candidates. Your answer should be strictly inside <answer></answer> tags. }"

# API settings
: "${API_BASE:=http://localhost:8011/v1}"
: "${API_KEY:=}"
: "${REQUEST_TIMEOUT:=240}"
: "${MAX_RETRIES:=4}"
: "${API_TEMPERATURE:=0.2}"
: "${API_TOP_P:=0.9}"
: "${API_MAX_NEW_TOKENS:=2048}"
: "${API_PRESENCE_PENALTY:=1.0}"

# Concurrency / Streaming
: "${GPU_DEVICES:=0}"
: "${MAX_CONCURRENT:=4}"
export MAX_CONCURRENT

: "${STREAM_JSONL:=true}"
: "${TRUNCATE_JSONL:=true}"
: "${JSONL_PATH:=${OUTPUT_JSON%.json}.jsonl}"
JSONL_DIR="$(dirname "${JSONL_PATH}")"
: "${HF_UPLOAD_PATH:=${JSONL_PATH}}"
HF_UPLOAD_DIR="$(dirname "${HF_UPLOAD_PATH}")"

# Push to HF
: "${PUSH_HF:=true}"
: "${PUSH_ONLY:=false}"
__out_base="$(basename "${OUTPUT_JSON}")"
: "${HF_OUT_FILE:=$(basename "${HF_UPLOAD_PATH}")}"
: "${HF_OUT_REPO:=${__out_base%.*}}"
: "${HF_README_PATH:=${OUTPUT_DIR}/README.md}"
: "${HF_README_EXTRA:=}"
: "${HF_COMMIT_MESSAGE:=}"
: "${HF_DELETE_REMOTE:=}"

# Resume/skip controls
: "${SKIP_IF_EXISTS:=true}"

echo "Model repo   : ${MODEL_REPO}"
echo "Dataset repo : ${DATASET_REPO} (${DATASET_FILE:-auto})"
echo "Output json  : ${OUTPUT_JSON}"
echo "Media base   : ${MEDIA_BASE}"
echo "Template     : ${TEMPLATE} | fps=${VIDEO_FPS} | maxlen=${VIDEO_MAXLEN}"
echo "Max samples  : ${MAX_SAMPLES} | Concurrency=${MAX_CONCURRENT}"
echo "API base     : ${API_BASE} | timeout=${REQUEST_TIMEOUT}s | retries=${MAX_RETRIES}"
echo "Video frames : request nframes=${VIDEO_NUM_FRAMES}"
echo "Video pixels : total=${VIDEO_TOTAL_PIXELS} | min=${VIDEO_MIN_PIXELS}"
echo "Prepend prompt (first 60 chars): ${PROMPT:0:60}"
echo "Stream jsonl : ${STREAM_JSONL} -> ${JSONL_PATH} (truncate=${TRUNCATE_JSONL})"
echo "Push HF      : ${PUSH_HF} -> ${HF_OUT_REPO}/${HF_OUT_FILE}"
echo "Skip if exists: ${SKIP_IF_EXISTS}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${JSONL_DIR}"
mkdir -p "${HF_UPLOAD_DIR}"

dataset_args=()
if [[ -n "${DATASET_FILE}" ]]; then
  dataset_args+=(--dataset_file "${DATASET_FILE}")
fi

video_meta_args=(
  --video_num_frames "${VIDEO_NUM_FRAMES}"
  --video_total_pixels "${VIDEO_TOTAL_PIXELS}"
  --video_min_pixels "${VIDEO_MIN_PIXELS}"
)
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  video_meta_args+=(--system_prompt "${SYSTEM_PROMPT}")
fi
if [[ "${LOG_VIDEO_FRAMES}" == "true" ]]; then
  video_meta_args+=(--log_video_frames)
fi

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

split_and_run=false
IFS=',' read -ra __gpus <<< "${GPU_DEVICES}"
if [[ ${#__gpus[@]} -gt 1 ]]; then
  split_and_run=true
fi

if [[ "${PUSH_ONLY}" == "true" ]]; then
  echo "[push-only] Skipping inference run and reusing ${OUTPUT_JSON}"
  if [[ ! -s "${OUTPUT_JSON}" ]]; then
    echo "[push-only][error] Missing ${OUTPUT_JSON}. Run inference once before push-only mode." >&2
    exit 1
  fi
else
  if [[ "${split_and_run}" == "true" ]]; then
    echo "Sharded run across GPUs: ${GPU_DEVICES}"
    SHARD_COUNT=${#__gpus[@]}
    shard_jsons=()
    for idx in "${!__gpus[@]}"; do
      gpu="${__gpus[$idx]}"
      shard_out="${OUTPUT_JSON%.json}.shard${idx}.json"
      shard_jsonl="${JSONL_PATH%.jsonl}.shard${idx}.jsonl"
      if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${shard_out}" ]]; then
        echo "[skip] shard ${idx} -> found ${shard_out}"
        shard_jsons+=("${shard_out}")
        continue
      fi
      shard_jsons+=("${shard_out}")
      (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_with_vllm_api.py" \
          --model_repo "${MODEL_REPO}" \
          --dataset_repo "${DATASET_REPO}" \
          --output_json "${shard_out}" \
          --media_base "${MEDIA_BASE}" \
          --template "${TEMPLATE}" \
          --video_fps "${VIDEO_FPS}" \
          --video_maxlen "${VIDEO_MAXLEN}" \
          --max_samples "${MAX_SAMPLES}" \
          --prepend_prompt "${PROMPT}" \
          --concurrency "${MAX_CONCURRENT}" \
          --num_shards "${SHARD_COUNT}" \
          --shard_index "${idx}" \
          --echo_ic \
          --ic_prefix "[shard ${idx}|gpu ${gpu}] ic| " \
          $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
          --jsonl_path "${shard_jsonl}" \
          $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
          "${dataset_args[@]}" \
          "${video_meta_args[@]}" \
          "${api_common_args[@]}"
      ) &
    done
    wait
    echo "Merging shards → ${OUTPUT_JSON}"
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/merge_infer_results.py" \
      --inputs "${shard_jsons[@]}" \
      --output "${OUTPUT_JSON}" \
      --model_repo "${MODEL_REPO}" \
      --dataset_repo "${DATASET_REPO}"
  else
    if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${OUTPUT_JSON}" ]]; then
      echo "[skip] single run -> found ${OUTPUT_JSON}"
    else
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
        --concurrency "${MAX_CONCURRENT}" \
        --echo_ic \
        --ic_prefix "[single|api] ic| " \
        $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
        --jsonl_path "${JSONL_PATH}" \
        $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
        "${dataset_args[@]}" \
        "${video_meta_args[@]}" \
        "${api_common_args[@]}"
    fi
  fi
fi

if [[ "${PUSH_HF}" == "true" ]]; then
  if [[ ! -s "${OUTPUT_JSON}" ]]; then
    echo "[push] Missing ${OUTPUT_JSON}; skipping HF upload."
  else
    push_args=(
      --input_json "${OUTPUT_JSON}"
      --upload_path "${HF_UPLOAD_PATH}"
      --readme_path "${HF_README_PATH}"
      --readme_extra "${HF_README_EXTRA}"
    )
    if [[ -n "${HF_OUT_REPO}" ]]; then
      push_args+=(--hf_out_repo "${HF_OUT_REPO}")
    fi
    if [[ -n "${HF_OUT_FILE}" ]]; then
      push_args+=(--hf_out_file "${HF_OUT_FILE}")
    fi
    if [[ -n "${HF_COMMIT_MESSAGE}" ]]; then
      push_args+=(--commit_message "${HF_COMMIT_MESSAGE}")
    fi
    if [[ -n "${HF_DELETE_REMOTE}" ]]; then
      IFS=',' read -ra __del_list <<< "${HF_DELETE_REMOTE}"
      for del_path in "${__del_list[@]}"; do
        del_path="${del_path//[$'\t\n\r ']/}"
        [[ -z "${del_path}" ]] && continue
        push_args+=(--delete_remote "${del_path}")
      done
    fi
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/push_results_to_hf.py" "${push_args[@]}"
  fi
fi

echo "✅ Inference via vLLM API finished: ${OUTPUT_JSON}"
