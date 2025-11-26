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

# Defaults (override via env)
: "${MODEL_REPO:=happy8825/sft-20251124}"
: "${DATASET_REPO:=happy8825/activitynet_validset_all}"
: "${DATASET_FILE:=anet_val_all_sft_llamafactory_gt.json}"
: "${OUTPUT_JSON:=${ROOT_DIR}/video_retrieval/output_1124_sft_easy_prompt/output_valid_all_sft1124.json}"

: "${MEDIA_BASE:=/hub_data2/dohwan/data/retrieval}"
: "${TEMPLATE:=qwen3_vl}"
: "${VIDEO_FPS:=2.0}"
: "${VIDEO_MAXLEN:=16}"
: "${MAX_SAMPLES:=4600}"

# Prompt to prepend before the user Query. Example override:
#   PROMPT=$'System Prompt: ...' bash t3_infer_valid_all.bash
: "${PROMPT:=You will receive a Query and N candidates. Each candidate is a video. Your task is to identify the most relevant video to the Query. Output only the index of the most relevant video among the candidates. Your answer should be strictly inside <answer></answer> tags. }"

# Concurrency / Streaming
: "${GPU_DEVICES:=4,5,6,7}"
: "${MAX_CONCURRENT:=2}"
export MAX_CONCURRENT

: "${STREAM_JSONL:=true}"
: "${TRUNCATE_JSONL:=true}"
: "${JSONL_PATH:=${OUTPUT_JSON%.json}.jsonl}"

# Push to HF
: "${PUSH_HF:=true}"
__out_base="$(basename "${OUTPUT_JSON}")"
: "${HF_OUT_FILE:=${__out_base}}"
: "${HF_OUT_REPO:=${__out_base%.*}}"

# Resume/skip controls
: "${SKIP_IF_EXISTS:=true}"

echo "Model repo   : ${MODEL_REPO}"
echo "Dataset repo : ${DATASET_REPO} (${DATASET_FILE})"
echo "Output json  : ${OUTPUT_JSON}"
echo "Media base   : ${MEDIA_BASE}"
echo "Template     : ${TEMPLATE} | fps=${VIDEO_FPS} | maxlen=${VIDEO_MAXLEN}"
echo "Max samples  : ${MAX_SAMPLES} | Concurrency=${MAX_CONCURRENT}"
echo "Prepend prompt (first 60 chars): ${PROMPT:0:60}"
echo "Stream jsonl : ${STREAM_JSONL} -> ${JSONL_PATH} (truncate=${TRUNCATE_JSONL})"
echo "Push HF      : ${PUSH_HF} -> ${HF_OUT_REPO}/${HF_OUT_FILE}"
echo "Skip if exists: ${SKIP_IF_EXISTS}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

split_and_run=false
IFS=',' read -ra __gpus <<< "${GPU_DEVICES}"
if [[ ${#__gpus[@]} -gt 1 ]]; then
  split_and_run=true
fi

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
      python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_valid_all.py" \
        --model_repo "${MODEL_REPO}" \
        --dataset_repo "${DATASET_REPO}" \
        --dataset_file "${DATASET_FILE}" \
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
        $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl )
    ) &
  done
  wait
  echo "Merging shards → ${OUTPUT_JSON}"
  python3 "${ROOT_DIR}/SFT/sft_trainer/infer/merge_infer_results.py" \
    --inputs "${shard_jsons[@]}" \
    --output "${OUTPUT_JSON}" \
    --model_repo "${MODEL_REPO}" \
    --dataset_repo "${DATASET_REPO}" \
    $( [[ "${PUSH_HF}" == "true" ]] && echo --push_hf ) \
    --hf_out_repo "${HF_OUT_REPO}" \
    --hf_out_file "${HF_OUT_FILE}"
else
  if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${OUTPUT_JSON}" ]]; then
    echo "[skip] single run -> found ${OUTPUT_JSON}"
  else
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_valid_all.py" \
      --model_repo "${MODEL_REPO}" \
      --dataset_repo "${DATASET_REPO}" \
      --dataset_file "${DATASET_FILE}" \
      --output_json "${OUTPUT_JSON}" \
      --media_base "${MEDIA_BASE}" \
      --template "${TEMPLATE}" \
      --video_fps "${VIDEO_FPS}" \
      --video_maxlen "${VIDEO_MAXLEN}" \
      --max_samples "${MAX_SAMPLES}" \
      --prepend_prompt "${PROMPT}" \
      --concurrency "${MAX_CONCURRENT}" \
      --echo_ic \
      --ic_prefix "[single|gpu ${GPU_DEVICES}] ic| " \
      $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
      --jsonl_path "${JSONL_PATH}" \
      $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
      $( [[ "${PUSH_HF}" == "true" ]] && echo --push_hf ) \
      --hf_out_repo "${HF_OUT_REPO}" \
      --hf_out_file "${HF_OUT_FILE}"
  fi
fi

echo "✅ Inference+Eval finished: ${OUTPUT_JSON}"
