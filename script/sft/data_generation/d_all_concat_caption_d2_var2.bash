#!/usr/bin/env bash

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

########################################
# 공통 setting (d_all with concat mode, RzenEmbed rerank)
########################################

# Data root for inputs/outputs
: "${DATA_DIR:=${ROOT_DIR}/data}"

# video directory
: "${VIDEO_BASE:=/hub_data2/dohwan/data/retrieval/activitynet/videos}"

# Hugging Face upload settings
: "${HF_USER:=happy8825}"
# If empty, defaults to "${BASE_NAME}_sft" in d3-3
: "${HF_REPO:=}"

# Optional unified base name across steps (leave empty to auto-detect)
: "${BASE_NAME:=}"

# d0_0: Filter JSON to entries whose videos exist
: "${RUN_D0_0:=true}"
: "${D0_0_JSON_PATH:=${ROOT_DIR}/data/anet_ret_val_1.json}"
: "${D0_0_OUTPUT_JSON:=${D0_0_JSON_PATH}}"
: "${D0_0_LIST_MISSING:=false}"
: "${D0_0_VALIDATE_DECODE:=true}"
: "${D0_0_TEST_FRAMES:=3}"
: "${D0_0_WORKERS:=50}"

# Toggle steps (set to true/false)
: "${RUN_D0:=true}"
: "${RUN_D1:=true}"
: "${RUN_D2:=true}"
: "${RUN_D2_ANALYZE:=true}"
: "${RUN_D3_1:=false}"
: "${RUN_D3_2:=false}"
: "${RUN_D3_3:=false}"

# Dry run: echo commands instead of executing
: "${DRY_RUN:=false}"

# Optional
# INPUT_JSON=""         # d2: d1 결과 글로벌 캡션 JSON 경로 (ex: data/d1/xxx_global.json)
# SIMILARITY_PKL=""     # d2-analyze: similarity_matrix.pkl 경로
# RERANK_JSON=""         # d3-1: reranking_train_hard.json 경로
# SFT_JSON=""            # d3-2: *_sft_training_data_kept.json 경로

# d1/generation and d3-1 SFT generation model controls (optional)
# MODEL_NAME="gpt-4o"    # overrides per-script default if exported
MAX_WORKERS=30         # overrides per-script default if exported

########################################
# End of CONFIG
########################################

export DATA_DIR VIDEO_BASE HF_USER HF_REPO BASE_NAME

run_step() {
  local step_name="$1"; shift
  local script_path="$1"; shift

  if [[ ! -f "${script_path}" ]]; then
    echo "❌ Missing script: ${script_path}" >&2
    exit 1
  fi

  echo "\n===== [${step_name}] ${script_path} ====="
  echo "DATA_DIR=${DATA_DIR} | VIDEO_BASE=${VIDEO_BASE} | BASE_NAME=${BASE_NAME:-<auto>}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: ${script_path} $*"
  else
    bash "${script_path}" "$@"
  fi
}

echo "Working dir : ${SCRIPT_DIR}"
echo "Project root: ${ROOT_DIR}"

# Optional environment setup
if [[ "${RUN_D0}" == "true" ]]; then
  run_step "d0-env" "${SCRIPT_DIR}/d0_env_setting.bash"
fi

# Existence check before generating anything downstream
if [[ "${RUN_D0_0}" == "true" ]]; then
  echo "\n===== [d0_0-existence-check] ${SCRIPT_DIR}/d0_0_data_existence_check.bash ====="
  echo "JSON_PATH=${D0_0_JSON_PATH} | VIDEO_BASE=${VIDEO_BASE} | OUTPUT_JSON=${D0_0_OUTPUT_JSON} | WORKERS=${D0_0_WORKERS}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN: JSON_PATH=\"${D0_0_JSON_PATH}\" VIDEO_BASE=\"${VIDEO_BASE}\" OUTPUT_JSON=\"${D0_0_OUTPUT_JSON}\" LIST_MISSING=\"${D0_0_LIST_MISSING}\" VALIDATE_DECODE=\"${D0_0_VALIDATE_DECODE}\" TEST_FRAMES=\"${D0_0_TEST_FRAMES}\" WORKERS=\"${D0_0_WORKERS}\" bash \"${SCRIPT_DIR}/d0_0_data_existence_check.bash\""
  else
    JSON_PATH="${D0_0_JSON_PATH}" \
    VIDEO_BASE="${VIDEO_BASE}" \
    OUTPUT_JSON="${D0_0_OUTPUT_JSON}" \
    LIST_MISSING="${D0_0_LIST_MISSING}" \
    VALIDATE_DECODE="${D0_0_VALIDATE_DECODE}" \
    TEST_FRAMES="${D0_0_TEST_FRAMES}" \
    WORKERS="${D0_0_WORKERS}" \
      bash "${SCRIPT_DIR}/d0_0_data_existence_check.bash"
  fi
fi

# Global caption generation (concat mode)
if [[ "${RUN_D1}" == "true" ]]; then
  run_step "d1-global-caption" "${SCRIPT_DIR}/d1_run_global_caption_batch.sh" --concat_caption
fi

if [[ "${RUN_D2}" == "true" ]]; then
  # Use var2 (RzenEmbed) reranking generator
  run_step "d2-generate-reranking-var2" "${SCRIPT_DIR}/d2_var2_rzenembed_generate_reranking.bash"
fi

if [[ "${RUN_D2_ANALYZE}" == "true" ]]; then
  run_step "d2-analyze-reranking" "${SCRIPT_DIR}/d2-analyze_rerank_analyze.bash"
fi

if [[ "${RUN_D3_1}" == "true" ]]; then
  run_step "d3-1-sft-generate" "${SCRIPT_DIR}/d3-1-sft_generate.bash"
fi

if [[ "${RUN_D3_2}" == "true" ]]; then
  run_step "d3-2-sft-convert" "${SCRIPT_DIR}/d3-2-sft_convert_to_training.bash"
fi

if [[ "${RUN_D3_3}" == "true" ]]; then
  run_step "d3-3-sft-push-hf" "${SCRIPT_DIR}/d3-3-sft_push_hf.bash"
fi

echo "\n✅ Pipeline completed."

