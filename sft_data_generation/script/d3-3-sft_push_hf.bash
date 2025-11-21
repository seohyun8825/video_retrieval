#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"

if [[ -z "${BASE_NAME:-}" ]]; then
  mapfile -t __llama_files < <(find "${DATA_DIR}/d3" -maxdepth 1 -type f -name "*_sft_llamafactory.json" | sort)
  if [[ ${#__llama_files[@]} -eq 0 ]]; then
    echo "❌ No *_sft_llamafactory.json found in ${DATA_DIR}/d3. Set BASE_NAME or FILE_PATH explicitly." >&2
    exit 1
  fi
  FILE_PATH="${__llama_files[0]}"
  BASE_NAME="$(basename "${FILE_PATH}" _sft_llamafactory.json)"
else
  FILE_PATH="${FILE_PATH:-${DATA_DIR}/d3/${BASE_NAME}_sft_llamafactory.json}"
fi

if [[ ! -f "${FILE_PATH}" ]]; then
  echo "❌ File not found: ${FILE_PATH}" >&2
  exit 1
fi

HF_USER="${HF_USER:-happy8825}"
HF_REPO="${HF_REPO:-${BASE_NAME}_sft}"
REPO_ID="${HF_USER}/${HF_REPO}"
REMOTE_PATH="${REMOTE_PATH:-${BASE_NAME}_sft_llamafactory.json}"

echo "Uploading ${FILE_PATH} to huggingface dataset ${REPO_ID} as ${REMOTE_PATH}"

huggingface-cli repo create "${REPO_ID}" --type dataset --exist-ok >/dev/null

huggingface-cli upload "${REPO_ID}" "${FILE_PATH}" "${REMOTE_PATH}" --repo-type dataset --commit-message "Add ${BASE_NAME} SFT data"

echo "✅ Uploaded to https://huggingface.co/datasets/${REPO_ID}"
