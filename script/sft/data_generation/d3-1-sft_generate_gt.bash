#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_DIR="${ROOT_DIR}/SFT/sft_data_generation"
PREPROCESS_DIR="${SFT_DIR}/preprocess/sft_data_generate"

# Optional: activate the same env used elsewhere
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "video-colbert" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate video-colbert || true
  fi
fi

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
RERANK_DIR="${RERANK_DIR:-${DATA_DIR}/d2_reranked}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/d3}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${RERANK_DIR}" ]]; then
  echo "❌ d2 reranked directory not found: ${RERANK_DIR}" >&2
  exit 1
fi

if [[ -z "${RERANK_JSON:-}" ]]; then
  mapfile -t __rerank_jsons < <(find "${RERANK_DIR}" -maxdepth 2 -type f -name "reranking_train_hard.json" | sort)
  if [[ ${#__rerank_jsons[@]} -eq 0 ]]; then
    echo "❌ No reranking_train_hard.json found under ${RERANK_DIR}. Set RERANK_JSON explicitly." >&2
    exit 1
  fi
  RERANK_JSON="${__rerank_jsons[0]}"
fi

if [[ -z "${BASE_NAME:-}" ]]; then
  BASE_NAME="$(basename "$(dirname "${RERANK_JSON}")")"
  BASE_NAME="${BASE_NAME%_global}"
  BASE_NAME="${BASE_NAME%_reranked}"
fi

LIMIT="${LIMIT:-0}"
QUERY_OVERRIDE="${QUERY_OVERRIDE:-}"

CMD=(
  python "${PREPROCESS_DIR}/generate_sft_ranking_data_gt.py"
  --reranking_json "${RERANK_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --base_name "${BASE_NAME}"
)

if [[ "${LIMIT}" != "0" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

if [[ -n "${QUERY_OVERRIDE}" ]]; then
  CMD+=(--query_override "${QUERY_OVERRIDE}")
fi

echo "[gt] Generating SFT (user-only) from: ${RERANK_JSON}"
"${CMD[@]}"

OUT_FILE="${OUTPUT_DIR}/${BASE_NAME}_sft_llamafactory_gt.json"
if [[ ! -f "${OUT_FILE}" ]]; then
  echo "❌ Expected output file not found: ${OUT_FILE}" >&2
  exit 1
fi

# Optional push to Hugging Face
PUSH_TO_HF="${PUSH_TO_HF:-true}"
if [[ "${PUSH_TO_HF,,}" == "true" ]]; then
  HF_REPO_ID="${HF_REPO_ID:-happy8825/activitynet_validset_all}"
  HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-$(basename "${OUT_FILE}")}"
  echo "[gt] Pushing to HF dataset: ${HF_REPO_ID}:${HF_PATH_IN_REPO}"

  # Prefer CLI. Install if missing, then attempt upload.
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[gt] huggingface-cli not found. Installing huggingface_hub…"
    python - <<'PY'
import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'huggingface_hub'])
PY
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    # Ensure repo exists (idempotent) then upload file.
    huggingface-cli repo create "${HF_REPO_ID}" --repo-type dataset || true
    huggingface-cli upload "${HF_REPO_ID}" "${OUT_FILE}" "${HF_PATH_IN_REPO}" --repo-type dataset \
      && echo "[gt] Uploaded via CLI." || echo "[gt] CLI upload reported a non-zero exit code."
  else
    # Fallback to Python uploader if CLI still unavailable
    python "${ROOT_DIR}/SFT/sft_trainer/LLaMA-Factory/scripts/push_hf_dataset.py" \
      --file "${OUT_FILE}" \
      --repo-id "${HF_REPO_ID}" \
      --path-in-repo "${HF_PATH_IN_REPO}"
  fi
fi

echo "[gt] Done. Output: ${OUT_FILE}"
