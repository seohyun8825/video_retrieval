#!/usr/bin/env bash

set -euo pipefail

# One-time environment setup for VLM2Vec embedder.
# - Creates conda env `vlm2vec`
# - Clones TIGER-AI-Lab/VLM2Vec under the extractor folder
# - Installs requirements

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SFT_DIR="${ROOT_DIR}/SFT/sft_data_generation"
PREPROCESS_DIR="${SFT_DIR}/preprocess/extract_similarity_matrix"

ENV_NAME="${ENV_NAME:-vlm2vec}"
VLM2VEC_REPO_DIR="${PREPROCESS_DIR}/VLM2Vec"
VLM2VEC_SUBDIR="${VLM2VEC_REPO_DIR}"

echo "Working dir  : ${SCRIPT_DIR}"
echo "Project root : ${ROOT_DIR}"
echo "Extractor dir: ${PREPROCESS_DIR}"
echo "Conda env    : ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda not found in PATH. Please install Miniconda/Anaconda." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env: ${ENV_NAME} (python=3.10)"
  conda create -y -n "${ENV_NAME}" python=3.10
else
  echo "Conda env already exists: ${ENV_NAME}"
fi

conda activate "${ENV_NAME}"
python -V
pip -V

mkdir -p "${PREPROCESS_DIR}"
if [[ ! -d "${VLM2VEC_REPO_DIR}" ]]; then
  echo "Cloning VLM2Vec into ${VLM2VEC_REPO_DIR}"
  git clone https://github.com/TIGER-AI-Lab/VLM2Vec.git "${VLM2VEC_REPO_DIR}"
else
  echo "VLM2Vec already present at ${VLM2VEC_REPO_DIR}"
fi

REQ_FILE="${VLM2VEC_SUBDIR}/requirements.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "❌ requirements.txt not found at ${REQ_FILE}" >&2
  exit 1
fi

echo "Installing Python dependencies from ${REQ_FILE}"
python -m pip install --upgrade pip
pip install -r "${REQ_FILE}"

echo "✅ VLM2Vec environment setup complete. Activate with: conda activate ${ENV_NAME}"

