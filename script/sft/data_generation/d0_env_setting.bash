#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PREPROCESS_DIR="${ROOT_DIR}/SFT/sft_data_generation/preprocess/extract_similarity_matrix"

echo "Working directory: ${PREPROCESS_DIR}"

if [[ ! -d "${PREPROCESS_DIR}/Video-ColBERT" ]]; then
  git clone https://github.com/yogesh-iitj/Video-ColBERT "${PREPROCESS_DIR}/Video-ColBERT"
else
  echo "Video-ColBERT already exists at ${PREPROCESS_DIR}/Video-ColBERT"
fi

ENV_NAME="${ENV_NAME:-video-colbert}"

echo "Creating Conda environment ${ENV_NAME} with Python 3.10..."
conda create -y -n "${ENV_NAME}" python=3.10

echo "Activating environment and installing dependencies..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install torch torchvision numpy pillow einops decord
pip install git+https://github.com/openai/CLIP.git
pip install openai
pip install matplotlib
pip install opencv-python

echo "Environment setup complete. Activate later with: conda activate ${ENV_NAME}"
