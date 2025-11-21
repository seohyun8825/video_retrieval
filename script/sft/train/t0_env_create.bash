#!/usr/bin/env bash

set -euo pipefail

# Create and prepare a Conda env for LLaMA-Factory training
# - Env name: llama_factory (override via ENV_NAME)
# - Python: 3.10
# - Installs extras: [torch,metrics,qwen,deepspeed]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/SFT/LLaMA-Factory"

ENV_NAME="${ENV_NAME:-llama_factory}"

echo "Project root : ${ROOT_DIR}"
echo "LLaMA-Factory: ${LLAMA_DIR}"
echo "Env name     : ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install Anaconda/Miniconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if missing
ENV_DIR="$(conda info --base)/envs/${ENV_NAME}"
if [[ ! -d "${ENV_DIR}" ]]; then
  echo "Creating Conda environment ${ENV_NAME} with Python 3.10..."
  conda create -y -n "${ENV_NAME}" python=3.10
else
  echo "Conda environment already exists: ${ENV_DIR}"
fi

echo "Activating environment ${ENV_NAME}..."
conda activate "${ENV_NAME}"

echo "Changing directory to ${LLAMA_DIR}"
cd "${LLAMA_DIR}"

echo "Installing LLaMA-Factory with extras (torch,metrics,qwen,deepspeed)..."
pip install -e ".[torch,metrics,qwen,deepspeed]" --no-build-isolation

echo "✅ Environment ready. Activate later with: conda activate ${ENV_NAME}"
