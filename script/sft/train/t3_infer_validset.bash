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

# Hard-coded defaults for current workflow
MODEL_REPO="happy8825/sft-20251121"
DATASET_REPO="happy8825/anet_sampled_sft_valid"
OUTPUT_JSON="/home/seohyun/vid_understanding/video_retrieval/output/output_sft-20251121.json"

MEDIA_BASE="/hub_data2/dohwan/data/retrieval"   # joined with sample video paths if not absolute
TEMPLATE="qwen3_vl"
VIDEO_FPS="2.0"
VIDEO_MAXLEN="16"
MAX_SAMPLES="50"

PUSH_HF="true"

# Streaming JSONL controls
STREAM_JSONL="${STREAM_JSONL:-true}"
TRUNCATE_JSONL="${TRUNCATE_JSONL:-true}"
JSONL_PATH="${JSONL_PATH:-${OUTPUT_JSON%.json}.jsonl}"

# Derive default output repo/file from OUTPUT_JSON
__out_base="$(basename "${OUTPUT_JSON}")"         # e.g., output_sft-20251121.json
HF_OUT_FILE="${__out_base}"
HF_OUT_REPO="${__out_base%.*}"     # strip extension

echo "Model repo   : ${MODEL_REPO}"
echo "Dataset repo : ${DATASET_REPO}"
echo "Output json  : ${OUTPUT_JSON}"
echo "Media base   : ${MEDIA_BASE}"
echo "Template     : ${TEMPLATE} | fps=${VIDEO_FPS} | maxlen=${VIDEO_MAXLEN}"
echo "Max samples  : ${MAX_SAMPLES}"
echo "Push HF      : ${PUSH_HF} -> ${HF_OUT_REPO}/${HF_OUT_FILE}"

# GPUs / concurrency
GPU_DEVICES="7"
MAX_CONCURRENT="2"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
export MAX_CONCURRENT

mkdir -p "$(dirname "${OUTPUT_JSON}")"

python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_validset.py" \
  --model_repo "${MODEL_REPO}" \
  --dataset_repo "${DATASET_REPO}" \
  --output_json "${OUTPUT_JSON}" \
  --media_base "${MEDIA_BASE}" \
  --template "${TEMPLATE}" \
  --video_fps "${VIDEO_FPS}" \
  --video_maxlen "${VIDEO_MAXLEN}" \
  --max_samples "${MAX_SAMPLES}" \
  --concurrency "${MAX_CONCURRENT}" \
  $( [[ "${PUSH_HF}" == "true" ]] && echo --push_hf ) \
  --hf_out_repo "${HF_OUT_REPO}" \
  --hf_out_file "${HF_OUT_FILE}" \
  $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
  --jsonl_path "${JSONL_PATH}" \
  $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl )

echo "✅ Inference finished: ${OUTPUT_JSON}"

if [[ "${PUSH_HF}" == "true" ]]; then
  # Determine HF user
  HF_USER="${HF_USER:-}"
  if [[ -z "${HF_USER}" ]]; then
    set +e
    HF_USER=$(python3 - << 'PY'
try:
    from huggingface_hub import HfApi, HfFolder
    token = HfFolder.get_token()
    if not token:
        raise SystemExit(1)
    info = HfApi().whoami(token)
    print(info.get('name',''))
except Exception:
    raise SystemExit(2)
PY
    )
    set -e
    if [[ -z "${HF_USER}" ]] && command -v huggingface-cli >/dev/null 2>&1; then
      set +e
      HF_USER=$(huggingface-cli whoami -s 2>/dev/null | head -n1)
      set -e
    fi
  fi
  if [[ -z "${HF_USER}" ]]; then
    echo " Could not determine HF_USER; skipping push." >&2
    exit 0
  fi

  REPO_ID="${HF_USER}/${HF_OUT_REPO}"
  echo "Pushing results to HF dataset: ${REPO_ID} as ${HF_OUT_FILE}"
  huggingface-cli repo create "${REPO_ID}" --type dataset --exist-ok >/dev/null
  huggingface-cli upload "${REPO_ID}" "${OUTPUT_JSON}" "${HF_OUT_FILE}" --repo-type dataset --commit-message "Add inference results from ${MODEL_REPO} on ${DATASET_REPO}"
  echo " Uploaded to https://huggingface.co/datasets/${REPO_ID}"
fi
