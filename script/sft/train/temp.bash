#!/usr/bin/env bash

set -euo pipefail

# One‑shot trainer for LLaMA‑Factory with handy overrides and
# auto‑registering a HF dataset into dataset_info.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Defaults (override via env or CLI args below)
DEFAULT_CONFIG_PATH="${ROOT_DIR}/script/sft/train/config/train_full/temp.yaml"

# Lightweight CLI parser to support --analyze_token while keeping
# backward compatibility with the original positional CONFIG_PATH.
ANALYZE_TOKEN=0
CONFIG_PATH=""
__args=("$@")
for ((i=0; i<${#__args[@]}; i++)); do
  a="${__args[$i]}"
  case "${a}" in
    --analyze_token)
      ANALYZE_TOKEN=1
      ;;
    -c|--config)
      # next token is config path if available
      if (( i + 1 < ${#__args[@]} )); then
        CONFIG_PATH="${__args[$((i+1))]}"
        ((i++))
      fi
      ;;
    --*)
      # ignore other flags (reserved)
      ;;
    *)
      # first non-flag is treated as CONFIG_PATH if not set yet
      if [[ -z "${CONFIG_PATH}" ]]; then
        CONFIG_PATH="${a}"
      fi
      ;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  CONFIG_PATH="${DEFAULT_CONFIG_PATH}"
fi
# GPU selection (comma separated)
GPU_DEVICES="0,1,2,3"  # e.g., "4,5"; empty = keep current

# Common overrides you asked to manage from bash
VIDEO_MAXLEN="${VIDEO_MAXLEN:-36}"
DATASET="${DATASET:-}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/script/sft/train/data_config}"
MEDIA_DIR="/hub_data3/seohyun"

LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
REPORT_TO="${REPORT_TO:-wandb}"

OUTPUT_DIR="${OUTPUT_DIR:-/hub_data3/seohyun/saves/ecva_instruct/full/sft}"
OUTPUT_DEBUG_DIR="${OUTPUT_DIR}/output_debug"

# Optional: register/override a dataset entry by only specifying HF_HUB_URL
# Example: HF_HUB_URL="happy8825/test_mllm_video_demo"
HF_HUB_URL="happy8825/train_ecva_clean_no_tag"

# If DATASET is empty and HF_HUB_URL is provided, infer dataset name from repo (last segment)
if [[ -z "${DATASET}" && -n "${HF_HUB_URL}" ]]; then
  DATASET="$(basename "${HF_HUB_URL}")"
fi

echo "Config       : ${CONFIG_PATH}"
echo "Dataset      : ${DATASET:-<none>}"
echo "Dataset dir  : ${DATASET_DIR}"
echo "Media dir    : ${MEDIA_DIR}"
echo "HF hub url   : ${HF_HUB_URL:-<leave as is>}"
echo "video_maxlen : ${VIDEO_MAXLEN}"
echo "logging_steps: ${LOGGING_STEPS} | save_steps: ${SAVE_STEPS} | report_to: ${REPORT_TO}"
echo "output_dir   : ${OUTPUT_DIR}"
echo "output_debug : ${OUTPUT_DEBUG_DIR} (analyze_token=${ANALYZE_TOKEN})"
echo "GPU devices  : ${GPU_DEVICES:-<inherit>}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${DATASET}" ]]; then
  echo "❌ DATASET is empty and HF_HUB_URL not provided to infer from. Set DATASET=... or HF_HUB_URL=org/repo." >&2
  exit 1
fi

mkdir -p "${DATASET_DIR}"
DATASET_INFO_JSON="${DATASET_DIR}/dataset_info.json"

# Ensure dataset_info.json exists
if [[ ! -f "${DATASET_INFO_JSON}" ]]; then
  echo "{}" > "${DATASET_INFO_JSON}"
fi

# If HF_HUB_URL is provided, (up)insert the dataset entry with default schema.
if [[ -n "${HF_HUB_URL}" ]]; then
  python3 - "$DATASET_INFO_JSON" "$DATASET" "$HF_HUB_URL" << 'PY'
import json, sys

path, name, hub = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
if not isinstance(data, dict):
    data = {}

entry = {
    "hf_hub_url": hub,
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "videos": "videos"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}

# upsert
data[name] = {**data.get(name, {}), **entry}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"[dataset_info.json] upserted entry '{name}' with hf_hub_url='{hub}' -> {path}")
PY
fi

# Optional: enable in-trainer token analysis (first 3 samples)
if [[ "${ANALYZE_TOKEN}" == "1" ]]; then
  echo "[token-analyze] Enabled (trainer hook). Logs: ${OUTPUT_DEBUG_DIR}/first3_from_trainer.jsonl"
  mkdir -p "${OUTPUT_DEBUG_DIR}"
fi

# Configure GPUs
if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
  # Derive nproc from the number of listed devices
  IFS=',' read -r -a __gpus <<< "${GPU_DEVICES}"
  export NPROC_PER_NODE="${#__gpus[@]}"
fi

# Launch training with overrides merged into YAML (OmegaConf CLI style)
set -x
LMF_ANALYZE_TOKEN="${ANALYZE_TOKEN}" LMF_ANALYZE_MAX_SAMPLES=3 \
  llamafactory-cli train "${CONFIG_PATH}" \
  dataset="${DATASET}" \
  dataset_dir="${DATASET_DIR}" \
  media_dir="${MEDIA_DIR}" \
  video_maxlen="${VIDEO_MAXLEN}" \
  logging_steps="${LOGGING_STEPS}" \
  save_steps="${SAVE_STEPS}" \
  report_to="${REPORT_TO}" \
  output_dir="${OUTPUT_DIR}"
set +x
