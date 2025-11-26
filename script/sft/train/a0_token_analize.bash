#!/usr/bin/env bash

set -euo pipefail

# Analyze per-sample token usage around <video> placeholders for Qwen3-VL SFT data.
# Outputs JSONL logs with counts per sample.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Defaults (override via env or CLI args)
DEFAULT_CONFIG_PATH="${ROOT_DIR}/script/sft/train/config/train_full/qwen3vl_2b_thinking_sft.yaml"
CONFIG_PATH="${1:-${DEFAULT_CONFIG_PATH}}"

# The dataset to inspect. If empty, infer from HF_HUB_URL below
DATASET="${DATASET:-}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/script/sft/train/data_config}"
MEDIA_DIR="${MEDIA_DIR:-/hub_data2/dohwan/data/retrieval}"

# Inference of dataset name from HF hub repo id (same logic as t1_train.bash)
HF_HUB_URL="${HF_HUB_URL:-happy8825/anet_caption_concat_sft}"
if [[ -z "${DATASET}" && -n "${HF_HUB_URL}" ]]; then
  DATASET="$(basename "${HF_HUB_URL}")"
fi

# Video processing overrides
VIDEO_MAXLEN="${VIDEO_MAXLEN:-64}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"

# Where to write logs
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output/token_analysis}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"

echo "Config      : ${CONFIG_PATH}"
echo "Dataset     : ${DATASET:-<none>} (hf=${HF_HUB_URL:-<n/a>})"
echo "Dataset dir : ${DATASET_DIR}"
echo "Media dir   : ${MEDIA_DIR}"
echo "video_maxlen: ${VIDEO_MAXLEN} | fps=${VIDEO_FPS}"
echo "Output dir  : ${OUT_DIR}"
echo "Max samples : ${MAX_SAMPLES}"

if [[ -z "${DATASET}" ]]; then
  echo "❌ DATASET is empty and HF_HUB_URL not provided to infer from. Set DATASET=... or HF_HUB_URL=org/repo." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

PYTHONPATH_ADD="${ROOT_DIR}/SFT/sft_trainer/LLaMA-Factory/src"
export PYTHONPATH="${PYTHONPATH_ADD}:${PYTHONPATH:-}"

# Ensure dataset_info.json contains this dataset (mirror logic from t1_train.bash)
mkdir -p "${DATASET_DIR}"
DATASET_INFO_JSON="${DATASET_DIR}/dataset_info.json"
if [[ ! -f "${DATASET_INFO_JSON}" ]]; then
  echo "{}" > "${DATASET_INFO_JSON}"
fi

python3 - "${DATASET_INFO_JSON}" "${DATASET}" "${HF_HUB_URL}" << 'PY'
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

data[name] = {**data.get(name, {}), **entry}
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"[dataset_info.json] upserted entry '{name}' with hf_hub_url='{hub}' -> {path}")
PY

python3 "${SCRIPT_DIR}/a0_token_analyze.py" \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --media_dir "${MEDIA_DIR}" \
  --video_maxlen "${VIDEO_MAXLEN}" \
  --video_fps "${VIDEO_FPS}" \
  --out_dir "${OUT_DIR}" \
  --max_samples "${MAX_SAMPLES}"

echo "Done. Logs written under: ${OUT_DIR}"
