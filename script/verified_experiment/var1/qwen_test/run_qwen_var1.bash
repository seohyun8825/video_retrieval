#!/usr/bin/env bash

# Run the Qwen vLLM agent on retrieval outputs (top-1 grounding or rerank@K + grounding).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
AGENT_PY="${ROOT_DIR}/var1_exp/agent/base_agent.py"

# I/O paths (edit as needed)
RANKINGS="${ROOT_DIR}/var1_exp/retriever_outputs/activitynet_fig_val_2_rzen/per_query_rankings.jsonl"
VIDEO_INDEX="${ROOT_DIR}/var1_exp/retriever_outputs/activitynet_fig_val_2_rzen/video_candidates.json"
OUTPUT_JSONL="${ROOT_DIR}/var1_exp/agent_outputs/qwen_var1_ground_gt.jsonl"

# Mode: ground_top1 | rerank5_ground | ground_gt (default)
MODE="${MODE:-ground_gt}"
RERANK_K="${RERANK_K:-5}"
LIMIT="${LIMIT:-0}"                # 0 = all queries
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-false}"
DEBUG="${DEBUG:-false}"

# vLLM / model settings
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8100/v1}"
API_KEY="${API_KEY:-}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-240}"
MAX_RETRIES="${MAX_RETRIES:-3}"
API_TEMPERATURE="${API_TEMPERATURE:-0.7}"
API_TOP_P="${API_TOP_P:-0.9}"
API_TOP_K="${API_TOP_K:--1}"
API_MAX_NEW_TOKENS="${API_MAX_NEW_TOKENS:-1024}"
API_PRESENCE_PENALTY="${API_PRESENCE_PENALTY:-1.0}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-}"
READ_TIMEOUT="${READ_TIMEOUT:-600}"

# Video metadata hints (keep consistent with server limits)
VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:-30}"
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-52144}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-0}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

echo "Mode            : ${MODE}"
echo "Rankings        : ${RANKINGS}"
echo "Video index     : ${VIDEO_INDEX}"
echo "Output JSONL    : ${OUTPUT_JSONL}"
echo "Model repo      : ${MODEL_REPO}"
echo "API base        : ${API_BASE}"
echo "Rerank@K        : ${RERANK_K}"
echo "Limit queries   : ${LIMIT}"
echo "Debug logging   : ${DEBUG}"
echo "Connect timeout : ${CONNECT_TIMEOUT:-none}"
echo "Read timeout    : ${READ_TIMEOUT:-none}"

if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${OUTPUT_JSONL}" ]]; then
  echo "[skip] ${OUTPUT_JSONL} exists"
  exit 0
fi

CMD=(
  python3 "${AGENT_PY}"
  --rankings "${RANKINGS}"
  --video_index "${VIDEO_INDEX}"
  --mode "${MODE}"
  --output_jsonl "${OUTPUT_JSONL}"
  --model_repo "${MODEL_REPO}"
  --api_base "${API_BASE}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --temperature "${API_TEMPERATURE}"
  --top_p "${API_TOP_P}"
  --top_k "${API_TOP_K}"
  --max_new_tokens "${API_MAX_NEW_TOKENS}"
  --presence_penalty "${API_PRESENCE_PENALTY}"
  --video_num_frames "${VIDEO_NUM_FRAMES}"
  --video_total_pixels "${VIDEO_TOTAL_PIXELS}"
  --video_min_pixels "${VIDEO_MIN_PIXELS}"
  --video_fps "${VIDEO_FPS}"
  --rerank_k "${RERANK_K}"
)
if [[ -n "${CONNECT_TIMEOUT}" ]]; then
  CMD+=(--connect_timeout "${CONNECT_TIMEOUT}")
fi
if [[ -n "${READ_TIMEOUT}" ]]; then
  CMD+=(--read_timeout "${READ_TIMEOUT}")
fi

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api_key "${API_KEY}")
fi
if [[ "${LIMIT}" -gt 0 ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${DEBUG}" == "true" ]]; then
  CMD+=(--debug)
fi

"${CMD[@]}"
