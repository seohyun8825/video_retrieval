#!/usr/bin/env bash
set -euo pipefail

# Hardcoded settings (edit here)
#JSONL_PATH="/home/seohyun/vid_understanding/video_retrieval/video_retrieval/qweninstruct_tuned/qweninstruct_tuned.jsonl"
JSONL_PATH="/home/seohyun/vid_understanding/video_retrieval/video_retrieval/output_vllm/qweninstruct2b_infer_result.jsonl"
LIMIT=8700

# Optional overrides via positional args
# Usage: ./eval_jsonl.bash [JSONL_PATH] [LIMIT]
if [[ ${1-} ]]; then JSONL_PATH="$1"; fi
if [[ ${2-} ]]; then LIMIT="$2"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/eval_jsonl.py" "$JSONL_PATH" --limit "$LIMIT"
