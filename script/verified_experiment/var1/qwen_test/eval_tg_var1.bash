#!/usr/bin/env bash

# Evaluate ground_top1 JSONL and write metrics JSON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-${ROOT_DIR}/var1_exp/agent_outputs/qwen_var1_tg.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT_DIR}/var1_exp/agent_outputs/qwen_var1_tg.metrics.json}"

echo "Input   : ${INPUT_JSONL}"
echo "Output  : ${OUTPUT_JSON}"

python3 "${ROOT_DIR}/var1_exp/agent/eval_ground_top1_metrics.py" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSON}"
