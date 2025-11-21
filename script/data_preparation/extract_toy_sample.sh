#!/usr/bin/env bash
set -euo pipefail

# Default values; edit here or override via CLI arguments.
SOURCE_JSON_DEFAULT="/hub_data1/seohyun/hub_data/anet_ret_train.json"
COUNT_DEFAULT=100
OUTPUT_JSON_DEFAULT="/home/seohyun/video_retrieval/toy.json"

usage() {
  cat <<'EOF' >&2
Usage:
  extract_toy_sample.sh [source_json] [count] [output_json] [extra python args...]

Defaults are defined at the top of this script. Additional arguments are passed
through to extract_toy_sample.py (e.g., --shuffle --seed 42).
EOF
}

if [[ $# -ge 1 && ( $1 == "-h" || $1 == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -ge 1 ]]; then
  SOURCE_JSON=$1
  shift
else
  SOURCE_JSON=$SOURCE_JSON_DEFAULT
fi

if [[ $# -ge 1 ]]; then
  COUNT_VALUE=$1
  shift
else
  COUNT_VALUE=$COUNT_DEFAULT
fi

if [[ $# -ge 1 ]]; then
  OUTPUT_JSON=$1
  shift
else
  OUTPUT_JSON=$OUTPUT_JSON_DEFAULT
fi

if [[ -z "$SOURCE_JSON" || -z "$OUTPUT_JSON" ]]; then
  usage
  exit 1
fi

if ! [[ "$COUNT_VALUE" =~ ^[0-9]+$ ]]; then
  echo "Count must be a non-negative integer" >&2
  exit 1
fi

EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${ROOT_DIR}/SFT/LLaMA-Factory/util/extract_toy_sample.py}"

python3 "${PYTHON_SCRIPT}" "$SOURCE_JSON" "$OUTPUT_JSON" --count "$COUNT_VALUE" "${EXTRA_ARGS[@]}"
