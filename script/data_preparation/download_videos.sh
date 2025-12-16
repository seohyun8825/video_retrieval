#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults if you want to hardcode paths.
JSON_PATH_DEFAULT="/hub_data2/dohwan/data/retrieval/activitynet/anet_ret_val_1.json"
DEST_DIR_DEFAULT="/hub_data3/seohyun/activitynet/videos"

usage() {
  cat <<'EOF' >&2
Usage:
  download_videos.sh [json_path] [destination_dir] [extra python args...]

If json_path or destination_dir are omitted, the defaults in this script are used.
By default, downloads skip files that already exist and a filtered JSON
(<json_path>._existing.json) is written containing only the videos that exist locally.
EOF
}

if [[ $# -ge 1 && ( $1 == "-h" || $1 == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -ge 1 ]]; then
  JSON_PATH=$1
  shift
else
  JSON_PATH="$JSON_PATH_DEFAULT"
fi

if [[ $# -ge 1 ]]; then
  DEST_DIR=$1
  shift
else
  DEST_DIR="$DEST_DIR_DEFAULT"
fi

if [[ -z "$JSON_PATH" || -z "$DEST_DIR" ]]; then
  usage
  exit 1
fi

EXTRA_ARGS=("$@")

if [[ "$JSON_PATH" == *.json ]]; then
  EXISTING_JSON="${JSON_PATH%.json}._existing.json"
else
  EXISTING_JSON="${JSON_PATH}._existing.json"
fi

add_skip_flag=true
for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --skip-existing|--no-skip-existing)
      add_skip_flag=false
      break
      ;;
  esac
done
if $add_skip_flag; then
  EXTRA_ARGS=(--skip-existing "${EXTRA_ARGS[@]}")
fi

add_existing_arg=true
for arg in "${EXTRA_ARGS[@]}"; do
  case "$arg" in
    --existing-json|--existing-json=*)
      add_existing_arg=false
      break
      ;;
  esac
done
if $add_existing_arg; then
  EXTRA_ARGS=(--existing-json "$EXISTING_JSON" "${EXTRA_ARGS[@]}")
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${ROOT_DIR}/SFT/sft_trainer/LLaMA-Factory/util/download_videos_from_json.py}"

python3 "${PYTHON_SCRIPT}" "$JSON_PATH" "$DEST_DIR" "${EXTRA_ARGS[@]}"
