#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   clean_from_json_list.bash anet_ret_val_1.json /in/base/videos /out/base/videos [workers]
#
# Reads the HF-style JSON list with a `video` field (e.g., v_xxx.mp4),
# resolves to an existing source by trying mp4 then mkv, and transcodes
# each to MP4 (no audio) using the same ffmpeg flags as clean_frame.bash.

JSON_PATH="${1:-}"
IN_DIR="${2:-}"
OUT_DIR="${3:-}"
WORKERS="${4:-64}"

if [[ -z "${JSON_PATH}" || -z "${IN_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: $0 anet_ret_val_1.json /in/base/videos /out/base/videos [workers]" >&2
  exit 2
fi

# jq가 없으면 Python으로 JSON을 읽는 폴백을 사용합니다.

mkdir -p "${OUT_DIR}"

# Build a NUL-separated list of src,dst pairs. For each entry in JSON, try
# mp4 first; if missing, try mkv. Destination is always .mp4.
gen_pairs() {
  if command -v jq >/dev/null 2>&1; then
    jq -r '.[].video' "${JSON_PATH}"
  else
    python3 - "${JSON_PATH}" << 'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    sys.stderr.write(f"[error] failed to read {path}: {e}\n")
    sys.exit(1)
seen = set()
if isinstance(data, list):
    for it in data:
        if isinstance(it, dict) and 'video' in it:
            v = it['video']
            if isinstance(v, str) and v not in seen:
                seen.add(v)
                print(v)
else:
    try:
        for it in data.values():
            if isinstance(it, dict) and 'video' in it:
                v = it['video']
                if isinstance(v, str) and v not in seen:
                    seen.add(v)
                    print(v)
    except Exception:
        pass
PY
  fi |
  sort -u | while read -r name; do
    base_noext="${name%.*}"
    src_mp4="${IN_DIR}/${base_noext}.mp4"
    src_mkv="${IN_DIR}/${base_noext}.mkv"
    if [[ -f "${src_mp4}" ]]; then
      src="${src_mp4}"
    elif [[ -f "${src_mkv}" ]]; then
      src="${src_mkv}"
    else
      echo "[skip] not found: ${name} (tried ${src_mp4} and ${src_mkv})" >&2
      continue
    fi
    dst="${OUT_DIR}/${base_noext}.mp4"
    printf '%s\0%s\0' "${src}" "${dst}"
  done
}

export OUT_DIR

transcode_pair() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  echo ">> ${src} -> ${dst}"
  ffmpeg \
    -v error \
    -err_detect ignore_err \
    -i "${src}" \
    -an \
    -c:v libx264 \
    -preset fast \
    -crf 23 \
    -movflags +faststart \
    "${dst}"
}

export -f transcode_pair

gen_pairs | xargs -0 -n2 -P "${WORKERS}" bash -lc 'transcode_pair "$@"' _

echo "Done. Output under: ${OUT_DIR}"
