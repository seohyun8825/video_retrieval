#!/usr/bin/env bash
set -euo pipefail

# 입력 / 출력 디렉토리
IN_DIR="/hub_data4/seohyun/ecva/after_incident"
OUT_DIR="/hub_data3/seohyun/ecva/after_incident"

# 동시에 돌릴 워커 수 (원하면 nproc 사용)
WORKERS="80"          # 또는 숫자 고정: WORKERS=8

mkdir -p "$OUT_DIR"

export IN_DIR OUT_DIR

process_one() {
  SRC="$1"
  REL="${SRC#$IN_DIR/}"
  DST="$OUT_DIR/${REL%.*}.mp4"

  mkdir -p "$(dirname "$DST")"

  echo "Processing: $SRC -> $DST"

  ffmpeg \
    -v error \
    -err_detect ignore_err \
    -i "$SRC" \
    -an \
    -c:v libx264 \
    -preset fast \
    -crf 23 \
    -movflags +faststart \
    "$DST"
}

export -f process_one

find "$IN_DIR" -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.avi' \) -print0 |
  xargs -0 -n1 -P"$WORKERS" bash -lc 'process_one "$@"' _
