#!/usr/bin/env bash
set -euo pipefail

# Re-encode ECVA videos to avoid zero-frame decoding issues.
# - Default targets: abnormal_video, normal_video, after_incident
# - Re-encodes with H.264 (libx264), fast preset, CRF 23, +faststart
# - Skips existing outputs unless --overwrite
# - Safe in-place when IN_DIR == OUT_DIR (writes to temp and moves)
#
# Usage:
#   bash clean_ecva.bash [--in-dir DIR] [--out-dir DIR] [--workers N]
#                        [--subdirs CSV] [--overwrite] [--keep-audio]
#
# Examples:
#   # Re-encode cropped clips in-place (recommended):
#   bash clean_ecva.bash --in-dir /hub_data4/seohyun/ecva --out-dir /hub_data4/seohyun/ecva --workers 8
#
#   # Only abnormal clips to a separate folder:
#   bash clean_ecva.bash --subdirs abnormal_video --out-dir /hub_data4/seohyun/ecva_clean --workers 6

IN_DIR="/hub_data4/seohyun/ecva"
OUT_DIR="/hub_data4/seohyun/ecva"
WORKERS="30"
SUBDIRS="abnormal_video,normal_video,after_incident"
OVERWRITE=0
KEEP_AUDIO=0
ONLY_BROKEN=0

usage() {
  echo "Usage: $0 [--in-dir DIR] [--out-dir DIR] [--workers N] [--subdirs CSV] [--overwrite] [--keep-audio] [--only-broken]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir) IN_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --subdirs) SUBDIRS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --keep-audio) KEEP_AUDIO=1; shift ;;
    --only-broken) ONLY_BROKEN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] ffmpeg not found in PATH. Please install ffmpeg." >&2
  exit 1
fi
if [[ "$ONLY_BROKEN" == "1" ]] && ! command -v ffprobe >/dev/null 2>&1; then
  echo "[!] ffprobe not found in PATH but --only-broken specified. Please install ffmpeg/ffprobe." >&2
  exit 1
fi

echo "[+] IN_DIR=$IN_DIR"
echo "[+] OUT_DIR=$OUT_DIR"
echo "[+] WORKERS=$WORKERS"
echo "[+] SUBDIRS=$SUBDIRS"
echo "[+] OVERWRITE=$OVERWRITE (1=overwrite existing)"
echo "[+] KEEP_AUDIO=$KEEP_AUDIO (1=keep audio tracks)"
echo "[+] ONLY_BROKEN=$ONLY_BROKEN (1=probe and re-encode only broken clips)"

IFS=',' read -r -a SUBDIR_ARR <<< "$SUBDIRS"

process_one() {
  local SRC="$1"
  local IN_ROOT="$2"
  local OUT_ROOT="$3"

  local REL="${SRC#${IN_ROOT}/}"
  local BASE_NOEXT="${REL%.*}"
  local DST_DIR="${OUT_ROOT}/$(dirname "$REL")"
  local DST="${OUT_ROOT}/${BASE_NOEXT}.mp4"

  mkdir -p "$DST_DIR"

  if [[ -f "$DST" && "$OVERWRITE" != "1" ]]; then
    echo "[=] Skip existing: $DST"
    return 0
  fi

  # If only re-encoding broken files, probe frame count first
  if [[ "$ONLY_BROKEN" == "1" ]]; then
    local frames
    frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
      -of default=nokey=1:noprint_wrappers=1 "$SRC" 2>/dev/null || true)
    # Treat as healthy if frames is a positive integer
    if [[ -n "$frames" && "$frames" != "N/A" ]]; then
      if [[ "$frames" =~ ^[0-9]+$ ]] && [[ "$frames" -ge 1 ]]; then
        echo "[=] Healthy (frames=$frames): $SRC"
        return 0
      fi
    fi
    echo "[!] Broken or unknown frames (='$frames'), re-encoding: $SRC"
  fi

  # Safe in-place: write to a temp file then move
  local TMP_DST
  if [[ "$SRC" == "$DST" ]]; then
    TMP_DST="$(mktemp --suffix .mp4 "${DST}.XXXXXX")"
  else
    TMP_DST="$DST"
  fi

  echo "[+] Re-encode: $SRC -> $DST"

  if [[ "$KEEP_AUDIO" == "1" ]]; then
    ffmpeg -hide_banner -loglevel error -nostdin \
      -y \
      -err_detect ignore_err \
      -i "$SRC" \
      -c:v libx264 -preset fast -crf 23 \
      -movflags +faststart \
      -pix_fmt yuv420p \
      "$TMP_DST"
  else
    ffmpeg -hide_banner -loglevel error -nostdin \
      -y \
      -err_detect ignore_err \
      -i "$SRC" \
      -an \
      -c:v libx264 -preset fast -crf 23 \
      -movflags +faststart \
      -pix_fmt yuv420p \
      "$TMP_DST"
  fi

  if [[ "$TMP_DST" != "$DST" ]]; then
    mv -f "$TMP_DST" "$DST"
  fi
}

export -f process_one
export IN_DIR OUT_DIR OVERWRITE KEEP_AUDIO

found_any=0
for sd in "${SUBDIR_ARR[@]}"; do
  SRC_DIR="$IN_DIR/$sd"
  if [[ -d "$SRC_DIR" ]]; then
    found_any=1
    echo "[+] Scanning: $SRC_DIR"
    find "$SRC_DIR" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' \) -print0 |
      xargs -0 -n1 -P"$WORKERS" bash -lc 'process_one "$3" "$1" "$2"' _ "$IN_DIR" "$OUT_DIR"
  else
    echo "[=] Skip missing subdir: $SRC_DIR"
  fi
done

# If no known subdir found, process IN_DIR itself
if [[ "$found_any" == "0" ]]; then
  echo "[=] No known subdirs present. Processing IN_DIR directly: $IN_DIR"
  find "$IN_DIR" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' \) -print0 |
    xargs -0 -n1 -P"$WORKERS" bash -lc 'process_one "$3" "$1" "$2"' _ "$IN_DIR" "$OUT_DIR"
fi

echo "[+] Done."
