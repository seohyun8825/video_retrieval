#!/usr/bin/env bash
set -euo pipefail

# Scan ECVA folders and re-encode videos while dropping corrupted frames.
# - Detects issues via ffprobe/ffmpeg and re-encodes only when necessary.
# - Re-encode uses: H.264, CFR at --fps, drop corrupt packets.
#
# Defaults:
#   DIRS=/hub_data4/seohyun/ecva/abnormal_video,/hub_data4/seohyun/ecva/after_incident,/hub_data4/seohyun/ecva/normal_video
#   FPS=2
#   WORKERS=32
#   ONLY_BROKEN=1   (skip files that look OK)
#   DRY_RUN=0       (1 = just list files that would be re-encoded)
#   EXTRA_CODECS=av1 (also re-encode when codec is in this list)
#
# Usage:
#   bash fix_ecva_dirs_drop_corrupt.bash \
#     [--dirs dir1,dir2,...] [--fps 2] [--workers 32] [--no-only-broken] [--dry-run] [--extra-codecs av1,vp9]

DIRS="/hub_data4/seohyun/ecva/abnormal_video,/hub_data4/seohyun/ecva/after_incident,/hub_data4/seohyun/ecva/normal_video"
FPS=2
WORKERS=128
ONLY_BROKEN=1
DRY_RUN=0
EXTRA_CODECS="av1"
# When set, do NOT treat unknown frame count (<=0) as broken
SKIP_UNKNOWN=0

usage(){
  echo "Usage: $0 [--dirs dir1,dir2,...] [--fps N] [--workers N] [--no-only-broken] [--dry-run] [--extra-codecs av1,vp9] [--skip-unknown]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dirs) DIRS="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --no-only-broken) ONLY_BROKEN=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --extra-codecs) EXTRA_CODECS="$2"; shift 2 ;;
    --skip-unknown) SKIP_UNKNOWN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -r -a DIR_ARR <<< "$DIRS"
echo "[+] DIRS=${DIRS} | FPS=${FPS} | WORKERS=${WORKERS} | ONLY_BROKEN=${ONLY_BROKEN} | DRY_RUN=${DRY_RUN} | EXTRA_CODECS=${EXTRA_CODECS}"

if ! command -v ffprobe >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] ffprobe/ffmpeg not found in PATH" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
LIST="$TMP_DIR/files.txt"
> "$LIST"

# Gather candidate files
for d in "${DIR_ARR[@]}"; do
  if [[ -d "$d" ]]; then
    find "$d" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.avi' \) -print >> "$LIST"
  else
    echo "[=] Skip missing dir: $d"
  fi
done

TOTAL=$(wc -l < "$LIST" | awk '{print $1}')
echo "[+] Found $TOTAL files to inspect"

EXTRA_SET="$(echo "$EXTRA_CODECS" | tr 'A-Z,' 'a-z\n' | awk 'NF{print}')"

# simple progress watcher (prints once per second when count changes)
progress_watch() {
  local total="$1"; shift
  local pfile="$1"; shift
  local label="${1:-progress}"
  local last=-1
  while true; do
    local curr=0
    if [[ -f "$pfile" ]]; then curr=$(wc -l < "$pfile" | awk '{print $1}'); fi
    if [[ "$curr" != "$last" ]]; then
      echo "[progress][$label] processed=${curr}/${total} remaining=$((total-curr))"
      last="$curr"
    fi
    sleep 1
  done
}

inspect_one() {
  local src="$1"
  # codec name
  local codec
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1 "$src" 2>/dev/null | tr 'A-Z' 'a-z' | head -n1 || true)
  # frame count (decoding)
  local nb
  nb=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$src" 2>/dev/null | head -n1 || true)
  local frames
  if [[ -z "$nb" || "$nb" == "N/A" ]]; then
    frames=-1
  else
    frames=$nb
  fi
  # quick decode test (error output non-empty => treat as broken)
  local errlog
  errlog=$(ffmpeg -v error -nostdin -threads 1 -i "$src" -map 0:v:0 -f null - 2>&1 | head -n 1 || true)

  # Decide whether to re-encode
  local need=0
  # codec trigger
  if echo "$EXTRA_SET" | grep -qx "$codec" 2>/dev/null; then
    need=1
  fi
  # decoding failure or corrupt
  if [[ $frames -le 0 ]]; then
    if [[ "$SKIP_UNKNOWN" != "1" ]]; then
      need=1
    fi
  fi
  if [[ -n "$errlog" ]]; then
    need=1
  fi

  if [[ $ONLY_BROKEN -eq 1 && $need -eq 0 ]]; then
    echo "SKIP|$src|codec=$codec|frames=$frames"; return 0
  fi
  echo "FIX |$src|codec=$codec|frames=$frames"; return 0
}

export -f inspect_one

echo "[+] Inspecting files ..."
INSPECT_LOG="$TMP_DIR/inspect.tsv"
PROG_INSPECT="$TMP_DIR/prog_inspect"
> "$INSPECT_LOG"
> "$PROG_INSPECT"
export PROG_INSPECT INSPECT_LOG SKIP_UNKNOWN EXTRA_SET ONLY_BROKEN FPS
progress_watch "$TOTAL" "$PROG_INSPECT" "inspect" & WATCH_I=$!
trap 'kill $WATCH_I 2>/dev/null || true; rm -rf "$TMP_DIR"' EXIT
cat "$LIST" | xargs -I{} -P "$WORKERS" bash -lc 'inspect_one "$1" >> "$INSPECT_LOG"; echo . >> "$PROG_INSPECT"' _ {}
kill $WATCH_I 2>/dev/null || true

TOFIX="$TMP_DIR/tofix.txt"
awk -F'|' '/^FIX /{print $2}' "$INSPECT_LOG" > "$TOFIX"
N_FIX=$(wc -l < "$TOFIX" | awk '{print $1}')
echo "[+] To re-encode: $N_FIX"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[=] DRY_RUN set. Listing files to re-encode:"
  cat "$TOFIX"
  exit 0
fi

if [[ "$N_FIX" -eq 0 ]]; then
  echo "[=] Nothing to do."
  exit 0
fi

echo "[+] Re-encoding with: drop corrupt packets @ fps=$FPS (libx264) ..."
reencode_one() {
  local src="$1"
  local tmp="${src}.tmp.mp4"
  echo "[re-encode] $src"
  ffmpeg -hide_banner -loglevel error -nostdin \
    -fflags +discardcorrupt -err_detect ignore_err \
    -i "$src" -an -c:v libx264 -preset fast -crf 23 -vf "fps=${FPS}" -vsync vfr -movflags +faststart \
    "$tmp" && mv -f "$tmp" "$src"
}
export -f reencode_one

PROG_TRANSCODE="$TMP_DIR/prog_transcode"
> "$PROG_TRANSCODE"
progress_watch "$N_FIX" "$PROG_TRANSCODE" "transcode" & WATCH_T=$!
trap 'kill $WATCH_T $WATCH_I 2>/dev/null || true; rm -rf "$TMP_DIR"' EXIT
xargs -I{} -P "$WORKERS" bash -lc 'reencode_one "$1"; echo . >> "$PROG_TRANSCODE"' _ {} < "$TOFIX"
kill $WATCH_T 2>/dev/null || true

echo "[+] Done. See details: $INSPECT_LOG"
