#!/usr/bin/env bash
set -euo pipefail

# Check ECVA videos for broken clips using ffprobe.
# - Classifies as BROKEN if nb_read_frames is empty/N\A or < MIN_FRAMES,
#   or if duration < MIN_DURATION (when MIN_DURATION > 0).
# - Scans subfolders: abnormal_video, normal_video, after_incident by default.
# - Runs in parallel and writes a TSV report.
#
# Usage:
#   bash check_ecva.bash [--in-dir DIR] [--subdirs CSV] [--workers N]
#                        [--min-frames N] [--min-duration S] [--report PATH]
#
# Examples:
#   bash check_ecva.bash --in-dir /hub_data4/seohyun/ecva --workers 16
#   bash check_ecva.bash --subdirs abnormal_video --min-frames 1 --report /tmp/ab_check.tsv

IN_DIR="/hub_data4/seohyun"
SUBDIRS="abnormal_video,normal_video,after_incident"
WORKERS="64"
MIN_FRAMES="1"
MIN_DURATION="0"
REPORT=""
VERBOSE=0

usage() {
  echo "Usage: $0 [--in-dir DIR] [--subdirs CSV] [--workers N] [--min-frames N] [--min-duration S] [--report PATH] [--verbose]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir) IN_DIR="$2"; shift 2 ;;
    --subdirs) SUBDIRS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --min-frames) MIN_FRAMES="$2"; shift 2 ;;
    --min-duration) MIN_DURATION="$2"; shift 2 ;;
    --report) REPORT="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "[!] ffprobe not found in PATH. Please install ffmpeg/ffprobe." >&2
  exit 1
fi

if [[ -z "$REPORT" ]]; then
  REPORT="$IN_DIR/_ecva_check_report.tsv"
fi

echo "[+] IN_DIR=$IN_DIR"
echo "[+] SUBDIRS=$SUBDIRS"
echo "[+] WORKERS=$WORKERS"
echo "[+] MIN_FRAMES=$MIN_FRAMES"
echo "[+] MIN_DURATION=$MIN_DURATION"
echo "[+] REPORT=$REPORT"
echo "[+] VERBOSE=$VERBOSE (1=print OK entries too)"

REPORT_DIR="$(dirname "$REPORT")"
mkdir -p "$REPORT_DIR"
REPORT_TMP="$(mktemp "${REPORT}.tmp.XXXXXX")"
echo -e "STATUS\tREL_PATH\tFRAMES\tDURATION" > "$REPORT_TMP"

IFS=',' read -r -a SUBDIR_ARR <<< "$SUBDIRS"

# Build file list first to know TOTAL
FILES_TMP="$(mktemp "${IN_DIR}/_ecva_files.XXXXXX")"
found_any=0
for sd in "${SUBDIR_ARR[@]}"; do
  SRC_DIR="$IN_DIR/$sd"
  if [[ -d "$SRC_DIR" ]]; then
    found_any=1
    echo "[+] Scanning: $SRC_DIR"
    find "$SRC_DIR" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.webm' \) -print0 >> "$FILES_TMP"
  else
    echo "[=] Skip missing subdir: $SRC_DIR"
  fi
done

if [[ "$found_any" == "0" ]]; then
  echo "[=] No known subdirs present. Processing IN_DIR directly: $IN_DIR"
  find "$IN_DIR" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.webm' \) -print0 >> "$FILES_TMP"
fi

TOTAL=$(tr -cd '\0' < "$FILES_TMP" | wc -c | awk '{print $1}')
echo "[+] TOTAL files=$TOTAL"

# Prepare counters and lock for progress updates
COUNTERS_FILE="$(mktemp "${IN_DIR}/_ecva_counters.XXXXXX")"
echo "0 0 0" > "$COUNTERS_FILE"  # processed ok broken
LOCK_FILE="$(mktemp "${IN_DIR}/_ecva_lock.XXXXXX")"

check_one() {
  local SRC="$1"
  local IN_ROOT="$2"
  local MIN_F="$3"
  local MIN_D="$4"

  local REL
  if [[ "$SRC" == ${IN_ROOT}/* ]]; then
    REL="${SRC#${IN_ROOT}/}"
  else
    REL="$SRC"
  fi

  local frames dur
  frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
    -of default=nokey=1:noprint_wrappers=1 "$SRC" 2>/dev/null || true)
  dur=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 \
    "$SRC" 2>/dev/null || true)

  # Normalize values
  if [[ -z "$frames" || "$frames" == "N/A" ]]; then
    frames="0"
  fi
  if [[ -z "$dur" || "$dur" == "N/A" ]]; then
    dur="-1"
  fi

  local status="OK"
  # Non-integer frames treated as 0
  if ! [[ "$frames" =~ ^[0-9]+$ ]]; then
    frames="0"
  fi
  # Decide status
  if [[ "$frames" -lt "$MIN_F" ]]; then
    status="BROKEN"
  fi
  if [[ "$MIN_D" != "0" ]]; then
    # require duration >= MIN_DURATION when MIN_DURATION>0; if unknown (-1), mark broken
    if [[ "$dur" == "-1" ]]; then
      status="BROKEN"
    else
      # compare as float using awk
      awk -v d="$dur" -v min="$MIN_D" 'BEGIN{ if (d+0 < min+0) exit 1; else exit 0 }'
      if [[ $? -ne 0 ]]; then
        status="BROKEN"
      fi
    fi
  fi

  # Atomically append report, update counters and show progress
  flock "$LOCK_FILE" bash -lc '
    # Append to report
    echo -e '"${status}\t${REL}\t${frames}\t${dur}"' >> '"$REPORT_TMP"'
    # Read counters
    if read p o b < '"$COUNTERS_FILE"' 2>/dev/null; then :; else p=0; o=0; b=0; fi
    p=$((p+1))
    if [ '"$status"' = "OK" ]; then o=$((o+1)); else b=$((b+1)); fi
    echo "$p $o $b" > '"$COUNTERS_FILE"'
    # Verbose per-file logs
    if [ '"$VERBOSE"' = "1" ]; then
      if [ '"$status"' = "OK" ]; then
        echo "[OK] '"$REL"' (frames='"$frames"', dur='"$dur"')" >&2
      else
        echo "[!] BROKEN: '"$REL"' (frames='"$frames"', dur='"$dur"')" >&2
      fi
    else
      if [ '"$status"' = "BROKEN" ]; then
        echo "[!] BROKEN: '"$REL"' (frames='"$frames"', dur='"$dur"')" >&2
      fi
    fi
    rem=$(( '"$TOTAL"' - p ))
    printf "\r[progress] processed=%d/%d ok=%d broken=%d remaining=%d" "$p" '"$TOTAL"' "$o" "$b" "$rem" >&2
  '
}

export -f check_one
export IN_DIR MIN_FRAMES MIN_DURATION REPORT_TMP COUNTERS_FILE LOCK_FILE TOTAL VERBOSE

# Process the unified file list
xargs -0 -n1 -P"$WORKERS" -I{} bash -lc 'check_one "$1" "$2" "$3" "$4"' _ "{}" "$IN_DIR" "$MIN_FRAMES" "$MIN_DURATION" < "$FILES_TMP"

# Summarize
mv -f "$REPORT_TMP" "$REPORT"
total=$(awk 'NR>1{c++} END{print c+0}' "$REPORT")
broken=$(awk -F"\t" 'NR>1 && $1=="BROKEN"{c++} END{print c+0}' "$REPORT")
ok=$(awk -F"\t" 'NR>1 && $1=="OK"{c++} END{print c+0}' "$REPORT")

echo >&2  # newline after progress
echo "[+] Summary: total=$total, ok=$ok, broken=$broken"
echo "[+] Report written to: $REPORT"

if [[ "$broken" -gt 0 ]]; then
  exit 1
fi
exit 0
