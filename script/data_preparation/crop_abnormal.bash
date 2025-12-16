#!/usr/bin/env bash
set -euo pipefail

# Crop abnormal video moments based on A3,moment timestamps in the CSV.
#
# Defaults (override via env vars or flags):
#   CSV_PATH=/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv
#   SRC_DIR=/hub_data4/seohyun/ecva/video
#   OUT_DIR=/hub_data4/seohyun/ecva/abnormal_video
#   REENCODE=0   # 0 = fast stream copy, 1 = re-encode (more accurate but slower)
#   OVERWRITE=0  # 0 = skip existing outputs, 1 = overwrite outputs
#   WORKERS=1    # parallel ffmpeg jobs
#
# Usage:
#   bash crop_abnormal.bash [--csv PATH] [--src-dir DIR] [--out-dir DIR] [--reencode] [--overwrite] [--workers N]

CSV_PATH_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv"
SRC_DIR_DEFAULT="/hub_data4/seohyun/ecva/video"
OUT_DIR_DEFAULT="/hub_data4/seohyun/ecva/abnormal_video"

CSV_PATH="${CSV_PATH:-$CSV_PATH_DEFAULT}"
SRC_DIR="${SRC_DIR:-$SRC_DIR_DEFAULT}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
REENCODE="${REENCODE:-0}"
OVERWRITE="${OVERWRITE:-0}"
WORKERS="${WORKERS:-50}"

usage() {
  echo "Usage: $0 [--csv PATH] [--src-dir DIR] [--out-dir DIR] [--reencode] [--overwrite] [--workers N]" >&2
  echo "  --csv PATH       Path to Video_Annotation.csv (default: $CSV_PATH_DEFAULT)" >&2
  echo "  --src-dir DIR    Source videos directory with <id>.mp4 (default: $SRC_DIR_DEFAULT)" >&2
  echo "  --out-dir DIR    Output directory (default: $OUT_DIR_DEFAULT)" >&2
  echo "  --reencode       Re-encode instead of -c copy (more accurate, slower)" >&2
  echo "  --overwrite      Overwrite existing output files" >&2
  echo "  --workers N      Number of parallel ffmpeg jobs (default: $WORKERS)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)
      CSV_PATH="$2"; shift 2 ;;
    --src-dir)
      SRC_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --reencode)
      REENCODE=1; shift ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --workers)
      WORKERS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage; exit 1 ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] ffmpeg not found in PATH. Please install ffmpeg." >&2
  exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[!] CSV file not found: $CSV_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[+] CSV_PATH=$CSV_PATH"
echo "[+] SRC_DIR=$SRC_DIR"
echo "[+] OUT_DIR=$OUT_DIR"
echo "[+] REENCODE=$REENCODE (0=copy, 1=reencode)"
echo "[+] OVERWRITE=$OVERWRITE"
echo "[+] WORKERS=$WORKERS"

# Use embedded Python for robust CSV parsing (handles quotes, commas, etc.).
# Pass shell options to Python via environment for correct behavior.
REENCODE="$REENCODE" OVERWRITE="$OVERWRITE" WORKERS="$WORKERS" \
python3 - "$CSV_PATH" "$SRC_DIR" "$OUT_DIR" << 'PY'
import csv
import os
import re
import shlex
import subprocess
import sys
import concurrent.futures

CSV_PATH = sys.argv[1]
SRC_DIR = sys.argv[2]
OUT_DIR = sys.argv[3]

REENCODE = os.environ.get('REENCODE', '0') in ('1','true','TRUE','yes','YES')
OVERWRITE = os.environ.get('OVERWRITE', '0') in ('1','true','TRUE','yes','YES')
try:
    WORKERS = int(os.environ.get('WORKERS', '1') or '1')
    if WORKERS < 1:
        WORKERS = 1
except Exception:
    WORKERS = 1

def mmss_to_seconds(s: str) -> int:
    s = re.sub(r'\D', '', s)
    if not s:
        return 0
    if len(s) <= 2:
        return int(s)
    minutes = int(s[:-2]) if len(s) > 2 else 0
    seconds = int(s[-2:])
    return minutes * 60 + seconds

def run_ffmpeg(src, start_s, end_s, out_fp):
    duration = max(0.0, float(end_s) - float(start_s))
    if duration <= 0:
        print(f"[!] Skip (non-positive duration): {out_fp} from {start_s} to {end_s}", file=sys.stderr)
        return False
    base = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin']
    ow = ['-y'] if OVERWRITE else ['-n']
    # Place -ss before -i for speed. For accuracy, REENCODE=True re-encodes frames.
    if REENCODE:
        cmd = base + ['-ss', str(start_s), '-i', src, '-t', str(duration),
                      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                      '-c:a', 'aac', '-b:a', '128k'] + ow + [out_fp]
    else:
        cmd = base + ['-ss', str(start_s), '-i', src, '-t', str(duration),
                      '-c', 'copy'] + ow + [out_fp]
    try:
        print("[ffmpeg]", ' '.join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] ffmpeg failed for {out_fp}: {e}", file=sys.stderr)
        return False

def normalize_header(h: str) -> str:
    return h.strip()

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # Find exact header names (tolerate whitespace differences)
    headers = {normalize_header(h): h for h in reader.fieldnames or []}

    def find_col(targets):
        for t in targets:
            for k, original in headers.items():
                if k == t:
                    return original
        return None

    col_id = find_col(['video_id', 'id'])
    col_moment = find_col(['A3,moment', 'A3 - moment', 'A3 moment', 'moment'])

    if not col_id or not col_moment:
        print("[!] Required columns not found. Have headers:", file=sys.stderr)
        print(list(reader.fieldnames or []), file=sys.stderr)
        sys.exit(1)

    pair_re = re.compile(r"\[(\d{3,}),\s*(\d{3,})\]")

    rows = 0
    scheduled = 0
    missing_src = 0
    skip_existing = 0
    errors = 0
    tasks = []  # (src_path, s_sec, e_sec, out_path)
    for row in reader:
        rows += 1
        vid_raw = (row.get(col_id) or '').strip()
        if not vid_raw:
            continue
        # Accept numeric IDs (strip decimals if any)
        vid = re.sub(r'\D', '', vid_raw)
        if not vid:
            continue
        src_path = os.path.join(SRC_DIR, f"{vid}.mp4")
        if not os.path.isfile(src_path):
            print(f"[!] Missing source: {src_path}", file=sys.stderr)
            missing_src += 1
            continue

        moment = row.get(col_moment) or ''
        moment = moment.strip()
        if not moment:
            continue

        # Extract all [mmss,mmss] pairs
        pairs = pair_re.findall(moment)
        if not pairs:
            continue

        seen = set()
        for s_mmss, e_mmss in pairs:
            # Normalize to 4+ digits, keep as-is for naming
            s_label = re.sub(r'\D', '', s_mmss)
            e_label = re.sub(r'\D', '', e_mmss)
            if not s_label or not e_label:
                continue
            key = (s_label, e_label)
            if key in seen:
                continue
            seen.add(key)

            s_sec = mmss_to_seconds(s_label)
            e_sec = mmss_to_seconds(e_label)
            if e_sec <= s_sec:
                # swap if reversed
                s_sec, e_sec = e_sec, s_sec
                s_label, e_label = e_label, s_label

            out_name = f"{vid}_{s_label}{e_label}.mp4"
            out_path = os.path.join(OUT_DIR, out_name)

            # Skip if already cropped and not overwriting
            if os.path.exists(out_path) and not OVERWRITE:
                print(f"[=] Skip existing: {out_path}")
                skip_existing += 1
                continue

            tasks.append((src_path, s_sec, e_sec, out_path))
            scheduled += 1

    # Execute ffmpeg tasks (possibly in parallel)
    processed = 0
    if tasks:
        if WORKERS == 1:
            for src_path, s_sec, e_sec, out_path in tasks:
                ok = run_ffmpeg(src_path, s_sec, e_sec, out_path)
                processed += 1
                if not ok:
                    errors += 1
        else:
            def _job(t):
                src_path, s_sec, e_sec, out_path = t
                return run_ffmpeg(src_path, s_sec, e_sec, out_path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for ok in ex.map(_job, tasks):
                    processed += 1
                    if not ok:
                        errors += 1

    print(f"[+] Done. Parsed rows: {rows}, scheduled: {scheduled}, processed: {processed}, missing_src: {missing_src}, skip_existing: {skip_existing}, errors: {errors}")
PY

echo "[+] Finished cropping abnormal segments. Outputs in: $OUT_DIR"
