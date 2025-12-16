#!/usr/bin/env bash
set -euo pipefail

# Crop normal video suffix: from the last A3,moment end to the end of the video.
#
# Defaults (override via env vars or flags):
#   CSV_PATH=/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv
#   SRC_DIR=/hub_data4/seohyun/ecva/video
#   OUT_DIR=/hub_data4/seohyun/ecva/after_incident
#   REENCODE=0     # 0 = fast stream copy, 1 = re-encode (more accurate but slower)
#   OVERWRITE=0    # 0 = skip existing outputs, 1 = overwrite outputs
#   WORKERS=1      # number of parallel ffmpeg jobs
#   MIN_DURATION=10  # skip if (video_end - last_end) <= MIN_DURATION seconds
#
# Usage:
#   bash normal_crop_after.sh [--csv PATH] [--src-dir DIR] [--out-dir DIR] \
#                             [--reencode] [--overwrite] [--workers N] [--min-duration S]

CSV_PATH_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv"
SRC_DIR_DEFAULT="/hub_data4/seohyun/ecva/video"
OUT_DIR_DEFAULT="/hub_data4/seohyun/ecva/after_incident"

CSV_PATH="${CSV_PATH:-$CSV_PATH_DEFAULT}"
SRC_DIR="${SRC_DIR:-$SRC_DIR_DEFAULT}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
REENCODE="${REENCODE:-0}"
OVERWRITE="${OVERWRITE:-0}"
WORKERS="${WORKERS:-1}"
MIN_DURATION="${MIN_DURATION:-10}"

usage() {
  echo "Usage: $0 [--csv PATH] [--src-dir DIR] [--out-dir DIR] [--reencode] [--overwrite] [--workers N] [--min-duration S]" >&2
  echo "  --csv PATH        Path to Video_Annotation.csv (default: $CSV_PATH_DEFAULT)" >&2
  echo "  --src-dir DIR     Source videos directory with <id>.mp4 (default: $SRC_DIR_DEFAULT)" >&2
  echo "  --out-dir DIR     Output directory (default: $OUT_DIR_DEFAULT)" >&2
  echo "  --reencode        Re-encode instead of -c copy (more accurate, slower)" >&2
  echo "  --overwrite       Overwrite existing output files" >&2
  echo "  --workers N       Number of parallel ffmpeg jobs (default: $WORKERS)" >&2
  echo "  --min-duration S  Minimum duration in seconds to keep (<=S will be skipped, default: $MIN_DURATION)" >&2
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
    --min-duration)
      MIN_DURATION="$2"; shift 2 ;;
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

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "[!] ffprobe not found in PATH. Please install ffmpeg/ffprobe." >&2
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
echo "[+] MIN_DURATION=$MIN_DURATION"

# Use embedded Python for robust CSV parsing and parallel execution.
REENCODE="$REENCODE" OVERWRITE="$OVERWRITE" WORKERS="$WORKERS" MIN_DURATION="$MIN_DURATION" \
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
try:
    MIN_DURATION = int(float(os.environ.get('MIN_DURATION', '10')))
except Exception:
    MIN_DURATION = 10

def mmss_to_seconds(s: str) -> int:
    s = re.sub(r'\D', '', s)
    if not s:
        return 0
    if len(s) <= 2:
        return int(s)
    minutes = int(s[:-2]) if len(s) > 2 else 0
    seconds = int(s[-2:])
    return minutes * 60 + seconds

def get_duration(src):
    try:
        out = subprocess.check_output([
            'ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=nokey=1:noprint_wrappers=1', src
        ], stderr=subprocess.DEVNULL, text=True).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None

def run_ffmpeg(src, start_s, end_s, out_fp):
    duration = max(0.0, float(end_s) - float(start_s)) if end_s is not None else None
    if duration is not None and duration <= 0:
        print(f"[!] Skip (non-positive duration): {out_fp} from {start_s} to {end_s}", file=sys.stderr)
        return False
    base = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin']
    ow = ['-y'] if OVERWRITE else ['-n']
    if REENCODE:
        args = ['-ss', str(start_s), '-i', src]
        if duration is not None:
            args += ['-t', str(duration)]
        args += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                 '-c:a', 'aac', '-b:a', '128k']
        cmd = base + args + ow + [out_fp]
    else:
        args = ['-ss', str(start_s), '-i', src]
        if duration is not None:
            args += ['-t', str(duration)]
        args += ['-c', 'copy']
        cmd = base + args + ow + [out_fp]
    try:
        print("[ffmpeg]", ' '.join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] ffmpeg failed for {out_fp}: {e}", file=sys.stderr)
        return False

def normalize_header(h: str) -> str:
    return h.strip()

# Collect latest abnormal end per video id
id_to_last_end = {}
missing_src = 0

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
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
    for row in reader:
        rows += 1
        vid_raw = (row.get(col_id) or '').strip()
        if not vid_raw:
            continue
        vid = re.sub(r'\D', '', vid_raw)
        if not vid:
            continue
        moment = (row.get(col_moment) or '').strip()
        if not moment:
            continue
        pairs = pair_re.findall(moment)
        if not pairs:
            continue
        for s_mmss, e_mmss in pairs:
            s_sec = mmss_to_seconds(s_mmss)
            e_sec = mmss_to_seconds(e_mmss)
            end_sec = max(s_sec, e_sec)
            if vid not in id_to_last_end or end_sec > id_to_last_end[vid]:
                id_to_last_end[vid] = end_sec

# Build tasks: cut [last_end, video_duration)
tasks = []
skip_existing = 0
too_short = 0
for vid, last_end in id_to_last_end.items():
    src_path = os.path.join(SRC_DIR, f"{vid}.mp4")
    if not os.path.isfile(src_path):
        if vid not in id_to_last_end:
            pass
        print(f"[!] Missing source: {src_path}", file=sys.stderr)
        missing_src += 1
        continue
    video_dur = get_duration(src_path)
    if video_dur is None:
        print(f"[!] Could not determine duration via ffprobe: {src_path}", file=sys.stderr)
        continue
    # Ensure sane bounds
    if last_end >= video_dur:
        print(f"[=] Skip tail (no time after incident): id={vid}, last_end={last_end:.2f}s, video_dur={video_dur:.2f}s")
        continue
    tail = video_dur - last_end
    if tail <= MIN_DURATION:
        too_short += 1
        print(f"[=] Skip too short tail (<= {MIN_DURATION}s): id={vid}, tail={int(tail)}s")
        continue

    out_name = f"{vid}_after.mp4"
    out_path = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_path) and not OVERWRITE:
        print(f"[=] Skip existing: {out_path}")
        skip_existing += 1
        continue

    # Provide explicit duration for consistency
    tasks.append((src_path, last_end, last_end + tail, out_path))

# Execute tasks (possibly in parallel)
processed = 0
errors = 0
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

print(f"[+] Done. unique_ids: {len(id_to_last_end)}, scheduled: {len(tasks)}, processed: {processed}, skip_existing: {skip_existing}, too_short: {too_short}, missing_src: {missing_src}, errors: {errors}")
PY

echo "[+] Finished cropping after-incident segments. Outputs in: $OUT_DIR"

