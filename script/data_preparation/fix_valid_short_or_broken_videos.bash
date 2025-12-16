#!/usr/bin/env bash
set -euo pipefail

# Fix only the short/broken videos referenced by a valid HF dataset repo.
# Strategy: for each rel path in the repo JSON(L), probe via ffprobe:
#   - codec name and frame count
#   - Re-encode to H.264 (CFR) if codec is AV1, or frame count <= 0 (unreadable),
#     or frame count < NFRAMES_REQ.
#
# Usage:
#   bash fix_valid_short_or_broken_videos.bash --repo happy8825/valid_ecva_clean \
#       [--media-root /hub_data4/seohyun] [--fps 2] [--nframes 48] [--workers 32]

REPO=""
MEDIA_ROOT="/hub_data4/seohyun"
FPS=2
NFRAMES=48
WORKERS=32
# Skip re-encoding if frame count unknown (<=0). Default: 0 (re-encode unknowns)
SKIP_UNKNOWN=0
# Extra codec names (comma-separated, lower-case) to force re-encode. Default: av1
EXTRA_CODECS="av1"
# Report path (TSV): STATUS, REL_PATH, ABS_PATH, CODEC, FRAMES, DECODE_ERR
REPORT="/home/seohyun/vid_understanding/video_retrieval/data/valid_fix_report.tsv"
# Try quick decode test (ffmpeg to null) to detect errors
DECODE_TEST=1
# If provided, use local dataset file (json/jsonl) instead of querying HF.
DATASET_FILE=""
# Only check and report (do not re-encode).
CHECK_ONLY=0

usage(){
  echo "Usage: $0 --repo ORG/NAME [--media-root DIR] [--fps N] [--nframes N] [--workers N]" >&2
  echo "            [--dataset-file PATH.json[l]] [--check-only] [--skip-unknown] [--extra-codecs av1,vp9]" >&2
  echo "            [--report PATH.tsv] [--no-decode-test]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --media-root) MEDIA_ROOT="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --nframes) NFRAMES="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --dataset-file) DATASET_FILE="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --skip-unknown) SKIP_UNKNOWN=1; shift ;;
    --extra-codecs) EXTRA_CODECS="$2"; shift 2 ;;
    --report) REPORT="$2"; shift 2 ;;
    --no-decode-test) DECODE_TEST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "[!] --repo is required" >&2; usage; exit 1
fi

echo "[+] REPO=$REPO | MEDIA_ROOT=$MEDIA_ROOT | FPS=$FPS | NFRAMES_REQ=$NFRAMES | WORKERS=$WORKERS | SKIP_UNKNOWN=$SKIP_UNKNOWN | EXTRA_CODECS=$EXTRA_CODECS | DATASET_FILE=${DATASET_FILE:-auto} | CHECK_ONLY=$CHECK_ONLY | REPORT=$REPORT | DECODE_TEST=$DECODE_TEST"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
LIST0="$TMP_DIR/tofix.list0"

REPO="$REPO" MEDIA_ROOT="$MEDIA_ROOT" NFRAMES="$NFRAMES" SKIP_UNKNOWN="$SKIP_UNKNOWN" EXTRA_CODECS="$EXTRA_CODECS" DATASET_FILE="$DATASET_FILE" REPORT="$REPORT" DECODE_TEST="$DECODE_TEST" \
python3 - << 'PY' > "$LIST0"
import json, os, subprocess, sys, time
from huggingface_hub import hf_hub_download, list_repo_files

REPO=os.environ['REPO']
MEDIA_ROOT=os.environ['MEDIA_ROOT']
N=int(os.environ.get('NFRAMES','48'))
SKIP_UNKNOWN=os.environ.get('SKIP_UNKNOWN','0') in ('1','true','TRUE','yes','YES')
EXTRA_CODECS=set([s.strip().lower() for s in (os.environ.get('EXTRA_CODECS','av1') or 'av1').split(',') if s.strip()])
DATASET_FILE=os.environ.get('DATASET_FILE','')
REPORT=os.environ.get('REPORT','')
DO_DECODE=os.environ.get('DECODE_TEST','0') in ('1','true','TRUE','yes','YES')

def log(msg: str):
    print(f"[py] {msg}", file=sys.stderr, flush=True)

def retry(fn, tries=5, delay=1.5):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            log(f"retry {i+1}/{tries} after error: {e}")
            time.sleep(delay)
    raise last

def find_name(repo:str)->str:
    log("listing repo files ...")
    files = retry(lambda: list_repo_files(repo, repo_type='dataset'))
    log(f"found {len(files)} files in repo")
    if 'data.json' in files: return 'data.json'
    if 'data.jsonl' in files: return 'data.jsonl'
    for f in files:
        if f.lower().endswith('.json') or f.lower().endswith('.jsonl'):
            return f
    raise FileNotFoundError('no json/jsonl found')

def ffprobe_value(args):
    try:
        out = subprocess.check_output(args, text=True).strip()
        return out
    except Exception:
        return ''

def get_codec_and_frames(path: str):
    if not os.path.exists(path):
        return '', -1
    codec = ffprobe_value([
        'ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name','-of','default=nokey=1:noprint_wrappers=1',path
    ]).strip().lower()
    nb = ffprobe_value([
        'ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','default=nokey=1:noprint_wrappers=1',path
    ]).strip()
    try:
        frames = int(nb) if nb and nb != 'N/A' else -1
    except Exception:
        frames = -1
    return codec, frames

if DATASET_FILE and os.path.isfile(DATASET_FILE):
    name = os.path.basename(DATASET_FILE)
    src = DATASET_FILE
    log(f"using local dataset file: {src}")
else:
    name=find_name(REPO)
    log(f"downloading {name} ...")
    src=retry(lambda: hf_hub_download(REPO, filename=name, repo_type='dataset', force_download=True))
    log(f"parsing {src}")

def iter_samples():
    if name.lower().endswith('.jsonl'):
        with open(src,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    yield json.loads(line)
                except Exception:
                    pass
    else:
        data=json.load(open(src,'r',encoding='utf-8'))
        for o in data:
            yield o

seen=set(); total=0; emit=0
codec_forced=0; frames_short=0; frames_unknown=0; decode_errs=0

if REPORT:
    try:
        with open(REPORT, 'w', encoding='utf-8') as rf:
            rf.write("STATUS\tREL_PATH\tABS_PATH\tCODEC\tFRAMES\tDECODE_ERR\n")
    except Exception as e:
        print(f"[py] warn: cannot open report for write: {REPORT}: {e}", file=sys.stderr)
        REPORT = ''
for obj in iter_samples():
    for rel in obj.get('videos') or []:
        if not isinstance(rel,str):
            continue
        path=os.path.join(MEDIA_ROOT, rel)
        if path in seen:
            continue
        seen.add(path); total += 1
        codec, frames = get_codec_and_frames(path)
        # Optional decode error test using ffmpeg
        dec_err = 0
        if DO_DECODE:
            try:
                cmd = ['ffmpeg','-v','error','-nostdin','-threads','1','-i',path,'-map','0:v:0','-f','null','-']
                out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if out.stderr.strip():
                    dec_err = 1
            except Exception:
                dec_err = 1

        status = 'SKIP'
        need = False
        if codec in EXTRA_CODECS:
            codec_forced += 1
            need = True
        if frames <= 0:
            frames_unknown += 1
            if not SKIP_UNKNOWN:
                need = True
        elif frames < N:
            frames_short += 1
            need = True
        if dec_err:
            decode_errs += 1
            need = True

        if need:
            status = 'FIX'
            print(path)
            emit += 1

        if REPORT:
            try:
                with open(REPORT, 'a', encoding='utf-8') as rf:
                    rf.write(f"{status}\t{rel}\t{path}\t{codec}\t{frames}\t{dec_err}\n")
            except Exception:
                pass
log(f"candidates total={total}, to_fix={emit}")
log(f"reasons: codec_forced={codec_forced}, frames_short={frames_short}, frames_unknown={frames_unknown}, decode_errs={decode_errs}")
PY

TOTAL=$(wc -l < "$LIST0" | awk '{print $1}')
echo "[+] To re-encode: $TOTAL file(s)"
echo "[+] Report: $REPORT"

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[=] CHECK_ONLY is set. Listing candidates:"
  cat "$LIST0"
  exit 0
fi

if [[ "$TOTAL" -eq 0 ]]; then
  echo "[=] Nothing to re-encode."
  exit 0
fi

echo "[+] Starting parallel transcode (P=$WORKERS) ..."
xargs -I{} -P "$WORKERS" bash -lc 'src="{}"; tmp="${src}.tmp.mp4"; echo "[re-encode] $src"; ffmpeg -v error -y -i "$src" -vf "fps=$FPS" -an -c:v libx264 -preset fast -crf 23 -movflags +faststart "$tmp" && mv -f "$tmp" "$src"' < "$LIST0"

echo "[+] Done fixing short/broken videos for: $REPO"
