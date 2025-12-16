#!/usr/bin/env bash
set -euo pipefail

# Diagnose which samples in an HF JSONL dataset fail video decoding.
# - Downloads data.jsonl from --repo
# - For each sample (up to --max), joins videos with --media-root and tries to decode
#   the first frame via PyAV; falls back to ffprobe if PyAV unavailable.
# - Prints BAD entries with sample index and offending path; summary at end.
#
# Usage:
#   bash diagnose_hf_jsonl_videos.bash \
#     --repo happy8825/ecva_clean --media-root /hub_data4/seohyun [--max 200] [--verbose] [--force]

REPO=""
MEDIA_ROOT="/hub_data4/seohyun"
MAX="0"       # 0 = all
VERBOSE=0
FORCE=0

usage(){
  echo "Usage: $0 --repo org/name [--media-root DIR] [--max N] [--verbose] [--force]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --media-root) MEDIA_ROOT="$2"; shift 2 ;;
    --max) MAX="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "[!] --repo is required" >&2; usage; exit 1
fi

echo "[+] repo=$REPO | media_root=$MEDIA_ROOT | max=${MAX} | force=${FORCE}"

REPO="$REPO" MEDIA_ROOT="$MEDIA_ROOT" MAX="$MAX" VERBOSE="$VERBOSE" FORCE="$FORCE" \
python3 - << 'PY'
import json, os, sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception as e:
    print('[!] huggingface_hub not installed. pip install -U huggingface_hub', file=sys.stderr)
    sys.exit(1)

REPO = os.environ['REPO']
MEDIA_ROOT = os.environ['MEDIA_ROOT']
MAX = int(os.environ.get('MAX','0'))
VERBOSE = os.environ.get('VERBOSE','0') == '1'
FORCE = os.environ.get('FORCE','0') == '1'

def find_jsonl(repo_id: str) -> str:
    files = list_repo_files(repo_id, repo_type='dataset')
    if 'data.jsonl' in files:
        return 'data.jsonl'
    for f in files:
        if f.lower().endswith('.jsonl'):
            return f
    raise FileNotFoundError('No JSONL found in repo')

name = find_jsonl(REPO)
print(f"[+] downloading: {REPO}/{name}")
src = hf_hub_download(REPO, filename=name, repo_type='dataset', force_download=FORCE)

def can_decode_first_frame(path: str) -> bool:
    try:
        import av  # PyAV
        try:
            with av.open(path) as container:
                for frame in container.decode(video=0):
                    # got first frame
                    return True
                return False
        except Exception:
            return False
    except Exception:
        # fallback to ffprobe presence
        import subprocess
        try:
            out = subprocess.check_output([
                'ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=nb_read_frames','-of','default=nokey=1:noprint_wrappers=1',
                path
            ], text=True).strip()
            if not out or out == 'N/A':
                return False
            return int(out) > 0
        except Exception:
            return False

bad = 0
ok = 0
total = 0
first_bad = None

with open(src, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f, start=1):
        if MAX and total >= MAX:
            break
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            bad += 1
            print(f"[BAD][{idx}] invalid json")
            if first_bad is None: first_bad = idx
            continue
        vids = obj.get('videos') or []
        if not vids:
            bad += 1
            print(f"[BAD][{idx}] no videos array")
            if first_bad is None: first_bad = idx
            continue
        sample_ok = True
        for rel in vids:
            if not isinstance(rel, str):
                sample_ok = False; why = 'non-string path'; break
            path = os.path.join(MEDIA_ROOT, rel)
            if not os.path.isfile(path):
                sample_ok = False; why = 'missing file'; break
            if not can_decode_first_frame(path):
                sample_ok = False; why = 'cannot decode first frame'; bad_path = path; break
        if sample_ok:
            ok += 1
            if VERBOSE:
                print(f"[OK][{idx}] {vids[0]}")
        else:
            bad += 1
            if first_bad is None: first_bad = idx
            print(f"[BAD][{idx}] {vids[0] if vids else ''} -> {why}{' : '+bad_path if 'bad_path' in locals() else ''}")

print(f"[+] summary: total={total}, ok={ok}, bad={bad}")
if first_bad is not None:
    print(f"[!] first bad sample index: {first_bad}")
PY
