#!/usr/bin/env bash
set -euo pipefail

# Sample a quick clean subset (default N=20) from an HF dataset by
# verifying local video files, then push to a new HF dataset repo.
#
# Defaults (override via flags or env):
#   REPO_IN=happy8825/train_ecva
#   REPO_OUT=happy8825/ecva_clean
#   MEDIA_ROOT=/hub_data4/seohyun
#   JSONL_OUT=/home/seohyun/vid_understanding/video_retrieval/data/ecva_clean_subset.jsonl
#   N=20
#   MIN_FRAMES=1
#   MIN_DURATION=0   # seconds; 0 disables duration filter
#   PUSH=1
#
# Usage:
#   bash sample_ecva_clean20.bash \
#     [--repo-in REPO] [--repo-out REPO] [--media-root DIR] [--jsonl-out PATH] \
#     [--n N] [--min-frames N] [--min-duration S] [--no-push]

REPO_IN_DEFAULT="happy8825/train_ecva"
REPO_OUT_DEFAULT="happy8825/ecva_clean"
MEDIA_ROOT_DEFAULT="/hub_data4/seohyun"
JSONL_OUT_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/ecva_clean_subset.jsonl"

REPO_IN="${REPO_IN:-$REPO_IN_DEFAULT}"
REPO_OUT="${REPO_OUT:-$REPO_OUT_DEFAULT}"
MEDIA_ROOT="${MEDIA_ROOT:-$MEDIA_ROOT_DEFAULT}"
JSONL_OUT="${JSONL_OUT:-$JSONL_OUT_DEFAULT}"
N="${N:-10}"
MIN_FRAMES="${MIN_FRAMES:-1}"
MIN_DURATION="${MIN_DURATION:-0}"
PUSH="${PUSH:-1}"

usage() {
  echo "Usage: $0 [--repo-in REPO] [--repo-out REPO] [--media-root DIR] [--jsonl-out PATH] [--n N] [--min-frames N] [--min-duration S] [--no-push]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-in) REPO_IN="$2"; shift 2 ;;
    --repo-out) REPO_OUT="$2"; shift 2 ;;
    --media-root) MEDIA_ROOT="$2"; shift 2 ;;
    --jsonl-out) JSONL_OUT="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --min-frames) MIN_FRAMES="$2"; shift 2 ;;
    --min-duration) MIN_DURATION="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "[!] python3 not found." >&2
  exit 1
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "[!] ffprobe not found (ffmpeg). Please install to validate frames/duration." >&2
  exit 1
fi

echo "[+] REPO_IN=$REPO_IN"
echo "[+] REPO_OUT=$REPO_OUT"
echo "[+] MEDIA_ROOT=$MEDIA_ROOT"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] N=$N | MIN_FRAMES=$MIN_FRAMES | MIN_DURATION=$MIN_DURATION | PUSH=$PUSH"

# Python: stream the source JSONL and stop after collecting N clean items.
MEDIA_ROOT="$MEDIA_ROOT" N="$N" MIN_FRAMES="$MIN_FRAMES" MIN_DURATION="$MIN_DURATION" \
python3 - "$REPO_IN" "$JSONL_OUT" << 'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception as e:
    print("[!] huggingface_hub not installed. pip install -U huggingface_hub", file=sys.stderr)
    sys.exit(1)

REPO_IN = sys.argv[1]
JSONL_OUT = Path(sys.argv[2])
MEDIA_ROOT = os.environ.get("MEDIA_ROOT")
TARGET_N = int(os.environ.get("N", "20"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "1"))
MIN_DURATION = float(os.environ.get("MIN_DURATION", "0"))

def find_data_jsonl(repo_id: str) -> str:
    files = list_repo_files(repo_id, repo_type="dataset")
    if "data.jsonl" in files:
        return "data.jsonl"
    for f in files:
        if f.lower().endswith(".jsonl"):
            return f
    raise FileNotFoundError("No JSONL found in repo.")

def probe_frames(path: str) -> int | None:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
            path
        ], text=True).strip()
        if not out or out == "N/A":
            return 0
        return int(out)
    except Exception:
        return None

def probe_duration(path: str) -> float | None:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", path
        ], text=True).strip()
        if not out or out == "N/A":
            return None
        return float(out)
    except Exception:
        return None

target = find_data_jsonl(REPO_IN)
print(f"[+] Downloading: {REPO_IN}/{target}")
src_path = hf_hub_download(REPO_IN, filename=target, repo_type="dataset")

picked = 0
total = 0
JSONL_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(src_path, "r", encoding="utf-8") as fin, open(JSONL_OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        if picked >= TARGET_N:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        videos = obj.get("videos") or []
        if not isinstance(videos, list) or len(videos) == 0:
            continue
        ok = True
        for rel in videos:
            if not isinstance(rel, str):
                ok = False; break
            abs_path = os.path.join(MEDIA_ROOT, rel)
            if not os.path.isfile(abs_path):
                ok = False; break
            fr = probe_frames(abs_path)
            if fr is None or fr < MIN_FRAMES:
                ok = False; break
            if MIN_DURATION > 0:
                dur = probe_duration(abs_path)
                if dur is None or dur < MIN_DURATION:
                    ok = False; break
        if ok:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            picked += 1
            print(f"[OK] kept={picked}/{TARGET_N} | file={videos[0]}")

print(f"[+] Done. total_scanned={total}, kept={picked}")
PY

echo "[+] Subset JSONL at: $JSONL_OUT"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO_OUT (hf)"
    hf repo create "$REPO_OUT" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload subset JSONL (hf)"
    hf upload "$REPO_OUT" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO_OUT (huggingface-cli)"
    huggingface-cli repo create "$REPO_OUT" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload subset JSONL (huggingface-cli)"
    huggingface-cli upload "$REPO_OUT" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  else
    echo "[!] Neither 'hf' nor 'huggingface-cli' found. Install with: pip install -U huggingface_hub" >&2
    exit 1
  fi
  echo "[+] Pushed to: https://huggingface.co/datasets/$REPO_OUT"
else
  echo "[=] PUSH disabled; skipping upload."
fi

echo "[i] Train with: HF_HUB_URL=$REPO_OUT bash ${BASH_SOURCE[0]%/*}/../sft/train/t1_train_ecva.bash"

