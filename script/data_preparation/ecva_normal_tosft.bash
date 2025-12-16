#!/usr/bin/env bash
set -euo pipefail

# Build SFT-style JSONL for normal ECVA clips and (optionally) push to HF Hub.
#
# Output schema per line (JSON):
# {
#   "messages": [
#     {"role": "user", "content": "<random prompt(optional)>\n<video>\n\nQuestion: <random question>"},
#     {"role": "assistant", "content": "<normal variation>"}
#   ],
#   "videos": ["ecva/normal_video/<file>.mp4"]
# }
#
# Defaults (override via flags or env):
#   CSV_PATH=/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv
#   NORMAL_DIR=/hub_data4/seohyun/ecva/normal_video
#   REPO=happy8825/normal_ecva
#   JSONL_OUT=/home/seohyun/vid_understanding/video_retrieval/data/normal_sft.jsonl
#   PUSH=1           # 1=upload to HF Hub, 0=only generate JSONL locally
#   PUSH_VIDEOS=0    # 1=also upload mp4 files (default: skip video upload)
#
# Usage:
#   bash ecva_normal_tosft.bash \
#     [--csv PATH] [--normal-dir DIR] [--repo REPO] [--jsonl-out PATH] [--no-push] [--push-videos]

CSV_PATH_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv"
NORMAL_DIR_DEFAULT="/hub_data4/seohyun/ecva/normal_video"
REPO_DEFAULT="happy8825/normal_ecva_sft"
JSONL_OUT_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/normal_sft.jsonl"

CSV_PATH="${CSV_PATH:-$CSV_PATH_DEFAULT}"
NORMAL_DIR="${NORMAL_DIR:-$NORMAL_DIR_DEFAULT}"
REPO="${REPO:-$REPO_DEFAULT}"
JSONL_OUT="${JSONL_OUT:-$JSONL_OUT_DEFAULT}"
PUSH="${PUSH:-1}"
PUSH_VIDEOS="${PUSH_VIDEOS:-0}"

usage() {
  echo "Usage: $0 [--csv PATH] [--normal-dir DIR] [--repo REPO] [--jsonl-out PATH] [--no-push] [--push-videos]" >&2
  echo "  --csv PATH       Path to Video_Annotation.csv (default: $CSV_PATH_DEFAULT)" >&2
  echo "  --normal-dir D   Normal videos dir with <id>_normal.mp4 (default: $NORMAL_DIR_DEFAULT)" >&2
  echo "  --repo REPO      HF dataset repo id (default: $REPO_DEFAULT)" >&2
  echo "  --jsonl-out P    Output JSONL path (default: $JSONL_OUT_DEFAULT)" >&2
  echo "  --no-push        Do not upload to HF Hub (generate JSONL only)" >&2
  echo "  --push-videos    Also upload mp4 videos (default: disabled)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV_PATH="$2"; shift 2;;
    --normal-dir) NORMAL_DIR="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --jsonl-out) JSONL_OUT="$2"; shift 2;;
    --no-push) PUSH=0; shift;;
    --push-videos) PUSH_VIDEOS=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ ! -d "$NORMAL_DIR" ]]; then
  echo "[!] Normal videos dir not found: $NORMAL_DIR" >&2
  exit 1
fi

JSONL_DIR="$(dirname "$JSONL_OUT")"
mkdir -p "$JSONL_DIR"

echo "[+] CSV_PATH=$CSV_PATH"
echo "[+] NORMAL_DIR=$NORMAL_DIR"
echo "[+] REPO=$REPO"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] PUSH=$PUSH"
echo "[+] PUSH_VIDEOS=$PUSH_VIDEOS (0=skip mp4 uploads)"

# Generate JSONL
python3 - "$NORMAL_DIR" "$JSONL_OUT" << 'PY'
import json
import os
import random
import re
import sys
from pathlib import Path

NORMAL_DIR = Path(sys.argv[1])
JSONL_OUT = Path(sys.argv[2])

# Randomness: each run differs by default. Set SEED env var for reproducibility.
seed = os.environ.get('SEED')
random.seed(None if seed is None else seed)

# Prompt/Question variations (copied from abnormal script), with an empty option for header
prompt_heads = [
    "Please watch the following video and answer the question.",
    "Examine the clip below and respond concisely.",
    "Analyze the video and provide a brief answer.",
    "Review the following video and answer the question.",
    "Consider the video and reply.",
    "",
]
questions = [
    "Does this video actually show an abnormal incident happening on-screen? If yes, briefly describe what happens.",
    "Is an abnormal incident visibly occurring in this clip? If yes, describe it briefly.",
    "Do you see the abnormal activity itself unfolding in the video? If yes, summarize it.",
    "Is there a moment where an anomaly is directly happening in-frame? If yes, briefly state what it is.",
    "Does the clip depict the unusual act or event itself occurring? If yes, identify it.",
    "Is anything abnormal actively happening in this scene? If yes, briefly describe the abnormal action.",
    "Are any anomalies directly occurring in this clip? If yes, identify them briefly.",
    "Does this video show suspicious or abnormal behavior taking place on-screen? If yes, briefly explain.",
    "Is an abnormal incident happening during the clip? If yes, provide a short description.",
    "Is there any irregular event actively occurring in this video? If yes, briefly say what it is.",
    "Is the clip normal, or is there a moment where an abnormal event happens on-screen? If so, briefly describe it.",
    "Does anything directly deviate from normal behavior in-frame? If yes, briefly identify the deviation.",
    "Is everything in this clip typical, or do we see the abnormal activity itself occur? If so, briefly describe it.",
    "Is there any unexpected event visibly happening in this clip? If yes, briefly summarize it.",
    "Is there any concerning activity occurring on-screen in this clip? If yes, briefly identify it.",
    "Does an abnormal event occur in this clip? If yes, briefly note what happens and where and when it occurs.",
    "Do you see an anomaly happening in any part of the clip as an action or event? If yes, describe it briefly.",
    "Does anything abnormal occur during the clip? If yes, briefly identify the event.",
    "Is there an abnormal moment where the incident is actively happening in the scene? If yes, describe it briefly.",
    "Is there an irregular action or incident occurring on-screen? If yes, briefly state what it is.",
    "Is there any unsafe or abnormal situation actively unfolding in this clip? If yes, briefly describe it.",
    "Does the clip show any abnormal or risky behavior occurring in-frame? If yes, briefly identify it.",
    "Is there any suspicious activity taking place on-screen in this video? If yes, briefly describe it.",
    "Is an incident or anomaly visibly happening in this clip? If yes, briefly summarize the incident.",
    "Does the scene include any abnormal interaction actually occurring on-screen? If yes, briefly identify it.",
    "Any abnormality actively occurring in this clip? If yes, briefly identify it.",
    "Anything unusual happening on-screen in this clip? If yes, briefly describe it.",
    "Any anomaly directly happening during the clip? If yes, identify it briefly.",
    "Is there an abnormal incident occurring on-screen here? If yes, describe briefly.",
    "Abnormal event present and visible while it happens? If yes, briefly summarize.",
]

# Normal variants supplied by user
normal_variants = [
    "Nothing in this clip looks particularly abnormal.",
    "Nothing in this video appears clearly abnormal.",
    "There is nothing notably abnormal shown in this scene.",
    "No clearly abnormal incident is shown happening in this clip.",
    "No obvious abnormal event is visible in this video.",
    "No clear anomaly is shown in this clip.",
    "No clearly unusual activity is visible in this scene.",
    "Nothing stands out as an abnormal incident in this video.",
    "Nothing in-frame appears distinctly abnormal.",
    "No clearly suspicious behavior is shown taking place, as far as can be seen.",

    "This scene appears mostly normal.",
    "This clip appears generally normal.",
    "This video appears normal overall, with nothing clearly abnormal shown.",
    "The clip looks ordinary, with no clear abnormal activity visible.",
    "The scene does not seem to show a clear abnormal incident.",
    "The video does not appear to show an obvious anomaly.",
    "No clear concerning incident is visible in this clip.",
    "All visible activity looks typical, with nothing clearly abnormal.",
    "The scene shows routine activity with no obvious abnormal moment.",
    "The clip shows typical behavior with nothing notably abnormal.",

    "Nothing clearly abnormal has happened in this clip so far.",
    "Nothing clearly abnormal has happened in this video so far.",
    "No clearly abnormal incident has happened in this clip yet.",
    "No clearly abnormal incident has happened in this video yet.",
    "No obvious abnormal event has happened in this clip yet.",
    "No obvious abnormal event has happened in this video yet.",
    "No clear anomaly has appeared in this clip yet.",
    "No clear anomaly has appeared in this video yet.",
    "No clearly suspicious activity has occurred in this clip so far.",
    "No clearly suspicious activity has occurred in this video so far.",
    "So far, nothing stands out as clearly abnormal in this clip.",
    "Up to now, nothing in this video looks clearly abnormal.",
    "At this point, no clearly abnormal event appears to have happened.",
    "There has not been a clearly abnormal moment in this clip yet.",
]


def pick(lst):
    return random.choice(lst)

def build_user_content():
    head = pick(prompt_heads)
    if head:
        return f"{head}\n<video>\n\nQuestion: {pick(questions)}"
    else:
        return f"<video>\n\nQuestion: {pick(questions)}"

written = 0

with JSONL_OUT.open('w', encoding='utf-8') as out:
    for p in sorted(NORMAL_DIR.glob('*.mp4')):
        name = p.name
        user = build_user_content()
        assistant = pick(normal_variants)
        obj = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "videos": [f"ecva/normal_video/{name}"]
        }
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        written += 1

print(f"[+] JSONL written: {JSONL_OUT} (samples={written})")
PY

echo "[+] JSONL created at: $JSONL_OUT"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repo exists: $REPO (hf)"
    hf repo create "$REPO" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload JSONL to repo root (hf)"
    hf upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset

    if [[ "$PUSH_VIDEOS" == "1" ]]; then
      echo "[+] Upload normal videos folder → ecva/normal_video/ (hf; folder upload)"
      # 'hf upload' uploads folders recursively by default.
      hf upload "$REPO" "$NORMAL_DIR" "ecva/normal_video" --repo-type dataset
    else
      echo "[=] Skipping mp4 upload (PUSH_VIDEOS=0)."
    fi

  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repo exists: $REPO (huggingface-cli)"
    huggingface-cli repo create "$REPO" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload JSONL to repo root (huggingface-cli)"
    huggingface-cli upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset

    if [[ "$PUSH_VIDEOS" == "1" ]]; then
      echo "[+] Upload normal videos folder → ecva/normal_video/ (file-by-file fallback)"
      find "$NORMAL_DIR" -type f -name "*.mp4" -print0 | while IFS= read -r -d '' f; do
        rel="ecva/normal_video/$(basename "$f")"
        echo "  - $rel"
        huggingface-cli upload "$REPO" "$f" "$rel" --repo-type dataset
      done
    else
      echo "[=] Skipping mp4 upload (PUSH_VIDEOS=0)."
    fi

  else
    echo "[!] Neither 'hf' nor 'huggingface-cli' found. Install with: pip install huggingface_hub" >&2
    exit 1
  fi

  echo "[+] Push complete: https://huggingface.co/datasets/$REPO"
else
  echo "[=] PUSH disabled. Skipped upload to HF Hub."
fi

