#!/usr/bin/env bash
set -euo pipefail

# Build SFT-style JSONL for after-incident ECVA clips and (optionally) push to HF Hub.
#
# Output schema per line (JSON):
# {
#   "messages": [
#     {"role": "user", "content": "<random prompt(optional)>\n<video>\n\nQuestion: <random question>"},
#     {"role": "assistant", "content": "<normal variation>"}
#   ],
#   "videos": ["ecva/after_incident/<file>.mp4"]
# }
#
# Defaults (override via flags or env):
#   AFTER_DIR=/hub_data4/seohyun/ecva/after_incident
#   REPO=happy8825/after_incident_normal
#   JSONL_OUT=/home/seohyun/vid_understanding/video_retrieval/data/after_incident_sft.jsonl
#   PUSH=1           # 1=upload to HF Hub, 0=only generate JSONL locally
#   PUSH_VIDEOS=0    # 1=also upload mp4 files (default: skip video upload)
#
# Usage:
#   bash ecva_after_incident_tosft.bash \
#     [--after-dir DIR] [--repo REPO] [--jsonl-out PATH] [--no-push] [--push-videos]

AFTER_DIR_DEFAULT="/hub_data4/seohyun/ecva/after_incident"
REPO_DEFAULT="happy8825/after_incident_normal"
JSONL_OUT_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/after_incident_sft.jsonl"

AFTER_DIR="${AFTER_DIR:-$AFTER_DIR_DEFAULT}"
REPO="${REPO:-$REPO_DEFAULT}"
JSONL_OUT="${JSONL_OUT:-$JSONL_OUT_DEFAULT}"
PUSH="${PUSH:-1}"
PUSH_VIDEOS="${PUSH_VIDEOS:-0}"

usage() {
  echo "Usage: $0 [--after-dir DIR] [--repo REPO] [--jsonl-out PATH] [--no-push] [--push-videos]" >&2
  echo "  --after-dir DIR  After-incident videos dir with <id>_after.mp4 (default: $AFTER_DIR_DEFAULT)" >&2
  echo "  --repo REPO      HF dataset repo id (default: $REPO_DEFAULT)" >&2
  echo "  --jsonl-out P    Output JSONL path (default: $JSONL_OUT_DEFAULT)" >&2
  echo "  --no-push        Do not upload to HF Hub (generate JSONL only)" >&2
  echo "  --push-videos    Also upload mp4 videos (default: disabled)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --after-dir) AFTER_DIR="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --jsonl-out) JSONL_OUT="$2"; shift 2;;
    --no-push) PUSH=0; shift;;
    --push-videos) PUSH_VIDEOS=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ ! -d "$AFTER_DIR" ]]; then
  echo "[!] After-incident videos dir not found: $AFTER_DIR" >&2
  exit 1
fi

JSONL_DIR="$(dirname "$JSONL_OUT")"
mkdir -p "$JSONL_DIR"

echo "[+] AFTER_DIR=$AFTER_DIR"
echo "[+] REPO=$REPO"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] PUSH=$PUSH"
echo "[+] PUSH_VIDEOS=$PUSH_VIDEOS (0=skip mp4 uploads)"

# Generate JSONL
python3 - "$AFTER_DIR" "$JSONL_OUT" << 'PY'
import json
import os
import random
import sys
from pathlib import Path

AFTER_DIR = Path(sys.argv[1])
JSONL_OUT = Path(sys.argv[2])

# Randomness: each run differs by default. Set SEED env var for reproducibility.
seed = os.environ.get('SEED')
random.seed(None if seed is None else seed)

# Prompt/Question variations (same as abnormal; header may be empty)
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


# Normal variants
normal_variants = [
    "No abnormal act is shown in this scene.",
    "No abnormal act is shown in this video.",
    "No abnormal act is shown in this clip.",
    "No abnormal incident is shown happening on-screen.",
    "No abnormal event is shown occurring in this clip.",
    "No anomaly is shown occurring in-frame.",
    "No unusual activity is shown occurring in this scene.",
    "No irregular activity is shown occurring in this video.",
    "No suspicious behavior is shown taking place on-screen.",
    "No concerning incident is shown in this clip.",

    "The scene appears mostly normal.",
    "The clip appears generally normal.",
    "The visible activity appears typical.",
    "The actions shown appear typical.",
    "Nothing clearly abnormal is shown in this clip.",
    "Nothing clearly unusual is shown in this video.",
    "No clear abnormal moment is visible in this scene.",
    "No obvious anomaly is visible in this clip.",
    "No clear suspicious activity is shown in this video.",
    "No clear irregular event is shown in this clip.",

    "No abnormal incident is shown at this time.",
    "No abnormal event is shown yet in this clip.",
    "No anomaly is shown yet in this video.",
    "No unusual activity is shown yet in this scene.",
    "No suspicious behavior is shown yet in this clip.",
    "No clear abnormal moment has appeared yet.",
    "So far, no abnormal act is shown in this clip.",
    "So far, no anomaly is shown in this video.",
    "Up to now, no abnormal incident is shown occurring on-screen.",
    "At this point, no clear abnormal activity is shown.",
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
    for p in sorted(AFTER_DIR.glob('*.mp4')):
        name = p.name
        user = build_user_content()
        assistant = pick(normal_variants)
        obj = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "videos": [f"ecva/after_incident/{name}"]
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
      echo "[+] Upload after-incident videos folder → ecva/after_incident/ (hf; folder upload)"
      # 'hf upload' uploads folders recursively by default.
      hf upload "$REPO" "$AFTER_DIR" "ecva/after_incident" --repo-type dataset
    else
      echo "[=] Skipping mp4 upload (PUSH_VIDEOS=0)."
    fi

  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repo exists: $REPO (huggingface-cli)"
    huggingface-cli repo create "$REPO" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload JSONL to repo root (huggingface-cli)"
    huggingface-cli upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset

    if [[ "$PUSH_VIDEOS" == "1" ]]; then
      echo "[+] Upload after-incident videos folder → ecva/after_incident/ (file-by-file fallback)"
      find "$AFTER_DIR" -type f -name "*.mp4" -print0 | while IFS= read -r -d '' f; do
        rel="ecva/after_incident/$(basename "$f")"
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

