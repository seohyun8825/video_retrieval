#!/usr/bin/env bash
set -euo pipefail

# Build SFT-style JSONL for abnormal ECVA clips and push to HF Hub.
#
# Output schema per line (JSON):
# {
#   "messages": [
#     {"role": "user", "content": "<random prompt>\n<video>\n\nQuestion: <random question>"},
#     {"role": "assistant", "content": "<abnormal variation>. In the video, <a3_description (first letter lowercased)>"}
#   ],
#   "videos": ["ecva/abnormal_video/<file>.mp4"]
# }
#
# Defaults (override via flags or env):
#   CSV_PATH=/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv
#   AB_DIR=/hub_data4/seohyun/ecva/abnormal_video
#   REPO=happy8825/abnormal_ecva
#   JSONL_OUT=/home/seohyun/vid_understanding/video_retrieval/data/abnormal_sft.jsonl
#   PUSH=1          # 1=upload to HF Hub, 0=only generate JSONL locally
#   PUSH_VIDEOS=0   # 1=also upload mp4 files (default: skip video upload)
#
# Usage:
#   bash ecva_abnormal_tosft.bash \
#     [--csv PATH] [--ab-dir DIR] [--repo REPO] [--jsonl-out PATH] [--no-push] [--push-videos]

CSV_PATH_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/Video_Annotation.csv"
AB_DIR_DEFAULT="/hub_data4/seohyun/ecva/abnormal_video"
REPO_DEFAULT="happy8825/abnormal_ecva_sft"
JSONL_OUT_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/abnormal_sft.jsonl"

CSV_PATH="${CSV_PATH:-$CSV_PATH_DEFAULT}"
AB_DIR="${AB_DIR:-$AB_DIR_DEFAULT}"
REPO="${REPO:-$REPO_DEFAULT}"
JSONL_OUT="${JSONL_OUT:-$JSONL_OUT_DEFAULT}"
PUSH="${PUSH:-1}"
PUSH_VIDEOS="${PUSH_VIDEOS:-0}"

usage() {
  echo "Usage: $0 [--csv PATH] [--ab-dir DIR] [--repo REPO] [--jsonl-out PATH] [--no-push] [--push-videos]" >&2
  echo "  --csv PATH      Path to Video_Annotation.csv (default: $CSV_PATH_DEFAULT)" >&2
  echo "  --ab-dir DIR    Abnormal videos dir with <id>_<mmss><mmss>.mp4 (default: $AB_DIR_DEFAULT)" >&2
  echo "  --repo REPO     HF dataset repo id (default: $REPO_DEFAULT)" >&2
  echo "  --jsonl-out P   Output JSONL path (default: $JSONL_OUT_DEFAULT)" >&2
  echo "  --no-push       Do not upload to HF Hub (generate JSONL only)" >&2
  echo "  --push-videos   Also upload mp4 videos (default: disabled)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV_PATH="$2"; shift 2;;
    --ab-dir) AB_DIR="$2"; shift 2;;
    --repo) REPO="$2"; shift 2;;
    --jsonl-out) JSONL_OUT="$2"; shift 2;;
    --no-push) PUSH=0; shift;;
    --push-videos) PUSH_VIDEOS=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[!] CSV not found: $CSV_PATH" >&2
  exit 1
fi
if [[ ! -d "$AB_DIR" ]]; then
  echo "[!] Abnormal videos dir not found: $AB_DIR" >&2
  exit 1
fi

JSONL_DIR="$(dirname "$JSONL_OUT")"
mkdir -p "$JSONL_DIR"

echo "[+] CSV_PATH=$CSV_PATH"
echo "[+] AB_DIR=$AB_DIR"
echo "[+] REPO=$REPO"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] PUSH=$PUSH"
echo "[+] PUSH_VIDEOS=$PUSH_VIDEOS (0=skip mp4 uploads)"

# Generate JSONL
python3 - "$CSV_PATH" "$AB_DIR" "$JSONL_OUT" << 'PY'
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

CSV_PATH = Path(sys.argv[1])
AB_DIR = Path(sys.argv[2])
JSONL_OUT = Path(sys.argv[3])

# Randomness: each run differs by default. Set SEED env var for reproducibility.
seed = os.environ.get('SEED')
random.seed(None if seed is None else seed)

def norm_header(h: str) -> str:
    return h.strip()

def extract_pairs(moment: str):
    # Returns list of (s_label, e_label) like ('0004', '0035')
    if not moment:
        return []
    pairs = re.findall(r"\[(\d{3,}),\s*(\d{3,})\]", moment)
    return [(s, e) for s, e in pairs]

def extract_desc_list(desc_field: str):
    # Field can be like: "[Text 1],[Text 2]" or "[Text]"
    if not desc_field:
        return []
    # Grab top-level bracket contents
    parts = re.findall(r"\[([^\[\]]+)\]", desc_field, flags=re.S)
    cleaned = [p.strip() for p in parts if p.strip()]
    if cleaned:
        return cleaned
    # Fallback: use whole string without brackets
    return [desc_field.strip().strip('[]').strip()]

def lower_first_alpha(s: str) -> str:
    # Lowercase the first alphabetic character only
    # for i, ch in enumerate(s):
    #     if ch.isalpha():
    #         return s[:i] + ch.lower() + s[i+1:]
    return s

# Time conversion helpers
def mmss_to_seconds(label: str) -> int:
    # label like '0004', '0035', can be 3+ digits (e.g., '00010')
    d = re.sub(r"\D", "", label)
    if not d:
        return 0
    if len(d) <= 2:
        return int(d)
    minutes = int(d[:-2]) if len(d) > 2 else 0
    seconds = int(d[-2:])
    return minutes * 60 + seconds

def norm_pair_secs(a: int, b: int):
    return (a, b) if a <= b else (b, a)

# Build mapping by seconds: (video_id, s_sec, e_sec) -> description
mapping_secs = {}
with CSV_PATH.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = {norm_header(h): h for h in (reader.fieldnames or [])}

    def find_col(candidates):
        for t in candidates:
            for k, orig in headers.items():
                if k == t:
                    return orig
        return None

    col_id = find_col(['video_id', 'id'])
    col_moment = find_col(['A3,moment', 'A3 - moment', 'A3 moment', 'moment'])
    col_desc = find_col(['A3 - description', 'A3,description', 'A3 description', 'description'])

    if not (col_id and col_moment and col_desc):
        print('[!] Required columns not found. Headers:', list(reader.fieldnames or []), file=sys.stderr)
        sys.exit(1)

    for row in reader:
        vid_raw = (row.get(col_id) or '').strip()
        if not vid_raw:
            continue
        vid = re.sub(r'\D', '', vid_raw)
        if not vid:
            continue
        pairs = extract_pairs((row.get(col_moment) or '').strip())
        descs = extract_desc_list((row.get(col_desc) or '').strip())
        if not pairs:
            continue
        for i, (s, e) in enumerate(pairs):
            desc = descs[i] if i < len(descs) and descs[i].strip() else (descs[0] if descs else '')
            s_sec = mmss_to_seconds(s)
            e_sec = mmss_to_seconds(e)
            s_sec, e_sec = norm_pair_secs(s_sec, e_sec)
            mapping_secs[(vid, s_sec, e_sec)] = desc

# Prompt/Question variations (English)
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

abn_variants = [
    "This scene is abnormal.",
    "This video is abnormal.",
    "This clip is abnormal.",
    "An abnormal event occurs in this video.",
    "An abnormal incident occurs in this clip.",
    "This video contains an abnormal event.",
    "This clip contains an abnormal event.",
    "Abnormal behavior is present in this video.",
    "Abnormal activity is present in this clip."
    "An anomaly is present in this video.",
    "An anomaly is present in this clip.",
    "This scene depicts an abnormal event.",
    "The clip shows an abnormal incident.",
    "The video shows abnormal behavior.",
    "The video shows abnormal activity.",
    "An abnormal situation is shown in this clip.",
    "This scene includes unusual behavior.",
    "This clip includes unusual activity.",
    "This scene contains irregular behavior.",
    "This video contains irregular activity.",
    "This clip contains an irregular incident.",
    "This clip depicts abnormal activity.",
    "Abnormal activity is present in this video.",
    "This video shows a concerning incident.",
    "A concerning event occurs in this clip.",
    "The scene contains unsafe event.",
    "There is an abnormal event in this scene.",
    "There is abnormal behavior in this clip.",
    "There is unusual activity in this video.",
    "An abnormal moment occurs in this footage.",
    "An irregular event occurs in this scene.",
    "Unexpected behavior occurs in this clip.",
    "Something unusual happens in this video.",
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
missing_desc = 0
unmatched = 0

with JSONL_OUT.open('w', encoding='utf-8') as out:
    for p in sorted(AB_DIR.glob('*.mp4')):
        name = p.name
        # Accept variable-length label tail, try to infer the split by trying
        # all splits that leave 3+ digits on each side, then match by seconds.
        m = re.match(r'^(\d+)_([0-9]+)\.mp4$', name)
        if not m:
            unmatched += 1
            print(f"[!] Skip unmatched filename pattern: {name}", file=sys.stderr)
            continue
        vid, tail = m.group(1), m.group(2)

        desc = None
        # Try all possible splits: first part length k from 3 to len(tail)-3
        for k in range(3, max(3, len(tail) - 2)):
            s_lab = tail[:k]
            e_lab = tail[k:]
            if len(e_lab) < 3:
                continue
            s_sec = mmss_to_seconds(s_lab)
            e_sec = mmss_to_seconds(e_lab)
            key = (vid, *norm_pair_secs(s_sec, e_sec))
            desc = mapping_secs.get(key)
            if desc:
                break

        if not desc:
            missing_desc += 1
            # Try to print a human-friendly pair guess for debugging
            if len(tail) >= 6:
                guess = (tail[:len(tail)//2], tail[len(tail)//2:])
                print(f"[!] No A3 description for {name} (id={vid}, guess={guess[0]}-{guess[1]})", file=sys.stderr)
            else:
                print(f"[!] No A3 description for {name} (id={vid})", file=sys.stderr)
            continue

        assistant = f"{pick(abn_variants)} {lower_first_alpha(desc).rstrip('.')} .".replace(' .', '.')
        user = build_user_content()
        obj = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "videos": [f"ecva/abnormal_video/{name}"]
        }
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        written += 1

print(f"[+] JSONL written: {JSONL_OUT} (samples={written}, missing_desc={missing_desc}, unmatched_files={unmatched})")
PY

echo "[+] JSONL created at: $JSONL_OUT"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repo exists: $REPO (hf)"
    hf repo create "$REPO" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload JSONL to repo root (hf)"
    hf upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset

    if [[ "$PUSH_VIDEOS" == "1" ]]; then
      echo "[+] Upload abnormal videos folder → ecva/abnormal_video/ (hf; folder upload)"
      # Note: 'hf upload' uploads folders recursively by default.
      hf upload "$REPO" "$AB_DIR" "ecva/abnormal_video" --repo-type dataset
    else
      echo "[=] Skipping mp4 upload (PUSH_VIDEOS=0)."
    fi

  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repo exists: $REPO (huggingface-cli)"
    huggingface-cli repo create "$REPO" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload JSONL to repo root (huggingface-cli)"
    huggingface-cli upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset

    if [[ "$PUSH_VIDEOS" == "1" ]]; then
      echo "[+] Upload abnormal videos folder → ecva/abnormal_video/ (file-by-file fallback)"
      find "$AB_DIR" -type f -name "*.mp4" -print0 | while IFS= read -r -d '' f; do
        rel="ecva/abnormal_video/$(basename "$f")"
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
