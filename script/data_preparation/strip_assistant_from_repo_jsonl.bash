#!/usr/bin/env bash
set -euo pipefail

# Strip assistant messages from an HF dataset JSONL and push back.
# - Keeps other fields (e.g., videos) intact; only filters messages to role=="user".
# - Optionally trims user content to only the single "Question: ...?" line (drops prompt heads, <video>, trailing guidance).
# - Or, with --video-only, force user content to just "<video>" (i.e., remove everything except the placeholder).
#
# Usage:
#   bash strip_assistant_from_repo_jsonl.bash --repo ORG/NAME [--jsonl-out PATH] [--no-push]

REPO=""
JSONL_OUT="/home/seohyun/vid_understanding/video_retrieval/data/cleaned_no_assistant.jsonl"
PUSH=1
# By default, keep only the 'Question: ...?' part from user content.
TRIM_TO_QUESTION=${TRIM_TO_QUESTION:-1}
# If set, override trimming and keep only <video>
VIDEO_ONLY=${VIDEO_ONLY:-0}

usage(){
  echo "Usage: $0 --repo ORG/NAME [--jsonl-out PATH] [--no-push] [--no-question-only] [--video-only]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --jsonl-out) JSONL_OUT="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --no-question-only) TRIM_TO_QUESTION=0; shift ;;
    --video-only) VIDEO_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "[!] --repo is required" >&2; usage; exit 1
fi

echo "[+] REPO=$REPO"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] PUSH=$PUSH"
echo "[+] TRIM_TO_QUESTION=$TRIM_TO_QUESTION"
echo "[+] VIDEO_ONLY=$VIDEO_ONLY"

python3 - "$REPO" "$JSONL_OUT" << 'PY'
import json, sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception as e:
    print('[!] huggingface_hub not installed. pip install -U huggingface_hub', file=sys.stderr)
    raise

REPO = sys.argv[1]
OUT = Path(sys.argv[2])
import os, re
TRIM_TO_QUESTION = os.environ.get('TRIM_TO_QUESTION', '0') in ('1','true','TRUE','yes','YES')
VIDEO_ONLY = os.environ.get('VIDEO_ONLY', '0') in ('1','true','TRUE','yes','YES')

def find_jsonl(repo_id: str) -> str:
    files = list_repo_files(repo_id, repo_type='dataset')
    if 'data.jsonl' in files:
        return 'data.jsonl'
    for f in files:
        if f.lower().endswith('.jsonl'):
            return f
    raise FileNotFoundError('No JSONL found in repo')

name = find_jsonl(REPO)
print(f"[+] Downloading: {REPO}/{name}")
src = hf_hub_download(REPO, filename=name, repo_type='dataset')

OUT.parent.mkdir(parents=True, exist_ok=True)
kept = 0
total = 0
with open(src, 'r', encoding='utf-8') as fin, OUT.open('w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        msgs = obj.get('messages')
        if isinstance(msgs, list):
            user_msgs = [m for m in msgs if isinstance(m, dict) and m.get('role') == 'user']
            if user_msgs:
                if VIDEO_ONLY:
                    only_vid = []
                    for m in user_msgs:
                        mm = dict(m)
                        mm['content'] = "\n<video>\n"
                        only_vid.append(mm)
                    obj['messages'] = only_vid
                elif TRIM_TO_QUESTION:
                    trimmed = []
                    for m in user_msgs:
                        c = m.get('content')
                        if isinstance(c, str):
                            # Keep only the first "Question: ...?" segment
                            match = re.search(r'(Question:\s*.+?\?)', c, flags=re.S)
                            if match:
                                new_c = match.group(1).strip()
                                m = dict(m)
                                m['content'] = new_c
                        trimmed.append(m)
                    obj['messages'] = trimmed if trimmed else user_msgs
                else:
                    obj['messages'] = user_msgs
            else:
                # If no user messages, keep as-is
                pass
        # Write back
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        kept += 1

print(f"[+] Stripped assistant messages: written={kept}, total_lines={total}")
PY

echo "[+] Cleaned JSONL at: $JSONL_OUT"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO (hf)"
    hf repo create "$REPO" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload cleaned JSONL (hf)"
    hf upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO (huggingface-cli)"
    huggingface-cli repo create "$REPO" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload cleaned JSONL (huggingface-cli)"
    huggingface-cli upload "$REPO" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  else
    echo "[!] Neither 'hf' nor 'huggingface-cli' found. Install with: pip install -U huggingface_hub" >&2
    exit 1
  fi
  echo "[+] Pushed to: https://huggingface.co/datasets/$REPO"
else
  echo "[=] PUSH disabled; skipping upload."
fi
