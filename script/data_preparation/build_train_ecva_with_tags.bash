#!/usr/bin/env bash
set -euo pipefail

# Build a new HF dataset from an existing ECVA JSON/JSONL by:
# - Replace user message with a standardized instruction:
#   "<video>\n\nQuestion: Is there a moment where an anomaly is directly happening in-frame? Briefly state what it is inside <think> tag, and give your final answer inside <answer> tag."
# - Preserve the original assistant content by wrapping it with <think> ... </think>,
#   and append an exact answer tag: <answer>normal|abnormal</answer> based on video path.
#   based on the folder of the first video:
#     - ecva/abnormal_video/...   -> abnormal
#     - ecva/normal_video/...     -> normal
#     - ecva/after_incident/...   -> normal
# - Push the resulting JSONL back to HF as a new dataset repo.
#
# Usage:
#   bash build_train_ecva_with_tags.bash \
#     --repo-in happy8825/train_ecva_clean \
#     --repo-out happy8825/train_ecva_clean_with_tag \
#     [--jsonl-out /path/to/out.jsonl] [--no-push]

REPO_IN=""
REPO_OUT=""
JSONL_OUT="/home/seohyun/vid_understanding/video_retrieval/data/train_ecva_with_tag.jsonl"
PUSH=1

usage(){
  echo "Usage: $0 --repo-in ORG/NAME --repo-out ORG/NAME [--jsonl-out PATH] [--no-push]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-in) REPO_IN="$2"; shift 2 ;;
    --repo-out) REPO_OUT="$2"; shift 2 ;;
    --jsonl-out) JSONL_OUT="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$REPO_IN" || -z "$REPO_OUT" ]]; then
  echo "[!] --repo-in and --repo-out are required" >&2
  usage; exit 1
fi

echo "[+] REPO_IN=$REPO_IN"
echo "[+] REPO_OUT=$REPO_OUT"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] PUSH=$PUSH"

python3 - "$REPO_IN" "$JSONL_OUT" << 'PY'
import json, os, re, sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception as e:
    print('[!] huggingface_hub not installed. pip install -U huggingface_hub', file=sys.stderr)
    raise

REPO_IN = sys.argv[1]
OUT_PATH = Path(sys.argv[2])

def find_input(repo_id: str) -> str:
    files = list_repo_files(repo_id, repo_type='dataset')
    # prefer JSONL if present; else any JSON
    if 'data.jsonl' in files:
        return 'data.jsonl'
    if 'data.json' in files:
        return 'data.json'
    for f in files:
        if f.lower().endswith('.jsonl'):
            return f
    for f in files:
        if f.lower().endswith('.json'):
            return f
    raise FileNotFoundError('no data.json(l) found')

name = find_input(REPO_IN)
src = hf_hub_download(REPO_IN, filename=name, repo_type='dataset')

def iter_samples(path: str, name: str):
    if name.lower().endswith('.jsonl'):
        with open(path,'r',encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue
    else:
        data = json.load(open(path,'r',encoding='utf-8'))
        if isinstance(data, list):
            for o in data:
                if isinstance(o, dict):
                    yield o

def label_from_videos(videos):
    vids = videos or []
    if not vids:
        return None
    v0 = (vids[0] or '').lower()
    if '/abnormal_video/' in v0:
        return 'abnormal'
    if '/normal_video/' in v0 or '/after_incident/' in v0:
        return 'normal'
    # fallback: majority voting across all videos
    ab = any(('/abnormal_video/' in (v or '').lower()) for v in vids)
    if ab:
        return 'abnormal'
    nm = any(('/normal_video/' in (v or '').lower() or '/after_incident/' in (v or '').lower()) for v in vids)
    if nm:
        return 'normal'
    return None

STANDARD_USER = (
    "<video>\n\n"
    "Question: Is there a moment where an anomaly is directly happening in-frame? "
    "Briefly state what it is inside <think> tag, and give your final answer inside <answer> tag."
)

def build_assistant_from(old_text, label: str) -> str:
    text = (old_text or '').strip()
    if not text:
        # Sensible fallbacks if original assistant is missing
        text = 'An abnormal event is directly happening in-frame.' if label == 'abnormal' else 'All visible activity looks typical, with nothing clearly abnormal.'
    return f"<think> {text} </think> <answer>{label}</answer>"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
written = 0
with open(OUT_PATH,'w',encoding='utf-8') as fout:
    for ex in iter_samples(src, name):
        msgs = ex.get('messages') or []
        user = next((m for m in msgs if isinstance(m, dict) and m.get('role')=='user'), None)
        asst = next((m for m in msgs if isinstance(m, dict) and m.get('role')=='assistant'), None)
        videos = ex.get('videos') or []
        label = label_from_videos(videos)
        # Build new messages
        new_msgs = []
        # Always use standardized user prompt
        new_msgs.append({"role":"user","content": STANDARD_USER})
        # Add assistant with exact tag
        if label in ('normal','abnormal'):
            old_text = (asst or {}).get('content') if isinstance(asst, dict) else ''
            new_msgs.append({"role":"assistant","content": build_assistant_from(old_text, label)})
        else:
            # Unknown label; keep without assistant
            pass
        out_obj = dict(ex)
        out_obj['messages'] = new_msgs
        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        written += 1

print(f"[+] Wrote {written} lines to {OUT_PATH}")
PY

echo "[+] JSONL written: $JSONL_OUT"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO_OUT (hf)"
    hf repo create "$REPO_OUT" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload JSONL (hf)"
    hf upload "$REPO_OUT" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO_OUT (huggingface-cli)"
    huggingface-cli repo create "$REPO_OUT" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload JSONL (huggingface-cli)"
    huggingface-cli upload "$REPO_OUT" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  else
    echo "[!] Neither 'hf' nor 'huggingface-cli' found. Install with: pip install -U huggingface_hub" >&2
    exit 1
  fi
  echo "[+] Pushed to: https://huggingface.co/datasets/$REPO_OUT"
else
  echo "[=] PUSH disabled (no-push)."
fi
