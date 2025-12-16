#!/usr/bin/env bash
set -euo pipefail

# Filter a HF dataset JSONL using an ECVA check report (TSV) and push cleaned data.
# - Reads BROKEN entries from the report (2nd column = REL_PATH)
# - Downloads data.jsonl from --repo-in and drops samples whose any video matches a BROKEN path
# - Uploads filtered JSONL to --repo-out as data.jsonl
#
# Defaults (override via flags or env):
#   REPO_IN=happy8825/train_ecva
#   REPO_OUT=happy8825/train_ecva_clean
#   REPORT=/hub_data4/seohyun/_ecva_check_report.tsv
#   JSONL_OUT=/home/seohyun/vid_understanding/video_retrieval/data/train_ecva_filtered.jsonl
#   PUSH=1
#
# Usage:
#   bash filter_train_by_report.bash \
#     [--repo-in REPO] [--repo-out REPO] [--report PATH] [--jsonl-out PATH] [--no-push]

REPO_IN_DEFAULT="happy8825/train_ecva"
REPO_OUT_DEFAULT="happy8825/train_ecva_clean"
REPORT_DEFAULT="/hub_data4/seohyun/_ecva_check_report.tsv"
JSONL_OUT_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/train_ecva_filtered.jsonl"

REPO_IN="${REPO_IN:-$REPO_IN_DEFAULT}"
REPO_OUT="${REPO_OUT:-$REPO_OUT_DEFAULT}"
REPORT="${REPORT:-$REPORT_DEFAULT}"
JSONL_OUT="${JSONL_OUT:-$JSONL_OUT_DEFAULT}"
PUSH="${PUSH:-1}"

usage() {
  echo "Usage: $0 [--repo-in REPO] [--repo-out REPO] [--report PATH] [--jsonl-out PATH] [--no-push]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-in) REPO_IN="$2"; shift 2 ;;
    --repo-out) REPO_OUT="$2"; shift 2 ;;
    --report) REPORT="$2"; shift 2 ;;
    --jsonl-out) JSONL_OUT="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "$REPORT" ]]; then
  echo "[!] Report not found: $REPORT" >&2
  exit 1
fi

echo "[+] REPO_IN=$REPO_IN"
echo "[+] REPO_OUT=$REPO_OUT"
echo "[+] REPORT=$REPORT"
echo "[+] JSONL_OUT=$JSONL_OUT"
echo "[+] PUSH=$PUSH"

python3 - "$REPO_IN" "$REPORT" "$JSONL_OUT" << 'PY'
import csv
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    print("[!] huggingface_hub not installed. pip install -U huggingface_hub", file=sys.stderr)
    sys.exit(1)

REPO_IN = sys.argv[1]
REPORT = Path(sys.argv[2])
OUT = Path(sys.argv[3])

def find_jsonl(repo_id: str) -> str:
    files = list_repo_files(repo_id, repo_type="dataset")
    if "data.jsonl" in files:
        return "data.jsonl"
    for f in files:
        if f.lower().endswith(".jsonl"):
            return f
    raise FileNotFoundError("No JSONL found in repo.")

# Load broken set from TSV report (STATUS, REL_PATH, FRAMES, DURATION)
broken = set()
with REPORT.open('r', encoding='utf-8') as rf:
    reader = csv.reader(rf, delimiter='\t')
    for i, row in enumerate(reader):
        if not row:
            continue
        if i == 0 and row[0].upper().startswith('STATUS'):
            continue
        status = (row[0] if len(row) > 0 else '').strip().upper()
        rel = (row[1] if len(row) > 1 else '').strip()
        if status == 'BROKEN' and rel:
            broken.add(rel)

print(f"[+] Loaded broken paths: {len(broken)}")

name = find_jsonl(REPO_IN)
print(f"[+] Downloading: {REPO_IN}/{name}")
src = hf_hub_download(REPO_IN, filename=name, repo_type="dataset")

kept = 0
dropped = 0
total = 0
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(src, 'r', encoding='utf-8') as fin, OUT.open('w', encoding='utf-8') as fout:
    for line in fin:
        total += 1
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            dropped += 1
            continue
        vids = obj.get('videos') or []
        if any((v in broken) for v in vids if isinstance(v, str)):
            dropped += 1
            continue
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print(f"[+] Filtered: kept={kept}, dropped={dropped}, total={total}")
PY

echo "[+] Cleaned JSONL at: $JSONL_OUT"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO_OUT (hf)"
    hf repo create "$REPO_OUT" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload cleaned JSONL (hf)"
    hf upload "$REPO_OUT" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensure repo exists: $REPO_OUT (huggingface-cli)"
    huggingface-cli repo create "$REPO_OUT" --type dataset >/dev/null 2>&1 || true
    echo "[+] Upload cleaned JSONL (huggingface-cli)"
    huggingface-cli upload "$REPO_OUT" "$JSONL_OUT" "data.jsonl" --repo-type dataset
  else
    echo "[!] Neither 'hf' nor 'huggingface-cli' found. Install with: pip install -U huggingface_hub" >&2
    exit 1
  fi
  echo "[+] Pushed to: https://huggingface.co/datasets/$REPO_OUT"
else
  echo "[=] PUSH disabled; skipping upload."
fi

