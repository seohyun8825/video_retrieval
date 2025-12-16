#!/usr/bin/env bash
set -euo pipefail

# Auto-diagnose an HF dataset repo (JSONL) against local videos and
# re-push a cleaned JSONL with BAD video samples removed to the same repo.
#
# It chains:
#   1) diagnose_hf_jsonl_videos.bash  --repo <REPO> --media-root <DIR>
#   2) Build a temporary TSV report from BAD entries
#   3) filter_train_by_report.bash     --repo-in <REPO> --repo-out <REPO> --report <TSV>
#   4) Re-diagnose to confirm cleanup
#
# Usage:
#   bash auto_clean_hf_repo.bash --repo happy8825/train_ecva_clean \
#       [--media-root /hub_data4/seohyun] [--max 0]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO=""
MEDIA_ROOT="/hub_data4/seohyun"
MAX="0"   # 0 = all

usage(){
  echo "Usage: $0 --repo ORG/NAME [--media-root DIR] [--max N]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --media-root) MEDIA_ROOT="$2"; shift 2 ;;
    --max) MAX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "[!] --repo is required" >&2; usage; exit 1
fi

DIAG="$SCRIPT_DIR/diagnose_hf_jsonl_videos.bash"
FILTER="$SCRIPT_DIR/filter_train_by_report.bash"
if [[ ! -x "$DIAG" ]]; then
  echo "[!] Not found: $DIAG" >&2; exit 1
fi
if [[ ! -x "$FILTER" ]]; then
  echo "[!] Not found: $FILTER" >&2; exit 1
fi

echo "[+] Repo=$REPO | media_root=$MEDIA_ROOT | max=$MAX"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
DIAG_LOG="$TMP_DIR/diag.txt"
BAD_LIST="$TMP_DIR/bad_relpaths.txt"
TSV="$TMP_DIR/block.tsv"

echo "[+] Diagnosing current repo..."
bash "$DIAG" --repo "$REPO" --media-root "$MEDIA_ROOT" ${MAX:+--max "$MAX"} | tee "$DIAG_LOG"

awk '/^\[BAD\]/{print $2}' "$DIAG_LOG" | sort -u > "$BAD_LIST" || true
BAD_N=$(wc -l < "$BAD_LIST" | awk '{print $1}')
echo "[+] BAD entries: $BAD_N"

if [[ "$BAD_N" -eq 0 ]]; then
  echo "[=] No BAD samples detected. Nothing to clean."
  exit 0
fi

{ echo -e "STATUS\tREL_PATH\tFRAMES\tDURATION"; awk '{printf "BROKEN\t%s\t0\t-1\n", $0}' "$BAD_LIST"; } > "$TSV"
echo "[+] Built block TSV: $TSV"

echo "[+] Filtering and re-pushing cleaned JSONL back to $REPO ..."
bash "$FILTER" --repo-in "$REPO" --repo-out "$REPO" --report "$TSV"

echo "[+] Re-diagnosing after cleanup..."
bash "$DIAG" --repo "$REPO" --media-root "$MEDIA_ROOT" ${MAX:+--max "$MAX"} | tee "$TMP_DIR/diag_after.txt"

echo "[+] Done."

