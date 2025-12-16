#!/usr/bin/env bash
set -euo pipefail

# Combine ECVA SFT JSONLs, balance abnormal to normal count, split 70/30,
# add ground-truth column for valid set only, and push to HF Hub.
#
# Inputs (defaults, override via flags/env):
#   AB_JSON=/home/seohyun/vid_understanding/video_retrieval/data/abnormal_sft.jsonl
#   NORM_JSON=/home/seohyun/vid_understanding/video_retrieval/data/normal_sft.jsonl
#   AFTER_JSON=/home/seohyun/vid_understanding/video_retrieval/data/after_incident_sft.jsonl
#   OUT_TRAIN=/home/seohyun/vid_understanding/video_retrieval/data/train_ecva.jsonl
#   OUT_VALID=/home/seohyun/vid_understanding/video_retrieval/data/valid_ecva.jsonl
#   REPO_TRAIN=happy8825/train_ecva
#   REPO_VALID=happy8825/valid_ecva
#   VALID_RATIO=0.3             # fraction for valid split
#   SEED=42                     # set for deterministic sampling/shuffle
#   PUSH=1                      # 1=push to HF, 0=only generate locally
#   AB_COUNT=0                  # if >0, select exactly this many abnormal; else match normals
#
# Usage:
#   bash split_ecva_train_val.sh \
#     [--ab-json PATH] [--norm-json PATH] [--after-json PATH] \
#     [--out-train PATH] [--out-valid PATH] \
#     [--repo-train REPO] [--repo-valid REPO] \
#     [--valid-ratio R] [--seed N] [--no-push] [--ab-count N]

AB_JSON_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/abnormal_sft.jsonl"
NORM_JSON_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/normal_sft.jsonl"
AFTER_JSON_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/after_incident_sft.jsonl"

OUT_TRAIN_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/train_ecva.jsonl"
OUT_VALID_DEFAULT="/home/seohyun/vid_understanding/video_retrieval/data/valid_ecva.jsonl"

REPO_TRAIN_DEFAULT="happy8825/train_ecva"
REPO_VALID_DEFAULT="happy8825/valid_ecva"

AB_JSON="${AB_JSON:-$AB_JSON_DEFAULT}"
NORM_JSON="${NORM_JSON:-$NORM_JSON_DEFAULT}"
AFTER_JSON="${AFTER_JSON:-$AFTER_JSON_DEFAULT}"
OUT_TRAIN="${OUT_TRAIN:-$OUT_TRAIN_DEFAULT}"
OUT_VALID="${OUT_VALID:-$OUT_VALID_DEFAULT}"
REPO_TRAIN="${REPO_TRAIN:-$REPO_TRAIN_DEFAULT}"
REPO_VALID="${REPO_VALID:-$REPO_VALID_DEFAULT}"
VALID_RATIO="${VALID_RATIO:-0.3}"
SEED_VAL="${SEED:-42}"
PUSH="${PUSH:-1}"
AB_COUNT="${AB_COUNT:-0}"

usage() {
  echo "Usage: $0 [--ab-json PATH] [--norm-json PATH] [--after-json PATH] [--out-train PATH] [--out-valid PATH] [--repo-train REPO] [--repo-valid REPO] [--valid-ratio R] [--seed N] [--no-push] [--ab-count N]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ab-json) AB_JSON="$2"; shift 2;;
    --norm-json) NORM_JSON="$2"; shift 2;;
    --after-json) AFTER_JSON="$2"; shift 2;;
    --out-train) OUT_TRAIN="$2"; shift 2;;
    --out-valid) OUT_VALID="$2"; shift 2;;
    --repo-train) REPO_TRAIN="$2"; shift 2;;
    --repo-valid) REPO_VALID="$2"; shift 2;;
    --valid-ratio) VALID_RATIO="$2"; shift 2;;
    --seed) SEED_VAL="$2"; shift 2;;
    --no-push) PUSH=0; shift;;
    --ab-count) AB_COUNT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

for p in "$AB_JSON" "$NORM_JSON" "$AFTER_JSON"; do
  if [[ ! -f "$p" ]]; then
    echo "[!] Missing input JSONL: $p" >&2
    exit 1
  fi
done

for outp in "$OUT_TRAIN" "$OUT_VALID"; do
  mkdir -p "$(dirname "$outp")"
done

echo "[+] AB_JSON=$AB_JSON"
echo "[+] NORM_JSON=$NORM_JSON"
echo "[+] AFTER_JSON=$AFTER_JSON"
echo "[+] OUT_TRAIN=$OUT_TRAIN"
echo "[+] OUT_VALID=$OUT_VALID"
echo "[+] REPO_TRAIN=$REPO_TRAIN"
echo "[+] REPO_VALID=$REPO_VALID"
echo "[+] VALID_RATIO=$VALID_RATIO"
echo "[+] SEED=$SEED_VAL"
echo "[+] AB_COUNT=$AB_COUNT (0=match normals)"
echo "[+] PUSH=$PUSH"

SEED="$SEED_VAL" AB_COUNT="$AB_COUNT" VALID_RATIO="$VALID_RATIO" \
python3 - "$AB_JSON" "$NORM_JSON" "$AFTER_JSON" "$OUT_TRAIN" "$OUT_VALID" << 'PY'
import json
import os
import random
import sys
from pathlib import Path

AB_JSON = Path(sys.argv[1])
NORM_JSON = Path(sys.argv[2])
AFTER_JSON = Path(sys.argv[3])
OUT_TRAIN = Path(sys.argv[4])
OUT_VALID = Path(sys.argv[5])

seed = os.environ.get('SEED')
random.seed(None if seed is None else int(seed))

try:
    valid_ratio = float(os.environ.get('VALID_RATIO', '0.3'))
except Exception:
    valid_ratio = 0.3

try:
    ab_count_target = int(os.environ.get('AB_COUNT', '0'))
except Exception:
    ab_count_target = 0

def read_jsonl(p: Path):
    data = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

ab = read_jsonl(AB_JSON)
nr = read_jsonl(NORM_JSON)
af = read_jsonl(AFTER_JSON)

normals = nr + af
num_normals = len(normals)
num_ab = len(ab)

if ab_count_target > 0:
    sel_ab_count = min(ab_count_target, num_ab)
else:
    sel_ab_count = min(num_normals, num_ab)

# sample abnormal subset
ab_idx = list(range(num_ab))
random.shuffle(ab_idx)
ab_sel = [ab[i] for i in ab_idx[:sel_ab_count]]

combined = []
for x in normals:
    combined.append((x, 'normal'))
for x in ab_sel:
    combined.append((x, 'abnormal'))

random.shuffle(combined)
total = len(combined)
valid_n = int(total * valid_ratio)
train_n = total - valid_n

train_items = combined[:train_n]
valid_items = combined[train_n:]

with OUT_TRAIN.open('w', encoding='utf-8') as ft:
    for obj, label in train_items:
        # Train set: leave as-is (no gt column requested)
        ft.write(json.dumps(obj, ensure_ascii=False) + "\n")

with OUT_VALID.open('w', encoding='utf-8') as fv:
    for obj, label in valid_items:
        obj2 = dict(obj)
        obj2['gt'] = label  # add ground-truth only to valid set
        fv.write(json.dumps(obj2, ensure_ascii=False) + "\n")

print(f"[+] Stats: normals={num_normals}, abnormal_total={num_ab}, abnormal_selected={sel_ab_count}, total_combined={total}, train={train_n}, valid={valid_n}")
PY

echo "[+] Train/Valid JSONL written:"
echo "    train → $OUT_TRAIN"
echo "    valid → $OUT_VALID"

if [[ "$PUSH" == "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repos (hf)"
    hf repo create "$REPO_TRAIN" --type dataset >/dev/null 2>&1 || true
    hf repo create "$REPO_VALID" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload train JSONL (hf)"
    hf upload "$REPO_TRAIN" "$OUT_TRAIN" "data.jsonl" --repo-type dataset

    echo "[+] Upload valid JSONL (hf)"
    hf upload "$REPO_VALID" "$OUT_VALID" "data.jsonl" --repo-type dataset

  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[+] Ensuring dataset repos (huggingface-cli)"
    huggingface-cli repo create "$REPO_TRAIN" --type dataset >/dev/null 2>&1 || true
    huggingface-cli repo create "$REPO_VALID" --type dataset >/dev/null 2>&1 || true

    echo "[+] Upload train JSONL (huggingface-cli)"
    huggingface-cli upload "$REPO_TRAIN" "$OUT_TRAIN" "data.jsonl" --repo-type dataset

    echo "[+] Upload valid JSONL (huggingface-cli)"
    huggingface-cli upload "$REPO_VALID" "$OUT_VALID" "data.jsonl" --repo-type dataset

  else
    echo "[!] Neither 'hf' nor 'huggingface-cli' found. Install with: pip install -U huggingface_hub" >&2
    exit 1
  fi

  echo "[+] Push complete:"
  echo "    train → https://huggingface.co/datasets/$REPO_TRAIN"
    echo "    valid → https://huggingface.co/datasets/$REPO_VALID"
else
  echo "[=] PUSH disabled (use --no-push to skip)."
fi

