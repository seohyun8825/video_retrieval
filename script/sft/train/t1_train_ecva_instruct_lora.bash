#!/usr/bin/env bash

# Ensure this script runs with bash even if invoked via `sh`.
if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then
    exec bash "$0" "$@"
  else
    echo "Please run this script with bash" >&2
    exit 1
  fi
fi

set -euo pipefail

# One‑shot trainer for LLaMA‑Factory with handy overrides and
# auto‑registering a HF dataset into dataset_info.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Defaults (edit below block to change behavior; no --args needed)
DEFAULT_CONFIG_PATH="${ROOT_DIR}/script/sft/train/config/train_full/qwen3vl_2b_instruct_sft_lora.yaml"
CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG_PATH}}"

# User toggles (edit values; env can override via VAR=...)
ANALYZE_TOKEN="${ANALYZE_TOKEN:-0}"                 # 1=첫 3개 샘플 토큰 디버깅
MID_EVAL_ON_SAVE="${MID_EVAL_ON_SAVE:-0}"           # 1=중간평가 활성화
MID_EVAL_FRACTION="${MID_EVAL_FRACTION:-1}"       # 0.5=전체 step의 절반 지점에서 1회 평가
EVAL_RESULT_PUSH="${EVAL_RESULT_PUSH:-1}"            # 1=중간평가 결과 HF push
EVAL_PARSE_MODE="${EVAL_PARSE_MODE:-answer_tag}"    # answer_tag | last_sentence
EVAL_METHOD="${EVAL_METHOD:-exact_match}"           # 현재 exact_match만 사용
EVAL_DATASET_REPO="${EVAL_DATASET_REPO:-happy8825/valid_ecva_clean}"
EVAL_PROMPT="${EVAL_PROMPT:-Are any anomalies directly occurring in this clip? If yes, identify them briefly. }"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-10}"
EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-3}"            # 예: "3" 또는 "2,3" (빈 값이면 학습 GPU 마지막 사용)
EVAL_DEBUG_TIME="${EVAL_DEBUG_TIME:-0}"              # 1=prep/generate/total 시간(ms) 기록
EVAL_DEBUG_MEMORY="${EVAL_DEBUG_MEMORY:-0}"          # 1=CUDA/CPU 메모리(before/after/delta) 기록
# GPU selection (comma separated)
GPU_DEVICES="0,1,2"  # e.g., "4,5"; empty = keep current

# Common overrides you asked to manage from bash
VIDEO_MAXLEN="${VIDEO_MAXLEN:-30}"
DATASET="${DATASET:-}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/script/sft/train/data_config}"
MEDIA_DIR="/hub_data3/seohyun"

LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-70}"
REPORT_TO="${REPORT_TO:-wandb}"

OUTPUT_DIR="${OUTPUT_DIR:-/hub_data4/seohyun/saves/ecva_instruct/lora}"
OUTPUT_DEBUG_DIR="${OUTPUT_DIR}/output_debug"

# Optional: register/override a dataset entry by only specifying HF_HUB_URL
# Example: HF_HUB_URL="happy8825/test_mllm_video_demo"
HF_HUB_URL="happy8825/train_ecva_clean_no_tag"

# If DATASET is empty and HF_HUB_URL is provided, infer dataset name from repo (last segment)
if [[ -z "${DATASET}" && -n "${HF_HUB_URL}" ]]; then
  DATASET="$(basename "${HF_HUB_URL}")"
fi

echo "Config       : ${CONFIG_PATH}"
echo "Dataset      : ${DATASET:-<none>}"
echo "Dataset dir  : ${DATASET_DIR}"
echo "Media dir    : ${MEDIA_DIR}"
echo "HF hub url   : ${HF_HUB_URL:-<leave as is>}"
echo "video_maxlen : ${VIDEO_MAXLEN}"
echo "logging_steps: ${LOGGING_STEPS} | save_steps: ${SAVE_STEPS} | report_to: ${REPORT_TO}"
echo "output_dir   : ${OUTPUT_DIR}"
echo "output_debug : ${OUTPUT_DEBUG_DIR} (analyze_token=${ANALYZE_TOKEN})"
echo "mid_eval     : on_save=${MID_EVAL_ON_SAVE} | push=${EVAL_RESULT_PUSH} | parse=${EVAL_PARSE_MODE} | method=${EVAL_METHOD}"
echo "eval dataset : ${EVAL_DATASET_REPO} | max_samples=${EVAL_MAX_SAMPLES}"
echo "eval prompt  : ${EVAL_PROMPT}"
echo "mid_eval frac: ${MID_EVAL_FRACTION} (0.5=절반 지점)"
echo "eval debug   : time=${EVAL_DEBUG_TIME} | memory=${EVAL_DEBUG_MEMORY}"
echo "GPU devices  : ${GPU_DEVICES:-<inherit>}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${DATASET}" ]]; then
  echo "DATASET is empty and HF_HUB_URL not provided to infer from. Set DATASET=... or HF_HUB_URL=org/repo." >&2
  exit 1
fi

mkdir -p "${DATASET_DIR}"
DATASET_INFO_JSON="${DATASET_DIR}/dataset_info.json"

# Ensure dataset_info.json exists
if [[ ! -f "${DATASET_INFO_JSON}" ]]; then
  echo "{}" > "${DATASET_INFO_JSON}"
fi

# If HF_HUB_URL is provided, (up)insert the dataset entry with default schema.
if [[ -n "${HF_HUB_URL}" ]]; then
  python3 - "$DATASET_INFO_JSON" "$DATASET" "$HF_HUB_URL" << 'PY'
import json, sys

path, name, hub = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
if not isinstance(data, dict):
    data = {}

entry = {
    "hf_hub_url": hub,
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "videos": "videos"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}

# upsert
data[name] = {**data.get(name, {}), **entry}

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"[dataset_info.json] upserted entry '{name}' with hf_hub_url='{hub}' -> {path}")
PY
fi

# Optional: enable in-trainer token analysis (first 3 samples)
if [[ "${ANALYZE_TOKEN}" == "1" ]]; then
  echo "[token-analyze] Enabled (trainer hook). Logs: ${OUTPUT_DEBUG_DIR}/first3_from_trainer.jsonl"
  mkdir -p "${OUTPUT_DEBUG_DIR}"
fi

# Optional: mid-training eval watcher (runs on each new checkpoint-*)
run_eval_for_ckpt() {
  # ensure no xtrace noise inside eval runner
  set +x 2>/dev/null || true
  local ckpt_dir="$1"
  local step_tag
  step_tag="$(basename "${ckpt_dir}")"  # checkpoint-500
  local step_num
  step_num="${step_tag#checkpoint-}"

  local MODEL_NAME_TAG
  MODEL_NAME_TAG="${MODEL_NAME_TAG:-$(basename "${OUTPUT_DIR}")}"
  local EV_DIR
  EV_DIR="${OUTPUT_DIR}/eval"
  mkdir -p "${EV_DIR}"
  local OUT_JSON
  OUT_JSON="${EV_DIR}/${MODEL_NAME_TAG}_step${step_num}.json"
  if [[ -s "${OUT_JSON}" ]]; then
    echo "[mid-eval] skip existing ${OUT_JSON}"
    return 0
  fi

  # Determine eval GPUs; support sharded multi-GPU runs like "2,3"
  local eval_cuda
  if [[ -n "${EVAL_GPU_DEVICES}" ]]; then
    eval_cuda="${EVAL_GPU_DEVICES}"
  else
    IFS=',' read -r -a __g <<< "${GPU_DEVICES}"
    if (( ${#__g[@]} >= 1 ));
    then eval_cuda="${__g[-1]}"; else eval_cuda="${GPU_DEVICES}"; fi
  fi

  IFS=',' read -r -a __eval_gpus <<< "${eval_cuda}"
  local shard_count=${#__eval_gpus[@]}

  echo "[mid-eval] step=${step_num} ckpt=${ckpt_dir} → ${OUT_JSON} (CUDA=${eval_cuda}; shards=${shard_count})"
  export PYTHONPATH="${ROOT_DIR}/SFT/sft_trainer/LLaMA-Factory/src:${ROOT_DIR}/SFT/sft_trainer:${PYTHONPATH:-}"
  extra_parse=()
  if [[ "${EVAL_PARSE_MODE}" == "last_sentence" ]]; then
    extra_parse+=(--last_sentence_parsing)
  else
    extra_parse+=(--answer_tag_parsing)
  fi
  debug_flags=()
  if [[ "${EVAL_DEBUG_TIME}" == "1" ]]; then
    debug_flags+=(--debug_time)
  fi
  if [[ "${EVAL_DEBUG_MEMORY}" == "1" ]]; then
    debug_flags+=(--debug_memory)
  fi

  if (( shard_count > 1 )); then
    # Launch one shard per GPU then merge
    local shard_jsons=()
    local idx=0
    for gpu in "${__eval_gpus[@]}"; do
      local shard_out="${OUT_JSON%.json}.shard${idx}.json"
      shard_jsons+=("${shard_out}")
      (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_valid_all.py" \
          --model_repo "${ckpt_dir}" \
          --dataset_repo "${EVAL_DATASET_REPO}" \
          --output_json "${shard_out}" \
          --media_base "${MEDIA_DIR}" \
          --template "qwen3_vl" \
          --video_fps 2.0 \
          --video_maxlen "${VIDEO_MAXLEN}" \
          --max_samples "${EVAL_MAX_SAMPLES}" \
          --prepend_prompt "${EVAL_PROMPT}" \
          --gt_is_label \
          --evqa \
          --num_shards "${shard_count}" \
          --shard_index "${idx}" \
          "${extra_parse[@]}" \
          "${debug_flags[@]}"
      ) &
      idx=$((idx+1))
    done
    wait || true
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/merge_infer_results.py" \
      --inputs "${shard_jsons[@]}" \
      --output "${OUT_JSON}" \
      --model_repo "${ckpt_dir}" \
      --dataset_repo "${EVAL_DATASET_REPO}"
    # Optional: write per-sample predictions as JSONL next to OUT_JSON
    if [[ "${EVAL_SAVE_PREDS_JSONL:-1}" == "1" ]]; then
      python3 - << 'PY' "${OUT_JSON}"
import json, os, sys
inp = sys.argv[1]
out = os.path.splitext(inp)[0] + ".jsonl"
try:
    with open(inp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data.get('items') or []
    # each item is expected to contain fields like: id, video(s), prompt, pred, gt, correct, etc.
    with open(out, 'w', encoding='utf-8') as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[mid-eval] wrote per-sample JSONL: {out} ({len(items)} records)")
except Exception as e:
    print(f"[mid-eval][warn] jsonl write failed: {e}")
PY
    fi
  else
    (
      export CUDA_VISIBLE_DEVICES="${eval_cuda}"
      python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_valid_all.py" \
        --model_repo "${ckpt_dir}" \
        --dataset_repo "${EVAL_DATASET_REPO}" \
        --output_json "${OUT_JSON}" \
        --media_base "${MEDIA_DIR}" \
        --template "qwen3_vl" \
        --video_maxlen "${VIDEO_MAXLEN}" \
        --max_samples "${EVAL_MAX_SAMPLES}" \
        --prepend_prompt "${EVAL_PROMPT}" \
        --gt_is_label \
        --evqa \
        "${extra_parse[@]}" \
        "${debug_flags[@]}"
    ) || echo "[mid-eval][warn] eval failed for ${ckpt_dir}"
    # Optional: write per-sample predictions as JSONL next to OUT_JSON
    if [[ "${EVAL_SAVE_PREDS_JSONL:-1}" == "1" && -s "${OUT_JSON}" ]]; then
      python3 - << 'PY' "${OUT_JSON}"
import json, os, sys
inp = sys.argv[1]
out = os.path.splitext(inp)[0] + ".jsonl"
try:
    with open(inp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data.get('items') or []
    with open(out, 'w', encoding='utf-8') as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[mid-eval] wrote per-sample JSONL: {out} ({len(items)} records)")
except Exception as e:
    print(f"[mid-eval][warn] jsonl write failed: {e}")
PY
    fi
  fi

  if [[ "${EVAL_RESULT_PUSH}" == "1" ]]; then
    # Push JSON + README with metrics to HF using the same helper as t3 script
    local HF_README_EXTRA
    HF_README_EXTRA="\n\nMid-training eval for ${MODEL_NAME_TAG} at step ${step_num}."
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/push_results_to_hf.py" \
      --input_json "${OUT_JSON}" \
      --readme_extra "${HF_README_EXTRA}"
  fi

  # Log to W&B if training uses wandb
  if [[ "${REPORT_TO}" == "wandb" ]]; then
    python3 - "$OUT_JSON" "$step_num" << 'PY'
import json, os, sys
out, step = sys.argv[1], int(sys.argv[2])
try:
    import wandb
except Exception:
    sys.exit(0)
with open(out, 'r', encoding='utf-8') as f:
    data = json.load(f)
m = data.get('metrics') or {}
run = wandb.init(project=os.getenv('WANDB_PROJECT', None), name=f"mid-eval_step{step}", group=os.getenv('WANDB_RUN_GROUP', 'mid-eval'), reinit=True)
to_log = {}
for k in ('top1_acc','recall_at_5','mrr','evqa_acc','evqa_total','evqa_with_gt_label'):
    if k in m:
        to_log[f"eval/{k}"] = m[k]
if to_log:
    wandb.log(to_log, step=step)
wandb.finish()
PY
  fi
}

start_eval_watcher() {
  # disable xtrace for background watcher to avoid noisy logs
  set +x 2>/dev/null || true
  echo "[mid-eval] watcher enabled (evaluate every new checkpoint); dataset=${EVAL_DATASET_REPO} parse=${EVAL_PARSE_MODE} push=${EVAL_RESULT_PUSH}"
  mkdir -p "${OUTPUT_DIR}/eval"
  # Evaluate each newly detected checkpoint. run_eval_for_ckpt skips if output JSON exists.
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    for d in "${OUTPUT_DIR}"/checkpoint-*; do
      [[ -d "${d}" ]] || continue
      run_eval_for_ckpt "${d}"
    done
    sleep 60
  done
}

# Configure GPUs
if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
  # Derive nproc from the number of listed devices
  IFS=',' read -r -a __gpus <<< "${GPU_DEVICES}"
  export NPROC_PER_NODE="${#__gpus[@]}"
fi

# Launch training with overrides merged into YAML (OmegaConf CLI style)
set -x
# Run training (background for watcher), then wait
LMF_ANALYZE_TOKEN="${ANALYZE_TOKEN}" LMF_ANALYZE_MAX_SAMPLES=3 \
  llamafactory-cli train "${CONFIG_PATH}" \
  dataset="${DATASET}" \
  dataset_dir="${DATASET_DIR}" \
  media_dir="${MEDIA_DIR}" \
  video_maxlen="${VIDEO_MAXLEN}" \
  logging_steps="${LOGGING_STEPS}" \
  save_steps="${SAVE_STEPS}" \
  report_to="${REPORT_TO}" \
  output_dir="${OUTPUT_DIR}" &
TRAIN_PID=$!

set +x

if [[ "${MID_EVAL_ON_SAVE}" == "1" ]]; then
  # Ensure eval log dir exists before redirect (redirection opens file before function runs)
  mkdir -p "${OUTPUT_DIR}/eval"
  # Route watcher chatter into a log file to keep terminal clean
  start_eval_watcher >"${OUTPUT_DIR}/eval/watcher.log" 2>&1 &
  WATCHER_PID=$!
fi

wait "${TRAIN_PID}"

if [[ "${MID_EVAL_ON_SAVE}" == "1" ]]; then
  # Give watcher a final pass for any last checkpoint, then stop
  sleep 5
  if [[ -n "${WATCHER_PID:-}" ]]; then
    kill "${WATCHER_PID}" 2>/dev/null || true
  fi
  # If target step never computed, fallback: run on the latest checkpoint once
  if [[ ! -f "${OUTPUT_DIR}/eval/.mid_eval_done" ]]; then
    last_ckpt="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
    if [[ -n "${last_ckpt}" ]]; then
      run_eval_for_ckpt "${last_ckpt}"
      : > "${OUTPUT_DIR}/eval/.mid_eval_done"
    fi
  fi
fi
