#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Conda env activation (optional)
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "llama_factory" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # Conda activation scripts are not nounset-safe; temporarily disable -u.
    nounset_was_set=0
    case $- in
      *u*) nounset_was_set=1 ;;
    esac
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llama_factory || true
    if [[ "${nounset_was_set}" -eq 1 ]]; then
      set -u
    fi
  fi
fi

# Ensure local LLaMA-Factory sources are on PYTHONPATH (for 'llamafactory' imports)
export PYTHONPATH="${ROOT_DIR}/SFT/sft_trainer/LLaMA-Factory/src:${ROOT_DIR}/SFT/sft_trainer:${PYTHONPATH:-}"

# Ensure we have a writable temp directory (avoid /tmp quota issues)
if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR="/hub_data1/seohyun/tmp"
fi
mkdir -p "${TMPDIR}"
export TMPDIR

# Simple flags parsing (script-level)
ECVA_MODE=0
for arg in "$@"; do
  case "$arg" in
    --ecva) ECVA_MODE=1 ;;
  esac
done

# Defaults (override via env)
: "${MODEL_REPO:=Qwen/Qwen3-VL-2B-Instruct}"
: "${DATASET_REPO:=happy8825/valid_ecva_clean}"
: "${DATASET_FILE:=}"
: "${OUTPUT_JSON:=${ROOT_DIR}/video_retrieval/output_ecva_lora/lora_0.9.json}"
OUTPUT_DIR="$(dirname "${OUTPUT_JSON}")"

: "${MEDIA_BASE:=/hub_data3/seohyun}"
: "${TEMPLATE:=qwen3_vl}"
: "${VIDEO_FPS:=2.0}"
# Follow t1_train settings: video_maxlen=30
: "${VIDEO_MAXLEN:=20}"
: "${MAX_SAMPLES:=5000}"
: "${VIDEO_NUM_FRAMES:=20}"
: "${VIDEO_TOTAL_PIXELS:=402144}"
: "${VIDEO_MIN_PIXELS:=0}"
: "${LOG_VIDEO_FRAMES:=false}"
# Safer decoding: request <= actual frames to avoid API 500s like
# "Expected reading 48 frames, but only loaded 47 frames".
# When true, the Python runner will probe frame counts (OpenCV/ffprobe)
# and cap nframes (with an off-by-one guard).
: "${SAFE_INFER:=true}"
# Prefer .mp4 over .mkv when both exist (often more reliably decodable)
: "${PREFER_MP4:=true}"
: "${SYSTEM_PROMPT:=}"
: "${TMP_VIDEO_DIR:=}"

# Prompt to prepend before the user Query. (Follow t1_train prompt)
: "${PROMPT:=Are any anomalies directly occurring in this clip? If yes, identify them briefly. }"

# API settings (vLLM server for tuned instruct model)
: "${API_BASE:=http://localhost:8630/v1}"
: "${API_KEY:=}"
: "${REQUEST_TIMEOUT:=240}"
: "${MAX_RETRIES:=4}"
: "${API_TEMPERATURE:=0.2}"
: "${API_TOP_P:=0.9}"
: "${API_MAX_NEW_TOKENS:=2048}"
: "${API_PRESENCE_PENALTY:=1.0}"

# Concurrency / Streaming
: "${GPU_DEVICES:=0}"
: "${MAX_CONCURRENT:=4}"
export MAX_CONCURRENT

: "${STREAM_JSONL:=true}"
: "${TRUNCATE_JSONL:=true}"
: "${JSONL_PATH:=${OUTPUT_JSON%.json}.jsonl}"
JSONL_DIR="$(dirname "${JSONL_PATH}")"
: "${HF_UPLOAD_PATH:=${JSONL_PATH}}"
HF_UPLOAD_DIR="$(dirname "${HF_UPLOAD_PATH}")"

# Push to HF
: "${PUSH_HF:=true}"
: "${PUSH_ONLY:=false}"
__out_base="$(basename "${OUTPUT_JSON}")"
: "${HF_OUT_FILE:=$(basename "${HF_UPLOAD_PATH}")}"
: "${HF_OUT_REPO:=${__out_base%.*}}"
: "${HF_README_PATH:=${OUTPUT_DIR}/README.md}"
: "${HF_README_EXTRA:=}"
: "${HF_COMMIT_MESSAGE:=}"
: "${HF_DELETE_REMOTE:=}"

# Resume/skip controls
: "${SKIP_IF_EXISTS:=true}"

echo "Model repo   : ${MODEL_REPO}"
echo "Dataset repo : ${DATASET_REPO} (${DATASET_FILE:-auto})"
echo "Output json  : ${OUTPUT_JSON}"
echo "Media base   : ${MEDIA_BASE}"
echo "Template     : ${TEMPLATE} | fps=${VIDEO_FPS} | maxlen=${VIDEO_MAXLEN}"
echo "Max samples  : ${MAX_SAMPLES} | Concurrency=${MAX_CONCURRENT}"
echo "API base     : ${API_BASE} | timeout=${REQUEST_TIMEOUT}s | retries=${MAX_RETRIES}"
echo "Video frames : request nframes=${VIDEO_NUM_FRAMES}"
echo "Video pixels : total=${VIDEO_TOTAL_PIXELS} | min=${VIDEO_MIN_PIXELS}"
echo "Prepend prompt (first 60 chars): ${PROMPT:0:60}"
echo "Stream jsonl : ${STREAM_JSONL} -> ${JSONL_PATH} (truncate=${TRUNCATE_JSONL})"
echo "Push HF      : ${PUSH_HF} -> ${HF_OUT_REPO}/${HF_OUT_FILE}"
echo "Skip if exists: ${SKIP_IF_EXISTS}"
echo "ECVA mode    : ${ECVA_MODE} (0=rank/idx parsing, 1=normal/abnormal parsing)"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${JSONL_DIR}"
mkdir -p "${HF_UPLOAD_DIR}"

dataset_args=()
if [[ -n "${DATASET_FILE}" ]]; then
  dataset_args+=(--dataset_file "${DATASET_FILE}")
fi

video_meta_args=(
  --video_num_frames "${VIDEO_NUM_FRAMES}"
  --video_total_pixels "${VIDEO_TOTAL_PIXELS}"
  --video_min_pixels "${VIDEO_MIN_PIXELS}"
)
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  video_meta_args+=(--system_prompt "${SYSTEM_PROMPT}")
fi
if [[ "${LOG_VIDEO_FRAMES}" == "true" ]]; then
  video_meta_args+=(--log_video_frames)
fi

api_common_args=(
  --api_base "${API_BASE}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --temperature "${API_TEMPERATURE}"
  --top_p "${API_TOP_P}"
  --max_new_tokens "${API_MAX_NEW_TOKENS}"
  --presence_penalty "${API_PRESENCE_PENALTY}"
)
if [[ -n "${API_KEY}" ]]; then
  api_common_args+=(--api_key "${API_KEY}")
fi

# Optional safety/compat flags
safe_flags=()
if [[ "${SAFE_INFER}" == "true" ]]; then
  safe_flags+=(--safe_infer)
fi
if [[ "${PREFER_MP4}" == "true" ]]; then
  safe_flags+=(--prefer_mp4)
fi

split_and_run=false
IFS=',' read -ra __gpus <<< "${GPU_DEVICES}"
if [[ ${#__gpus[@]} -gt 1 ]]; then
  split_and_run=true
fi

if [[ "${PUSH_ONLY}" == "true" ]]; then
  echo "[push-only] Skipping inference run and reusing ${OUTPUT_JSON}"
  if [[ ! -s "${OUTPUT_JSON}" ]]; then
    echo "[push-only][error] Missing ${OUTPUT_JSON}. Run inference once before push-only mode." >&2
    exit 1
  fi
else
  if [[ "${split_and_run}" == "true" ]]; then
    echo "Sharded run across GPUs: ${GPU_DEVICES}"
    SHARD_COUNT=${#__gpus[@]}
    shard_jsons=()
    for idx in "${!__gpus[@]}"; do
      gpu="${__gpus[$idx]}"
      shard_out="${OUTPUT_JSON%.json}.shard${idx}.json"
      shard_jsonl="${JSONL_PATH%.jsonl}.shard${idx}.jsonl"
      if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${shard_out}" ]]; then
        echo "[skip] shard ${idx} -> found ${shard_out}"
        shard_jsons+=("${shard_out}")
        continue
      fi
      shard_jsons+=("${shard_out}")
      (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_with_vllm_api.py" \
          --model_repo "${MODEL_REPO}" \
          --dataset_repo "${DATASET_REPO}" \
          --output_json "${shard_out}" \
          --media_base "${MEDIA_BASE}" \
          --template "${TEMPLATE}" \
          --video_fps "${VIDEO_FPS}" \
          --video_maxlen "${VIDEO_MAXLEN}" \
          --max_samples "${MAX_SAMPLES}" \
          --prepend_prompt "${PROMPT}" \
          --concurrency "${MAX_CONCURRENT}" \
          --num_shards "${SHARD_COUNT}" \
          --shard_index "${idx}" \
          --gt_is_label \
          --evqa \
          --echo_ic \
          --ic_prefix "[shard ${idx}|gpu ${gpu}] ic| " \
          $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
          --jsonl_path "${shard_jsonl}" \
          $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
          "${dataset_args[@]}" \
          "${video_meta_args[@]}" \
          "${api_common_args[@]}" \
          "${safe_flags[@]}"
      ) &
    done
    wait
    echo "Merging shards → ${OUTPUT_JSON}"
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/merge_infer_results.py" \
      --inputs "${shard_jsons[@]}" \
      --output "${OUTPUT_JSON}" \
      --model_repo "${MODEL_REPO}" \
      --dataset_repo "${DATASET_REPO}"
  else
    if [[ "${SKIP_IF_EXISTS}" == "true" && -s "${OUTPUT_JSON}" ]]; then
      echo "[skip] single run -> found ${OUTPUT_JSON}"
    else
      python3 "${ROOT_DIR}/SFT/sft_trainer/infer/run_infer_with_vllm_api.py" \
        --model_repo "${MODEL_REPO}" \
        --dataset_repo "${DATASET_REPO}" \
        --output_json "${OUTPUT_JSON}" \
        --media_base "${MEDIA_BASE}" \
        --template "${TEMPLATE}" \
        --video_fps "${VIDEO_FPS}" \
        --video_maxlen "${VIDEO_MAXLEN}" \
        --max_samples "${MAX_SAMPLES}" \
        --prepend_prompt "${PROMPT}" \
        --concurrency "${MAX_CONCURRENT}" \
        --gt_is_label \
        --evqa \
        --echo_ic \
        --ic_prefix "[single|api] ic| " \
        $( [[ "${STREAM_JSONL}" == "true" ]] && echo --stream_jsonl ) \
        --jsonl_path "${JSONL_PATH}" \
        $( [[ "${TRUNCATE_JSONL}" == "true" ]] && echo --truncate_jsonl ) \
        "${dataset_args[@]}" \
        "${video_meta_args[@]}" \
        "${api_common_args[@]}" \
        "${safe_flags[@]}"
    fi
  fi
fi

if [[ "${PUSH_HF}" == "true" ]]; then
  if [[ ! -s "${OUTPUT_JSON}" ]]; then
    echo "[push] Missing ${OUTPUT_JSON}; skipping HF upload."
  else
    push_args=(
      --input_json "${OUTPUT_JSON}"
      --upload_path "${HF_UPLOAD_PATH}"
      --readme_path "${HF_README_PATH}"
      --readme_extra "${HF_README_EXTRA}"
    )
    if [[ -n "${HF_OUT_REPO}" ]]; then
      push_args+=(--hf_out_repo "${HF_OUT_REPO}")
    fi
    if [[ -n "${HF_OUT_FILE}" ]]; then
      push_args+=(--hf_out_file "${HF_OUT_FILE}")
    fi
    if [[ -n "${HF_COMMIT_MESSAGE}" ]]; then
      push_args+=(--commit_message "${HF_COMMIT_MESSAGE}")
    fi
    if [[ -n "${HF_DELETE_REMOTE}" ]]; then
      IFS=',' read -ra __del_list <<< "${HF_DELETE_REMOTE}"
      for del_path in "${__del_list[@]}"; do
        del_path="${del_path//[$'\t\n\r ']/}"
        [[ -z "${del_path}" ]] && continue
        push_args+=(--delete_remote "${del_path}")
      done
    fi
    python3 "${ROOT_DIR}/SFT/sft_trainer/infer/push_results_to_hf.py" "${push_args[@]}"
  fi
fi

echo "✅ Inference via vLLM API finished: ${OUTPUT_JSON}"

# ECVA post-processing: parse <answer> for normal/abnormal and score vs dataset 'gt'.
if [[ "${ECVA_MODE}" == "1" ]]; then
  echo "[ECVA] Post-processing predictions as normal/abnormal and scoring..."
  python3 - "$OUTPUT_JSON" "$DATASET_REPO" << 'PY'
import json, os, re, sys
from typing import Dict, Any

try:
    from huggingface_hub import list_repo_files, hf_hub_download
except Exception as e:
    print(f"[warn] huggingface_hub not available: {e}", file=sys.stderr)
    list_repo_files = None
    hf_hub_download = None

RESULT_PATH = sys.argv[1]
DATASET_REPO = sys.argv[2]

def find_jsonl(repo_id: str) -> str:
    if list_repo_files is None:
        raise RuntimeError("huggingface_hub not installed")
    files = list_repo_files(repo_id, repo_type='dataset')
    if 'data.jsonl' in files:
        return 'data.jsonl'
    for f in files:
        if f.lower().endswith('.jsonl'):
            return f
    raise FileNotFoundError('No JSONL found in dataset repo')

def load_gt_map(repo_id: str) -> Dict[str, str]:
    """Build mapping from first video path to gt label ('abnormal'|'normal')."""
    if hf_hub_download is None:
        return {}
    try:
        name = find_jsonl(repo_id)
        src = hf_hub_download(repo_id, filename=name, repo_type='dataset')
    except Exception as e:
        print(f"[warn] cannot download dataset jsonl: {e}")
        return {}
    gt_map: Dict[str, str] = {}
    import json
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vids = obj.get('videos') or []
            if not vids:
                continue
            gt = (obj.get('gt') or '').strip().lower()
            if gt in ('abnormal','normal'):
                key = vids[0]
                gt_map[key] = gt
    return gt_map

def parse_answer_to_label(text: str) -> str:
    if not isinstance(text, str):
        return ''
    # Try to extract inside <answer> ... </answer>
    m = re.search(r"<\s*answer[^>]*>(.*?)<\s*/\s*answer\s*>", text, flags=re.I|re.S)
    target = m.group(1) if m else text
    t = target.lower()
    # Normalize: keep letters only to be robust to punctuation/spaces
    t_norm = re.sub(r"[^a-z]", "", t)
    # Heuristic: if 'ab' appears anywhere, count as abnormal; else normal
    if 'ab' in t_norm:  # catches abnormal, abnorm, etc.
        return 'abnormal'
    if 'norm' in t_norm:
        return 'normal'
    # fallback: look for keywords
    for k in ('anomaly','unusual','irregular','suspicious','unsafe','incident'):
        if k in t_norm:
            return 'abnormal'
    return 'normal'

with open(RESULT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

items = data.get('items') or []
gt_map = load_gt_map(DATASET_REPO)

total = 0
has_gt = 0
correct = 0
ab_n = 0
nm_n = 0
for it in items:
    pred_text = it.get('predict') or ''
    pred_label = parse_answer_to_label(pred_text)
    it['ecva_pred'] = pred_label
    total += 1
    if pred_label == 'abnormal':
        ab_n += 1
    else:
        nm_n += 1
    vids = it.get('videos') or []
    gt_label = gt_map.get(vids[0]) if vids else None
    if gt_label in ('abnormal','normal'):
        it['ecva_gt'] = gt_label
        has_gt += 1
        if gt_label == pred_label:
            it['ecva_correct'] = True
            correct += 1
        else:
            it['ecva_correct'] = False

metrics = data.get('metrics') or {}
if has_gt:
    metrics['ecva_total'] = total
    metrics['ecva_with_gt'] = has_gt
    metrics['ecva_abnormal_preds'] = ab_n
    metrics['ecva_normal_preds'] = nm_n
    metrics['ecva_acc'] = (correct / has_gt) if has_gt else 0.0
else:
    metrics['ecva_total'] = total
    metrics['ecva_with_gt'] = 0
    metrics['ecva_abnormal_preds'] = ab_n
    metrics['ecva_normal_preds'] = nm_n

data['metrics'] = metrics

with open(RESULT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(json.dumps({
    'ecva_total': total,
    'ecva_with_gt': has_gt,
    'ecva_acc': (correct/has_gt) if has_gt else None,
    'abnormal_preds': ab_n,
    'normal_preds': nm_n,
}, ensure_ascii=False))
PY
fi
