#!/usr/bin/env bash

set -euo pipefail

# Push a trained model directory to Hugging Face Hub.
# - Login should already be done (huggingface-cli login)
# - Repo is created if missing (type: model)
#
# Usage:
#   t2_push_model.bash [MODEL_DIR]
#
# Config (env-overridable):
#   HF_USER        : Hugging Face username/org (auto from `whoami` if empty)
#   HF_REPO        : Repository name (default: sft-YYYYMMDD)
#   HF_PRIVATE     : true/false to create private repo (default: false)
#   HF_BRANCH      : branch/revision to upload to (default: main)
#   INCLUDE        : include glob (optional, e.g., "*.safetensors")
#   EXCLUDE        : exclude glob (optional, e.g., "*.pt")
#   COMMIT_MESSAGE : commit message (default: "Add ${HF_REPO}")
#   DRY_RUN        : true to print commands only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_DIR="${1:-${MODEL_DIR:-/hub_data4/seohyun/saves/ecva_instruct/full/sft/checkpoint-350}}"

# Defaults
HF_USER="${HF_USER:-}"
HF_REPO="${HF_REPO:-}"
HF_PRIVATE="${HF_PRIVATE:-false}"
HF_BRANCH="${HF_BRANCH:-main}"
INCLUDE="${INCLUDE:-}"
EXCLUDE="${EXCLUDE:-}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${HF_REPO}" ]]; then
  HF_REPO="best77ecva_tuned-$(date +%Y%m%d)"
fi

if [[ -z "${HF_USER}" ]]; then
  # Prefer robust Python API query
  set +e
  HF_USER=$(python3 - << 'PY'
try:
    from huggingface_hub import HfApi, HfFolder
    token = HfFolder.get_token()
    if not token:
        raise SystemExit(1)
    info = HfApi().whoami(token)
    print(info.get('name',''))
except Exception:
    raise SystemExit(2)
PY
  )
  status=$?
  set -e
  if [[ $status -ne 0 || -z "${HF_USER}" ]]; then
    # Fallback to CLI short form
    if command -v huggingface-cli >/dev/null 2>&1; then
      set +e
      HF_USER=$(huggingface-cli whoami -s 2>/dev/null | head -n1)
      set -e
    fi
  fi
fi

# Sanitize to valid repo id charset
sanitize_id() {
  local s="$1"
  s="$(printf %s "$s" | tr -cd 'A-Za-z0-9._-')"
  s="$(printf %s "$s" | sed -E 's/^[-.]+//; s/[-.]+$//')"
  printf %s "$s"
}

if [[ -z "${HF_USER}" ]]; then
  echo "❌ Could not determine HF_USER. Set HF_USER=your-name or ensure huggingface login is active." >&2
  exit 1
fi
HF_USER_ORIG="${HF_USER}"
HF_USER="$(sanitize_id "${HF_USER}")"
if [[ -z "${HF_USER}" ]]; then
  echo "❌ Invalid HF_USER ('${HF_USER_ORIG}'). Provide a valid alphanumeric username (with . _ - allowed)." >&2
  exit 1
fi

HF_REPO_ORIG="${HF_REPO}"
HF_REPO="$(sanitize_id "${HF_REPO}")"
if [[ -z "${HF_REPO}" ]]; then
  echo "❌ Invalid HF_REPO ('${HF_REPO_ORIG}'). Use [A-Za-z0-9._-] and avoid starting/ending with '-' or '.'." >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "❌ MODEL_DIR not found: ${MODEL_DIR}" >&2
  exit 1
fi

REPO_ID="${HF_USER}/${HF_REPO}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Add ${HF_REPO}}"

echo "Model dir  : ${MODEL_DIR}"
echo "Repo id    : ${REPO_ID} (type=model)"
echo "Branch     : ${HF_BRANCH}"
echo "Private    : ${HF_PRIVATE}"
if [[ -n "${INCLUDE}" ]]; then echo "Include    : ${INCLUDE}"; fi
if [[ -n "${EXCLUDE}" ]]; then echo "Exclude    : ${EXCLUDE}"; fi
echo "Message    : ${COMMIT_MESSAGE}"

create_cmd=(
  huggingface-cli repo create "${REPO_ID}"
  --type model
  --exist-ok
)

upload_cmd=(
  huggingface-cli upload "${REPO_ID}" "${MODEL_DIR}" "."
  --repo-type model
  --commit-message "${COMMIT_MESSAGE}"
  --revision "${HF_BRANCH}"
)

if [[ "${HF_PRIVATE,,}" == "true" ]]; then
  create_cmd+=(--private)
fi
if [[ -n "${INCLUDE}" ]]; then
  upload_cmd+=(--include "${INCLUDE}")
fi
if [[ -n "${EXCLUDE}" ]]; then
  upload_cmd+=(--exclude "${EXCLUDE}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN: ${create_cmd[*]}"
  echo "DRY_RUN: ${upload_cmd[*]}"
else
  "${create_cmd[@]}" >/dev/null
  "${upload_cmd[@]}"
fi

echo "✅ Pushed to https://huggingface.co/${REPO_ID}/tree/${HF_BRANCH}"
