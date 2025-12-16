#!/usr/bin/env bash

set -euo pipefail

# Upload a folder of inference outputs (e.g., JSONL files) to a Hugging Face
# dataset repo. Overwrites existing files with the same names.
#
# Usage:
#   push_infer.bash <local_dir> <hf_repo_id|repo_name> [pattern] [commit_message]
#
# Examples:
#   push_infer.bash \
#     /home/seohyun/vid_understanding/video_retrieval/video_retrieval/output_sft \
#     seohyun/output-sft \
#     "*.jsonl" \
#     "Upload SFT outputs"
#
#   # If you pass just a repo name (no slash), it will prefix with your HF user
#   # based on your `huggingface-cli login` session.

LOCAL_DIR="/home/seohyun/vid_understanding/video_retrieval/video_retrieval/output_1124_sft_orig_prompt"
HF_REPO="qwenutil1126infer"
ALLOW_PATTERN="${3:-*.jsonl}"
COMMIT_MESSAGE="${4:-Upload inference outputs}"

if [[ -z "${LOCAL_DIR}" || -z "${HF_REPO}" ]]; then
  echo "Usage: $0 <local_dir> <hf_repo_id|repo_name> [pattern] [commit_message]" >&2
  exit 2
fi

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "[error] Local dir not found: ${LOCAL_DIR}" >&2
  exit 3
fi

echo "Local dir   : ${LOCAL_DIR}"
echo "Repo        : ${HF_REPO} (dataset)"
echo "Allow patt. : ${ALLOW_PATTERN}"
echo "Commit msg  : ${COMMIT_MESSAGE}"

# Run the upload via huggingface_hub
python3 - "$LOCAL_DIR" "$HF_REPO" "$ALLOW_PATTERN" "$COMMIT_MESSAGE" << 'PY'
import os, sys, glob, tempfile
from datetime import datetime
from huggingface_hub import HfApi, HfFolder, upload_folder

local_dir, repo_spec, allow_pattern, commit_message = sys.argv[1:5]
api = HfApi()

def resolve_repo(repo: str) -> str:
    if "/" in repo:
        return repo
    token = HfFolder.get_token()
    if not token:
        raise SystemExit("[error] Missing HF token. Please run `huggingface-cli login` or pass user/repo explicitly.")
    who = api.whoami(token)
    user = who.get("name") or who.get("username")
    if not user:
        raise SystemExit("[error] Could not resolve HF username; please pass full 'user/repo'.")
    def _san(s: str) -> str:
        keep = [ch for ch in s if ch.isalnum() or ch in "._-"]
        s = "".join(keep).strip("-.")
        return s or "results"
    return f"{_san(user)}/{_san(repo)}"

repo_id = resolve_repo(repo_spec)
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# Upload files in folder matching allow_pattern (e.g., *.jsonl)
print(f"[hf] Uploading from {local_dir} to {repo_id} (pattern={allow_pattern})")
upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[allow_pattern] if allow_pattern else None,
    commit_message=commit_message,
)

# Add or update a lightweight README with file listing
paths = sorted([p for p in glob.glob(os.path.join(local_dir, allow_pattern))])
ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
lines = [
    "---",
    f"pretty_name: \"{os.path.basename(repo_id)}\"",
    "language:",
    "- en",
    "tags:",
    "- video-retrieval",
    "- evaluation",
    "---",
    "",
    f"# {repo_id}",
    "",
    f"Uploaded on {ts}",
    "",
    "## Files",
]
for p in paths:
    lines.append(f"- `{os.path.basename(p)}`")

body = "\n".join(lines) + "\n"
with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as tf:
    tf.write(body)
    tmp_path = tf.name

api.upload_file(
    path_or_fileobj=tmp_path,
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Update README (auto-generated)",
)
print(f"[hf] Pushed to https://huggingface.co/datasets/{repo_id}")
PY

echo "✅ Done."

