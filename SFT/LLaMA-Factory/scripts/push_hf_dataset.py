#!/usr/bin/env python3
"""
Upload a local JSON file to a Hugging Face dataset repository.

Usage examples:

  python scripts/push_hf_dataset.py \
      --file /home/seohyun/vid_understanding/LLaMA-Factory/data/mllm_video_demo.json \
      --repo-id happy8825/test_mllm_video_demo \
      --path-in-repo mllm_video_demo.json

Notes:
- You must be logged in (`huggingface-cli login`) or have `HF_TOKEN` in env.
- The repo is created if it does not exist (repo_type=dataset).
"""

from __future__ import annotations

import argparse
import os
from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload local JSON to HF dataset repo")
    p.add_argument("--file", required=True, help="Local JSON file to upload")
    p.add_argument("--repo-id", required=True, help="Target repo id, e.g. user/repo")
    p.add_argument(
        "--path-in-repo",
        default=None,
        help="Destination path inside repo (default: basename of --file)")
    p.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist")
    p.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message (default: auto-generated)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    local_path = os.path.expanduser(args.file)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    repo_id = args.repo_id
    repo_type = "dataset"
    path_in_repo = args.path_in_repo or os.path.basename(local_path)
    commit_message = args.commit_message or f"Upload {os.path.basename(local_path)}"

    # Ensure repo exists (idempotent)
    create_repo(repo_id=repo_id, repo_type=repo_type, private=args.private, exist_ok=True)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )

    print(f"Uploaded {local_path} -> {repo_id}:{path_in_repo} (type={repo_type})")


if __name__ == "__main__":
    main()

