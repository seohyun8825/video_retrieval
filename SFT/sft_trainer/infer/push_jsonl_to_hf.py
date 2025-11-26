#!/usr/bin/env python3
"""
Push a local JSONL file to a Hugging Face dataset repository.

Typical usage:
  python /home/seohyun/vid_understanding/video_retrieval/SFT/sft_trainer/infer/push_jsonl_to_hf.py \
    --input_jsonl /home/seohyun/vid_understanding/video_retrieval/output_1124/output_sft1124.jsonl \
    --repo output_sft1124 \
    --remote_path output_sft1124.jsonl

Notes:
- If --repo does not include a username/org prefix, the script will try to
  resolve the current HF user and prefix automatically.
- The dataset repo is created if it does not exist.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, HfFolder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push a JSONL file to HF dataset repo")
    p.add_argument("--input_jsonl", required=True, help="Local JSONL path to upload")
    p.add_argument("--repo", help="Target repo id or name (dataset). If name only, prefix with current user.")
    p.add_argument("--remote_path", help="Path in repo (default: basename of input_jsonl)")
    p.add_argument("--branch", default="main", help="Target branch (default: main)")
    p.add_argument("--private", action="store_true", help="Create repo as private (when creating)")
    p.add_argument("--message", default=None, help="Commit message (default: auto)")
    return p.parse_args()


def _whoami_user() -> str | None:
    try:
        token = HfFolder.get_token()
        if not token:
            return None
        info = HfApi().whoami(token)
        return info.get("name") or None
    except Exception:
        return None


def _sanitize(s: str) -> str:
    # Keep only A-Za-z0-9._- and strip leading/trailing '-' or '.'
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
    return s.lstrip("-.").rstrip("-.")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        print(f"❌ input_jsonl not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    repo = args.repo
    if not repo:
        repo = input_path.stem  # default to filename without extension

    # Prefix user if missing
    if "/" not in repo:
        user = _whoami_user()
        if not user:
            print("❌ Cannot resolve HF user. Please login or provide --repo with 'user/repo'.", file=sys.stderr)
            sys.exit(1)
        repo = f"{_sanitize(user)}/{_sanitize(repo)}"

    remote_path = args.remote_path or input_path.name
    api = HfApi()

    # Create dataset repo if missing
    api.create_repo(repo_id=repo, repo_type="dataset", private=args.private, exist_ok=True)

    commit_message = args.message or f"Add {input_path.name}"
    api.upload_file(
        path_or_fileobj=str(input_path),
        path_in_repo=remote_path,
        repo_id=repo,
        repo_type="dataset",
        revision=args.branch,
        commit_message=commit_message,
    )

    print(f"✅ Uploaded {input_path} to https://huggingface.co/datasets/{repo}/blob/{args.branch}/{remote_path}")


if __name__ == "__main__":
    main()

