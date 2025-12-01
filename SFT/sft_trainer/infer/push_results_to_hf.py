#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from huggingface_hub import HfApi, HfFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an existing inference JSON to the HF Hub and author a README with metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_json", required=True, help="Path to the inference results JSON")
    parser.add_argument(
        "--hf_out_repo",
        default=None,
        help="Target HF dataset repo id (user/repo). When omitted, derive the repo name from the JSON basename and the active HF user.",
    )
    parser.add_argument(
        "--hf_out_file",
        default=None,
        help="Remote filename (defaults to the basename of the input JSON)",
    )
    parser.add_argument(
        "--readme_path",
        default=None,
        help="Optional path to write the generated README. Defaults to README.md next to the JSON file.",
    )
    parser.add_argument(
        "--readme_extra",
        default="",
        help="Extra Markdown snippet appended to the README (useful for provenance notes).",
    )
    parser.add_argument(
        "--commit_message",
        default=None,
        help="Override the commit message for the JSON upload (README gets its own short message).",
    )
    parser.add_argument(
        "--upload_path",
        default=None,
        help="Local file to upload to the HF dataset repo (defaults to --input_json).",
    )
    parser.add_argument(
        "--delete_remote",
        action="append",
        default=None,
        help="Remote repo-relative path to delete before uploading (can be provided multiple times).",
    )
    return parser.parse_args()


def sanitize_fragment(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
    sanitized = "".join(keep).strip("-.")
    return sanitized or "results"


def resolve_repo(hf_repo: Optional[str], default_name: str) -> Optional[str]:
    if hf_repo:
        hf_repo = hf_repo.strip()
    if hf_repo and "/" in hf_repo:
        return hf_repo

    api = HfApi()
    token = HfFolder.get_token()
    if not token:
        return None

    try:
        who = api.whoami(token)
    except Exception:
        return None

    user = who.get("name") or who.get("username")
    if not user:
        return None

    repo_name = hf_repo or sanitize_fragment(default_name)
    return f"{sanitize_fragment(user)}/{sanitize_fragment(repo_name)}"


def fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def build_readme(data: Dict[str, Any], metrics: Dict[str, Any], extra: str) -> str:
    model = data.get("model") or "Unknown model"
    dataset = data.get("dataset") or "Unknown dataset"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    total = metrics.get("total")
    with_gt = metrics.get("with_gt")
    parsed = metrics.get("with_parsed_answer")
    top1 = metrics.get("top1_acc")
    recall5 = metrics.get("recall_at_5")
    mrr = metrics.get("mrr")

    pretty_name = f"{model} · {dataset} results".replace('"', "'")
    lines = [
        "---",
        f"pretty_name: \"{pretty_name}\"",
        "language:",
        "- en",
        "tags:",
        "- video-retrieval",
        "- evaluation",
        "- vllm",
        "---",
        "",
        f"# {pretty_name}",
        "",
        f"- **Model**: `{model}`",
        f"- **Dataset**: `{dataset}`",
        f"- **Generated**: `{timestamp}`",
        "",
        "## Metrics",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total samples | {fmt_metric(total)} |",
        f"| With GT | {fmt_metric(with_gt)} |",
        f"| Parsed answers | {fmt_metric(parsed)} |",
        f"| Top-1 accuracy | {fmt_metric(top1)} |",
        f"| Recall@5 | {fmt_metric(recall5)} |",
        f"| MRR | {fmt_metric(mrr)} |",
        "",
        "The uploaded JSON contains full per-sample predictions produced via `t3_infer_with_vllm.bash`.",
    ]

    extra = (extra or "").strip()
    if extra:
        lines.extend(["", extra])

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics") or {}

    default_repo = os.path.splitext(os.path.basename(args.input_json))[0]
    repo_id = resolve_repo(args.hf_out_repo, default_repo)
    if not repo_id:
        raise SystemExit(
            "Could not resolve HF repo id. Provide --hf_out_repo as 'user/repo' or login with `huggingface-cli login`."
        )

    upload_path = args.upload_path or args.input_json
    out_file = args.hf_out_file or os.path.basename(upload_path)
    readme_path = args.readme_path or os.path.join(os.path.dirname(args.input_json), "README.md")
    if readme_path:
        os.makedirs(os.path.dirname(readme_path) or ".", exist_ok=True)
    readme_body = build_readme(data, metrics, args.readme_extra)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_body)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    if args.delete_remote:
        for remote_path in args.delete_remote:
            cleaned = (remote_path or "").strip()
            if not cleaned:
                continue
            try:
                api.delete_file(path_in_repo=cleaned, repo_id=repo_id, repo_type="dataset")
                print(f"[hf] deleted remote file: {cleaned}")
            except Exception as exc:
                print(f"[hf][warn] failed to delete {cleaned}: {exc}")

    commit_message = args.commit_message or f"Upload {out_file} results"
    api.upload_file(
        path_or_fileobj=upload_path,
        path_in_repo=out_file,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update README with metrics",
    )
    print(f"Pushed {out_file} and README to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
