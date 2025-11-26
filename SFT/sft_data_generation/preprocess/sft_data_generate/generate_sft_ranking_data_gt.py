#!/usr/bin/env python3
"""
Generate SFT ranking data (ground-truth only, no GPT) from a reranking JSON.

Input reranking JSON format (produced by video_colbert_generate_reranking.py):
{
  "metadata": { ... },
  "training_samples": [
    {
      "text_id": str,
      "caption": str,
      "videos": [
        {"video_id": str, "video_path": str, "similarity_score": float, "is_correct": bool}, ...
      ]
    }, ...
  ]
}

Output SFT format (array of records) e.g.:
[
  {
    "messages": [
      {"role": "user", "content": "Query: \"...\"\n\nCandidates:\n[1] video:<video>\n..."}
    ],
    "videos": ["activitynet/videos/v_....mp4", ...]
  }, ...
]

Notes:
- Only a single user message is emitted; no assistant message is included.
- The videos array contains relative paths derived from the basename of input paths.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def load_reranking(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_rel_video_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.join("activitynet", "videos", base).replace("\\", "/")


def build_user_content(query: str, num_candidates: int) -> str:
    lines = []
    lines.append(f"Query: \"{query}\"")
    lines.append("")
    lines.append("Candidates:")
    for i in range(1, num_candidates + 1):
        lines.append(f"[{i}] video:<video>")
    return "\n".join(lines)


def convert_samples(rerank: Dict[str, Any], limit: int | None = None, query_override: str | None = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    samples = rerank.get("training_samples", [])
    if limit is not None and limit > 0:
        samples = samples[:limit]

    for sample in samples:
        caption: str = sample.get("caption", "")
        query_text = query_override if query_override is not None else caption

        vids = sample.get("videos", [])
        rel_videos = [to_rel_video_path(v.get("video_path", "")) for v in vids]
        # Compute 1-based GT index: position where sample text matches candidate
        gt_idx = 0
        text_id = sample.get("text_id")
        for i, v in enumerate(vids, start=1):
            vid = v.get("video_id")
            if v.get("is_correct") or (text_id and vid == text_id):
                gt_idx = i
                break
        content = build_user_content(query_text, len(rel_videos))

        out.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "videos": rel_videos,
                "gt": gt_idx,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert reranking JSON into SFT user-only format (no GPT)")
    ap.add_argument("--reranking_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_name", default="sft")
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--query_override", default=None, help="if set, use this text for all Query")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rerank = load_reranking(args.reranking_json)
    data = convert_samples(rerank, limit=args.limit or None, query_override=args.query_override)

    out_file = os.path.join(args.output_dir, f"{args.base_name}_sft_llamafactory_gt.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved SFT GT JSON: {out_file}")


if __name__ == "__main__":
    main()
