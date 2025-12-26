#!/usr/bin/env python3
"""Compute grounding metrics from ground_top1 JSONL outputs."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _ratio(num: float, den: float) -> float | None:
        return (num / den) if den else None

    total = len(rows)
    gt_known = 0
    gt_in_topk = 0
    top1_match = 0
    iou_sum = 0.0
    iou_count = 0
    wrong_video = 0
    wrong_video_marked_no = 0
    top1_gt_but_no = 0
    avg_latency_sum = 0.0
    avg_latency_count = 0

    for row in rows:
        gt_vid = row.get("gt_video_id")
        if gt_vid:
            gt_known += 1
            if row.get("gt_in_candidates"):
                gt_in_topk += 1

        grounding = row.get("grounding") or {}
        match_gt = bool(grounding.get("match_gt"))
        verdict = (grounding.get("verdict") or "").lower()
        iou = grounding.get("iou")
        latency = grounding.get("latency_ms")

        if latency is not None:
            try:
                avg_latency_sum += float(latency)
                avg_latency_count += 1
            except Exception:
                pass

        if match_gt:
            top1_match += 1
            if iou is not None:
                try:
                    iou_sum += float(iou)
                    iou_count += 1
                except Exception:
                    pass
            # GT를 맞췄지만 NO라고 답한 케이스 카운트
            if verdict == "no":
                top1_gt_but_no += 1
        else:
            # Only count as wrong when GT exists.
            if gt_vid:
                wrong_video += 1
                if verdict == "no":
                    wrong_video_marked_no += 1

    metrics = {
        "total_rows": total,
        "gt_in_topk": gt_in_topk,
        "top1_match_gt": top1_match,
        "wrong_video_total": wrong_video,
        "wrong_video_marked_no_rate": _ratio(wrong_video_marked_no, wrong_video),
        "mean_iou_when_match": (iou_sum / iou_count) if iou_count else None,
        "top1_gt_match_but_no": top1_gt_but_no,
        "top1_gt_match_but_no_rate": _ratio(top1_gt_but_no, top1_match),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute metrics for ground_top1 JSONL outputs.")
    p.add_argument("--input", required=True, help="Path to ground_top1 JSONL (e.g., qwen_var1_tg.jsonl)")
    p.add_argument(
        "--output",
        help="Output metrics JSON path. Default: alongside input with .metrics.json suffix",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.output or (os.path.splitext(args.input)[0] + ".metrics.json")

    rows = load_jsonl(args.input)
    metrics = compute_metrics(rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # human-readable summary
    def pct(x):
        return f"{x*100:.2f}%" if x is not None else "n/a"

    print("=== GroundTop1 Metrics (subset) ===")
    print(f"total_rows               : {metrics['total_rows']}")
    print(f"gt_in_topk               : {metrics['gt_in_topk']}")
    print(f"top1_match_gt            : {metrics['top1_match_gt']}")
    print(f"wrong_video_total        : {metrics['wrong_video_total']}")
    print(f"wrong_video_marked_no_rate: {pct(metrics['wrong_video_marked_no_rate'])}")
    print(f"mean_iou_when_match      : {metrics['mean_iou_when_match']}")
    print(f"top1_gt_match_but_no     : {metrics['top1_gt_match_but_no']} "
          f"({pct(metrics['top1_gt_match_but_no_rate'])})")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
