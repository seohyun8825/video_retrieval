#!/usr/bin/env python3
"""
Evaluate running/finished inference from shard JSONL files in a directory.

Usage:
  python eval_jsonl_folder.py --dir /path/to/folder \
    --pattern "*.jsonl" \
    [--per_file] [--save /path/to/summary.json]

Assumes each JSONL line is a record like:
  {"user":..., "videos":..., "gt": int, "predict": str,
   "pred_order": [int,...], "top1_correct": bool, "gt_pos": int, ...}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval JSONL shards in a folder")
    p.add_argument("--dir", required=True, help="Directory containing *.jsonl files")
    p.add_argument("--pattern", default="*.jsonl", help="Glob pattern for files (default: *.jsonl)")
    p.add_argument("--per_file", action="store_true", help="Print per-file metrics as well")
    p.add_argument("--save", default=None, help="Optional path to save combined summary JSON")
    return p.parse_args()


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                # skip malformed lines
                pass
    return rows


def _metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    with_gt = [r for r in items if int(r.get("gt") or 0) > 0]
    n_gt = len(with_gt)
    # Use provided fields when available; else compute from pred_order/gt
    top1_correct_count = 0
    recall_hits = 0
    mrr_sum = 0.0
    mrr_cnt = 0
    for r in with_gt:
        gt = int(r.get("gt") or 0)
        pred_order = r.get("pred_order") or []
        if isinstance(pred_order, list):
            if pred_order:
                top1_correct_count += int(pred_order[0] == gt)
                if gt in pred_order:
                    recall_hits += 1
                    pos = pred_order.index(gt) + 1
                    mrr_sum += 1.0 / float(pos)
                    mrr_cnt += 1
        else:
            # If pred_order missing but top1_correct present, count it
            if r.get("top1_correct"):
                top1_correct_count += 1

    top1_acc = (top1_correct_count / n_gt) if n_gt else 0.0
    recall_at_5 = (recall_hits / n_gt) if n_gt else 0.0
    mrr = (mrr_sum / mrr_cnt) if mrr_cnt else 0.0
    return {
        "total_lines": total,
        "with_gt": n_gt,
        "top1_acc": top1_acc,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
    }


def main() -> None:
    args = parse_args()
    folder = os.path.abspath(args.dir)
    files = sorted(glob.glob(os.path.join(folder, args.pattern)))
    if not files:
        print(f"No files matched: {folder}/{args.pattern}")
        return

    if args.per_file:
        print("Per-file metrics:")

    all_rows: List[Dict[str, Any]] = []
    for fp in files:
        rows = _load_jsonl(fp)
        all_rows.extend(rows)
        if args.per_file:
            m = _metrics(rows)
            print(f"  - {os.path.basename(fp)} -> total={m['total_lines']} with_gt={m['with_gt']} top1={m['top1_acc']:.3f} R@5={m['recall_at_5']:.3f} MRR={m['mrr']:.3f}")

    m_all = _metrics(all_rows)
    print("\nCombined metrics:")
    print(
        f"  total={m_all['total_lines']} with_gt={m_all['with_gt']} "
        f"top1={m_all['top1_acc']:.4f} R@5={m_all['recall_at_5']:.4f} MRR={m_all['mrr']:.4f}"
    )

    if args.save:
        out = {
            "dir": folder,
            "pattern": args.pattern,
            "files": [os.path.basename(f) for f in files],
            "metrics": m_all,
            "count_files": len(files),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved summary: {args.save}")


if __name__ == "__main__":
    main()

