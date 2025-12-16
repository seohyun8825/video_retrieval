#!/usr/bin/env python3
import argparse
import json
import re
from typing import Optional, Tuple


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.S)
INDEX_IN_BRACKETS_RE = re.compile(r"\[(\d+)\]")
ANY_INT_RE = re.compile(r"(\d+)")


def extract_answer_content(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = ANSWER_RE.search(s)
    return m.group(1).strip() if m else s.strip()


def parse_top1_from_predict(predict_value) -> Optional[int]:
    """Try to get the top-1 predicted index from various field formats.

    Supported formats:
    - pred_order: [3, 1, 5, ...] -> returns 3
    - predict: "<answer>3</answer>" -> returns 3
    - predict: "<answer>[3] > [1] > [5]</answer>" -> returns 3
    - predict: "3" or "[3]" -> returns 3
    """
    # If predict_value is already a list (pred_order)
    if isinstance(predict_value, list) and predict_value:
        try:
            return int(predict_value[0])
        except Exception:
            return None

    # If it's a string, try to parse inside <answer> first
    if isinstance(predict_value, str):
        content = extract_answer_content(predict_value)
        # Prefer [n] pattern first
        m = INDEX_IN_BRACKETS_RE.search(content)
        if m:
            return int(m.group(1))
        # Fall back to any integer in the content
        m2 = ANY_INT_RE.search(content)
        if m2:
            return int(m2.group(1))

    return None


def get_top1_and_gt(obj: dict) -> Tuple[Optional[int], Optional[int]]:
    # predicted top1
    pred_top1 = None
    if "pred_order" in obj:
        pred_top1 = parse_top1_from_predict(obj.get("pred_order"))
    if pred_top1 is None:
        pred_top1 = parse_top1_from_predict(obj.get("predict"))

    # ground truth (preferred key: gt)
    gt = obj.get("gt")
    if isinstance(gt, str):
        # tolerate gt as string
        if gt.isdigit():
            gt = int(gt)
        else:
            # try parse from "<answer>..." style
            content = extract_answer_content(gt)
            m = INDEX_IN_BRACKETS_RE.search(content) or ANY_INT_RE.search(content)
            gt = int(m.group(1)) if m else None
    elif not isinstance(gt, int):
        # try label string like "<answer>[3] > ...</answer>"
        label = obj.get("label")
        if isinstance(label, str):
            content = extract_answer_content(label)
            m = INDEX_IN_BRACKETS_RE.search(content) or ANY_INT_RE.search(content)
            gt = int(m.group(1)) if m else None
        else:
            gt = None

    return pred_top1, gt


def main():
    parser = argparse.ArgumentParser(description="Compute top-1 correct count from JSONL.")
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N lines")
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-line correctness diagnostics"
    )
    args = parser.parse_args()

    total = 0
    correct = 0
    skipped = 0

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if args.limit is not None and total >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if args.verbose:
                    print(f"[WARN] Line {line_no}: JSON decode error, skip.")
                continue

            # Prefer explicit boolean if present
            if isinstance(obj.get("top1_correct"), bool):
                is_correct = obj["top1_correct"]
                # still count total only for evaluable items
                total += 1
                if is_correct:
                    correct += 1
                if args.verbose:
                    gt = obj.get("gt")
                    pred_top1 = None
                    if "pred_order" in obj:
                        pred_top1 = parse_top1_from_predict(obj.get("pred_order"))
                    if pred_top1 is None:
                        pred_top1 = parse_top1_from_predict(obj.get("predict"))
                    print(f"[Line {line_no}] pred={pred_top1} gt={gt} -> {is_correct}")
                continue

            # Otherwise, try to compute from fields
            pred_top1, gt = get_top1_and_gt(obj)
            if pred_top1 is None or gt is None:
                skipped += 1
                if args.verbose:
                    print(
                        f"[WARN] Line {line_no}: missing pred_top1 ({pred_top1}) or gt ({gt}), skip."
                    )
                continue

            total += 1
            is_correct = (pred_top1 == gt)
            if is_correct:
                correct += 1
            if args.verbose:
                print(f"[Line {line_no}] pred={pred_top1} gt={gt} -> {is_correct}")

    acc = (correct / total) if total > 0 else 0.0
    print("=== Top-1 Accuracy ===")
    print(f"file:    {args.jsonl_path}")
    if args.limit is not None:
        print(f"limit:   {args.limit}")
    print(f"total:   {total}")
    print(f"correct: {correct}")
    print(f"acc:     {acc:.4f} ({acc*100:.2f}%)")
    if skipped:
        print(f"skipped: {skipped} (invalid/missing fields)")


if __name__ == "__main__":
    main()

