#!/usr/bin/env bash

# Compute simple R@0.5 / R@0.7 and mIoU-from-zero metrics from a ground/rerank JSONL.
# - match_gt가 false이거나 span이 없으면 IoU=0으로 처리해 평균/정확도에 포함.
# - ground_top1 과 rerank_then_ground 두 모드 모두 지원.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# 첫 번째 인자로 입력 경로를 받을 수 있고, 없으면 기본 ground_top1 출력 경로를 사용.
INPUT_JSONL="${1:-${INPUT_JSONL:-${ROOT_DIR}/var1_exp/agent_outputs/qwen_var1_rerank5_ground.jsonl}}"
# 두 번째 인자나 OUTPUT_JSON 환경변수로 JSON 저장 경로를 지정할 수 있음(생략 시 콘솔만 출력).
OUTPUT_JSON="${2:-${OUTPUT_JSON:-}}"

echo "Input JSONL : ${INPUT_JSONL}"
[[ -n "${OUTPUT_JSON}" ]] && echo "Output JSON : ${OUTPUT_JSON}"

python3 - "$INPUT_JSONL" "$OUTPUT_JSON" <<'PY'
import json
import os
import sys
from typing import Dict, Any, Tuple, Optional


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def interval_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    p0, p1 = pred
    g0, g1 = gt
    if p1 < p0 or g1 < g0:
        return 0.0
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    return (inter / union) if union > 0 else 0.0


def extract_span(grounding: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    span = grounding.get("span")
    if isinstance(span, dict):
        s, e = span.get("start"), span.get("end")
        if s is not None and e is not None:
            try:
                s_f, e_f = float(s), float(e)
                if s_f <= e_f:
                    return s_f, e_f
            except Exception:
                return None
    return None


def compute_metrics(rows):
    total_eligible = 0  # GT 있는 샘플 수
    missing_gt = 0
    matched = 0
    span_available = 0
    sum_iou_zeroed = 0.0
    hits_05 = 0
    hits_07 = 0

    for row in rows:
        gt_vid = row.get("gt_video_id") or row.get("video_id")
        gt_time = row.get("gt_time") or row.get("time")
        if not gt_vid or not isinstance(gt_time, (list, tuple)) or len(gt_time) != 2:
            missing_gt += 1
            continue
        total_eligible += 1

        grounding = row.get("grounding") or {}
        pred_vid = grounding.get("video_id")
        match_gt = grounding.get("match_gt")
        if match_gt is None and pred_vid:
            match_gt = str(pred_vid) == str(gt_vid)

        span = extract_span(grounding)
        if span:
            span_available += 1
            try:
                iou = interval_iou(span, (float(gt_time[0]), float(gt_time[1])))
            except Exception:
                iou = 0.0
        else:
            iou = 0.0

        if not match_gt:
            iou = 0.0  # 틀리면 0점 처리
        else:
            matched += 1

        sum_iou_zeroed += iou
        hits_05 += 1 if iou >= 0.5 else 0
        hits_07 += 1 if iou >= 0.7 else 0

    metrics = {
        "total_with_gt": total_eligible,
        "missing_gt": missing_gt,
        "matched_gt_count": matched,
        "span_available": span_available,
        "miou_with_mismatch_zero": (sum_iou_zeroed / total_eligible) if total_eligible else None,
        "r1_at_0_5": hits_05,
        "r1_at_0_5_rate": (hits_05 / total_eligible) if total_eligible else None,
        "r1_at_0_7": hits_07,
        "r1_at_0_7_rate": (hits_07 / total_eligible) if total_eligible else None,
    }
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py INPUT_JSONL [OUTPUT_JSON]", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    rows = list(load_jsonl(input_path))
    metrics = compute_metrics(rows)
    print("=== Eval (zeroed on mismatch) ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
PY
