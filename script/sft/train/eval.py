import argparse
import json
import math
import re
from typing import List, Dict, Any

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.S)
INDEX_RE = re.compile(r"\[(\d+)\]")


def extract_answer_string(s: str) -> str:
    """<answer> ... </answer> 안의 내용을 추출."""
    if not isinstance(s, str):
        return ""
    m = ANSWER_RE.search(s)
    return m.group(1) if m else ""


def parse_order(answer_str: str) -> List[int]:
    """'[2] > [4] > [1] > ...' 문자열을 [2, 4, 1, ...] 리스트로 변환."""
    return [int(x) for x in INDEX_RE.findall(answer_str)]


def dcg_at_k(pred_order: List[int], rel_grades: Dict[int, int], k: int) -> float:
    """주어진 예측 순서와 relevance grade로 DCG@k 계산."""
    dcg = 0.0
    for i, idx in enumerate(pred_order[:k], start=1):
        rel = rel_grades.get(idx, 0)
        if rel <= 0:
            continue
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def idcg_at_k(relevant_list: List[int], rel_grades: Dict[int, int], k: int) -> float:
    """이상적인 순서에서의 IDCG@k 계산."""
    k_eff = min(k, len(relevant_list))
    # relevant_list가 이미 label 순서이므로, grade는 앞에서부터 큰 값이라고 가정
    ideal_grades = [rel_grades[idx] for idx in relevant_list[:k_eff]]
    idcg = 0.0
    for i, rel in enumerate(ideal_grades, start=1):
        if rel <= 0:
            continue
        idcg += (2 ** rel - 1) / math.log2(i + 1)
    return idcg


def eval_jsonl(path: str, num_relevant: int = 1) -> Dict[str, Any]:
    # 평가할 K 값들
    Ks = [1, 3, 5]

    hit_at_k = {k: 0 for k in Ks}
    prec_at_k_sum = {k: 0.0 for k in Ks}
    recall_at_k_sum = {k: 0.0 for k in Ks}
    ndcg_at_k_sum = {k: 0.0 for k in Ks}

    mrr_sum = 0.0
    map_sum = 0.0
    n = 0  # 유효한 쿼리 수

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Line {line_no}: JSON decode error, skip.")
                continue

            predict_raw = obj.get("predict", "")
            label_raw = obj.get("label", "")

            pred_str = extract_answer_string(predict_raw)
            label_str = extract_answer_string(label_raw)

            pred_order = parse_order(pred_str)
            label_order = parse_order(label_str)

            if not pred_order or not label_order:
                print(f"[WARN] Line {line_no}: cannot parse answer, skip.")
                continue

            # label 상위 num_relevant개를 'relevant'로 사용
            num_rel_actual = min(num_relevant, len(label_order))
            if num_rel_actual == 0:
                print(f"[WARN] Line {line_no}: no relevant items, skip.")
                continue

            relevant_list = label_order[:num_rel_actual]
            relevant_set = set(relevant_list)

            # relevance grade (nDCG용) - 상위일수록 더 높은 점수
            # 예: num_rel_actual=3 -> grades: 3, 2, 1
            rel_grades = {
                idx: (num_rel_actual - i)
                for i, idx in enumerate(relevant_list)
            }

            # MRR: 가장 먼저 나온 relevant의 거꾸로 순위
            first_rank = None
            for i, idx in enumerate(pred_order, start=1):
                if idx in relevant_set:
                    first_rank = i
                    break
            if first_rank is None:
                # 하나도 못 찾으면 MRR 기여 0
                first_rank = None
            else:
                mrr_sum += 1.0 / first_rank

            # MAP: Average Precision
            hit_count = 0
            ap_sum = 0.0
            for i, idx in enumerate(pred_order, start=1):
                if idx in relevant_set:
                    hit_count += 1
                    ap_sum += hit_count / i
            if hit_count > 0:
                map_sum += ap_sum / num_rel_actual
            else:
                # relevant를 하나도 못 찾으면 AP=0
                map_sum += 0.0

            # Hit@K, Precision@K, Recall@K, nDCG@K
            for k in Ks:
                k_eff = min(k, len(pred_order))
                top_k = pred_order[:k_eff]
                num_rel_ret = len(relevant_set.intersection(top_k))

                if num_rel_ret > 0:
                    hit_at_k[k] += 1

                prec_at_k_sum[k] += num_rel_ret / float(k_eff)
                recall_at_k_sum[k] += num_rel_ret / float(num_rel_actual)

                # nDCG@K
                dcg = dcg_at_k(pred_order, rel_grades, k_eff)
                idcg = idcg_at_k(relevant_list, rel_grades, k_eff)
                if idcg > 0:
                    ndcg_at_k_sum[k] += dcg / idcg
                else:
                    ndcg_at_k_sum[k] += 0.0

            n += 1

    if n == 0:
        return {
            "num_queries": 0,
            "num_relevant": num_relevant,
            "Hit@K": {},
            "Precision@K": {},
            "Recall@K": {},
            "nDCG@K": {},
            "MRR": 0.0,
            "MAP": 0.0,
        }

    metrics = {
        "num_queries": n,
        "num_relevant": num_relevant,
        "Hit@K": {},
        "Precision@K": {},
        "Recall@K": {},
        "nDCG@K": {},
        "MRR": mrr_sum / n,
        "MAP": map_sum / n,
    }

    for k in Ks:
        metrics["Hit@K"][k] = hit_at_k[k] / n
        metrics["Precision@K"][k] = prec_at_k_sum[k] / n
        metrics["Recall@K"][k] = recall_at_k_sum[k] / n
        metrics["nDCG@K"][k] = ndcg_at_k_sum[k] / n

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RankLLM-style JSONL (predict vs label ranking)."
    )
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file")
    parser.add_argument(
        "--num_relevant",
        type=int,
        default=1,
        help="Number of top label-ranked items to treat as relevant (default: 1)",
    )
    args = parser.parse_args()

    metrics = eval_jsonl(args.jsonl_path, num_relevant=args.num_relevant)

    print("=== Evaluation Result ===")
    print(f"#queries:     {metrics['num_queries']}")
    print(f"#relevant/qp: {metrics['num_relevant']}")
    print()

    print("Hit@K / Precision@K / Recall@K / nDCG@K")
    for k in sorted(metrics["Hit@K"].keys()):
        hit = metrics["Hit@K"][k]
        prec = metrics["Precision@K"][k]
        rec = metrics["Recall@K"][k]
        ndcg = metrics["nDCG@K"][k]
        print(
            f"K={k}: Hit={hit:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, nDCG={ndcg:.4f}"
        )

    print()
    print(f"MRR: {metrics['MRR']:.4f}")
    print(f"MAP: {metrics['MAP']:.4f}")


if __name__ == "__main__":
    main()