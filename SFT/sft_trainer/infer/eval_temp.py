import json
import re

def extract_pred_from_answer_tag(text):
    """
    Find ALL <answer>...</answer> tags,
    extract the FIRST number that appears inside ANY of them.
    If no <answer> tag or no number inside them → return 0.
    """

    # find all <answer>...</answer> blocks
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)

    if not answer_blocks:
        return 0  # no answer tag at all

    # search for the first number inside answer tags (left to right)
    for block in answer_blocks:
        m = re.search(r"\d+", block)
        if m:
            return int(m.group(0))

    return 0  # tags existed but no number inside any tag


def compute_recall(jsonl_path):
    total = 0
    correct = 0
    no_answer_tag_count = 0
    no_number_inside_tag_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            pred = extract_pred_from_answer_tag(item["predict"])
            gt = item["gt"]

            # stats
            if "<answer>" not in item["predict"]:
                no_answer_tag_count += 1
            elif pred == 0:
                no_number_inside_tag_count += 1

            total += 1
            if pred == gt:
                correct += 1

    recall = correct / total if total > 0 else 0

    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Recall@1: {recall:.4f}")
    print(f"No <answer> tag: {no_answer_tag_count}")
    print(f"<answer> exists but no number: {no_number_inside_tag_count}")

    return recall


# 실행
jsonl_path = "/home/seohyun/vid_understanding/video_retrieval/video_retrieval/output_1124_sft_orig_prompt/output_valid_all_orig_sft1124.jsonl"
compute_recall(jsonl_path)