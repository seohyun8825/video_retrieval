#!/usr/bin/env python3
import re
from datasets import load_dataset, DatasetDict

SRC_REPO = "happy8825/anet_ret_train_global_rzen_sft"
DST_REPO = "happy8825/anet_ret_train_global_rzen_sft_nothinking"


def strip_bullets(text: str) -> str:
    pattern = (
        r"\n- Inside <think>, use ONLY <content>, <contrast>, <summary>\.\n"
        r"- Do NOT use per-candidate XML tags\."
    )
    return re.sub(pattern, "", text)


def strip_think_block(text: str) -> str:
    no_think = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return no_think


def process_example(example):
    msgs = example["messages"]
    new_msgs = []

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")

        if role == "user":
            # bullet 두 줄 삭제
            content = strip_bullets(content)
        elif role == "assistant":
            # <think> 블록 통째 삭제
            content = strip_think_block(content)

        m["content"] = content
        new_msgs.append(m)

    example["messages"] = new_msgs
    return example


def main():
    print(f"🔹 Loading dataset: {SRC_REPO}")
    ds_dict = load_dataset(SRC_REPO)  # 모든 split (train/validation 등) 같이 로드

    new_dict = DatasetDict()
    for split_name, ds in ds_dict.items():
        print(f"✨ Processing split: {split_name} (num_rows={len(ds)})")
        new_ds = ds.map(process_example, desc=f"strip think / bullets [{split_name}]")
        new_dict[split_name] = new_ds

    print(f"🚀 Pushing to new repo: {DST_REPO}")
    new_dict.push_to_hub(DST_REPO)
    print("✅ Done!")


if __name__ == "__main__":
    main()