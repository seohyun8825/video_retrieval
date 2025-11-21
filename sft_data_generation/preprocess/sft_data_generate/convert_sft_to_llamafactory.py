#!/usr/bin/env python3
"""
Convert SFT ranking data to LLaMA-Factory JSON format.
"""

import argparse
import json
import os
from typing import List


def load_sft_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_user_content(system_prompt: str, query: str, num_candidates: int = 5) -> str:
    user_content = f'{system_prompt}\n\nQuery: "{query}"\n\nCandidates:\n'
    for idx in range(1, num_candidates + 1):
        user_content += f"[{idx}] video: <|vision_start|><video><|vision_end|>\n"
    return user_content


def convert_samples(
    sft_data_path: str,
    video_prefix: str,
    output_path: str,
) -> List[dict]:
    data = load_sft_data(sft_data_path)
    samples = data.get("samples", [])

    system_prompt = """System Prompt: You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is a video described via frames. 

Decision rules (concise):
- Judge relevance ONLY from what is visible in the frames.
- Down-rank clear mismatches (wrong domain/scene/action). Consider temporal coherence.
- Tie-breakers (in order): action visibility/close-up > temporal coherence > low occlusion/clutter > overall coverage.

ABSOLUTE OUTPUT CONTRACT:
<think>
<content>
[1] ...
[2] ...
[3] ...
[4] ...
[5] ...
</content>

<contrast>
...
</contrast>

<summary>
1st=[i] ...
2nd=[j] ...
3rd=[k] ...
4th=[m] ...
5th=[n] ...
</summary>
</think>
<answer> [i] > [j] > [k] > [m] > [n] </answer>

Constraints:
- Inside <think>, use ONLY <content>, <contrast>, <summary>.
- Do NOT use per-candidate XML tags.
- Always output a total order in <answer>.
"""

    converted = []
    for sample in samples:
        assistant_content = sample.get("answer") or sample.get("o3_analysis") or ""
        if not assistant_content.strip():
            continue

        query = sample.get("text", "")
        user_message = create_user_content(system_prompt, query)

        videos = sample.get("videos", [])
        image_paths = []
        for vid in videos:
            video_id = os.path.basename(vid.get("video_path", ""))
            if not video_id:
                continue
            rel_path = os.path.join(video_prefix, video_id)
            image_paths.append(rel_path)

        converted.append(
            {
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_content.strip()},
                ],
                "videos": image_paths,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} samples to {output_path}")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to LLaMA-Factory format.")
    parser.add_argument("--sft_json", required=True, help="Path to sft_training_data_kept.json")
    parser.add_argument("--video_prefix", default="activitynet/videos", help="Prefix to prepend to video IDs")
    parser.add_argument("--output_path", required=True, help="Output JSON path for LLaMA-Factory")
    args = parser.parse_args()

    convert_samples(
        sft_data_path=args.sft_json,
        video_prefix=args.video_prefix,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
