#!/usr/bin/env python3
"""
Generate SFT reranking data using GPT from reranking JSON.
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from openai import OpenAI


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_reranking_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_frames(video_path: str, num_frames: int = 12, verbose: bool = False) -> List[np.ndarray]:
    if not os.path.exists(video_path):
        if verbose:
            print(f"[extract_frames] File not found: {video_path}")
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"[extract_frames] Cannot open: {video_path}")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        if verbose:
            print(f"[extract_frames] No frames: {video_path}")
        cap.release()
        return []
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            if verbose:
                print(f"[extract_frames] Failed to read frame {idx} from {video_path}")
    cap.release()
    return frames


def combine_frames(frames: List[np.ndarray]) -> np.ndarray:
    if len(frames) != 12:
        return None
    h = min(frame.shape[0] for frame in frames)
    w = min(frame.shape[1] for frame in frames)
    resized = [cv2.resize(frame, (w, h)) for frame in frames]
    rows = []
    for i in range(3):
        row = np.hstack(resized[i * 4:(i + 1) * 4])
        rows.append(row)
    combined = np.vstack(rows)
    return combined


def frames_to_base64(frames: List[np.ndarray]) -> str:
    combined = combine_frames(frames)
    if combined is None:
        return ""
    h, w = combined.shape[:2]
    if w > 1200:
        scale = 1200 / w
        combined = cv2.resize(combined, (1200, int(h * scale)))
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


def parse_answer(answer_text: str) -> List[int]:
    if not answer_text:
        return []
    content = answer_text
    if "<answer>" in content and "</answer>" in content:
        content = content.split("<answer>", 1)[1].split("</answer>", 1)[0]
    parts = [p.strip() for p in content.split(">")]
    order = []
    for p in parts:
        if "[" in p and "]" in p:
            try:
                order.append(int(p[p.index("[") + 1:p.index("]")]))
            except Exception:
                pass
    return order


def build_gt_order(videos: List[Dict]) -> List[int]:
    enriched = []
    for i, v in enumerate(videos, start=1):
        enriched.append((i, bool(v.get("is_correct", False)), float(v.get("similarity_score", -1e9))))
    enriched.sort(key=lambda x: (not x[1], -x[2], x[0]))
    return [i for (i, _, _) in enriched]


def evaluate_order(pred: List[int], gt: List[int]) -> Dict[str, Any]:
    if not pred or not gt:
        return {"keep": False, "reason": "invalid"}
    top_match = pred[0] == gt[0]
    gt_pos = {item: pos for pos, item in enumerate(gt)}
    diff = sum(1 for pos, item in enumerate(pred) if gt_pos.get(item, -1) != pos)
    keep = top_match and diff <= 2
    reason = "ok" if keep else ("top1" if not top_match else "pos_diff")
    return {"keep": keep, "reason": reason, "pos_diff": diff, "top1": top_match}


def main():
    parser = argparse.ArgumentParser(description="Generate SFT reranking data.")
    parser.add_argument("--reranking_json", required=True)
    parser.add_argument("--video_base", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--api_key_path", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--wait_sec", type=float, default=2.0)
    parser.add_argument("--save_frames", action="store_true", help="Save extracted frames to disk.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print detailed progress logs.")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers when using batch mode.")
    parser.add_argument("--use_batch_api", action="store_true", help="Enable simple concurrent processing.")
    parser.add_argument("--base_name", default="sft", help="Base name prefix for output files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("frames",):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    with open(args.api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)

    prompt = load_prompt(args.prompt_file)
    reranking_data = load_reranking_data(args.reranking_json)
    samples = reranking_data.get("training_samples", [])[: args.num_samples]

    def dbg(msg: str):
        if args.debug:
            print(msg)

    dbg(f"[INFO] Loaded prompt from {args.prompt_file}")
    dbg(f"[INFO] Using reranking data: {args.reranking_json} ({len(samples)} samples)")

    def handle_sample(sample_idx: int, sample: Dict[str, Any], client_obj: OpenAI = None) -> Tuple[int, Dict[str, Any]]:
        dbg(f"\n[Sample {sample_idx + 1}/{len(samples)}] start")
        if args.verbose:
            print(f"\n[Sample {sample_idx+1}/{len(samples)}]")
        caption = sample["caption"]
        videos = sample["videos"]

        image_payloads = []
        for vid in videos:
            video_file = os.path.join(args.video_base, os.path.basename(vid["video_path"]))
            dbg(f"  - extracting frames: {video_file}")
            frames = extract_frames(video_file, args.num_frames, verbose=args.verbose)
            if frames:
                dbg(f"    extracted {len(frames)} frames")
                image_payloads.append(frames_to_base64(frames))
                if args.save_frames:
                    frame_dir = os.path.join(args.output_dir, "frames", f"sample_{sample_idx}_{vid['video_id']}")
                    os.makedirs(frame_dir, exist_ok=True)
                    for j, frame in enumerate(frames):
                        cv2.imwrite(os.path.join(frame_dir, f"frame_{j:02d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                image_payloads.append("")
                if args.verbose:
                    print(f"  - No frames extracted for {video_file}")
                dbg("    failed to extract frames")

        user_content = [{"type": "text", "text": f"Query: {caption}"}]
        for i, img in enumerate(image_payloads):
            if not img:
                continue
            user_content.append({"type": "text", "text": f"[{i+1}] image"})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

        answer = ""
        try:
            dbg("  - calling OpenAI API …")
            client_to_use = client_obj or OpenAI(api_key=api_key)
            response = client_to_use.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": prompt}]},
                    {"role": "user", "content": user_content},
                ],
            )
            answer = response.choices[0].message.content
            dbg("  - API call succeeded")
        except Exception as e:
            if args.verbose:
                print(f"[API error] {e}")
            dbg(f"  - API call failed: {e}")

        pred = parse_answer(answer)
        gt = build_gt_order(videos)
        eval_res = evaluate_order(pred, gt)
        dbg(f"  - evaluation: keep={eval_res.get('keep')} reason={eval_res.get('reason')} pos_diff={eval_res.get('pos_diff')}")

        result = {
            "sample_index": sample_idx,
            "text": caption,
            "videos": videos,
            "answer": answer,
            "pred_order": pred,
            "gt_order": gt,
            "evaluation": eval_res,
        }
        return sample_idx, result

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results_buffer = [None] * len(samples)

    if args.use_batch_api and args.max_workers > 1:
        dbg(f"[INFO] Running in parallel with {args.max_workers} workers")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for idx, sample in enumerate(samples):
                futures.append(executor.submit(handle_sample, idx, sample, None))
            for fut in as_completed(futures):
                idx, res = fut.result()
                results_buffer[idx] = res
    else:
        dbg("[INFO] Running sequentially")
        for idx, sample in enumerate(samples):
            _, res = handle_sample(idx, sample, client)
            results_buffer[idx] = res
            if idx < len(samples) - 1:
                dbg(f"  - waiting {args.wait_sec}s before next sample")
                time.sleep(args.wait_sec)

    all_results = []
    kept = []
    discarded = []
    for res in results_buffer:
        all_results.append(res)
        if res["evaluation"].get("keep"):
            kept.append(res)
        else:
            discarded.append(res)

    metadata = {
        "model": args.model,
        "total_samples": len(all_results),
        "kept_samples": len(kept),
        "discarded_samples": len(discarded),
        "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_all = {"metadata": metadata, "samples": all_results}
    out_kept = {"metadata": metadata, "samples": kept}
    out_discarded = {"metadata": metadata, "samples": discarded}

    all_path = os.path.join(args.output_dir, f"{args.base_name}_sft_training_data_all.json")
    kept_path = os.path.join(args.output_dir, f"{args.base_name}_sft_training_data_kept.json")
    discarded_path = os.path.join(args.output_dir, f"{args.base_name}_sft_training_data_discarded.json")

    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(out_all, f, ensure_ascii=False, indent=2)
    with open(kept_path, "w", encoding="utf-8") as f:
        json.dump(out_kept, f, ensure_ascii=False, indent=2)
    with open(discarded_path, "w", encoding="utf-8") as f:
        json.dump(out_discarded, f, ensure_ascii=False, indent=2)

    dbg(f"[INFO] Saved files:\n  - {all_path}\n  - {kept_path}\n  - {discarded_path}")


if __name__ == "__main__":
    main()
