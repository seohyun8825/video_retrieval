#!/usr/bin/env python3
"""
Run inference on a HF dataset of LLaMA-Factory video samples using a trained model.

Outputs a JSON with per-sample:
  { user, videos, predict, label }

Optionally pushes the output file to a HF dataset repo for viewing.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List
from tqdm import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download

from llamafactory.hparams import get_infer_args
from llamafactory.chat.hf_engine import HuggingfaceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer on HF dataset and save results")
    p.add_argument("--model_repo", required=True, help="HF model repo id (e.g., user/sft-20251121)")
    p.add_argument("--dataset_repo", required=True, help="HF dataset repo id (e.g., user/anet_sampled_sft_valid)")
    p.add_argument("--output_json", required=True, help="Local path to write results JSON")
    p.add_argument("--media_base", default="/hub_data2/dohwan/data/retrieval", help="Base dir for resolving relative video paths")
    p.add_argument("--template", default="qwen3_vl")
    p.add_argument("--video_fps", type=float, default=2.0)
    p.add_argument("--video_maxlen", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--push_hf", action="store_true")
    p.add_argument("--hf_out_repo", help="HF dataset repo id to push results (default: derived from output filename)")
    p.add_argument("--hf_out_file", help="Remote filename when uploading (default: basename of output_json)")
    # Streaming JSONL options
    p.add_argument("--stream_jsonl", action="store_true", help="Append each sample to JSONL as soon as it's ready")
    p.add_argument("--jsonl_path", help="Path to JSONL stream file (default: output_json with .jsonl)")
    p.add_argument("--truncate_jsonl", action="store_true", help="Truncate JSONL at start if exists")
    # Concurrency
    p.add_argument("--concurrency", type=int, default=1, help="Number of concurrent inferences to overlap CPU video decoding with GPU generation")
    return p.parse_args()


def load_dataset_from_hf(repo_id: str) -> List[dict]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [f for f in files if f.lower().endswith(".json")]
    if not candidates:
        raise RuntimeError(f"No JSON files found in {repo_id}")
    # Prefer *_valid.json, else first json
    remote_file = next((f for f in candidates if f.endswith("_valid.json")), candidates[0])
    path = hf_hub_download(repo_id=repo_id, filename=remote_file, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("Expected LLaMA-Factory dataset JSON (list)")
    return data


def build_engine(model_repo: str, template: str, video_fps: float, video_maxlen: int):
    model_args, data_args, finetuning_args, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_repo,
            template=template,
            cutoff_len=204800,
            default_system=None,
            enable_thinking=True,
            image_max_pixels=8_000_000,
            video_max_pixels=16_384,
            video_fps=video_fps,
            video_maxlen=video_maxlen,
            temperature=0.2,
            top_p=0.9,
            top_k=50,
            max_new_tokens=4096,
            repetition_penalty=1.0,
        )
    )
    return HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)


def to_abs(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    data = load_dataset_from_hf(args.dataset_repo)
    samples = data[: args.max_samples] if args.max_samples > 0 else data

    # Hint engine to allow this concurrency internally
    os.environ.setdefault("MAX_CONCURRENT", str(max(1, args.concurrency)))
    engine = build_engine(args.model_repo, args.template, args.video_fps, args.video_maxlen)

    # Prepare JSONL streaming
    stream_path = args.jsonl_path or os.path.splitext(args.output_json)[0] + ".jsonl"
    if args.stream_jsonl and args.truncate_jsonl:
        os.makedirs(os.path.dirname(stream_path), exist_ok=True)
        with open(stream_path, "w", encoding="utf-8"):
            pass

    results: List[Dict[str, Any]] = [None] * len(samples)
    total = len(samples)
    pbar = tqdm(total=total, desc="Infer validset", unit="sample")
    done = 0

    import asyncio

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def run_one(idx: int, sample: Dict[str, Any]):
        nonlocal done
        # Collect messages
        msgs = sample.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), None)
        assistant = next((m for m in msgs if m.get("role") == "assistant"), None)
        user_content = (user or {}).get("content", "")
        label = (assistant or {}).get("content", "")
        videos_rel = sample.get("videos") or []
        videos_abs = [to_abs(v, args.media_base) for v in videos_rel]

        async with sem:
            # Use async chat to allow overlap
            responses = await engine.chat(
                messages=[{"role": "user", "content": user_content}],
                videos=videos_abs,
            )
            text = responses[0].response_text if responses else ""
            record = {
                "user": user_content,
                "videos": videos_rel,
                "predict": text,
                "label": label,
            }
            results[idx] = record

            if args.stream_jsonl:
                with open(stream_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            done += 1
            pbar.set_postfix_str(f"done={done} remaining={total-done}")
            pbar.update(1)

    async def run_all():
        tasks = [run_one(i, s) for i, s in enumerate(samples)]
        # Run with concurrency limit via semaphore
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    pbar.close()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_repo, "dataset": args.dataset_repo, "items": results}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {args.output_json}")

    if args.push_hf:
        api = HfApi()
        out_repo = args.hf_out_repo or os.path.splitext(os.path.basename(args.output_json))[0]
        out_file = args.hf_out_file or os.path.basename(args.output_json)

        # If no namespace is provided, prefix with current HF user to avoid 404s
        if "/" not in out_repo:
            try:
                token = HfFolder.get_token()
                if token:
                    who = api.whoami(token)
                    user = who.get("name") or who.get("username")
                else:
                    user = None
            except Exception:
                user = None

            if not user:
                print(
                    "[warn] Could not resolve HF username. "
                    "Provide --hf_out_repo as 'user/repo' or login via huggingface-cli.",
                    flush=True,
                )
                # Gracefully skip upload instead of hard error to not break the pipeline
                return
            # Basic sanitization to keep repo id valid
            def _sanitize(s: str) -> str:
                s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
                return s.lstrip("-.").rstrip("-.")

            out_repo = f"{_sanitize(user)}/{_sanitize(out_repo)}"

        # Ensure repo exists then upload
        api.create_repo(repo_id=out_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=args.output_json,
            path_in_repo=out_file,
            repo_id=out_repo,
            repo_type="dataset",
            commit_message=f"Add inference results from {args.model_repo} on {args.dataset_repo}",
        )
        print(f"Pushed to https://huggingface.co/datasets/{out_repo}")


if __name__ == "__main__":
    main()
