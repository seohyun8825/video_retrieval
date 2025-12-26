#!/usr/bin/env python3
"""
Run inference on a HF dataset (user-only + videos + optional gt),
optionally prepend a custom prompt to the user content, stream JSONL,
push results to HF, and compute simple evaluation vs. gt.

Input dataset format (list of samples):
  {
    "messages": [ {"role":"user", "content": "Query: ...\n\nCandidates: ..."} ],
    "videos": ["activitynet/videos/v_xxx.mp4", ...],
    "gt": 3  # 1-based index of the correct video among candidates (optional)
  }

Outputs a JSON: { "model":..., "dataset":..., "items":[...], "metrics":{...} }
Each item: { user, videos, gt, predict, pred_order, top1_correct, gt_pos }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from huggingface_hub import HfApi, HfFolder, hf_hub_download

from llamafactory.hparams import get_infer_args
from llamafactory.chat.hf_engine import HuggingfaceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer on HF dataset with prompt prepend and GT eval")
    p.add_argument("--model_repo", required=True, help="HF model repo id (e.g., user/sft-model)")
    p.add_argument("--dataset_repo", required=True, help="HF dataset repo id (e.g., user/validset)")
    p.add_argument("--output_json", required=True, help="Local path to write results JSON")
    p.add_argument("--media_base", default="/hub_data2/dohwan/data/retrieval", help="Base dir for resolving relative video paths")
    p.add_argument("--template", default="qwen3_vl")
    p.add_argument("--video_fps", type=float, default=2.0)
    p.add_argument("--video_maxlen", type=int, default=16)
    p.add_argument("--quantization_bit", type=int, choices=[4, 8], default=None, help="Bitsandbytes quantization (4 or 8).")
    p.add_argument("--video_max_pixels", type=int, default=16_384, help="Max pixels for video preprocessing (passed to engine)")
    p.add_argument("--image_max_pixels", type=int, default=8_000_000, help="Max pixels for images (passed to engine)")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=1)
    # Sharding
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--shard_index", type=int, default=0, help="Zero-based shard index")

    # New options
    p.add_argument("--prepend_prompt", default="hi", help="Text to prepend before the original user query")
    p.add_argument("--dataset_file", default=None, help="Specific JSON filename in the HF dataset repo")

    # Streaming JSONL options
    p.add_argument("--stream_jsonl", action="store_true", help="Append each sample to JSONL as soon as it's ready")
    p.add_argument("--jsonl_path", help="Path to JSONL stream file (default: output_json with .jsonl)")
    p.add_argument("--truncate_jsonl", action="store_true", help="Truncate JSONL at start if exists")

    # Push to HF options
    p.add_argument("--push_hf", action="store_true")
    p.add_argument("--hf_out_repo", help="HF dataset repo id to push results (default: derived from output filename)")
    p.add_argument("--hf_out_file", help="Remote filename when uploading (default: basename of output_json)")
    # Live echo
    p.add_argument("--echo_ic", action="store_true", help="Print per-sample results using icecream (or simple print fallback)")
    p.add_argument("--ic_prefix", default="ic| ", help="Prefix for icecream print lines")
    # ECVA/EVQA label-based eval
    p.add_argument("--gt_is_label", action="store_true", help="Treat sample['gt'] as 'normal'|'abnormal' string and record in 'gt_label'.")
    p.add_argument("--evqa", action="store_true", help="Enable exact-match eval (pred vs gt_label).")
    # Parsing modes for pred label
    g = p.add_mutually_exclusive_group()
    g.add_argument("--answer_tag_parsing", action="store_true", help="Parse pred label from <answer>...</answer> (default)")
    g.add_argument("--last_sentence_parsing", action="store_true", help="Parse pred label from the last sentence of the output")
    # Debug toggles
    p.add_argument("--debug_time", action="store_true", help="Record timing breakdown (prep/generate/total) per sample")
    p.add_argument("--debug_memory", action="store_true", help="Record memory usage (CUDA/CPU) per sample")
    return p.parse_args()


def load_dataset_from_hf(repo_id: str, filename: str | None) -> List[dict]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [f for f in files if f.lower().endswith(".json")]
    if not candidates:
        raise RuntimeError(f"No JSON files found in {repo_id}")
    remote_file = filename or next((f for f in candidates if f.endswith("_valid.json")), candidates[0])
    path = hf_hub_download(repo_id=repo_id, filename=remote_file, repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError("Expected dataset JSON as a list of samples")
    return data


def _parse_pred_answer(text: str, *, mode: str = "answer_tag") -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    t = text
    if mode == "last_sentence":
        # Split by sentence terminators; pick the last non-empty segment
        import re as _re
        parts = [s.strip() for s in _re.split(r"[\.!?\n]+", t) if s and s.strip()]
        last = parts[-1].lower() if parts else t.lower()
        if "abnormal" in last:
            return "abnormal"
        if "normal" in last:
            return "normal"
        return None
    # default: answer_tag mode
    import re as _re
    m = _re.search(r"<\s*answer[^>]*>(.*?)<\s*/\s*answer\s*>", t, flags=_re.I | _re.S)
    if m:
        inner = (m.group(1) or "").strip().lower()
        if "abnormal" in inner:
            return "abnormal"
        if "normal" in inner:
            return "normal"
    return None


def build_engine(
    model_repo: str,
    template: str,
    video_fps: float,
    video_maxlen: int,
    *,
    video_max_pixels: int,
    image_max_pixels: int,
    quantization_bit: Optional[int],
):
    model_args, data_args, finetuning_args, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_repo,
            template=template,
            cutoff_len=204800,
            default_system=None,
            enable_thinking=True,
            image_max_pixels=image_max_pixels,
            video_max_pixels=video_max_pixels,
            video_fps=video_fps,
            video_maxlen=video_maxlen,
            temperature=0.2,
            top_p=0.9,
            top_k=50,
            max_new_tokens=1024,
            repetition_penalty=1.0,
            quantization_bit=quantization_bit,
        )
    )
    return HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)


def to_abs(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


def parse_answer_order(text: str, n_candidates: int = 5) -> List[int]:
    """Strict parser: use only the first number inside <answer> ... </answer>.

    - Searches the first <answer> block.
    - Picks the first bracketed number [i] if present; else the first bare digit.
    - Returns a one-element list [i]; empty if not found or out of 1..N range.
    """
    if not text:
        return []

    m = re.search(r"<answer[^>]*>(.*?)</answer>", text, flags=re.S | re.I)
    if not m:
        return []
    scope = m.group(1)

    # 1) First bracketed number [i]
    m_br = re.search(r"\[(\d+)\]", scope)
    if m_br:
        i = int(m_br.group(1))
        if 1 <= i <= max(1, n_candidates):
            return [i]
        return []

    # 2) First bare digit within <answer>
    m_d = re.search(r"\b([0-9])\b", scope)
    if m_d:
        i = int(m_d.group(1))
        if 1 <= i <= max(1, n_candidates):
            return [i]
        return []

    return []


def main() -> None:
    t_start = time.perf_counter()
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    data = load_dataset_from_hf(args.dataset_repo, args.dataset_file)
    # Apply sharding first (deterministic by position)
    ns, si = max(1, args.num_shards), max(0, args.shard_index)
    if si >= ns:
        raise ValueError(f"shard_index {si} must be < num_shards {ns}")
    sharded = [s for i, s in enumerate(data) if (i % ns) == si]
    # Then apply max_samples limit per shard
    samples = sharded[: args.max_samples] if args.max_samples > 0 else sharded

    # Concurrency hint for engine
    os.environ.setdefault("MAX_CONCURRENT", str(max(1, args.concurrency)))
    engine = build_engine(
        args.model_repo,
        args.template,
        args.video_fps,
        args.video_maxlen,
        video_max_pixels=args.video_max_pixels,
        image_max_pixels=args.image_max_pixels,
        quantization_bit=args.quantization_bit,
    )
    t_engine_ready = time.perf_counter()

    # Prepare JSONL streaming
    stream_path = args.jsonl_path or os.path.splitext(args.output_json)[0] + ".jsonl"
    if args.stream_jsonl and args.truncate_jsonl:
        os.makedirs(os.path.dirname(stream_path), exist_ok=True)
        with open(stream_path, "w", encoding="utf-8"):
            pass

    results: List[Dict[str, Any]] = [None] * len(samples)
    parse_mode = "last_sentence" if args.last_sentence_parsing else "answer_tag"
    total = len(samples)
    pbar = tqdm(total=total, desc="Infer valid(all)", unit="sample")
    done = 0

    import asyncio

    sem = asyncio.Semaphore(max(1, args.concurrency))

    # optional icecream setup
    if args.echo_ic:
        try:
            from icecream import ic as _ic
            _ic.configureOutput(prefix=args.ic_prefix)
            def _echo(**kw):
                _ic(kw)
        except Exception:
            def _echo(**kw):
                print(args.ic_prefix + json.dumps(kw, ensure_ascii=False))
    else:
        def _echo(**kw):
            pass

    async def run_one(idx: int, sample: Dict[str, Any]):
        nonlocal done
        # Extract fields
        msgs = sample.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), None)
        original = (user or {}).get("content", "")
        prepend = args.prepend_prompt or ""
        user_content = (prepend + "\n\n" + original) if prepend else original
        videos_rel = sample.get("videos") or []
        videos_abs = [to_abs(v, args.media_base) for v in videos_rel]
        # GT may be ranking index or label string
        if args.gt_is_label:
            val = sample.get("gt")
            gt_label = val.strip().lower() if isinstance(val, str) else (str(val).strip().lower() if val is not None else None)
            gt = 0
        else:
            gt_label = None
            gt = int(sample.get("gt") or 0)

        async with sem:
            t_one_start = time.perf_counter()
            if args.echo_ic:
                _echo(event="start", idx=idx, vids=len(videos_rel))
            responses = await engine.chat(
                messages=[{"role": "user", "content": user_content}],
                videos=videos_abs,
                debug_time=args.debug_time,
                debug_memory=args.debug_memory,
            )
            resp0 = responses[0] if responses else None
            text = resp0.response_text if resp0 else ""
            if not args.gt_is_label:
                order = parse_answer_order(text, n_candidates=len(videos_rel) if videos_rel else 5)
                top1 = order[0] if order else 0
                gt_pos = order.index(gt) + 1 if (gt and gt in order) else 0
                top1_correct = (top1 == gt and gt != 0)
            else:
                order, top1, gt_pos, top1_correct = [], 0, 0, False

            pred_answer = None
            if args.evqa and args.gt_is_label:
                pred_answer = _parse_pred_answer(text, mode=parse_mode)

            record = {
                "user": user_content,
                "videos": videos_rel,
                **({"gt": gt} if not args.gt_is_label else {}),
                **({"gt_label": gt_label} if args.gt_is_label and gt_label else {}),
                "predict": text,
                **({"pred_answer": pred_answer} if pred_answer is not None else {}),
                "pred_order": order,
                "top1_correct": top1_correct,
                "gt_pos": gt_pos,
                "index": idx,
            }
            if resp0 is not None:
                record["prompt_tokens"] = int(getattr(resp0, "prompt_length", 0) or 0)
                record["output_tokens"] = int(getattr(resp0, "response_length", 0) or 0)
                record["finish_reason"] = getattr(resp0, "finish_reason", "")
            # Attach optional debug info from engine response (if present)
            if args.debug_time or args.debug_memory:
                try:
                    dbg = getattr(resp0, "debug", None)
                    if isinstance(dbg, dict):
                        if args.debug_time and isinstance(dbg.get("time_ms"), dict):
                            record["timing_ms"] = dbg["time_ms"]
                        if args.debug_memory and isinstance(dbg.get("memory"), dict):
                            record["memory"] = dbg["memory"]
                except Exception:
                    pass

            results[idx] = record

            if args.stream_jsonl:
                with open(stream_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Echo to terminal
            if not args.gt_is_label:
                _echo(idx=idx, gt=gt, top1=top1, correct=top1_correct)
            else:
                dbg = {"idx": idx}
                if gt_label:
                    dbg.update({"gt_label": gt_label})
                if pred_answer is not None:
                    dbg.update({"pred_answer": pred_answer})
                    if gt_label in ("normal","abnormal"):
                        dbg.update({"em_correct": pred_answer == gt_label})
                _echo(**dbg)
            if args.echo_ic:
                _echo(event="done", idx=idx, elapsed_ms=int((time.perf_counter() - t_one_start) * 1000))

            done += 1
            pbar.set_postfix_str(f"done={done} remaining={total-done}")
            pbar.update(1)

    async def run_all():
        tasks = [run_one(i, s) for i, s in enumerate(samples)]
        await asyncio.gather(*tasks)

    t_infer_start = time.perf_counter()
    asyncio.run(run_all())
    t_infer_end = time.perf_counter()
    pbar.close()

    # Aggregate metrics
    parsed = [r for r in results if r is not None]
    n = len(parsed)
    n_gt = sum(1 for r in parsed if r.get("gt"))
    n_parsed = sum(1 for r in parsed if r.get("pred_order"))
    top1_acc = (sum(1 for r in parsed if r.get("top1_correct")) / n_gt) if n_gt else 0.0
    # Recall@5 here equals fraction where GT appears in any predicted rank (given 5 candidates)
    recall5 = (sum(1 for r in parsed if r.get("gt") and r.get("gt_pos", 0) > 0) / n_gt) if n_gt else 0.0
    # Mean reciprocal rank (MRR) based on GT position
    import math
    mrr = 0.0
    cnt = 0
    for r in parsed:
        pos = r.get("gt_pos", 0)
        if pos > 0:
            mrr += 1.0 / float(pos)
            cnt += 1
    mrr = (mrr / cnt) if cnt else 0.0

    metrics = {
        "total": n,
        "with_gt": n_gt,
        "with_parsed_answer": n_parsed,
        "top1_acc": top1_acc,
        "recall_at_5": recall5,
        "mrr": mrr,
        "num_shards": ns,
        "shard_index": si,
        "wall_clock_s": float(time.perf_counter() - t_start),
        "engine_load_s": float(t_engine_ready - t_start),
        "infer_wall_clock_s": float(t_infer_end - t_infer_start),
    }

    # EVQA/ECVA exact-match metrics (optional)
    if args.evqa and args.gt_is_label:
        has_label = [r for r in parsed if r.get("gt_label") in ("normal","abnormal")]
        n_label = len(has_label)
        if n_label:
            em_correct = sum(1 for r in has_label if isinstance(r.get("pred_answer"), str) and r.get("pred_answer").strip().lower() == r.get("gt_label"))
            metrics["evqa_total"] = n
            metrics["evqa_with_gt_label"] = n_label
            metrics["evqa_acc"] = em_correct / n_label if n_label else 0.0

    # Timing/memory/token aggregates (only if present)
    time_totals = [float(r.get("timing_ms", {}).get("total")) for r in parsed if isinstance(r.get("timing_ms"), dict) and r["timing_ms"].get("total") is not None]
    if time_totals:
        metrics["sample_time_ms"] = {
            "avg": float(statistics.mean(time_totals)),
            "max": float(max(time_totals)),
            "p95": float(statistics.quantiles(time_totals, n=20, method="inclusive")[-1]) if len(time_totals) > 1 else float(time_totals[0]),
        }

    prompt_tokens = [int(r.get("prompt_tokens", 0)) for r in parsed if r.get("prompt_tokens") is not None]
    output_tokens = [int(r.get("output_tokens", 0)) for r in parsed if r.get("output_tokens") is not None]
    if prompt_tokens:
        metrics["prompt_tokens"] = {"avg": float(statistics.mean(prompt_tokens)), "max": int(max(prompt_tokens))}
    if output_tokens:
        metrics["output_tokens"] = {"avg": float(statistics.mean(output_tokens)), "max": int(max(output_tokens))}

    mem_after = [int(r.get("memory", {}).get("cuda_alloc_bytes_after", 0)) for r in parsed if isinstance(r.get("memory"), dict)]
    mem_delta = [int(r.get("memory", {}).get("cuda_alloc_bytes_delta", 0)) for r in parsed if isinstance(r.get("memory"), dict)]
    rss_after = [int(r.get("memory", {}).get("rss_bytes_after", 0)) for r in parsed if isinstance(r.get("memory"), dict)]
    if mem_after or mem_delta or rss_after:
        metrics["memory_peaks"] = {}
        if mem_after:
            metrics["memory_peaks"]["cuda_alloc_bytes_after_max"] = int(max(mem_after))
        if mem_delta:
            metrics["memory_peaks"]["cuda_alloc_bytes_delta_max"] = int(max(mem_delta))
        if rss_after:
            metrics["memory_peaks"]["rss_bytes_after_max"] = int(max(rss_after))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model_repo, "dataset": args.dataset_repo, "items": results, "metrics": metrics},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {n} results to {args.output_json}")
    print("Metrics:", json.dumps(metrics, indent=2))

    if args.push_hf:
        api = HfApi()
        out_repo = args.hf_out_repo or os.path.splitext(os.path.basename(args.output_json))[0]
        out_file = args.hf_out_file or os.path.basename(args.output_json)

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
                print("[warn] No HF user. Provide --hf_out_repo as 'user/repo' or login.")
                return
            def _sanitize(s: str) -> str:
                s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
                return s.lstrip("-.").rstrip("-.")
            out_repo = f"{_sanitize(user)}/{_sanitize(out_repo)}"

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
