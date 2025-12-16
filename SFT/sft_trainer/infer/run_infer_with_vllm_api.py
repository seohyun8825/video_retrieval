#!/usr/bin/env python3
"""
Run inference on an HF dataset by routing prompts/videos to an OpenAI-compatible
HTTP endpoint (e.g., LLaMA-Factory API server backed by vLLM).

The output format matches run_infer_valid_all.py to keep downstream tooling.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import re

import requests
from huggingface_hub import HfApi, HfFolder, hf_hub_download, list_repo_files
from tqdm import tqdm

from run_infer_valid_all import load_dataset_from_hf, parse_answer_order, to_abs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer on HF dataset via OpenAI-compatible API")
    p.add_argument("--model_repo", required=True, help="Model name to send in API payloads")
    p.add_argument("--dataset_repo", required=True, help="HF dataset repo id (e.g., user/validset)")
    p.add_argument("--output_json", required=True, help="Local path to write results JSON")
    p.add_argument("--media_base", default="/hub_data2/dohwan/data/retrieval", help="Base dir for resolving relative video paths")
    p.add_argument("--template", default="qwen3_vl", help="Unused but kept for parity with other scripts")
    p.add_argument("--video_fps", type=float, default=2.0, help="Video FPS hint (used for padding metadata)")
    p.add_argument("--video_maxlen", type=int, default=16, help="Unused placeholder for logging")
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=1, help="Max in-flight HTTP requests")
    # Sharding
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--shard_index", type=int, default=0, help="Zero-based shard index")

    p.add_argument("--prepend_prompt", default="", help="Text to prepend before the original user query")
    p.add_argument("--prepend_only", action="store_true", help="Use only prepend_prompt; ignore dataset user content")
    p.add_argument("--dataset_file", default=None, help="Specific JSON filename in the HF dataset repo")
    p.add_argument("--system_prompt", default="", help="Optional system prompt sent to the API")

    # Streaming JSONL options
    p.add_argument("--stream_jsonl", action="store_true", help="Append each sample to JSONL as soon as it's ready")
    p.add_argument("--jsonl_path", help="Path to JSONL stream file (default: output_json with .jsonl)")
    p.add_argument("--truncate_jsonl", action="store_true", help="Truncate JSONL at start if exists")

    # Push to HF options
    p.add_argument("--push_hf", action="store_true")
    p.add_argument("--hf_out_repo", help="HF dataset repo id to push results (default: derived from output filename)")
    p.add_argument("--hf_out_file", help="Remote filename when uploading (default: basename of output_json)")

    # API options
    p.add_argument("--api_base", default="http://localhost:8010/v1", help="Base URL to OpenAI-compatible server")
    p.add_argument("--api_key", default=None, help="API key for Authorization header (optional)")
    p.add_argument("--request_timeout", type=float, default=240.0, help="HTTP timeout (seconds)")
    p.add_argument("--max_retries", type=int, default=3, help="Retries per sample before giving up")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=-1, help="Top-k sampling (if supported by server)")
    p.add_argument("--max_new_tokens", type=int, default=1024, help="max_tokens in API payload")
    p.add_argument("--presence_penalty", type=float, default=1.0, help="Mapped to repetition_penalty on server")
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Alias for repetition_penalty; overrides presence_penalty when provided",
    )
    p.add_argument(
        "--enable_thinking",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="If false, strip <think>...</think> block from prepend prompt",
    )
    p.add_argument("--video_num_frames", type=int, default=48, help="nframes metadata per video (default: 48)")
    p.add_argument("--video_total_pixels", type=int, default=224 * 224, help="total_pixels metadata per video")
    p.add_argument(
        "--video_min_pixels",
        type=int,
        default=0,
        help="min_pixels metadata per video (0 to skip)",
    )
    p.add_argument("--prefer_mp4", action="store_true", help="Prefer .mp4 path when both .mp4 and .mkv exist")
    p.add_argument("--log_video_frames", action="store_true", help="Record per-video frame counts in the output JSON")

    # Live echo
    p.add_argument("--echo_ic", action="store_true", help="Print per-sample results using icecream (or simple print fallback)")
    p.add_argument("--ic_prefix", default="ic| ", help="Prefix for icecream print lines")
    # ECVA: gt stored as label (normal/abnormal) instead of index
    p.add_argument("--gt_is_label", action="store_true", help="Treat sample['gt'] as 'normal'|'abnormal' string and record in 'gt_label'.")
    # EVQA/ECVA exact-match evaluation: parse <answer>...</answer> as pred_answer and compare to gt_label exactly
    p.add_argument("--evqa", action="store_true", help="Enable exact-match eval of <answer> vs gt_label (normal|abnormal).")
    # Safe infer: cap requested nframes to available frames to avoid server decode errors
    p.add_argument("--safe_infer", action="store_true", help="Cap nframes per video to actual decodable frames (prevents 48>43 errors).")
    return p.parse_args()


class OpenAIClient:
    """Minimal OpenAI-compatible client using requests."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        timeout: float,
        max_retries: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        presence_penalty: float,
        system_prompt: str = "",
        video_num_frames: int = 48,
        video_total_pixels: int = 224 * 224,
        video_min_pixels: int = 0,
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = int(top_k) if top_k is not None else -1
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.video_num_frames = video_num_frames
        self.video_total_pixels = video_total_pixels
        self.video_min_pixels = video_min_pixels
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @staticmethod
    def _path_to_file_url(path: str) -> str:
        abs_path = os.path.abspath(path)
        return "file://" + abs_path

    def _build_messages(self, prompt: str, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        def _video_item_payload(item: Dict[str, Any]) -> Dict[str, Any]:
            path = item.get("path")
            file_url = self._path_to_file_url(path)
            meta = item.get("meta") or {}
            nframes = meta.get("nframes", self.video_num_frames)
            total_pixels = meta.get("total_pixels", self.video_total_pixels)
            min_pixels = meta.get("min_pixels", self.video_min_pixels)
            video_payload: Dict[str, Any] = {"url": file_url}
            if nframes:
                video_payload["nframes"] = nframes
            if total_pixels:
                video_payload["total_pixels"] = total_pixels
            if min_pixels:
                video_payload["min_pixels"] = min_pixels
            return {"type": "video_url", "video_url": video_payload}

        content: List[Dict[str, Any]] = []
        placeholder = "<video>"
        if videos and prompt and placeholder in prompt:
            # Interleave text and videos at each <video> anchor.
            parts = re.split(r"(\<video\>)", prompt)
            vid_idx = 0
            for part in parts:
                if part == placeholder:
                    if vid_idx < len(videos):
                        content.append(_video_item_payload(videos[vid_idx]))
                        vid_idx += 1
                    else:
                        # No matching video left; keep literal token to preserve fidelity.
                        content.append({"type": "text", "text": placeholder})
                elif part:
                    content.append({"type": "text", "text": part})
            # If there are extra videos beyond placeholders, append them at the end (rare).
            while vid_idx < len(videos):
                content.append(_video_item_payload(videos[vid_idx]))
                vid_idx += 1
        else:
            # Fallback: text first then all videos (previous behavior).
            if prompt:
                content.append({"type": "text", "text": prompt})
            for item in videos or []:
                content.append(_video_item_payload(item))

        messages.append({"role": "user", "content": content})
        return messages

    def chat(self, prompt: str, videos: List[Dict[str, Any]]) -> Tuple[str, float, Optional[str], List[Dict[str, Any]]]:
        """Returns (text, latency_ms, error, request_messages)."""
        messages = self._build_messages(prompt, videos)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
        }
        # Optional sampling knobs (server may ignore if unsupported)
        if self.top_k and self.top_k > 0:
            payload["top_k"] = self.top_k
            payload["do_sample"] = True
        start = time.perf_counter()
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    try:
                        err_json = resp.json()
                        err_detail = err_json.get("error") if isinstance(err_json, dict) else err_json
                    except ValueError:
                        err_detail = resp.text.strip()
                    if isinstance(err_detail, dict):
                        err_detail = err_detail.get("message") or err_detail
                    last_err = f"APIError {resp.status_code}: {err_detail}"
                    raise ValueError(last_err)

                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise ValueError("No choices in API response")
                message = choices[0].get("message") or {}
                text = message.get("content") or ""
                latency_ms = (time.perf_counter() - start) * 1000.0
                return text, latency_ms, None, messages
            except Exception as exc:  # pylint: disable=broad-except
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries - 1:
                    time.sleep(min(5.0, 1.5 * (attempt + 1)))
                else:
                    break
        latency_ms = (time.perf_counter() - start) * 1000.0
        return "", latency_ms, last_err, messages


def resolve_video_path(path: str, base: str, prefer_mp4: bool = False) -> str:
    """Return a local video path, trying alternate extensions if needed."""
    abs_path = to_abs(path, base)
    root, ext = os.path.splitext(abs_path)

    # Prefer .mp4 when available (even if the original points to .mkv)
    if prefer_mp4:
        mp4_path = root + ".mp4"
        if os.path.exists(mp4_path) or os.path.islink(mp4_path):
            return mp4_path

    # If the original exists, keep it
    if os.path.exists(abs_path) or os.path.islink(abs_path):
        return abs_path

    # Otherwise, try a sensible alternate extension
    alt_candidates = []
    if ext.lower() == ".mp4":
        alt_candidates.append(root + ".mkv")
    elif ext.lower() == ".mkv":
        alt_candidates.append(root + ".mp4")
    for candidate in alt_candidates:
        if os.path.exists(candidate) or os.path.islink(candidate):
            return candidate
    return abs_path


_warned_cv2 = False


def _ffprobe_frame_count(path: str) -> int:
    """Try to count frames via ffprobe (robust for codecs like AV1)."""
    import subprocess, json as _json
    try:
        # First attempt: JSON output of nb_read_frames and nb_frames
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,nb_frames",
            "-of", "json",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        info = _json.loads(out.decode("utf-8", "ignore"))
        streams = info.get("streams") or []
        if streams:
            s = streams[0]
            vals = []
            for k in ("nb_read_frames", "nb_frames"):
                v = s.get(k)
                if isinstance(v, str) and v.isdigit():
                    vals.append(int(v))
                elif isinstance(v, int):
                    vals.append(int(v))
            if vals:
                return max(x for x in vals if isinstance(x, int))
    except Exception:
        pass
    try:
        # Fallback: CSV output (sometimes available when json parsing fails)
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,nb_frames",
            "-of", "csv=p=0",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "ignore").strip()
        # Output may be like: "N/A,43" or "48,48"
        parts = [p for p in out.replace("\n", ",").split(",") if p]
        nums = []
        for p in parts:
            p = p.strip()
            if p.isdigit():
                nums.append(int(p))
        if nums:
            return max(nums)
    except Exception:
        pass
    return 0


def get_frame_count(path: str) -> int:
    global _warned_cv2  # pylint: disable=global-statement
    if not os.path.exists(path):
        return 0
    # Try OpenCV first (fast when available)
    try:
        import cv2  # type: ignore
    except ImportError:
        if not _warned_cv2:
            print("[warn] OpenCV not available; falling back to ffprobe for frame counting.")
            _warned_cv2 = True
        return _ffprobe_frame_count(path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        # Fall back to ffprobe (handles AV1, etc.)
        return _ffprobe_frame_count(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frame_count and frame_count > 0:
        return frame_count
    # If OpenCV reports 0 (e.g., codec not supported), try ffprobe
    return _ffprobe_frame_count(path)


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    data = load_dataset_from_hf(args.dataset_repo, args.dataset_file)
    # Build GT label map from the HF repo JSONL when requested.
    gt_repo_map: Dict[str, str] = {}
    if args.gt_is_label:
        try:
            files = list_repo_files(args.dataset_repo, repo_type="dataset")
            name = "data.jsonl" if "data.jsonl" in files else next((f for f in files if f.lower().endswith(".jsonl")), None)
            if name:
                src = hf_hub_download(args.dataset_repo, filename=name, repo_type="dataset")
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        vids = obj.get("videos") or []
                        if not vids:
                            continue
                        g = obj.get("gt")
                        if isinstance(g, str):
                            lab = g.strip().lower()
                            if lab in ("normal", "abnormal"):
                                gt_repo_map[vids[0]] = lab
        except Exception:
            gt_repo_map = {}
    ns, si = max(1, args.num_shards), max(0, args.shard_index)
    if si >= ns:
        raise ValueError(f"shard_index {si} must be < num_shards {ns}")
    sharded = [s for i, s in enumerate(data) if (i % ns) == si]
    samples = sharded[: args.max_samples] if args.max_samples > 0 else sharded

    stream_path = args.jsonl_path or os.path.splitext(args.output_json)[0] + ".jsonl"
    if args.stream_jsonl and args.truncate_jsonl:
        os.makedirs(os.path.dirname(stream_path), exist_ok=True)
        with open(stream_path, "w", encoding="utf-8"):
            pass

    # Resolve repetition penalty mapping: server uses presence_penalty as repetition_penalty
    presence_for_server = args.repetition_penalty if args.repetition_penalty is not None else args.presence_penalty

    client = OpenAIClient(
        base_url=args.api_base,
        model=args.model_repo,
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        presence_penalty=presence_for_server,
        system_prompt=args.system_prompt,
        video_num_frames=args.video_num_frames,
        video_total_pixels=args.video_total_pixels,
        video_min_pixels=args.video_min_pixels,
    )

    results: List[Dict[str, Any]] = [None] * len(samples)
    total = len(samples)
    pbar = tqdm(total=total, desc="Infer valid(all) via API", unit="sample")
    done = 0
    sem = asyncio.Semaphore(max(1, args.concurrency))

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
        msgs = sample.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), None)
        original = (user or {}).get("content", "")
        prepend = args.prepend_prompt or ""
        if not args.enable_thinking and prepend:
            # Strip any explicit <think>...</think> contract from the prepend prompt
            prepend = re.sub(r"<think>.*?</think>\s*", "", prepend, flags=re.S | re.I)
        if args.prepend_only:
            user_content = prepend
        else:
            user_content = (prepend + "\n\n" + original) if prepend else original
        videos_rel = sample.get("videos") or []
        video_payloads = []
        video_frame_info = []
        for rel_path in videos_rel:
            resolved_path = resolve_video_path(rel_path, args.media_base, args.prefer_mp4)
            frames_total = 0
            if args.safe_infer or args.log_video_frames:
                frames_total = get_frame_count(resolved_path)
            nframes = args.video_num_frames
            if args.safe_infer:
                if frames_total and frames_total > 0:
                    # Cap to available frames; also apply off-by-one guard to avoid
                    # server-side "Expected X but only loaded X-1" failures on edge cases.
                    nframes = max(1, min(nframes, frames_total))
                    if nframes == frames_total:
                        nframes = max(1, nframes - 1)
                else:
                    # Unknown total frames (OpenCV+ffprobe failed). Use a conservative
                    # request (nframes-1) to reduce chance of server-side short reads.
                    nframes = max(1, nframes - 1)
            meta = {
                "nframes": nframes,
                "total_pixels": args.video_total_pixels,
                "min_pixels": args.video_min_pixels,
                "fps": args.video_fps,
            }
            video_payloads.append({"path": resolved_path, "meta": meta})
            if args.log_video_frames:
                video_frame_info.append(
                    {
                        "video": rel_path,
                        "frames_total": frames_total,
                        "requested_nframes": args.video_num_frames,
                        "used_nframes": nframes,
                        "requested_fps": args.video_fps,
                    }
                )
        # GT handling: either numeric index (ranking) or string label (ECVA)
        gt_label = None
        if args.gt_is_label:
            val = sample.get("gt")
            if isinstance(val, str):
                gt_label = val.strip().lower()
            elif val is not None:
                gt_label = str(val).strip().lower()
            if (not gt_label) and videos_rel:
                gt_label = gt_repo_map.get(videos_rel[0])
            gt = 0
        else:
            gt = int(sample.get("gt") or 0)

        async with sem:
            text, latency_ms, error, request_messages = await asyncio.to_thread(client.chat, user_content, video_payloads)
            order = parse_answer_order(text, n_candidates=len(videos_rel) if videos_rel else 5)
            if args.gt_is_label:
                top1 = 0
                gt_pos = 0
                top1_correct = False
            else:
                top1 = order[0] if order else 0
                gt_pos = order.index(gt) + 1 if (gt and gt in order) else 0
                top1_correct = (top1 == gt and gt != 0)

            pred_answer = None
            if args.evqa and isinstance(text, str):
                # 1) Primary: extract inside <answer> ... </answer>
                m = re.search(r"<\s*answer[^>]*>(.*?)<\s*/\s*answer\s*>", text, flags=re.I | re.S)
                if m:
                    pred_answer = (m.group(1) or "").strip().lower()
                else:
                    # 2) Fallbacks to handle models that output: </think>\n\nnormal or </think>\n\nabnormal
                    tail = text
                    m_think = re.search(r"/\s*think\s*>", text, flags=re.I)
                    if m_think:
                        tail = text[m_think.end():]
                    # Try to find the last whole-word occurrence of abnormal|normal after </think>
                    m_label_iter = list(re.finditer(r"\b(abnormal|normal)\b", tail, flags=re.I))
                    if m_label_iter:
                        pred_answer = m_label_iter[-1].group(1).strip().lower()
                    else:
                        # As a backup, search the entire text for a clear label
                        m_label_iter = list(re.finditer(r"\b(abnormal|normal)\b", text, flags=re.I))
                        if m_label_iter:
                            pred_answer = m_label_iter[-1].group(1).strip().lower()
                        else:
                            # Last resort heuristic: normalize letters and look for patterns
                            t_norm = re.sub(r"[^a-z]", "", tail.lower())
                            if "ab" in t_norm:
                                pred_answer = "abnormal"
                            elif "norm" in t_norm:
                                pred_answer = "normal"

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
                "latency_ms": latency_ms,
                "index": idx,
                "request_messages": request_messages,
            }
            if args.log_video_frames:
                record["video_frame_counts"] = video_frame_info
            if error:
                record["error"] = error
            results[idx] = record

            if args.stream_jsonl:
                with open(stream_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Debug echo (icecream). Include EVQA fields when enabled.
            dbg = {"idx": idx, "error": bool(error)}
            if not args.gt_is_label:
                dbg.update({"gt": gt, "top1": top1, "correct": top1_correct})
            else:
                if gt_label:
                    dbg.update({"gt_label": gt_label})
                if pred_answer is not None:
                    dbg.update({"pred_answer": pred_answer})
                    if gt_label in ("normal","abnormal"):
                        dbg.update({"em_correct": pred_answer == gt_label})
            _echo(**dbg)

            done += 1
            pbar.set_postfix_str(f"done={done} remaining={total-done}")
            pbar.update(1)

    async def run_all():
        tasks = [run_one(i, s) for i, s in enumerate(samples)]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    pbar.close()

    parsed = [r for r in results if r is not None]
    n = len(parsed)
    n_gt = sum(1 for r in parsed if r.get("gt"))
    n_parsed = sum(1 for r in parsed if r.get("pred_order"))
    top1_acc = (sum(1 for r in parsed if r.get("top1_correct")) / n_gt) if n_gt else 0.0
    recall5 = (sum(1 for r in parsed if r.get("gt") and r.get("gt_pos", 0) > 0) / n_gt) if n_gt else 0.0
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
    }

    # EVQA exact-match metrics (optional)
    if args.evqa:
        has_label = [r for r in parsed if isinstance(r.get("gt_label"), str) and r.get("gt_label") in ("normal", "abnormal")]
        n_label = len(has_label)
        if n_label:
            em_correct = sum(1 for r in has_label if isinstance(r.get("pred_answer"), str) and r.get("pred_answer").strip().lower() == r.get("gt_label"))
            metrics["evqa_total"] = n
            metrics["evqa_with_gt_label"] = n_label
            metrics["evqa_acc"] = em_correct / n_label if n_label else 0.0

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
            commit_message=f"Add inference results from {args.model_repo} (API) on {args.dataset_repo}",
        )
        print(f"Pushed to https://huggingface.co/datasets/{out_repo}")


if __name__ == "__main__":
    main()
