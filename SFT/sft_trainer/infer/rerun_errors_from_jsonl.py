#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import HfApi, HfFolder

# Reuse utilities from the local inference modules
from run_infer_valid_all import load_dataset_from_hf, parse_answer_order


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rerun only error samples found in a JSONL stream and merge results into a new JSON.\n"
            "- Parses JSONL for entries with an 'error' field and collects their 'index' values.\n"
            "- Re-infers those indices via an OpenAI-compatible API with optional fallback.\n"
            "- Overwrites the corresponding entries in a base JSON and recomputes metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Inputs
    p.add_argument("--jsonl_in", required=True, help="Path to the original streaming JSONL (contains 'index' and optional 'error')")
    p.add_argument("--base_json_in", required=True, help="Path to the base combined JSON to patch/overwrite (full items list)")
    p.add_argument("--output_json", required=True, help="Path to write the merged 'without error' JSON")

    # Dataset/model/global settings
    p.add_argument("--model_repo", required=True, help="Model name to send in API payloads")
    p.add_argument("--dataset_repo", required=True, help="HF dataset repo id (e.g., user/validset)")
    p.add_argument("--dataset_file", default=None, help="Specific JSON filename in the HF dataset repo")
    p.add_argument("--media_base", default="/hub_data3/seohyun", help="Base dir for resolving relative video paths")
    p.add_argument("--system_prompt", default="", help="Optional system prompt sent to the API")
    p.add_argument("--prepend_prompt", default="", help="Text to prepend before the original user query")
    p.add_argument(
        "--enable_thinking",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="If false, strip <think>...</think> from any prepend prompt",
    )

    # Fallback/video settings
    p.add_argument("--video_num_frames", type=int, default=48, help="Requested nframes per video for the first attempt")
    p.add_argument("--video_total_pixels", type=int, default=224 * 224, help="Total pixels hint per video for the first attempt")
    p.add_argument("--video_min_pixels", type=int, default=0, help="Min pixels hint per video (0 to omit)")
    p.add_argument("--prefer_mp4", action="store_true", help="Prefer .mp4 path when both .mp4 and .mkv exist")
    p.add_argument(
        "--fallback_frames",
        default="24,16,8",
        help="Comma-separated fallback nframes tried after a length error (applied per-sample)",
    )
    p.add_argument(
        "--fallback_pixels",
        default="16384,12544,9216",
        help="Comma-separated fallback total_pixels tried after a length error (applied per-sample)",
    )

    # API settings
    p.add_argument("--api_base", default="http://localhost:8019/v1", help="Base URL to OpenAI-compatible server")
    p.add_argument("--api_key", default=None, help="API key (optional)")
    p.add_argument("--request_timeout", type=float, default=240.0, help="HTTP timeout (seconds)")
    p.add_argument("--max_retries", type=int, default=3, help="Retries per HTTP attempt")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--presence_penalty", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=None)

    # Concurrency
    p.add_argument("--concurrency", type=int, default=4)

    # JSONL streaming for the rerun (optional)
    p.add_argument("--stream_jsonl_out", default=None, help="If set, append rerun results here as JSONL")
    p.add_argument("--truncate_stream_out", action="store_true", help="Truncate --stream_jsonl_out at start if exists")
    p.add_argument("--stream_existing", action="store_true", help="When streaming, first write all existing non-error items from base JSON")

    # Echo progress
    p.add_argument("--echo_ic", action="store_true", help="Print per-sample progress using icecream (or print fallback)")
    p.add_argument("--ic_prefix", default="[rerun] ", help="Prefix for progress lines")

    # Push to HF (optional)
    p.add_argument("--push_hf", action="store_true")
    p.add_argument("--hf_out_repo", default=None, help="HF dataset repo id (user/repo). Defaults to user/<basename>")
    p.add_argument("--hf_out_file", default=None, help="Remote filename (defaults to basename of output JSON)")
    p.add_argument("--commit_message", default=None, help="Upload commit message")
    return p.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def collect_error_indices(jsonl_path: str) -> List[int]:
    idxs: Set[int] = set()
    for row in iter_jsonl(jsonl_path):
        if "error" in row:
            try:
                i = int(row.get("index"))
                idxs.add(i)
            except Exception:
                pass
    return sorted(idxs)


class OpenAIClient:
    """Minimal OpenAI-compatible client using requests (copied to allow custom fallback)."""

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
    ) -> None:
        import requests  # local import to avoid global hard dependency

        self._requests = requests
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
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @staticmethod
    def _video_payload(path: str, nframes: int, total_pixels: int, min_pixels: int) -> Dict[str, Any]:
        abs_path = os.path.abspath(path)
        payload: Dict[str, Any] = {
            "url": "file://" + abs_path,
        }
        if nframes:
            payload["nframes"] = nframes
        if total_pixels:
            payload["total_pixels"] = total_pixels
        if min_pixels:
            payload["min_pixels"] = min_pixels
        return {"type": "video_url", "video_url": payload}

    def build_messages(
        self,
        prompt: str,
        videos_abs: List[str],
        nframes: int,
        total_pixels: int,
        min_pixels: int,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        placeholder = "<video>"
        if videos_abs and prompt and placeholder in prompt:
            parts = re.split(r"(\<video\>)", prompt)
            vid_idx = 0
            for part in parts:
                if part == placeholder:
                    if vid_idx < len(videos_abs):
                        content.append(self._video_payload(videos_abs[vid_idx], nframes, total_pixels, min_pixels))
                        vid_idx += 1
                    else:
                        content.append({"type": "text", "text": placeholder})
                elif part:
                    content.append({"type": "text", "text": part})
            while vid_idx < len(videos_abs):
                content.append(self._video_payload(videos_abs[vid_idx], nframes, total_pixels, min_pixels))
                vid_idx += 1
        else:
            if prompt:
                content.append({"type": "text", "text": prompt})
            for p in videos_abs:
                content.append(self._video_payload(p, nframes, total_pixels, min_pixels))

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def chat_once(self, messages: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
        }
        if self.top_k and self.top_k > 0:
            payload["top_k"] = self.top_k
            payload["do_sample"] = True

        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
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
                return text, None
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries - 1:
                    time.sleep(min(5.0, 1.5 * (attempt + 1)))
                else:
                    break
        return "", last_err


def recompute_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [r for r in items if r is not None]
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
    return {
        "total": n,
        "with_gt": n_gt,
        "with_parsed_answer": n_parsed,
        "top1_acc": top1_acc,
        "recall_at_5": recall5,
        "mrr": mrr,
    }


def fix_prompt(prepend: str, original: str, enable_thinking: bool) -> str:
    if not enable_thinking and prepend:
        prepend = re.sub(r"<think>.*?</think>\s*", "", prepend, flags=re.S | re.I)
    if prepend:
        return prepend + "\n\n" + (original or "")
    return original or ""


async def run_sample(
    client: OpenAIClient,
    sample: Dict[str, Any],
    index: int,
    prepend_prompt: str,
    enable_thinking: bool,
    media_base: str,
    nframes_seq: List[int],
    pixels_seq: List[int],
    min_pixels: int,
    prefer_mp4: bool,
) -> Dict[str, Any]:
    user_msg = next((m for m in (sample.get("messages") or []) if m.get("role") == "user"), None)
    original = (user_msg or {}).get("content", "")
    user_content = fix_prompt(prepend_prompt, original, enable_thinking)
    videos_rel: List[str] = sample.get("videos") or []
    def _resolve(p: str) -> str:
        ap = p if os.path.isabs(p) else os.path.join(media_base, p)
        root, ext = os.path.splitext(ap)
        if prefer_mp4:
            mp4p = root + ".mp4"
            if os.path.exists(mp4p) or os.path.islink(mp4p):
                return mp4p
        if os.path.exists(ap) or os.path.islink(ap):
            return ap
        # try alternate
        if ext.lower() == ".mkv":
            cand = root + ".mp4"
        else:
            cand = root + ".mkv"
        if os.path.exists(cand) or os.path.islink(cand):
            return cand
        return ap
    videos_abs: List[str] = [_resolve(v) for v in videos_rel]
    gt = int(sample.get("gt") or 0)

    # Try multiple (nframes, total_pixels) pairs to avoid decoder-length errors
    last_err: Optional[str] = None
    request_messages: Optional[List[Dict[str, Any]]] = None
    for nf in nframes_seq:
        for px in pixels_seq:
            messages = client.build_messages(user_content, videos_abs, nf, px, min_pixels)
            text, err = await asyncio.to_thread(client.chat_once, messages)
            if not err:
                order = parse_answer_order(text, n_candidates=len(videos_rel) if videos_rel else 5)
                top1 = order[0] if order else 0
                gt_pos = order.index(gt) + 1 if (gt and gt in order) else 0
                top1_correct = (top1 == gt and gt != 0)
                record = {
                    "user": user_content,
                    "videos": videos_rel,
                    "gt": gt,
                    "predict": text,
                    "pred_order": order,
                    "top1_correct": top1_correct,
                    "gt_pos": gt_pos,
                    "index": index,
                    "request_messages": messages,
                }
                return record
            last_err = err
            # Only fall back on explicit length-related errors; otherwise retry loop already handled
            if "maximum model length" not in err and "decoder prompt" not in err:
                # break inner/outer loops
                nf = -1  # type: ignore[assignment]
                break
        if last_err and ("maximum model length" not in last_err and "decoder prompt" not in last_err):
            break

    # If all attempts failed, return a record carrying the last error
    return {
        "user": user_content,
        "videos": videos_rel,
        "gt": gt,
        "predict": "",
        "pred_order": [],
        "top1_correct": False,
        "gt_pos": 0,
        "index": index,
        "request_messages": request_messages or [],
        "error": last_err or "Unknown error",
    }


def ensure_repo_id(hf_repo: Optional[str], default_name: str) -> Optional[str]:
    if hf_repo and "/" in hf_repo:
        return hf_repo.strip()
    try:
        api = HfApi()
        token = HfFolder.get_token()
        if not token:
            return None
        who = api.whoami(token)
        user = who.get("name") or who.get("username")
        if not user:
            return None
        name = hf_repo or default_name
        # light sanitize
        def _s(s: str) -> str:
            s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
            return s.strip("-.") or "results"

        return f"{_s(user)}/{_s(name)}"
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    # 1) Collect error indices
    err_indices = collect_error_indices(args.jsonl_in)
    if not err_indices:
        raise SystemExit("No error entries found in JSONL; nothing to rerun.")

    # 2) Load base JSON and dataset
    with open(args.base_json_in, "r", encoding="utf-8") as f:
        base = json.load(f)
    items: List[Dict[str, Any]] = base.get("items") or []

    data = load_dataset_from_hf(args.dataset_repo, args.dataset_file)
    if len(items) != len(data):
        print(f"[warn] base items ({len(items)}) != dataset length ({len(data)}); proceeding by position.")

    # 3) Prepare API client and fallback schedules
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
    )

    nframes_seq: List[int] = [int(x) for x in str(args.video_num_frames).split(",") if str(x).strip()]
    # extend with fallback frames
    nframes_seq.extend(int(x) for x in (args.fallback_frames.split(",") if args.fallback_frames else []) if str(x).strip())
    # unique while preserving order
    seen_nf: Set[int] = set()
    nframes_seq = [x for x in nframes_seq if not (x in seen_nf or seen_nf.add(x))]  # type: ignore[arg-type]

    pixels_seq: List[int] = [int(x) for x in str(args.video_total_pixels).split(",") if str(x).strip()]
    pixels_seq.extend(int(x) for x in (args.fallback_pixels.split(",") if args.fallback_pixels else []) if str(x).strip())
    seen_px: Set[int] = set()
    pixels_seq = [x for x in pixels_seq if not (x in seen_px or seen_px.add(x))]  # type: ignore[arg-type]

    # 4) Optionally prepare JSONL stream out
    if args.stream_jsonl_out:
        os.makedirs(os.path.dirname(args.stream_jsonl_out) or ".", exist_ok=True)
        if args.truncate_stream_out:
            with open(args.stream_jsonl_out, "w", encoding="utf-8"):
                pass
        # Stream all existing non-error items immediately if requested
        if args.stream_existing:
            streamed = 0
            with open(args.stream_jsonl_out, "a", encoding="utf-8") as jf:
                for i, rec in enumerate(items):
                    if not rec or not isinstance(rec, dict):
                        continue
                    if rec.get("error"):
                        continue
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    streamed += 1
            print(f"[stream] wrote {streamed} existing non-error items to {args.stream_jsonl_out}")

    # 5) Rerun in parallel
    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    results: Dict[int, Dict[str, Any]] = {}

    # Echo setup
    if args.echo_ic:
        try:
            from icecream import ic as _ic  # type: ignore

            _ic.configureOutput(prefix=args.ic_prefix)

            def _echo(**kw):
                _ic(kw)

        except Exception:

            def _echo(**kw):
                print(args.ic_prefix + json.dumps(kw, ensure_ascii=False))

    else:

        def _echo(**kw):
            pass

    async def _run(idx: int):
        async with sem:
            rec = await run_sample(
                client=client,
                sample=data[idx],
                index=idx,
                prepend_prompt=args.prepend_prompt,
                enable_thinking=args.enable_thinking,
                media_base=args.media_base,
                nframes_seq=nframes_seq,
                pixels_seq=pixels_seq,
                min_pixels=args.video_min_pixels,
                prefer_mp4=args.prefer_mp4,
            )
            results[idx] = rec
            if args.stream_jsonl_out:
                with open(args.stream_jsonl_out, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # progress echo
            _echo(idx=idx)

    async def _main_all():
        total = len(err_indices)
        done = 0
        # Wrap tasks to update counters
        async def _wrap(i: int):
            nonlocal done
            await _run(i)
            done += 1
            _echo(done=done, remaining=total - done, total=total)

        await asyncio.gather(*[_wrap(i) for i in err_indices])

    asyncio.run(_main_all())

    # 6) Merge into base items
    for i, rec in results.items():
        # Ensure no lingering error field makes it into the final output unless we truly failed again
        if rec.get("error"):
            # keep as-is to reflect persistent failure
            items[i] = rec
        else:
            rec.pop("error", None)
            items[i] = rec

    # 7) Recompute metrics and write output
    metrics = recompute_metrics(items)
    out = {
        "model": base.get("model") or args.model_repo,
        "dataset": base.get("dataset") or args.dataset_repo,
        "items": items,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Wrote without-error JSON ->", args.output_json)
    print("Metrics:", json.dumps(metrics, indent=2))

    # 8) Optional push to HF
    if args.push_hf:
        repo = ensure_repo_id(args.hf_out_repo, os.path.splitext(os.path.basename(args.output_json))[0])
        if not repo:
            print("[warn] Could not resolve HF repo id; skip push.")
            return
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
        out_file = args.hf_out_file or os.path.basename(args.output_json)
        commit_msg = args.commit_message or f"Upload without-error results ({len(err_indices)} repaired)"
        api.upload_file(
            path_or_fileobj=args.output_json,
            path_in_repo=out_file,
            repo_id=repo,
            repo_type="dataset",
            commit_message=commit_msg,
        )
        print(f"Pushed to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
