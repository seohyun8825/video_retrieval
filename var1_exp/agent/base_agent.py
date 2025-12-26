#!/usr/bin/env python3
"""
Lightweight agent that consumes retrieval outputs (per_query_rankings.jsonl),
calls a vLLM OpenAI-compatible endpoint, and performs:
1) Top-1 temporal grounding (or "NO" when the clip does not match).
2) Top-5 reranking with the MLLM followed by temporal grounding on the winner.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from tqdm import tqdm


RERANK_PROMPT_TEMPLATE = """You are ranking candidate videos for the query below.
Return only the best index using the exact format <answer>[i]</answer> where i is 1-based.
Add a 1-2 sentence justification after the answer.
If none of the videos match the query, respond with <answer>NONE</answer> and explain briefly.

Query:
{query}

Candidates (ordered to match attached videos):
{candidates_block}
"""

GROUND_PROMPT_TEMPLATE = """You are a temporal grounding assistant.
Given the query, watch the attached video and identify the tight time span in seconds.
- If the event is present, respond with <answer>[start_sec, end_sec]</answer> using seconds with one decimal when possible.
- If the video does not contain the event or is too ambiguous, respond with exactly NO.
Do not add any other text after the answer.

Query:
{query}
"""

GROUND_PROMPT_STRICT_TEMPLATE = """You are a temporal grounding assistant.
Given the query, watch the attached video and return only the tight time span in seconds.
- Respond strictly with <answer>[start_sec, end_sec]</answer> (start <= end).
- Do not answer NO or any other text besides the bracketed span.

Query:
{query}
"""


@dataclass
class APIConfig:
    model: str
    api_base: str = "http://localhost:8099/v1"
    api_key: Optional[str] = None
    timeout: float = 240.0  # backwards compatibility (used for both if others not set)
    connect_timeout: Optional[float] = None
    read_timeout: Optional[float] = None
    max_retries: int = 3
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = -1
    max_tokens: int = 1024
    presence_penalty: float = 1.0
    video_num_frames: int = 30
    video_total_pixels: int = 402144
    video_min_pixels: int = 0
    debug: bool = False


def load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_video_index(path: str) -> Dict[str, dict]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    index: Dict[str, dict] = {}
    for row in data:
        vid = str(row.get("video_id", "")).strip()
        if not vid:
            continue
        index[vid] = row
    return index


def parse_answer_order(text: str, n_candidates: int) -> List[int]:
    """Strict parser for <answer>[i]</answer>."""
    if not text:
        return []
    m = re.search(r"<answer[^>]*>(.*?)</answer>", text, flags=re.S | re.I)
    if not m:
        return []
    scope = m.group(1)
    m_br = re.search(r"\[(\d+)\]", scope)
    if m_br:
        i = int(m_br.group(1))
        if 1 <= i <= max(1, n_candidates):
            return [i]
        return []
    m_d = re.search(r"\b([0-9])\b", scope)
    if m_d:
        i = int(m_d.group(1))
        if 1 <= i <= max(1, n_candidates):
            return [i]
    return []


def is_rerank_none(text: str) -> bool:
    """Detect if rerank response declares no matching video."""
    if not isinstance(text, str):
        return False
    m = re.search(r"<answer[^>]*>(.*?)</answer>", text, flags=re.S | re.I)
    scope = m.group(1) if m else text
    return bool(re.search(r"\bnone\b", scope, flags=re.I))


def parse_temporal_span(text: str) -> Optional[Tuple[float, float]]:
    """[start, end] 구간을 텍스트에서 추출한다."""
    if not isinstance(text, str):
        return None
    scope = text
    m = re.search(r"<answer[^>]*>(.*?)</answer>", text, flags=re.S | re.I)
    if m:
        scope = m.group(1)
    # [start, end]
    m = re.search(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]", scope)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a <= b:
            return a, b
    # start=..., end=...
    start = re.search(r"start\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", scope, flags=re.I)
    end = re.search(r"end\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", scope, flags=re.I)
    if start and end:
        a, b = float(start.group(1)), float(end.group(1))
        if a <= b:
            return a, b
    return None


def interval_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Compute IoU for 1D intervals [start, end]."""
    p0, p1 = pred
    g0, g1 = gt
    if p1 < p0 or g1 < g0:
        return 0.0
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    if union <= 0:
        return 0.0
    return inter / union


def load_existing_desc_ids(path: str) -> Set[str]:
    """Return desc_id set from an existing output JSONL (for skip/resume)."""
    if not os.path.exists(path):
        return set()
    desc_ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                did = obj.get("desc_id")
                if did is not None:
                    desc_ids.add(str(did))
            except Exception:
                continue
    return desc_ids


class OpenAIClient:
    """Minimal OpenAI-compatible client using requests."""

    def __init__(self, cfg: APIConfig) -> None:
        self.endpoint = cfg.api_base.rstrip("/") + "/chat/completions"
        self.model = cfg.model
        self.timeout = cfg.timeout
        self.max_retries = max(1, cfg.max_retries)
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = int(cfg.top_k) if cfg.top_k is not None else -1
        self.max_tokens = cfg.max_tokens
        self.presence_penalty = cfg.presence_penalty
        self.system_prompt = ""
        self.video_num_frames = cfg.video_num_frames
        self.video_total_pixels = cfg.video_total_pixels
        self.video_min_pixels = cfg.video_min_pixels
        self.debug = cfg.debug
        # requests timeout supports tuple(connect, read); fall back to single value
        if cfg.connect_timeout is not None or cfg.read_timeout is not None:
            self.timeout = (cfg.connect_timeout or cfg.timeout, cfg.read_timeout or cfg.timeout)
        else:
            self.timeout = cfg.timeout
        self.headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            self.headers["Authorization"] = f"Bearer {cfg.api_key}"

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    @staticmethod
    def _path_to_file_url(path: str) -> str:
        abs_path = os.path.abspath(path)
        return "file://" + abs_path

    def _video_item_payload(self, item: Dict[str, Any]) -> Dict[str, Any]:
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

    def _build_messages(self, prompt: str, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        content: List[Dict[str, Any]] = []
        placeholder = "<video>"
        if videos and prompt and placeholder in prompt:
            parts = re.split(r"(\<video\>)", prompt)
            vid_idx = 0
            for part in parts:
                if part == placeholder:
                    if vid_idx < len(videos):
                        content.append(self._video_item_payload(videos[vid_idx]))
                        vid_idx += 1
                    else:
                        content.append({"type": "text", "text": placeholder})
                elif part:
                    content.append({"type": "text", "text": part})
            while vid_idx < len(videos):
                content.append(self._video_item_payload(videos[vid_idx]))
                vid_idx += 1
        else:
            if prompt:
                content.append({"type": "text", "text": prompt})
            for item in videos or []:
                content.append(self._video_item_payload(item))
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
        if self.top_k and self.top_k > 0:
            payload["top_k"] = self.top_k
            payload["do_sample"] = True
        start = time.perf_counter()
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            self._dbg(
                f"[DEBUG][API] attempt={attempt + 1}/{self.max_retries} url={self.endpoint} timeout={self.timeout} videos={len(videos)}"
            )
            try:
                resp = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=self.timeout)
                self._dbg(f"[DEBUG][API] status={resp.status_code} elapsed_ms={resp.elapsed.total_seconds() * 1000:.1f}")
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
                self._dbg(f"[DEBUG][API] success latency_ms={latency_ms:.1f}")
                return text, latency_ms, None, messages
            except Exception as exc:  # pylint: disable=broad-except
                last_err = f"{type(exc).__name__}: {exc}"
                self._dbg(f"[DEBUG][API] error={last_err}")
                if attempt < self.max_retries - 1:
                    time.sleep(min(5.0, 1.5 * (attempt + 1)))
                else:
                    break
        latency_ms = (time.perf_counter() - start) * 1000.0
        return "", latency_ms, last_err, messages


class BaseAgent:
    # vLLM API를 호출해 rerank/grounding 두 가지 작업을 수행하는 경량 에이전트
    def __init__(
        self,
        client: OpenAIClient,
        video_index: Dict[str, dict],
        rerank_prompt: str,
        ground_prompt: str,
        ground_prompt_strict: str,
        video_num_frames: int,
        video_total_pixels: int,
        video_min_pixels: int,
        video_fps: float,
        debug: bool = False,
    ) -> None:
        self.client = client
        self.video_index = video_index
        self.rerank_prompt = rerank_prompt
        self.ground_prompt = ground_prompt
        self.ground_prompt_strict = ground_prompt_strict
        self.video_num_frames = video_num_frames
        self.video_total_pixels = video_total_pixels
        self.video_min_pixels = video_min_pixels
        self.video_fps = video_fps
        self.debug = debug

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _video_payload_from_id(self, video_id: str) -> Optional[Dict[str, Any]]:
        # video_id로부터 vLLM에 보낼 로컬 파일 URL/메타를 구성
        meta = self.video_index.get(video_id)
        if not meta:
            return None
        path = meta.get("video_path") or meta.get("path") or meta.get("video_rel")
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return {
            "path": path,
            "meta": {
                "nframes": self.video_num_frames,
                "total_pixels": self.video_total_pixels,
                "min_pixels": self.video_min_pixels,
                "fps": self.video_fps,
            },
            "video_id": video_id,
        }

    def _format_rerank_prompt(self, query: str, k: int) -> str:
        # 후보 개수(k)에 맞춰 <video> 플래이스홀더를 생성
        candidates_block = "\n".join([f"[{i}] <video>" for i in range(1, k + 1)])
        return self.rerank_prompt.format(query=query, candidates_block=candidates_block)

    def _format_ground_prompt(self, query: str) -> str:
        return self.ground_prompt.format(query=query)

    def _format_ground_prompt_strict(self, query: str) -> str:
        return self.ground_prompt_strict.format(query=query)

    def _write_record(self, fh, record: dict) -> None:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_ground_top1(
        self,
        rankings: List[dict],
        output_jsonl: str,
        *,
        output_json: Optional[str] = None,
        metrics_json: Optional[str] = None,
    ) -> None:
        # (태스크1) top-1 후보만 넣어 바로 temporal grounding 수행
        existing = load_existing_desc_ids(output_jsonl)
        if self.debug and existing:
            print(f"[DEBUG] Skipping {len(existing)} existing records (ground_top1)")
        total = 0
        gt_known = 0
        gt_in_topk = 0
        gt_matched = 0
        error_no_topk = 0
        error_missing_path = 0
        iou_sum_matched = 0.0
        iou_count_matched = 0
        wrong_video = 0
        wrong_video_marked_no = 0
        records: List[dict] = []
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for row in tqdm(rankings, desc="Top1 grounding", unit="query"):
                query = row.get("query", "")
                topk = row.get("topk") or []
                gt_vid = str(row.get("video_id")) if row.get("video_id") else None
                gt_time = row.get("time")
                base_rec = {
                    "mode": "ground_top1",
                    "desc_id": row.get("desc_id"),
                    "query": query,
                    "retrieved_topk": topk[:5],
                    **({"gt_video_id": gt_vid} if gt_vid else {}),
                    **({"gt_time": gt_time} if gt_time else {}),
                }
                desc_id = base_rec.get("desc_id")
                if desc_id is not None and str(desc_id) in existing:
                    continue
                if not topk:
                    base_rec["error"] = "No candidates in topk."
                    error_no_topk += 1
                    self._write_record(out_f, base_rec)
                    records.append(base_rec)
                    continue
                total += 1

                top1_vid = str(topk[0].get("video_id"))
                self._dbg(f"[DEBUG][Top1] desc_id={row.get('desc_id')} query={query[:120]}")
                if gt_vid:
                    gt_known += 1
                    in_cands = any(str(c.get("video_id")) == gt_vid for c in topk)
                    if in_cands:
                        gt_in_topk += 1
                    base_rec["gt_in_candidates"] = in_cands

                payload = self._video_payload_from_id(top1_vid)
                if not payload:
                    base_rec["error"] = f"Missing video path for {top1_vid}"
                    error_missing_path += 1
                    self._write_record(out_f, base_rec)
                    records.append(base_rec)
                    continue
                self._dbg(f"[DEBUG][Top1] payload video_id={top1_vid} path={payload.get('path')}")

                prompt = self._format_ground_prompt(query)
                text, latency_ms, error, _ = self.client.chat(prompt, [payload])
                span = parse_temporal_span(text)
                verdict = "no" if re.search(r"\bNO\b", str(text).strip(), flags=re.I) else "match"
                self._dbg(
                    f"[DEBUG][Top1][response] latency_ms={latency_ms:.1f} error={error} text={str(text)[:300]}"
                )
                iou = None
                if span and gt_time and isinstance(gt_time, (list, tuple)) and len(gt_time) == 2:
                    try:
                        gt_start, gt_end = float(gt_time[0]), float(gt_time[1])
                        iou = interval_iou((span[0], span[1]), (gt_start, gt_end))
                    except Exception:
                        iou = None

                base_rec["grounding"] = {
                    "video_id": top1_vid,
                    "response": text,
                    "latency_ms": latency_ms,
                    "error": error,
                    "span": {"start": span[0], "end": span[1]} if span else None,
                    "verdict": verdict,
                    **({"iou": iou} if iou is not None else {}),
                }
                if gt_vid:
                    matched = top1_vid == gt_vid
                    base_rec["grounding"]["match_gt"] = matched
                    if matched:
                        gt_matched += 1
                        if iou is not None:
                            iou_sum_matched += iou
                            iou_count_matched += 1
                    else:
                        wrong_video += 1
                        if verdict == "no":
                            wrong_video_marked_no += 1
                self._write_record(out_f, base_rec)
                records.append(base_rec)
        # 간단한 메트릭 로그
        if gt_known:
            print(
                f"[Top1] processed={total}, gt_known={gt_known}, "
                f"gt_in_topk={gt_in_topk}, top1_match={gt_matched}, "
                f"mean_tIoU_match={ (iou_sum_matched / iou_count_matched) if iou_count_matched else 'n/a' }"
            )
        metrics = {
            "mode": "ground_top1",
            "total_rows": len(rankings),
            "processed": total,
            "errors_no_topk": error_no_topk,
            "errors_missing_path": error_missing_path,
            "gt_known": gt_known,
            "gt_in_topk": gt_in_topk,
            "top1_match_gt": gt_matched,
            "mean_iou_when_match": (iou_sum_matched / iou_count_matched) if iou_count_matched else None,
            "wrong_video_total": wrong_video,
            "wrong_video_marked_no": wrong_video_marked_no,
        }
        # 전체 JSON/메트릭 저장
        if output_json:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"items": records, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        if metrics_json:
            os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    def run_ground_gt(
        self,
        rankings: List[dict],
        output_jsonl: str,
        *,
        output_json: Optional[str] = None,
        metrics_json: Optional[str] = None,
    ) -> None:
        # (태스크3) GT 영상에 대해서만 temporal grounding 수행 (NO 금지, strict span)
        existing = load_existing_desc_ids(output_jsonl)
        if self.debug and existing:
            print(f"[DEBUG] Skipping {len(existing)} existing records (ground_gt)")
        total = 0
        errors_no_gt = 0
        error_missing_path = 0
        iou_sum = 0.0
        iou_count = 0
        records: List[dict] = []
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for row in tqdm(rankings, desc="Ground GT", unit="query"):
                query = row.get("query", "")
                gt_vid = str(row.get("video_id")) if row.get("video_id") else None
                gt_time = row.get("time")
                base_rec = {
                    "mode": "ground_gt",
                    "desc_id": row.get("desc_id"),
                    "query": query,
                    **({"gt_video_id": gt_vid} if gt_vid else {}),
                    **({"gt_time": gt_time} if gt_time else {}),
                }
                desc_id = base_rec.get("desc_id")
                if desc_id is not None and str(desc_id) in existing:
                    continue
                if not gt_vid:
                    base_rec["error"] = "Missing gt_video_id"
                    errors_no_gt += 1
                    self._write_record(out_f, base_rec)
                    records.append(base_rec)
                    continue
                total += 1

                payload = self._video_payload_from_id(gt_vid)
                if not payload:
                    base_rec["error"] = f"Missing video path for {gt_vid}"
                    error_missing_path += 1
                    self._write_record(out_f, base_rec)
                    records.append(base_rec)
                    continue

                prompt = self._format_ground_prompt_strict(query)
                text, latency_ms, error, _ = self.client.chat(prompt, [payload])
                span = parse_temporal_span(text)
                verdict = "match"
                self._dbg(
                    f"[DEBUG][GroundGT] desc_id={row.get('desc_id')} latency_ms={latency_ms:.1f} error={error} text={str(text)[:300]}"
                )

                iou = None
                if span and gt_time and isinstance(gt_time, (list, tuple)) and len(gt_time) == 2:
                    try:
                        gt_start, gt_end = float(gt_time[0]), float(gt_time[1])
                        iou = interval_iou((span[0], span[1]), (gt_start, gt_end))
                        iou_sum += iou
                        iou_count += 1
                    except Exception:
                        iou = None

                base_rec["grounding"] = {
                    "video_id": gt_vid,
                    "response": text,
                    "latency_ms": latency_ms,
                    "error": error,
                    "span": {"start": span[0], "end": span[1]} if span else None,
                    "verdict": verdict,
                    "match_gt": True,
                    **({"iou": iou} if iou is not None else {}),
                }
                self._write_record(out_f, base_rec)
                records.append(base_rec)

        if total:
            print(
                f"[GroundGT] processed={total}, errors_no_gt={errors_no_gt}, "
                f"errors_missing_path={error_missing_path}, mean_tIoU={ (iou_sum / iou_count) if iou_count else 'n/a' }"
            )
        metrics = {
            "mode": "ground_gt",
            "total_rows": len(rankings),
            "processed": total,
            "errors_no_gt": errors_no_gt,
            "errors_missing_path": error_missing_path,
            "mean_iou": (iou_sum / iou_count) if iou_count else None,
        }
        if output_json:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"items": records, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        if metrics_json:
            os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    def run_rerank_and_ground(
        self,
        rankings: List[dict],
        output_jsonl: str,
        *,
        output_json: Optional[str] = None,
        metrics_json: Optional[str] = None,
        k: int = 5,
    ) -> None:
        # (태스크2) top-k 전체를 rerank 후 선택된 1개에 대해 temporal grounding 수행
        existing = load_existing_desc_ids(output_jsonl)
        if self.debug and existing:
            print(f"[DEBUG] Skipping {len(existing)} existing records (rerank_then_ground)")
        total = 0
        gt_known = 0
        gt_in_topk = 0
        gt_not_in_topk = 0
        gt_selected = 0
        rerank_match = 0
        rerank_none_correct = 0
        rerank_none_wrong = 0
        error_no_payloads = 0
        error_missing_path = 0
        iou_sum_selected = 0.0
        iou_count_selected = 0
        wrong_selected = 0
        wrong_selected_marked_no = 0
        records: List[dict] = []
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for row in tqdm(rankings, desc="Rerank@5 + grounding", unit="query"):
                query = row.get("query", "")
                topk = row.get("topk") or []
                candidates = topk[:k]
                gt_vid = str(row.get("video_id")) if row.get("video_id") else None
                gt_time = row.get("time")
                in_cands = False
                total += 1
                if gt_vid:
                    gt_known += 1
                    in_cands = any(str(c.get("video_id")) == gt_vid for c in candidates)
                    if in_cands:
                        gt_in_topk += 1
                    else:
                        gt_not_in_topk += 1
                # 1단계: top-k 영상으로 rerank -> best index 선정
                base_rec = {
                    "mode": "rerank_then_ground",
                    "desc_id": row.get("desc_id"),
                    "query": query,
                    "retrieved_topk": candidates,
                    **({"gt_video_id": gt_vid} if gt_vid else {}),
                    **({"gt_time": gt_time} if gt_time else {}),
                    **({"gt_in_candidates": in_cands} if gt_vid else {}),
                }
                desc_id = base_rec.get("desc_id")
                if desc_id is not None and str(desc_id) in existing:
                    continue
                payloads: List[Dict[str, Any]] = []
                missing: List[str] = []
                for cand in candidates:
                    vid = str(cand.get("video_id"))
                    payload = self._video_payload_from_id(vid)
                    if payload:
                        payloads.append(payload)
                    else:
                        missing.append(vid)
                self._dbg(
                    f"[DEBUG][Rerank] desc_id={row.get('desc_id')} query={query[:120]} candidates={[c.get('video_id') for c in candidates]}"
                )
                if missing:
                    self._dbg(f"[DEBUG][Rerank] missing video payloads={missing}")
                if not payloads:
                    base_rec["error"] = f"No usable candidates (missing videos: {missing})"
                    error_no_payloads += 1
                    self._write_record(out_f, base_rec)
                    records.append(base_rec)
                    continue

                rerank_prompt = self._format_rerank_prompt(query, len(payloads))
                rerank_text, rerank_latency, rerank_error, _ = self.client.chat(rerank_prompt, payloads)
                pred_none = is_rerank_none(rerank_text)
                order = [] if pred_none else parse_answer_order(rerank_text, len(payloads))
                choice_idx = order[0] - 1 if order else 0
                choice_idx = max(0, min(choice_idx, len(payloads) - 1)) if payloads else 0
                self._dbg(
                    f"[DEBUG][Rerank][response] latency_ms={rerank_latency:.1f} choice_idx={choice_idx + 1 if payloads else 'n/a'} pred_none={pred_none} error={rerank_error} text={str(rerank_text)[:300]}"
                )

                rerank_applicable = bool(gt_vid and in_cands)
                rerank_hit = rerank_applicable and not pred_none and bool(str(candidates[choice_idx].get("video_id")) == gt_vid)
                rerank_pred_none_correct = bool(pred_none and gt_vid and not in_cands)
                rerank_pred_none_wrong = bool(pred_none and gt_vid and in_cands)
                if rerank_hit:
                    rerank_match += 1
                if rerank_pred_none_correct:
                    rerank_none_correct += 1
                if rerank_pred_none_wrong:
                    rerank_none_wrong += 1

                base_rec["rerank"] = {
                    "response": rerank_text,
                    "latency_ms": rerank_latency,
                    "error": rerank_error,
                    "choice_index": (choice_idx + 1) if (not pred_none and payloads) else None,  # 1-based
                    "pred_none": pred_none,
                    **({"match_gt": rerank_hit if rerank_applicable else None} if gt_vid else {}),
                    **({"applicable": rerank_applicable} if gt_vid else {}),
                    **({"none_correct": rerank_pred_none_correct} if gt_vid and not in_cands else {}),
                }

                if rerank_hit:
                    chosen = payloads[choice_idx]
                    # 2단계: 선택된 한 개 영상에 대해 temporal grounding (strict)
                    ground_prompt = self._format_ground_prompt_strict(query)
                    g_text, g_latency, g_error, _ = self.client.chat(ground_prompt, [chosen])
                    span = parse_temporal_span(g_text)
                    verdict = "match"  # strict 모드에서는 NO 금지
                    self._dbg(
                        f"[DEBUG][Ground] video_id={chosen.get('video_id')} latency_ms={g_latency:.1f} error={g_error} text={str(g_text)[:300]}"
                    )
                    iou = None
                    if span and gt_time and isinstance(gt_time, (list, tuple)) and len(gt_time) == 2:
                        try:
                            gt_start, gt_end = float(gt_time[0]), float(gt_time[1])
                            iou = interval_iou((span[0], span[1]), (gt_start, gt_end))
                        except Exception:
                            iou = None

                    base_rec["grounding"] = {
                        "video_id": chosen.get("video_id"),
                        "response": g_text,
                        "latency_ms": g_latency,
                        "error": g_error,
                        "span": {"start": span[0], "end": span[1]} if span else None,
                        "verdict": verdict,
                        **({"iou": iou} if iou is not None else {}),
                    }
                    if gt_vid:
                        matched = chosen.get("video_id") == gt_vid
                        base_rec["grounding"]["match_gt"] = matched
                        if matched:
                            gt_selected += 1
                            if iou is not None:
                                iou_sum_selected += iou
                                iou_count_selected += 1
                else:
                    base_rec["grounding"] = {"skipped": True, "reason": "rerank_not_correct_or_none"}
                self._write_record(out_f, base_rec)
                records.append(base_rec)
        # 간단한 메트릭 로그
        if gt_known:
            print(
                f"[Rerank] processed={total}, gt_known={gt_known}, "
                f"gt_in_topk={gt_in_topk}, rerank_match={rerank_match}, "
                f"gt_not_in_topk={gt_not_in_topk}, rerank_pred_none_correct={rerank_none_correct}, "
                f"gt_selected={gt_selected}, "
                f"mean_tIoU_selected={ (iou_sum_selected / iou_count_selected) if iou_count_selected else 'n/a' }"
            )
        metrics = {
            "mode": "rerank_then_ground",
            "total_rows": len(rankings),
            "processed": total,
            "errors_no_payloads": error_no_payloads,
            "errors_missing_path": error_missing_path,
            "gt_known": gt_known,
            "gt_in_topk": gt_in_topk,
            "gt_not_in_topk": gt_not_in_topk,
            "gt_selected": gt_selected,
            "rerank_match_gt": rerank_match,
            "rerank_pred_none_correct": rerank_none_correct,
            "rerank_pred_none_wrong": rerank_none_wrong,
            "mean_iou_when_selected": (iou_sum_selected / iou_count_selected) if iou_count_selected else None,
            "wrong_selected_total": wrong_selected,
            "wrong_selected_marked_no": wrong_selected_marked_no,
        }
        if output_json:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"items": records, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        if metrics_json:
            os.makedirs(os.path.dirname(metrics_json), exist_ok=True)
            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video retrieval agent using vLLM API")
    p.add_argument("--rankings", required=True, help="per_query_rankings.jsonl path")
    p.add_argument("--video_index", required=True, help="video_candidates.json path (maps video_id to path)")
    p.add_argument("--mode", choices=["ground_top1", "ground_gt", "rerank5_ground"], required=True)
    p.add_argument("--output_jsonl", required=True, help="Output JSONL path (per-sample)")
    p.add_argument("--output_json", help="Output JSON (all samples + metrics). Default: output_jsonl with .json")
    p.add_argument("--metrics_json", help="Output metrics-only JSON. Default: output_jsonl with .metrics.json")
    p.add_argument("--limit", type=int, default=0, help="Max queries to process (0 = all)")
    p.add_argument("--rerank_k", type=int, default=5, help="How many candidates to rerank")
    p.add_argument("--model_repo", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--api_base", default="http://localhost:8099/v1")
    p.add_argument("--api_key", default=None)
    p.add_argument("--request_timeout", type=float, default=240.0, help="Legacy overall timeout (seconds)")
    p.add_argument("--connect_timeout", type=float, default=None, help="Optional connect timeout (seconds)")
    p.add_argument("--read_timeout", type=float, default=None, help="Optional read timeout (seconds)")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--presence_penalty", type=float, default=1.0)
    p.add_argument("--video_num_frames", type=int, default=30)
    p.add_argument("--video_total_pixels", type=int, default=402144)
    p.add_argument("--video_min_pixels", type=int, default=0)
    p.add_argument("--video_fps", type=float, default=2.0)
    p.add_argument("--rerank_prompt", default=RERANK_PROMPT_TEMPLATE)
    p.add_argument("--ground_prompt", default=GROUND_PROMPT_TEMPLATE)
    p.add_argument("--ground_prompt_strict", default=GROUND_PROMPT_STRICT_TEMPLATE)
    p.add_argument("--debug", action="store_true", help="Print verbose debug logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 기본 출력 경로 유도
    out_json = args.output_json or (os.path.splitext(args.output_jsonl)[0] + ".json")
    metrics_json = args.metrics_json or (os.path.splitext(args.output_jsonl)[0] + ".metrics.json")

    rankings = load_jsonl(args.rankings)
    if args.limit and args.limit > 0:
        rankings = rankings[: args.limit]
    video_index = load_video_index(args.video_index)

    cfg = APIConfig(
        model=args.model_repo,
        api_base=args.api_base,
        api_key=args.api_key,
        timeout=args.request_timeout,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
        video_num_frames=args.video_num_frames,
        video_total_pixels=args.video_total_pixels,
        video_min_pixels=args.video_min_pixels,
        debug=args.debug,
    )
    client = OpenAIClient(cfg)

    agent = BaseAgent(
        client=client,
        video_index=video_index,
        rerank_prompt=args.rerank_prompt,
        ground_prompt=args.ground_prompt,
        ground_prompt_strict=args.ground_prompt_strict,
        video_num_frames=args.video_num_frames,
        video_total_pixels=args.video_total_pixels,
        video_min_pixels=args.video_min_pixels,
        video_fps=args.video_fps,
        debug=args.debug,
    )

    if args.mode == "ground_top1":
        agent.run_ground_top1(rankings, args.output_jsonl, output_json=out_json, metrics_json=metrics_json)
    elif args.mode == "ground_gt":
        agent.run_ground_gt(rankings, args.output_jsonl, output_json=out_json, metrics_json=metrics_json)
    else:
        agent.run_rerank_and_ground(rankings, args.output_jsonl, output_json=out_json, metrics_json=metrics_json, k=args.rerank_k)

    print(f"Saved results to {args.output_jsonl}")


if __name__ == "__main__":
    main()
