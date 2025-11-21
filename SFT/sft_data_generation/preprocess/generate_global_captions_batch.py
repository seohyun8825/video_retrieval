#!/usr/bin/env python3
"""
Generate global captions for timestamp-level captions using the OpenAI Batch API.

This script:
1. Reads timestamp-based caption data from an input JSON file.
2. Builds per-video prompts from a template file (with <INSERT CAPTION ARRAY HERE> placeholder).
3. Splits prompts into chunked batch requests to the OpenAI Responses endpoint.
4. Polls the created batches concurrently until they complete, then extracts each global caption.
5. Writes the input JSON back out with an added "global_caption" field per entry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


PLACEHOLDER = "<INSERT CAPTION ARRAY HERE>"


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}")
    return data


def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"Prompt template must include placeholder {PLACEHOLDER}")
    return template


def read_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"No API key found in {path}")
    return key


def build_prompt(template: str, captions: List[str]) -> str:
    caption_text = json.dumps(captions, ensure_ascii=False, indent=2)
    return template.replace(PLACEHOLDER, caption_text)


def create_batch_requests(
    data: List[Dict[str, Any]], template: str, model: str, start_index: int = 0
) -> List[Dict[str, Any]]:
    requests = []
    for offset, item in enumerate(data):
        idx = start_index + offset
        captions = item.get("caption")
        if not captions:
            raise ValueError(f"Entry {idx} missing 'caption' array")
        prompt = build_prompt(template, captions)
        request = {
            "custom_id": f"global_caption_{idx}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "input": prompt,
            },
        }
        requests.append(request)
    return requests


def upload_batch_file(client: OpenAI, requests: List[Dict[str, Any]]) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for req in requests:
            tmp.write(json.dumps(req, ensure_ascii=False) + "\n")
        temp_path = tmp.name
    try:
        with open(temp_path, "rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")
    finally:
        os.remove(temp_path)
    return input_file.id


def wait_for_batches(
    client: OpenAI,
    batch_jobs: List[Dict[str, Any]],
    poll_interval: int = 15,
    timeout: int = 3600,
) -> None:
    start_time = time.time()
    pending = {job["batch_id"] for job in batch_jobs}
    while pending:
        for job in batch_jobs:
            batch_id = job["batch_id"]
            if batch_id not in pending:
                continue
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            print(f"[batch:{batch_id}] status={status}")
            if status == "completed":
                job["completed_batch"] = batch
                pending.remove(batch_id)
            elif status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch {batch_id} ended with status {status}: {batch}")
        if pending:
            if time.time() - start_time > timeout:
                raise TimeoutError("Batches did not finish within the allotted time.")
            time.sleep(poll_interval)


def _read_file_content(response: Any) -> str:
    """Handle different shapes of file content objects from the OpenAI SDK."""
    if hasattr(response, "text") and response.text is not None:
        return response.text
    if hasattr(response, "content") and isinstance(response.content, (bytes, bytearray)):
        return response.content.decode("utf-8")
    if hasattr(response, "body") and isinstance(response.body, (bytes, bytearray)):
        return response.body.decode("utf-8")
    if hasattr(response, "read"):
        data = response.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    raise RuntimeError("Unable to read batch output file content from response object")


def download_batch_output(client: OpenAI, batch) -> Tuple[List[Dict[str, Any]], str]:
    if not getattr(batch, "output_file_id", None):
        raise RuntimeError("Batch completed but no output_file_id found")
    response = client.files.content(batch.output_file_id)
    text = _read_file_content(response)
    lines = [line for line in text.splitlines() if line.strip()]
    return [json.loads(line) for line in lines], text


def save_raw_text(raw_text: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(raw_text, encoding="utf-8")
    print(f"Saved raw output to {output_path}")
    return output_path


def request_single_global_caption(
    client: OpenAI, template: str, model: str, captions: List[str]
) -> Tuple[Dict[str, Any], str]:
    prompt = build_prompt(template, captions)
    response = client.responses.create(model=model, input=prompt)
    response_dict = json.loads(response.model_dump_json())
    raw_text = json.dumps(response_dict, ensure_ascii=False, indent=2)
    return response_dict, raw_text


def extract_text_from_response(response_body: Dict[str, Any]) -> str:
    outputs = response_body.get("output") or response_body.get("outputs")
    if not outputs:
        return ""
    chunks: List[str] = []
    for output in outputs:
        for content in output.get("content", []):
            if content.get("type") in ("output_text", "text"):
                chunks.append(content.get("text", ""))
    return "".join(chunks).strip()


GLOBAL_PATTERN = re.compile(r"<global_caption>(.*?)</global_caption>", re.DOTALL | re.IGNORECASE)


def strip_xml_tag(text: str) -> str:
    match = GLOBAL_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def merge_results(
    data: List[Dict[str, Any]], responses: List[Dict[str, Any]], fail_on_missing: bool = True
) -> None:
    lookup: Dict[str, str] = {}
    for entry in responses:
        custom_id = entry.get("custom_id")
        body = entry.get("response", {}).get("body", {})
        raw_text = extract_text_from_response(body)
        lookup[custom_id] = strip_xml_tag(raw_text)

    missing = []
    for idx, item in enumerate(data):
        key = f"global_caption_{idx}"
        caption = lookup.get(key)
        if caption:
            item["global_caption"] = caption
        else:
            missing.append(key)
    if missing and fail_on_missing:
        raise RuntimeError(f"Missing batch outputs for: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate global captions via OpenAI batch processing.")
    parser.add_argument("--input_json", required=True, type=Path, help="Path to the input JSON file.")
    parser.add_argument("--output_json", required=True, type=Path, help="Path to save the output JSON file.")
    parser.add_argument(
        "--prompt_file",
        default=Path(__file__).resolve().parent.parent / "prompt" / "global_caption_generate.txt",
        type=Path,
        help="Prompt template file path.",
    )
    parser.add_argument(
        "--api_key_path",
        default=Path(__file__).resolve().parents[2] / "openai",
        type=Path,
        help="Path to the file that stores the OpenAI API key.",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name to use.")
    parser.add_argument("--poll_interval", type=int, default=15, help="Seconds between batch status polls.")
    parser.add_argument("--timeout", type=int, default=3600, help="Max seconds to wait for batch completion.")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=30,
        help="Number of samples to include in a single batch submission.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many entries (from the start) to process.",
    )
    parser.add_argument(
        "--raw_output_dir",
        type=Path,
        default=None,
        help="Optional directory to store raw batch outputs (JSONL).",
    )
    parser.add_argument(
        "--batch_api",
        "--batch-api",
        dest="batch_api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the OpenAI Batch API (default). Pass --no-batch-api to use standard responses with threading.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum parallel workers when using standard (non-batch) API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json(args.input_json)
    template = load_prompt_template(args.prompt_file)
    api_key = read_api_key(args.api_key_path)

    client = OpenAI(api_key=api_key)

    total_entries = len(data)
    if total_entries == 0:
        raise ValueError("Input JSON is empty.")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive.")

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive.")
        process_total = min(total_entries, args.limit)
        print(f"Limiting processing to first {process_total} entries (of {total_entries}).")
    else:
        process_total = total_entries

    if process_total == 0:
        raise ValueError("No entries to process after applying limit.")

    if args.batch_api:
        print("Using OpenAI Batch API mode.")
        batch_jobs: List[Dict[str, Any]] = []
        for start in range(0, process_total, args.chunk_size):
            end = min(start + args.chunk_size, process_total)
            chunk = data[start:end]
            print(f"Processing entries {start}-{end - 1} (size={len(chunk)})")

            requests = create_batch_requests(chunk, template, args.model, start_index=start)
            input_file_id = upload_batch_file(client, requests)
            print(f"Uploaded batch request file for entries {start}-{end - 1}: {input_file_id}")

            batch = client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/responses",
                completion_window="24h",
            )
            print(f"Created batch: {batch.id}")

            batch_jobs.append(
                {
                    "start": start,
                    "end": end,
                    "batch_id": batch.id,
                }
            )

        wait_for_batches(client, batch_jobs, poll_interval=args.poll_interval, timeout=args.timeout)

        for job in batch_jobs:
            batch = job.get("completed_batch")
            if not batch:
                raise RuntimeError(f"Batch {job['batch_id']} did not complete successfully.")
            responses, raw_text = download_batch_output(client, batch)
            if args.raw_output_dir:
                inclusive_end = max(job["start"], job["end"] - 1)
                filename = f"{job['start']:05d}-{inclusive_end:05d}_{job['batch_id']}.jsonl"
                save_raw_text(raw_text, args.raw_output_dir, filename)
            merge_results(data, responses)
    else:
        print(
            f"Using standard Responses API mode with up to {args.max_workers} parallel workers "
            f"for {process_total} entries."
        )
        responses_payload: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {}
            for idx in range(process_total):
                captions = data[idx].get("caption")
                if not captions:
                    raise ValueError(f"Entry {idx} missing 'caption' array")
                future = executor.submit(
                    request_single_global_caption,
                    client,
                    template,
                    args.model,
                    captions,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                response_body, raw_text = future.result()
                responses_payload.append(
                    {
                        "custom_id": f"global_caption_{idx}",
                        "response": {"body": response_body},
                    }
                )
                if args.raw_output_dir:
                    filename = f"standard_{idx:05d}.json"
                    save_raw_text(raw_text, args.raw_output_dir, filename)

        merge_results(data, responses_payload)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote updated data to {args.output_json}")


if __name__ == "__main__":
    main()
