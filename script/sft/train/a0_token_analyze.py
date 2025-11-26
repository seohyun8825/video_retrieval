#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Ensure LLaMA-Factory is on PYTHONPATH via the caller bash wrapper.
from datasets import Dataset
from transformers import AutoProcessor, AutoTokenizer, Seq2SeqTrainingArguments

# LLaMA-Factory internals (local copy under SFT/sft_trainer)
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.loader import _load_single_dataset
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.model_args import ModelArguments


def _read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please install pyyaml.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tokenizer_and_template(model_name: str, template_name: str, enable_thinking: Optional[bool]) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    data_args = DataArguments(template=template_name, enable_thinking=enable_thinking)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer, template


def build_processor(model_name: str, video_max_pixels: Optional[int], video_maxlen: Optional[int], video_fps: Optional[float]):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # Attach runtime overrides that LLaMA-Factory plugins read
    if video_max_pixels is not None:
        setattr(processor, "video_max_pixels", int(video_max_pixels))
    if video_maxlen is not None:
        setattr(processor, "video_maxlen", int(video_maxlen))
    if video_fps is not None:
        setattr(processor, "video_fps", float(video_fps))
    return processor


def load_aligned_dataset(dataset_name: str, dataset_dir: str, media_dir: Optional[str], max_samples: Optional[int], cache_dir: Optional[str] = None) -> Dataset:
    # Minimal Model/Data/Training args for loader
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen3-VL-2B-Thinking", cache_dir=cache_dir)
    data_args = DataArguments(dataset_dir=dataset_dir, media_dir=media_dir, max_samples=max_samples)
    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(dataset_dir, ".tmp_token_analysis"), dataloader_num_workers=4)

    dataset_attr = get_dataset_list([dataset_name], dataset_dir)[0]
    ds = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
    assert isinstance(ds, Dataset), "Only regular Dataset is supported for analysis."
    return ds


def tokenize_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def analyze_sample(
    sample: Dict[str, Any],
    tokenizer,
    template,
    processor,
    video_token_symbol: str,
    cutoff_len: int,
    template_name: str,
) -> Dict[str, Any]:
    # Build convenience structures
    prompt: List[Dict[str, str]] = sample.get("_prompt", [])
    response: List[Dict[str, str]] = sample.get("_response", [])
    system: Optional[str] = sample.get("_system", None)
    tools: Optional[str] = sample.get("_tools", None)
    videos: List[Any] = (sample.get("_videos") or [])

    # Count text tokens segmented by <video> per message
    import re

    segments: List[Dict[str, Any]] = []
    video_placeholder = "<video>"
    global_video_index = 0

    # Pre-compute per-video visual tokens analytically via plugin mm inputs
    # and also per-video frame counts for reporting.
    # Use the same multimodal plugin as the active template
    plugin = template.mm_plugin

    video_visual_token_counts: List[int] = []  # count of <|video_pad|> tokens only
    video_frames: List[int] = []
    video_wrap_tokens: List[int] = []  # count of vision BOS/EOS tokens around frames
    video_total_tokens: List[int] = []  # visual + wraps (no timestamps)

    for vid in videos:
        # We avoid building a gigantic expanded string. Query plugin internals via _get_mm_inputs.
        mm_inputs = plugin._get_mm_inputs(images=[], videos=[vid], audios=[], processor=processor)  # type: ignore[attr-defined]
        # video_grid_thw: shape (T, H, W) per video, packed in a tensor
        v_thw = mm_inputs.get("video_grid_thw")
        if hasattr(v_thw, "tolist"):
            v_thw = v_thw.tolist()
        # Expect [[T, H, W]] for single video
        if isinstance(v_thw, list) and len(v_thw) > 0:
            T, H, W = map(int, v_thw[0])
            # merge_size from video_processor
            video_processor = getattr(processor, "video_processor", None)
            merge_size = int(getattr(video_processor, "merge_size", 2))
            seqlen_per_frame = (H // merge_size) * (W // merge_size)
            total_visual_tokens = T * seqlen_per_frame
            wraps = 2 * T  # one vision_bos + one vision_eos per frame
            video_visual_token_counts.append(int(total_visual_tokens))
            video_frames.append(int(T))
            video_wrap_tokens.append(int(wraps))
            video_total_tokens.append(int(total_visual_tokens + wraps))
        else:
            video_visual_token_counts.append(0)
            video_frames.append(0)
            video_wrap_tokens.append(0)
            video_total_tokens.append(0)

    # Iterate messages in order: prompt (multi-turn) then response (one turn)
    for mi, m in enumerate(prompt + response):
        role = m.get("role", "")
        content = m.get("content", "")
        if video_placeholder not in content:
            if content:
                segments.append({
                    "message_index": mi,
                    "role": role,
                    "kind": "text",
                    "position": "all",
                    "token_count": tokenize_len(tokenizer, content),
                })
            continue

        parts = re.split(r"(\<video\>)", content)
        seg_idx = 0
        for part in parts:
            if part == video_placeholder:
                # Report using pre-computed visual token count for this placeholder index
                vtoks = video_visual_token_counts[global_video_index] if global_video_index < len(video_visual_token_counts) else 0
                segments.append({
                    "message_index": mi,
                    "role": role,
                    "kind": "video",
                    "video_index": global_video_index,
                    "frames": video_frames[global_video_index] if global_video_index < len(video_frames) else 0,
                    "visual_tokens": vtoks,
                    "wrap_tokens": video_wrap_tokens[global_video_index] if global_video_index < len(video_wrap_tokens) else 0,
                    "total_tokens_no_timestamps": video_total_tokens[global_video_index] if global_video_index < len(video_total_tokens) else 0,
                })
                global_video_index += 1
                seg_idx += 1
            elif part:
                segments.append({
                    "message_index": mi,
                    "role": role,
                    "kind": "text",
                    "position": "pre" if seg_idx == 0 else "between",
                    "token_count": tokenize_len(tokenizer, part),
                })

    # Build full input_ids using the same processor as training for a length reference
    data_args_for_len = DataArguments(
        cutoff_len=cutoff_len,
        template=template_name,
        enable_thinking=getattr(template, "enable_thinking", None),
    )
    dataset_proc = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args_for_len)
    input_ids, _ = dataset_proc._encode_data_example(prompt, response, system, tools, [], videos, [])

    # Summaries
    total_text_tokens = sum(s.get("token_count", 0) for s in segments if s["kind"] == "text")
    total_video_visual_tokens = sum(s.get("visual_tokens", 0) for s in segments if s["kind"] == "video")

    return {
        "num_videos": len(videos),
        "video_frames": video_frames,
        "video_visual_token_counts": video_visual_token_counts,
        "video_wrap_token_counts": video_wrap_tokens,
        "video_total_tokens_no_timestamps": video_total_tokens,
        "segments": segments,
        "total_text_tokens": int(total_text_tokens),
        "total_video_visual_tokens": int(total_video_visual_tokens),
        "input_ids_total_len": int(len(input_ids)),
    }


def main():
    p = argparse.ArgumentParser(description="Per-sample token analysis for Qwen3-VL SFT data")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--media_dir", required=False, default=None)
    p.add_argument("--video_maxlen", type=int, default=None)
    p.add_argument("--video_fps", type=float, default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=50)
    args = p.parse_args()

    cfg = _read_yaml(args.config)
    model_name = cfg.get("model_name_or_path", "Qwen/Qwen3-VL-2B-Thinking")
    template_name = cfg.get("template", "qwen3_vl")
    enable_thinking = cfg.get("enable_thinking", True)
    cutoff_len = int(cfg.get("cutoff_len", 2048000))
    video_max_pixels = cfg.get("video_max_pixels", None)

    # Build components
    tokenizer, template = build_tokenizer_and_template(model_name, template_name, enable_thinking)
    processor = build_processor(model_name, video_max_pixels, args.video_maxlen, args.video_fps)

    # Load dataset aligned to _prompt/_response/_videos
    ds = load_aligned_dataset(args.dataset, args.dataset_dir, args.media_dir, max_samples=args.max_samples)

    # Output file
    _ensure_dir(args.out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"token_analysis_{args.dataset}_{ts}.jsonl")

    # video token symbol for Qwen3-VL template
    video_token_symbol = "<|video_pad|>"

    total_stats = {
        "dataset": args.dataset,
        "model": model_name,
        "template": template_name,
        "cutoff_len": cutoff_len,
        "video_max_pixels": int(video_max_pixels) if video_max_pixels is not None else None,
        "video_maxlen": args.video_maxlen,
        "video_fps": args.video_fps,
        "num_samples": len(ds),
    }

    with open(out_path, "w", encoding="utf-8") as w:
        w.write(json.dumps({"meta": total_stats}, ensure_ascii=False) + "\n")
        for idx, ex in enumerate(ds):
            try:
                result = analyze_sample(ex, tokenizer, template, processor, video_token_symbol, cutoff_len, template_name)
                result.update({"index": idx})
            except Exception as e:  # robust to any single-sample failure
                result = {"index": idx, "error": str(e)}
            w.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
