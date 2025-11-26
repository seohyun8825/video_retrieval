#!/usr/bin/env python3
"""
Generate reranking datasets (easy/medium/hard) using RzenEmbed similarities.

Input JSON format (same as d1 global caption output):
  [ {"video": "relative/path.mp4", "global_caption": "..."}, ... ]

Outputs (same as other pipelines):
  - similarity_matrix.pkl
  - reranking_train_easy.json
  - reranking_train_medium.json
  - reranking_train_hard.json
"""

from __future__ import annotations

# Force-disable FlashAttention2 in this process to avoid ImportError inside transformers
import os as _os
_os.environ["USE_FLASH_ATTENTION_2"] = "0"
_os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
_os.environ["FLASH_ATTENTION_2"] = "0"
_os.environ["TRANSFORMERS_FLASH_ATTENTION_2_ENABLED"] = "0"
_os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
_os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
_os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"
# Silence tokenizers fork parallelism warning
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import pickle
import random
from typing import List, Tuple
import sys
from tqdm import tqdm
import torch

import numpy as np


def load_items(input_json: str) -> List[dict]:
    with open(input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of video entries.")
    required_keys = {"video", "global_caption"}
    for idx, entry in enumerate(items):
        if not required_keys.issubset(entry.keys()):
            missing = required_keys - set(entry.keys())
            raise ValueError(f"Entry {idx} is missing keys: {missing}")
    return items


def extract_frames_cv2(video_path: str, num_frames: int):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        return []
    # If we decoded fewer than requested, repeat last
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def compute_embeddings(
    items: List[dict],
    video_base: str,
    num_frames: int,
    query_instruction: str,
    candidate_instruction: str,
):
    # Try to import from installed pkg first, then fall back to local repo under extractor dir
    try:
        from rzen_embed_inference import RzenEmbed  # type: ignore
    except Exception:
        # Fallback: allow local path via env or default relative folder
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.environ.get("RZENEMBED_ROOT", ""),
            os.path.join(SCRIPT_DIR, "RzenEmbed"),
        ]
        for root in candidates:
            if root and os.path.isdir(root):
                if root not in sys.path:
                    sys.path.insert(0, root)
                try:
                    from rzen_embed_inference import RzenEmbed  # type: ignore
                    break
                except Exception:
                    continue
        else:
            raise RuntimeError(
                "rzen_embed_inference is not available. Set RZENEMBED_ROOT to the folder containing rzen_embed_inference.py, "
                "or `pip install rzen-embed-inference`."
            )

    model_name = os.environ.get("RZEN_MODEL", "qihoo360/RzenEmbed")
    attn_impl = os.environ.get("RZEN_ATTN_IMPL", "sdpa")
    rzen = RzenEmbed(model_name, attn_implementation=attn_impl)

    # Prepare data
    texts = []
    video_paths = []
    all_frames = []
    for entry in tqdm(items, desc="Decoding + sampling frames", dynamic_ncols=True):
        video_file = entry["video"]
        caption = entry["global_caption"]
        vpath = os.path.join(video_base, video_file)
        if not os.path.exists(vpath):
            raise FileNotFoundError(f"Video file not found: {vpath}")
        frames = extract_frames_cv2(vpath, num_frames=num_frames)
        if not frames:
            # skip if we cannot decode
            continue
        texts.append(caption)
        video_paths.append(vpath)
        all_frames.append(frames)

    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32), [], []

    # Compute embeddings with visible progress (chunked)
    batch = int(os.environ.get("RZEN_BATCH", "32") or 32)
    q_chunks, v_chunks = [], []
    for i in tqdm(range(0, len(texts), batch), desc="Encoding text", dynamic_ncols=True):
        q_emb = rzen.get_fused_embeddings(instruction=query_instruction, texts=texts[i : i + batch])
        q_chunks.append(q_emb)
    for i in tqdm(range(0, len(all_frames), batch), desc="Encoding video", dynamic_ncols=True):
        v_emb = rzen.get_fused_embeddings(instruction=candidate_instruction, images=all_frames[i : i + batch])
        v_chunks.append(v_emb)
    # Concatenate chunks
    import numpy as _np
    def _to_np(x):
        # Convert torch tensors (incl. bfloat16) to float32 numpy arrays safely
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32).cpu().numpy()
        arr = _np.asarray(x)
        # Some NumPy builds may not have dtype("bfloat16"); use name-based check.
        dt_name = getattr(arr.dtype, "name", str(arr.dtype))
        if dt_name in ("bfloat16", "float16"):
            arr = arr.astype(_np.float32)
        return arr
    query_embeds = _to_np(_np.concatenate([_to_np(x) for x in q_chunks], axis=0))
    candidate_embeds = _to_np(_np.concatenate([_to_np(x) for x in v_chunks], axis=0))

    # Convert to numpy
    # Normalize to be safe
    def _l2n(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return x / n

    Q = _l2n(np.asarray(query_embeds))
    V = _l2n(np.asarray(candidate_embeds))
    sim_matrix = (Q @ V.T).astype(np.float32)

    # Build metadata lists aligned to rows/cols
    video_data = []
    text_data = []
    for vpath, caption in zip(video_paths, texts):
        vid = os.path.splitext(os.path.basename(vpath))[0]
        video_data.append({"video_id": vid, "video_path": vpath, "split": "train"})
        text_data.append({"video_id": vid, "caption": caption, "split": "train"})

    return sim_matrix, video_data, text_data


def save_similarity_pickle(output_dir: str, similarity_matrix, video_data, text_data):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "similarity_matrix": similarity_matrix,
        "video_data": video_data,
        "text_data": text_data,
        "video_text_mapping": list(range(len(video_data))),
    }
    with open(os.path.join(output_dir, "similarity_matrix.pkl"), "wb") as f:
        pickle.dump(payload, f)


def _build_sample(
    selected_indices: List[int],
    text_idx: int,
    text_data: List[dict],
    train_indices: List[int],
    video_data: List[dict],
    similarities_row: np.ndarray,
) -> dict:
    videos = []
    for local_idx in selected_indices:
        video_global_idx = train_indices[local_idx]
        video_info = video_data[video_global_idx]
        videos.append(
            {
                "video_id": video_info["video_id"],
                "video_path": video_info["video_path"],
                "similarity_score": float(similarities_row[local_idx]),
                "is_correct": video_info["video_id"] == text_data[text_idx]["video_id"],
            }
        )
    return {
        "text_id": text_data[text_idx]["video_id"],
        "caption": text_data[text_idx]["caption"],
        "videos": videos,
    }


def create_easy_dataset(sim_matrix, video_data, text_data, train_indices):
    data = {
        "metadata": {
            "mode": "easy",
            "description": "정답 1개 + 랜덤 4개",
            "videos_per_text": 5,
            "total_texts": len(train_indices),
        },
        "training_samples": [],
    }
    for local_idx, text_idx in enumerate(train_indices):
        similarities_row = sim_matrix[local_idx]
        other_indices = [i for i in range(len(train_indices)) if i != local_idx]
        sampled = random.sample(other_indices, k=min(4, len(other_indices))) if other_indices else []
        selected = [local_idx] + sampled
        random.shuffle(selected)
        selected = selected[:5]
        data["training_samples"].append(
            _build_sample(selected, text_idx, text_data, train_indices, video_data, similarities_row)
        )
    return data


def create_medium_dataset(sim_matrix, video_data, text_data, train_indices):
    data = {
        "metadata": {
            "mode": "medium",
            "description": "정답 + 상위/중간/하위 혼합",
            "videos_per_text": 5,
            "total_texts": len(train_indices),
        },
        "training_samples": [],
    }
    total = len(train_indices)
    for local_idx, text_idx in enumerate(train_indices):
        similarities_row = sim_matrix[local_idx]
        sorted_idx = np.argsort(similarities_row)[::-1]
        correct_idx = local_idx
        selected = [correct_idx]
        top_candidates = [idx for idx in sorted_idx[: min(20, total)] if idx != correct_idx]
        selected.extend(top_candidates[:2])
        mid_start = total // 3
        mid_end = 2 * total // 3
        mid_candidates = [idx for idx in sorted_idx[mid_start:mid_end] if idx not in selected]
        if mid_candidates:
            selected.append(random.choice(mid_candidates))
        low_candidates = [idx for idx in sorted_idx[-max(50, total // 5):] if idx not in selected]
        if low_candidates:
            selected.append(random.choice(low_candidates))
        while len(selected) < 5:
            remaining = [idx for idx in range(total) if idx not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))
        random.shuffle(selected)
        data["training_samples"].append(
            _build_sample(selected[:5], text_idx, text_data, train_indices, video_data, similarities_row)
        )
    return data


def create_hard_dataset(sim_matrix, video_data, text_data, train_indices):
    data = {
        "metadata": {
            "mode": "hard",
            "description": "상위 후보에서만 샘플링",
            "videos_per_text": 5,
            "total_texts": len(train_indices),
        },
        "training_samples": [],
    }
    total = len(train_indices)
    top_k = min(20, total)
    for local_idx, text_idx in enumerate(train_indices):
        similarities_row = sim_matrix[local_idx]
        sorted_idx = np.argsort(similarities_row)[::-1][:top_k]
        correct_idx = local_idx
        if correct_idx in sorted_idx:
            others = [idx for idx in sorted_idx if idx != correct_idx]
            k = min(4, len(others))
            selected = [correct_idx] + random.sample(others, k=k)
        else:
            selected = [correct_idx] + list(sorted_idx[: min(4, len(sorted_idx))])
        while len(selected) < 5 and len(selected) < len(sorted_idx):
            remaining = [idx for idx in sorted_idx if idx not in selected]
            if not remaining:
                break
            selected.append(remaining[0])
        random.shuffle(selected)
        data["training_samples"].append(
            _build_sample(selected[:5], text_idx, text_data, train_indices, video_data, similarities_row)
        )
    return data


def main():
    ap = argparse.ArgumentParser(description="Generate reranking datasets using RzenEmbed")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--video_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--query_instruction",
        default="Find the video snippet that corresponds to the given caption:",
    )
    ap.add_argument(
        "--candidate_instruction",
        default="Understand the content of the provided video.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    items = load_items(args.input_json)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    sim_matrix, video_data, text_data = compute_embeddings(
        items,
        args.video_base,
        args.num_frames,
        args.query_instruction,
        args.candidate_instruction,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_similarity_pickle(args.output_dir, sim_matrix, video_data, text_data)

    train_indices = list(range(len(text_data)))
    easy = create_easy_dataset(sim_matrix, video_data, text_data, train_indices)
    medium = create_medium_dataset(sim_matrix, video_data, text_data, train_indices)
    hard = create_hard_dataset(sim_matrix, video_data, text_data, train_indices)

    def _save_json(obj, name):
        with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    _save_json(easy, "reranking_train_easy.json")
    _save_json(medium, "reranking_train_medium.json")
    _save_json(hard, "reranking_train_hard.json")

    print("All reranking datasets generated at:", args.output_dir)


if __name__ == "__main__":
    main()
