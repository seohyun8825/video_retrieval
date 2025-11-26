#!/usr/bin/env python3
"""
Generate reranking datasets (easy/medium/hard) using VLM2Vec similarities.

Input JSON format (same as d1 global caption output):
  [ {"video": "relative/path.mp4", "global_caption": "..."}, ... ]

Outputs (same as Video-ColBERT pipeline):
  - similarity_matrix.pkl
  - reranking_train_easy.json
  - reranking_train_medium.json
  - reranking_train_hard.json
"""

from __future__ import annotations

# Avoid forcing FlashAttention2 when not installed (force override)
import os as _os
_os.environ["USE_FLASH_ATTENTION_2"] = "0"
_os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
_os.environ["FLASH_ATTENTION_2"] = "0"
_os.environ["TRANSFORMERS_FLASH_ATTENTION_2_ENABLED"] = "0"
_os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
_os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
_os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

import argparse
import json
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# VLM2Vec repo path: prefer env VLM2VEC_ROOT, else probe common local paths
ENV_ROOT = os.environ.get("VLM2VEC_ROOT", "")
candidate_roots = [
    ENV_ROOT,
    os.path.join(SCRIPT_DIR, "VLM2Vec"),
    os.path.join(SCRIPT_DIR, "VLM2Vec", "VLM2Vec-V2.0"),
]
VLM2VEC_ROOT = None
for root in candidate_roots:
    if root and os.path.isdir(os.path.join(root, "src")):
        VLM2VEC_ROOT = root
        break
if VLM2VEC_ROOT is None:
    # last resort: if env root set, still append it
    VLM2VEC_ROOT = ENV_ROOT or os.path.join(SCRIPT_DIR, "VLM2Vec")
if VLM2VEC_ROOT not in os.sys.path:
    os.sys.path.insert(0, VLM2VEC_ROOT)
print(f"[vlm2vec] using repo root: {VLM2VEC_ROOT}")

# Imports from VLM2Vec
from src.arguments import ModelArguments, DataArguments  # type: ignore
from src.model.model import MMEBModel  # type: ignore
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS  # type: ignore
from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info  # type: ignore


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


def build_vlm2vec(device: torch.device):
    """Load VLM2Vec model/processor according to user's preferences."""
    model_args = ModelArguments(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        checkpoint_path="TIGER-Lab/VLM2Vec-Qwen2VL-7B",
        pooling="last",
        normalize=True,
        model_backbone="qwen2_vl",
        lora=True,
    )
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)
    # Prefer bf16 for speed & stability on Ampere+
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device, dtype=dtype)
    model.eval()
    return model, processor


@torch.inference_mode()
def compute_reps(
    model: MMEBModel,
    processor,
    items: List[dict],
    video_base: str,
    fps: float,
    max_pixels: int,
    device: torch.device,
    nframes: int,
    loader_workers: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[dict], List[dict]]:
    """Return (text_reps, video_reps, video_data, text_data)."""
    text_reps: List[torch.Tensor] = []
    video_reps: List[torch.Tensor] = []
    video_data: List[dict] = []
    text_data: List[dict] = []

    from concurrent.futures import ThreadPoolExecutor

    def _prepare(entry: dict):
        video_file = entry["video"]
        caption = entry["global_caption"]
        video_path = os.path.join(video_base, video_file)
        if not os.path.exists(video_path):
            return None
        video_ele = {"type": "video", "video": video_path, "max_pixels": int(max_pixels)}
        use_n = int(nframes or 0) > 0
        if use_n:
            video_ele["nframes"] = int(nframes)
        else:
            video_ele["fps"] = float(fps)
        messages = [{
            "role": "user",
            "content": [video_ele, {"type": "text", "text": "Describe this video."}],
        }]
        try:
            _, video_inputs = process_vision_info(messages)
        except Exception:
            if use_n:
                video_ele_fb = {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": int(max_pixels),
                    "fps": float(fps),
                    "min_frames": int(nframes),
                    "max_frames": int(nframes),
                }
                messages = [{"role": "user", "content": [video_ele_fb, {"type": "text", "text": "Describe this video."}]}]
                _, video_inputs = process_vision_info(messages)
            else:
                return None
        v_inputs = processor(
            text=f"{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.",
            videos=video_inputs,
            return_tensors="pt",
        )
        if "pixel_values_videos" in v_inputs:
            v_inputs["pixel_values_videos"] = v_inputs["pixel_values_videos"].unsqueeze(0)
        if "video_grid_thw" in v_inputs:
            v_inputs["video_grid_thw"] = v_inputs["video_grid_thw"].unsqueeze(0)
        t_inputs = processor(text=caption, images=None, return_tensors="pt")
        return video_file, caption, video_path, v_inputs, t_inputs

    # Prepare inputs in parallel (CPU-bound)
    prepared = []
    if loader_workers and loader_workers > 1:
        with ThreadPoolExecutor(max_workers=loader_workers) as ex:
            for res in tqdm(ex.map(_prepare, items), total=len(items), desc="Preparing inputs"):
                if res is not None:
                    prepared.append(res)
    else:
        for it in tqdm(items, desc="Preparing inputs"):
            res = _prepare(it)
            if res is not None:
                prepared.append(res)

    # Encode sequentially on GPU (safe)
    for video_file, caption, video_path, v_inputs, t_inputs in tqdm(prepared, desc="Encoding reps"):
        v_inputs = {k: v.to(device) for k, v in v_inputs.items()}
        t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
        v_rep = model(qry=v_inputs)["qry_reps"].detach().float().cpu().squeeze(0)
        video_reps.append(v_rep)
        t_rep = model(tgt=t_inputs)["tgt_reps"].detach().float().cpu().squeeze(0)
        text_reps.append(t_rep)

        video_id = os.path.splitext(os.path.basename(video_file))[0]
        video_data.append({"video_id": video_id, "video_path": video_path, "split": "train"})
        text_data.append({"video_id": video_id, "caption": caption, "split": "train"})

    return text_reps, video_reps, video_data, text_data


def compute_similarity_matrix(text_reps: List[torch.Tensor], video_reps: List[torch.Tensor]) -> np.ndarray:
    """Compute NxN similarity where rows=texts, cols=videos."""
    if len(text_reps) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    T = torch.stack(text_reps)  # [N, D]
    V = torch.stack(video_reps)  # [N, D]
    # If reps are normalized, dot = cosine. Otherwise normalize here.
    # Normalize to be safe.
    T = torch.nn.functional.normalize(T, dim=-1)
    V = torch.nn.functional.normalize(V, dim=-1)
    sim = T @ V.T  # [N, N]
    return sim.detach().cpu().numpy().astype(np.float32)


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
    ap = argparse.ArgumentParser(description="Generate reranking datasets using VLM2Vec")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--video_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--nframes", type=int, default=64, help="If >0, request exact nframes per video (falls back when too short)")
    ap.add_argument("--loader_workers", type=int, default=1, help="Parallel CPU workers for input preparation (0/1=off)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = load_items(args.input_json)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = build_vlm2vec(device)
    (
        text_reps,
        video_reps,
        video_data,
        text_data,
    ) = compute_reps(
        model,
        processor,
        items,
        args.video_base,
        args.fps,
        args.max_pixels,
        device,
        args.nframes,
        loader_workers=max(0, int(args.loader_workers or 0)),
    )

    sim_matrix = compute_similarity_matrix(text_reps, video_reps)
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
