#!/usr/bin/env python3
"""
Generate reranking datasets (easy/medium/hard) from a simple video-caption JSON
using Video-ColBERT similarities.
"""

import argparse
import json
import os
import pickle
import random
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_COLBERT_ROOT = os.path.join(SCRIPT_DIR, "Video-ColBERT")
VIDEO_COLBERT_ROOT = os.environ.get("VIDEO_COLBERT_ROOT", DEFAULT_VIDEO_COLBERT_ROOT)
if VIDEO_COLBERT_ROOT and VIDEO_COLBERT_ROOT not in os.sys.path:
    os.sys.path.insert(0, VIDEO_COLBERT_ROOT)

from video_colbert import VideoColBERT, load_video_frames


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


def compute_features(
    model: VideoColBERT,
    items: List[dict],
    video_base: str,
    num_frames: int,
    device: torch.device,
    frame_size: int,
    batch_size: int = 8,
    decode_threads: int | None = None,
    loader_workers: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[dict], List[dict]]:
    """Return (query_features_list, frame_features_list, video_features_list, video_data, text_data).

    Batches videos to better utilize GPU and reduce Python overhead.
    """
    model.eval()

    query_features_list: List[torch.Tensor] = []
    frame_features_list: List[torch.Tensor] = []
    video_features_list: List[torch.Tensor] = []
    video_data: List[dict] = []
    text_data: List[dict] = []

    # Optionally control decord threads globally
    if decode_threads is not None:
        os.environ["DECORD_THREADS"] = str(int(decode_threads))

    def _load_one(entry: dict) -> Tuple[str, str, str, Optional[torch.Tensor]]:
        video_file = entry["video"]
        caption = entry["global_caption"]
        video_path = os.path.join(video_base, video_file)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        try:
            frames = load_video_frames(video_path, num_frames=num_frames, frame_size=frame_size)
            return video_file, caption, video_path, frames
        except Exception as e:
            print(f"[warn] Skipping video due to frame load failure: {video_path} -> {e}")
            return video_file, caption, video_path, None

    for start in tqdm(range(0, len(items), batch_size), desc="Computing embeddings"):
        batch = items[start : start + batch_size]

        # Load frames: optionally in parallel across multiple videos
        if loader_workers and loader_workers > 1:
            with ThreadPoolExecutor(max_workers=loader_workers) as ex:
                loaded = list(ex.map(_load_one, batch))
        else:
            loaded = [_load_one(entry) for entry in batch]

        # Filter out failures
        ok_indices: List[int] = [i for i, (_, _, _, fr) in enumerate(loaded) if fr is not None]
        if not ok_indices:
            continue

        captions: List[str] = [loaded[i][1] for i in ok_indices]
        video_files: List[str] = [loaded[i][0] for i in ok_indices]
        video_paths: List[str] = [loaded[i][2] for i in ok_indices]
        frames_batch: torch.Tensor = torch.stack([loaded[i][3] for i in ok_indices])  # type: ignore[arg-type]
        frames_batch = frames_batch.to(device, non_blocking=True)

        with torch.no_grad():
            # Batch text encode
            q_feats_b = model.encode_query(captions).detach().cpu()  # [B, Q, D]
            # Batch video encode
            f_feats_b, v_feats_b = model.encode_video(frames_batch)
            f_feats_b = f_feats_b.detach().cpu()
            v_feats_b = v_feats_b.detach().cpu()

        # Split and append per-item
        for i in range(len(ok_indices)):
            query_features_list.append(q_feats_b[i])
            frame_features_list.append(f_feats_b[i])
            video_features_list.append(v_feats_b[i])

            video_id = os.path.splitext(os.path.basename(video_files[i]))[0]
            video_data.append({"video_id": video_id, "video_path": video_paths[i], "split": "train"})
            text_data.append({"video_id": video_id, "caption": captions[i], "split": "train"})

    return (
        query_features_list,
        frame_features_list,
        video_features_list,
        video_data,
        text_data,
    )


def compute_similarity_matrix(
    query_features_list,
    frame_features_list,
    video_features_list,
    model: VideoColBERT,
    device: torch.device,
    chunk_size: int = 16,
):
    model.eval()
    total = len(query_features_list)
    sim_matrix = np.zeros((total, total), dtype=np.float32)

    for i, query_feat in enumerate(tqdm(query_features_list, desc="Building similarity matrix")):
        scores = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            frame_chunk = torch.stack(frame_features_list[start:end]).to(device)
            video_chunk = torch.stack(video_features_list[start:end]).to(device)

            q = query_feat.unsqueeze(0).to(device)
            q = q.expand(frame_chunk.size(0), -1, -1)
            with torch.no_grad():
                _, frame_sim, video_sim = model.compute_similarity(q, frame_chunk, video_chunk)
                chunk_scores = (frame_sim + video_sim).cpu().tolist()
            scores.extend(chunk_scores)

            del frame_chunk, video_chunk, q, frame_sim, video_sim
            torch.cuda.empty_cache()

        sim_matrix[i, : len(scores)] = scores

    return sim_matrix


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
        other_indices = [
            i for i in range(len(train_indices)) if i != local_idx
        ]
        if other_indices:
            k = min(4, len(other_indices))
            sampled = random.sample(other_indices, k=k)
        else:
            sampled = []
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


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate reranking datasets using Video-ColBERT.")
    parser.add_argument("--input_json", required=True, help="Path to toy JSON file.")
    parser.add_argument("--video_base", required=True, help="Base directory that stores video files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--num_frames", type=int, default=12, help="Frames to sample per video.")
    parser.add_argument("--frame_size", type=int, default=224, help="Frame resolution for CLIP input (default: 224).")
    parser.add_argument("--chunk_size", type=int, default=16, help="Batch size when building similarity matrix (default: 16).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction (videos per forward).")
    parser.add_argument("--decode_threads", type=int, default=0, help="Override decord decode threads (0=keep default).")
    parser.add_argument("--loader_workers", type=int, default=0, help="Parallel video loaders per batch (0/1=off).")
    parser.add_argument("--limit", type=int, default=0, help="If >0, limit number of items processed (for speed/debug).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = load_items(args.input_json)
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoColBERT(backbone_name="ViT-B/32", device=str(device))
    print(f"[video-colbert] device={device} amp={getattr(model, 'use_amp', None)} "
          f"batch_size={args.batch_size} chunk_size={args.chunk_size} "
          f"num_frames={args.num_frames} frame_size={args.frame_size} "
          f"decode_threads={args.decode_threads}")
    (
        query_features_list,
        frame_features_list,
        video_features_list,
        video_data,
        text_data,
    ) = compute_features(
        model,
        items,
        args.video_base,
        args.num_frames,
        device,
        args.frame_size,
        batch_size=max(1, int(args.batch_size)),
        decode_threads=(None if int(args.decode_threads or 0) <= 0 else int(args.decode_threads)),
        loader_workers=max(0, int(args.loader_workers or 0)),
    )
    sim_matrix = compute_similarity_matrix(
        query_features_list, frame_features_list, video_features_list, model, device, chunk_size=args.chunk_size
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_similarity_pickle(args.output_dir, sim_matrix, video_data, text_data)

    train_indices = list(range(len(text_data)))

    easy = create_easy_dataset(sim_matrix, video_data, text_data, train_indices)
    medium = create_medium_dataset(sim_matrix, video_data, text_data, train_indices)
    hard = create_hard_dataset(sim_matrix, video_data, text_data, train_indices)

    save_json(easy, os.path.join(args.output_dir, "reranking_train_easy.json"))
    save_json(medium, os.path.join(args.output_dir, "reranking_train_medium.json"))
    save_json(hard, os.path.join(args.output_dir, "reranking_train_hard.json"))

    print("All reranking datasets generated at:", args.output_dir)


if __name__ == "__main__":
    main()
