#!/usr/bin/env python3
"""
Evaluate text-to-video retrieval on ActivityNet-style JSONL using RzenEmbed.
"""
from __future__ import annotations

import os as _os

# Force-disable FlashAttention2 to avoid import issues
_os.environ["USE_FLASH_ATTENTION_2"] = "0"
_os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
_os.environ["FLASH_ATTENTION_2"] = "0"
_os.environ["TRANSFORMERS_FLASH_ATTENTION_2_ENABLED"] = "0"
_os.environ["TRANSFORMERS_NO_FLASH_ATTENTION"] = "1"
_os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
_os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_video_id(raw: str) -> str:
    base = os.path.basename(str(raw))
    low = base.lower()
    if low.endswith(VIDEO_EXTS):
        return os.path.splitext(base)[0]
    return base


def resolve_video_path(video_base: str, raw: str) -> Tuple[str, str, str]:
    raw = str(raw)
    base = os.path.basename(raw)
    low = base.lower()
    if low.endswith(VIDEO_EXTS):
        rel = raw
    else:
        rel = raw + ".mp4"
    if os.path.isabs(rel):
        path = rel
    else:
        path = os.path.join(video_base, rel)
    vid = os.path.splitext(os.path.basename(rel))[0]
    return vid, path, rel


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
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def get_rzen_embed():
    try:
        from rzen_embed_inference import RzenEmbed  # type: ignore
        return RzenEmbed
    except Exception:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        candidates = [
            os.environ.get("RZENEMBED_ROOT", ""),
            os.path.join(
                root_dir,
                "SFT",
                "sft_data_generation",
                "preprocess",
                "extract_similarity_matrix",
                "RzenEmbed",
            ),
        ]
        for root in candidates:
            if root and os.path.isdir(root):
                if root not in sys.path:
                    sys.path.insert(0, root)
                try:
                    from rzen_embed_inference import RzenEmbed  # type: ignore
                    return RzenEmbed
                except Exception:
                    continue
        raise RuntimeError(
            "rzen_embed_inference is not available. Set RZENEMBED_ROOT or install rzen-embed-inference."
        )


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32).cpu().numpy()
    arr = np.asarray(x)
    dt_name = getattr(arr.dtype, "name", str(arr.dtype))
    if dt_name in ("bfloat16", "float16"):
        arr = arr.astype(np.float32)
    return arr


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


def encode_texts(
    rzen,
    texts: List[str],
    instruction: str,
    batch_size: int,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text", dynamic_ncols=True):
        emb = rzen.get_fused_embeddings(instruction=instruction, texts=texts[i : i + batch_size])
        chunks.append(_to_np(emb))
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.concatenate(chunks, axis=0)
    return _l2n(arr.astype(np.float32))


def encode_videos(
    rzen,
    video_items: List[dict],
    num_frames: int,
    instruction: str,
    batch_size: int,
) -> Tuple[np.ndarray, List[dict], List[dict]]:
    frames_batch = []
    meta_batch: List[dict] = []
    chunks: List[np.ndarray] = []
    kept_meta: List[dict] = []
    skipped: List[dict] = []

    for item in tqdm(video_items, desc="Decoding videos", dynamic_ncols=True):
        vpath = item["video_path"]
        if not os.path.exists(vpath):
            skipped.append({"video_id": item["video_id"], "video_path": vpath, "reason": "missing"})
            continue
        frames = extract_frames_cv2(vpath, num_frames=num_frames)
        if not frames:
            skipped.append({"video_id": item["video_id"], "video_path": vpath, "reason": "decode_failed"})
            continue
        frames_batch.append(frames)
        meta_batch.append(item)
        if len(frames_batch) >= batch_size:
            emb = rzen.get_fused_embeddings(instruction=instruction, images=frames_batch)
            chunks.append(_to_np(emb))
            kept_meta.extend(meta_batch)
            frames_batch = []
            meta_batch = []

    if frames_batch:
        emb = rzen.get_fused_embeddings(instruction=instruction, images=frames_batch)
        chunks.append(_to_np(emb))
        kept_meta.extend(meta_batch)

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32), [], skipped
    arr = np.concatenate(chunks, axis=0)
    return _l2n(arr.astype(np.float32)), kept_meta, skipped


def build_video_candidates(
    items: Iterable[dict],
    video_field: str,
    video_base: str,
    limit_videos: int,
) -> List[dict]:
    seen = set()
    candidates: List[dict] = []
    for item in items:
        raw = item.get(video_field)
        if raw is None:
            continue
        vid, path, rel = resolve_video_path(video_base, raw)
        if vid in seen:
            continue
        seen.add(vid)
        candidates.append({"video_id": vid, "video_path": path, "video_rel": rel})
        if limit_videos > 0 and len(candidates) >= limit_videos:
            break
    return candidates


def build_queries(
    items: Iterable[dict],
    query_field: str,
    video_field: str,
    id_field: str,
    limit: int,
) -> List[dict]:
    queries: List[dict] = []
    for item in items:
        if query_field not in item or video_field not in item:
            continue
        text = str(item[query_field]).strip()
        if not text:
            continue
        vid = normalize_video_id(item[video_field])
        desc_id = str(item.get(id_field, ""))
        entry = {
            "desc_id": desc_id,
            "video_id": vid,
            "query": text,
            "time": item.get("time"),
        }
        queries.append(entry)
        if limit > 0 and len(queries) >= limit:
            break
    return queries


def compute_metrics(ranks: List[int], topks: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not ranks:
        for k in topks:
            metrics[f"recall@{k}"] = 0.0
        metrics["mean_rank"] = 0.0
        metrics["median_rank"] = 0.0
        metrics["max_rank"] = 0.0
        metrics["mrr"] = 0.0
        return metrics

    ranks_arr = np.asarray(ranks, dtype=np.int32)
    for k in topks:
        metrics[f"recall@{k}"] = float(np.mean(ranks_arr < k))
    metrics["mean_rank"] = float(np.mean(ranks_arr))
    metrics["median_rank"] = float(np.median(ranks_arr))
    metrics["max_rank"] = float(np.max(ranks_arr))
    metrics["mrr"] = float(np.mean(1.0 / (ranks_arr + 1)))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Text-to-video retrieval eval with RzenEmbed")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--video_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--query_field", default="fig_desc")
    ap.add_argument("--video_field", default="video")
    ap.add_argument("--id_field", default="desc_id")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit_videos", type=int, default=0)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--topks", type=int, nargs="*", default=[1, 5, 10, 100])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--query_instruction",
        default="Find the video snippet that corresponds to the given caption:",
    )
    ap.add_argument(
        "--candidate_instruction",
        default="Understand the content of the provided video.",
    )
    ap.add_argument("--save_similarity", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = load_jsonl(args.input_jsonl)

    video_candidates = build_video_candidates(
        items,
        args.video_field,
        args.video_base,
        args.limit_videos,
    )
    queries = build_queries(
        items,
        args.query_field,
        args.video_field,
        args.id_field,
        args.limit,
    )

    if not queries:
        raise ValueError("No valid queries found. Check query_field and input_jsonl.")
    if not video_candidates:
        raise ValueError("No candidate videos found. Check video_field and input_jsonl.")

    os.makedirs(args.output_dir, exist_ok=True)

    RzenEmbed = get_rzen_embed()
    model_name = os.environ.get("RZEN_MODEL", "qihoo360/RzenEmbed")
    attn_impl = os.environ.get("RZEN_ATTN_IMPL", "sdpa")
    rzen = RzenEmbed(model_name, attn_implementation=attn_impl)

    query_texts = [q["query"] for q in queries]
    query_embeds = encode_texts(rzen, query_texts, args.query_instruction, args.batch_size)

    video_embeds, video_meta, skipped_videos = encode_videos(
        rzen,
        video_candidates,
        args.num_frames,
        args.candidate_instruction,
        args.batch_size,
    )

    if len(video_meta) == 0:
        raise ValueError("No videos could be encoded. Check video paths.")

    video_id_to_idx = {v["video_id"]: i for i, v in enumerate(video_meta)}

    ranks: List[int] = []
    output_path = os.path.join(args.output_dir, "per_query_rankings.jsonl")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(tqdm(queries, desc="Scoring queries", dynamic_ncols=True)):
            row = query_embeds[i] @ video_embeds.T
            sorted_idx = np.argsort(row)[::-1]
            gt_idx = video_id_to_idx.get(q["video_id"])
            if gt_idx is None:
                gt_rank = None
                gt_score = None
                gt_in_candidates = False
            else:
                gt_rank = int(np.where(sorted_idx == gt_idx)[0][0]) + 1
                gt_score = float(row[gt_idx])
                gt_in_candidates = True
                ranks.append(gt_rank - 1)

            topk = args.topk if args.topk > 0 else len(sorted_idx)
            top_indices = sorted_idx[:topk]
            ranked = [
                {"video_id": video_meta[j]["video_id"], "score": float(row[j])}
                for j in top_indices
            ]

            record = {
                "desc_id": q["desc_id"],
                "video_id": q["video_id"],
                "query": q["query"],
                "time": q.get("time"),
                "gt_in_candidates": gt_in_candidates,
                "gt_rank": gt_rank,
                "gt_score": gt_score,
                "topk": ranked,
            }
            out_f.write(json.dumps(record, ensure_ascii=True) + "\n")

    metrics = compute_metrics(ranks, args.topks)
    summary = {
        "total_queries": len(queries),
        "valid_queries": len(ranks),
        "skipped_gt_missing": len(queries) - len(ranks),
        "candidate_videos": len(video_meta),
        "skipped_videos": len(skipped_videos),
        "metrics": metrics,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    video_index_path = os.path.join(args.output_dir, "video_candidates.json")
    with open(video_index_path, "w", encoding="utf-8") as f:
        json.dump(video_meta, f, ensure_ascii=True, indent=2)

    if skipped_videos:
        skipped_path = os.path.join(args.output_dir, "skipped_videos.jsonl")
        with open(skipped_path, "w", encoding="utf-8") as f:
            for rec in skipped_videos:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    if args.save_similarity:
        sim_path = os.path.join(args.output_dir, "similarity_matrix.npy")
        sim_matrix = query_embeds @ video_embeds.T
        np.save(sim_path, sim_matrix.astype(np.float32))

    print("=== Retrieval Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("\nOutputs:")
    print(f"- {output_path}")
    print(f"- {metrics_path}")
    print(f"- {video_index_path}")
    if skipped_videos:
        print(f"- {skipped_path}")
    if args.save_similarity:
        print(f"- {sim_path}")


if __name__ == "__main__":
    main()
