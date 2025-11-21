#!/usr/bin/env python3
"""
Analyze similarity_matrix.pkl to compute retrieval metrics and simple visualizations.

Example:
  python analyze_similarity_matrix.py \
      --similarity_pkl outputs/toy_reranking/similarity_matrix.pkl \
      --output_dir outputs/toy_reranking/analysis \
      --topks 1 5 10 \
      --heatmap_size 20
"""

import argparse
import json
import os
import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def compute_rank_metrics(sim_matrix: np.ndarray, topks: List[int]):
    """Compute rank of the correct video (diagonal) for each text and recall@k."""
    total = sim_matrix.shape[0]
    ranks = np.empty(total, dtype=np.int32)

    for i in range(total):
        row = sim_matrix[i]
        sorted_idx = np.argsort(row)[::-1]
        rank = int(np.where(sorted_idx == i)[0][0])
        ranks[i] = rank

    metrics = {}
    for k in topks:
        recall = float(np.mean(ranks < k))
        metrics[f"recall@{k}"] = recall

    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["median_rank"] = float(np.median(ranks))
    metrics["max_rank"] = int(np.max(ranks))

    return ranks, metrics


def plot_rank_histogram(ranks: np.ndarray, output_dir: str):
    plt.figure(figsize=(8, 4))
    plt.hist(ranks, bins=50, color="steelblue", edgecolor="black")
    plt.xlabel("Rank of Correct Video")
    plt.ylabel("Count")
    plt.title("Distribution of Correct Video Ranks")
    plt.tight_layout()
    path = os.path.join(output_dir, "rank_histogram.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_heatmap(sim_matrix: np.ndarray, output_dir: str, sample_size: int = 20, seed: int = 42):
    total = sim_matrix.shape[0]
    size = min(sample_size, total)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    sample_idx = sorted(indices[:size])

    subset = sim_matrix[np.ix_(sample_idx, sample_idx)]

    plt.figure(figsize=(6, 5))
    plt.imshow(subset, cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.title(f"Similarity Heatmap (subset size={size})")
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_heatmap.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Analyze similarity_matrix.pkl results.")
    parser.add_argument("--similarity_pkl", required=True, help="Path to similarity_matrix.pkl")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots/results")
    parser.add_argument("--topks", type=int, nargs="*", default=[1, 5, 10], help="Top-k values for recall")
    parser.add_argument("--heatmap_size", type=int, default=20, help="Subset size for heatmap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.similarity_pkl, "rb") as f:
        payload = pickle.load(f)
    sim_matrix = payload["similarity_matrix"]

    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")

    ranks, metrics = compute_rank_metrics(sim_matrix, args.topks)

    print("=== Retrieval Metrics ===")
    for k, value in metrics.items():
        print(f"{k}: {value:.4f}" if isinstance(value, float) else f"{k}: {value}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        hist_path = plot_rank_histogram(ranks, args.output_dir)
        heatmap_path = plot_heatmap(sim_matrix, args.output_dir, args.heatmap_size, args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "histogram": hist_path, "heatmap": heatmap_path}, f, indent=2)

        ranks_path = os.path.join(args.output_dir, "ranks.npy")
        np.save(ranks_path, ranks)

        print(f"\nSaved artifacts to {args.output_dir}:")
        print(f"- {metrics_path}")
        print(f"- {hist_path}")
        print(f"- {heatmap_path}")
        print(f"- {ranks_path}")


if __name__ == "__main__":
    main()
