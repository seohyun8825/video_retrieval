#!/usr/bin/env python3
"""Extract a subset of entries from an ActivityNet-style JSON file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a toy/sample JSON subset")
    parser.add_argument("source_json", help="Path to the input JSON file")
    parser.add_argument("output_json", help="Path where the sampled JSON will be written")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of samples to keep (default: 10)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle entries before selecting the first N (default: keep original order)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when --shuffle is set",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for the output file (default: 2)",
    )
    return parser


def load_entries(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a list of entries")
    return data


def select_entries(entries: List[Any], count: int, shuffle: bool, seed: int | None) -> List[Any]:
    if count <= 0:
        return []
    if shuffle:
        rng = Random(seed)
        entries = entries.copy()
        rng.shuffle(entries)
    return entries[: min(count, len(entries))]


def write_entries(entries: List[Any], path: Path, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=indent)
    print(f"Wrote {len(entries)} entries to {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source = Path(args.source_json).expanduser().resolve()
    target = Path(args.output_json).expanduser().resolve()
    entries = load_entries(source)
    subset = select_entries(entries, args.count, args.shuffle, args.seed)
    write_entries(subset, target, args.indent)


if __name__ == "__main__":
    main()
