#!/usr/bin/env python3
"""Download ActivityNet videos listed in a JSON manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

Entry = Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download videos referenced in a JSON file using gcloud storage cp",
    )
    parser.add_argument(
        "json_path",
        help="Path to the JSON file containing ActivityNet samples (list of dicts with a 'video' key)",
    )
    parser.add_argument(
        "destination",
        help="Local directory where downloaded videos will be stored",
    )
    parser.add_argument(
        "--gcs-prefix",
        default="gs://mlv_doc_under/activitynet/videos",
        help="GCS prefix where video files are stored",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip videos that already exist at the destination (default: enabled)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Download even if the target file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gcloud commands without executing them",
    )
    parser.add_argument(
        "--existing-json",
        help="Optional path to write a JSON file containing only the entries that were successfully downloaded or already present",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel downloads to run (default: 4, set to 1 for sequential)",
    )
    parser.set_defaults(skip_existing=True)
    return parser


def read_entries(json_path: Path) -> List[Entry]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected {json_path} to contain a JSON list")
    return data


def unique_entries(entries: Sequence[Entry]) -> List[Entry]:
    unique: List[Entry] = []
    seen: Set[str] = set()
    for entry in entries:
        video = entry.get("video")
        if not isinstance(video, str):
            continue
        video = video.strip()
        if not video:
            continue
        if video in seen:
            continue
        seen.add(video)
        unique.append(entry)
    return unique


def run_downloads(
    entries: Sequence[Entry],
    prefix: str,
    dest: Path,
    skip_existing: bool,
    dry_run: bool,
    workers: int,
) -> List[Entry]:
    dest.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/")
    successful: List[Entry] = []
    workers = max(1, workers)
    download_queue: List[tuple[Entry, List[str], str]] = []
    for entry in entries:
        video = entry.get("video")
        if not isinstance(video, str):
            continue
        video = video.strip()
        if not video:
            continue
        gcs_path = f"{prefix}/{video}"
        local_path = dest / video
        if skip_existing and local_path.exists():
            print(f"[skip] {local_path} already exists")
            successful.append(entry)
            continue
        cmd: List[str] = [
            "gcloud",
            "storage",
            "cp",
            gcs_path,
            str(dest),
        ]
        print("Executing:", " ".join(cmd))
        if dry_run:
            continue
        download_queue.append((entry, cmd, gcs_path))

    if dry_run:
        return successful

    def download_task(entry: Entry, cmd: List[str], gcs_path: str) -> bool:
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[warn] Failed to download {gcs_path}: {exc}")
            return False

    if workers == 1:
        for entry, cmd, gcs_path in download_queue:
            if download_task(entry, cmd, gcs_path):
                successful.append(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(download_task, entry, cmd, gcs_path)
                for entry, cmd, gcs_path in download_queue
            ]
            for (entry, _, _), future in zip(download_queue, futures):
                if future.result():
                    successful.append(entry)

    return successful


def write_existing_entries(entries: Sequence[Entry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(entries)} entries to {output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    json_path = Path(args.json_path).expanduser().resolve()
    dest = Path(args.destination).expanduser().resolve()

    entries = unique_entries(read_entries(json_path))
    if not entries:
        raise SystemExit(f"No video entries found in {json_path}")

    successful = run_downloads(
        entries=entries,
        prefix=args.gcs_prefix,
        dest=dest,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        workers=args.workers,
    )

    if args.existing_json:
        output_path = Path(args.existing_json).expanduser().resolve()
        write_existing_entries(successful, output_path)


if __name__ == "__main__":
    main()
