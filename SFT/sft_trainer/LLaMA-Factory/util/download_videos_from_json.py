#!/usr/bin/env python3
"""Download ActivityNet videos listed in a JSON manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

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


Variant = Tuple[str, Path, bool]


def build_variants(video: str, dest: Path) -> List[Variant]:
    """Return candidate (remote_rel, local_path, is_primary) tuples."""
    variants: List[Variant] = []
    rel = video.strip()
    primary_local = (dest / rel).resolve()
    variants.append((rel, primary_local, True))
    path_obj = Path(rel)
    if path_obj.suffix.lower() == ".mp4":
        alt_rel = str(path_obj.with_suffix(".mkv"))
        alt_local = (dest / alt_rel).resolve()
        variants.append((alt_rel, alt_local, False))
    return variants


def ensure_primary_alias(primary: Path, actual: Path) -> None:
    """Create a symlink at primary pointing to actual if needed."""
    if primary == actual:
        return
    if primary.exists() or primary.is_symlink():
        return
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(actual, primary)
        print(f"[info] Linked {primary} -> {actual}")
    except OSError as exc:  # pragma: no cover - best effort
        print(f"[warn] Could not create symlink {primary} -> {actual}: {exc}")


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
    download_queue: List[tuple[Entry, List[Variant]]] = []
    for entry in entries:
        video = entry.get("video")
        if not isinstance(video, str):
            continue
        video = video.strip()
        if not video:
            continue
        variants = build_variants(video, dest)
        primary_local = variants[0][1]
        existing_variant: Variant | None = None
        if skip_existing:
            for variant in variants:
                rel_path, local_path, is_primary_variant = variant
                if local_path.exists() or local_path.is_symlink():
                    existing_variant = variant
                    break
        if existing_variant:
            rel_path, local_path, is_primary = existing_variant
            print(f"[skip] {local_path} already exists (variant: {rel_path})")
            if not is_primary:
                ensure_primary_alias(primary_local, local_path)
            successful.append(entry)
            continue

        if dry_run:
            for rel_path, local_path, _ in variants:
                gcs_path = f"{prefix}/{rel_path}"
                print(
                    "Executing:",
                    " ".join(
                        [
                            "gcloud",
                            "storage",
                            "cp",
                            gcs_path,
                            str(local_path),
                        ]
                    ),
                )
            continue

        download_queue.append((entry, variants))

    if dry_run:
        return successful

    def download_task(entry: Entry, variants: List[Variant]) -> bool:
        primary_local = variants[0][1]
        for rel_path, local_path, is_primary in variants:
            gcs_path = f"{prefix}/{rel_path}"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            cmd: List[str] = [
                "gcloud",
                "storage",
                "cp",
                gcs_path,
                str(local_path),
            ]
            print("Executing:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                if not is_primary:
                    ensure_primary_alias(primary_local, local_path)
                return True
            except subprocess.CalledProcessError as exc:
                print(f"[warn] Failed to download {gcs_path}: {exc}")
        return False

    if workers == 1:
        for entry, variants in download_queue:
            if download_task(entry, variants):
                successful.append(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_task, entry, variants) for entry, variants in download_queue]
            for (entry, _), future in zip(download_queue, futures):
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
