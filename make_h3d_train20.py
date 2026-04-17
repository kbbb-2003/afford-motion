#!/usr/bin/env python3
"""Create and optionally activate a subset split for H3D/HumanML3D."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find split file: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_ids(path: Path, ids: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {path}. Use --force if you want to overwrite it."
        )
    path.write_text("\n".join(ids) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixed-random subset split from data/H3D/train.txt or "
            "data/H3D/train_full.txt."
        )
    )
    parser.add_argument("--h3d-dir", default="data/H3D", help="Path to the H3D directory.")
    parser.add_argument(
        "--source-name",
        default="train_full.txt",
        help="Preferred source split filename. Falls back to train.txt if missing.",
    )
    parser.add_argument(
        "--backup-name",
        default="train_full.txt",
        help="Backup filename to create from train.txt when needed.",
    )
    parser.add_argument(
        "--output-name",
        default="train_20.txt",
        help="Output subset filename.",
    )
    parser.add_argument("--ratio", type=float, default=0.2, help="Subset ratio, e.g. 0.2 for 20%%.")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for subset sampling.")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Also copy the generated subset to train.txt after creation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output or backup files if needed.",
    )
    args = parser.parse_args()

    if not (0.0 < args.ratio <= 1.0):
        raise ValueError(f"--ratio must be in (0, 1], but got {args.ratio}.")

    h3d_dir = Path(args.h3d_dir)
    train_path = h3d_dir / "train.txt"
    backup_path = h3d_dir / args.backup_name
    source_path = h3d_dir / args.source_name
    output_path = h3d_dir / args.output_name

    if not h3d_dir.exists():
        raise FileNotFoundError(f"H3D directory does not exist: {h3d_dir}")
    if not train_path.exists():
        raise FileNotFoundError(f"Current train split does not exist: {train_path}")

    if not backup_path.exists():
        if backup_path.exists() and not args.force:
            raise FileExistsError(f"Backup already exists: {backup_path}")
        shutil.copy2(train_path, backup_path)
        print(f"[backup] Created backup split: {backup_path}")

    if not source_path.exists():
        source_path = train_path
        print(f"[info] Preferred source split not found. Falling back to: {source_path}")

    ids = read_ids(source_path)
    if not ids:
        raise RuntimeError(f"No ids found in source split: {source_path}")

    random.seed(args.seed)
    shuffled_ids = ids[:]
    random.shuffle(shuffled_ids)

    subset_size = max(1, int(len(shuffled_ids) * args.ratio))
    subset_ids = shuffled_ids[:subset_size]

    write_ids(output_path, subset_ids, force=args.force)
    print(f"[done] Source split: {source_path}")
    print(f"[done] Total ids: {len(ids)}")
    print(f"[done] Subset ids: {len(subset_ids)}")
    print(f"[done] Saved subset split to: {output_path}")

    if args.activate:
        shutil.copy2(output_path, train_path)
        print(f"[done] Activated subset split as current train split: {train_path}")


if __name__ == "__main__":
    main()
