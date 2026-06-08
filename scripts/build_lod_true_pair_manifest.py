#!/usr/bin/env python3
"""Build the canonical true-LOD pair manifest.

The split unit is the long/normal exposure odd id.  The short/dark exposure
id is always normal_id + 1, matching the public LOD naming convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_LOD_ROOT = Path("/home/caq/6666_raw/0000_dataset/LOD")
DEFAULT_MANIFEST_NAME = "lod_true_pairs_2118_112_seed42.csv"
EXPECTED_HW = (800, 1200)
EXPECTED_RGB_DTYPE = np.uint8
EXPECTED_RAW_DTYPE = np.uint16
FIELDNAMES = (
    "pair_id",
    "split",
    "normal_id",
    "dark_id",
    "rgb_normal_path",
    "rgb_dark_path",
    "raw_normal_path",
    "raw_dark_path",
    "label_space",
    "height",
    "width",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-root", type=Path, default=DEFAULT_LOD_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=2118)
    parser.add_argument("--val-count", type=int, default=112)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Smoke/debug limit. When set, split counts are derived for the limited subset.",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Only validate paths and ids. Intended for development, not canonical output.",
    )
    return parser.parse_args()


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected numeric filename stem, got {path.name!r}") from exc


def collect_ids(directory: Path, suffix: str) -> set[int]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing required LOD directory: {directory}")
    ids = {numeric_stem(path) for path in directory.glob(f"*{suffix}")}
    if not ids:
        raise ValueError(f"No {suffix} files found in {directory}")
    return ids


def validate_image(path: Path, *, expected_dtype: np.dtype, expected_hw: tuple[int, int], label: str) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV failed to read {label}: {path}")
    if image.shape != (expected_hw[0], expected_hw[1], 3):
        raise ValueError(f"{label} has shape {image.shape}, expected {(expected_hw[0], expected_hw[1], 3)}: {path}")
    if image.dtype != expected_dtype:
        raise ValueError(f"{label} has dtype {image.dtype}, expected {expected_dtype}: {path}")


def split_rows(
    normal_ids: list[int],
    *,
    seed: int,
    train_count: int,
    val_count: int,
    smoke_limited: bool,
) -> dict[int, str]:
    total = len(normal_ids)
    if train_count + val_count != total:
        raise ValueError(f"Split counts {train_count}+{val_count} do not match pair count {total}")
    shuffled = list(normal_ids)
    random.Random(seed).shuffle(shuffled)
    train_ids = set(shuffled[:train_count])
    val_ids = set(shuffled[train_count:])
    if len(train_ids) != train_count or len(val_ids) != val_count or train_ids & val_ids:
        raise AssertionError("Internal split construction error")
    split_name = {normal_id: "00Train" for normal_id in train_ids}
    split_name.update({normal_id: "01Valid" for normal_id in val_ids})
    if not smoke_limited and (train_count, val_count) != (2118, 112):
        raise ValueError("Canonical full manifest must use train_count=2118 and val_count=112")
    return split_name


def derive_smoke_counts(total: int) -> tuple[int, int]:
    if total < 2:
        raise ValueError("--max-pairs smoke mode needs at least 2 pairs")
    val_count = max(1, round(total * 0.05))
    train_count = total - val_count
    return train_count, val_count


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_rows(
    *,
    lod_root: Path,
    normal_ids: Iterable[int],
    split_by_id: dict[int, str],
    validate_pixels: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for normal_id in sorted(normal_ids):
        dark_id = normal_id + 1
        rgb_normal = lod_root / "RGB_normal" / f"{normal_id}.JPG"
        rgb_dark = lod_root / "RGB_Dark" / f"{dark_id}.JPG"
        raw_normal = lod_root / "RAW_normal" / f"{normal_id}.png"
        raw_dark = lod_root / "RAW_Dark" / f"{dark_id}.png"
        for path in (rgb_normal, rgb_dark, raw_normal, raw_dark):
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired LOD file: {path}")
        if validate_pixels:
            validate_image(rgb_normal, expected_dtype=EXPECTED_RGB_DTYPE, expected_hw=EXPECTED_HW, label="RGB_normal")
            validate_image(rgb_dark, expected_dtype=EXPECTED_RGB_DTYPE, expected_hw=EXPECTED_HW, label="RGB_Dark")
            validate_image(raw_normal, expected_dtype=EXPECTED_RAW_DTYPE, expected_hw=EXPECTED_HW, label="RAW_normal")
            validate_image(raw_dark, expected_dtype=EXPECTED_RAW_DTYPE, expected_hw=EXPECTED_HW, label="RAW_Dark")
        rows.append(
            {
                "pair_id": f"lod-{normal_id:04d}-{dark_id:04d}",
                "split": split_by_id[normal_id],
                "normal_id": normal_id,
                "dark_id": dark_id,
                "rgb_normal_path": rel(rgb_normal, lod_root),
                "rgb_dark_path": rel(rgb_dark, lod_root),
                "raw_normal_path": rel(raw_normal, lod_root),
                "raw_dark_path": rel(raw_dark, lod_root),
                "label_space": "inverse_relative",
                "height": EXPECTED_HW[0],
                "width": EXPECTED_HW[1],
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    manifest_path: Path,
    *,
    lod_root: Path,
    rows: list[dict[str, object]],
    seed: int,
    train_count: int,
    val_count: int,
    max_pairs: int | None,
    image_validation: bool,
) -> Path:
    split_counts = Counter(str(row["split"]) for row in rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lod_root": str(lod_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "pair_count": len(rows),
        "split_unit": "pair_id=normal odd id",
        "split_policy": "python_random_shuffle_seed42_exact_counts",
        "seed": seed,
        "train_count": train_count,
        "val_count": val_count,
        "split_counts": dict(sorted(split_counts.items())),
        "split_names": ["00Train", "01Valid"],
        "max_pairs": max_pairs,
        "is_smoke": max_pairs is not None,
        "official_split_note": (
            "No public official LOD split list was found. This manifest uses a reproducible pair-random "
            "split and is not the BMVC paper's 1830/400 detection split."
        ),
        "expected_rgb": {"height": EXPECTED_HW[0], "width": EXPECTED_HW[1], "channels": 3, "dtype": "uint8"},
        "expected_raw": {"height": EXPECTED_HW[0], "width": EXPECTED_HW[1], "channels": 3, "dtype": "uint16"},
        "image_validation": image_validation,
        "path_columns_relative_to_lod_root": True,
        "label_space": "inverse_relative",
    }
    meta_path = manifest_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return meta_path


def main() -> None:
    args = parse_args()
    lod_root = args.lod_root.expanduser().resolve()
    output = args.output
    if output is None:
        output = lod_root / "manifests" / DEFAULT_MANIFEST_NAME
    output = output.expanduser().resolve()

    rgb_normal_ids = collect_ids(lod_root / "RGB_normal", ".JPG")
    rgb_dark_ids = collect_ids(lod_root / "RGB_Dark", ".JPG")
    raw_normal_ids = collect_ids(lod_root / "RAW_normal", ".png")
    raw_dark_ids = collect_ids(lod_root / "RAW_Dark", ".png")

    normal_ids = sorted(rgb_normal_ids)
    if normal_ids != sorted(raw_normal_ids):
        raise ValueError("RGB_normal ids and RAW_normal ids differ")
    if any(normal_id % 2 != 1 for normal_id in normal_ids):
        raise ValueError("Expected all normal ids to be odd")
    expected_dark_ids = {normal_id + 1 for normal_id in normal_ids}
    if rgb_dark_ids != expected_dark_ids:
        raise ValueError("RGB_Dark ids do not exactly match normal_id+1")
    if raw_dark_ids != expected_dark_ids:
        raise ValueError("RAW_Dark ids do not exactly match normal_id+1")

    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            raise ValueError("--max-pairs must be positive")
        normal_ids = normal_ids[: min(args.max_pairs, len(normal_ids))]
        train_count, val_count = derive_smoke_counts(len(normal_ids))
    else:
        train_count = int(args.train_count)
        val_count = int(args.val_count)

    split_by_id = split_rows(
        normal_ids,
        seed=int(args.seed),
        train_count=train_count,
        val_count=val_count,
        smoke_limited=args.max_pairs is not None,
    )
    rows = build_rows(
        lod_root=lod_root,
        normal_ids=normal_ids,
        split_by_id=split_by_id,
        validate_pixels=not bool(args.skip_image_validation),
    )
    write_manifest(output, rows)
    meta_path = write_metadata(
        output,
        lod_root=lod_root,
        rows=rows,
        seed=int(args.seed),
        train_count=train_count,
        val_count=val_count,
        max_pairs=args.max_pairs,
        image_validation=not bool(args.skip_image_validation),
    )
    split_counts = Counter(str(row["split"]) for row in rows)
    print(
        "[done] lod_true_pair_manifest "
        f"pairs={len(rows)} splits={dict(sorted(split_counts.items()))} "
        f"manifest={output} metadata={meta_path}"
    )


if __name__ == "__main__":
    main()
