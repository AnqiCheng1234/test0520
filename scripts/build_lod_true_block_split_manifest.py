#!/usr/bin/env python3
"""Build a neighbor-excluded block split view for true LOD pseudo labels.

This rewrites only the manifest split column. It keeps pseudo_depth_path values
pointing at the already generated DAv2-L labels, so teacher inference does not
need to be rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOD_ROOT = Path("/home/caq/6666_raw/0000_dataset/LOD")
DEFAULT_SOURCE_MANIFEST = (
    DEFAULT_LOD_ROOT
    / "pseudo_depth_dav2l_rgb_normal_rel_1200x800"
    / "lod_true_rgb_normal_dav2l_rel_manifest.csv"
)
DEFAULT_OUTPUT = (
    DEFAULT_LOD_ROOT
    / "pseudo_depth_dav2l_rgb_normal_rel_1200x800"
    / "lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv"
)
REQUIRED_COLUMNS = (
    "pair_id",
    "split",
    "normal_id",
    "dark_id",
    "rgb_normal_path",
    "rgb_dark_path",
    "raw_normal_path",
    "raw_dark_path",
    "pseudo_depth_path",
    "label_space",
    "height",
    "width",
    "teacher_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-root", type=Path, default=DEFAULT_LOD_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-count", type=int, default=112)
    parser.add_argument("--val-blocks", type=int, default=8)
    parser.add_argument(
        "--neighbor-exclude-radius",
        type=int,
        default=10,
        help="Number of non-val pair indices excluded from train on each side of every validation block.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Smoke/debug limit. Uses the first N pairs from the source manifest.",
    )
    parser.add_argument(
        "--skip-path-validation",
        action="store_true",
        help="Skip checking image and pseudo-depth paths exist.",
    )
    return parser.parse_args()


def resolve_data_path(root: Path, value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def load_rows(path: Path, *, max_pairs: int | None) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    rows.sort(key=lambda row: int(row["normal_id"]))
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("--max-pairs must be positive")
        rows = rows[: min(max_pairs, len(rows))]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    seen = set()
    for row in rows:
        normal_id = int(row["normal_id"])
        dark_id = int(row["dark_id"])
        if dark_id != normal_id + 1:
            raise ValueError(f"{row['pair_id']} violates dark_id=normal_id+1")
        if normal_id in seen:
            raise ValueError(f"Duplicate normal_id={normal_id}")
        seen.add(normal_id)
    return rows, fieldnames


def choose_uniform_block_starts(
    *,
    total: int,
    val_count: int,
    val_blocks: int,
    exclude_radius: int,
) -> list[int]:
    if val_blocks <= 0:
        raise ValueError("--val-blocks must be positive")
    if val_count <= 0:
        raise ValueError("--val-count must be positive")
    if val_count % val_blocks != 0:
        raise ValueError("--val-count must be divisible by --val-blocks for uniform block split")
    block_size = val_count // val_blocks
    if block_size <= 0:
        raise ValueError("Validation block size must be positive")
    if exclude_radius < 0:
        raise ValueError("--neighbor-exclude-radius must be >= 0")
    needed = val_count + 2 * exclude_radius * val_blocks
    if needed >= total:
        raise ValueError(
            "Not enough pairs for requested val blocks and neighbor exclusion: "
            f"need at least >{needed}, got {total}"
        )
    first = exclude_radius
    last = total - exclude_radius - block_size
    if val_blocks == 1:
        return [round((total - block_size) / 2)]
    starts = [round(first + i * (last - first) / (val_blocks - 1)) for i in range(val_blocks)]
    previous_end: int | None = None
    for start in starts:
        end = start + block_size - 1
        if previous_end is not None and start <= previous_end + 2 * exclude_radius:
            raise ValueError("Validation blocks overlap after applying neighbor exclusion")
        if start < 0 or end >= total:
            raise ValueError("Validation block placement exceeded available pair range")
        previous_end = end
    return starts


def assign_splits(
    rows: list[dict[str, str]],
    *,
    val_count: int,
    val_blocks: int,
    exclude_radius: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    total = len(rows)
    block_size = val_count // val_blocks
    starts = choose_uniform_block_starts(
        total=total,
        val_count=val_count,
        val_blocks=val_blocks,
        exclude_radius=exclude_radius,
    )
    split_by_index = {index: "00Train" for index in range(total)}
    blocks: list[dict[str, Any]] = []
    for block_id, start in enumerate(starts):
        end = start + block_size - 1
        for index in range(max(0, start - exclude_radius), min(total, end + exclude_radius + 1)):
            split_by_index[index] = "02HoldoutBuffer"
        for index in range(start, end + 1):
            split_by_index[index] = "01Valid"
        blocks.append(
            {
                "block_id": block_id,
                "start_index": start,
                "end_index": end,
                "start_pair_id": rows[start]["pair_id"],
                "end_pair_id": rows[end]["pair_id"],
                "start_normal_id": int(rows[start]["normal_id"]),
                "end_normal_id": int(rows[end]["normal_id"]),
            }
        )
    return split_by_index, blocks


def validate_paths(rows: list[dict[str, str]], *, lod_root: Path) -> None:
    for row in rows:
        for key in ("rgb_normal_path", "rgb_dark_path", "raw_normal_path", "raw_dark_path", "pseudo_depth_path"):
            path = resolve_data_path(lod_root, row[key])
            if not path.is_file():
                raise FileNotFoundError(f"{row['pair_id']} missing {key}: {path}")


def nearest_val_train_distance(split_by_index: dict[int, str]) -> dict[str, int | None]:
    val_indices = [idx for idx, split in split_by_index.items() if split == "01Valid"]
    train_indices = [idx for idx, split in split_by_index.items() if split == "00Train"]
    if not val_indices or not train_indices:
        return {"min": None, "max": None}
    nearest = []
    for val_idx in val_indices:
        nearest.append(min(abs(val_idx - train_idx) for train_idx in train_indices))
    return {"min": min(nearest), "max": max(nearest)}


def write_outputs(
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    output: Path,
    source_manifest: Path,
    lod_root: Path,
    split_by_index: dict[int, str],
    blocks: list[dict[str, Any]],
    val_count: int,
    val_blocks: int,
    exclude_radius: int,
    max_pairs: int | None,
    path_validation: bool,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for index, row in enumerate(rows):
        new_row = dict(row)
        new_row["split"] = split_by_index[index]
        new_rows.append(new_row)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    counts = Counter(row["split"] for row in new_rows)
    audit = nearest_val_train_distance(split_by_index)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lod_root": str(lod_root.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "manifest": str(output.resolve()),
        "pair_count": len(rows),
        "split_policy": "uniform_contiguous_blocks_with_neighbor_exclusion",
        "split_unit": "sorted normal_id pair index",
        "val_count": val_count,
        "val_blocks": val_blocks,
        "val_block_size": val_count // val_blocks,
        "neighbor_exclude_radius_pairs": exclude_radius,
        "split_counts": dict(sorted(counts.items())),
        "split_names": ["00Train", "01Valid", "02HoldoutBuffer"],
        "validation_blocks": blocks,
        "nearest_val_to_train_distance_pairs": audit,
        "path_validation": path_validation,
        "max_pairs": max_pairs,
        "is_smoke": max_pairs is not None,
        "pseudo_depth_paths_reused_from_source_manifest": True,
    }
    meta_path = output.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return meta_path


def main() -> None:
    args = parse_args()
    lod_root = args.lod_root.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows, fieldnames = load_rows(source_manifest, max_pairs=args.max_pairs)
    if not args.skip_path_validation:
        validate_paths(rows, lod_root=lod_root)
    split_by_index, blocks = assign_splits(
        rows,
        val_count=int(args.val_count),
        val_blocks=int(args.val_blocks),
        exclude_radius=int(args.neighbor_exclude_radius),
    )
    meta_path = write_outputs(
        rows=rows,
        fieldnames=fieldnames,
        output=output,
        source_manifest=source_manifest,
        lod_root=lod_root,
        split_by_index=split_by_index,
        blocks=blocks,
        val_count=int(args.val_count),
        val_blocks=int(args.val_blocks),
        exclude_radius=int(args.neighbor_exclude_radius),
        max_pairs=args.max_pairs,
        path_validation=not bool(args.skip_path_validation),
    )
    counts = Counter(split_by_index.values())
    audit = nearest_val_train_distance(split_by_index)
    print(
        "[done] lod_true_block_split_manifest "
        f"pairs={len(rows)} splits={dict(sorted(counts.items()))} "
        f"nearest_val_train_pair_distance={audit} "
        f"manifest={output} metadata={meta_path}"
    )


if __name__ == "__main__":
    main()
