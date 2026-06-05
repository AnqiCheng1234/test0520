#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError(f"Expected a non-empty comma-separated list, got {value!r}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LED-HB absolute four-column filelists.")
    parser.add_argument("--led-root", required=True)
    parser.add_argument("--illum", required=True, choices=["HB"])
    parser.add_argument("--train-maps", required=True)
    parser.add_argument("--val-maps", required=True)
    parser.add_argument("--val-stride", type=int, required=True)
    parser.add_argument("--val-n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def frame_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Non-integer frame stem in {path}") from exc


def collect_map_samples(root: Path, split: str, map_name: str) -> list[tuple[Path, Path, Path, Path]]:
    map_root = root / "extracted" / "HB" / split / map_name
    if not map_root.is_dir():
        raise FileNotFoundError(f"Missing LED-HB map directory: {map_root}")

    rgb_dir = map_root / "ldr_color"
    depth_dir = map_root / "distance_to_image_plane"
    camera_dir = map_root / "camera_params"
    transforms_dir = map_root / "transforms"
    for directory in (rgb_dir, depth_dir, camera_dir, transforms_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing required LED-HB directory: {directory}")

    rgb_frames = {frame_id(path): path for path in rgb_dir.glob("*.png")}
    depth_frames = {frame_id(path): path for path in depth_dir.glob("*.exr")}
    camera_frames = {frame_id(path): path for path in camera_dir.glob("*.json")}
    transform_frames = {frame_id(path): path for path in transforms_dir.glob("*.p")}

    frame_sets = {
        "ldr_color": set(rgb_frames),
        "distance_to_image_plane": set(depth_frames),
        "camera_params": set(camera_frames),
        "transforms": set(transform_frames),
    }
    common = set.intersection(*frame_sets.values()) if frame_sets else set()
    for name, frames in frame_sets.items():
        missing = sorted(common.symmetric_difference(frames))[:10]
        if frames != common:
            raise RuntimeError(
                f"Frame mismatch for {split}/{map_name}/{name}: "
                f"common={len(common)} {name}={len(frames)} example_diff={missing}"
            )

    return [
        (
            rgb_frames[frame].resolve(),
            depth_frames[frame].resolve(),
            camera_frames[frame].resolve(),
            transform_frames[frame].resolve(),
        )
        for frame in sorted(common)
    ]


def write_filelist(path: Path, rows: list[tuple[Path, Path, Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(" ".join(str(value) for value in row) + "\n")


def sample_records(rows: list[tuple[Path, Path, Path, Path]]) -> list[dict[str, str]]:
    return [
        {
            "rgb": str(row[0]),
            "depth": str(row[1]),
            "camera_params": str(row[2]),
            "transforms": str(row[3]),
        }
        for row in rows
    ]


def summarize_split(
    *,
    args: argparse.Namespace,
    train_maps: list[str],
    val_maps: list[str],
    train_rows: list[tuple[Path, Path, Path, Path]],
    val_rows: list[tuple[Path, Path, Path, Path]],
    map_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    return {
        "dataset_name": "led_hb",
        "illumination": args.illum,
        "led_root": str(Path(args.led_root).expanduser().resolve()),
        "train_split": "hb_train_all",
        "val_split": f"hb_val_stride{int(args.val_stride)}_n{int(args.val_n)}_seed{int(args.seed)}",
        "train_maps": train_maps,
        "val_maps": val_maps,
        "val_stride": int(args.val_stride),
        "val_n": int(args.val_n),
        "seed": int(args.seed),
        "selection": {
            "train": "all numerically sorted frames from each train map",
            "val": "position stride over numerically sorted frames, then truncated to val_n",
            "seed_semantics": "provenance only; no random sampling",
        },
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "by_map": map_counts,
        },
        "first_10": {
            "train": sample_records(train_rows[:10]),
            "val": sample_records(val_rows[:10]),
        },
        "last_10": {
            "train": sample_records(train_rows[-10:]),
            "val": sample_records(val_rows[-10:]),
        },
    }


def main() -> None:
    args = parse_args()
    if args.val_stride <= 0:
        raise ValueError(f"--val-stride must be positive, got {args.val_stride}")
    if args.val_n <= 0:
        raise ValueError(f"--val-n must be positive, got {args.val_n}")

    led_root = Path(args.led_root).expanduser().resolve()
    train_maps = parse_csv(args.train_maps)
    val_maps = parse_csv(args.val_maps)
    out_dir = Path(args.out_dir).expanduser().resolve()

    train_rows: list[tuple[Path, Path, Path, Path]] = []
    val_rows_all: list[tuple[Path, Path, Path, Path]] = []
    map_counts: dict[str, dict[str, int]] = {"train": {}, "val_raw": {}, "val_selected": {}}

    for map_name in train_maps:
        rows = collect_map_samples(led_root, "train", map_name)
        train_rows.extend(rows)
        map_counts["train"][map_name] = len(rows)

    for map_name in val_maps:
        rows = collect_map_samples(led_root, "val", map_name)
        val_rows_all.extend(rows)
        map_counts["val_raw"][map_name] = len(rows)

    val_rows = val_rows_all[:: int(args.val_stride)][: int(args.val_n)]
    for map_name in val_maps:
        selected = [
            row
            for row in val_rows
            if row[0].parent.parent.name == map_name
        ]
        map_counts["val_selected"][map_name] = len(selected)

    train_path = out_dir / "hb_train_all.txt"
    val_path = out_dir / f"hb_val_stride{int(args.val_stride)}_n{int(args.val_n)}_seed{int(args.seed)}.txt"
    summary_path = out_dir / "hb_split_summary.json"

    write_filelist(train_path, train_rows)
    write_filelist(val_path, val_rows)
    summary = summarize_split(
        args=args,
        train_maps=train_maps,
        val_maps=val_maps,
        train_rows=train_rows,
        val_rows=val_rows,
        map_counts=map_counts,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"train": len(train_rows), "val": len(val_rows), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
