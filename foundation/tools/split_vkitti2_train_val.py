from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic VKITTI2 train/val split for M-series runs.")
    parser.add_argument("--input", required=True, help="Input VKITTI2 split file with RGB and depth paths.")
    parser.add_argument("--train-output", required=True, help="Output train split path.")
    parser.add_argument("--val-output", required=True, help="Output validation split path.")
    parser.add_argument("--summary-output", required=True, help="Output JSON summary path.")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail if any RGB/depth file is missing.")
    return parser.parse_args()


def stratum_key(rgb_path: str) -> tuple[str, str, str]:
    parts = Path(rgb_path).parts
    try:
        frames_idx = parts.index("frames")
    except ValueError as exc:
        raise ValueError(f"Cannot parse VKITTI2 path without 'frames': {rgb_path}") from exc

    try:
        scene = parts[frames_idx - 2]
        condition = parts[frames_idx - 1]
        camera = parts[frames_idx + 2]
    except IndexError as exc:
        raise ValueError(f"Cannot parse scene/condition/camera from VKITTI2 path: {rgb_path}") from exc

    if not scene.startswith("Scene") or not camera.startswith("Camera_"):
        raise ValueError(
            f"Unexpected VKITTI2 scene/camera in path: scene={scene!r} camera={camera!r} path={rgb_path}"
        )
    return scene, condition, camera


def read_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Expected two fields at {path}:{line_idx + 1}, got {len(fields)}: {line}")
            rgb_path, depth_path = fields
            scene, condition, camera = stratum_key(rgb_path)
            entries.append(
                {
                    "index": line_idx,
                    "line": line,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "scene": scene,
                    "condition": condition,
                    "camera": camera,
                    "key": (scene, condition, camera),
                }
            )
    if not entries:
        raise ValueError(f"No split entries found in {path}")
    return entries


def write_split(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry["line"])
            f.write("\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError(f"--val-fraction must be in (0, 1), got {args.val_fraction}")

    input_path = Path(args.input).expanduser()
    train_output = Path(args.train_output).expanduser()
    val_output = Path(args.val_output).expanduser()
    summary_output = Path(args.summary_output).expanduser()

    entries = read_entries(input_path)
    missing_rgb = sum(1 for entry in entries if not Path(entry["rgb_path"]).is_file())
    missing_depth = sum(1 for entry in entries if not Path(entry["depth_path"]).is_file())
    if args.strict and (missing_rgb or missing_depth):
        raise FileNotFoundError(f"Missing files: missing_rgb={missing_rgb} missing_depth={missing_depth}")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["key"]].append(entry)

    rng = random.Random(args.seed)
    train_indices: set[int] = set()
    val_indices: set[int] = set()
    per_key: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = list(grouped[key])
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        val_count = max(1, round(n * args.val_fraction)) if n >= 2 else 0
        val_group = shuffled[:val_count]
        train_group = shuffled[val_count:]
        train_indices.update(int(entry["index"]) for entry in train_group)
        val_indices.update(int(entry["index"]) for entry in val_group)
        scene, condition, camera = key
        per_key.append(
            {
                "scene": scene,
                "condition": condition,
                "camera": camera,
                "total_count": n,
                "train_count": len(train_group),
                "val_count": len(val_group),
            }
        )

    overlap_count = len(train_indices & val_indices)
    if overlap_count:
        raise RuntimeError(f"Internal split error: overlap_count={overlap_count}")

    train_entries = [entry for entry in entries if int(entry["index"]) in train_indices]
    val_entries = [entry for entry in entries if int(entry["index"]) in val_indices]
    write_split(train_output, train_entries)
    write_split(val_output, val_entries)

    summary = {
        "input": str(input_path),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "total": len(entries),
        "train": len(train_entries),
        "val": len(val_entries),
        "missing_rgb": missing_rgb,
        "missing_depth": missing_depth,
        "overlap_count": overlap_count,
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "per_scene_condition_camera": per_key,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        "wrote split: "
        f"total={summary['total']} train={summary['train']} val={summary['val']} "
        f"missing_rgb={missing_rgb} missing_depth={missing_depth} overlap_count={overlap_count}"
    )


if __name__ == "__main__":
    main()
