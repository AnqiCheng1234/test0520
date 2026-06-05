from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic VKITTI2 scene-holdout split.")
    parser.add_argument("--input", required=True, help="Input VKITTI2 split file with RGB and depth paths.")
    parser.add_argument("--train-output", required=True, help="Output train split path.")
    parser.add_argument("--val-output", required=True, help="Output validation split path.")
    parser.add_argument("--summary-output", required=True, help="Output JSON summary path.")
    parser.add_argument(
        "--holdout-scene",
        action="append",
        required=True,
        help="Scene to hold out, e.g. Scene20. May be passed multiple times.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Optionally sample this many validation entries from the held-out scene entries.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used when --val-samples is set.")
    parser.add_argument("--strict", action="store_true", help="Fail if any RGB/depth file is missing.")
    return parser.parse_args()


def parse_vkitti_path(rgb_path: str) -> tuple[str, str, str, int]:
    parts = Path(rgb_path).parts
    try:
        frames_idx = parts.index("frames")
    except ValueError as exc:
        raise ValueError(f"Cannot parse VKITTI2 path without 'frames': {rgb_path}") from exc

    try:
        scene = parts[frames_idx - 2]
        condition = parts[frames_idx - 1]
        camera = parts[frames_idx + 2]
        frame_name = parts[frames_idx + 3]
    except IndexError as exc:
        raise ValueError(f"Cannot parse scene/condition/camera/frame from VKITTI2 path: {rgb_path}") from exc

    if not scene.startswith("Scene") or not camera.startswith("Camera_"):
        raise ValueError(
            f"Unexpected VKITTI2 scene/camera in path: scene={scene!r} camera={camera!r} path={rgb_path}"
        )
    if not frame_name.startswith("rgb_") or not frame_name.endswith(".jpg"):
        raise ValueError(f"Unexpected VKITTI2 frame filename: {frame_name!r} path={rgb_path}")
    frame_idx = int(frame_name.removeprefix("rgb_").removesuffix(".jpg"))
    return scene, condition, camera, frame_idx


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
            scene, condition, camera, frame_idx = parse_vkitti_path(rgb_path)
            entries.append(
                {
                    "index": line_idx,
                    "line": line,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "scene": scene,
                    "condition": condition,
                    "camera": camera,
                    "frame_idx": frame_idx,
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


def summarize_counts(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene = Counter(entry["scene"] for entry in entries)
    by_condition = Counter(entry["condition"] for entry in entries)
    by_camera = Counter(entry["camera"] for entry in entries)
    by_scene_condition_camera: dict[tuple[str, str, str], int] = Counter(
        (entry["scene"], entry["condition"], entry["camera"]) for entry in entries
    )
    return {
        "by_scene": dict(sorted(by_scene.items())),
        "by_condition": dict(sorted(by_condition.items())),
        "by_camera": dict(sorted(by_camera.items())),
        "per_scene_condition_camera": [
            {
                "scene": scene,
                "condition": condition,
                "camera": camera,
                "count": count,
            }
            for (scene, condition, camera), count in sorted(by_scene_condition_camera.items())
        ],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    train_output = Path(args.train_output).expanduser()
    val_output = Path(args.val_output).expanduser()
    summary_output = Path(args.summary_output).expanduser()

    holdout_scenes = sorted(set(args.holdout_scene))
    if not holdout_scenes:
        raise ValueError("At least one --holdout-scene is required.")
    for scene in holdout_scenes:
        if not scene.startswith("Scene"):
            raise ValueError(f"Unexpected holdout scene name: {scene!r}")

    entries = read_entries(input_path)
    available_scenes = sorted({entry["scene"] for entry in entries})
    unknown = sorted(set(holdout_scenes) - set(available_scenes))
    if unknown:
        raise ValueError(f"Holdout scenes not found in input: {unknown}; available={available_scenes}")

    missing_rgb = sum(1 for entry in entries if not Path(entry["rgb_path"]).is_file())
    missing_depth = sum(1 for entry in entries if not Path(entry["depth_path"]).is_file())
    if args.strict and (missing_rgb or missing_depth):
        raise FileNotFoundError(f"Missing files: missing_rgb={missing_rgb} missing_depth={missing_depth}")

    train_entries = [entry for entry in entries if entry["scene"] not in holdout_scenes]
    holdout_entries = [entry for entry in entries if entry["scene"] in holdout_scenes]
    if args.val_samples is not None:
        if args.val_samples <= 0:
            raise ValueError(f"--val-samples must be positive when set, got {args.val_samples}")
        if args.val_samples > len(holdout_entries):
            raise ValueError(f"--val-samples={args.val_samples} exceeds holdout entries={len(holdout_entries)}")
        rng = random.Random(args.seed)
        selected = list(holdout_entries)
        rng.shuffle(selected)
        selected_indices = {int(entry["index"]) for entry in selected[: args.val_samples]}
        val_entries = [entry for entry in entries if int(entry["index"]) in selected_indices]
    else:
        val_entries = holdout_entries
    unused_holdout_entries = [
        entry
        for entry in holdout_entries
        if int(entry["index"]) not in {int(val_entry["index"]) for val_entry in val_entries}
    ]
    if not train_entries or not val_entries:
        raise ValueError(f"Invalid scene holdout split: train={len(train_entries)} val={len(val_entries)}")

    train_indices = {int(entry["index"]) for entry in train_entries}
    val_indices = {int(entry["index"]) for entry in val_entries}
    overlap_count = len(train_indices & val_indices)
    if overlap_count:
        raise RuntimeError(f"Internal split error: overlap_count={overlap_count}")

    train_scenes = sorted({entry["scene"] for entry in train_entries})
    val_scenes = sorted({entry["scene"] for entry in val_entries})
    scene_overlap = sorted(set(train_scenes) & set(val_scenes))
    if scene_overlap:
        raise RuntimeError(f"Scene holdout failed; train/val scene overlap: {scene_overlap}")

    write_split(train_output, train_entries)
    write_split(val_output, val_entries)

    summary = {
        "input": str(input_path),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "split_type": "scene_holdout",
        "holdout_scenes": holdout_scenes,
        "available_scenes": available_scenes,
        "val_sampling": None
        if args.val_samples is None
        else {
            "sampled_from_holdout": True,
            "requested_val_samples": int(args.val_samples),
            "seed": int(args.seed),
            "holdout_total": len(holdout_entries),
            "unused_holdout": len(unused_holdout_entries),
        },
        "total": len(entries),
        "train": len(train_entries),
        "val": len(val_entries),
        "unused_holdout": len(unused_holdout_entries),
        "train_fraction": len(train_entries) / len(entries),
        "val_fraction": len(val_entries) / len(entries),
        "missing_rgb": missing_rgb,
        "missing_depth": missing_depth,
        "overlap_count": overlap_count,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "scene_overlap": scene_overlap,
        "train_counts": summarize_counts(train_entries),
        "val_counts": summarize_counts(val_entries),
        "unused_holdout_counts": summarize_counts(unused_holdout_entries),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        "wrote scene-holdout split: "
        f"holdout_scenes={holdout_scenes} total={summary['total']} "
        f"train={summary['train']} val={summary['val']} "
        f"missing_rgb={missing_rgb} missing_depth={missing_depth} overlap_count={overlap_count}"
    )


if __name__ == "__main__":
    main()
