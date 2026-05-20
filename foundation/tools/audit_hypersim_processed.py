from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np


DEFAULT_PROCESSED_BASE = Path("/mnt/drive/1111_new_works/hypersim_marigold_processed/hypersim")
DEFAULT_RAW_ROOT = Path("/mnt/drive/1111_new_works/hypersim")
SPLIT_DIR_NAMES = {
    "train": "hypersim_processed_train",
    "val": "hypersim_processed_val",
    "test": "hypersim_processed_test",
}


def hypersim_distance_to_depth(distance: np.ndarray) -> np.ndarray:
    width, height, focal = 1024, 768, 886.81
    imageplane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width).reshape(1, width)
    imageplane_x = np.repeat(imageplane_x, height, axis=0).astype(np.float32)[:, :, None]
    imageplane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height).reshape(height, 1)
    imageplane_y = np.repeat(imageplane_y, width, axis=1).astype(np.float32)[:, :, None]
    imageplane_z = np.full((height, width, 1), focal, np.float32)
    imageplane = np.concatenate([imageplane_x, imageplane_y, imageplane_z], axis=2)
    return distance / np.linalg.norm(imageplane, ord=2, axis=2) * focal


def parse_processed_depth_rel(depth_rel: str) -> Tuple[str, str, int]:
    scene, name = depth_rel.split("/", 1)
    stem = Path(name).stem
    # depth_plane_cam_00_fr0000 -> cam_00, 0
    parts = stem.split("_")
    if len(parts) < 5 or parts[0] != "depth" or parts[1] != "plane" or parts[2] != "cam":
        raise ValueError(f"Unexpected processed depth path format: {depth_rel}")
    camera = f"{parts[2]}_{parts[3]}"
    frame = int(parts[4].removeprefix("fr"))
    return scene, camera, frame


def raw_hdf5_path(raw_root: Path, depth_rel: str) -> Path:
    scene, camera, frame = parse_processed_depth_rel(depth_rel)
    return raw_root / scene / "images" / f"scene_{camera}_geometry_hdf5" / f"frame.{frame:04d}.depth_meters.hdf5"


def read_filelist(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Expected '<rgb> <depth>' row in {path}, got: {line!r}")
            rows.append((parts[0], parts[1]))
    return rows


def summarize_depths(split_root: Path, rows: List[Tuple[str, str]], *, max_samples: int, pixel_stride: int) -> Dict[str, object]:
    sampled_values = []
    invalid_pixels = 0
    total_pixels = 0
    used = rows[:max_samples] if max_samples > 0 else rows
    for _, depth_rel in used:
        depth_path = split_root / depth_rel
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise ValueError(f"Failed to read depth PNG: {depth_path}")
        depth_m = depth_raw.astype(np.float32) / 1000.0
        finite = np.isfinite(depth_m) & (depth_m > 0)
        total_pixels += int(depth_m.size)
        invalid_pixels += int(depth_m.size - np.count_nonzero(finite))
        if pixel_stride > 1:
            depth_m = depth_m[::pixel_stride, ::pixel_stride]
            finite = finite[::pixel_stride, ::pixel_stride]
        if np.any(finite):
            sampled_values.append(depth_m[finite])

    if sampled_values:
        values = np.concatenate(sampled_values)
        quantiles = np.quantile(values, [0.001, 0.01, 0.5, 0.95, 0.99])
        stats = {
            "min": float(values.min()),
            "p0_1": float(quantiles[0]),
            "p1": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "p99": float(quantiles[4]),
            "max": float(values.max()),
            "sampled_pixels": int(values.size),
        }
    else:
        stats = {
            "min": None,
            "p0_1": None,
            "p1": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "sampled_pixels": 0,
        }

    stats["samples"] = int(len(used))
    stats["invalid_ratio"] = float(invalid_pixels / max(total_pixels, 1))
    return stats


def compare_to_raw(split_root: Path, raw_root: Path, rows: List[Tuple[str, str]], *, compare_samples: int) -> List[Dict[str, object]]:
    results = []
    for rgb_rel, depth_rel in rows[:compare_samples]:
        processed_depth_path = split_root / depth_rel
        raw_depth_path = raw_hdf5_path(raw_root, depth_rel)
        depth_raw = cv2.imread(str(processed_depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise ValueError(f"Failed to read processed depth: {processed_depth_path}")
        if not raw_depth_path.is_file():
            raise FileNotFoundError(f"Missing raw HyperSim HDF5 depth: {raw_depth_path}")

        processed_depth = depth_raw.astype(np.float32) / 1000.0
        with h5py.File(raw_depth_path, "r") as f:
            distance = np.asarray(f["dataset"], dtype=np.float32)
        raw_plane = hypersim_distance_to_depth(distance).astype(np.float32)

        valid = np.isfinite(processed_depth) & np.isfinite(raw_plane) & (processed_depth > 0) & (raw_plane > 0)
        diff = np.abs(processed_depth[valid] - raw_plane[valid])
        results.append(
            {
                "rgb_rel": rgb_rel,
                "depth_rel": depth_rel,
                "processed_depth_path": str(processed_depth_path),
                "raw_depth_path": str(raw_depth_path),
                "valid_pixels": int(np.count_nonzero(valid)),
                "mean_abs_diff_m": float(diff.mean()) if diff.size else None,
                "p95_abs_diff_m": float(np.quantile(diff, 0.95)) if diff.size else None,
                "max_abs_diff_m": float(diff.max()) if diff.size else None,
                "processed_min_m": float(processed_depth[valid].min()) if diff.size else None,
                "processed_max_m": float(processed_depth[valid].max()) if diff.size else None,
                "raw_plane_min_m": float(raw_plane[valid].min()) if diff.size else None,
                "raw_plane_max_m": float(raw_plane[valid].max()) if diff.size else None,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Marigold-processed HyperSim RGB/depth files.")
    parser.add_argument("--processed-base", type=Path, default=DEFAULT_PROCESSED_BASE)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--split", choices=sorted(SPLIT_DIR_NAMES), default="train")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--pixel-stride", type=int, default=16)
    parser.add_argument("--compare-samples", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    split_root = args.processed_base / SPLIT_DIR_NAMES[args.split]
    filelist_path = split_root / f"filename_list_{args.split}.txt"
    rows = read_filelist(filelist_path)
    payload = {
        "processed_base": str(args.processed_base),
        "split_root": str(split_root),
        "filelist_path": str(filelist_path),
        "split": args.split,
        "total_rows": len(rows),
        "depth_histogram": summarize_depths(
            split_root,
            rows,
            max_samples=args.max_samples,
            pixel_stride=max(int(args.pixel_stride), 1),
        ),
        "raw_plane_comparison": compare_to_raw(
            split_root,
            args.raw_root,
            rows,
            compare_samples=max(int(args.compare_samples), 0),
        ),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
