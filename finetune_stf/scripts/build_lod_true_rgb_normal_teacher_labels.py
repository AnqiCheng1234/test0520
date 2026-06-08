#!/usr/bin/env python3
"""Build true-LOD RGB_normal DAv2 teacher pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEFAULT_LOD_ROOT = Path("/home/caq/6666_raw/0000_dataset/LOD")
DEFAULT_PAIR_MANIFEST = DEFAULT_LOD_ROOT / "manifests" / "lod_true_pairs_2118_112_seed42.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_LOD_ROOT / "pseudo_depth_dav2l_rgb_normal_rel_1200x800"
DEFAULT_CHECKPOINT = Path("/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth")
MANIFEST_NAME = "lod_true_rgb_normal_dav2l_rel_manifest.csv"
NATIVE_HW = (800, 1200)
REQUIRED_PAIR_COLUMNS = (
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
OUTPUT_MANIFEST_COLUMNS = (
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
    parser.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--input-size", type=int, default=812)
    parser.add_argument("--splits", nargs="+", default=["00Train", "01Valid"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size-sweep", action="store_true")
    parser.add_argument("--sweep-input-sizes", nargs="+", type=int, default=[812, 924])
    parser.add_argument("--vis-cmap", default="magma_r", choices=("magma", "magma_r"))
    parser.add_argument("--vis-vmin-pct", type=float, default=1.0)
    parser.add_argument("--vis-vmax-pct", type=float, default=99.0)
    parser.add_argument(
        "--cleanup-success",
        action="store_true",
        help="Remove output-root after success when it is clearly a smoke/debug/tmp/codex_smoke path.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def rel_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def read_pair_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    with args.pair_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_PAIR_COLUMNS if name not in set(reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{args.pair_manifest} missing required columns: {missing}")
        rows: list[dict[str, Any]] = []
        wanted_splits = set(args.splits)
        per_split_counts: Counter[str] = Counter()
        for row in reader:
            split = row["split"].strip()
            if split not in wanted_splits:
                continue
            if args.max_samples_per_split is not None and per_split_counts[split] >= args.max_samples_per_split:
                continue
            item = dict(row)
            for key in ("rgb_normal_path", "rgb_dark_path", "raw_normal_path", "raw_dark_path"):
                item[key] = resolve_path(args.lod_root, item[key])
                if not item[key].is_file():
                    raise FileNotFoundError(f"Missing LOD pair file for {row['pair_id']}: {item[key]}")
            if item["label_space"] != "inverse_relative":
                raise ValueError(f"Unsupported label_space for {row['pair_id']}: {item['label_space']!r}")
            if (int(item["height"]), int(item["width"])) != NATIVE_HW:
                raise ValueError(f"Unexpected native HW for {row['pair_id']}: {(item['height'], item['width'])}")
            rows.append(item)
            per_split_counts[split] += 1
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
    if not rows:
        raise ValueError("No LOD rows selected")
    return rows


def load_model(args: argparse.Namespace) -> DepthAnythingV2:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAv2 checkpoint: {args.checkpoint}")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state = torch.load(str(args.checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device).eval()
    return model


def read_rgb_normal_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV failed to read RGB_normal: {path}")
    if image.shape != (NATIVE_HW[0], NATIVE_HW[1], 3):
        raise ValueError(f"RGB_normal shape {image.shape}, expected {(NATIVE_HW[0], NATIVE_HW[1], 3)}: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"RGB_normal dtype {image.dtype}, expected uint8: {path}")
    return image


def infer_teacher(model: DepthAnythingV2, rgb_normal_path: Path, input_size: int) -> np.ndarray:
    bgr = read_rgb_normal_bgr(rgb_normal_path)
    pred = model.infer_image(bgr, input_size=input_size).astype(np.float32, copy=False)
    if pred.shape != NATIVE_HW:
        raise ValueError(f"Expected prediction shape {NATIVE_HW}, got {tuple(pred.shape)} for {rgb_normal_path}")
    if not np.isfinite(pred).all():
        raise ValueError(f"Prediction contains non-finite values for {rgb_normal_path}")
    if float(np.max(pred)) <= 0.0:
        raise ValueError(f"Prediction is non-positive for {rgb_normal_path}")
    return pred


def colorize_prediction(pred: np.ndarray, *, cmap_name: str, vmin_pct: float, vmax_pct: float) -> np.ndarray:
    valid = pred[np.isfinite(pred)]
    if valid.size == 0:
        scaled = np.zeros_like(pred, dtype=np.uint8)
    else:
        vmin = float(np.percentile(valid, vmin_pct))
        vmax = float(np.percentile(valid, vmax_pct))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        scaled = np.clip((pred - vmin) / (vmax - vmin), 0.0, 1.0)
        if cmap_name.endswith("_r"):
            scaled = 1.0 - scaled
        scaled = (scaled * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_MAGMA)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.save(tmp, array.astype(np.float32, copy=False))
    tmp.replace(path)


def atomic_save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    ok = cv2.imwrite(str(tmp), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {tmp}")
    tmp.replace(path)


def array_stats(pred: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(pred) & (pred > 0)
    payload: dict[str, Any] = {
        "shape": list(pred.shape),
        "valid_positive_coverage": float(valid.mean()),
    }
    vals = pred[valid]
    if vals.size:
        payload.update(
            {
                "min": float(vals.min()),
                "mean": float(vals.mean()),
                "p50": float(np.percentile(vals, 50.0)),
                "p99": float(np.percentile(vals, 99.0)),
                "max": float(vals.max()),
            }
        )
    return payload


def output_stem(row: dict[str, Any]) -> str:
    return f"{int(row['normal_id']):04d}_{int(row['dark_id']):04d}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_output_manifest(args: argparse.Namespace, rows: list[dict[str, str]]) -> Path:
    manifest_path = args.output_root / MANIFEST_NAME
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def cleanup_if_requested(args: argparse.Namespace) -> None:
    if not args.cleanup_success:
        return
    path = args.output_root.resolve()
    marker_ok = any(marker in str(path) for marker in ("smoke", "debug", "tmp", "codex_smoke"))
    if not marker_ok:
        raise ValueError(f"Refusing cleanup-success for non-temporary output path: {path}")
    shutil.rmtree(path)
    print(f"[CLEANUP] removed successful temporary output: {path}", flush=True)


def run_sweep(args: argparse.Namespace) -> None:
    rows = read_pair_rows(args)
    model = load_model(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    start = time.time()
    for index, row in enumerate(rows, start=1):
        for input_size in args.sweep_input_sizes:
            pred = infer_teacher(model, row["rgb_normal_path"], input_size)
            stem = f"{output_stem(row)}_input{input_size}"
            npy_path = args.output_root / f"{stem}.npy"
            png_path = args.output_root / f"{stem}.png"
            atomic_save_npy(npy_path, pred)
            atomic_save_png(
                png_path,
                colorize_prediction(
                    pred,
                    cmap_name=args.vis_cmap,
                    vmin_pct=args.vis_vmin_pct,
                    vmax_pct=args.vis_vmax_pct,
                ),
            )
            record = {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "normal_id": int(row["normal_id"]),
                "dark_id": int(row["dark_id"]),
                "input_size": int(input_size),
                "npy_path": str(npy_path),
                "png_path": str(png_path),
                **array_stats(pred),
            }
            summary_rows.append(record)
            print(f"[SWEEP] {index}/{len(rows)} pair={row['pair_id']} input_size={input_size} -> {npy_path}", flush=True)
    write_json(
        args.output_root / "input_size_sweep_summary.json",
        {
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.time() - start,
            "lod_root": str(args.lod_root),
            "pair_manifest": str(args.pair_manifest),
            "checkpoint": str(args.checkpoint),
            "encoder": args.encoder,
            "device": args.device,
            "sweep_input_sizes": [int(v) for v in args.sweep_input_sizes],
            "sample_count": len(rows),
            "rows": summary_rows,
        },
    )
    cleanup_if_requested(args)


def run_label_build(args: argparse.Namespace) -> None:
    rows = read_pair_rows(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    manifest_rows: list[dict[str, str]] = []
    stats_rows = []
    start = time.time()
    for index, row in enumerate(rows, start=1):
        stem = output_stem(row)
        label_dir = args.output_root / row["split"]
        vis_dir = args.output_root / "vis" / row["split"]
        npy_path = label_dir / f"{stem}.npy"
        png_path = vis_dir / f"{stem}.png"
        if npy_path.exists() and not args.overwrite:
            pred = np.load(npy_path).astype(np.float32, copy=False)
            if pred.shape != NATIVE_HW:
                raise ValueError(f"Existing label has wrong shape {pred.shape}: {npy_path}")
        else:
            pred = infer_teacher(model, row["rgb_normal_path"], args.input_size)
            atomic_save_npy(npy_path, pred)
            atomic_save_png(
                png_path,
                colorize_prediction(
                    pred,
                    cmap_name=args.vis_cmap,
                    vmin_pct=args.vis_vmin_pct,
                    vmax_pct=args.vis_vmax_pct,
                ),
            )
        manifest_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "normal_id": str(row["normal_id"]),
                "dark_id": str(row["dark_id"]),
                "rgb_normal_path": rel_to_root(args.lod_root, row["rgb_normal_path"]),
                "rgb_dark_path": rel_to_root(args.lod_root, row["rgb_dark_path"]),
                "raw_normal_path": rel_to_root(args.lod_root, row["raw_normal_path"]),
                "raw_dark_path": rel_to_root(args.lod_root, row["raw_dark_path"]),
                "pseudo_depth_path": rel_to_root(args.lod_root, npy_path),
                "label_space": "inverse_relative",
                "height": str(NATIVE_HW[0]),
                "width": str(NATIVE_HW[1]),
                "teacher_source": "LOD_RGB_normal_DAv2L",
            }
        )
        stats = array_stats(pred)
        stats_rows.append(stats)
        print(f"[LABEL] {index}/{len(rows)} {row['split']} pair={row['pair_id']} -> {npy_path}", flush=True)

    manifest_path = write_output_manifest(args, manifest_rows)
    elapsed = time.time() - start
    split_counts = Counter(row["split"] for row in manifest_rows)
    coverages = np.asarray([item["valid_positive_coverage"] for item in stats_rows], dtype=np.float64)
    means = np.asarray([item.get("mean", np.nan) for item in stats_rows], dtype=np.float64)
    run_config = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "lod_root": str(args.lod_root),
        "pair_manifest": str(args.pair_manifest),
        "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint),
        "encoder": args.encoder,
        "input_size": int(args.input_size),
        "target_hw": list(NATIVE_HW),
        "label_space": "inverse_relative",
        "teacher_source": "LOD_RGB_normal_DAv2L",
        "teacher_input": "rgb_normal_path",
        "splits": list(args.splits),
        "sample_count": len(manifest_rows),
        "max_samples": args.max_samples,
        "max_samples_per_split": args.max_samples_per_split,
        "manifest_path": str(manifest_path),
        "device": args.device,
        "overwrite": bool(args.overwrite),
        "vis_cmap": args.vis_cmap,
        "vis_vmin_pct": args.vis_vmin_pct,
        "vis_vmax_pct": args.vis_vmax_pct,
        "elapsed_seconds": elapsed,
    }
    run_summary = {
        "sample_count": len(manifest_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "elapsed_seconds": elapsed,
        "target_shape": list(NATIVE_HW),
        "valid_positive_coverage": {
            "mean": float(np.nanmean(coverages)),
            "min": float(np.nanmin(coverages)),
            "max": float(np.nanmax(coverages)),
        },
        "target_value_mean": {
            "mean": float(np.nanmean(means)),
            "min": float(np.nanmin(means)),
            "max": float(np.nanmax(means)),
        },
    }
    write_json(args.output_root / "run_config.json", run_config)
    write_json(args.output_root / "run_summary.json", run_summary)
    print(f"[OK] labels={len(manifest_rows)} manifest={manifest_path} elapsed={elapsed:.1f}s", flush=True)
    cleanup_if_requested(args)


def main() -> None:
    args = parse_args()
    args.lod_root = args.lod_root.expanduser().resolve()
    args.pair_manifest = args.pair_manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.input_size % 14 != 0:
        raise ValueError("--input-size must be a multiple of 14")
    for input_size in args.sweep_input_sizes:
        if input_size % 14 != 0:
            raise ValueError("--sweep-input-sizes values must be multiples of 14")
    if args.input_size_sweep:
        run_sweep(args)
    else:
        run_label_build(args)


if __name__ == "__main__":
    main()
