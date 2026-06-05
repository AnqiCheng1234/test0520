#!/usr/bin/env python3
"""Build ROD-night DAv2-L teacher-bright inverse-relative pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from finetune_stf.dataset.rod_raw_rgb import (  # noqa: E402
    PIPELINE_PARAMS,
    ROD_NATIVE_HW,
    teacher_bright_degreen_v1,
    unpack_raw24,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


DEFAULT_ROD_ROOT = Path("/mnt/drive/3333_raw/ROD")
DEFAULT_OUTPUT_ROOT = DEFAULT_ROD_ROOT / "pseudo_depth_dav2l_night_teacherbright_rel_1440x928"
DEFAULT_CHECKPOINT = Path("/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth")
MANIFEST_NAME = "rod_night_dav2_rel_manifest.csv"
RAW_EXPECTED_BYTES = 1856 * 2880 * 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROD-night teacher-bright DAv2-L pseudo labels.")
    parser.add_argument("--rod-root", type=Path, default=DEFAULT_ROD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--input-size", type=int, default=924)
    parser.add_argument("--splits", nargs="+", default=["00Train", "01Valid"])
    parser.add_argument("--sample-prefix", default="night")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--input-size-sweep",
        action="store_true",
        help="Write a small comparison set for --sweep-input-sizes instead of the formal label manifest.",
    )
    parser.add_argument("--sweep-input-sizes", nargs="+", type=int, default=[700, 924])
    parser.add_argument("--vis-cmap", default="magma_r")
    parser.add_argument("--vis-vmin-pct", type=float, default=1.0)
    parser.add_argument("--vis-vmax-pct", type=float, default=99.0)
    parser.add_argument(
        "--cleanup-success",
        action="store_true",
        help="Remove output-root after a successful run when output-root is clearly temporary.",
    )
    return parser.parse_args()


def raw_path_for_split(rod_root: Path, split: str, sample_prefix: str) -> list[Path]:
    raw_dir = rod_root / f"{split}-raws" / split
    paths = sorted(raw_dir.glob(f"{sample_prefix}-*.raw"))
    if not paths:
        raise FileNotFoundError(f"No RAW files found under {raw_dir} with prefix {sample_prefix!r}")
    return paths


def rggb_path_for_raw(rod_root: Path, split: str, raw_path: Path) -> Path:
    return rod_root / f"{split}-rggb" / split / f"{raw_path.stem}.npy"


def collect_raw_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for split in args.splits:
        split_paths = raw_path_for_split(args.rod_root, split, args.sample_prefix)
        if args.max_samples_per_split is not None:
            split_paths = split_paths[: args.max_samples_per_split]
        selected.extend((split, path) for path in split_paths)
    if args.max_samples is not None:
        selected = selected[: args.max_samples]
    if not selected:
        raise ValueError("No ROD raw paths selected")
    return selected


def load_model(args: argparse.Namespace) -> DepthAnythingV2:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device).eval()
    return model


def infer_teacher(model: DepthAnythingV2, raw_path: Path, input_size: int) -> np.ndarray:
    raw = unpack_raw24(raw_path)
    teacher_rgb = teacher_bright_degreen_v1(raw)
    teacher_bgr = cv2.cvtColor(teacher_rgb, cv2.COLOR_RGB2BGR)
    pred = model.infer_image(teacher_bgr, input_size=input_size).astype(np.float32, copy=False)
    if pred.shape != ROD_NATIVE_HW:
        raise ValueError(f"Expected prediction shape {ROD_NATIVE_HW}, got {tuple(pred.shape)} for {raw_path}")
    if not np.isfinite(pred).all():
        raise ValueError(f"Prediction contains non-finite values for {raw_path}")
    return pred


def colorize_prediction(pred: np.ndarray, cmap_name: str, vmin_pct: float, vmax_pct: float) -> np.ndarray:
    valid = pred[np.isfinite(pred)]
    if valid.size == 0:
        scaled = np.zeros_like(pred, dtype=np.uint8)
    else:
        vmin = float(np.percentile(valid, vmin_pct))
        vmax = float(np.percentile(valid, vmax_pct))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        scaled = np.clip((pred - vmin) / (vmax - vmin), 0.0, 1.0)
        if str(cmap_name).endswith("_r"):
            scaled = 1.0 - scaled
        scaled = (scaled * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_MAGMA)
    return colored


def rel_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_run_config(args: argparse.Namespace, sample_count: int, elapsed: float) -> None:
    run_config = {
        "rod_root": str(args.rod_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "splits": list(args.splits),
        "sample_prefixes": [args.sample_prefix],
        "encoder": args.encoder,
        "input_size": int(args.input_size),
        "target_hw": list(ROD_NATIVE_HW),
        "label_space": "inverse_relative",
        "teacher_rgb_pipeline": "teacher_bright_degreen_v1",
        "teacher_rgb_pipeline_params": PIPELINE_PARAMS["teacher_bright_degreen_v1"],
        "raw_bayer": "true_rggb",
        "raw_projection": "R_meanGrGb_B",
        "vis_cmap": args.vis_cmap,
        "vis_vmin_pct": args.vis_vmin_pct,
        "vis_vmax_pct": args.vis_vmax_pct,
        "max_samples": args.max_samples,
        "max_samples_per_split": args.max_samples_per_split,
        "overwrite": bool(args.overwrite),
        "device": args.device,
        "manifest_path": str((args.output_root / MANIFEST_NAME).resolve()),
        "sample_count": int(sample_count),
        "elapsed_seconds": float(elapsed),
    }
    with (args.output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)


def write_manifest(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    manifest_path = args.output_root / MANIFEST_NAME
    fieldnames = [
        "sample_id",
        "split",
        "scene",
        "raw24_path",
        "rggb_path",
        "pseudo_depth_path",
        "label_space",
        "height",
        "width",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_raw_size(path: Path) -> None:
    size = path.stat().st_size
    if size != RAW_EXPECTED_BYTES:
        raise ValueError(f"{path} has {size} bytes, expected {RAW_EXPECTED_BYTES}")


def run_sweep(args: argparse.Namespace) -> None:
    selected = collect_raw_paths(args)
    model = load_model(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    start = time.time()
    for split, raw_path in selected:
        assert_raw_size(raw_path)
        for input_size in args.sweep_input_sizes:
            pred = infer_teacher(model, raw_path, input_size)
            stem = f"{split}_{raw_path.stem}_input{input_size}"
            npy_path = args.output_root / f"{stem}.npy"
            png_path = args.output_root / f"{stem}_spectral_r.png"
            np.save(npy_path, pred)
            cv2.imwrite(str(png_path), colorize_prediction(pred, args.vis_cmap, args.vis_vmin_pct, args.vis_vmax_pct))
            summary.append(
                {
                    "split": split,
                    "sample_id": raw_path.stem,
                    "input_size": int(input_size),
                    "npy_path": str(npy_path),
                    "png_path": str(png_path),
                    "min": float(pred.min()),
                    "max": float(pred.max()),
                    "mean": float(pred.mean()),
                }
            )
            print(f"[SWEEP] {split} {raw_path.name} input_size={input_size} saved={npy_path}", flush=True)
    with (args.output_root / "input_size_sweep_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "elapsed_seconds": time.time() - start,
                "rows": summary,
                "sweep_input_sizes": args.sweep_input_sizes,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def run_label_build(args: argparse.Namespace) -> None:
    selected = collect_raw_paths(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    rows = []
    start = time.time()
    for idx, (split, raw_path) in enumerate(selected, start=1):
        assert_raw_size(raw_path)
        label_dir = args.output_root / split
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = label_dir / f"{raw_path.stem}.npy"
        if out_path.exists() and not args.overwrite:
            pred = np.load(out_path)
            if pred.shape != ROD_NATIVE_HW:
                raise ValueError(f"Existing label has wrong shape {pred.shape}: {out_path}")
        else:
            pred = infer_teacher(model, raw_path, args.input_size)
            np.save(out_path, pred)

        rggb_path = rggb_path_for_raw(args.rod_root, split, raw_path)
        if not rggb_path.is_file():
            raise FileNotFoundError(f"Missing ROD RGGB path for manifest: {rggb_path}")
        if not str(raw_path.resolve()).startswith(str(args.rod_root.resolve())):
            raise ValueError(f"RAW path is outside ROD root: {raw_path}")
        rows.append(
            {
                "sample_id": raw_path.stem,
                "split": split,
                "scene": split,
                "raw24_path": rel_to_root(args.rod_root, raw_path),
                "rggb_path": rel_to_root(args.rod_root, rggb_path),
                "pseudo_depth_path": rel_to_root(args.rod_root, out_path),
                "label_space": "inverse_relative",
                "height": str(ROD_NATIVE_HW[0]),
                "width": str(ROD_NATIVE_HW[1]),
            }
        )
        print(f"[LABEL] {idx}/{len(selected)} {split} {raw_path.name} -> {out_path}", flush=True)

    write_manifest(args, rows)
    write_run_config(args, sample_count=len(rows), elapsed=time.time() - start)
    print(f"[OK] wrote labels={len(rows)} manifest={args.output_root / MANIFEST_NAME}", flush=True)


def main() -> None:
    args = parse_args()
    args.rod_root = args.rod_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.input_size % 14 != 0:
        raise ValueError("--input-size must be a multiple of 14")
    if args.input_size_sweep:
        run_sweep(args)
    else:
        run_label_build(args)
    if args.cleanup_success:
        output_str = str(args.output_root)
        if not any(marker in output_str for marker in ("codex_smoke", "debug", "tmp", "smoke")):
            raise ValueError(f"Refusing to cleanup non-temporary output path: {args.output_root}")
        shutil.rmtree(args.output_root)
        print(f"[OK] removed temporary output {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
