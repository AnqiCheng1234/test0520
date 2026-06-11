#!/usr/bin/env python3
"""Build small ROD DNG + RawPy default RGB preview panels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from finetune_stf.dataset.rod_raw_rgb import (  # noqa: E402
    PIPELINE_PARAMS,
    render_pipeline,
    unpack_raw24,
)
from finetune_stf.dataset.rod_rawpy_render import (  # noqa: E402
    DNG_CFA_PATTERN,
    get_rawpy_runtime_versions,
    rawpy_postprocess_default,
    resize_to_native,
    write_dng,
)
from finetune_stf.dataset.rod_raw_student_rgb import (  # noqa: E402
    DEFAULT_ROD_NIGHT_TEACHER_MANIFEST,
    DEFAULT_ROD_ROOT,
    _load_rod_manifest_rows,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEFAULT_DAV2L_CHECKPOINT = Path("/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROD RawPy default debug preview panels.")
    parser.add_argument("--rod-root", type=Path, default=Path(DEFAULT_ROD_ROOT))
    parser.add_argument("--manifest-path", type=Path, default=Path(DEFAULT_ROD_NIGHT_TEACHER_MANIFEST))
    parser.add_argument("--split", default="01Valid")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection", choices=["spread", "first"], default="spread")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-dng", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-direct-rawpy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--skip-dav2", action="store_true", help="Only build RGB panels, without DAV2-L predictions.")
    parser.add_argument("--dav2-checkpoint", type=Path, default=DEFAULT_DAV2L_CHECKPOINT)
    parser.add_argument("--dav2-encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--dav2-device", default="cuda")
    parser.add_argument("--dav2-input-size", type=int, default=924)
    parser.add_argument("--dav2-cmap", default="Spectral_r")
    parser.add_argument("--dav2-vmin-pct", type=float, default=1.0)
    parser.add_argument("--dav2-vmax-pct", type=float, default=99.0)
    return parser.parse_args()


def default_output_root(limit: int) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M")
    return PROJECT_ROOT / "finetune_stf" / "analysis" / "rod_rawpy_debug" / f"{stamp}_rod_rawpy_preview_n{limit}"


def select_rows(rows: list[dict[str, object]], limit: int, selection: str) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}")
    if limit >= len(rows):
        return rows
    if selection == "first":
        return rows[:limit]
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64).tolist()
    return [rows[int(idx)] for idx in indices]


def try_direct_rawpy(raw_path: Path) -> dict[str, object]:
    try:
        rgb = rawpy_postprocess_default(raw_path)
        return {"ok": True, "shape": list(rgb.shape), "dtype": str(rgb.dtype)}
    except Exception as exc:  # LibRaw returns precise subclasses here.
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def image_stats(rgb: np.ndarray) -> dict[str, object]:
    values = rgb.astype(np.float32, copy=False)
    return {
        "shape": list(rgb.shape),
        "dtype": str(rgb.dtype),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "percentiles": {
            f"p{q:g}": float(np.percentile(values, q)) for q in (0, 1, 50, 99, 99.9, 100)
        },
        "channel_mean": [float(values[..., idx].mean()) for idx in range(3)],
    }


def depth_stats(depth: np.ndarray) -> dict[str, object]:
    values = depth.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "shape": list(depth.shape),
            "dtype": str(depth.dtype),
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "percentiles": {},
        }
    return {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "percentiles": {f"p{q:g}": float(np.percentile(finite, q)) for q in (0, 1, 50, 99, 100)},
    }


def resolve_colormap_name(cmap_name: str) -> str:
    if cmap_name in colormaps:
        return cmap_name
    requested = str(cmap_name).lower()
    for name in colormaps:
        if name.lower() == requested:
            return name
    raise ValueError(f"Unknown matplotlib colormap: {cmap_name!r}")


def load_dav2_model(checkpoint: Path, encoder: str, device: str) -> DepthAnythingV2:
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for DAV2-L but torch.cuda.is_available() is false")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAV2 checkpoint: {checkpoint}")
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def infer_dav2_rgb(model: DepthAnythingV2, rgb: np.ndarray, input_size: int) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected uint8 RGB image, got shape={rgb.shape} dtype={rgb.dtype}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with torch.inference_mode():
        pred = model.infer_image(bgr, input_size=int(input_size)).astype(np.float32, copy=False)
    if pred.shape != rgb.shape[:2]:
        raise ValueError(f"DAV2 output shape {pred.shape} != image shape {rgb.shape[:2]}")
    if not np.isfinite(pred).all():
        raise ValueError("DAV2 output contains non-finite values")
    return pred


def colorize_depth_shared(depth: np.ndarray, *, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    denom = max(float(vmax) - float(vmin), 1e-9)
    norm = np.clip((depth.astype(np.float32, copy=False) - float(vmin)) / denom, 0.0, 1.0)
    cmap = colormaps.get_cmap(cmap_name)
    rgb = np.full((*depth.shape, 3), 235, dtype=np.uint8)
    finite = np.isfinite(depth)
    rgb[finite] = np.clip(cmap(norm[finite])[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)


def make_tile(image: np.ndarray, label: str, width: int) -> Image.Image:
    h, w = image.shape[:2]
    tile_h = max(1, int(round(h * width / w)))
    resized = cv2.resize(image, (width, tile_h), interpolation=cv2.INTER_AREA)
    canvas = Image.new("RGB", (width, tile_h + 34), "white")
    canvas.paste(Image.fromarray(resized), (0, 34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), label, fill=(20, 20, 20), font=font)
    return canvas


def make_panel(
    *,
    sample_id: str,
    student_rgb: np.ndarray,
    teacher_rgb: np.ndarray,
    rawpy_rgb: np.ndarray,
    student_depth_vis: np.ndarray | None = None,
    teacher_depth_vis: np.ndarray | None = None,
    rawpy_depth_vis: np.ndarray | None = None,
    dav2_cmap: str | None = None,
    out_path: Path,
    tile_width: int,
) -> None:
    if student_depth_vis is None or teacher_depth_vis is None or rawpy_depth_vis is None:
        labels = [
            f"{sample_id} student_dark",
            "teacher_bright",
            "DNG rawpy.postprocess()",
        ]
        images = [student_rgb, teacher_rgb, rawpy_rgb]
    else:
        cmap_label = dav2_cmap or "DAV2-L"
        labels = [
            f"{sample_id} student_dark RGB",
            f"student_dark DAV2-L {cmap_label}",
            "teacher_bright RGB",
            f"teacher_bright DAV2-L {cmap_label}",
            "DNG rawpy RGB",
            f"rawpy DAV2-L {cmap_label}",
        ]
        images = [
            student_rgb,
            student_depth_vis,
            teacher_rgb,
            teacher_depth_vis,
            rawpy_rgb,
            rawpy_depth_vis,
        ]
    tiles = [make_tile(image, label, tile_width) for image, label in zip(images, labels)]
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), "white")
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, 0))
        x += tile.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_contact_sheet(panel_paths: list[Path], out_path: Path, columns: int = 2) -> None:
    if not panel_paths:
        return
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    panel_w = max(panel.width for panel in panels)
    panel_h = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (panel_w * columns, panel_h * rows), "white")
    for idx, panel in enumerate(panels):
        x = (idx % columns) * panel_w
        y = (idx // columns) * panel_h
        sheet.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def safe_remove_dng(path: Path, output_root: Path) -> None:
    markers = ("debug", "tmp", "smoke", "codex_smoke")
    if not any(marker in str(output_root) for marker in markers):
        raise ValueError(f"Refusing to remove DNG outside a clearly temporary/debug output root: {output_root}")
    path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root(args.limit)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists; pass --overwrite to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_rod_manifest_rows(args.manifest_path, args.rod_root, args.split)
    selected = select_rows(rows, args.limit, args.selection)
    if not selected:
        raise ValueError(f"No rows selected for split={args.split}")

    dav2_enabled = not bool(args.skip_dav2)
    dav2_model: DepthAnythingV2 | None = None
    dav2_cmap: str | None = None
    if dav2_enabled:
        if int(args.dav2_input_size) % 14 != 0:
            raise ValueError("--dav2-input-size must be a multiple of 14")
        dav2_cmap = resolve_colormap_name(args.dav2_cmap)
        if str(args.dav2_device) == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        dav2_model = load_dav2_model(args.dav2_checkpoint, args.dav2_encoder, args.dav2_device)

    dirs = {
        "dng": output_root / "dng_uint16",
        "rawpy_full": output_root / "rawpy_default_fullres",
        "rawpy_native": output_root / "rawpy_default_928x1440",
        "student": output_root / "student_dark_degreen_v1",
        "teacher": output_root / "teacher_bright_degreen_v1",
        "panel": output_root / "panels",
    }
    if dav2_enabled:
        cmap_token = str(dav2_cmap).lower()
        dirs.update(
            {
                "student_dav2_pred": output_root / "dav2l_pred" / "student_dark_degreen_v1",
                "teacher_dav2_pred": output_root / "dav2l_pred" / "teacher_bright_degreen_v1",
                "rawpy_dav2_pred": output_root / "dav2l_pred" / "rawpy_default_928x1440",
                "student_dav2_vis": output_root / f"dav2l_vis_{cmap_token}" / "student_dark_degreen_v1",
                "teacher_dav2_vis": output_root / f"dav2l_vis_{cmap_token}" / "teacher_bright_degreen_v1",
                "rawpy_dav2_vis": output_root / f"dav2l_vis_{cmap_token}" / "rawpy_default_928x1440",
            }
        )
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest_rawpy_preview.jsonl"
    panel_paths: list[Path] = []
    records: list[dict[str, object]] = []
    dav2_values: list[np.ndarray] = []
    for idx, row in enumerate(selected):
        sample_id = str(row["sample_id"])
        split = str(row["split"])
        raw_path = Path(row["raw_path"])
        stem = f"{idx:04d}_{split}_{sample_id}"
        dng_path = dirs["dng"] / f"{stem}.dng"
        rawpy_full_path = dirs["rawpy_full"] / f"{stem}.png"
        rawpy_native_path = dirs["rawpy_native"] / f"{stem}.png"
        student_path = dirs["student"] / f"{stem}.png"
        teacher_path = dirs["teacher"] / f"{stem}.png"
        panel_path = dirs["panel"] / f"{stem}_panel.png"

        direct_status = try_direct_rawpy(raw_path) if args.check_direct_rawpy else {"checked": False}
        raw_norm = unpack_raw24(raw_path)
        mosaic16 = write_dng(raw_norm, dng_path)
        rawpy_full = rawpy_postprocess_default(dng_path)
        rawpy_native = resize_to_native(rawpy_full)
        student_rgb = render_pipeline(raw_norm, "student_dark_degreen_v1")
        teacher_rgb = render_pipeline(raw_norm, "teacher_bright_degreen_v1")

        save_rgb(rawpy_full_path, rawpy_full)
        save_rgb(rawpy_native_path, rawpy_native)
        save_rgb(student_path, student_rgb)
        save_rgb(teacher_path, teacher_rgb)

        record = {
            "sample_id": sample_id,
            "split": split,
            "raw_path": str(raw_path),
            "dng_path": str(dng_path) if args.keep_dng else None,
            "rawpy_fullres_rgb_path": str(rawpy_full_path),
            "rawpy_928x1440_rgb_path": str(rawpy_native_path),
            "student_dark_degreen_v1_path": str(student_path),
            "teacher_bright_degreen_v1_path": str(teacher_path),
            "panel_path": str(panel_path),
            "direct_rawpy_read_raw24": direct_status,
            "raw_shape": list(raw_norm.shape),
            "raw_norm_min": float(raw_norm.min()),
            "raw_norm_max": float(raw_norm.max()),
            "raw_quantization": "existing_unpack_raw24_float01_to_uint16_65535",
            "mosaic16_min": int(mosaic16.min()),
            "mosaic16_max": int(mosaic16.max()),
            "cfa": "RGGB",
            "black_level": 0,
            "white_level": 65535,
            "postprocess_kwargs": {},
            "rawpy_fullres_stats": image_stats(rawpy_full),
            "rawpy_928x1440_stats": image_stats(rawpy_native),
            "student_dark_stats": image_stats(student_rgb),
            "teacher_bright_stats": image_stats(teacher_rgb),
        }

        if dav2_enabled:
            if dav2_model is None:
                raise RuntimeError("DAV2-L model was not initialized")
            student_pred = infer_dav2_rgb(dav2_model, student_rgb, args.dav2_input_size)
            teacher_pred = infer_dav2_rgb(dav2_model, teacher_rgb, args.dav2_input_size)
            rawpy_pred = infer_dav2_rgb(dav2_model, rawpy_native, args.dav2_input_size)
            student_pred_path = dirs["student_dav2_pred"] / f"{stem}.npy"
            teacher_pred_path = dirs["teacher_dav2_pred"] / f"{stem}.npy"
            rawpy_pred_path = dirs["rawpy_dav2_pred"] / f"{stem}.npy"
            np.save(student_pred_path, student_pred.astype(np.float32, copy=False))
            np.save(teacher_pred_path, teacher_pred.astype(np.float32, copy=False))
            np.save(rawpy_pred_path, rawpy_pred.astype(np.float32, copy=False))
            for pred in (student_pred, teacher_pred, rawpy_pred):
                finite = pred[np.isfinite(pred)]
                if finite.size:
                    dav2_values.append(finite.astype(np.float32, copy=False))
            record.update(
                {
                    "student_dark_dav2l_pred_path": str(student_pred_path),
                    "teacher_bright_dav2l_pred_path": str(teacher_pred_path),
                    "rawpy_928x1440_dav2l_pred_path": str(rawpy_pred_path),
                    "student_dark_dav2l_stats": depth_stats(student_pred),
                    "teacher_bright_dav2l_stats": depth_stats(teacher_pred),
                    "rawpy_928x1440_dav2l_stats": depth_stats(rawpy_pred),
                }
            )

        if not args.keep_dng:
            safe_remove_dng(dng_path, output_root)
        records.append(record)
        status = "rgb+dav2" if dav2_enabled else "rgb"
        print(
            f"[{idx + 1}/{len(selected)}] {split}/{sample_id} status={status} "
            f"rawpy_mean={record['rawpy_928x1440_stats']['mean']:.3f}",
            flush=True,
        )

    dav2_vis_range: dict[str, float] | None = None
    if dav2_enabled:
        if not dav2_values:
            raise ValueError("DAV2-L produced no finite values for visualization")
        dav2_concat = np.concatenate([values.reshape(-1) for values in dav2_values]).astype(np.float32, copy=False)
        vmin = float(np.percentile(dav2_concat, args.dav2_vmin_pct))
        vmax = float(np.percentile(dav2_concat, args.dav2_vmax_pct))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        dav2_vis_range = {
            "vmin": vmin,
            "vmax": vmax,
            "vmin_pct": float(args.dav2_vmin_pct),
            "vmax_pct": float(args.dav2_vmax_pct),
        }

    for record in records:
        sample_id = str(record["sample_id"])
        stem = Path(str(record["panel_path"])).name.replace("_panel.png", "")
        student_rgb = iio.imread(record["student_dark_degreen_v1_path"])
        teacher_rgb = iio.imread(record["teacher_bright_degreen_v1_path"])
        rawpy_rgb = iio.imread(record["rawpy_928x1440_rgb_path"])

        if dav2_enabled:
            if dav2_cmap is None or dav2_vis_range is None:
                raise RuntimeError("DAV2-L visualization settings were not initialized")
            student_pred = np.load(record["student_dark_dav2l_pred_path"])
            teacher_pred = np.load(record["teacher_bright_dav2l_pred_path"])
            rawpy_pred = np.load(record["rawpy_928x1440_dav2l_pred_path"])
            student_vis = colorize_depth_shared(
                student_pred,
                vmin=dav2_vis_range["vmin"],
                vmax=dav2_vis_range["vmax"],
                cmap_name=dav2_cmap,
            )
            teacher_vis = colorize_depth_shared(
                teacher_pred,
                vmin=dav2_vis_range["vmin"],
                vmax=dav2_vis_range["vmax"],
                cmap_name=dav2_cmap,
            )
            rawpy_vis = colorize_depth_shared(
                rawpy_pred,
                vmin=dav2_vis_range["vmin"],
                vmax=dav2_vis_range["vmax"],
                cmap_name=dav2_cmap,
            )
            student_vis_path = dirs["student_dav2_vis"] / f"{stem}.png"
            teacher_vis_path = dirs["teacher_dav2_vis"] / f"{stem}.png"
            rawpy_vis_path = dirs["rawpy_dav2_vis"] / f"{stem}.png"
            save_rgb(student_vis_path, student_vis)
            save_rgb(teacher_vis_path, teacher_vis)
            save_rgb(rawpy_vis_path, rawpy_vis)
            record.update(
                {
                    "student_dark_dav2l_vis_path": str(student_vis_path),
                    "teacher_bright_dav2l_vis_path": str(teacher_vis_path),
                    "rawpy_928x1440_dav2l_vis_path": str(rawpy_vis_path),
                    "dav2l_vis_cmap": dav2_cmap,
                    "dav2l_vis_range": dav2_vis_range,
                }
            )
            make_panel(
                sample_id=sample_id,
                student_rgb=student_rgb,
                teacher_rgb=teacher_rgb,
                rawpy_rgb=rawpy_rgb,
                student_depth_vis=student_vis,
                teacher_depth_vis=teacher_vis,
                rawpy_depth_vis=rawpy_vis,
                dav2_cmap=dav2_cmap,
                out_path=Path(str(record["panel_path"])),
                tile_width=args.tile_width,
            )
        else:
            make_panel(
                sample_id=sample_id,
                student_rgb=student_rgb,
                teacher_rgb=teacher_rgb,
                rawpy_rgb=rawpy_rgb,
                out_path=Path(str(record["panel_path"])),
                tile_width=args.tile_width,
            )
        panel_paths.append(Path(str(record["panel_path"])))

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, sort_keys=True) + "\n")

    contact_sheet = output_root / "rod_rawpy_preview_contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet)

    rawpy_versions = get_rawpy_runtime_versions()
    run_config = {
        "script": str(Path(__file__).resolve()),
        "rod_root": str(Path(args.rod_root).expanduser().resolve()),
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "output_root": str(output_root.resolve()),
        "split": args.split,
        "limit": int(args.limit),
        "selection": args.selection,
        "sample_count": len(selected),
        "check_direct_rawpy": bool(args.check_direct_rawpy),
        "keep_dng": bool(args.keep_dng),
        "rawpy_version": rawpy_versions["rawpy_version"],
        "libraw_version": rawpy_versions["libraw_version"],
        "tifffile_version": rawpy_versions["tifffile_version"],
        "imageio_version": getattr(sys.modules.get("imageio"), "__version__", "unknown"),
        "student_pipeline_params": PIPELINE_PARAMS["student_dark_degreen_v1"],
        "teacher_pipeline_params": PIPELINE_PARAMS["teacher_bright_degreen_v1"],
        "dav2l_prediction": {
            "enabled": dav2_enabled,
            "checkpoint": str(Path(args.dav2_checkpoint).expanduser().resolve()) if dav2_enabled else None,
            "encoder": args.dav2_encoder if dav2_enabled else None,
            "device": args.dav2_device if dav2_enabled else None,
            "input_size": int(args.dav2_input_size) if dav2_enabled else None,
            "cmap": dav2_cmap if dav2_enabled else None,
            "shared_visualization_range": dav2_vis_range,
            "label_space": "relative_inverse_depth_from_dav2l",
            "input_views": [
                "student_dark_degreen_v1",
                "teacher_bright_degreen_v1",
                "rawpy_default_928x1440",
            ],
        },
        "dng_metadata": {
            "cfa": "RGGB",
            "cfa_pattern": list(DNG_CFA_PATTERN),
            "black_level": 0,
            "white_level": 65535,
            "color_matrix_1": "standard_xyz_to_srgb_d65",
            "as_shot_neutral": [1, 1, 1],
            "calibration_illuminant_1": "D65",
        },
        "rawpy_postprocess_kwargs": {},
        "contact_sheet": str(contact_sheet),
        "manifest": str(manifest_path),
    }
    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    rawpy_means = [float(item["rawpy_928x1440_stats"]["mean"]) for item in records]
    summary = {
        "output_root": str(output_root.resolve()),
        "contact_sheet": str(contact_sheet),
        "manifest": str(manifest_path),
        "sample_count": len(records),
        "direct_rawpy_ok_count": sum(1 for item in records if item["direct_rawpy_read_raw24"].get("ok") is True),
        "rawpy_928x1440_mean_avg": float(np.mean(rawpy_means)),
        "rawpy_928x1440_mean_min": float(np.min(rawpy_means)),
        "rawpy_928x1440_mean_max": float(np.max(rawpy_means)),
        "dav2l_enabled": dav2_enabled,
        "dav2l_cmap": dav2_cmap if dav2_enabled else None,
        "dav2l_shared_visualization_range": dav2_vis_range,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[DONE] output_root={output_root}", flush=True)
    print(f"[DONE] contact_sheet={contact_sheet}", flush=True)
    print(f"[DONE] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
