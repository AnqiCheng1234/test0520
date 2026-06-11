#!/usr/bin/env python3
"""Build LOD RGB/RAW dark-normal panels with LinearRaw DNG + RawPy output."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import rawpy
import tifffile
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_true import (  # noqa: E402
    DEFAULT_LOD_TRUE_MANIFEST,
    DEFAULT_LOD_TRUE_ROOT,
    LOD_TRUE_NATIVE_HW,
    _load_lod_true_manifest_rows,
)


DNG_COLOR_MATRIX_XYZ_TO_SRGB = (
    3240454,
    1000000,
    -1537138,
    1000000,
    -498531,
    1000000,
    -969266,
    1000000,
    1876011,
    1000000,
    41556,
    1000000,
    55643,
    1000000,
    -204026,
    1000000,
    1057225,
    1000000,
)
DNG_AS_SHOT_NEUTRAL = (1, 1, 1, 1, 1, 1)
DNG_PHOTOMETRIC_LINEAR_RAW = 34892
RAWPY_POSTPROCESS_KWARGS = {"use_auto_wb": True}
RAWPY_POSTPROCESS_LABEL = "rawpy.postprocess(use_auto_wb=True)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LOD RawPy debug preview panels.")
    parser.add_argument("--lod-root", type=Path, default=Path(DEFAULT_LOD_TRUE_ROOT))
    parser.add_argument("--manifest-path", type=Path, default=Path(DEFAULT_LOD_TRUE_MANIFEST))
    parser.add_argument("--split", default="01Valid")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection", choices=["spread", "first"], default="spread")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-dng", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-direct-rawpy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-width", type=int, default=300)
    return parser.parse_args()


def default_output_root(limit: int) -> Path:
    stamp = datetime.now().strftime("%m%d_%H%M")
    return PROJECT_ROOT / "finetune_stf" / "analysis" / "lod_rawpy_debug" / f"{stamp}_lod_rawpy_preview_n{limit}"


def select_rows(rows: list[dict[str, object]], limit: int, selection: str) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}")
    if limit >= len(rows):
        return rows
    if selection == "first":
        return rows[:limit]
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64).tolist()
    return [rows[int(idx)] for idx in indices]


def load_rgb_u8(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV failed to read RGB image: {path}")
    if tuple(image_bgr.shape) != (*LOD_TRUE_NATIVE_HW, 3):
        raise ValueError(f"Expected LOD RGB shape {(*LOD_TRUE_NATIVE_HW, 3)}, got {tuple(image_bgr.shape)}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_raw_rgb16(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw_bgr is None:
        raise ValueError(f"OpenCV failed to read RAW image: {path}")
    if tuple(raw_bgr.shape) != (*LOD_TRUE_NATIVE_HW, 3):
        raise ValueError(f"Expected LOD RAW shape {(*LOD_TRUE_NATIVE_HW, 3)}, got {tuple(raw_bgr.shape)}")
    if raw_bgr.dtype != np.uint16:
        raise ValueError(f"Expected LOD RAW dtype uint16, got {raw_bgr.dtype}: {path}")
    raw_rgb16 = raw_bgr[..., ::-1]
    raw_norm = raw_rgb16.astype(np.float32) / 65535.0
    return raw_rgb16, raw_norm


def dng_extratags() -> list[tuple[int, str, int, object, bool]]:
    return [
        (271, "s", 0, "LOD", False),  # Make
        (272, "s", 0, "LOD_RAW_RGB16_LINEAR_DEBUG", False),  # Model
        (274, "H", 1, 1, False),  # Orientation
        (50706, "B", 4, (1, 4, 0, 0), False),  # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), False),  # DNGBackwardVersion
        (50708, "s", 0, "LOD_RAW_RGB16_LINEAR_DEBUG", False),  # UniqueCameraModel
        (50710, "B", 3, (0, 1, 2), False),  # CFAPlaneColor; used here as RGB color plane metadata
        (50714, "H", 3, (0, 0, 0), False),  # BlackLevel
        (50717, "H", 3, (65535, 65535, 65535), False),  # WhiteLevel
        (50721, "2i", 9, DNG_COLOR_MATRIX_XYZ_TO_SRGB, False),  # ColorMatrix1
        (50728, "2I", 3, DNG_AS_SHOT_NEUTRAL, False),  # AsShotNeutral
        (50778, "H", 1, 21, False),  # CalibrationIlluminant1, D65
    ]


def write_linear_raw_dng(raw_rgb16: np.ndarray, dng_path: Path) -> None:
    if tuple(raw_rgb16.shape) != (*LOD_TRUE_NATIVE_HW, 3):
        raise ValueError(f"Expected LOD RAW RGB16 HWC image, got {tuple(raw_rgb16.shape)}")
    if raw_rgb16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 LOD RAW image, got {raw_rgb16.dtype}")
    dng_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        dng_path,
        raw_rgb16,
        photometric=DNG_PHOTOMETRIC_LINEAR_RAW,
        compression=None,
        metadata=None,
        extratags=dng_extratags(),
    )


def rawpy_postprocess_lod(dng_path: Path) -> np.ndarray:
    with rawpy.imread(str(dng_path)) as raw:
        return raw.postprocess(**RAWPY_POSTPROCESS_KWARGS)


def try_direct_rawpy(path: Path) -> dict[str, object]:
    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess()
        return {"ok": True, "shape": list(rgb.shape), "dtype": str(rgb.dtype)}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


def linear_preview(raw_norm: np.ndarray) -> np.ndarray:
    return np.clip(raw_norm * 255.0, 0, 255).astype(np.uint8)


def robust_preview(raw_norm: np.ndarray) -> np.ndarray:
    values = raw_norm[np.isfinite(raw_norm)]
    if values.size == 0:
        return np.zeros(raw_norm.shape, dtype=np.uint8)
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(raw_norm))
        hi = float(np.nanmax(raw_norm))
    if hi <= lo:
        return np.zeros(raw_norm.shape, dtype=np.uint8)
    stretched = np.clip((raw_norm - lo) / (hi - lo), 0.0, 1.0)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def image_stats(image: np.ndarray) -> dict[str, object]:
    values = image.astype(np.float32, copy=False)
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "percentiles": {f"p{q:g}": float(np.percentile(values, q)) for q in (0, 1, 50, 99, 99.9, 100)},
        "channel_mean": [float(values[..., idx].mean()) for idx in range(3)],
    }


def raw_norm_stats(raw_norm: np.ndarray) -> dict[str, object]:
    values = raw_norm.astype(np.float32, copy=False)
    return {
        "shape": list(raw_norm.shape),
        "dtype": str(raw_norm.dtype),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "percentiles": {f"p{q:g}": float(np.percentile(values, q)) for q in (0, 1, 50, 99, 99.9, 100)},
        "channel_mean": [float(values[..., idx].mean()) for idx in range(3)],
    }


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)


def make_tile(image: np.ndarray, label: str, width: int) -> Image.Image:
    h, w = image.shape[:2]
    tile_h = max(1, int(round(h * width / w)))
    resized = cv2.resize(image, (width, tile_h), interpolation=cv2.INTER_AREA)
    canvas = Image.new("RGB", (width, tile_h + 38), "white")
    canvas.paste(Image.fromarray(resized), (0, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), label, fill=(20, 20, 20), font=ImageFont.load_default())
    return canvas


def make_panel(sample_id: str, images: list[tuple[str, np.ndarray]], out_path: Path, tile_width: int) -> None:
    tiles = [make_tile(image, label, tile_width) for label, image in images]
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), "white")
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, 0))
        x += tile.width
    draw = ImageDraw.Draw(panel)
    draw.text((8, height - 16), sample_id, fill=(20, 20, 20), font=ImageFont.load_default())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_contact_sheet(panel_paths: list[Path], out_path: Path, columns: int = 1) -> None:
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

    rows = _load_lod_true_manifest_rows(args.manifest_path, args.lod_root, args.split)
    selected = select_rows(rows, args.limit, args.selection)
    if not selected:
        raise ValueError(f"No LOD rows selected for split={args.split}")

    dirs = {
        "dng": output_root / "linearraw_dng",
        "rgb_dark": output_root / "rgb_dark",
        "rgb_normal": output_root / "rgb_normal",
        "raw_dark_linear": output_root / "raw_dark_linear_preview",
        "raw_normal_linear": output_root / "raw_normal_linear_preview",
        "raw_dark_robust": output_root / "raw_dark_p1p99_preview",
        "raw_normal_robust": output_root / "raw_normal_p1p99_preview",
        "rawpy_dark": output_root / "raw_dark_lineardng_rawpy_auto_wb",
        "rawpy_normal": output_root / "raw_normal_lineardng_rawpy_auto_wb",
        "panel": output_root / "panels",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    panel_paths: list[Path] = []
    manifest_path = output_root / "manifest_lod_rawpy_preview.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for idx, row in enumerate(selected):
            pair_id = str(row["pair_id"])
            stem = f"{idx:04d}_{pair_id}"
            rgb_dark = load_rgb_u8(Path(row["rgb_dark_path"]))
            rgb_normal = load_rgb_u8(Path(row["rgb_normal_path"]))
            raw_dark_rgb16, raw_dark_norm = load_raw_rgb16(Path(row["raw_dark_path"]))
            raw_normal_rgb16, raw_normal_norm = load_raw_rgb16(Path(row["raw_normal_path"]))

            dng_dark_path = dirs["dng"] / f"{stem}_RAW_Dark_linearrgb.dng"
            dng_normal_path = dirs["dng"] / f"{stem}_RAW_normal_linearrgb.dng"
            write_linear_raw_dng(raw_dark_rgb16, dng_dark_path)
            write_linear_raw_dng(raw_normal_rgb16, dng_normal_path)
            rawpy_dark = rawpy_postprocess_lod(dng_dark_path)
            rawpy_normal = rawpy_postprocess_lod(dng_normal_path)

            rgb_dark_path = dirs["rgb_dark"] / f"{stem}.png"
            rgb_normal_path = dirs["rgb_normal"] / f"{stem}.png"
            raw_dark_linear_path = dirs["raw_dark_linear"] / f"{stem}.png"
            raw_normal_linear_path = dirs["raw_normal_linear"] / f"{stem}.png"
            raw_dark_robust_path = dirs["raw_dark_robust"] / f"{stem}.png"
            raw_normal_robust_path = dirs["raw_normal_robust"] / f"{stem}.png"
            rawpy_dark_path = dirs["rawpy_dark"] / f"{stem}.png"
            rawpy_normal_path = dirs["rawpy_normal"] / f"{stem}.png"
            panel_path = dirs["panel"] / f"{stem}_panel.png"

            raw_dark_linear = linear_preview(raw_dark_norm)
            raw_normal_linear = linear_preview(raw_normal_norm)
            raw_dark_robust = robust_preview(raw_dark_norm)
            raw_normal_robust = robust_preview(raw_normal_norm)
            save_rgb(rgb_dark_path, rgb_dark)
            save_rgb(rgb_normal_path, rgb_normal)
            save_rgb(raw_dark_linear_path, raw_dark_linear)
            save_rgb(raw_normal_linear_path, raw_normal_linear)
            save_rgb(raw_dark_robust_path, raw_dark_robust)
            save_rgb(raw_normal_robust_path, raw_normal_robust)
            save_rgb(rawpy_dark_path, rawpy_dark)
            save_rgb(rawpy_normal_path, rawpy_normal)

            make_panel(
                pair_id,
                [
                    ("RGB_Dark JPG", rgb_dark),
                    ("RGB_Normal JPG", rgb_normal),
                    ("RAW_Dark linear", raw_dark_linear),
                    ("RAW_Normal linear", raw_normal_linear),
                    ("RAW_Dark RawPy auto_wb", rawpy_dark),
                    ("RAW_Normal RawPy auto_wb", rawpy_normal),
                ],
                panel_path,
                args.tile_width,
            )
            panel_paths.append(panel_path)

            direct = {}
            if args.check_direct_rawpy:
                for key in ("rgb_dark_path", "rgb_normal_path", "raw_dark_path", "raw_normal_path"):
                    direct[key] = try_direct_rawpy(Path(row[key]))
            record = {
                "pair_id": pair_id,
                "split": str(row["split"]),
                "normal_id": int(row["normal_id"]),
                "dark_id": int(row["dark_id"]),
                "source_paths": {
                    "rgb_dark": str(row["rgb_dark_path"]),
                    "rgb_normal": str(row["rgb_normal_path"]),
                    "raw_dark": str(row["raw_dark_path"]),
                    "raw_normal": str(row["raw_normal_path"]),
                },
                "output_paths": {
                    "rgb_dark": str(rgb_dark_path),
                    "rgb_normal": str(rgb_normal_path),
                    "raw_dark_linear_preview": str(raw_dark_linear_path),
                    "raw_normal_linear_preview": str(raw_normal_linear_path),
                    "raw_dark_p1p99_preview": str(raw_dark_robust_path),
                    "raw_normal_p1p99_preview": str(raw_normal_robust_path),
                    "raw_dark_lineardng_rawpy_auto_wb": str(rawpy_dark_path),
                    "raw_normal_lineardng_rawpy_auto_wb": str(rawpy_normal_path),
                    "panel": str(panel_path),
                    "dng_dark": str(dng_dark_path) if args.keep_dng else None,
                    "dng_normal": str(dng_normal_path) if args.keep_dng else None,
                },
                "direct_rawpy_read_sources": direct,
                "rawpy_route": f"LOD 3-channel uint16 PNG -> LinearRaw RGB DNG photometric=34892 -> {RAWPY_POSTPROCESS_LABEL}",
                "rawpy_postprocess_kwargs": dict(RAWPY_POSTPROCESS_KWARGS),
                "semantic_warning": (
                    "LOD RAW is already 3-channel uint16 PNG, not Bayer mosaic; this is LinearRaw RGB DNG "
                    "RawPy processing, not a Bayer demosaic ISP baseline like ROD."
                ),
                "rgb_dark_stats": image_stats(rgb_dark),
                "rgb_normal_stats": image_stats(rgb_normal),
                "raw_dark_norm_stats": raw_norm_stats(raw_dark_norm),
                "raw_normal_norm_stats": raw_norm_stats(raw_normal_norm),
                "rawpy_dark_stats": image_stats(rawpy_dark),
                "rawpy_normal_stats": image_stats(rawpy_normal),
            }
            if not args.keep_dng:
                safe_remove_dng(dng_dark_path, output_root)
                safe_remove_dng(dng_normal_path, output_root)
            manifest_file.write(json.dumps(record, sort_keys=True) + "\n")
            records.append(record)
            print(
                f"[{idx + 1}/{len(selected)}] {pair_id} "
                f"rawpy_dark_mean={record['rawpy_dark_stats']['mean']:.3f} "
                f"rawpy_normal_mean={record['rawpy_normal_stats']['mean']:.3f} panel={panel_path}",
                flush=True,
            )

    contact_sheet = output_root / "lod_rawpy_preview_contact_sheet.png"
    make_contact_sheet(panel_paths, contact_sheet)

    summary = {
        "output_root": str(output_root.resolve()),
        "contact_sheet": str(contact_sheet),
        "manifest": str(manifest_path),
        "sample_count": len(records),
        "rawpy_route": f"3-channel LinearRaw RGB DNG wrapper, photometric=34892, {RAWPY_POSTPROCESS_LABEL}",
        "rawpy_postprocess_kwargs": dict(RAWPY_POSTPROCESS_KWARGS),
        "semantic_warning": (
            "This LOD output is not directly comparable to ROD Bayer DNG RawPy as a demosaic ISP; "
            "LOD RAW files are 3-channel uint16 PNGs."
        ),
        "direct_rawpy_ok_count": sum(
            1
            for record in records
            for status in record["direct_rawpy_read_sources"].values()
            if status.get("ok") is True
        ),
        "rawpy_dark_mean_avg": float(np.mean([r["rawpy_dark_stats"]["mean"] for r in records])),
        "rawpy_normal_mean_avg": float(np.mean([r["rawpy_normal_stats"]["mean"] for r in records])),
        "rgb_dark_mean_avg": float(np.mean([r["rgb_dark_stats"]["mean"] for r in records])),
        "rgb_normal_mean_avg": float(np.mean([r["rgb_normal_stats"]["mean"] for r in records])),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    run_config = {
        "script": str(Path(__file__).resolve()),
        "lod_root": str(Path(args.lod_root).expanduser().resolve()),
        "manifest_path": str(Path(args.manifest_path).expanduser().resolve()),
        "output_root": str(output_root.resolve()),
        "split": args.split,
        "limit": int(args.limit),
        "selection": args.selection,
        "sample_count": len(records),
        "check_direct_rawpy": bool(args.check_direct_rawpy),
        "keep_dng": bool(args.keep_dng),
        "rawpy_version": getattr(rawpy, "__version__", "unknown"),
        "libraw_version": getattr(rawpy, "libraw_version", "unknown"),
        "tifffile_version": getattr(tifffile, "__version__", "unknown"),
        "dng_metadata": {
            "photometric": "LinearRaw",
            "photometric_code": DNG_PHOTOMETRIC_LINEAR_RAW,
            "channels": "RGB",
            "black_level": [0, 0, 0],
            "white_level": [65535, 65535, 65535],
            "color_matrix_1": "standard_xyz_to_srgb_d65",
            "as_shot_neutral": [1, 1, 1],
            "calibration_illuminant_1": "D65",
        },
        "rawpy_postprocess_kwargs": dict(RAWPY_POSTPROCESS_KWARGS),
        "contact_sheet": str(contact_sheet),
        "manifest": str(manifest_path),
    }
    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    print(f"[DONE] output_root={output_root}", flush=True)
    print(f"[DONE] contact_sheet={contact_sheet}", flush=True)
    print(f"[DONE] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
