#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DAYTIMES = ("night", "day", "twilight")
GROUPS = (
    (
        "group1_clear_dry_incity",
        {
            "fog": "none",
            "precipitation": "none",
            "road_state": "dry",
            "infrastructure": "inCity",
        },
    ),
    (
        "group2_rain_wet_incity",
        {
            "fog": "none",
            "precipitation": "rain",
            "road_state": "wet",
            "infrastructure": "inCity",
        },
    ),
)
REQUIRED_PATH_COLUMNS = ("lut_preview", "raw_left_tiff", "lidar_proj_left")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_has_files(row: dict[str, str]) -> bool:
    return all(row.get(name) and Path(row[name]).is_file() for name in REQUIRED_PATH_COLUMNS)


def stable_pick(rows: list[dict[str, str]], key: str, seed: int) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(ordered)
    return ordered[index]


def select_group_rows(
    rows: list[dict[str, str]],
    label: str,
    conditions: dict[str, str],
    seed: int,
) -> dict[str, dict[str, str]]:
    selected = {}
    for daytime in DAYTIMES:
        candidates = [
            row
            for row in rows
            if row.get("daytime") == daytime
            and row_has_files(row)
            and all(row.get(name) == value for name, value in conditions.items())
        ]
        if not candidates:
            raise RuntimeError(f"No STF sample for {label} daytime={daytime}: {conditions}")
        selected[daytime] = stable_pick(candidates, f"{label}:{daytime}", seed)
    return selected


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def load_rgb(path: str, size: tuple[int, int]) -> Image.Image:
    return resize_image(Image.open(path).convert("RGB"), size)


def load_raw(path: str) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32)


def raw_to_preview(raw: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.clip(raw / max(scale, 1.0), 0.0, 1.0)
    scaled = np.sqrt(scaled)
    image = (scaled * 255.0).astype(np.uint8)
    return np.stack([image, image, image], axis=-1)


def colorize_depth(depth: np.ndarray, valid: np.ndarray, vmax: float) -> np.ndarray:
    normalized = np.clip(depth / vmax, 0.0, 1.0)
    image_u8 = ((1.0 - normalized) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def lidar_preview(path: str, size: tuple[int, int], vmax: float, kernel_size: int) -> Image.Image:
    with np.load(path, allow_pickle=False) as data:
        depth = np.asarray(data["arr_0"], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    colored = colorize_depth(depth, valid, vmax)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        valid_u8 = (valid.astype(np.uint8) * 255)
        expanded_mask = cv2.dilate(valid_u8, kernel, iterations=1) > 0
        expanded_color = cv2.dilate(colored, kernel, iterations=1)
        background = np.full_like(expanded_color, 32, dtype=np.uint8)
        background[expanded_mask] = expanded_color[expanded_mask]
        colored = background
    return resize_image(Image.fromarray(colored), size)


def make_colorbar(width: int, height: int, vmax: float) -> Image.Image:
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    image_u8 = ((1.0 - gradient) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored = np.repeat(colored, height, axis=0)
    return Image.fromarray(colored)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int] = (235, 235, 235),
) -> None:
    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def make_group_panel(
    out_path: Path,
    label: str,
    conditions: dict[str, str],
    rows_by_daytime: dict[str, dict[str, str]],
    *,
    cell_size: tuple[int, int],
    depth_vmax: float,
    lidar_kernel_size: int,
    raw_percentile: float,
) -> None:
    raw_arrays = [load_raw(rows_by_daytime[daytime]["raw_left_tiff"]) for daytime in DAYTIMES]
    raw_scale = float(np.percentile(np.concatenate([arr.reshape(-1) for arr in raw_arrays]), raw_percentile))

    rgb_cells = [load_rgb(rows_by_daytime[daytime]["lut_preview"], cell_size) for daytime in DAYTIMES]
    raw_cells = [
        resize_image(Image.fromarray(raw_to_preview(raw, raw_scale)), cell_size)
        for raw in raw_arrays
    ]
    lidar_cells = [
        lidar_preview(
            rows_by_daytime[daytime]["lidar_proj_left"],
            cell_size,
            depth_vmax,
            lidar_kernel_size,
        )
        for daytime in DAYTIMES
    ]

    left_w = 138
    gap = 16
    margin = 20
    header_h = 86
    column_h = 36
    row_gap = 20
    colorbar_h = 48
    cell_w, cell_h = cell_size
    canvas_w = margin * 2 + left_w + gap + len(DAYTIMES) * cell_w + (len(DAYTIMES) - 1) * gap
    canvas_h = margin + header_h + column_h + 3 * cell_h + 2 * row_gap + colorbar_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    condition_text = ", ".join(f"{key}={value}" for key, value in conditions.items())
    draw_text(draw, (margin, margin), label)
    draw_text(draw, (margin, margin + 20), condition_text, fill=(210, 210, 210))
    draw_text(
        draw,
        (margin, margin + 40),
        f"RAW shared p{raw_percentile:g} scale={raw_scale:.1f}, gamma=0.5 | LiDAR vmax={depth_vmax:g}m",
        fill=(190, 190, 190),
    )

    x0 = margin + left_w + gap
    y = margin + header_h
    for idx, daytime in enumerate(DAYTIMES):
        row = rows_by_daytime[daytime]
        x = x0 + idx * (cell_w + gap)
        draw_text(draw, (x, y), daytime.upper())
        draw_text(draw, (x, y + 16), row["sample_id"], fill=(190, 190, 190))

    y += column_h
    rows = (
        ("RGB LUT", rgb_cells),
        ("RAW shared", raw_cells),
        ("LiDAR depth", lidar_cells),
    )
    for row_idx, (row_label, cells) in enumerate(rows):
        row_y = y + row_idx * (cell_h + row_gap)
        draw_text(draw, (margin, row_y + 8), row_label)
        for idx, image in enumerate(cells):
            x = x0 + idx * (cell_w + gap)
            canvas.paste(image, (x, row_y))

    colorbar_y = y + 3 * cell_h + 2 * row_gap + 16
    colorbar = make_colorbar(260, 16, depth_vmax)
    canvas.paste(colorbar, (x0, colorbar_y))
    draw_text(draw, (x0, colorbar_y + 20), "near")
    draw_text(draw, (x0 + 205, colorbar_y + 20), f"far {depth_vmax:g}m+")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def stack_overview(paths: list[Path], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_manifest.csv",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument("--cell-height", type=int, default=192)
    parser.add_argument("--depth-vmax", type=float, default=80.0)
    parser.add_argument("--lidar-kernel-size", type=int, default=5)
    parser.add_argument("--raw-percentile", type=float, default=99.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path("finetune_stf/analysis/stf_daytime_compare") / datetime.now().strftime("%m%d_%H%M")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.manifest))
    panel_paths = []
    summary_rows = []
    for label, conditions in GROUPS:
        rows_by_daytime = select_group_rows(rows, label, conditions, args.seed)
        out_path = out_dir / f"{label}.png"
        make_group_panel(
            out_path,
            label,
            conditions,
            rows_by_daytime,
            cell_size=(args.cell_width, args.cell_height),
            depth_vmax=args.depth_vmax,
            lidar_kernel_size=args.lidar_kernel_size,
            raw_percentile=args.raw_percentile,
        )
        panel_paths.append(out_path)
        for daytime, row in rows_by_daytime.items():
            summary_rows.append(
                {
                    "group": label,
                    "daytime": daytime,
                    "sample_id": row["sample_id"],
                    "official_split": row["official_split"],
                    "fog": row["fog"],
                    "precipitation": row["precipitation"],
                    "road_state": row["road_state"],
                    "infrastructure": row["infrastructure"],
                    "lut_preview": row["lut_preview"],
                    "raw_left_tiff": row["raw_left_tiff"],
                    "lidar_proj_left": row["lidar_proj_left"],
                    "panel_path": str(out_path),
                }
            )

    overview_path = out_dir / "overview.png"
    stack_overview(panel_paths, overview_path)

    with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "source_manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "groups": [
            {"label": label, "conditions": conditions}
            for label, conditions in GROUPS
        ],
        "panels": [str(path) for path in panel_paths],
        "overview": str(overview_path),
        "samples_csv": str(out_dir / "samples.csv"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
