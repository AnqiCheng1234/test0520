#!/usr/bin/env python3
"""Create per-image KITTI RAW-like unprocessing panels with value distributions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


CHANNEL_COLORS_3 = [(230, 80, 80), (80, 220, 100), (80, 150, 255)]
CHANNEL_COLORS_4 = [(230, 80, 80), (70, 230, 110), (160, 230, 120), (80, 150, 255)]
CHANNEL_NAMES_3 = ("R", "G", "B")
CHANNEL_NAMES_4 = ("R", "G1", "G2", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-image unprocessing viz_dump-style panels.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=["normal", "dark", "over"])
    parser.add_argument("--tile-width", type=int, default=400)
    parser.add_argument("--tile-height", type=int, default=120)
    parser.add_argument("--dist-height", type=int, default=220)
    parser.add_argument("--max-images", type=int, default=0, help="Use 0 for all manifest samples.")
    return parser.parse_args()


def sample_name(sample: dict) -> str:
    return sample.get("sample_name") or f"{sample['split']}_{sample['drive']}_{sample['camera']}_{sample['frame']}"


def preview_path(raw_output_dir: Path, variant: str, link_path: Path, sample_root: Path) -> Path:
    rel = link_path.relative_to(sample_root).with_suffix(".png")
    return raw_output_dir / "preview" / variant / rel


def npz_path(raw_output_dir: Path, variant: str, link_path: Path, sample_root: Path) -> Path:
    rel = link_path.relative_to(sample_root).with_suffix(".npz")
    return raw_output_dir / variant / rel


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def load_raw_packed(path: Path) -> np.ndarray:
    data = np.load(path)
    if "raw_packed" not in data:
        raise KeyError(f"{path} does not contain raw_packed")
    x = data["raw_packed"].astype(np.float32)
    if x.ndim != 3 or x.shape[0] != 4:
        raise ValueError(f"Expected raw_packed [4,H,W], got {x.shape} in {path}")
    return x


def fit_image(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(np.clip(image_rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)
    pil = Image.fromarray(image, mode="RGB")
    fitted = ImageOps.contain(pil, size, method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (245, 245, 245))
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return np.asarray(canvas)


def draw_tile(canvas: np.ndarray, image_rgb: np.ndarray, title: str, x0: int, y0: int, width: int, height: int) -> None:
    tile = fit_image(image_rgb, (width, height))
    canvas[y0 : y0 + height, x0 : x0 + width] = tile
    overlay_h = 24
    canvas[y0 : y0 + overlay_h, x0 : x0 + width] = (
        0.45 * canvas[y0 : y0 + overlay_h, x0 : x0 + width]
    ).astype(np.uint8)
    cv2.putText(
        canvas,
        str(title)[:72],
        (x0 + 8, y0 + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def format_stat(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    if abs(value) < 1e-3 and value != 0.0:
        return f"{value:.1e}"
    return f"{value:.3g}"


def channel_first(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.shape}")
    if x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
        return x
    if x.shape[-1] in (3, 4):
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"Cannot infer channel axis from {x.shape}")


def distribution_axis_limits(channels: np.ndarray, axis_mode: str) -> tuple[float, float, tuple[float, ...], str]:
    if axis_mode == "unit":
        return 0.0, 1.0, (0.0, 0.5, 1.0), "x-axis fixed 0..1"
    values = channels[np.isfinite(channels)]
    if values.size == 0:
        return 0.0, 1.0, (0.0, 0.5, 1.0), "x-axis n/a"
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    mid = 0.5 * (float(lo) + float(hi))
    return float(lo), float(hi), (float(lo), mid, float(hi)), "x-axis p1..p99"


def draw_hist_axes(
    canvas: np.ndarray,
    hist_x: int,
    hist_y: int,
    hist_w: int,
    hist_h: int,
    lo: float,
    hi: float,
    tick_values: Sequence[float],
) -> None:
    base_y = hist_y + hist_h
    cv2.line(canvas, (hist_x, hist_y), (hist_x, base_y), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, (hist_x, base_y), (hist_x + hist_w, base_y), (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "1", (hist_x - 20, hist_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (hist_x - 20, base_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1, cv2.LINE_AA)
    denom = max(float(hi) - float(lo), 1e-12)
    for tick in tick_values:
        tick = float(tick)
        if not math.isfinite(tick) or tick < lo or tick > hi:
            continue
        px = hist_x + int(round(((tick - lo) / denom) * hist_w))
        cv2.line(canvas, (px, hist_y), (px, base_y), (62, 62, 62), 1, cv2.LINE_AA)
        cv2.line(canvas, (px, base_y), (px, base_y + 4), (150, 150, 150), 1, cv2.LINE_AA)
        label = format_stat(tick)
        (label_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        label_x = int(np.clip(px - label_w // 2, hist_x, hist_x + hist_w - label_w))
        cv2.putText(canvas, label, (label_x, base_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 190), 1, cv2.LINE_AA)


def channel_stats(values: np.ndarray) -> dict[str, float] | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def draw_distribution_block(
    canvas: np.ndarray,
    array: np.ndarray,
    title: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    *,
    axis_mode: str,
) -> None:
    cv2.rectangle(canvas, (x0, y0), (x0 + width - 1, y0 + height - 1), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, title[:68], (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
    try:
        channels = channel_first(array)
    except ValueError as exc:
        cv2.putText(canvas, str(exc)[:60], (x0 + 8, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
        return
    c = int(channels.shape[0])
    names = CHANNEL_NAMES_4 if c == 4 else CHANNEL_NAMES_3
    colors = CHANNEL_COLORS_4 if c == 4 else CHANNEL_COLORS_3
    hist_x = x0 + 36
    hist_y = y0 + 30
    hist_w = width - 46
    hist_h = 64
    lo, hi, tick_values, axis_note = distribution_axis_limits(channels, axis_mode)
    cv2.rectangle(canvas, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h), (38, 38, 38), thickness=-1)
    draw_hist_axes(canvas, hist_x, hist_y, hist_w, hist_h, lo, hi, tick_values)

    bins = 96
    for idx in range(c):
        values = channels[idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        counts, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
        if counts.max() <= 0:
            continue
        points = []
        for bidx, count in enumerate(counts):
            px = hist_x + int(round((bidx / max(bins - 1, 1)) * (hist_w - 1)))
            py = hist_y + hist_h - 1 - int(round((float(count) / float(counts.max())) * (hist_h - 4)))
            points.append((px, py))
        for left, right in zip(points[:-1], points[1:]):
            cv2.line(canvas, left, right, colors[idx], 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{axis_note}: {format_stat(lo)}..{format_stat(hi)}",
        (x0 + 8, y0 + 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    stat_y = y0 + 144
    for idx in range(c):
        stat = channel_stats(channels[idx])
        if stat is None:
            line = f"{names[idx]}: n/a"
        else:
            line = (
                f"{names[idx]} min/p50/max "
                f"{format_stat(stat['min'])}/{format_stat(stat['p50'])}/{format_stat(stat['max'])} "
                f"p99 {format_stat(stat['p99'])}"
            )
        cv2.putText(canvas, line[:72], (x0 + 8, stat_y + idx * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[idx], 1, cv2.LINE_AA)


def build_panel(
    *,
    sample: dict,
    sample_root: Path,
    raw_output_dir: Path,
    output_dir: Path,
    variants: Sequence[str],
    tile_size: tuple[int, int],
    dist_height: int,
) -> dict[str, object]:
    name = sample_name(sample)
    link = Path(sample["link_path"])
    rgb = load_rgb(link)
    previews = [load_rgb(preview_path(raw_output_dir, variant, link, sample_root)) for variant in variants]
    raw_arrays = [load_raw_packed(npz_path(raw_output_dir, variant, link, sample_root)) for variant in variants]

    columns = ["original", *variants]
    arrays_for_dist = [rgb, *raw_arrays]
    dist_titles = ["RGB value distribution", *[f"{variant} RAW packed distribution" for variant in variants]]
    axis_modes = ["unit", *["auto" for _ in variants]]

    gap = 8
    meta_h = 50
    tile_w, tile_h = tile_size
    width = len(columns) * tile_w + (len(columns) + 1) * gap
    height = meta_h + tile_h + dist_height + gap * 4
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :, :] = (14, 14, 14)

    cv2.putText(canvas, name[:150], (gap, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"rgb={link}  raw_output={raw_output_dir.name}"[:190],
        (gap, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    y_tile = meta_h + gap
    y_dist = y_tile + tile_h + gap
    images = [rgb, *previews]
    for idx, (title, image) in enumerate(zip(columns, images)):
        x = gap + idx * (tile_w + gap)
        draw_tile(canvas, image, title, x, y_tile, tile_w, tile_h)
        draw_distribution_block(
            canvas,
            arrays_for_dist[idx],
            dist_titles[idx],
            x,
            y_dist,
            tile_w,
            dist_height,
            axis_mode=axis_modes[idx],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_panel.jpg"
    Image.fromarray(canvas).save(out_path, quality=92)

    raw_summaries = {}
    for variant, raw in zip(variants, raw_arrays):
        raw_summaries[variant] = {
            "shape": [int(v) for v in raw.shape],
            "min": float(np.min(raw)),
            "p50": float(np.percentile(raw, 50)),
            "p99": float(np.percentile(raw, 99)),
            "max": float(np.max(raw)),
            "mean": float(np.mean(raw)),
        }
    return {"sample_name": name, "panel": str(out_path), "raw": raw_summaries}


def main() -> None:
    args = parse_args()
    manifest_path = args.sample_manifest.expanduser().resolve()
    raw_output_dir = args.raw_output_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else raw_output_dir / "viz_dump"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_root = Path(manifest["output_dir"]).expanduser().resolve()
    samples = list(manifest["samples"])
    if args.max_images and args.max_images > 0:
        samples = samples[: args.max_images]

    records = []
    for sample in samples:
        records.append(
            build_panel(
                sample=sample,
                sample_root=sample_root,
                raw_output_dir=raw_output_dir,
                output_dir=output_dir,
                variants=args.variants,
                tile_size=(args.tile_width, args.tile_height),
                dist_height=args.dist_height,
            )
        )

    manifest_jsonl = output_dir / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "sample_manifest": str(manifest_path),
        "raw_output_dir": str(raw_output_dir),
        "output_dir": str(output_dir),
        "num_panels": len(records),
        "manifest_jsonl": str(manifest_jsonl),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
