from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RGB_COLORS = ((224, 64, 64), (64, 190, 96), (72, 128, 232))
DEFAULT_RAW_COLORS = ((224, 64, 64), (86, 200, 102), (72, 170, 120), (70, 120, 232))


def channel_stats(values: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "p50": None, "p99": None, "max": None}
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def summarize_channels(
    data: np.ndarray,
    *,
    channels: Sequence[str],
    channel_axis: int = 0,
) -> dict[str, dict[str, float | None]]:
    arr = np.asarray(data, dtype=np.float32)
    if channel_axis < 0:
        channel_axis += arr.ndim
    if channel_axis != 0:
        arr = np.moveaxis(arr, channel_axis, 0)
    if arr.shape[0] != len(channels):
        raise ValueError(f"Expected {len(channels)} channels, got shape={arr.shape}")
    return {name: channel_stats(arr[i]) for i, name in enumerate(channels)}


def _format_stat(value: float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def draw_distribution_tile(
    data: np.ndarray,
    *,
    channels: Sequence[str],
    colors: Sequence[tuple[int, int, int]] | None = None,
    channel_axis: int = 0,
    bins: int = 128,
    x_range: tuple[float, float] = (0.0, 1.0),
    width: int = 414,
    height: int = 125,
    font: ImageFont.ImageFont | None = None,
    small_font: ImageFont.ImageFont | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    arr = np.asarray(data, dtype=np.float32)
    if channel_axis < 0:
        channel_axis += arr.ndim
    if channel_axis != 0:
        arr = np.moveaxis(arr, channel_axis, 0)
    if arr.shape[0] != len(channels):
        raise ValueError(f"Expected {len(channels)} channels, got shape={arr.shape}")

    if colors is None:
        colors = DEFAULT_RGB_COLORS if len(channels) == 3 else DEFAULT_RAW_COLORS
    if len(colors) < len(channels):
        raise ValueError("Not enough colors for distribution channels.")

    image = Image.new("RGB", (int(width), int(height)), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    font = font or ImageFont.load_default()
    small_font = small_font or font

    left, top, right, bottom = 38, 10, int(width) - 8, int(height) - 34
    draw.rectangle([left, top, right, bottom], outline=(160, 160, 160), width=1)

    histograms: list[np.ndarray] = []
    stats = summarize_channels(arr, channels=channels, channel_axis=0)
    for c in range(len(channels)):
        values = arr[c]
        values = values[np.isfinite(values)]
        hist, _ = np.histogram(values, bins=int(bins), range=x_range)
        hist = hist.astype(np.float32)
        if hist.max() > 0:
            hist = hist / hist.max()
        histograms.append(hist)

    graph_w = max(right - left, 1)
    graph_h = max(bottom - top, 1)
    for c, hist in enumerate(histograms):
        points = []
        for i, y in enumerate(hist):
            x = left + int(round((i / max(len(hist) - 1, 1)) * graph_w))
            yy = bottom - int(round(float(y) * graph_h))
            points.append((x, yy))
        if len(points) >= 2:
            draw.line(points, fill=colors[c], width=2)

    for tick, label in ((0.0, f"{x_range[0]:.0f}"), (0.5, "0.5"), (1.0, f"{x_range[1]:.0f}")):
        x = left + int(round(tick * graph_w))
        draw.line([(x, bottom), (x, bottom + 4)], fill=(90, 90, 90), width=1)
        draw.text((x - 8, bottom + 5), label, fill=(60, 60, 60), font=small_font)

    legend_x = left
    legend_y = int(height) - 22
    for c, name in enumerate(channels):
        x0 = legend_x + c * 52
        draw.rectangle([x0, legend_y + 2, x0 + 10, legend_y + 12], fill=colors[c])
        draw.text((x0 + 14, legend_y), str(name), fill=(35, 35, 35), font=small_font)

    stat_lines = []
    for name in channels[:4]:
        s = stats[str(name)]
        stat_lines.append(
            f"{name}: {_format_stat(s['min'])}/{_format_stat(s['p50'])}/"
            f"{_format_stat(s['p99'])}/{_format_stat(s['max'])}"
        )
    stat_text = "  ".join(stat_lines[:2])
    draw.text((left, 0), stat_text, fill=(35, 35, 35), font=small_font)
    if len(stat_lines) > 2:
        draw.text((left, 12), "  ".join(stat_lines[2:4]), fill=(35, 35, 35), font=small_font)

    metadata = {
        "channels": list(channels),
        "x_range": [float(x_range[0]), float(x_range[1])],
        "bins": int(bins),
        "normalization": "per_channel_max",
        "stats": stats,
    }
    return image, metadata
