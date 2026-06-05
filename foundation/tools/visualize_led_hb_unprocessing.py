#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from foundation.engine.datasets.led_hb import (
    LEDHBRaw,
    default_led_hb_raw_adapter_config,
)


CHANNEL_NAMES = ("R", "Gr", "Gb", "B")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _srgb_from_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _raw_camera_rgb(packed: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            packed[0],
            0.5 * (packed[1] + packed[2]),
            packed[3],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _wb_debug_preview(packed: np.ndarray, red_gain: float, blue_gain: float) -> np.ndarray:
    camera_rgb = _raw_camera_rgb(packed)
    camera_rgb[..., 0] *= red_gain
    camera_rgb[..., 2] *= blue_gain
    return _srgb_from_linear(camera_rgb)


def _display_stretch(x: np.ndarray, *, gamma: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    hi = float(np.percentile(x[finite], 99.5))
    if hi <= 1e-8:
        hi = float(np.max(x[finite]))
    if hi <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    out = np.clip(x / hi, 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, gamma)
    return out.astype(np.float32, copy=False)


def _nice_upper_bound(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.1
    for candidate in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        if value <= candidate:
            return candidate
    return 1.0


def _packed_to_mosaic(packed: np.ndarray) -> np.ndarray:
    _, height, width = packed.shape
    mosaic = np.zeros((height * 2, width * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 0::2] = packed[2]
    mosaic[1::2, 1::2] = packed[3]
    return mosaic


def _sample_indices(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted({int(round(x)) for x in np.linspace(0, total - 1, count)})


def _plot_hist_lines(
    ax: Any,
    channels: np.ndarray,
    *,
    names: tuple[str, ...],
    colors: tuple[str, ...],
    xlim: tuple[float, float],
    bins: int = 96,
) -> None:
    lo, hi = xlim
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for channel, name, color in zip(channels, names, colors):
        values = np.asarray(channel, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        counts, _ = np.histogram(np.clip(values, lo, hi), bins=edges)
        if counts.max() > 0:
            counts = counts.astype(np.float32) / float(counts.max())
        ax.plot(centers, counts, color=color, linewidth=1.25, label=name)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, 1.04)
    ax.grid(True, color="#3a3a3a", linewidth=0.5, alpha=0.75)
    ax.tick_params(colors="#d5d5d5", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color("#777777")
    ax.set_facecolor("#202020")
    ax.legend(loc="upper right", fontsize=6, frameon=False, labelcolor="#e7e7e7")


def _plot_onepage(
    dataset: LEDHBRaw,
    indices: Iterable[int],
    stats: dict[str, Any],
    output_path: Path,
    *,
    raw_preview_display: str = "stretch",
) -> tuple[list[dict[str, Any]], float]:
    if raw_preview_display not in {"stretch", "linear"}:
        raise ValueError(f"raw_preview_display must be stretch or linear, got {raw_preview_display!r}")
    indices = list(indices)
    raw_axis_max = _nice_upper_bound(float(np.quantile(stats["raw_all"], 0.99)))
    raw_axis_max = min(max(raw_axis_max, 0.02), 1.0)
    rows: list[dict[str, Any]] = []
    raw_display_note = (
        "P99.5 display stretch only."
        if raw_preview_display == "stretch"
        else "No display stretch: raw sensor-linear values shown directly."
    )

    titles = [
        ("LDR RGB", "sRGB LED input after fixed 378x672 geometry. This is the RGB/C2 appearance."),
        ("RGB dist", "Per-channel RGB histogram. X axis is fixed to 0..1."),
        ("RAW camera RGB", f"raw4 shown as [R, mean(Gr,Gb), B]. {raw_display_note}"),
        ("RAW dist", f"Packed RGGB raw4 histogram. X axis fixed 0..{raw_axis_max:g}; larger values are clipped into the last bin."),
        ("WB+sRGB debug", "RAW preview after fixed WB gains and sRGB conversion. Sanity check only."),
        ("RGGB mosaic", f"Single-channel Bayer mosaic from raw4. {raw_display_note}"),
    ]
    fig = plt.figure(figsize=(24, 2.1 + 2.25 * max(1, len(indices))), facecolor="#111111")
    grid = fig.add_gridspec(
        nrows=len(indices) + 1,
        ncols=6,
        height_ratios=[0.72] + [1.0] * len(indices),
        hspace=0.18,
        wspace=0.05,
    )

    for col_idx, (title, description) in enumerate(titles):
        ax = fig.add_subplot(grid[0, col_idx])
        ax.set_facecolor("#1c1c1c")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#4a4a4a")
        ax.text(0.02, 0.78, title, color="#ffffff", fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.02,
            0.50,
            textwrap.fill(description, width=38),
            color="#d7d7d7",
            fontsize=8,
            va="top",
            transform=ax.transAxes,
            linespacing=1.15,
        )

    for row_idx, sample_idx in enumerate(indices, start=1):
        sample = dataset.build_sample(sample_idx, include_rgb_preview=True)
        packed = _to_numpy(sample["raw"])
        rgb = _to_numpy(sample["rgb_preview"])
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        isp_params = sample["isp_params"]
        red_gain = _to_float(isp_params["red_gain"])
        blue_gain = _to_float(isp_params["blue_gain"])
        camera_rgb = _raw_camera_rgb(packed)
        wb_preview = _wb_debug_preview(packed, red_gain, blue_gain)
        mosaic = _packed_to_mosaic(packed)
        if raw_preview_display == "stretch":
            raw_camera_view = _display_stretch(camera_rgb, gamma=1.0 / 2.2)
            mosaic_view = _display_stretch(mosaic, gamma=1.0 / 2.2)
        else:
            raw_camera_view = np.clip(camera_rgb, 0.0, 1.0)
            mosaic_view = np.clip(mosaic, 0.0, 1.0)

        image_columns = [
            rgb_hwc,
            None,
            raw_camera_view,
            None,
            np.clip(wb_preview, 0.0, 1.0),
            mosaic_view,
        ]
        for col_idx, image in enumerate(image_columns):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.set_facecolor("#050505")
            if col_idx == 1:
                _plot_hist_lines(
                    ax,
                    rgb,
                    names=("R", "G", "B"),
                    colors=("#e65050", "#54d66a", "#4f93ff"),
                    xlim=(0.0, 1.0),
                )
                ax.set_title(
                    f"p50={np.quantile(rgb, 0.5):.3g} p99={np.quantile(rgb, 0.99):.3g}",
                    color="#e7e7e7",
                    fontsize=8,
                    pad=2,
                )
            elif col_idx == 3:
                _plot_hist_lines(
                    ax,
                    packed,
                    names=("R", "Gr", "Gb", "B"),
                    colors=("#e65050", "#61e882", "#9bd75a", "#4f93ff"),
                    xlim=(0.0, raw_axis_max),
                )
                ax.set_title(
                    f"p50={np.quantile(packed, 0.5):.3g} p99={np.quantile(packed, 0.99):.3g} max={packed.max():.3g}",
                    color="#e7e7e7",
                    fontsize=8,
                    pad=2,
                )
            else:
                if image is not None and image.ndim == 2:
                    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
                elif image is not None:
                    ax.imshow(np.clip(image, 0.0, 1.0))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{sample['sample_name']}\nrg={red_gain:.2f} bg={blue_gain:.2f}",
                    color="#e7e7e7",
                    fontsize=7,
                    rotation=0,
                    labelpad=48,
                    va="center",
                )

        rows.append(
            {
                "dataset_index": int(sample_idx),
                "sample_name": sample["sample_name"],
                "image_path": sample["image_path"],
                "red_gain": red_gain,
                "blue_gain": blue_gain,
                "light_scale": _to_float(isp_params["light_scale"]),
                "raw_min": float(packed.min()),
                "raw_max": float(packed.max()),
                "raw_mean": float(packed.mean()),
                "raw_p99": float(np.quantile(packed, 0.99)),
            }
        )

    fig.suptitle(
        "LED-HB val RA0 inverse-ISP one-page view: image pairs and value distributions",
        color="#ffffff",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        (
            "Formal RAW input is packed raw4 [R, Gr, Gb, B]. "
            f"RAW preview display={raw_preview_display}; histograms always use actual tensor values."
        ),
        ha="center",
        color="#d7d7d7",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return rows, raw_axis_max


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    qs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    return {f"q{q:g}": float(np.quantile(values, q)) for q in qs}


def _rgb_luma(rgb_chw: np.ndarray) -> np.ndarray:
    rgb = np.transpose(rgb_chw, (1, 2, 0))
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _plot_preview_grid(dataset: LEDHBRaw, indices: Iterable[int], output_path: Path) -> list[dict[str, Any]]:
    rows = []
    indices = list(indices)
    fig, axes = plt.subplots(len(indices), 7, figsize=(21, 3.0 * max(1, len(indices))), squeeze=False)
    for row_idx, sample_idx in enumerate(indices):
        sample = dataset.build_sample(sample_idx, include_rgb_preview=True)
        packed = _to_numpy(sample["raw"])
        rgb = _to_numpy(sample["rgb_preview"])
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        isp_params = sample["isp_params"]
        red_gain = _to_float(isp_params["red_gain"])
        blue_gain = _to_float(isp_params["blue_gain"])
        camera_rgb = _raw_camera_rgb(packed)
        wb_preview = _wb_debug_preview(packed, red_gain, blue_gain)
        mosaic = _packed_to_mosaic(packed)

        columns = [
            ("LDR RGB", rgb_hwc),
            ("raw4 camera RGB\np99.5 stretch", _display_stretch(camera_rgb, gamma=1.0 / 2.2)),
            ("WB+sRGB debug", np.clip(wb_preview, 0.0, 1.0)),
            ("RGGB mosaic\np99.5 stretch", _display_stretch(mosaic, gamma=1.0 / 2.2)),
            ("R", _display_stretch(packed[0], gamma=1.0 / 2.2)),
            ("Gr", _display_stretch(packed[1], gamma=1.0 / 2.2)),
            ("B", _display_stretch(packed[3], gamma=1.0 / 2.2)),
        ]
        for col_idx, (title, image) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            if image.ndim == 2:
                ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.clip(image, 0.0, 1.0))
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
        axes[row_idx, 0].set_ylabel(
            f"{sample['sample_name']}\nrg={red_gain:.2f} bg={blue_gain:.2f}",
            fontsize=8,
        )
        rows.append(
            {
                "dataset_index": int(sample_idx),
                "sample_name": sample["sample_name"],
                "image_path": sample["image_path"],
                "red_gain": red_gain,
                "blue_gain": blue_gain,
                "light_scale": _to_float(isp_params["light_scale"]),
                "raw_min": float(packed.min()),
                "raw_max": float(packed.max()),
                "raw_mean": float(packed.mean()),
                "raw_p99": float(np.quantile(packed, 0.99)),
            }
        )
    fig.suptitle("LED-HB val: formal RA0 inverse-ISP / packed RGGB raw4 visualization", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return rows


def _collect_stats(dataset: LEDHBRaw, *, max_samples: int, pixel_stride: int) -> dict[str, Any]:
    total = len(dataset)
    count = total if max_samples <= 0 else min(total, max_samples)
    channel_values = [[] for _ in CHANNEL_NAMES]
    rgb_luma_values = []
    per_sample = []
    for i in range(count):
        if i % 50 == 0:
            print(f"[STATS] {i}/{count}", flush=True)
        sample = dataset.build_sample(i, include_rgb_preview=True)
        packed = _to_numpy(sample["raw"])[:, ::pixel_stride, ::pixel_stride].astype(np.float32, copy=False)
        rgb = _to_numpy(sample["rgb_preview"])[:, ::pixel_stride, ::pixel_stride].astype(np.float32, copy=False)
        luma = _rgb_luma(rgb)
        rgb_luma_values.append(luma.reshape(-1))
        for channel_idx in range(4):
            channel_values[channel_idx].append(packed[channel_idx].reshape(-1))
        raw_flat = packed.reshape(-1)
        per_sample.append(
            {
                "dataset_index": int(i),
                "sample_name": sample["sample_name"],
                "raw_mean": float(raw_flat.mean()),
                "raw_std": float(raw_flat.std()),
                "raw_p01": float(np.quantile(raw_flat, 0.01)),
                "raw_p50": float(np.quantile(raw_flat, 0.50)),
                "raw_p99": float(np.quantile(raw_flat, 0.99)),
                "raw_zero_frac_lt_1e_4": float((raw_flat < 1e-4).mean()),
                "raw_sat_frac_gt_0p999": float((raw_flat > 0.999).mean()),
                "rgb_luma_mean": float(luma.mean()),
                "rgb_luma_p99": float(np.quantile(luma, 0.99)),
            }
        )
    channel_arrays = [np.concatenate(values) if values else np.empty((0,), dtype=np.float32) for values in channel_values]
    rgb_luma_array = np.concatenate(rgb_luma_values) if rgb_luma_values else np.empty((0,), dtype=np.float32)
    raw_all = np.concatenate(channel_arrays) if channel_arrays else np.empty((0,), dtype=np.float32)
    return {
        "count": count,
        "pixel_stride": int(pixel_stride),
        "raw_all": raw_all,
        "channels": dict(zip(CHANNEL_NAMES, channel_arrays)),
        "rgb_luma": rgb_luma_array,
        "per_sample": per_sample,
    }


def _plot_distribution(stats: dict[str, Any], output_path: Path) -> None:
    raw_all = stats["raw_all"]
    channels = stats["channels"]
    rgb_luma = stats["rgb_luma"]
    per_sample = stats["per_sample"]
    bins = np.linspace(0.0, 1.0, 201)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    ax.hist(raw_all, bins=bins, density=True, color="#555555", alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("raw4 all values")
    ax.set_xlabel("value")
    ax.set_ylabel("density, log")

    ax = axes[0, 1]
    for name, values in channels.items():
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.4, label=name)
    ax.set_yscale("log")
    ax.set_title("raw4 per-channel values")
    ax.set_xlabel("value")
    ax.legend(frameon=False)

    ax = axes[0, 2]
    ax.hist(rgb_luma, bins=bins, density=True, histtype="stepfilled", alpha=0.35, label="RGB luma")
    ax.hist(raw_all, bins=bins, density=True, histtype="step", linewidth=1.3, label="raw4 all")
    ax.set_yscale("log")
    ax.set_title("RGB luma vs raw4")
    ax.set_xlabel("value")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.boxplot([channels[name] for name in CHANNEL_NAMES], tick_labels=CHANNEL_NAMES, showfliers=False)
    ax.set_title("raw4 channel value boxes")
    ax.set_ylabel("value")

    ax = axes[1, 1]
    means = np.asarray([row["raw_mean"] for row in per_sample], dtype=np.float32)
    p99 = np.asarray([row["raw_p99"] for row in per_sample], dtype=np.float32)
    luma_mean = np.asarray([row["rgb_luma_mean"] for row in per_sample], dtype=np.float32)
    ax.hist(means, bins=60, alpha=0.6, label="raw mean")
    ax.hist(p99, bins=60, alpha=0.6, label="raw p99")
    ax.hist(luma_mean, bins=60, alpha=0.5, label="RGB luma mean")
    ax.set_title("per-frame summary distribution")
    ax.set_xlabel("value")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    zero_frac = np.asarray([row["raw_zero_frac_lt_1e_4"] for row in per_sample], dtype=np.float32)
    sat_frac = np.asarray([row["raw_sat_frac_gt_0p999"] for row in per_sample], dtype=np.float32)
    ax.hist(zero_frac, bins=60, alpha=0.7, label="raw < 1e-4")
    ax.hist(sat_frac, bins=60, alpha=0.7, label="raw > 0.999")
    ax.set_title("near-zero / saturation fractions")
    ax.set_xlabel("fraction")
    ax.legend(frameon=False)

    fig.suptitle(
        f"LED-HB formal RA0 inverse-ISP distribution, n={stats['count']}, pixel_stride={stats['pixel_stride']}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_summary(
    output_path: Path,
    *,
    dataset: LEDHBRaw,
    preview_rows: list[dict[str, Any]],
    stats: dict[str, Any],
    args: argparse.Namespace,
    raw_axis_max: float | None = None,
) -> None:
    channels = stats["channels"]
    per_sample = stats["per_sample"]
    summary = {
        "input_list": str(Path(args.led_list).resolve()),
        "num_dataset_samples": len(dataset),
        "num_stats_samples": stats["count"],
        "pixel_stride": stats["pixel_stride"],
        "onepage_raw_axis_max": raw_axis_max,
        "preview_samples": preview_rows,
        "geometry": dataset.describe_geometry(),
        "unprocessing": dataset.describe_unprocessing(),
        "notes": [
            "This uses the same LEDHBRaw formal RA0 online inverse-ISP path as N2/N7.",
            "randomize_unprocessing=false fixes red/blue gains and light scale.",
            "noise parameters are kept in config, but noise realization is disabled when randomize_unprocessing=false.",
            "raw4 channel order is R,Gr,Gb,B packed from RGGB.",
        ],
        "raw_all_quantiles": _quantiles(stats["raw_all"]),
        "rgb_luma_quantiles": _quantiles(stats["rgb_luma"]),
        "channel_quantiles": {name: _quantiles(values) for name, values in channels.items()},
        "per_sample_summary": {
            "raw_mean": _quantiles(np.asarray([row["raw_mean"] for row in per_sample], dtype=np.float32)),
            "raw_p99": _quantiles(np.asarray([row["raw_p99"] for row in per_sample], dtype=np.float32)),
            "raw_zero_frac_lt_1e_4": _quantiles(
                np.asarray([row["raw_zero_frac_lt_1e_4"] for row in per_sample], dtype=np.float32)
            ),
            "raw_sat_frac_gt_0p999": _quantiles(
                np.asarray([row["raw_sat_frac_gt_0p999"] for row in per_sample], dtype=np.float32)
            ),
            "rgb_luma_mean": _quantiles(np.asarray([row["rgb_luma_mean"] for row in per_sample], dtype=np.float32)),
        },
    }
    output_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize LED-HB RAW-Adapter-style inverse ISP output and value distributions.")
    parser.add_argument("--led-list", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["train", "val"], default="val")
    parser.add_argument("--num-panel-samples", type=int, default=6)
    parser.add_argument("--stats-max-samples", type=int, default=1000)
    parser.add_argument("--pixel-stride", type=int, default=16)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--raw-preview-display", choices=["stretch", "linear"], default="stretch")
    parser.add_argument("--write-separate", action="store_true", help="Also write the old separate preview-grid and distribution images.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.pixel_stride <= 0:
        raise ValueError("--pixel-stride must be positive")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_config = default_led_hb_raw_adapter_config()
    dataset = LEDHBRaw(
        args.led_list,
        mode=args.mode,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        hflip_prob=0.0,
        include_rgb_preview=True,
        unprocessing_config=raw_config,
    )
    stats = _collect_stats(dataset, max_samples=args.stats_max_samples, pixel_stride=args.pixel_stride)
    indices = _sample_indices(len(dataset), args.num_panel_samples)
    preview_rows, raw_axis_max = _plot_onepage(
        dataset,
        indices,
        stats,
        out_dir / "led_hb_invisp_unprocessing_onepage.png",
        raw_preview_display=args.raw_preview_display,
    )
    if args.write_separate:
        _plot_preview_grid(dataset, indices, out_dir / "led_hb_invisp_unprocessing_preview_grid.png")
        _plot_distribution(stats, out_dir / "led_hb_invisp_unprocessing_distribution.png")
    _write_summary(
        out_dir / "led_hb_invisp_unprocessing_summary.json",
        dataset=dataset,
        preview_rows=preview_rows,
        stats=stats,
        args=args,
        raw_axis_max=raw_axis_max,
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "onepage": str(out_dir / "led_hb_invisp_unprocessing_onepage.png"),
                "summary": str(out_dir / "led_hb_invisp_unprocessing_summary.json"),
                "num_stats_samples": int(stats["count"]),
                "pixel_stride": int(stats["pixel_stride"]),
                "raw_axis_max": float(raw_axis_max),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
