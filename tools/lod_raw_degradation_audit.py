#!/usr/bin/env python3
"""Audit paired LOD RAW_Dark / RAW_normal degradation statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_raw import _apply_crop  # noqa: E402
from finetune_stf.dataset.lod_true import (  # noqa: E402
    LOD_TRUE_NATIVE_HW,
    _load_lod_true_manifest_rows,
    _load_raw_rgb16_png,
)


EPS = 1e-8
PERCENTILES = (0.1, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 99.0, 99.9)
CHANNELS = ("R", "G", "B")
SUMMARY_FIELDS = (
    "ev_gap_luma",
    "gain_p95_luma",
    "dark_luma_p99",
    "normal_luma_p99",
    "dark_luma_dynamic_range_98",
    "normal_luma_dynamic_range_98",
    "noise_sigma_dark_luma_flat",
    "noise_sigma_normal_luma_flat",
    "noise_sigma_ratio_dark_over_normal",
    "snr_ratio_lowbin",
    "snr_ratio_midbin",
    "snr_ratio_highbin",
    "gradient_ratio",
    "gradient_corr",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "chromaticity_delta_l1",
    "linear_residual_sigma",
    "gamma_residual_sigma",
    "gamma_value",
    "gamma_fit_r2",
    "phase_corr_shift_abs_max",
    "phase_corr_response",
    "registration_outlier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", default="01Valid")
    parser.add_argument(
        "--crop-mode",
        default="center512x960",
        choices=("center512x960", "full"),
        help="Audit crop. center512x960 matches train.py val crop from 800x1200.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=-1,
        help="-1 means all samples; otherwise deterministic random sample per split.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--panel-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def split_names(value: str) -> list[str]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        raise ValueError("--splits must contain at least one split name")
    return names


def crop_box_for_mode(crop_mode: str) -> tuple[int, int, int, int]:
    if crop_mode == "full":
        return 0, 0, int(LOD_TRUE_NATIVE_HW[0]), int(LOD_TRUE_NATIVE_HW[1])
    if crop_mode == "center512x960":
        return 144, 120, 512, 960
    raise ValueError(f"Unsupported crop_mode={crop_mode!r}")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def percentile_field_name(pct: float) -> str:
    text = ("%g" % pct).replace(".", "_")
    return f"p{text}"


def describe_array(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    payload: dict[str, float] = {}
    pct_values = np.percentile(arr, PERCENTILES)
    for pct, value in zip(PERCENTILES, pct_values):
        payload[f"{prefix}_{percentile_field_name(pct)}"] = float(value)
    payload[f"{prefix}_mean"] = float(np.mean(arr))
    payload[f"{prefix}_std"] = float(np.std(arr))
    payload[f"{prefix}_max"] = float(np.max(arr))
    payload[f"{prefix}_black_ratio"] = float(np.mean(arr <= (1.0 / 65535.0)))
    payload[f"{prefix}_sat_ratio"] = float(np.mean(arr >= (65534.0 / 65535.0)))
    p1 = payload[f"{prefix}_p1"]
    p99 = payload[f"{prefix}_p99"]
    payload[f"{prefix}_dynamic_range_98"] = float(p99 - p1)
    return payload


def channel_and_luma_stats(prefix: str, rgb: np.ndarray) -> dict[str, float]:
    payload: dict[str, float] = {}
    for idx, channel in enumerate(CHANNELS):
        payload.update(describe_array(f"{prefix}_{channel}", rgb[..., idx]))
    payload.update(describe_array(f"{prefix}_luma", luma(rgb)))
    return payload


def median_residual(values: np.ndarray, ksize: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    blurred = cv2.medianBlur(arr, ksize)
    return arr - blurred


def mad_sigma(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    p1, p99 = np.percentile(arr, (1.0, 99.0))
    return np.clip((arr - p1) / (p99 - p1 + EPS), 0.0, 1.0).astype(np.float32)


def sobel_magnitude(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def corrcoef_safe(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    else:
        valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid].reshape(-1)
    y = y[valid].reshape(-1)
    if x.size < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < EPS or sy < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_exposure_and_color(row: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    gains: dict[str, float] = {}
    for channel in (*CHANNELS, "luma"):
        normal_p95 = row[f"normal_{channel}_p95"]
        dark_p95 = row[f"dark_{channel}_p95"]
        gain = float(normal_p95 / max(dark_p95, EPS))
        gains[channel] = gain
        payload[f"gain_p95_{channel}"] = gain
        payload[f"ev_gap_{channel}"] = float(np.log2(max(gain, EPS)))
    payload["channel_gain_ratio_R_over_G"] = float(gains["R"] / max(gains["G"], EPS))
    payload["channel_gain_ratio_B_over_G"] = float(gains["B"] / max(gains["G"], EPS))
    payload["channel_gain_range_ratio"] = float(max(gains[c] for c in CHANNELS) / max(min(gains[c] for c in CHANNELS), EPS))
    return payload


def compute_chromaticity(dark: np.ndarray, normal: np.ndarray) -> dict[str, float]:
    dark_sum = np.sum(dark, axis=2, keepdims=True) + EPS
    normal_sum = np.sum(normal, axis=2, keepdims=True) + EPS
    dark_chroma = dark / dark_sum
    normal_chroma = normal / normal_sum
    dark_mean = np.mean(dark_chroma.reshape(-1, 3), axis=0)
    normal_mean = np.mean(normal_chroma.reshape(-1, 3), axis=0)
    delta = dark_mean - normal_mean
    payload = {
        "chromaticity_delta_l1": float(np.sum(np.abs(delta))),
        "chromaticity_delta_l2": float(np.sqrt(np.sum(delta**2))),
    }
    for idx, channel in enumerate(CHANNELS):
        payload[f"dark_chromaticity_mean_{channel}"] = float(dark_mean[idx])
        payload[f"normal_chromaticity_mean_{channel}"] = float(normal_mean[idx])
        payload[f"chromaticity_delta_{channel}"] = float(delta[idx])
    return payload


def compute_noise_metrics(dark_luma: np.ndarray, normal_luma: np.ndarray) -> dict[str, float]:
    normal_grad = sobel_magnitude(normal_luma)
    threshold = float(np.percentile(normal_grad, 20.0))
    flat_mask = normal_grad < threshold
    dark_residual = median_residual(dark_luma, ksize=5)
    normal_residual = median_residual(normal_luma, ksize=5)
    sigma_dark = mad_sigma(dark_residual, flat_mask)
    sigma_normal = mad_sigma(normal_residual, flat_mask)
    payload: dict[str, float] = {
        "flat_mask_ratio": float(np.mean(flat_mask)),
        "flat_grad_threshold_normal_luma_p20": threshold,
        "noise_sigma_dark_luma_flat": sigma_dark,
        "noise_sigma_normal_luma_flat": sigma_normal,
        "noise_sigma_ratio_dark_over_normal": float(sigma_dark / max(sigma_normal, EPS)),
    }

    edges = np.percentile(normal_luma, (0, 20, 40, 60, 80, 100))
    edges[0] = -np.inf
    edges[-1] = np.inf
    snr_ratios: list[float] = []
    for idx in range(5):
        bin_mask = (normal_luma >= edges[idx]) & (normal_luma <= edges[idx + 1])
        if idx > 0:
            bin_mask = (normal_luma > edges[idx]) & (normal_luma <= edges[idx + 1])
        dark_sigma_bin = mad_sigma(dark_residual, bin_mask)
        normal_sigma_bin = mad_sigma(normal_residual, bin_mask)
        dark_mean_bin = float(np.mean(dark_luma[bin_mask])) if np.any(bin_mask) else float("nan")
        normal_mean_bin = float(np.mean(normal_luma[bin_mask])) if np.any(bin_mask) else float("nan")
        dark_snr = float(dark_mean_bin / max(dark_sigma_bin, EPS))
        normal_snr = float(normal_mean_bin / max(normal_sigma_bin, EPS))
        snr_ratio = float(dark_snr / max(normal_snr, EPS))
        payload[f"noise_sigma_dark_luma_bin{idx}"] = dark_sigma_bin
        payload[f"noise_sigma_normal_luma_bin{idx}"] = normal_sigma_bin
        payload[f"snr_proxy_dark_bin{idx}"] = dark_snr
        payload[f"snr_proxy_normal_bin{idx}"] = normal_snr
        payload[f"snr_ratio_bin{idx}"] = snr_ratio
        snr_ratios.append(snr_ratio)
    payload["snr_proxy_dark_lowbin"] = payload["snr_proxy_dark_bin0"]
    payload["snr_proxy_normal_lowbin"] = payload["snr_proxy_normal_bin0"]
    payload["snr_ratio_lowbin"] = snr_ratios[0]
    payload["snr_ratio_midbin"] = snr_ratios[2]
    payload["snr_ratio_highbin"] = snr_ratios[4]
    return payload


def compute_edge_metrics(dark_luma: np.ndarray, normal_luma: np.ndarray) -> dict[str, float]:
    dark_norm = robust_normalize(dark_luma)
    normal_norm = robust_normalize(normal_luma)
    sobel_dark = sobel_magnitude(dark_norm)
    sobel_normal = sobel_magnitude(normal_norm)
    sobel_p90_dark = float(np.percentile(sobel_dark, 90.0))
    sobel_p90_normal = float(np.percentile(sobel_normal, 90.0))
    edge_dark = sobel_dark > sobel_p90_dark
    edge_normal = sobel_normal > sobel_p90_normal
    intersection = int(np.count_nonzero(edge_dark & edge_normal))
    dark_count = int(np.count_nonzero(edge_dark))
    normal_count = int(np.count_nonzero(edge_normal))
    precision = float(intersection / max(dark_count, 1))
    recall = float(intersection / max(normal_count, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, EPS))
    return {
        "sobel_mean_dark": float(np.mean(sobel_dark)),
        "sobel_mean_normal": float(np.mean(sobel_normal)),
        "sobel_p90_dark": sobel_p90_dark,
        "sobel_p90_normal": sobel_p90_normal,
        "gradient_ratio": float(sobel_p90_dark / max(sobel_p90_normal, EPS)),
        "gradient_corr": corrcoef_safe(sobel_dark, sobel_normal),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_dark_pixels": float(dark_count),
        "edge_normal_pixels": float(normal_count),
        "edge_intersection_pixels": float(intersection),
    }


def fit_linear_and_gamma(dark_luma: np.ndarray, normal_luma: np.ndarray, *, seed: int, max_points: int = 50000) -> dict[str, float]:
    x = np.asarray(dark_luma, dtype=np.float64).reshape(-1)
    y = np.asarray(normal_luma, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 16:
        return {
            "linear_residual_sigma": float("nan"),
            "linear_residual_mad": float("nan"),
            "gamma_residual_sigma": float("nan"),
            "gamma_residual_mad": float("nan"),
            "gamma_value": float("nan"),
            "gamma_fit_r2": float("nan"),
        }
    x = x[valid]
    y = y[valid]
    p5, p95 = np.percentile(x, (5.0, 95.0))
    mid = (x >= p5) & (x <= p95)
    if int(np.count_nonzero(mid)) >= 16:
        x = x[mid]
        y = y[mid]
    if x.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    def solve_from_feature(feature: np.ndarray) -> tuple[float, float, np.ndarray]:
        design = np.stack([feature, np.ones_like(feature)], axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        return float(coef[0]), float(coef[1]), pred

    a_lin, b_lin, pred_lin = solve_from_feature(x)
    residual_lin = y - pred_lin
    best: tuple[float, float, float, np.ndarray] | None = None
    x_clip = np.clip(x, EPS, None)
    for gamma in np.linspace(0.35, 2.80, 50):
        feature = np.power(x_clip, float(gamma))
        a, b, pred = solve_from_feature(feature)
        sse = float(np.mean((y - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(gamma), a, pred)
    assert best is not None
    _, gamma_value, a_gamma, pred_gamma = best
    residual_gamma = y - pred_gamma
    y_var = float(np.var(y))
    gamma_r2 = 1.0 - float(np.var(residual_gamma) / max(y_var, EPS))
    return {
        "linear_a": a_lin,
        "linear_b": b_lin,
        "linear_residual_sigma": float(np.std(residual_lin)),
        "linear_residual_mad": mad_sigma(residual_lin),
        "gamma_a": a_gamma,
        "gamma_b": float(np.mean(y - a_gamma * np.power(x_clip, gamma_value))),
        "gamma_residual_sigma": float(np.std(residual_gamma)),
        "gamma_residual_mad": mad_sigma(residual_gamma),
        "gamma_value": gamma_value,
        "gamma_fit_r2": gamma_r2,
    }


def compute_registration(dark_luma: np.ndarray, normal_luma: np.ndarray) -> dict[str, float]:
    dark_norm = robust_normalize(dark_luma)
    normal_norm = robust_normalize(normal_luma)
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            normal_norm.astype(np.float32),
            dark_norm.astype(np.float32),
        )
    except cv2.error:
        shift_x, shift_y, response = float("nan"), float("nan"), float("nan")
    grad_dark = sobel_magnitude(dark_norm)
    grad_normal = sobel_magnitude(normal_norm)
    shift_abs_max = float(max(abs(shift_y), abs(shift_x))) if math.isfinite(shift_x) and math.isfinite(shift_y) else float("nan")
    outlier = bool(math.isfinite(shift_abs_max) and shift_abs_max > 2.0)
    return {
        "phase_corr_shift_y": float(shift_y),
        "phase_corr_shift_x": float(shift_x),
        "phase_corr_shift_abs_max": shift_abs_max,
        "phase_corr_response": float(response),
        "gradient_ncc": corrcoef_safe(grad_dark, grad_normal),
        "registration_outlier": float(outlier),
    }


def process_row(
    manifest_row: dict[str, Any],
    *,
    lod_root: Path,
    split: str,
    split_sample_index: int,
    crop_box: tuple[int, int, int, int],
    crop_mode: str,
    sample_seed: int,
) -> dict[str, Any]:
    raw_dark = _load_raw_rgb16_png(Path(manifest_row["raw_dark_path"]), norm_mode="uint16_div_65535")
    raw_normal = _load_raw_rgb16_png(Path(manifest_row["raw_normal_path"]), norm_mode="uint16_div_65535")
    raw_dark = _apply_crop(raw_dark, crop_box)
    raw_normal = _apply_crop(raw_normal, crop_box)
    dark_luma = luma(raw_dark)
    normal_luma = luma(raw_normal)
    row: dict[str, Any] = {
        "pair_id": manifest_row["pair_id"],
        "split": split,
        "split_sample_index": int(split_sample_index),
        "normal_id": int(manifest_row["normal_id"]),
        "dark_id": int(manifest_row["dark_id"]),
        "rgb_normal_path": str(manifest_row["rgb_normal_path"]),
        "rgb_dark_path": str(manifest_row["rgb_dark_path"]),
        "raw_normal_path": str(manifest_row["raw_normal_path"]),
        "raw_dark_path": str(manifest_row["raw_dark_path"]),
        "pseudo_depth_path": str(manifest_row["target_path"]),
        "crop_mode": crop_mode,
        "crop_top": int(crop_box[0]),
        "crop_left": int(crop_box[1]),
        "crop_height": int(crop_box[2]),
        "crop_width": int(crop_box[3]),
        "lod_root": str(lod_root),
    }
    row.update(channel_and_luma_stats("dark", raw_dark))
    row.update(channel_and_luma_stats("normal", raw_normal))
    row.update(compute_exposure_and_color(row))
    row.update(compute_chromaticity(raw_dark, raw_normal))
    row.update(compute_noise_metrics(dark_luma, normal_luma))
    row.update(compute_edge_metrics(dark_luma, normal_luma))
    row.update(fit_linear_and_gamma(dark_luma, normal_luma, seed=sample_seed))
    row.update(compute_registration(dark_luma, normal_luma))
    return row


def choose_rows(rows: list[dict[str, Any]], max_samples: int, *, seed: int) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if max_samples < 0 or max_samples >= len(indexed):
        return indexed
    rng = random.Random(int(seed))
    selected = rng.sample(indexed, k=int(max_samples))
    return sorted(selected, key=lambda item: item[0])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        payload: dict[str, Any] = {
            "samples": len(split_rows),
            "registration_outliers": int(sum(int(safe_float(row.get("registration_outlier", 0.0))) for row in split_rows)),
            "metrics": {},
        }
        for field in SUMMARY_FIELDS:
            payload["metrics"][field] = numeric_summary([row.get(field, float("nan")) for row in split_rows])
        summary[split] = payload
    return summary


def format_float(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_summary_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LOD RAW Degradation Audit Summary",
        "",
        "Crop: `center512x960` matches val center crop when used.",
        "",
        "## Split Summary",
        "",
        "| split | samples | reg outliers | ev_gap_luma p50 | snr_ratio_lowbin p50 | edge_recall p50 | chroma_delta_l1 p50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, payload in summary.items():
        metrics = payload["metrics"]
        lines.append(
            "| {split} | {samples} | {outliers} | {ev} | {snr} | {edge} | {chroma} |".format(
                split=split,
                samples=payload["samples"],
                outliers=payload["registration_outliers"],
                ev=format_float(metrics["ev_gap_luma"].get("p50")),
                snr=format_float(metrics["snr_ratio_lowbin"].get("p50")),
                edge=format_float(metrics["edge_recall"].get("p50")),
                chroma=format_float(metrics["chromaticity_delta_l1"].get("p50")),
            )
        )
    lines.extend(["", "## Key Metrics", ""])
    for split, payload in summary.items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| metric | mean | p10 | p50 | p90 |")
        lines.append("|---|---:|---:|---:|---:|")
        for field in SUMMARY_FIELDS:
            stat = payload["metrics"][field]
            lines.append(
                f"| `{field}` | {format_float(stat.get('mean'))} | {format_float(stat.get('p10'))} | "
                f"{format_float(stat.get('p50'))} | {format_float(stat.get('p90'))} |"
            )
        lines.append("")
    outliers = [row for row in rows if safe_float(row.get("registration_outlier")) >= 0.5]
    lines.extend(["## Registration Outliers", ""])
    if outliers:
        lines.append("| split | pair_id | shift_y | shift_x | response |")
        lines.append("|---|---|---:|---:|---:|")
        for row in outliers[:50]:
            lines.append(
                f"| {row['split']} | {row['pair_id']} | {format_float(row.get('phase_corr_shift_y'))} | "
                f"{format_float(row.get('phase_corr_shift_x'))} | {format_float(row.get('phase_corr_response'))} |"
            )
    else:
        lines.append("No samples exceeded `abs(phase_corr_shift) > 2 px`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_hist(path: Path, series: list[tuple[str, list[float]]], *, xlabel: str, title: str, bins: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in series:
        arr = np.asarray([safe_float(v) for v in values], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            plt.hist(arr, bins=bins, alpha=0.55, label=f"{label} (n={arr.size})")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_histograms(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    hist_dir = output_dir / "histograms"
    plot_hist(
        hist_dir / "raw_dark_vs_normal_percentiles.png",
        [
            ("dark_luma_p99", [row["dark_luma_p99"] for row in rows]),
            ("normal_luma_p99", [row["normal_luma_p99"] for row in rows]),
            ("dark_luma_p95", [row["dark_luma_p95"] for row in rows]),
            ("normal_luma_p95", [row["normal_luma_p95"] for row in rows]),
        ],
        xlabel="luma percentile value",
        title="RAW_Dark vs RAW_normal luma high percentiles",
    )
    plot_hist(
        hist_dir / "ev_gap_distribution.png",
        [("ev_gap_luma", [row["ev_gap_luma"] for row in rows])],
        xlabel="EV gap from p95 luma",
        title="EV gap distribution",
    )
    plot_hist(
        hist_dir / "noise_proxy_distribution.png",
        [
            ("noise_sigma_ratio", [row["noise_sigma_ratio_dark_over_normal"] for row in rows]),
            ("snr_ratio_lowbin", [row["snr_ratio_lowbin"] for row in rows]),
        ],
        xlabel="ratio",
        title="Noise / SNR proxy distributions",
    )
    plot_hist(
        hist_dir / "edge_retention_distribution.png",
        [
            ("edge_recall", [row["edge_recall"] for row in rows]),
            ("gradient_corr", [row["gradient_corr"] for row in rows]),
        ],
        xlabel="edge metric",
        title="Edge retention distributions",
    )


def preview_image(raw_rgb: np.ndarray) -> Image.Image:
    arr = robust_normalize(np.asarray(raw_rgb, dtype=np.float32))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return Image.fromarray(np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))


def heat_image(values: np.ndarray) -> Image.Image:
    arr = np.asarray(values, dtype=np.float32)
    arr = robust_normalize(arr)
    cmap = plt.get_cmap("magma")
    rgba = cmap(arr)
    rgb = np.round(rgba[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def edge_image(values: np.ndarray) -> Image.Image:
    edge = sobel_magnitude(robust_normalize(values))
    return heat_image(edge)


def resize_tile(image: Image.Image, width: int = 280, height: int = 160) -> Image.Image:
    return image.resize((width, height), Image.Resampling.BILINEAR)


def write_panel(path: Path, row: dict[str, Any], crop_box: tuple[int, int, int, int], title: str) -> None:
    raw_dark = _apply_crop(_load_raw_rgb16_png(Path(row["raw_dark_path"]), norm_mode="uint16_div_65535"), crop_box)
    raw_normal = _apply_crop(_load_raw_rgb16_png(Path(row["raw_normal_path"]), norm_mode="uint16_div_65535"), crop_box)
    dark_l = luma(raw_dark)
    normal_l = luma(raw_normal)
    diff = np.abs(normal_l - dark_l)
    tiles = [
        ("RAW_normal", preview_image(raw_normal)),
        ("RAW_dark", preview_image(raw_dark)),
        ("abs luma diff", heat_image(diff)),
        ("normal edges", edge_image(normal_l)),
        ("dark edges", edge_image(dark_l)),
    ]
    tile_w, tile_h = 280, 160
    header_h = 62
    label_h = 22
    canvas = Image.new("RGB", (tile_w * len(tiles), header_h + label_h + tile_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))
    metric_line = (
        f"pair={row['pair_id']} split={row['split']} ev={format_float(row.get('ev_gap_luma'))} "
        f"snr_low={format_float(row.get('snr_ratio_lowbin'))} edge_rec={format_float(row.get('edge_recall'))} "
        f"shift=({format_float(row.get('phase_corr_shift_y'))},{format_float(row.get('phase_corr_shift_x'))})"
    )
    draw.text((8, 28), metric_line, fill=(0, 0, 0))
    for idx, (label, image) in enumerate(tiles):
        x = idx * tile_w
        draw.text((x + 8, header_h), label, fill=(0, 0, 0))
        canvas.paste(resize_tile(image, tile_w, tile_h), (x, header_h + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def safe_name(value: Any) -> str:
    text = str(value)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def write_panels(output_dir: Path, rows: list[dict[str, Any]], crop_box: tuple[int, int, int, int], count: int) -> None:
    if count <= 0:
        return
    categories: list[tuple[str, list[dict[str, Any]], str]] = []
    categories.append(
        (
            "high_noise_top20",
            sorted(rows, key=lambda r: safe_float(r.get("noise_sigma_ratio_dark_over_normal")), reverse=True)[:count],
            "High noise ratio",
        )
    )
    categories.append(
        (
            "high_ev_gap_top20",
            sorted(rows, key=lambda r: safe_float(r.get("ev_gap_luma")), reverse=True)[:count],
            "High EV gap",
        )
    )
    categories.append(
        (
            "low_edge_retention_top20",
            sorted(rows, key=lambda r: safe_float(r.get("edge_recall")))[:count],
            "Low edge retention",
        )
    )
    val_rows = [row for row in rows if str(row.get("split")) == "01Valid"]
    rng = random.Random(42)
    if len(val_rows) > count:
        val_rows = rng.sample(val_rows, k=count)
    categories.append(("random_val", sorted(val_rows, key=lambda r: str(r.get("pair_id"))), "Random val"))
    for category, selected, title in categories:
        for idx, row in enumerate(selected):
            filename = f"{idx:03d}_{safe_name(row['pair_id'])}.png"
            write_panel(output_dir / "panels" / category / filename, row, crop_box, title)


def main() -> None:
    args = parse_args()
    lod_root = args.lod_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_box = crop_box_for_mode(args.crop_mode)
    all_rows: list[dict[str, Any]] = []
    splits = split_names(args.splits)
    for split_idx, split in enumerate(splits):
        manifest_rows = _load_lod_true_manifest_rows(manifest, lod_root, split)
        selected = choose_rows(
            manifest_rows,
            int(args.max_samples_per_split),
            seed=int(args.sample_seed) + split_idx * 1009,
        )
        print(f"[AUDIT] split={split} selected={len(selected)}/{len(manifest_rows)} crop={args.crop_mode}", flush=True)
        for local_idx, (manifest_index, manifest_row) in enumerate(selected):
            row = process_row(
                manifest_row,
                lod_root=lod_root,
                split=split,
                split_sample_index=manifest_index,
                crop_box=crop_box,
                crop_mode=args.crop_mode,
                sample_seed=int(args.sample_seed) * 1000003 + manifest_index,
            )
            all_rows.append(row)
            if (local_idx + 1) % 50 == 0 or local_idx + 1 == len(selected):
                print(f"[AUDIT] split={split} processed={local_idx + 1}/{len(selected)}", flush=True)

    write_csv(output_dir / "per_sample_raw_stats.csv", all_rows)
    summary = build_summary(all_rows)
    with (output_dir / "summary_by_split.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_summary_md(output_dir / "summary_by_split.md", summary, all_rows)
    write_histograms(output_dir, all_rows)
    write_panels(output_dir, all_rows, crop_box, int(args.panel_count))
    run_meta = {
        "lod_root": str(lod_root),
        "manifest": str(manifest),
        "splits": splits,
        "crop_mode": args.crop_mode,
        "crop_box": list(crop_box),
        "max_samples_per_split": int(args.max_samples_per_split),
        "sample_seed": int(args.sample_seed),
        "rows_written": len(all_rows),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2, sort_keys=True)
    print(f"[AUDIT] wrote {len(all_rows)} rows to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
