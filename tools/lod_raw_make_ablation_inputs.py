#!/usr/bin/env python3
"""Generate LOD RAW ablation PNG16 inputs and manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_true import (  # noqa: E402
    LOD_TRUE_NATIVE_HW,
    LOD_TRUE_REQUIRED_COLUMNS,
    _load_raw_rgb16_png,
)


EPS = 1e-8
DEFAULT_TRANSFORMS = (
    "raw_dark_identity",
    "raw_dark_denoise_median3",
    "raw_dark_bilateral_weak_trainfit_sigma",
    "raw_dark_bilateral_medium_trainfit_sigma",
    "raw_dark_bilateral_strong_trainfit_sigma",
    "raw_dark_tone_percentile_trainfit",
    "raw_dark_tone_percentile_luma_trainfit",
    "raw_dark_tone_gamma_trainfit",
    "raw_dark_denoise_then_tone_trainfit",
    "raw_dark_tone_then_denoise_trainfit",
    "raw_dark_tone_percentile_oracle",
    "raw_dark_tone_denoise_oracle",
    "raw_normal_identity",
    "raw_normal_to_dark_exposure_trainfit",
    "raw_normal_to_dark_noise_only_trainfit",
    "raw_normal_to_dark_exposure_noise_trainfit",
)
DARK_TRANSFORMS = {
    "raw_dark_identity",
    "raw_dark_denoise_median3",
    "raw_dark_bilateral_weak_trainfit_sigma",
    "raw_dark_bilateral_medium_trainfit_sigma",
    "raw_dark_bilateral_strong_trainfit_sigma",
    "raw_dark_tone_percentile_trainfit",
    "raw_dark_tone_percentile_luma_trainfit",
    "raw_dark_tone_gamma_trainfit",
    "raw_dark_denoise_then_tone_trainfit",
    "raw_dark_tone_then_denoise_trainfit",
    "raw_dark_tone_percentile_oracle",
    "raw_dark_tone_denoise_oracle",
}
NORMAL_TRANSFORMS = {
    "raw_normal_identity",
    "raw_normal_to_dark_exposure_trainfit",
    "raw_normal_to_dark_noise_only_trainfit",
    "raw_normal_to_dark_exposure_noise_trainfit",
}
CHANNELS = ("R", "G", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transforms", default=",".join(DEFAULT_TRANSFORMS))
    parser.add_argument("--fit-split", default="00Train")
    parser.add_argument("--fit-max-samples", type=int, default=-1)
    parser.add_argument("--fit-pixels-per-sample", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=-1, help="Smoke/debug limiter. -1 processes every manifest row.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_transform_names(value: str) -> list[str]:
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        raise ValueError("--transforms produced an empty list")
    known = DARK_TRANSFORMS | NORMAL_TRANSFORMS
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"Unknown transform(s): {', '.join(unknown)}")
    return names


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in LOD_TRUE_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def select_fit_rows(rows: list[dict[str, str]], split: str, max_samples: int, seed: int) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"].strip() == split]
    if max_samples >= 0 and max_samples < len(selected):
        rng = random.Random(int(seed))
        selected = rng.sample(selected, k=int(max_samples))
        selected = sorted(selected, key=lambda row: int(row["normal_id"]))
    if not selected:
        raise ValueError(f"No fit rows selected for split={split!r}")
    return selected


def select_process_rows(rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    if max_rows < 0 or max_rows >= len(rows):
        return rows
    return rows[: int(max_rows)]


def load_raw_rgb(path: Path) -> np.ndarray:
    return _load_raw_rgb16_png(path, norm_mode="uint16_div_65535")


def write_raw_rgb(path: Path, raw_rgb: np.ndarray, compression: int) -> None:
    arr = np.asarray(raw_rgb, dtype=np.float32)
    if arr.shape != (LOD_TRUE_NATIVE_HW[0], LOD_TRUE_NATIVE_HW[1], 3):
        raise ValueError(f"Expected raw RGB shape {(*LOD_TRUE_NATIVE_HW, 3)}, got {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_u16_rgb = np.round(np.clip(arr, 0.0, 1.0) * 65535.0).astype(np.uint16)
    ok = cv2.imwrite(str(path), raw_u16_rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise IOError(f"cv2.imwrite failed: {path}")


def luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def sobel_magnitude(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def median_residual(values: np.ndarray, ksize: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr - cv2.medianBlur(arr, ksize)


def mad_sigma(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def sample_pixel_indices(height: int, width: int, count: int, rng: np.random.Generator, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if mask is not None:
        flat = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
        if flat.size == 0:
            flat = np.arange(height * width)
    else:
        flat = np.arange(height * width)
    if count > 0 and flat.size > count:
        flat = rng.choice(flat, size=int(count), replace=False)
    yy, xx = np.divmod(flat, width)
    return yy.astype(np.int64), xx.astype(np.int64)


def percentile_mapping(x: np.ndarray, src_p1: np.ndarray, src_p99: np.ndarray, dst_p1: np.ndarray, dst_p99: np.ndarray) -> np.ndarray:
    src_p1 = np.asarray(src_p1, dtype=np.float32)
    src_p99 = np.asarray(src_p99, dtype=np.float32)
    dst_p1 = np.asarray(dst_p1, dtype=np.float32)
    dst_p99 = np.asarray(dst_p99, dtype=np.float32)
    z = (np.asarray(x, dtype=np.float32) - src_p1) / (src_p99 - src_p1 + EPS)
    y = z * (dst_p99 - dst_p1) + dst_p1
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def fit_gamma(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 32:
        return {"gamma": 1.0, "a": 1.0, "b": 0.0, "residual_sigma": float("nan"), "r2": float("nan")}
    p5, p95 = np.percentile(x, (5, 95))
    mid = (x >= p5) & (x <= p95)
    if int(np.count_nonzero(mid)) >= 32:
        x = x[mid]
        y = y[mid]
    x = np.clip(x, EPS, None)
    best: dict[str, float] | None = None
    for gamma in np.linspace(0.35, 2.80, 50):
        feat = np.power(x, float(gamma))
        design = np.stack([feat, np.ones_like(feat)], axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        residual = y - pred
        sse = float(np.mean(residual**2))
        if best is None or sse < best["sse"]:
            y_var = float(np.var(y))
            best = {
                "sse": sse,
                "gamma": float(gamma),
                "a": float(coef[0]),
                "b": float(coef[1]),
                "residual_sigma": float(np.std(residual)),
                "r2": float(1.0 - np.var(residual) / max(y_var, EPS)),
            }
    assert best is not None
    best.pop("sse", None)
    return best


def fit_noise_line(x: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    signal = np.asarray(x, dtype=np.float64).reshape(-1)
    var = np.asarray(residual, dtype=np.float64).reshape(-1) ** 2
    valid = np.isfinite(signal) & np.isfinite(var)
    signal = signal[valid]
    var = var[valid]
    if signal.size < 32:
        return {"a": 0.0, "b": float(np.var(residual))}
    design = np.stack([signal, np.ones_like(signal)], axis=1)
    coef, *_ = np.linalg.lstsq(design, var, rcond=None)
    a = max(float(coef[0]), 0.0)
    b = max(float(coef[1]), 0.0)
    return {"a": a, "b": b}


def fit_train_params(rows: list[dict[str, str]], lod_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.sample_seed))
    dark_samples: list[np.ndarray] = []
    normal_samples: list[np.ndarray] = []
    dark_luma_samples: list[np.ndarray] = []
    normal_luma_samples: list[np.ndarray] = []
    flat_dark_noise_sigmas: list[float] = []
    noise_signal_by_channel: list[list[np.ndarray]] = [[], [], []]
    noise_residual_by_channel: list[list[np.ndarray]] = [[], [], []]
    print(f"[FIT] rows={len(rows)} pixels_per_sample={args.fit_pixels_per_sample}", flush=True)
    for idx, row in enumerate(rows):
        dark = load_raw_rgb(resolve_path(lod_root, row["raw_dark_path"]))
        normal = load_raw_rgb(resolve_path(lod_root, row["raw_normal_path"]))
        h, w = dark.shape[:2]
        yy, xx = sample_pixel_indices(h, w, int(args.fit_pixels_per_sample), rng)
        dark_pix = dark[yy, xx, :]
        normal_pix = normal[yy, xx, :]
        dark_samples.append(dark_pix)
        normal_samples.append(normal_pix)
        dark_l = luma(dark)
        normal_l = luma(normal)
        dark_luma_samples.append(dark_l[yy, xx])
        normal_luma_samples.append(normal_l[yy, xx])
        normal_grad = sobel_magnitude(normal_l)
        flat_mask = normal_grad < float(np.percentile(normal_grad, 20.0))
        flat_dark_noise_sigmas.append(mad_sigma(median_residual(dark_l), flat_mask))
        fyy, fxx = sample_pixel_indices(h, w, int(args.fit_pixels_per_sample), rng, mask=flat_mask)
        dark_flat = dark[fyy, fxx, :]
        normal_flat = normal[fyy, fxx, :]
        # Temporary per-row percentile exposure estimate for conservative residual sampling.
        d_p1, d_p99 = np.percentile(dark.reshape(-1, 3), (1, 99), axis=0)
        n_p1, n_p99 = np.percentile(normal.reshape(-1, 3), (1, 99), axis=0)
        exposed_flat = percentile_mapping(normal_flat, n_p1, n_p99, d_p1, d_p99)
        residual = dark_flat - exposed_flat
        for c in range(3):
            noise_signal_by_channel[c].append(exposed_flat[:, c])
            noise_residual_by_channel[c].append(residual[:, c])
        if (idx + 1) % 100 == 0 or idx + 1 == len(rows):
            print(f"[FIT] processed={idx + 1}/{len(rows)}", flush=True)
    dark_pixels = np.concatenate(dark_samples, axis=0)
    normal_pixels = np.concatenate(normal_samples, axis=0)
    dark_luma = np.concatenate(dark_luma_samples, axis=0)
    normal_luma = np.concatenate(normal_luma_samples, axis=0)
    percentile = {
        "dark_p1_rgb": np.percentile(dark_pixels, 1, axis=0).astype(float).tolist(),
        "dark_p99_rgb": np.percentile(dark_pixels, 99, axis=0).astype(float).tolist(),
        "normal_p1_rgb": np.percentile(normal_pixels, 1, axis=0).astype(float).tolist(),
        "normal_p99_rgb": np.percentile(normal_pixels, 99, axis=0).astype(float).tolist(),
        "dark_p1_luma": float(np.percentile(dark_luma, 1)),
        "dark_p99_luma": float(np.percentile(dark_luma, 99)),
        "normal_p1_luma": float(np.percentile(normal_luma, 1)),
        "normal_p99_luma": float(np.percentile(normal_luma, 99)),
    }
    gamma = {}
    for idx, channel in enumerate(CHANNELS):
        gamma[channel] = fit_gamma(dark_pixels[:, idx], normal_pixels[:, idx])
    noise_model = {}
    for idx, channel in enumerate(CHANNELS):
        signal = np.concatenate(noise_signal_by_channel[idx], axis=0)
        residual = np.concatenate(noise_residual_by_channel[idx], axis=0)
        noise_model[channel] = fit_noise_line(signal, residual)
    noise_model["dark_luma_flat_sigma_median"] = float(np.nanmedian(np.asarray(flat_dark_noise_sigmas, dtype=np.float64)))
    return {
        "fit_split": str(args.fit_split),
        "fit_rows": len(rows),
        "fit_pixels_per_sample": int(args.fit_pixels_per_sample),
        "percentile": percentile,
        "gamma": gamma,
        "noise_model": noise_model,
    }


def median3(raw: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw)
    for c in range(3):
        out[..., c] = cv2.medianBlur(raw[..., c].astype(np.float32), 3)
    return np.clip(out, 0.0, 1.0)


def bilateral(raw: np.ndarray, sigma_color: float) -> np.ndarray:
    return np.clip(
        cv2.bilateralFilter(
            raw.astype(np.float32),
            d=5,
            sigmaColor=float(max(sigma_color, 1e-5)),
            sigmaSpace=3.0,
        ),
        0.0,
        1.0,
    )


def tone_percentile_dark_to_normal(raw: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    p = params["percentile"]
    return percentile_mapping(
        raw,
        np.asarray(p["dark_p1_rgb"], dtype=np.float32),
        np.asarray(p["dark_p99_rgb"], dtype=np.float32),
        np.asarray(p["normal_p1_rgb"], dtype=np.float32),
        np.asarray(p["normal_p99_rgb"], dtype=np.float32),
    )


def tone_percentile_luma_dark_to_normal(raw: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    p = params["percentile"]
    old_luma = luma(raw)
    new_luma = percentile_mapping(
        old_luma[..., None],
        np.asarray([p["dark_p1_luma"]], dtype=np.float32),
        np.asarray([p["dark_p99_luma"]], dtype=np.float32),
        np.asarray([p["normal_p1_luma"]], dtype=np.float32),
        np.asarray([p["normal_p99_luma"]], dtype=np.float32),
    )[..., 0]
    ratio = new_luma / np.maximum(old_luma, EPS)
    return np.clip(raw * ratio[..., None], 0.0, 1.0)


def tone_gamma_dark_to_normal(raw: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    out = np.empty_like(raw)
    for idx, channel in enumerate(CHANNELS):
        gp = params["gamma"][channel]
        out[..., idx] = float(gp["a"]) * np.power(np.clip(raw[..., idx], EPS, None), float(gp["gamma"])) + float(gp["b"])
    return np.clip(out, 0.0, 1.0)


def normal_to_dark_exposure(raw: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    p = params["percentile"]
    return percentile_mapping(
        raw,
        np.asarray(p["normal_p1_rgb"], dtype=np.float32),
        np.asarray(p["normal_p99_rgb"], dtype=np.float32),
        np.asarray(p["dark_p1_rgb"], dtype=np.float32),
        np.asarray(p["dark_p99_rgb"], dtype=np.float32),
    )


def tone_percentile_dark_to_normal_oracle(raw_dark: np.ndarray, raw_normal: np.ndarray) -> np.ndarray:
    dark_p1, dark_p99 = np.percentile(raw_dark.reshape(-1, 3), (1, 99), axis=0)
    normal_p1, normal_p99 = np.percentile(raw_normal.reshape(-1, 3), (1, 99), axis=0)
    return percentile_mapping(raw_dark, dark_p1, dark_p99, normal_p1, normal_p99)


def tone_denoise_dark_to_normal_oracle(raw_dark: np.ndarray, raw_normal: np.ndarray) -> np.ndarray:
    toned = tone_percentile_dark_to_normal_oracle(raw_dark, raw_normal)
    normal_l = luma(raw_normal)
    flat_mask = sobel_magnitude(normal_l) < float(np.percentile(sobel_magnitude(normal_l), 20.0))
    residual_sigma = mad_sigma(luma(toned) - normal_l, flat_mask)
    if not np.isfinite(residual_sigma) or residual_sigma <= 0:
        residual_sigma = 0.01
    return bilateral(toned, sigma_color=2.0 * float(residual_sigma))


def add_noise(raw: np.ndarray, params: dict[str, Any], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = raw.astype(np.float32).copy()
    for idx, channel in enumerate(CHANNELS):
        model = params["noise_model"][channel]
        sigma = np.sqrt(np.maximum(float(model["a"]) * out[..., idx] + float(model["b"]), 0.0)).astype(np.float32)
        out[..., idx] += rng.normal(0.0, 1.0, size=out[..., idx].shape).astype(np.float32) * sigma
    return np.clip(out, 0.0, 1.0)


def transform_source_and_fn(
    name: str,
    params: dict[str, Any],
    noise_seed: int,
    lod_root: Path,
) -> tuple[str, Callable[[np.ndarray, dict[str, str]], np.ndarray]]:
    sigma_base = float(params["noise_model"].get("dark_luma_flat_sigma_median", 0.03))
    sigma_by_name = {
        "raw_dark_bilateral_weak_trainfit_sigma": 1.0 * sigma_base,
        "raw_dark_bilateral_medium_trainfit_sigma": 2.0 * sigma_base,
        "raw_dark_bilateral_strong_trainfit_sigma": 4.0 * sigma_base,
    }
    if name == "raw_dark_identity":
        return "raw_dark_path", lambda raw, row: raw
    if name == "raw_dark_denoise_median3":
        return "raw_dark_path", lambda raw, row: median3(raw)
    if name in sigma_by_name:
        return "raw_dark_path", lambda raw, row, sigma=sigma_by_name[name]: bilateral(raw, sigma)
    if name == "raw_dark_tone_percentile_trainfit":
        return "raw_dark_path", lambda raw, row: tone_percentile_dark_to_normal(raw, params)
    if name == "raw_dark_tone_percentile_luma_trainfit":
        return "raw_dark_path", lambda raw, row: tone_percentile_luma_dark_to_normal(raw, params)
    if name == "raw_dark_tone_gamma_trainfit":
        return "raw_dark_path", lambda raw, row: tone_gamma_dark_to_normal(raw, params)
    if name == "raw_dark_denoise_then_tone_trainfit":
        return "raw_dark_path", lambda raw, row: tone_percentile_dark_to_normal(bilateral(raw, 2.0 * sigma_base), params)
    if name == "raw_dark_tone_then_denoise_trainfit":
        return "raw_dark_path", lambda raw, row: bilateral(tone_percentile_dark_to_normal(raw, params), 2.0 * sigma_base)
    if name == "raw_dark_tone_percentile_oracle":
        return "raw_dark_path", lambda raw, row: tone_percentile_dark_to_normal_oracle(
            raw,
            load_raw_rgb(resolve_path(lod_root, row["raw_normal_path"])),
        )
    if name == "raw_dark_tone_denoise_oracle":
        return "raw_dark_path", lambda raw, row: tone_denoise_dark_to_normal_oracle(
            raw,
            load_raw_rgb(resolve_path(lod_root, row["raw_normal_path"])),
        )
    if name == "raw_normal_identity":
        return "raw_normal_path", lambda raw, row: raw
    if name == "raw_normal_to_dark_exposure_trainfit":
        return "raw_normal_path", lambda raw, row: normal_to_dark_exposure(raw, params)
    if name == "raw_normal_to_dark_noise_only_trainfit":
        return "raw_normal_path", lambda raw, row: add_noise(raw, params, noise_seed + int(row["normal_id"]))
    if name == "raw_normal_to_dark_exposure_noise_trainfit":
        return "raw_normal_path", lambda raw, row: add_noise(
            normal_to_dark_exposure(raw, params),
            params,
            noise_seed + int(row["normal_id"]),
        )
    raise ValueError(f"Unsupported transform={name}")


def output_png_path(output_dir: Path, transform_name: str, src_path: Path) -> Path:
    return output_dir / "png16" / transform_name / src_path.name


def generate_transform(
    *,
    name: str,
    rows: list[dict[str, str]],
    all_rows: list[dict[str, str]],
    fieldnames: list[str],
    lod_root: Path,
    output_dir: Path,
    params: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_column, fn = transform_source_and_fn(name, params, int(args.noise_seed), lod_root)
    generated_by_pair: dict[str, str] = {}
    for idx, row in enumerate(rows):
        src_path = resolve_path(lod_root, row[source_column])
        out_path = output_png_path(output_dir, name, src_path)
        if args.overwrite or not out_path.is_file():
            raw = load_raw_rgb(src_path)
            transformed = fn(raw, row)
            write_raw_rgb(out_path, transformed, int(args.png_compression))
        generated_by_pair[str(row["pair_id"])] = str(out_path)
        if (idx + 1) % 100 == 0 or idx + 1 == len(rows):
            print(f"[GEN][{name}] processed={idx + 1}/{len(rows)}", flush=True)
    manifest_rows = [dict(row) for row in all_rows]
    for row in manifest_rows:
        pair_id = str(row["pair_id"])
        if pair_id in generated_by_pair:
            row[source_column] = generated_by_pair[pair_id]
    manifest_path = output_dir / "manifests" / f"{name}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "transform": name,
        "source_column": source_column,
        "manifest": str(manifest_path),
        "png_dir": str(output_dir / "png16" / name),
        "generated_rows": len(rows),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    lod_root = args.lod_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms = parse_transform_names(args.transforms)
    fieldnames, all_rows = read_manifest(manifest)
    process_rows = select_process_rows(all_rows, int(args.max_rows))
    fit_rows = select_fit_rows(
        all_rows,
        split=str(args.fit_split),
        max_samples=int(args.fit_max_samples),
        seed=int(args.sample_seed),
    )
    params = fit_train_params(fit_rows, lod_root, args)
    write_json(output_dir / "transform_params" / "trainfit_percentile_params.json", params["percentile"])
    write_json(output_dir / "transform_params" / "trainfit_gamma_params.json", params["gamma"])
    write_json(output_dir / "transform_params" / "trainfit_noise_model.json", params["noise_model"])
    write_json(output_dir / "transform_params" / "trainfit_all_params.json", params)
    results = []
    print(
        f"[GEN] transforms={len(transforms)} rows={len(process_rows)}/{len(all_rows)} output_dir={output_dir}",
        flush=True,
    )
    for name in transforms:
        result = generate_transform(
            name=name,
            rows=process_rows,
            all_rows=all_rows,
            fieldnames=fieldnames,
            lod_root=lod_root,
            output_dir=output_dir,
            params=params,
            args=args,
        )
        results.append(result)
    run_meta = {
        "lod_root": str(lod_root),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "transforms": transforms,
        "fit_split": str(args.fit_split),
        "fit_rows": len(fit_rows),
        "processed_rows": len(process_rows),
        "manifest_rows": len(all_rows),
        "max_rows": int(args.max_rows),
        "fit_max_samples": int(args.fit_max_samples),
        "fit_pixels_per_sample": int(args.fit_pixels_per_sample),
        "noise_seed": int(args.noise_seed),
        "results": results,
    }
    write_json(output_dir / "run_metadata.json", run_meta)
    print(f"[GEN] completed transforms={len(results)} output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
