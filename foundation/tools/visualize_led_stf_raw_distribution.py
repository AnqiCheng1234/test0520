#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finetune_stf.dataset.raw_storage import get_raw_storage_spec
from finetune_stf.dataset.raw_utils import (
    DEFAULT_RAW_NPZ_ROOT,
    RECTIFIED_BAYER_KEY,
    decode_stf_raw_by_storage_format,
    load_rectified_bayer_npz,
)
from finetune_stf.dataset.stf import DEFAULT_STF_ROOT
from foundation.engine.datasets.led_hb import (
    LEDHBRaw,
    default_led_hb_raw_adapter_config,
)


CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
RGB_CHANNEL_NAMES = ("R", "G", "B")
RAW_PARAM_KEYS = (
    "raw_adapter_red_gain_range",
    "raw_adapter_blue_gain_range",
    "raw_adapter_fixed_red_gain",
    "raw_adapter_fixed_blue_gain",
    "raw_adapter_shot_noise",
    "raw_adapter_read_noise",
    "raw_adapter_black_level",
    "raw_adapter_white_level",
    "raw_adapter_fixed_light_scale",
    "raw_adapter_dark_light_scale_range",
    "raw_adapter_over_light_scale_range",
    "raw_adapter_variant_policy",
    "raw_adapter_variant_weights",
    "raw_adapter_rgb_transfer",
    "raw_adapter_inverse_tone",
    "raw_adapter_ccm",
    "raw_adapter_cfa_pattern",
    "raw_adapter_packed_channel_order",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quantiles(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    qs = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)
    return {f"q{q:g}": float(np.quantile(vals, q)) for q in qs}


def sample_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted({int(round(v)) for v in np.linspace(0, total - 1, count)})


def packed_to_base_rgb_chw(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float32)
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected CHW packed raw4, got {packed.shape}")
    return np.stack([packed[0], 0.5 * (packed[1] + packed[2]), packed[3]], axis=0)


def luma_from_rgb_chw(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def display_stretch(image: np.ndarray, *, gamma: float = 1.0 / 2.2) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    hi = float(np.percentile(image[finite], 99.5))
    if hi <= 1e-8:
        hi = float(np.max(image[finite]))
    if hi <= 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    out = np.clip(image / hi, 0.0, 1.0)
    return np.power(out, gamma).astype(np.float32, copy=False)


def hwc_from_chw(chw: np.ndarray) -> np.ndarray:
    return np.transpose(chw, (1, 2, 0))


def _hist_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, np.zeros(len(centers), dtype=np.float32)
    counts, _ = np.histogram(np.clip(vals, bins[0], bins[-1]), bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, counts.astype(np.float32, copy=False)


def _plot_hist_overlay(
    ax: Any,
    led_values: np.ndarray,
    stf_values: np.ndarray,
    *,
    title: str,
    xlim: tuple[float, float],
    bins: int = 160,
    log_y: bool = True,
    led_label: str = "LED calibrated",
    stf_label: str = "STF decoded",
) -> None:
    edges = np.linspace(float(xlim[0]), float(xlim[1]), int(bins) + 1)
    x, y = _hist_density(led_values, edges)
    ax.plot(x, y, color="#f0a13a", linewidth=1.6, label=led_label)
    x, y = _hist_density(stf_values, edges)
    ax.plot(x, y, color="#5aa7ff", linewidth=1.6, label=stf_label)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def _plot_channel_overlay(ax: Any, led_channels: dict[str, np.ndarray], stf_channels: dict[str, np.ndarray], xlim: tuple[float, float]) -> None:
    colors = {
        "R": "#e94f4f",
        "Gr": "#40c463",
        "Gb": "#8bc34a",
        "B": "#4f83ff",
    }
    edges = np.linspace(float(xlim[0]), float(xlim[1]), 161)
    for name in CHANNEL_NAMES:
        x, y = _hist_density(led_channels[name], edges)
        ax.plot(x, y, color=colors[name], linewidth=1.3, linestyle="-", label=f"LED {name}")
        x, y = _hist_density(stf_channels[name], edges)
        ax.plot(x, y, color=colors[name], linewidth=1.3, linestyle="--", label=f"STF {name}")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_title("per-channel raw4, solid=LED dashed=STF")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=6, ncols=2)


def _plot_cdf(
    ax: Any,
    led_values: np.ndarray,
    stf_values: np.ndarray,
    *,
    title: str,
    xlim: tuple[float, float],
    led_label: str = "LED calibrated",
    stf_label: str = "STF decoded",
) -> None:
    for values, label, color in (
        (led_values, led_label, "#f0a13a"),
        (stf_values, stf_label, "#5aa7ff"),
    ):
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals = np.sort(np.clip(vals, xlim[0], xlim[1]))
        y = np.linspace(0.0, 1.0, vals.size, endpoint=True)
        ax.plot(vals, y, color=color, linewidth=1.6, label=label)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def read_stf_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename_stem", "lut_preview"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No STF rows found in {path}")
    return rows


def filter_stf_rows(rows: list[dict[str, str]], daytime_filter: str) -> list[dict[str, str]]:
    key = str(daytime_filter).strip().lower()
    if key in ("", "all"):
        return rows
    filtered = [row for row in rows if str(row.get("daytime", "")).strip().lower() == key]
    if not filtered:
        raise ValueError(f"No STF rows matched --stf-daytime-filter={daytime_filter!r}")
    return filtered


def resolve_stf_path(stf_root: Path, value: str) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path
    return (stf_root / path).resolve()


def read_lut_preview(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read STF lut_preview: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if tuple(rgb.shape[:2]) != target_hw:
        rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32, copy=False)


def load_stf_decoded_raw(raw_npz_root: Path, sample_name: str, raw_storage_format: str) -> np.ndarray:
    path = (raw_npz_root / f"{sample_name}.npz").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing STF RAW NPZ: {path}")
    storage = load_rectified_bayer_npz(path, key=RECTIFIED_BAYER_KEY)
    decoded = decode_stf_raw_by_storage_format(storage, raw_storage_format)
    if decoded.ndim != 3 or decoded.shape[-1] != 4:
        raise ValueError(f"Expected decoded STF raw HWC4, got {decoded.shape} from {path}")
    return np.transpose(decoded.astype(np.float32, copy=False), (2, 0, 1))


def append_raw_values(stats: dict[str, Any], packed_chw: np.ndarray, *, pixel_stride: int) -> None:
    raw = np.asarray(packed_chw[:, ::pixel_stride, ::pixel_stride], dtype=np.float32)
    base = packed_to_base_rgb_chw(raw)
    for idx, name in enumerate(CHANNEL_NAMES):
        stats["channel_values"][name].append(raw[idx].reshape(-1))
    stats["raw_all_values"].append(raw.reshape(-1))
    stats["base_luma_values"].append(luma_from_rgb_chw(base).reshape(-1))
    flat = raw.reshape(-1)
    stats["per_sample"].append(
        {
            "raw_mean": float(flat.mean()),
            "raw_std": float(flat.std()),
            "raw_p50": float(np.quantile(flat, 0.50)),
            "raw_p99": float(np.quantile(flat, 0.99)),
            "raw_zero_frac_lt_1e_4": float((flat < 1e-4).mean()),
            "raw_sat_frac_gt_0p999": float((flat > 0.999).mean()),
            "base_luma_mean": float(luma_from_rgb_chw(base).mean()),
        }
    )


def append_rgb_values(stats: dict[str, Any], rgb_chw: np.ndarray, *, pixel_stride: int) -> None:
    rgb = np.asarray(rgb_chw[:, ::pixel_stride, ::pixel_stride], dtype=np.float32)
    stats["rgb_all_values"].append(rgb.reshape(-1))
    stats["rgb_luma_values"].append(luma_from_rgb_chw(rgb).reshape(-1))


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    channels = {
        name: np.concatenate(values) if values else np.empty((0,), dtype=np.float32)
        for name, values in stats["channel_values"].items()
    }
    raw_all = np.concatenate(stats["raw_all_values"]) if stats["raw_all_values"] else np.empty((0,), dtype=np.float32)
    base_luma = np.concatenate(stats["base_luma_values"]) if stats["base_luma_values"] else np.empty((0,), dtype=np.float32)
    rgb_all = np.concatenate(stats["rgb_all_values"]) if stats["rgb_all_values"] else np.empty((0,), dtype=np.float32)
    rgb_luma = np.concatenate(stats["rgb_luma_values"]) if stats["rgb_luma_values"] else np.empty((0,), dtype=np.float32)
    return {
        "count": int(stats["count"]),
        "pixel_stride": int(stats["pixel_stride"]),
        "channels": channels,
        "raw_all": raw_all,
        "base_luma": base_luma,
        "rgb_all": rgb_all,
        "rgb_luma": rgb_luma,
        "per_sample": stats["per_sample"],
        "preview_samples": stats["preview_samples"],
    }


def collect_led_stats(
    dataset: LEDHBRaw,
    *,
    max_samples: int,
    pixel_stride: int,
    panel_samples: int,
) -> dict[str, Any]:
    total = len(dataset)
    count = total if max_samples <= 0 else min(total, max_samples)
    indices = sample_indices(total, count)
    panel_index_set = set(sample_indices(len(indices), min(panel_samples, len(indices))))
    stats = {
        "count": 0,
        "pixel_stride": pixel_stride,
        "channel_values": {name: [] for name in CHANNEL_NAMES},
        "raw_all_values": [],
        "base_luma_values": [],
        "rgb_all_values": [],
        "rgb_luma_values": [],
        "per_sample": [],
        "preview_samples": [],
    }
    for local_idx, dataset_idx in enumerate(indices):
        if local_idx % 50 == 0:
            print(f"[LED] {local_idx}/{len(indices)}", flush=True)
        sample = dataset.build_sample(int(dataset_idx), include_rgb_preview=True)
        packed = to_numpy(sample["raw"]).astype(np.float32, copy=False)
        rgb = to_numpy(sample["rgb_preview"]).astype(np.float32, copy=False)
        append_rgb_values(stats, rgb, pixel_stride=pixel_stride)
        append_raw_values(stats, packed, pixel_stride=pixel_stride)
        if local_idx in panel_index_set:
            stats["preview_samples"].append(
                {
                    "dataset_index": int(dataset_idx),
                    "sample_name": str(sample["sample_name"]),
                    "rgb": rgb,
                    "raw": packed,
                    "image_path": str(sample["image_path"]),
                    "isp_params": jsonable(sample["isp_params"]),
                }
            )
        stats["count"] += 1
    return finalize_stats(stats)


def collect_stf_stats(
    rows: list[dict[str, str]],
    *,
    stf_root: Path,
    raw_npz_root: Path,
    raw_storage_format: str,
    max_samples: int,
    pixel_stride: int,
    panel_samples: int,
    target_hw: tuple[int, int],
) -> dict[str, Any]:
    total = len(rows)
    count = total if max_samples <= 0 else min(total, max_samples)
    indices = sample_indices(total, count)
    panel_index_set = set(sample_indices(len(indices), min(panel_samples, len(indices))))
    stats = {
        "count": 0,
        "pixel_stride": pixel_stride,
        "channel_values": {name: [] for name in CHANNEL_NAMES},
        "raw_all_values": [],
        "base_luma_values": [],
        "rgb_all_values": [],
        "rgb_luma_values": [],
        "per_sample": [],
        "preview_samples": [],
    }
    for local_idx, row_idx in enumerate(indices):
        if local_idx % 50 == 0:
            print(f"[STF] {local_idx}/{len(indices)}", flush=True)
        row = rows[int(row_idx)]
        sample_name = str(row["filename_stem"])
        packed = load_stf_decoded_raw(raw_npz_root, sample_name, raw_storage_format)
        rgb = read_lut_preview(resolve_stf_path(stf_root, row["lut_preview"]), target_hw=target_hw)
        rgb_chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
        append_rgb_values(stats, rgb_chw, pixel_stride=pixel_stride)
        append_raw_values(stats, packed, pixel_stride=pixel_stride)
        if local_idx in panel_index_set:
            stats["preview_samples"].append(
                {
                    "manifest_index": int(row_idx),
                    "sample_name": sample_name,
                    "rgb": rgb_chw,
                    "raw": packed,
                    "lut_preview": str(resolve_stf_path(stf_root, row["lut_preview"])),
                    "daytime": row.get("daytime", ""),
                    "fog": row.get("fog", ""),
                    "precipitation": row.get("precipitation", ""),
                    "road_state": row.get("road_state", ""),
                }
            )
        stats["count"] += 1
    return finalize_stats(stats)


def _find_candidate_dicts(payload: dict[str, Any], calibration_group: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [payload]
    for key in (
        "raw_adapter_overrides",
        "recommended_raw_adapter_overrides",
        "led_raw_adapter_overrides",
        "raw_adapter_config",
        "resolved_raw_adapter_config",
        "config",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    group_reports = payload.get("group_reports")
    if isinstance(group_reports, dict):
        group = group_reports.get(calibration_group)
        if isinstance(group, dict):
            candidates.append(group)
            ranges = group.get("recommended_brooks_ranges")
            if isinstance(ranges, dict):
                candidates.append(ranges)
    ranges = payload.get("recommended_brooks_ranges")
    if isinstance(ranges, dict):
        candidates.append(ranges)
    return candidates


def _describe_applied_calibration_group(payload: dict[str, Any], requested_group: str) -> str:
    if isinstance(payload.get("raw_adapter_overrides"), dict):
        recommended = payload.get("recommended_group_for_led_hb_night")
        if recommended:
            return str(recommended)
        return "top_level_raw_adapter_overrides"
    return str(requested_group)


def _midpoint(values: Any, *, field: str) -> float:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{field} must be a two-value range, got {values!r}")
    return 0.5 * (float(values[0]) + float(values[1]))


def _apply_exact_raw_adapter_values(config: dict[str, Any], candidates: Iterable[dict[str, Any]]) -> list[str]:
    applied: list[str] = []
    for candidate in candidates:
        for key in RAW_PARAM_KEYS:
            if key in candidate and candidate[key] is not None:
                config[key] = candidate[key]
                applied.append(key)
    return sorted(set(applied))


def _apply_ranges_as_fixed(
    config: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
) -> list[str]:
    applied: list[str] = []
    ranges: dict[str, Any] = {}
    for candidate in candidates:
        for key in (
            "red_gain_range",
            "blue_gain_range",
            "black_level_range",
            "read_noise_std_range",
            "shot_log_gain_range",
            "raw_adapter_red_gain_range",
            "raw_adapter_blue_gain_range",
        ):
            if key in candidate and candidate[key] is not None:
                ranges[key] = candidate[key]

    if "red_gain_range" in ranges:
        config["raw_adapter_red_gain_range"] = [float(v) for v in ranges["red_gain_range"]]
        config["raw_adapter_fixed_red_gain"] = _midpoint(ranges["red_gain_range"], field="red_gain_range")
        applied.extend(["raw_adapter_red_gain_range", "raw_adapter_fixed_red_gain"])
    if "blue_gain_range" in ranges:
        config["raw_adapter_blue_gain_range"] = [float(v) for v in ranges["blue_gain_range"]]
        config["raw_adapter_fixed_blue_gain"] = _midpoint(ranges["blue_gain_range"], field="blue_gain_range")
        applied.extend(["raw_adapter_blue_gain_range", "raw_adapter_fixed_blue_gain"])
    if "raw_adapter_red_gain_range" in ranges:
        config["raw_adapter_red_gain_range"] = [float(v) for v in ranges["raw_adapter_red_gain_range"]]
        config["raw_adapter_fixed_red_gain"] = _midpoint(ranges["raw_adapter_red_gain_range"], field="raw_adapter_red_gain_range")
        applied.extend(["raw_adapter_red_gain_range", "raw_adapter_fixed_red_gain"])
    if "raw_adapter_blue_gain_range" in ranges:
        config["raw_adapter_blue_gain_range"] = [float(v) for v in ranges["raw_adapter_blue_gain_range"]]
        config["raw_adapter_fixed_blue_gain"] = _midpoint(ranges["raw_adapter_blue_gain_range"], field="raw_adapter_blue_gain_range")
        applied.extend(["raw_adapter_blue_gain_range", "raw_adapter_fixed_blue_gain"])
    if "black_level_range" in ranges:
        config["raw_adapter_black_level"] = _midpoint(ranges["black_level_range"], field="black_level_range")
        applied.append("raw_adapter_black_level")
    if "read_noise_std_range" in ranges:
        config["raw_adapter_read_noise"] = _midpoint(ranges["read_noise_std_range"], field="read_noise_std_range")
        applied.append("raw_adapter_read_noise")
    if "shot_log_gain_range" in ranges:
        config["raw_adapter_shot_noise"] = math.exp(_midpoint(ranges["shot_log_gain_range"], field="shot_log_gain_range"))
        applied.append("raw_adapter_shot_noise")
    return sorted(set(applied))


def _ensure_fixed_values_are_inside_ranges(config: dict[str, Any]) -> None:
    for fixed_key, range_key in (
        ("raw_adapter_fixed_red_gain", "raw_adapter_red_gain_range"),
        ("raw_adapter_fixed_blue_gain", "raw_adapter_blue_gain_range"),
    ):
        fixed = float(config[fixed_key])
        lo, hi = [float(v) for v in config[range_key]]
        if lo <= fixed <= hi:
            continue
        config[range_key] = [min(lo, fixed), max(hi, fixed)]


def build_led_raw_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = default_led_hb_raw_adapter_config()
    source_meta: dict[str, Any] = {
        "calibration_source": None,
        "calibration_source_sha256": None,
        "calibration_group": None,
        "applied_exact_keys": [],
        "applied_range_derived_keys": [],
        "derive_fixed_from_ranges": bool(args.derive_fixed_from_ranges),
        "note": "No calibration JSON was provided; using default LED-HB raw_adapter config.",
    }
    if args.calibration_json:
        path = Path(args.calibration_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration JSON: {path}")
        payload = load_json(path)
        candidates = _find_candidate_dicts(payload, str(args.calibration_group))
        applied_group = _describe_applied_calibration_group(payload, str(args.calibration_group))
        exact = _apply_exact_raw_adapter_values(config, candidates)
        derived: list[str] = []
        if args.derive_fixed_from_ranges:
            derived = _apply_ranges_as_fixed(config, candidates)
        fixed_ready = "raw_adapter_fixed_red_gain" in config and "raw_adapter_fixed_blue_gain" in config
        if args.calibration_json and not fixed_ready:
            raise ValueError(
                "Calibration JSON did not provide fixed red/blue gains. "
                "Provide exact raw_adapter_* overrides or pass --derive-fixed-from-ranges for a visual-only midpoint preview."
            )
        source_meta.update(
            {
                "calibration_source": str(path),
                "calibration_source_sha256": sha256_file(path),
                "calibration_group": applied_group,
                "requested_calibration_group": str(args.calibration_group),
                "applied_exact_keys": exact,
                "applied_range_derived_keys": derived,
                "note": (
                    "Exact raw_adapter_* values were used where present. "
                    "Range-derived values are midpoint visual previews and must not be treated as a formal launch config."
                    if derived
                    else "Exact raw_adapter_* values were used where present."
                ),
            }
        )

    if args.raw_adapter_fixed_red_gain is not None:
        config["raw_adapter_fixed_red_gain"] = float(args.raw_adapter_fixed_red_gain)
    if args.raw_adapter_fixed_blue_gain is not None:
        config["raw_adapter_fixed_blue_gain"] = float(args.raw_adapter_fixed_blue_gain)
    if args.raw_adapter_black_level is not None:
        config["raw_adapter_black_level"] = float(args.raw_adapter_black_level)
    if args.raw_adapter_white_level is not None:
        config["raw_adapter_white_level"] = float(args.raw_adapter_white_level)
    if args.raw_adapter_shot_noise is not None:
        config["raw_adapter_shot_noise"] = float(args.raw_adapter_shot_noise)
    if args.raw_adapter_read_noise is not None:
        config["raw_adapter_read_noise"] = float(args.raw_adapter_read_noise)

    config["randomize_unprocessing"] = False
    _ensure_fixed_values_are_inside_ranges(config)
    return config, source_meta


def plot_distribution_overlay(led: dict[str, Any], stf: dict[str, Any], output_path: Path) -> None:
    all_values = np.concatenate([led["raw_all"], stf["raw_all"]])
    finite = all_values[np.isfinite(all_values)]
    axis_hi = float(np.quantile(finite, 0.999)) if finite.size else 1.0
    axis_hi = min(max(axis_hi, 0.05), 1.0)
    xlim = (0.0, axis_hi)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    _plot_hist_overlay(axes[0, 0], led["raw_all"], stf["raw_all"], title="raw4 all values", xlim=xlim)
    _plot_channel_overlay(axes[0, 1], led["channels"], stf["channels"], xlim=xlim)
    _plot_cdf(axes[0, 2], led["raw_all"], stf["raw_all"], title="raw4 all CDF", xlim=xlim)
    _plot_hist_overlay(axes[1, 0], led["base_luma"], stf["base_luma"], title="base_rgb luma", xlim=xlim)

    led_mean = np.asarray([row["raw_mean"] for row in led["per_sample"]], dtype=np.float32)
    stf_mean = np.asarray([row["raw_mean"] for row in stf["per_sample"]], dtype=np.float32)
    led_p99 = np.asarray([row["raw_p99"] for row in led["per_sample"]], dtype=np.float32)
    stf_p99 = np.asarray([row["raw_p99"] for row in stf["per_sample"]], dtype=np.float32)
    _plot_hist_overlay(axes[1, 1], led_mean, stf_mean, title="per-frame raw mean", xlim=xlim, bins=80, log_y=False)
    _plot_hist_overlay(axes[1, 2], led_p99, stf_p99, title="per-frame raw p99", xlim=xlim, bins=80, log_y=False)

    fig.suptitle(
        f"Calibrated LED synthetic raw4 vs STF decoded raw4 distribution, pixel_stride={led['pixel_stride']}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _imshow(ax: Any, image: np.ndarray, title: str, *, gray: bool = False) -> None:
    if gray:
        ax.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def plot_sample_contact_sheet(led: dict[str, Any], stf: dict[str, Any], output_path: Path) -> None:
    led_samples = list(led["preview_samples"])
    stf_samples = list(stf["preview_samples"])
    rows = min(len(led_samples), len(stf_samples))
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 12, figsize=(24, 2.7 * rows), squeeze=False)
    for row_idx in range(rows):
        led_sample = led_samples[row_idx]
        stf_sample = stf_samples[row_idx]
        led_rgb = hwc_from_chw(led_sample["rgb"])
        led_raw = np.asarray(led_sample["raw"], dtype=np.float32)
        led_base = hwc_from_chw(display_stretch(packed_to_base_rgb_chw(led_raw)))
        stf_rgb = hwc_from_chw(stf_sample["rgb"])
        stf_raw = np.asarray(stf_sample["raw"], dtype=np.float32)
        stf_base = hwc_from_chw(display_stretch(packed_to_base_rgb_chw(stf_raw)))

        images: list[tuple[str, np.ndarray, bool]] = [
            ("LED RGB", led_rgb, False),
            ("LED base_rgb", led_base, False),
            ("LED R", display_stretch(led_raw[0]), True),
            ("LED Gr", display_stretch(led_raw[1]), True),
            ("LED Gb", display_stretch(led_raw[2]), True),
            ("LED B", display_stretch(led_raw[3]), True),
            ("STF lut", stf_rgb, False),
            ("STF base_rgb", stf_base, False),
            ("STF R", display_stretch(stf_raw[0]), True),
            ("STF Gr", display_stretch(stf_raw[1]), True),
            ("STF Gb", display_stretch(stf_raw[2]), True),
            ("STF B", display_stretch(stf_raw[3]), True),
        ]
        for col_idx, (title, image, gray) in enumerate(images):
            ax = axes[row_idx, col_idx]
            _imshow(ax, image, title if row_idx == 0 else "", gray=gray)
        axes[row_idx, 0].set_ylabel(
            f"{led_sample['sample_name']}\nvs\n{stf_sample['sample_name']}",
            fontsize=7,
            rotation=0,
            labelpad=58,
            va="center",
        )
    fig.suptitle("Visual-only sample comparison: LED calibrated synthetic raw vs STF decoded raw", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _single_hist(ax: Any, values: np.ndarray, *, title: str, color: str, xlim: tuple[float, float]) -> None:
    edges = np.linspace(float(xlim[0]), float(xlim[1]), 96)
    x, y = _hist_density(values, edges)
    ax.plot(x, y, color=color, linewidth=1.2)
    ax.set_xlim(*xlim)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=6)


def _raw_axis_xlim(*stats: dict[str, Any]) -> tuple[float, float]:
    primary = np.asarray(stats[0]["raw_all"], dtype=np.float32) if stats else np.asarray([1.0], dtype=np.float32)
    primary = primary[np.isfinite(primary)]
    if primary.size == 0:
        primary = np.asarray([1.0], dtype=np.float32)
    hi = float(np.quantile(primary, 0.995))
    hi = min(max(hi, 0.05), 1.0)
    return 0.0, hi


def _raw_preview_hwc(raw_chw: np.ndarray) -> np.ndarray:
    return hwc_from_chw(display_stretch(packed_to_base_rgb_chw(raw_chw)))


def plot_requested_comparison_board(
    *,
    stf: dict[str, Any],
    led_calibrated: dict[str, Any],
    led_invisp: dict[str, Any],
    output_path: Path,
) -> None:
    rows = min(len(stf["preview_samples"]), len(led_calibrated["preview_samples"]), len(led_invisp["preview_samples"]))
    if rows <= 0:
        return
    raw_xlim = _raw_axis_xlim(stf, led_calibrated, led_invisp)
    fig = plt.figure(figsize=(30, 3.0 + 2.55 * rows))
    grid = fig.add_gridspec(
        nrows=rows + 1,
        ncols=10,
        height_ratios=[1.15] + [1.0] * rows,
        hspace=0.42,
        wspace=0.18,
    )

    top_axes = [fig.add_subplot(grid[0, i * 2 : (i + 1) * 2]) for i in range(5)]
    _plot_hist_overlay(
        top_axes[0],
        led_calibrated["rgb_all"],
        stf["rgb_all"],
        title="RGB all values: LED vs STF",
        xlim=(0.0, 1.0),
        log_y=True,
        led_label="LED RGB",
        stf_label="STF RGB",
    )
    _plot_hist_overlay(
        top_axes[1],
        led_invisp["raw_all"],
        stf["raw_all"],
        title="RAW all: Invisp default vs STF",
        xlim=raw_xlim,
        log_y=True,
        led_label="Invisp default RAW",
        stf_label="STF RAW",
    )
    _plot_hist_overlay(
        top_axes[2],
        led_calibrated["raw_all"],
        stf["raw_all"],
        title="RAW all: calibrated vs STF",
        xlim=raw_xlim,
        log_y=True,
        led_label="Calibrated RAW",
        stf_label="STF RAW",
    )
    _plot_cdf(
        top_axes[3],
        led_invisp["raw_all"],
        stf["raw_all"],
        title="RAW CDF: Invisp vs STF",
        xlim=raw_xlim,
        led_label="Invisp default RAW",
        stf_label="STF RAW",
    )
    _plot_cdf(
        top_axes[4],
        led_calibrated["raw_all"],
        stf["raw_all"],
        title="RAW CDF: calibrated vs STF",
        xlim=raw_xlim,
        led_label="Calibrated RAW",
        stf_label="STF RAW",
    )

    headers = (
        "STF RGB",
        "STF RGB dist",
        "STF RAW",
        "STF RAW dist",
        "LED RGB",
        "LED RGB dist",
        "Invisp RAW",
        "Invisp RAW dist",
        "Calibrated RAW",
        "Calibrated RAW dist",
    )
    for row_idx in range(rows):
        stf_sample = stf["preview_samples"][row_idx]
        cal_sample = led_calibrated["preview_samples"][row_idx]
        invisp_sample = led_invisp["preview_samples"][row_idx]
        cells: list[tuple[str, str, np.ndarray | None, str, tuple[float, float] | None]] = [
            ("image", headers[0], hwc_from_chw(stf_sample["rgb"]), "", None),
            ("hist", headers[1], stf_sample["rgb"].reshape(-1), "#5aa7ff", (0.0, 1.0)),
            ("image", headers[2], _raw_preview_hwc(stf_sample["raw"]), "", None),
            ("hist", headers[3], stf_sample["raw"].reshape(-1), "#5aa7ff", raw_xlim),
            ("image", headers[4], hwc_from_chw(cal_sample["rgb"]), "", None),
            ("hist", headers[5], cal_sample["rgb"].reshape(-1), "#f0a13a", (0.0, 1.0)),
            ("image", headers[6], _raw_preview_hwc(invisp_sample["raw"]), "", None),
            ("hist", headers[7], invisp_sample["raw"].reshape(-1), "#8a8a8a", raw_xlim),
            ("image", headers[8], _raw_preview_hwc(cal_sample["raw"]), "", None),
            ("hist", headers[9], cal_sample["raw"].reshape(-1), "#f0a13a", raw_xlim),
        ]
        for col_idx, (kind, title, payload, color, xlim) in enumerate(cells):
            ax = fig.add_subplot(grid[row_idx + 1, col_idx])
            if kind == "image":
                assert payload is not None
                ax.imshow(np.clip(payload, 0.0, 1.0))
                ax.set_axis_off()
                if row_idx == 0:
                    ax.set_title(title, fontsize=8)
            else:
                assert payload is not None and xlim is not None
                _single_hist(ax, payload, title=title if row_idx == 0 else "", color=color, xlim=xlim)
        fig.axes[-10].set_ylabel(
            f"STF {stf_sample['sample_name']}\nLED {cal_sample['sample_name']}",
            fontsize=7,
            rotation=0,
            labelpad=58,
            va="center",
        )

    fig.suptitle(
        f"STF RGB/RAW vs LED RGB, Invisp default RAW, and calibrated RAW; RAW x-axis={raw_xlim[0]:.3g}..{raw_xlim[1]:.3g}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stats_summary(stats: dict[str, Any]) -> dict[str, Any]:
    per_sample = stats["per_sample"]
    return {
        "count": int(stats["count"]),
        "pixel_stride": int(stats["pixel_stride"]),
        "raw_all_quantiles": quantiles(stats["raw_all"]),
        "base_luma_quantiles": quantiles(stats["base_luma"]),
        "rgb_all_quantiles": quantiles(stats["rgb_all"]),
        "rgb_luma_quantiles": quantiles(stats["rgb_luma"]),
        "channel_quantiles": {name: quantiles(values) for name, values in stats["channels"].items()},
        "per_sample_summary": {
            key: quantiles(np.asarray([row[key] for row in per_sample], dtype=np.float32))
            for key in (
                "raw_mean",
                "raw_std",
                "raw_p50",
                "raw_p99",
                "raw_zero_frac_lt_1e_4",
                "raw_sat_frac_gt_0p999",
                "base_luma_mean",
            )
        },
        "preview_samples": [
            {k: v for k, v in row.items() if k not in {"rgb", "raw"}}
            for row in stats["preview_samples"]
        ],
    }


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    calib = payload["calibration"]
    led = payload["led"]
    stf = payload["stf"]
    lines = [
        "# Section 11 Calibrated LED vs STF RAW Distribution Review",
        "",
        "This package is visual-only. It does not include depth quantitative results.",
        "",
        "## Inputs",
        "",
        f"- Calibration source: `{calib.get('calibration_source')}`",
        f"- Calibration group: `{calib.get('calibration_group')}`",
        f"- LED samples: {led['count']} from `{payload['paths']['led_list']}`",
        f"- STF samples: {stf['count']} from `{payload['paths']['stf_manifest']}`",
        f"- STF raw format: `{payload['stf_raw']['raw_storage_format']}`",
        f"- Pixel stride for distribution collection: {led['pixel_stride']}",
        "",
        "## Generated Files",
        "",
        "- `distribution_overlay.png`",
        "- `sample_contact_sheet.png`",
        "- `summary.json`",
        "",
        "## Notes",
        "",
        f"- {calib.get('note')}",
        "- LED raw is generated through the same `LEDHBRaw` raw_adapter path used by N2/N7.",
        "- Current LED-HB validation keeps `randomize_unprocessing=false`; noise parameters are recorded, but noise realization is not applied in this visual package.",
        "- STF raw is decoded as `legacy_bggR_decomp16`: storage `[B,Gr,Gb,R]` -> model `[R,Gr,Gb,B]` with STF LUT decompand.",
        "- Use this as the section-11 gate before lineage-B training or quantitative STF eval.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visual-only comparison of calibrated LED synthetic RAW and STF decoded RAW distributions."
    )
    parser.add_argument("--led-list", type=Path, default=ROOT / "finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt")
    parser.add_argument("--stf-root", type=Path, default=Path(DEFAULT_STF_ROOT))
    parser.add_argument("--stf-manifest", type=Path, default=None)
    parser.add_argument("--stf-daytime-filter", default="all", choices=["all", "day", "night", "twilight"])
    parser.add_argument("--raw-npz-root", type=Path, default=Path(DEFAULT_RAW_NPZ_ROOT))
    parser.add_argument("--raw-storage-format", default="legacy_bggR_decomp16", choices=["legacy_bggR_decomp16"])
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--calibration-group", default="overall")
    parser.add_argument(
        "--derive-fixed-from-ranges",
        action="store_true",
        help="Use range midpoints from calibration JSON for a visual-only preview when exact raw_adapter_* values are absent.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--led-max-samples", type=int, default=200)
    parser.add_argument("--stf-max-samples", type=int, default=200)
    parser.add_argument("--pixel-stride", type=int, default=16)
    parser.add_argument("--panel-samples", type=int, default=4)
    parser.add_argument("--comparison-board", action="store_true", help="Also write requested STF/LED/Invisp/calibrated comparison board.")
    parser.add_argument("--raw-adapter-fixed-red-gain", type=float, default=None)
    parser.add_argument("--raw-adapter-fixed-blue-gain", type=float, default=None)
    parser.add_argument("--raw-adapter-black-level", type=float, default=None)
    parser.add_argument("--raw-adapter-white-level", type=float, default=None)
    parser.add_argument("--raw-adapter-shot-noise", type=float, default=None)
    parser.add_argument("--raw-adapter-read-noise", type=float, default=None)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=200.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.pixel_stride <= 0:
        raise ValueError("--pixel-stride must be positive")
    if args.led_max_samples == 0 or args.stf_max_samples == 0:
        raise ValueError("--led-max-samples and --stf-max-samples must be non-zero")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stf_root = Path(args.stf_root).expanduser().resolve()
    stf_manifest = Path(args.stf_manifest).expanduser().resolve() if args.stf_manifest else stf_root / "manifests/stf_raw_depth_v1_val.csv"
    raw_root = Path(args.raw_npz_root).expanduser().resolve()
    led_list = Path(args.led_list).expanduser().resolve()

    raw_config, calibration_meta = build_led_raw_config(args)
    dataset = LEDHBRaw(
        led_list,
        mode="val",
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        hflip_prob=0.0,
        include_rgb_preview=True,
        unprocessing_config=raw_config,
    )
    rows_all = read_stf_manifest(stf_manifest)
    rows = filter_stf_rows(rows_all, str(args.stf_daytime_filter))
    led_stats = collect_led_stats(
        dataset,
        max_samples=int(args.led_max_samples),
        pixel_stride=int(args.pixel_stride),
        panel_samples=int(args.panel_samples),
    )
    stf_stats = collect_stf_stats(
        rows,
        stf_root=stf_root,
        raw_npz_root=raw_root,
        raw_storage_format=str(args.raw_storage_format),
        max_samples=int(args.stf_max_samples),
        pixel_stride=int(args.pixel_stride),
        panel_samples=int(args.panel_samples),
        target_hw=(512, 960),
    )

    plot_distribution_overlay(led_stats, stf_stats, out_dir / "distribution_overlay.png")
    plot_sample_contact_sheet(led_stats, stf_stats, out_dir / "sample_contact_sheet.png")
    led_invisp_stats = None
    if args.comparison_board:
        invisp_dataset = LEDHBRaw(
            led_list,
            mode="val",
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            hflip_prob=0.0,
            include_rgb_preview=True,
            unprocessing_config=default_led_hb_raw_adapter_config(),
        )
        led_invisp_stats = collect_led_stats(
            invisp_dataset,
            max_samples=int(args.led_max_samples),
            pixel_stride=int(args.pixel_stride),
            panel_samples=int(args.panel_samples),
        )
        plot_requested_comparison_board(
            stf=stf_stats,
            led_calibrated=led_stats,
            led_invisp=led_invisp_stats,
            output_path=out_dir / "requested_comparison_board.png",
        )

    raw_spec = get_raw_storage_spec(str(args.raw_storage_format))
    payload = {
        "stage": "section11_calibrated_led_vs_stf_distribution_review",
        "paths": {
            "led_list": str(led_list),
            "stf_root": str(stf_root),
            "stf_manifest": str(stf_manifest),
            "stf_daytime_filter": str(args.stf_daytime_filter),
            "stf_manifest_rows_total": len(rows_all),
            "stf_manifest_rows_after_filter": len(rows),
            "raw_npz_root": str(raw_root),
            "output_dir": str(out_dir),
        },
        "calibration": calibration_meta,
        "led_raw_adapter_config": raw_config,
        "led_raw_adapter_summary": dataset.describe_unprocessing(),
        "led_geometry": dataset.describe_geometry(),
        "stf_raw": {
            "raw_storage_format": str(args.raw_storage_format),
            "raw_key": RECTIFIED_BAYER_KEY,
            "raw_storage_channel_order": list(raw_spec.storage_channel_order),
            "raw_model_channel_order": list(raw_spec.model_channel_order),
            "channel_reorder": list(raw_spec.channel_reorder),
            "raw_decompand": raw_spec.decompand,
            "raw_post_decode_norm": raw_spec.post_decode_norm,
        },
        "led": stats_summary(led_stats),
        "stf": stats_summary(stf_stats),
        "led_invisp_default": stats_summary(led_invisp_stats) if led_invisp_stats is not None else None,
        "notes": [
            "Visual distribution review only; do not treat this as a depth quantitative result.",
            "LED synthetic raw and STF real raw are not sample-paired; compare distribution and gross appearance.",
            "Current LED-HB raw_adapter validation fixes randomize_unprocessing=false, so noise parameters are recorded but not realized.",
        ],
    }
    write_json(out_dir / "summary.json", payload)
    write_markdown_summary(out_dir / "summary.md", payload)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "distribution_overlay": str(out_dir / "distribution_overlay.png"),
                "sample_contact_sheet": str(out_dir / "sample_contact_sheet.png"),
                "requested_comparison_board": str(out_dir / "requested_comparison_board.png") if args.comparison_board else None,
                "summary": str(out_dir / "summary.json"),
                "led_samples": int(led_stats["count"]),
                "stf_samples": int(stf_stats["count"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
