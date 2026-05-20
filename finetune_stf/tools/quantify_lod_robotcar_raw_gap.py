from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_raw import DEFAULT_LOD_DAY_MANIFEST, DEFAULT_LOD_NIGHT_MANIFEST, DEFAULT_LOD_ROOT
from finetune_stf.dataset.raw_utils import RECTIFIED_BAYER_KEY, load_rectified_bayer_npz, normalize_raw_4ch


DEFAULT_ROBOTCAR_DAY_MANIFEST = (
    "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100/manifests/"
    "robotcar_raw_depth_v1_val.csv"
)
DEFAULT_ROBOTCAR_NIGHT_MANIFEST = (
    "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo/manifests/"
    "robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv"
)
CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
PERCENTILES = (0.1, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 99.0, 99.9)
ROBOTCAR_PACK_ORDER = "[R,Gr,Gb,B]"
LOD_RAW24_HEIGHT = 1856
LOD_RAW24_WIDTH = 2880
LOD_RAW24_INPUT_BYTES = LOD_RAW24_HEIGHT * LOD_RAW24_WIDTH * 3
LOD_RAW24_MAX_VALUE = float(2**24 - 1)
STAGE_DESCRIPTIONS = {
    "lod_day_S0_source_raw24": "LoD day source .raw, unpacked uint24 / (2^24 - 1), packed RGGB",
    "lod_day_S1_processed_npy": "LoD day model-facing rggb_path .npy",
    "lod_night_S0_source_raw24": "LoD night source .raw, unpacked uint24 / (2^24 - 1), packed RGGB",
    "lod_night_S1_processed_npy": "LoD night model-facing rggb_path .npy",
    "robotcar_day_S0_source_png": "RobotCar day raw_src_path 8-bit Bayer PNG, packed GBRG->[R,Gr,Gb,B], /255",
    "robotcar_day_S1_native_rectified_npz": "RobotCar day raw_native_path .npz after per-plane rectification",
    "robotcar_day_S2_eval_npz_model_facing": "RobotCar day raw_eval_path .npz used by RobotCarValRaw",
    "robotcar_night_S0_source_png": "RobotCar night raw_src_path 8-bit Bayer PNG, packed GBRG->[R,Gr,Gb,B], /255",
    "robotcar_night_S1_native_rectified_npz": "RobotCar night raw_native_path .npz after per-plane rectification",
    "robotcar_night_S2_eval_npz_model_facing": "RobotCar night raw_eval_path .npz used by RobotCarValRaw",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quantify LoD RAW vs RobotCar RAW distribution gap.")
    parser.add_argument("--lod-root", default=DEFAULT_LOD_ROOT)
    parser.add_argument("--lod-day-manifest", default=DEFAULT_LOD_DAY_MANIFEST)
    parser.add_argument("--lod-night-manifest", default=DEFAULT_LOD_NIGHT_MANIFEST)
    parser.add_argument("--robotcar-day-manifest", default=DEFAULT_ROBOTCAR_DAY_MANIFEST)
    parser.add_argument("--robotcar-night-manifest", default=DEFAULT_ROBOTCAR_NIGHT_MANIFEST)
    parser.add_argument("--max-lod-day-samples", type=int, default=None)
    parser.add_argument("--max-lod-night-samples", type=int, default=None)
    parser.add_argument("--max-robotcar-day-samples", type=int, default=None)
    parser.add_argument("--max-robotcar-night-samples", type=int, default=None)
    parser.add_argument(
        "--pixels-per-image",
        type=int,
        default=0,
        help="0 means all pixels; positive values sample this many Bayer pixels per image for hist/CDF.",
    )
    parser.add_argument("--hist-bins", type=int, default=65536)
    parser.add_argument("--unique-max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--sample-mode", default="first", choices=("first", "random"))
    parser.add_argument(
        "--mask-remap-zero-border",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, pixels whose four channels are all zero are excluded from hist/CDF.",
    )
    parser.add_argument("--fit-lut-source", default=None, choices=("lod_day", "lod_night", "robotcar_day", "robotcar_night"))
    parser.add_argument("--fit-lut-target", default=None, choices=("lod_day", "lod_night", "robotcar_day", "robotcar_night"))
    parser.add_argument("--lut-size", type=int, default=4096)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.pixels_per_image < 0:
        parser.error("--pixels-per-image must be >= 0")
    if args.hist_bins < 16:
        parser.error("--hist-bins must be >= 16")
    if args.unique_max_samples < 0:
        parser.error("--unique-max-samples must be >= 0")
    if bool(args.fit_lut_source) != bool(args.fit_lut_target):
        parser.error("--fit-lut-source and --fit-lut-target must be provided together")
    return args


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_lod_path(lod_root: str | Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(lod_root).expanduser().resolve() / path).resolve()


def select_rows(rows: list[dict[str, str]], max_samples: int | None, *, rng: np.random.Generator, mode: str):
    if max_samples is None or max_samples >= len(rows):
        return rows
    if mode == "first":
        return rows[:max_samples]
    indices = np.sort(rng.choice(len(rows), size=max_samples, replace=False))
    return [rows[int(idx)] for idx in indices]


def canonicalize_robotcar(raw: np.ndarray, pack_order: str) -> np.ndarray:
    if pack_order == ROBOTCAR_PACK_ORDER:
        return raw
    if pack_order == "[R,Gb,Gr,B]":
        return raw[..., [0, 2, 1, 3]]
    raise ValueError(f"Unsupported RobotCar pack_order: {pack_order}")


def pack_robotcar_gbrg_uint8(raw_full: np.ndarray) -> np.ndarray:
    if raw_full.ndim != 2:
        raise ValueError(f"Expected RobotCar raw PNG as 2D Bayer image, got {raw_full.shape}")
    return np.stack(
        [
            raw_full[1::2, 0::2],
            raw_full[1::2, 1::2],
            raw_full[0::2, 0::2],
            raw_full[0::2, 1::2],
        ],
        axis=-1,
    )


def unpack_lod_raw24(path: Path) -> np.ndarray:
    raw_bytes = np.fromfile(path, dtype=np.uint8)
    if raw_bytes.size != LOD_RAW24_INPUT_BYTES:
        raise ValueError(
            f"{path} has {raw_bytes.size} bytes, expected {LOD_RAW24_INPUT_BYTES} "
            f"for {LOD_RAW24_HEIGHT}x{LOD_RAW24_WIDTH}x3 raw24"
        )
    raw24 = (
        raw_bytes[0::3].astype(np.uint32)
        + raw_bytes[1::3].astype(np.uint32) * 256
        + raw_bytes[2::3].astype(np.uint32) * 65536
    )
    return raw24.reshape((LOD_RAW24_HEIGHT, LOD_RAW24_WIDTH)).astype(np.float32) / LOD_RAW24_MAX_VALUE


def pack_lod_rggb(raw_full: np.ndarray) -> np.ndarray:
    if raw_full.shape != (LOD_RAW24_HEIGHT, LOD_RAW24_WIDTH):
        raise ValueError(f"Expected LoD full raw shape {(LOD_RAW24_HEIGHT, LOD_RAW24_WIDTH)}, got {raw_full.shape}")
    return np.stack(
        [
            raw_full[0::2, 0::2],
            raw_full[0::2, 1::2],
            raw_full[1::2, 0::2],
            raw_full[1::2, 1::2],
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def derive_lod_raw24_path(rggb_path: Path) -> Path:
    text = str(rggb_path)
    if "-rggb/" not in text:
        raise ValueError(f"Cannot derive LoD raw24 path from rggb path: {rggb_path}")
    return Path(text.replace("-rggb/", "-raws/")).with_suffix(".raw")


class HistogramAccumulator:
    def __init__(self, *, bins: int, rng: np.random.Generator, pixels_per_image: int, mask_zero_border: bool):
        self.edges = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
        self.hist = np.zeros((4, int(bins)), dtype=np.int64)
        self.count = np.zeros(4, dtype=np.int64)
        self.sum = np.zeros(4, dtype=np.float64)
        self.sumsq = np.zeros(4, dtype=np.float64)
        self.min = np.full(4, np.inf, dtype=np.float64)
        self.max = np.full(4, -np.inf, dtype=np.float64)
        self.zero_count = np.zeros(4, dtype=np.int64)
        self.sat_count = np.zeros(4, dtype=np.int64)
        self.image_count = 0
        self.rng = rng
        self.pixels_per_image = int(pixels_per_image)
        self.mask_zero_border = bool(mask_zero_border)

    def add(self, raw_4ch: np.ndarray) -> None:
        raw = np.asarray(raw_4ch, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[-1] != 4:
            raise ValueError(f"Expected raw shape (H,W,4), got {raw.shape}")
        flat = raw.reshape(-1, 4)
        finite = np.isfinite(flat).all(axis=1)
        if self.mask_zero_border:
            finite &= np.any(flat != 0.0, axis=1)
        flat = flat[finite]
        if flat.size == 0:
            return
        if self.pixels_per_image > 0 and flat.shape[0] > self.pixels_per_image:
            indices = self.rng.choice(flat.shape[0], size=self.pixels_per_image, replace=False)
            flat = flat[indices]
        for channel in range(4):
            vals = flat[:, channel].astype(np.float64, copy=False)
            self.hist[channel] += np.histogram(vals, bins=self.edges)[0]
            self.count[channel] += vals.size
            self.sum[channel] += vals.sum()
            self.sumsq[channel] += np.square(vals).sum()
            self.min[channel] = min(self.min[channel], float(vals.min()))
            self.max[channel] = max(self.max[channel], float(vals.max()))
            self.zero_count[channel] += int(np.count_nonzero(vals <= 1e-8))
            self.sat_count[channel] += int(np.count_nonzero(vals >= 1.0 - 1e-8))
        self.image_count += 1

    def percentile(self, channel: int, q: float) -> float:
        total = int(self.count[channel])
        if total <= 0:
            return float("nan")
        rank = max(1, int(math.ceil((float(q) / 100.0) * total)))
        idx = int(np.searchsorted(np.cumsum(self.hist[channel]), rank, side="left"))
        idx = min(max(idx, 0), len(self.edges) - 2)
        return float((self.edges[idx] + self.edges[idx + 1]) * 0.5)

    def cdf(self, channel: int) -> tuple[np.ndarray, np.ndarray]:
        total = int(self.count[channel])
        x = (self.edges[:-1] + self.edges[1:]) * 0.5
        if total <= 0:
            return x, np.zeros_like(x)
        return x, np.cumsum(self.hist[channel]) / float(total)

    def summary(self) -> dict[str, object]:
        channels = {}
        for idx, name in enumerate(CHANNEL_NAMES):
            count = int(self.count[idx])
            if count <= 0:
                channels[name] = {"count": 0}
                continue
            mean = float(self.sum[idx] / count)
            var = max(float(self.sumsq[idx] / count - mean * mean), 0.0)
            channels[name] = {
                "count": count,
                "min": float(self.min[idx]),
                "max": float(self.max[idx]),
                "mean": mean,
                "std": float(math.sqrt(var)),
                "zero_ratio": float(self.zero_count[idx] / count),
                "saturation_ratio": float(self.sat_count[idx] / count),
                "percentiles": {f"p{q:g}": self.percentile(idx, q) for q in PERCENTILES},
            }
        return {
            "images_accumulated": int(self.image_count),
            "hist_bins": int(len(self.edges) - 1),
            "pixels_per_image": int(self.pixels_per_image),
            "mask_zero_border": bool(self.mask_zero_border),
            "channels": channels,
        }


def compute_unique_counts(raw_4ch: np.ndarray) -> dict[str, int]:
    return {name: int(np.unique(raw_4ch[..., idx]).size) for idx, name in enumerate(CHANNEL_NAMES)}


def summarize_unique(unique_rows: list[dict[str, int]]) -> dict[str, object]:
    if not unique_rows:
        return {"num_images": 0}
    result: dict[str, object] = {"num_images": len(unique_rows)}
    for name in CHANNEL_NAMES:
        vals = np.asarray([row[name] for row in unique_rows], dtype=np.float64)
        result[name] = {
            "min": int(vals.min()),
            "p50": float(np.percentile(vals, 50)),
            "mean": float(vals.mean()),
            "max": int(vals.max()),
        }
    return result


def make_accumulator(args, *, seed_offset: int) -> HistogramAccumulator:
    return HistogramAccumulator(
        bins=args.hist_bins,
        rng=np.random.default_rng(args.seed + seed_offset),
        pixels_per_image=args.pixels_per_image,
        mask_zero_border=args.mask_remap_zero_border,
    )


def run_lod_group(
    *,
    name: str,
    rows: list[dict[str, str]],
    lod_root: str | Path,
    accumulator: HistogramAccumulator,
    stage_accumulators: dict[str, HistogramAccumulator],
    unique_max_samples: int,
) -> dict[str, object]:
    unique_rows = []
    missing = 0
    source_missing = 0
    for idx, row in enumerate(rows):
        raw_path = resolve_lod_path(lod_root, row["rggb_path"])
        if not raw_path.is_file():
            missing += 1
            continue
        source_path = derive_lod_raw24_path(raw_path)
        if source_path.is_file():
            stage_accumulators[f"{name}_S0_source_raw24"].add(pack_lod_rggb(unpack_lod_raw24(source_path)))
        else:
            source_missing += 1
        raw = normalize_raw_4ch(np.load(raw_path).astype(np.float32, copy=False), norm_mode="sensor_linear")
        accumulator.add(raw)
        stage_accumulators[f"{name}_S1_processed_npy"].add(raw)
        if idx < unique_max_samples:
            unique_rows.append(compute_unique_counts(raw))
        if (idx + 1) == 1 or (idx + 1) % 100 == 0 or (idx + 1) == len(rows):
            print(f"[stage][{name}] processed {idx + 1}/{len(rows)}", flush=True)
    return {
        "group": name,
        "distribution_stage": "model_facing_processed_asset",
        "path_field": "rggb_path",
        "source_unit": "LoD packed RGGB npy produced by raw24_to_rggb_npy.py, already scaled to unit interval",
        "source_raw24_path_rule": "replace '-rggb/' with '-raws/' and '.npy' with '.raw'",
        "rows_selected": len(rows),
        "missing_files": missing,
        "source_raw24_missing_files": source_missing,
        "unique_counts": summarize_unique(unique_rows),
    }


def run_robotcar_group(
    *,
    name: str,
    rows: list[dict[str, str]],
    accumulator: HistogramAccumulator,
    stage_accumulators: dict[str, HistogramAccumulator],
    unique_max_samples: int,
) -> dict[str, object]:
    pre_unique_rows = []
    post_unique_rows = []
    missing = 0
    source_missing = 0
    native_missing = 0
    for idx, row in enumerate(rows):
        raw_eval_path = Path(row["raw_eval_path"]).expanduser().resolve()
        raw_src_path = Path(row["raw_src_path"]).expanduser().resolve()
        raw_native_path = Path(row["raw_native_path"]).expanduser().resolve()
        if not raw_eval_path.is_file():
            missing += 1
            continue
        if raw_src_path.is_file():
            src = cv2.imread(str(raw_src_path), cv2.IMREAD_UNCHANGED)
            if src is not None:
                src_packed = pack_robotcar_gbrg_uint8(src).astype(np.float32) / 255.0
                stage_accumulators[f"{name}_S0_source_png"].add(src_packed)
            else:
                source_missing += 1
        else:
            source_missing += 1
        if raw_native_path.is_file():
            native = load_rectified_bayer_npz(raw_native_path, key=RECTIFIED_BAYER_KEY)
            native = canonicalize_robotcar(native, row.get("pack_order", ROBOTCAR_PACK_ORDER))
            native = normalize_raw_4ch(native, norm_mode="sensor_linear")
            stage_accumulators[f"{name}_S1_native_rectified_npz"].add(native)
        else:
            native_missing += 1
        raw = load_rectified_bayer_npz(raw_eval_path, key=RECTIFIED_BAYER_KEY)
        raw = canonicalize_robotcar(raw, row.get("pack_order", ROBOTCAR_PACK_ORDER))
        raw = normalize_raw_4ch(raw, norm_mode="sensor_linear")
        accumulator.add(raw)
        stage_accumulators[f"{name}_S2_eval_npz_model_facing"].add(raw)
        if idx < unique_max_samples:
            post_unique_rows.append(compute_unique_counts(raw))
            if raw_src_path.is_file():
                src = cv2.imread(str(raw_src_path), cv2.IMREAD_UNCHANGED)
                if src is not None:
                    pre_unique_rows.append(compute_unique_counts(pack_robotcar_gbrg_uint8(src)))
        if (idx + 1) == 1 or (idx + 1) % 100 == 0 or (idx + 1) == len(rows):
            print(f"[stage][{name}] processed {idx + 1}/{len(rows)}", flush=True)
    return {
        "group": name,
        "distribution_stage": "model_facing_processed_asset",
        "path_field": "raw_eval_path",
        "pre_rectification_path_field": "raw_src_path",
        "source_unit": "RobotCar raw_eval npz produced from public 8-bit PNG by /255 and per-plane rectification",
        "pre_rectification_source_unit": "RobotCar public raw_src_path 8-bit Bayer PNG packed to [R,Gr,Gb,B] only for source quantization diagnostics",
        "rows_selected": len(rows),
        "missing_files": missing,
        "pre_rectification_missing_files": source_missing,
        "native_rectified_missing_files": native_missing,
        "pre_rectification_uint8_unique_counts": summarize_unique(pre_unique_rows),
        "post_rectification_float_unique_counts": summarize_unique(post_unique_rows),
    }


def write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    rows = []
    for group_name, group_summary in summary["groups"].items():
        channels = group_summary["distribution"]["channels"]
        for channel_name in CHANNEL_NAMES:
            stats = channels[channel_name]
            row = {
                "group": group_name,
                "channel": channel_name,
                "count": stats["count"],
                "min": stats["min"],
                "mean": stats["mean"],
                "std": stats["std"],
                "max": stats["max"],
                "zero_ratio": stats["zero_ratio"],
                "saturation_ratio": stats["saturation_ratio"],
            }
            row.update(stats["percentiles"])
            rows.append(row)
    if not rows:
        raise ValueError("No distribution rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_stage_summary_csv(path: Path, summary: dict[str, object]) -> None:
    rows = []
    for stage_name, stage_summary in summary["stage_groups"].items():
        channels = stage_summary["distribution"]["channels"]
        for channel_name in CHANNEL_NAMES:
            stats = channels[channel_name]
            if stats["count"] <= 0:
                continue
            row = {
                "stage": stage_name,
                "description": stage_summary["description"],
                "channel": channel_name,
                "images": stage_summary["distribution"]["images_accumulated"],
                "count": stats["count"],
                "min": stats["min"],
                "mean": stats["mean"],
                "std": stats["std"],
                "max": stats["max"],
                "zero_ratio": stats["zero_ratio"],
                "saturation_ratio": stats["saturation_ratio"],
            }
            row.update(stats["percentiles"])
            rows.append(row)
    if not rows:
        raise ValueError("No stage distribution rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_cdf(path: Path, accumulators: dict[str, HistogramAccumulator], *, channel: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for name, acc in accumulators.items():
        x, y = acc.cdf(channel)
        ax.plot(x, y, label=name)
    ax.set_xlabel(f"{CHANNEL_NAMES[channel]} value")
    ax.set_ylabel("CDF")
    ax.set_title(f"LoD vs RobotCar RAW CDF ({CHANNEL_NAMES[channel]})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hist(path: Path, accumulators: dict[str, HistogramAccumulator], *, channel: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for name, acc in accumulators.items():
        total = max(int(acc.count[channel]), 1)
        density = acc.hist[channel].astype(np.float64) / total
        x = (acc.edges[:-1] + acc.edges[1:]) * 0.5
        ax.plot(x, density, label=name)
    ax.set_yscale("log")
    ax.set_xlabel(f"{CHANNEL_NAMES[channel]} value")
    ax.set_ylabel("Probability per bin (log)")
    ax.set_title(f"LoD vs RobotCar RAW Histogram ({CHANNEL_NAMES[channel]})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fit_lut_from_hist(source: HistogramAccumulator, target: HistogramAccumulator, *, channel: int, lut_size: int) -> np.ndarray:
    sx, scdf = source.cdf(channel)
    tx, tcdf = target.cdf(channel)
    quantiles = np.linspace(0.0, 1.0, int(lut_size), dtype=np.float64)
    source_values = np.interp(quantiles, scdf, sx)
    target_values = np.interp(quantiles, tcdf, tx)
    x_grid = np.linspace(0.0, 1.0, int(lut_size), dtype=np.float64)
    lut = np.interp(x_grid, source_values, target_values)
    lut = np.maximum.accumulate(lut)
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def build_group_rows(args, rng: np.random.Generator) -> dict[str, dict[str, object]]:
    lod_day_rows = select_rows(read_csv_rows(args.lod_day_manifest), args.max_lod_day_samples, rng=rng, mode=args.sample_mode)
    lod_night_rows = select_rows(read_csv_rows(args.lod_night_manifest), args.max_lod_night_samples, rng=rng, mode=args.sample_mode)
    robotcar_day_rows = select_rows(
        read_csv_rows(args.robotcar_day_manifest),
        args.max_robotcar_day_samples,
        rng=rng,
        mode=args.sample_mode,
    )
    robotcar_night_rows = select_rows(
        read_csv_rows(args.robotcar_night_manifest),
        args.max_robotcar_night_samples,
        rng=rng,
        mode=args.sample_mode,
    )
    return {
        "lod_day": {"rows": lod_day_rows, "manifest": args.lod_day_manifest},
        "lod_night": {"rows": lod_night_rows, "manifest": args.lod_night_manifest},
        "robotcar_day": {"rows": robotcar_day_rows, "manifest": args.robotcar_day_manifest},
        "robotcar_night": {"rows": robotcar_night_rows, "manifest": args.robotcar_night_manifest},
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    groups = build_group_rows(args, rng)
    accumulators = {
        name: make_accumulator(args, seed_offset=idx + 1)
        for idx, name in enumerate(groups)
    }
    stage_accumulators = {
        name: make_accumulator(args, seed_offset=1000 + idx)
        for idx, name in enumerate(STAGE_DESCRIPTIONS)
    }

    group_meta: dict[str, object] = {}
    for name in ("lod_day", "lod_night"):
        group_meta[name] = run_lod_group(
            name=name,
            rows=groups[name]["rows"],
            lod_root=args.lod_root,
            accumulator=accumulators[name],
            stage_accumulators=stage_accumulators,
            unique_max_samples=args.unique_max_samples,
        )
    for name in ("robotcar_day", "robotcar_night"):
        group_meta[name] = run_robotcar_group(
            name=name,
            rows=groups[name]["rows"],
            accumulator=accumulators[name],
            stage_accumulators=stage_accumulators,
            unique_max_samples=args.unique_max_samples,
        )

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "host": socket.gethostname(),
        "script": str(Path(__file__).resolve()),
        "config": {
            "lod_root": str(Path(args.lod_root).expanduser()),
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "pixels_per_image": args.pixels_per_image,
            "hist_bins": args.hist_bins,
            "unique_max_samples": args.unique_max_samples,
            "mask_remap_zero_border": args.mask_remap_zero_border,
            "manifests": {name: groups[name]["manifest"] for name in groups},
        },
        "groups": {},
        "stage_groups": {},
        "artifacts": {},
    }
    for name in groups:
        meta = dict(group_meta[name])
        meta["manifest"] = groups[name]["manifest"]
        meta["distribution"] = accumulators[name].summary()
        summary["groups"][name] = meta
    for name, description in STAGE_DESCRIPTIONS.items():
        summary["stage_groups"][name] = {
            "description": description,
            "distribution": stage_accumulators[name].summary(),
        }

    cdf_path = output_dir / "raw_cdf_gr.png"
    hist_path = output_dir / "raw_hist_gr_log.png"
    plot_cdf(cdf_path, accumulators, channel=1)
    plot_hist(hist_path, accumulators, channel=1)
    summary["artifacts"]["cdf_gr_png"] = str(cdf_path)
    summary["artifacts"]["hist_gr_png"] = str(hist_path)

    if args.fit_lut_source and args.fit_lut_target:
        lut = np.stack(
            [
                fit_lut_from_hist(
                    accumulators[args.fit_lut_source],
                    accumulators[args.fit_lut_target],
                    channel=channel,
                    lut_size=args.lut_size,
                )
                for channel in range(4)
            ],
            axis=0,
        )
        lut_path = output_dir / f"lut_{args.fit_lut_source}_to_{args.fit_lut_target}.npy"
        np.save(lut_path, lut)
        lut_meta_path = output_dir / f"lut_{args.fit_lut_source}_to_{args.fit_lut_target}.meta.json"
        lut_meta = {
            "source": args.fit_lut_source,
            "target": args.fit_lut_target,
            "lut_shape": list(lut.shape),
            "lut_path": str(lut_path),
            "source_manifest": groups[args.fit_lut_source]["manifest"],
            "target_manifest": groups[args.fit_lut_target]["manifest"],
            "source_rows_selected": int(len(groups[args.fit_lut_source]["rows"])),
            "target_rows_selected": int(len(groups[args.fit_lut_target]["rows"])),
            "all_rows_selected": {name: int(len(group["rows"])) for name, group in groups.items()},
            "hist_bins": args.hist_bins,
            "pixels_per_image": args.pixels_per_image,
            "created_at": summary["created_at"],
        }
        lut_meta_path.write_text(json.dumps(lut_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["artifacts"]["lut_npy"] = str(lut_path)
        summary["artifacts"]["lut_meta_json"] = str(lut_meta_path)

    csv_path = output_dir / "raw_distribution_summary.csv"
    stage_csv_path = output_dir / "raw_stage_summary.csv"
    json_path = output_dir / "raw_distribution_summary.json"
    write_summary_csv(csv_path, summary)
    write_stage_summary_csv(stage_csv_path, summary)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {stage_csv_path}")
    print(f"Wrote {cdf_path}")
    print(f"Wrote {hist_path}")
    if "lut_npy" in summary["artifacts"]:
        print(f"Wrote {summary['artifacts']['lut_npy']}")


if __name__ == "__main__":
    main()
