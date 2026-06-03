#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune_stf.dataset.raw_utils import (  # noqa: E402
    DEFAULT_RAW_NPZ_ROOT,
    RECTIFIED_BAYER_KEY,
    decode_stf_raw_by_storage_format,
    load_rectified_bayer_npz,
)
from finetune_stf.dataset.stf import DEFAULT_STF_ROOT  # noqa: E402
from foundation.engine.datasets.led_hb import (  # noqa: E402
    LEDHBRaw,
    default_led_hb_raw_adapter_config,
)


CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
DEFAULT_OUTPUT = ROOT / "finetune_stf/tools/realraw_unprocessing_calibration.json"
DEFAULT_LED_LIST = ROOT / "finetune_stf/dataset/splits/led_hb/hb_train_all.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def quantiles(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    qs = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)
    return {f"q{q:g}": float(np.quantile(vals, q)) for q in qs}


def sample_indices(total: int, count: int, *, seed: int) -> list[int]:
    if count <= 0 or count >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), count))


def packed_base_luma(raw_chw: np.ndarray) -> np.ndarray:
    return 0.2126 * raw_chw[0] + 0.7152 * (0.5 * (raw_chw[1] + raw_chw[2])) + 0.0722 * raw_chw[3]


def append_stats(bucket: dict[str, Any], raw_chw: np.ndarray, *, pixel_stride: int, meta: dict[str, Any] | None = None) -> None:
    raw = np.asarray(raw_chw[:, ::pixel_stride, ::pixel_stride], dtype=np.float32)
    for idx, name in enumerate(CHANNEL_NAMES):
        bucket["channels"][name].append(raw[idx].reshape(-1))
    bucket["raw_all"].append(raw.reshape(-1))
    bucket["base_luma"].append(packed_base_luma(raw).reshape(-1))
    r_mean = float(raw[0].mean())
    g_mean = float((0.5 * (raw[1] + raw[2])).mean())
    b_mean = float(raw[3].mean())
    flat = raw.reshape(-1)
    sample = {
        "mean_R": r_mean,
        "mean_G": g_mean,
        "mean_B": b_mean,
        "mean_all": float(flat.mean()),
        "p50_all": float(np.quantile(flat, 0.50)),
        "p90_all": float(np.quantile(flat, 0.90)),
        "p99_all": float(np.quantile(flat, 0.99)),
        "zero_frac_lt_1e_4": float((flat < 1e-4).mean()),
        "sat_frac_gt_0p999": float((flat > 0.999).mean()),
        "G_over_R": float(g_mean / max(r_mean, 1e-8)),
        "G_over_B": float(g_mean / max(b_mean, 1e-8)),
    }
    if meta:
        sample.update(meta)
    bucket["per_sample"].append(sample)
    bucket["count"] += 1


def empty_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "channels": {name: [] for name in CHANNEL_NAMES},
        "raw_all": [],
        "base_luma": [],
        "per_sample": [],
    }


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    channels = {
        name: np.concatenate(values) if values else np.empty((0,), dtype=np.float32)
        for name, values in bucket["channels"].items()
    }
    raw_all = np.concatenate(bucket["raw_all"]) if bucket["raw_all"] else np.empty((0,), dtype=np.float32)
    base_luma = np.concatenate(bucket["base_luma"]) if bucket["base_luma"] else np.empty((0,), dtype=np.float32)
    per_sample = bucket["per_sample"]
    return {
        "count": int(bucket["count"]),
        "channels": channels,
        "raw_all": raw_all,
        "base_luma": base_luma,
        "per_sample": per_sample,
    }


def summary_from_final(stats: dict[str, Any]) -> dict[str, Any]:
    per_sample = stats["per_sample"]
    keys = (
        "mean_R",
        "mean_G",
        "mean_B",
        "mean_all",
        "p50_all",
        "p90_all",
        "p99_all",
        "zero_frac_lt_1e_4",
        "sat_frac_gt_0p999",
        "G_over_R",
        "G_over_B",
    )
    return {
        "count": int(stats["count"]),
        "raw_all_quantiles": quantiles(stats["raw_all"]),
        "base_luma_quantiles": quantiles(stats["base_luma"]),
        "channel_quantiles": {name: quantiles(values) for name, values in stats["channels"].items()},
        "per_sample": {
            key: quantiles(np.asarray([row[key] for row in per_sample], dtype=np.float32))
            for key in keys
        },
    }


def read_manifest_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"filename_stem", "daytime"}
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            for row in reader:
                row = dict(row)
                row["_manifest_path"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError(f"No STF rows loaded from manifests: {paths}")
    return rows


def load_stf_decoded_raw(raw_root: Path, sample_name: str, raw_storage_format: str) -> np.ndarray:
    path = raw_root / f"{sample_name}.npz"
    storage = load_rectified_bayer_npz(path, key=RECTIFIED_BAYER_KEY)
    decoded = decode_stf_raw_by_storage_format(storage, raw_storage_format)
    return np.transpose(decoded.astype(np.float32, copy=False), (2, 0, 1))


def collect_stf_stats(
    rows: list[dict[str, str]],
    *,
    raw_root: Path,
    raw_storage_format: str,
    pixel_stride: int,
) -> dict[str, dict[str, Any]]:
    buckets = {"overall": empty_bucket(), "day": empty_bucket(), "night": empty_bucket(), "twilight": empty_bucket()}
    skipped = 0
    for idx, row in enumerate(rows, start=1):
        sample_name = row["filename_stem"]
        try:
            raw = load_stf_decoded_raw(raw_root, sample_name, raw_storage_format)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[WARN] skip STF {sample_name}: {exc}", flush=True)
            continue
        meta = {
            "sample_name": sample_name,
            "daytime": row.get("daytime", ""),
            "manifest_path": row.get("_manifest_path", ""),
        }
        append_stats(buckets["overall"], raw, pixel_stride=pixel_stride, meta=meta)
        daytime = str(row.get("daytime", "")).strip().lower()
        if daytime in buckets and daytime != "overall":
            append_stats(buckets[daytime], raw, pixel_stride=pixel_stride, meta=meta)
        if idx % 250 == 0 or idx == len(rows):
            print(f"[STF] scanned {idx}/{len(rows)} skipped={skipped}", flush=True)
    return {name: finalize_bucket(bucket) for name, bucket in buckets.items() if bucket["count"] > 0}


def probe_config() -> dict[str, Any]:
    cfg = default_led_hb_raw_adapter_config()
    cfg.update(
        {
            "raw_adapter_red_gain_range": [1.0, 1.0],
            "raw_adapter_blue_gain_range": [1.0, 1.0],
            "raw_adapter_fixed_red_gain": 1.0,
            "raw_adapter_fixed_blue_gain": 1.0,
            "raw_adapter_shot_noise": 0.0,
            "raw_adapter_read_noise": 0.0,
            "raw_adapter_black_level": 0.0,
            "raw_adapter_white_level": 1.0,
            "raw_adapter_fixed_light_scale": 1.0,
            "raw_adapter_variant_policy": "normal",
            "raw_adapter_variant_weights": "normal=1.0,dark=0.0,over=0.0",
            "randomize_unprocessing": False,
        }
    )
    return cfg


def collect_led_probe_stats(
    led_list: Path,
    *,
    num_led: int,
    pixel_stride: int,
    seed: int,
) -> dict[str, Any]:
    dataset = LEDHBRaw(
        led_list,
        mode="train",
        min_depth=1.0,
        max_depth=200.0,
        hflip_prob=0.0,
        include_rgb_preview=False,
        unprocessing_config=probe_config(),
    )
    indices = sample_indices(len(dataset), num_led, seed=seed)
    bucket = empty_bucket()
    for local_idx, dataset_idx in enumerate(indices, start=1):
        sample = dataset.build_sample(dataset_idx, include_rgb_preview=False)
        raw = sample["raw"].detach().cpu().numpy().astype(np.float32, copy=False)
        append_stats(
            bucket,
            raw,
            pixel_stride=pixel_stride,
            meta={"dataset_index": int(dataset_idx), "sample_name": str(sample["sample_name"])},
        )
        if local_idx % 100 == 0 or local_idx == len(indices):
            print(f"[LED] processed {local_idx}/{len(indices)}", flush=True)
    return finalize_bucket(bucket)


def median_stat(stats: dict[str, Any], key: str) -> float:
    vals = np.asarray([row[key] for row in stats["per_sample"]], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"No finite values for {key}")
    return float(np.median(vals))


def apply_gains_to_led_probe(led_stats: dict[str, Any], *, red_gain: float, blue_gain: float) -> dict[str, Any]:
    bucket = empty_bucket()
    channels = led_stats["channels"]
    r = channels["R"] / float(red_gain)
    gr = channels["Gr"]
    gb = channels["Gb"]
    b = channels["B"] / float(blue_gain)
    raw_all = np.concatenate([r, gr, gb, b]).astype(np.float32, copy=False)
    base_luma = (
        0.2126 * r
        + 0.7152 * (0.5 * (gr + gb))
        + 0.0722 * b
    ).astype(np.float32, copy=False)
    bucket["count"] = int(led_stats["count"])
    bucket["channels"] = {"R": [r], "Gr": [gr], "Gb": [gb], "B": [b]}
    bucket["raw_all"] = [raw_all]
    bucket["base_luma"] = [base_luma]
    bucket["per_sample"] = []
    # Per-sample summaries are approximated from the aggregate arrays for brightness diagnostics only.
    return finalize_bucket(bucket)


def calibrate_group(
    *,
    led_probe: dict[str, Any],
    stf_target: dict[str, Any],
    group_name: str,
) -> dict[str, Any]:
    led_g_over_r = median_stat(led_probe, "G_over_R")
    led_g_over_b = median_stat(led_probe, "G_over_B")
    target_g_over_r = median_stat(stf_target, "G_over_R")
    target_g_over_b = median_stat(stf_target, "G_over_B")
    red_gain = float(np.clip(target_g_over_r / max(led_g_over_r, 1e-6), 0.1, 8.0))
    blue_gain = float(np.clip(target_g_over_b / max(led_g_over_b, 1e-6), 0.1, 8.0))

    led_wb = apply_gains_to_led_probe(led_probe, red_gain=red_gain, blue_gain=blue_gain)
    led_q = summary_from_final(led_wb)["raw_all_quantiles"]
    stf_q = summary_from_final(stf_target)["raw_all_quantiles"]
    required_light_scale = {
        "q0.5": float(stf_q["q0.5"] / max(led_q["q0.5"], 1e-8)),
        "q0.9": float(stf_q["q0.9"] / max(led_q["q0.9"], 1e-8)),
        "q0.99": float(stf_q["q0.99"] / max(led_q["q0.99"], 1e-8)),
    }
    selected_light_scale = float(np.clip(required_light_scale["q0.9"], 0.05, 2.5))
    if abs(selected_light_scale - 1.0) <= 1e-3:
        variant_policy = "normal"
        variant_weights = "normal=1.0,dark=0.0,over=0.0"
        dark_range = [0.05, 0.4]
        over_range = [1.5, 2.5]
        selected_light_scale = 1.0
    elif selected_light_scale < 1.0:
        variant_policy = "dark"
        variant_weights = "normal=0.0,dark=1.0,over=0.0"
        dark_range = [selected_light_scale, selected_light_scale]
        over_range = [1.5, 2.5]
    else:
        variant_policy = "over"
        variant_weights = "normal=0.0,dark=0.0,over=1.0"
        dark_range = [0.05, 0.4]
        over_range = [selected_light_scale, selected_light_scale]
    black_level = float(np.clip(stf_q.get("q0.001", 0.0), 0.0, 0.02))
    config = default_led_hb_raw_adapter_config()
    config.update(
        {
            "raw_adapter_red_gain_range": [red_gain, red_gain],
            "raw_adapter_blue_gain_range": [blue_gain, blue_gain],
            "raw_adapter_fixed_red_gain": red_gain,
            "raw_adapter_fixed_blue_gain": blue_gain,
            "raw_adapter_fixed_light_scale": selected_light_scale,
            "raw_adapter_dark_light_scale_range": dark_range,
            "raw_adapter_over_light_scale_range": over_range,
            "raw_adapter_variant_policy": variant_policy,
            "raw_adapter_variant_weights": variant_weights,
            "raw_adapter_shot_noise": 0.0,
            "raw_adapter_read_noise": 0.0,
            "raw_adapter_black_level": black_level,
            "raw_adapter_white_level": 1.0,
            "randomize_unprocessing": False,
        }
    )
    return {
        "group": group_name,
        "target_stf_samples": int(stf_target["count"]),
        "led_probe_samples": int(led_probe["count"]),
        "led_probe_median_G_over_R": led_g_over_r,
        "led_probe_median_G_over_B": led_g_over_b,
        "stf_target_median_G_over_R": target_g_over_r,
        "stf_target_median_G_over_B": target_g_over_b,
        "raw_adapter_overrides": config,
        "launch_args": [
            "--unprocessing-method", "raw_adapter_style",
            "--vkitti-unprocessing-preset", "not_applicable",
            "--no-randomize-unprocessing",
            "--raw-adapter-backend", "analytic",
            "--raw-adapter-cfa-pattern", "RGGB",
            "--raw-adapter-packed-channel-order", "R_Gr_Gb_B",
            "--raw-adapter-rgb-transfer", str(config["raw_adapter_rgb_transfer"]),
            "--raw-adapter-inverse-tone", str(config["raw_adapter_inverse_tone"]),
            "--raw-adapter-ccm", str(config["raw_adapter_ccm"]),
            "--raw-adapter-red-gain-range", f"{red_gain:.10g}", f"{red_gain:.10g}",
            "--raw-adapter-blue-gain-range", f"{blue_gain:.10g}", f"{blue_gain:.10g}",
            "--raw-adapter-fixed-red-gain", f"{red_gain:.10g}",
            "--raw-adapter-fixed-blue-gain", f"{blue_gain:.10g}",
            "--raw-adapter-fixed-light-scale", f"{selected_light_scale:.10g}",
            "--raw-adapter-dark-light-scale-range", f"{dark_range[0]:.10g}", f"{dark_range[1]:.10g}",
            "--raw-adapter-over-light-scale-range", f"{over_range[0]:.10g}", f"{over_range[1]:.10g}",
            "--raw-adapter-shot-noise", "0.0",
            "--raw-adapter-read-noise", "0.0",
            "--raw-adapter-noise-mean-mode", "zero",
            "--raw-adapter-black-level", f"{black_level:.10g}",
            "--raw-adapter-white-level", "1.0",
            "--raw-adapter-random-seed-policy", "dataloader_generator",
            "--raw-adapter-variant-policy", variant_policy,
            "--raw-adapter-variant-weights", variant_weights,
        ],
        "brightness_diagnostic": {
            "led_after_wb_raw_all_quantiles": led_q,
            "stf_target_raw_all_quantiles": stf_q,
            "required_light_scale_by_quantile": required_light_scale,
            "selected_light_scale": selected_light_scale,
            "selected_light_scale_quantile": "q0.9",
            "selected_variant_policy": variant_policy,
            "requires_light_scale_relaxation": bool(abs(selected_light_scale - 1.0) > 1e-3),
            "note": (
                "The selected light scale uses q0.9 as a robust brightness anchor. This is an explicit experiment-semantic "
                "raw_adapter parameter and must be written into formal launch scripts."
            ),
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate LED raw_adapter params against decoded STF RAW.")
    parser.add_argument("--stf-root", type=Path, default=Path(DEFAULT_STF_ROOT))
    parser.add_argument("--stf-manifest", type=Path, action="append", default=None)
    parser.add_argument("--raw-npz-root", type=Path, default=Path(DEFAULT_RAW_NPZ_ROOT))
    parser.add_argument("--raw-storage-format", default="legacy_bggR_decomp16", choices=["legacy_bggR_decomp16"])
    parser.add_argument("--led-list", type=Path, default=DEFAULT_LED_LIST)
    parser.add_argument("--num-led", type=int, default=1000)
    parser.add_argument("--num-stf", type=int, default=-1)
    parser.add_argument("--pixel-stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.pixel_stride <= 0:
        raise ValueError("--pixel-stride must be positive")
    stf_root = Path(args.stf_root).expanduser().resolve()
    manifests = args.stf_manifest or [
        stf_root / "manifests/stf_raw_depth_v1_train.csv",
        stf_root / "manifests/stf_raw_depth_v1_val.csv",
        stf_root / "manifests/stf_raw_depth_v1_test.csv",
    ]
    manifests = [Path(path).expanduser().resolve() for path in manifests]
    rows = read_manifest_rows(manifests)
    if args.num_stf > 0 and args.num_stf < len(rows):
        selected = sample_indices(len(rows), args.num_stf, seed=args.seed)
        rows = [rows[idx] for idx in selected]
    print(f"[SETUP] STF rows={len(rows)} manifests={len(manifests)}", flush=True)

    led_list = Path(args.led_list).expanduser().resolve()
    led_probe = collect_led_probe_stats(
        led_list,
        num_led=int(args.num_led),
        pixel_stride=int(args.pixel_stride),
        seed=int(args.seed),
    )
    stf_groups = collect_stf_stats(
        rows,
        raw_root=Path(args.raw_npz_root).expanduser().resolve(),
        raw_storage_format=str(args.raw_storage_format),
        pixel_stride=int(args.pixel_stride),
    )
    calibrations = {
        name: calibrate_group(led_probe=led_probe, stf_target=stats, group_name=name)
        for name, stats in stf_groups.items()
    }
    payload = {
        "tool": "finetune_stf/tools/calibrate_unprocessing_to_stf_decoded.py",
        "args": {
            "stf_root": str(stf_root),
            "stf_manifests": [str(path) for path in manifests],
            "raw_npz_root": str(Path(args.raw_npz_root).expanduser().resolve()),
            "raw_storage_format": str(args.raw_storage_format),
            "led_list": str(led_list),
            "num_led": int(args.num_led),
            "num_stf": int(args.num_stf),
            "pixel_stride": int(args.pixel_stride),
            "seed": int(args.seed),
        },
        "input_hashes": {
            "led_list_sha256": sha256_file(led_list),
            "stf_manifest_sha256": {str(path): sha256_file(path) for path in manifests},
        },
        "decode_contract": {
            "raw_key": RECTIFIED_BAYER_KEY,
            "raw_storage_format": str(args.raw_storage_format),
            "raw_storage_channel_order": ["B", "Gr", "Gb", "R"],
            "raw_model_channel_order": ["R", "Gr", "Gb", "B"],
            "channel_reorder": [3, 1, 2, 0],
            "raw_decompand": "stf_lut_to_0_1",
            "raw_post_decode_norm": "passthrough",
        },
        "led_probe_summary": summary_from_final(led_probe),
        "stf_group_summaries": {name: summary_from_final(stats) for name, stats in stf_groups.items()},
        "calibrations": calibrations,
        "recommended_group_for_led_hb_night": "night" if "night" in calibrations else "overall",
        "raw_adapter_overrides": calibrations["night" if "night" in calibrations else "overall"]["raw_adapter_overrides"],
        "launch_args": calibrations["night" if "night" in calibrations else "overall"]["launch_args"],
        "notes": [
            "This calibration uses decoded STF RAW, not companded raw/3967.",
            "The top-level raw_adapter_overrides choose the night group for LED-HB if available, because LED-HB is a night/high-beam training set.",
            "The top-level raw_adapter_overrides include an explicit fixed_light_scale selected from the night-group q0.9 brightness ratio.",
            "Noise values are set to zero because no-randomize unprocessing does not realize noise in the current LED-HB N-series training path.",
        ],
    }
    output = Path(args.output).expanduser().resolve()
    write_json(output, payload)
    print(f"[DONE] wrote {output}", flush=True)
    rec = payload["recommended_group_for_led_hb_night"]
    print(json.dumps({
        "recommended_group": rec,
        "raw_adapter_overrides": payload["raw_adapter_overrides"],
        "brightness_diagnostic": calibrations[rec]["brightness_diagnostic"],
    }, indent=2))


if __name__ == "__main__":
    main()
