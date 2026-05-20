from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.raw_utils import DEFAULT_RAW_NPZ_ROOT, load_rectified_bayer_npz, normalize_raw_4ch
from foundation.engine.datasets import DEFAULT_TRAIN_LIST as DEFAULT_VKITTI_TRAIN_LIST, VKITTI2Raw


DEFAULT_STF_MANIFESTS = (
    "/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_train.csv",
    "/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_val.csv",
    "/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_test.csv",
)
CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
PERCENTILES = (1, 10, 50, 90, 99)
INTENSITY_BINS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.01)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantify STF real RAW vs VKITTI pseudo-RAW gap in 4ch packed space.")
    parser.add_argument("--stf-manifest", nargs="+", default=list(DEFAULT_STF_MANIFESTS))
    parser.add_argument("--raw-npz-root", default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--vkitti-train-list", default=str(DEFAULT_VKITTI_TRAIN_LIST))
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--pixels-per-image", type=int, default=512)
    parser.add_argument("--max-stf-samples", type=int, default=None)
    parser.add_argument("--max-vkitti-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--vkitti-randomize-unprocessing", action="store_true", default=True)
    parser.add_argument("--no-vkitti-randomize-unprocessing", action="store_false", dest="vkitti_randomize_unprocessing")
    parser.add_argument(
        "--vkitti-unprocessing-preset",
        default="stf_legacy",
        help=(
            "Pseudo-raw preset for VKITTI2Raw. "
            "Defaults to stf_legacy so historical STF-gap reports remain comparable."
        ),
    )
    parser.add_argument(
        "--vkitti-unprocessing-mix-weights",
        default=None,
        help=(
            "Optional mix weights for dual preset. "
            "Examples: '0.3,0.7' or "
            "'eth3d_sensor_linear=0.3,robotcar_subset100_sensor_linear=0.7'."
        ),
    )
    parser.add_argument("--vkitti-hflip-prob", type=float, default=0.5)
    parser.add_argument("--output-json", type=str, required=True)
    return parser.parse_args()


def read_stf_rows(manifest_paths: Iterable[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for manifest_path in manifest_paths:
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def smooth_raw(raw_4ch: np.ndarray) -> np.ndarray:
    return cv2.blur(raw_4ch, (3, 3))


def sample_pixels(raw_4ch: np.ndarray, *, pixels_per_image: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, c = raw_4ch.shape
    if c != 4:
        raise ValueError(f"Expected packed RAW with 4 channels, got {raw_4ch.shape}")
    smooth = smooth_raw(raw_4ch)
    total = h * w
    count = min(int(pixels_per_image), total)
    indices = rng.choice(total, size=count, replace=False)
    raw_flat = raw_4ch.reshape(total, 4)
    smooth_flat = smooth.reshape(total, 4)
    raw_vals = raw_flat[indices]
    smooth_vals = smooth_flat[indices]
    residual_vals = raw_vals - smooth_vals
    return raw_vals, smooth_vals, residual_vals


def append_group(store: Dict[str, List[np.ndarray]], key: str, raw_vals: np.ndarray, smooth_vals: np.ndarray, residual_vals: np.ndarray) -> None:
    bucket = store.setdefault(key, [])
    bucket.append(np.concatenate([raw_vals, smooth_vals, residual_vals], axis=1))


def summarize_group(chunks: List[np.ndarray]) -> Dict[str, object]:
    if not chunks:
        return {"num_points": 0}
    merged = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    raw_vals = merged[:, 0:4]
    smooth_vals = merged[:, 4:8]
    residual_vals = merged[:, 8:12]

    channel_stats: Dict[str, object] = {}
    for idx, name in enumerate(CHANNEL_NAMES):
        v = raw_vals[:, idx]
        s = smooth_vals[:, idx]
        r = residual_vals[:, idx]
        noise_bins = []
        for lo, hi in zip(INTENSITY_BINS[:-1], INTENSITY_BINS[1:]):
            mask = (s >= lo) & (s < hi)
            if mask.any():
                noise_bins.append(
                    {
                        "range": [float(lo), float(hi)],
                        "count": int(mask.sum()),
                        "residual_std": float(r[mask].std()),
                        "residual_mean_abs": float(np.abs(r[mask]).mean()),
                    }
                )
            else:
                noise_bins.append(
                    {
                        "range": [float(lo), float(hi)],
                        "count": 0,
                        "residual_std": 0.0,
                        "residual_mean_abs": 0.0,
                    }
                )

        channel_stats[name] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "zero_frac": float((v <= 1e-8).mean()),
            "sat_frac": float((v >= 1.0 - 1e-8).mean()),
            "percentiles": {f"p{p}": float(np.percentile(v, p)) for p in PERCENTILES},
            "noise": {
                "residual_std": float(r.std()),
                "residual_mean_abs": float(np.abs(r).mean()),
                "by_intensity_bin": noise_bins,
            },
        }

    g = 0.5 * (raw_vals[:, 1] + raw_vals[:, 2])
    r = raw_vals[:, 0]
    b = raw_vals[:, 3]
    ratio_floor = 0.02
    valid_r = (g > ratio_floor) & (r > ratio_floor)
    valid_b = (g > ratio_floor) & (b > ratio_floor)
    log_g_over_r = np.log(np.maximum(g[valid_r], 1e-6)) - np.log(np.maximum(r[valid_r], 1e-6))
    log_g_over_b = np.log(np.maximum(g[valid_b], 1e-6)) - np.log(np.maximum(b[valid_b], 1e-6))

    pooled_residual = residual_vals.reshape(-1)
    summary = {
        "num_points": int(raw_vals.shape[0]),
        "channels": channel_stats,
        "channel_ratios": {
            "log_g_over_r_median": float(np.median(log_g_over_r)) if log_g_over_r.size else 0.0,
            "log_g_over_b_median": float(np.median(log_g_over_b)) if log_g_over_b.size else 0.0,
            "log_g_over_r_mean": float(log_g_over_r.mean()) if log_g_over_r.size else 0.0,
            "log_g_over_b_mean": float(log_g_over_b.mean()) if log_g_over_b.size else 0.0,
            "ratio_floor": ratio_floor,
        },
        "pooled_noise": {
            "residual_std": float(pooled_residual.std()),
            "residual_mean_abs": float(np.abs(pooled_residual).mean()),
        },
    }
    return summary


def distance_between_groups(stf_summary: Dict[str, object], vk_summary: Dict[str, object]) -> Dict[str, float]:
    percentile_diffs = []
    mean_std_diffs = []
    zero_sat_diffs = []
    noise_diffs = []
    for name in CHANNEL_NAMES:
        stf_ch = stf_summary["channels"][name]
        vk_ch = vk_summary["channels"][name]
        for p in PERCENTILES:
            percentile_diffs.append(abs(stf_ch["percentiles"][f"p{p}"] - vk_ch["percentiles"][f"p{p}"]))
        mean_std_diffs.append(abs(stf_ch["mean"] - vk_ch["mean"]))
        mean_std_diffs.append(abs(stf_ch["std"] - vk_ch["std"]))
        zero_sat_diffs.append(abs(stf_ch["zero_frac"] - vk_ch["zero_frac"]))
        zero_sat_diffs.append(abs(stf_ch["sat_frac"] - vk_ch["sat_frac"]))
        noise_diffs.append(abs(stf_ch["noise"]["residual_std"] - vk_ch["noise"]["residual_std"]))
        noise_diffs.append(abs(stf_ch["noise"]["residual_mean_abs"] - vk_ch["noise"]["residual_mean_abs"]))
        stf_bins = stf_ch["noise"]["by_intensity_bin"]
        vk_bins = vk_ch["noise"]["by_intensity_bin"]
        for stf_bin, vk_bin in zip(stf_bins, vk_bins):
            noise_diffs.append(abs(stf_bin["residual_std"] - vk_bin["residual_std"]))

    ratio_diffs = [
        abs(stf_summary["channel_ratios"]["log_g_over_r_median"] - vk_summary["channel_ratios"]["log_g_over_r_median"]),
        abs(stf_summary["channel_ratios"]["log_g_over_b_median"] - vk_summary["channel_ratios"]["log_g_over_b_median"]),
        abs(stf_summary["channel_ratios"]["log_g_over_r_mean"] - vk_summary["channel_ratios"]["log_g_over_r_mean"]),
        abs(stf_summary["channel_ratios"]["log_g_over_b_mean"] - vk_summary["channel_ratios"]["log_g_over_b_mean"]),
    ]

    result = {
        "tone_percentile_l1": float(np.mean(percentile_diffs)),
        "mean_std_l1": float(np.mean(mean_std_diffs)),
        "zero_sat_l1": float(np.mean(zero_sat_diffs)),
        "channel_ratio_l1": float(np.mean(ratio_diffs)),
        "noise_l1": float(np.mean(noise_diffs)),
    }
    result["total_score"] = (
        result["tone_percentile_l1"]
        + result["mean_std_l1"]
        + result["zero_sat_l1"]
        + result["channel_ratio_l1"]
        + result["noise_l1"]
    )
    return result


def build_stf_raw_path(filename_stem: str, raw_npz_root: str) -> Path:
    return Path(raw_npz_root).expanduser().resolve() / f"{filename_stem}.npz"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    stf_rows = read_stf_rows(args.stf_manifest)
    if args.max_stf_samples is not None:
        stf_rows = stf_rows[: args.max_stf_samples]

    stf_groups: Dict[str, List[np.ndarray]] = {"overall": [], "day": [], "night": []}
    for row in stf_rows:
        raw_path = build_stf_raw_path(row["filename_stem"], args.raw_npz_root)
        raw_4ch = normalize_raw_4ch(load_rectified_bayer_npz(raw_path))
        raw_vals, smooth_vals, residual_vals = sample_pixels(raw_4ch, pixels_per_image=args.pixels_per_image, rng=rng)
        append_group(stf_groups, "overall", raw_vals, smooth_vals, residual_vals)
        daytime = row.get("daytime", "").strip().lower()
        if daytime in ("day", "night"):
            append_group(stf_groups, daytime, raw_vals, smooth_vals, residual_vals)

    vkitti = VKITTI2Raw(
        args.vkitti_train_list,
        mode="train",
        size=(args.input_height, args.input_width),
        randomize_unprocessing=args.vkitti_randomize_unprocessing,
        unprocessing_preset=args.vkitti_unprocessing_preset,
        unprocessing_mix_weights=args.vkitti_unprocessing_mix_weights,
        hflip_prob=args.vkitti_hflip_prob,
    )
    max_vkitti = min(args.max_vkitti_samples, len(vkitti))
    vkitti_groups: Dict[str, List[np.ndarray]] = {"overall": []}
    for idx in range(max_vkitti):
        py_rng = random.Random(args.seed + idx)
        torch_generator = torch.Generator().manual_seed(args.seed + idx)
        sample = vkitti.build_sample(idx, py_rng=py_rng, torch_generator=torch_generator)
        raw_4ch = sample["raw"].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
        raw_vals, smooth_vals, residual_vals = sample_pixels(raw_4ch, pixels_per_image=args.pixels_per_image, rng=rng)
        append_group(vkitti_groups, "overall", raw_vals, smooth_vals, residual_vals)

    summary = {
        "config": {
            "pixels_per_image": args.pixels_per_image,
            "max_stf_samples": args.max_stf_samples,
            "max_vkitti_samples": args.max_vkitti_samples,
            "seed": args.seed,
            "vkitti_randomize_unprocessing": args.vkitti_randomize_unprocessing,
            "vkitti_unprocessing_preset": args.vkitti_unprocessing_preset,
            "vkitti_unprocessing_mix_weights": args.vkitti_unprocessing_mix_weights,
            "vkitti_hflip_prob": args.vkitti_hflip_prob,
            "input_hw": [args.input_height, args.input_width],
            "stf_manifests": list(args.stf_manifest),
            "raw_npz_root": args.raw_npz_root,
            "vkitti_train_list": args.vkitti_train_list,
            "vkitti_unprocessing": vkitti.describe_unprocessing(),
            "unprocessing_defaults": {
                "red_gain_range": list(vkitti.unprocessing.red_gain_range),
                "blue_gain_range": list(vkitti.unprocessing.blue_gain_range),
                "black_level_range": list(vkitti.unprocessing.black_level_range),
                "shot_log_gain_range": list(vkitti.unprocessing.shot_log_gain_range),
                "read_noise_std_range": list(vkitti.unprocessing.read_noise_std_range),
                "exposure_gain_range": list(vkitti.unprocessing.exposure_gain_range),
                "cfa_patterns": list(vkitti.unprocessing.cfa_patterns),
            },
        },
        "stf": {key: summarize_group(val) for key, val in stf_groups.items()},
        "vkitti": {key: summarize_group(val) for key, val in vkitti_groups.items()},
        "distance": {},
    }
    summary["distance"]["overall"] = distance_between_groups(summary["stf"]["overall"], summary["vkitti"]["overall"])
    if summary["stf"]["day"]["num_points"] > 0:
        summary["distance"]["day_vs_vkitti"] = distance_between_groups(summary["stf"]["day"], summary["vkitti"]["overall"])
    if summary["stf"]["night"]["num_points"] > 0:
        summary["distance"]["night_vs_vkitti"] = distance_between_groups(summary["stf"]["night"], summary["vkitti"]["overall"])

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote summary to {output_path}")
    for key, dist in summary["distance"].items():
        print(
            f"{key}: total={dist['total_score']:.6f} "
            f"tone={dist['tone_percentile_l1']:.6f} "
            f"ratio={dist['channel_ratio_l1']:.6f} "
            f"noise={dist['noise_l1']:.6f}"
        )


if __name__ == "__main__":
    main()
