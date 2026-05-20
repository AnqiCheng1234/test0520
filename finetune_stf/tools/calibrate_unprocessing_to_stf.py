#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune_stf.dataset.raw_utils import COMPANDED_MAX, load_rectified_bayer_npz, normalize_raw_4ch  # noqa: E402
from foundation.engine.transforms.unprocessing import build_unprocessing_transform_from_preset  # noqa: E402


DEFAULT_STF_MANIFESTS = [
    "/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_train.csv",
    "/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_val.csv",
    "/home/caq/6666_raw/seeingthroughfog/manifests/stf_raw_depth_v1_test.csv",
]
DEFAULT_RAW_NPZ_ROOT = "/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz"
DEFAULT_VKITTI_LIST = str(ROOT / "finetune_stf" / "dataset" / "splits" / "vkitti2" / "train.txt")
DEFAULT_OUTPUT = str(ROOT / "finetune_stf" / "tools" / "stf_unprocessing_calibration_v2.json")


def percentile_summary(values: List[float], ps: Tuple[int, ...] = (10, 50, 90)) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def sanitise_range(
    low: float,
    high: float,
    *,
    min_low: float | None = None,
    min_width: float | None = None,
) -> List[float]:
    if min_low is not None and low < min_low:
        low = min_low
    if min_width is not None and (high - low) < min_width:
        high = low + min_width
    return [float(low), float(high)]


def compute_basic_stats(packed_01: np.ndarray) -> Dict[str, float]:
    r, gr, gb, b = (packed_01[..., i] for i in range(4))
    g = 0.5 * (gr + gb)
    eps = 1e-6
    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    return {
        "mean_R": mean_r,
        "mean_G": mean_g,
        "mean_B": mean_b,
        "mean_all": float(packed_01.mean()),
        "p1_all": float(np.percentile(packed_01, 1)),
        "p5_all": float(np.percentile(packed_01, 5)),
        "p99_all": float(np.percentile(packed_01, 99)),
        "G_over_R": mean_g / (mean_r + eps),
        "G_over_B": mean_g / (mean_b + eps),
    }


def extract_patch_stats(
    mean_source: np.ndarray,
    var_source: np.ndarray,
    *,
    patch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = mean_source.shape
    ph = h // patch_size
    pw = w // patch_size
    if ph < 2 or pw < 2:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    mean_img = mean_source[: ph * patch_size, : pw * patch_size, :].reshape(ph, patch_size, pw, patch_size, 4)
    var_img = var_source[: ph * patch_size, : pw * patch_size, :].reshape(ph, patch_size, pw, patch_size, 4)
    patch_mean = mean_img.mean(axis=(1, 3))
    patch_var = var_img.mean(axis=(1, 3))
    return patch_mean.reshape(-1, 4), patch_var.reshape(-1, 4)


def extract_patch_mean_var_from_image(
    packed_01: np.ndarray,
    *,
    patch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = packed_01.shape
    ph = h // patch_size
    pw = w // patch_size
    if ph < 2 or pw < 2:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    img = packed_01[: ph * patch_size, : pw * patch_size, :].reshape(ph, patch_size, pw, patch_size, 4)
    patch_mean = img.mean(axis=(1, 3))
    patch_var = img.var(axis=(1, 3))
    return patch_mean.reshape(-1, 4), patch_var.reshape(-1, 4)


def fit_lower_envelope_line(
    patch_means: np.ndarray,
    patch_vars: np.ndarray,
    *,
    num_bins: int = 20,
    min_bins: int = 5,
    min_range: float = 0.03,
    lower_env_pct: float = 10.0,
) -> Tuple[float | None, float | None]:
    fits = []
    for ch in range(4):
        m = patch_means[:, ch]
        v = patch_vars[:, ch]
        if m.size < 16 or (m.max() - m.min()) < min_range:
            continue
        edges = np.linspace(m.min(), m.max(), num_bins + 1)
        bin_indices = np.clip(np.digitize(m, edges[1:-1]), 0, num_bins - 1)
        bin_means, bin_vars = [], []
        for b in range(num_bins):
            mask = bin_indices == b
            if mask.sum() < 3:
                continue
            bin_means.append(float(m[mask].mean()))
            bin_vars.append(float(np.percentile(v[mask], lower_env_pct)))
        if len(bin_means) < min_bins:
            continue
        slope, intercept = np.polyfit(bin_means, bin_vars, 1)
        fits.append((max(float(intercept), 0.0), max(float(slope), 1e-8)))
    if not fits:
        return None, None
    arr = np.asarray(fits)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def per_image_ptc(packed_01: np.ndarray, *, patch_size: int) -> Tuple[float | None, float | None]:
    patch_means, patch_vars = extract_patch_mean_var_from_image(packed_01, patch_size=patch_size)
    if patch_means.shape[0] == 0:
        return None, None
    return fit_lower_envelope_line(patch_means, patch_vars)


def per_image_residual_noise_fit(packed_01: np.ndarray, *, patch_size: int) -> Tuple[float | None, float | None]:
    smooth = cv2.blur(packed_01, (3, 3))
    residual_sq = (packed_01 - smooth) ** 2
    patch_means, patch_vars = extract_patch_stats(smooth, residual_sq, patch_size=patch_size)
    if patch_means.shape[0] == 0:
        return None, None
    return fit_lower_envelope_line(patch_means, patch_vars, min_range=0.02)


def build_vkitti_reference(vkitti_image_paths: List[str], num_samples: int, device: str) -> Dict[str, float | List[float] | int]:
    transform = build_unprocessing_transform_from_preset("stf_legacy", randomize=False).to(device)
    transform.eval()
    per_ch_means = []
    per_image_all_mean = []

    used = 0
    print(f"[VKITTI] building canonical Brooks reference on up to {num_samples} images", flush=True)
    for path in vkitti_image_paths:
        if used >= num_samples:
            break
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            packed, _ = transform(tensor)
        arr = packed[0].detach().cpu().numpy()
        per_ch_means.append(arr.mean(axis=(1, 2)))
        per_image_all_mean.append(float(arr.mean()))
        used += 1
        if used % 25 == 0 or used == num_samples:
            print(f"[VKITTI] processed {used}/{num_samples}", flush=True)

    if not per_ch_means:
        raise RuntimeError("No VKITTI images were successfully processed.")

    per_ch_means_np = np.stack(per_ch_means, axis=0)
    return {
        "canonical_exposure_gain": float(transform.canonical_params.exposure_gain),
        "per_ch_mean_avg": [float(v) for v in per_ch_means_np.mean(axis=0)],
        "R_mean_avg": float(per_ch_means_np[:, 0].mean()),
        "G_mean_avg": float(per_ch_means_np[:, 1:3].mean()),
        "B_mean_avg": float(per_ch_means_np[:, 3].mean()),
        "all_mean_avg": float(np.mean(per_image_all_mean)),
        "num_samples": int(len(per_ch_means)),
    }


def load_manifest_rows(paths: Iterable[str]) -> List[Dict[str, str]]:
    rows = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "stem": row["filename_stem"],
                        "cfa": row.get("cfa_pattern", ""),
                        "daytime": row.get("daytime", "").strip().lower(),
                    }
                )
    return rows


def make_group_recommendation(
    basic_summary: Dict[str, Dict[str, float]],
    noise_summary: Dict[str, object],
    vkitti_ref: Dict[str, float | List[float] | int],
    cfa_rec: List[str],
) -> Dict[str, object]:
    vkitti_g_ref = max(float(vkitti_ref["G_mean_avg"]), 1e-8)
    stf_g = basic_summary["mean_G"]
    exposure_rec = [stf_g["p10"] / vkitti_g_ref, stf_g["p90"] / vkitti_g_ref]

    shot_scale_floor = 1e-5
    read_var_floor = 1e-10
    read_noise_rec = [
        math.sqrt(max(noise_summary["read_var"]["p10"], read_var_floor)),
        math.sqrt(max(noise_summary["read_var"]["p90"], read_var_floor)),
    ]
    shot_log_gain_rec = [
        math.log(max(noise_summary["shot_scale"]["p10"], shot_scale_floor)),
        math.log(max(noise_summary["shot_scale"]["p90"], shot_scale_floor)),
    ]

    return {
        "red_gain_range": sanitise_range(basic_summary["G_over_R"]["p10"], basic_summary["G_over_R"]["p90"], min_low=1e-2, min_width=1e-2),
        "blue_gain_range": sanitise_range(basic_summary["G_over_B"]["p10"], basic_summary["G_over_B"]["p90"], min_low=1e-2, min_width=1e-2),
        "black_level_range": sanitise_range(basic_summary["p1_all"]["p10"], basic_summary["p1_all"]["p90"], min_low=0.0, min_width=1e-4),
        "exposure_gain_range": sanitise_range(exposure_rec[0], exposure_rec[1], min_low=1e-3, min_width=1e-3),
        "read_noise_std_range": sanitise_range(read_noise_rec[0], read_noise_rec[1], min_low=0.0, min_width=1e-4),
        "shot_log_gain_range": sanitise_range(shot_log_gain_rec[0], shot_log_gain_rec[1], min_width=1e-2),
        "cfa_patterns": cfa_rec,
    }


def summarise_basic_stats(items: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = list(items[0].keys()) if items else []
    return {k: percentile_summary([x[k] for x in items]) for k in keys}


def summarise_noise_fits(fits: List[Tuple[float, float]]) -> Dict[str, object]:
    read_vars = [x[0] for x in fits]
    shot_scales = [x[1] for x in fits]
    return {
        "num_images_fit": len(fits),
        "read_var": percentile_summary(read_vars),
        "shot_scale": percentile_summary(shot_scales),
    }


def print_group_table(group_name: str, current: Dict[str, object], recommended: Dict[str, object]) -> None:
    print(f"\n[{group_name}]")
    print(f"{'param':<26} {'current':>24} {'recommended':>28}")
    print("-" * 82)
    for k in (
        "red_gain_range",
        "blue_gain_range",
        "black_level_range",
        "exposure_gain_range",
        "read_noise_std_range",
        "shot_log_gain_range",
    ):
        cur = current[k]
        rec = recommended[k]
        print(f"{k:<26} [{cur[0]:>8.4f}, {cur[1]:>8.4f}]   [{rec[0]:>8.4f}, {rec[1]:>8.4f}]")
    print(f"{'cfa_patterns':<26} {str(current['cfa_patterns']):>24}   {str(recommended['cfa_patterns']):>28}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate Brooks unprocessing ranges to STF real raw statistics.")
    parser.add_argument("--stf-manifest", action="append", default=None, help="Repeatable STF manifest csv input.")
    parser.add_argument("--raw-npz-root", default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--vkitti-list", default=DEFAULT_VKITTI_LIST)
    parser.add_argument("--num-stf", type=int, default=-1, help="Cap on STF samples (-1 means all).")
    parser.add_argument("--num-vkitti", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-interval", type=int, default=250)
    args = parser.parse_args()

    manifests = args.stf_manifest or DEFAULT_STF_MANIFESTS
    rng = random.Random(args.seed)
    rows = load_manifest_rows(manifests)
    if not rows:
        raise RuntimeError(f"No STF rows loaded from {manifests}")
    rng.shuffle(rows)
    if args.num_stf > 0:
        rows = rows[: args.num_stf]
    print(f"[STF] loaded {len(rows)} rows from {len(manifests)} manifest(s)", flush=True)

    cfa_counter: Dict[str, int] = {}
    for row in rows:
        cfa_counter[row["cfa"]] = cfa_counter.get(row["cfa"], 0) + 1
    print(f"[STF] CFA distribution: {cfa_counter}", flush=True)

    groups = ("overall", "day", "night")
    basic_by_group: Dict[str, List[Dict[str, float]]] = {k: [] for k in groups}
    ptc_by_group: Dict[str, List[Tuple[float, float]]] = {k: [] for k in groups}
    residual_by_group: Dict[str, List[Tuple[float, float]]] = {k: [] for k in groups}

    raw_root = Path(args.raw_npz_root)
    for idx, row in enumerate(rows, start=1):
        npz_path = raw_root / f"{row['stem']}.npz"
        if not npz_path.is_file():
            continue
        packed_01 = normalize_raw_4ch(load_rectified_bayer_npz(npz_path))
        basic = compute_basic_stats(packed_01)
        ptc_fit = per_image_ptc(packed_01, patch_size=args.patch_size)
        residual_fit = per_image_residual_noise_fit(packed_01, patch_size=args.patch_size)

        basic_by_group["overall"].append(basic)
        if ptc_fit[0] is not None and ptc_fit[1] is not None:
            ptc_by_group["overall"].append((ptc_fit[0], ptc_fit[1]))
        if residual_fit[0] is not None and residual_fit[1] is not None:
            residual_by_group["overall"].append((residual_fit[0], residual_fit[1]))

        daytime = row["daytime"]
        if daytime in ("day", "night"):
            basic_by_group[daytime].append(basic)
            if ptc_fit[0] is not None and ptc_fit[1] is not None:
                ptc_by_group[daytime].append((ptc_fit[0], ptc_fit[1]))
            if residual_fit[0] is not None and residual_fit[1] is not None:
                residual_by_group[daytime].append((residual_fit[0], residual_fit[1]))
        if idx % args.progress_interval == 0 or idx == len(rows):
            print(f"[STF] processed {idx}/{len(rows)} rows", flush=True)

    if not basic_by_group["overall"]:
        raise RuntimeError("No STF samples loaded successfully from NPZ files.")

    with open(args.vkitti_list, "r", encoding="utf-8") as f:
        vkitti_image_paths = [line.strip().split()[0] for line in f if line.strip()]
    rng.shuffle(vkitti_image_paths)
    print(f"[VKITTI] loaded {len(vkitti_image_paths)} candidate RGB images from split", flush=True)
    vkitti_ref = build_vkitti_reference(vkitti_image_paths, args.num_vkitti, device=args.device)

    current = {
        "red_gain_range": [1.2, 2.4],
        "blue_gain_range": [1.2, 2.4],
        "black_level_range": [0.0, 0.02],
        "exposure_gain_range": [0.25, 0.70],
        "read_noise_std_range": [0.001, 0.02],
        "shot_log_gain_range": [-2.0, 0.0],
        "cfa_patterns": ["RGGB", "BGGR", "GRBG", "GBRG"],
    }
    cfa_rec = sorted({c for c in cfa_counter if c}, key=lambda c: -cfa_counter[c])

    group_reports: Dict[str, object] = {}
    for group in groups:
        if not basic_by_group[group]:
            continue
        basic_summary = summarise_basic_stats(basic_by_group[group])
        ptc_summary = summarise_noise_fits(ptc_by_group[group])
        residual_summary = summarise_noise_fits(residual_by_group[group])
        noise_source = "residual_noise" if residual_summary["num_images_fit"] > 0 else "patch_ptc"
        selected_noise_summary = residual_summary if noise_source == "residual_noise" else ptc_summary
        recommended = make_group_recommendation(basic_summary, selected_noise_summary, vkitti_ref, cfa_rec)
        group_reports[group] = {
            "n_stf_samples": len(basic_by_group[group]),
            "basic_summary": basic_summary,
            "patch_ptc_summary": ptc_summary,
            "residual_noise_summary": residual_summary,
            "selected_noise_source_for_recommendation": noise_source,
            "recommended_brooks_ranges": recommended,
        }

    report = {
        "args": vars(args),
        "stf_manifests": manifests,
        "companded_max": COMPANDED_MAX,
        "stf_cfa_distribution": cfa_counter,
        "vkitti_canonical_reference": vkitti_ref,
        "current_brooks_ranges": current,
        "group_reports": group_reports,
        "notes": [
            "This v2 calibration keeps STF->parameter inversion as the primary goal.",
            "Recommendations are reported separately for overall/day/night STF subsets.",
            "Noise recommendations prefer residual-noise fits (raw minus weak blur), falling back to patch PTC when residual fits are unavailable.",
            "All ranges are reported in the normalized packed-Bayer [0,1] domain after STF companded values are divided by 3967.",
            "G_over_R and G_over_B are direct estimates of Brooks red_gain and blue_gain because Brooks invert-WB divides R and B while leaving G unchanged.",
            "Exposure gain is still estimated relative to a canonical VKITTI Brooks reference at exposure_gain=1.",
            "Noise parameters remain model-input-domain fits, not exact physical sensor parameters.",
        ],
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[DONE] wrote report JSON to {out_path}", flush=True)

    sep = "=" * 88
    print(sep)
    print(f"Loaded STF rows: {len(rows)}   CFA distribution: {cfa_counter}")
    print(f"VKITTI canonical samples: {vkitti_ref['num_samples']}")
    print(sep)
    for group in ("overall", "day", "night"):
        if group not in group_reports:
            continue
        print_group_table(group, current, group_reports[group]["recommended_brooks_ranges"])
        print(
            f"noise source={group_reports[group]['selected_noise_source_for_recommendation']} "
            f"n_stf={group_reports[group]['n_stf_samples']} "
            f"residual_fit_n={group_reports[group]['residual_noise_summary']['num_images_fit']} "
            f"ptc_fit_n={group_reports[group]['patch_ptc_summary']['num_images_fit']}"
        )
    print(sep)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
