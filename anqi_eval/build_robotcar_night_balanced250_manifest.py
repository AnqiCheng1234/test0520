#!/usr/bin/env python3
"""Build a motion-gated, time-balanced, scene-interleaved RobotCar-night val manifest.

For each night scene in the input val manifest, we re-compute window cumulative VO
translation per row, drop static rows, then sample `samples_per_scene` rows
uniformly across the timestamp range of the surviving candidate pool. The two
scenes' selections are written out round-robin so that any prefix cap stays
scene-balanced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.robotcar import RobotCarValRGB, RobotCarValRaw  # noqa: E402


REQUIRED_LOADER_COLUMNS = (
    "scene",
    "sample_name",
    "rgb_src_path",
    "rgb_eval_path",
    "raw_src_path",
    "raw_native_path",
    "raw_eval_path",
    "depth_src_path",
    "depth_proxy_path",
    "meta_src_path",
    "rgb_eval_hw",
    "raw_native_hw",
    "raw_eval_hw",
    "depth_full_hw",
    "depth_fast_hw",
    "pack_order",
)
PATH_COLUMNS = (
    "rgb_src_path",
    "rgb_eval_path",
    "raw_src_path",
    "raw_native_path",
    "raw_eval_path",
    "depth_src_path",
    "depth_proxy_path",
    "meta_src_path",
)
ALLOWED_POSES_TYPE = "vo"
ALLOWED_LASER_SENSORS = "lms_front"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotcar-dataset-root",
        type=Path,
        required=True,
        help="Root that contains downloads/<scene>/vo/vo.csv (e.g. /mnt/drive/3333_raw/robotcar)",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Source full night val manifest CSV.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Destination balanced250 manifest CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Destination sidecar summary JSON.",
    )
    parser.add_argument(
        "--samples-per-scene",
        type=int,
        default=125,
        help="Number of samples to select per scene (default 125).",
    )
    parser.add_argument(
        "--window-half-sec",
        type=float,
        default=10.0,
        help="Half-window seconds; total motion window = 2 * value (default 10.0 -> 20s window).",
    )
    parser.add_argument(
        "--min-window-travel-m",
        type=float,
        default=5.0,
        help="Minimum cumulative VO translation within the window for a row to be kept (meters).",
    )
    parser.add_argument(
        "--prefix-checks",
        type=str,
        default="10,50,100,250",
        help="Comma-separated prefix sizes for scene-balance verification.",
    )
    parser.add_argument(
        "--smoke-loader-samples",
        type=int,
        default=2,
        help="Number of samples to instantiate from each dataset class as a final cross-check.",
    )
    return parser.parse_args()


def load_vo_prefix(vo_path: Path) -> tuple[list[int], list[float]]:
    timestamps: list[int] = []
    magnitudes: list[float] = []
    with vo_path.open() as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return [], [0.0]
        for row in reader:
            if not row:
                continue
            t = int(row[0])
            dx = float(row[2])
            dy = float(row[3])
            dz = float(row[4])
            timestamps.append(t)
            magnitudes.append(math.sqrt(dx * dx + dy * dy + dz * dz))
    order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
    timestamps = [timestamps[i] for i in order]
    magnitudes = [magnitudes[i] for i in order]
    prefix = [0.0] * (len(magnitudes) + 1)
    for i, mag in enumerate(magnitudes):
        prefix[i + 1] = prefix[i] + mag
    return timestamps, prefix


def window_travel(ts_sorted: list[int], prefix_sum: list[float], t_start: int, t_end: int) -> float:
    if not ts_sorted:
        return 0.0
    left = bisect_left(ts_sorted, t_start)
    right = bisect_right(ts_sorted, t_end)
    return float(prefix_sum[right] - prefix_sum[left])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))


def load_manifest_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in input manifest: {path}")
    missing = [col for col in REQUIRED_LOADER_COLUMNS if col not in fieldnames]
    if missing:
        raise ValueError(
            f"Input manifest {path} missing required columns: {missing}; loader contracts will fail downstream."
        )
    return fieldnames, rows


def time_uniform_indices(n_candidates: int, ts_values: list[int], k: int) -> list[int]:
    if k <= 0:
        return []
    if n_candidates <= k:
        return list(range(n_candidates))
    t_min = float(ts_values[0])
    t_max = float(ts_values[-1])
    if t_max <= t_min:
        return [round(i * (n_candidates - 1) / (k - 1)) if k > 1 else 0 for i in range(k)]
    selected: list[int] = []
    used: set[int] = set()
    for j in range(k):
        target = t_min + (t_max - t_min) * (j / (k - 1)) if k > 1 else (t_min + t_max) / 2.0
        idx = bisect_left(ts_values, int(round(target)))
        if idx >= n_candidates:
            idx = n_candidates - 1
        elif idx > 0 and abs(ts_values[idx - 1] - target) < abs(ts_values[idx] - target):
            idx = idx - 1
        if idx in used:
            forward = idx + 1
            backward = idx - 1
            chosen = None
            while forward < n_candidates or backward >= 0:
                if forward < n_candidates and forward not in used:
                    chosen = forward
                    break
                if backward >= 0 and backward not in used:
                    chosen = backward
                    break
                forward += 1
                backward -= 1
            if chosen is None:
                continue
            idx = chosen
        used.add(idx)
        selected.append(idx)
    selected.sort()
    return selected


def main() -> int:
    args = parse_args()
    fieldnames, rows = load_manifest_rows(args.input_manifest)
    extra_field = "window_travel_m"
    output_fieldnames = list(fieldnames)
    if extra_field not in output_fieldnames:
        output_fieldnames.append(extra_field)

    scenes: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    for row in rows:
        scenes.setdefault(row["scene"], []).append(row)
    scene_count_raw = {scene: len(scene_rows) for scene, scene_rows in scenes.items()}
    if len(scenes) != 2:
        raise ValueError(f"Expected exactly 2 scenes, got {list(scenes.keys())}")

    window_us = int(args.window_half_sec * 1e6)
    min_travel = float(args.min_window_travel_m)
    samples_per_scene = int(args.samples_per_scene)
    prefix_sizes = [int(x) for x in args.prefix_checks.split(",") if x.strip()]

    selected_per_scene: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    candidate_stats_per_scene: dict[str, dict] = {}

    for scene, scene_rows in scenes.items():
        scene_rows_sorted = sorted(scene_rows, key=lambda r: int(r["timestamp"]))
        vo_path = args.robotcar_dataset_root / "downloads" / scene / "vo" / "vo.csv"
        if not vo_path.is_file():
            raise FileNotFoundError(f"Missing VO file for scene {scene}: {vo_path}")
        vo_ts, vo_prefix = load_vo_prefix(vo_path)
        if not vo_ts:
            raise RuntimeError(f"VO file empty for scene {scene}: {vo_path}")

        candidates: list[dict[str, str]] = []
        candidate_travels: list[float] = []
        candidate_ts: list[int] = []
        for row in scene_rows_sorted:
            if str(row.get("quality_ok", "")).lower() != "true":
                continue
            if str(row.get("poses_type", "")).strip() != ALLOWED_POSES_TYPE:
                continue
            if str(row.get("laser_sensors", "")).strip() != ALLOWED_LASER_SENSORS:
                continue
            ts = int(row["timestamp"])
            travel = window_travel(vo_ts, vo_prefix, ts - window_us, ts + window_us)
            if travel < min_travel:
                continue
            row_copy = dict(row)
            row_copy[extra_field] = f"{travel:.6f}"
            candidates.append(row_copy)
            candidate_travels.append(travel)
            candidate_ts.append(ts)
        if len(candidates) < samples_per_scene:
            raise RuntimeError(
                f"Scene {scene}: only {len(candidates)} motion-gated candidates < required {samples_per_scene}; "
                f"loosen --min-window-travel-m or fix VO coverage."
            )

        ts_min = float(candidate_ts[0])
        ts_max = float(candidate_ts[-1])
        candidate_stats_per_scene[scene] = {
            "input_rows": scene_count_raw[scene],
            "motion_gated_candidates": len(candidates),
            "candidate_window_travel_m": {
                "min": float(min(candidate_travels)),
                "p10": percentile(candidate_travels, 10),
                "median": percentile(candidate_travels, 50),
                "p90": percentile(candidate_travels, 90),
                "max": float(max(candidate_travels)),
            },
            "candidate_timestamp_us_range": [int(ts_min), int(ts_max)],
            "candidate_timespan_s": (ts_max - ts_min) / 1e6,
        }

        chosen_idx = time_uniform_indices(len(candidates), candidate_ts, samples_per_scene)
        selected_rows = [candidates[i] for i in chosen_idx]
        selected_per_scene[scene] = selected_rows

    scenes_order = list(selected_per_scene.keys())
    interleaved: list[dict[str, str]] = []
    iters = [iter(selected_per_scene[scene]) for scene in scenes_order]
    while True:
        progress = False
        for scene_iter in iters:
            try:
                interleaved.append(next(scene_iter))
                progress = True
            except StopIteration:
                continue
        if not progress:
            break

    if len(interleaved) != samples_per_scene * len(scenes_order):
        raise RuntimeError(
            f"Interleave produced {len(interleaved)} rows, expected {samples_per_scene * len(scenes_order)}"
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in interleaved:
            writer.writerow({key: row.get(key, "") for key in output_fieldnames})

    prefix_distribution: dict[str, dict[str, int]] = {}
    for size in prefix_sizes:
        slice_rows = interleaved[:size]
        counts: dict[str, int] = {scene: 0 for scene in scenes_order}
        for row in slice_rows:
            counts[row["scene"]] = counts.get(row["scene"], 0) + 1
        prefix_distribution[f"prefix_{size}"] = counts

    selected_stats_per_scene: dict[str, dict] = {}
    for scene, selected_rows in selected_per_scene.items():
        ts_list = [int(row["timestamp"]) for row in selected_rows]
        gaps = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
        gaps_s = [g / 1e6 for g in gaps]
        travels = [float(row[extra_field]) for row in selected_rows]
        selected_stats_per_scene[scene] = {
            "selected_count": len(selected_rows),
            "timestamp_us_range": [ts_list[0], ts_list[-1]],
            "timestamp_span_s": (ts_list[-1] - ts_list[0]) / 1e6,
            "timestamp_gap_s": {
                "min": float(min(gaps_s)) if gaps_s else 0.0,
                "median": percentile(gaps_s, 50),
                "max": float(max(gaps_s)) if gaps_s else 0.0,
            },
            "selected_window_travel_m": {
                "min": float(min(travels)),
                "median": percentile(travels, 50),
                "max": float(max(travels)),
            },
        }

    prefix_diversity: dict[str, dict] = {}
    for size in (10, 50):
        if size > len(interleaved):
            continue
        per_scene_ts: dict[str, list[int]] = {scene: [] for scene in scenes_order}
        for row in interleaved[:size]:
            per_scene_ts[row["scene"]].append(int(row["timestamp"]))
        spans_s = {
            scene: ((max(ts_list) - min(ts_list)) / 1e6) if ts_list else 0.0
            for scene, ts_list in per_scene_ts.items()
        }
        prefix_diversity[f"prefix_{size}_scene_span_s"] = spans_s

    coverage_ratios = {}
    for scene in scenes_order:
        sel_span = selected_stats_per_scene[scene]["timestamp_span_s"]
        cand_span = candidate_stats_per_scene[scene]["candidate_timespan_s"]
        coverage_ratios[scene] = (sel_span / cand_span) if cand_span > 0 else 0.0

    summary = {
        "source_manifest": str(args.input_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "robotcar_dataset_root": str(args.robotcar_dataset_root.resolve()),
        "scenes": list(scenes_order),
        "scene_input_counts": scene_count_raw,
        "candidate_stats_per_scene": candidate_stats_per_scene,
        "selected_stats_per_scene": selected_stats_per_scene,
        "selection_strategy": "vo_motion_gated_time_uniform_per_scene_then_round_robin_interleaved",
        "min_window_travel_m": min_travel,
        "window_half_sec": float(args.window_half_sec),
        "samples_per_scene": samples_per_scene,
        "total_rows": len(interleaved),
        "prefix_distribution": prefix_distribution,
        "prefix_scene_span_s": prefix_diversity,
        "selected_to_candidate_span_ratio": coverage_ratios,
    }

    acceptance_failures: list[str] = []
    expected_total = samples_per_scene * len(scenes_order)
    if summary["total_rows"] != expected_total:
        acceptance_failures.append(f"total_rows={summary['total_rows']} expected={expected_total}")
    for scene in scenes_order:
        if selected_stats_per_scene[scene]["selected_count"] != samples_per_scene:
            acceptance_failures.append(
                f"scene {scene} selected_count={selected_stats_per_scene[scene]['selected_count']} expected={samples_per_scene}"
            )
        if min(float(row[extra_field]) for row in selected_per_scene[scene]) < min_travel:
            acceptance_failures.append(f"scene {scene} contains row with window_travel_m < {min_travel}")
    for size in (10, 50, 100, 250):
        prefix = prefix_distribution.get(f"prefix_{size}")
        if not prefix:
            continue
        expected = size // 2
        for scene, count in prefix.items():
            if count != expected:
                acceptance_failures.append(f"prefix_{size} scene {scene} count={count} expected={expected}")
    span_thresholds = {10: 45.0, 50: 300.0}
    for size, threshold in span_thresholds.items():
        spans = prefix_diversity.get(f"prefix_{size}_scene_span_s", {})
        for scene, span in spans.items():
            if span < threshold:
                acceptance_failures.append(
                    f"prefix_{size} scene {scene} span_s={span:.1f} < threshold {threshold}"
                )
    for scene, ratio in coverage_ratios.items():
        if ratio < 0.80:
            acceptance_failures.append(
                f"scene {scene} selected_to_candidate_span_ratio={ratio:.3f} < 0.80"
            )

    summary["acceptance_failures"] = acceptance_failures

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if acceptance_failures:
        for line in acceptance_failures:
            print(f"[ACCEPT-FAIL] {line}", file=sys.stderr, flush=True)
        return 1

    for col in REQUIRED_LOADER_COLUMNS:
        if col not in output_fieldnames:
            print(f"[FATAL] output manifest missing loader-required column: {col}", file=sys.stderr, flush=True)
            return 2
    for row in interleaved:
        for col in PATH_COLUMNS:
            value = row.get(col, "")
            if not value:
                print(f"[FATAL] empty path column {col} in selected row {row.get('sample_name')}", file=sys.stderr, flush=True)
                return 2
            if not Path(value).is_file():
                print(f"[FATAL] path missing on disk: {col}={value}", file=sys.stderr, flush=True)
                return 2

    manifest_root = args.output_manifest.parent.parent
    manifest_name = args.output_manifest.name

    raw_dataset = RobotCarValRaw(robotcar_root=manifest_root, manifest_name=manifest_name)
    if len(raw_dataset) != expected_total:
        print(f"[FATAL] RobotCarValRaw length={len(raw_dataset)} expected={expected_total}", file=sys.stderr, flush=True)
        return 2
    for idx in range(min(args.smoke_loader_samples, len(raw_dataset))):
        sample = raw_dataset[idx]
        if "image" not in sample or "depth" not in sample:
            print(f"[FATAL] RobotCarValRaw[{idx}] missing image/depth", file=sys.stderr, flush=True)
            return 2

    rgb_dataset = RobotCarValRGB(robotcar_root=manifest_root, manifest_name=manifest_name)
    if len(rgb_dataset) != expected_total:
        print(f"[FATAL] RobotCarValRGB length={len(rgb_dataset)} expected={expected_total}", file=sys.stderr, flush=True)
        return 2
    for idx in range(min(args.smoke_loader_samples, len(rgb_dataset))):
        sample = rgb_dataset[idx]
        if "image" not in sample or "depth" not in sample:
            print(f"[FATAL] RobotCarValRGB[{idx}] missing image/depth", file=sys.stderr, flush=True)
            return 2

    print(json.dumps({
        "status": "ok",
        "output_manifest": str(args.output_manifest.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "total_rows": summary["total_rows"],
        "scene_input_counts": scene_count_raw,
        "selected_count_per_scene": {
            scene: selected_stats_per_scene[scene]["selected_count"] for scene in scenes_order
        },
        "selected_to_candidate_span_ratio": coverage_ratios,
        "prefix_distribution": prefix_distribution,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
