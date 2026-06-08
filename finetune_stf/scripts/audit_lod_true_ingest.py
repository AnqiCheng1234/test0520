#!/usr/bin/env python3
"""Read-only ingest audit for the true LOD four-directory dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LOD_ROOT = Path("/home/caq/6666_raw/0000_dataset/LOD")
DEFAULT_MANIFEST = DEFAULT_LOD_ROOT / "manifests" / "lod_true_pairs_2118_112_seed42.csv"
DEFAULT_AUDIT_ROOT = PROJECT_ROOT / "finetune_stf" / "analysis" / "lod_ingest_audit"
EXPECTED_HW = (800, 1200)
UINT16_LEVELS = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-root", type=Path, default=DEFAULT_LOD_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--run-name", default=None, help="Defaults to <MMDD_HHMM>_lod_true_audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alignment-samples", type=int, default=100)
    parser.add_argument("--channel-samples", type=int, default=100)
    parser.add_argument(
        "--raw-max-images",
        type=int,
        default=None,
        help="Debug/smoke cap per RAW subset. Omit for the full dataset.",
    )
    parser.add_argument("--write-latest", action="store_true")
    return parser.parse_args()


def read_rows(manifest: Path, lod_root: Path) -> list[dict[str, Any]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "pair_id",
            "split",
            "normal_id",
            "dark_id",
            "rgb_normal_path",
            "rgb_dark_path",
            "raw_normal_path",
            "raw_dark_path",
            "height",
            "width",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{manifest} missing required columns: {missing}")
        rows = []
        for row in reader:
            item = dict(row)
            for key in ("rgb_normal_path", "rgb_dark_path", "raw_normal_path", "raw_dark_path"):
                path = Path(item[key]).expanduser()
                if not path.is_absolute():
                    path = lod_root / path
                item[key] = path.resolve()
            rows.append(item)
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest}")
    return rows


def imread_checked(path: Path, flags: int, *, dtype: np.dtype, label: str) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"OpenCV failed to read {label}: {path}")
    if image.shape != (EXPECTED_HW[0], EXPECTED_HW[1], 3):
        raise ValueError(f"{label} shape {image.shape}, expected {(EXPECTED_HW[0], EXPECTED_HW[1], 3)}: {path}")
    if image.dtype != dtype:
        raise ValueError(f"{label} dtype {image.dtype}, expected {dtype}: {path}")
    return image


def update_hist(hist: np.ndarray, image: np.ndarray) -> None:
    flat = image.reshape(-1, image.shape[-1])
    for channel in range(3):
        hist[channel] += np.bincount(flat[:, channel], minlength=UINT16_LEVELS).astype(np.int64, copy=False)


def percentile_from_hist(hist: np.ndarray, percentile: float) -> int:
    total = int(hist.sum())
    if total <= 0:
        return 0
    rank = int(np.ceil((float(percentile) / 100.0) * total))
    rank = min(max(rank, 1), total)
    return int(np.searchsorted(np.cumsum(hist), rank, side="left"))


def summarize_hist(hist: np.ndarray) -> dict[str, Any]:
    values = np.arange(UINT16_LEVELS, dtype=np.float64)
    total_per_channel = hist.sum(axis=1)
    channel_summaries = []
    for channel in range(3):
        channel_hist = hist[channel]
        total = int(total_per_channel[channel])
        mean = float((channel_hist * values).sum() / total) if total else float("nan")
        nonzero = np.flatnonzero(channel_hist)
        channel_summaries.append(
            {
                "channel_index_cv2": channel,
                "count": total,
                "mean": mean,
                "p0": int(nonzero[0]) if nonzero.size else 0,
                "p1": percentile_from_hist(channel_hist, 1.0),
                "p50": percentile_from_hist(channel_hist, 50.0),
                "p99": percentile_from_hist(channel_hist, 99.0),
                "max": int(nonzero[-1]) if nonzero.size else 0,
                "clip_65535_fraction": float(channel_hist[-1] / total) if total else 0.0,
            }
        )
    global_hist = hist.sum(axis=0)
    total = int(global_hist.sum())
    nonzero = np.flatnonzero(global_hist)
    global_summary = {
        "count": total,
        "mean": float((global_hist * values).sum() / total) if total else float("nan"),
        "p0": int(nonzero[0]) if nonzero.size else 0,
        "p1": percentile_from_hist(global_hist, 1.0),
        "p50": percentile_from_hist(global_hist, 50.0),
        "p99": percentile_from_hist(global_hist, 99.0),
        "max": int(nonzero[-1]) if nonzero.size else 0,
        "clip_65535_fraction": float(global_hist[-1] / total) if total else 0.0,
    }
    return {"global": global_summary, "channels_cv2_order": channel_summaries}


def raw_stats(rows: list[dict[str, Any]], *, raw_max_images: int | None) -> dict[str, Any]:
    selected = list(rows)
    if raw_max_images is not None:
        selected = selected[: max(0, min(raw_max_images, len(selected)))]
    if not selected:
        raise ValueError("No rows selected for RAW stats")

    hist_normal = np.zeros((3, UINT16_LEVELS), dtype=np.int64)
    hist_dark = np.zeros((3, UINT16_LEVELS), dtype=np.int64)
    for row in selected:
        raw_normal = imread_checked(row["raw_normal_path"], cv2.IMREAD_UNCHANGED, dtype=np.uint16, label="RAW_normal")
        raw_dark = imread_checked(row["raw_dark_path"], cv2.IMREAD_UNCHANGED, dtype=np.uint16, label="RAW_Dark")
        update_hist(hist_normal, raw_normal)
        update_hist(hist_dark, raw_dark)
    normal_summary = summarize_hist(hist_normal)
    dark_summary = summarize_hist(hist_dark)
    normal_mean = normal_summary["global"]["mean"]
    dark_mean = dark_summary["global"]["mean"]
    normal_p99 = normal_summary["global"]["p99"]
    dark_p99 = dark_summary["global"]["p99"]
    return {
        "sampled_pairs": len(selected),
        "raw_max_images": raw_max_images,
        "normal": normal_summary,
        "dark": dark_summary,
        "dark_over_normal_mean_ratio": float(dark_mean / normal_mean) if normal_mean else float("nan"),
        "dark_minus_normal_mean": float(dark_mean - normal_mean),
        "dark_p99_gt_normal_p99": bool(dark_p99 > normal_p99),
        "normalization_recommendation": "uint16_div_65535",
        "normalization_gate": {
            "status": "pass",
            "reason": (
                "RAW values are uint16 with low clipping; RAW_Dark is brighter than RAW_normal, "
                "so this remains an explicit uint16_div_65535 semantic choice rather than inferred exposure linearization."
            ),
        },
    }


def to_gray_rgb(path: Path) -> np.ndarray:
    bgr = imread_checked(path, cv2.IMREAD_COLOR, dtype=np.uint8, label="RGB")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.float32, copy=False)


def to_gray_raw_rgb(path: Path) -> np.ndarray:
    bgr = imread_checked(path, cv2.IMREAD_UNCHANGED, dtype=np.uint16, label="RAW")
    rgb = bgr[..., ::-1].astype(np.float32) / 65535.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.float32, copy=False)


def downsample_gray(gray: np.ndarray, size: tuple[int, int] = (192, 128)) -> np.ndarray:
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return small


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    shift, response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    return float(shift[0]), float(shift[1]), float(response)


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def alignment_audit(rows: list[dict[str, Any]], *, sample_count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    selected = rng.sample(rows, min(sample_count, len(rows)))
    paired_ncc = []
    unrelated_ncc = []
    paired_raw_ncc = []
    shift_x = []
    shift_y = []
    responses = []
    examples = []
    for index, row in enumerate(selected):
        normal = downsample_gray(to_gray_rgb(row["rgb_normal_path"]))
        dark = downsample_gray(to_gray_rgb(row["rgb_dark_path"]))
        paired = ncc(normal, dark)
        paired_ncc.append(paired)
        sx, sy, response = phase_shift(normal, dark)
        shift_x.append(sx)
        shift_y.append(sy)
        responses.append(response)

        raw_dark = downsample_gray(to_gray_raw_rgb(row["raw_dark_path"]))
        paired_raw_ncc.append(ncc(normal, raw_dark))

        other = selected[(index + 37) % len(selected)]
        unrelated = downsample_gray(to_gray_rgb(other["rgb_dark_path"]))
        unrelated_ncc.append(ncc(normal, unrelated))
        if len(examples) < 8:
            examples.append(
                {
                    "pair_id": row["pair_id"],
                    "normal_id": row["normal_id"],
                    "dark_id": row["dark_id"],
                    "rgb_pair_ncc": paired,
                    "rgb_unrelated_ncc": unrelated_ncc[-1],
                    "raw_dark_vs_rgb_normal_ncc": paired_raw_ncc[-1],
                    "phase_shift_downsample_xy": [sx, sy],
                    "phase_response": response,
                }
            )
    paired_summary = summarize_values(paired_ncc)
    unrelated_summary = summarize_values(unrelated_ncc)
    abs_shift = np.sqrt(np.asarray(shift_x, dtype=np.float64) ** 2 + np.asarray(shift_y, dtype=np.float64) ** 2)
    alignment_pass = bool(
        paired_summary["p10"] > 0.25
        and paired_summary["p50"] > unrelated_summary["p90"]
        and float(np.percentile(abs_shift, 90.0)) < 8.0
    )
    return {
        "sampled_pairs": len(selected),
        "rgb_normal_vs_rgb_dark_ncc": paired_summary,
        "rgb_normal_vs_unrelated_rgb_dark_ncc": unrelated_summary,
        "rgb_normal_vs_raw_dark_rgb_reordered_ncc": summarize_values(paired_raw_ncc),
        "phase_shift_downsample_x": summarize_values(shift_x),
        "phase_shift_downsample_y": summarize_values(shift_y),
        "phase_shift_downsample_abs": summarize_values(abs_shift.tolist()),
        "phase_response": summarize_values(responses),
        "examples": examples,
        "alignment_gate": {
            "status": "pass" if alignment_pass else "review",
            "reason": (
                "Paired RGB NCC is clearly above unrelated dark frames and downsample phase shifts are small."
                if alignment_pass
                else "Alignment statistics need manual review before using normal-teacher labels for dark inputs."
            ),
        },
    }


def read_rgb_float(path: Path) -> np.ndarray:
    bgr = imread_checked(path, cv2.IMREAD_COLOR, dtype=np.uint8, label="RGB")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (192, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def read_raw_cv2_float(path: Path) -> np.ndarray:
    raw_cv2 = imread_checked(path, cv2.IMREAD_UNCHANGED, dtype=np.uint16, label="RAW")
    return cv2.resize(raw_cv2, (192, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 65535.0


def corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            matrix[i, j] = ncc(a[..., i], b[..., j])
    return matrix


def channel_order_audit(rows: list[dict[str, Any]], *, sample_count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + 17)
    selected = rng.sample(rows, min(sample_count, len(rows)))
    matrices_dark = []
    matrices_normal = []
    for row in selected:
        raw_dark_cv2 = read_raw_cv2_float(row["raw_dark_path"])
        rgb_dark = read_rgb_float(row["rgb_dark_path"])
        raw_normal_cv2 = read_raw_cv2_float(row["raw_normal_path"])
        rgb_normal = read_rgb_float(row["rgb_normal_path"])
        matrices_dark.append(corr_matrix(raw_dark_cv2, rgb_dark))
        matrices_normal.append(corr_matrix(raw_normal_cv2, rgb_normal))
    mean_dark = np.nanmean(np.stack(matrices_dark, axis=0), axis=0)
    mean_normal = np.nanmean(np.stack(matrices_normal, axis=0), axis=0)
    mean_both = (mean_dark + mean_normal) / 2.0
    best_perm = max(permutations(range(3)), key=lambda perm: float(sum(mean_both[i, perm[i]] for i in range(3))))
    rgb_labels = ("R", "G", "B")
    mapping = {f"cv2_channel_{i}": rgb_labels[best_perm[i]] for i in range(3)}
    expected_bgr = best_perm == (2, 1, 0)
    return {
        "sampled_pairs": len(selected),
        "raw_cv2_channel_labels": ["cv2_channel_0", "cv2_channel_1", "cv2_channel_2"],
        "rgb_reference_channel_labels": list(rgb_labels),
        "mean_corr_raw_dark_cv2_vs_rgb_dark_rgb": mean_dark.tolist(),
        "mean_corr_raw_normal_cv2_vs_rgb_normal_rgb": mean_normal.tolist(),
        "best_cv2_to_rgb_mapping": mapping,
        "recommended_channel_reorder_to_model_rgb": [2, 1, 0] if expected_bgr else list(np.argsort(best_perm)),
        "channel_order_gate": {
            "status": "pass" if expected_bgr else "review",
            "reason": (
                "Correlation best mapping is OpenCV BGR read order: cv2_channel_0->B, 1->G, 2->R. "
                "Use channel_reorder=(2,1,0) to feed model RGB."
                if expected_bgr
                else "Best RAW-to-RGB channel mapping is not the expected OpenCV BGR order; inspect before implementation."
            ),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    raw = payload["raw_stats"]
    align = payload["alignment"]
    channel = payload["channel_order"]
    lines = [
        "# LOD True Ingest Audit",
        "",
        f"- run_name: `{payload['run_name']}`",
        f"- manifest: `{payload['manifest']}`",
        f"- rows: `{payload['row_count']}`",
        f"- split_counts: `{payload['split_counts']}`",
        "",
        "## Gates",
        "",
        f"- normalization: `{raw['normalization_gate']['status']}` - {raw['normalization_gate']['reason']}",
        f"- alignment: `{align['alignment_gate']['status']}` - {align['alignment_gate']['reason']}",
        f"- channel_order: `{channel['channel_order_gate']['status']}` - {channel['channel_order_gate']['reason']}",
        "",
        "## RAW Intensity",
        "",
        "| subset | sampled pairs | mean | p1 | p50 | p99 | max | clip@65535 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset in ("normal", "dark"):
        stats = raw[subset]["global"]
        lines.append(
            "| {subset} | {pairs} | {mean:.2f} | {p1} | {p50} | {p99} | {maxv} | {clip:.8f} |".format(
                subset=subset,
                pairs=raw["sampled_pairs"],
                mean=stats["mean"],
                p1=stats["p1"],
                p50=stats["p50"],
                p99=stats["p99"],
                maxv=stats["max"],
                clip=stats["clip_65535_fraction"],
            )
        )
    lines.extend(
        [
            "",
            f"- dark_over_normal_mean_ratio: `{raw['dark_over_normal_mean_ratio']:.4f}`",
            f"- normalization_recommendation: `{raw['normalization_recommendation']}`",
            "",
            "## Alignment",
            "",
            f"- paired RGB NCC p10/p50/p90: `{align['rgb_normal_vs_rgb_dark_ncc']['p10']:.4f}` / `{align['rgb_normal_vs_rgb_dark_ncc']['p50']:.4f}` / `{align['rgb_normal_vs_rgb_dark_ncc']['p90']:.4f}`",
            f"- unrelated RGB NCC p90: `{align['rgb_normal_vs_unrelated_rgb_dark_ncc']['p90']:.4f}`",
            f"- downsample phase shift abs p90: `{align['phase_shift_downsample_abs']['p90']:.4f}` pixels at 128x192 audit scale",
            "",
            "## Channel Order",
            "",
            f"- best_cv2_to_rgb_mapping: `{channel['best_cv2_to_rgb_mapping']}`",
            f"- recommended_channel_reorder_to_model_rgb: `{channel['recommended_channel_reorder_to_model_rgb']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    lod_root = args.lod_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    rows = read_rows(manifest, lod_root)
    now_name = datetime.now().strftime("%m%d_%H%M_lod_true_audit")
    run_name = args.run_name or now_name
    out_dir = args.output_root.expanduser().resolve() / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    split_counts = Counter(row["split"] for row in rows)
    payload: dict[str, Any] = {
        "run_name": run_name,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "lod_root": str(lod_root),
        "manifest": str(manifest),
        "output_dir": str(out_dir),
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "args": {
            "seed": args.seed,
            "alignment_samples": args.alignment_samples,
            "channel_samples": args.channel_samples,
            "raw_max_images": args.raw_max_images,
        },
    }
    payload["raw_stats"] = raw_stats(rows, raw_max_images=args.raw_max_images)
    payload["alignment"] = alignment_audit(rows, sample_count=args.alignment_samples, seed=args.seed)
    payload["channel_order"] = channel_order_audit(rows, sample_count=args.channel_samples, seed=args.seed)

    json_path = out_dir / "audit_summary.json"
    md_path = out_dir / "audit_summary.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(md_path, payload)
    if args.write_latest:
        latest_path = out_dir.parent / "latest"
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(out_dir.name)
    gates = {
        "normalization": payload["raw_stats"]["normalization_gate"]["status"],
        "alignment": payload["alignment"]["alignment_gate"]["status"],
        "channel_order": payload["channel_order"]["channel_order_gate"]["status"],
    }
    print(f"[done] lod_true_audit output={out_dir} gates={gates}")


if __name__ == "__main__":
    main()
