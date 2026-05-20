#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.raw_audit_common import (
    DEVICE,
    build_image_edge_band,
    build_labeled_grid,
    build_raw_student_model,
    build_rgb_reference_model,
    choose_depth_vis_range,
    colorize_depth,
    colorize_error,
    colorize_scalar,
    compute_metrics_or_nan,
    denorm_rgb_tensor,
    depth_gradient_magnitude,
    infer_batched,
    load_experiment_config,
    mask_to_rgb,
    masked_stats,
    raw_tensor_to_preview,
    raw_tensor_to_pseudo_rgb_norm,
    relative_error,
    resize_rgb,
    resolve_checkpoint,
    summarize_metric_records,
    teacher_student_disagreement,
    write_json,
)
from anqi_eval.eval_rel_depth_strict import affine_align_disp
from finetune_stf.dataset.eth3d import ETH3DValRGB, ETH3DValRaw
from finetune_stf.dataset.kitti_eval import DEFAULT_KITTI_BASE, DEFAULT_KITTI_VAL_SPLIT, KITTIEval
from finetune_stf.dataset.robotcar import RobotCarValRGB, RobotCarValRaw


SPLIT_CHOICES = ("eth3d_fast", "robotcar_day_fast", "robotcar_night_fast", "kitti_val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 0 qualitative failure audit: fixed samples, 9-panel outputs, and regional metrics."
    )
    parser.add_argument("--exp-dir", required=True, type=Path, help="Experiment directory containing config.json.")
    parser.add_argument("--checkpoint", default="last", help='Checkpoint alias: "last", "current", "best", or a .pth path.')
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_CHOICES), choices=SPLIT_CHOICES)
    parser.add_argument("--sample-plan", type=Path, default=None, help="Optional JSON mapping split names to indices/sample names.")
    parser.add_argument("--samples-per-split", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--roi-file", type=Path, default=None, help="Optional manual ROI JSON for night object metrics.")
    parser.add_argument("--panel-width", type=int, default=480, help="Width of each tile in saved 9-panel images.")
    parser.add_argument("--error-max", type=float, default=0.5)
    parser.add_argument("--disagreement-threshold", type=float, default=3.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def dataset_sample_name(dataset: Any, idx: int, split_name: str) -> str:
    row = dataset.rows[int(idx)]
    if split_name == "kitti_val":
        return Path(row["image_path"]).stem
    scene = row.get("scene")
    stem = row.get("sample_name")
    if scene and stem:
        return f"{scene}/{stem}"
    return str(stem or idx)


def parse_sample_plan(path: Path | None) -> dict[str, list[Any]]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    out: dict[str, list[Any]] = {}
    for split, value in payload.items():
        if isinstance(value, dict):
            value = value.get("indices", value.get("samples"))
        if not isinstance(value, list):
            raise ValueError(f"Sample plan entry for {split!r} must be a list or object with indices/samples")
        out[str(split)] = value
    return out


def select_indices(dataset: Any, split: str, sample_plan: dict[str, list[Any]], samples_per_split: int) -> list[int]:
    if split not in sample_plan:
        count = max(1, min(int(samples_per_split), len(dataset)))
        return np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()

    name_to_idx = {dataset_sample_name(dataset, idx, split): idx for idx in range(len(dataset))}
    stem_to_idx = {name.split("/")[-1]: idx for name, idx in name_to_idx.items()}
    indices = []
    for item in sample_plan[split]:
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str) and item.isdigit():
            idx = int(item)
        elif isinstance(item, str) and item in name_to_idx:
            idx = name_to_idx[item]
        elif isinstance(item, str) and item in stem_to_idx:
            idx = stem_to_idx[item]
        else:
            raise ValueError(f"Could not resolve sample {item!r} for split {split}")
        if idx < 0 or idx >= len(dataset):
            raise ValueError(f"Sample index {idx} out of range for split {split} length {len(dataset)}")
        indices.append(int(idx))
    return sorted(dict.fromkeys(indices))


def resolve_raw_input_mode(input_type: str) -> str:
    if input_type in {
        "raw_packed",
        "raw_ram",
        "raw_ram_rgb",
        "raw_ram_residual",
        "raw_ram_feature_adapter",
        "raw_ram_bridge_feature_adapter",
        "raw_ram_bridge_feature_adapter_lora",
        "raw_ram_bridge",
        "raw_ram_bridge_lora",
        "raw_ram_rgb_bridge",
        "raw_ram_rgb_bridge_lora",
    }:
        return "raw_ram"
    return "raw_naive"


def build_split_datasets(split: str, cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    input_hw = (int(cfg.get("input_height", 644)), int(cfg.get("input_width", 1008)))
    input_type = cfg.get("input_type", "raw_ram_bridge")
    raw_input_mode = resolve_raw_input_mode(input_type)
    channel_mode = cfg.get("channel_mode", "rgb_avg_g")
    use_imagenet_norm = bool(cfg.get("use_imagenet_norm", True))

    if split == "eth3d_fast":
        meta = {
            "min_depth": float(cfg.get("eth3d_min_depth", 0.1)),
            "max_depth": float(cfg.get("eth3d_max_depth", 80.0)),
            "diagnostic_note": "real paired ETH3D RGB and RAW",
        }
        common = dict(
            eth3d_root=cfg.get("eth3d_root", "/mnt/drive/3333_raw/eth3d_raw_depth_640960"),
            depth_mode="fast",
            fast_eval_backend=cfg.get("eth3d_fast_eval_backend", "proxy"),
            min_depth=meta["min_depth"],
            max_depth=meta["max_depth"],
        )
        rgb_dataset = ETH3DValRGB(**common)
        raw_dataset = ETH3DValRaw(
            **common,
            norm_mode=cfg.get("eth3d_norm_mode", "sensor_linear"),
            channel_mode=channel_mode,
            use_imagenet_norm=use_imagenet_norm,
            input_mode=raw_input_mode,
        )
        return rgb_dataset, raw_dataset, meta

    if split in {"robotcar_day_fast", "robotcar_night_fast"}:
        is_night = split == "robotcar_night_fast"
        meta = {
            "min_depth": float(cfg.get("robotcar_night_min_depth" if is_night else "robotcar_min_depth", 0.1)),
            "max_depth": float(cfg.get("robotcar_night_max_depth" if is_night else "robotcar_max_depth", 50.0)),
            "diagnostic_note": "real paired RobotCar RGB and RAW",
        }
        common = dict(
            robotcar_root=cfg.get(
                "robotcar_night_root" if is_night else "robotcar_root",
                "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100",
            ),
            depth_mode="fast",
            fast_eval_backend=cfg.get("robotcar_night_fast_eval_backend" if is_night else "robotcar_fast_eval_backend", "sparse"),
            min_depth=meta["min_depth"],
            max_depth=meta["max_depth"],
        )
        if is_night:
            common["manifest_name"] = cfg.get("robotcar_night_manifest_name", "robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv")
        rgb_dataset = RobotCarValRGB(**common)
        raw_dataset = RobotCarValRaw(
            **common,
            norm_mode=cfg.get("robotcar_night_norm_mode" if is_night else "robotcar_norm_mode", "sensor_linear"),
            channel_mode=channel_mode,
            use_imagenet_norm=use_imagenet_norm,
            input_mode=raw_input_mode,
        )
        return rgb_dataset, raw_dataset, meta

    if split == "kitti_val":
        meta = {
            "min_depth": float(cfg.get("kitti_min_depth", 0.1)),
            "max_depth": float(cfg.get("kitti_max_depth", 80.0)),
            "diagnostic_note": "KITTI has no real RAW; RAW panels use synthetic unprocessed RGB diagnostic",
        }
        common = dict(
            filelist_path=cfg.get("kitti_val_split", str(DEFAULT_KITTI_VAL_SPLIT)),
            kitti_base=cfg.get("kitti_base", DEFAULT_KITTI_BASE),
            size=input_hw,
            min_depth=meta["min_depth"],
            max_depth=meta["max_depth"],
        )
        rgb_dataset = KITTIEval(input_type="rgb", **common)
        raw_dataset = KITTIEval(input_type="raw_ram", **common)
        return rgb_dataset, raw_dataset, meta

    raise ValueError(f"Unsupported split: {split}")


def get_raw_tensor(sample: dict[str, Any]) -> torch.Tensor:
    if "raw" in sample:
        return sample["raw"].float()
    image = sample.get("image")
    if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[0] == 4:
        return image.float()
    raise KeyError("Sample does not contain a 4-channel raw tensor")


def compute_depth_contrast(depth: np.ndarray, edge_band: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(depth)
    if finite.any():
        fill = float(np.median(depth[finite]))
    else:
        fill = 0.0
    dense = np.where(finite, depth, fill).astype(np.float32, copy=False)
    dx = np.abs(dense[:, 1:] - dense[:, :-1])
    dy = np.abs(dense[1:, :] - dense[:-1, :])
    finite_x = finite[:, 1:] & finite[:, :-1]
    finite_y = finite[1:, :] & finite[:-1, :]
    edge_x = (edge_band[:, 1:] | edge_band[:, :-1]) & finite_x
    edge_y = (edge_band[1:, :] | edge_band[:-1, :]) & finite_y
    non_x = (~(edge_band[:, 1:] | edge_band[:, :-1])) & finite_x
    non_y = (~(edge_band[1:, :] | edge_band[:-1, :])) & finite_y
    edge_vals = np.concatenate([dx[edge_x], dy[edge_y]]) if edge_x.any() or edge_y.any() else np.asarray([], dtype=np.float32)
    non_vals = np.concatenate([dx[non_x], dy[non_y]]) if non_x.any() or non_y.any() else np.asarray([], dtype=np.float32)
    edge_mean = float(edge_vals.mean()) if edge_vals.size else float("nan")
    non_mean = float(non_vals.mean()) if non_vals.size else float("nan")
    return {
        "edge_mean_abs_depth_diff": edge_mean,
        "non_edge_mean_abs_depth_diff": non_mean,
        "edge_to_non_edge_ratio": float(edge_mean / max(non_mean, 1e-6)) if np.isfinite(edge_mean) and np.isfinite(non_mean) else float("nan"),
    }


def compute_roi_metric(
    roi_payload: dict[str, Any] | None,
    *,
    split: str,
    sample_name: str,
    gt: np.ndarray,
    pred_depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> dict[str, Any]:
    if roi_payload is None:
        return {"status": "not_available", "reason": "no --roi-file provided"}
    split_payload = roi_payload.get(split, {})
    rois = split_payload.get(sample_name) or split_payload.get(sample_name.split("/")[-1])
    if not rois:
        return {"status": "not_available", "reason": "no ROI for sample"}
    roi_mask = np.zeros_like(valid_mask, dtype=bool)
    h, w = valid_mask.shape
    for roi in rois:
        x0 = int(max(0, min(w, roi["x0"])))
        x1 = int(max(0, min(w, roi["x1"])))
        y0 = int(max(0, min(h, roi["y0"])))
        y1 = int(max(0, min(h, roi["y1"])))
        if x1 > x0 and y1 > y0:
            roi_mask[y0:y1, x0:x1] = True
    metrics = compute_metrics_or_nan(gt, pred_depth, valid_mask & roi_mask, min_depth=min_depth, max_depth=max_depth)
    return {"status": "ok", "roi_count": len(rois), "metrics": metrics}


def make_tile_hw(sample_hw: tuple[int, int], panel_width: int) -> tuple[int, int]:
    h, w = sample_hw
    width = max(64, int(panel_width))
    height = max(64, int(round(h * width / max(w, 1))))
    return height, width


def add_sample_metadata(fixed_samples: list[dict[str, Any]], split: str, indices: list[int], dataset: Any) -> None:
    for order, idx in enumerate(indices, start=1):
        fixed_samples.append(
            {
                "split": split,
                "order": order,
                "dataset_index": int(idx),
                "sample_name": dataset_sample_name(dataset, idx, split),
            }
        )


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    cfg = load_experiment_config(exp_dir)
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint, cfg)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else exp_dir / f"round0_failure_audit_{args.checkpoint}"
    panel_dir = output_dir / "panels_9up"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp
    sample_plan = parse_sample_plan(args.sample_plan)
    roi_payload = None
    if args.roi_file is not None:
        with args.roi_file.expanduser().open("r", encoding="utf-8") as handle:
            roi_payload = json.load(handle)

    split_state: dict[str, dict[str, Any]] = {}
    fixed_samples: list[dict[str, Any]] = []
    for split in args.splits:
        rgb_dataset, raw_dataset, meta = build_split_datasets(split, cfg)
        if len(rgb_dataset) != len(raw_dataset):
            raise RuntimeError(f"{split}: RGB/RAW dataset length mismatch: {len(rgb_dataset)} vs {len(raw_dataset)}")
        indices = select_indices(rgb_dataset, split, sample_plan, args.samples_per_split)
        split_state[split] = {
            "rgb_dataset": rgb_dataset,
            "raw_dataset": raw_dataset,
            "meta": meta,
            "indices": indices,
        }
        add_sample_metadata(fixed_samples, split, indices, rgb_dataset)
    write_json(output_dir / "fixed_samples.json", {"samples": fixed_samples})

    print(f"[audit] device={DEVICE} exp={exp_dir.name} checkpoint={checkpoint_path}", flush=True)
    print("[audit] loading frozen RGB DAv2 reference", flush=True)
    rgb_model = build_rgb_reference_model(cfg)
    teacher_cache: dict[tuple[str, int], dict[str, Any]] = {}
    channel_mode = cfg.get("channel_mode", "rgb_avg_g")

    for split, state in split_state.items():
        rgb_dataset = state["rgb_dataset"]
        raw_dataset = state["raw_dataset"]
        meta = state["meta"]
        for idx in state["indices"]:
            rgb_sample = rgb_dataset[idx]
            raw_sample = raw_dataset[idx]
            gt = rgb_sample["depth"].numpy().astype(np.float32, copy=False)
            valid = rgb_sample["valid_mask"].numpy().astype(bool)
            target_hw = tuple(int(v) for v in gt.shape)
            true_rgb = (denorm_rgb_tensor(rgb_sample["image"]) * 255.0).round().astype(np.uint8)
            true_rgb = resize_rgb(true_rgb, target_hw)
            raw_tensor = get_raw_tensor(raw_sample)
            raw_preview = resize_rgb(raw_tensor_to_preview(raw_tensor), target_hw)
            rgb_disp = infer_batched(rgb_model, rgb_sample["image"], target_hw, use_amp=use_amp)
            raw_as_rgb_input = raw_tensor_to_pseudo_rgb_norm(raw_tensor, channel_mode=channel_mode)
            raw_as_rgb_disp = infer_batched(rgb_model, raw_as_rgb_input, target_hw, use_amp=use_amp)
            rgb_depth, rgb_align = affine_align_disp(gt, rgb_disp, valid)
            raw_as_rgb_depth, raw_as_rgb_align = affine_align_disp(gt, raw_as_rgb_disp, valid)
            edge_band = build_image_edge_band(true_rgb)
            teacher_cache[(split, int(idx))] = {
                "sample_name": dataset_sample_name(rgb_dataset, idx, split),
                "gt": gt,
                "valid": valid,
                "target_hw": target_hw,
                "true_rgb": true_rgb,
                "raw_preview": raw_preview,
                "rgb_disp": rgb_disp,
                "rgb_depth": rgb_depth,
                "rgb_align": rgb_align,
                "raw_as_rgb_disp": raw_as_rgb_disp,
                "raw_as_rgb_depth": raw_as_rgb_depth,
                "raw_as_rgb_align": raw_as_rgb_align,
                "edge_band": edge_band,
                "meta": meta,
            }
            print(f"[audit][teacher] {split} idx={idx} {teacher_cache[(split, int(idx))]['sample_name']}", flush=True)

    del rgb_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("[audit] loading RAW student checkpoint", flush=True)
    raw_model = build_raw_student_model(cfg, checkpoint_path)
    per_sample: list[dict[str, Any]] = []
    summary_buckets: dict[str, dict[str, list[dict[str, float]]]] = {}

    for split, state in split_state.items():
        raw_dataset = state["raw_dataset"]
        meta = state["meta"]
        min_depth = float(meta["min_depth"])
        max_depth = float(meta["max_depth"])
        summary_buckets.setdefault(
            split,
            {
                "valid": [],
                "edge_band": [],
                "non_edge": [],
                "gradient": [],
                "contrast": [],
                "disagreement": [],
            },
        )
        for idx in state["indices"]:
            cache = teacher_cache[(split, int(idx))]
            raw_sample = raw_dataset[idx]
            raw_tensor = get_raw_tensor(raw_sample)
            student_disp = infer_batched(raw_model, raw_tensor, cache["target_hw"], use_amp=use_amp)
            student_depth, student_align = affine_align_disp(cache["gt"], student_disp, cache["valid"])
            valid_eval = cache["valid"] & np.isfinite(student_depth) & (student_depth > 0)
            edge_eval = valid_eval & cache["edge_band"]
            non_edge_eval = valid_eval & ~cache["edge_band"]
            valid_metrics = compute_metrics_or_nan(cache["gt"], student_depth, valid_eval, min_depth=min_depth, max_depth=max_depth)
            edge_metrics = compute_metrics_or_nan(cache["gt"], student_depth, edge_eval, min_depth=min_depth, max_depth=max_depth)
            non_edge_metrics = compute_metrics_or_nan(cache["gt"], student_depth, non_edge_eval, min_depth=min_depth, max_depth=max_depth)
            rgb_metrics = compute_metrics_or_nan(cache["gt"], cache["rgb_depth"], cache["valid"], min_depth=min_depth, max_depth=max_depth)
            raw_as_rgb_metrics = compute_metrics_or_nan(cache["gt"], cache["raw_as_rgb_depth"], cache["valid"], min_depth=min_depth, max_depth=max_depth)

            invalid_mask = ~cache["valid"]
            top_rows = max(1, int(round(cache["target_hw"][0] * 0.2)))
            top_band = np.zeros_like(invalid_mask, dtype=bool)
            top_band[:top_rows, :] = True
            grad = depth_gradient_magnitude(student_depth)
            finite_student = np.isfinite(student_depth)
            grad_edge = masked_stats(grad, cache["edge_band"] & finite_student)
            grad_non_edge = masked_stats(grad, (~cache["edge_band"]) & finite_student)
            grad_ratio = {
                "edge_mean": float(grad_edge["mean"]),
                "non_edge_mean": float(grad_non_edge["mean"]),
                "edge_to_non_edge_ratio": (
                    float(grad_edge["mean"] / max(float(grad_non_edge["mean"]), 1e-6))
                    if np.isfinite(float(grad_edge["mean"])) and np.isfinite(float(grad_non_edge["mean"]))
                    else float("nan")
                ),
            }
            contrast = compute_depth_contrast(student_depth, cache["edge_band"])
            disagreement_map, disagreement_stats = teacher_student_disagreement(
                student_disp,
                cache["rgb_disp"],
                threshold=args.disagreement_threshold,
            )
            roi_metric = compute_roi_metric(
                roi_payload,
                split=split,
                sample_name=cache["sample_name"],
                gt=cache["gt"],
                pred_depth=student_depth,
                valid_mask=cache["valid"],
                min_depth=min_depth,
                max_depth=max_depth,
            )

            rel_err = relative_error(cache["gt"], student_depth, valid_eval)
            vmin, vmax = choose_depth_vis_range(cache["gt"], cache["valid"], min_depth=min_depth, max_depth=max_depth)
            tile_hw = make_tile_hw(cache["target_hw"], args.panel_width)
            panels = [
                ("input/raw preview", cache["raw_preview"]),
                ("GT valid mask", mask_to_rgb(cache["valid"], dilate=3 if cache["valid"].mean() < 0.02 else 1)),
                ("RGB DAv2 depth", colorize_depth(cache["rgb_depth"], np.isfinite(cache["rgb_depth"]), vmin=vmin, vmax=vmax)),
                ("RAW-as-RGB ZS depth", colorize_depth(cache["raw_as_rgb_depth"], np.isfinite(cache["raw_as_rgb_depth"]), vmin=vmin, vmax=vmax)),
                ("RAW student depth", colorize_depth(student_depth, np.isfinite(student_depth), vmin=vmin, vmax=vmax)),
                ("valid-pixel error", colorize_error(rel_err, valid_eval, vmax=args.error_max)),
                ("edge-band error", colorize_error(rel_err, edge_eval, vmax=args.error_max)),
                ("depth gradient", colorize_scalar(grad, np.isfinite(grad))),
                ("student-teacher disagreement", colorize_scalar(disagreement_map, np.isfinite(disagreement_map))),
            ]
            footer = f"{split} idx={idx} {cache['sample_name']} | {meta['diagnostic_note']}"
            panel = build_labeled_grid(panels, cols=3, tile_hw=tile_hw, footer=footer)
            safe_name = cache["sample_name"].replace("/", "__")
            panel_path = panel_dir / f"{split}_{idx:05d}_{safe_name}_9up.jpg"
            panel.save(panel_path, quality=95)

            record = {
                "split": split,
                "dataset_index": int(idx),
                "sample_name": cache["sample_name"],
                "panel_path": str(panel_path),
                "diagnostic_note": meta["diagnostic_note"],
                "rgb_reference_metrics": rgb_metrics,
                "raw_as_rgb_zero_shot_metrics": raw_as_rgb_metrics,
                "raw_student_valid_metrics": valid_metrics,
                "raw_student_edge_band_metrics": edge_metrics,
                "raw_student_non_edge_metrics": non_edge_metrics,
                "student_align_to_gt": student_align,
                "rgb_align_to_gt": cache["rgb_align"],
                "raw_as_rgb_align_to_gt": cache["raw_as_rgb_align"],
                "invalid_depth_stats": {
                    "student_invalid": masked_stats(student_depth, invalid_mask),
                    "rgb_teacher_invalid": masked_stats(cache["rgb_depth"], invalid_mask),
                    "student_top_band_invalid_proxy": masked_stats(student_depth, invalid_mask & top_band),
                    "rgb_teacher_top_band_invalid_proxy": masked_stats(cache["rgb_depth"], invalid_mask & top_band),
                    "top_band_fraction": 0.2,
                },
                "depth_gradient_edge_ratio": grad_ratio,
                "image_edge_conditioned_depth_contrast": contrast,
                "teacher_student_disagreement_ratio": disagreement_stats,
                "night_object_roi_metric": roi_metric,
            }
            per_sample.append(record)
            summary_buckets[split]["valid"].append(valid_metrics)
            summary_buckets[split]["edge_band"].append(edge_metrics)
            summary_buckets[split]["non_edge"].append(non_edge_metrics)
            summary_buckets[split]["gradient"].append(grad_ratio)
            summary_buckets[split]["contrast"].append(contrast)
            summary_buckets[split]["disagreement"].append(disagreement_stats)
            print(f"[audit][student] {split} idx={idx} panel={panel_path.name}", flush=True)

    split_summary: dict[str, Any] = {}
    for split, buckets in summary_buckets.items():
        split_summary[split] = {
            "n": len(buckets["valid"]),
            "valid_metrics": summarize_metric_records(buckets["valid"]),
            "edge_band_metrics": summarize_metric_records(buckets["edge_band"]),
            "non_edge_metrics": summarize_metric_records(buckets["non_edge"]),
            "depth_gradient_edge_ratio": summarize_metric_records(buckets["gradient"]),
            "image_edge_conditioned_depth_contrast": summarize_metric_records(buckets["contrast"]),
            "teacher_student_disagreement_ratio": summarize_metric_records(buckets["disagreement"]),
        }

    metrics_payload = {
        "exp_dir": str(exp_dir),
        "checkpoint": str(checkpoint_path),
        "splits": list(args.splits),
        "samples_per_split": int(args.samples_per_split),
        "panel_dir": str(panel_dir),
        "fixed_samples_path": str(output_dir / "fixed_samples.json"),
        "per_sample": per_sample,
        "summary": split_summary,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    md_lines = [
        "# Round 0 Failure Audit",
        "",
        f"- exp_dir: `{exp_dir}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- panel_dir: `{panel_dir}`",
        "",
        "| split | n | valid abs_rel/d1 | edge abs_rel/d1 | non-edge abs_rel/d1 | disagreement area |",
        "|---|---:|---|---|---|---:|",
    ]
    for split, summary in split_summary.items():
        valid = summary["valid_metrics"]
        edge = summary["edge_band_metrics"]
        non_edge = summary["non_edge_metrics"]
        disagree = summary["teacher_student_disagreement_ratio"]
        md_lines.append(
            "| {split} | {n} | {va:.4f}/{vd:.4f} | {ea:.4f}/{ed:.4f} | {na:.4f}/{nd:.4f} | {dr:.4f} |".format(
                split=split,
                n=summary["n"],
                va=valid.get("abs_rel", float("nan")),
                vd=valid.get("d1", float("nan")),
                ea=edge.get("abs_rel", float("nan")),
                ed=edge.get("d1", float("nan")),
                na=non_edge.get("abs_rel", float("nan")),
                nd=non_edge.get("d1", float("nan")),
                dr=disagree.get("area_ratio", float("nan")),
            )
        )
    md_lines.extend(
        [
            "",
            "Notes:",
            "- KITTI RAW-related panels are synthetic unprocessed RGB diagnostics because KITTI has no real RAW.",
            "- `night_object_roi_metric` is `not_available` unless a manual `--roi-file` is supplied.",
            "",
        ]
    )
    (output_dir / "metrics.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "panel_dir": str(panel_dir), "summary": split_summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
