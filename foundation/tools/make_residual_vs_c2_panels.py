#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from foundation.engine.datasets import VKITTI2HalfresRGBDepth, VKITTI2Raw
from foundation.engine.models import (
    build_c2_frozen_incremental_residual_model,
    build_dav2_residual_control_model,
    build_raw_residual_dav2_model,
)
from foundation.engine.transforms import resolve_unprocessing_config
from foundation.tools.make_vkitti_raw_residual_qual_panels import (
    choose_depth_range,
    clip_metric_depth_for_eval,
    colorize_depth,
    colorize_error,
    colorize_gate,
    colorize_improvement,
    colorize_signed,
    draw_tile,
    image_from_array,
    load_font,
)
from foundation.tools.residual_training_common import resolve_model_state, save_json, strip_module_prefix


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make VKITTI method-vs-C2 residual panels.")
    parser.add_argument("--c2-run-dir", required=True)
    parser.add_argument("--c2-checkpoint", required=True)
    parser.add_argument("--method-run-dir", required=True)
    parser.add_argument("--method-checkpoint", required=True)
    parser.add_argument("--method-kind", required=True, choices=["raw", "nseries", "control"])
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--selection-mode", default=None, choices=["fixed", "linspace", "scan", "mixed"])
    parser.add_argument("--topk-better", type=int, default=4)
    parser.add_argument("--topk-worse", type=int, default=4)
    parser.add_argument("--max-scan-samples", type=int, default=None)
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--error-max-abs-rel", type=float, default=0.75)
    parser.add_argument("--depth-pmin", type=float, default=1.0)
    parser.add_argument("--depth-pmax", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=414)
    parser.add_argument("--tile-height", type=int, default=125)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def format_float(value: Any, digits: int = 6) -> str:
    out = finite_float(value)
    return "" if out is None else f"{out:.{digits}f}"


def build_dataset(config: dict[str, Any]):
    if str(config.get("input_domain")) == "raw4":
        val_config = dict(config)
        if val_config.get("unprocessing_method") == "old_brooks_preset":
            val_config["randomize_unprocessing"] = False
            val_config = resolve_unprocessing_config(val_config)
        return VKITTI2Raw(
            filelist_path=config["vkitti_val_list"],
            mode="val",
            size=(int(config["input_height"]), int(config["input_width"])),
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            unprocessing_config=val_config,
            hflip_prob=0.0,
            include_rgb_input=True,
            include_rgb_preview=True,
            include_geometry=True,
            raw_storage_format=config["raw_storage_format"],
            fullres_even_policy=config["fullres_even_policy"],
            rgb_input_space=config["rgb_input_space"],
            depth_target_space=config["depth_target_space"],
        )
    return VKITTI2HalfresRGBDepth(
        filelist_path=config["vkitti_val_list"],
        mode="val",
        size=(int(config["input_height"]), int(config["input_width"])),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        hflip_prob=0.0,
        include_geometry=True,
        raw_storage_format=config["raw_storage_format"],
        fullres_even_policy=config["fullres_even_policy"],
        rgb_input_space=config["rgb_input_space"],
        depth_target_space=config["depth_target_space"],
    )


def build_c2_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_dav2_residual_control_model(
        base,
        residual_feature_source="d0",
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def build_method_model(config: dict[str, Any], checkpoint: Path, kind: str, device: torch.device) -> torch.nn.Module:
    if kind == "raw":
        base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
        model = build_raw_residual_dav2_model(
            base,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_head_d0_mode=str(config.get("residual_head_d0_mode", "concat")),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    elif kind == "control":
        base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
        model = build_dav2_residual_control_model(
            base,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    else:
        c2_config = dict(config)
        c2_model = build_c2_model(c2_config, Path(str(config["c2_checkpoint"])).expanduser().resolve(), torch.device("cpu"))
        model = build_c2_frozen_incremental_residual_model(
            c2_model,
            method_id=str(config["method_id"]),
            incremental_feature_source=str(config["incremental_feature_source"]),
            delta_condition=str(config["delta_condition"]),
            gate_condition=str(config["gate_condition"]),
            raw_feature_encoder_trainable=str(config["raw_feature_encoder_trainable"]),
            residual_alpha=float(config["residual_alpha"]),
            lambda_lp=float(config["lambda_lp"]),
            lowpass_kernel=int(config["lowpass_kernel"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def parse_index_list(arg: str | None, dataset_len: int) -> list[int]:
    if not arg:
        return []
    indices = [int(x.strip()) for x in arg.split(",") if x.strip()]
    out = []
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for validation set length {dataset_len}")
        if idx not in out:
            out.append(idx)
    return out


def linspace_indices(dataset_len: int, max_panels: int) -> list[int]:
    if dataset_len <= 0:
        return []
    if max_panels <= 1:
        indices = [0]
    else:
        indices = [int(round(x)) for x in np.linspace(0, dataset_len - 1, max(int(max_panels), 1))]
    out = []
    for idx in indices:
        if idx not in out:
            out.append(idx)
    return out[: int(max_panels)]


def parse_indices(arg: str | None, dataset_len: int, max_panels: int) -> list[int]:
    if arg:
        return parse_index_list(arg, dataset_len)[: int(max_panels)]
    return linspace_indices(dataset_len, max_panels)


def make_panel(record: dict[str, Any], args: argparse.Namespace, residual_vlim: float) -> Image.Image:
    tile_w = int(args.tile_width)
    tile_h = int(args.tile_height)
    header_h = int(args.header_height)
    font = load_font(12)
    small_font = load_font(10)
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + header_h) * 3), (0, 0, 0))
    valid = record["valid"]
    depth_range = f"{record['depth_vmin']:.2f}..{record['depth_vmax']:.2f}m"
    error_range = f"0..{float(args.error_max_abs_rel):.2f} absrel"
    residual_range = f"+/-{float(residual_vlim):.3f}"
    tiles = [
        ("RGB", "", np.clip(record["rgb"] * 255.0, 0.0, 255.0).round().astype(np.uint8)),
        ("GT depth", depth_range, colorize_depth(record["depth"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"])),
        ("C2 depth", depth_range, colorize_depth(record["aligned_c2"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"])),
        ("Method depth", depth_range, colorize_depth(record["aligned_method"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"])),
        ("C2 abs_rel error", error_range, colorize_error(record["err_c2"], valid, vmax=float(args.error_max_abs_rel))),
        ("Method abs_rel error", error_range, colorize_error(record["err_method"], valid, vmax=float(args.error_max_abs_rel))),
        (
            "Improve over C2",
            "green: method better, red: method worse",
            colorize_improvement(record["err_c2"] - record["err_method"], valid, vlim=float(args.error_max_abs_rel)),
        ),
        ("Method gate*delta", residual_range, colorize_signed(record["gate_delta"], valid, vlim=residual_vlim, cmap_name="coolwarm")),
        ("Method gate", "0..1", colorize_gate(record["gate"], valid)),
    ]
    for i, (title, subtitle, rgb) in enumerate(tiles):
        draw_tile(
            canvas,
            col=i % 3,
            row=i // 3,
            tile=image_from_array(rgb, tile_width=tile_w, tile_height=tile_h),
            title=title,
            subtitle=subtitle,
            tile_width=tile_w,
            tile_height=tile_h,
            header_height=header_h,
            font=font,
            small_font=small_font,
        )
    return canvas


def collect_record(
    *,
    sample: dict[str, Any],
    idx: int,
    c2_model: torch.nn.Module,
    method_model: torch.nn.Module,
    method_config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    image = sample["image"].unsqueeze(0).to(device).float()
    raw = sample.get("raw")
    if raw is not None:
        raw = raw.unsqueeze(0).to(device).float()
    depth_t = sample["depth"].unsqueeze(0).to(device).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device).bool()
    valid_t = valid_t & (depth_t >= float(method_config["min_depth"])) & (depth_t <= float(method_config["max_depth"]))
    if int(valid_t[0].sum().item()) < 128:
        return None
    method_batch = {"image": image, "valid_mask": valid_t}
    if raw is not None:
        method_batch["raw"] = raw
    amp_enabled = bool(method_config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if method_config.get("amp_dtype") == "fp16" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        c2_out = c2_model({"image": image, "valid_mask": valid_t})
        method_out = method_model(method_batch)
    depth = depth_t[0].detach().cpu().numpy().astype(np.float32)
    valid = valid_t[0].detach().cpu().numpy().astype(bool)
    c2_disp = c2_out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    method_disp = method_out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    aligned_c2, _ = affine_align_disp(depth, c2_disp, valid)
    aligned_method, _ = affine_align_disp(depth, method_disp, valid)
    c2_metrics = compute_metrics(depth, aligned_c2, valid, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    method_metrics = compute_metrics(depth, aligned_method, valid, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    if c2_metrics is None or method_metrics is None:
        return None
    aligned_c2_eval = clip_metric_depth_for_eval(aligned_c2, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    aligned_method_eval = clip_metric_depth_for_eval(aligned_method, min_depth=float(method_config["min_depth"]), max_depth=float(method_config["max_depth"]))
    eval_valid = valid & np.isfinite(aligned_c2_eval) & np.isfinite(aligned_method_eval) & (aligned_c2_eval > 0.0) & (aligned_method_eval > 0.0) & (depth > 0.0)
    err_c2 = np.zeros_like(depth, dtype=np.float32)
    err_method = np.zeros_like(depth, dtype=np.float32)
    err_c2[eval_valid] = np.abs(aligned_c2_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)
    err_method[eval_valid] = np.abs(aligned_method_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)
    gate_t = method_out.get("gate")
    has_gate = gate_t is not None
    if gate_t is None:
        gate_t = torch.zeros_like(method_out["pred"])
    has_delta_effective = "delta_effective" in method_out
    has_delta = "delta" in method_out
    if has_delta_effective:
        delta_t = method_out["delta_effective"]
    elif has_delta:
        delta_t = method_out["delta"]
    else:
        delta_t = torch.zeros_like(method_out["pred"])
    gate = gate_t[0].float().detach().cpu().numpy().astype(np.float32)
    gate_delta_t = gate_t.float() * delta_t.float()
    gate_delta = gate_delta_t[0].detach().cpu().numpy().astype(np.float32)
    depth_vmin, depth_vmax = choose_depth_range(
        depth,
        valid,
        min_depth=float(method_config["min_depth"]),
        max_depth=float(method_config["max_depth"]),
        pmin=float(args.depth_pmin),
        pmax=float(args.depth_pmax),
    )
    return {
        "dataset_index": int(idx),
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "rgb": sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32),
        "depth": depth,
        "valid": eval_valid,
        "aligned_c2": aligned_c2.astype(np.float32),
        "aligned_method": aligned_method.astype(np.float32),
        "err_c2": err_c2,
        "err_method": err_method,
        "gate": gate,
        "gate_delta": gate_delta,
        "depth_vmin": depth_vmin,
        "depth_vmax": depth_vmax,
        "c2_abs_rel": float(c2_metrics["abs_rel"]),
        "method_abs_rel": float(method_metrics["abs_rel"]),
        "method_minus_c2_abs_rel": float(method_metrics["abs_rel"] - c2_metrics["abs_rel"]),
        "method_better_than_c2": bool(float(method_metrics["abs_rel"] - c2_metrics["abs_rel"]) < 0.0),
        "has_gate": bool(has_gate),
        "has_delta_effective": bool(has_delta_effective),
        "has_delta": bool(has_delta),
    }


def add_selection_reason(reasons: dict[int, list[str]], idx: int, reason: str) -> None:
    bucket = reasons.setdefault(int(idx), [])
    if reason not in bucket:
        bucket.append(reason)


def collect_records_for_indices(
    *,
    indices: list[int],
    dataset: Any,
    c2_model: torch.nn.Module,
    method_model: torch.nn.Module,
    method_config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    records_by_idx: dict[int, dict[str, Any]],
) -> None:
    for idx in indices:
        if idx in records_by_idx:
            continue
        record = collect_record(
            sample=dataset[idx],
            idx=idx,
            c2_model=c2_model,
            method_model=method_model,
            method_config=method_config,
            device=device,
            args=args,
        )
        if record is not None:
            records_by_idx[int(idx)] = record


def resolve_selection_mode(args: argparse.Namespace) -> str:
    if args.selection_mode:
        return str(args.selection_mode)
    return "fixed" if args.sample_indices else "linspace"


def select_records(
    *,
    dataset: Any,
    c2_model: torch.nn.Module,
    method_model: torch.nn.Module,
    method_config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_len = len(dataset)
    mode = resolve_selection_mode(args)
    if mode == "fixed" and not args.sample_indices:
        raise ValueError("--selection-mode fixed requires --sample-indices")
    if int(args.topk_better) < 0 or int(args.topk_worse) < 0:
        raise ValueError("--topk-better and --topk-worse must be non-negative")
    if args.max_scan_samples is not None and int(args.max_scan_samples) <= 0:
        raise ValueError("--max-scan-samples must be positive when provided")

    requested_indices: list[int] = []
    reasons: dict[int, list[str]] = {}
    records_by_idx: dict[int, dict[str, Any]] = {}
    scan_metadata: dict[str, Any] = {
        "enabled": mode in ("scan", "mixed"),
        "scan_sample_count": 0,
        "scan_valid_record_count": 0,
        "topk_better": int(args.topk_better),
        "topk_worse": int(args.topk_worse),
        "max_scan_samples": args.max_scan_samples,
    }

    if mode in ("fixed", "mixed"):
        for idx in parse_index_list(args.sample_indices, dataset_len):
            requested_indices.append(idx)
            add_selection_reason(reasons, idx, "fixed")
    elif mode == "linspace":
        for idx in linspace_indices(dataset_len, int(args.max_panels)):
            requested_indices.append(idx)
            add_selection_reason(reasons, idx, "linspace")

    if mode in ("scan", "mixed"):
        scan_count = dataset_len if args.max_scan_samples is None else min(dataset_len, int(args.max_scan_samples))
        scan_metadata["scan_sample_count"] = int(scan_count)
        scan_indices = list(range(scan_count))
        collect_records_for_indices(
            indices=scan_indices,
            dataset=dataset,
            c2_model=c2_model,
            method_model=method_model,
            method_config=method_config,
            device=device,
            args=args,
            records_by_idx=records_by_idx,
        )
        scan_records = [records_by_idx[idx] for idx in scan_indices if idx in records_by_idx]
        scan_metadata["scan_valid_record_count"] = int(len(scan_records))
        sorted_records = sorted(scan_records, key=lambda record: float(record["method_minus_c2_abs_rel"]))
        better_records = sorted_records[: int(args.topk_better)]
        worse_records = list(reversed(sorted_records))[: int(args.topk_worse)]
        scan_metadata["top_better_indices"] = [int(record["dataset_index"]) for record in better_records]
        scan_metadata["top_worse_indices"] = [int(record["dataset_index"]) for record in worse_records]
        for rank, record in enumerate(better_records, start=1):
            idx = int(record["dataset_index"])
            requested_indices.append(idx)
            add_selection_reason(reasons, idx, f"top_better_rank_{rank}")
        for rank, record in enumerate(worse_records, start=1):
            idx = int(record["dataset_index"])
            requested_indices.append(idx)
            add_selection_reason(reasons, idx, f"top_worse_rank_{rank}")

    ordered_indices: list[int] = []
    for idx in requested_indices:
        if idx not in ordered_indices:
            ordered_indices.append(idx)

    collect_records_for_indices(
        indices=ordered_indices,
        dataset=dataset,
        c2_model=c2_model,
        method_model=method_model,
        method_config=method_config,
        device=device,
        args=args,
        records_by_idx=records_by_idx,
    )
    records = [records_by_idx[idx] for idx in ordered_indices if idx in records_by_idx]
    for record in records:
        idx = int(record["dataset_index"])
        record["selection_reason"] = ",".join(reasons.get(idx, ["selected"]))
    metadata = {
        "selection_mode": mode,
        "requested_indices": ordered_indices,
        "selected_indices": [int(record["dataset_index"]) for record in records],
        "selection_reason": {str(idx): ",".join(reason_list) for idx, reason_list in sorted(reasons.items())},
        "scan": scan_metadata,
    }
    return records, metadata


def records_to_rows(records: list[dict[str, Any]], *, method_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "method_label": method_label,
                "dataset_index": int(record["dataset_index"]),
                "sample_name": record["sample_name"],
                "image_path": record["image_path"],
                "depth_path": record["depth_path"],
                "panel_path": record.get("panel_path", ""),
                "c2_abs_rel": record["c2_abs_rel"],
                "method_abs_rel": record["method_abs_rel"],
                "method_minus_c2_abs_rel": record["method_minus_c2_abs_rel"],
                "method_better_than_c2": bool(record["method_better_than_c2"]),
                "selection_reason": record.get("selection_reason", ""),
                "has_gate": bool(record.get("has_gate", False)),
                "has_delta_effective": bool(record.get("has_delta_effective", False)),
            }
        )
    return rows


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method_label",
        "dataset_index",
        "sample_name",
        "image_path",
        "depth_path",
        "panel_path",
        "c2_abs_rel",
        "method_abs_rel",
        "method_minus_c2_abs_rel",
        "method_better_than_c2",
        "selection_reason",
        "has_gate",
        "has_delta_effective",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], *, method_label: str, output_dir: Path) -> dict[str, Any]:
    deltas = [float(row["method_minus_c2_abs_rel"]) for row in rows if finite_float(row["method_minus_c2_abs_rel"]) is not None]
    sorted_by_delta = sorted(rows, key=lambda row: float(row["method_minus_c2_abs_rel"])) if rows else []
    return {
        "method_label": method_label,
        "num_panels": int(len(rows)),
        "num_method_better_than_c2": int(sum(1 for row in rows if bool(row["method_better_than_c2"]))),
        "num_method_worse_than_c2": int(sum(1 for row in rows if float(row["method_minus_c2_abs_rel"]) > 0.0)),
        "mean_method_minus_c2_abs_rel": float(np.mean(deltas)) if deltas else None,
        "median_method_minus_c2_abs_rel": float(np.median(deltas)) if deltas else None,
        "best_sample_by_method_minus_c2": sorted_by_delta[0] if sorted_by_delta else None,
        "worst_sample_by_method_minus_c2": sorted_by_delta[-1] if sorted_by_delta else None,
        "output_dir": str(output_dir),
    }


def write_summary_md(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Improvement over C2 panels\n\n")
        f.write(f"- method_label: {summary.get('method_label')}\n")
        f.write(f"- num_panels: {summary.get('num_panels')}\n")
        f.write(f"- num_method_better_than_c2: {summary.get('num_method_better_than_c2')}\n")
        f.write(f"- num_method_worse_than_c2: {summary.get('num_method_worse_than_c2')}\n")
        f.write(f"- mean_method_minus_c2_abs_rel: {format_float(summary.get('mean_method_minus_c2_abs_rel'))}\n")
        f.write(f"- median_method_minus_c2_abs_rel: {format_float(summary.get('median_method_minus_c2_abs_rel'))}\n\n")
        f.write("| idx | sample | C2 abs_rel | method abs_rel | method-C2 | better | panel |\n")
        f.write("|---:|---|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row.get('dataset_index')} | {row.get('sample_name')} | "
                f"{format_float(row.get('c2_abs_rel'))} | {format_float(row.get('method_abs_rel'))} | "
                f"{format_float(row.get('method_minus_c2_abs_rel'))} | {row.get('method_better_than_c2')} | "
                f"{row.get('panel_path')} |\n"
            )
        f.write("\nNegative method-C2 means method is better than C2.\n")
        f.write("Positive method-C2 means method is worse than C2.\n")


def main() -> None:
    args = parse_args()
    c2_run_dir = Path(args.c2_run_dir).expanduser().resolve()
    method_run_dir = Path(args.method_run_dir).expanduser().resolve()
    c2_checkpoint = Path(args.c2_checkpoint).expanduser().resolve()
    method_checkpoint = Path(args.method_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    method_label = str(args.method_label) if args.method_label else method_run_dir.name
    c2_config = load_json(c2_run_dir / "config.json")
    method_config = load_json(method_run_dir / "config.json")
    method_c2_checkpoint_override = None
    if args.method_kind == "nseries":
        configured_c2 = method_config.get("c2_checkpoint")
        configured_c2_path = Path(str(configured_c2)).expanduser().resolve() if configured_c2 else None
        if configured_c2_path is not None and configured_c2_path.is_file() and configured_c2_path != c2_checkpoint:
            raise ValueError(
                "N-series method config c2_checkpoint differs from --c2-checkpoint: "
                f"{configured_c2_path} vs {c2_checkpoint}"
            )
        if configured_c2_path is None or configured_c2_path != c2_checkpoint:
            method_c2_checkpoint_override = {
                "configured": str(configured_c2) if configured_c2 else None,
                "used": str(c2_checkpoint),
                "reason": "use explicit --c2-checkpoint for frozen C2 in N-series visualization",
            }
            method_config["c2_checkpoint"] = str(c2_checkpoint)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(method_config)
    c2_model = build_c2_model(c2_config, c2_checkpoint, device)
    method_model = build_method_model(method_config, method_checkpoint, args.method_kind, device)
    records, selection_metadata = select_records(
        dataset=dataset,
        c2_model=c2_model,
        method_model=method_model,
        method_config=method_config,
        device=device,
        args=args,
    )
    print(
        "selection "
        f"mode={selection_metadata['selection_mode']} "
        f"selected={selection_metadata['selected_indices']} "
        f"scan_samples={selection_metadata['scan']['scan_sample_count']}",
        flush=True,
    )
    residual_values = [np.abs(record["gate_delta"][record["valid"]]) for record in records if bool(record["valid"].any())]
    residual_vlim = float(np.percentile(np.concatenate(residual_values), 99.0)) if residual_values else 1.0
    residual_vlim = max(residual_vlim, 1e-6)
    manifest_records = []
    for order, record in enumerate(records, start=1):
        safe_name = record["sample_name"].replace("/", "_")
        panel_path = output_dir / f"{order:02d}_validx{record['dataset_index']:04d}_{safe_name}_vs_c2.jpg"
        panel = make_panel(record, args, residual_vlim)
        panel.save(panel_path, quality=95)
        record["panel_path"] = str(panel_path)
        manifest_records.append(
            {
                "order": int(order),
                "dataset_index": int(record["dataset_index"]),
                "sample_name": record["sample_name"],
                "image_path": record["image_path"],
                "depth_path": record["depth_path"],
                "panel_path": str(panel_path),
                "c2_abs_rel": record["c2_abs_rel"],
                "method_abs_rel": record["method_abs_rel"],
                "method_minus_c2_abs_rel": record["method_minus_c2_abs_rel"],
                "method_better_than_c2": record["method_better_than_c2"],
                "selection_reason": record.get("selection_reason", ""),
                "has_gate": record.get("has_gate", False),
                "has_delta_effective": record.get("has_delta_effective", False),
                "has_delta": record.get("has_delta", False),
            }
        )
        print(f"wrote {panel_path}", flush=True)
    manifest = {
        "c2_run_dir": str(c2_run_dir),
        "c2_checkpoint": str(c2_checkpoint),
        "method_run_dir": str(method_run_dir),
        "method_checkpoint": str(method_checkpoint),
        "method_kind": args.method_kind,
        "method_label": method_label,
        "method_c2_checkpoint_override": method_c2_checkpoint_override,
        "selection_mode": selection_metadata["selection_mode"],
        "requested_indices": selection_metadata["requested_indices"],
        "selected_indices": selection_metadata["selected_indices"],
        "selection_reason": selection_metadata["selection_reason"],
        "scan": selection_metadata["scan"],
        "records": manifest_records,
        "improvement_definition": "improvement = C2_absrel_error - method_absrel_error; positive means method better",
        "panel_layout": [
            "RGB",
            "GT depth",
            "C2 depth",
            "Method depth",
            "C2 abs_rel error",
            "Method abs_rel error",
            "Improve over C2",
            "Method gate*delta",
            "Method gate",
        ],
    }
    save_json(output_dir / "manifest.json", manifest)
    if args.write_summary:
        rows = records_to_rows(records, method_label=method_label)
        summary = summarize_rows(rows, method_label=method_label, output_dir=output_dir)
        save_json(output_dir / "summary.json", {"summary": summary, "records": rows, "manifest": manifest})
        write_records_csv(output_dir / "records.csv", rows)
        write_summary_md(output_dir / "summary.md", summary, rows)
        print(f"wrote {output_dir / 'summary.json'}", flush=True)
        print(f"wrote {output_dir / 'records.csv'}", flush=True)
        print(f"wrote {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
