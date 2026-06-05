#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.models.spatial_adapter import CenterPadCropAdapter
from finetune_stf.util.loss import build_training_target, robust_normalize_target_per_sample
from foundation.engine.datasets.led_hb import LEDHBHalfresRGBDepth, LEDHBRaw
from foundation.tools.eval_led_hb_d0 import LED_REGION_KEYS, sample_led_d0_region_metrics
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    average_dicts,
    resolve_model_state,
    save_json,
    strip_module_prefix,
)
from foundation.tools.train_led_hb_incremental_residual import (
    LED_NSERIES_REGION_KEYS,
    add_dataset_raw_donor_if_needed,
    build_model as build_nseries_model,
    feature_ablation_active,
    feature_ablation_mode,
    forward_incremental_model,
    load_incremental_checkpoint,
    sample_led_region_metrics_three,
    subtract_dicts,
)
from foundation.tools.train_led_hb_residual_control import MODEL_CONFIGS, build_dav2_residual_control_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate LED-HB D0/C2/N-series runs with max-depth override.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-kind", required=True, choices=["d0", "c2", "nseries"])
    parser.add_argument("--max-depth", type=float, required=True, choices=[80.0, 200.0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--feature-ablation-mode", default="true", choices=["true", "none", "zero", "mean", "shuffle"])
    parser.add_argument("--feature-ablation-scope", default="both", choices=["both", "delta", "gate"])
    parser.add_argument("--feature-ablation-key", default="x3", choices=["x3"])
    parser.add_argument("--feature-ablation-donor-offset", type=int, default=1)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_logger() -> logging.Logger:
    logger = logging.getLogger("eval_led_hb_formal")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s"))
        logger.addHandler(handler)
    return logger


def build_rgb_dataset(config: dict[str, Any], max_depth: float) -> LEDHBHalfresRGBDepth:
    return LEDHBHalfresRGBDepth(
        filelist_path=config["led_val_list"],
        mode="val",
        size=(int(config["input_height"]), int(config["input_width"])),
        min_depth=float(config["min_depth"]),
        max_depth=float(max_depth),
        hflip_prob=0.0,
        include_geometry=True,
        dataset_name=str(config.get("dataset_name", "led_hb")),
        illumination=str(config.get("illumination", "HB")),
        led_geometry_mode=str(config.get("led_geometry_mode", config.get("dataset_geometry_mode"))),
        raw_storage_format="not_applicable",
        rgb_input_space=str(config["rgb_input_space"]),
        depth_target_space=str(config["depth_target_space"]),
        depth_label=str(config["depth_label"]),
        depth_unit=str(config["depth_unit"]),
    )


def build_raw_dataset(config: dict[str, Any], max_depth: float) -> LEDHBRaw:
    return LEDHBRaw(
        filelist_path=config["led_val_list"],
        mode="val",
        size=(int(config["input_height"]), int(config["input_width"])),
        min_depth=float(config["min_depth"]),
        max_depth=float(max_depth),
        hflip_prob=0.0,
        include_rgb_input=True,
        include_rgb_preview=True,
        include_geometry=True,
        dataset_name=str(config.get("dataset_name", "led_hb")),
        illumination=str(config.get("illumination", "HB")),
        led_geometry_mode=str(config.get("led_geometry_mode", config.get("dataset_geometry_mode"))),
        raw_storage_format=str(config["raw_storage_format"]),
        rgb_input_space=str(config["rgb_input_space"]),
        depth_target_space=str(config["depth_target_space"]),
        depth_label=str(config["depth_label"]),
        depth_unit=str(config["depth_unit"]),
        unprocessing_config=config["resolved_unprocessing_config"],
    )


def loader_for(dataset: Any, batch_size: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")


def aggregate_c2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    final = average_dicts([row["final"] for row in ok], METRIC_KEYS)
    d0 = average_dicts([row["D0"] for row in ok], METRIC_KEYS)
    region_final = average_dicts([row["region"]["final"] for row in ok], LED_REGION_KEYS)
    region_d0 = average_dicts([row["region"]["D0"] for row in ok], LED_REGION_KEYS)
    return {
        "samples": len(ok),
        "overall": {"final": final, "D0": d0, "delta_final_minus_D0": subtract_dicts(final, d0, METRIC_KEYS)},
        "region": {"final": region_final, "D0": region_d0, "delta_final_minus_D0": subtract_dicts(region_final, region_d0, LED_REGION_KEYS)},
    }


def aggregate_nseries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    final = average_dicts([row["final"] for row in ok], METRIC_KEYS)
    d1 = average_dicts([row["D1"] for row in ok], METRIC_KEYS)
    d0 = average_dicts([row["D0"] for row in ok], METRIC_KEYS)
    region_final = average_dicts([row["region"]["final"] for row in ok], LED_NSERIES_REGION_KEYS)
    region_d1 = average_dicts([row["region"]["D1"] for row in ok], LED_NSERIES_REGION_KEYS)
    region_d0 = average_dicts([row["region"]["D0"] for row in ok], LED_NSERIES_REGION_KEYS)
    return {
        "samples": len(ok),
        "overall": {
            "final": final,
            "D1": d1,
            "D0": d0,
            "delta_final_minus_D1": subtract_dicts(final, d1, METRIC_KEYS),
            "delta_D1_minus_D0": subtract_dicts(d1, d0, METRIC_KEYS),
        },
        "region": {
            "final": region_final,
            "D1": region_d1,
            "D0": region_d0,
            "delta_final_minus_D1": subtract_dicts(region_final, region_d1, LED_NSERIES_REGION_KEYS),
            "delta_D1_minus_D0": subtract_dicts(region_d1, region_d0, LED_NSERIES_REGION_KEYS),
        },
    }


def evaluate_c2(config: dict[str, Any], checkpoint: Path, max_depth: float, device: torch.device, max_samples: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = build_rgb_dataset(config, max_depth)
    loader = loader_for(dataset, 1, device)
    base_model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_dav2_residual_control_model(
        base_model,
        residual_feature_source=str(config["residual_feature_source"]),
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    model = model.to(device).eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_samples is not None and len(rows) >= max_samples:
                break
            row: dict[str, Any] = {"dataset_index": int(idx), "sample_name": batch["sample_name"][0], "status": "ok"}
            image = batch["image"].to(device).float()
            depth_t = batch["depth"].to(device).float()
            valid_t = batch["valid_mask"].to(device).bool() & (depth_t >= float(config["min_depth"])) & (depth_t <= max_depth)
            if int(valid_t[0].sum().item()) < 128:
                row["status"] = "skipped_invalid_pixels"
                rows.append(row)
                continue
            out = model({"image": image, "valid_mask": valid_t})
            inv_gt = build_training_target(depth_t, valid_t, target_space="metric_depth")
            y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_t, min_valid_pixels=128)
            depth_np = depth_t[0].detach().cpu().numpy().astype(np.float32)
            valid_np = valid_t[0].detach().cpu().numpy().astype(bool)
            pred_np = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
            d0_np = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
            aligned_final, _ = affine_align_disp(depth_np, pred_np, valid_np)
            aligned_d0, _ = affine_align_disp(depth_np, d0_np, valid_np)
            metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=float(config["min_depth"]), max_depth=max_depth)
            metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=float(config["min_depth"]), max_depth=max_depth)
            if metrics_final is None or metrics_d0 is None:
                row["status"] = "skipped_metric_failure"
                rows.append(row)
                continue
            rgb_preview = batch["rgb_preview"][0].permute(1, 2, 0).numpy().astype(np.float32)
            region_final = sample_led_d0_region_metrics(depth_np=depth_np, valid_np=valid_np, aligned_d0=aligned_final, d0_norm_np=out["D0_norm"][0].float().detach().cpu().numpy(), y_norm_np=y_norm[0].float().detach().cpu().numpy(), rgb_preview_np=rgb_preview, min_depth=float(config["min_depth"]), max_depth=max_depth)
            region_d0 = sample_led_d0_region_metrics(depth_np=depth_np, valid_np=valid_np, aligned_d0=aligned_d0, d0_norm_np=out["D0_norm"][0].float().detach().cpu().numpy(), y_norm_np=y_norm[0].float().detach().cpu().numpy(), rgb_preview_np=rgb_preview, min_depth=float(config["min_depth"]), max_depth=max_depth)
            row.update({"final": {k: float(metrics_final[k]) for k in METRIC_KEYS if k in metrics_final}, "D0": {k: float(metrics_d0[k]) for k in METRIC_KEYS if k in metrics_d0}, "region": {"final": region_final, "D0": region_d0}})
            rows.append(row)
    summary = aggregate_c2(rows)
    summary.update({"dataset_geometry": dataset.describe_geometry()})
    return summary, rows


def evaluate_nseries(args: argparse.Namespace, config: dict[str, Any], checkpoint: Path, max_depth: float, device: torch.device, max_samples: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_config = dict(config)
    eval_config["max_depth"] = float(max_depth)
    dataset = build_raw_dataset(eval_config, max_depth) if str(eval_config["input_domain"]) == "raw4" else build_rgb_dataset(eval_config, max_depth)
    args_ns = argparse.Namespace(**eval_config)
    args_ns.eval_only = True
    args_ns.eval_feature_ablation_mode = args.feature_ablation_mode
    args_ns.feature_ablation_scope = args.feature_ablation_scope
    args_ns.feature_ablation_key = args.feature_ablation_key
    args_ns.feature_ablation_seed = int(eval_config.get("feature_ablation_seed", 42))
    args_ns.feature_ablation_donor_offset = int(args.feature_ablation_donor_offset)
    args_ns.train_feature_ablation_mode = "true"
    args_ns.bs = int(eval_config.get("bs", 1))
    args_ns.amp = bool(eval_config.get("amp", False))
    args_ns.lowpass_kernel = int(eval_config["lowpass_kernel"])
    loader = loader_for(dataset, max(int(args_ns.bs), 1), device)
    model = build_nseries_model(args_ns)
    load_incremental_checkpoint(model, checkpoint, strict=True)
    model = model.to(device).eval()
    rows: list[dict[str, Any]] = []
    visited = 0
    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and len(rows) >= max_samples:
                break
            image = batch["image"].to(device).float()
            raw = batch.get("raw")
            if raw is not None:
                raw = raw.to(device).float()
            depth_t = batch["depth"].to(device).float()
            valid_t = batch["valid_mask"].to(device).bool() & (depth_t >= float(eval_config["min_depth"])) & (depth_t <= max_depth)
            model_batch: dict[str, torch.Tensor] = {"image": image, "valid_mask": valid_t}
            if raw is not None:
                model_batch["raw"] = raw
            batch_size = int(image.shape[0])
            add_dataset_raw_donor_if_needed(model_batch=model_batch, dataset=dataset, sample_indices=[visited + i for i in range(batch_size)], args=args_ns, device=device)
            out = forward_incremental_model(model, model_batch, args_ns, phase="eval")
            inv_gt = build_training_target(depth_t, valid_t, target_space="metric_depth")
            y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_t, min_valid_pixels=128)
            for sample_idx in range(batch_size):
                if max_samples is not None and len(rows) >= max_samples:
                    break
                row: dict[str, Any] = {"dataset_index": int(visited + sample_idx), "sample_name": batch["sample_name"][sample_idx], "status": "ok"}
                valid_i = valid_t[sample_idx]
                if int(valid_i.sum().item()) < 128:
                    row["status"] = "skipped_invalid_pixels"
                    rows.append(row)
                    continue
                depth_np = depth_t[sample_idx].detach().cpu().numpy().astype(np.float32)
                valid_np = valid_i.detach().cpu().numpy().astype(bool)
                final_disp = out["pred"][sample_idx].float().detach().cpu().numpy().astype(np.float32)
                d1_disp = out["D1_norm"][sample_idx].float().detach().cpu().numpy().astype(np.float32)
                d0_disp = (float(eval_config["d0_sign"]) * out["D0"][sample_idx].float()).detach().cpu().numpy().astype(np.float32)
                aligned_final, _ = affine_align_disp(depth_np, final_disp, valid_np)
                aligned_d1, _ = affine_align_disp(depth_np, d1_disp, valid_np)
                aligned_d0, _ = affine_align_disp(depth_np, d0_disp, valid_np)
                metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=float(eval_config["min_depth"]), max_depth=max_depth)
                metrics_d1 = compute_metrics(depth_np, aligned_d1, valid_np, min_depth=float(eval_config["min_depth"]), max_depth=max_depth)
                metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=float(eval_config["min_depth"]), max_depth=max_depth)
                if metrics_final is None or metrics_d1 is None or metrics_d0 is None:
                    row["status"] = "skipped_metric_failure"
                    rows.append(row)
                    continue
                rgb_preview = batch["rgb_preview"][sample_idx].permute(1, 2, 0).numpy().astype(np.float32)
                regions = sample_led_region_metrics_three(depth_np=depth_np, valid_np=valid_np, aligned_final=aligned_final, aligned_d1=aligned_d1, aligned_d0=aligned_d0, d0_norm_np=out["D0_norm"][sample_idx].float().detach().cpu().numpy(), d1_norm_np=out["D1_norm"][sample_idx].float().detach().cpu().numpy(), y_norm_np=y_norm[sample_idx].float().detach().cpu().numpy(), rgb_preview_np=rgb_preview, min_depth=float(eval_config["min_depth"]), max_depth=max_depth)
                row.update({
                    "final": {k: float(metrics_final[k]) for k in METRIC_KEYS if k in metrics_final},
                    "D1": {k: float(metrics_d1[k]) for k in METRIC_KEYS if k in metrics_d1},
                    "D0": {k: float(metrics_d0[k]) for k in METRIC_KEYS if k in metrics_d0},
                    "region": regions,
                    "feature_ablation_mode": feature_ablation_mode(args_ns, phase="eval"),
                    "feature_ablation_scope": str(args_ns.feature_ablation_scope),
                    "feature_ablation_applied": bool(feature_ablation_active(args_ns, phase="eval")),
                })
                rows.append(row)
            visited += batch_size
    summary = aggregate_nseries(rows)
    summary.update({
        "dataset_geometry": dataset.describe_geometry(),
        "feature_ablation_mode": feature_ablation_mode(args_ns, phase="eval"),
        "feature_ablation_scope": str(args_ns.feature_ablation_scope),
        "feature_ablation_key": str(args_ns.feature_ablation_key),
        "feature_ablation_donor_offset": int(args_ns.feature_ablation_donor_offset),
        "feature_ablation_applied": bool(feature_ablation_active(args_ns, phase="eval")),
    })
    return summary, rows


def evaluate_d0(config: dict[str, Any], checkpoint: Path, max_depth: float, device: torch.device, max_samples: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = build_rgb_dataset(config, max_depth)
    loader = loader_for(dataset, 1, device)
    model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    model = model.to(device).eval()
    adapter = CenterPadCropAdapter(sensor_hw=(int(config["input_height"]), int(config["input_width"])), backbone_hw=None).to(device)
    sign = int(config.get("recommended_d0_sign", config.get("d0_sign", 1)))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_samples is not None and len(rows) >= max_samples:
                break
            image = batch["image"].to(device).float()
            depth_t = batch["depth"].float()
            valid_t = batch["valid_mask"].bool() & (depth_t >= float(config["min_depth"])) & (depth_t <= max_depth)
            row: dict[str, Any] = {"dataset_index": int(idx), "sample_name": batch["sample_name"][0], "status": "ok"}
            if int(valid_t[0].sum().item()) < 128:
                row["status"] = "skipped_invalid_pixels"
                rows.append(row)
                continue
            d0 = sign * adapter.crop_depth(model(adapter.pad_rgb(image)))[0].detach().cpu().numpy().astype(np.float32)
            depth_np = depth_t[0].numpy().astype(np.float32)
            valid_np = valid_t[0].numpy().astype(bool)
            aligned, _ = affine_align_disp(depth_np, d0, valid_np)
            metrics = compute_metrics(depth_np, aligned, valid_np, min_depth=float(config["min_depth"]), max_depth=max_depth)
            if metrics is None:
                row["status"] = "skipped_metric_failure"
            else:
                row["D0"] = {k: float(metrics[k]) for k in METRIC_KEYS if k in metrics}
            rows.append(row)
    ok = [row for row in rows if row["status"] == "ok" and "D0" in row]
    return {"samples": len(ok), "overall": {"D0": average_dicts([row["D0"] for row in ok], METRIC_KEYS)}, "dataset_geometry": dataset.describe_geometry(), "selected_sign": sign}, rows


def main() -> None:
    args = parse_args()
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise ValueError("--max-val-samples must be positive when provided.")
    logger = make_logger()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable.")
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_kind == "d0":
        summary_path = run_dir / "d0_summary.json"
        config = load_json(summary_path) if summary_path.is_file() else {}
        config.setdefault("encoder", "vits")
        config.setdefault("led_val_list", str(PROJECT_ROOT / "finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt"))
        config.setdefault("input_height", 378)
        config.setdefault("input_width", 672)
        config.setdefault("min_depth", 1.0)
        config.setdefault("max_depth", 200.0)
        config.setdefault("rgb_input_space", "resize_area_756x1344_then_2x2_area")
        config.setdefault("depth_target_space", "resize_nearest_756x1344_then_2x2_valid_mean")
        config.setdefault("depth_label", "distance_to_image_plane")
        config.setdefault("depth_unit", "meter")
        summary, rows = evaluate_d0(config, checkpoint, float(args.max_depth), device, args.max_val_samples)
    else:
        config = load_json(run_dir / "config.json")
        start = time.time()
        if args.run_kind == "c2":
            summary, rows = evaluate_c2(config, checkpoint, float(args.max_depth), device, args.max_val_samples)
        else:
            summary, rows = evaluate_nseries(args, config, checkpoint, float(args.max_depth), device, args.max_val_samples)
        summary["elapsed_seconds"] = float(time.time() - start)

    summary.update({
        "run_kind": args.run_kind,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "max_depth": float(args.max_depth),
        "max_val_samples": args.max_val_samples,
        "eval_protocol": "per_image_affine_disp_depth_anything_v2",
    })
    save_json(output_dir / "metrics.json", summary)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    logger.info("wrote %s and %s", output_dir / "metrics.json", output_dir / "per_sample.jsonl")


if __name__ == "__main__":
    main()
