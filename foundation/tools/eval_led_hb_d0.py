#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from foundation.engine.datasets.led_hb import LEDHBHalfresRGBDepth
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    average_dicts,
    region_abs_rel,
    save_json,
    strip_module_prefix,
    resolve_model_state,
    top_fraction_mask,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

LED_REGION_KEYS = (
    "boundary_abs_rel",
    "d0_high_error_abs_rel",
    "near_1_20_abs_rel",
    "mid_20_50_abs_rel",
    "far50_abs_rel",
    "far100_abs_rel",
    "dark_q20_abs_rel",
    "saturated_abs_rel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen DAv2 D0 baseline on LED-HB.")
    parser.add_argument("--encoder", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--led-val-list", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def sample_led_d0_region_metrics(
    *,
    depth_np: np.ndarray,
    valid_np: np.ndarray,
    aligned_d0: np.ndarray,
    d0_norm_np: np.ndarray,
    y_norm_np: np.ndarray,
    rgb_preview_np: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> dict[str, float]:
    grad_y, grad_x = np.gradient(depth_np.astype(np.float32))
    boundary_score = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    luma = 0.2126 * rgb_preview_np[..., 0] + 0.7152 * rgb_preview_np[..., 1] + 0.0722 * rgb_preview_np[..., 2]
    valid_luma = luma[valid_np & np.isfinite(luma)]
    dark_threshold = float(np.quantile(valid_luma, 0.20)) if valid_luma.size else 0.0
    masks = {
        "boundary_abs_rel": top_fraction_mask(boundary_score, valid_np, 0.10),
        "d0_high_error_abs_rel": top_fraction_mask(np.abs(d0_norm_np - y_norm_np), valid_np, 0.20),
        "near_1_20_abs_rel": valid_np & (depth_np >= 1.0) & (depth_np <= 20.0),
        "mid_20_50_abs_rel": valid_np & (depth_np > 20.0) & (depth_np <= 50.0),
        "far50_abs_rel": valid_np & (depth_np > 50.0),
        "far100_abs_rel": valid_np & (depth_np > 100.0),
        "dark_q20_abs_rel": valid_np & (luma <= dark_threshold),
        "saturated_abs_rel": valid_np & (np.max(rgb_preview_np, axis=-1) > 0.95),
    }
    return {
        key: region_abs_rel(depth_np, aligned_d0, mask, min_depth=min_depth, max_depth=max_depth)
        for key, mask in masks.items()
    }


def evaluate_signs(
    model: torch.nn.Module,
    adapter: CenterPadCropAdapter,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    metrics_by_sign = {"+1": [], "-1": []}
    regions_by_sign = {"+1": [], "-1": []}
    processed = 0
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            if args.max_val_samples is not None and processed >= int(args.max_val_samples):
                break
            image = batch["image"].to(device, non_blocking=True).float()
            depth_t = batch["depth"].float()
            valid_t = batch["valid_mask"].bool() & (depth_t >= args.min_depth) & (depth_t <= args.max_depth)
            if int(valid_t[0].sum().item()) < 128:
                continue
            d0_raw = model(adapter.pad_rgb(image))
            d0_raw = adapter.crop_depth(d0_raw)[0].detach().cpu().numpy().astype(np.float32)

            depth_np = depth_t[0].numpy().astype(np.float32)
            valid_np = valid_t[0].numpy().astype(bool)
            inv_gt = build_training_target(depth_t, valid_t, target_space="metric_depth")
            y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_t, min_valid_pixels=128)
            y_norm_np = y_norm[0].numpy().astype(np.float32)
            rgb_preview_np = batch["rgb_preview"][0].permute(1, 2, 0).numpy().astype(np.float32)
            for sign in (1, -1):
                key = f"{sign:+d}"
                signed = float(sign) * d0_raw
                aligned, _ = affine_align_disp(depth_np, signed, valid_np)
                metrics = compute_metrics(depth_np, aligned, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
                if metrics is None:
                    continue
                metrics_by_sign[key].append({name: float(metrics[name]) for name in METRIC_KEYS if name in metrics})
                regions_by_sign[key].append(
                    sample_led_d0_region_metrics(
                        depth_np=depth_np,
                        valid_np=valid_np,
                        aligned_d0=aligned,
                        d0_norm_np=signed,
                        y_norm_np=y_norm_np,
                        rgb_preview_np=rgb_preview_np,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                    )
                )
            processed += 1
    if processed == 0:
        raise RuntimeError("LED-HB D0 eval produced zero valid samples.")
    metrics_mean = {key: average_dicts(rows, METRIC_KEYS) for key, rows in metrics_by_sign.items()}
    regions_mean = {key: average_dicts(rows, LED_REGION_KEYS) for key, rows in regions_by_sign.items()}
    plus = metrics_mean["+1"]["abs_rel"]
    minus = metrics_mean["-1"]["abs_rel"]
    if plus is None and minus is None:
        raise RuntimeError("No finite D0 AbsRel for either sign.")
    recommended = 1 if minus is None or (plus is not None and float(plus) <= float(minus)) else -1
    selected_key = f"{recommended:+d}"
    return {
        "processed_samples": int(processed),
        "elapsed_seconds": float(time.time() - start),
        "recommended_d0_sign": int(recommended),
        "alignment_protocol": "per_image_affine_disp_depth_anything_v2",
        "metrics_by_sign": metrics_mean,
        "region_by_sign": regions_mean,
        "selected": {
            "sign": int(recommended),
            "overall": metrics_mean[selected_key],
            "region": regions_mean[selected_key],
        },
    }


def main() -> None:
    args = parse_args()
    if (int(args.input_height), int(args.input_width)) != (378, 672):
        raise ValueError(f"LED-HB D0 eval expects input size (378, 672), got {(args.input_height, args.input_width)}")
    if not (0.0 < float(args.min_depth) < float(args.max_depth)):
        raise ValueError(f"Expected 0 < min_depth < max_depth, got {args.min_depth}, {args.max_depth}")
    if args.max_val_samples is not None and int(args.max_val_samples) <= 0:
        raise ValueError("--max-val-samples must be positive when provided.")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LEDHBHalfresRGBDepth(
        filelist_path=args.led_val_list,
        mode="val",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        hflip_prob=0.0,
        include_geometry=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    ckpt_obj = torch.load(args.pretrained_from, map_location="cpu")
    status = model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    if status.missing_keys or status.unexpected_keys:
        raise RuntimeError(f"Unexpected DAv2 load status: {status}")
    model = model.to(device).eval()
    adapter = CenterPadCropAdapter(sensor_hw=(args.input_height, args.input_width), backbone_hw=None).to(device)
    result = evaluate_signs(model, adapter, loader, args, device)
    payload = {
        "dataset_name": "led_hb",
        "illumination": "HB",
        "encoder": args.encoder,
        "pretrained_from": str(Path(args.pretrained_from).expanduser().resolve()),
        "led_val_list": str(Path(args.led_val_list).expanduser().resolve()),
        "input_height": int(args.input_height),
        "input_width": int(args.input_width),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "max_val_samples": args.max_val_samples,
        "dataset_geometry": dataset.describe_geometry(),
        **result,
    }
    save_json(output_dir / "d0_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
