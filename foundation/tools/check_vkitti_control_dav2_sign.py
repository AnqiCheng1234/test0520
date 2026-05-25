from __future__ import annotations

import argparse
import json
import math
import sys
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
from foundation.engine.datasets.vkitti2_halfres_rgb_depth import VKITTI2HalfresRGBDepth


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
QUALITY_KEYS = ("abs_rel", "d1", "silog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DAv2 sign on VKITTI2HalfresRGBDepth control data.")
    parser.add_argument("--encoder", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--vkitti-val-list", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--fullres-even-policy", required=True, choices=["crop_bottom_to_even"])
    parser.add_argument("--rgb-input-space", required=True, choices=["halfres_2x2_area"])
    parser.add_argument("--depth-target-space", required=True, choices=["halfres_2x2_valid_mean"])
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (args.input_height, args.input_width) != (187, 621):
        raise ValueError(f"Control sign check expects input size (187, 621), got {(args.input_height, args.input_width)}")
    if not (0.0 < args.min_depth < args.max_depth):
        raise ValueError(f"Expected 0 < min_depth < max_depth, got {args.min_depth}, {args.max_depth}")
    if args.max_samples <= 0:
        raise ValueError(f"--max-samples must be positive, got {args.max_samples}")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if x.size < 10:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (x_std * y_std))


def compute_d0_quality(depth: np.ndarray, valid: np.ndarray, d0: np.ndarray, args: argparse.Namespace) -> dict[str, float] | None:
    if int(valid.sum()) < 128:
        return None
    aligned, _ = affine_align_disp(depth, d0, valid)
    metrics = compute_metrics(depth, aligned, valid, min_depth=args.min_depth, max_depth=args.max_depth)
    if metrics is None:
        return None
    return {key: float(metrics[key]) for key in QUALITY_KEYS}


def mean_quality(rows: list[dict[str, float]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    if not values:
        return None
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This sign check expects CUDA.")

    dataset = VKITTI2HalfresRGBDepth(
        filelist_path=args.vkitti_val_list,
        mode="val",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        hflip_prob=0.0,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    ckpt_obj = torch.load(args.pretrained_from, map_location="cpu")
    state = strip_module_prefix(resolve_model_state(ckpt_obj))
    status = model.load_state_dict(state, strict=True)
    if status.missing_keys or status.unexpected_keys:
        raise RuntimeError(f"Unexpected DAv2 load status: {status}")
    model = model.to(device).eval()
    adapter = CenterPadCropAdapter(sensor_hw=(args.input_height, args.input_width), backbone_hw=None).to(device)

    sample_corrs: list[float] = []
    halfres_quality: list[dict[str, float]] = []
    global_x: list[np.ndarray] = []
    global_y: list[np.ndarray] = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            if processed >= args.max_samples:
                break
            image = batch["image"].to(device, non_blocking=True).float()
            depth = batch["depth"][0].numpy().astype(np.float32)
            valid = batch["valid_mask"][0].numpy().astype(bool)
            valid = valid & np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
            if int(valid.sum()) < 128:
                continue
            d0 = model(adapter.pad_rgb(image))
            d0 = adapter.crop_depth(d0)[0].detach().cpu().numpy().astype(np.float32)
            inv = np.zeros_like(depth, dtype=np.float32)
            inv[valid] = 1.0 / np.clip(depth[valid], 1e-6, None)
            corr = corrcoef(d0[valid], inv[valid])
            if math.isfinite(corr):
                half_metrics = compute_d0_quality(depth, valid, d0, args)
                if half_metrics is not None:
                    halfres_quality.append(half_metrics)
                sample_corrs.append(corr)
                global_x.append(d0[valid])
                global_y.append(inv[valid])
                processed += 1

    if not sample_corrs:
        raise RuntimeError("No valid samples for sign check.")

    d0_all = np.concatenate(global_x)
    inv_all = np.concatenate(global_y)
    global_corr = corrcoef(d0_all, inv_all)
    mean_corr = float(np.mean(sample_corrs))
    recommended_sign = 1 if mean_corr >= 0.0 else -1
    halfres_quality_mean = {key: mean_quality(halfres_quality, key) for key in QUALITY_KEYS}
    halfres_tag = f"halfres_{int(args.input_height)}x{int(args.input_width)}"
    payload = {
        "encoder": args.encoder,
        "pretrained_from": args.pretrained_from,
        "vkitti_val_list": args.vkitti_val_list,
        "input_height": int(args.input_height),
        "input_width": int(args.input_width),
        "raw_storage_format": "not_applicable",
        "fullres_even_policy": args.fullres_even_policy,
        "rgb_input_space": args.rgb_input_space,
        "depth_target_space": args.depth_target_space,
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "max_samples": int(args.max_samples),
        "processed_samples": int(processed),
        "mean_sample_corr": mean_corr,
        "median_sample_corr": float(np.median(sample_corrs)),
        "global_pixel_corr": float(global_corr),
        "recommended_d0_sign": int(recommended_sign),
        "alignment_protocol": "per_image_affine_disp_depth_anything_v2",
        "dataset_geometry": dataset.describe_geometry(),
        "quality": {halfres_tag: halfres_quality_mean},
    }
    for key in QUALITY_KEYS:
        payload[f"{halfres_tag}_D0_{key}_mean_over_{int(args.max_samples)}"] = halfres_quality_mean[key]

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
