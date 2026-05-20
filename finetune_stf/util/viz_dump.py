from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from finetune_stf.util.loss import (
    SigLoss,
    _grad_matching_from_aligned,
    _ssi_mse_from_aligned,
    align_prediction_to_inverse_gt,
    build_training_target,
    robust_normalize_target_per_sample,
)
from finetune_stf.models.raw_ram import phase1b_tanh_tail_squash
from foundation.engine.transforms import packed_bayer_to_base_rgb


_FIXED_SAMPLE_LOADERS = (
    ("kitti", "kitti_val_loader"),
    ("eth3d", "eth3d_val_fast_loader"),
    ("robotcar", "robotcar_val_fast_loader"),
    ("robotcar_night", "robotcar_night_val_fast_loader"),
)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _clone_sample_to_cpu(sample):
    cloned = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, tuple):
            cloned[key] = list(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def collect_fixed_samples(train_state, n_per_split=8):
    fixed_samples = {}
    for split_name, loader_key in _FIXED_SAMPLE_LOADERS:
        loader = train_state.get(loader_key)
        if loader is None:
            continue

        samples = []
        for batch in loader:
            samples.append(_clone_sample_to_cpu(batch))
            if len(samples) >= int(n_per_split):
                break
        fixed_samples[split_name] = samples
    return fixed_samples


def _first_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    if torch.is_tensor(value) and value.ndim == 0:
        return value.item()
    return value


def _sample_stem(sample, fallback):
    for key in ("sample_stem", "sample_name", "image_path", "depth_path"):
        value = _first_value(sample.get(key))
        if value:
            stem = Path(str(value)).stem if key.endswith("_path") else str(value)
            stem = stem.replace(os.sep, "__").replace("/", "__")
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
            if stem:
                return stem
    return f"sample_{fallback:02d}"


def _select_model_input(sample, input_type):
    if str(input_type) != "rgb" and "raw" in sample:
        tensor = sample["raw"]
    else:
        tensor = sample["image"]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _colorize_disp(disp):
    disp = np.asarray(disp, dtype=np.float32)
    finite = np.isfinite(disp)
    if not np.any(finite):
        norm = np.zeros(disp.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(disp[finite], [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(disp[finite]))
            hi = float(np.max(disp[finite]))
        if hi <= lo:
            norm = np.zeros(disp.shape, dtype=np.uint8)
        else:
            clipped = np.clip(disp, lo, hi)
            clipped = np.where(np.isfinite(clipped), clipped, lo)
            norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    cmap = matplotlib.colormaps.get_cmap("Spectral")
    return (cmap(norm)[:, :, :3] * 255.0)[:, :, ::-1].astype(np.uint8)


def _colorize_depth(depth, valid_mask, *, vmin, vmax):
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    denom = max(float(vmax) - float(vmin), 1e-6)
    values = np.clip(depth, float(vmin), float(vmax))
    norm = (values - float(vmin)) / denom
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    bgr = (cmap(norm)[:, :, :3] * 255.0)[:, :, ::-1].astype(np.uint8)
    bgr[~valid_mask] = 0
    return bgr


def _tensor_to_first_sample(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    tensor = tensor.detach()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _as_2d_numpy(value, *, dtype=np.float32):
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr.astype(dtype, copy=False)


def _rgb_tensor_to_bgr_preview(x_rgb):
    if x_rgb is None or not torch.is_tensor(x_rgb):
        return None
    x_rgb = _tensor_to_first_sample(x_rgb.detach().float().cpu())
    if not torch.is_tensor(x_rgb) or x_rgb.ndim != 3 or x_rgb.shape[0] != 3:
        return None
    preview = x_rgb.numpy().transpose(1, 2, 0)
    preview = np.clip(preview, 0.0, 1.0) ** (1.0 / 2.2)
    rgb = (preview * 255.0).round().astype(np.uint8)
    return rgb[:, :, ::-1]


def _infer_ram_rgb(active_model, x_raw):
    if x_raw.ndim != 4 or x_raw.shape[1] != 4:
        return None

    module = _module(active_model)
    if hasattr(module, "image_bridge") and hasattr(module, "ram_core"):
        x4, _ = module.ram_core.forward_with_features(x_raw)
        return module.image_bridge(x4, x_raw=x_raw)
    if hasattr(module, "rgb_head") and hasattr(module, "ram_core"):
        x4 = module.ram_core(x_raw)
        return module.rgb_head(x4, x_raw=x_raw)
    if hasattr(module, "residual_head") and hasattr(module, "ram_core"):
        x4 = module.ram_core(x_raw)
        delta_rgb = module.residual_head(x4)
        base_rgb = packed_bayer_to_base_rgb(x_raw)
        residual_scale = float(getattr(module, "residual_scale", 0.1))
        return torch.clamp(base_rgb + residual_scale * torch.tanh(delta_rgb), min=0.0, max=1.0)
    if hasattr(module, "ram_core"):
        in_channels = _ram_core_input_channels(module.ram_core)
        if in_channels == 3:
            x3_in = packed_bayer_to_base_rgb(x_raw)
            return module.ram_core(x3_in)
    if hasattr(module, "input_stem"):
        x_rgb = module.input_stem(x_raw)
        if bool(getattr(module, "clip_rgb", True)):
            x_rgb = torch.clamp(x_rgb, min=0.0, max=1.0)
        return x_rgb
    return None


def _resize_2d_bilinear(values, target_hw):
    values = np.asarray(values, dtype=np.float32)
    if tuple(values.shape) == tuple(target_hw):
        return values
    tensor = torch.from_numpy(values)[None, None]
    resized = F.interpolate(tensor, size=tuple(target_hw), mode="bilinear", align_corners=True)
    return resized[0, 0].numpy()


def _sample_bilinear_disparity_at_mask_np(pred_disp, valid_mask, full_hw):
    pred_disp = np.asarray(pred_disp, dtype=np.float64)
    coords = np.argwhere(np.asarray(valid_mask, dtype=bool))
    if coords.size == 0:
        return coords, np.zeros((0,), dtype=np.float64)

    src_h, src_w = pred_disp.shape
    full_h, full_w = int(full_hw[0]), int(full_hw[1])
    ys = coords[:, 0].astype(np.float64)
    xs = coords[:, 1].astype(np.float64)
    if full_h > 1:
        ys *= float(src_h - 1) / float(full_h - 1)
    else:
        ys.fill(0.0)
    if full_w > 1:
        xs *= float(src_w - 1) / float(full_w - 1)
    else:
        xs.fill(0.0)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = ys - y0.astype(np.float64)
    wx = xs - x0.astype(np.float64)

    samples = (
        pred_disp[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + pred_disp[y0, x1] * (1.0 - wy) * wx
        + pred_disp[y1, x0] * wy * (1.0 - wx)
        + pred_disp[y1, x1] * wy * wx
    )
    return coords, samples


def _affine_align_disp_1d(gt_depth, pred_disp):
    gt_depth = np.asarray(gt_depth, dtype=np.float64).reshape(-1)
    pred_disp = np.asarray(pred_disp, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(pred_disp)
    if int(valid.sum()) < 2:
        aligned_depth = np.full(pred_disp.shape, np.nan, dtype=np.float64)
        return aligned_depth, {"scale": 0.0, "shift": 0.0, "invalid_aligned_pixels": int(valid.size), "invalid_aligned_ratio": 1.0}

    gt_disp = 1.0 / np.clip(gt_depth[valid], a_min=1e-9, a_max=None)
    x = pred_disp[valid]
    design = np.stack([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(design, gt_disp, rcond=None)
    scale, shift = float(coef[0]), float(coef[1])
    aligned_depth = _apply_disp_alignment(pred_disp, scale, shift)
    pos = np.isfinite(pred_disp * scale + shift) & ((pred_disp * scale + shift) > 0)
    invalid_count = int(valid.sum() - np.count_nonzero(valid & pos))
    return aligned_depth, {
        "scale": scale,
        "shift": shift,
        "invalid_aligned_pixels": invalid_count,
        "invalid_aligned_ratio": float(invalid_count / max(int(valid.sum()), 1)),
    }


def _affine_align_disp(gt_depth, pred_disp, valid_mask):
    gt_depth = np.asarray(gt_depth, dtype=np.float64)
    pred_disp = np.asarray(pred_disp, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(pred_disp)
    if int(valid.sum()) < 2:
        aligned_depth = np.full(pred_disp.shape, np.nan, dtype=np.float64)
        return aligned_depth, {"scale": 0.0, "shift": 0.0, "invalid_aligned_pixels": int(valid.size), "invalid_aligned_ratio": 1.0}

    gt_disp = 1.0 / np.clip(gt_depth[valid], a_min=1e-9, a_max=None)
    x = pred_disp[valid]
    design = np.stack([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(design, gt_disp, rcond=None)
    scale, shift = float(coef[0]), float(coef[1])
    aligned_depth = _apply_disp_alignment(pred_disp, scale, shift)
    pos = np.isfinite(pred_disp * scale + shift) & ((pred_disp * scale + shift) > 0)
    invalid_count = int(valid.sum() - np.count_nonzero(valid & pos))
    return aligned_depth, {
        "scale": scale,
        "shift": shift,
        "invalid_aligned_pixels": invalid_count,
        "invalid_aligned_ratio": float(invalid_count / max(int(valid.sum()), 1)),
    }


def _apply_disp_alignment(pred_disp, scale, shift):
    aligned_disp = np.asarray(pred_disp, dtype=np.float64) * float(scale) + float(shift)
    aligned_depth = np.full(aligned_disp.shape, np.nan, dtype=np.float64)
    pos = np.isfinite(aligned_disp) & (aligned_disp > 0.0)
    aligned_depth[pos] = 1.0 / aligned_disp[pos]
    return aligned_depth


def _fixed_eval_depth_bounds(args, split_name):
    split_name = str(split_name)
    if split_name == "kitti":
        return (
            float(getattr(args, "kitti_min_depth", getattr(args, "min_depth", 0.1))),
            float(getattr(args, "kitti_max_depth", getattr(args, "max_depth", 80.0))),
        )
    if split_name == "eth3d":
        return (
            float(getattr(args, "eth3d_min_depth", getattr(args, "min_depth", 0.1))),
            float(getattr(args, "eth3d_max_depth", getattr(args, "max_depth", 80.0))),
        )
    if split_name == "robotcar_night":
        return (
            float(getattr(args, "robotcar_night_min_depth", getattr(args, "min_depth", 0.1))),
            float(getattr(args, "robotcar_night_max_depth", getattr(args, "max_depth", 80.0))),
        )
    if split_name == "robotcar":
        return (
            float(getattr(args, "robotcar_min_depth", getattr(args, "min_depth", 0.1))),
            float(getattr(args, "robotcar_max_depth", getattr(args, "max_depth", 80.0))),
        )
    return (
        float(getattr(args, "min_depth", 0.1)),
        float(getattr(args, "max_depth", 80.0)),
    )


def _compute_depth_metrics(depth, aligned_depth, valid_mask, *, min_depth, max_depth):
    eval_depth = np.asarray(aligned_depth, dtype=np.float64).copy()
    finite = np.isfinite(eval_depth)
    eval_depth[finite] = np.clip(eval_depth[finite], float(min_depth), float(max_depth))
    gt = np.asarray(depth, dtype=np.float64)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(eval_depth)
        & (eval_depth > 0)
        & np.isfinite(gt)
        & (gt > 0)
    )
    if int(valid.sum()) < 10:
        return None
    diff = eval_depth[valid] - gt[valid]
    diff_log = np.log(eval_depth[valid]) - np.log(gt[valid])
    thresh = np.maximum(gt[valid] / eval_depth[valid], eval_depth[valid] / gt[valid])
    return {
        "abs_rel": float(np.mean(np.abs(diff) / gt[valid])),
        "sq_rel": float(np.mean(diff ** 2 / gt[valid])),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(np.mean(diff_log ** 2))),
        "silog": float(np.sqrt(max(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2, 0.0))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25 ** 2)),
        "d3": float(np.mean(thresh < 1.25 ** 3)),
        "valid_eval_pixels": int(valid.sum()),
    }


def _fixed_eval_metrics(sample, pred_disp, args, split_name):
    depth = _as_2d_numpy(sample.get("depth"), dtype=np.float32)
    valid_mask = _as_2d_numpy(sample.get("valid_mask"), dtype=bool)
    pred = _as_2d_numpy(pred_disp, dtype=np.float32)
    if depth is None or valid_mask is None or pred is None:
        return None

    min_depth, max_depth = _fixed_eval_depth_bounds(args, split_name)
    valid_mask = (
        valid_mask.astype(bool, copy=False)
        & np.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )
    if int(valid_mask.sum()) < 10:
        return None

    depth_mode = str(_first_value(sample.get("depth_mode"), "full"))
    fast_eval_backend = str(_first_value(sample.get("fast_eval_backend"), "proxy"))
    use_sparse_fast_eval = depth_mode == "fast" and fast_eval_backend == "sparse"
    if use_sparse_fast_eval:
        coords, pred_samples = _sample_bilinear_disparity_at_mask_np(pred, valid_mask, depth.shape[-2:])
        depth_samples = depth[valid_mask]
        aligned_samples, align_stats = _affine_align_disp_1d(depth_samples, pred_samples)
        metric_aligned_depth = np.full(depth.shape, np.nan, dtype=np.float64)
        if coords.size:
            metric_aligned_depth[coords[:, 0], coords[:, 1]] = aligned_samples
        panel_aligned_depth = _apply_disp_alignment(pred, align_stats["scale"], align_stats["shift"])
    else:
        pred_dense = _resize_2d_bilinear(pred, depth.shape[-2:])
        metric_aligned_depth, align_stats = _affine_align_disp(depth, pred_dense, valid_mask)
        panel_aligned_depth = metric_aligned_depth

    metrics = _compute_depth_metrics(
        depth,
        metric_aligned_depth,
        valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    if metrics is None:
        return None

    gt_valid = depth[valid_mask]
    if gt_valid.size:
        vmin = max(min_depth, float(np.nanmin(gt_valid)))
        vmax = min(max_depth, float(np.nanmax(gt_valid)))
    else:
        vmin, vmax = min_depth, max_depth
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = min_depth, max_depth

    return {
        "metrics": metrics,
        "align": align_stats,
        "panel_aligned_depth": panel_aligned_depth,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "depth_mode": depth_mode,
        "fast_eval_backend": fast_eval_backend,
        "sparse_fast_eval": bool(use_sparse_fast_eval),
    }


def _json_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _write_fixed_panel_extras(
    *,
    active_model,
    rgb_baseline_model,
    rgb_baseline_label,
    tensor,
    pred_disp,
    disp_np,
    sample,
    args,
    split_name,
    stem,
    split_dir,
    sample_index,
    disp_path,
    color_path,
    input_type,
    amp_dtype,
    use_amp,
):
    show_rgb_baseline = rgb_baseline_model is not None
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and tensor.is_cuda):
        ram_rgb = _infer_ram_rgb(active_model, tensor) if tensor.ndim == 4 and tensor.shape[1] == 4 else None
        ram_rgb_pre_norm = (
            _infer_front_rgb_pre_norm(active_model, tensor)
            if tensor.ndim == 4 and tensor.shape[1] == 4
            else None
        )
    ram_rgb_bgr = _rgb_tensor_to_bgr_preview(ram_rgb)

    metrics_payload = _fixed_eval_metrics(sample, pred_disp, args, split_name)
    ram_rgb_path = split_dir / f"{stem}_ram_rgb.png"
    panel_path = split_dir / f"{stem}_panel.png"
    metrics_path = split_dir / f"{stem}_metrics.json"

    extra = {}
    if ram_rgb_bgr is not None:
        cv2.imwrite(str(ram_rgb_path), ram_rgb_bgr)
        extra["ram_rgb"] = str(ram_rgb_path)

    metrics_json = {
        "split": split_name,
        "sample_name": str(_first_value(sample.get("sample_name"), stem)),
        "sample_stem": stem,
        "disp": str(disp_path),
        "color": str(color_path),
        "panel": str(panel_path),
    }
    if ram_rgb_bgr is not None:
        metrics_json["ram_rgb"] = str(ram_rgb_path)

    if metrics_payload is not None:
        aligned_depth = metrics_payload["panel_aligned_depth"]
        depth_mask = np.isfinite(aligned_depth) & (aligned_depth > 0)
        depth_bgr = _colorize_depth(
            aligned_depth,
            depth_mask,
            vmin=metrics_payload["vmin"],
            vmax=metrics_payload["vmax"],
        )
        depth_color_path = split_dir / f"{stem}_depth_color.png"
        depth_npz_path = split_dir / f"{stem}_aligned_depth.npz"
        cv2.imwrite(str(depth_color_path), depth_bgr)
        np.savez_compressed(
            depth_npz_path,
            aligned_depth=aligned_depth.astype(np.float32, copy=False),
            abs_rel=np.float32(metrics_payload["metrics"]["abs_rel"]),
            rmse=np.float32(metrics_payload["metrics"]["rmse"]),
            d1=np.float32(metrics_payload["metrics"]["d1"]),
            align_scale=np.float32(metrics_payload["align"]["scale"]),
            align_shift=np.float32(metrics_payload["align"]["shift"]),
        )
        abs_rel = metrics_payload["metrics"]["abs_rel"]
        extra.update(
            {
                "depth_color": str(depth_color_path),
                "aligned_depth": str(depth_npz_path),
                "abs_rel": float(abs_rel),
                "rmse": float(metrics_payload["metrics"]["rmse"]),
                "d1": float(metrics_payload["metrics"]["d1"]),
            }
        )
        metrics_json.update(
            {
                "abs_rel": _json_float(abs_rel),
                "sq_rel": _json_float(metrics_payload["metrics"]["sq_rel"]),
                "rmse": _json_float(metrics_payload["metrics"]["rmse"]),
                "rmse_log": _json_float(metrics_payload["metrics"]["rmse_log"]),
                "silog": _json_float(metrics_payload["metrics"]["silog"]),
                "d1": _json_float(metrics_payload["metrics"]["d1"]),
                "d2": _json_float(metrics_payload["metrics"]["d2"]),
                "d3": _json_float(metrics_payload["metrics"]["d3"]),
                "valid_eval_pixels": int(metrics_payload["metrics"]["valid_eval_pixels"]),
                "align_scale": _json_float(metrics_payload["align"]["scale"]),
                "align_shift": _json_float(metrics_payload["align"]["shift"]),
                "invalid_aligned_ratio": _json_float(metrics_payload["align"]["invalid_aligned_ratio"]),
                "depth_color": str(depth_color_path),
                "aligned_depth": str(depth_npz_path),
                "vmin": _json_float(metrics_payload["vmin"]),
                "vmax": _json_float(metrics_payload["vmax"]),
                "min_depth": _json_float(metrics_payload["min_depth"]),
                "max_depth": _json_float(metrics_payload["max_depth"]),
                "depth_mode": metrics_payload["depth_mode"],
                "fast_eval_backend": metrics_payload["fast_eval_backend"],
                "sparse_fast_eval": bool(metrics_payload["sparse_fast_eval"]),
            }
        )
        target_view = _masked_depth_view(sample)
        current_view = aligned_depth.astype(np.float32, copy=False)
        current_title = "Current aligned"
        current_subtitle = f"abs_rel {_format_panel_metric(abs_rel)}"
    else:
        abs_rel = None
        target_view = _masked_depth_view(sample)
        current_view = np.asarray(disp_np, dtype=np.float32)
        current_title = "Pred disparity"
        current_subtitle = "abs_rel n/a"
        metrics_json.update(
            {
                "abs_rel": None,
                "rmse": None,
                "note": "abs_rel unavailable; missing depth/valid mask or fewer than 10 valid eval pixels",
            }
        )

    rgb_baseline_payload = None
    rgb_baseline_view = None
    rgb_baseline_title = str(rgb_baseline_label or "RGB DAv2")
    rgb_baseline_subtitle = "baseline n/a"
    if show_rgb_baseline:
        baseline_tensor = _fixed_rgb_baseline_input_from_sample(sample)
        if baseline_tensor is None:
            metrics_json["rgb_baseline"] = {
                "available": False,
                "note": "missing RGB path/tensor for fixed-viz RGB DAv2 baseline",
            }
        else:
            if baseline_tensor.ndim == 3:
                baseline_tensor = baseline_tensor.unsqueeze(0)
            baseline_device = next(_module(rgb_baseline_model).parameters()).device
            baseline_tensor = baseline_tensor.to(device=baseline_device, non_blocking=True).float()
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp and baseline_tensor.is_cuda,
            ):
                rgb_baseline_pred = rgb_baseline_model(baseline_tensor)
            rgb_baseline_disp = rgb_baseline_pred.detach().float()
            if rgb_baseline_disp.ndim == 4 and rgb_baseline_disp.shape[1] == 1:
                rgb_baseline_disp = rgb_baseline_disp[:, 0]
            rgb_baseline_disp_np = _as_2d_numpy(rgb_baseline_disp, dtype=np.float32)
            baseline_info = {
                "available": True,
                "label": rgb_baseline_title,
            }
            if rgb_baseline_disp_np is not None:
                rgb_baseline_disp_path = split_dir / f"{stem}_rgb_dav2_disp.npz"
                rgb_baseline_color_path = split_dir / f"{stem}_rgb_dav2_color.png"
                np.savez_compressed(
                    rgb_baseline_disp_path,
                    disp=rgb_baseline_disp_np.astype(np.float32, copy=False),
                )
                cv2.imwrite(str(rgb_baseline_color_path), _colorize_disp(rgb_baseline_disp_np))
                baseline_info.update(
                    {
                        "disp": str(rgb_baseline_disp_path),
                        "color": str(rgb_baseline_color_path),
                    }
                )
                extra["rgb_baseline_disp"] = str(rgb_baseline_disp_path)
                extra["rgb_baseline_color"] = str(rgb_baseline_color_path)

            rgb_baseline_payload = _fixed_eval_metrics(sample, rgb_baseline_disp, args, split_name)
            if rgb_baseline_payload is not None:
                rgb_baseline_aligned_depth = rgb_baseline_payload["panel_aligned_depth"]
                rgb_baseline_depth_mask = np.isfinite(rgb_baseline_aligned_depth) & (rgb_baseline_aligned_depth > 0)
                rgb_baseline_depth_bgr = _colorize_depth(
                    rgb_baseline_aligned_depth,
                    rgb_baseline_depth_mask,
                    vmin=rgb_baseline_payload["vmin"],
                    vmax=rgb_baseline_payload["vmax"],
                )
                rgb_baseline_depth_color_path = split_dir / f"{stem}_rgb_dav2_depth_color.png"
                rgb_baseline_depth_npz_path = split_dir / f"{stem}_rgb_dav2_aligned_depth.npz"
                cv2.imwrite(str(rgb_baseline_depth_color_path), rgb_baseline_depth_bgr)
                np.savez_compressed(
                    rgb_baseline_depth_npz_path,
                    aligned_depth=rgb_baseline_aligned_depth.astype(np.float32, copy=False),
                    abs_rel=np.float32(rgb_baseline_payload["metrics"]["abs_rel"]),
                    rmse=np.float32(rgb_baseline_payload["metrics"]["rmse"]),
                    d1=np.float32(rgb_baseline_payload["metrics"]["d1"]),
                    align_scale=np.float32(rgb_baseline_payload["align"]["scale"]),
                    align_shift=np.float32(rgb_baseline_payload["align"]["shift"]),
                )
                rgb_baseline_abs_rel = rgb_baseline_payload["metrics"]["abs_rel"]
                rgb_baseline_view = rgb_baseline_aligned_depth.astype(np.float32, copy=False)
                rgb_baseline_subtitle = f"abs_rel {_format_panel_metric(rgb_baseline_abs_rel)}"
                baseline_info.update(
                    {
                        "abs_rel": _json_float(rgb_baseline_abs_rel),
                        "sq_rel": _json_float(rgb_baseline_payload["metrics"]["sq_rel"]),
                        "rmse": _json_float(rgb_baseline_payload["metrics"]["rmse"]),
                        "rmse_log": _json_float(rgb_baseline_payload["metrics"]["rmse_log"]),
                        "silog": _json_float(rgb_baseline_payload["metrics"]["silog"]),
                        "d1": _json_float(rgb_baseline_payload["metrics"]["d1"]),
                        "d2": _json_float(rgb_baseline_payload["metrics"]["d2"]),
                        "d3": _json_float(rgb_baseline_payload["metrics"]["d3"]),
                        "valid_eval_pixels": int(rgb_baseline_payload["metrics"]["valid_eval_pixels"]),
                        "align_scale": _json_float(rgb_baseline_payload["align"]["scale"]),
                        "align_shift": _json_float(rgb_baseline_payload["align"]["shift"]),
                        "invalid_aligned_ratio": _json_float(rgb_baseline_payload["align"]["invalid_aligned_ratio"]),
                        "depth_color": str(rgb_baseline_depth_color_path),
                        "aligned_depth": str(rgb_baseline_depth_npz_path),
                        "vmin": _json_float(rgb_baseline_payload["vmin"]),
                        "vmax": _json_float(rgb_baseline_payload["vmax"]),
                    }
                )
                extra.update(
                    {
                        "rgb_baseline_depth_color": str(rgb_baseline_depth_color_path),
                        "rgb_baseline_aligned_depth": str(rgb_baseline_depth_npz_path),
                        "rgb_baseline_abs_rel": float(rgb_baseline_abs_rel),
                        "rgb_baseline_rmse": float(rgb_baseline_payload["metrics"]["rmse"]),
                        "rgb_baseline_d1": float(rgb_baseline_payload["metrics"]["d1"]),
                    }
                )
            elif rgb_baseline_disp_np is not None:
                rgb_baseline_view = rgb_baseline_disp_np.astype(np.float32, copy=False)
                rgb_baseline_title = f"{rgb_baseline_title} disp"
            metrics_json["rgb_baseline"] = baseline_info

    panel = _make_fixed_viz_panel(
        sample=sample,
        split_name=split_name,
        sample_index=sample_index,
        input_type=input_type,
        target_view=target_view,
        current_view=current_view,
        current_title=current_title,
        current_subtitle=current_subtitle,
        metrics_payload=metrics_payload,
        ram_rgb_pre_norm=ram_rgb_pre_norm[0] if torch.is_tensor(ram_rgb_pre_norm) and ram_rgb_pre_norm.ndim == 4 else ram_rgb_pre_norm,
        show_rgb_baseline=show_rgb_baseline,
        rgb_baseline_view=rgb_baseline_view,
        rgb_baseline_title=rgb_baseline_title,
        rgb_baseline_subtitle=rgb_baseline_subtitle,
    )
    cv2.imwrite(str(panel_path), panel[:, :, ::-1])
    extra["panel"] = str(panel_path)
    metrics_json["panel"] = str(panel_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2, sort_keys=True)
        handle.write("\n")
    extra["metrics"] = str(metrics_path)
    return extra


def _epoch_dir_name(epoch):
    if isinstance(epoch, int):
        return f"epoch_{epoch:02d}"
    return f"epoch_{epoch}"


def _module(model):
    return model.module if hasattr(model, "module") else model


_TRAIN_SOURCE_DATASET_KEYS = {
    "vkitti": "vkitti_train",
    "lod": "lod_train",
    "lod_day": "lod_day_train",
    "lod_night": "lod_night_train",
    "hypersim": "hypersim_train",
}
_LOD_BASELINE_RELATION = "independent_rgb_baseline_vs_lod_dav2_pseudo_depth"
_INDEPENDENT_BASELINE_RELATION = "independent_rgb_baseline"
_TRAIN_VIZ_CSV_FIELDS = (
    "epoch",
    "source",
    "sample_index",
    "sample_name",
    "target_space",
    "panel_path",
    "npz_path",
    "current_loss_total",
    "current_loss_ssi",
    "current_loss_grad",
    "current_loss_grad_weighted",
    "baseline_loss_total",
    "baseline_loss_ssi",
    "baseline_loss_grad",
    "baseline_loss_grad_weighted",
    "current_align_scale",
    "current_align_shift",
    "baseline_align_scale",
    "baseline_align_shift",
    "baseline_target_relation",
)


def _stable_int_seed(*parts):
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def _jsonable(value):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        if value.numel() <= 256:
            return value.detach().cpu().tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
        }
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 256:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _metadata_record(source, dataset_key, idx, sample, sample_seed, baseline_relation):
    record = {
        "source": source,
        "dataset_key": dataset_key,
        "index": int(idx),
        "sample_index": int(idx),
        "sample_name": str(_first_value(sample.get("sample_name"), f"sample_{idx:06d}")),
        "image_path": str(_first_value(sample.get("image_path"), "")),
        "raw_path": str(_first_value(sample.get("raw_path"), sample.get("image_path", ""))),
        "rgb_path": str(_first_value(sample.get("rgb_path"), "")),
        "depth_path": str(_first_value(sample.get("depth_path"), "")),
        "split": str(_first_value(sample.get("split"), "")),
        "target_space": str(_first_value(sample.get("target_space"), "")),
        "train_viz_seed": int(sample_seed),
        "geometry_params": _jsonable(sample.get("geometry_params", {})),
        "unprocessing_metadata": _jsonable(sample.get("isp_params", {})),
        "baseline_target_relation": baseline_relation,
    }
    return record


def _baseline_relation_for_source(source):
    return _LOD_BASELINE_RELATION if str(source).startswith("lod") else _INDEPENDENT_BASELINE_RELATION


def _resolve_train_viz_sources(train_state, datasets, requested):
    if str(requested).strip().lower() == "auto":
        requested_sources = list(train_state.get("train_sources", ()))
        explicit = False
    else:
        requested_sources = [item.strip() for item in str(requested).split(",") if item.strip()]
        explicit = True

    resolved = []
    warnings = []

    def add_source(source, dataset_key):
        if dataset_key in datasets:
            pair = (source, dataset_key)
            if pair not in resolved:
                resolved.append(pair)
            return True
        return False

    for source in requested_sources:
        if source == "lod":
            if add_source("lod", "lod_train"):
                continue
            added_split = False
            for split_source in ("lod_day", "lod_night"):
                added_split = add_source(split_source, _TRAIN_SOURCE_DATASET_KEYS[split_source]) or added_split
            if not added_split and explicit:
                warnings.append("requested source 'lod' but no lod_train/lod_day_train/lod_night_train dataset exists")
            continue

        dataset_key = _TRAIN_SOURCE_DATASET_KEYS.get(source)
        if dataset_key is None:
            if explicit:
                warnings.append(f"unknown train-viz source {source!r}")
            continue
        if not add_source(source, dataset_key) and explicit:
            warnings.append(f"requested source {source!r} but dataset key {dataset_key!r} is missing")

    return resolved, warnings


def _build_fixed_source_sample(dataset, idx, sample_seed, *, include_geometry):
    if hasattr(dataset, "build_sample"):
        build_sample = dataset.build_sample
        params = inspect.signature(build_sample).parameters
        kwargs = {}
        if "py_rng" in params:
            kwargs["py_rng"] = random.Random(sample_seed)
        if "rng" in params:
            kwargs["rng"] = random.Random(sample_seed)
        if "torch_generator" in params:
            kwargs["torch_generator"] = torch.Generator().manual_seed(int(sample_seed))
        if "include_geometry" in params:
            kwargs["include_geometry"] = bool(include_geometry)
        return build_sample(idx, **kwargs)
    return dataset[idx]


def _coerce_baseline_output(output):
    if isinstance(output, dict):
        tensor = output.get("tensor", output.get("image", output.get("rgb")))
        preview = output.get("preview", output.get("rgb_preview"))
        return tensor, preview
    if isinstance(output, (list, tuple)) and len(output) == 2:
        return output[0], output[1]
    raise TypeError("build_rgb_baseline_input must return (tensor, preview) or a dict with tensor/preview")


def _sample_hw(sample):
    if "depth" in sample and torch.is_tensor(sample["depth"]):
        return tuple(int(v) for v in sample["depth"].shape[-2:])
    for key in ("raw", "image"):
        tensor = sample.get(key)
        if torch.is_tensor(tensor):
            return tuple(int(v) for v in tensor.shape[-2:])
    raise ValueError("Could not infer sample spatial size")


def collect_fixed_train_source_samples(train_state, datasets, args, *, logger=None):
    resolved_sources, warnings = _resolve_train_viz_sources(
        train_state,
        datasets,
        getattr(args, "train_viz_sources", "auto"),
    )
    for message in warnings:
        if logger is not None:
            logger.warning("[TRAIN_VIZ] %s", message)

    if not resolved_sources:
        if logger is not None:
            logger.info("[TRAIN_VIZ] disabled: no train source datasets resolved from %s", args.train_viz_sources)
        return {}

    baseline_enabled = bool(
        getattr(args, "train_viz_rgb_baseline", True)
        or getattr(args, "train_viz_rgb_baseline_checkpoint", None)
    )
    n_per_source = int(getattr(args, "train_viz_n_per_source", 8))
    seed = int(getattr(args, "train_viz_seed", getattr(args, "seed", 42)))
    root = Path(args.save_path) / "train_viz"
    root.mkdir(parents=True, exist_ok=True)

    fixed_samples = {}
    fixed_manifest = []
    source_counts = {}
    has_lod = False

    for source, dataset_key in resolved_sources:
        dataset = datasets[dataset_key]
        dataset_len = len(dataset)
        count = min(max(n_per_source, 0), dataset_len)
        if count <= 0:
            continue

        selection_rng = random.Random(_stable_int_seed(seed, source, dataset_key, "indices"))
        if count >= dataset_len:
            indices = list(range(dataset_len))
        else:
            indices = selection_rng.sample(range(dataset_len), count)

        records = []
        relation = _baseline_relation_for_source(source)
        has_lod = has_lod or source.startswith("lod")
        for order, idx in enumerate(indices):
            sample_seed = _stable_int_seed(seed, source, idx, order)
            sample = _build_fixed_source_sample(dataset, idx, sample_seed, include_geometry=True)
            record = {
                "source": source,
                "dataset_key": dataset_key,
                "index": int(idx),
                "sample_seed": int(sample_seed),
                "sample": sample,
                "baseline_target_relation": relation,
            }
            if baseline_enabled and hasattr(dataset, "build_rgb_baseline_input"):
                geometry = sample.get("geometry_params")
                if not geometry:
                    raise ValueError(
                        f"Train-viz RGB baseline requested, but fixed sample has no geometry_params: "
                        f"source={source} idx={idx}"
                    )
                baseline_output = dataset.build_rgb_baseline_input(idx, geometry, target_hw=_sample_hw(sample))
                baseline_tensor, rgb_preview = _coerce_baseline_output(baseline_output)
                record["rgb_baseline_input"] = baseline_tensor.detach().cpu().float()
                if rgb_preview is not None:
                    record["rgb_preview"] = rgb_preview.detach().cpu().float()
            elif "rgb_preview" in sample and torch.is_tensor(sample["rgb_preview"]):
                record["rgb_preview"] = sample["rgb_preview"].detach().cpu().float()

            records.append(record)
            fixed_manifest.append(
                _metadata_record(source, dataset_key, idx, sample, sample_seed, relation)
            )

        fixed_samples[source] = records
        source_counts[source] = len(records)

    with (root / "fixed_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(fixed_manifest, handle, indent=2, sort_keys=True)

    if logger is not None:
        logger.info(
            "[TRAIN_VIZ] fixed samples=%s baseline=%s root=%s",
            source_counts,
            "enabled" if baseline_enabled else "disabled",
            root,
        )
        if baseline_enabled and has_lod:
            logger.info(
                "[TRAIN_VIZ][NOTE] LOD pseudo-depth target and train-viz RGB baseline are separate "
                "predictions; compare both panels instead of treating the target as the RGB baseline."
            )
    return fixed_samples


def _select_train_model_input(sample, input_type):
    if str(input_type) != "rgb":
        tensor = sample.get("raw", sample.get("image"))
    else:
        tensor = sample.get("image")
    if tensor is None:
        raise KeyError("Sample has no model input tensor")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def _ensure_pred_hw(pred_disp, target_hw):
    if pred_disp.ndim == 4 and pred_disp.shape[1] == 1:
        pred_disp = pred_disp[:, 0]
    if pred_disp.ndim != 3:
        raise ValueError(f"Expected prediction with shape (B,H,W) or (B,1,H,W), got {tuple(pred_disp.shape)}")
    if tuple(pred_disp.shape[-2:]) != tuple(target_hw):
        pred_disp = F.interpolate(
            pred_disp[:, None],
            tuple(target_hw),
            mode="bilinear",
            align_corners=True,
        )[:, 0]
    return pred_disp


def _solve_scale_shift_with_params(pred, target, valid_mask, *, min_valid_pixels=128, eps=1e-6):
    if pred.shape != target.shape or pred.shape != valid_mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape} target={target.shape} mask={valid_mask.shape}")
    bsz = pred.shape[0]
    device = pred.device
    dtype = pred.dtype
    scales = torch.zeros(bsz, device=device, dtype=dtype)
    shifts = torch.zeros(bsz, device=device, dtype=dtype)
    ok = torch.zeros(bsz, device=device, dtype=torch.bool)
    with torch.no_grad():
        for bidx in range(bsz):
            mb = valid_mask[bidx]
            if int(mb.sum().item()) < min_valid_pixels:
                continue
            x = pred[bidx][mb].reshape(-1, 1)
            y = target[bidx][mb].reshape(-1, 1)
            a_mat = torch.cat([x, torch.ones_like(x)], dim=1)
            try:
                sol = torch.linalg.lstsq(a_mat, y).solution
            except RuntimeError:
                continue
            scale = sol[0, 0]
            shift = sol[1, 0]
            if not torch.isfinite(scale) or not torch.isfinite(shift) or scale.abs() < eps:
                continue
            scales[bidx] = scale
            shifts[bidx] = shift
            ok[bidx] = True
    pred_aligned = scales.view(bsz, 1, 1) * pred + shifts.view(bsz, 1, 1)
    effective_mask = valid_mask & ok.view(bsz, 1, 1)
    return pred_aligned, effective_mask, scales, shifts, ok


def _nan_tensor_like(tensor):
    return torch.full_like(tensor, float("nan"))


def _compute_sample_viz_loss(pred_disp, depth, valid_mask, args, target_space):
    min_valid = 128
    eps = 1e-6
    pred_disp = pred_disp.float()
    depth = depth.float()
    valid_mask = valid_mask.bool()
    original_target = build_training_target(depth, valid_mask, target_space=target_space, eps=eps)
    target_for_loss = original_target
    norm_centers = torch.zeros(depth.shape[0], device=depth.device, dtype=depth.dtype)
    norm_scales = torch.ones(depth.shape[0], device=depth.device, dtype=depth.dtype)
    normalized = torch.zeros(depth.shape[0], device=depth.device, dtype=torch.bool)

    metrics = {
        "loss_total": float("nan"),
        "loss_ssi": float("nan"),
        "loss_grad": float("nan"),
        "loss_grad_weighted": float("nan"),
        "align_scale": float("nan"),
        "align_shift": float("nan"),
    }

    if str(args.loss_type) == "aligned_sig":
        try:
            aligned = align_prediction_to_inverse_gt(pred_disp[0], original_target[0], valid_mask[0], eps=eps)
        except RuntimeError:
            aligned = None
        if aligned is None:
            aligned_original = _nan_tensor_like(original_target)
            return metrics, aligned_original, original_target, valid_mask
        pred_loss, aligned_gt, aligned_mask, scale, shift = aligned
        if int(aligned_mask.sum().item()) < min_valid:
            aligned_original = _nan_tensor_like(original_target)
            return metrics, aligned_original, original_target, aligned_mask.unsqueeze(0)
        sig_loss = SigLoss(warm_up=False, eps=1e-3)(pred_loss, aligned_gt, aligned_mask)
        metrics["loss_total"] = float(sig_loss.detach().item())
        metrics["align_scale"] = float(scale.detach().item())
        metrics["align_shift"] = float(shift.detach().item())
        aligned_original = metrics["align_scale"] * pred_disp + metrics["align_shift"]
        return metrics, aligned_original, original_target, aligned_mask.unsqueeze(0)

    if bool(getattr(args, "loss_target_normalization", True)):
        target_for_loss, norm_stats = robust_normalize_target_per_sample(
            original_target,
            valid_mask,
            min_valid_pixels=min_valid,
            min_scale=float(getattr(args, "loss_norm_min_scale", 1e-3)),
        )
        norm_centers = norm_stats["norm_centers"]
        norm_scales = norm_stats["norm_scales"]
        normalized = norm_stats["normalized_mask"]

    pred_aligned, effective_mask, scales, shifts, ok = _solve_scale_shift_with_params(
        pred_disp,
        target_for_loss,
        valid_mask,
        min_valid_pixels=min_valid,
        eps=eps,
    )
    if not bool(ok[0].item()) or int(effective_mask[0].sum().item()) < min_valid:
        return metrics, _nan_tensor_like(original_target), target_for_loss, effective_mask

    loss_ssi = _ssi_mse_from_aligned(pred_aligned, target_for_loss, effective_mask, min_valid)
    metrics["loss_ssi"] = float(loss_ssi.detach().item())
    if str(args.loss_type) == "ssi":
        metrics["loss_total"] = metrics["loss_ssi"]
    elif str(args.loss_type) == "ssi_grad":
        loss_grad = _grad_matching_from_aligned(
            pred_aligned,
            target_for_loss,
            effective_mask,
            n_scales=int(getattr(args, "loss_grad_scales", 4)),
            mask_downsample=str(getattr(args, "loss_mask_downsample", "strict")),
            min_valid=min_valid,
        )
        metrics["loss_grad"] = float(loss_grad.detach().item())
        metrics["loss_grad_weighted"] = float(getattr(args, "loss_lambda_grad", 2.0)) * metrics["loss_grad"]
        metrics["loss_total"] = metrics["loss_ssi"] + metrics["loss_grad_weighted"]
    else:
        raise ValueError(f"Unsupported loss_type={args.loss_type!r}")

    metrics["align_scale"] = float(scales[0].detach().item())
    metrics["align_shift"] = float(shifts[0].detach().item())
    if bool(normalized[0].item()):
        aligned_original = pred_aligned * norm_scales.view(-1, 1, 1) + norm_centers.view(-1, 1, 1)
    else:
        aligned_original = pred_aligned
    return metrics, aligned_original, target_for_loss, effective_mask


def _tensor_2d_np(tensor, valid_mask=None):
    array = tensor.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim == 3:
        array = array[0]
    if valid_mask is not None:
        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        array = array.copy()
        array[~mask_np] = np.nan
    return array


def _target_and_prediction_views(target_space, depth, valid_mask, aligned):
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    if valid_np.ndim == 3:
        valid_np = valid_np[0]
    if target_space == "metric_depth":
        target_view = _tensor_2d_np(depth, valid_mask)
        aligned_inv = _tensor_2d_np(aligned, valid_mask)
        pred_view = np.full_like(aligned_inv, np.nan, dtype=np.float32)
        good = valid_np & np.isfinite(aligned_inv) & (aligned_inv > 1e-6)
        pred_view[good] = 1.0 / aligned_inv[good]
        return target_view, pred_view
    target_view = _tensor_2d_np(depth, valid_mask)
    pred_view = _tensor_2d_np(aligned, valid_mask)
    return target_view, pred_view


def _to_rgb_preview(tensor, *, clip=True):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        array = tensor.detach().float().cpu().numpy()
    else:
        array = np.asarray(tensor, dtype=np.float32)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        tensor_chw = torch.from_numpy(np.transpose(array, (2, 0, 1))).float()
        array = packed_bayer_to_base_rgb(tensor_chw).detach().cpu().numpy()
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        array = np.repeat(array[..., None], 3, axis=-1)
    if clip:
        return np.clip(array, 0.0, 1.0)
    return array


def _maybe_denormalize_imagenet(image_rgb):
    if image_rgb is None:
        return None
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        return image_rgb
    finite = image[np.isfinite(image)]
    if finite.size and (float(np.nanmin(finite)) < -0.05 or float(np.nanmax(finite)) > 1.25):
        image = image * _IMAGENET_STD.reshape(1, 1, 3) + _IMAGENET_MEAN.reshape(1, 1, 3)
    return np.clip(image, 0.0, 1.0)


def _load_rgb_preview_from_path(path_value):
    path_value = _first_value(path_value)
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image


def _imagenet_normalized_rgb_tensor(image_rgb):
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    image = np.clip(image, 0.0, 1.0)
    normalized = (image - _IMAGENET_MEAN.reshape(1, 1, 3)) / _IMAGENET_STD.reshape(1, 1, 3)
    chw = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))).astype(np.float32, copy=False)
    return torch.from_numpy(chw)


def _fixed_rgb_baseline_input_from_sample(sample):
    for key in ("rgb_eval_path", "rgb_640_path", "rgb_src_path"):
        preview = _load_rgb_preview_from_path(sample.get(key))
        if preview is not None:
            return _imagenet_normalized_rgb_tensor(preview)

    tensor = sample.get("image")
    if tensor is None or not torch.is_tensor(tensor):
        return None
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        return None

    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() and (float(finite.min()) < -0.05 or float(finite.max()) > 1.25):
        return tensor
    image = tensor.numpy().transpose(1, 2, 0)
    return _imagenet_normalized_rgb_tensor(image)


def _fixed_rgb_preview_from_sample(sample, input_type):
    if "rgb_preview" in sample:
        preview = _to_rgb_preview(sample.get("rgb_preview"))
        if preview is not None:
            return preview

    for key in ("rgb_eval_path", "rgb_640_path", "rgb_src_path", "image_path"):
        preview = _load_rgb_preview_from_path(sample.get(key))
        if preview is not None:
            return preview

    if str(input_type) == "rgb" and "image" in sample:
        return _maybe_denormalize_imagenet(_to_rgb_preview(sample.get("image"), clip=False))
    if "raw" in sample:
        return _to_rgb_preview(sample.get("raw"))
    return _maybe_denormalize_imagenet(_to_rgb_preview(sample.get("image"), clip=False))


def _fixed_input_preview_from_sample(sample, input_type):
    if str(input_type) != "rgb" and "raw" in sample:
        return _to_rgb_preview(sample.get("raw"))
    if "raw" in sample:
        return _to_rgb_preview(sample.get("raw"))
    preview = _to_rgb_preview(sample.get("image"), clip=False)
    if str(input_type) == "rgb":
        return _maybe_denormalize_imagenet(preview)
    return np.clip(preview, 0.0, 1.0) if preview is not None else None


def _raw_preview_from_sample(sample):
    tensor = sample.get("raw", sample.get("image"))
    if tensor is None:
        return None
    return _to_rgb_preview(tensor)


def _preview_percentile_stretch(image_rgb, *, lower=1.0, upper=99.0, gamma=1.0):
    if image_rgb is None:
        return None
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        return image_rgb
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)
    values = image[finite]
    lo, hi = np.percentile(values, [float(lower), float(upper)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        return np.clip(image, 0.0, 1.0)
    stretched = (np.clip(image, lo, hi) - lo) / (hi - lo)
    stretched = np.where(np.isfinite(stretched), stretched, 0.0)
    if gamma != 1.0:
        stretched = np.clip(stretched, 0.0, 1.0) ** (1.0 / float(gamma))
    return np.clip(stretched, 0.0, 1.0)


def _ram_core_input_channels(ram_core):
    encoder = getattr(ram_core, "encoder", None)
    features = getattr(encoder, "features", None)
    if features is None:
        return None
    first = features[0] if len(features) else None
    return int(getattr(first, "in_channels", 0)) or None


def _rgb_interface_pre_clamp(rgb_head, x4, *, x_raw):
    mode = getattr(rgb_head, "mode", None)
    conv = getattr(rgb_head, "conv", None)
    if mode is None or conv is None:
        return rgb_head(x4, x_raw=x_raw)

    if mode == "sigmoid":
        return torch.sigmoid(conv(x4))
    if mode == "linear_clamp":
        return conv(x4)
    if mode == "tanh01":
        return 0.5 + 0.5 * torch.tanh(conv(x4))

    if x_raw is None:
        return rgb_head(x4, x_raw=x_raw)
    base_rgb = packed_bayer_to_base_rgb(x_raw)
    delta_rgb = conv(x4)
    if mode == "residual_tanh":
        delta_rgb = torch.tanh(delta_rgb)
    elif mode != "residual_linear":
        return rgb_head(x4, x_raw=x_raw)
    residual_scale = float(getattr(rgb_head, "residual_scale", 0.1))
    return base_rgb + residual_scale * delta_rgb


def _infer_front_rgb_pre_norm(active_model, x_raw):
    if x_raw is None or not torch.is_tensor(x_raw):
        return None
    if x_raw.ndim == 3:
        x_raw = x_raw.unsqueeze(0)
    if x_raw.ndim != 4:
        return None

    module = _module(active_model)
    ram_core = getattr(module, "ram_core", None)

    if hasattr(module, "image_bridge") and ram_core is not None:
        x4 = ram_core(x_raw)
        rgb_head = getattr(module.image_bridge, "rgb_head", None)
        if rgb_head is not None:
            return _rgb_interface_pre_clamp(rgb_head, x4, x_raw=x_raw)
        return module.image_bridge(x4, x_raw=x_raw)

    if hasattr(module, "rgb_head") and ram_core is not None:
        x4 = ram_core(x_raw)
        return _rgb_interface_pre_clamp(module.rgb_head, x4, x_raw=x_raw)

    if hasattr(module, "residual_head") and ram_core is not None:
        x4 = ram_core(x_raw)
        delta_rgb = module.residual_head(x4)
        base_rgb = packed_bayer_to_base_rgb(x_raw)
        residual_scale = float(getattr(module, "residual_scale", 0.1))
        return base_rgb + residual_scale * torch.tanh(delta_rgb)

    if ram_core is not None:
        in_channels = _ram_core_input_channels(ram_core)
        if in_channels == 3 and x_raw.shape[1] == 4:
            x3_in = packed_bayer_to_base_rgb(x_raw)
            x3 = ram_core(x3_in)
            return phase1b_tanh_tail_squash(x3)
        if in_channels == 4 and x_raw.shape[1] == 4:
            x4 = ram_core(x_raw)
            if x4.ndim == 4 and x4.shape[1] == 4:
                return packed_bayer_to_base_rgb(x4)

    return None


def _color_limits(arrays):
    finite_values = []
    for array in arrays:
        if array is None:
            continue
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return 0.0, 1.0
    values = np.concatenate(finite_values)
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _colorize_array_rgb(array, lo, hi):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    norm = np.zeros(array.shape, dtype=np.uint8)
    if hi > lo:
        clipped = np.clip(array, lo, hi)
        clipped = np.where(np.isfinite(clipped), clipped, lo)
        norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral")
    rgb = (cmap(norm)[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~finite] = 0
    return rgb


def _format_loss(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4g}"


def _make_tile(image_rgb, title, subtitle="", tile_hw=(230, 360)):
    tile_h, tile_w = tile_hw
    if image_rgb is None:
        image_uint8 = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    else:
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0.0, 1.0)
            image_uint8 = (image * 255.0).astype(np.uint8)
        else:
            image_uint8 = image
        image_uint8 = cv2.resize(image_uint8, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    overlay_h = 42 if subtitle else 26
    image_uint8[:overlay_h, :, :] = (0.55 * image_uint8[:overlay_h, :, :]).astype(np.uint8)
    cv2.putText(image_uint8, str(title), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(image_uint8, str(subtitle), (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return image_uint8


def _metadata_strip(record, current_metrics, width, *, height=58):
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:, :, :] = (18, 18, 18)
    sample = record["sample"]
    crop_box = sample.get("geometry_params", {}).get("crop_box", "")
    text_left = (
        f"source={record['source']} sample={sample.get('sample_name', record['index'])} "
        f"idx={record['index']} target={sample.get('target_space', 'n/a')}"
    )
    text_right = (
        f"current loss={_format_loss(current_metrics['loss_total'])} "
        f"scale={_format_loss(current_metrics['align_scale'])} "
        f"shift={_format_loss(current_metrics['align_shift'])} crop={crop_box}"
    )
    cv2.putText(strip, text_left[:170], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(strip, text_right[:170], (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    return strip


def _channel_percentiles(image_rgb):
    if image_rgb is None:
        return None
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    stats = []
    for cidx in range(3):
        values = image[..., cidx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            stats.append(None)
            continue
        stats.append(
            {
                "min": float(np.min(values)),
                "p1": float(np.percentile(values, 1)),
                "p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
        )
    return stats


def _format_stat_value(value):
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if abs(value) < 1e-3 and value != 0.0:
        return f"{value:.1e}"
    return f"{value:.3g}"


def _nice_upper_bound(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    normalized = value / scale
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return float(candidate * scale)
    return float(10.0 * scale)


def _distribution_axis_limits(image, axis_mode):
    if axis_mode == "unit":
        return 0.0, 1.0, (0.0, 0.5, 1.0), "x-axis fixed 0..1"

    finite_values = np.asarray(image, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0, (0.0, 0.5, 1.0), "x-axis n/a"

    if axis_mode == "zero_center":
        robust_abs = float(np.percentile(np.abs(finite_values), 99.0))
        limit = max(2.5, _nice_upper_bound(robust_abs))
        return -limit, limit, (-limit, 0.0, limit), "x-axis centered at 0"

    lo, hi = np.percentile(finite_values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values))
    if hi <= lo:
        hi = lo + 1.0
    mid = 0.5 * (float(lo) + float(hi))
    return float(lo), float(hi), (float(lo), mid, float(hi)), "x-axis p1..p99"


def _draw_hist_axes(canvas, hist_x, hist_y, hist_w, hist_h, lo, hi, tick_values, *, zero_center=False):
    axis_color = (150, 150, 150)
    grid_color = (62, 62, 62)
    label_color = (190, 190, 190)
    base_y = hist_y + hist_h

    cv2.line(canvas, (hist_x, hist_y), (hist_x, base_y), axis_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (hist_x, base_y), (hist_x + hist_w, base_y), axis_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "1", (hist_x - 20, hist_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, label_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (hist_x - 20, base_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, label_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "rel", (hist_x - 30, hist_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1, cv2.LINE_AA)

    denom = max(float(hi) - float(lo), 1e-12)
    for tick in tick_values:
        tick = float(tick)
        if not math.isfinite(tick) or tick < lo or tick > hi:
            continue
        px = hist_x + int(round(((tick - lo) / denom) * hist_w))
        cv2.line(canvas, (px, hist_y), (px, base_y), grid_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (px, base_y), (px, base_y + 4), axis_color, 1, cv2.LINE_AA)
        label = _format_stat_value(tick)
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        label_x = int(np.clip(px - label_w // 2, hist_x, hist_x + hist_w - label_w))
        cv2.putText(
            canvas,
            label,
            (label_x, base_y + 17 + label_h // 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            label_color,
            1,
            cv2.LINE_AA,
        )

    if zero_center and lo < 0.0 < hi:
        zero_x = hist_x + int(round(((0.0 - lo) / denom) * hist_w))
        cv2.line(canvas, (zero_x, hist_y), (zero_x, base_y), (215, 215, 215), 1, cv2.LINE_AA)


def _draw_distribution_block(canvas, image_rgb, title, x0, y0, width, height, *, axis_mode="auto"):
    cv2.rectangle(canvas, (x0, y0), (x0 + width - 1, y0 + height - 1), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, title, (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    if image_rgb is None:
        cv2.putText(canvas, "no data", (x0 + 8, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
        return

    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        cv2.putText(canvas, "invalid shape", (x0 + 8, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
        return

    hist_x = x0 + 36
    hist_y = y0 + 30
    hist_w = width - 46
    hist_h = 54
    stats = _channel_percentiles(image)
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        cv2.putText(canvas, "no finite values", (x0 + 8, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
        return

    lo, hi, tick_values, axis_note = _distribution_axis_limits(image, axis_mode)

    cv2.rectangle(canvas, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h), (38, 38, 38), thickness=-1)
    _draw_hist_axes(
        canvas,
        hist_x,
        hist_y,
        hist_w,
        hist_h,
        lo,
        hi,
        tick_values,
        zero_center=(axis_mode == "zero_center"),
    )
    channel_colors = [(230, 80, 80), (80, 220, 100), (80, 150, 255)]
    bins = 80
    for cidx, color in enumerate(channel_colors):
        values = image[..., cidx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        counts, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
        if counts.max() <= 0:
            continue
        points = []
        for bidx, count in enumerate(counts):
            px = hist_x + int(round((bidx / max(bins - 1, 1)) * (hist_w - 1)))
            py = hist_y + hist_h - 1 - int(round((float(count) / float(counts.max())) * (hist_h - 4)))
            points.append((px, py))
        for left, right in zip(points[:-1], points[1:]):
            cv2.line(canvas, left, right, color, 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{axis_note}: {_format_stat_value(lo)}..{_format_stat_value(hi)}",
        (x0 + 8, y0 + 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    labels = ("R", "G", "B")
    for cidx, stat in enumerate(stats):
        if stat is None:
            line = f"{labels[cidx]}: n/a"
        else:
            line = (
                f"{labels[cidx]} min/p50/max "
                f"{_format_stat_value(stat['min'])}/{_format_stat_value(stat['p50'])}/{_format_stat_value(stat['max'])} "
                f"p99 {_format_stat_value(stat['p99'])}"
            )
        cv2.putText(
            canvas,
            line[:70],
            (x0 + 8, y0 + 130 + cidx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            channel_colors[cidx],
            1,
            cv2.LINE_AA,
        )


def _distribution_strip(rgb_preview, raw_preview, ram_preview, width, *, height=190, titles=None, axis_modes=None):
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:, :, :] = (14, 14, 14)
    gap = 8
    block_w = (width - gap * 4) // 3
    x_positions = [gap, gap * 2 + block_w, gap * 3 + block_w * 2]
    if titles is None:
        titles = ("RGB value distribution", "RAW preview distribution", "RAM BN+tanh2.5 distribution")
    if axis_modes is None:
        axis_modes = ("unit", "unit", "zero_center")
    items = zip(x_positions, (rgb_preview, raw_preview, ram_preview), titles)
    for idx, (x0, image, title) in enumerate(items):
        axis_mode = axis_modes[min(len(axis_modes) - 1, idx)]
        _draw_distribution_block(strip, image, title, x0, 8, block_w, height - 16, axis_mode=axis_mode)
    return strip


def _make_train_viz_panel(
    record,
    target_view,
    current_view,
    current_metrics,
    ram_rgb_pre_norm=None,
    baseline_view=None,
    baseline_metrics=None,
    baseline_label="RGB DAv2",
):
    rgb_preview = _to_rgb_preview(record.get("rgb_preview"))
    raw_preview = _raw_preview_from_sample(record["sample"])
    ram_preview = _to_rgb_preview(ram_rgb_pre_norm, clip=False)
    color_limit_inputs = [target_view, current_view]
    if baseline_view is not None:
        color_limit_inputs.append(baseline_view)
    lo, hi = _color_limits(color_limit_inputs)
    target_rgb = _colorize_array_rgb(target_view, lo, hi)
    baseline_rgb = _colorize_array_rgb(baseline_view, lo, hi) if baseline_view is not None else None
    current_rgb = _colorize_array_rgb(current_view, lo, hi)
    source = record["source"]
    target_label = "Target depth" if record["sample"].get("target_space") == "metric_depth" else "Target rel inv"
    tiles = [
        _make_tile(rgb_preview, "RGB", source),
        _make_tile(_preview_percentile_stretch(raw_preview), "RAW preview", "R, avg(G), B"),
        _make_tile(_preview_percentile_stretch(ram_preview), "RAM BN+tanh2.5", "no clamp/norm"),
        _make_tile(target_rgb, target_label, f"range {lo:.3g}..{hi:.3g}"),
    ]
    if baseline_view is not None:
        baseline_loss = baseline_metrics.get("loss_total") if isinstance(baseline_metrics, dict) else float("nan")
        tiles.append(_make_tile(baseline_rgb, baseline_label, f"loss {_format_loss(baseline_loss)}"))
    tiles.append(_make_tile(current_rgb, "Current aligned", f"loss {_format_loss(current_metrics['loss_total'])}"))
    image_row = np.concatenate(tiles, axis=1)
    return np.concatenate(
        [
            image_row,
            _metadata_strip(record, current_metrics, image_row.shape[1]),
            _distribution_strip(rgb_preview, raw_preview, ram_preview, image_row.shape[1]),
        ],
        axis=0,
    )


def _format_panel_metric(value, spec=".4f"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return format(value, spec)


def _masked_depth_view(sample):
    depth = _as_2d_numpy(sample.get("depth"), dtype=np.float32)
    valid_mask = _as_2d_numpy(sample.get("valid_mask"), dtype=bool)
    if depth is None:
        return None
    view = depth.astype(np.float32, copy=True)
    if valid_mask is not None:
        view[~valid_mask.astype(bool, copy=False)] = np.nan
    return view


def _fixed_metadata_strip(split_name, sample, sample_index, metrics_payload, width, *, height=58):
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:, :, :] = (18, 18, 18)
    sample_name = str(_first_value(sample.get("sample_name"), _sample_stem(sample, sample_index)))
    depth_mode = str(_first_value(sample.get("depth_mode"), "full"))
    backend = str(_first_value(sample.get("fast_eval_backend"), "proxy"))
    text_left = f"split={split_name} sample={sample_name} idx={sample_index} depth_mode={depth_mode} backend={backend}"

    metrics = metrics_payload.get("metrics", {}) if metrics_payload else {}
    align = metrics_payload.get("align", {}) if metrics_payload else {}
    text_right = (
        f"abs_rel={_format_panel_metric(metrics.get('abs_rel'))} "
        f"rmse={_format_panel_metric(metrics.get('rmse'))} "
        f"d1={_format_panel_metric(metrics.get('d1'))} "
        f"scale={_format_panel_metric(align.get('scale'), '.4g')} "
        f"shift={_format_panel_metric(align.get('shift'), '.4g')}"
    )
    cv2.putText(strip, text_left[:170], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(strip, text_right[:170], (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    return strip


def _make_fixed_viz_panel(
    *,
    sample,
    split_name,
    sample_index,
    input_type,
    target_view,
    current_view,
    current_title,
    current_subtitle,
    metrics_payload,
    ram_rgb_pre_norm=None,
    show_rgb_baseline=False,
    rgb_baseline_view=None,
    rgb_baseline_title="RGB DAv2",
    rgb_baseline_subtitle="",
):
    rgb_preview = _fixed_rgb_preview_from_sample(sample, input_type)
    input_preview = _fixed_input_preview_from_sample(sample, input_type)
    ram_preview = _to_rgb_preview(ram_rgb_pre_norm, clip=False)
    color_limit_inputs = [target_view, current_view]
    if show_rgb_baseline and rgb_baseline_view is not None:
        color_limit_inputs.append(rgb_baseline_view)
    lo, hi = _color_limits(color_limit_inputs)
    target_rgb = _colorize_array_rgb(target_view, lo, hi) if target_view is not None else None
    rgb_baseline_rgb = (
        _colorize_array_rgb(rgb_baseline_view, lo, hi)
        if show_rgb_baseline and rgb_baseline_view is not None
        else None
    )
    current_rgb = _colorize_array_rgb(current_view, lo, hi) if current_view is not None else None
    tiles = [
        _make_tile(rgb_preview, "RGB", split_name),
        _make_tile(_preview_percentile_stretch(input_preview), "Input preview", "RAW/base RGB"),
        _make_tile(_preview_percentile_stretch(ram_preview), "RAM BN+tanh2.5", "no clamp/norm"),
        _make_tile(target_rgb, "GT depth", f"range {lo:.3g}..{hi:.3g}"),
    ]
    if show_rgb_baseline:
        tiles.append(_make_tile(rgb_baseline_rgb, rgb_baseline_title, rgb_baseline_subtitle))
    tiles.append(_make_tile(current_rgb, current_title, current_subtitle))
    image_row = np.concatenate(tiles, axis=1)
    return np.concatenate(
        [
            image_row,
            _fixed_metadata_strip(split_name, sample, sample_index, metrics_payload, image_row.shape[1]),
            _distribution_strip(
                rgb_preview,
                input_preview,
                ram_preview,
                image_row.shape[1],
                titles=("RGB preview distribution", "Input preview distribution", "RAM BN+tanh2.5 distribution"),
            ),
        ],
        axis=0,
    )


def _float_or_nan(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _csv_value(value):
    value = _float_or_nan(value)
    if math.isfinite(value):
        return f"{value:.9g}"
    return "nan"


def dump_train_source_samples(
    model,
    fixed_samples,
    args,
    epoch,
    save_root,
    *,
    writer=None,
    baseline_model=None,
    baseline_label="rgb_baseline",
    logger=None,
):
    if not fixed_samples:
        return {}

    root = Path(save_root) / "train_viz"
    epoch_root = root / _epoch_dir_name(epoch)
    epoch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"
    csv_path = root / "loss_by_sample.csv"
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    use_amp = bool(getattr(args, "amp", False))
    device = next(_module(model).parameters()).device
    models = {id(model): model}
    if baseline_model is not None:
        models[id(baseline_model)] = baseline_model
    training_states = {key: active_model.training for key, active_model in models.items()}
    outputs = {}
    csv_needs_header = not csv_path.exists()

    try:
        for active_model in models.values():
            active_model.eval()

        with manifest_path.open("a", encoding="utf-8") as manifest_handle, csv_path.open("a", newline="", encoding="utf-8") as csv_handle:
            csv_writer = csv.DictWriter(csv_handle, fieldnames=_TRAIN_VIZ_CSV_FIELDS)
            if csv_needs_header:
                csv_writer.writeheader()

            with torch.no_grad():
                for source, records in fixed_samples.items():
                    source_dir = epoch_root / source
                    source_dir.mkdir(parents=True, exist_ok=True)
                    saved = []
                    for sample_order, record in enumerate(records):
                        sample = record["sample"]
                        target_space = str(_first_value(sample.get("target_space"), "metric_depth"))
                        depth = sample["depth"]
                        valid_mask = sample["valid_mask"]
                        if depth.ndim == 2:
                            depth = depth.unsqueeze(0)
                        if valid_mask.ndim == 2:
                            valid_mask = valid_mask.unsqueeze(0)
                        depth = depth.to(device=device, non_blocking=True).float()
                        valid_mask = valid_mask.to(device=device, non_blocking=True).bool()

                        model_input = _select_train_model_input(sample, getattr(args, "input_type", "rgb"))
                        model_input = model_input.to(device=device, non_blocking=True).float()
                        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and model_input.is_cuda):
                            current_pred = model(model_input)
                            ram_rgb_pre_norm = _infer_front_rgb_pre_norm(model, model_input)
                        current_pred = _ensure_pred_hw(current_pred.float(), depth.shape[-2:])
                        current_metrics, current_aligned, target_loss_space, current_effective_mask = _compute_sample_viz_loss(
                            current_pred,
                            depth,
                            valid_mask,
                            args,
                            target_space,
                        )

                        baseline_has_input = baseline_model is not None and record.get("rgb_baseline_input") is not None
                        if baseline_has_input:
                            baseline_input = record.get("rgb_baseline_input")
                            if baseline_input.ndim == 3:
                                baseline_input = baseline_input.unsqueeze(0)
                            baseline_input = baseline_input.to(device=device, non_blocking=True).float()
                            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and baseline_input.is_cuda):
                                baseline_pred = baseline_model(baseline_input)
                            baseline_pred = _ensure_pred_hw(baseline_pred.float(), depth.shape[-2:])
                            baseline_metrics, baseline_aligned, _, baseline_effective_mask = _compute_sample_viz_loss(
                                baseline_pred,
                                depth,
                                valid_mask,
                                args,
                                target_space,
                            )
                        else:
                            baseline_pred = torch.full_like(current_pred, float("nan"))
                            baseline_aligned = torch.full_like(current_aligned, float("nan"))
                            baseline_effective_mask = torch.zeros_like(valid_mask)
                            baseline_metrics = {
                                "loss_total": float("nan"),
                                "loss_ssi": float("nan"),
                                "loss_grad": float("nan"),
                                "loss_grad_weighted": float("nan"),
                                "align_scale": float("nan"),
                                "align_shift": float("nan"),
                            }

                        target_view, current_view = _target_and_prediction_views(
                            target_space,
                            depth[0],
                            valid_mask[0],
                            current_aligned[0],
                        )
                        baseline_view = None
                        if baseline_has_input:
                            _, baseline_view = _target_and_prediction_views(
                                target_space,
                                depth[0],
                                valid_mask[0],
                                baseline_aligned[0],
                            )
                        panel_rgb = _make_train_viz_panel(
                            record,
                            target_view,
                            current_view,
                            current_metrics,
                            ram_rgb_pre_norm=ram_rgb_pre_norm[0] if torch.is_tensor(ram_rgb_pre_norm) else None,
                            baseline_view=baseline_view,
                            baseline_metrics=baseline_metrics,
                            baseline_label=baseline_label,
                        )

                        stem = _sample_stem(sample, sample_order)
                        npz_path = source_dir / f"{sample_order:02d}_{stem}.npz"
                        panel_path = source_dir / f"{sample_order:02d}_{stem}_panel.jpg"
                        valid_np = valid_mask[0].detach().cpu().numpy().astype(bool)
                        target_original_np = _tensor_2d_np(depth[0], valid_mask[0])
                        current_aligned_np = _tensor_2d_np(current_aligned[0], valid_mask[0])
                        baseline_aligned_np = _tensor_2d_np(baseline_aligned[0], valid_mask[0])
                        npz_payload = {
                            "target_space": np.array(target_space),
                            "valid_mask": valid_np,
                            "target_original_space": target_original_np,
                            "target_loss_space": _tensor_2d_np(target_loss_space[0], valid_mask[0]),
                            "current_raw_pred_disp": _tensor_2d_np(current_pred[0]),
                            "current_aligned_target_space": current_aligned_np,
                            "baseline_raw_pred_disp": _tensor_2d_np(baseline_pred[0]),
                            "baseline_aligned_target_space": baseline_aligned_np,
                            "current_align_scale": np.array(current_metrics["align_scale"], dtype=np.float32),
                            "current_align_shift": np.array(current_metrics["align_shift"], dtype=np.float32),
                            "baseline_align_scale": np.array(baseline_metrics["align_scale"], dtype=np.float32),
                            "baseline_align_shift": np.array(baseline_metrics["align_shift"], dtype=np.float32),
                            "current_loss_total": np.array(current_metrics["loss_total"], dtype=np.float32),
                            "current_loss_ssi": np.array(current_metrics["loss_ssi"], dtype=np.float32),
                            "current_loss_grad": np.array(current_metrics["loss_grad"], dtype=np.float32),
                            "current_loss_grad_weighted": np.array(current_metrics["loss_grad_weighted"], dtype=np.float32),
                            "baseline_loss_total": np.array(baseline_metrics["loss_total"], dtype=np.float32),
                            "baseline_loss_ssi": np.array(baseline_metrics["loss_ssi"], dtype=np.float32),
                            "baseline_loss_grad": np.array(baseline_metrics["loss_grad"], dtype=np.float32),
                            "baseline_loss_grad_weighted": np.array(baseline_metrics["loss_grad_weighted"], dtype=np.float32),
                            "baseline_target_relation": np.array(record["baseline_target_relation"]),
                            "baseline_label": np.array(str(baseline_label)),
                            "current_effective_mask": current_effective_mask[0].detach().cpu().numpy().astype(bool),
                            "baseline_effective_mask": baseline_effective_mask[0].detach().cpu().numpy().astype(bool),
                        }
                        if target_space == "metric_depth":
                            current_depth = np.full_like(current_aligned_np, np.nan, dtype=np.float32)
                            baseline_depth = np.full_like(baseline_aligned_np, np.nan, dtype=np.float32)
                            current_good = valid_np & np.isfinite(current_aligned_np) & (current_aligned_np > 1e-6)
                            baseline_good = valid_np & np.isfinite(baseline_aligned_np) & (baseline_aligned_np > 1e-6)
                            current_depth[current_good] = 1.0 / current_aligned_np[current_good]
                            baseline_depth[baseline_good] = 1.0 / baseline_aligned_np[baseline_good]
                            npz_payload["current_aligned_depth_m"] = current_depth
                            npz_payload["baseline_aligned_depth_m"] = baseline_depth
                        np.savez_compressed(npz_path, **npz_payload)
                        cv2.imwrite(str(panel_path), panel_rgb[:, :, ::-1])

                        if writer is not None:
                            tag_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
                            writer.add_image(
                                f"train_viz/{source}/{tag_stem}",
                                np.transpose(panel_rgb, (2, 0, 1)),
                                epoch,
                                dataformats="CHW",
                            )
                            writer.add_scalar(
                                f"train_viz_loss/{source}/{tag_stem}/current_total",
                                _float_or_nan(current_metrics["loss_total"]),
                                epoch,
                            )
                            writer.add_scalar(
                                f"train_viz_loss/{source}/{tag_stem}/baseline_total",
                                _float_or_nan(baseline_metrics["loss_total"]),
                                epoch,
                            )

                        manifest_record = {
                            "epoch": int(epoch),
                            "source": source,
                            "sample_index": int(record["index"]),
                            "sample_name": str(sample.get("sample_name", stem)),
                            "target_space": target_space,
                            "panel_path": str(panel_path),
                            "npz_path": str(npz_path),
                            "current_loss_total": _float_or_nan(current_metrics["loss_total"]),
                            "current_loss_ssi": _float_or_nan(current_metrics["loss_ssi"]),
                            "current_loss_grad": _float_or_nan(current_metrics["loss_grad"]),
                            "current_loss_grad_weighted": _float_or_nan(current_metrics["loss_grad_weighted"]),
                            "baseline_loss_total": _float_or_nan(baseline_metrics["loss_total"]),
                            "baseline_loss_ssi": _float_or_nan(baseline_metrics["loss_ssi"]),
                            "baseline_loss_grad": _float_or_nan(baseline_metrics["loss_grad"]),
                            "baseline_loss_grad_weighted": _float_or_nan(baseline_metrics["loss_grad_weighted"]),
                            "current_align_scale": _float_or_nan(current_metrics["align_scale"]),
                            "current_align_shift": _float_or_nan(current_metrics["align_shift"]),
                            "baseline_align_scale": _float_or_nan(baseline_metrics["align_scale"]),
                            "baseline_align_shift": _float_or_nan(baseline_metrics["align_shift"]),
                            "baseline_target_relation": record["baseline_target_relation"],
                        }
                        manifest_handle.write(json.dumps(manifest_record, sort_keys=True) + "\n")
                        csv_writer.writerow(
                            {
                                key: (
                                    _csv_value(manifest_record[key])
                                    if key.endswith("loss_total")
                                    or key.endswith("loss_ssi")
                                    or key.endswith("loss_grad")
                                    or key.endswith("loss_grad_weighted")
                                    or key.endswith("align_scale")
                                    or key.endswith("align_shift")
                                    else manifest_record[key]
                                )
                                for key in _TRAIN_VIZ_CSV_FIELDS
                            }
                        )
                        saved.append({"panel": str(panel_path), "npz": str(npz_path)})
                    outputs[source] = saved
    finally:
        for key, active_model in models.items():
            if training_states.get(key):
                active_model.train()

    if logger is not None:
        logger.info(
            "[TRAIN_VIZ] dumped epoch=%d sources=%s root=%s",
            epoch,
            {source: len(paths) for source, paths in outputs.items()},
            epoch_root,
        )
    return outputs


def dump_fixed_samples(
    model,
    fixed_samples,
    args,
    epoch,
    save_root,
    *,
    model_overrides=None,
    input_type_overrides=None,
    rgb_baseline_model=None,
    rgb_baseline_splits=None,
    rgb_baseline_label="RGB DAv2",
):
    if not fixed_samples:
        return {}

    model_overrides = model_overrides or {}
    input_type_overrides = input_type_overrides or {}
    rgb_baseline_splits = set(rgb_baseline_splits or ())
    root = Path(save_root) / "viz_fixed" / _epoch_dir_name(epoch)
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    use_amp = bool(getattr(args, "amp", False))
    outputs = {}

    models = {id(model): model}
    for override in model_overrides.values():
        if override is not None:
            models[id(override)] = override
    if rgb_baseline_model is not None:
        models[id(rgb_baseline_model)] = rgb_baseline_model
    training_states = {key: value.training for key, value in models.items()}

    try:
        for active_model in models.values():
            active_model.eval()

        with torch.no_grad():
            for split_name, samples in fixed_samples.items():
                split_dir = root / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                active_model = model_overrides.get(split_name) or model
                input_type = input_type_overrides.get(split_name, getattr(args, "input_type", "rgb"))
                device = next(_module(active_model).parameters()).device
                saved = []

                for idx, sample in enumerate(samples):
                    tensor = _select_model_input(sample, input_type).to(device=device, non_blocking=True).float()
                    with torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                        enabled=use_amp and tensor.is_cuda,
                    ):
                        pred_disp = active_model(tensor)
                    disp = pred_disp.detach().float()
                    if disp.ndim == 4 and disp.shape[1] == 1:
                        disp = disp[:, 0]
                    if disp.ndim == 3:
                        disp_np = disp[0].cpu().numpy()
                    else:
                        disp_np = disp.cpu().numpy()

                    stem = _sample_stem(sample, idx)
                    npz_path = split_dir / f"{stem}_disp.npz"
                    png_path = split_dir / f"{stem}_color.png"
                    np.savez_compressed(npz_path, disp=disp_np.astype(np.float32, copy=False))
                    cv2.imwrite(str(png_path), _colorize_disp(disp_np))
                    saved_record = {"disp": str(npz_path), "color": str(png_path)}
                    saved_record.update(
                        _write_fixed_panel_extras(
                            active_model=active_model,
                            rgb_baseline_model=(
                                rgb_baseline_model if split_name in rgb_baseline_splits else None
                            ),
                            rgb_baseline_label=rgb_baseline_label,
                            tensor=tensor,
                            pred_disp=disp,
                            disp_np=disp_np,
                            sample=sample,
                            args=args,
                            split_name=split_name,
                            stem=stem,
                            split_dir=split_dir,
                            sample_index=idx,
                            disp_path=npz_path,
                            color_path=png_path,
                            input_type=input_type,
                            amp_dtype=amp_dtype,
                            use_amp=use_amp,
                        )
                    )
                    saved.append(saved_record)
                outputs[split_name] = saved
    finally:
        for key, active_model in models.items():
            if training_states.get(key):
                active_model.train()
    return outputs
