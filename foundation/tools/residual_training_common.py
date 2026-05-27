from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from finetune_stf.util.loss import build_training_target, robust_normalize_target_per_sample


METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3")
REGION_KEYS = (
    "boundary_abs_rel",
    "dav2_high_error_abs_rel",
    "far50_abs_rel",
    "dark_abs_rel",
    "saturated_abs_rel",
)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def attach_file_logger(logger: logging.Logger, log_path: Path) -> None:
    log_path = log_path.expanduser().resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            return
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logger.level)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s"))
    logger.addHandler(handler)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def float_or_none(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def mean_finite(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(finite))


def average_dicts(rows: list[dict[str, Any]], keys: tuple[str, ...] | list[str]) -> dict[str, float | None]:
    return {key: mean_finite([row.get(key, float("nan")) for row in rows]) for key in keys}


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError(f"masked_mean shape mismatch: value={value.shape} mask={mask.shape}")
    if not bool(mask.any().item()):
        return value.sum() * 0.0
    return value[mask].mean()


def build_error_mask(e0: torch.Tensor, valid_mask: torch.Tensor, *, min_valid_pixels: int = 128) -> torch.Tensor:
    out = torch.zeros_like(e0)
    with torch.no_grad():
        for b in range(e0.shape[0]):
            mask = valid_mask[b]
            if int(mask.sum().item()) < min_valid_pixels:
                continue
            vals = e0[b][mask].float()
            q80 = torch.quantile(vals, 0.80)
            q95 = torch.quantile(vals, 0.95)
            out[b] = torch.clamp((e0[b].float() - q80) / (q95 - q80 + 1e-6), 0.0, 1.0).to(dtype=e0.dtype)
    return out


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    mask_x = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]
    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    target_dy = target[:, 1:, :] - target[:, :-1, :]
    mask_y = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
    return masked_mean((pred_dx - target_dx).abs(), mask_x) + masked_mean((pred_dy - target_dy).abs(), mask_y)


def compute_residual_loss(
    out: dict[str, torch.Tensor],
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    min_valid_pixels: int = 128,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = out["pred"].float()
    d0_norm = out["D0_norm"].float()
    gate = out["gate"].float()
    delta = out["delta"].float()
    valid_mask = valid_mask.bool()
    sample_ok = valid_mask.flatten(1).sum(dim=1) >= int(min_valid_pixels)
    effective_mask = valid_mask & sample_ok[:, None, None]
    used_samples = int(sample_ok.sum().item())
    skipped_samples = int((~sample_ok).sum().item())
    if used_samples == 0:
        zero = pred.sum() * 0.0
        return zero, {
            "used_samples": 0,
            "skipped_samples": skipped_samples,
            "loss_total": 0.0,
            "L_depth": 0.0,
            "L_grad": 0.0,
            "L_keep": 0.0,
            "L_res": 0.0,
            "L_gate": 0.0,
            "L_gate_sup": 0.0,
            "mean_gate": 0.0,
            "max_gate": 0.0,
            "mean_abs_delta": 0.0,
            "mean_abs_gate_delta": 0.0,
            "mean_abs_final_minus_d0_norm": 0.0,
        }

    inv_gt = build_training_target(depth.float(), valid_mask, target_space="metric_depth")
    y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_mask, min_valid_pixels=min_valid_pixels)
    y_norm = y_norm.float()

    e0 = (d0_norm - y_norm).abs()
    m_error = build_error_mask(e0, effective_mask, min_valid_pixels=min_valid_pixels)
    gate_delta = gate * delta

    l_depth = masked_mean((pred - y_norm).abs(), effective_mask)
    l_grad = gradient_l1(pred, y_norm, effective_mask)
    l_keep = masked_mean((1.0 - m_error) * gate_delta.abs(), effective_mask)
    l_res = masked_mean(gate_delta.abs(), effective_mask)
    l_gate = masked_mean(gate, effective_mask)
    gate_prob = gate.float().clamp(1e-6, 1.0 - 1e-6)
    m_error_float = m_error.float()
    bce = -(m_error_float * torch.log(gate_prob) + (1.0 - m_error_float) * torch.log1p(-gate_prob))
    l_gate_sup = masked_mean(bce, effective_mask)
    loss = l_depth + 0.5 * l_grad + 0.1 * l_keep + 0.01 * l_res + 0.005 * l_gate + 0.05 * l_gate_sup

    with torch.no_grad():
        info = {
            "used_samples": used_samples,
            "skipped_samples": skipped_samples,
            "loss_total": float(loss.detach().item()),
            "L_depth": float(l_depth.detach().item()),
            "L_grad": float(l_grad.detach().item()),
            "L_keep": float(l_keep.detach().item()),
            "L_res": float(l_res.detach().item()),
            "L_gate": float(l_gate.detach().item()),
            "L_gate_sup": float(l_gate_sup.detach().item()),
            "mean_gate": float(gate[effective_mask].mean().detach().item()),
            "max_gate": float(gate[effective_mask].max().detach().item()),
            "mean_abs_delta": float(delta[effective_mask].abs().mean().detach().item()),
            "mean_abs_gate_delta": float(gate_delta[effective_mask].abs().mean().detach().item()),
            "mean_abs_final_minus_d0_norm": float((pred - d0_norm)[effective_mask].abs().mean().detach().item()),
        }
    return loss, info


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    train_summary: dict[str, Any],
    val_summary: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "args": vars(args),
            "train_summary": train_summary,
            "val_summary": val_summary,
        },
        path,
    )


def top_fraction_mask(values: np.ndarray, valid_mask: np.ndarray, fraction: float) -> np.ndarray:
    valid_values = values[valid_mask & np.isfinite(values)]
    if valid_values.size == 0:
        return np.zeros_like(valid_mask, dtype=bool)
    threshold = float(np.quantile(valid_values, 1.0 - float(fraction)))
    return valid_mask & np.isfinite(values) & (values >= threshold)


def region_abs_rel(
    gt: np.ndarray,
    aligned_depth: np.ndarray,
    mask: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
) -> float:
    eval_depth = np.asarray(aligned_depth, dtype=np.float64).copy()
    finite = np.isfinite(eval_depth)
    eval_depth[finite] = np.clip(eval_depth[finite], float(min_depth), float(max_depth))
    vm = (
        mask
        & np.isfinite(gt)
        & np.isfinite(eval_depth)
        & (gt >= float(min_depth))
        & (gt <= float(max_depth))
        & (gt > 0)
        & (eval_depth > 0)
    )
    if int(vm.sum()) < 10:
        return float("nan")
    return float(np.mean(np.abs(eval_depth[vm] - gt[vm]) / gt[vm]))


def sample_region_metrics(
    *,
    depth_np: np.ndarray,
    valid_np: np.ndarray,
    aligned_final: np.ndarray,
    aligned_d0: np.ndarray,
    d0_norm_np: np.ndarray,
    y_norm_np: np.ndarray,
    rgb_preview_np: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[dict[str, float], dict[str, float]]:
    grad_y, grad_x = np.gradient(depth_np.astype(np.float32))
    boundary_score = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    boundary = top_fraction_mask(boundary_score, valid_np, 0.10)
    high_error = top_fraction_mask(np.abs(d0_norm_np - y_norm_np), valid_np, 0.20)
    far50 = valid_np & (depth_np > 50.0)
    luma = 0.2126 * rgb_preview_np[..., 0] + 0.7152 * rgb_preview_np[..., 1] + 0.0722 * rgb_preview_np[..., 2]
    dark = valid_np & (luma < 0.15)
    saturated = valid_np & (np.max(rgb_preview_np, axis=-1) > 0.95)
    masks = {
        "boundary_abs_rel": boundary,
        "dav2_high_error_abs_rel": high_error,
        "far50_abs_rel": far50,
        "dark_abs_rel": dark,
        "saturated_abs_rel": saturated,
    }
    final = {
        key: region_abs_rel(depth_np, aligned_final, mask, min_depth=min_depth, max_depth=max_depth)
        for key, mask in masks.items()
    }
    d0 = {
        key: region_abs_rel(depth_np, aligned_d0, mask, min_depth=min_depth, max_depth=max_depth)
        for key, mask in masks.items()
    }
    return final, d0
