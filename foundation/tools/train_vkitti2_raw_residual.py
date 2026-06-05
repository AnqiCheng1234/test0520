from __future__ import annotations

import argparse
import json
import logging
import math
import pprint
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.util.loss import build_training_target, robust_normalize_target_per_sample
from finetune_stf.util.utils import init_log
from foundation.engine.datasets import (
    DEPTH_TARGET_SPACE_CHOICES,
    FULLRES_EVEN_POLICY_CHOICES,
    RAW_STORAGE_FORMAT_CHOICES,
    RGB_INPUT_SPACE_CHOICES,
    VKITTI2Raw,
    validate_vkitti_raw_semantics,
)
from foundation.engine.models import build_raw_residual_dav2_model
from foundation.engine.transforms import (
    NOT_APPLICABLE,
    RAW_ADAPTER_PACKED_CHANNEL_ORDER,
    assert_unprocessing_summaries_compatible,
    assert_unprocessing_summary_matches_config,
    resolve_unprocessing_config,
)
from foundation.tools.eval_raw_residual_kitti import (
    KittiHalfresRawDataset,
    average_metrics as average_kitti_metrics,
    collect_eval_for_sample as collect_kitti_eval_for_sample,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3")
REGION_KEYS = (
    "boundary_abs_rel",
    "dav2_high_error_abs_rel",
    "far50_abs_rel",
    "dark_abs_rel",
    "saturated_abs_rel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VKITTI2 RAW residual refinement over frozen RGB DAv2.")
    parser.add_argument("--input-domain", required=True, choices=["raw4"])
    parser.add_argument("--model-input-tensor", required=True, choices=["raw"])
    parser.add_argument("--raw-storage-format", required=True, choices=RAW_STORAGE_FORMAT_CHOICES)
    parser.add_argument("--fullres-even-policy", required=True, choices=FULLRES_EVEN_POLICY_CHOICES)
    parser.add_argument("--rgb-input-space", required=True, choices=RGB_INPUT_SPACE_CHOICES)
    parser.add_argument("--depth-target-space", required=True, choices=DEPTH_TARGET_SPACE_CHOICES)
    parser.add_argument("--front-end", required=True, choices=["raw_to_base_rgb_ram3"])
    parser.add_argument("--encoder", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--vkitti-train-list", required=True)
    parser.add_argument("--vkitti-val-list", required=True)
    parser.add_argument("--eval-kitti", action="store_true", help="Also run KITTI val after each eval epoch.")
    parser.add_argument("--kitti-val-split", default=None)
    parser.add_argument("--kitti-base", default=None)
    parser.add_argument(
        "--kitti-eval-protocol",
        default=None,
        choices=["halfres_raw_canonical_even_pad_crop_affine_disp"],
    )
    parser.add_argument("--kitti-expected-val-samples", type=int, default=None)
    parser.add_argument("--kitti-num-workers", type=int, default=None)
    parser.add_argument("--max-kitti-val-samples", type=int, default=None)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--residual-feature-source", required=True, choices=["ffm_mid", "x3", "x3_ffm_mid"])
    parser.add_argument(
        "--residual-head-d0-mode",
        required=True,
        choices=["concat", "none"],
        help="Whether residual head input explicitly concatenates D0_norm.",
    )
    parser.add_argument("--residual-alpha", type=float, required=True)
    parser.add_argument("--d0-sign", type=int, required=True, choices=[-1, 1])
    parser.add_argument("--unprocessing-method", default="old_brooks_preset", choices=["old_brooks_preset", "raw_adapter_style"])
    parser.add_argument("--vkitti-unprocessing-preset", default=NOT_APPLICABLE)
    parser.add_argument("--vkitti-unprocessing-mix-weights", default=None)
    parser.add_argument("--randomize-unprocessing", action="store_true", default=None)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument("--raw-adapter-backend", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "analytic", "external_raw_rgb_cache"])
    parser.add_argument("--raw-adapter-cfa-pattern", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "RGGB"])
    parser.add_argument(
        "--raw-adapter-packed-channel-order",
        default=NOT_APPLICABLE,
        choices=[NOT_APPLICABLE, RAW_ADAPTER_PACKED_CHANNEL_ORDER],
    )
    parser.add_argument("--raw-adapter-rgb-transfer", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "srgb_piecewise"])
    parser.add_argument("--raw-adapter-inverse-tone", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "none", "global_0p15"])
    parser.add_argument("--raw-adapter-ccm", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "identity", "generic_d65"])
    parser.add_argument("--raw-adapter-red-gain-range", nargs=2, type=float, default=None)
    parser.add_argument("--raw-adapter-blue-gain-range", nargs=2, type=float, default=None)
    parser.add_argument("--raw-adapter-fixed-red-gain", type=float, default=None)
    parser.add_argument("--raw-adapter-fixed-blue-gain", type=float, default=None)
    parser.add_argument("--raw-adapter-variant-policy", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "normal", "dark", "over", "mix"])
    parser.add_argument("--raw-adapter-variant-weights", default=None)
    parser.add_argument("--raw-adapter-fixed-light-scale", type=float, default=None)
    parser.add_argument("--raw-adapter-dark-light-scale-range", nargs=2, type=float, default=None)
    parser.add_argument("--raw-adapter-over-light-scale-range", nargs=2, type=float, default=None)
    parser.add_argument("--raw-adapter-shot-noise", type=float, default=None)
    parser.add_argument("--raw-adapter-read-noise", type=float, default=None)
    parser.add_argument("--raw-adapter-noise-mean-mode", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "zero", "rawadapter_text"])
    parser.add_argument("--raw-adapter-black-level", type=float, default=None)
    parser.add_argument("--raw-adapter-white-level", type=float, default=None)
    parser.add_argument(
        "--raw-adapter-random-seed-policy",
        default=NOT_APPLICABLE,
        choices=[NOT_APPLICABLE, "dataloader_generator", "path_hash"],
    )
    parser.add_argument("--raw-adapter-external-raw-rgb-root", default=None)
    parser.add_argument("--raw-adapter-external-key", default=None)
    parser.add_argument("--raw-adapter-external-cache-space", default=None)
    parser.add_argument("--hflip-prob", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--heavy-save-path", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    expected = {
        "input_domain": "raw4",
        "model_input_tensor": "raw",
        "front_end": "raw_to_base_rgb_ram3",
    }
    for attr, value in expected.items():
        if getattr(args, attr) != value:
            raise ValueError(f"{attr} must be {value!r}, got {getattr(args, attr)!r}")
    validate_vkitti_raw_semantics(
        raw_storage_format=args.raw_storage_format,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    if args.randomize_unprocessing is None:
        raise ValueError("Pass either --randomize-unprocessing or --no-randomize-unprocessing explicitly.")
    resolved_unprocessing_config = resolve_unprocessing_config(vars(args))
    if (
        resolved_unprocessing_config["unprocessing_method"] == "raw_adapter_style"
        and resolved_unprocessing_config["raw_adapter_backend"] != "analytic"
    ):
        raise ValueError("Phase A only supports --raw-adapter-backend analytic for online training.")
    for key, value in resolved_unprocessing_config.items():
        setattr(args, key, value)
    args.resolved_unprocessing_config = dict(resolved_unprocessing_config)
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError(f"Invalid input size {(args.input_height, args.input_width)}")
    if not (0.0 < args.min_depth < args.max_depth):
        raise ValueError(f"Expected 0 < min_depth < max_depth, got {args.min_depth}, {args.max_depth}")
    if args.residual_alpha <= 0.0:
        raise ValueError(f"--residual-alpha must be positive, got {args.residual_alpha}")
    if args.residual_head_d0_mode == "none" and args.residual_feature_source == "x3_ffm_mid":
        raise ValueError(
            "--residual-head-d0-mode none with --residual-feature-source x3_ffm_mid is not part of the "
            "defined ablation matrix; use ffm_mid or x3, or confirm a new semantic experiment."
        )
    if not (0.0 <= args.hflip_prob <= 1.0):
        raise ValueError(f"--hflip-prob must be in [0, 1], got {args.hflip_prob}")
    if args.bs <= 0 or args.accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("bs, accum-steps, and epochs must be positive.")
    if args.save_interval <= 0 or args.eval_interval <= 0:
        raise ValueError("save-interval and eval-interval must be positive.")
    kitti_args = [
        args.kitti_val_split,
        args.kitti_base,
        args.kitti_eval_protocol,
        args.kitti_expected_val_samples,
        args.kitti_num_workers,
        args.max_kitti_val_samples,
    ]
    if not args.eval_kitti and any(value is not None for value in kitti_args):
        raise ValueError("KITTI eval parameters require --eval-kitti.")
    if args.max_kitti_val_samples is not None and args.max_kitti_val_samples <= 0:
        raise ValueError("--max-kitti-val-samples must be positive when provided.")
    if args.kitti_num_workers is not None and args.kitti_num_workers < 0:
        raise ValueError("--kitti-num-workers must be non-negative when provided.")
    if args.kitti_expected_val_samples is not None and args.kitti_expected_val_samples <= 0:
        raise ValueError("--kitti-expected-val-samples must be positive when provided.")
    if args.eval_kitti:
        if not args.kitti_val_split or not args.kitti_base:
            raise ValueError("--eval-kitti requires --kitti-val-split and --kitti-base.")
        if args.kitti_eval_protocol != "halfres_raw_canonical_even_pad_crop_affine_disp":
            raise ValueError(
                "--eval-kitti requires --kitti-eval-protocol "
                "halfres_raw_canonical_even_pad_crop_affine_disp."
            )
        kitti_geometry_expected = {
            "raw_storage_format": "synthetic_packed_bayer_4ch_halfres",
            "fullres_even_policy": "crop_bottom_to_even",
            "rgb_input_space": "halfres_2x2_area",
            "depth_target_space": "halfres_2x2_valid_mean",
            "input_height": 187,
            "input_width": 621,
        }
        for attr, value in kitti_geometry_expected.items():
            if getattr(args, attr) != value:
                raise ValueError(f"KITTI eval requires {attr}={value!r}, got {getattr(args, attr)!r}")
        if not Path(args.kitti_val_split).expanduser().is_file():
            raise FileNotFoundError(f"Missing KITTI val split: {args.kitti_val_split}")
        if not Path(args.kitti_base).expanduser().is_dir():
            raise FileNotFoundError(f"Missing KITTI base directory: {args.kitti_base}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    optimizer: AdamW,
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


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epoch: int,
    amp_dtype: torch.dtype,
    logger: logging.Logger,
) -> dict[str, Any]:
    model.eval()
    final_metrics: list[dict[str, float]] = []
    d0_metrics: list[dict[str, float]] = []
    final_regions: list[dict[str, float]] = []
    d0_regions: list[dict[str, float]] = []
    diagnostics: list[dict[str, float]] = []
    processed = 0
    start = time.time()

    logger.info("[EVAL] start epoch=%d max_val_samples=%s", epoch, args.max_val_samples)
    for batch in dataloader:
        if args.max_val_samples is not None and processed >= args.max_val_samples:
            break

        image = batch["image"].to(device, non_blocking=True).float()
        raw = batch["raw"].to(device, non_blocking=True).float()
        depth = batch["depth"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        valid_mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)
        if int(valid_mask[0].sum().item()) < 128:
            continue

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            out = model({"image": image, "raw": raw, "valid_mask": valid_mask})
        pred = out["pred"].float()
        d0_disp = (float(args.d0_sign) * out["D0"].float()).detach()

        inv_gt = build_training_target(depth.float(), valid_mask, target_space="metric_depth")
        y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_mask, min_valid_pixels=128)

        depth_np = depth[0].detach().cpu().numpy().astype(np.float32)
        valid_np = valid_mask[0].detach().cpu().numpy().astype(bool)
        pred_np = pred[0].detach().cpu().numpy().astype(np.float32)
        d0_np = d0_disp[0].detach().cpu().numpy().astype(np.float32)

        aligned_final, _ = affine_align_disp(depth_np, pred_np, valid_np)
        aligned_d0, _ = affine_align_disp(depth_np, d0_np, valid_np)
        metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
        metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
        if metrics_final is None or metrics_d0 is None:
            continue

        rgb_preview = batch["rgb_preview"][0].permute(1, 2, 0).numpy().astype(np.float32)
        region_final, region_d0 = sample_region_metrics(
            depth_np=depth_np,
            valid_np=valid_np,
            aligned_final=aligned_final,
            aligned_d0=aligned_d0,
            d0_norm_np=out["D0_norm"][0].float().detach().cpu().numpy(),
            y_norm_np=y_norm[0].float().detach().cpu().numpy(),
            rgb_preview_np=rgb_preview,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        final_metrics.append({key: float(metrics_final[key]) for key in METRIC_KEYS if key in metrics_final})
        d0_metrics.append({key: float(metrics_d0[key]) for key in METRIC_KEYS if key in metrics_d0})
        final_regions.append(region_final)
        d0_regions.append(region_d0)

        gate = out["gate"].float()
        delta = out["delta"].float()
        gate_delta = gate * delta
        diagnostics.append(
            {
                "mean_gate": float(gate[valid_mask].mean().detach().item()),
                "max_gate": float(gate[valid_mask].max().detach().item()),
                "mean_abs_delta": float(delta[valid_mask].abs().mean().detach().item()),
                "mean_abs_gate_delta": float(gate_delta[valid_mask].abs().mean().detach().item()),
                "mean_abs_final_minus_d0_norm": float((pred - out["D0_norm"].float())[valid_mask].abs().mean().detach().item()),
            }
        )
        processed += 1

    if processed == 0:
        raise RuntimeError("Validation produced zero valid samples.")

    overall_final = average_dicts(final_metrics, METRIC_KEYS)
    overall_d0 = average_dicts(d0_metrics, METRIC_KEYS)
    region_final = average_dicts(final_regions, REGION_KEYS)
    region_d0 = average_dicts(d0_regions, REGION_KEYS)
    diag = average_dicts(
        diagnostics,
        ["mean_gate", "max_gate", "mean_abs_delta", "mean_abs_gate_delta", "mean_abs_final_minus_d0_norm"],
    )
    delta = {
        "final_abs_rel_minus_D0_abs_rel": (
            None
            if overall_final["abs_rel"] is None or overall_d0["abs_rel"] is None
            else overall_final["abs_rel"] - overall_d0["abs_rel"]
        ),
        "final_d1_minus_D0_d1": (
            None
            if overall_final["d1"] is None or overall_d0["d1"] is None
            else overall_final["d1"] - overall_d0["d1"]
        ),
    }
    region_delta = {
        key: None if region_final[key] is None or region_d0[key] is None else region_final[key] - region_d0[key]
        for key in REGION_KEYS
    }
    summary = {
        "epoch": int(epoch),
        "samples": int(processed),
        "max_val_samples": args.max_val_samples,
        "overall": {"final": overall_final, "D0": overall_d0, "delta": delta},
        "region": {"final": region_final, "D0": region_d0, "delta": region_delta},
        "diagnostics": diag,
        "elapsed_seconds": float(time.time() - start),
    }
    logger.info(
        "[EVAL] done epoch=%d samples=%d final_abs_rel=%.5f D0_abs_rel=%.5f delta_abs_rel=%.5f final_d1=%.5f D0_d1=%.5f elapsed=%s",
        epoch,
        processed,
        float(overall_final["abs_rel"]),
        float(overall_d0["abs_rel"]),
        float(delta["final_abs_rel_minus_D0_abs_rel"]),
        float(overall_final["d1"]),
        float(overall_d0["d1"]),
        format_seconds(summary["elapsed_seconds"]),
    )
    return summary


def collate_single_sample(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError(f"Expected batch_size=1 for KITTI eval, got {len(batch)}")
    return batch[0]


def build_kitti_val_loader(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[KittiHalfresRawDataset | None, DataLoader | None]:
    if not args.eval_kitti:
        return None, None

    dataset = KittiHalfresRawDataset(
        filelist_path=args.kitti_val_split,
        kitti_base=args.kitti_base,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        unprocessing_config=args.resolved_unprocessing_config,
    )
    assert_unprocessing_summary_matches_config(
        dataset.describe_unprocessing(),
        args.resolved_unprocessing_config,
        context="KITTI val dataset vs training resolved unprocessing config",
    )
    if args.kitti_expected_val_samples is not None and len(dataset) != int(args.kitti_expected_val_samples):
        raise RuntimeError(
            f"Expected KITTI val length {int(args.kitti_expected_val_samples)}, got {len(dataset)}"
        )
    workers = int(args.kitti_num_workers) if args.kitti_num_workers is not None else max(min(args.num_workers, 2), 0)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_single_sample,
        persistent_workers=workers > 0,
    )
    return dataset, loader


def evaluate_kitti_model(
    model: torch.nn.Module,
    dataset: KittiHalfresRawDataset,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epoch: int,
    amp_dtype: torch.dtype,
    logger: logging.Logger,
    output_dir: Path,
) -> dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_visit = len(dataset) if args.max_kitti_val_samples is None else min(int(args.max_kitti_val_samples), len(dataset))
    amp_enabled = bool(args.amp) and device.type == "cuda"
    config = dict(vars(args))
    rows: list[dict[str, Any]] = []
    ok_metric_rows: list[dict[str, Any]] = []
    start = time.time()
    per_sample_path = output_dir / "per_sample.jsonl"

    logger.info(
        "[EVAL][KITTI] start epoch=%d dataset_samples=%d max_visit=%d max_kitti_val_samples=%s",
        epoch,
        len(dataset),
        max_visit,
        args.max_kitti_val_samples,
    )
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for visited, sample in enumerate(dataloader):
            if visited >= max_visit:
                break
            row, _ = collect_kitti_eval_for_sample(
                sample=sample,
                model=model,
                config=config,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                args=args,
                collect_panel=False,
            )
            row["epoch"] = int(epoch)
            rows.append(row)
            if row["status"] == "ok":
                ok_metric_rows.append(row)
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
            if (visited + 1) % 50 == 0 or visited + 1 == max_visit:
                logger.info(
                    "[EVAL][KITTI] processed=%d/%d ok=%d elapsed=%s",
                    visited + 1,
                    max_visit,
                    len(ok_metric_rows),
                    format_seconds(time.time() - start),
                )

    if not ok_metric_rows:
        raise RuntimeError("KITTI eval produced zero ok samples.")

    final_metrics = [row["final"] for row in ok_metric_rows]
    d0_metrics = [row["D0"] for row in ok_metric_rows]
    overall_final = average_kitti_metrics(final_metrics)
    overall_d0 = average_kitti_metrics(d0_metrics)
    delta = {
        "final_abs_rel_minus_D0_abs_rel": (
            None
            if overall_final["abs_rel"] is None or overall_d0["abs_rel"] is None
            else overall_final["abs_rel"] - overall_d0["abs_rel"]
        ),
        "final_d1_minus_D0_d1": (
            None
            if overall_final["d1"] is None or overall_d0["d1"] is None
            else overall_final["d1"] - overall_d0["d1"]
        ),
    }
    status_counts = Counter(str(row["status"]) for row in rows)
    elapsed_seconds = time.time() - start
    summary = {
        "dataset": "kitti_val_halfres_raw",
        "epoch": int(epoch),
        "dataset_samples": int(len(dataset)),
        "visited_samples": int(len(rows)),
        "samples": int(len(ok_metric_rows)),
        "max_val_samples": args.max_kitti_val_samples,
        "kitti_val_split": str(Path(args.kitti_val_split).expanduser().resolve()),
        "kitti_base": str(Path(args.kitti_base).expanduser().resolve()),
        "eval_protocol": args.kitti_eval_protocol,
        "validate_run_config_branch": str(args.unprocessing_method),
        "unprocessing_policy_source": "training_resolved_config",
        "unprocessing_policy": dataset.describe_unprocessing(),
        "note": (
            "KITTI val is evaluated with canonical_even_pad_crop to match the VKITTI-trained "
            "fixed 187x621 RAW residual model; scores are not KITTI public benchmark settings."
        ),
        "geometry_policy": dataset.describe_geometry(),
        "status_counts": dict(status_counts),
        "overall": {"final": overall_final, "D0": overall_d0, "delta": delta},
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_visited_sample": float(elapsed_seconds / max(len(rows), 1)),
        "per_sample_path": str(per_sample_path),
    }
    save_json(output_dir / "metrics.json", summary)
    logger.info(
        "[EVAL][KITTI] done epoch=%d visited=%d ok=%d final_abs_rel=%.5f D0_abs_rel=%.5f "
        "delta_abs_rel=%.5f final_d1=%.5f D0_d1=%.5f elapsed=%s",
        epoch,
        len(rows),
        len(ok_metric_rows),
        float(overall_final["abs_rel"]),
        float(overall_d0["abs_rel"]),
        float(delta["final_abs_rel_minus_D0_abs_rel"]),
        float(overall_final["d1"]),
        float(overall_d0["d1"]),
        format_seconds(elapsed_seconds),
    )
    return summary


def build_loaders(args: argparse.Namespace) -> tuple[VKITTI2Raw, VKITTI2Raw, DataLoader, DataLoader]:
    train_unprocessing_config = dict(args.resolved_unprocessing_config)
    val_unprocessing_config = dict(args.resolved_unprocessing_config)
    if train_unprocessing_config["unprocessing_method"] == "old_brooks_preset":
        val_unprocessing_config["randomize_unprocessing"] = False
        val_unprocessing_config = resolve_unprocessing_config(val_unprocessing_config)

    train_dataset = VKITTI2Raw(
        filelist_path=args.vkitti_train_list,
        mode="train",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        unprocessing_config=train_unprocessing_config,
        hflip_prob=args.hflip_prob,
        include_rgb_input=True,
        include_rgb_preview=False,
        raw_storage_format=args.raw_storage_format,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    val_dataset = VKITTI2Raw(
        filelist_path=args.vkitti_val_list,
        mode="val",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        unprocessing_config=val_unprocessing_config,
        hflip_prob=0.0,
        include_rgb_input=True,
        include_rgb_preview=True,
        raw_storage_format=args.raw_storage_format,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(min(args.num_workers, 2), 0),
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    assert_unprocessing_summaries_compatible(
        train_dataset.describe_unprocessing(),
        val_dataset.describe_unprocessing(),
        context="VKITTI train vs VKITTI val unprocessing",
    )
    assert_unprocessing_summary_matches_config(
        train_dataset.describe_unprocessing(),
        args.resolved_unprocessing_config,
        context="VKITTI train dataset vs training resolved unprocessing config",
    )
    assert_unprocessing_summary_matches_config(
        val_dataset.describe_unprocessing(),
        args.resolved_unprocessing_config,
        context="VKITTI val dataset vs training resolved unprocessing config",
    )
    return train_dataset, val_dataset, train_loader, val_loader


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training entry expects CUDA.")

    save_path = Path(args.save_path).expanduser().resolve()
    heavy_save_path = Path(args.heavy_save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    heavy_save_path.mkdir(parents=True, exist_ok=True)

    logger = init_log("vkitti2_raw_residual", logging.INFO) or logging.getLogger("vkitti2_raw_residual")
    logger.propagate = False
    attach_file_logger(logger, save_path / "train.log")
    logger.info("%s\n", pprint.pformat({**vars(args), "device": str(device)}))

    cudnn.enabled = True
    cudnn.benchmark = True
    set_random_seed(args.seed)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    kitti_val_dataset, kitti_val_loader = build_kitti_val_loader(args, device)
    train_unprocessing_desc = train_dataset.describe_unprocessing()
    val_unprocessing_desc = val_dataset.describe_unprocessing()
    if args.unprocessing_method == "raw_adapter_style":
        expected_hash = args.resolved_unprocessing_config["raw_adapter_config_hash"]
        for label, desc in (("train", train_unprocessing_desc), ("vkitti_val", val_unprocessing_desc)):
            if desc.get("raw_adapter_config_hash") != expected_hash:
                raise RuntimeError(
                    f"{label} raw_adapter_config_hash={desc.get('raw_adapter_config_hash')!r} "
                    f"does not match config hash {expected_hash!r}"
                )
    config_payload = dict(vars(args))
    vkitti_val_geometry = val_dataset.describe_geometry()
    config_payload["dataset_geometry"] = {
        "train": train_dataset.describe_geometry(),
        "val": vkitti_val_geometry,
        "vkitti_val": vkitti_val_geometry,
    }
    config_payload["eval_protocol"] = {
        "vkitti_val": "per_image_affine_disp_depth_anything_v2",
        "kitti_val": args.kitti_eval_protocol if args.eval_kitti else "disabled",
    }
    config_payload["unprocessing_policy"] = {
        "train": train_unprocessing_desc,
        "vkitti_val": val_unprocessing_desc,
    }
    if kitti_val_dataset is not None:
        config_payload["dataset_geometry"]["kitti_val"] = kitti_val_dataset.describe_geometry()
        config_payload["unprocessing_policy"]["kitti_val"] = kitti_val_dataset.describe_unprocessing()
    save_json(save_path / "config.json", config_payload)
    base_model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model = build_raw_residual_dav2_model(
        base_model,
        residual_feature_source=args.residual_feature_source,
        residual_head_d0_mode=args.residual_head_d0_mode,
        residual_alpha=args.residual_alpha,
        d0_sign=args.d0_sign,
        sensor_hw=(args.input_height, args.input_width),
        backbone_hw=None,
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(strip_module_prefix(resolve_model_state(resume)), strict=True)
        start_epoch = int(resume.get("epoch", -1)) + 1
        global_step = int(resume.get("global_step", 0))
        logger.info("[INIT] resumed model from %s", args.resume_from)
    else:
        ckpt_obj = torch.load(args.pretrained_from, map_location="cpu")
        model.load_base_dav2_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)))
        logger.info("[INIT] loaded frozen DAv2 weights from %s", args.pretrained_from)

    model = model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")
    total_params, trainable_param_count = count_parameters(model)
    logger.info(
        "[MODEL] total_params=%d trainable_params=%d frozen_params=%d residual_feature_source=%s "
        "residual_head_d0_mode=%s d0_sign=%d",
        total_params,
        trainable_param_count,
        total_params - trainable_param_count,
        args.residual_feature_source,
        args.residual_head_d0_mode,
        args.d0_sign,
    )
    logger.info("[DATASET] train_samples=%d vkitti_val_samples=%d", len(train_dataset), len(val_dataset))
    if kitti_val_dataset is not None:
        logger.info("[DATASET][KITTI] val_samples=%d source_shapes=%s", len(kitti_val_dataset), kitti_val_dataset.source_shape_counts)
    logger.info("[DATASET] geometry=%s", config_payload["dataset_geometry"])
    logger.info("[DATASET] train_unprocessing=%s", train_unprocessing_desc)
    logger.info("[DATASET] vkitti_val_unprocessing=%s", val_unprocessing_desc)
    if kitti_val_dataset is not None:
        logger.info("[DATASET][KITTI] val_unprocessing=%s", kitti_val_dataset.describe_unprocessing())

    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    kitti_val_history: list[dict[str, Any]] = []
    best_abs_rel = float("inf")
    best_kitti_abs_rel = float("inf")
    steps_per_epoch = len(train_loader)
    if args.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        last_log_time = epoch_start
        running: dict[str, float] = {}
        used_steps = 0
        optimizer_steps = 0
        pending_gradients = False

        for step_idx, batch in enumerate(train_loader):
            if step_idx >= steps_per_epoch:
                break
            image = batch["image"].to(device, non_blocking=True).float()
            raw = batch["raw"].to(device, non_blocking=True).float()
            depth = batch["depth"].to(device, non_blocking=True).float()
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()

            if epoch == start_epoch and step_idx == 0:
                logger.info(
                    "[BATCH] image=%s raw=%s depth=%s valid=%s samples=%s",
                    tuple(image.shape),
                    tuple(raw.shape),
                    tuple(depth.shape),
                    tuple(valid_mask.shape),
                    batch["sample_name"][: min(2, len(batch["sample_name"]))],
                )

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                out = model({"image": image, "raw": raw, "valid_mask": valid_mask})
                loss, loss_info = compute_residual_loss(out, depth, valid_mask)

            if loss_info["used_samples"] > 0:
                accum_denom = min(args.accum_steps, steps_per_epoch - step_idx)
                loss_scaled = loss / float(accum_denom)
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                pending_gradients = True
                used_steps += 1
                for key, value in loss_info.items():
                    if isinstance(value, (int, float)):
                        running[key] = running.get(key, 0.0) + float(value)

            is_boundary = ((step_idx + 1) % args.accum_steps == 0) or ((step_idx + 1) >= steps_per_epoch)
            if is_boundary and pending_gradients:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                pending_gradients = False
            elif is_boundary:
                optimizer.zero_grad(set_to_none=True)

            if (step_idx + 1) % args.log_interval == 0 or (step_idx + 1) == steps_per_epoch:
                now = time.time()
                denom = max(used_steps, 1)
                step_per_sec = (step_idx + 1) / max(now - epoch_start, 1e-6)
                eta_seconds = (steps_per_epoch - (step_idx + 1)) / max(step_per_sec, 1e-6)
                x3 = out["x3"].float().detach()
                ffm_mid = out["ffm_mid"].float().detach()
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d opt_step=%d loss_total=%.5f L_depth=%.5f L_grad=%.5f "
                    "L_keep=%.5f L_res=%.5f L_gate=%.5f L_gate_sup=%.5f mean_gate=%.5f max_gate=%.5f "
                    "mean_abs_delta=%.5f mean_abs_gate_delta=%.5f mean_abs_Dfinal_minus_D0norm=%.5f "
                    "x3_mean=%.5f x3_std=%.5f x3_min=%.5f x3_max=%.5f ffm_mid_mean=%.5f ffm_mid_std=%.5f "
                    "lr=%.7f max_mem_mb=%.0f used=%d skipped=%d step_per_sec=%.2f elapsed=%s eta=%s",
                    epoch,
                    step_idx + 1,
                    steps_per_epoch,
                    optimizer_steps,
                    running.get("loss_total", 0.0) / denom,
                    running.get("L_depth", 0.0) / denom,
                    running.get("L_grad", 0.0) / denom,
                    running.get("L_keep", 0.0) / denom,
                    running.get("L_res", 0.0) / denom,
                    running.get("L_gate", 0.0) / denom,
                    running.get("L_gate_sup", 0.0) / denom,
                    running.get("mean_gate", 0.0) / denom,
                    running.get("max_gate", 0.0) / denom,
                    running.get("mean_abs_delta", 0.0) / denom,
                    running.get("mean_abs_gate_delta", 0.0) / denom,
                    running.get("mean_abs_final_minus_d0_norm", 0.0) / denom,
                    float(x3.mean().item()),
                    float(x3.std(unbiased=False).item()),
                    float(x3.min().item()),
                    float(x3.max().item()),
                    float(ffm_mid.mean().item()),
                    float(ffm_mid.std(unbiased=False).item()),
                    optimizer.param_groups[0]["lr"],
                    max_mem,
                    int(loss_info["used_samples"]),
                    int(loss_info["skipped_samples"]),
                    step_per_sec,
                    format_seconds(now - epoch_start),
                    format_seconds(eta_seconds),
                )
                last_log_time = now

        denom = max(used_steps, 1)
        train_summary = {
            "epoch": int(epoch),
            "used_steps": int(used_steps),
            "optimizer_steps": int(optimizer_steps),
            "elapsed_seconds": float(time.time() - epoch_start),
        }
        for key, value in running.items():
            train_summary[key] = float(value / denom)
        train_history.append(train_summary)
        save_json(save_path / "train_loss_summary.json", {"epochs": train_history})
        logger.info(
            "[EPOCH] done epoch=%d avg_loss=%.5f used_steps=%d elapsed=%s",
            epoch,
            train_summary.get("loss_total", 0.0),
            used_steps,
            format_seconds(train_summary["elapsed_seconds"]),
        )

        val_summary = None
        kitti_val_summary = None
        if ((epoch + 1) % args.eval_interval) == 0:
            val_summary = evaluate_model(
                model,
                val_loader,
                args,
                device,
                epoch=epoch,
                amp_dtype=amp_dtype,
                logger=logger,
            )
            val_history.append(val_summary)
            save_json(save_path / "val_metrics.json", {"epochs": val_history, "latest": val_summary})
            current_abs_rel = val_summary["overall"]["final"]["abs_rel"]
            if current_abs_rel is not None and float(current_abs_rel) < best_abs_rel:
                best_abs_rel = float(current_abs_rel)
                save_json(save_path / "best_val_metrics.json", val_summary)
                if args.save_best_checkpoint:
                    save_checkpoint(
                        heavy_save_path / "best_abs_rel.pth",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        args=args,
                        train_summary=train_summary,
                        val_summary=val_summary,
                    )
            if args.eval_kitti:
                if kitti_val_dataset is None or kitti_val_loader is None:
                    raise RuntimeError("KITTI eval was enabled but the KITTI val loader was not built.")
                kitti_val_summary = evaluate_kitti_model(
                    model,
                    kitti_val_dataset,
                    kitti_val_loader,
                    args,
                    device,
                    epoch=epoch,
                    amp_dtype=amp_dtype,
                    logger=logger,
                    output_dir=save_path / "kitti_val" / f"epoch_{epoch:02d}",
                )
                kitti_val_history.append(kitti_val_summary)
                save_json(
                    save_path / "kitti_val_metrics.json",
                    {"epochs": kitti_val_history, "latest": kitti_val_summary},
                )
                current_kitti_abs_rel = kitti_val_summary["overall"]["final"]["abs_rel"]
                if current_kitti_abs_rel is not None and float(current_kitti_abs_rel) < best_kitti_abs_rel:
                    best_kitti_abs_rel = float(current_kitti_abs_rel)
                    save_json(save_path / "best_kitti_val_metrics.json", kitti_val_summary)

        if ((epoch + 1) % args.save_interval) == 0:
            save_checkpoint(
                heavy_save_path / f"epoch_{epoch:02d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                train_summary=train_summary,
                val_summary=val_summary,
            )
        save_checkpoint(
            heavy_save_path / "latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            train_summary=train_summary,
            val_summary=val_summary,
        )

    save_json(
        save_path / "run_summary.json",
        {
            "config": config_payload,
            "train": train_history,
            "val": val_history,
            "vkitti_val": val_history,
            "kitti_val": kitti_val_history,
            "best_abs_rel": float_or_none(best_abs_rel),
            "best_kitti_abs_rel": float_or_none(best_kitti_abs_rel),
            "heavy_save_path": str(heavy_save_path),
        },
    )


if __name__ == "__main__":
    main()
