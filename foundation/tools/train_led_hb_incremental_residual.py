#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import pprint
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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
from foundation.engine.datasets.led_hb import (
    LED_DEPTH_LABEL_CHOICES,
    LED_DEPTH_TARGET_SPACE_CHOICES,
    LED_DEPTH_UNIT_CHOICES,
    LED_GEOMETRY_MODE_CHOICES,
    LED_RAW_STORAGE_FORMAT_CHOICES,
    LED_RGB_INPUT_SPACE_CHOICES,
    LEDHBHalfresRGBDepth,
    LEDHBRaw,
    validate_led_hb_raw_semantics,
    validate_led_hb_rgb_depth_semantics,
)
from foundation.engine.models import (
    build_c2_frozen_incremental_residual_model,
    build_dav2_residual_control_model,
)
from foundation.engine.models.dav2_incremental_residual import (
    DELTA_CONDITIONS,
    FEATURE_ABLATION_MODES,
    FEATURE_ABLATION_SCOPES,
    GATE_CONDITIONS,
    INCREMENTAL_FEATURE_SOURCES,
    RAW_FEATURE_ENCODER_TRAINABLE,
    validate_incremental_contract,
)
from foundation.engine.transforms import (
    NOT_APPLICABLE,
    RAW_ADAPTER_PACKED_CHANNEL_ORDER,
    assert_unprocessing_summaries_compatible,
    assert_unprocessing_summary_matches_config,
    resolve_unprocessing_config,
)
from foundation.tools.eval_led_hb_d0 import LED_REGION_KEYS, sample_led_d0_region_metrics
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    attach_file_logger,
    average_dicts,
    compute_incremental_residual_loss,
    count_parameters,
    float_or_none,
    format_seconds,
    lowpass_avgpool,
    mean_finite,
    region_abs_rel,
    resolve_model_state,
    save_checkpoint,
    save_json,
    strip_module_prefix,
    top_fraction_mask,
)
from foundation.tools.train_vkitti2_incremental_residual import (
    add_dataset_raw_donor_if_needed,
    feature_ablation_active,
    feature_ablation_mode,
    forward_incremental_model,
    n6_eval_batch_size,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
METHOD_IDS = ("N2", "N3", "N5", "N7")
FRONT_END_CHOICES = ("c2_frozen_raw_ram_incremental", "c2_frozen_rgb_incremental", "c2_frozen_d1_incremental")
EVAL_PROTOCOL_CHOICES = ("per_image_affine_disp_depth_anything_v2",)
LED_NSERIES_REGION_KEYS = (
    "boundary_abs_rel",
    "d0_high_error_abs_rel",
    "d1_high_error_abs_rel",
    "near_1_20_abs_rel",
    "mid_20_50_abs_rel",
    "far50_abs_rel",
    "far100_abs_rel",
    "dark_q20_abs_rel",
    "saturated_abs_rel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LED-HB C2-frozen N-series incremental residual training.")
    parser.add_argument("--method-id", required=True, choices=METHOD_IDS)
    parser.add_argument("--dataset-name", required=True, choices=["led_hb"])
    parser.add_argument("--illumination", required=True, choices=["HB"])
    parser.add_argument("--input-domain", required=True, choices=["raw4", "rgb"])
    parser.add_argument("--model-input-tensor", required=True, choices=["raw", "image"])
    parser.add_argument("--dataset-geometry-mode", required=True, choices=LED_GEOMETRY_MODE_CHOICES)
    parser.add_argument("--raw-storage-format", required=True, choices=LED_RAW_STORAGE_FORMAT_CHOICES + (NOT_APPLICABLE,))
    parser.add_argument("--rgb-input-space", required=True, choices=LED_RGB_INPUT_SPACE_CHOICES)
    parser.add_argument("--depth-target-space", required=True, choices=LED_DEPTH_TARGET_SPACE_CHOICES)
    parser.add_argument("--depth-label", required=True, choices=LED_DEPTH_LABEL_CHOICES)
    parser.add_argument("--depth-unit", required=True, choices=LED_DEPTH_UNIT_CHOICES)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--eval-protocol", required=True, choices=EVAL_PROTOCOL_CHOICES)
    parser.add_argument("--front-end", required=True, choices=FRONT_END_CHOICES)
    parser.add_argument("--encoder", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--c2-checkpoint", required=True)
    parser.add_argument("--c2-run-dir", required=True)
    parser.add_argument("--led-train-list", required=True)
    parser.add_argument("--led-val-list", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--incremental-feature-source", required=True, choices=INCREMENTAL_FEATURE_SOURCES)
    parser.add_argument("--delta-condition", required=True, choices=DELTA_CONDITIONS)
    parser.add_argument("--gate-condition", required=True, choices=GATE_CONDITIONS)
    parser.add_argument("--raw-feature-encoder-trainable", required=True, choices=RAW_FEATURE_ENCODER_TRAINABLE)
    parser.add_argument("--residual-alpha", type=float, required=True)
    parser.add_argument("--d0-sign", type=int, required=True, choices=[-1, 1])
    parser.add_argument("--lambda-lp", type=float, required=True)
    parser.add_argument("--lowpass-kernel", type=int, required=True)
    parser.add_argument("--q-good", type=float, required=True)
    parser.add_argument("--train-feature-ablation-mode", default="true", choices=["true", "none", "zero", "mean"])
    parser.add_argument("--eval-feature-ablation-mode", default="true", choices=FEATURE_ABLATION_MODES)
    parser.add_argument("--feature-ablation-scope", default="both", choices=FEATURE_ABLATION_SCOPES)
    parser.add_argument("--feature-ablation-key", default="x3", choices=["x3"])
    parser.add_argument("--feature-ablation-seed", type=int, default=42)
    parser.add_argument("--feature-ablation-donor-offset", type=int, default=1)
    parser.add_argument("--n6-output-dir", default=None)
    parser.add_argument("--experiment-label", default=None)
    parser.add_argument("--lambda-final", type=float, required=True)
    parser.add_argument("--lambda-boundary", type=float, required=True)
    parser.add_argument("--lambda-grad", type=float, required=True)
    parser.add_argument("--lambda-keep-good-d1", type=float, required=True)
    parser.add_argument("--lambda-gate-sparse", type=float, required=True)
    parser.add_argument("--lambda-lowfreq-loss", type=float, required=True)
    parser.add_argument("--lambda-invalid-keep", type=float, required=True)
    parser.add_argument("--unprocessing-method", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "raw_adapter_style"])
    parser.add_argument("--vkitti-unprocessing-preset", default=NOT_APPLICABLE)
    parser.add_argument("--vkitti-unprocessing-mix-weights", default=None)
    parser.add_argument("--randomize-unprocessing", action="store_true", default=None)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument("--raw-adapter-backend", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "analytic"])
    parser.add_argument("--raw-adapter-cfa-pattern", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "RGGB"])
    parser.add_argument("--raw-adapter-packed-channel-order", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, RAW_ADAPTER_PACKED_CHANNEL_ORDER])
    parser.add_argument("--raw-adapter-rgb-transfer", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "srgb_piecewise"])
    parser.add_argument("--raw-adapter-inverse-tone", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "none", "global_0p15"])
    parser.add_argument("--raw-adapter-ccm", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "identity", "generic_d65"])
    parser.add_argument("--raw-adapter-red-gain-range", nargs="+", default=None)
    parser.add_argument("--raw-adapter-blue-gain-range", nargs="+", default=None)
    parser.add_argument("--raw-adapter-fixed-red-gain", default=None)
    parser.add_argument("--raw-adapter-fixed-blue-gain", default=None)
    parser.add_argument("--raw-adapter-fixed-light-scale", default=None)
    parser.add_argument("--raw-adapter-dark-light-scale-range", nargs="+", default=None)
    parser.add_argument("--raw-adapter-over-light-scale-range", nargs="+", default=None)
    parser.add_argument("--raw-adapter-shot-noise", default=None)
    parser.add_argument("--raw-adapter-read-noise", default=None)
    parser.add_argument("--raw-adapter-noise-mean-mode", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "zero", "rawadapter_text"])
    parser.add_argument("--raw-adapter-black-level", default=None)
    parser.add_argument("--raw-adapter-white-level", default=None)
    parser.add_argument("--raw-adapter-random-seed-policy", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "dataloader_generator", "path_hash"])
    parser.add_argument("--raw-adapter-external-raw-rgb-root", default=None)
    parser.add_argument("--raw-adapter-external-key", default=None)
    parser.add_argument("--raw-adapter-external-cache-space", default=None)
    parser.add_argument("--raw-adapter-variant-policy", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "normal", "dark", "over", "mix"])
    parser.add_argument("--raw-adapter-variant-weights", default=None)
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


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value == NOT_APPLICABLE:
        return None
    return float(value)


def _none_or_float_range(value: Any) -> list[float] | None:
    if value is None:
        return None
    values = list(value)
    if len(values) == 1 and str(values[0]) == NOT_APPLICABLE:
        return None
    if len(values) != 2:
        raise ValueError(f"Expected two floats or {NOT_APPLICABLE!r}, got {value!r}")
    return [float(values[0]), float(values[1])]


def normalize_optional_numeric_args(args: argparse.Namespace) -> None:
    for key in (
        "raw_adapter_fixed_red_gain",
        "raw_adapter_fixed_blue_gain",
        "raw_adapter_fixed_light_scale",
        "raw_adapter_shot_noise",
        "raw_adapter_read_noise",
        "raw_adapter_black_level",
        "raw_adapter_white_level",
    ):
        setattr(args, key, _none_or_float(getattr(args, key)))
    for key in (
        "raw_adapter_red_gain_range",
        "raw_adapter_blue_gain_range",
        "raw_adapter_dark_light_scale_range",
        "raw_adapter_over_light_scale_range",
    ):
        setattr(args, key, _none_or_float_range(getattr(args, key)))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_path_str(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve()) if value not in (None, "") else ""


def _meta_value(meta: Mapping[str, Any], key: str) -> Any:
    if key == "led_geometry_mode":
        return meta.get("led_geometry_mode", meta.get("dataset_geometry_mode"))
    return meta.get(key)


def validate_c2_metadata(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = Path(args.c2_checkpoint).expanduser().resolve()
    run_dir = Path(args.c2_run_dir).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing C2 checkpoint: {ckpt_path}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing C2 run dir: {run_dir}")
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = ckpt_obj.get("args") if isinstance(ckpt_obj, dict) else None
    run_config = load_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else None
    meta = dict(run_config or {})
    if isinstance(ckpt_args, dict):
        meta.update(ckpt_args)
    if not meta:
        raise ValueError("C2 checkpoint and run dir do not contain metadata; refusing N-series run.")
    if str(meta.get("experiment_id", "")).lower() != "c2":
        raise ValueError(f"C2 experiment_id must be C2, got {meta.get('experiment_id')!r}")
    expected = {
        "dataset_name": "led_hb",
        "illumination": "HB",
        "led_geometry_mode": args.dataset_geometry_mode,
        "input_height": 378,
        "input_width": 672,
        "min_depth": 1.0,
        "max_depth": 200.0,
        "depth_label": "distance_to_image_plane",
        "depth_unit": "meter",
        "encoder": args.encoder,
        "d0_sign": args.d0_sign,
        "pretrained_from": args.pretrained_from,
        "residual_alpha": args.residual_alpha,
    }
    for key, expected_value in expected.items():
        actual = _meta_value(meta, key)
        if actual is None:
            raise ValueError(f"C2 metadata missing required key {key!r}")
        if key == "pretrained_from":
            if _as_path_str(actual) != _as_path_str(expected_value):
                raise ValueError(f"C2 metadata {key} must match current setting: {actual!r} vs {expected_value!r}")
        elif isinstance(expected_value, float):
            if not math.isclose(float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"C2 metadata {key} must be {expected_value!r}, got {actual!r}")
        elif isinstance(expected_value, int):
            if int(actual) != int(expected_value):
                raise ValueError(f"C2 metadata {key} must be {expected_value!r}, got {actual!r}")
        else:
            if str(actual) != str(expected_value):
                raise ValueError(f"C2 metadata {key} must be {expected_value!r}, got {actual!r}")
    for key in ("led_train_list", "led_val_list"):
        actual = meta.get(key)
        expected_value = getattr(args, key)
        if actual is None:
            raise ValueError(f"C2 metadata missing required key {key!r}")
        if _as_path_str(actual) != _as_path_str(expected_value):
            raise ValueError(f"C2 metadata {key} must match current setting: {actual!r} vs {expected_value!r}")
    return {"checkpoint_args": ckpt_args, "run_config": run_config, "source": "merged_config_checkpoint"}


def method_is_raw(args: argparse.Namespace) -> bool:
    return str(args.incremental_feature_source) == "x3"


def validate_args(args: argparse.Namespace) -> None:
    normalize_optional_numeric_args(args)
    args.method_id = str(args.method_id).upper()
    validate_incremental_contract(
        method_id=args.method_id,
        incremental_feature_source=args.incremental_feature_source,
        delta_condition=args.delta_condition,
        gate_condition=args.gate_condition,
        raw_feature_encoder_trainable=args.raw_feature_encoder_trainable,
    )
    if args.train_split != "hb_train_all" or args.val_split != "hb_val_stride5_n1000_seed42":
        raise ValueError(f"Unexpected LED split tags: train={args.train_split!r} val={args.val_split!r}")
    if args.eval_protocol != "per_image_affine_disp_depth_anything_v2":
        raise ValueError(f"Unsupported eval_protocol={args.eval_protocol!r}")
    if not (0.0 < args.q_good < 1.0):
        raise ValueError(f"--q-good must be in (0,1), got {args.q_good}")
    if args.lowpass_kernel <= 0 or args.lowpass_kernel % 2 == 0:
        raise ValueError(f"--lowpass-kernel must be positive odd, got {args.lowpass_kernel}")
    if args.bs <= 0 or args.accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("bs, accum-steps, and epochs must be positive.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive when provided.")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise ValueError("--max-val-samples must be positive when provided.")
    if not (0.0 <= args.hflip_prob <= 1.0):
        raise ValueError(f"--hflip-prob must be in [0,1], got {args.hflip_prob}")
    if args.eval_only and not args.resume_from:
        raise ValueError("--eval-only requires --resume-from.")
    for path_attr in ("led_train_list", "led_val_list", "pretrained_from", "c2_checkpoint"):
        if not Path(getattr(args, path_attr)).expanduser().is_file():
            raise FileNotFoundError(f"Missing required file for {path_attr}: {getattr(args, path_attr)}")
    if not Path(args.c2_run_dir).expanduser().is_dir():
        raise FileNotFoundError(f"Missing C2 run dir: {args.c2_run_dir}")
    if args.resume_from is not None and not Path(args.resume_from).expanduser().is_file():
        raise FileNotFoundError(f"Missing resume checkpoint: {args.resume_from}")
    if str(args.eval_feature_ablation_mode) == "shuffle" and int(args.feature_ablation_donor_offset) == 0:
        raise ValueError("--feature-ablation-donor-offset must be nonzero for shuffle eval.")

    active_ablation_modes = {str(args.train_feature_ablation_mode), str(args.eval_feature_ablation_mode)} - {"true", "none"}
    if active_ablation_modes and (args.method_id not in ("N2", "N7") or args.incremental_feature_source != "x3"):
        raise ValueError("x3 feature ablation requires method_id in ('N2', 'N7') and incremental_feature_source='x3'.")

    if method_is_raw(args):
        validate_led_hb_raw_semantics(
            dataset_name=args.dataset_name,
            illumination=args.illumination,
            dataset_geometry_mode=args.dataset_geometry_mode,
            input_height=args.input_height,
            input_width=args.input_width,
            depth_label=args.depth_label,
            depth_unit=args.depth_unit,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            input_domain=args.input_domain,
            model_input_tensor=args.model_input_tensor,
            raw_storage_format=args.raw_storage_format,
            unprocessing_method=args.unprocessing_method,
            raw_adapter_backend=args.raw_adapter_backend,
            randomize_unprocessing=bool(args.randomize_unprocessing),
            raw_adapter_fixed_light_scale=float(args.raw_adapter_fixed_light_scale),
            raw_adapter_variant_policy=args.raw_adapter_variant_policy,
        )
        expected = {
            "input_domain": "raw4",
            "model_input_tensor": "raw",
            "front_end": "c2_frozen_raw_ram_incremental",
            "raw_feature_encoder_trainable": "true",
        }
        for attr, value in expected.items():
            if str(getattr(args, attr)) != value:
                raise ValueError(f"{args.method_id} requires {attr}={value!r}, got {getattr(args, attr)!r}")
        if args.randomize_unprocessing is None:
            raise ValueError("RAW N-series requires explicit --randomize-unprocessing or --no-randomize-unprocessing.")
        resolved = resolve_unprocessing_config(vars(args))
        for key, value in resolved.items():
            setattr(args, key, value)
        args.resolved_unprocessing_config = dict(resolved)
    else:
        validate_led_hb_rgb_depth_semantics(
            dataset_name=args.dataset_name,
            illumination=args.illumination,
            dataset_geometry_mode=args.dataset_geometry_mode,
            input_height=args.input_height,
            input_width=args.input_width,
            depth_label=args.depth_label,
            depth_unit=args.depth_unit,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            input_domain=args.input_domain,
            model_input_tensor=args.model_input_tensor,
            raw_storage_format=args.raw_storage_format,
            unprocessing_method=args.unprocessing_method,
        )
        expected_front_end = "c2_frozen_rgb_incremental" if args.incremental_feature_source == "rgb" else "c2_frozen_d1_incremental"
        if args.front_end != expected_front_end:
            raise ValueError(f"{args.method_id} requires front_end={expected_front_end!r}, got {args.front_end!r}")
        if args.raw_feature_encoder_trainable != NOT_APPLICABLE:
            raise ValueError(f"{args.method_id} requires raw_feature_encoder_trainable={NOT_APPLICABLE!r}")
        args.resolved_unprocessing_config = {"unprocessing_method": NOT_APPLICABLE}

    args.c2_metadata = validate_c2_metadata(args)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subtract_dicts(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...] | list[str]) -> dict[str, float | None]:
    return {key: None if a.get(key) is None or b.get(key) is None else float(a[key]) - float(b[key]) for key in keys}


def sample_led_region_metrics_three(
    *,
    depth_np: np.ndarray,
    valid_np: np.ndarray,
    aligned_final: np.ndarray,
    aligned_d1: np.ndarray,
    aligned_d0: np.ndarray,
    d0_norm_np: np.ndarray,
    d1_norm_np: np.ndarray,
    y_norm_np: np.ndarray,
    rgb_preview_np: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> dict[str, dict[str, float]]:
    grad_y, grad_x = np.gradient(depth_np.astype(np.float32))
    boundary_score = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    luma = 0.2126 * rgb_preview_np[..., 0] + 0.7152 * rgb_preview_np[..., 1] + 0.0722 * rgb_preview_np[..., 2]
    valid_luma = luma[valid_np & np.isfinite(luma)]
    dark_threshold = float(np.quantile(valid_luma, 0.20)) if valid_luma.size else 0.0
    masks = {
        "boundary_abs_rel": top_fraction_mask(boundary_score, valid_np, 0.10),
        "d0_high_error_abs_rel": top_fraction_mask(np.abs(d0_norm_np - y_norm_np), valid_np, 0.20),
        "d1_high_error_abs_rel": top_fraction_mask(np.abs(d1_norm_np - y_norm_np), valid_np, 0.20),
        "near_1_20_abs_rel": valid_np & (depth_np >= 1.0) & (depth_np <= 20.0),
        "mid_20_50_abs_rel": valid_np & (depth_np > 20.0) & (depth_np <= 50.0),
        "far50_abs_rel": valid_np & (depth_np > 50.0),
        "far100_abs_rel": valid_np & (depth_np > 100.0),
        "dark_q20_abs_rel": valid_np & (luma <= dark_threshold),
        "saturated_abs_rel": valid_np & (np.max(rgb_preview_np, axis=-1) > 0.95),
    }
    aligned = {"final": aligned_final, "D1": aligned_d1, "D0": aligned_d0}
    return {
        name: {
            key: region_abs_rel(depth_np, pred, mask, min_depth=min_depth, max_depth=max_depth)
            for key, mask in masks.items()
        }
        for name, pred in aligned.items()
    }


def target_region_score(summary: dict[str, Any]) -> float | None:
    delta = summary.get("region", {}).get("delta_final_minus_D1", {})
    return mean_finite([
        delta.get("boundary_abs_rel"),
        delta.get("d1_high_error_abs_rel"),
        delta.get("dark_q20_abs_rel"),
        delta.get("saturated_abs_rel"),
        delta.get("far50_abs_rel"),
    ])


def build_c2_model(args: argparse.Namespace) -> torch.nn.Module:
    c2_base = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    c2_model = build_dav2_residual_control_model(
        c2_base,
        residual_feature_source="d0",
        residual_alpha=float(args.residual_alpha),
        d0_sign=int(args.d0_sign),
        sensor_hw=(int(args.input_height), int(args.input_width)),
        backbone_hw=None,
    )
    ckpt_obj = torch.load(str(Path(args.c2_checkpoint).expanduser().resolve()), map_location="cpu")
    c2_model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    c2_model.eval()
    for param in c2_model.parameters():
        param.requires_grad = False
    return c2_model


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return build_c2_frozen_incremental_residual_model(
        build_c2_model(args),
        method_id=args.method_id,
        incremental_feature_source=args.incremental_feature_source,
        delta_condition=args.delta_condition,
        gate_condition=args.gate_condition,
        raw_feature_encoder_trainable=args.raw_feature_encoder_trainable,
        residual_alpha=args.residual_alpha,
        lambda_lp=args.lambda_lp,
        lowpass_kernel=args.lowpass_kernel,
        sensor_hw=(args.input_height, args.input_width),
        backbone_hw=None,
    )


def load_incremental_checkpoint(model: torch.nn.Module, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(str(Path(path).expanduser().resolve()), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(checkpoint)), strict=strict)
    return checkpoint


def build_loaders(args: argparse.Namespace) -> tuple[Any, Any, DataLoader, DataLoader]:
    common = {
        "size": (args.input_height, args.input_width),
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "dataset_name": args.dataset_name,
        "illumination": args.illumination,
        "led_geometry_mode": args.dataset_geometry_mode,
        "rgb_input_space": args.rgb_input_space,
        "depth_target_space": args.depth_target_space,
        "depth_label": args.depth_label,
        "depth_unit": args.depth_unit,
    }
    if method_is_raw(args):
        train_dataset = LEDHBRaw(
            args.led_train_list,
            mode="train",
            hflip_prob=args.hflip_prob,
            include_rgb_input=True,
            include_rgb_preview=False,
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            unprocessing_config=args.resolved_unprocessing_config,
            **common,
        )
        val_dataset = LEDHBRaw(
            args.led_val_list,
            mode="val",
            hflip_prob=0.0,
            include_rgb_input=True,
            include_rgb_preview=True,
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            unprocessing_config=args.resolved_unprocessing_config,
            **common,
        )
        assert_unprocessing_summaries_compatible(train_dataset.describe_unprocessing(), val_dataset.describe_unprocessing(), context="LED train vs LED val unprocessing")
        assert_unprocessing_summary_matches_config(train_dataset.describe_unprocessing(), args.resolved_unprocessing_config, context="LED train dataset vs config")
        assert_unprocessing_summary_matches_config(val_dataset.describe_unprocessing(), args.resolved_unprocessing_config, context="LED val dataset vs config")
    else:
        train_dataset = LEDHBHalfresRGBDepth(
            args.led_train_list,
            mode="train",
            hflip_prob=args.hflip_prob,
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            **common,
        )
        val_dataset = LEDHBHalfresRGBDepth(
            args.led_val_list,
            mode="val",
            hflip_prob=0.0,
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            **common,
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
    val_batch_size = n6_eval_batch_size(args)
    val_workers = 0 if bool(getattr(args, "eval_only", False)) else max(min(args.num_workers, 2), 0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=val_workers > 0,
    )
    return train_dataset, val_dataset, train_loader, val_loader


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
    d1_metrics: list[dict[str, float]] = []
    d0_metrics: list[dict[str, float]] = []
    final_regions: list[dict[str, float]] = []
    d1_regions: list[dict[str, float]] = []
    d0_regions: list[dict[str, float]] = []
    diagnostics: list[dict[str, float]] = []
    processed = 0
    visited = 0
    start = time.time()
    logger.info("[EVAL] start epoch=%d max_val_samples=%s batch_size=%d ablation=%s", epoch, args.max_val_samples, n6_eval_batch_size(args), feature_ablation_mode(args, phase="eval"))
    for batch in dataloader:
        if args.max_val_samples is not None and processed >= int(args.max_val_samples):
            break
        image = batch["image"].to(device, non_blocking=True).float()
        raw = batch.get("raw")
        if raw is not None:
            raw = raw.to(device, non_blocking=True).float()
        depth = batch["depth"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        valid_mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)
        model_batch: dict[str, torch.Tensor] = {"image": image, "valid_mask": valid_mask}
        if raw is not None:
            model_batch["raw"] = raw
        batch_size = int(image.shape[0])
        add_dataset_raw_donor_if_needed(
            model_batch=model_batch,
            dataset=dataloader.dataset,
            sample_indices=[visited + idx for idx in range(batch_size)],
            args=args,
            device=device,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            out = forward_incremental_model(model, model_batch, args, phase="eval")
        inv_gt = build_training_target(depth.float(), valid_mask, target_space="metric_depth")
        y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_mask, min_valid_pixels=128)
        for sample_idx in range(batch_size):
            if args.max_val_samples is not None and processed >= int(args.max_val_samples):
                break
            valid_i = valid_mask[sample_idx]
            if int(valid_i.sum().item()) < 128:
                continue
            depth_np = depth[sample_idx].detach().cpu().numpy().astype(np.float32)
            valid_np = valid_i.detach().cpu().numpy().astype(bool)
            final_disp = out["pred"][sample_idx].float().detach().cpu().numpy().astype(np.float32)
            d1_disp = out["D1_norm"][sample_idx].float().detach().cpu().numpy().astype(np.float32)
            d0_disp = (float(args.d0_sign) * out["D0"][sample_idx].float()).detach().cpu().numpy().astype(np.float32)
            aligned_final, _ = affine_align_disp(depth_np, final_disp, valid_np)
            aligned_d1, _ = affine_align_disp(depth_np, d1_disp, valid_np)
            aligned_d0, _ = affine_align_disp(depth_np, d0_disp, valid_np)
            metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
            metrics_d1 = compute_metrics(depth_np, aligned_d1, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
            metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
            if metrics_final is None or metrics_d1 is None or metrics_d0 is None:
                continue
            rgb_preview = batch["rgb_preview"][sample_idx].permute(1, 2, 0).numpy().astype(np.float32)
            regions = sample_led_region_metrics_three(
                depth_np=depth_np,
                valid_np=valid_np,
                aligned_final=aligned_final,
                aligned_d1=aligned_d1,
                aligned_d0=aligned_d0,
                d0_norm_np=out["D0_norm"][sample_idx].float().detach().cpu().numpy(),
                d1_norm_np=out["D1_norm"][sample_idx].float().detach().cpu().numpy(),
                y_norm_np=y_norm[sample_idx].float().detach().cpu().numpy(),
                rgb_preview_np=rgb_preview,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
            final_metrics.append({key: float(metrics_final[key]) for key in METRIC_KEYS if key in metrics_final})
            d1_metrics.append({key: float(metrics_d1[key]) for key in METRIC_KEYS if key in metrics_d1})
            d0_metrics.append({key: float(metrics_d0[key]) for key in METRIC_KEYS if key in metrics_d0})
            final_regions.append(regions["final"])
            d1_regions.append(regions["D1"])
            d0_regions.append(regions["D0"])
            gate = out["gate"][sample_idx].float()
            delta = out["delta"][sample_idx].float()
            delta_effective = out["delta_effective"][sample_idx].float()
            gate_delta = gate * delta_effective
            low = lowpass_avgpool(gate_delta.unsqueeze(0), kernel_size=args.lowpass_kernel)[0]
            denom = gate_delta[valid_i].abs().mean()
            diagnostics.append(
                {
                    "mean_gate": float(gate[valid_i].mean().detach().item()),
                    "max_gate": float(gate[valid_i].max().detach().item()),
                    "mean_abs_delta": float(delta[valid_i].abs().mean().detach().item()),
                    "mean_abs_delta_effective": float(delta_effective[valid_i].abs().mean().detach().item()),
                    "mean_abs_gate_delta": float(gate_delta[valid_i].abs().mean().detach().item()),
                    "low_ratio": float((low[valid_i].abs().mean() / (denom + 1e-6)).detach().item()),
                    "high_ratio": float(((gate_delta - low)[valid_i].abs().mean() / (denom + 1e-6)).detach().item()),
                }
            )
            processed += 1
        visited += batch_size
    if processed == 0:
        raise RuntimeError("Validation produced zero valid samples.")
    overall_final = average_dicts(final_metrics, METRIC_KEYS)
    overall_d1 = average_dicts(d1_metrics, METRIC_KEYS)
    overall_d0 = average_dicts(d0_metrics, METRIC_KEYS)
    region_final = average_dicts(final_regions, LED_NSERIES_REGION_KEYS)
    region_d1 = average_dicts(d1_regions, LED_NSERIES_REGION_KEYS)
    region_d0 = average_dicts(d0_regions, LED_NSERIES_REGION_KEYS)
    diag = average_dicts(diagnostics, ["mean_gate", "max_gate", "mean_abs_delta", "mean_abs_delta_effective", "mean_abs_gate_delta", "low_ratio", "high_ratio"])
    summary = {
        "epoch": int(epoch),
        "samples": int(processed),
        "max_val_samples": args.max_val_samples,
        "alignment_protocol": args.eval_protocol,
        "overall": {
            "final": overall_final,
            "D1": overall_d1,
            "D0": overall_d0,
            "delta_final_minus_D1": subtract_dicts(overall_final, overall_d1, METRIC_KEYS),
            "delta_D1_minus_D0": subtract_dicts(overall_d1, overall_d0, METRIC_KEYS),
        },
        "region": {
            "final": region_final,
            "D1": region_d1,
            "D0": region_d0,
            "delta_final_minus_D1": subtract_dicts(region_final, region_d1, LED_NSERIES_REGION_KEYS),
            "delta_D1_minus_D0": subtract_dicts(region_d1, region_d0, LED_NSERIES_REGION_KEYS),
        },
        "diagnostics": diag,
        "elapsed_seconds": float(time.time() - start),
        "feature_ablation_mode": feature_ablation_mode(args, phase="eval"),
        "feature_ablation_scope": str(getattr(args, "feature_ablation_scope", "both")),
        "feature_ablation_key": str(getattr(args, "feature_ablation_key", "x3")),
        "feature_ablation_seed": int(getattr(args, "feature_ablation_seed", 42)),
        "feature_ablation_donor_offset": int(getattr(args, "feature_ablation_donor_offset", 1)),
        "feature_ablation_mean_kind": "batch_spatial_mean" if feature_ablation_mode(args, phase="eval") == "mean" else None,
        "feature_ablation_applied": bool(feature_ablation_active(args, phase="eval")),
    }
    summary["target_region_score"] = target_region_score(summary)
    logger.info(
        "[EVAL] done epoch=%d samples=%d final_abs_rel=%.5f D1_abs_rel=%.5f D0_abs_rel=%.5f final_minus_D1=%.5f elapsed=%s",
        epoch,
        processed,
        float(overall_final["abs_rel"]),
        float(overall_d1["abs_rel"]),
        float(overall_d0["abs_rel"]),
        float(summary["overall"]["delta_final_minus_D1"]["abs_rel"]),
        format_seconds(summary["elapsed_seconds"]),
    )
    return summary


def build_config_payload(args: argparse.Namespace, train_dataset: Any, val_dataset: Any, total_params: int, trainable_params: int) -> dict[str, Any]:
    payload = dict(vars(args))
    payload["led_geometry_mode"] = args.dataset_geometry_mode
    payload["dataset_geometry"] = {
        "train": train_dataset.describe_geometry(),
        "val": val_dataset.describe_geometry(),
        "led_val": val_dataset.describe_geometry(),
    }
    if method_is_raw(args):
        payload["unprocessing_policy"] = {
            "train": train_dataset.describe_unprocessing(),
            "led_val": val_dataset.describe_unprocessing(),
        }
    payload["model_param_counts"] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
    }
    return payload


def build_feature_ablation_summary(args: argparse.Namespace, led_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": str(args.experiment_label or args.method_id),
        "source_checkpoint": str(Path(args.resume_from).expanduser().resolve()) if args.resume_from else None,
        "feature_ablation_mode": feature_ablation_mode(args, phase="eval"),
        "feature_ablation_scope": str(args.feature_ablation_scope),
        "feature_ablation_key": str(args.feature_ablation_key),
        "feature_ablation_seed": int(args.feature_ablation_seed),
        "feature_ablation_donor_offset": int(args.feature_ablation_donor_offset),
        "method_id": str(args.method_id),
        "incremental_feature_source": str(args.incremental_feature_source),
        "c2_checkpoint": str(Path(args.c2_checkpoint).expanduser().resolve()),
        "c2_run_dir": str(Path(args.c2_run_dir).expanduser().resolve()),
        "processed_samples": {"led_hb": int(led_summary.get("samples", 0))},
        "led_hb": led_summary,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training entry expects CUDA.")

    requested_save_path = str(args.save_path)
    if args.eval_only:
        eval_mode = feature_ablation_mode(args, phase="eval")
        eval_scope = str(args.feature_ablation_scope)
        output_dir = Path(args.n6_output_dir).expanduser().resolve() if args.n6_output_dir else Path(args.save_path).expanduser().resolve() / f"eval_only_{eval_mode}_{eval_scope}"
        args.requested_save_path = requested_save_path
        args.save_path = str(output_dir)
        args.n6_output_dir = str(output_dir)
    save_path = Path(args.save_path).expanduser().resolve()
    heavy_save_path = Path(args.heavy_save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        heavy_save_path.mkdir(parents=True, exist_ok=True)

    logger = init_log("led_hb_incremental_residual", logging.INFO) or logging.getLogger("led_hb_incremental_residual")
    logger.propagate = False
    attach_file_logger(logger, save_path / "train.log")
    logger.info("%s\n", pprint.pformat({**vars(args), "device": str(device)}))
    cudnn.enabled = True
    cudnn.benchmark = not bool(args.eval_only)
    cudnn.deterministic = bool(args.eval_only)
    set_random_seed(args.seed)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    model = build_model(args)
    start_epoch = 0
    global_step = 0
    resume = None
    if args.resume_from:
        resume = load_incremental_checkpoint(model, args.resume_from, strict=True)
        start_epoch = int(resume.get("epoch", -1)) + 1
        global_step = int(resume.get("global_step", 0))
        logger.info("[INIT] resumed model from %s", args.resume_from)
    model = model.to(device)
    total_params, trainable_param_count = count_parameters(model)
    config_payload = build_config_payload(args, train_dataset, val_dataset, total_params, trainable_param_count)
    save_json(save_path / "config.json", config_payload)
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    if args.eval_only:
        resume_epoch = int(start_epoch - 1)
        val_summary = evaluate_model(model, val_loader, args, device, epoch=resume_epoch, amp_dtype=amp_dtype, logger=logger)
        eval_mode = feature_ablation_mode(args, phase="eval")
        eval_scope = str(args.feature_ablation_scope)
        save_json(save_path / f"eval_only_led_hb_{eval_mode}_{eval_scope}.json", val_summary)
        save_json(save_path / "val_metrics.json", {"epochs": [val_summary], "latest": val_summary})
        n6_summary = build_feature_ablation_summary(args, val_summary)
        save_json(save_path / "feature_ablation_summary.json", n6_summary)
        save_json(save_path / "n6_summary.json", n6_summary)
        save_json(save_path / "run_summary.json", {"config": config_payload, "train": [], "val": [val_summary], "led_val": [val_summary], "feature_ablation_summary": n6_summary, "heavy_save_path": str(heavy_save_path)})
        logger.info("[EVAL_ONLY] wrote outputs to %s", save_path)
        return

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if resume is not None and "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")
    logger.info("[MODEL] total_params=%d trainable_params=%d frozen_params=%d method=%s", total_params, trainable_param_count, total_params - trainable_param_count, args.method_id)
    logger.info("[DATASET] train_samples=%d led_val_samples=%d", len(train_dataset), len(val_dataset))

    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    best_abs_rel = float("inf")
    best_target_score = float("inf")
    steps_per_epoch = len(train_loader)
    if args.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, int(args.max_train_steps))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running: dict[str, float] = {}
        used_steps = 0
        optimizer_steps = 0
        pending_gradients = False
        for step_idx, batch in enumerate(train_loader):
            if step_idx >= steps_per_epoch:
                break
            image = batch["image"].to(device, non_blocking=True).float()
            raw = batch.get("raw")
            if raw is not None:
                raw = raw.to(device, non_blocking=True).float()
            depth = batch["depth"].to(device, non_blocking=True).float()
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
            if epoch == start_epoch and step_idx == 0:
                logger.info("[BATCH] image=%s raw=%s depth=%s valid=%s samples=%s", tuple(image.shape), None if raw is None else tuple(raw.shape), tuple(depth.shape), tuple(valid_mask.shape), batch["sample_name"][: min(2, len(batch["sample_name"]))])
            model_batch: dict[str, torch.Tensor] = {"image": image, "valid_mask": valid_mask}
            if raw is not None:
                model_batch["raw"] = raw
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                out = forward_incremental_model(model, model_batch, args, phase="train")
                loss, loss_info = compute_incremental_residual_loss(
                    out,
                    depth,
                    valid_mask,
                    q_good=args.q_good,
                    lambda_final=args.lambda_final,
                    lambda_boundary=args.lambda_boundary,
                    lambda_grad=args.lambda_grad,
                    lambda_keep_good_d1=args.lambda_keep_good_d1,
                    lambda_gate_sparse=args.lambda_gate_sparse,
                    lambda_lowfreq_loss=args.lambda_lowfreq_loss,
                    lambda_invalid_keep=args.lambda_invalid_keep,
                    lowpass_kernel=args.lowpass_kernel,
                )
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
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d opt_step=%d loss=%.5f mean_gate=%.5f mean_abs_gate_delta=%.5f lr=%.7f max_mem_mb=%.0f elapsed=%s",
                    epoch,
                    step_idx + 1,
                    steps_per_epoch,
                    optimizer_steps,
                    running.get("loss_total", 0.0) / denom,
                    running.get("mean_gate", 0.0) / denom,
                    running.get("mean_abs_gate_delta", 0.0) / denom,
                    optimizer.param_groups[0]["lr"],
                    max_mem,
                    format_seconds(now - epoch_start),
                )
        denom = max(used_steps, 1)
        train_summary = {"epoch": int(epoch), "used_steps": int(used_steps), "optimizer_steps": int(optimizer_steps), "elapsed_seconds": float(time.time() - epoch_start)}
        for key, value in running.items():
            train_summary[key] = float(value / denom)
        train_history.append(train_summary)
        save_json(save_path / "train_loss_summary.json", {"epochs": train_history})

        val_summary = None
        if ((epoch + 1) % args.eval_interval) == 0:
            val_summary = evaluate_model(model, val_loader, args, device, epoch=epoch, amp_dtype=amp_dtype, logger=logger)
            val_history.append(val_summary)
            save_json(save_path / "val_metrics.json", {"epochs": val_history, "latest": val_summary})
            current_abs_rel = val_summary["overall"]["final"]["abs_rel"]
            current_target = val_summary.get("target_region_score")
            if current_abs_rel is not None and float(current_abs_rel) < best_abs_rel:
                best_abs_rel = float(current_abs_rel)
                save_json(save_path / "best_val_metrics.json", val_summary)
                if args.save_best_checkpoint:
                    save_checkpoint(heavy_save_path / "best_abs_rel.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)
            if current_target is not None and float(current_target) < best_target_score:
                best_target_score = float(current_target)
                save_json(save_path / "best_target_region_metrics.json", val_summary)
                if args.save_best_checkpoint:
                    save_checkpoint(heavy_save_path / "best_target_region_score.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)

        if ((epoch + 1) % args.save_interval) == 0:
            save_checkpoint(heavy_save_path / f"epoch_{epoch:02d}.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)
        save_checkpoint(heavy_save_path / "latest.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)

    save_json(
        save_path / "run_summary.json",
        {
            "config": config_payload,
            "train": train_history,
            "val": val_history,
            "led_val": val_history,
            "best_abs_rel": float_or_none(best_abs_rel),
            "best_target_region_score": float_or_none(best_target_score),
            "heavy_save_path": str(heavy_save_path),
        },
    )


if __name__ == "__main__":
    main()
