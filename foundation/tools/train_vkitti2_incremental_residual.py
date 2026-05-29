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
    VKITTI2HalfresRGBDepth,
    VKITTI2Raw,
    validate_vkitti_halfres_rgb_depth_semantics,
    validate_vkitti_raw_semantics,
)
from foundation.engine.models import (
    build_c2_frozen_incremental_residual_model,
    build_dav2_residual_control_model,
)
from foundation.engine.models.dav2_incremental_residual import (
    DELTA_CONDITIONS,
    GATE_CONDITIONS,
    INCREMENTAL_FEATURE_SOURCES,
    INCREMENTAL_METHOD_IDS,
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
from foundation.tools.eval_raw_residual_kitti import KittiHalfresRawDataset
from foundation.tools.residual_control_kitti_eval import (
    CONTROL_KITTI_EVAL_PROTOCOL,
    KittiHalfresRGBDepthDataset,
    collate_single_sample,
)
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    REGION_KEYS,
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


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DATASET_GEOMETRY_CHOICES = ("vkitti2_even_fullres_halfres_2x2",)
FRONT_END_CHOICES = (
    "c2_frozen_raw_ram_incremental",
    "c2_frozen_rgb_incremental",
    "c2_frozen_d1_incremental",
)
EVAL_PROTOCOL_CHOICES = ("per_image_affine_disp_depth_anything_v2",)
RAW_KITTI_EVAL_PROTOCOL = "halfres_raw_canonical_even_pad_crop_affine_disp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VKITTI2 N-series C2-frozen incremental residual training.")
    parser.add_argument("--method-id", required=True, choices=INCREMENTAL_METHOD_IDS)
    parser.add_argument("--input-domain", required=True, choices=["raw4", "rgb"])
    parser.add_argument("--model-input-tensor", required=True, choices=["raw", "image"])
    parser.add_argument("--dataset-geometry-mode", required=True, choices=DATASET_GEOMETRY_CHOICES)
    parser.add_argument("--raw-storage-format", required=True, choices=tuple(RAW_STORAGE_FORMAT_CHOICES) + (NOT_APPLICABLE,))
    parser.add_argument("--fullres-even-policy", required=True, choices=FULLRES_EVEN_POLICY_CHOICES)
    parser.add_argument("--rgb-input-space", required=True, choices=RGB_INPUT_SPACE_CHOICES)
    parser.add_argument("--depth-target-space", required=True, choices=DEPTH_TARGET_SPACE_CHOICES)
    parser.add_argument("--front-end", required=True, choices=FRONT_END_CHOICES)
    parser.add_argument("--encoder", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--c2-checkpoint", required=True)
    parser.add_argument("--c2-run-dir", required=True)
    parser.add_argument("--vkitti-train-list", required=True)
    parser.add_argument("--vkitti-val-list", required=True)
    parser.add_argument("--eval-protocol", required=True, choices=EVAL_PROTOCOL_CHOICES)
    parser.add_argument("--eval-kitti", action="store_true")
    parser.add_argument("--kitti-val-split", default=None)
    parser.add_argument("--kitti-base", default=None)
    parser.add_argument("--kitti-eval-protocol", default=None, choices=[RAW_KITTI_EVAL_PROTOCOL, CONTROL_KITTI_EVAL_PROTOCOL])
    parser.add_argument("--kitti-expected-val-samples", type=int, default=None)
    parser.add_argument("--kitti-num-workers", type=int, default=None)
    parser.add_argument("--max-kitti-val-samples", type=int, default=None)
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
    parser.add_argument("--lambda-final", type=float, required=True)
    parser.add_argument("--lambda-boundary", type=float, required=True)
    parser.add_argument("--lambda-grad", type=float, required=True)
    parser.add_argument("--lambda-keep-good-d1", type=float, required=True)
    parser.add_argument("--lambda-gate-sparse", type=float, required=True)
    parser.add_argument("--lambda-lowfreq-loss", type=float, required=True)
    parser.add_argument("--lambda-invalid-keep", type=float, required=True)
    parser.add_argument("--unprocessing-method", default=NOT_APPLICABLE, choices=[NOT_APPLICABLE, "old_brooks_preset", "raw_adapter_style"])
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
    parser.add_argument(
        "--raw-adapter-random-seed-policy",
        default=NOT_APPLICABLE,
        choices=[NOT_APPLICABLE, "dataloader_generator", "path_hash"],
    )
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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_path_str(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve()) if value not in (None, "") else ""


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
    meta = dict(ckpt_args or run_config or {})
    if not meta:
        raise ValueError("C2 checkpoint and run dir do not contain args/config metadata; refusing formal N-series run.")

    experiment_id = meta.get("experiment_id")
    if experiment_id is not None and str(experiment_id).lower() != "c2":
        raise ValueError(f"C2 checkpoint experiment_id must be C2, got {experiment_id!r}")
    expected = {
        "residual_feature_source": "d0",
        "front_end": "dav2_rgb_frozen",
        "raw_storage_format": NOT_APPLICABLE,
        "input_domain": "rgb",
        "model_input_tensor": "image",
        "encoder": args.encoder,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "d0_sign": args.d0_sign,
        "fullres_even_policy": args.fullres_even_policy,
        "rgb_input_space": args.rgb_input_space,
        "depth_target_space": args.depth_target_space,
    }
    for key, expected_value in expected.items():
        actual = meta.get(key)
        if actual is None:
            raise ValueError(f"C2 metadata missing required key {key!r}")
        if isinstance(expected_value, float):
            if not math.isclose(float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"C2 metadata {key} must be {expected_value!r}, got {actual!r}")
        elif isinstance(expected_value, int):
            if int(actual) != int(expected_value):
                raise ValueError(f"C2 metadata {key} must be {expected_value!r}, got {actual!r}")
        else:
            if str(actual) != str(expected_value):
                raise ValueError(f"C2 metadata {key} must be {expected_value!r}, got {actual!r}")

    for key in ("vkitti_train_list", "vkitti_val_list", "pretrained_from"):
        actual = meta.get(key)
        expected_value = getattr(args, key)
        if actual is None:
            raise ValueError(f"C2 metadata missing required key {key!r}")
        if _as_path_str(actual) != _as_path_str(expected_value):
            raise ValueError(f"C2 metadata {key} must match current setting: {actual!r} vs {expected_value!r}")
    if "residual_alpha" in meta and not math.isclose(float(meta["residual_alpha"]), float(args.residual_alpha), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"C2 residual_alpha must match current residual_alpha={args.residual_alpha}, got {meta['residual_alpha']}")
    return {"checkpoint_args": ckpt_args, "run_config": run_config, "source": "checkpoint_args" if ckpt_args else "run_config"}


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
    if args.dataset_geometry_mode != "vkitti2_even_fullres_halfres_2x2":
        raise ValueError(f"Unsupported dataset_geometry_mode={args.dataset_geometry_mode!r}")
    if (args.input_height, args.input_width) != (187, 621):
        raise ValueError(f"N-series halfres requires input size (187, 621), got {(args.input_height, args.input_width)}")
    if not (0.0 < args.min_depth < args.max_depth):
        raise ValueError(f"Expected 0 < min_depth < max_depth, got {args.min_depth}, {args.max_depth}")
    if args.residual_alpha <= 0.0:
        raise ValueError(f"--residual-alpha must be positive, got {args.residual_alpha}")
    if not (0.0 < args.q_good < 1.0):
        raise ValueError(f"--q-good must be in (0,1), got {args.q_good}")
    if args.lowpass_kernel <= 0 or args.lowpass_kernel % 2 == 0:
        raise ValueError(f"--lowpass-kernel must be positive odd, got {args.lowpass_kernel}")
    if not (0.0 <= args.hflip_prob <= 1.0):
        raise ValueError(f"--hflip-prob must be in [0,1], got {args.hflip_prob}")
    if args.bs <= 0 or args.accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("bs, accum-steps, and epochs must be positive.")
    if args.save_interval <= 0 or args.eval_interval <= 0:
        raise ValueError("save-interval and eval-interval must be positive.")

    is_raw = args.incremental_feature_source in ("x3", "ffm_mid")
    if is_raw:
        expected = {
            "input_domain": "raw4",
            "model_input_tensor": "raw",
            "front_end": "c2_frozen_raw_ram_incremental",
            "raw_storage_format": "synthetic_packed_bayer_4ch_halfres",
        }
        for attr, value in expected.items():
            if getattr(args, attr) != value:
                raise ValueError(f"{args.method_id} requires {attr}={value!r}, got {getattr(args, attr)!r}")
        validate_vkitti_raw_semantics(
            raw_storage_format=args.raw_storage_format,
            fullres_even_policy=args.fullres_even_policy,
            rgb_input_space=args.rgb_input_space,
            depth_target_space=args.depth_target_space,
        )
        if args.randomize_unprocessing is None:
            raise ValueError("RAW N-series requires explicit --randomize-unprocessing or --no-randomize-unprocessing.")
        resolved = resolve_unprocessing_config(vars(args))
        if resolved["unprocessing_method"] == "raw_adapter_style" and resolved["raw_adapter_backend"] != "analytic":
            raise ValueError("RAW N-series only supports raw_adapter_backend=analytic for online training.")
        for key, value in resolved.items():
            setattr(args, key, value)
        args.resolved_unprocessing_config = dict(resolved)
    else:
        expected = {
            "input_domain": "rgb",
            "model_input_tensor": "image",
            "raw_storage_format": NOT_APPLICABLE,
            "front_end": "c2_frozen_rgb_incremental" if args.incremental_feature_source == "rgb" else "c2_frozen_d1_incremental",
        }
        for attr, value in expected.items():
            if getattr(args, attr) != value:
                raise ValueError(f"{args.method_id} requires {attr}={value!r}, got {getattr(args, attr)!r}")
        validate_vkitti_halfres_rgb_depth_semantics(
            raw_storage_format=args.raw_storage_format,
            fullres_even_policy=args.fullres_even_policy,
            rgb_input_space=args.rgb_input_space,
            depth_target_space=args.depth_target_space,
        )
        if args.unprocessing_method != NOT_APPLICABLE:
            raise ValueError("RGB/D1 N-series requires --unprocessing-method not_applicable.")
        args.resolved_unprocessing_config = {"unprocessing_method": NOT_APPLICABLE}

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
    if args.eval_kitti:
        if not args.kitti_val_split or not args.kitti_base:
            raise ValueError("--eval-kitti requires --kitti-val-split and --kitti-base.")
        expected_protocol = RAW_KITTI_EVAL_PROTOCOL if is_raw else CONTROL_KITTI_EVAL_PROTOCOL
        if args.kitti_eval_protocol != expected_protocol:
            raise ValueError(f"{args.method_id} requires --kitti-eval-protocol {expected_protocol}.")
        if args.kitti_expected_val_samples is not None and args.kitti_expected_val_samples <= 0:
            raise ValueError("--kitti-expected-val-samples must be positive when provided.")
        if args.kitti_num_workers is not None and args.kitti_num_workers < 0:
            raise ValueError("--kitti-num-workers must be non-negative when provided.")
        if args.max_kitti_val_samples is not None and args.max_kitti_val_samples <= 0:
            raise ValueError("--max-kitti-val-samples must be positive when provided.")
        if not Path(args.kitti_val_split).expanduser().is_file():
            raise FileNotFoundError(f"Missing KITTI val split: {args.kitti_val_split}")
        if not Path(args.kitti_base).expanduser().is_dir():
            raise FileNotFoundError(f"Missing KITTI base directory: {args.kitti_base}")

    args.c2_metadata = validate_c2_metadata(args)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    c2_model = build_c2_model(args)
    return build_c2_frozen_incremental_residual_model(
        c2_model,
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


def build_loaders(args: argparse.Namespace) -> tuple[Any, Any, DataLoader, DataLoader]:
    is_raw = args.incremental_feature_source in ("x3", "ffm_mid")
    if is_raw:
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
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            fullres_even_policy=args.fullres_even_policy,
            rgb_input_space=args.rgb_input_space,
            depth_target_space=args.depth_target_space,
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
    else:
        train_dataset = VKITTI2HalfresRGBDepth(
            filelist_path=args.vkitti_train_list,
            mode="train",
            size=(args.input_height, args.input_width),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            hflip_prob=args.hflip_prob,
            raw_storage_format=args.raw_storage_format,
            fullres_even_policy=args.fullres_even_policy,
            rgb_input_space=args.rgb_input_space,
            depth_target_space=args.depth_target_space,
        )
        val_dataset = VKITTI2HalfresRGBDepth(
            filelist_path=args.vkitti_val_list,
            mode="val",
            size=(args.input_height, args.input_width),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            hflip_prob=0.0,
            include_geometry=True,
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
    val_workers = max(min(args.num_workers, 2), 0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=val_workers > 0,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_kitti_val_loader(args: argparse.Namespace, device: torch.device) -> tuple[Any | None, DataLoader | None]:
    if not args.eval_kitti:
        return None, None
    if args.incremental_feature_source in ("x3", "ffm_mid"):
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
    else:
        dataset = KittiHalfresRGBDepthDataset(
            filelist_path=args.kitti_val_split,
            kitti_base=args.kitti_base,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
    if args.kitti_expected_val_samples is not None and len(dataset) != int(args.kitti_expected_val_samples):
        raise RuntimeError(f"Expected KITTI val length {int(args.kitti_expected_val_samples)}, got {len(dataset)}")
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


def sample_region_metrics_three(
    *,
    depth_np: np.ndarray,
    valid_np: np.ndarray,
    aligned_final: np.ndarray,
    aligned_d1: np.ndarray,
    aligned_d0: np.ndarray,
    base_norm_np: np.ndarray,
    y_norm_np: np.ndarray,
    rgb_preview_np: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> dict[str, dict[str, float]]:
    grad_y, grad_x = np.gradient(depth_np.astype(np.float32))
    boundary_score = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    masks = {
        "boundary_abs_rel": top_fraction_mask(boundary_score, valid_np, 0.10),
        "dav2_high_error_abs_rel": top_fraction_mask(np.abs(base_norm_np - y_norm_np), valid_np, 0.20),
        "far50_abs_rel": valid_np & (depth_np > 50.0),
        "dark_abs_rel": valid_np & (
            (0.2126 * rgb_preview_np[..., 0] + 0.7152 * rgb_preview_np[..., 1] + 0.0722 * rgb_preview_np[..., 2])
            < 0.15
        ),
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


def subtract_dicts(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...] | list[str]) -> dict[str, float | None]:
    return {key: None if a.get(key) is None or b.get(key) is None else float(a[key]) - float(b[key]) for key in keys}


def target_region_score(summary: dict[str, Any]) -> float | None:
    region = summary.get("region", {})
    delta = region.get("delta_final_minus_D1", {})
    values = [
        delta.get("boundary_abs_rel"),
        delta.get("far50_abs_rel"),
        delta.get("dark_abs_rel"),
        delta.get("saturated_abs_rel"),
        delta.get("fog_low_contrast_abs_rel"),
    ]
    return mean_finite(values)


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
    start = time.time()
    logger.info("[EVAL] start epoch=%d max_val_samples=%s", epoch, args.max_val_samples)

    for batch in dataloader:
        if args.max_val_samples is not None and processed >= args.max_val_samples:
            break
        image = batch["image"].to(device, non_blocking=True).float()
        raw = batch.get("raw")
        if raw is not None:
            raw = raw.to(device, non_blocking=True).float()
        depth = batch["depth"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        valid_mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)
        if int(valid_mask[0].sum().item()) < 128:
            continue
        model_batch = {"image": image, "valid_mask": valid_mask}
        if raw is not None:
            model_batch["raw"] = raw
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            out = model(model_batch)

        inv_gt = build_training_target(depth.float(), valid_mask, target_space="metric_depth")
        y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_mask, min_valid_pixels=128)

        depth_np = depth[0].detach().cpu().numpy().astype(np.float32)
        valid_np = valid_mask[0].detach().cpu().numpy().astype(bool)
        final_disp = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
        d1_disp = out["D1_norm"][0].float().detach().cpu().numpy().astype(np.float32)
        d0_disp = (float(args.d0_sign) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
        aligned_final, _ = affine_align_disp(depth_np, final_disp, valid_np)
        aligned_d1, _ = affine_align_disp(depth_np, d1_disp, valid_np)
        aligned_d0, _ = affine_align_disp(depth_np, d0_disp, valid_np)
        metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
        metrics_d1 = compute_metrics(depth_np, aligned_d1, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
        metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=args.min_depth, max_depth=args.max_depth)
        if metrics_final is None or metrics_d1 is None or metrics_d0 is None:
            continue
        rgb_preview = batch["rgb_preview"][0].permute(1, 2, 0).numpy().astype(np.float32)
        regions = sample_region_metrics_three(
            depth_np=depth_np,
            valid_np=valid_np,
            aligned_final=aligned_final,
            aligned_d1=aligned_d1,
            aligned_d0=aligned_d0,
            base_norm_np=out["D1_norm"][0].float().detach().cpu().numpy(),
            y_norm_np=y_norm[0].float().detach().cpu().numpy(),
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

        gate = out["gate"].float()
        delta = out["delta"].float()
        delta_effective = out["delta_effective"].float()
        gate_delta = gate * delta_effective
        low = lowpass_avgpool(gate_delta, kernel_size=args.lowpass_kernel)
        denom = gate_delta[valid_mask].abs().mean()
        diagnostics.append(
            {
                "mean_gate": float(gate[valid_mask].mean().detach().item()),
                "max_gate": float(gate[valid_mask].max().detach().item()),
                "mean_abs_delta": float(delta[valid_mask].abs().mean().detach().item()),
                "mean_abs_delta_effective": float(delta_effective[valid_mask].abs().mean().detach().item()),
                "mean_abs_gate_delta": float(gate_delta[valid_mask].abs().mean().detach().item()),
                "low_ratio": float((low[valid_mask].abs().mean() / (denom + 1e-6)).detach().item()),
                "high_ratio": float(((gate_delta - low)[valid_mask].abs().mean() / (denom + 1e-6)).detach().item()),
            }
        )
        processed += 1

    if processed == 0:
        raise RuntimeError("Validation produced zero valid samples.")
    overall_final = average_dicts(final_metrics, METRIC_KEYS)
    overall_d1 = average_dicts(d1_metrics, METRIC_KEYS)
    overall_d0 = average_dicts(d0_metrics, METRIC_KEYS)
    region_final = average_dicts(final_regions, REGION_KEYS)
    region_d1 = average_dicts(d1_regions, REGION_KEYS)
    region_d0 = average_dicts(d0_regions, REGION_KEYS)
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
            "delta_final_minus_D1": subtract_dicts(region_final, region_d1, REGION_KEYS),
            "delta_D1_minus_D0": subtract_dicts(region_d1, region_d0, REGION_KEYS),
        },
        "diagnostics": diag,
        "target_region_score": target_region_score({"region": {"delta_final_minus_D1": subtract_dicts(region_final, region_d1, REGION_KEYS)}}),
        "elapsed_seconds": float(time.time() - start),
    }
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


def filter_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    if metrics is None:
        return {key: None for key in METRIC_KEYS}
    return {key: float_or_none(metrics.get(key, float("nan"))) for key in METRIC_KEYS}


def collect_kitti_nseries_sample(
    *,
    sample: dict[str, Any],
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    row = {
        "dataset_index": int(sample["dataset_index"]),
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "status": str(sample.get("status", "ok")),
        "valid_pixels": None,
        "final": filter_metrics(None),
        "D1": filter_metrics(None),
        "D0": filter_metrics(None),
        "diagnostics": {"mean_gate": None, "max_gate": None, "mean_abs_delta": None, "mean_abs_gate_delta": None},
    }
    if sample.get("status") != "ok":
        row["error"] = sample.get("error")
        return row
    image = sample["image"].unsqueeze(0).to(device, non_blocking=True).float()
    raw = sample.get("raw")
    if raw is not None:
        raw = raw.unsqueeze(0).to(device, non_blocking=True).float()
    depth_t = sample["depth"].unsqueeze(0).to(device, non_blocking=True).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device, non_blocking=True).bool()
    valid_t = valid_t & (depth_t >= float(config["min_depth"])) & (depth_t <= float(config["max_depth"]))
    valid_pixels = int(valid_t[0].sum().item())
    row["valid_pixels"] = valid_pixels
    if valid_pixels < 128:
        row["status"] = "skipped_invalid_pixels"
        row["error"] = f"valid_pixels={valid_pixels} < 128"
        return row
    try:
        model_batch = {"image": image, "valid_mask": valid_t}
        if raw is not None:
            model_batch["raw"] = raw
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
            out = model(model_batch)
        depth_np = depth_t[0].detach().cpu().numpy().astype(np.float32)
        valid_np = valid_t[0].detach().cpu().numpy().astype(bool)
        final_disp = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
        d1_disp = out["D1_norm"][0].float().detach().cpu().numpy().astype(np.float32)
        d0_disp = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
        aligned_final, _ = affine_align_disp(depth_np, final_disp, valid_np)
        aligned_d1, _ = affine_align_disp(depth_np, d1_disp, valid_np)
        aligned_d0, _ = affine_align_disp(depth_np, d0_disp, valid_np)
        metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=float(config["min_depth"]), max_depth=float(config["max_depth"]))
        metrics_d1 = compute_metrics(depth_np, aligned_d1, valid_np, min_depth=float(config["min_depth"]), max_depth=float(config["max_depth"]))
        metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=float(config["min_depth"]), max_depth=float(config["max_depth"]))
        if metrics_final is None or metrics_d1 is None or metrics_d0 is None:
            row["status"] = "skipped_metric_failure"
            row["error"] = "compute_metrics returned None"
            return row
        gate = out["gate"].float()
        delta = out["delta"].float()
        gate_delta = gate * out["delta_effective"].float()
        row["status"] = "ok"
        row["final"] = filter_metrics(metrics_final)
        row["D1"] = filter_metrics(metrics_d1)
        row["D0"] = filter_metrics(metrics_d0)
        row["diagnostics"] = {
            "mean_gate": float(gate[valid_t].mean().detach().item()),
            "max_gate": float(gate[valid_t].max().detach().item()),
            "mean_abs_delta": float(delta[valid_t].abs().mean().detach().item()),
            "mean_abs_gate_delta": float(gate_delta[valid_t].abs().mean().detach().item()),
        }
        return row
    except Exception as exc:  # noqa: BLE001
        row["status"] = "skipped_metric_failure"
        row["error"] = str(exc)
        return row


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {key: mean_finite([row.get(key) for row in rows]) for key in METRIC_KEYS}


def evaluate_kitti_model(
    model: torch.nn.Module,
    dataset: Any,
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
    ok_rows: list[dict[str, Any]] = []
    start = time.time()
    per_sample_path = output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for visited, sample in enumerate(dataloader):
            if visited >= max_visit:
                break
            row = collect_kitti_nseries_sample(
                sample=sample,
                model=model,
                config=config,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            row["epoch"] = int(epoch)
            rows.append(row)
            if row["status"] == "ok":
                ok_rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if (visited + 1) % 50 == 0 or visited + 1 == max_visit:
                logger.info("[EVAL][KITTI] processed=%d/%d ok=%d elapsed=%s", visited + 1, max_visit, len(ok_rows), format_seconds(time.time() - start))
    if not ok_rows:
        raise RuntimeError("KITTI eval produced zero ok samples.")
    overall_final = average_metrics([row["final"] for row in ok_rows])
    overall_d1 = average_metrics([row["D1"] for row in ok_rows])
    overall_d0 = average_metrics([row["D0"] for row in ok_rows])
    status_counts = Counter(str(row["status"]) for row in rows)
    elapsed_seconds = time.time() - start
    summary = {
        "dataset": "kitti_val_nseries",
        "epoch": int(epoch),
        "dataset_samples": int(len(dataset)),
        "visited_samples": int(len(rows)),
        "samples": int(len(ok_rows)),
        "max_val_samples": args.max_kitti_val_samples,
        "kitti_val_split": str(Path(args.kitti_val_split).expanduser().resolve()),
        "kitti_base": str(Path(args.kitti_base).expanduser().resolve()),
        "eval_protocol": args.kitti_eval_protocol,
        "status_counts": dict(status_counts),
        "overall": {
            "final": overall_final,
            "D1": overall_d1,
            "D0": overall_d0,
            "delta_final_minus_D1": subtract_dicts(overall_final, overall_d1, METRIC_KEYS),
            "delta_D1_minus_D0": subtract_dicts(overall_d1, overall_d0, METRIC_KEYS),
        },
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_visited_sample": float(elapsed_seconds / max(len(rows), 1)),
        "per_sample_path": str(per_sample_path),
    }
    save_json(output_dir / "metrics.json", summary)
    logger.info(
        "[EVAL][KITTI] done epoch=%d ok=%d final_abs_rel=%.5f D1_abs_rel=%.5f delta=%.5f elapsed=%s",
        epoch,
        len(ok_rows),
        float(overall_final["abs_rel"]),
        float(overall_d1["abs_rel"]),
        float(summary["overall"]["delta_final_minus_D1"]["abs_rel"]),
        format_seconds(elapsed_seconds),
    )
    return summary


def update_best_record(
    current: dict[str, Any] | None,
    *,
    key_value: float | None,
    minimize: bool,
    epoch: int,
    checkpoint_path: Path,
    val_summary: dict[str, Any] | None,
    kitti_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if key_value is None or not math.isfinite(float(key_value)):
        return current
    if current is not None:
        previous = current.get("value")
        if previous is not None:
            better = float(key_value) < float(previous) if minimize else float(key_value) > float(previous)
            if not better:
                return current
    return {
        "value": float(key_value),
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "vkitti": val_summary,
        "kitti": kitti_summary,
    }


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

    logger = init_log("vkitti2_incremental_residual", logging.INFO) or logging.getLogger("vkitti2_incremental_residual")
    logger.propagate = False
    attach_file_logger(logger, save_path / "train.log")
    logger.info("%s\n", pprint.pformat({**vars(args), "device": str(device)}))

    cudnn.enabled = True
    cudnn.benchmark = True
    set_random_seed(args.seed)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    kitti_val_dataset, kitti_val_loader = build_kitti_val_loader(args, device)
    model = build_model(args)
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(strip_module_prefix(resolve_model_state(resume)), strict=True)
        start_epoch = int(resume.get("epoch", -1)) + 1
        global_step = int(resume.get("global_step", 0))
        logger.info("[INIT] resumed model from %s", args.resume_from)

    model = model.to(device)
    total_params, trainable_param_count = count_parameters(model)
    config_payload = dict(vars(args))
    config_payload["dataset_geometry"] = {
        "train": train_dataset.describe_geometry(),
        "val": val_dataset.describe_geometry(),
        "vkitti_val": val_dataset.describe_geometry(),
    }
    if args.incremental_feature_source in ("x3", "ffm_mid"):
        config_payload["unprocessing_policy"] = {
            "train": train_dataset.describe_unprocessing(),
            "vkitti_val": val_dataset.describe_unprocessing(),
        }
    if kitti_val_dataset is not None:
        config_payload["dataset_geometry"]["kitti_val"] = kitti_val_dataset.describe_geometry()
        if hasattr(kitti_val_dataset, "describe_unprocessing"):
            config_payload.setdefault("unprocessing_policy", {})["kitti_val"] = kitti_val_dataset.describe_unprocessing()
    config_payload["model_param_counts"] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_param_count),
        "frozen_params": int(total_params - trainable_param_count),
    }
    config_payload["eval_protocol"] = {
        "vkitti_val": args.eval_protocol,
        "kitti_val": args.kitti_eval_protocol if args.eval_kitti else "disabled",
    }
    save_json(save_path / "config.json", config_payload)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")
    logger.info(
        "[MODEL] total_params=%d trainable_params=%d frozen_params=%d method=%s feature=%s lambda_lp=%.3f q_good=%.3f",
        total_params,
        trainable_param_count,
        total_params - trainable_param_count,
        args.method_id,
        args.incremental_feature_source,
        args.lambda_lp,
        args.q_good,
    )
    logger.info("[DATASET] train_samples=%d vkitti_val_samples=%d", len(train_dataset), len(val_dataset))
    if kitti_val_dataset is not None:
        logger.info("[DATASET][KITTI] val_samples=%d", len(kitti_val_dataset))

    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    kitti_val_history: list[dict[str, Any]] = []
    best_abs_rel_record: dict[str, Any] | None = None
    best_kitti_abs_rel_record: dict[str, Any] | None = None
    best_boundary_record: dict[str, Any] | None = None
    best_target_region_record: dict[str, Any] | None = None
    steps_per_epoch = len(train_loader)
    if args.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)

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
                logger.info(
                    "[BATCH] image=%s raw=%s depth=%s valid=%s samples=%s",
                    tuple(image.shape),
                    None if raw is None else tuple(raw.shape),
                    tuple(depth.shape),
                    tuple(valid_mask.shape),
                    batch["sample_name"][: min(2, len(batch["sample_name"]))],
                )
            model_batch = {"image": image, "valid_mask": valid_mask}
            if raw is not None:
                model_batch["raw"] = raw
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                out = model(model_batch)
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
                step_per_sec = (step_idx + 1) / max(now - epoch_start, 1e-6)
                eta_seconds = (steps_per_epoch - (step_idx + 1)) / max(step_per_sec, 1e-6)
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d opt_step=%d loss=%.5f L_final=%.5f L_boundary=%.5f "
                    "L_grad=%.5f L_keepD1=%.5f L_gate=%.5f L_lowfreq=%.5f L_invalid=%.5f "
                    "mean_gate=%.5f mean_abs_gate_delta=%.5f low_ratio=%.5f high_ratio=%.5f "
                    "lr=%.7f max_mem_mb=%.0f used=%d skipped=%d step_per_sec=%.2f elapsed=%s eta=%s",
                    epoch,
                    step_idx + 1,
                    steps_per_epoch,
                    optimizer_steps,
                    running.get("loss_total", 0.0) / denom,
                    running.get("L_final", 0.0) / denom,
                    running.get("L_boundary", 0.0) / denom,
                    running.get("L_grad", 0.0) / denom,
                    running.get("L_keep_good_D1", 0.0) / denom,
                    running.get("L_gate_sparse", 0.0) / denom,
                    running.get("L_lowfreq", 0.0) / denom,
                    running.get("L_invalid_keep", 0.0) / denom,
                    running.get("mean_gate", 0.0) / denom,
                    running.get("mean_abs_gate_delta", 0.0) / denom,
                    running.get("low_ratio", 0.0) / denom,
                    running.get("high_ratio", 0.0) / denom,
                    optimizer.param_groups[0]["lr"],
                    max_mem,
                    int(loss_info["used_samples"]),
                    int(loss_info["skipped_samples"]),
                    step_per_sec,
                    format_seconds(now - epoch_start),
                    format_seconds(eta_seconds),
                )

        denom = max(used_steps, 1)
        train_summary = {"epoch": int(epoch), "used_steps": int(used_steps), "optimizer_steps": int(optimizer_steps), "elapsed_seconds": float(time.time() - epoch_start)}
        for key, value in running.items():
            train_summary[key] = float(value / denom)
        train_history.append(train_summary)
        save_json(save_path / "train_loss_summary.json", {"epochs": train_history})

        val_summary = None
        kitti_val_summary = None
        epoch_ckpt_path = heavy_save_path / f"epoch_{epoch:02d}.pth"
        if ((epoch + 1) % args.eval_interval) == 0:
            val_summary = evaluate_model(model, val_loader, args, device, epoch=epoch, amp_dtype=amp_dtype, logger=logger)
            val_history.append(val_summary)
            save_json(save_path / "val_metrics.json", {"epochs": val_history, "latest": val_summary})
            if args.eval_kitti:
                if kitti_val_dataset is None or kitti_val_loader is None:
                    raise RuntimeError("KITTI eval requested but loader was not built.")
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
                save_json(save_path / "kitti_val_metrics.json", {"epochs": kitti_val_history, "latest": kitti_val_summary})

            current_abs_rel = val_summary["overall"]["final"]["abs_rel"]
            current_boundary = val_summary["region"]["final"]["boundary_abs_rel"]
            current_target_score = val_summary.get("target_region_score")
            current_kitti_abs_rel = None if kitti_val_summary is None else kitti_val_summary["overall"]["final"]["abs_rel"]
            best_abs_rel_record = update_best_record(best_abs_rel_record, key_value=current_abs_rel, minimize=True, epoch=epoch, checkpoint_path=epoch_ckpt_path, val_summary=val_summary, kitti_summary=kitti_val_summary)
            best_boundary_record = update_best_record(best_boundary_record, key_value=current_boundary, minimize=True, epoch=epoch, checkpoint_path=epoch_ckpt_path, val_summary=val_summary, kitti_summary=kitti_val_summary)
            best_target_region_record = update_best_record(best_target_region_record, key_value=current_target_score, minimize=True, epoch=epoch, checkpoint_path=epoch_ckpt_path, val_summary=val_summary, kitti_summary=kitti_val_summary)
            best_kitti_abs_rel_record = update_best_record(best_kitti_abs_rel_record, key_value=current_kitti_abs_rel, minimize=True, epoch=epoch, checkpoint_path=epoch_ckpt_path, val_summary=val_summary, kitti_summary=kitti_val_summary)
            save_json(save_path / "best_val_metrics.json", best_abs_rel_record or {})
            save_json(save_path / "best_boundary_metrics.json", best_boundary_record or {})
            save_json(save_path / "best_target_region_metrics.json", best_target_region_record or {})
            if best_kitti_abs_rel_record is not None:
                save_json(save_path / "best_kitti_val_metrics.json", best_kitti_abs_rel_record)
            if args.save_best_checkpoint:
                if best_abs_rel_record is not None and best_abs_rel_record["epoch"] == epoch:
                    save_checkpoint(heavy_save_path / "best_abs_rel.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)
                if best_target_region_record is not None and best_target_region_record["epoch"] == epoch:
                    save_checkpoint(heavy_save_path / "best_target_region_score.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)

        if ((epoch + 1) % args.save_interval) == 0:
            save_checkpoint(epoch_ckpt_path, model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)
        save_checkpoint(heavy_save_path / "latest.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)

    save_json(
        save_path / "run_summary.json",
        {
            "config": config_payload,
            "train": train_history,
            "val": val_history,
            "vkitti_val": val_history,
            "kitti_val": kitti_val_history,
            "best_abs_rel": best_abs_rel_record,
            "best_kitti_abs_rel": best_kitti_abs_rel_record,
            "best_boundary_abs_rel": best_boundary_record,
            "best_target_region_score": best_target_region_record,
            "heavy_save_path": str(heavy_save_path),
        },
    )


if __name__ == "__main__":
    main()
