#!/usr/bin/env python3
"""Run true-LOD RAW_dark/RAW_normal checkpoint cross-evaluation.

The matrix is:
  M_DD: C_dark   on I_dark
  M_DN: C_dark   on I_normal
  M_NN: C_normal on I_normal
  M_ND: C_normal on I_dark

Each cell can be evaluated strictly, or after front-end BN recalibration on the
matching train input domain. Recalibration updates only RAM/front-end BN running
statistics; no parameters are changed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.lod_true import (  # noqa: E402
    LOD_RAW_RGB16_NORM_MODE,
    LOD_RAW_RGB16_STORAGE_FORMAT,
    LODTrueRawDarkRGB16,
)
from finetune_stf.train import (  # noqa: E402
METRIC_KEYS,
    build_model,
    parse_args as train_parse_args,
    resolve_model_state,
    set_random_seed,
    strip_module_prefix,
)
from finetune_stf.util.metric import (  # noqa: E402
    affine_align_to_inverse_target,
    compute_inverse_relative_metrics,
)
from finetune_stf.util.model_input import select_model_input  # noqa: E402


DARK_RUN_ID = "0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"
NORMAL_RUN_ID = "0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"
DEFAULT_CKPT_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")
DEFAULT_EXP_ROOT = PROJECT_ROOT / "finetune_stf" / "exp"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "finetune_stf" / "analysis" / "lod_raw_cross_eval"
POST_RAM_DEBUG_KEYS = (
    "post_ram_enabled",
    "post_ram_scale",
    "post_ram_mean_abs_xram",
    "post_ram_mean_abs_delta",
    "post_ram_delta_ratio",
    "post_ram_xram_p1",
    "post_ram_xram_p50",
    "post_ram_xram_p99",
    "post_ram_xclean_p1",
    "post_ram_xclean_p50",
    "post_ram_xclean_p99",
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    run_id: str
    run_dir: Path
    checkpoint: Path
    train_dataset_family: str
    train_dataset_input_mode: str


@dataclass(frozen=True)
class InputSpec:
    label: str
    dataset_family: str
    dataset_input_mode: str


@dataclass(frozen=True)
class ComboSpec:
    matrix_id: str
    checkpoint: CheckpointSpec
    eval_input: InputSpec


INPUT_DARK = InputSpec(
    label="I_dark",
    dataset_family="lod_true_raw_dark_rgb16",
    dataset_input_mode="raw_rgb16_dark",
)
INPUT_NORMAL = InputSpec(
    label="I_normal",
    dataset_family="lod_true_raw_normal_rgb16",
    dataset_input_mode="raw_rgb16_normal",
)


def parse_args() -> argparse.Namespace:
    timestamp = dt.datetime.now().strftime("%m%d_%H%M")
    default_output = DEFAULT_ANALYSIS_ROOT / f"{timestamp}_lod_raw_dark_normal_cross_eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dark-run-dir", type=Path, default=DEFAULT_EXP_ROOT / DARK_RUN_ID)
    parser.add_argument("--normal-run-dir", type=Path, default=DEFAULT_EXP_ROOT / NORMAL_RUN_ID)
    parser.add_argument(
        "--dark-checkpoint",
        type=Path,
        default=DEFAULT_CKPT_ROOT / DARK_RUN_ID / "best_model.pth",
    )
    parser.add_argument(
        "--normal-checkpoint",
        type=Path,
        default=DEFAULT_CKPT_ROOT / NORMAL_RUN_ID / "best_model.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--lod-root", type=Path, default=None)
    parser.add_argument("--lod-manifest", type=Path, default=None)
    parser.add_argument("--pretrained-from", type=Path, default=None)
    parser.add_argument("--eval-split", default="01Valid")
    parser.add_argument("--bn-recalib-split", default="00Train")
    parser.add_argument("--variants", default="strict,bn_recalib")
    parser.add_argument("--combos", default="M_DD,M_DN,M_NN,M_ND")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--bn-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-bn-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default=None)
    parser.add_argument(
        "--bn-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset front-end BN running stats before cumulative recalibration.",
    )
    parser.add_argument("--progress-interval", type=int, default=50)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = run_dir.expanduser().resolve() / "config.json"
    resolved_path = run_dir.expanduser().resolve() / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {resolved_path}")
    return load_json(config_path), load_json(resolved_path)


def validate_checkpoint_run(spec: CheckpointSpec, resolved: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {
        "dataset_family": spec.train_dataset_family,
        "dataset_input_mode": spec.train_dataset_input_mode,
        "input_domain": "raw3",
        "front_end": "raw_rgb16_ram3",
        "model_input_tensor": "raw",
        "raw_storage_format": LOD_RAW_RGB16_STORAGE_FORMAT,
    }
    mismatches = []
    for key, expected_value in expected.items():
        got = resolved.get(key)
        if str(got) != str(expected_value):
            mismatches.append(f"{key}: got={got!r} expected={expected_value!r}")
    if str(config.get("raw_ram_rgb_tail")) != "identity":
        mismatches.append(f"raw_ram_rgb_tail: got={config.get('raw_ram_rgb_tail')!r} expected='identity'")
    if mismatches:
        raise ValueError(f"{spec.label} run semantics do not match expected cross-eval setup:\n" + "\n".join(mismatches))
    if not spec.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"Missing {spec.label} checkpoint: {spec.checkpoint}")


def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def train_args_from_run_config(
    *,
    config: dict[str, Any],
    eval_input: InputSpec,
    output_dir: Path,
    args: argparse.Namespace,
) -> argparse.Namespace:
    pretrained = args.pretrained_from or Path(str(config_value(config, "pretrained_from", "")))
    lod_root = args.lod_root or Path(str(config_value(config, "lod_root", "")))
    lod_manifest = args.lod_manifest or Path(str(config_value(config, "lod_manifest", "")))
    if not str(pretrained):
        raise ValueError("Could not resolve --pretrained-from from CLI or run config")
    if not str(lod_root) or not str(lod_manifest):
        raise ValueError("Could not resolve LOD root/manifest from CLI or run config")

    argv = [
        "finetune_stf/train.py",
        "--stage",
        "eval_only",
        "--encoder",
        str(config_value(config, "encoder", "vits")),
        "--dataset-family",
        eval_input.dataset_family,
        "--dataset-input-mode",
        eval_input.dataset_input_mode,
        "--input-domain",
        str(config_value(config, "input_domain", "raw3")),
        "--front-end",
        str(config_value(config, "front_end", "raw_rgb16_ram3")),
        "--model-input-tensor",
        str(config_value(config, "model_input_tensor", "raw")),
        "--bridge",
        str(config_value(config, "bridge", "none")),
        "--decoder-feature-adapter",
        str(config_value(config, "decoder_feature_adapter", "none")),
        "--lora",
        str(config_value(config, "lora", "none")),
        "--pretrained-from",
        str(Path(pretrained).expanduser()),
        "--lod-root",
        str(Path(lod_root).expanduser()),
        "--lod-manifest",
        str(Path(lod_manifest).expanduser()),
        "--lod-label-space",
        str(config_value(config, "lod_label_space", "inverse_relative")),
        "--lod-train-crop-mode",
        str(config_value(config, "lod_train_crop_mode", "random")),
        "--lod-val-crop-mode",
        str(config_value(config, "lod_val_crop_mode", "center")),
        "--raw-storage-format",
        str(config_value(config, "raw_storage_format", LOD_RAW_RGB16_STORAGE_FORMAT)),
        "--lod-raw-norm-mode",
        str(config_value(config, "lod_raw_norm_mode", LOD_RAW_RGB16_NORM_MODE)),
        "--raw-ram-rgb-tail",
        str(config_value(config, "raw_ram_rgb_tail", "identity")),
        "--post-ram-cleanup",
        str(config_value(config, "post_ram_cleanup", "none")),
        "--post-ram-cleanup-channels",
        str(config_value(config, "post_ram_cleanup_channels", "n_a")),
        "--post-ram-cleanup-blocks",
        str(config_value(config, "post_ram_cleanup_blocks", "n_a")),
        "--post-ram-cleanup-scale",
        str(config_value(config, "post_ram_cleanup_scale", "n_a")),
        "--post-ram-cleanup-norm",
        str(config_value(config, "post_ram_cleanup_norm", "n_a")),
        "--post-ram-cleanup-zero-init",
        str(config_value(config, "post_ram_cleanup_zero_init", "n_a")),
        "--post-ram-cleanup-lr",
        str(config_value(config, "post_ram_cleanup_lr", "n_a")),
        "--post-ram-external-denoiser",
        str(config_value(config, "post_ram_external_denoiser", "none")),
        "--post-ram-denoiser-sigma",
        str(config_value(config, "post_ram_denoiser_sigma", "n_a")),
        "--post-ram-denoiser-alpha",
        str(config_value(config, "post_ram_denoiser_alpha", "n_a")),
        "--post-ram-denoiser-affine",
        str(config_value(config, "post_ram_denoiser_affine", "n_a")),
        "--post-ram-denoiser-frozen",
        str(config_value(config, "post_ram_denoiser_frozen", "n_a")),
        "--post-ram-operation-position",
        str(config_value(config, "post_ram_operation_position", "n_a")),
        "--dav2-train-mode",
        str(config_value(config, "dav2_train_mode", "decoder")),
        "--raw-front-end-lr",
        str(float(config_value(config, "raw_front_end_lr", 5e-5))),
        "--lr",
        str(float(config_value(config, "lr", 1e-5))),
        "--input-height",
        str(int(config_value(config, "input_height", 512))),
        "--input-width",
        str(int(config_value(config, "input_width", 960))),
        "--bs",
        str(int(config_value(config, "bs", 8))),
        "--num-workers",
        str(int(args.num_workers)),
        "--eval-lod",
        "--no-eval-stf",
        "--no-enable-fixed-viz-dump",
        "--no-enable-train-source-viz-dump",
        "--save-path",
        str(output_dir / "_build_args"),
        "--heavy-save-root",
        str(output_dir / "_heavy_build_args"),
    ]

    if str(config_value(config, "lora", "none")) == "dav2_lora":
        argv.extend(
            [
                "--lora-block-mode",
                str(config_value(config, "lora_block_mode", "tap")),
                "--lora-rank",
                str(int(config_value(config, "lora_rank", 8))),
                "--lora-alpha",
                str(float(config_value(config, "lora_alpha", 16.0))),
                "--lora-lr",
                str(float(config_value(config, "lora_lr", 5e-5))),
            ]
        )
        tap_layers = config.get("lora_tap_layers")
        if tap_layers:
            argv.extend(["--lora-tap-layers", *[str(int(v)) for v in tap_layers]])

    amp = bool(config_value(config, "amp", True)) if args.amp is None else bool(args.amp)
    amp_dtype = str(config_value(config, "amp_dtype", "bf16")) if args.amp_dtype is None else str(args.amp_dtype)
    argv.extend(["--amp" if amp else "--no-amp", "--amp-dtype", amp_dtype])

    old_argv = sys.argv
    try:
        sys.argv = argv
        train_args = train_parse_args()
    finally:
        sys.argv = old_argv
    return train_args


def make_dataset(
    train_args: argparse.Namespace,
    *,
    input_spec: InputSpec,
    split: str,
    crop_mode: str,
) -> LODTrueRawDarkRGB16:
    return LODTrueRawDarkRGB16(
        lod_root=train_args.lod_root,
        manifest_path=train_args.lod_manifest,
        split=split,
        size=(int(train_args.input_height), int(train_args.input_width)),
        mode="val",
        label_space=train_args.lod_label_space,
        crop_mode=crop_mode,
        raw_storage_format=train_args.raw_storage_format,
        lod_raw_norm_mode=train_args.lod_raw_norm_mode,
        raw_input_mode=input_spec.dataset_input_mode,
    )


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    ckpt_obj = torch.load(checkpoint.expanduser(), map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_state_dict(state_dict, strict=True)
    if not isinstance(ckpt_obj, dict):
        return {"path": str(checkpoint), "epoch": None, "best_metric": None, "best_metrics": None}
    return {
        "path": str(checkpoint),
        "epoch": ckpt_obj.get("epoch"),
        "best_metric": ckpt_obj.get("best_metric"),
        "best_metrics": ckpt_obj.get("best_metrics"),
        "best_lod_d1": ckpt_obj.get("best_lod_d1"),
    }


def amp_dtype_from_train_args(train_args: argparse.Namespace) -> torch.dtype:
    return torch.float16 if str(getattr(train_args, "amp_dtype", "bf16")) == "fp16" else torch.bfloat16


def make_loader(dataset: torch.utils.data.Dataset, *, batch_size: int, num_workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        **kwargs,
    )


def collated_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.ndim == 0 else item
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def metric_average(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def scalar_debug_values(debug: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in POST_RAM_DEBUG_KEYS:
        if key not in debug:
            continue
        value = debug[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float()
            if tensor.numel() != 1:
                continue
            values[key] = float(tensor.cpu().item())
            continue
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def evaluate_model(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    train_args: argparse.Namespace,
    combo: ComboSpec,
    variant: str,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    max_samples: int | None,
    progress_interval: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = make_loader(dataset, batch_size=eval_batch_size, num_workers=num_workers)
    model.eval()
    use_amp = device.type == "cuda" and bool(getattr(train_args, "amp", False))
    amp_dtype = amp_dtype_from_train_args(train_args)
    rows: list[dict[str, Any]] = []
    start = time.time()
    processed = 0

    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and processed >= int(max_samples):
                break
            model_input = select_model_input(
                batch,
                train_args.resolved_config.model_input_tensor,
                dataset_family=train_args.resolved_config.dataset_family,
                sample_source=f"{combo.matrix_id}_{variant}",
            )
            model_input = model_input.to(device=device, non_blocking=True).float()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(model_input, return_features=True)
                pred = output["depth"].float()
                batch_debug = scalar_debug_values(output.get("ram_debug", {}))
            if pred.ndim != 3:
                raise ValueError(f"Expected model output shape (B,H,W), got {tuple(pred.shape)}")
            depth_batch = batch["depth"].float()
            valid_batch = batch["valid_mask"].bool()
            batch_size = int(depth_batch.shape[0])
            if tuple(pred.shape[-2:]) != tuple(depth_batch.shape[-2:]):
                pred = F.interpolate(
                    pred[:, None],
                    tuple(int(v) for v in depth_batch.shape[-2:]),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
            pred_np = pred.detach().cpu().numpy()
            depth_np = depth_batch.detach().cpu().numpy()
            valid_np = valid_batch.detach().cpu().numpy().astype(bool)

            for i in range(batch_size):
                if max_samples is not None and processed >= int(max_samples):
                    break
                target = depth_np[i]
                valid = valid_np[i] & np.isfinite(target) & (target > 0)
                if int(np.count_nonzero(valid)) < 10:
                    continue
                aligned_inverse, align_stats = affine_align_to_inverse_target(pred_np[i], target, valid)
                metrics = compute_inverse_relative_metrics(aligned_inverse, target, valid)
                if metrics is None:
                    continue
                row = {
                    "matrix_id": combo.matrix_id,
                    "variant": variant,
                    "checkpoint_label": combo.checkpoint.label,
                    "eval_input_label": combo.eval_input.label,
                    "checkpoint_train_dataset_family": combo.checkpoint.train_dataset_family,
                    "checkpoint_train_dataset_input_mode": combo.checkpoint.train_dataset_input_mode,
                    "eval_dataset_family": combo.eval_input.dataset_family,
                    "eval_dataset_input_mode": combo.eval_input.dataset_input_mode,
                    "sample_index": int(processed),
                    "sample_id": str(collated_item(batch.get("sample_id", ""), i)),
                    "normal_id": int(collated_item(batch.get("normal_id", -1), i)),
                    "dark_id": int(collated_item(batch.get("dark_id", -1), i)),
                    "valid_pixels": int(align_stats.get("valid_pixels", int(np.count_nonzero(valid)))),
                    "align_scale": float(align_stats.get("scale", float("nan"))),
                    "align_shift": float(align_stats.get("shift", float("nan"))),
                }
                for key in METRIC_KEYS:
                    if key in metrics:
                        row[key] = float(metrics[key])
                for key, value in batch_debug.items():
                    row[key] = float(value)
                rows.append(row)
                processed += 1
            if progress_interval > 0 and (processed % int(progress_interval) == 0) and processed > 0:
                print(f"[EVAL] {combo.matrix_id}/{variant} processed={processed}", flush=True)

    summary = {
        "matrix_id": combo.matrix_id,
        "variant": variant,
        "checkpoint_label": combo.checkpoint.label,
        "checkpoint_run_id": combo.checkpoint.run_id,
        "checkpoint_path": str(combo.checkpoint.checkpoint),
        "eval_input_label": combo.eval_input.label,
        "eval_dataset_family": combo.eval_input.dataset_family,
        "eval_dataset_input_mode": combo.eval_input.dataset_input_mode,
        "samples": len(rows),
        "dataset_size": len(dataset),
        "elapsed_sec": float(time.time() - start),
    }
    for key in METRIC_KEYS:
        summary[key] = metric_average(rows, key)
    for key in POST_RAM_DEBUG_KEYS:
        summary[key] = metric_average(rows, key)
    return summary, rows


def bn_modules(module: torch.nn.Module) -> list[torch.nn.modules.batchnorm._BatchNorm]:
    return [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def recalibrate_frontend_bn(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    train_args: argparse.Namespace,
    combo: ComboSpec,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    reset: bool,
) -> dict[str, Any]:
    if not hasattr(model, "ram_core"):
        raise ValueError("BN recalibration currently requires a model with ram_core")
    front_end = model.ram_core
    modules = bn_modules(front_end)
    if not modules:
        raise ValueError("No front-end BatchNorm modules found to recalibrate")

    old_momenta = [m.momentum for m in modules]
    if reset:
        for module in modules:
            module.reset_running_stats()
    for module in modules:
        module.momentum = None

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    front_end.train()
    loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    start = time.time()
    batches = 0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and batches >= int(max_batches):
                break
            model_input = select_model_input(
                batch,
                train_args.resolved_config.model_input_tensor,
                dataset_family=train_args.resolved_config.dataset_family,
                sample_source=f"{combo.matrix_id}_bn_recalib",
            )
            model_input = model_input.to(device=device, non_blocking=True).float()
            _ = front_end(model_input)
            batches += 1
            samples += int(model_input.shape[0])

    for module, old_momentum in zip(modules, old_momenta):
        module.momentum = old_momentum
    front_end.eval()
    model.eval()
    return {
        "bn_recalib_split": str(getattr(dataset, "split", "unknown")),
        "bn_recalib_samples": int(samples),
        "bn_recalib_batches": int(batches),
        "bn_recalib_bn_modules": int(len(modules)),
        "bn_recalib_reset": bool(reset),
        "bn_recalib_elapsed_sec": float(time.time() - start),
        "bn_recalib_mode": "front_end_ram_core_only",
    }


def build_combos(dark: CheckpointSpec, normal: CheckpointSpec) -> dict[str, ComboSpec]:
    return {
        "M_DD": ComboSpec("M_DD", dark, INPUT_DARK),
        "M_DN": ComboSpec("M_DN", dark, INPUT_NORMAL),
        "M_NN": ComboSpec("M_NN", normal, INPUT_NORMAL),
        "M_ND": ComboSpec("M_ND", normal, INPUT_DARK),
    }


def recovery_rows(matrix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {}
    for row in matrix_rows:
        by_variant.setdefault(str(row["variant"]), {})[str(row["matrix_id"])] = row

    rows = []
    for variant, values in sorted(by_variant.items()):
        required = {"M_DD", "M_DN", "M_NN", "M_ND"}
        if not required.issubset(values):
            continue
        m_dd = float(values["M_DD"]["d1"])
        m_dn = float(values["M_DN"]["d1"])
        m_nn = float(values["M_NN"]["d1"])
        m_nd = float(values["M_ND"]["d1"])
        gap = m_nn - m_dd
        r_dn = (m_dn - m_dd) / gap if abs(gap) > 1e-12 else float("nan")
        rows.append(
            {
                "variant": variant,
                "M_DD_d1": m_dd,
                "M_DN_d1": m_dn,
                "M_NN_d1": m_nn,
                "M_ND_d1": m_nd,
                "G_M_NN_minus_M_DD": gap,
                "R_DN": r_dn,
                "M_DN_minus_M_DD": m_dn - m_dd,
                "M_ND_minus_M_DD": m_nd - m_dd,
            }
        )
    return rows


def write_report(path: Path, matrix_rows: list[dict[str, Any]], recovery: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# LOD RAW Cross-Eval",
        "",
        f"- output_dir: `{meta['output_dir']}`",
        f"- eval_split: `{meta['eval_split']}`",
        f"- bn_recalib_split: `{meta['bn_recalib_split']}`",
        f"- bn_recalib_definition: reset running stats, cumulative average, RAM/front-end only, no gradient",
        "",
        "## Matrix",
        "",
        "| variant | matrix | checkpoint | input | D1 | AbsRel | RMSE | post_delta_ratio | samples | BN samples |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matrix_rows:
        lines.append(
            "| {variant} | {matrix_id} | {checkpoint_label} | {eval_input_label} | {d1:.6f} | "
            "{abs_rel:.6f} | {rmse:.6f} | {post_delta_ratio:.6f} | {samples} | {bn_samples} |".format(
                variant=row["variant"],
                matrix_id=row["matrix_id"],
                checkpoint_label=row["checkpoint_label"],
                eval_input_label=row["eval_input_label"],
                d1=float(row["d1"]),
                abs_rel=float(row["abs_rel"]),
                rmse=float(row["rmse"]),
                post_delta_ratio=float(row.get("post_ram_delta_ratio", float("nan"))),
                samples=int(row["samples"]),
                bn_samples=int(row.get("bn_recalib_samples") or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Recovery",
            "",
            "| variant | G=M_NN-M_DD | R_DN | M_DN-M_DD | M_ND-M_DD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in recovery:
        lines.append(
            "| {variant} | {gap:.6f} | {r_dn:.6f} | {dn_delta:.6f} | {nd_delta:.6f} |".format(
                variant=row["variant"],
                gap=float(row["G_M_NN_minus_M_DD"]),
                r_dn=float(row["R_DN"]),
                dn_delta=float(row["M_DN_minus_M_DD"]),
                nd_delta=float(row["M_ND_minus_M_DD"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dark = CheckpointSpec(
        label="C_dark",
        run_id=args.dark_run_dir.name,
        run_dir=args.dark_run_dir.expanduser().resolve(),
        checkpoint=args.dark_checkpoint.expanduser().resolve(),
        train_dataset_family="lod_true_raw_dark_rgb16",
        train_dataset_input_mode="raw_rgb16_dark",
    )
    normal = CheckpointSpec(
        label="C_normal",
        run_id=args.normal_run_dir.name,
        run_dir=args.normal_run_dir.expanduser().resolve(),
        checkpoint=args.normal_checkpoint.expanduser().resolve(),
        train_dataset_family="lod_true_raw_normal_rgb16",
        train_dataset_input_mode="raw_rgb16_normal",
    )

    run_payloads: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for spec in (dark, normal):
        config, resolved = get_config(spec.run_dir)
        validate_checkpoint_run(spec, resolved, config)
        run_payloads[spec.label] = (config, resolved)

    variants = split_csv(args.variants)
    allowed_variants = {"strict", "bn_recalib"}
    unknown_variants = sorted(set(variants) - allowed_variants)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}; expected {sorted(allowed_variants)}")

    combos = build_combos(dark, normal)
    requested_combos = split_csv(args.combos)
    unknown_combos = sorted(set(requested_combos) - set(combos))
    if unknown_combos:
        raise ValueError(f"Unknown combos: {unknown_combos}; expected {sorted(combos)}")

    print(f"[CROSS_EVAL] output_dir={output_dir}", flush=True)
    print(f"[CROSS_EVAL] device={device} variants={variants} combos={requested_combos}", flush=True)

    matrix_rows: list[dict[str, Any]] = []
    all_per_sample_rows: list[dict[str, Any]] = []
    combo_meta: dict[str, Any] = {}

    for combo_id in requested_combos:
        combo = combos[combo_id]
        config, _ = run_payloads[combo.checkpoint.label]
        train_args = train_args_from_run_config(
            config=config,
            eval_input=combo.eval_input,
            output_dir=output_dir / "build_args" / combo.matrix_id,
            args=args,
        )
        eval_dataset = make_dataset(
            train_args,
            input_spec=combo.eval_input,
            split=args.eval_split,
            crop_mode=str(config_value(config, "lod_val_crop_mode", "center")),
        )
        bn_dataset = make_dataset(
            train_args,
            input_spec=combo.eval_input,
            split=args.bn_recalib_split,
            crop_mode="center",
        )

        for variant in variants:
            print(
                f"[CROSS_EVAL] start {combo.matrix_id}/{variant} "
                f"ckpt={combo.checkpoint.label} input={combo.eval_input.label}",
                flush=True,
            )
            model = build_model(train_args)
            ckpt_meta = load_checkpoint_into_model(model, combo.checkpoint.checkpoint)
            model.to(device)
            recalib_meta: dict[str, Any] = {}
            if variant == "bn_recalib":
                recalib_meta = recalibrate_frontend_bn(
                    model=model,
                    dataset=bn_dataset,
                    train_args=train_args,
                    combo=combo,
                    device=device,
                    batch_size=args.bn_batch_size,
                    num_workers=args.num_workers,
                    max_batches=args.max_bn_batches,
                    reset=args.bn_reset,
                )
                print(
                    f"[CROSS_EVAL] recalibrated {combo.matrix_id}: "
                    f"samples={recalib_meta['bn_recalib_samples']} "
                    f"batches={recalib_meta['bn_recalib_batches']}",
                    flush=True,
                )

            summary, per_sample_rows = evaluate_model(
                model=model,
                dataset=eval_dataset,
                train_args=train_args,
                combo=combo,
                variant=variant,
                device=device,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_eval_samples,
                progress_interval=args.progress_interval,
            )
            summary.update(recalib_meta)
            summary["checkpoint_epoch"] = ckpt_meta.get("epoch")
            summary["checkpoint_best_metric"] = ckpt_meta.get("best_metric")
            summary["checkpoint_best_lod_d1"] = ckpt_meta.get("best_lod_d1")
            matrix_rows.append(summary)
            all_per_sample_rows.extend(per_sample_rows)
            combo_meta[f"{combo.matrix_id}_{variant}"] = {
                "train_args_resolved_config": train_args.resolved_config.to_dict(),
                "checkpoint_meta": ckpt_meta,
                "dataset_size": len(eval_dataset),
                "bn_dataset_size": len(bn_dataset),
            }
            per_sample_path = output_dir / "per_sample" / f"{combo.matrix_id}_{variant}.csv"
            write_csv(per_sample_path, per_sample_rows)
            print(
                f"[CROSS_EVAL] done {combo.matrix_id}/{variant} "
                f"d1={summary['d1']:.6f} abs_rel={summary['abs_rel']:.6f} samples={summary['samples']}",
                flush=True,
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    recovery = recovery_rows(matrix_rows)
    meta = {
        "output_dir": str(output_dir),
        "eval_split": str(args.eval_split),
        "bn_recalib_split": str(args.bn_recalib_split),
        "variants": variants,
        "combos": requested_combos,
        "dark_run": {
            "run_dir": str(dark.run_dir),
            "checkpoint": str(dark.checkpoint),
        },
        "normal_run": {
            "run_dir": str(normal.run_dir),
            "checkpoint": str(normal.checkpoint),
        },
        "device": str(device),
        "seed": int(args.seed),
        "max_eval_samples": args.max_eval_samples,
        "max_bn_batches": args.max_bn_batches,
        "combo_meta": combo_meta,
    }
    write_csv(output_dir / "matrix.csv", matrix_rows)
    write_csv(output_dir / "recovery.csv", recovery)
    write_csv(output_dir / "per_sample_all.csv", all_per_sample_rows)
    write_json(output_dir / "summary.json", {"meta": meta, "matrix": matrix_rows, "recovery": recovery})
    write_report(output_dir / "report.md", matrix_rows, recovery, meta)
    print(f"[CROSS_EVAL] wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
