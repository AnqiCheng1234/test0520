#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import pprint
import random
import sys
import time
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
from foundation.engine.datasets.led_hb import (
    LED_DEPTH_LABEL_CHOICES,
    LED_DEPTH_TARGET_SPACE_CHOICES,
    LED_DEPTH_UNIT_CHOICES,
    LED_GEOMETRY_MODE_CHOICES,
    LED_RGB_INPUT_SPACE_CHOICES,
    LEDHBHalfresRGBDepth,
    validate_led_hb_rgb_depth_semantics,
)
from foundation.engine.models.dav2_residual_control import (
    CONTROL_FEATURE_SOURCES,
    build_dav2_residual_control_model,
)
from foundation.tools.eval_led_hb_d0 import LED_REGION_KEYS, sample_led_d0_region_metrics
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    attach_file_logger,
    average_dicts,
    compute_residual_loss,
    count_parameters,
    float_or_none,
    format_seconds,
    resolve_model_state,
    save_checkpoint,
    save_json,
    strip_module_prefix,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LED-HB RGB/D0 control residual refinement over frozen RGB DAv2.")
    parser.add_argument("--experiment-id", required=True, choices=["C2"])
    parser.add_argument("--dataset-name", required=True, choices=["led_hb"])
    parser.add_argument("--illumination", required=True, choices=["HB"])
    parser.add_argument("--input-domain", required=True, choices=["rgb"])
    parser.add_argument("--model-input-tensor", required=True, choices=["image"])
    parser.add_argument("--dataset-geometry-mode", required=True, choices=LED_GEOMETRY_MODE_CHOICES)
    parser.add_argument("--raw-storage-format", required=True, choices=["not_applicable"])
    parser.add_argument("--rgb-input-space", required=True, choices=LED_RGB_INPUT_SPACE_CHOICES)
    parser.add_argument("--depth-target-space", required=True, choices=LED_DEPTH_TARGET_SPACE_CHOICES)
    parser.add_argument("--depth-label", required=True, choices=LED_DEPTH_LABEL_CHOICES)
    parser.add_argument("--depth-unit", required=True, choices=LED_DEPTH_UNIT_CHOICES)
    parser.add_argument("--front-end", required=True, choices=["dav2_rgb_frozen"])
    parser.add_argument("--encoder", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--pretrained-from", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--led-train-list", required=True)
    parser.add_argument("--led-val-list", required=True)
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--residual-feature-source", required=True, choices=CONTROL_FEATURE_SOURCES)
    parser.add_argument("--residual-alpha", type=float, required=True)
    parser.add_argument("--d0-sign", type=int, required=True, choices=[-1, 1])
    parser.add_argument("--hflip-prob", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=20)
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
        unprocessing_method="not_applicable",
    )
    expected = {
        "experiment_id": "C2",
        "front_end": "dav2_rgb_frozen",
        "residual_feature_source": "d0",
        "train_split": "hb_train_all",
        "val_split": "hb_val_stride5_n1000_seed42",
    }
    for attr, value in expected.items():
        if str(getattr(args, attr)) != value:
            raise ValueError(f"{attr} must be {value!r}, got {getattr(args, attr)!r}")
    if not (0.0 <= args.hflip_prob <= 1.0):
        raise ValueError(f"--hflip-prob must be in [0,1], got {args.hflip_prob}")
    if args.bs <= 0 or args.accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("bs, accum-steps, and epochs must be positive.")
    if args.save_interval <= 0 or args.eval_interval <= 0:
        raise ValueError("save-interval and eval-interval must be positive.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive when provided.")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise ValueError("--max-val-samples must be positive when provided.")
    if args.residual_alpha <= 0.0:
        raise ValueError(f"--residual-alpha must be positive, got {args.residual_alpha}")
    for path_attr in ("led_train_list", "led_val_list", "pretrained_from"):
        if not Path(getattr(args, path_attr)).expanduser().is_file():
            raise FileNotFoundError(f"Missing required file for {path_attr}: {getattr(args, path_attr)}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subtract_dicts(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...] | list[str]) -> dict[str, float | None]:
    return {key: None if a.get(key) is None or b.get(key) is None else float(a[key]) - float(b[key]) for key in keys}


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
        if args.max_val_samples is not None and processed >= int(args.max_val_samples):
            break
        image = batch["image"].to(device, non_blocking=True).float()
        depth = batch["depth"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
        valid_mask = valid_mask & (depth >= args.min_depth) & (depth <= args.max_depth)
        if int(valid_mask[0].sum().item()) < 128:
            continue
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
            out = model({"image": image, "valid_mask": valid_mask})
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
        region_d0 = sample_led_d0_region_metrics(
            depth_np=depth_np,
            valid_np=valid_np,
            aligned_d0=aligned_d0,
            d0_norm_np=out["D0_norm"][0].float().detach().cpu().numpy(),
            y_norm_np=y_norm[0].float().detach().cpu().numpy(),
            rgb_preview_np=rgb_preview,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        region_final = sample_led_d0_region_metrics(
            depth_np=depth_np,
            valid_np=valid_np,
            aligned_d0=aligned_final,
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
    region_final = average_dicts(final_regions, LED_REGION_KEYS)
    region_d0 = average_dicts(d0_regions, LED_REGION_KEYS)
    delta = subtract_dicts(overall_final, overall_d0, METRIC_KEYS)
    region_delta = subtract_dicts(region_final, region_d0, LED_REGION_KEYS)
    diag = average_dicts(diagnostics, ["mean_gate", "max_gate", "mean_abs_delta", "mean_abs_gate_delta", "mean_abs_final_minus_d0_norm"])
    summary = {
        "epoch": int(epoch),
        "samples": int(processed),
        "max_val_samples": args.max_val_samples,
        "alignment_protocol": "per_image_affine_disp_depth_anything_v2",
        "overall": {
            "final": overall_final,
            "D0": overall_d0,
            "delta": {"final_minus_D0": delta},
            "delta_final_minus_D0": delta,
        },
        "region": {
            "final": region_final,
            "D0": region_d0,
            "delta": {"final_minus_D0": region_delta},
            "delta_final_minus_D0": region_delta,
        },
        "diagnostics": diag,
        "elapsed_seconds": float(time.time() - start),
    }
    logger.info(
        "[EVAL] done epoch=%d samples=%d final_abs_rel=%.5f D0_abs_rel=%.5f delta_abs_rel=%.5f elapsed=%s",
        epoch,
        processed,
        float(overall_final["abs_rel"]),
        float(overall_d0["abs_rel"]),
        float(delta["abs_rel"]),
        format_seconds(summary["elapsed_seconds"]),
    )
    return summary


def build_loaders(args: argparse.Namespace) -> tuple[LEDHBHalfresRGBDepth, LEDHBHalfresRGBDepth, DataLoader, DataLoader]:
    train_dataset = LEDHBHalfresRGBDepth(
        filelist_path=args.led_train_list,
        mode="train",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        hflip_prob=args.hflip_prob,
        include_geometry=True,
        dataset_name=args.dataset_name,
        illumination=args.illumination,
        led_geometry_mode=args.dataset_geometry_mode,
        raw_storage_format=args.raw_storage_format,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
        depth_label=args.depth_label,
        depth_unit=args.depth_unit,
    )
    val_dataset = LEDHBHalfresRGBDepth(
        filelist_path=args.led_val_list,
        mode="val",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        hflip_prob=0.0,
        include_geometry=True,
        dataset_name=args.dataset_name,
        illumination=args.illumination,
        led_geometry_mode=args.dataset_geometry_mode,
        raw_storage_format=args.raw_storage_format,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
        depth_label=args.depth_label,
        depth_unit=args.depth_unit,
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


def build_config_payload(args: argparse.Namespace, train_dataset: LEDHBHalfresRGBDepth, val_dataset: LEDHBHalfresRGBDepth, total_params: int, trainable_params: int) -> dict[str, Any]:
    payload = dict(vars(args))
    payload["led_geometry_mode"] = args.dataset_geometry_mode
    payload["eval_protocol"] = "per_image_affine_disp_depth_anything_v2"
    payload["dataset_geometry"] = {
        "train": train_dataset.describe_geometry(),
        "val": val_dataset.describe_geometry(),
        "led_val": val_dataset.describe_geometry(),
    }
    payload["model_param_counts"] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
    }
    explicit = {
        "dataset_name": args.dataset_name,
        "illumination": args.illumination,
        "led_train_list": args.led_train_list,
        "led_val_list": args.led_val_list,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "led_geometry_mode": args.dataset_geometry_mode,
        "depth_label": args.depth_label,
        "depth_unit": args.depth_unit,
        "rgb_input_space": args.rgb_input_space,
        "depth_target_space": args.depth_target_space,
        "eval_protocol": "per_image_affine_disp_depth_anything_v2",
        "input_domain": args.input_domain,
        "model_input_tensor": args.model_input_tensor,
        "raw_storage_format": args.raw_storage_format,
        "input_height": int(args.input_height),
        "input_width": int(args.input_width),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "encoder": args.encoder,
        "pretrained_from": args.pretrained_from,
        "residual_feature_source": args.residual_feature_source,
        "residual_alpha": float(args.residual_alpha),
        "d0_sign": int(args.d0_sign),
    }
    payload.update(explicit)
    return payload


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

    logger = init_log("led_hb_residual_control", logging.INFO) or logging.getLogger("led_hb_residual_control")
    logger.propagate = False
    attach_file_logger(logger, save_path / "train.log")
    logger.info("%s\n", pprint.pformat({**vars(args), "device": str(device)}))

    cudnn.enabled = True
    cudnn.benchmark = True
    set_random_seed(args.seed)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    base_model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model = build_dav2_residual_control_model(
        base_model,
        residual_feature_source=args.residual_feature_source,
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
    total_params, trainable_param_count = count_parameters(model)
    config_payload = build_config_payload(args, train_dataset, val_dataset, total_params, trainable_param_count)
    save_json(save_path / "config.json", config_payload)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")
    logger.info("[MODEL] total_params=%d trainable_params=%d frozen_params=%d", total_params, trainable_param_count, total_params - trainable_param_count)
    logger.info("[DATASET] train_samples=%d led_val_samples=%d", len(train_dataset), len(val_dataset))
    logger.info("[DATASET] geometry=%s", config_payload["dataset_geometry"])

    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    best_abs_rel = float("inf")
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
            depth = batch["depth"].to(device, non_blocking=True).float()
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()
            if epoch == start_epoch and step_idx == 0:
                logger.info("[BATCH] image=%s depth=%s valid=%s samples=%s", tuple(image.shape), tuple(depth.shape), tuple(valid_mask.shape), batch["sample_name"][: min(2, len(batch["sample_name"]))])
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
                out = model({"image": image, "valid_mask": valid_mask})
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
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d opt_step=%d loss_total=%.5f mean_gate=%.5f mean_abs_gate_delta=%.5f lr=%.7f max_mem_mb=%.0f elapsed=%s",
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
            if current_abs_rel is not None and float(current_abs_rel) < best_abs_rel:
                best_abs_rel = float(current_abs_rel)
                save_json(save_path / "best_val_metrics.json", val_summary)
                if args.save_best_checkpoint:
                    save_checkpoint(heavy_save_path / "best_abs_rel.pth", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, args=args, train_summary=train_summary, val_summary=val_summary)

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
            "heavy_save_path": str(heavy_save_path),
        },
    )


if __name__ == "__main__":
    main()
