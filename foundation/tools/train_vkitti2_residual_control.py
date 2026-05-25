from __future__ import annotations

import argparse
import json
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
from foundation.engine.datasets.vkitti2_halfres_rgb_depth import (
    VKITTI2HalfresRGBDepth,
    validate_vkitti_halfres_rgb_depth_semantics,
)
from foundation.engine.models.dav2_residual_control import (
    CONTROL_FEATURE_SOURCES,
    build_dav2_residual_control_model,
)
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    REGION_KEYS,
    attach_file_logger,
    average_dicts,
    compute_residual_loss,
    count_parameters,
    float_or_none,
    format_seconds,
    resolve_model_state,
    sample_region_metrics,
    save_checkpoint,
    save_json,
    strip_module_prefix,
)
from foundation.tools.residual_control_kitti_eval import (
    CONTROL_KITTI_EVAL_PROTOCOL,
    build_kitti_val_loader,
    evaluate_control_kitti_model,
)


CONTROL_FRONT_END_CHOICES = ("dav2_rgb_frozen",)
CONTROL_RAW_STORAGE_CHOICES = ("not_applicable",)
CONTROL_FULLRES_EVEN_POLICY_CHOICES = ("crop_bottom_to_even",)
CONTROL_RGB_INPUT_SPACE_CHOICES = ("halfres_2x2_area",)
CONTROL_DEPTH_TARGET_SPACE_CHOICES = ("halfres_2x2_valid_mean",)
CONTROL_INPUT_DOMAIN_CHOICES = ("rgb",)
CONTROL_MODEL_INPUT_CHOICES = ("image",)
CONTROL_DATASET_GEOMETRY_CHOICES = ("vkitti2_even_fullres_halfres_2x2",)

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VKITTI2 RGB/D0 control residual refinement over frozen RGB DAv2.")
    parser.add_argument("--experiment-id", required=True, choices=["C1", "C2"])
    parser.add_argument("--input-domain", required=True, choices=CONTROL_INPUT_DOMAIN_CHOICES)
    parser.add_argument("--model-input-tensor", required=True, choices=CONTROL_MODEL_INPUT_CHOICES)
    parser.add_argument("--dataset-geometry-mode", required=True, choices=CONTROL_DATASET_GEOMETRY_CHOICES)
    parser.add_argument("--raw-storage-format", required=True, choices=CONTROL_RAW_STORAGE_CHOICES)
    parser.add_argument("--fullres-even-policy", required=True, choices=CONTROL_FULLRES_EVEN_POLICY_CHOICES)
    parser.add_argument("--rgb-input-space", required=True, choices=CONTROL_RGB_INPUT_SPACE_CHOICES)
    parser.add_argument("--depth-target-space", required=True, choices=CONTROL_DEPTH_TARGET_SPACE_CHOICES)
    parser.add_argument("--front-end", required=True, choices=CONTROL_FRONT_END_CHOICES)
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
        choices=[CONTROL_KITTI_EVAL_PROTOCOL],
    )
    parser.add_argument("--kitti-expected-val-samples", type=int, default=None)
    parser.add_argument("--kitti-num-workers", type=int, default=None)
    parser.add_argument("--max-kitti-val-samples", type=int, default=None)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--residual-feature-source", required=True, choices=CONTROL_FEATURE_SOURCES)
    parser.add_argument("--residual-alpha", type=float, required=True)
    parser.add_argument("--d0-sign", type=int, required=True, choices=[-1, 1])
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
    expected_common = {
        "input_domain": "rgb",
        "model_input_tensor": "image",
        "dataset_geometry_mode": "vkitti2_even_fullres_halfres_2x2",
        "raw_storage_format": "not_applicable",
        "front_end": "dav2_rgb_frozen",
        "fullres_even_policy": "crop_bottom_to_even",
        "rgb_input_space": "halfres_2x2_area",
        "depth_target_space": "halfres_2x2_valid_mean",
    }
    for attr, value in expected_common.items():
        if getattr(args, attr) != value:
            raise ValueError(f"{attr} must be {value!r}, got {getattr(args, attr)!r}")

    validate_vkitti_halfres_rgb_depth_semantics(
        raw_storage_format=args.raw_storage_format,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    expected_feature = {"C1": "rgb", "C2": "d0"}[args.experiment_id]
    if args.residual_feature_source != expected_feature:
        raise ValueError(
            f"{args.experiment_id} requires residual_feature_source={expected_feature!r}, "
            f"got {args.residual_feature_source!r}"
        )
    if (args.input_height, args.input_width) != (187, 621):
        raise ValueError(f"C-series halfres control requires input size (187, 621), got {(args.input_height, args.input_width)}")
    if not (0.0 < args.min_depth < args.max_depth):
        raise ValueError(f"Expected 0 < min_depth < max_depth, got {args.min_depth}, {args.max_depth}")
    if args.residual_alpha <= 0.0:
        raise ValueError(f"--residual-alpha must be positive, got {args.residual_alpha}")
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
        if args.kitti_eval_protocol != CONTROL_KITTI_EVAL_PROTOCOL:
            raise ValueError(f"--eval-kitti requires --kitti-eval-protocol {CONTROL_KITTI_EVAL_PROTOCOL}.")
        if not Path(args.kitti_val_split).expanduser().is_file():
            raise FileNotFoundError(f"Missing KITTI val split: {args.kitti_val_split}")
        if not Path(args.kitti_base).expanduser().is_dir():
            raise FileNotFoundError(f"Missing KITTI base directory: {args.kitti_base}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        region_final, region_d0 = sample_region_metrics(
            depth_np=depth_np,
            valid_np=valid_np,
            aligned_final=aligned_final,
            aligned_d0=aligned_d0,
            d0_norm_np=out["D0_norm"][0].float().detach().cpu().numpy(),
            y_norm_np=y_norm[0].float().detach().cpu().numpy(),
            rgb_preview_np=rgb_preview,
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
        "alignment_protocol": "per_image_affine_disp_depth_anything_v2",
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


def build_loaders(args: argparse.Namespace) -> tuple[VKITTI2HalfresRGBDepth, VKITTI2HalfresRGBDepth, DataLoader, DataLoader]:
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

    logger = init_log("vkitti2_residual_control", logging.INFO) or logging.getLogger("vkitti2_residual_control")
    logger.propagate = False
    attach_file_logger(logger, save_path / "train.log")
    logger.info("%s\n", pprint.pformat({**vars(args), "device": str(device)}))

    cudnn.enabled = True
    cudnn.benchmark = True
    set_random_seed(args.seed)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    kitti_val_dataset, kitti_val_loader = build_kitti_val_loader(args, device)
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
    config_payload = dict(vars(args))
    vkitti_val_geometry = val_dataset.describe_geometry()
    config_payload["dataset_geometry"] = {
        "train": train_dataset.describe_geometry(),
        "val": vkitti_val_geometry,
        "vkitti_val": vkitti_val_geometry,
    }
    if kitti_val_dataset is not None:
        config_payload["dataset_geometry"]["kitti_val"] = kitti_val_dataset.describe_geometry()
    config_payload["model_param_counts"] = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_param_count),
        "frozen_params": int(total_params - trainable_param_count),
    }
    config_payload["eval_protocol"] = {
        "vkitti_val": "per_image_affine_disp_depth_anything_v2",
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
        "[MODEL] total_params=%d trainable_params=%d frozen_params=%d residual_feature_source=%s d0_sign=%d",
        total_params,
        trainable_param_count,
        total_params - trainable_param_count,
        args.residual_feature_source,
        args.d0_sign,
    )
    logger.info("[DATASET] train_samples=%d vkitti_val_samples=%d", len(train_dataset), len(val_dataset))
    if kitti_val_dataset is not None:
        logger.info("[DATASET][KITTI] val_samples=%d source_shapes=%s", len(kitti_val_dataset), kitti_val_dataset.source_shape_counts)
    logger.info("[DATASET] geometry=%s", config_payload["dataset_geometry"])
    logger.info("[CONTROL] x3_mean=n/a ffm_mid_mean=n/a")

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
                logger.info(
                    "[BATCH] image=%s depth=%s valid=%s samples=%s",
                    tuple(image.shape),
                    tuple(depth.shape),
                    tuple(valid_mask.shape),
                    batch["sample_name"][: min(2, len(batch["sample_name"]))],
                )

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
                step_per_sec = (step_idx + 1) / max(now - epoch_start, 1e-6)
                eta_seconds = (steps_per_epoch - (step_idx + 1)) / max(step_per_sec, 1e-6)
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d opt_step=%d loss_total=%.5f L_depth=%.5f L_grad=%.5f "
                    "L_keep=%.5f L_res=%.5f L_gate=%.5f L_gate_sup=%.5f mean_gate=%.5f max_gate=%.5f "
                    "mean_abs_delta=%.5f mean_abs_gate_delta=%.5f mean_abs_Dfinal_minus_D0norm=%.5f "
                    "x3_mean=n/a ffm_mid_mean=n/a lr=%.7f max_mem_mb=%.0f used=%d skipped=%d "
                    "step_per_sec=%.2f elapsed=%s eta=%s",
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
                    optimizer.param_groups[0]["lr"],
                    max_mem,
                    int(loss_info["used_samples"]),
                    int(loss_info["skipped_samples"]),
                    step_per_sec,
                    format_seconds(now - epoch_start),
                    format_seconds(eta_seconds),
                )

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
                    raise RuntimeError("KITTI eval requested but KITTI dataloader was not built.")
                kitti_val_summary = evaluate_control_kitti_model(
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
