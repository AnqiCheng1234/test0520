from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pprint
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.util.loss import AlignedInverseSigLoss
from finetune_stf.util.utils import init_log
from foundation.engine.datasets import DEFAULT_TRAIN_LIST, VKITTI2Raw
from foundation.engine.models import build_dav2_raw_naive_depth_model


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 VKITTI2 naive RAW training (online pseudo-RAW).")
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument(
        "--pretrained-from",
        default="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth",
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--vkitti-train-list", default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--input-height", type=int, default=518)
    parser.add_argument("--input-width", type=int, default=966)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--train-backbone", action="store_false", dest="freeze_backbone")
    parser.add_argument("--randomize-unprocessing", action="store_true", default=True)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument(
        "--vkitti-unprocessing-preset",
        default="sensor_linear_dual",
        help=(
            "Pseudo-raw preset for VKITTI2Raw. "
            "Supported presets include: stf_legacy, eth3d_sensor_linear, "
            "robotcar_public_gbrg_generic (public GBRG + generic ranges), "
            "robotcar_subset100_sensor_linear, robotcar_subset100_sensor_linear_fixccm, "
            "robotcar_night_sensor_linear, sensor_linear_dual, "
            "robotcar_day_night_sensor_linear_dual. The robotcar_subset100*, "
            "robotcar_night_sensor_linear, sensor_linear_dual, and "
            "robotcar_day_night_sensor_linear_dual presets use RobotCar statistics "
            "and are not public-only."
        ),
    )
    parser.add_argument(
        "--vkitti-unprocessing-mix-weights",
        default=None,
        help=(
            "Optional mix weights for dual preset. "
            "Examples: '0.3,0.7' or "
            "'eth3d_sensor_linear=0.3,robotcar_subset100_sensor_linear=0.7'."
        ),
    )
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj):
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


def attach_file_logger(logger: logging.Logger, log_path: str) -> None:
    log_path = os.path.abspath(log_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == log_path:
            return

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logger.level)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def save_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def update_optimizer_lrs(optimizer: AdamW, scale: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def save_checkpoint(path: Path, *, model, optimizer, epoch: int, global_step: int, args, epoch_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "epoch_loss": float(epoch_loss),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training entry currently expects CUDA.")

    save_path = Path(args.save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    logger = init_log("phase1_vkitti2_naive", logging.INFO)
    logger.propagate = 0
    attach_file_logger(logger, save_path / "train.log")
    logger.info("%s\n", pprint.pformat({**vars(args), "device": str(device)}))

    save_json(save_path / "config.json", vars(args))

    cudnn.enabled = True
    cudnn.benchmark = True
    set_random_seed(args.seed)

    dataset = VKITTI2Raw(
        filelist_path=args.vkitti_train_list,
        mode="train",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        randomize_unprocessing=args.randomize_unprocessing,
        unprocessing_preset=args.vkitti_unprocessing_preset,
        unprocessing_mix_weights=args.vkitti_unprocessing_mix_weights,
        hflip_prob=args.hflip_prob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    base_model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model = build_dav2_raw_naive_depth_model(
        base_model,
        freeze_backbone=args.freeze_backbone,
    )

    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(strip_module_prefix(resolve_model_state(resume)), strict=True)
        logger.info("[INIT] resumed full model from %s", args.resume_from)
        start_epoch = int(resume.get("epoch", -1)) + 1
        global_step = int(resume.get("global_step", 0))
    else:
        ckpt_obj = torch.load(args.pretrained_from, map_location="cpu")
        state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
        model.load_base_dav2_state_dict(state_dict)
        logger.info("[INIT] loaded base DAv2 weights from %s", args.pretrained_from)
        start_epoch = 0
        global_step = 0

    model = model.to(device)
    criterion = AlignedInverseSigLoss(min_valid_pixels_per_sample=128).to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        [{"params": trainable_params, "lr": args.lr, "initial_lr": args.lr}],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")

    total_params, trainable_param_count = count_parameters(model)
    logger.info(
        "[MODEL] total_params=%d trainable_params=%d frozen_params=%d freeze_backbone=%s",
        total_params,
        trainable_param_count,
        total_params - trainable_param_count,
        args.freeze_backbone,
    )
    logger.info("[DATASET] train_samples=%d batch_size=%d accum_steps=%d", len(dataset), args.bs, args.accum_steps)
    logger.info("[DATASET] vkitti_unprocessing=%s", dataset.describe_unprocessing())

    steps_per_epoch = len(dataloader)
    if args.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_steps)
    optimizer_steps_per_epoch = max(math.ceil(steps_per_epoch / args.accum_steps), 1)
    total_iters = max(args.epochs * optimizer_steps_per_epoch, 1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.dav2.pretrained.eval()
        optimizer.zero_grad(set_to_none=True)

        epoch_start = time.time()
        running_loss = 0.0
        used_steps = 0
        optimizer_steps = 0
        pending_gradients = False
        last_log_time = epoch_start

        for step_idx, batch in enumerate(dataloader):
            if step_idx >= steps_per_epoch:
                break

            raw = batch["raw"].to(device, non_blocking=True).float()
            depth = batch["depth"].to(device, non_blocking=True).float()
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).bool()

            if epoch == start_epoch and step_idx == 0:
                logger.info(
                    "[BATCH] raw_shape=%s depth_shape=%s valid_shape=%s sample_preview=%s",
                    tuple(raw.shape),
                    tuple(depth.shape),
                    tuple(valid_mask.shape),
                    batch["sample_name"][: min(2, len(batch["sample_name"]))],
                )

            is_boundary = ((step_idx + 1) % args.accum_steps == 0) or ((step_idx + 1) >= steps_per_epoch)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                pred_disp = model(raw)
            loss, loss_info = criterion(pred_disp.float(), depth, valid_mask)

            if loss_info["used_samples"] > 0:
                loss_scaled = loss / min(args.accum_steps, steps_per_epoch - (step_idx // args.accum_steps) * args.accum_steps)
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                pending_gradients = True
                running_loss += float(loss.item())
                used_steps += 1

            if is_boundary and pending_gradients:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_gradients = False
                current_iter = epoch * optimizer_steps_per_epoch + optimizer_steps - 1
                scale = (1 - min(current_iter, total_iters - 1) / total_iters) ** 0.9
                update_optimizer_lrs(optimizer, scale)
                global_step += 1
            elif is_boundary:
                optimizer.zero_grad(set_to_none=True)

            if (step_idx + 1) % args.log_interval == 0:
                now = time.time()
                step_per_sec = args.log_interval / max(now - last_log_time, 1e-6)
                eta_seconds = (steps_per_epoch - (step_idx + 1)) / max(step_per_sec, 1e-6)
                logger.info(
                    "[TRAIN] epoch=%d step=%d/%d opt_step=%d/%d lr=%.7f loss=%.4f running_avg=%.4f used=%d skipped=%d "
                    "step_per_sec=%.2f elapsed=%s eta=%s max_mem_mb=%.0f",
                    epoch,
                    step_idx + 1,
                    steps_per_epoch,
                    optimizer_steps,
                    optimizer_steps_per_epoch,
                    optimizer.param_groups[0]["lr"],
                    float(loss.item()),
                    running_loss / max(used_steps, 1),
                    loss_info["used_samples"],
                    loss_info["skipped_samples"],
                    step_per_sec,
                    format_seconds(now - epoch_start),
                    format_seconds(eta_seconds),
                    torch.cuda.max_memory_allocated(device=device) / (1024 ** 2),
                )
                last_log_time = now

        epoch_loss = running_loss / max(used_steps, 1)
        logger.info(
            "[EPOCH] done epoch=%d avg_loss=%.4f used_steps=%d elapsed=%s",
            epoch,
            epoch_loss,
            used_steps,
            format_seconds(time.time() - epoch_start),
        )

        save_checkpoint(
            save_path / "latest.pth",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            epoch_loss=epoch_loss,
        )
        if ((epoch + 1) % args.save_interval) == 0:
            save_checkpoint(
                save_path / f"epoch_{epoch:02d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                epoch_loss=epoch_loss,
            )


if __name__ == "__main__":
    main()
