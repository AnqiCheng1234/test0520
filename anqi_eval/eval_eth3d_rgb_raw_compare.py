#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
from matplotlib import colormaps
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.dataset.eth3d import ETH3DValRGB, ETH3DValRaw
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    RAW_RAM_BRIDGE_INPUT_TYPES,
    RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
    build_raw_ram_bridge_depth_model,
)
from finetune_stf.models.raw_ram import build_raw_ram_depth_model
from finetune_stf.models.raw_feature_adapter import (
    DEFAULT_FEATURE_ADAPTER_KEYS,
    RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    build_raw_ram_feature_adapter_depth_model,
)
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model
from foundation.engine.models import build_dav2_raw_naive_depth_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3")
RAW_PACKED_INPUT_TYPES = ("raw_packed",)
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
DEPTH_CMAP = colormaps["Spectral_r"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ETH3D fast proxy with RGB reference and RAW checkpoint, and save 5-panel comparisons.")
    parser.add_argument("exp_dir", type=Path, help="Experiment directory containing config.json.")
    parser.add_argument("--checkpoint", default="best", help='Checkpoint to use: "best", "last", or a custom .pth path.')
    parser.add_argument("--eth3d-root", default="/mnt/drive/3333_raw/eth3d_raw_depth_640960")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults inside exp_dir.")
    parser.add_argument("--fast-eval-backend", default="proxy", choices=["proxy", "sparse"])
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples. Combined with --scene-diverse (default on) to pick at most one sample per scene up to this cap.")
    parser.add_argument(
        "--scene-diverse",
        action="store_true",
        default=True,
        help="Default: pick the first sample of each scene in manifest order, capped by --max-samples. Use --no-scene-diverse for the old behavior (first N samples regardless of scene).",
    )
    parser.add_argument("--no-scene-diverse", action="store_false", dest="scene_diverse")
    parser.add_argument("--save-panels", action="store_true", default=True)
    parser.add_argument("--no-save-panels", action="store_false", dest="save_panels")
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Save only 5-panel comparisons and metadata; skip individual RGB/RAW image writes.",
    )
    return parser.parse_args()


def resolve_checkpoint(exp_dir: Path, checkpoint_arg: str) -> Path:
    if checkpoint_arg == "best":
        return exp_dir / "best_model.pth"
    if checkpoint_arg == "last":
        return exp_dir / "last_epoch_model.pth"
    return Path(checkpoint_arg)


def unwrap_model_state(state_obj):
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        state_obj = state_obj["model"]
    if isinstance(state_obj, dict) and state_obj and all(str(k).startswith("module.") for k in state_obj):
        state_obj = {k[len("module."):]: v for k, v in state_obj.items()}
    return state_obj


def remap_legacy_ffm_keys(state_dict: dict) -> dict:
    remap = {
        "ram_core.ffm.fuse.0.": "ram_core.ffm.conv1.",
        "ram_core.ffm.fuse.1.": "ram_core.ffm.conv2.",
        "ram_core.ffm.fuse.2.": "ram_core.ffm.conv3.",
        "ram_core.ffm.fuse.3.": "ram_core.ffm.out_conv.",
        "ram_core.ffm.fuse.4.": "ram_core.ffm.out_bn.",
    }
    output = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remap.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        output[new_key] = value
    return output


def build_raw_model(cfg: dict, checkpoint_path: Path):
    input_type = cfg.get("input_type", "rgb")
    sensor_hw = (cfg.get("input_height", 512), cfg.get("input_width", 960))
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[cfg["encoder"]])

    if input_type in (*RAW_RAM_BRIDGE_INPUT_TYPES, *RAW_RAM_RGB_BRIDGE_INPUT_TYPES):
        model = build_raw_ram_bridge_depth_model(
            dav2,
            input_type=input_type,
            bridge_source=cfg.get("bridge_source", "ram_core"),
            bridge_feature_keys=cfg.get("bridge_feature_keys", list(DEFAULT_BRIDGE_FEATURE_KEYS)),
            bridge_layers=cfg.get("bridge_layers"),
            rgb_interface_mode=cfg.get("rgb_interface_mode", "residual_tanh"),
            rgb_residual_scale=cfg.get("rgb_residual_scale", 0.1),
            lora_block_mode=cfg.get("lora_block_mode", DEFAULT_LORA_BLOCK_MODE),
            lora_rank=cfg.get("lora_rank", 8),
            lora_alpha=cfg.get("lora_alpha", 16.0),
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif input_type in RAW_PACKED_INPUT_TYPES:
        model = build_dav2_raw_naive_depth_model(
            dav2,
            freeze_backbone=True,
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif input_type in ("raw_ram", "raw_ram_residual"):
        model = build_raw_ram_depth_model(
            dav2,
            input_type=input_type,
            rgb_interface_mode=cfg.get("rgb_interface_mode", "residual_tanh"),
            rgb_residual_scale=cfg.get("rgb_residual_scale", 0.1),
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
        feat_keys = cfg.get("bridge_feature_keys") or list(DEFAULT_FEATURE_ADAPTER_KEYS)
        model = build_raw_ram_feature_adapter_depth_model(
            dav2,
            input_type=input_type,
            feature_keys=feat_keys,
            bridge_source=cfg.get("bridge_source", "ram_core"),
            bridge_layers=cfg.get("bridge_layers"),
            rgb_interface_mode=cfg.get("rgb_interface_mode", "residual_tanh"),
            rgb_residual_scale=cfg.get("rgb_residual_scale", 0.1),
            lora_block_mode=cfg.get("lora_block_mode", DEFAULT_LORA_BLOCK_MODE),
            lora_rank=cfg.get("lora_rank", 8),
            lora_alpha=cfg.get("lora_alpha", 16.0),
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif input_type in ("rgb", "raw"):
        model = build_dav2_padded_rgb_depth_model(
            dav2,
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    else:
        raise ValueError(f"Unsupported input_type for ETH3D compare script: {input_type}")

    state_dict = unwrap_model_state(torch.load(checkpoint_path, map_location="cpu"))
    if any(k.startswith("ram_core.ffm.fuse.") for k in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    if hasattr(model, "load_compatible_state_dict"):
        model.load_compatible_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()


def build_rgb_reference_model(cfg: dict):
    rgb_ckpt = Path(cfg["pretrained_from"]).expanduser().resolve()
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[cfg["encoder"]])
    model = build_dav2_padded_rgb_depth_model(
        dav2,
        sensor_hw=(640, 960),
        backbone_hw=None,
    )
    state_dict = unwrap_model_state(torch.load(rgb_ckpt, map_location="cpu"))
    model.load_base_dav2_state_dict(state_dict)
    return model.to(DEVICE).eval()


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray, *, vmin: float, vmax: float) -> Image.Image:
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    value = np.clip(depth, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = (value - vmin) / denom
    rgb = (DEPTH_CMAP(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    rgb[~valid_mask] = 0
    return Image.fromarray(rgb)


def build_raw_preview(raw_4ch_path: Path) -> Image.Image:
    with np.load(raw_4ch_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_4ch"], dtype=np.float32)
    preview = np.stack(
        [
            raw[..., 0],
            0.5 * (raw[..., 1] + raw[..., 2]),
            raw[..., 3],
        ],
        axis=-1,
    )
    preview = np.clip(preview, 0.0, 1.0) ** (1.0 / 2.2)
    return Image.fromarray((preview * 255.0).round().astype(np.uint8))


def build_panel(images: list[Image.Image], labels: list[str], footer: str) -> Image.Image:
    panel_w, panel_h = images[0].size
    header_h = 32
    footer_h = 24
    canvas = Image.new("RGB", (panel_w * len(images), panel_h + header_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(zip(images, labels)):
        x0 = idx * panel_w
        canvas.paste(image, (x0, header_h))
        draw.text((x0 + 12, 8), label, fill="black")
    draw.text((12, header_h + panel_h + 4), footer, fill="black")
    return canvas


def infer_batched(model, image_tensor: torch.Tensor, target_hw: tuple[int, int], *, use_amp: bool = True) -> np.ndarray:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    amp_enabled = use_amp and DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(image_tensor).float()
    if tuple(pred.shape[-2:]) != tuple(target_hw):
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0]
    return pred[0].detach().cpu().numpy()


def summarize_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {key: float("nan") for key in METRIC_KEYS}
    return {
        key: float(sum(record[key] for record in records) / len(records))
        for key in METRIC_KEYS
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    config_path = exp_dir / "config.json"
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    input_type = cfg.get("input_type", "rgb")
    if input_type == "rgb":
        raise ValueError("This script expects a RAW-like experiment checkpoint, not an RGB experiment.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (exp_dir / "eth3d_fast_rgb_raw_compare_best")
    )
    rgb_dir = output_dir / "rgb_reference"
    raw_dir = output_dir / "raw_model"
    panel_dir = output_dir / "panels_5up"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.panels_only:
        panel_dir.mkdir(parents=True, exist_ok=True)
    else:
        for path in (rgb_dir, raw_dir, panel_dir):
            path.mkdir(parents=True, exist_ok=True)

    raw_dataset = ETH3DValRaw(
        eth3d_root=args.eth3d_root,
        depth_mode="fast",
        fast_eval_backend=args.fast_eval_backend,
        min_depth=0.1,
        max_depth=80.0,
        norm_mode="sensor_linear",
        channel_mode=cfg.get("channel_mode", "rgb_avg_g"),
        use_imagenet_norm=cfg.get("use_imagenet_norm", True),
        input_mode=(
            "raw_ram"
            if input_type in (
                *RAW_PACKED_INPUT_TYPES,
                "raw_ram",
                "raw_ram_residual",
                *RAW_RAM_BRIDGE_INPUT_TYPES,
                *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
                *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
            )
            else "raw_naive"
        ),
    )
    rgb_dataset = ETH3DValRGB(
        eth3d_root=args.eth3d_root,
        depth_mode="fast",
        fast_eval_backend=args.fast_eval_backend,
        min_depth=0.1,
        max_depth=80.0,
    )
    if len(raw_dataset) != len(rgb_dataset):
        raise RuntimeError(f"ETH3D raw/rgb dataset length mismatch: {len(raw_dataset)} vs {len(rgb_dataset)}")

    dataset_size = len(raw_dataset)
    indices = list(range(dataset_size))
    if args.max_samples is not None:
        if args.scene_diverse:
            scenes_seen = set()
            diverse_indices = []
            for idx in indices:
                scene = raw_dataset.rows[idx].get("scene")
                if scene in scenes_seen:
                    continue
                scenes_seen.add(scene)
                diverse_indices.append(idx)
                if len(diverse_indices) >= args.max_samples:
                    break
            indices = diverse_indices
        else:
            indices = indices[: args.max_samples]

    rgb_model = build_rgb_reference_model(cfg)
    rgb_metrics = []
    panel_manifest = []
    panel_images = {}
    print(f"[ETH3D][RGB] start samples={len(indices)} pretrained_from={cfg['pretrained_from']}", flush=True)
    for count, idx in enumerate(indices, start=1):
        sample = rgb_dataset[idx]
        depth = sample["depth"].numpy()
        valid = sample["valid_mask"].numpy().astype(bool)
        pred_disp = infer_batched(rgb_model, sample["image"], depth.shape[-2:])
        aligned_depth, _ = affine_align_disp(depth, pred_disp, valid)
        metrics = compute_metrics(depth, aligned_depth, valid, min_depth=0.1, max_depth=80.0)
        if metrics is None:
            continue
        rgb_metrics.append({key: float(metrics[key]) for key in METRIC_KEYS})

        row = rgb_dataset.rows[idx]
        sample_name = sample["sample_name"]
        valid_depth = depth[valid]
        vmin = max(0.1, float(valid_depth.min()))
        vmax = min(80.0, float(valid_depth.max()))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = 0.1, 80.0

        rgb_image = Image.open(row["rgb_640_path"]).convert("RGB")
        gt_image = colorize_depth(depth, valid, vmin=vmin, vmax=vmax)
        rgb_pred = colorize_depth(aligned_depth, valid, vmin=vmin, vmax=vmax)

        safe_name = sample_name.replace("/", "__")
        record = {
            "index": idx,
            "sample_name": sample_name,
            "scene": sample["scene"],
            "vmin": vmin,
            "vmax": vmax,
            "rgb_abs_rel": float(metrics["abs_rel"]),
            "rgb_rmse": float(metrics["rmse"]),
        }
        if args.panels_only:
            panel_images[idx] = {
                "rgb": rgb_image.copy(),
                "rgb_pred": rgb_pred.copy(),
                "gt": gt_image.copy(),
            }
            record["rgb_path"] = None
            record["rgb_pred_path"] = None
            record["gt_path"] = None
        else:
            rgb_image_path = rgb_dir / f"{idx:04d}_{safe_name}_rgb.jpg"
            rgb_pred_path = rgb_dir / f"{idx:04d}_{safe_name}_rgb_pred.jpg"
            gt_path = rgb_dir / f"{idx:04d}_{safe_name}_gt.jpg"
            rgb_image.save(rgb_image_path, quality=95)
            rgb_pred.save(rgb_pred_path, quality=95)
            gt_image.save(gt_path, quality=95)
            record["rgb_path"] = str(rgb_image_path)
            record["rgb_pred_path"] = str(rgb_pred_path)
            record["gt_path"] = str(gt_path)
        panel_manifest.append(record)
        if count == 1 or count % 25 == 0 or count == len(indices):
            print(f"[ETH3D][RGB] processed {count}/{len(indices)} {sample_name}", flush=True)

    del rgb_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    raw_model = build_raw_model(cfg, checkpoint_path)
    raw_metrics = []
    manifest_by_index = {record["index"]: record for record in panel_manifest}
    print(f"[ETH3D][RAW] start samples={len(indices)} checkpoint={checkpoint_path.name}", flush=True)
    for count, idx in enumerate(indices, start=1):
        sample = raw_dataset[idx]
        depth = sample["depth"].numpy()
        valid = sample["valid_mask"].numpy().astype(bool)
        image_tensor = sample["raw"] if "raw" in sample else sample["image"]
        pred_disp = infer_batched(raw_model, image_tensor, depth.shape[-2:])
        aligned_depth, _ = affine_align_disp(depth, pred_disp, valid)
        metrics = compute_metrics(depth, aligned_depth, valid, min_depth=0.1, max_depth=80.0)
        if metrics is None or idx not in manifest_by_index:
            continue
        raw_metrics.append({key: float(metrics[key]) for key in METRIC_KEYS})

        record = manifest_by_index[idx]
        row = raw_dataset.rows[idx]
        sample_name = sample["sample_name"]
        raw_image = build_raw_preview(Path(row["raw_640_path"]))
        raw_pred = colorize_depth(aligned_depth, valid, vmin=float(record["vmin"]), vmax=float(record["vmax"]))
        if args.panels_only:
            record["raw_path"] = None
            record["raw_pred_path"] = None
        else:
            raw_image_path = raw_dir / f"{idx:04d}_{sample_name.replace('/', '__')}_raw.jpg"
            raw_pred_path = raw_dir / f"{idx:04d}_{sample_name.replace('/', '__')}_raw_pred.jpg"
            raw_image.save(raw_image_path, quality=95)
            raw_pred.save(raw_pred_path, quality=95)
            record["raw_path"] = str(raw_image_path)
            record["raw_pred_path"] = str(raw_pred_path)
        record["raw_abs_rel"] = float(metrics["abs_rel"])
        record["raw_rmse"] = float(metrics["rmse"])

        if args.save_panels:
            footer = (
                f"{sample_name} | rgb abs_rel={record['rgb_abs_rel']:.4f} rmse={record['rgb_rmse']:.4f} | "
                f"raw abs_rel={record['raw_abs_rel']:.4f} rmse={record['raw_rmse']:.4f}"
            )
            if args.panels_only:
                cached = panel_images[idx]
                rgb_panel = cached["rgb"]
                rgb_pred_panel = cached["rgb_pred"]
                gt_panel = cached["gt"]
            else:
                rgb_panel = Image.open(record["rgb_path"]).convert("RGB")
                rgb_pred_panel = Image.open(record["rgb_pred_path"]).convert("RGB")
                gt_panel = Image.open(record["gt_path"]).convert("RGB")
            panel = build_panel(
                [
                    rgb_panel,
                    rgb_pred_panel,
                    raw_image,
                    raw_pred,
                    gt_panel,
                ],
                ["rgb", "rgb_pred", "raw", "raw_pred", "gt"],
                footer,
            )
            panel.save(panel_dir / f"{idx:04d}_{sample_name.replace('/', '__')}_5up.jpg", quality=95)
        if count == 1 or count % 25 == 0 or count == len(indices):
            print(f"[ETH3D][RAW] processed {count}/{len(indices)} {sample_name}", flush=True)

    summary = {
        "exp_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "pretrained_from": cfg["pretrained_from"],
        "eth3d_root": str(Path(args.eth3d_root).expanduser().resolve()),
        "fast_eval_backend": args.fast_eval_backend,
        "processed_samples": len(raw_metrics),
        "input_type": input_type,
        "rgb_interface_mode": cfg.get("rgb_interface_mode"),
        "rgb_residual_scale": cfg.get("rgb_residual_scale"),
        "rgb_reference_metrics": summarize_metrics(rgb_metrics),
        "raw_checkpoint_metrics": summarize_metrics(raw_metrics),
        "panel_dir": str(panel_dir),
        "rgb_dir": None if args.panels_only else str(rgb_dir),
        "raw_dir": None if args.panels_only else str(raw_dir),
        "panels_only": bool(args.panels_only),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "panel_manifest.json", panel_manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
