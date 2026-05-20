#!/usr/bin/env python3
"""
Dump STF RGB triplets for appendix-A diagnosis:

    1. base_rgb    = packed_bayer_to_base_rgb(x_raw)
    2. adapted_rgb = the 3ch image actually sent toward DAv2
    3. real_rgb    = STF manifest lut_preview image

This is a lightweight standalone diagnostic script. It does not modify the
existing qualitative pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.raw_utils import DEFAULT_RAW_NPZ_ROOT, load_rectified_bayer_npz, normalize_raw_4ch
from finetune_stf.models.lora_bridge import RAW_RAM_BRIDGE_INPUT_TYPES
from finetune_stf.models.raw_feature_adapter import RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES
from finetune_stf.models.raw_ram import packed_bayer_to_base_rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_STF_ROOT = "/home/caq/6666_raw/seeingthroughfog"
RAW_PACKED_INPUT_TYPES = ("raw_packed",)
SUPPORTED_INPUT_TYPES = (
    *RAW_PACKED_INPUT_TYPES,
    "raw_ram",
    "raw_ram_residual",
    *RAW_RAM_BRIDGE_INPUT_TYPES,
    *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
)
VALID_SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Dump STF base/adapted/real RGB triplets for raw-model diagnosis.")
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Path to an experiment directory containing config.json.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help='Checkpoint to load: "best", "latest", or a custom .pth path. Default: best',
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=VALID_SPLITS,
        help="Dataset split to visualize. Default: val",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many evenly spaced samples to dump. Default: 10",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit sample indices. Overrides --num-samples.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional custom output subdirectory name inside exp_dir.",
    )
    return parser.parse_args()


def make_indices(dataset_size: int, num_samples: int, explicit_indices: list[int] | None) -> list[int]:
    if explicit_indices:
        indices = [int(idx) for idx in explicit_indices]
    else:
        num_samples = max(1, min(num_samples, dataset_size))
        indices = np.linspace(0, dataset_size - 1, num_samples, dtype=int).tolist()

    deduped = []
    seen = set()
    for idx in indices:
        if idx < 0 or idx >= dataset_size:
            raise IndexError(f"Sample index {idx} is out of range for dataset size {dataset_size}")
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def resolve_checkpoint(exp_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "best":
        return exp_dir / "best_model.pth"
    if checkpoint == "latest":
        return exp_dir / "latest_model.pth"
    return Path(checkpoint)


def resolve_output_dir(exp_dir: Path, split: str, checkpoint: str, output_subdir: str | None) -> Path:
    if output_subdir:
        return exp_dir / output_subdir
    ckpt_name = Path(checkpoint).stem if checkpoint not in {"best", "latest"} else checkpoint
    return exp_dir / f"diagnostic_rgb_triplets_{split}_{ckpt_name}"


def load_manifest_rows(stf_root: str, split: str, raw_npz_root: str):
    manifest_path = Path(stf_root).expanduser().resolve() / "manifests" / f"stf_raw_depth_v1_{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")

    raw_root = Path(raw_npz_root).expanduser().resolve()
    stf_root_path = Path(stf_root).expanduser().resolve()
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sample_name = row["filename_stem"].strip()
            raw_npz_path = (raw_root / f"{sample_name}.npz").resolve()
            rgb_rel = row["lut_preview"].strip()
            rgb_path = Path(rgb_rel)
            if not rgb_path.is_absolute():
                rgb_path = (stf_root_path / rgb_rel).resolve()
            depth_rel = row["lidar_proj_left"].strip()
            depth_path = Path(depth_rel)
            if not depth_path.is_absolute():
                depth_path = (stf_root_path / depth_rel).resolve()
            rows.append(
                {
                    "index": idx,
                    "sample_name": sample_name,
                    "raw_npz_path": raw_npz_path,
                    "rgb_path": rgb_path.resolve(),
                    "depth_path": depth_path.resolve(),
                    "daytime": row.get("daytime", "").strip().lower(),
                }
            )
    if not rows:
        raise ValueError(f"No STF samples found in {manifest_path}")
    return rows


def unwrap_model_state(state_obj):
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        state_obj = state_obj["model"]
    if isinstance(state_obj, dict) and state_obj and all(k.startswith("module.") for k in state_obj):
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
    new_sd = {}
    for key, val in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remap.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        new_sd[new_key] = val
    return new_sd


def load_model(cfg: dict, checkpoint_path: str):
    from depth_anything_v2.dpt import DepthAnythingV2
    from finetune_stf.models.lora_bridge import (
        DEFAULT_BRIDGE_FEATURE_KEYS,
        DEFAULT_LORA_BLOCK_MODE,
        build_raw_ram_bridge_depth_model,
    )
    from finetune_stf.models.raw_feature_adapter import (
        DEFAULT_FEATURE_ADAPTER_KEYS,
        build_raw_ram_feature_adapter_depth_model,
    )
    from finetune_stf.models.raw_ram import build_raw_ram_depth_model
    from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model
    from foundation.engine.models import build_dav2_raw_naive_depth_model

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    encoder = cfg["encoder"]
    input_type = cfg.get("input_type", "rgb")
    dav2 = DepthAnythingV2(**model_configs[encoder])

    if input_type in RAW_RAM_BRIDGE_INPUT_TYPES:
        model = build_raw_ram_bridge_depth_model(
            dav2,
            input_type=input_type,
            bridge_source=cfg.get("bridge_source", "ram_core"),
            bridge_feature_keys=cfg.get("bridge_feature_keys", list(DEFAULT_BRIDGE_FEATURE_KEYS)),
            bridge_layers=cfg.get("bridge_layers"),
            lora_block_mode=cfg.get("lora_block_mode", DEFAULT_LORA_BLOCK_MODE),
            lora_rank=cfg.get("lora_rank", 8),
            lora_alpha=cfg.get("lora_alpha", 16.0),
        )
    elif input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
        model = build_raw_ram_feature_adapter_depth_model(
            dav2,
            input_type=input_type,
            feature_keys=cfg.get("bridge_feature_keys", list(DEFAULT_FEATURE_ADAPTER_KEYS)),
            bridge_source=cfg.get("bridge_source", "ram_core"),
            bridge_layers=cfg.get("bridge_layers"),
            rgb_interface_mode=cfg.get("rgb_interface_mode", "residual_tanh"),
            rgb_residual_scale=cfg.get("rgb_residual_scale", 0.1),
            lora_block_mode=cfg.get("lora_block_mode", DEFAULT_LORA_BLOCK_MODE),
            lora_rank=cfg.get("lora_rank", 8),
            lora_alpha=cfg.get("lora_alpha", 16.0),
        )
    elif input_type in RAW_PACKED_INPUT_TYPES:
        model = build_dav2_raw_naive_depth_model(
            dav2,
            upsample_mode=cfg.get("upsample_mode", "bilinear"),
            clip_rgb=cfg.get("clip_rgb", True),
            freeze_backbone=True,
        )
    elif input_type in ("raw_ram", "raw_ram_residual"):
        model = build_raw_ram_depth_model(dav2, input_type=input_type)
    else:
        model = build_dav2_padded_rgb_depth_model(dav2)

    state_dict = unwrap_model_state(torch.load(checkpoint_path, map_location="cpu"))
    if any(k.startswith("ram_core.ffm.fuse.") for k in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if not hasattr(model, "load_base_dav2_state_dict"):
            raise
        model.load_base_dav2_state_dict(state_dict)
    return model.to(DEVICE).eval()


def rgb_tensor_to_pil(x_rgb: torch.Tensor, *, target_hw: tuple[int, int]) -> Image.Image:
    if x_rgb.ndim == 4:
        x_rgb = x_rgb[0]
    if x_rgb.ndim != 3 or x_rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor with shape (3,H,W) or (1,3,H,W), got {tuple(x_rgb.shape)}")

    target_h, target_w = target_hw
    x_rgb = (
        x_rgb.detach()
        .float()
        .clamp(0.0, 1.0)
        .unsqueeze(0)
    )
    if tuple(x_rgb.shape[-2:]) != (target_h, target_w):
        x_rgb = torch.nn.functional.interpolate(
            x_rgb,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    rgb_u8 = (
        x_rgb[0]
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
    )
    h, w, c = rgb_u8.shape
    if c != 3:
        raise ValueError(f"Expected RGB uint8 tensor with 3 channels, got {tuple(rgb_u8.shape)}")
    data = bytes(rgb_u8.view(-1).tolist())
    return Image.frombytes("RGB", (w, h), data)


def build_panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    panel_w, panel_h = images[0].size
    header_h = 30
    canvas = Image.new("RGB", (panel_w * len(images), panel_h + header_h), "white")
    for idx, image in enumerate(images):
        canvas.paste(image, (idx * panel_w, header_h))
    draw = ImageDraw.Draw(canvas)
    for idx, label in enumerate(labels):
        draw.text((idx * panel_w + 12, 8), label, fill="black")
    return canvas


def extract_base_and_adapted_rgb(model, input_type: str, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base_rgb = packed_bayer_to_base_rgb(x_raw)

    with torch.no_grad():
        if input_type in RAW_PACKED_INPUT_TYPES:
            features = model.forward_features(x_raw)
            adapted_rgb = features["rgb"]
        elif input_type == "raw_ram":
            x4 = model.ram_core(x_raw)
            adapted_rgb = model.rgb_head(x4)
        elif input_type == "raw_ram_residual":
            x4 = model.ram_core(x_raw)
            delta_rgb = model.residual_head(x4)
            adapted_rgb = torch.clamp(
                base_rgb + model.residual_scale * torch.tanh(delta_rgb),
                min=0.0,
                max=1.0,
            )
        elif input_type in RAW_RAM_BRIDGE_INPUT_TYPES:
            x4, _ = model.ram_core.forward_with_features(x_raw)
            adapted_rgb = model.rgb_head(x4)
        elif input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
            features = model.forward_features(x_raw)
            adapted_rgb = features["rgb"]
        else:
            raise ValueError(f"Unsupported input_type for RGB triplet dump: {input_type}")

    return base_rgb, adapted_rgb


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    config_path = exp_dir / "config.json"
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint).resolve()
    output_dir = resolve_output_dir(exp_dir, args.split, args.checkpoint, args.output_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    input_type = cfg.get("input_type", "rgb")
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(f"This diagnostic only supports raw-like packed inputs, got input_type={input_type}")

    stf_root = cfg.get("stf_root", DEFAULT_STF_ROOT)
    raw_npz_root = cfg.get("raw_npz_root", DEFAULT_RAW_NPZ_ROOT)
    norm_mode = cfg.get("norm_mode", "companded")

    print(f"Loading model from {checkpoint_path} on {DEVICE} ...", flush=True)
    model = load_model(cfg, str(checkpoint_path))

    rows = load_manifest_rows(stf_root, args.split, raw_npz_root)
    indices = make_indices(len(rows), args.num_samples, args.indices)

    sample_lines = ["index\tsample_name\traw_npz_path\trgb_path"]
    for idx in indices:
        row = rows[idx]
        raw_npz_path = row["raw_npz_path"]
        rgb_path = row["rgb_path"]

        if not raw_npz_path.is_file():
            raise FileNotFoundError(f"Missing raw npz: {raw_npz_path}")
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")

        bayer_rect = load_rectified_bayer_npz(raw_npz_path)
        bayer_4ch_norm = normalize_raw_4ch(bayer_rect, norm_mode=norm_mode)
        chw = np.ascontiguousarray(bayer_4ch_norm.transpose(2, 0, 1)).astype(np.float32, copy=False)
        x_raw = torch.frombuffer(chw.tobytes(), dtype=torch.float32).view(chw.shape).unsqueeze(0).to(DEVICE)

        real_rgb = Image.open(rgb_path).convert("RGB")
        target_hw = (int(bayer_4ch_norm.shape[0]), int(bayer_4ch_norm.shape[1]))
        base_rgb, adapted_rgb = extract_base_and_adapted_rgb(model, input_type, x_raw)

        base_rgb_pil = rgb_tensor_to_pil(base_rgb, target_hw=target_hw)
        adapted_rgb_pil = rgb_tensor_to_pil(adapted_rgb, target_hw=target_hw)
        real_rgb = real_rgb.resize((target_hw[1], target_hw[0]), resample=Image.Resampling.BILINEAR)

        base_rgb_pil.save(output_dir / f"base_rgb_{idx:04d}.jpg", quality=95)
        adapted_rgb_pil.save(output_dir / f"adapted_rgb_{idx:04d}.jpg", quality=95)
        real_rgb.save(output_dir / f"real_rgb_{idx:04d}.jpg", quality=95)
        build_panel(
            [base_rgb_pil, adapted_rgb_pil, real_rgb],
            ["base_rgb", "adapted_rgb", "real_rgb"],
        ).save(output_dir / f"triptych_{idx:04d}.jpg", quality=95)

        sample_lines.append(f"{idx}\t{row['sample_name']}\t{raw_npz_path}\t{rgb_path}")
        print(f"  [{idx}] {row['sample_name']}", flush=True)

    (output_dir / "samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    print(f"\nSaved {len(indices)} RGB triplets to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
