#!/usr/bin/env python3
"""
Companion to compare_rgb_raw_qualitative.py:
Re-generate 4-panel RGB-vs-RAW quadtychs but recolorize the raw depth prediction
using its own p2-p98 percentile range instead of the GT full range, to reveal
structure when the aligned pred occupies only a narrow slice of the GT depth span.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "anqi_eval"))

from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.dataset.raw_utils import load_rectified_bayer_npz, normalize_raw_4ch
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    build_raw_ram_bridge_depth_model,
)
from visualize_stf_predictions import (  # type: ignore
    affine_align_disp,
    colorize_depth,
    load_depth_npz,
    load_manifest_rows,
    rgb_tensor_to_pil,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CONFIGS = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir", type=Path)
    p.add_argument("--checkpoint", default="best_model_eth3d.pth")
    p.add_argument("--rgb-dir", type=Path, required=True)
    p.add_argument("--output-subdir", default="qualitative_compare_rgb_zeroshot_raw_best_pred_range")
    p.add_argument("--num-samples", type=int, default=12)
    return p.parse_args()


def load_model(cfg):
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[cfg["encoder"]])
    model = build_raw_ram_bridge_depth_model(
        dav2,
        input_type=cfg["input_type"],
        bridge_source=cfg.get("bridge_source", "ram_core"),
        bridge_feature_keys=cfg.get("bridge_feature_keys", list(DEFAULT_BRIDGE_FEATURE_KEYS)),
        bridge_layers=cfg.get("bridge_layers"),
        lora_block_mode=cfg.get("lora_block_mode", DEFAULT_LORA_BLOCK_MODE),
        lora_rank=cfg.get("lora_rank", 8),
        lora_alpha=cfg.get("lora_alpha", 16.0),
    )
    return model


def build_panel(images, labels):
    pw, ph = images[0].size
    header = 30
    canvas = Image.new("RGB", (pw * len(images), ph + header), "white")
    for i, img in enumerate(images):
        canvas.paste(img, (i * pw, header))
    draw = ImageDraw.Draw(canvas)
    for i, lab in enumerate(labels):
        draw.text((i * pw + 12, 8), lab, fill="black")
    return canvas


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def main():
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    cfg = json.load(open(exp_dir / "config.json"))
    ckpt = exp_dir / args.checkpoint

    model = load_model(cfg)
    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model = model.to(DEVICE).eval()

    rows = load_manifest_rows(
        cfg["stf_root"], "val", input_type=cfg["input_type"], raw_npz_root=cfg["raw_npz_root"]
    )
    indices = np.linspace(0, len(rows) - 1, args.num_samples, dtype=int).tolist()

    rgb_dir = args.rgb_dir.resolve()
    out_dir = exp_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    for idx in indices:
        row = rows[idx]
        gt = load_depth_npz(row["depth_path"])
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt >= cfg.get("min_depth", 1.0)) & (gt <= cfg.get("max_depth", 80.0))
        if not valid.any():
            print(f"skip {idx}: no valid gt"); continue

        b4 = normalize_raw_4ch(load_rectified_bayer_npz(row["image_path"]), norm_mode=cfg.get("norm_mode", "sensor_linear"))
        t = torch.from_numpy(np.ascontiguousarray(b4.transpose(2, 0, 1)).astype(np.float32)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            x4, feat = model.ram_core.forward_with_features(t)
            front_rgb = model.rgb_head(x4)
            x_norm = (front_rgb - model.img_mean) / model.img_std
            x_norm = model.spatial_adapter.pad_rgb(x_norm)
            ph = model.dav2.pretrained.patch_size
            patch_hw = (x_norm.shape[-2] // ph, x_norm.shape[-1] // ph)
            bridge = model.bridge_adapter(feat, patch_hw=patch_hw)
            pred = model.dav2(x_norm, bridge_injections=bridge)
            pred = model.spatial_adapter.crop_depth(pred)
            pred = F.interpolate(pred[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()

        aligned = affine_align_disp(gt, pred, valid)
        fin = np.isfinite(aligned) & (aligned > 0)
        if not fin.any():
            print(f"skip {idx}: no finite aligned"); continue

        # pred-range percentiles (p2-p98) so structure is visible even when pred span << GT span
        finv = aligned[valid & fin]
        p2, p98 = float(np.percentile(finv, 2)), float(np.percentile(finv, 98))
        if p98 <= p2:
            p98 = p2 + 1e-3
        raw_pred_pil = colorize_depth(aligned, p2, p98).resize((w, h))
        raw_front_pil = rgb_tensor_to_pil(front_rgb, target_hw=(h, w))

        rgb_image = open_rgb(rgb_dir / f"image_{idx:04d}.jpg")
        rgb_pred = open_rgb(rgb_dir / f"pred_{idx:04d}.jpg")

        label = f"raw_pred (p2={p2:.1f}m p98={p98:.1f}m)"
        canvas = build_panel(
            [rgb_image, rgb_pred, raw_front_pil, raw_pred_pil],
            ["rgb", "rgb_pred (GT range)", "raw_front_rgb", label],
        )
        canvas.save(out_dir / f"quadtych_{idx:04d}.jpg", quality=95)
        print(f"[{idx}] aligned range [{finv.min():.2f},{finv.max():.2f}]m, vis [{p2:.2f},{p98:.2f}]m")

    print("saved to", out_dir)


if __name__ == "__main__":
    main()
