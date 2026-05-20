#!/usr/bin/env python3
"""
Save STF qualitative assets for a trained DAv2 experiment.

Reads config.json from the experiment directory to determine model architecture,
loads the requested checkpoint, runs inference on the specified split, performs
affine alignment (disparity -> metric depth), and saves:
    image_XXXX.jpg
    pred_XXXX.jpg
    front_rgb_XXXX.jpg   (raw-like models only)

If --compare-rgb-dir is provided for a raw-like experiment, the script skips
the intermediate raw qualitative directory and directly saves 4-panel
comparisons:
    rgb | rgb_pred | raw_front_rgb | raw_pred

Usage:
    python anqi_eval/visualize_stf_predictions.py \
        finetune_stf/exp/e3_raw_ram_bridge_full_dav3_bs4_acc4_20260413_233926

    python anqi_eval/visualize_stf_predictions.py \
        finetune_stf/exp/e3_raw_ram_bridge_full_dav3_bs4_acc4_20260413_233926 \
        --split val --checkpoint best --num-samples 12
"""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.raw_utils import (
    DEFAULT_RAW_NPZ_ROOT,
    bayer_to_3ch,
    load_rectified_bayer_npz,
    normalize_raw,
    normalize_raw_4ch,
    pseudo_rgb_to_bgr,
)
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    RAW_RAM_BRIDGE_INPUT_TYPES,
)
from finetune_stf.models.raw_feature_adapter import (
    DEFAULT_FEATURE_ADAPTER_KEYS,
    RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    build_raw_ram_feature_adapter_depth_model,
)
from finetune_stf.models.raw_ram import packed_bayer_to_base_rgb
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model
from foundation.engine.models import build_dav2_raw_naive_depth_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_STF_ROOT = "/home/caq/6666_raw/seeingthroughfog"
DEFAULT_MIN_DEPTH = 1.0
DEFAULT_MAX_DEPTH = 80.0
VALID_SPLITS = ("train", "val", "test")
RAW_PACKED_INPUT_TYPES = ("raw_packed",)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Save STF qualitative depth visualizations for a trained DAv2 experiment.",
    )
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Path to the experiment output directory (must contain config.json).",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help='Which checkpoint to load: "best", "latest", or a custom .pth path. Default: best',
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
        default=12,
        help="How many evenly spaced samples to save. Default: 12",
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
    parser.add_argument(
        "--compare-rgb-dir",
        type=Path,
        default=None,
        help="Optional RGB zero-shot qualitative directory. If set, save only 4-panel comparisons.",
    )
    parser.add_argument(
        "--gt-vis-dilate-kernel",
        type=int,
        default=5,
        help="Odd kernel size to dilate sparse GT points for visualization. Default: 5",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint(exp_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "best":
        return exp_dir / "best_model.pth"
    if checkpoint == "latest":
        return exp_dir / "latest_model.pth"
    return Path(checkpoint)


def resolve_output_dir(
    exp_dir: Path,
    split: str,
    checkpoint: str,
    output_subdir: str | None,
    *,
    compare_mode: bool = False,
) -> Path:
    if output_subdir:
        return exp_dir / output_subdir
    if compare_mode:
        return exp_dir / "qualitative_compare_rgb_zeroshot_raw_best"
    ckpt_name = Path(checkpoint).stem if checkpoint not in {"best", "latest"} else checkpoint
    return exp_dir / f"qualitative_{split}_{ckpt_name}"


def make_indices(dataset_size: int, num_samples: int, explicit_indices: list[int] | None) -> list[int]:
    if explicit_indices:
        indices = [int(idx) for idx in explicit_indices]
    else:
        num_samples = max(1, min(num_samples, dataset_size))
        indices = np.linspace(0, dataset_size - 1, num_samples, dtype=int).tolist()

    deduped: list[int] = []
    seen = set()
    for idx in indices:
        if idx < 0 or idx >= dataset_size:
            raise IndexError(f"Sample index {idx} is out of range for dataset size {dataset_size}")
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    """Map metric depth to plasma colormap, returning a PIL RGB image."""
    value = np.asarray(depth, dtype=np.float32)
    value = np.clip(value, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = (value - vmin) / denom
    rgba = colormaps["plasma"](norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


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


def rgb_tensor_to_pil(x_rgb: torch.Tensor, *, target_hw: tuple[int, int]) -> Image.Image:
    """Convert a BCHW/CHW float tensor in [0,1] to a resized PIL RGB image."""
    if x_rgb.ndim == 4:
        x_rgb = x_rgb[0]
    if x_rgb.ndim != 3 or x_rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor with shape (3,H,W) or (1,3,H,W), got {tuple(x_rgb.shape)}")

    rgb = (
        x_rgb.detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb_u8)


def dilate_mask(vis_arr: np.ndarray, valid_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate sparse GT visualization points so they are more visible."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or not valid_mask.any():
        return vis_arr
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Convert PIL-style RGB uint8 to BGR for cv2 dilate, then back
    bgr = vis_arr[..., ::-1].copy()
    dilated = cv2.dilate(bgr, kernel, iterations=1)
    dilated_valid = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=1) > 0

    output = np.zeros_like(vis_arr)
    output[dilated_valid] = dilated[dilated_valid][..., ::-1]
    return output


# ---------------------------------------------------------------------------
# Manifest loading (same logic as eval_stf_rel_depth.py)
# ---------------------------------------------------------------------------

def load_manifest_rows(stf_root: str, split: str, *, input_type: str, raw_npz_root: str):
    import csv

    manifest_path = Path(stf_root) / "manifests" / f"stf_raw_depth_v1_{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")

    raw_npz_root = Path(raw_npz_root).expanduser().resolve()
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sample_name = row["filename_stem"]
            if input_type == "rgb":
                rel = row["lut_preview"].strip()
                image_path = (Path(stf_root) / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            else:
                image_path = (raw_npz_root / f"{sample_name}.npz").resolve()

            depth_rel = row["lidar_proj_left"].strip()
            depth_path = (Path(stf_root) / depth_rel).resolve() if not Path(depth_rel).is_absolute() else Path(depth_rel)

            rows.append({
                "index": idx,
                "sample_name": sample_name,
                "image_path": image_path,
                "depth_path": depth_path,
            })
    if not rows:
        raise ValueError(f"No STF samples found in {manifest_path}")
    return rows


def load_depth_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return np.array(data["arr_0"], dtype=np.float32, copy=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def unwrap_model_state(state_obj):
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        state_obj = state_obj["model"]
    if isinstance(state_obj, dict) and state_obj and all(k.startswith("module.") for k in state_obj):
        state_obj = {k[len("module."):]: v for k, v in state_obj.items()}
    return state_obj


def remap_legacy_ffm_keys(state_dict: dict) -> dict:
    """Remap old FFM key format (ffm.fuse.N) to current (ffm.conv1/conv2/conv3/out_conv/out_bn)."""
    REMAP = {
        "ram_core.ffm.fuse.0.": "ram_core.ffm.conv1.",
        "ram_core.ffm.fuse.1.": "ram_core.ffm.conv2.",
        "ram_core.ffm.fuse.2.": "ram_core.ffm.conv3.",
        "ram_core.ffm.fuse.3.": "ram_core.ffm.out_conv.",
        "ram_core.ffm.fuse.4.": "ram_core.ffm.out_bn.",
    }
    new_sd = {}
    for key, val in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in REMAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        new_sd[new_key] = val
    return new_sd


def load_model(cfg: dict, checkpoint_path: str):
    from depth_anything_v2.dpt import DepthAnythingV2
    from finetune_stf.models.lora_bridge import build_raw_ram_bridge_depth_model
    from finetune_stf.models.raw_feature_adapter import build_raw_ram_feature_adapter_depth_model
    from finetune_stf.models.raw_ram import build_raw_ram_depth_model

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    encoder = cfg["encoder"]
    input_type = cfg.get("input_type", "rgb")
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[encoder])

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
    # Handle legacy FFM key format from older checkpoints
    if any(k.startswith("ram_core.ffm.fuse.") for k in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if not hasattr(model, "load_base_dav2_state_dict"):
            raise
        model.load_base_dav2_state_dict(state_dict)
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_preprocessed_rgb(model, image_rgb, input_height, input_width, target_height, target_width, *, use_imagenet_norm=True):
    """Run inference on a 3-channel float32 RGB image in [0,1]. Returns disparity (H, W) numpy."""
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet

    image_resized = cv2.resize(image_rgb, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    sample = {"image": image_resized.astype(np.float32)}
    if use_imagenet_norm:
        sample = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(sample)
    image = PrepareForNet()(sample)["image"]
    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_disp = model.forward(tensor)
        pred_disp = F.interpolate(
            pred_disp[:, None],
            (target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    return pred_disp.cpu().numpy()


def infer_raw_with_front_rgb(model, input_type, bayer_4ch_norm, input_height, input_width, target_height, target_width):
    """Run raw-like inference and also return the 3-channel front-end RGB sent toward the backbone."""
    # Raw checkpoints can be evaluated at runtime sizes different from the
    # training crop (e.g. STF 512x960 vs LOD 644x1008), so only enforce the
    # packed-Bayer layout here.
    if bayer_4ch_norm.ndim != 3 or bayer_4ch_norm.shape[2] != 4:
        raise ValueError(
            f"Expected packed Bayer with shape (H, W, 4), got {tuple(bayer_4ch_norm.shape)}"
        )
    tensor = torch.from_numpy(
        np.ascontiguousarray(bayer_4ch_norm.transpose(2, 0, 1)).astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if input_type in RAW_PACKED_INPUT_TYPES:
            features = model.forward_features(tensor)
            front_rgb = features["rgb"]
            pred_disp = features["depth"]
        elif input_type == "raw_ram":
            x4 = model.ram_core(tensor)
            front_rgb = model.rgb_head(x4)
            x_norm = (front_rgb - model.img_mean) / model.img_std
            x_norm = model.spatial_adapter.pad_rgb(x_norm)
            pred_disp = model.dav2(x_norm)
            pred_disp = model.spatial_adapter.crop_depth(pred_disp)
        elif input_type == "raw_ram_residual":
            x4 = model.ram_core(tensor)
            delta_rgb = model.residual_head(x4)
            base_rgb = packed_bayer_to_base_rgb(tensor)
            front_rgb = torch.clamp(
                base_rgb + model.residual_scale * torch.tanh(delta_rgb),
                min=0.0,
                max=1.0,
            )
            x_norm = (front_rgb - model.img_mean) / model.img_std
            x_norm = model.spatial_adapter.pad_rgb(x_norm)
            pred_disp = model.dav2(x_norm)
            pred_disp = model.spatial_adapter.crop_depth(pred_disp)
        elif input_type in RAW_RAM_BRIDGE_INPUT_TYPES:
            x4, feature_dict = model.ram_core.forward_with_features(tensor)
            front_rgb = model.rgb_head(x4)
            x_norm = (front_rgb - model.img_mean) / model.img_std
            x_norm = model.spatial_adapter.pad_rgb(x_norm)
            patch_hw = (
                x_norm.shape[-2] // model.dav2.pretrained.patch_size,
                x_norm.shape[-1] // model.dav2.pretrained.patch_size,
            )
            bridge_injections = model.bridge_adapter(feature_dict, patch_hw=patch_hw)
            pred_disp = model.dav2(x_norm, bridge_injections=bridge_injections)
            pred_disp = model.spatial_adapter.crop_depth(pred_disp)
        elif input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
            features = model.forward_features(tensor)
            front_rgb = features["rgb"]
            pred_disp = features["depth"]
        else:
            raise ValueError(f"Unsupported raw-like input_type for front RGB extraction: {input_type}")

        pred_disp = F.interpolate(
            pred_disp[:, None],
            (target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    return pred_disp.cpu().numpy(), rgb_tensor_to_pil(front_rgb, target_hw=(target_height, target_width))


def affine_align_disp(gt_depth, pred_disp, valid_mask):
    """Fit s * pred_disp + t ~= 1/gt_depth on valid pixels, return aligned metric depth."""
    gt_disp = np.zeros_like(gt_depth, dtype=np.float64)
    gt_disp[valid_mask] = 1.0 / np.clip(gt_depth[valid_mask], a_min=1e-9, a_max=None)

    y = gt_disp[valid_mask].reshape(-1, 1).astype(np.float64)
    x = pred_disp[valid_mask].reshape(-1, 1).astype(np.float64)
    A = np.concatenate([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale, shift = float(coef[0].item()), float(coef[1].item())

    aligned_disp = pred_disp.astype(np.float64) * scale + shift
    aligned_depth = np.full(pred_disp.shape, np.nan, dtype=np.float64)
    pos = aligned_disp > 0
    aligned_depth[pos] = 1.0 / aligned_disp[pos]
    return aligned_depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    config_path = exp_dir / "config.json"
    checkpoint_path = resolve_checkpoint(exp_dir, args.checkpoint).resolve()
    compare_rgb_dir = args.compare_rgb_dir.resolve() if args.compare_rgb_dir else None
    output_dir = resolve_output_dir(
        exp_dir,
        args.split,
        args.checkpoint,
        args.output_subdir,
        compare_mode=compare_rgb_dir is not None,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    input_type = cfg.get("input_type", "rgb")
    raw_like_compare = compare_rgb_dir is not None
    if raw_like_compare and input_type not in (
        *RAW_PACKED_INPUT_TYPES,
        "raw_ram",
        "raw_ram_residual",
        *RAW_RAM_BRIDGE_INPUT_TYPES,
        *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    ):
        raise ValueError("--compare-rgb-dir is only supported for raw-like experiments.")
    if compare_rgb_dir is not None and not compare_rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB qualitative directory: {compare_rgb_dir}")

    input_height = cfg.get("input_height", 512)
    input_width = cfg.get("input_width", 960)
    min_depth = cfg.get("min_depth", DEFAULT_MIN_DEPTH)
    max_depth = cfg.get("max_depth", DEFAULT_MAX_DEPTH)
    norm_mode = cfg.get("norm_mode", "companded")
    stf_root = cfg.get("stf_root", DEFAULT_STF_ROOT)
    raw_npz_root = cfg.get("raw_npz_root", DEFAULT_RAW_NPZ_ROOT)

    # ---- load model ----
    print(f"Loading model from {checkpoint_path} ...", flush=True)
    model = load_model(cfg, str(checkpoint_path))

    # ---- load dataset manifest ----
    rows = load_manifest_rows(stf_root, args.split, input_type=input_type, raw_npz_root=raw_npz_root)
    indices = make_indices(len(rows), args.num_samples, args.indices)

    # ---- clean old outputs ----
    for old_file in output_dir.glob("*.jpg"):
        old_file.unlink()

    if raw_like_compare:
        sample_lines = ["index\trgb_image\trgb_pred\traw_front_rgb\traw_pred"]
    else:
        sample_lines = ["index\tsample_name\timage_path\tdepth_path"]

    for idx in indices:
        sample = rows[idx]
        image_path = sample["image_path"]
        depth_path = sample["depth_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing depth: {depth_path}")

        gt_depth = load_depth_npz(depth_path)
        valid_mask = np.isfinite(gt_depth) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
        h, w = gt_depth.shape

        # ---- inference ----
        front_rgb_pil = None
        if input_type in (
            *RAW_PACKED_INPUT_TYPES,
            "raw_ram",
            "raw_ram_residual",
            *RAW_RAM_BRIDGE_INPUT_TYPES,
            *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
        ):
            bayer_rect = load_rectified_bayer_npz(image_path)
            bayer_4ch_norm = normalize_raw_4ch(bayer_rect, norm_mode=norm_mode)

            pred_disp, front_rgb_pil = infer_raw_with_front_rgb(
                model,
                input_type,
                bayer_4ch_norm,
                input_height,
                input_width,
                h,
                w,
            )

            # Pseudo-RGB for visualization
            image_rgb_vis = bayer_to_3ch(bayer_rect, channel_mode="rgb_avg_g")
            image_rgb_vis = normalize_raw(image_rgb_vis, norm_mode=norm_mode)
            vis_bgr = pseudo_rgb_to_bgr(image_rgb_vis)
            vis_bgr = cv2.resize(vis_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        elif input_type == "raw":
            # Naive raw: load Bayer NPZ -> 3ch pseudo-RGB -> standard DAv2 inference
            bayer_rect = load_rectified_bayer_npz(image_path)
            channel_mode = cfg.get("channel_mode", "rgb_avg_g")
            image_rgb = bayer_to_3ch(bayer_rect, channel_mode=channel_mode)
            image_rgb = normalize_raw(image_rgb, norm_mode=norm_mode)
            vis_bgr = pseudo_rgb_to_bgr(image_rgb)
            vis_bgr = cv2.resize(vis_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

            pred_disp = infer_preprocessed_rgb(
                model, image_rgb, input_height, input_width, h, w,
                use_imagenet_norm=cfg.get("use_imagenet_norm", True),
            )
        else:
            # Standard RGB input
            raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                raise ValueError(f"Failed to read image: {image_path}")
            vis_bgr = cv2.resize(raw_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            image_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0

            pred_disp = infer_preprocessed_rgb(
                model, image_rgb, input_height, input_width, h, w,
                use_imagenet_norm=cfg.get("use_imagenet_norm", True),
            )

        # ---- affine alignment: disparity -> metric depth ----
        if valid_mask.sum() < 10:
            print(f"  [warn] idx={idx} {sample['sample_name']}: too few valid GT pixels, skipping")
            continue

        aligned_depth = affine_align_disp(gt_depth, pred_disp, valid_mask)
        aligned_depth = np.clip(aligned_depth, min_depth, max_depth)

        # ---- visualization range from valid GT ----
        if valid_mask.any():
            valid_gt = gt_depth[valid_mask]
            vmin = max(min_depth, float(valid_gt.min()))
            vmax = min(max_depth, float(valid_gt.max()))
            if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = min_depth, max_depth
        else:
            vmin, vmax = min_depth, max_depth

        # ---- colorize ----
        pred_pil = colorize_depth(aligned_depth, vmin=vmin, vmax=vmax)

        # Input image: convert BGR -> RGB for PIL
        image_pil = Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
        panel_size = image_pil.size  # (W, H)
        pred_pil = pred_pil.resize(panel_size, resample=Image.Resampling.BILINEAR)

        if raw_like_compare:
            rgb_image_path = compare_rgb_dir / f"image_{idx:04d}.jpg"
            rgb_pred_path = compare_rgb_dir / f"pred_{idx:04d}.jpg"
            if not rgb_image_path.is_file() or not rgb_pred_path.is_file():
                raise FileNotFoundError(
                    f"Missing RGB qualitative assets for idx={idx}: {rgb_image_path}, {rgb_pred_path}"
                )
            if front_rgb_pil is None:
                raise RuntimeError("Expected front_rgb_pil for raw-like comparison mode.")

            raw_front_rgb_pil = front_rgb_pil.resize(panel_size, resample=Image.Resampling.BILINEAR)
            panel = build_panel(
                [
                    open_rgb(rgb_image_path),
                    open_rgb(rgb_pred_path),
                    raw_front_rgb_pil,
                    pred_pil,
                ],
                ["rgb", "rgb_pred", "raw_front_rgb", "raw_pred"],
            )
            panel.save(output_dir / f"quadtych_{idx:04d}.jpg", quality=95)
            sample_lines.append(
                f"{idx}\t{rgb_image_path}\t{rgb_pred_path}\tfront_rgb_{idx:04d}.jpg\tpred_{idx:04d}.jpg"
            )
        else:
            sample_lines.append(f"{idx}\t{sample['sample_name']}\t{image_path}\t{depth_path}")
            image_pil.save(output_dir / f"image_{idx:04d}.jpg", quality=95)
            pred_pil.save(output_dir / f"pred_{idx:04d}.jpg", quality=95)
            if front_rgb_pil is not None:
                front_rgb_pil.resize(panel_size, resample=Image.Resampling.BILINEAR).save(
                    output_dir / f"front_rgb_{idx:04d}.jpg",
                    quality=95,
                )

        print(f"  [{idx}] {sample['sample_name']}  vrange=({vmin:.1f}, {vmax:.1f})m", flush=True)

    (output_dir / "samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    print(f"\n{output_dir}")
    if raw_like_compare:
        print(f"Saved {len(indices)} 4-panel comparisons from {args.split} using {checkpoint_path.name}")
    else:
        print(f"Saved {len(indices)} samples from {args.split} using {checkpoint_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
