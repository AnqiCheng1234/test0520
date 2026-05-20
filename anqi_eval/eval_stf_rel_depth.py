#!/usr/bin/env python3
"""
Evaluate DAv2 on Seeing Through Fog (STF) raw-depth splits.

The model prediction is treated as inverse relative depth / disparity-like
output, so we keep the same benchmark-style per-image affine alignment in
disparity space that is already used for the other DAv2 relative-depth evals.
"""

import argparse
import csv
from datetime import date
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_rel_depth_strict import (
    FILE_DIR,
    affine_align_disp,
    check_sample_shapes,
    compute_metrics,
    load_model,
    resolve_checkpoint_path,
)
from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
from finetune_stf.dataset.raw_utils import (
    DEFAULT_RAW_NPZ_ROOT,
    STF_RAW_DECODE_MODES,
    bayer_to_3ch,
    decode_stf_raw_4ch,
    load_rectified_bayer_npz,
    normalize_raw,
    normalize_raw_4ch,
    pseudo_rgb_to_bgr,
)
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    LORA_BLOCK_MODE_CHOICES,
    RAW_RAM_BRIDGE_INPUT_TYPES,
)


DEFAULT_STF_ROOT = "/home/caq/6666_raw/seeingthroughfog"
DEFAULT_MIN_DEPTH = 1.0
DEFAULT_MAX_DEPTH = 80.0
BASE_REQUIRED_COLUMNS = ("filename_stem", "lidar_proj_left")
RGB_REQUIRED_COLUMNS = BASE_REQUIRED_COLUMNS + ("lut_preview",)
VALID_SPLITS = ("train", "val", "test", "adverse_only")
DEFAULT_BRIDGE_LAYERS_BY_ENCODER = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
    "vitg": [9, 19, 29, 39],
}
SPECTRAL_R_RGB = np.array(
    [
        [94, 79, 162],
        [50, 136, 189],
        [102, 194, 165],
        [171, 221, 164],
        [230, 245, 152],
        [255, 255, 191],
        [254, 224, 139],
        [253, 174, 97],
        [244, 109, 67],
        [213, 62, 79],
        [158, 1, 66],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "DAv2 STF Relative Depth Evaluation (strict disparity-space affine)"
    )
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--input-type",
        default="rgb",
        choices=["rgb", "raw", "raw_ram", "raw_ram_residual", *RAW_RAM_BRIDGE_INPUT_TYPES],
    )
    parser.add_argument("--stf-root", default=DEFAULT_STF_ROOT)
    parser.add_argument("--raw-npz-root", default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--split", default="test", choices=VALID_SPLITS)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--norm-mode", default="companded")
    parser.add_argument("--stf-raw-decode-mode", default="legacy_companded", choices=STF_RAW_DECODE_MODES)
    parser.add_argument("--channel-mode", default="rgb_avg_g")
    parser.add_argument("--bridge-source", default="ram_core", choices=["ram_core"])
    parser.add_argument(
        "--bridge-feature-keys",
        nargs="+",
        default=list(DEFAULT_BRIDGE_FEATURE_KEYS),
        choices=list(DEFAULT_BRIDGE_FEATURE_KEYS),
    )
    parser.add_argument("--bridge-layers", nargs="+", type=int, default=None)
    parser.add_argument("--lora-block-mode", default=DEFAULT_LORA_BLOCK_MODE, choices=LORA_BLOCK_MODE_CHOICES)
    parser.add_argument("--lora-rank", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16.0, type=float)
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", default=FILE_DIR)
    parser.add_argument("--no-save-results", action="store_true")
    parser.add_argument("--no-save-vis", action="store_true")
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--vis-num-samples", type=int, default=10)
    parser.add_argument("--vis-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--gt-vis-dilate-kernel",
        type=int,
        default=5,
        help="Odd kernel size used to slightly enlarge sparse STF GT points in visualization only.",
    )
    parser.add_argument(
        "--error-vis-max",
        type=float,
        default=0.5,
        help="Maximum relative error shown in the error map; larger values are clipped.",
    )
    parser.set_defaults(use_imagenet_norm=True)
    args = parser.parse_args()
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be > 0")
    args.bridge_feature_keys = list(dict.fromkeys(args.bridge_feature_keys))
    if args.bridge_layers is not None:
        args.bridge_layers = list(dict.fromkeys(args.bridge_layers))
    elif args.input_type in RAW_RAM_BRIDGE_INPUT_TYPES:
        args.bridge_layers = list(DEFAULT_BRIDGE_LAYERS_BY_ENCODER[args.encoder])
    if args.input_type != "rgb" and args.stf_raw_decode_mode != "legacy_companded" and args.norm_mode != "passthrough":
        parser.error(
            f"{args.stf_raw_decode_mode} already returns [0,1] decompanded RAW; "
            "use --norm-mode passthrough to avoid a second normalization."
        )
    return args


def resolve_manifest_path(stf_root, split, manifest_path_arg):
    if manifest_path_arg:
        manifest_path = Path(manifest_path_arg).expanduser().resolve()
    else:
        manifest_path = Path(stf_root).expanduser().resolve() / "manifests" / f"stf_raw_depth_v1_{split}.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")
    return manifest_path


def resolve_data_path(stf_root, path_str):
    path = Path(path_str.strip()).expanduser()
    if path.is_absolute():
        return path
    return (Path(stf_root).expanduser().resolve() / path).resolve()


def load_manifest_rows(manifest_path, stf_root, max_samples=None, *, input_type="rgb", raw_npz_root=DEFAULT_RAW_NPZ_ROOT):
    rows = []
    raw_npz_root = Path(raw_npz_root).expanduser().resolve()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required_columns = RGB_REQUIRED_COLUMNS if input_type == "rgb" else BASE_REQUIRED_COLUMNS
        missing = [name for name in required_columns if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required STF columns: {', '.join(missing)}"
            )

        for idx, row in enumerate(reader):
            sample_name = row["filename_stem"]
            if input_type == "rgb":
                image_path = resolve_data_path(stf_root, row["lut_preview"])
            else:
                image_path = (raw_npz_root / f"{sample_name}.npz").resolve()
            rows.append(
                {
                    "index": idx,
                    "sample_name": sample_name,
                    "image_path": image_path,
                    "depth_path": resolve_data_path(stf_root, row["lidar_proj_left"]),
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No STF samples found in {manifest_path}")
    return rows


def make_indices(dataset_size, num_samples, explicit_indices):
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


def load_depth_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" not in data.files:
            raise KeyError(f"{path} does not contain arr_0")
        depth = np.array(data["arr_0"], dtype=np.float32, copy=True)
    return depth


def resolve_input_shape(args):
    if (args.input_height is None) ^ (args.input_width is None):
        raise ValueError("Pass both --input-height and --input-width together.")

    if args.input_height is not None and args.input_width is not None:
        return int(args.input_height), int(args.input_width)

    return int(args.input_size), int(args.input_size)


def build_infer_transform(input_height, input_width, *, keep_aspect_ratio, use_imagenet_norm):
    transforms = [
        Resize(
            width=input_width,
            height=input_height,
            resize_target=False,
            keep_aspect_ratio=keep_aspect_ratio,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
    ]
    if use_imagenet_norm:
        transforms.append(
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    transforms.append(PrepareForNet())
    return Compose(transforms)


def infer_preprocessed_rgb(
    model,
    image_rgb,
    input_height,
    input_width,
    target_height,
    target_width,
    *,
    keep_aspect_ratio,
    use_imagenet_norm,
):
    transform = build_infer_transform(
        input_height,
        input_width,
        keep_aspect_ratio=keep_aspect_ratio,
        use_imagenet_norm=use_imagenet_norm,
    )
    image = transform({"image": image_rgb})["image"]
    device = next(model.parameters()).device
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_disp = model.forward(image)
        pred_disp = F.interpolate(
            pred_disp[:, None],
            (target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return pred_disp.cpu().numpy()


def infer_rectangular(model, raw_bgr, input_height, input_width):
    h, w = raw_bgr.shape[:2]
    image_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB) / 255.0
    return infer_preprocessed_rgb(
        model,
        image_rgb,
        input_height,
        input_width,
        h,
        w,
        keep_aspect_ratio=False,
        use_imagenet_norm=True,
    )


def infer_rectangular_raw(
    model,
    image_rgb,
    input_height,
    input_width,
    target_height,
    target_width,
    *,
    use_imagenet_norm,
):
    return infer_preprocessed_rgb(
        model,
        image_rgb,
        input_height,
        input_width,
        target_height,
        target_width,
        keep_aspect_ratio=False,
        use_imagenet_norm=use_imagenet_norm,
    )


def infer_square_raw(
    model,
    image_rgb,
    input_size,
    target_height,
    target_width,
    *,
    use_imagenet_norm,
):
    return infer_preprocessed_rgb(
        model,
        image_rgb,
        input_size,
        input_size,
        target_height,
        target_width,
        keep_aspect_ratio=True,
        use_imagenet_norm=use_imagenet_norm,
    )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unwrap_model_state(state_obj):
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        state_obj = state_obj["model"]
    if isinstance(state_obj, dict) and state_obj and all(key.startswith("module.") for key in state_obj.keys()):
        state_obj = {key[len("module."):]: value for key, value in state_obj.items()}
    return state_obj


def load_raw_adapter_model(
    encoder,
    checkpoint,
    *,
    input_type="raw_ram",
    bridge_source="ram_core",
    bridge_feature_keys=None,
    bridge_layers=None,
    lora_block_mode=DEFAULT_LORA_BLOCK_MODE,
    lora_rank=8,
    lora_alpha=16.0,
):
    """Load a raw-family checkpoint for evaluation."""
    from depth_anything_v2.dpt import DepthAnythingV2
    from finetune_stf.models.lora_bridge import build_raw_ram_bridge_depth_model
    from finetune_stf.models.raw_ram import build_raw_ram_depth_model

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    if input_type in RAW_RAM_BRIDGE_INPUT_TYPES:
        model = build_raw_ram_bridge_depth_model(
            dav2,
            input_type=input_type,
            bridge_source=bridge_source,
            bridge_feature_keys=bridge_feature_keys or DEFAULT_BRIDGE_FEATURE_KEYS,
            bridge_layers=bridge_layers,
            lora_block_mode=lora_block_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
    else:
        model = build_raw_ram_depth_model(dav2, input_type=input_type)
    state_dict = unwrap_model_state(torch.load(checkpoint, map_location="cpu"))
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


def infer_raw_ram(model, bayer_4ch_norm, input_height, input_width, target_height, target_width):
    """Run inference with RawRamDepthModel on 4-channel packed Bayer input."""
    # bayer_4ch_norm: (H, W, 4) float32 in [0, 1]
    # Resize to model input resolution
    resized = cv2.resize(
        bayer_4ch_norm,
        (input_width, input_height),
        interpolation=cv2.INTER_CUBIC,
    )
    # (H, W, 4) -> (4, H, W) -> (1, 4, H, W)
    tensor = torch.from_numpy(
        np.ascontiguousarray(resized.transpose(2, 0, 1)).astype(np.float32)
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_disp = model(tensor)
        pred_disp = F.interpolate(
            pred_disp[:, None],
            (target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

    return pred_disp.cpu().numpy()


def spectral_r_rgb(norm):
    anchors = np.linspace(0.0, 1.0, len(SPECTRAL_R_RGB), dtype=np.float32)
    flat = np.clip(norm.reshape(-1), 0.0, 1.0)
    channels = [
        np.interp(flat, anchors, SPECTRAL_R_RGB[:, channel]).reshape(norm.shape)
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def colorize_depth(depth, valid_mask, vmin, vmax):
    clipped = np.clip(depth.astype(np.float32), vmin, vmax)
    norm = (clipped - vmin) / max(vmax - vmin, 1e-6)
    rgb = spectral_r_rgb(norm)
    bgr = rgb[..., ::-1].copy()
    bgr[~valid_mask] = 0
    return bgr


def dilate_sparse_colormap(vis_bgr, valid_mask, kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1 or not valid_mask.any():
        return vis_bgr

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(vis_bgr, kernel, iterations=1)
    dilated_mask = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=1) > 0

    output = np.zeros_like(vis_bgr)
    output[dilated_mask] = dilated[dilated_mask]
    return output


def colorize_error_map(error_map, valid_mask, error_vis_max):
    error_vis_max = max(float(error_vis_max), 1e-6)
    clipped = np.clip(error_map.astype(np.float32), 0.0, error_vis_max)
    scaled = np.round(clipped / error_vis_max * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    heatmap[~valid_mask] = 0
    return heatmap


def make_depth_eval_mask(gt_depth, aligned_depth, valid_mask):
    return valid_mask & np.isfinite(aligned_depth) & (aligned_depth > 0) & (gt_depth > 0)


def make_depth_color_bar(width, height):
    gradient = np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))
    rgb = spectral_r_rgb(gradient)
    return rgb[..., ::-1].copy()


def make_error_color_bar(width, height):
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)


def draw_color_bar(canvas, x, y, title, bar_bgr, left_label, right_label):
    bar_h, bar_w = bar_bgr.shape[:2]
    title_y = max(y - 6, 12)
    cv2.putText(
        canvas,
        title,
        (x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    canvas[y : y + bar_h, x : x + bar_w] = bar_bgr
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (0, 0, 0), 1)
    label_y = y + bar_h + 14
    cv2.putText(
        canvas,
        left_label,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    right_size = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        canvas,
        right_label,
        (x + bar_w - right_size[0], label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def choose_vis_range(gt_depth, valid_mask, min_depth, max_depth):
    if valid_mask.any():
        valid_gt = gt_depth[valid_mask]
        vmin = max(min_depth, float(valid_gt.min()))
        vmax = min(max_depth, float(valid_gt.max()))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = min_depth, max_depth
    else:
        vmin, vmax = min_depth, max_depth
    return vmin, vmax


def save_visualization(
    vis_dir,
    idx,
    raw_bgr,
    aligned_depth,
    gt_depth,
    gt_valid_mask,
    sample_info,
    metrics,
    align_stats,
    min_depth,
    max_depth,
    gt_vis_dilate_kernel,
    error_vis_max,
):
    pred_valid_mask = np.isfinite(aligned_depth) & (aligned_depth > 0)
    eval_valid_mask = make_depth_eval_mask(gt_depth, aligned_depth, gt_valid_mask)
    vmin, vmax = choose_vis_range(
        gt_depth=gt_depth,
        valid_mask=gt_valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    pred_vis = colorize_depth(aligned_depth, pred_valid_mask, vmin, vmax)
    gt_vis = colorize_depth(gt_depth, gt_valid_mask, vmin, vmax)
    gt_vis = dilate_sparse_colormap(gt_vis, gt_valid_mask, gt_vis_dilate_kernel)
    error_map = np.zeros_like(gt_depth, dtype=np.float32)
    error_map[eval_valid_mask] = (
        np.abs(aligned_depth[eval_valid_mask] - gt_depth[eval_valid_mask])
        / np.clip(gt_depth[eval_valid_mask], a_min=1e-6, a_max=None)
    )
    error_vis = colorize_error_map(error_map, eval_valid_mask, error_vis_max)
    error_vis = dilate_sparse_colormap(error_vis, eval_valid_mask, gt_vis_dilate_kernel)

    header_h = 78
    h, w = raw_bgr.shape[:2]
    canvas = np.full((h + header_h, w * 4, 3), 255, dtype=np.uint8)
    canvas[header_h:, 0:w] = raw_bgr
    canvas[header_h:, w : 2 * w] = pred_vis
    canvas[header_h:, 2 * w : 3 * w] = gt_vis
    canvas[header_h:, 3 * w : 4 * w] = error_vis

    cv2.putText(canvas, f"input | idx={idx} | {sample_info['sample_name']}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"pred aligned | abs_rel={metrics['abs_rel']:.3f} | inv_align={align_stats['invalid_aligned_ratio']:.3f}", (w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "gt | shared depth scale with pred", (2 * w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"error map | rel clip={error_vis_max:.2f}", (3 * w + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    bar_y = 34
    bar_w = min(320, max(180, w - 120))
    bar_h = 12
    depth_bar = make_depth_color_bar(bar_w, bar_h)
    error_bar = make_error_color_bar(bar_w, bar_h)
    draw_color_bar(
        canvas,
        x=w + 12,
        y=bar_y,
        title="depth scale (shared by pred and gt)",
        bar_bgr=depth_bar,
        left_label=f"{vmin:.1f}m",
        right_label=f"{vmax:.1f}m",
    )
    draw_color_bar(
        canvas,
        x=3 * w + 12,
        y=bar_y,
        title="error scale",
        bar_bgr=error_bar,
        left_label="0.00",
        right_label=f"{error_vis_max:.2f}",
    )

    cv2.imwrite(str(vis_dir / f"triptych_{idx:04d}.jpg"), canvas)
    cv2.imwrite(str(vis_dir / f"image_{idx:04d}.jpg"), raw_bgr)
    cv2.imwrite(str(vis_dir / f"pred_{idx:04d}.jpg"), pred_vis)
    cv2.imwrite(str(vis_dir / f"gt_{idx:04d}.jpg"), gt_vis)
    cv2.imwrite(str(vis_dir / f"error_{idx:04d}.jpg"), error_vis)


def resolve_vis_dir(save_dir, split, checkpoint_path, vis_dir_arg):
    if vis_dir_arg:
        return Path(vis_dir_arg).expanduser().resolve()
    checkpoint_stem = Path(checkpoint_path).stem
    return (Path(save_dir).expanduser().resolve() / f"stf_vis_{split}_{checkpoint_stem}").resolve()


def aggregate_metrics(records):
    metric_keys = [k for k in records[0]["metrics"].keys() if k != "valid_eval_pixels"]
    summary = {
        key: float(np.mean([record["metrics"][key] for record in records]))
        for key in metric_keys
    }
    summary["avg_valid_eval_pixels"] = float(
        np.mean([record["metrics"]["valid_eval_pixels"] for record in records])
    )
    summary["avg_invalid_aligned_ratio"] = float(
        np.mean([record["align_stats"]["invalid_aligned_ratio"] for record in records])
    )
    return summary


def write_results(save_dir, split, checkpoint_path, manifest_path, records, summary, vis_dir, min_depth, max_depth):
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    txt_path = save_dir / f"eval_stf_rel_depth_{split}_{stamp}.txt"
    csv_path = save_dir / f"eval_stf_rel_depth_{split}_{stamp}.csv"

    lines = [
        f"STF split: {split}",
        f"Samples evaluated: {len(records)}",
        f"Checkpoint: {checkpoint_path}",
        f"Manifest: {manifest_path}",
        f"Visualization dir: {vis_dir if vis_dir is not None else 'disabled'}",
        f"Metric depth clip: [{min_depth:.2f}, {max_depth:.2f}]",
        "",
        f"abs_rel={summary['abs_rel']:.4f}",
        f"sq_rel={summary['sq_rel']:.4f}",
        f"rmse={summary['rmse']:.4f}",
        f"rmse_log={summary['rmse_log']:.4f}",
        f"log10={summary['log10']:.4f}",
        f"silog={summary['silog']:.4f}",
        f"silog_x100={summary['silog_x100']:.2f}",
        f"d1={summary['d1']:.4f}",
        f"d2={summary['d2']:.4f}",
        f"d3={summary['d3']:.4f}",
        f"avg_valid_eval_pixels={summary['avg_valid_eval_pixels']:.1f}",
        f"avg_invalid_aligned_ratio={summary['avg_invalid_aligned_ratio']:.4f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    base_fieldnames = [
        "index",
        "sample_name",
        "image_path",
        "depth_path",
        "scale",
        "shift",
        "invalid_aligned_ratio",
        "valid_eval_pixels",
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "log10",
        "silog",
        "silog_x100",
        "d1",
        "d2",
        "d3",
    ]
    base_fieldname_set = set(base_fieldnames)
    extra_metric_fieldnames = sorted(
        {
            key
            for record in records
            for key in record["metrics"].keys()
            if key not in base_fieldname_set
        }
    )
    fieldnames = base_fieldnames + extra_metric_fieldnames
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "index": record["index"],
                    "sample_name": record["sample_name"],
                    "image_path": record["image_path"],
                    "depth_path": record["depth_path"],
                    "scale": record["align_stats"]["scale"],
                    "shift": record["align_stats"]["shift"],
                    "invalid_aligned_ratio": record["align_stats"]["invalid_aligned_ratio"],
                    **record["metrics"],
                }
            )

    return txt_path, csv_path


def main():
    args = parse_args()
    input_height, input_width = resolve_input_shape(args)

    args.checkpoint = resolve_checkpoint_path(args.encoder, args.checkpoint)
    manifest_path = resolve_manifest_path(args.stf_root, args.split, args.manifest_path)
    rows = load_manifest_rows(
        manifest_path,
        args.stf_root,
        max_samples=args.max_samples,
        input_type=args.input_type,
        raw_npz_root=args.raw_npz_root,
    )

    vis_indices = set()
    vis_dir = None
    if not args.no_save_vis:
        vis_indices = set(make_indices(len(rows), args.vis_num_samples, args.vis_indices))
        vis_dir = resolve_vis_dir(args.save_dir, args.split, args.checkpoint, args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.encoder} from {args.checkpoint} (input_type={args.input_type}) ...", flush=True)
    if args.input_type in ("raw_ram", "raw_ram_residual", *RAW_RAM_BRIDGE_INPUT_TYPES):
        model = load_raw_adapter_model(
            args.encoder,
            args.checkpoint,
            input_type=args.input_type,
            bridge_source=args.bridge_source,
            bridge_feature_keys=args.bridge_feature_keys,
            bridge_layers=args.bridge_layers,
            lora_block_mode=args.lora_block_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
    else:
        model = load_model(args.encoder, args.checkpoint)

    records = []
    vis_sample_lines = ["index\tsample_name\timage_path\tdepth_path"]
    for eval_idx, sample_info in enumerate(rows):
        image_path = sample_info["image_path"]
        depth_path = sample_info["depth_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing STF image: {image_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing STF depth: {depth_path}")

        gt_depth = load_depth_npz(depth_path)
        valid_mask = np.isfinite(gt_depth) & (gt_depth >= args.min_depth) & (gt_depth <= args.max_depth)
        vis_bgr = None

        if args.input_type in ("raw_ram", "raw_ram_residual", *RAW_RAM_BRIDGE_INPUT_TYPES):
            bayer_rect = load_rectified_bayer_npz(image_path)
            bayer_rect = decode_stf_raw_4ch(bayer_rect, decode_mode=args.stf_raw_decode_mode)
            bayer_4ch_norm = normalize_raw_4ch(bayer_rect, norm_mode=args.norm_mode)
            # For visualization, create a pseudo-RGB from the 4ch Bayer
            image_rgb_vis = bayer_to_3ch(bayer_rect, channel_mode="rgb_avg_g")
            image_rgb_vis = normalize_raw(image_rgb_vis, norm_mode=args.norm_mode)
            raw_bgr = pseudo_rgb_to_bgr(image_rgb_vis)
            vis_bgr = cv2.resize(
                raw_bgr,
                (gt_depth.shape[1], gt_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            pred_disp = infer_raw_ram(
                model,
                bayer_4ch_norm,
                input_height,
                input_width,
                gt_depth.shape[0],
                gt_depth.shape[1],
            )
        elif args.input_type == "raw":
            bayer_rect = load_rectified_bayer_npz(image_path)
            bayer_rect = decode_stf_raw_4ch(bayer_rect, decode_mode=args.stf_raw_decode_mode)
            image_rgb = bayer_to_3ch(bayer_rect, channel_mode=args.channel_mode)
            image_rgb = normalize_raw(image_rgb, norm_mode=args.norm_mode)
            raw_bgr = pseudo_rgb_to_bgr(image_rgb)
            vis_bgr = cv2.resize(
                raw_bgr,
                (gt_depth.shape[1], gt_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            if args.input_height is not None and args.input_width is not None:
                pred_disp = infer_rectangular_raw(
                    model,
                    image_rgb,
                    input_height,
                    input_width,
                    gt_depth.shape[0],
                    gt_depth.shape[1],
                    use_imagenet_norm=args.use_imagenet_norm,
                )
            else:
                pred_disp = infer_square_raw(
                    model,
                    image_rgb,
                    args.input_size,
                    gt_depth.shape[0],
                    gt_depth.shape[1],
                    use_imagenet_norm=args.use_imagenet_norm,
                )
        else:
            raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                raise ValueError(f"Failed to read STF image: {image_path}")
            vis_bgr = raw_bgr

            if args.input_height is not None and args.input_width is not None:
                pred_disp = infer_rectangular(model, raw_bgr, input_height, input_width)
            else:
                pred_disp = model.infer_image(raw_bgr, args.input_size)

        check_sample_shapes(pred_disp, gt_depth, valid_mask, sample_info["sample_name"])
        if valid_mask.sum() < 10:
            print(f"  [warn] {sample_info['sample_name']}: too few valid GT pixels, skipping", flush=True)
            continue

        aligned_depth, align_stats = affine_align_disp(gt_depth, pred_disp, valid_mask)
        metrics = compute_metrics(
            gt_depth,
            aligned_depth,
            valid_mask,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        if metrics is None:
            print(f"  [warn] {sample_info['sample_name']}: no valid pixels after alignment, skipping", flush=True)
            continue

        records.append(
            {
                "index": eval_idx,
                "sample_name": sample_info["sample_name"],
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "align_stats": align_stats,
                "metrics": metrics,
            }
        )

        if align_stats["invalid_aligned_ratio"] > 0.01:
            print(
                f"  [warn] {sample_info['sample_name']}: invalid_aligned_ratio={align_stats['invalid_aligned_ratio']:.4f}",
                flush=True,
            )

        if eval_idx in vis_indices and vis_dir is not None:
            save_visualization(
                vis_dir=vis_dir,
                idx=eval_idx,
                raw_bgr=vis_bgr,
                aligned_depth=aligned_depth,
                gt_depth=gt_depth,
                gt_valid_mask=valid_mask,
                sample_info=sample_info,
                metrics=metrics,
                align_stats=align_stats,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                gt_vis_dilate_kernel=args.gt_vis_dilate_kernel,
                error_vis_max=args.error_vis_max,
            )
            vis_sample_lines.append(
                f"{eval_idx}\t{sample_info['sample_name']}\t{image_path}\t{depth_path}"
            )

        if (eval_idx + 1) % 50 == 0:
            print(f"  [{eval_idx + 1}/{len(rows)}] ...", flush=True)

    if not records:
        raise RuntimeError("No valid STF samples were evaluated.")

    summary = aggregate_metrics(records)
    print(f"\n=== STF {args.split.upper()} ===", flush=True)
    print(
        f"  n={len(records)}  abs_rel={summary['abs_rel']:.4f}  rmse={summary['rmse']:.4f}  "
        f"silog={summary['silog']:.4f}  silog_x100={summary['silog_x100']:.2f}  "
        f"d1={summary['d1']:.4f}  d2={summary['d2']:.4f}  d3={summary['d3']:.4f}  "
        f"invalid_align={summary['avg_invalid_aligned_ratio']:.4f}",
        flush=True,
    )

    if vis_dir is not None:
        (vis_dir / "samples.txt").write_text("\n".join(vis_sample_lines) + "\n", encoding="utf-8")
        print(f"Saved STF visualizations to {vis_dir}", flush=True)

    if not args.no_save_results:
        txt_path, csv_path = write_results(
            save_dir=args.save_dir,
            split=args.split,
            checkpoint_path=args.checkpoint,
            manifest_path=manifest_path,
            records=records,
            summary=summary,
            vis_dir=vis_dir,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        print(f"Saved STF report to {txt_path}", flush=True)
        print(f"Saved STF per-sample CSV to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
