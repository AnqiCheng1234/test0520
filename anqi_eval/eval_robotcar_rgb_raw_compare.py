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
from finetune_stf.dataset.robotcar import RobotCarValRGB, RobotCarValRaw
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    RAW_RAM_BRIDGE_INPUT_TYPES,
    RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
    build_raw_ram_bridge_depth_model,
)
from finetune_stf.models.raw_ram import build_raw_ram_depth_model
from finetune_stf.models.raw_ram import packed_bayer_to_base_rgb
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
RECTIFIED_BAYER_KEY = "bayer_rect"
DEPTH_CMAP = colormaps["Spectral_r"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RobotCar sparse depth with RGB zero-shot reference and RAW checkpoint, and save comparison panels."
    )
    parser.add_argument("exp_dir", type=Path, help="Experiment directory containing config.json.")
    parser.add_argument("--checkpoint", default="best", help='Checkpoint to use: "best", "last", or a custom .pth path.')
    parser.add_argument("--robotcar-root", default="/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640")
    parser.add_argument("--manifest-name", default=None, help="Optional manifest filename inside <robotcar-root>/manifests/. Defaults to dataset's default day manifest; use the night balanced manifest for night runs.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults inside exp_dir.")
    parser.add_argument("--fast-eval-backend", default="sparse", choices=["proxy", "sparse"])
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--subset-indices",
        default=None,
        help="Optional JSON file containing {'indices': [...]} to evaluate an exact fixed subset.",
    )
    parser.add_argument("--raw-source", default="native", choices=["eval", "native"])
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument(
        "--error-max",
        type=float,
        default=0.5,
        help="Relative error value mapped to the top of the error colormap.",
    )
    parser.add_argument("--save-panels", action="store_true", default=True)
    parser.add_argument("--no-save-panels", action="store_false", dest="save_panels")
    parser.add_argument(
        "--panels-only",
        action="store_true",
        help="Save only comparison panels and metadata; skip individual RGB/RAW image writes.",
    )
    parser.add_argument(
        "--include-ram-rgb-preview",
        action="store_true",
        help="Add a panel showing the clamped 3-channel output of RamCore3 before ImageNet normalization.",
    )
    parser.add_argument(
        "--stratified-eval",
        action="store_true",
        default=False,
        help="Compute baseline_80m / refit_max40 / refit_max50 / strat_near40 in one pass; skips per-sample image and panel writes.",
    )
    return parser.parse_args()


def build_protocol_defs(min_depth: float, max_depth: float) -> tuple[tuple[str, float, float, float, float], ...]:
    max_tag = int(round(float(max_depth)))
    protocol_defs = [
        (f"baseline_{max_tag}m", float(min_depth), float(max_depth), float(min_depth), float(max_depth)),
    ]
    if float(max_depth) > 40.0:
        protocol_defs.append(("refit_max40", float(min_depth), 40.0, float(min_depth), 40.0))
        protocol_defs.append((f"strat_near40_fit{max_tag}m", float(min_depth), float(max_depth), float(min_depth), 40.0))
    if float(max_depth) > 50.0:
        protocol_defs.append(("refit_max50", float(min_depth), 50.0, float(min_depth), 50.0))
    return tuple(protocol_defs)


def compute_stratified_metrics(depth, pred_disp, valid_mask, protocol_defs):
    """Return {protocol_name: metrics_dict or None} for the requested RobotCar protocols."""
    out = {}
    aligned_cache = {}
    for name, fit_min, fit_max, eval_min, eval_max in protocol_defs:
        fit_key = (float(fit_min), float(fit_max))
        if fit_key not in aligned_cache:
            fit_mask = valid_mask & (depth >= fit_min) & (depth <= fit_max)
            if int(fit_mask.sum()) < 10:
                aligned_cache[fit_key] = None
            else:
                aligned, _ = affine_align_disp(depth, pred_disp, fit_mask)
                aligned_cache[fit_key] = aligned
        aligned = aligned_cache[fit_key]
        if aligned is None:
            out[name] = None
            continue
        eval_mask = valid_mask & (depth >= eval_min) & (depth <= eval_max)
        out[name] = compute_metrics(depth, aligned, eval_mask, eval_min, eval_max)
    return out


def affine_align_disp_1d(gt_depth: np.ndarray, pred_disp: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gt_depth = np.asarray(gt_depth, dtype=np.float64).reshape(-1)
    pred_disp = np.asarray(pred_disp, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt_depth) & (gt_depth > 0.0) & np.isfinite(pred_disp)
    if int(valid.sum()) < 2:
        aligned_depth = np.full(gt_depth.shape, np.nan, dtype=np.float64)
        return aligned_depth, {
            "scale": 0.0,
            "shift": 0.0,
            "invalid_aligned_pixels": int(valid.size),
            "invalid_aligned_ratio": 1.0,
        }

    gt_disp = 1.0 / np.clip(gt_depth[valid], a_min=1e-9, a_max=None)
    x = pred_disp[valid]
    A = np.stack([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(A, gt_disp, rcond=None)
    scale, shift = float(coef[0]), float(coef[1])

    aligned_disp = pred_disp * scale + shift
    aligned_depth = np.full(pred_disp.shape, np.nan, dtype=np.float64)
    pos = np.isfinite(aligned_disp) & (aligned_disp > 0.0)
    aligned_depth[pos] = 1.0 / aligned_disp[pos]
    invalid_count = int(valid.sum() - np.count_nonzero(valid & pos))
    return aligned_depth, {
        "scale": scale,
        "shift": shift,
        "invalid_aligned_pixels": invalid_count,
        "invalid_aligned_ratio": float(invalid_count / max(int(valid.sum()), 1)),
    }


def sample_bilinear_disparity_at_mask_np(
    pred_disp: np.ndarray,
    valid_mask: np.ndarray,
    full_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pred_disp = np.asarray(pred_disp, dtype=np.float32)
    if pred_disp.ndim != 2:
        raise ValueError(f"Expected 2D disparity map, got shape {pred_disp.shape}")
    coords = np.argwhere(np.asarray(valid_mask, dtype=bool))
    if coords.size == 0:
        return coords, np.zeros((0,), dtype=pred_disp.dtype)

    src_h, src_w = pred_disp.shape
    full_h, full_w = int(full_hw[0]), int(full_hw[1])
    ys = coords[:, 0].astype(np.float32)
    xs = coords[:, 1].astype(np.float32)
    if full_h > 1:
        ys *= float(src_h - 1) / float(full_h - 1)
    else:
        ys.fill(0.0)
    if full_w > 1:
        xs *= float(src_w - 1) / float(full_w - 1)
    else:
        xs.fill(0.0)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)

    wy = ys - y0.astype(np.float32)
    wx = xs - x0.astype(np.float32)
    samples = (
        pred_disp[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + pred_disp[y0, x1] * (1.0 - wy) * wx
        + pred_disp[y1, x0] * wy * (1.0 - wx)
        + pred_disp[y1, x1] * wy * wx
    )
    return coords, samples


def apply_disp_alignment(pred_disp: np.ndarray, scale: float, shift: float) -> np.ndarray:
    aligned_disp = np.asarray(pred_disp, dtype=np.float64) * float(scale) + float(shift)
    aligned_depth = np.full(aligned_disp.shape, np.nan, dtype=np.float64)
    pos = np.isfinite(aligned_disp) & (aligned_disp > 0.0)
    aligned_depth[pos] = 1.0 / aligned_disp[pos]
    return aligned_depth


def align_prediction_for_eval_and_panel(
    depth: np.ndarray,
    pred_disp_dense: np.ndarray,
    pred_disp_native: np.ndarray,
    valid_mask: np.ndarray,
    *,
    use_sparse_fast_eval: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not use_sparse_fast_eval:
        aligned_depth, stats = affine_align_disp(depth, pred_disp_dense, valid_mask)
        return aligned_depth, aligned_depth, stats

    coords, pred_samples = sample_bilinear_disparity_at_mask_np(pred_disp_native, valid_mask, depth.shape[-2:])
    depth_samples = depth[valid_mask]
    aligned_samples, stats = affine_align_disp_1d(depth_samples, pred_samples)
    metric_aligned_depth = np.full(depth.shape, np.nan, dtype=np.float64)
    if coords.size:
        metric_aligned_depth[coords[:, 0], coords[:, 1]] = aligned_samples
    panel_aligned_depth = apply_disp_alignment(pred_disp_dense, stats["scale"], stats["shift"])
    return metric_aligned_depth, panel_aligned_depth, stats


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


def resolve_raw_input_mode(input_type: str) -> str:
    if input_type in (
        *RAW_PACKED_INPUT_TYPES,
        "raw_ram",
        "raw_ram_residual",
        *RAW_RAM_BRIDGE_INPUT_TYPES,
        *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
        *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    ):
        return "raw_ram"
    return "raw_naive"


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
        raise ValueError(f"Unsupported input_type for RobotCar compare script: {input_type}")

    state_dict = unwrap_model_state(torch.load(checkpoint_path, map_location="cpu"))
    if any(k.startswith("ram_core.ffm.fuse.") for k in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    if hasattr(model, "load_compatible_state_dict"):
        model.load_compatible_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()


def build_rgb_reference_model(cfg: dict, sensor_hw: tuple[int, int]):
    rgb_ckpt = Path(cfg["pretrained_from"]).expanduser().resolve()
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[cfg["encoder"]])
    model = build_dav2_padded_rgb_depth_model(
        dav2,
        sensor_hw=sensor_hw,
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


def make_dense_pred_mask(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    return np.isfinite(depth) & (depth > 0.0)


def relative_error_map(depth: np.ndarray, pred_depth: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    pred_depth = np.asarray(pred_depth, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    pred_valid = make_dense_pred_mask(pred_depth)
    err_valid = valid_mask & pred_valid & np.isfinite(depth) & (depth > 0.0)
    err = np.zeros_like(depth, dtype=np.float32)
    err[err_valid] = np.abs(pred_depth[err_valid] - depth[err_valid]) / np.clip(depth[err_valid], 1e-6, None)
    return err, err_valid


def colorize_error(error_map: np.ndarray, valid_mask: np.ndarray, *, max_rel_error: float) -> Image.Image:
    error_map = np.asarray(error_map, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    scale = max(float(max_rel_error), 1e-6)
    values = np.clip(error_map, 0.0, scale)
    encoded = np.round(values / scale * 255.0).astype(np.uint8)
    rgb = cv2.applyColorMap(encoded, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~valid_mask] = 0
    return Image.fromarray(rgb)


def build_raw_preview(raw_4ch_path: Path) -> Image.Image:
    with np.load(raw_4ch_path, allow_pickle=False) as data:
        raw = np.asarray(data[RECTIFIED_BAYER_KEY], dtype=np.float32)
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


def tensor_rgb_to_preview(x_rgb: torch.Tensor) -> Image.Image:
    if x_rgb.ndim != 3 or x_rgb.shape[0] != 3:
        raise ValueError(f"Expected CHW 3-channel tensor, got {tuple(x_rgb.shape)}")
    preview = x_rgb.detach().float().cpu().numpy().transpose(1, 2, 0)
    preview = np.clip(preview, 0.0, 1.0) ** (1.0 / 2.2)
    return Image.fromarray((preview * 255.0).round().astype(np.uint8))


def infer_ram_rgb_preview(model, image_tensor: torch.Tensor) -> Image.Image:
    if not hasattr(model, "ram_core"):
        raise ValueError("--include-ram-rgb-preview requires a model with ram_core")
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 4:
        raise ValueError(f"RAM RGB preview expects a CHW 4-channel RAW tensor, got {tuple(image_tensor.shape)}")

    x_raw = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    with torch.no_grad():
        x3_in = packed_bayer_to_base_rgb(x_raw)
        x3 = model.ram_core(x3_in)
        x_rgb = torch.clamp(x3, min=0.0, max=1.0)[0]
    return tensor_rgb_to_preview(x_rgb)


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


def resize_for_display(image: Image.Image, display_size: tuple[int, int]) -> Image.Image:
    if image.size == display_size:
        return image
    return image.resize(display_size, Image.Resampling.BILINEAR)


def infer_batched(
    model,
    image_tensor: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    use_amp: bool = True,
    return_native: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    amp_enabled = use_amp and DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(image_tensor).float()
    pred_native = pred[0].detach().cpu().numpy()
    if tuple(pred.shape[-2:]) != tuple(target_hw):
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0]
    pred_dense = pred[0].detach().cpu().numpy()
    if return_native:
        return pred_dense, pred_native
    return pred_dense


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


def sample_indices(dataset_size: int, sample_count: int, seed: int) -> list[int]:
    sample_count = min(int(sample_count), int(dataset_size))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(dataset_size, size=sample_count, replace=False)
    return sorted(int(idx) for idx in indices.tolist())


def load_subset_indices(path: str | None, dataset_size: int, sample_count: int, seed: int) -> list[int]:
    if not path:
        return sample_indices(dataset_size, sample_count, seed)
    subset_path = Path(path).expanduser().resolve()
    with subset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "indices" not in payload:
        raise ValueError(f"{subset_path} must contain a JSON object with an 'indices' list")
    indices = sorted({int(idx) for idx in payload["indices"]})
    if any(idx < 0 or idx >= dataset_size for idx in indices):
        raise ValueError(f"{subset_path} contains out-of-range dataset indices for size={dataset_size}")
    return indices


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
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth > 0")
    protocol_defs = build_protocol_defs(args.min_depth, args.max_depth)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (exp_dir / "robotcar_sparse500_rgb_raw_compare_best")
    )
    rgb_dir = output_dir / "rgb_reference"
    raw_dir = output_dir / "raw_model"
    panel_dir = output_dir / ("panels_8up_ram3ch" if args.include_ram_rgb_preview else "panels_5up")
    if args.stratified_eval:
        # metrics-only mode: skip per-sample image writes and panel generation
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.panels_only:
            panel_dir.mkdir(parents=True, exist_ok=True)
        else:
            for path in (rgb_dir, raw_dir, panel_dir):
                path.mkdir(parents=True, exist_ok=True)

    raw_kwargs = dict(
        robotcar_root=args.robotcar_root,
        depth_mode="fast",
        fast_eval_backend=args.fast_eval_backend,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        norm_mode="sensor_linear",
        channel_mode=cfg.get("channel_mode", "rgb_avg_g"),
        use_imagenet_norm=cfg.get("use_imagenet_norm", True),
        input_mode=resolve_raw_input_mode(input_type),
    )
    if args.manifest_name is not None:
        raw_kwargs["manifest_name"] = args.manifest_name
    raw_dataset = RobotCarValRaw(**raw_kwargs)
    if args.raw_source == "native":
        for row in raw_dataset.rows:
            row["raw_eval_path"] = row["raw_native_path"]
            row["raw_eval_hw"] = row["raw_native_hw"]
    rgb_kwargs = dict(
        robotcar_root=args.robotcar_root,
        depth_mode="fast",
        fast_eval_backend=args.fast_eval_backend,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    if args.manifest_name is not None:
        rgb_kwargs["manifest_name"] = args.manifest_name
    rgb_dataset = RobotCarValRGB(**rgb_kwargs)
    if len(raw_dataset) != len(rgb_dataset):
        raise RuntimeError(f"RobotCar raw/rgb dataset length mismatch: {len(raw_dataset)} vs {len(rgb_dataset)}")

    dataset_size = len(raw_dataset)
    indices = load_subset_indices(args.subset_indices, dataset_size, args.sample_count, args.sample_seed)
    first_rgb_path = Path(rgb_dataset.rows[indices[0]]["rgb_eval_path"])
    first_rgb_size = Image.open(first_rgb_path).size
    sensor_hw = (int(first_rgb_size[1]), int(first_rgb_size[0]))
    display_size = first_rgb_size

    rgb_model = build_rgb_reference_model(cfg, sensor_hw=sensor_hw)
    rgb_metrics = []
    panel_manifest = []
    panel_images = {}
    rgb_strat = {p[0]: [] for p in protocol_defs}
    print(
        f"[RobotCar][RGB] start samples={len(indices)} seed={args.sample_seed} pretrained_from={cfg['pretrained_from']}",
        flush=True,
    )
    for count, idx in enumerate(indices, start=1):
        sample = rgb_dataset[idx]
        depth = sample["depth"].numpy()
        valid = sample["valid_mask"].numpy().astype(bool)
        pred_disp, pred_disp_native = infer_batched(
            rgb_model,
            sample["image"],
            depth.shape[-2:],
            return_native=True,
        )
        use_sparse_fast_eval = sample["depth_mode"] == "fast" and sample["fast_eval_backend"] == "sparse"
        metric_aligned_depth, panel_aligned_depth, _ = align_prediction_for_eval_and_panel(
            depth,
            pred_disp,
            pred_disp_native,
            valid,
            use_sparse_fast_eval=use_sparse_fast_eval,
        )
        metrics = compute_metrics(depth, metric_aligned_depth, valid, min_depth=args.min_depth, max_depth=args.max_depth)
        if metrics is None:
            continue
        rgb_metrics.append({key: float(metrics[key]) for key in METRIC_KEYS})

        if args.stratified_eval:
            strat = compute_stratified_metrics(depth, pred_disp, valid, protocol_defs)
            for proto_name, proto_metrics in strat.items():
                if proto_metrics is not None:
                    rgb_strat[proto_name].append({k: float(proto_metrics[k]) for k in METRIC_KEYS})
            if count == 1 or count % 100 == 0 or count == len(indices):
                print(f"[RobotCar][RGB] processed {count}/{len(indices)} {sample['sample_name']}", flush=True)
            continue

        row = rgb_dataset.rows[idx]
        sample_name = sample["sample_name"]
        valid_depth = depth[valid]
        vmin = max(float(args.min_depth), float(valid_depth.min()))
        vmax = min(float(args.max_depth), float(valid_depth.max()))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(args.min_depth), float(args.max_depth)

        rgb_image = resize_for_display(Image.open(row["rgb_eval_path"]).convert("RGB"), display_size)
        gt_image = resize_for_display(colorize_depth(depth, valid, vmin=vmin, vmax=vmax), display_size)
        # Keep GT sparse, but visualize predictions densely wherever aligned depth is valid.
        rgb_pred = resize_for_display(
            colorize_depth(panel_aligned_depth, make_dense_pred_mask(panel_aligned_depth), vmin=vmin, vmax=vmax),
            display_size,
        )
        rgb_error_map, rgb_error_valid = relative_error_map(depth, metric_aligned_depth, valid)
        rgb_error = resize_for_display(
            colorize_error(rgb_error_map, rgb_error_valid, max_rel_error=args.error_max),
            display_size,
        )

        safe_name = sample_name.replace("/", "__")
        record = {
            "dataset_index": idx,
            "order_index": count,
            "sample_name": sample_name,
            "scene": sample["scene"],
            "vmin": vmin,
            "vmax": vmax,
            "rgb_abs_rel": float(metrics["abs_rel"]),
            "rgb_rmse": float(metrics["rmse"]),
        }
        if args.panels_only:
            panel_images[count] = {
                "rgb": rgb_image.copy(),
                "rgb_pred": rgb_pred.copy(),
                "rgb_error": rgb_error.copy(),
                "gt": gt_image.copy(),
            }
            record["rgb_path"] = None
            record["rgb_pred_path"] = None
            record["rgb_error_path"] = None
            record["gt_path"] = None
        else:
            rgb_image_path = rgb_dir / f"{count:04d}_{safe_name}_rgb.jpg"
            rgb_pred_path = rgb_dir / f"{count:04d}_{safe_name}_rgb_pred.jpg"
            rgb_error_path = rgb_dir / f"{count:04d}_{safe_name}_rgb_error.jpg"
            gt_path = rgb_dir / f"{count:04d}_{safe_name}_gt.jpg"
            rgb_image.save(rgb_image_path, quality=95)
            rgb_pred.save(rgb_pred_path, quality=95)
            rgb_error.save(rgb_error_path, quality=95)
            gt_image.save(gt_path, quality=95)
            record["rgb_path"] = str(rgb_image_path)
            record["rgb_pred_path"] = str(rgb_pred_path)
            record["rgb_error_path"] = str(rgb_error_path)
            record["gt_path"] = str(gt_path)
        panel_manifest.append(record)
        if count == 1 or count % 25 == 0 or count == len(indices):
            print(f"[RobotCar][RGB] processed {count}/{len(indices)} {sample_name}", flush=True)

    del rgb_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    raw_model = build_raw_model(cfg, checkpoint_path)
    raw_metrics = []
    raw_strat = {p[0]: [] for p in protocol_defs}
    manifest_by_order = {record["order_index"]: record for record in panel_manifest}
    print(f"[RobotCar][RAW] start samples={len(indices)} checkpoint={checkpoint_path.name}", flush=True)
    for count, idx in enumerate(indices, start=1):
        sample = raw_dataset[idx]
        depth = sample["depth"].numpy()
        valid = sample["valid_mask"].numpy().astype(bool)
        image_tensor = sample["raw"] if "raw" in sample else sample["image"]
        pred_disp, pred_disp_native = infer_batched(raw_model, image_tensor, depth.shape[-2:], return_native=True)
        use_sparse_fast_eval = sample["depth_mode"] == "fast" and sample["fast_eval_backend"] == "sparse"
        metric_aligned_depth, panel_aligned_depth, _ = align_prediction_for_eval_and_panel(
            depth,
            pred_disp,
            pred_disp_native,
            valid,
            use_sparse_fast_eval=use_sparse_fast_eval,
        )
        metrics = compute_metrics(depth, metric_aligned_depth, valid, min_depth=args.min_depth, max_depth=args.max_depth)
        if metrics is None:
            continue
        if not args.stratified_eval and count not in manifest_by_order:
            continue
        raw_metrics.append({key: float(metrics[key]) for key in METRIC_KEYS})

        if args.stratified_eval:
            strat = compute_stratified_metrics(depth, pred_disp, valid, protocol_defs)
            for proto_name, proto_metrics in strat.items():
                if proto_metrics is not None:
                    raw_strat[proto_name].append({k: float(proto_metrics[k]) for k in METRIC_KEYS})
            if count == 1 or count % 100 == 0 or count == len(indices):
                print(f"[RobotCar][RAW] processed {count}/{len(indices)} {sample['sample_name']}", flush=True)
            continue

        record = manifest_by_order[count]
        row = raw_dataset.rows[idx]
        sample_name = sample["sample_name"]
        raw_image = resize_for_display(build_raw_preview(Path(row["raw_eval_path"])), display_size)
        ram_rgb_image = None
        if args.include_ram_rgb_preview:
            ram_rgb_image = resize_for_display(infer_ram_rgb_preview(raw_model, image_tensor), display_size)
        raw_pred = resize_for_display(
            colorize_depth(
                panel_aligned_depth,
                make_dense_pred_mask(panel_aligned_depth),
                vmin=float(record["vmin"]),
                vmax=float(record["vmax"]),
            ),
            display_size,
        )
        raw_error_map, raw_error_valid = relative_error_map(depth, metric_aligned_depth, valid)
        raw_error = resize_for_display(
            colorize_error(raw_error_map, raw_error_valid, max_rel_error=args.error_max),
            display_size,
        )
        if args.panels_only:
            record["raw_path"] = None
            record["ram_rgb_path"] = None if args.include_ram_rgb_preview else None
            record["raw_pred_path"] = None
            record["raw_error_path"] = None
        else:
            raw_image_path = raw_dir / f"{count:04d}_{sample_name.replace('/', '__')}_raw.jpg"
            ram_rgb_path = raw_dir / f"{count:04d}_{sample_name.replace('/', '__')}_ram_rgb.jpg"
            raw_pred_path = raw_dir / f"{count:04d}_{sample_name.replace('/', '__')}_raw_pred.jpg"
            raw_error_path = raw_dir / f"{count:04d}_{sample_name.replace('/', '__')}_raw_error.jpg"
            raw_image.save(raw_image_path, quality=95)
            if ram_rgb_image is not None:
                ram_rgb_image.save(ram_rgb_path, quality=95)
            raw_pred.save(raw_pred_path, quality=95)
            raw_error.save(raw_error_path, quality=95)
            record["raw_path"] = str(raw_image_path)
            record["ram_rgb_path"] = str(ram_rgb_path) if ram_rgb_image is not None else None
            record["raw_pred_path"] = str(raw_pred_path)
            record["raw_error_path"] = str(raw_error_path)
        record["raw_abs_rel"] = float(metrics["abs_rel"])
        record["raw_rmse"] = float(metrics["rmse"])

        if args.save_panels:
            footer = (
                f"{sample_name} | rgb abs_rel={record['rgb_abs_rel']:.4f} rmse={record['rgb_rmse']:.4f} | "
                f"raw abs_rel={record['raw_abs_rel']:.4f} rmse={record['raw_rmse']:.4f}"
            )
            if args.panels_only:
                cached = panel_images[count]
                rgb_panel = cached["rgb"]
                rgb_pred_panel = cached["rgb_pred"]
                rgb_error_panel = cached["rgb_error"]
                gt_panel = cached["gt"]
            else:
                rgb_panel = Image.open(record["rgb_path"]).convert("RGB")
                rgb_pred_panel = Image.open(record["rgb_pred_path"]).convert("RGB")
                rgb_error_panel = Image.open(record["rgb_error_path"]).convert("RGB")
                gt_panel = Image.open(record["gt_path"]).convert("RGB")
            panel_parts = [rgb_panel, rgb_pred_panel, rgb_error_panel, raw_image]
            panel_labels = ["rgb", "rgb_pred", "rgb_error", "raw"]
            if ram_rgb_image is not None:
                panel_parts.append(ram_rgb_image)
                panel_labels.append("ram_rgb")
            panel_parts.extend([raw_pred, raw_error, gt_panel])
            panel_labels.extend(["raw_pred", "raw_error", "gt"])
            panel = build_panel(panel_parts, panel_labels, footer)
            panel_suffix = "8up_ram3ch" if ram_rgb_image is not None else "5up"
            panel.save(panel_dir / f"{count:04d}_{sample_name.replace('/', '__')}_{panel_suffix}.jpg", quality=95)
        if count == 1 or count % 25 == 0 or count == len(indices):
            print(f"[RobotCar][RAW] processed {count}/{len(indices)} {sample_name}", flush=True)

    summary = {
        "exp_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "pretrained_from": cfg["pretrained_from"],
        "robotcar_root": str(Path(args.robotcar_root).expanduser().resolve()),
        "fast_eval_backend": args.fast_eval_backend,
        "sample_count_requested": int(args.sample_count),
        "sample_count_actual": len(indices),
        "sample_seed": int(args.sample_seed),
        "subset_indices": str(Path(args.subset_indices).expanduser().resolve()) if args.subset_indices else None,
        "raw_source": args.raw_source,
        "input_type": input_type,
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "error_max": float(args.error_max),
        "rgb_reference_metrics": summarize_metrics(rgb_metrics),
        "raw_checkpoint_metrics": summarize_metrics(raw_metrics),
        "panel_dir": str(panel_dir),
        "panel_layout": (
            ["rgb", "rgb_pred", "rgb_error", "raw", "ram_rgb", "raw_pred", "raw_error", "gt"]
            if args.include_ram_rgb_preview
            else ["rgb", "rgb_pred", "rgb_error", "raw", "raw_pred", "raw_error", "gt"]
        ),
        "ram_rgb_preview": bool(args.include_ram_rgb_preview),
        "ram_rgb_preview_policy": (
            "packed RAW [R,Gr,Gb,B] is first converted to [R,(Gr+Gb)/2,B], passed through model.ram_core "
            "(RamCore3), clamped to [0,1], then gamma 1/2.2 is applied for display."
            if args.include_ram_rgb_preview
            else None
        ),
        "rgb_dir": None if args.panels_only else str(rgb_dir),
        "raw_dir": None if args.panels_only else str(raw_dir),
        "stratified_eval": bool(args.stratified_eval),
        "panels_only": bool(args.panels_only),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "subset_indices.json", {"indices": indices})
    if args.stratified_eval:
        stratified_summary = {
            "exp_dir": str(exp_dir),
            "checkpoint": str(checkpoint_path),
            "pretrained_from": cfg["pretrained_from"],
            "robotcar_root": str(Path(args.robotcar_root).expanduser().resolve()),
            "fast_eval_backend": args.fast_eval_backend,
            "sample_count_requested": int(args.sample_count),
            "sample_count_actual": len(indices),
            "sample_seed": int(args.sample_seed),
            "subset_indices": str(Path(args.subset_indices).expanduser().resolve()) if args.subset_indices else str(output_dir / "subset_indices.json"),
            "raw_source": args.raw_source,
            "input_type": input_type,
            "min_depth": float(args.min_depth),
            "max_depth": float(args.max_depth),
            "protocols": [
                {"name": name, "fit_min": fmin, "fit_max": fmax, "eval_min": emin, "eval_max": emax}
                for (name, fmin, fmax, emin, emax) in protocol_defs
            ],
            "rgb": {name: summarize_metrics(rgb_strat[name]) for name in rgb_strat},
            "raw": {name: summarize_metrics(raw_strat[name]) for name in raw_strat},
            "rgb_n": {name: len(rgb_strat[name]) for name in rgb_strat},
            "raw_n": {name: len(raw_strat[name]) for name in raw_strat},
        }
        save_json(output_dir / "summary_stratified.json", stratified_summary)
        print(json.dumps(stratified_summary, indent=2, ensure_ascii=False))
    else:
        save_json(output_dir / "panel_manifest.json", panel_manifest)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
