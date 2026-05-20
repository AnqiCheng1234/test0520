#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    RAW_RAM_BRIDGE_INPUT_TYPES,
    RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
    build_raw_ram_bridge_depth_model,
)
from finetune_stf.models.raw_feature_adapter import (
    DEFAULT_FEATURE_ADAPTER_KEYS,
    RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    build_raw_ram_feature_adapter_depth_model,
)
from finetune_stf.models.raw_ram import build_raw_ram_depth_model
from finetune_stf.models.raw_ram import packed_bayer_to_base_rgb
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model
from foundation.engine.models import build_dav2_raw_naive_depth_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFICIAL_VITL_CKPT = "/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"
DEFAULT_HEAVY_SAVE_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")
RAW_PACKED_INPUT_TYPES = ("raw_packed",)
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3")
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def unwrap_model_state(state_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        state_obj = state_obj["model"]
    if isinstance(state_obj, dict) and state_obj and all(str(key).startswith("module.") for key in state_obj):
        state_obj = {key[len("module.") :]: value for key, value in state_obj.items()}
    if not isinstance(state_obj, dict):
        raise TypeError(f"Expected checkpoint state dict, got {type(state_obj).__name__}")
    return state_obj


def remap_legacy_ffm_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
                new_key = new_prefix + key[len(old_prefix) :]
                break
        output[new_key] = value
    return output


def load_experiment_config(exp_dir: str | Path) -> dict[str, Any]:
    exp_dir = Path(exp_dir).expanduser().resolve()
    config_path = exp_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return read_json(config_path)


def resolve_checkpoint(exp_dir: str | Path, checkpoint: str, cfg: dict[str, Any] | None = None) -> Path:
    exp_dir = Path(exp_dir).expanduser().resolve()
    checkpoint = str(checkpoint)
    aliases = {
        "best": "best_model.pth",
        "last": "last_epoch_model.pth",
        "current": "current_model.pth",
    }
    if checkpoint not in aliases:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = (exp_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        return path

    filename = aliases[checkpoint]
    candidates = [exp_dir / filename]
    if cfg:
        heavy_save_path = cfg.get("heavy_save_path")
        if heavy_save_path:
            candidates.append(Path(heavy_save_path).expanduser().resolve() / filename)
        heavy_save_root = cfg.get("heavy_save_root")
        if heavy_save_root:
            candidates.append(Path(heavy_save_root).expanduser().resolve() / exp_dir.name / filename)
    candidates.append(DEFAULT_HEAVY_SAVE_ROOT / exp_dir.name / filename)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not resolve checkpoint alias {checkpoint!r}. Checked:\n{checked}")


def build_dav2(encoder: str) -> DepthAnythingV2:
    return DepthAnythingV2(**MODEL_CONFIGS[encoder])


def build_rgb_reference_model(cfg: dict[str, Any], *, sensor_hw: tuple[int, int] | None = None):
    ckpt_path = Path(cfg.get("pretrained_from") or OFFICIAL_VITL_CKPT).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing RGB DAv2 checkpoint: {ckpt_path}")
    if sensor_hw is None:
        sensor_hw = (int(cfg.get("input_height", 644)), int(cfg.get("input_width", 1008)))
    model = build_dav2_padded_rgb_depth_model(
        build_dav2(cfg.get("encoder", "vitl")),
        sensor_hw=sensor_hw,
        backbone_hw=None,
    )
    model.load_base_dav2_state_dict(unwrap_model_state(torch.load(ckpt_path, map_location="cpu")))
    return model.to(DEVICE).eval()


def build_raw_student_model(cfg: dict[str, Any], checkpoint_path: str | Path):
    input_type = cfg.get("input_type", "rgb")
    sensor_hw = (int(cfg.get("input_height", 644)), int(cfg.get("input_width", 1008)))
    dav2 = build_dav2(cfg.get("encoder", "vitl"))

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
    elif input_type in ("raw_ram", "raw_ram_residual", "raw_ram_rgb"):
        model = build_raw_ram_depth_model(
            dav2,
            input_type=input_type,
            rgb_interface_mode=cfg.get("rgb_interface_mode", "residual_tanh"),
            rgb_residual_scale=cfg.get("rgb_residual_scale", 0.1),
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
        model = build_raw_ram_feature_adapter_depth_model(
            dav2,
            input_type=input_type,
            feature_keys=cfg.get("bridge_feature_keys") or list(DEFAULT_FEATURE_ADAPTER_KEYS),
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
    elif input_type == "rgb":
        model = build_dav2_padded_rgb_depth_model(dav2, sensor_hw=sensor_hw, backbone_hw=None)
    else:
        raise ValueError(f"Unsupported input_type for audit: {input_type}")

    state_dict = unwrap_model_state(torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu"))
    if any(key.startswith("ram_core.ffm.fuse.") for key in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    if hasattr(model, "load_compatible_state_dict"):
        model.load_compatible_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()


def infer_batched(model, image_tensor: torch.Tensor, target_hw: tuple[int, int], *, use_amp: bool = True) -> np.ndarray:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    amp_enabled = bool(use_amp and DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(image_tensor).float()
    if tuple(pred.shape[-2:]) != tuple(target_hw):
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0]
    return pred[0].detach().cpu().numpy().astype(np.float32, copy=False)


def _as_batched_device_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(image_tensor.shape)}")
    return image_tensor.to(DEVICE, non_blocking=True).float()


def extract_dav2_patch_features(
    dav2_model,
    x_norm_padded: torch.Tensor,
    layers: list[int] | tuple[int, ...],
    *,
    bridge_injections: dict[int, torch.Tensor] | None = None,
    use_amp: bool = True,
) -> dict[int, np.ndarray]:
    """Return DINOv2 patch feature maps as CPU numpy arrays keyed by block index."""
    layer_tuple = tuple(int(layer) for layer in layers)
    amp_enabled = bool(use_amp and DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        outputs = dav2_model.pretrained.get_intermediate_layers(
            x_norm_padded,
            layer_tuple,
            reshape=True,
            return_class_token=True,
            bridge_injections=bridge_injections,
        )
    features: dict[int, np.ndarray] = {}
    for layer, output in zip(layer_tuple, outputs):
        patch_features = output[0] if isinstance(output, tuple) else output
        features[int(layer)] = patch_features[0].float().detach().cpu().numpy().astype(np.float32, copy=False)
    return features


def extract_rgb_teacher_patch_features(
    rgb_model,
    image_tensor: torch.Tensor,
    layers: list[int] | tuple[int, ...],
    *,
    use_amp: bool = True,
) -> dict[str, Any]:
    image_batch = _as_batched_device_tensor(image_tensor)
    x_norm_padded = rgb_model.spatial_adapter.pad_rgb(image_batch)
    features = extract_dav2_patch_features(
        rgb_model.dav2,
        x_norm_padded,
        layers,
        use_amp=use_amp,
    )
    return {
        "sensor_hw": [int(v) for v in image_batch.shape[-2:]],
        "padded_hw": [int(v) for v in x_norm_padded.shape[-2:]],
        "patch_hw": [
            int(x_norm_padded.shape[-2] // rgb_model.dav2.pretrained.patch_size),
            int(x_norm_padded.shape[-1] // rgb_model.dav2.pretrained.patch_size),
        ],
        "features": features,
    }


def prepare_raw_student_vit_input(
    raw_model,
    raw_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
    """Build the padded ImageNet-normalized tensor consumed by the student's DAv2.

    For bridge models this returns both the padded tensor and the bridge token
    injections. Non-bridge RAW models return ``None`` for injections.
    """
    raw_batch = _as_batched_device_tensor(raw_tensor)
    if raw_batch.shape[1] != 4:
        raise ValueError(f"Expected 4-channel RAW tensor, got {tuple(raw_batch.shape)}")

    if hasattr(raw_model, "build_bridge_injections"):
        return raw_model.build_bridge_injections(raw_batch)

    if hasattr(raw_model, "rgb_head"):
        x4 = raw_model.ram_core(raw_batch)
        x_rgb = raw_model.rgb_head(x4, x_raw=raw_batch)
    elif hasattr(raw_model, "residual_head"):
        x4 = raw_model.ram_core(raw_batch)
        delta_rgb = raw_model.residual_head(x4)
        x_rgb = torch.clamp(
            packed_bayer_to_base_rgb(raw_batch) + raw_model.residual_scale * torch.tanh(delta_rgb),
            min=0.0,
            max=1.0,
        )
    elif hasattr(raw_model, "ram_core") and hasattr(raw_model, "img_mean") and hasattr(raw_model, "img_std"):
        x_rgb_in = packed_bayer_to_base_rgb(raw_batch)
        x_rgb = torch.clamp(raw_model.ram_core(x_rgb_in), min=0.0, max=1.0)
    elif hasattr(raw_model, "spatial_adapter") and raw_batch.shape[1] == 4:
        x_rgb = packed_bayer_to_base_rgb(raw_batch)
    else:
        raise TypeError(f"Unsupported RAW student model for feature extraction: {type(raw_model).__name__}")

    x_norm = (x_rgb - raw_model.img_mean) / raw_model.img_std
    x_norm = raw_model.spatial_adapter.pad_rgb(x_norm)
    return x_norm, None


def extract_raw_student_patch_features(
    raw_model,
    raw_tensor: torch.Tensor,
    layers: list[int] | tuple[int, ...],
    *,
    use_bridge: bool = True,
    use_amp: bool = True,
) -> dict[str, Any]:
    with torch.no_grad():
        x_norm_padded, bridge_injections = prepare_raw_student_vit_input(raw_model, raw_tensor)
    if not use_bridge:
        bridge_injections = None
    features = extract_dav2_patch_features(
        raw_model.dav2,
        x_norm_padded,
        layers,
        bridge_injections=bridge_injections,
        use_amp=use_amp,
    )
    return {
        "sensor_hw": [int(v) for v in raw_tensor.shape[-2:]],
        "padded_hw": [int(v) for v in x_norm_padded.shape[-2:]],
        "patch_hw": [
            int(x_norm_padded.shape[-2] // raw_model.dav2.pretrained.patch_size),
            int(x_norm_padded.shape[-1] // raw_model.dav2.pretrained.patch_size),
        ],
        "bridge_injections_used": bool(bridge_injections),
        "features": features,
    }


def raw_tensor_to_preview(raw_chw: torch.Tensor | np.ndarray) -> np.ndarray:
    raw = raw_chw.detach().cpu().numpy() if isinstance(raw_chw, torch.Tensor) else np.asarray(raw_chw)
    if raw.ndim != 3:
        raise ValueError(f"Expected raw tensor with 3 dims, got {raw.shape}")
    if raw.shape[0] == 4:
        raw = np.transpose(raw, (1, 2, 0))
    if raw.shape[-1] != 4:
        raise ValueError(f"Expected 4-channel raw tensor, got {raw.shape}")
    preview = np.stack([raw[..., 0], 0.5 * (raw[..., 1] + raw[..., 2]), raw[..., 3]], axis=-1)
    preview = np.clip(preview, 0.0, 1.0) ** (1.0 / 2.2)
    return (preview * 255.0).round().astype(np.uint8)


def raw_tensor_to_pseudo_rgb_norm(raw_chw: torch.Tensor, *, channel_mode: str = "rgb_avg_g") -> torch.Tensor:
    raw = raw_chw.float()
    if raw.ndim != 3 or raw.shape[0] != 4:
        raise ValueError(f"Expected raw CHW tensor with shape (4,H,W), got {tuple(raw.shape)}")
    if channel_mode == "rgb_avg_g":
        rgb = torch.stack([raw[0], 0.5 * (raw[1] + raw[2]), raw[3]], dim=0)
    elif channel_mode == "rggb":
        rgb = raw[[0, 1, 3]]
    else:
        raise ValueError(f"Unsupported channel_mode: {channel_mode}")
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
    return (rgb.clamp(0.0, 1.0) - mean) / std


def denorm_rgb_tensor(image_chw: torch.Tensor | np.ndarray) -> np.ndarray:
    image = image_chw.detach().cpu().numpy() if isinstance(image_chw, torch.Tensor) else np.asarray(image_chw)
    if image.ndim != 3:
        raise ValueError(f"Expected image CHW/HWC tensor, got {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.float32, copy=False) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    value = np.clip(depth, vmin, vmax)
    value = np.where(np.isfinite(value), value, vmin)
    denom = max(float(vmax - vmin), 1e-6)
    rgb = cv2.applyColorMap(((value - vmin) / denom * 255.0).round().astype(np.uint8), cv2.COLORMAP_PLASMA)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~valid_mask] = 0
    return rgb


def colorize_error(error_map: np.ndarray, valid_mask: np.ndarray, *, vmax: float = 0.5) -> np.ndarray:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    clipped = np.clip(np.asarray(error_map, dtype=np.float32), 0.0, max(float(vmax), 1e-6))
    rgb = cv2.applyColorMap((clipped / max(float(vmax), 1e-6) * 255.0).round().astype(np.uint8), cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~valid_mask] = 0
    return rgb


def colorize_scalar(value: np.ndarray, valid_mask: np.ndarray | None = None, *, vmax_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    valid = np.isfinite(arr) if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & np.isfinite(arr))
    if valid.any():
        vmax = float(np.percentile(arr[valid], vmax_pct))
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)
    scaled = np.clip(arr / vmax, 0.0, 1.0)
    rgb = cv2.applyColorMap((scaled * 255.0).round().astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if valid_mask is not None:
        rgb[~np.asarray(valid_mask, dtype=bool)] = 0
    return rgb


def mask_to_rgb(mask: np.ndarray, *, dilate: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if dilate > 1:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = (255, 255, 255)
    return rgb


def resize_rgb(image: np.ndarray, hw: tuple[int, int], *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    h, w = hw
    if tuple(image.shape[:2]) == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=interpolation)


def choose_depth_vis_range(depth: np.ndarray, valid_mask: np.ndarray, *, min_depth: float, max_depth: float) -> tuple[float, float]:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if valid.any():
        vals = depth[valid]
        vmin = max(float(min_depth), float(np.percentile(vals, 1.0)))
        vmax = min(float(max_depth), float(np.percentile(vals, 99.0)))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            return vmin, vmax
    return float(min_depth), float(max_depth)


def relative_error(gt: np.ndarray, pred_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    err = np.zeros_like(gt, dtype=np.float32)
    valid = mask & np.isfinite(pred_depth) & (pred_depth > 0) & (gt > 0)
    err[valid] = np.abs(pred_depth[valid] - gt[valid]) / np.clip(gt[valid], a_min=1e-6, a_max=None)
    return err


def compute_metrics_or_nan(gt: np.ndarray, pred_depth: np.ndarray, mask: np.ndarray, *, min_depth: float, max_depth: float) -> dict[str, float]:
    metrics = compute_metrics(gt, pred_depth, mask, min_depth=min_depth, max_depth=max_depth)
    if metrics is None:
        return {key: float("nan") for key in (*METRIC_KEYS, "valid_eval_pixels")}
    return {key: float(metrics[key]) for key in metrics}


def summarize_metric_records(records: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    keys = sorted({key for record in records for key in record})
    for key in keys:
        vals = np.asarray([record[key] for record in records if np.isfinite(record.get(key, np.nan))], dtype=np.float64)
        out[key] = float(vals.mean()) if vals.size else float("nan")
    return out


def build_image_edge_band(rgb_image: np.ndarray, *, low: int = 80, high: int = 160, dilate: int = 5) -> np.ndarray:
    rgb_u8 = np.clip(rgb_image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high)) > 0
    if dilate > 1:
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1) > 0
    return edges


def depth_gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if finite.any():
        fill = float(np.median(depth[finite]))
    else:
        fill = 0.0
    dense = np.where(finite, depth, fill).astype(np.float32, copy=False)
    gy, gx = np.gradient(dense)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)


def masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not valid.any():
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    vals = np.asarray(values[valid], dtype=np.float64)
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "p50": float(np.percentile(vals, 50.0)),
        "p90": float(np.percentile(vals, 90.0)),
        "p99": float(np.percentile(vals, 99.0)),
    }


def align_disp_to_reference(source_disp: np.ndarray, ref_disp: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(source_disp) & np.isfinite(ref_disp)
    if int(valid.sum()) < 10:
        return np.full_like(source_disp, np.nan, dtype=np.float32), {"scale": float("nan"), "shift": float("nan")}
    x = source_disp[valid].reshape(-1, 1).astype(np.float64)
    y = ref_disp[valid].reshape(-1, 1).astype(np.float64)
    A = np.concatenate([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale, shift = float(coef[0].item()), float(coef[1].item())
    aligned = source_disp.astype(np.float64) * scale + shift
    return aligned.astype(np.float32, copy=False), {"scale": scale, "shift": shift}


def teacher_student_disagreement(
    student_disp: np.ndarray,
    teacher_disp: np.ndarray,
    *,
    threshold: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.isfinite(student_disp) & np.isfinite(teacher_disp)
    aligned_student, align_stats = align_disp_to_reference(student_disp, teacher_disp, finite)
    finite = finite & np.isfinite(aligned_student)
    if not finite.any():
        empty = np.zeros_like(student_disp, dtype=np.float32)
        return empty, {"area_ratio": float("nan"), "threshold": float(threshold), **align_stats}
    teacher_vals = teacher_disp[finite].astype(np.float64)
    med = float(np.median(teacher_vals))
    mad = float(np.median(np.abs(teacher_vals - med)))
    scale = max(1.4826 * mad, 1e-6)
    normalized = np.zeros_like(student_disp, dtype=np.float32)
    normalized[finite] = np.abs(aligned_student[finite] - teacher_disp[finite]) / scale
    ratio = float(np.mean(normalized[finite] > float(threshold)))
    return normalized, {"area_ratio": ratio, "threshold": float(threshold), "teacher_mad_scale": scale, **align_stats}


def build_labeled_grid(
    panels: list[tuple[str, np.ndarray]],
    *,
    cols: int = 3,
    tile_hw: tuple[int, int] | None = None,
    footer: str | None = None,
) -> Image.Image:
    if not panels:
        raise ValueError("No panels provided")
    if tile_hw is None:
        tile_hw = panels[0][1].shape[:2]
    tile_h, tile_w = tile_hw
    rows = int(np.ceil(len(panels) / float(cols)))
    header_h = 28
    footer_h = 24 if footer else 0
    canvas = Image.new("RGB", (tile_w * cols, (tile_h + header_h) * rows + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x0 = col * tile_w
        y0 = row * (tile_h + header_h)
        image = resize_rgb(np.asarray(image), (tile_h, tile_w))
        canvas.paste(Image.fromarray(image.astype(np.uint8)), (x0, y0 + header_h))
        draw.rectangle((x0, y0, x0 + tile_w, y0 + header_h), fill="white")
        draw.text((x0 + 8, y0 + 8), label, fill="black")
    if footer:
        draw.text((8, (tile_h + header_h) * rows + 5), footer, fill="black")
    return canvas
