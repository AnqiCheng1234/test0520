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
from finetune_stf.dataset.kitti_eval import DEFAULT_KITTI_BASE, DEFAULT_KITTI_VAL_SPLIT, KITTIEval
from finetune_stf.models.lora_bridge import merge_lora_in_state_dict
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
DEPTH_CMAP = colormaps["Spectral_r"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare KITTI RGB zero-shot DAv2 vs our RGB-only decoder path and save 2x3 panels."
    )
    parser.add_argument("exp_dir", type=Path, help="Experiment directory containing config.json.")
    parser.add_argument("--ours-checkpoint", default="last", help='Ours checkpoint: "last", "best", or custom path.')
    parser.add_argument("--dav2-checkpoint", default="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth")
    parser.add_argument("--kitti-base", default=DEFAULT_KITTI_BASE)
    parser.add_argument("--kitti-split", default=str(DEFAULT_KITTI_VAL_SPLIT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit KITTI dataset indices.")
    parser.add_argument("--num-samples", type=int, default=5, help="Used only when --indices is omitted.")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--error-max", type=float, default=0.5, help="Relative error clip for error maps.")
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
    if isinstance(state_obj, dict) and state_obj and all(str(key).startswith("module.") for key in state_obj):
        state_obj = {key[len("module."):]: value for key, value in state_obj.items()}
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


def build_rgb_wrapper_model(
    *,
    encoder: str,
    sensor_hw: tuple[int, int],
    checkpoint_path: Path,
    lora_alpha: float = 16.0,
    lora_rank: int = 8,
):
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model = build_dav2_padded_rgb_depth_model(
        dav2,
        sensor_hw=sensor_hw,
        backbone_hw=None,
    )
    state_dict = unwrap_model_state(torch.load(checkpoint_path, map_location="cpu"))
    if any(key.startswith("ram_core.ffm.fuse.") for key in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    if any(key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight") for key in state_dict):
        state_dict = merge_lora_in_state_dict(state_dict, alpha=lora_alpha, rank=lora_rank)
    if any(key.startswith("dav2.") or key.startswith("spatial_adapter.") for key in state_dict):
        status = model.load_compatible_state_dict(state_dict, strict=False)
        load_mode = "wrapper_partial"
    else:
        status = model.load_base_dav2_state_dict(state_dict)
        load_mode = "base_dav2"
    return model.to(DEVICE).eval(), status, load_mode


def infer_batched(model, image_tensor: torch.Tensor, target_hw: tuple[int, int], *, use_amp: bool = True) -> np.ndarray:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    amp_enabled = use_amp and DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(image_tensor).float()
    if tuple(pred.shape[-2:]) != tuple(target_hw):
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0]
    return pred[0].detach().cpu().numpy()


def aligned_disp_to_display_depth(
    pred_disp: np.ndarray,
    align_stats: dict[str, float],
    *,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Build a full-frame depth image for visualization only.

    Metrics still use the stricter aligned depth returned by affine_align_disp.
    For display, non-positive aligned disparity is clipped to the far plane so
    sky/no-GT regions are still rendered instead of turning black.
    """
    scale = float(align_stats["scale"])
    shift = float(align_stats["shift"])
    aligned_disp = pred_disp.astype(np.float64) * scale + shift
    min_disp = 1.0 / max(float(max_depth), 1e-6)
    max_disp = 1.0 / max(float(min_depth), 1e-6)
    clipped_disp = np.clip(aligned_disp, min_disp, max_disp)
    clipped_disp = np.where(np.isfinite(clipped_disp), clipped_disp, min_disp)
    return (1.0 / clipped_disp).astype(np.float32)


def _background_array(background: Image.Image | np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if background is None:
        return None
    h, w = shape_hw
    if isinstance(background, Image.Image):
        if background.size != (w, h):
            background = background.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(background.convert("RGB"), dtype=np.uint8)
    bg = np.asarray(background)
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    if bg.ndim == 2:
        bg = np.repeat(bg[..., None], 3, axis=-1)
    return bg[..., :3].astype(np.uint8, copy=False)


def colorize_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    vmin: float,
    vmax: float,
    background: Image.Image | np.ndarray | None = None,
    invalid_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    value = np.clip(depth, vmin, vmax)
    value = np.where(np.isfinite(value), value, vmin)
    denom = max(vmax - vmin, 1e-6)
    norm = (value - vmin) / denom
    rgb = (DEPTH_CMAP(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    if valid_mask is not None:
        bg = _background_array(background, depth.shape)
        invalid = ~valid_mask
        if bg is not None:
            rgb[invalid] = bg[invalid]
        elif invalid_color is not None:
            rgb[invalid] = invalid_color
    return Image.fromarray(rgb)


def colorize_error(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_rel_error: float,
    background: Image.Image | np.ndarray | None = None,
    invalid_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    error_map = np.asarray(error_map, dtype=np.float32)
    clipped = np.clip(error_map, 0.0, max(float(max_rel_error), 1e-6))
    scaled = np.round(clipped / max(float(max_rel_error), 1e-6) * 255.0).astype(np.uint8)
    rgb = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    invalid = ~valid_mask
    bg = _background_array(background, error_map.shape)
    if bg is not None:
        rgb[invalid] = bg[invalid]
    elif invalid_color is not None:
        rgb[invalid] = invalid_color
    return Image.fromarray(rgb)


def make_indices(dataset_size: int, num_samples: int) -> list[int]:
    num_samples = max(1, min(int(num_samples), int(dataset_size)))
    return np.linspace(0, dataset_size - 1, num_samples, dtype=int).tolist()


def build_compare_panel(
    *,
    rgb_image: Image.Image,
    gt_image: Image.Image,
    dav2_pred: Image.Image,
    ours_pred: Image.Image,
    dav2_error: Image.Image,
    ours_error: Image.Image,
    sample_name: str,
    dav2_metrics: dict[str, float],
    ours_metrics: dict[str, float],
) -> Image.Image:
    panel_w, panel_h = rgb_image.size
    cols = 3
    rows = 2
    header_h = 64
    footer_h = 28
    canvas = Image.new("RGB", (panel_w * cols, panel_h * rows + header_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)

    titles = [
        ("rgb", "dav2_pred", "dav2_error"),
        ("gt_depth", "ours_pred", "ours_error"),
    ]
    images = [
        [rgb_image, dav2_pred, dav2_error],
        [gt_image, ours_pred, ours_error],
    ]
    for row_idx in range(rows):
        for col_idx in range(cols):
            x0 = col_idx * panel_w
            y0 = header_h + row_idx * panel_h
            canvas.paste(images[row_idx][col_idx], (x0, y0))
            draw.rectangle((x0 + 8, y0 + 8, x0 + 188, y0 + 34), fill=(0, 0, 0))
            draw.text((x0 + 16, y0 + 14), titles[row_idx][col_idx], fill="white")

    canvas_np = np.array(canvas)
    cv2.putText(
        canvas_np,
        sample_name,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_np,
        f"dav2 abs_rel={dav2_metrics['abs_rel']:.4f} d1={dav2_metrics['d1']:.4f}",
        (panel_w + 16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_np,
        f"ours abs_rel={ours_metrics['abs_rel']:.4f} d1={ours_metrics['d1']:.4f}",
        (panel_w + 16, header_h + panel_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_np,
        "KITTI invalid/no-GT regions are black; metrics use valid GT only",
        (16, header_h + panel_h * rows + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return Image.fromarray(canvas_np)


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    config_path = exp_dir / "config.json"
    ours_checkpoint = resolve_checkpoint(exp_dir, args.ours_checkpoint).expanduser().resolve()
    dav2_checkpoint = Path(args.dav2_checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not ours_checkpoint.is_file():
        raise FileNotFoundError(f"Missing ours checkpoint: {ours_checkpoint}")
    if not dav2_checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAv2 checkpoint: {dav2_checkpoint}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    sensor_hw = (cfg.get("input_height", 512), cfg.get("input_width", 960))
    dataset = KITTIEval(
        filelist_path=args.kitti_split,
        kitti_base=args.kitti_base,
        size=sensor_hw,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        input_type="rgb",
    )
    indices = list(args.indices) if args.indices else make_indices(len(dataset), args.num_samples)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else exp_dir / "kitti_rgb_compare_dav2_vs_ours_2x3"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dav2_model, dav2_status, dav2_load_mode = build_rgb_wrapper_model(
        encoder=cfg["encoder"],
        sensor_hw=sensor_hw,
        checkpoint_path=dav2_checkpoint,
        lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        lora_rank=int(cfg.get("lora_rank", 8)),
    )
    ours_model, ours_status, ours_load_mode = build_rgb_wrapper_model(
        encoder=cfg["encoder"],
        sensor_hw=sensor_hw,
        checkpoint_path=ours_checkpoint,
        lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        lora_rank=int(cfg.get("lora_rank", 8)),
    )

    manifest = []
    print(
        f"[KITTI][COMPARE] samples={len(indices)} "
        f"dav2_load={dav2_load_mode} ours_load={ours_load_mode}",
        flush=True,
    )
    for order_idx, dataset_idx in enumerate(indices, start=1):
        sample = dataset[dataset_idx]
        depth = sample["depth"].numpy()
        valid = sample["valid_mask"].numpy().astype(bool)
        sample_name = sample["sample_name"]

        dav2_disp = infer_batched(dav2_model, sample["image"], depth.shape[-2:])
        ours_disp = infer_batched(ours_model, sample["image"], depth.shape[-2:])
        dav2_aligned, dav2_align_stats = affine_align_disp(depth, dav2_disp, valid)
        ours_aligned, ours_align_stats = affine_align_disp(depth, ours_disp, valid)

        dav2_metrics = compute_metrics(depth, dav2_aligned, valid, min_depth=args.min_depth, max_depth=args.max_depth)
        ours_metrics = compute_metrics(depth, ours_aligned, valid, min_depth=args.min_depth, max_depth=args.max_depth)
        if dav2_metrics is None or ours_metrics is None:
            continue

        row = dataset.rows[dataset_idx]
        rgb_image = Image.open(row["image_path"]).convert("RGB")
        display_size = rgb_image.size
        if display_size != (depth.shape[1], depth.shape[0]):
            rgb_image = rgb_image.resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR)
            display_size = rgb_image.size

        valid_depth = depth[valid]
        vmin = max(float(args.min_depth), float(valid_depth.min()))
        vmax = min(float(args.max_depth), float(valid_depth.max()))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = float(args.min_depth), float(args.max_depth)

        dav2_valid = np.isfinite(dav2_aligned) & (dav2_aligned > 0.0)
        ours_valid = np.isfinite(ours_aligned) & (ours_aligned > 0.0)
        dav2_eval = valid & dav2_valid & np.isfinite(depth) & (depth > 0.0)
        ours_eval = valid & ours_valid & np.isfinite(depth) & (depth > 0.0)

        dav2_err = np.zeros_like(depth, dtype=np.float32)
        ours_err = np.zeros_like(depth, dtype=np.float32)
        dav2_err[dav2_eval] = np.abs(dav2_aligned[dav2_eval] - depth[dav2_eval]) / np.clip(depth[dav2_eval], 1e-6, None)
        ours_err[ours_eval] = np.abs(ours_aligned[ours_eval] - depth[ours_eval]) / np.clip(depth[ours_eval], 1e-6, None)

        dav2_display_depth = aligned_disp_to_display_depth(
            dav2_disp,
            dav2_align_stats,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        ours_display_depth = aligned_disp_to_display_depth(
            ours_disp,
            ours_align_stats,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

        gt_image = colorize_depth(depth, valid, vmin=vmin, vmax=vmax, invalid_color=(0, 0, 0)).resize(
            display_size, Image.Resampling.BILINEAR
        )
        dav2_pred = colorize_depth(dav2_display_depth, None, vmin=vmin, vmax=vmax).resize(
            display_size, Image.Resampling.BILINEAR
        )
        ours_pred = colorize_depth(ours_display_depth, None, vmin=vmin, vmax=vmax).resize(
            display_size, Image.Resampling.BILINEAR
        )
        dav2_error = colorize_error(dav2_err, dav2_eval, max_rel_error=args.error_max, invalid_color=(0, 0, 0)).resize(
            display_size, Image.Resampling.BILINEAR
        )
        ours_error = colorize_error(ours_err, ours_eval, max_rel_error=args.error_max, invalid_color=(0, 0, 0)).resize(
            display_size, Image.Resampling.BILINEAR
        )

        panel = build_compare_panel(
            rgb_image=rgb_image,
            gt_image=gt_image,
            dav2_pred=dav2_pred,
            ours_pred=ours_pred,
            dav2_error=dav2_error,
            ours_error=ours_error,
            sample_name=f"idx={dataset_idx} | {sample_name}",
            dav2_metrics=dav2_metrics,
            ours_metrics=ours_metrics,
        )
        panel_path = output_dir / f"{order_idx:04d}_{sample_name}_compare_2x3.jpg"
        panel.save(panel_path, quality=95)
        manifest.append(
            {
                "dataset_index": dataset_idx,
                "sample_name": sample_name,
                "image_path": str(row["image_path"]),
                "depth_path": str(row["depth_path"]),
                "panel_path": str(panel_path),
                "dav2_abs_rel": float(dav2_metrics["abs_rel"]),
                "dav2_d1": float(dav2_metrics["d1"]),
                "ours_abs_rel": float(ours_metrics["abs_rel"]),
                "ours_d1": float(ours_metrics["d1"]),
            }
        )
        print(f"[KITTI][COMPARE] saved {order_idx}/{len(indices)} idx={dataset_idx} {sample_name}", flush=True)

    summary = {
        "exp_dir": str(exp_dir),
        "ours_checkpoint": str(ours_checkpoint),
        "dav2_checkpoint": str(dav2_checkpoint),
        "kitti_base": str(Path(args.kitti_base).expanduser().resolve()),
        "kitti_split": str(Path(args.kitti_split).expanduser().resolve()),
        "indices": indices,
        "output_dir": str(output_dir),
        "ours_load_mode": ours_load_mode,
        "dav2_load_mode": dav2_load_mode,
        "ours_missing_keys_count": len(ours_status.missing_keys),
        "ours_unexpected_keys_count": len(ours_status.unexpected_keys),
        "dav2_missing_keys_count": len(dav2_status.missing_keys),
        "dav2_unexpected_keys_count": len(dav2_status.unexpected_keys),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
