#!/usr/bin/env python3
"""
Compare a trained raw-like STF checkpoint against a no-train base_rgb control:

    base_rgb = [R, (Gr + Gb)/2, B]
    base_rgb_control = base_rgb -> frozen ImageNet-pretrained DAv2

This is a lower-bound control for Appendix-A step 4. It does not replace a
trained residual-head experiment, but it helps answer whether preserving the
base Bayer contrast is directionally helpful before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.dump_stf_rgb_triplets import (
    DEVICE,
    load_manifest_rows,
    load_model,
    make_indices,
    resolve_checkpoint,
    rgb_tensor_to_pil,
    unwrap_model_state,
)
from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.dataset.raw_utils import load_rectified_bayer_npz, normalize_raw_4ch
from finetune_stf.models.lora_bridge import RAW_RAM_BRIDGE_INPUT_TYPES
from finetune_stf.models.raw_feature_adapter import RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES
from finetune_stf.models.raw_ram import IMAGENET_MEAN, IMAGENET_STD, packed_bayer_to_base_rgb
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model

RAW_PACKED_INPUT_TYPES = ("raw_packed",)
SUPPORTED_INPUT_TYPES = (
    *RAW_PACKED_INPUT_TYPES,
    "raw_ram",
    "raw_ram_residual",
    *RAW_RAM_BRIDGE_INPUT_TYPES,
    *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
)
VALID_SPLITS = ("train", "val", "test")
DISPLAY_HW = (512, 960)
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare STF checkpoint vs base_rgb passthrough control.")
    parser.add_argument("exp_dir", type=Path, help="Path to an experiment directory containing config.json.")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help='Checkpoint to load: "best", "latest", or a custom .pth path. Default: best',
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=VALID_SPLITS,
        help="Dataset split to analyze. Default: val",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many evenly spaced samples to use. Default: 10",
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


def resolve_output_dir(exp_dir: Path, split: str, checkpoint: str, output_subdir: str | None) -> Path:
    if output_subdir:
        return exp_dir / output_subdir
    ckpt_name = Path(checkpoint).stem if checkpoint not in {"best", "latest"} else checkpoint
    return exp_dir / f"diagnostic_base_rgb_control_{split}_{ckpt_name}"


def load_depth_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.array(data["arr_0"], dtype=np.float32, copy=True)


def make_raw_tensor(raw_npz_path: Path, *, norm_mode: str) -> torch.Tensor:
    bayer_rect = load_rectified_bayer_npz(raw_npz_path)
    bayer_4ch_norm = normalize_raw_4ch(bayer_rect, norm_mode=norm_mode)
    chw = np.ascontiguousarray(bayer_4ch_norm.transpose(2, 0, 1)).astype(np.float32, copy=False)
    return torch.frombuffer(chw.tobytes(), dtype=torch.float32).view(chw.shape).unsqueeze(0).to(DEVICE)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().contiguous()
    shape = tuple(x.shape)
    return np.array(x.view(-1).tolist(), dtype=np.float32).reshape(shape)


def build_rgb_reference_model(pretrained_from: str, encoder: str):
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model = build_dav2_padded_rgb_depth_model(dav2)
    state_dict = unwrap_model_state(torch.load(pretrained_from, map_location="cpu"))
    model.load_base_dav2_state_dict(state_dict)
    return model.to(DEVICE).eval()


def infer_current_model(model, x_raw: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    with torch.no_grad():
        pred_disp = model(x_raw)
        if pred_disp.ndim == 4:
            pred_disp = pred_disp[:, 0]
        pred_disp = F.interpolate(
            pred_disp[:, None],
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    return tensor_to_numpy(pred_disp)


def infer_base_rgb_control(rgb_model, x_raw: torch.Tensor, target_hw: tuple[int, int]) -> tuple[np.ndarray, torch.Tensor]:
    target_h, target_w = target_hw
    img_mean = torch.tensor(IMAGENET_MEAN, device=x_raw.device, dtype=x_raw.dtype).view(1, 3, 1, 1)
    img_std = torch.tensor(IMAGENET_STD, device=x_raw.device, dtype=x_raw.dtype).view(1, 3, 1, 1)
    with torch.no_grad():
        base_rgb = packed_bayer_to_base_rgb(x_raw)
        base_rgb_norm = (base_rgb - img_mean) / img_std
        pred_disp = rgb_model(base_rgb_norm)
        if pred_disp.ndim == 4:
            pred_disp = pred_disp[:, 0]
        pred_disp = F.interpolate(
            pred_disp[:, None],
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    return tensor_to_numpy(pred_disp), base_rgb


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_with_label(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), label, fill="black")
    return canvas


def build_panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    labeled = [image_with_label(img, label) for img, label in zip(images, labels)]
    width = sum(img.width for img in labeled)
    height = max(img.height for img in labeled)
    canvas = Image.new("RGB", (width, height), "white")
    x_offset = 0
    for image in labeled:
        canvas.paste(image, (x_offset, 0))
        x_offset += image.width
    return canvas


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    value = np.asarray(depth, dtype=np.float32)
    value = np.nan_to_num(value, nan=vmax, posinf=vmax, neginf=vmin)
    value = np.clip(value, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = ((value - vmin) / denom * 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_PLASMA)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_rgb)


def colorize_diff(diff: np.ndarray, vmax: float) -> Image.Image:
    value = np.asarray(diff, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=vmax, neginf=0.0)
    value = np.clip(value, 0.0, vmax)
    denom = max(vmax, 1e-6)
    norm = (value / denom * 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_rgb)


def aggregate_metric_dicts(records: list[dict], prefix: str) -> dict[str, float]:
    out = {}
    for key in ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "d1", "d2", "d3"):
        out[key] = float(np.mean([float(rec[f"{prefix}_{key}"]) for rec in records]))
    return out


def group_records(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = record.get("daytime") or "unknown"
        groups.setdefault(key, []).append(record)
    return groups


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"abs_rel={metrics['abs_rel']:.4f} rmse={metrics['rmse']:.4f} "
        f"silog={metrics['silog']:.4f} d1={metrics['d1']:.4f} "
        f"d2={metrics['d2']:.4f} d3={metrics['d3']:.4f}"
    )


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
        raise ValueError(f"This diagnostic only supports raw-like experiments, got input_type={input_type}")

    rows = load_manifest_rows(
        cfg.get("stf_root", "/home/caq/6666_raw/seeingthroughfog"),
        args.split,
        cfg.get("raw_npz_root", "/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz"),
    )
    indices = make_indices(len(rows), args.num_samples, args.indices)
    norm_mode = cfg.get("norm_mode", "companded")
    min_depth = float(cfg.get("min_depth", 1.0))
    max_depth = float(cfg.get("max_depth", 80.0))

    current_model = load_model(cfg, str(checkpoint_path))
    rgb_reference_model = build_rgb_reference_model(cfg["pretrained_from"], cfg["encoder"])

    for old_file in output_dir.glob("panel_*.jpg"):
        old_file.unlink()

    per_sample = []
    sample_lines = ["index\tsample_name\tdaytime\traw_npz_path"]
    for idx in indices:
        row = rows[idx]
        gt_depth = load_depth_npz(row["depth_path"])
        valid_mask = np.isfinite(gt_depth) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
        if valid_mask.sum() < 10:
            continue

        target_hw = tuple(int(v) for v in gt_depth.shape)
        x_raw = make_raw_tensor(row["raw_npz_path"], norm_mode=norm_mode)
        pred_current = infer_current_model(current_model, x_raw, target_hw)
        pred_base, base_rgb = infer_base_rgb_control(rgb_reference_model, x_raw, target_hw)

        aligned_current, align_current_stats = affine_align_disp(gt_depth, pred_current, valid_mask=valid_mask)
        aligned_base, align_base_stats = affine_align_disp(gt_depth, pred_base, valid_mask=valid_mask)

        metrics_current = compute_metrics(
            gt_depth,
            aligned_current,
            valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        metrics_base = compute_metrics(
            gt_depth,
            aligned_base,
            valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if metrics_current is None or metrics_base is None:
            continue

        diff_aligned = np.abs(aligned_current - aligned_base)
        diff_aligned[~np.isfinite(diff_aligned)] = 0.0
        diff_vmax = float(np.percentile(diff_aligned[valid_mask], 99)) if np.any(valid_mask) else 1.0
        diff_vmax = max(diff_vmax, 1e-3)

        valid_gt = gt_depth[valid_mask]
        vmin = float(np.percentile(valid_gt, 5))
        vmax = float(np.percentile(valid_gt, 95))
        vmin = max(vmin, min_depth)
        vmax = min(max(vmax, vmin + 1e-3), max_depth)

        real_rgb = open_rgb(row["rgb_path"]).resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)
        base_rgb_vis = rgb_tensor_to_pil(base_rgb, target_hw=target_hw)
        panel = build_panel(
            [
                real_rgb,
                base_rgb_vis,
                colorize_depth(aligned_current, vmin=vmin, vmax=vmax),
                colorize_depth(aligned_base, vmin=vmin, vmax=vmax),
                colorize_diff(diff_aligned, vmax=diff_vmax),
            ],
            [
                f"real_rgb [{row.get('daytime') or 'unknown'}]",
                "base_rgb",
                f"current abs_rel={metrics_current['abs_rel']:.3f}",
                f"base_ctrl abs_rel={metrics_base['abs_rel']:.3f}",
                "abs diff (aligned depth)",
            ],
        )
        panel.save(output_dir / f"panel_{idx:04d}.jpg", quality=95)

        record = {
            "index": int(idx),
            "sample_name": row["sample_name"],
            "daytime": row.get("daytime") or "unknown",
            "current_abs_rel": float(metrics_current["abs_rel"]),
            "current_sq_rel": float(metrics_current["sq_rel"]),
            "current_rmse": float(metrics_current["rmse"]),
            "current_rmse_log": float(metrics_current["rmse_log"]),
            "current_log10": float(metrics_current["log10"]),
            "current_silog": float(metrics_current["silog"]),
            "current_d1": float(metrics_current["d1"]),
            "current_d2": float(metrics_current["d2"]),
            "current_d3": float(metrics_current["d3"]),
            "base_abs_rel": float(metrics_base["abs_rel"]),
            "base_sq_rel": float(metrics_base["sq_rel"]),
            "base_rmse": float(metrics_base["rmse"]),
            "base_rmse_log": float(metrics_base["rmse_log"]),
            "base_log10": float(metrics_base["log10"]),
            "base_silog": float(metrics_base["silog"]),
            "base_d1": float(metrics_base["d1"]),
            "base_d2": float(metrics_base["d2"]),
            "base_d3": float(metrics_base["d3"]),
            "delta_abs_rel": float(metrics_base["abs_rel"] - metrics_current["abs_rel"]),
            "delta_rmse": float(metrics_base["rmse"] - metrics_current["rmse"]),
            "delta_silog": float(metrics_base["silog"] - metrics_current["silog"]),
            "delta_d1": float(metrics_base["d1"] - metrics_current["d1"]),
            "valid_pixels": int(valid_mask.sum()),
            "current_invalid_align_ratio": float(align_current_stats["invalid_aligned_ratio"]),
            "base_invalid_align_ratio": float(align_base_stats["invalid_aligned_ratio"]),
        }
        per_sample.append(record)
        sample_lines.append(f"{idx}\t{row['sample_name']}\t{row.get('daytime') or 'unknown'}\t{row['raw_npz_path']}")

    if not per_sample:
        raise RuntimeError("No valid samples were processed.")

    current_mean = aggregate_metric_dicts(per_sample, "current")
    base_mean = aggregate_metric_dicts(per_sample, "base")
    delta_mean = {
        "abs_rel": float(base_mean["abs_rel"] - current_mean["abs_rel"]),
        "rmse": float(base_mean["rmse"] - current_mean["rmse"]),
        "silog": float(base_mean["silog"] - current_mean["silog"]),
        "d1": float(base_mean["d1"] - current_mean["d1"]),
        "d2": float(base_mean["d2"] - current_mean["d2"]),
        "d3": float(base_mean["d3"] - current_mean["d3"]),
    }

    group_summary = {}
    for group_name, records in group_records(per_sample).items():
        group_current = aggregate_metric_dicts(records, "current")
        group_base = aggregate_metric_dicts(records, "base")
        group_summary[group_name] = {
            "count": len(records),
            "current": group_current,
            "base": group_base,
            "delta": {
                "abs_rel": float(group_base["abs_rel"] - group_current["abs_rel"]),
                "rmse": float(group_base["rmse"] - group_current["rmse"]),
                "silog": float(group_base["silog"] - group_current["silog"]),
                "d1": float(group_base["d1"] - group_current["d1"]),
            },
        }

    summary = {
        "exp_dir": str(exp_dir),
        "checkpoint": str(checkpoint_path),
        "input_type": input_type,
        "pretrained_from": cfg["pretrained_from"],
        "split": args.split,
        "indices": indices,
        "current_mean": current_mean,
        "base_rgb_control_mean": base_mean,
        "delta_base_minus_current": delta_mean,
        "group_summary": group_summary,
        "per_sample": per_sample,
    }

    summary_lines = [
        f"exp_dir: {exp_dir}",
        f"checkpoint: {checkpoint_path}",
        f"input_type: {input_type}",
        f"pretrained_from: {cfg['pretrained_from']}",
        f"split: {args.split}",
        f"indices: {indices}",
        "",
        f"current_mean: {format_metrics(current_mean)}",
        f"base_rgb_control_mean: {format_metrics(base_mean)}",
        (
            "delta_base_minus_current: "
            f"abs_rel={delta_mean['abs_rel']:+.4f} rmse={delta_mean['rmse']:+.4f} "
            f"silog={delta_mean['silog']:+.4f} d1={delta_mean['d1']:+.4f} "
            f"d2={delta_mean['d2']:+.4f} d3={delta_mean['d3']:+.4f}"
        ),
        "",
        "group_summary:",
    ]
    for group_name, payload in sorted(group_summary.items()):
        summary_lines.append(
            f"  {group_name} (n={payload['count']}): current[{format_metrics(payload['current'])}] "
            f"base[{format_metrics(payload['base'])}] "
            f"delta_abs_rel={payload['delta']['abs_rel']:+.4f} delta_d1={payload['delta']['d1']:+.4f}"
        )

    summary_lines.append("")
    summary_lines.append("per_sample:")
    for record in per_sample:
        summary_lines.append(
            f"  idx={record['index']:04d} {record['daytime']:<9} "
            f"current_abs_rel={record['current_abs_rel']:.4f} base_abs_rel={record['base_abs_rel']:.4f} "
            f"delta_abs_rel={record['delta_abs_rel']:+.4f} delta_d1={record['delta_d1']:+.4f}"
        )

    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    print(f"Saved base_rgb control comparison for {len(per_sample)} samples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
