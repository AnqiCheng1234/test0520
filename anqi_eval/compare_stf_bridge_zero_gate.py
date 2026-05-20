#!/usr/bin/env python3
"""
Compare STF predictions from a raw_ram_bridge checkpoint before/after zeroing
bridge gates.

Outputs:
    - per-sample panels: real_rgb | orig_depth | zero_gate_depth | abs_diff
    - per-sample CSV-like text summary
    - aggregate summary.txt / summary.json
"""

from __future__ import annotations

import argparse
import json
import math
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
    DEFAULT_RAW_NPZ_ROOT,
    DEFAULT_STF_ROOT,
    DEVICE,
    load_manifest_rows,
    load_model,
    make_indices,
    resolve_checkpoint,
)
from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from finetune_stf.dataset.raw_utils import load_rectified_bayer_npz, normalize_raw_4ch
from finetune_stf.models.lora_bridge import RAW_RAM_BRIDGE_INPUT_TYPES

VALID_SPLITS = ("train", "val", "test")
DISPLAY_HW = (512, 960)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare STF bridge orig vs zero-gate predictions.")
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
    return exp_dir / f"diagnostic_bridge_zero_gate_{split}_{ckpt_name}"


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


def infer_raw_ram_bridge(model, x_raw: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    with torch.no_grad():
        x4, feature_dict = model.ram_core.forward_with_features(x_raw)
        x_rgb = model.rgb_head(x4)
        x_norm = (x_rgb - model.img_mean) / model.img_std
        x_norm = model.spatial_adapter.pad_rgb(x_norm)
        patch_hw = (
            x_norm.shape[-2] // model.dav2.pretrained.patch_size,
            x_norm.shape[-1] // model.dav2.pretrained.patch_size,
        )
        bridge_injections = model.bridge_adapter(feature_dict, patch_hw=patch_hw)
        pred_disp = model.dav2(x_norm, bridge_injections=bridge_injections)
        pred_disp = model.spatial_adapter.crop_depth(pred_disp)
        pred_disp = F.interpolate(
            pred_disp[:, None],
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
    return tensor_to_numpy(pred_disp)


def snapshot_gate_state(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.bridge_adapter.gates.items()}


def restore_gate_state(model, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.bridge_adapter.gates.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def zero_gate_state(model) -> None:
    with torch.no_grad():
        for param in model.bridge_adapter.gates.values():
            param.zero_()


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
    x = 0
    for img in labeled:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


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


def aggregate_metric_dicts(records: list[dict]) -> dict[str, float]:
    keys = [k for k in records[0].keys() if isinstance(records[0][k], (int, float))]
    out = {}
    for key in keys:
        out[key] = float(np.mean([float(rec[key]) for rec in records]))
    return out


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
    if input_type not in RAW_RAM_BRIDGE_INPUT_TYPES:
        raise ValueError(f"This diagnostic only supports raw_ram_bridge models, got input_type={input_type}")

    stf_root = cfg.get("stf_root", DEFAULT_STF_ROOT)
    raw_npz_root = cfg.get("raw_npz_root", DEFAULT_RAW_NPZ_ROOT)
    norm_mode = cfg.get("norm_mode", "companded")
    min_depth = float(cfg.get("min_depth", 1.0))
    max_depth = float(cfg.get("max_depth", 80.0))

    print(f"Loading model from {checkpoint_path} on {DEVICE} ...", flush=True)
    model = load_model(cfg, str(checkpoint_path))
    gate_state = snapshot_gate_state(model)
    gate_values = {name: float(value.item()) for name, value in gate_state.items()}
    gate_tanh = {name: float(math.tanh(val)) for name, val in gate_values.items()}

    rows = load_manifest_rows(stf_root, args.split, raw_npz_root)
    indices = make_indices(len(rows), args.num_samples, args.indices)

    sample_lines = [
        "index\tsample_name\tdaytime\torig_abs_rel\tzero_abs_rel\tdelta_abs_rel\torig_d1\tzero_d1\tdelta_d1\tpred_l1\taligned_l1_valid"
    ]
    orig_metrics_all = []
    zero_metrics_all = []
    delta_rows = []

    for idx in indices:
        row = rows[idx]
        raw_npz_path = row["raw_npz_path"]
        rgb_path = row["rgb_path"]
        depth_path = row["depth_path"]

        if not raw_npz_path.is_file():
            raise FileNotFoundError(f"Missing raw npz: {raw_npz_path}")
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing rgb path: {rgb_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing depth path: {depth_path}")

        gt_depth = load_depth_npz(depth_path)
        valid_mask = np.isfinite(gt_depth) & (gt_depth >= min_depth) & (gt_depth <= max_depth)
        if int(valid_mask.sum()) < 10:
            print(f"  [warn] idx={idx} {row['sample_name']}: too few valid GT pixels, skipping", flush=True)
            continue

        x_raw = make_raw_tensor(raw_npz_path, norm_mode=norm_mode)

        restore_gate_state(model, gate_state)
        pred_orig = infer_raw_ram_bridge(model, x_raw, gt_depth.shape)
        zero_gate_state(model)
        pred_zero = infer_raw_ram_bridge(model, x_raw, gt_depth.shape)
        restore_gate_state(model, gate_state)

        aligned_orig, _ = affine_align_disp(gt_depth, pred_orig, valid_mask)
        aligned_zero, _ = affine_align_disp(gt_depth, pred_zero, valid_mask)
        metrics_orig = compute_metrics(gt_depth, aligned_orig, valid_mask, min_depth=min_depth, max_depth=max_depth)
        metrics_zero = compute_metrics(gt_depth, aligned_zero, valid_mask, min_depth=min_depth, max_depth=max_depth)
        if metrics_orig is None or metrics_zero is None:
            print(f"  [warn] idx={idx} {row['sample_name']}: no valid metrics after alignment, skipping", flush=True)
            continue

        pred_l1 = float(np.mean(np.abs(pred_orig - pred_zero)))
        both_valid = valid_mask & np.isfinite(aligned_orig) & np.isfinite(aligned_zero)
        aligned_l1_valid = float(np.mean(np.abs(aligned_orig[both_valid] - aligned_zero[both_valid]))) if both_valid.any() else float("nan")

        orig_metrics_all.append(metrics_orig)
        zero_metrics_all.append(metrics_zero)
        delta = {
            "idx": idx,
            "sample_name": row["sample_name"],
            "daytime": row.get("daytime", ""),
            "pred_l1": pred_l1,
            "aligned_l1_valid": aligned_l1_valid,
            "delta_abs_rel": float(metrics_zero["abs_rel"] - metrics_orig["abs_rel"]),
            "delta_d1": float(metrics_zero["d1"] - metrics_orig["d1"]),
            "orig_abs_rel": float(metrics_orig["abs_rel"]),
            "zero_abs_rel": float(metrics_zero["abs_rel"]),
            "orig_d1": float(metrics_orig["d1"]),
            "zero_d1": float(metrics_zero["d1"]),
        }
        delta_rows.append(delta)

        valid_gt = gt_depth[valid_mask]
        vmin = max(min_depth, float(valid_gt.min()))
        vmax = min(max_depth, float(valid_gt.max()))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = min_depth, max_depth

        diff_map = np.abs(aligned_orig - aligned_zero)
        diff_valid = diff_map[both_valid]
        diff_vmax = float(np.quantile(diff_valid, 0.99)) if diff_valid.size else 1.0
        diff_vmax = max(diff_vmax, 1e-6)

        real_rgb = open_rgb(rgb_path).resize((DISPLAY_HW[1], DISPLAY_HW[0]), resample=Image.Resampling.BILINEAR)
        orig_pil = colorize_depth(aligned_orig, vmin, vmax).resize((DISPLAY_HW[1], DISPLAY_HW[0]), resample=Image.Resampling.BILINEAR)
        zero_pil = colorize_depth(aligned_zero, vmin, vmax).resize((DISPLAY_HW[1], DISPLAY_HW[0]), resample=Image.Resampling.BILINEAR)
        diff_pil = colorize_diff(diff_map, diff_vmax).resize((DISPLAY_HW[1], DISPLAY_HW[0]), resample=Image.Resampling.BILINEAR)

        panel = build_panel(
            [real_rgb, orig_pil, zero_pil, diff_pil],
            [
                f"real_rgb ({row.get('daytime','')})",
                f"orig depth abs_rel={metrics_orig['abs_rel']:.3f} d1={metrics_orig['d1']:.3f}",
                f"zero-gate depth abs_rel={metrics_zero['abs_rel']:.3f} d1={metrics_zero['d1']:.3f}",
                f"|orig-zero| p99={diff_vmax:.2f}m",
            ],
        )
        panel.save(output_dir / f"panel_{idx:04d}.jpg", quality=95)

        sample_lines.append(
            f"{idx}\t{row['sample_name']}\t{row.get('daytime','')}\t"
            f"{metrics_orig['abs_rel']:.6f}\t{metrics_zero['abs_rel']:.6f}\t{delta['delta_abs_rel']:+.6f}\t"
            f"{metrics_orig['d1']:.6f}\t{metrics_zero['d1']:.6f}\t{delta['delta_d1']:+.6f}\t"
            f"{pred_l1:.6f}\t{aligned_l1_valid:.6f}"
        )
        print(
            f"  [{idx}] {row['sample_name']} daytime={row.get('daytime','')} "
            f"delta_abs_rel={delta['delta_abs_rel']:+.4f} delta_d1={delta['delta_d1']:+.4f} "
            f"pred_l1={pred_l1:.5f}",
            flush=True,
        )

    if not orig_metrics_all:
        raise RuntimeError("No valid samples processed.")

    orig_summary = aggregate_metric_dicts(orig_metrics_all)
    zero_summary = aggregate_metric_dicts(zero_metrics_all)
    mean_pred_l1 = float(np.mean([row["pred_l1"] for row in delta_rows]))
    mean_aligned_l1 = float(np.mean([row["aligned_l1_valid"] for row in delta_rows if math.isfinite(row["aligned_l1_valid"])]))
    delta_summary = {
        "abs_rel": float(zero_summary["abs_rel"] - orig_summary["abs_rel"]),
        "rmse": float(zero_summary["rmse"] - orig_summary["rmse"]),
        "silog": float(zero_summary["silog"] - orig_summary["silog"]),
        "d1": float(zero_summary["d1"] - orig_summary["d1"]),
        "d2": float(zero_summary["d2"] - orig_summary["d2"]),
        "d3": float(zero_summary["d3"] - orig_summary["d3"]),
        "mean_pred_l1": mean_pred_l1,
        "mean_aligned_l1_valid": mean_aligned_l1,
    }

    summary = {
        "checkpoint": str(checkpoint_path),
        "input_type": input_type,
        "split": args.split,
        "num_samples": len(delta_rows),
        "device": DEVICE,
        "gate_values": gate_values,
        "gate_tanh": gate_tanh,
        "orig_summary": orig_summary,
        "zero_summary": zero_summary,
        "delta_summary": delta_summary,
        "per_sample": delta_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

    lines = [
        f"checkpoint={checkpoint_path}",
        f"input_type={input_type}",
        f"split={args.split}",
        f"num_samples={len(delta_rows)}",
        f"gate_values={gate_values}",
        f"gate_tanh={gate_tanh}",
        "",
        "[orig]",
        f"abs_rel={orig_summary['abs_rel']:.6f} rmse={orig_summary['rmse']:.6f} silog={orig_summary['silog']:.6f} d1={orig_summary['d1']:.6f} d2={orig_summary['d2']:.6f} d3={orig_summary['d3']:.6f}",
        "[zero_gate]",
        f"abs_rel={zero_summary['abs_rel']:.6f} rmse={zero_summary['rmse']:.6f} silog={zero_summary['silog']:.6f} d1={zero_summary['d1']:.6f} d2={zero_summary['d2']:.6f} d3={zero_summary['d3']:.6f}",
        "[delta zero-orig]",
        f"abs_rel={delta_summary['abs_rel']:+.6f} rmse={delta_summary['rmse']:+.6f} silog={delta_summary['silog']:+.6f} d1={delta_summary['d1']:+.6f} d2={delta_summary['d2']:+.6f} d3={delta_summary['d3']:+.6f}",
        f"mean_pred_l1={delta_summary['mean_pred_l1']:.6f}",
        f"mean_aligned_l1_valid={delta_summary['mean_aligned_l1_valid']:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved bridge zero-gate comparison to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
