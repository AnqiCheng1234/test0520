#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.util.loss import build_training_target, robust_normalize_target_per_sample
from foundation.engine.datasets import VKITTI2HalfresRGBDepth, VKITTI2Raw
from foundation.engine.models import (
    build_c2_frozen_incremental_residual_model,
    build_dav2_residual_control_model,
    build_raw_residual_dav2_model,
)
from foundation.engine.transforms import resolve_unprocessing_config
from foundation.tools.residual_training_common import (
    format_seconds,
    lowpass_avgpool,
    mean_finite,
    resolve_model_state,
    save_json,
    strip_module_prefix,
    top_fraction_mask,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze residual/gate energy and frequency distribution.")
    parser.add_argument("--run-kind", required=True, choices=["control", "raw", "nseries"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lowpass-kernel", type=int, default=31)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(config: dict[str, Any], run_kind: str):
    input_domain = str(config.get("input_domain", "raw4" if run_kind == "raw" else "rgb"))
    if input_domain == "raw4":
        val_config = dict(config)
        if val_config.get("unprocessing_method") == "old_brooks_preset":
            val_config["randomize_unprocessing"] = False
            val_config = resolve_unprocessing_config(val_config)
        return VKITTI2Raw(
            filelist_path=config["vkitti_val_list"],
            mode="val",
            size=(int(config["input_height"]), int(config["input_width"])),
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            unprocessing_config=val_config,
            hflip_prob=0.0,
            include_rgb_input=True,
            include_rgb_preview=True,
            include_geometry=True,
            raw_storage_format=config["raw_storage_format"],
            fullres_even_policy=config["fullres_even_policy"],
            rgb_input_space=config["rgb_input_space"],
            depth_target_space=config["depth_target_space"],
        )
    return VKITTI2HalfresRGBDepth(
        filelist_path=config["vkitti_val_list"],
        mode="val",
        size=(int(config["input_height"]), int(config["input_width"])),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        hflip_prob=0.0,
        include_geometry=True,
        raw_storage_format=config["raw_storage_format"],
        fullres_even_policy=config["fullres_even_policy"],
        rgb_input_space=config["rgb_input_space"],
        depth_target_space=config["depth_target_space"],
    )


def build_c2_from_config(config: dict[str, Any]) -> torch.nn.Module:
    base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    return build_dav2_residual_control_model(
        base,
        residual_feature_source="d0",
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )


def build_model(config: dict[str, Any], checkpoint: Path, run_kind: str, device: torch.device) -> torch.nn.Module:
    if run_kind == "raw":
        base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
        model = build_raw_residual_dav2_model(
            base,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_head_d0_mode=str(config.get("residual_head_d0_mode", "concat")),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    elif run_kind == "control":
        base = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
        model = build_dav2_residual_control_model(
            base,
            residual_feature_source=str(config["residual_feature_source"]),
            residual_alpha=float(config["residual_alpha"]),
            d0_sign=int(config["d0_sign"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    else:
        c2_model = build_c2_from_config(config)
        c2_ckpt = Path(str(config["c2_checkpoint"])).expanduser().resolve()
        c2_obj = torch.load(str(c2_ckpt), map_location="cpu")
        c2_model.load_state_dict(strip_module_prefix(resolve_model_state(c2_obj)), strict=True)
        model = build_c2_frozen_incremental_residual_model(
            c2_model,
            method_id=str(config["method_id"]),
            incremental_feature_source=str(config["incremental_feature_source"]),
            delta_condition=str(config["delta_condition"]),
            gate_condition=str(config["gate_condition"]),
            raw_feature_encoder_trainable=str(config["raw_feature_encoder_trainable"]),
            residual_alpha=float(config["residual_alpha"]),
            lambda_lp=float(config["lambda_lp"]),
            lowpass_kernel=int(config["lowpass_kernel"]),
            sensor_hw=(int(config["input_height"]), int(config["input_width"])),
            backbone_hw=None,
        )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def mask_stat(
    *,
    mask: np.ndarray,
    gate: np.ndarray,
    residual_abs: np.ndarray,
    residual: np.ndarray,
    valid_den_gate: float,
    valid_den_res: float,
) -> dict[str, Any]:
    mask = mask.astype(bool)
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return {
            "empty_mask": True,
            "pixel_ratio": None,
            "gate_mass_ratio": None,
            "residual_energy_ratio": None,
            "mean_gate": None,
            "mean_abs_gate_delta": None,
        }
    total_pixels = int(mask.size)
    gate_sum = float(np.sum(gate[mask]))
    res_sum = float(np.sum(residual_abs[mask]))
    return {
        "empty_mask": False,
        "pixel_ratio": float(pixel_count / max(total_pixels, 1)),
        "gate_mass_ratio": None if valid_den_gate <= 0.0 else float(gate_sum / valid_den_gate),
        "residual_energy_ratio": None if valid_den_res <= 0.0 else float(res_sum / valid_den_res),
        "mean_gate": float(np.mean(gate[mask])),
        "mean_abs_gate_delta": float(np.mean(np.abs(residual[mask]))),
    }


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def analyze_sample(
    *,
    sample: dict[str, Any],
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    lowpass_kernel: int,
) -> dict[str, Any] | None:
    image = sample["image"].unsqueeze(0).to(device).float()
    raw = sample.get("raw")
    if raw is not None:
        raw = raw.unsqueeze(0).to(device).float()
    depth = sample["depth"].unsqueeze(0).to(device).float()
    valid = sample["valid_mask"].unsqueeze(0).to(device).bool()
    valid = valid & (depth >= float(config["min_depth"])) & (depth <= float(config["max_depth"]))
    if int(valid[0].sum().item()) < 128:
        return None
    batch = {"image": image, "valid_mask": valid}
    if raw is not None:
        batch["raw"] = raw
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if config.get("amp_dtype") == "fp16" else torch.bfloat16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        out = model(batch)
    gate_t = out["gate"].float()
    if "delta_effective" in out:
        residual_t = gate_t * out["delta_effective"].float()
        base_t = out.get("base_norm", out.get("D1_norm", out["D0_norm"])).float()
    else:
        residual_t = gate_t * out["delta"].float()
        base_t = out["D0_norm"].float()
    inv_gt = build_training_target(depth.float(), valid, target_space="metric_depth")
    y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid, min_valid_pixels=128)

    depth_np = depth[0].detach().cpu().numpy().astype(np.float32)
    valid_np = valid[0].detach().cpu().numpy().astype(bool)
    rgb = sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32)
    gate = gate_t[0].detach().cpu().numpy().astype(np.float32)
    residual = residual_t[0].detach().cpu().numpy().astype(np.float32)
    residual_abs = np.abs(residual)
    base_error = np.abs(base_t[0].detach().cpu().numpy().astype(np.float32) - y_norm[0].detach().cpu().numpy().astype(np.float32))
    grad_y, grad_x = np.gradient(depth_np.astype(np.float32))
    boundary = top_fraction_mask(np.sqrt(grad_x * grad_x + grad_y * grad_y), valid_np, 0.10)
    high_error = top_fraction_mask(base_error, valid_np, 0.20)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    masks = {
        "valid": valid_np,
        "invalid": ~valid_np,
        "boundary": boundary,
        "high_error": high_error,
        "far50": valid_np & (depth_np > 50.0),
        "dark": valid_np & (luma < 0.15),
        "saturated": valid_np & (np.max(rgb, axis=-1) > 0.95),
    }
    den_gate = float(np.sum(gate[valid_np]))
    den_res = float(np.sum(residual_abs[valid_np]))
    low = lowpass_avgpool(residual_t, kernel_size=lowpass_kernel)[0].detach().cpu().numpy().astype(np.float32)
    high = residual - low
    denom = float(np.mean(residual_abs[valid_np])) if bool(valid_np.any()) else 0.0
    row = {
        "dataset_index": int(sample.get("dataset_index", -1)),
        "sample_name": str(sample["sample_name"]),
        "valid_pixels": int(valid_np.sum()),
        "frequency": {
            "low_ratio": None if denom <= 0.0 else float(np.mean(np.abs(low[valid_np])) / (denom + 1e-6)),
            "high_ratio": None if denom <= 0.0 else float(np.mean(np.abs(high[valid_np])) / (denom + 1e-6)),
        },
        "masks": {
            name: mask_stat(
                mask=mask,
                gate=gate,
                residual_abs=residual_abs,
                residual=residual,
                valid_den_gate=den_gate,
                valid_den_res=den_res,
            )
            for name, mask in masks.items()
        },
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mask_names = ["valid", "invalid", "boundary", "high_error", "far50", "dark", "saturated"]
    metric_names = ["pixel_ratio", "gate_mass_ratio", "residual_energy_ratio", "mean_gate", "mean_abs_gate_delta"]
    mask_summary = {}
    for mask_name in mask_names:
        mask_summary[mask_name] = {
            metric: mean_finite([row["masks"][mask_name].get(metric) for row in rows])
            for metric in metric_names
        }
        mask_summary[mask_name]["nonempty_samples"] = int(
            sum(1 for row in rows if not bool(row["masks"][mask_name].get("empty_mask", True)))
        )
    return {
        "samples": int(len(rows)),
        "masks": mask_summary,
        "frequency": {
            "low_ratio": mean_finite([row["frequency"].get("low_ratio") for row in rows]),
            "high_ratio": mean_finite([row["frequency"].get("high_ratio") for row in rows]),
        },
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for mask_name, values in summary["masks"].items():
        row = {"mask": mask_name, **values}
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Residual Energy/Frequency Summary\n\n")
        f.write(f"Samples: {summary['samples']}\n\n")
        f.write(f"Low ratio: {summary['frequency']['low_ratio']}\n")
        f.write(f"High ratio: {summary['frequency']['high_ratio']}\n\n")
        f.write("| mask | pixel_ratio | gate_mass_ratio | residual_energy_ratio | mean_gate | mean_abs_gate_delta | nonempty |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for mask_name, row in summary["masks"].items():
            f.write(
                f"| {mask_name} | {row.get('pixel_ratio')} | {row.get('gate_mass_ratio')} | "
                f"{row.get('residual_energy_ratio')} | {row.get('mean_gate')} | "
                f"{row.get('mean_abs_gate_delta')} | {row.get('nonempty_samples')} |\n"
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_json(run_dir / "config.json")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config, args.run_kind)
    model = build_model(config, checkpoint, args.run_kind, device)
    max_samples = len(dataset) if args.max_val_samples is None else min(int(args.max_val_samples), len(dataset))
    rows: list[dict[str, Any]] = []
    start = time.time()
    for idx in range(max_samples):
        sample = dataset[idx]
        sample["dataset_index"] = int(idx)
        row = analyze_sample(
            sample=sample,
            model=model,
            config=config,
            device=device,
            lowpass_kernel=int(args.lowpass_kernel),
        )
        if row is not None:
            rows.append(row)
        if (idx + 1) % 50 == 0 or idx + 1 == max_samples:
            print(f"processed={idx + 1}/{max_samples} ok={len(rows)} elapsed={format_seconds(time.time() - start)}", flush=True)
    if not rows:
        raise RuntimeError("No valid samples for energy/frequency analysis.")
    per_sample_path = output_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(rows)
    summary.update(
        {
            "run_kind": args.run_kind,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "lowpass_kernel": int(args.lowpass_kernel),
            "max_val_samples": args.max_val_samples,
            "elapsed_seconds": float(time.time() - start),
            "per_sample_path": str(per_sample_path),
        }
    )
    save_json(output_dir / "summary.json", summary)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_summary_md(output_dir / "summary.md", summary)


if __name__ == "__main__":
    main()
