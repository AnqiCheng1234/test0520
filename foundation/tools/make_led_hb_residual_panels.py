#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp
from foundation.tools.eval_led_hb_formal import build_raw_dataset, build_rgb_dataset, load_json
from foundation.tools.residual_training_common import resolve_model_state, save_json, strip_module_prefix
from foundation.tools.train_led_hb_incremental_residual import (
    add_dataset_raw_donor_if_needed,
    build_model as build_nseries_model,
    feature_ablation_mode,
    forward_incremental_model,
    load_incremental_checkpoint,
)
from foundation.tools.train_led_hb_residual_control import MODEL_CONFIGS, build_dav2_residual_control_model
from depth_anything_v2.dpt import DepthAnythingV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make LED-HB residual qualitative panels.")
    parser.add_argument("--run-kind", required=True, choices=["c2", "nseries"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-panels", type=int, default=8)
    parser.add_argument("--selection-mode", default="linspace", choices=["fixed", "linspace", "scan_topk_better_worse_by_final_minus_D1_absrel"])
    parser.add_argument("--indices", default="")
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--feature-ablation-mode", default="true", choices=["true", "none", "zero", "mean", "shuffle"])
    parser.add_argument("--feature-ablation-scope", default="both", choices=["both", "delta", "gate"])
    parser.add_argument("--feature-ablation-donor-offset", type=int, default=1)
    return parser.parse_args()


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normalize_map(value: np.ndarray, valid: np.ndarray | None = None, *, symmetric: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    mask = np.isfinite(arr) if valid is None else (valid & np.isfinite(arr))
    if not mask.any():
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[mask]
    if symmetric:
        vmax = max(float(np.percentile(np.abs(vals), 98.0)), 1e-6)
        out = 0.5 + 0.5 * np.clip(arr / vmax, -1.0, 1.0)
    else:
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        out = (arr - lo) / max(hi - lo, 1e-6)
    out = np.where(mask, out, 0.0)
    return np.clip(out, 0.0, 1.0)


def colorize_error(error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    norm = normalize_map(error, valid)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm
    rgb[..., 1] = 1.0 - np.abs(norm - 0.5) * 2.0
    rgb[..., 2] = 1.0 - norm
    rgb[~valid] = 0.0
    return rgb


def tile(label: str, image: np.ndarray, width: int = 252, height: int = 142) -> Image.Image:
    pil = Image.fromarray(to_uint8_rgb(image)).resize((width, height), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height + 22), "white")
    canvas.paste(pil, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill=(0, 0, 0))
    return canvas


def make_grid(items: list[tuple[str, np.ndarray]]) -> Image.Image:
    tiles = [tile(label, image) for label, image in items]
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for idx, panel in enumerate(tiles):
        grid.paste(panel, ((idx % cols) * w, (idx // cols) * h))
    return grid


def select_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.indices.strip():
        return [int(item.strip()) for item in args.indices.split(",") if item.strip()][: int(args.num_panels)]
    if args.selection_mode == "fixed":
        return list(range(min(int(args.num_panels), dataset_len)))
    count = min(int(args.num_panels), dataset_len)
    if count <= 1:
        return [0]
    return sorted(set(int(round(v)) for v in np.linspace(0, dataset_len - 1, count)))


def build_c2(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base_model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_dav2_residual_control_model(
        base_model,
        residual_feature_source=str(config["residual_feature_source"]),
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt)), strict=True)
    return model.to(device).eval()


def build_nseries(config: dict[str, Any], checkpoint: Path, args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, argparse.Namespace]:
    args_ns = argparse.Namespace(**config)
    args_ns.eval_only = True
    args_ns.eval_feature_ablation_mode = args.feature_ablation_mode
    args_ns.feature_ablation_scope = args.feature_ablation_scope
    args_ns.feature_ablation_key = "x3"
    args_ns.feature_ablation_donor_offset = int(args.feature_ablation_donor_offset)
    args_ns.train_feature_ablation_mode = "true"
    model = build_nseries_model(args_ns)
    load_incremental_checkpoint(model, checkpoint, strict=True)
    return model.to(device).eval(), args_ns


def raw_visual(raw: torch.Tensor | None) -> np.ndarray:
    if raw is None:
        return np.zeros((378, 672, 3), dtype=np.float32)
    r = raw.detach().cpu().numpy().astype(np.float32)
    return np.stack([r[0], 0.5 * (r[1] + r[2]), r[3]], axis=-1)


def x3_visual(x3: torch.Tensor | None) -> np.ndarray:
    if x3 is None:
        return np.zeros((378, 672, 3), dtype=np.float32)
    x = x3.detach().cpu().numpy().astype(np.float32)
    x = np.transpose(x, (1, 2, 0))
    return normalize_map(x, None)


def panel_for_sample(sample: dict[str, Any], out: dict[str, torch.Tensor], config: dict[str, Any]) -> Image.Image:
    depth = sample["depth"].numpy().astype(np.float32)
    valid = sample["valid_mask"].numpy().astype(bool) & (depth >= float(config["min_depth"])) & (depth <= float(config["max_depth"]))
    rgb = sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32)
    final_disp = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
    d0_disp = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
    d1_disp = out.get("D1_norm", out.get("D0_norm"))[0].float().detach().cpu().numpy().astype(np.float32)
    final_depth, _ = affine_align_disp(depth, final_disp, valid)
    d0_depth, _ = affine_align_disp(depth, d0_disp, valid)
    d1_depth, _ = affine_align_disp(depth, d1_disp, valid)
    err_final = np.abs(final_depth - depth) / np.clip(depth, 1e-6, None)
    err_d0 = np.abs(d0_depth - depth) / np.clip(depth, 1e-6, None)
    err_d1 = np.abs(d1_depth - depth) / np.clip(depth, 1e-6, None)
    gate = out.get("gate")
    delta_eff = out.get("delta_effective", out.get("delta"))
    gate_map = np.zeros_like(depth) if gate is None else gate[0].float().detach().cpu().numpy()
    gate_delta = np.zeros_like(depth) if gate is None or delta_eff is None else (gate[0].float() * delta_eff[0].float()).detach().cpu().numpy()
    items = [
        ("RGB / ldr_color", rgb),
        ("raw4 visualization", raw_visual(sample.get("raw"))),
        ("x3 visualization", x3_visual(None if out.get("x3") is None else out["x3"][0])),
        ("GT depth", normalize_map(depth, valid)),
        ("D0 depth", normalize_map(d0_depth, valid)),
        ("D1 depth", normalize_map(d1_depth, valid)),
        ("Final depth", normalize_map(final_depth, valid)),
        ("D0 abs-rel error", colorize_error(err_d0, valid)),
        ("D1 abs-rel error", colorize_error(err_d1, valid)),
        ("Final abs-rel error", colorize_error(err_final, valid)),
        ("Final-D1 improvement", normalize_map(err_d1 - err_final, valid, symmetric=True)),
        ("gate * delta", normalize_map(gate_delta, valid, symmetric=True)),
        ("gate map", normalize_map(gate_map, valid)),
    ]
    return make_grid(items)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable.")
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(run_dir / "config.json")
    config["max_depth"] = float(args.max_depth)
    dataset = build_raw_dataset(config, float(args.max_depth)) if str(config.get("input_domain")) == "raw4" else build_rgb_dataset(config, float(args.max_depth))
    indices = select_indices(len(dataset), args)
    if args.run_kind == "c2":
        model = build_c2(config, checkpoint, device)
        args_ns = None
    else:
        model, args_ns = build_nseries(config, checkpoint, args, device)
    records = []
    with torch.no_grad():
        for order, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device).float()
            valid = sample["valid_mask"].unsqueeze(0).to(device).bool()
            model_batch: dict[str, torch.Tensor] = {"image": image, "valid_mask": valid}
            if sample.get("raw") is not None:
                model_batch["raw"] = sample["raw"].unsqueeze(0).to(device).float()
            if args_ns is not None:
                add_dataset_raw_donor_if_needed(model_batch=model_batch, dataset=dataset, sample_indices=[idx], args=args_ns, device=device)
                out = forward_incremental_model(model, model_batch, args_ns, phase="eval")
            else:
                out = model(model_batch)
            panel = panel_for_sample(sample, out, config)
            panel_path = output_dir / f"panel_{order:02d}_{sample['sample_name']}.png"
            panel.save(panel_path)
            records.append({"order": order, "dataset_index": int(idx), "sample_name": sample["sample_name"], "panel_path": str(panel_path)})
    manifest = {
        "run_kind": args.run_kind,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "selection_mode": args.selection_mode,
        "feature_ablation_mode": feature_ablation_mode(args_ns, phase="eval") if args_ns is not None else "not_applicable",
        "records": records,
    }
    save_json(output_dir / "panel_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
