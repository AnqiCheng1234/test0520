#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from foundation.tools.eval_led_hb_formal import build_raw_dataset, load_json
from foundation.tools.make_vkitti_raw_residual_qual_panels import (
    choose_depth_range,
    clip_metric_depth_for_eval,
    colorize_depth,
    colorize_error,
    colorize_improvement,
    draw_tile,
    image_from_array,
    load_font,
)
from foundation.tools.residual_training_common import resolve_model_state, save_json, strip_module_prefix
from foundation.tools.train_led_hb_incremental_residual import (
    add_dataset_raw_donor_if_needed,
    build_model as build_nseries_model,
    forward_incremental_model,
    load_incremental_checkpoint,
)
from foundation.tools.train_led_hb_residual_control import MODEL_CONFIGS, build_dav2_residual_control_model


DEFAULT_C2_RUN = (
    PROJECT_ROOT
    / "finetune_stf/exp/0531_2341_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e20"
)
DEFAULT_N5_RUN = (
    PROJECT_ROOT
    / "finetune_stf/exp/0601_0355_led_hb_n5_d1_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"
)
DEFAULT_N3_RUN = (
    PROJECT_ROOT
    / "finetune_stf/exp/0601_0550_led_hb_n3_rgb_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"
)
DEFAULT_N2_RUN = (
    PROJECT_ROOT
    / "finetune_stf/exp/0601_0751_led_hb_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"
)
DEFAULT_N7_RUN = (
    PROJECT_ROOT
    / "finetune_stf/exp/0601_1005_led_hb_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10"
)
HEAVY_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt")


def default_checkpoint(run_dir: Path) -> Path:
    return HEAVY_ROOT / run_dir.name / "best_abs_rel.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make LED-HB main comparison panels for C2 and N-series runs.")
    parser.add_argument("--c2-run-dir", default=str(DEFAULT_C2_RUN))
    parser.add_argument("--c2-checkpoint", default=str(default_checkpoint(DEFAULT_C2_RUN)))
    parser.add_argument("--n5-run-dir", default=str(DEFAULT_N5_RUN))
    parser.add_argument("--n5-checkpoint", default=str(default_checkpoint(DEFAULT_N5_RUN)))
    parser.add_argument("--n3-run-dir", default=str(DEFAULT_N3_RUN))
    parser.add_argument("--n3-checkpoint", default=str(default_checkpoint(DEFAULT_N3_RUN)))
    parser.add_argument("--n2-run-dir", default=str(DEFAULT_N2_RUN))
    parser.add_argument("--n2-checkpoint", default=str(default_checkpoint(DEFAULT_N2_RUN)))
    parser.add_argument("--n7-run-dir", default=str(DEFAULT_N7_RUN))
    parser.add_argument("--n7-checkpoint", default=str(default_checkpoint(DEFAULT_N7_RUN)))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-indices", default="0,250,500,750")
    parser.add_argument("--num-panels", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=200.0)
    parser.add_argument("--error-max-abs-rel", type=float, default=0.75)
    parser.add_argument("--depth-pmin", type=float, default=1.0)
    parser.add_argument("--depth-pmax", type=float, default=99.0)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=120)
    parser.add_argument("--header-height", type=int, default=30)
    parser.add_argument("--raw-context-method", default="n7", choices=["n2", "n7"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_indices(arg: str, *, dataset_len: int, count: int) -> list[int]:
    if arg.strip():
        indices = [int(item.strip()) for item in arg.split(",") if item.strip()]
    elif count <= 1:
        indices = [0]
    else:
        indices = [int(round(x)) for x in np.linspace(0, dataset_len - 1, count)]
    out: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Sample index {idx} out of range for validation set length {dataset_len}")
        if idx not in out:
            out.append(idx)
    return out[: int(count)]


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


def build_nseries(config: dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, argparse.Namespace]:
    args_ns = argparse.Namespace(**config)
    args_ns.eval_only = True
    args_ns.eval_feature_ablation_mode = "true"
    args_ns.feature_ablation_scope = "both"
    args_ns.feature_ablation_key = "x3"
    args_ns.feature_ablation_donor_offset = 1
    args_ns.train_feature_ablation_mode = "true"
    model = build_nseries_model(args_ns)
    load_incremental_checkpoint(model, checkpoint, strict=True)
    return model.to(device).eval(), args_ns


def raw_visual(raw: torch.Tensor) -> np.ndarray:
    arr = raw.detach().cpu().numpy().astype(np.float32)
    return np.stack([arr[0], 0.5 * (arr[1] + arr[2]), arr[3]], axis=-1)


def x3_visual(x3: torch.Tensor | None) -> np.ndarray:
    if x3 is None:
        return np.zeros((378, 672, 3), dtype=np.float32)
    arr = x3.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def align_depth_and_error(
    *,
    depth: np.ndarray,
    disp: np.ndarray,
    valid: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    aligned, _ = affine_align_disp(depth, disp, valid)
    metrics = compute_metrics(depth, aligned, valid, min_depth=min_depth, max_depth=max_depth)
    if metrics is None:
        raise RuntimeError("Metric computation failed for selected sample.")
    aligned_eval = clip_metric_depth_for_eval(aligned, min_depth=min_depth, max_depth=max_depth)
    eval_valid = valid & np.isfinite(aligned_eval) & (aligned_eval > 0.0) & np.isfinite(depth) & (depth > 0.0)
    err = np.zeros_like(depth, dtype=np.float32)
    err[eval_valid] = np.abs(aligned_eval[eval_valid] - depth[eval_valid]) / np.clip(depth[eval_valid], 1e-6, None)
    return aligned.astype(np.float32), err, {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def method_label(key: str) -> str:
    return {
        "n5": "N5 D1-only",
        "n3": "N3 RGB",
        "n2": "N2 x3",
        "n7": "N7 x3 stopgrad",
    }[key]


def make_panel(
    *,
    sample: dict[str, Any],
    record: dict[str, Any],
    args: argparse.Namespace,
    raw_context: str,
) -> Image.Image:
    tile_w = int(args.tile_width)
    tile_h = int(args.tile_height)
    header_h = int(args.header_height)
    font = load_font(12)
    small_font = load_font(10)
    cols = 4
    canvas = Image.new("RGB", (tile_w * cols, (tile_h + header_h) * 6), (0, 0, 0))

    valid = record["valid"]
    depth_range = f"{record['depth_vmin']:.1f}..{record['depth_vmax']:.1f}m"
    error_range = f"0..{float(args.error_max_abs_rel):.2f} absrel"
    improve_range = "green: better than C2"
    raw_title = f"RAW-like raw4 ({raw_context.upper()} context)"
    x3_title = f"x3 feature ({raw_context.upper()} context)"

    def depth_tile(name: str) -> Image.Image:
        return image_from_array(
            colorize_depth(record[name]["depth"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]),
            tile_width=tile_w,
            tile_height=tile_h,
        )

    def error_tile(name: str) -> Image.Image:
        return image_from_array(
            colorize_error(record[name]["error"], valid, vmax=float(args.error_max_abs_rel)),
            tile_width=tile_w,
            tile_height=tile_h,
        )

    def improve_tile(name: str) -> Image.Image:
        return image_from_array(
            colorize_improvement(record["c2"]["error"] - record[name]["error"], valid, vlim=float(args.error_max_abs_rel)),
            tile_width=tile_w,
            tile_height=tile_h,
        )

    def improve_vs_n3_tile(name: str) -> Image.Image:
        return image_from_array(
            colorize_improvement(record["n3"]["error"] - record[name]["error"], valid, vlim=float(args.error_max_abs_rel)),
            tile_width=tile_w,
            tile_height=tile_h,
        )

    def metric_subtitle(name: str) -> str:
        metrics = record[name]["metrics"]
        subtitle = f"absrel={metrics['abs_rel']:.3f} d1={metrics['d1']:.3f}"
        if name in ("n5", "n3", "n2", "n7"):
            delta = float(metrics["abs_rel"]) - float(record["c2"]["metrics"]["abs_rel"])
            subtitle = f"{subtitle} vsC2={delta:+.3f}"
        return subtitle

    tiles = [
        (
            "RGB input",
            str(sample["sample_name"]),
            image_from_array(np.clip(record["rgb"] * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h),
        ),
        (raw_title, "synthetic packed Bayer [R,Gr,Gb,B]", image_from_array(np.clip(raw_visual(sample["raw"]) * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h)),
        (x3_title, "normalized 2..98 percentile", image_from_array(np.clip(record[raw_context]["x3_visual"] * 255.0, 0.0, 255.0).round().astype(np.uint8), tile_width=tile_w, tile_height=tile_h)),
        ("GT depth", depth_range, image_from_array(colorize_depth(record["depth"], valid, vmin=record["depth_vmin"], vmax=record["depth_vmax"]), tile_width=tile_w, tile_height=tile_h)),
        ("D0 depth", metric_subtitle("d0"), depth_tile("d0")),
        ("D0 error", error_range, error_tile("d0")),
        ("C2 / D1 depth", metric_subtitle("c2"), depth_tile("c2")),
        ("C2 / D1 error", error_range, error_tile("c2")),
    ]
    for name in ("n5", "n3", "n2", "n7"):
        n3_subtitle = "reference" if name == "n3" else "green: better than N3"
        tiles.extend(
            [
                (method_label(name), metric_subtitle(name), depth_tile(name)),
                (f"{name.upper()} error", error_range, error_tile(name)),
                (f"{name.upper()} improve vs C2", improve_range, improve_tile(name)),
                (f"{name.upper()} improve vs N3", n3_subtitle, improve_vs_n3_tile(name)),
            ]
        )

    for i, (title, subtitle, tile) in enumerate(tiles):
        draw_tile(
            canvas,
            col=i % cols,
            row=i // cols,
            tile=tile,
            title=title,
            subtitle=subtitle,
            tile_width=tile_w,
            tile_height=tile_h,
            header_height=header_h,
            font=font,
            small_font=small_font,
        )
    return canvas


def main() -> None:
    args = parse_args()
    if int(args.num_panels) <= 0:
        raise ValueError("--num-panels must be positive")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable.")

    run_dirs = {
        "c2": Path(args.c2_run_dir).expanduser().resolve(),
        "n5": Path(args.n5_run_dir).expanduser().resolve(),
        "n3": Path(args.n3_run_dir).expanduser().resolve(),
        "n2": Path(args.n2_run_dir).expanduser().resolve(),
        "n7": Path(args.n7_run_dir).expanduser().resolve(),
    }
    checkpoints = {
        "c2": Path(args.c2_checkpoint).expanduser().resolve(),
        "n5": Path(args.n5_checkpoint).expanduser().resolve(),
        "n3": Path(args.n3_checkpoint).expanduser().resolve(),
        "n2": Path(args.n2_checkpoint).expanduser().resolve(),
        "n7": Path(args.n7_checkpoint).expanduser().resolve(),
    }
    for key, path in run_dirs.items():
        require_dir(path, f"{key} run dir")
    for key, path in checkpoints.items():
        require_file(path, f"{key} checkpoint")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = PROJECT_ROOT / "plans/0531_led_night/diagnostics" / f"{timestamp}_led_hb_main_comparison_panels"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    configs = {key: load_json(path / "config.json") for key, path in run_dirs.items()}
    raw_context = str(args.raw_context_method)
    raw_config = dict(configs[raw_context])
    raw_config["max_depth"] = float(args.max_depth)
    dataset = build_raw_dataset(raw_config, float(args.max_depth))
    indices = parse_indices(args.sample_indices, dataset_len=len(dataset), count=int(args.num_panels))

    c2_model = build_c2(configs["c2"], checkpoints["c2"], device)
    n_models: dict[str, tuple[torch.nn.Module, argparse.Namespace]] = {
        key: build_nseries(configs[key], checkpoints[key], device) for key in ("n5", "n3", "n2", "n7")
    }

    manifest_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for order, idx in enumerate(indices, start=1):
            sample = dataset[idx]
            depth = sample["depth"].numpy().astype(np.float32)
            valid = sample["valid_mask"].numpy().astype(bool)
            valid = valid & (depth >= float(raw_config["min_depth"])) & (depth <= float(args.max_depth))
            if int(valid.sum()) < 128:
                raise RuntimeError(f"Selected sample {idx} has too few valid pixels.")

            image = sample["image"].unsqueeze(0).to(device).float()
            raw = sample["raw"].unsqueeze(0).to(device).float()
            valid_t = torch.from_numpy(valid).unsqueeze(0).to(device).bool()
            model_batch: dict[str, torch.Tensor] = {"image": image, "raw": raw, "valid_mask": valid_t}

            c2_out = c2_model({"image": image, "valid_mask": valid_t})
            d0_disp = (float(configs["c2"]["d0_sign"]) * c2_out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
            c2_disp = c2_out["pred"][0].float().detach().cpu().numpy().astype(np.float32)

            depth_vmin, depth_vmax = choose_depth_range(
                depth,
                valid,
                min_depth=float(raw_config["min_depth"]),
                max_depth=float(args.max_depth),
                pmin=float(args.depth_pmin),
                pmax=float(args.depth_pmax),
            )
            record: dict[str, Any] = {
                "dataset_index": int(idx),
                "sample_name": str(sample["sample_name"]),
                "image_path": str(sample["image_path"]),
                "depth_path": str(sample["depth_path"]),
                "rgb": sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32),
                "depth": depth,
                "valid": valid,
                "depth_vmin": depth_vmin,
                "depth_vmax": depth_vmax,
            }
            d0_depth, d0_error, d0_metrics = align_depth_and_error(
                depth=depth,
                disp=d0_disp,
                valid=valid,
                min_depth=float(raw_config["min_depth"]),
                max_depth=float(args.max_depth),
            )
            c2_depth, c2_error, c2_metrics = align_depth_and_error(
                depth=depth,
                disp=c2_disp,
                valid=valid,
                min_depth=float(raw_config["min_depth"]),
                max_depth=float(args.max_depth),
            )
            record["d0"] = {"depth": d0_depth, "error": d0_error, "metrics": d0_metrics}
            record["c2"] = {"depth": c2_depth, "error": c2_error, "metrics": c2_metrics}

            for key, (model, args_ns) in n_models.items():
                batch = {"image": image, "valid_mask": valid_t}
                if str(configs[key].get("input_domain")) == "raw4":
                    batch["raw"] = raw
                add_dataset_raw_donor_if_needed(model_batch=batch, dataset=dataset, sample_indices=[idx], args=args_ns, device=device)
                out = forward_incremental_model(model, batch, args_ns, phase="eval")
                disp = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
                aligned, err, metrics = align_depth_and_error(
                    depth=depth,
                    disp=disp,
                    valid=valid,
                    min_depth=float(raw_config["min_depth"]),
                    max_depth=float(args.max_depth),
                )
                gate = out.get("gate")
                delta = out.get("delta_effective", out.get("delta"))
                gate_np = np.zeros_like(depth, dtype=np.float32) if gate is None else gate[0].float().detach().cpu().numpy().astype(np.float32)
                if gate is None or delta is None:
                    gate_delta_np = np.zeros_like(depth, dtype=np.float32)
                else:
                    gate_delta_np = (gate[0].float() * delta[0].float()).detach().cpu().numpy().astype(np.float32)
                record[key] = {
                    "depth": aligned,
                    "error": err,
                    "metrics": metrics,
                    "gate": gate_np,
                    "gate_delta": gate_delta_np,
                    "x3_visual": x3_visual(None if out.get("x3") is None else out["x3"][0]),
                }

            panel = make_panel(sample=sample, record=record, args=args, raw_context=raw_context)
            safe_sample = str(sample["sample_name"]).replace("/", "_")
            panel_path = output_dir / f"panel_{order:02d}_validx{idx:04d}_{safe_sample}.png"
            panel.save(panel_path)
            manifest_records.append(
                {
                    "order": int(order),
                    "dataset_index": int(idx),
                    "sample_name": str(sample["sample_name"]),
                    "image_path": str(sample["image_path"]),
                    "depth_path": str(sample["depth_path"]),
                    "panel_path": str(panel_path),
                    "metrics": {
                        name: {"abs_rel": float(record[name]["metrics"]["abs_rel"]), "d1": float(record[name]["metrics"]["d1"])}
                        for name in ("d0", "c2", "n5", "n3", "n2", "n7")
                    },
                }
            )
            print(f"wrote {panel_path}", flush=True)

    manifest = {
        "output_dir": str(output_dir),
        "sample_indices": indices,
        "max_depth": float(args.max_depth),
        "raw_context_method": raw_context,
        "run_dirs": {key: str(value) for key, value in run_dirs.items()},
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "improvement_definition": "improvement = C2_absrel_error - method_absrel_error; green means method is better than C2",
        "n3_comparison_definition": "improvement_vs_n3 = N3_absrel_error - method_absrel_error; green means method is better than N3 RGB control",
        "raw_semantics": "RAW panel is inverse-ISP synthetic RAW-like packed Bayer raw4 derived from LED LDR RGB, not real sensor RAW.",
        "panel_layout": [
            "Row 1: RGB / RAW-like raw4 / x3 / GT",
            "Row 2: D0 depth / D0 error / C2-D1 depth / C2-D1 error",
            "Rows 3-6: method depth / method error / method improve-vs-C2 / method improve-vs-N3",
        ],
        "records": manifest_records,
    }
    save_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
