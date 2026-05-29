#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.util.loss import build_training_target, robust_normalize_target_per_sample
from foundation.engine.datasets import VKITTI2Raw
from foundation.engine.models import build_raw_residual_dav2_model
from foundation.engine.transforms import packed_bayer_to_base_rgb, resolve_unprocessing_config
from foundation.tools.residual_training_common import (
    METRIC_KEYS,
    REGION_KEYS,
    average_dicts,
    float_or_none,
    format_seconds,
    resolve_model_state,
    sample_region_metrics,
    save_json,
    strip_module_prefix,
)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-time x3/ffm_mid feature ablation for RAW residual models.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-source", required=True, choices=["x3", "ffm_mid"])
    parser.add_argument("--feature-ablation-modes", default="true,zero,mean,shuffle")
    parser.add_argument("--shuffle-policy", default="stable_hash_far", choices=["stable_hash_far", "next"])
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(config: dict[str, Any]) -> VKITTI2Raw:
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


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    base_model = DepthAnythingV2(**MODEL_CONFIGS[str(config["encoder"])])
    model = build_raw_residual_dav2_model(
        base_model,
        residual_feature_source=str(config["residual_feature_source"]),
        residual_head_d0_mode=str(config.get("residual_head_d0_mode", "concat")),
        residual_alpha=float(config["residual_alpha"]),
        d0_sign=int(config["d0_sign"]),
        sensor_hw=(int(config["input_height"]), int(config["input_width"])),
        backbone_hw=None,
    )
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(ckpt_obj)), strict=True)
    return model.to(device).eval()


def parse_condition_frame(sample: dict[str, Any]) -> tuple[str, int | None]:
    path = Path(str(sample.get("image_path", "")))
    parts = path.parts
    condition = ""
    for idx, part in enumerate(parts):
        if part == "frames" and idx + 2 < len(parts):
            condition = parts[idx + 2]
            break
    frame = None
    try:
        frame = int(path.stem.split("_")[-1])
    except ValueError:
        frame = None
    return condition, frame


def stable_hash_far_mapping(dataset: VKITTI2Raw, *, seed: int) -> list[int]:
    meta: list[tuple[str, int | None, str]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition, frame = parse_condition_frame(sample)
        meta.append((condition, frame, str(sample["sample_name"])))
    mapping: list[int] = []
    for target_idx, (target_condition, target_frame, _) in enumerate(meta):
        tiers: list[list[int]] = []
        if target_frame is not None:
            tiers.append(
                [
                    idx
                    for idx, (cond, frame, _) in enumerate(meta)
                    if idx != target_idx and cond != target_condition and frame is not None and abs(frame - target_frame) >= 50
                ]
            )
        tiers.append([idx for idx, (cond, _, _) in enumerate(meta) if idx != target_idx and cond != target_condition])
        tiers.append([idx for idx in range(len(meta)) if idx != target_idx])
        candidates = next((tier for tier in tiers if tier), [])
        if not candidates:
            candidates = [target_idx]
        donor = min(
            candidates,
            key=lambda candidate_idx: hashlib.sha256(
                f"{seed}:{target_idx}:{candidate_idx}:{meta[candidate_idx][2]}".encode("utf-8")
            ).hexdigest(),
        )
        mapping.append(int(donor))
    return mapping


def donor_mapping(dataset: VKITTI2Raw, *, policy: str, seed: int) -> list[int]:
    if policy == "next":
        return [int((idx + 1) % len(dataset)) for idx in range(len(dataset))]
    if policy == "stable_hash_far":
        return stable_hash_far_mapping(dataset, seed=seed)
    raise ValueError(f"Unsupported shuffle policy: {policy}")


def mapping_sha256(mapping_records: list[dict[str, Any]]) -> str:
    payload = json.dumps(mapping_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_feature(model: torch.nn.Module, raw: torch.Tensor, feature_source: str) -> torch.Tensor:
    base_rgb = packed_bayer_to_base_rgb(raw)
    x3, ram_features = model.ram_core.forward_with_features(base_rgb)
    if feature_source == "x3":
        return x3.detach()
    if feature_source == "ffm_mid":
        return ram_features["ffm_mid"].detach()
    raise ValueError(f"Unsupported feature_source={feature_source!r}")


def compute_feature_mean(
    *,
    dataset: VKITTI2Raw,
    model: torch.nn.Module,
    feature_source: str,
    device: torch.device,
    max_samples: int | None,
) -> torch.Tensor:
    sums: torch.Tensor | None = None
    count = 0
    limit = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    with torch.no_grad():
        for idx in range(limit):
            sample = dataset[idx]
            raw = sample["raw"].unsqueeze(0).to(device).float()
            feat = extract_feature(model, raw, feature_source)
            reduce_dims = tuple(range(2, feat.ndim))
            channel_sum = feat.sum(dim=reduce_dims).sum(dim=0)
            pixels = int(np.prod(feat.shape[2:]))
            sums = channel_sum if sums is None else sums + channel_sum
            count += pixels
    if sums is None or count <= 0:
        raise RuntimeError("Cannot compute feature mean from zero samples.")
    return (sums / float(count)).view(1, -1, 1, 1)


def evaluate_mode(
    *,
    dataset: VKITTI2Raw,
    model: torch.nn.Module,
    config: dict[str, Any],
    args: argparse.Namespace,
    mode: str,
    device: torch.device,
    feature_mean: torch.Tensor | None,
    mapping: list[int],
    mapping_records: list[dict[str, Any]],
) -> dict[str, Any]:
    final_metrics: list[dict[str, float]] = []
    d0_metrics: list[dict[str, float]] = []
    final_regions: list[dict[str, float]] = []
    d0_regions: list[dict[str, float]] = []
    diagnostics: list[dict[str, float]] = []
    start = time.time()
    max_samples = len(dataset) if args.max_val_samples is None else min(int(args.max_val_samples), len(dataset))
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.float16 if config.get("amp_dtype") == "fp16" else torch.bfloat16
    per_sample: list[dict[str, Any]] = []

    with torch.no_grad():
        for idx in range(max_samples):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device).float()
            raw = sample["raw"].unsqueeze(0).to(device).float()
            depth = sample["depth"].unsqueeze(0).to(device).float()
            valid = sample["valid_mask"].unsqueeze(0).to(device).bool()
            valid = valid & (depth >= float(config["min_depth"])) & (depth <= float(config["max_depth"]))
            if int(valid[0].sum().item()) < 128:
                continue

            override = None
            donor_idx = None
            if mode == "zero":
                feat = extract_feature(model, raw, args.feature_source)
                override = {args.feature_source: torch.zeros_like(feat)}
            elif mode == "mean":
                if feature_mean is None:
                    raise RuntimeError("feature_mean is required for mean ablation")
                feat = extract_feature(model, raw, args.feature_source)
                override = {args.feature_source: feature_mean.to(device=device, dtype=feat.dtype).expand_as(feat)}
            elif mode == "shuffle":
                donor_idx = int(mapping[idx])
                donor_sample = dataset[donor_idx]
                donor_raw = donor_sample["raw"].unsqueeze(0).to(device).float()
                override = {args.feature_source: extract_feature(model, donor_raw, args.feature_source)}
            elif mode != "true":
                raise ValueError(f"Unsupported mode={mode!r}")

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                out = model.forward_with_feature_override(
                    {"image": image, "raw": raw, "valid_mask": valid},
                    feature_override=override,
                    feature_ablation_mode=mode,
                )

            inv_gt = build_training_target(depth.float(), valid, target_space="metric_depth")
            y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid, min_valid_pixels=128)
            depth_np = depth[0].detach().cpu().numpy().astype(np.float32)
            valid_np = valid[0].detach().cpu().numpy().astype(bool)
            pred_np = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
            d0_np = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
            aligned_final, _ = affine_align_disp(depth_np, pred_np, valid_np)
            aligned_d0, _ = affine_align_disp(depth_np, d0_np, valid_np)
            metrics_final = compute_metrics(depth_np, aligned_final, valid_np, min_depth=float(config["min_depth"]), max_depth=float(config["max_depth"]))
            metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, min_depth=float(config["min_depth"]), max_depth=float(config["max_depth"]))
            if metrics_final is None or metrics_d0 is None:
                continue
            rgb_preview = sample["rgb_preview"].permute(1, 2, 0).numpy().astype(np.float32)
            region_final, region_d0 = sample_region_metrics(
                depth_np=depth_np,
                valid_np=valid_np,
                aligned_final=aligned_final,
                aligned_d0=aligned_d0,
                d0_norm_np=out["D0_norm"][0].float().detach().cpu().numpy(),
                y_norm_np=y_norm[0].float().detach().cpu().numpy(),
                rgb_preview_np=rgb_preview,
                min_depth=float(config["min_depth"]),
                max_depth=float(config["max_depth"]),
            )
            gate = out["gate"].float()
            delta = out["delta"].float()
            gate_delta = gate * delta
            row = {
                "dataset_index": int(idx),
                "sample_name": str(sample["sample_name"]),
                "donor_index": donor_idx,
                "final_abs_rel": float(metrics_final["abs_rel"]),
                "D0_abs_rel": float(metrics_d0["abs_rel"]),
                "mean_gate": float(gate[valid].mean().detach().item()),
                "mean_abs_delta": float(delta[valid].abs().mean().detach().item()),
                "mean_abs_gate_delta": float(gate_delta[valid].abs().mean().detach().item()),
            }
            per_sample.append(row)
            final_metrics.append({key: float(metrics_final[key]) for key in METRIC_KEYS if key in metrics_final})
            d0_metrics.append({key: float(metrics_d0[key]) for key in METRIC_KEYS if key in metrics_d0})
            final_regions.append(region_final)
            d0_regions.append(region_d0)
            diagnostics.append(
                {
                    "mean_gate": row["mean_gate"],
                    "mean_abs_delta": row["mean_abs_delta"],
                    "mean_abs_gate_delta": row["mean_abs_gate_delta"],
                }
            )

    if not final_metrics:
        raise RuntimeError(f"Mode {mode!r} produced zero valid samples.")
    summary = {
        "feature_ablation_mode": mode,
        "feature_source": args.feature_source,
        "samples": len(final_metrics),
        "max_val_samples": args.max_val_samples,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "shuffle_policy": args.shuffle_policy if mode == "shuffle" else None,
        "shuffle_seed": args.shuffle_seed if mode == "shuffle" else None,
        "donor_mapping_sha256": mapping_sha256(mapping_records) if mode == "shuffle" else None,
        "D0_consistency": "feature override is applied after D0/D0_norm computation; target raw/image are unchanged",
        "overall": {"final": average_dicts(final_metrics, METRIC_KEYS), "D0": average_dicts(d0_metrics, METRIC_KEYS)},
        "region": {"final": average_dicts(final_regions, REGION_KEYS), "D0": average_dicts(d0_regions, REGION_KEYS)},
        "diagnostics": average_dicts(diagnostics, ["mean_gate", "mean_abs_delta", "mean_abs_gate_delta"]),
        "per_sample": per_sample,
        "elapsed_seconds": float(time.time() - start),
    }
    print(
        f"[{mode}] samples={summary['samples']} abs_rel={summary['overall']['final']['abs_rel']:.6f} "
        f"elapsed={format_seconds(summary['elapsed_seconds'])}",
        flush=True,
    )
    return summary


def write_summary(output_dir: Path, summaries: dict[str, dict[str, Any]], mapping_records: list[dict[str, Any]]) -> None:
    rows = []
    true_abs_rel = summaries.get("true", {}).get("overall", {}).get("final", {}).get("abs_rel")
    for mode, summary in summaries.items():
        overall = summary["overall"]["final"]
        region = summary["region"]["final"]
        row = {
            "feature_ablation_mode": mode,
            "feature_source": summary["feature_source"],
            "abs_rel": overall.get("abs_rel"),
            "d1": overall.get("d1"),
            "boundary_abs_rel": region.get("boundary_abs_rel"),
            "dav2_high_error_abs_rel": region.get("dav2_high_error_abs_rel"),
            "far50_abs_rel": region.get("far50_abs_rel"),
            "dark_abs_rel": region.get("dark_abs_rel"),
            "saturated_abs_rel": region.get("saturated_abs_rel"),
            "mean_gate": summary["diagnostics"].get("mean_gate"),
            "mean_abs_delta": summary["diagnostics"].get("mean_abs_delta"),
            "mean_abs_gate_delta": summary["diagnostics"].get("mean_abs_gate_delta"),
            "minus_true_abs_rel": None if true_abs_rel is None or overall.get("abs_rel") is None else overall["abs_rel"] - true_abs_rel,
            "shuffle_policy": summary.get("shuffle_policy"),
            "shuffle_seed": summary.get("shuffle_seed"),
            "donor_mapping_sha256": summary.get("donor_mapping_sha256"),
        }
        rows.append(row)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RAW Feature Ablation Summary\n\n")
        f.write("| mode | abs_rel | minus_true | boundary | far50 | dark | saturated | mean_gate |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['feature_ablation_mode']} | {float_or_none(row['abs_rel'])} | "
                f"{float_or_none(row['minus_true_abs_rel'])} | {float_or_none(row['boundary_abs_rel'])} | "
                f"{float_or_none(row['far50_abs_rel'])} | {float_or_none(row['dark_abs_rel'])} | "
                f"{float_or_none(row['saturated_abs_rel'])} | {float_or_none(row['mean_gate'])} |\n"
            )
        if mapping_records:
            f.write(f"\nDonor mapping SHA256: `{mapping_sha256(mapping_records)}`\n")
    save_json(output_dir / "donor_mapping.json", {"records": mapping_records, "sha256": mapping_sha256(mapping_records)} if mapping_records else {"records": []})


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_json(run_dir / "config.json")
    if args.feature_source not in str(config.get("residual_feature_source", "")):
        raise ValueError(
            f"Requested feature_source={args.feature_source!r} is not present in run residual_feature_source="
            f"{config.get('residual_feature_source')!r}"
        )
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config)
    model = build_model(config, checkpoint, device)
    modes = [mode.strip() for mode in args.feature_ablation_modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in ("true", "zero", "mean", "shuffle"):
            raise ValueError(f"Unsupported ablation mode={mode!r}")
    mapping = donor_mapping(dataset, policy=args.shuffle_policy, seed=args.shuffle_seed)
    max_records = len(dataset) if args.max_val_samples is None else min(args.max_val_samples, len(dataset))
    mapping_records = []
    for idx in range(max_records):
        donor_idx = mapping[idx]
        target = dataset[idx]
        donor = dataset[donor_idx]
        mapping_records.append(
            {
                "target_index": int(idx),
                "donor_index": int(donor_idx),
                "target_sample_name": str(target["sample_name"]),
                "donor_sample_name": str(donor["sample_name"]),
                "shuffle_policy": args.shuffle_policy,
                "shuffle_seed": int(args.shuffle_seed),
            }
        )
    feature_mean = None
    if "mean" in modes:
        feature_mean = compute_feature_mean(
            dataset=dataset,
            model=model,
            feature_source=args.feature_source,
            device=device,
            max_samples=args.max_val_samples,
        )
    summaries = {}
    for mode in modes:
        summary = evaluate_mode(
            dataset=dataset,
            model=model,
            config=config,
            args=args,
            mode=mode,
            device=device,
            feature_mean=feature_mean,
            mapping=mapping,
            mapping_records=mapping_records,
        )
        summaries[mode] = summary
        save_json(output_dir / f"{mode}_metrics.json", summary)
    write_summary(output_dir, summaries, mapping_records if "shuffle" in modes else [])


if __name__ == "__main__":
    main()
