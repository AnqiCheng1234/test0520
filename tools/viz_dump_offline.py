#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from finetune_stf.train import (
    METRIC_KEYS,
    affine_align_disp_1d,
    build_datasets,
    build_model,
    build_rgb_decoder_eval_model,
    build_rgb_reference_eval_model,
    get_single_sample_meta,
    resolve_model_state,
    sample_bilinear_disparity_at_mask,
    strip_module_prefix,
    sync_rgb_decoder_eval_model,
)
from finetune_stf.util.viz_dump import collect_fixed_samples, dump_fixed_samples


_VALID_SPLITS = {"kitti", "nyu", "eth3d", "robotcar", "robotcar_night"}
_EVAL_FLAG_BY_SPLIT = {
    "kitti": "eval_kitti",
    "nyu": "eval_nyu",
    "eth3d": "eval_eth3d",
    "robotcar": "eval_robotcar",
    "robotcar_night": "eval_robotcar_night",
}


def _parse_splits(value):
    if value is None or str(value).strip().lower() in {"", "all"}:
        return None
    splits = {item.strip().lower() for item in str(value).split(",") if item.strip()}
    unknown = sorted(splits - _VALID_SPLITS)
    if unknown:
        raise ValueError(f"Unknown split(s): {', '.join(unknown)}. Valid choices: {', '.join(sorted(_VALID_SPLITS))}")
    return splits


def _load_args(run_dir, *, splits=None):
    config_path = Path(run_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("enable_fixed_viz_dump", True)
    data["eval_only"] = False
    data["num_workers"] = 0
    data["save_path"] = str(run_dir)
    data["heavy_save_path"] = data.get("heavy_save_path") or str(run_dir)
    if splits is not None:
        for split, flag in _EVAL_FLAG_BY_SPLIT.items():
            data[flag] = split in splits
        data["eval_stf"] = False
        data["stage"] = "offline_viz"
    return SimpleNamespace(**data)


def _make_loaders(datasets):
    loader_keys = {
        "kitti_val": "kitti_val_loader",
        "nyu_val": "nyu_val_loader",
        "eth3d_val_fast": "eth3d_val_fast_loader",
        "robotcar_val_fast": "robotcar_val_fast_loader",
        "robotcar_night_val_fast": "robotcar_night_val_fast_loader",
    }
    state = {}
    for dataset_key, loader_key in loader_keys.items():
        dataset = datasets.get(dataset_key)
        if dataset is not None:
            state[loader_key] = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return state


def _select_input(sample, input_type):
    if str(input_type) != "rgb" and "raw" in sample:
        tensor = sample["raw"]
    else:
        tensor = sample["image"]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.cuda(non_blocking=True).float()


def _accumulate(summary, counts, metrics):
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            continue
        summary[key] += value
        counts[key] += 1


def _finalize(summary, counts):
    return {
        key: (float(summary[key] / counts[key]) if counts[key] > 0 else None)
        for key in METRIC_KEYS
    }


def evaluate_loader(model, loader, args, *, input_type, min_depth, max_depth, max_samples=None):
    model.eval()
    sums = {key: 0.0 for key in METRIC_KEYS}
    counts = {key: 0 for key in METRIC_KEYS}
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    processed = 0

    with torch.no_grad():
        for sample in loader:
            if max_samples is not None and processed >= max_samples:
                break
            img = _select_input(sample, input_type)
            depth = sample["depth"][0].cuda(non_blocking=True).float()
            valid_mask = sample["valid_mask"][0].cuda(non_blocking=True).bool()
            depth_mode = str(get_single_sample_meta(sample, "depth_mode", "full"))
            fast_eval_backend = str(get_single_sample_meta(sample, "fast_eval_backend", "proxy"))

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(getattr(args, "amp", False))):
                pred_disp = model(img).float()

            valid_mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)
            if int(valid_mask.sum().item()) < 10:
                continue

            if depth_mode == "fast" and fast_eval_backend == "sparse":
                _, pred_samples = sample_bilinear_disparity_at_mask(pred_disp[0], valid_mask, depth.shape[-2:])
                depth_np = depth[valid_mask].detach().cpu().numpy()
                pred_np = pred_samples.detach().cpu().numpy()
                valid_np = np.ones_like(depth_np, dtype=bool)
                aligned_depth, _ = affine_align_disp_1d(depth_np, pred_np)
            else:
                pred_disp = F.interpolate(
                    pred_disp[:, None],
                    depth.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]
                depth_np = depth.detach().cpu().numpy()
                pred_np = pred_disp.detach().cpu().numpy()
                valid_np = valid_mask.detach().cpu().numpy().astype(bool)
                aligned_depth, _ = affine_align_disp(depth_np, pred_np, valid_np)

            metrics = compute_metrics(depth_np, aligned_depth, valid_np, min_depth=min_depth, max_depth=max_depth)
            if metrics is None:
                continue
            _accumulate(sums, counts, metrics)
            processed += 1
    return {"samples": processed, "metrics": _finalize(sums, counts)}


def main():
    parser = argparse.ArgumentParser(description="Offline fixed-sample visualization dump for a train.py run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch-name", default="last")
    parser.add_argument("--eval-json", default=None)
    parser.add_argument("--splits", default=None, help="Comma-separated fixed-viz splits to dump, or all. Example: kitti")
    parser.add_argument("--n-per-split", default=8, type=int)
    args_cli = parser.parse_args()

    splits = _parse_splits(args_cli.splits)
    args = _load_args(args_cli.run_dir, splits=splits)
    datasets = build_datasets(args)
    train_state = _make_loaders(datasets)
    fixed_samples = collect_fixed_samples(train_state, n_per_split=args_cli.n_per_split)
    if splits is not None:
        fixed_samples = {split: samples for split, samples in fixed_samples.items() if split in splits}

    model = build_model(args).cuda().eval()
    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu")
    model.load_state_dict(strip_module_prefix(resolve_model_state(checkpoint)), strict=True)

    model_overrides = {}
    input_type_overrides = {}
    rgb_decoder_eval_model = None
    kitti_reference_eval_model = None
    fixed_viz_rgb_baseline_model = None
    if "kitti_val_loader" in train_state:
        if args.kitti_eval_protocol == "rgb_checkpoint_decoder":
            rgb_decoder_eval_model = build_rgb_decoder_eval_model(args).cuda().eval()
            sync_rgb_decoder_eval_model(rgb_decoder_eval_model, model, logger=None, rank=0, sync_tag="offline_dump")
            model_overrides["kitti"] = rgb_decoder_eval_model
        else:
            kitti_reference_eval_model = build_rgb_reference_eval_model(args).cuda().eval()
            model_overrides["kitti"] = kitti_reference_eval_model
        input_type_overrides["kitti"] = "rgb"

    if any(split_name in fixed_samples for split_name in ("robotcar", "robotcar_night")):
        fixed_viz_rgb_baseline_model = kitti_reference_eval_model or build_rgb_reference_eval_model(args).cuda().eval()

    dump_outputs = dump_fixed_samples(
        model,
        fixed_samples,
        args,
        args_cli.epoch_name,
        args_cli.run_dir,
        model_overrides=model_overrides,
        input_type_overrides=input_type_overrides,
        rgb_baseline_model=fixed_viz_rgb_baseline_model,
        rgb_baseline_splits=("robotcar", "robotcar_night"),
        rgb_baseline_label="RGB DAv2",
    )
    print(json.dumps({split: len(paths) for split, paths in dump_outputs.items()}, indent=2, sort_keys=True))

    if args_cli.eval_json:
        eval_model_by_tag = {
            "kitti": model_overrides.get("kitti"),
            "nyu": rgb_decoder_eval_model,
            "eth3d": model,
            "robotcar": model,
            "robotcar_night": model,
        }
        eval_specs = [
            ("kitti", "kitti_val_loader", "rgb", args.kitti_min_depth, args.kitti_max_depth, args.debug_max_kitti_samples),
            ("nyu", "nyu_val_loader", "rgb", args.nyu_min_depth, args.nyu_max_depth, args.nyu_max_samples),
            ("eth3d", "eth3d_val_fast_loader", args.input_type, args.eth3d_min_depth, args.eth3d_max_depth, args.eth3d_max_samples),
            ("robotcar", "robotcar_val_fast_loader", args.input_type, args.robotcar_min_depth, args.robotcar_max_depth, args.robotcar_max_samples),
            (
                "robotcar_night",
                "robotcar_night_val_fast_loader",
                args.input_type,
                args.robotcar_night_min_depth,
                args.robotcar_night_max_depth,
                args.robotcar_night_max_samples,
            ),
        ]
        eval_results = {}
        for tag, loader_key, input_type, min_depth, max_depth, max_samples in eval_specs:
            loader = train_state.get(loader_key)
            eval_model = eval_model_by_tag.get(tag)
            if loader is None or eval_model is None:
                continue
            eval_results[tag] = evaluate_loader(
                eval_model,
                loader,
                args,
                input_type=input_type,
                min_depth=min_depth,
                max_depth=max_depth,
                max_samples=max_samples,
            )
        output_path = Path(args_cli.eval_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, sort_keys=True)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
