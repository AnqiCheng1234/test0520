#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.datasets import (
    DEPTH_TARGET_SPACE_CHOICES,
    FULLRES_EVEN_POLICY_CHOICES,
    RAW_STORAGE_FORMAT_CHOICES,
    RGB_INPUT_SPACE_CHOICES,
    VKITTI2Raw,
)
from foundation.engine.transforms import (
    NOT_APPLICABLE,
    RAW_ADAPTER_PACKED_CHANNEL_ORDER,
    assert_unprocessing_summaries_compatible,
    assert_unprocessing_summary_matches_config,
    resolve_unprocessing_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test VKITTI2 raw_adapter_style dataset path.")
    parser.add_argument("--vkitti-train-list", required=True)
    parser.add_argument("--vkitti-val-list", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--raw-storage-format", required=True, choices=RAW_STORAGE_FORMAT_CHOICES)
    parser.add_argument("--fullres-even-policy", required=True, choices=FULLRES_EVEN_POLICY_CHOICES)
    parser.add_argument("--rgb-input-space", required=True, choices=RGB_INPUT_SPACE_CHOICES)
    parser.add_argument("--depth-target-space", required=True, choices=DEPTH_TARGET_SPACE_CHOICES)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--unprocessing-method", required=True, choices=["raw_adapter_style"])
    parser.add_argument("--vkitti-unprocessing-preset", default=NOT_APPLICABLE)
    parser.add_argument("--vkitti-unprocessing-mix-weights", default=None)
    parser.add_argument("--randomize-unprocessing", action="store_true", default=None)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument("--raw-adapter-backend", required=True, choices=["analytic"])
    parser.add_argument("--raw-adapter-cfa-pattern", required=True, choices=["RGGB"])
    parser.add_argument("--raw-adapter-packed-channel-order", required=True, choices=[RAW_ADAPTER_PACKED_CHANNEL_ORDER])
    parser.add_argument("--raw-adapter-rgb-transfer", required=True, choices=["srgb_piecewise"])
    parser.add_argument("--raw-adapter-inverse-tone", required=True, choices=["none", "global_0p15"])
    parser.add_argument("--raw-adapter-ccm", required=True, choices=["identity", "generic_d65"])
    parser.add_argument("--raw-adapter-red-gain-range", nargs=2, type=float, required=True)
    parser.add_argument("--raw-adapter-blue-gain-range", nargs=2, type=float, required=True)
    parser.add_argument("--raw-adapter-fixed-red-gain", type=float, required=True)
    parser.add_argument("--raw-adapter-fixed-blue-gain", type=float, required=True)
    parser.add_argument("--raw-adapter-variant-policy", required=True, choices=["normal", "dark", "over", "mix"])
    parser.add_argument("--raw-adapter-variant-weights", required=True)
    parser.add_argument("--raw-adapter-fixed-light-scale", type=float, required=True)
    parser.add_argument("--raw-adapter-dark-light-scale-range", nargs=2, type=float, required=True)
    parser.add_argument("--raw-adapter-over-light-scale-range", nargs=2, type=float, required=True)
    parser.add_argument("--raw-adapter-shot-noise", type=float, required=True)
    parser.add_argument("--raw-adapter-read-noise", type=float, required=True)
    parser.add_argument("--raw-adapter-noise-mean-mode", required=True, choices=["zero", "rawadapter_text"])
    parser.add_argument("--raw-adapter-black-level", type=float, required=True)
    parser.add_argument("--raw-adapter-white-level", type=float, required=True)
    parser.add_argument("--raw-adapter-random-seed-policy", required=True, choices=["dataloader_generator", "path_hash"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def check_sample(sample: dict[str, Any], *, expected_hw: tuple[int, int]) -> dict[str, Any]:
    h, w = expected_hw
    expected = {
        "raw": (4, h, w),
        "image": (3, h, w),
        "depth": (h, w),
        "valid_mask": (h, w),
    }
    for key, shape in expected.items():
        actual_shape = tuple(sample[key].shape)
        if actual_shape != shape:
            raise AssertionError(f"{key} shape mismatch: got={actual_shape} expected={shape}")
    raw = sample["raw"]
    if not torch.isfinite(raw).all():
        raise AssertionError("raw tensor contains non-finite values")
    if float(raw.min()) < -1e-7 or float(raw.max()) > 1.0 + 1e-7:
        raise AssertionError(f"raw tensor range invalid: {float(raw.min())}..{float(raw.max())}")
    isp = sample["isp_params"]
    if isp.get("unprocessing_method") != "raw_adapter_style":
        raise AssertionError(f"metadata unprocessing_method mismatch: {isp.get('unprocessing_method')!r}")
    if isp.get("noise_model") != "none" or bool(isp.get("noise_realization_applied")):
        raise AssertionError("fixed raw_adapter_style dataset smoke must not apply noise realization")
    if "hflip_applied" not in isp:
        raise AssertionError("metadata must record hflip_applied")
    return {
        "sample_name": str(sample["sample_name"]),
        "raw_shape": list(sample["raw"].shape),
        "image_shape": list(sample["image"].shape),
        "depth_shape": list(sample["depth"].shape),
        "hflip_applied": bool(isp["hflip_applied"]),
        "raw_adapter_config_hash": str(isp["raw_adapter_config_hash"]),
        "noise_model": str(isp["noise_model"]),
        "noise_realization_applied": bool(isp["noise_realization_applied"]),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if "codex_smoke" not in str(output):
        raise ValueError("--output must contain codex_smoke")
    if args.randomize_unprocessing is None:
        raise ValueError("Pass either --randomize-unprocessing or --no-randomize-unprocessing explicitly.")

    resolved = resolve_unprocessing_config(vars(args))
    train_dataset = VKITTI2Raw(
        filelist_path=args.vkitti_train_list,
        mode="train",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        unprocessing_config=resolved,
        hflip_prob=args.hflip_prob,
        include_rgb_input=True,
        include_rgb_preview=False,
        include_geometry=True,
        raw_storage_format=args.raw_storage_format,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    val_dataset = VKITTI2Raw(
        filelist_path=args.vkitti_val_list,
        mode="val",
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        unprocessing_config=resolved,
        hflip_prob=0.0,
        include_rgb_input=True,
        include_rgb_preview=False,
        include_geometry=True,
        raw_storage_format=args.raw_storage_format,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    train_desc = train_dataset.describe_unprocessing()
    val_desc = val_dataset.describe_unprocessing()
    assert_unprocessing_summaries_compatible(train_desc, val_desc, context="dataset smoke train vs val")
    assert_unprocessing_summary_matches_config(train_desc, resolved, context="dataset smoke train vs config")
    assert_unprocessing_summary_matches_config(val_desc, resolved, context="dataset smoke val vs config")

    max_samples = max(int(args.max_samples), 1)
    train_rows = [
        check_sample(train_dataset.build_sample(i, include_rgb_input=True), expected_hw=(args.input_height, args.input_width))
        for i in range(min(max_samples, len(train_dataset)))
    ]
    val_rows = [
        check_sample(val_dataset.build_sample(i, include_rgb_input=True), expected_hw=(args.input_height, args.input_width))
        for i in range(min(max_samples, len(val_dataset)))
    ]
    payload = {
        "status": "ok",
        "train_len": len(train_dataset),
        "val_len": len(val_dataset),
        "train_unprocessing": train_desc,
        "val_unprocessing": val_desc,
        "train_samples": train_rows,
        "val_samples": val_rows,
    }
    save_json(output, payload)
    print(json.dumps({"status": "ok", "output": str(output), "hash": train_desc["raw_adapter_config_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
