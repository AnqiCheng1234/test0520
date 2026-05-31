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

from foundation.engine.datasets.led_hb import (
    LEDHBHalfresRGBDepth,
    LEDHBRaw,
    default_led_hb_raw_adapter_config,
)
from foundation.tools.residual_training_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test LED-HB RGB/depth and RAW dataset shapes.")
    parser.add_argument("--led-train-list", required=True)
    parser.add_argument("--led-val-list", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--min-depth", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--led-geometry-mode", required=True, choices=["resize_fullres_756x1344_then_halfres_378x672"])
    parser.add_argument("--raw-storage-format", required=True, choices=["synthetic_packed_bayer_4ch_halfres"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    values = tensor.float()
    if mask is not None:
        values = values[mask.bool()]
    finite = torch.isfinite(values)
    if values.numel() == 0:
        return {"finite_ratio": 0.0, "min": None, "max": None, "mean": None}
    finite_values = values[finite]
    return {
        "finite_ratio": float(finite.float().mean().item()),
        "min": None if finite_values.numel() == 0 else float(finite_values.min().item()),
        "max": None if finite_values.numel() == 0 else float(finite_values.max().item()),
        "mean": None if finite_values.numel() == 0 else float(finite_values.mean().item()),
    }


def assert_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise RuntimeError(f"{name} shape mismatch: got={tuple(tensor.shape)} expected={expected}")


def inspect_rgb_sample(sample: dict[str, Any], *, expected_hw: tuple[int, int]) -> dict[str, Any]:
    assert_shape("image", sample["image"], (3, *expected_hw))
    assert_shape("rgb_preview", sample["rgb_preview"], (3, *expected_hw))
    assert_shape("depth", sample["depth"], expected_hw)
    assert_shape("valid_mask", sample["valid_mask"], expected_hw)
    valid_pixels = int(sample["valid_mask"].sum().item())
    if valid_pixels <= 0:
        raise RuntimeError(f"Sample has no valid pixels: {sample['sample_name']}")
    depth_valid = sample["depth"][sample["valid_mask"]]
    if not bool(torch.isfinite(depth_valid).all().item()):
        raise RuntimeError(f"Sample has non-finite valid depth: {sample['sample_name']}")
    return {
        "sample_name": sample["sample_name"],
        "image_path": sample["image_path"],
        "valid_pixels": valid_pixels,
        "valid_ratio": float(sample["valid_mask"].float().mean().item()),
        "depth": tensor_stats(sample["depth"], sample["valid_mask"]),
        "rgb_preview": tensor_stats(sample["rgb_preview"]),
        "geometry": sample.get("geometry_params", {}),
    }


def inspect_raw_sample(sample: dict[str, Any], *, expected_hw: tuple[int, int]) -> dict[str, Any]:
    rgb_report = inspect_rgb_sample(
        {
            **sample,
            "image": sample["image"],
            "rgb_preview": sample["rgb_preview"],
        },
        expected_hw=expected_hw,
    )
    assert_shape("raw", sample["raw"], (4, *expected_hw))
    if not bool(torch.isfinite(sample["raw"]).all().item()):
        raise RuntimeError(f"RAW has non-finite values: {sample['sample_name']}")
    rgb_report["raw"] = tensor_stats(sample["raw"])
    rgb_report["isp_params_keys"] = sorted(str(key) for key in sample.get("isp_params", {}).keys())
    return rgb_report


def main() -> None:
    args = parse_args()
    expected_hw = (int(args.input_height), int(args.input_width))
    common = {
        "size": expected_hw,
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "led_geometry_mode": args.led_geometry_mode,
    }
    raw_config = default_led_hb_raw_adapter_config()
    datasets = {
        "rgb_train": LEDHBHalfresRGBDepth(args.led_train_list, mode="train", hflip_prob=0.0, include_geometry=True, **common),
        "rgb_val": LEDHBHalfresRGBDepth(args.led_val_list, mode="val", hflip_prob=0.0, include_geometry=True, **common),
        "raw_train": LEDHBRaw(
            args.led_train_list,
            mode="train",
            hflip_prob=0.0,
            include_rgb_input=True,
            include_rgb_preview=True,
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            unprocessing_config=raw_config,
            **common,
        ),
        "raw_val": LEDHBRaw(
            args.led_val_list,
            mode="val",
            hflip_prob=0.0,
            include_rgb_input=True,
            include_rgb_preview=True,
            include_geometry=True,
            raw_storage_format=args.raw_storage_format,
            unprocessing_config=raw_config,
            **common,
        ),
    }
    summary: dict[str, Any] = {
        "lengths": {name: len(dataset) for name, dataset in datasets.items()},
        "geometry": {name: dataset.describe_geometry() for name, dataset in datasets.items()},
        "samples": {},
        "raw_unprocessing": datasets["raw_train"].describe_unprocessing(),
    }
    summary["samples"]["rgb_train_0"] = inspect_rgb_sample(datasets["rgb_train"][0], expected_hw=expected_hw)
    summary["samples"]["rgb_val_0"] = inspect_rgb_sample(datasets["rgb_val"][0], expected_hw=expected_hw)
    summary["samples"]["raw_train_0"] = inspect_raw_sample(datasets["raw_train"][0], expected_hw=expected_hw)
    summary["samples"]["raw_val_0"] = inspect_raw_sample(datasets["raw_val"][0], expected_hw=expected_hw)

    save_json(Path(args.output).expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
