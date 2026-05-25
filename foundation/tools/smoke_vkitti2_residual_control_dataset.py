from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.datasets.vkitti2_halfres_rgb_depth import VKITTI2HalfresRGBDepth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test VKITTI2 residual control dataset shapes and keys.")
    parser.add_argument("--vkitti-train-list", required=True)
    parser.add_argument("--vkitti-val-list", required=True)
    parser.add_argument("--input-height", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True)
    parser.add_argument("--fullres-even-policy", required=True, choices=["crop_bottom_to_even"])
    parser.add_argument("--rgb-input-space", required=True, choices=["halfres_2x2_area"])
    parser.add_argument("--depth-target-space", required=True, choices=["halfres_2x2_valid_mean"])
    parser.add_argument("--hflip-prob", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def assert_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise AssertionError(f"{name} shape mismatch: got={actual} expected={expected}")


def check_sample(sample: dict[str, Any], *, expected_hw: tuple[int, int], split: str) -> dict[str, Any]:
    if "raw" in sample:
        raise AssertionError(f"{split} sample unexpectedly contains raw key")
    if "isp_params" in sample:
        raise AssertionError(f"{split} sample unexpectedly contains isp_params key")
    h, w = expected_hw
    assert_shape(f"{split}.image", tuple(sample["image"].shape), (3, h, w))
    assert_shape(f"{split}.depth", tuple(sample["depth"].shape), (h, w))
    assert_shape(f"{split}.valid_mask", tuple(sample["valid_mask"].shape), (h, w))
    assert_shape(f"{split}.rgb_preview", tuple(sample["rgb_preview"].shape), (3, h, w))
    geometry = sample.get("geometry_params")
    if not isinstance(geometry, dict):
        raise AssertionError(f"{split} sample missing geometry_params")
    if geometry.get("original_hw") != [375, 1242]:
        raise AssertionError(f"{split} original_hw mismatch: {geometry.get('original_hw')}")
    if geometry.get("even_fullres_hw") != [374, 1242]:
        raise AssertionError(f"{split} even_fullres_hw mismatch: {geometry.get('even_fullres_hw')}")
    return {
        "sample_name": sample["sample_name"],
        "image_shape": list(sample["image"].shape),
        "depth_shape": list(sample["depth"].shape),
        "valid_shape": list(sample["valid_mask"].shape),
        "rgb_preview_shape": list(sample["rgb_preview"].shape),
        "valid_pixels": int(sample["valid_mask"].sum().item()),
        "geometry": geometry,
        "has_raw": "raw" in sample,
        "has_isp_params": "isp_params" in sample,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    expected_hw = (int(args.input_height), int(args.input_width))
    if expected_hw != (187, 621):
        raise ValueError(f"Control smoke expects halfres size (187, 621), got {expected_hw}")

    train_dataset = VKITTI2HalfresRGBDepth(
        filelist_path=args.vkitti_train_list,
        mode="train",
        size=expected_hw,
        hflip_prob=args.hflip_prob,
        include_geometry=True,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    val_dataset = VKITTI2HalfresRGBDepth(
        filelist_path=args.vkitti_val_list,
        mode="val",
        size=expected_hw,
        hflip_prob=0.0,
        include_geometry=True,
        fullres_even_policy=args.fullres_even_policy,
        rgb_input_space=args.rgb_input_space,
        depth_target_space=args.depth_target_space,
    )
    payload = {
        "status": "ok",
        "train_len": len(train_dataset),
        "val_len": len(val_dataset),
        "train_geometry": train_dataset.describe_geometry(),
        "val_geometry": val_dataset.describe_geometry(),
        "train_sample": check_sample(train_dataset[0], expected_hw=expected_hw, split="train"),
        "val_sample": check_sample(val_dataset[0], expected_hw=expected_hw, split="val"),
    }
    output = Path(args.output).expanduser()
    save_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
