#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_stf.dataset.raw_utils import (
    DEFAULT_RAW_NPZ_ROOT,
    bayer_to_3ch,
    load_rectified_bayer_npz,
    normalize_raw,
    pseudo_rgb_to_bgr,
)
from finetune_stf.dataset.stf import DEFAULT_STF_ROOT, STF, _load_depth_npz
from finetune_stf.dataset.stf_raw import STF_RAW


def parse_args():
    parser = argparse.ArgumentParser(description="Verify STF RAW dataset loading and alignment")
    parser.add_argument("--stf-root", default=DEFAULT_STF_ROOT)
    parser.add_argument("--raw-npz-root", default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "scripts" / "verify_output"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--input-height", type=int, default=518)
    parser.add_argument("--input-width", type=int, default=966)
    parser.add_argument("--norm-mode", default="companded")
    parser.add_argument("--channel-mode", default="rgb_avg_g")
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.set_defaults(use_imagenet_norm=True)
    return parser.parse_args()


def overlay_sparse_depth(bgr_image, depth, *, min_depth=1.0, max_depth=80.0, radius=2):
    vis = bgr_image.copy()
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return vis

    depth_values = depth[ys, xs]
    scaled = np.clip((depth_values - min_depth) / (max_depth - min_depth) * 255.0, 0, 255)
    colors = cv2.applyColorMap(scaled.astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        cv2.circle(vis, (int(x), int(y)), radius, colors[idx].tolist(), -1)
    return vis


def tensor_stats(sample):
    image = sample["image"].float()
    depth = sample["depth"].float()
    valid_mask = sample["valid_mask"]
    return {
        "image_shape": tuple(image.shape),
        "depth_shape": tuple(depth.shape),
        "image_min": float(image.min().item()),
        "image_max": float(image.max().item()),
        "image_mean": float(image.mean().item()),
        "valid_pixels": int(valid_mask.sum().item()),
    }


def select_indices(dataset_size, num_samples):
    num_samples = max(1, min(int(num_samples), int(dataset_size)))
    return np.linspace(0, dataset_size - 1, num_samples, dtype=int).tolist()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    size = (args.input_height, args.input_width)
    rgb_dataset = STF(
        "val",
        stf_root=args.stf_root,
        size=size,
        min_depth=1.0,
        max_depth=80.0,
        merge_test_into_train=False,
    )
    raw_dataset = STF_RAW(
        "val",
        stf_root=args.stf_root,
        raw_npz_root=args.raw_npz_root,
        size=size,
        min_depth=1.0,
        max_depth=80.0,
        merge_test_into_train=False,
        norm_mode=args.norm_mode,
        channel_mode=args.channel_mode,
        use_imagenet_norm=args.use_imagenet_norm,
    )

    if len(rgb_dataset) != len(raw_dataset):
        raise ValueError(f"Dataset length mismatch: rgb={len(rgb_dataset)} raw={len(raw_dataset)}")

    lines = [
        "index\tsample_name\trgb_image_shape\traw_image_shape\trgb_depth_shape\traw_depth_shape\t"
        "rgb_image_min\trgb_image_max\trgb_image_mean\traw_image_min\traw_image_max\traw_image_mean\t"
        "rgb_valid_pixels\traw_valid_pixels"
    ]

    for idx in select_indices(len(raw_dataset), args.num_samples):
        rgb_sample = rgb_dataset[idx]
        raw_sample = raw_dataset[idx]
        if rgb_sample["sample_name"] != raw_sample["sample_name"]:
            raise ValueError(
                f"Sample mismatch at idx={idx}: rgb={rgb_sample['sample_name']} raw={raw_sample['sample_name']}"
            )

        rgb_stats = tensor_stats(rgb_sample)
        raw_stats = tensor_stats(raw_sample)
        lines.append(
            "\t".join(
                [
                    str(idx),
                    raw_sample["sample_name"],
                    str(rgb_stats["image_shape"]),
                    str(raw_stats["image_shape"]),
                    str(rgb_stats["depth_shape"]),
                    str(raw_stats["depth_shape"]),
                    f"{rgb_stats['image_min']:.6f}",
                    f"{rgb_stats['image_max']:.6f}",
                    f"{rgb_stats['image_mean']:.6f}",
                    f"{raw_stats['image_min']:.6f}",
                    f"{raw_stats['image_max']:.6f}",
                    f"{raw_stats['image_mean']:.6f}",
                    str(rgb_stats["valid_pixels"]),
                    str(raw_stats["valid_pixels"]),
                ]
            )
        )

        rgb_bgr = cv2.imread(rgb_sample["image_path"], cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise ValueError(f"Failed to read RGB image: {rgb_sample['image_path']}")

        raw_bayer = load_rectified_bayer_npz(raw_sample["image_path"])
        raw_rgb = bayer_to_3ch(raw_bayer, channel_mode=args.channel_mode)
        raw_rgb = normalize_raw(raw_rgb, norm_mode=args.norm_mode)
        raw_bgr = pseudo_rgb_to_bgr(raw_rgb)
        raw_bgr = cv2.resize(raw_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        depth = _load_depth_npz(raw_sample["depth_path"])
        rgb_overlay = overlay_sparse_depth(rgb_bgr, depth)
        raw_overlay = overlay_sparse_depth(raw_bgr, depth)
        panel = np.hstack([rgb_overlay, raw_overlay])

        cv2.putText(
            panel,
            f"LUT RGB | {raw_sample['sample_name']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "RAW pseudo-RGB",
            (rgb_bgr.shape[1] + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / f"verify_{idx:04d}_{raw_sample['sample_name']}.jpg"), panel)

    stats_path = output_dir / "stats.tsv"
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nSaved verification outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
