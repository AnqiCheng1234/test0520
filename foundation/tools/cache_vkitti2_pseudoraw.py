from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.datasets import DEFAULT_TRAIN_LIST, VKITTI2Raw


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache one deterministic VKITTI2 pseudo-RAW set.")
    parser.add_argument("--filelist-path", default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", default="train", choices=["train", "val"])
    parser.add_argument("--input-height", type=int, default=518)
    parser.add_argument("--input-width", type=int, default=966)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--randomize-unprocessing", action="store_true", default=True)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument(
        "--vkitti-unprocessing-preset",
        default="sensor_linear_dual",
        help=(
            "Pseudo-raw preset for VKITTI2Raw. "
            "Supported presets include: stf_legacy, eth3d_sensor_linear, "
            "robotcar_public_gbrg_generic (public GBRG + generic ranges), "
            "robotcar_subset100_sensor_linear, robotcar_subset100_sensor_linear_fixccm, "
            "robotcar_night_sensor_linear, sensor_linear_dual, "
            "robotcar_day_night_sensor_linear_dual. The robotcar_subset100*, "
            "robotcar_night_sensor_linear, sensor_linear_dual, and "
            "robotcar_day_night_sensor_linear_dual presets use RobotCar statistics "
            "and are not public-only."
        ),
    )
    parser.add_argument(
        "--vkitti-unprocessing-mix-weights",
        default=None,
        help=(
            "Optional mix weights for dual preset. "
            "Examples: '0.3,0.7' or "
            "'eth3d_sensor_linear=0.3,robotcar_subset100_sensor_linear=0.7'."
        ),
    )
    parser.add_argument("--hflip-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--raw-dtype", choices=sorted(TORCH_DTYPES), default="float16")
    parser.add_argument("--depth-dtype", choices=sorted(TORCH_DTYPES), default="float16")
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate output size and validate free space without creating the cache.",
    )
    parser.add_argument(
        "--capacity-threshold",
        type=float,
        default=0.7,
        help="Abort when estimated bytes exceed this fraction of available bytes.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_config(args: argparse.Namespace, dataset: VKITTI2Raw, num_samples: int) -> Dict[str, Any]:
    unproc_desc = dataset.describe_unprocessing()
    return {
        "cache_version": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "filelist_path": str(dataset.filelist_path),
        "num_samples": num_samples,
        "mode": dataset.mode,
        "size": list(dataset.size),
        "min_depth": dataset.min_depth,
        "max_depth": dataset.max_depth,
        "randomize_unprocessing": bool(unproc_desc["active_transform"]["randomize"]),
        "hflip_prob": dataset.hflip_prob,
        "seed": args.seed,
        "sample_seed_formula": "seed + idx",
        "storage": {
            "format": "torch_pt_per_sample",
            "raw_dtype": args.raw_dtype,
            "depth_dtype": args.depth_dtype,
            "valid_mask_dtype": "uint8",
        },
        "unprocessing": {
            "preset_name": unproc_desc["unprocessing_preset"],
            "preset_kind": unproc_desc["unprocessing_kind"],
            "preset_version": unproc_desc["preset_version"],
            "preset_hash": unproc_desc["preset_hash"],
            "isp_profile_group": unproc_desc["isp_profile_group"],
            "sub_presets": unproc_desc["sub_presets"],
            "default_sub_preset": unproc_desc["default_sub_preset"],
            "mix_weights": unproc_desc["mix_weights"],
            "active_transform": to_jsonable(unproc_desc["active_transform"]),
            "sub_preset_transforms": to_jsonable(unproc_desc["sub_preset_transforms"]),
        },
        "notes": [
            "This cache freezes one crop/flip/unprocessing realization per source image.",
            "Online training remains available through foundation.engine.datasets.VKITTI2Raw.",
        ],
    }


def dtype_nbytes(dtype_name: str) -> int:
    return torch.empty((), dtype=TORCH_DTYPES[dtype_name]).element_size()


def estimate_payload_bytes(args: argparse.Namespace, num_samples: int) -> int:
    raw_bytes = 4 * dtype_nbytes(args.raw_dtype)
    depth_bytes = dtype_nbytes(args.depth_dtype)
    valid_mask_bytes = 1
    return int(num_samples) * int(args.input_height) * int(args.input_width) * (
        raw_bytes + depth_bytes + valid_mask_bytes
    )


def find_existing_parent(path: Path) -> Path:
    current = path.expanduser().resolve()
    if current.exists():
        return current
    for parent in current.parents:
        if parent.exists():
            return parent
    raise FileNotFoundError(f"No existing parent found for {path}")


def build_capacity_report(args: argparse.Namespace, output_dir: Path, num_samples: int) -> Dict[str, Any]:
    disk_path = find_existing_parent(output_dir)
    usage = shutil.disk_usage(disk_path)
    estimated_bytes = estimate_payload_bytes(args, num_samples)
    threshold_bytes = int(usage.free * float(args.capacity_threshold))
    return {
        "output_dir": str(output_dir),
        "disk_path": str(disk_path),
        "num_samples": int(num_samples),
        "size": [int(args.input_height), int(args.input_width)],
        "raw_dtype": args.raw_dtype,
        "depth_dtype": args.depth_dtype,
        "estimated_bytes": int(estimated_bytes),
        "available_bytes": int(usage.free),
        "capacity_threshold": float(args.capacity_threshold),
        "threshold_bytes": int(threshold_bytes),
        "would_abort": bool(estimated_bytes > threshold_bytes),
    }


def check_capacity_or_raise(args: argparse.Namespace, output_dir: Path, num_samples: int) -> Dict[str, Any]:
    report = build_capacity_report(args, output_dir, num_samples)
    print("[CAPACITY] " + json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["would_abort"]:
        raise RuntimeError(
            "Estimated cache payload exceeds capacity threshold: "
            f"estimated_bytes={report['estimated_bytes']} "
            f"available_bytes={report['available_bytes']} "
            f"threshold={report['capacity_threshold']}"
        )
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    samples_dir = output_dir / "samples"
    manifest_path = output_dir / "manifest.jsonl"
    config_path = output_dir / "config.json"

    dataset = VKITTI2Raw(
        filelist_path=args.filelist_path,
        mode=args.mode,
        size=(args.input_height, args.input_width),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        randomize_unprocessing=args.randomize_unprocessing,
        unprocessing_preset=args.vkitti_unprocessing_preset,
        unprocessing_mix_weights=args.vkitti_unprocessing_mix_weights,
        hflip_prob=args.hflip_prob,
    )
    num_samples = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
    capacity_report = check_capacity_or_raise(args, output_dir, num_samples)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **capacity_report}, indent=2, ensure_ascii=False))
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output dir must be empty or absent: {output_dir}")

    samples_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args, dataset, num_samples)
    config["capacity"] = capacity_report
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    raw_dtype = TORCH_DTYPES[args.raw_dtype]
    depth_dtype = TORCH_DTYPES[args.depth_dtype]

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for idx in range(num_samples):
            sample_seed = args.seed + idx
            py_rng = random.Random(sample_seed)
            torch_generator = torch.Generator(device="cpu")
            torch_generator.manual_seed(sample_seed)

            sample = dataset.build_sample(
                idx,
                py_rng=py_rng,
                torch_generator=torch_generator,
                include_geometry=True,
            )

            cache_filename = f"{idx:06d}_{sample['sample_name']}.pt"
            cache_path = samples_dir / cache_filename
            payload = {
                "raw": sample["raw"].to(dtype=raw_dtype).cpu().contiguous(),
                "depth": sample["depth"].to(dtype=depth_dtype).cpu().contiguous(),
                "valid_mask": sample["valid_mask"].to(dtype=torch.uint8).cpu().contiguous(),
                "sample_name": sample["sample_name"],
                "image_path": sample["image_path"],
                "depth_path": sample["depth_path"],
            }
            torch.save(payload, cache_path)

            record = {
                "idx": idx,
                "sample_seed": sample_seed,
                "sample_name": sample["sample_name"],
                "cache_path": str(cache_path),
                "image_path": sample["image_path"],
                "depth_path": sample["depth_path"],
                "raw_shape": list(sample["raw"].shape),
                "depth_shape": list(sample["depth"].shape),
                "geometry_params": to_jsonable(sample["geometry_params"]),
                "isp_params": to_jsonable(sample["isp_params"]),
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % args.log_interval == 0 or (idx + 1) == num_samples:
                print(
                    f"[CACHE] {idx + 1}/{num_samples} "
                    f"sample={sample['sample_name']} "
                    f"pattern={sample['isp_params']['cfa_pattern']}"
                )
                manifest_file.flush()

    summary = {
        "num_samples": num_samples,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
