from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.datasets import DEFAULT_TRAIN_LIST, VKITTI2Raw
from foundation.engine.transforms import list_unprocessing_presets


CHANNEL_NAMES = ("R", "Gr", "Gb", "B")


def percentile_summary(values: List[float], ps=(10, 50, 90)) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check VKITTI2 pseudo-raw distribution under one or more unprocessing presets. "
            "Reports p10/p50/p90 for mean_all, sat_ratio, and per-channel mean."
        )
    )
    parser.add_argument("--vkitti-train-list", default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--mode", default="val", choices=["train", "val"])
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--randomize-unprocessing", action="store_true", default=True)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument(
        "--preset",
        action="append",
        default=None,
        help=(
            "Repeatable preset name. If omitted, all presets are evaluated. "
            "Example: --preset eth3d_sensor_linear --preset sensor_linear_dual"
        ),
    )
    parser.add_argument(
        "--dual-mix-weights",
        default=None,
        help=(
            "Optional mix weights for sensor_linear_dual. "
            "Examples: '0.5,0.5' or "
            "'eth3d_sensor_linear=0.3,robotcar_subset100_sensor_linear=0.7'."
        ),
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def evaluate_preset(
    *,
    preset_name: str,
    args: argparse.Namespace,
    sample_indices: List[int],
) -> Dict[str, object]:
    mix_weights = args.dual_mix_weights if preset_name == "sensor_linear_dual" else None
    dataset = VKITTI2Raw(
        filelist_path=args.vkitti_train_list,
        mode=args.mode,
        size=(args.input_height, args.input_width),
        randomize_unprocessing=args.randomize_unprocessing,
        unprocessing_preset=preset_name,
        unprocessing_mix_weights=mix_weights,
        hflip_prob=args.hflip_prob,
    )

    mean_all_values: List[float] = []
    sat_ratio_values: List[float] = []
    per_channel_values: Dict[str, List[float]] = {name: [] for name in CHANNEL_NAMES}
    sub_preset_counter: Dict[str, int] = {}
    by_sub_preset: Dict[str, Dict[str, List[float]]] = {}

    for rank, sample_idx in enumerate(sample_indices):
        seed = args.seed + rank
        py_rng = random.Random(seed)
        torch_generator = torch.Generator(device="cpu")
        torch_generator.manual_seed(seed)

        sample = dataset.build_sample(
            sample_idx % len(dataset),
            py_rng=py_rng,
            torch_generator=torch_generator,
            include_geometry=False,
        )
        raw = sample["raw"].float()
        channel_means = raw.view(4, -1).mean(dim=1)
        mean_all = float(raw.mean().item())
        sat_ratio = float((raw >= (1.0 - 1e-6)).float().mean().item())
        selected_sub = str(sample["isp_params"].get("selected_sub_preset_name", preset_name))

        mean_all_values.append(mean_all)
        sat_ratio_values.append(sat_ratio)
        for idx, name in enumerate(CHANNEL_NAMES):
            per_channel_values[name].append(float(channel_means[idx].item()))

        sub_preset_counter[selected_sub] = sub_preset_counter.get(selected_sub, 0) + 1
        bucket = by_sub_preset.setdefault(
            selected_sub,
            {"mean_all": [], "sat_ratio": []},
        )
        bucket["mean_all"].append(mean_all)
        bucket["sat_ratio"].append(sat_ratio)

    sub_breakdown = {}
    for sub_name, bucket in by_sub_preset.items():
        sub_breakdown[sub_name] = {
            "num_samples": len(bucket["mean_all"]),
            "mean_all": percentile_summary(bucket["mean_all"]),
            "sat_ratio": percentile_summary(bucket["sat_ratio"]),
        }

    return {
        "num_samples": len(sample_indices),
        "preset_description": dataset.describe_unprocessing(),
        "selected_sub_preset_counts": sub_preset_counter,
        "mean_all": percentile_summary(mean_all_values),
        "sat_ratio": percentile_summary(sat_ratio_values),
        "channel_mean": {
            name: percentile_summary(values)
            for name, values in per_channel_values.items()
        },
        "by_sub_preset": sub_breakdown,
    }


def main() -> None:
    args = parse_args()
    preset_list = args.preset or list(list_unprocessing_presets())
    if not preset_list:
        raise ValueError("No presets to evaluate.")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    base_dataset = VKITTI2Raw(
        filelist_path=args.vkitti_train_list,
        mode=args.mode,
        size=(args.input_height, args.input_width),
        randomize_unprocessing=args.randomize_unprocessing,
        unprocessing_preset="stf_legacy",
        hflip_prob=args.hflip_prob,
    )
    sample_count = min(args.num_samples, len(base_dataset))
    sample_indices = list(range(sample_count))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "config": {
            "vkitti_train_list": str(Path(args.vkitti_train_list).expanduser().resolve()),
            "input_hw": [args.input_height, args.input_width],
            "mode": args.mode,
            "hflip_prob": args.hflip_prob,
            "num_samples": sample_count,
            "seed": args.seed,
            "randomize_unprocessing": args.randomize_unprocessing,
            "presets": preset_list,
            "dual_mix_weights": args.dual_mix_weights,
        },
        "results": {},
    }

    for preset_name in preset_list:
        print(f"[CHECK] preset={preset_name} samples={sample_count}", flush=True)
        result = evaluate_preset(
            preset_name=preset_name,
            args=args,
            sample_indices=sample_indices,
        )
        report["results"][preset_name] = result
        print(
            f"[CHECK] preset={preset_name} mean_all={result['mean_all']} sat_ratio={result['sat_ratio']} "
            f"sub_presets={result['selected_sub_preset_counts']}",
            flush=True,
        )

    output_path = args.output_json
    if output_path:
        path = Path(output_path).expanduser().resolve()
    else:
        path = (PROJECT_ROOT / "foundation" / "debug" / "unprocessing_presets_check.json").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
