from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundation.engine.datasets import HypersimProcessedRaw


DEFAULT_ETH3D_MANIFEST = Path("/mnt/drive/3333_raw/eth3d_raw_depth_640960/manifests/eth3d_raw_depth_v2_val.csv")
DEFAULT_ROBOTCAR_MANIFEST = Path(
    "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100/manifests/robotcar_raw_depth_v1_val.csv"
)
CHANNEL_NAMES = ("R", "Gr", "Gb", "B")


def raw4_to_rgb(raw_4ch: np.ndarray) -> np.ndarray:
    raw_4ch = np.asarray(raw_4ch, dtype=np.float32)
    if raw_4ch.ndim != 3 or raw_4ch.shape[-1] != 4:
        raise ValueError(f"Expected raw_4ch with shape HxWx4, got {raw_4ch.shape}")
    rgb = np.stack(
        [
            raw_4ch[..., 0],
            0.5 * (raw_4ch[..., 1] + raw_4ch[..., 2]),
            raw_4ch[..., 3],
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def to_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return out
    vals = depth[valid_mask]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[valid_mask] = colored[valid_mask]
    return out


def add_label(image_bgr: np.ndarray, label: str) -> np.ndarray:
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def resize_panel(image_bgr: np.ndarray, height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == height:
        return image_bgr
    new_w = max(int(round(w * height / h)), 1)
    return cv2.resize(image_bgr, (new_w, height), interpolation=cv2.INTER_AREA)


def raw_stats(raw_4ch: np.ndarray) -> Dict[str, object]:
    raw_4ch = np.asarray(raw_4ch, dtype=np.float32)
    flat = raw_4ch.reshape(-1, 4)
    return {
        "mean_all": float(raw_4ch.mean()),
        "sat_ratio": float((raw_4ch >= (1.0 - 1e-6)).mean()),
        "channel_mean": {name: float(flat[:, idx].mean()) for idx, name in enumerate(CHANNEL_NAMES)},
        "percentiles": {
            f"p{p}": float(np.percentile(raw_4ch, p))
            for p in (1, 10, 50, 90, 99)
        },
    }


def read_first_manifest_raw(manifest_path: Path, raw_key: str) -> Tuple[np.ndarray, str]:
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = Path(row[raw_key])
            if not raw_path.is_file():
                continue
            with np.load(raw_path, allow_pickle=False) as data:
                key = "raw_4ch" if "raw_4ch" in data.files else data.files[0]
                raw = np.asarray(data[key], dtype=np.float32)
            return np.clip(raw, 0.0, 1.0), str(raw_path)
    raise FileNotFoundError(f"No readable raw npz found in {manifest_path}")


def summarize_stats(items: Iterable[Dict[str, object]]) -> Dict[str, object]:
    items = list(items)
    if not items:
        return {}
    return {
        "mean_all": {
            "mean": float(np.mean([item["mean_all"] for item in items])),
            "min": float(np.min([item["mean_all"] for item in items])),
            "max": float(np.max([item["mean_all"] for item in items])),
        },
        "sat_ratio": {
            "mean": float(np.mean([item["sat_ratio"] for item in items])),
            "min": float(np.min([item["sat_ratio"] for item in items])),
            "max": float(np.max([item["sat_ratio"] for item in items])),
        },
        "channel_mean": {
            name: float(np.mean([item["channel_mean"][name] for item in items]))
            for name in CHANNEL_NAMES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize HyperSim pseudo-RAW with real RAW references.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--randomize-unprocessing", action="store_true", default=True)
    parser.add_argument("--no-randomize-unprocessing", action="store_false", dest="randomize_unprocessing")
    parser.add_argument("--unprocessing-preset", default="sensor_linear_dual")
    parser.add_argument("--unprocessing-mix-weights", default=None)
    parser.add_argument("--eth3d-manifest", type=Path, default=DEFAULT_ETH3D_MANIFEST)
    parser.add_argument("--robotcar-manifest", type=Path, default=DEFAULT_ROBOTCAR_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HypersimProcessedRaw(
        split="train",
        mode="val",
        size=(args.input_height, args.input_width),
        randomize_unprocessing=args.randomize_unprocessing,
        unprocessing_preset=args.unprocessing_preset,
        unprocessing_mix_weights=args.unprocessing_mix_weights,
        hflip_prob=0.0,
    )
    eth3d_raw, eth3d_path = read_first_manifest_raw(args.eth3d_manifest, "raw_640_path")
    robotcar_raw, robotcar_path = read_first_manifest_raw(args.robotcar_manifest, "raw_eval_path")

    eth3d_panel = add_label(to_u8(raw4_to_rgb(eth3d_raw))[..., ::-1], "ETH3D real RAW ref")
    robotcar_panel = add_label(to_u8(raw4_to_rgb(robotcar_raw))[..., ::-1], "RobotCar real RAW ref")
    real_ref_stats = {
        "eth3d": {"path": eth3d_path, **raw_stats(eth3d_raw)},
        "robotcar": {"path": robotcar_path, **raw_stats(robotcar_raw)},
    }

    sample_stats: List[Dict[str, object]] = []
    sub_preset_counts: Dict[str, int] = {}
    for rank in range(min(args.num_samples, len(dataset))):
        py_rng = random.Random(args.seed + rank)
        torch_generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
        sample = dataset.build_sample(rank, py_rng=py_rng, torch_generator=torch_generator, include_geometry=True)
        raw = sample["raw"].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
        depth = sample["depth"].cpu().numpy().astype(np.float32, copy=False)
        valid = sample["valid_mask"].cpu().numpy().astype(bool)
        rgb_preview = sample["rgb_preview"].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
        rgb_panel = to_u8(rgb_preview)[..., ::-1]
        raw_panel = to_u8(raw4_to_rgb(raw))[..., ::-1]
        depth_panel = colorize_depth(depth, valid)

        panels = [
            add_label(rgb_panel, "HyperSim RGB (same crop)"),
            add_label(raw_panel, "HyperSim pseudo RAW"),
            add_label(depth_panel, "HyperSim z-depth"),
            resize_panel(eth3d_panel, args.input_height),
            resize_panel(robotcar_panel, args.input_height),
        ]
        canvas = np.concatenate(panels, axis=1)
        out_path = args.output_dir / f"hypersim_pseudoraw_{rank:03d}.png"
        cv2.imwrite(str(out_path), canvas)

        stats = raw_stats(raw)
        stats["image_path"] = sample["image_path"]
        stats["depth_path"] = sample["depth_path"]
        stats["output_path"] = str(out_path)
        stats["geometry_params"] = sample["geometry_params"]
        selected_sub = str(sample["isp_params"].get("selected_sub_preset_name", "unknown"))
        stats["selected_sub_preset_name"] = selected_sub
        sub_preset_counts[selected_sub] = sub_preset_counts.get(selected_sub, 0) + 1
        sample_stats.append(stats)

    report = {
        "config": {
            "num_samples": len(sample_stats),
            "input_hw": [args.input_height, args.input_width],
            "unprocessing_preset": args.unprocessing_preset,
            "unprocessing_mix_weights": args.unprocessing_mix_weights,
            "randomize_unprocessing": args.randomize_unprocessing,
            "seed": args.seed,
        },
        "selected_sub_preset_counts": sub_preset_counts,
        "hypersim_summary": summarize_stats(sample_stats),
        "real_references": real_ref_stats,
        "samples": sample_stats,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[DONE] wrote {len(sample_stats)} visualizations to {args.output_dir}")
    print(f"[DONE] wrote {summary_path}")


if __name__ == "__main__":
    main()
