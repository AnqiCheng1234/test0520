#!/usr/bin/env python3
"""Dump a true-LOD RGB/RAW input sanity panel."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from finetune_stf.dataset.lod_raw import _apply_crop  # noqa: E402
from finetune_stf.dataset.lod_true import (  # noqa: E402
    DEFAULT_LOD_TRUE_MANIFEST,
    DEFAULT_LOD_TRUE_ROOT,
    LODTrueRGBDark,
    LODTrueRawDarkRGB16,
    _load_rgb_u8,
)
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model  # noqa: E402


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
DEFAULT_DAV2S_CHECKPOINT = "/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth"


CHANNEL_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
CHANNEL_LABELS = ("R", "G", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-root", default=DEFAULT_LOD_TRUE_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_LOD_TRUE_MANIFEST)
    parser.add_argument("--split", default="01Valid")
    parser.add_argument("--mode", default="val", choices=("train", "val"))
    parser.add_argument("--crop-mode", default="center", choices=("center", "random"))
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=960)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--dav2s-encoder", default="vits", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--dav2s-checkpoint", default=DEFAULT_DAV2S_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated dataset indices. Defaults to the first --num-samples samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to finetune_stf/analysis/lod_true_input_panel/<MMDD_HHMMSS>.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def resolve_model_state(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value
    return ckpt_obj


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_dav2s_rgb_model(args: argparse.Namespace, device: torch.device, sensor_hw: tuple[int, int]) -> torch.nn.Module:
    checkpoint = Path(args.dav2s_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DAv2-S checkpoint: {checkpoint}")
    base = DepthAnythingV2(**MODEL_CONFIGS[str(args.dav2s_encoder)])
    model = build_dav2_padded_rgb_depth_model(base, sensor_hw=sensor_hw, backbone_hw=None)
    ckpt_obj = torch.load(str(checkpoint), map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    model.load_base_dav2_state_dict(state_dict)
    model.to(device).eval()
    return model


def chw_to_hwc(chw: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(chw):
        chw = chw.detach().cpu().numpy()
    chw = np.asarray(chw, dtype=np.float32)
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got {chw.shape}")
    return np.transpose(chw, (1, 2, 0))


def hwc_to_chw(hwc: np.ndarray) -> np.ndarray:
    hwc = np.asarray(hwc, dtype=np.float32)
    if hwc.ndim != 3 or hwc.shape[2] != 3:
        raise ValueError(f"Expected HWC with 3 channels, got {hwc.shape}")
    return np.transpose(hwc, (2, 0, 1))


def robust_rgb_preview(hwc: np.ndarray) -> np.ndarray:
    """Make an RGB preview while leaving histograms to show true values."""
    hwc = np.asarray(hwc, dtype=np.float32)
    finite = hwc[np.isfinite(hwc)]
    if finite.size == 0:
        return np.zeros_like(hwc)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(hwc)), float(np.nanmax(hwc))
    if hi <= lo:
        return np.zeros_like(hwc)
    return np.clip((hwc - lo) / (hi - lo), 0.0, 1.0)


def colorize_label(label: np.ndarray, valid_mask: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    label = np.asarray(label, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    denom = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((label - float(vmin)) / denom, 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    rgb = cmap(norm)[..., :3].astype(np.float32)
    rgb[~valid_mask] = 0.0
    return rgb


def finite_stats(array: np.ndarray | torch.Tensor) -> dict[str, object]:
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    values = np.asarray(array, dtype=np.float64)
    finite = values[np.isfinite(values)]
    out: dict[str, object] = {
        "shape": [int(v) for v in values.shape],
        "finite_count": int(finite.size),
    }
    if finite.size == 0:
        return out
    quantiles = np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])
    out.update(
        {
            "min": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p05": float(quantiles[2]),
            "p50": float(quantiles[3]),
            "p95": float(quantiles[4]),
            "p99": float(quantiles[5]),
            "max": float(quantiles[6]),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
        }
    )
    return out


def stat_line(stats: dict[str, object]) -> str:
    if "mean" not in stats:
        return "no finite values"
    return (
        f"mean {stats['mean']:.4g}  std {stats['std']:.4g}\n"
        f"p1 {stats['p01']:.4g}  p50 {stats['p50']:.4g}  p99 {stats['p99']:.4g}"
    )


def sample_channel_values(chw: np.ndarray, max_values: int = 200_000) -> list[np.ndarray]:
    chw = np.asarray(chw, dtype=np.float32)
    flat = chw.reshape(3, -1)
    n = flat.shape[1]
    if n > max_values:
        idx = np.linspace(0, n - 1, max_values).astype(np.int64)
        flat = flat[:, idx]
    return [flat[channel][np.isfinite(flat[channel])] for channel in range(3)]


def draw_channel_hist(ax, chw: np.ndarray, *, title: str, xlim: tuple[float, float] = (0.0, 1.0)) -> None:
    for label, color, values in zip(CHANNEL_LABELS, CHANNEL_COLORS, sample_channel_values(chw)):
        if values.size:
            ax.hist(values, bins=80, range=xlim, histtype="step", linewidth=1.1, color=color, label=label)
    ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="x", labelsize=7, pad=1)


def parse_indices(value: str | None, num_samples: int) -> list[int]:
    if value is None:
        return list(range(int(num_samples)))
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--indices was provided but no valid integers were found")
    return indices


def infer_dav2s_prediction(model: torch.nn.Module, image_chw_norm: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        pred = model(image_chw_norm.unsqueeze(0).to(device, non_blocking=True))
    if pred.ndim == 3:
        pred = pred[0]
    elif pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[0, 0]
    else:
        raise ValueError(f"Unexpected DAv2-S prediction shape: {tuple(pred.shape)}")
    pred_np = pred.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(pred_np).all():
        raise ValueError("DAv2-S prediction contains non-finite values")
    return pred_np


def build_samples(args: argparse.Namespace, indices: list[int], model: torch.nn.Module, device: torch.device) -> list[dict[str, object]]:
    size = (int(args.input_height), int(args.input_width))
    rgb_dataset = LODTrueRGBDark(
        lod_root=args.lod_root,
        manifest_path=args.manifest,
        split=args.split,
        size=size,
        mode=args.mode,
        crop_mode=args.crop_mode,
    )
    raw_dataset = LODTrueRawDarkRGB16(
        lod_root=args.lod_root,
        manifest_path=args.manifest,
        split=args.split,
        size=size,
        mode=args.mode,
        crop_mode=args.crop_mode,
        raw_storage_format="raw_rgb16_png_3ch",
        lod_raw_norm_mode="uint16_div_65535",
    )
    if len(rgb_dataset) != len(raw_dataset):
        raise RuntimeError(f"RGB/RAW dataset length mismatch: {len(rgb_dataset)} vs {len(raw_dataset)}")

    samples = []
    for order, idx in enumerate(indices):
        if idx < 0 or idx >= len(rgb_dataset):
            raise IndexError(f"Index {idx} outside split length {len(rgb_dataset)}")
        rng = random.Random(int(args.seed) + idx)
        rgb_sample = rgb_dataset.build_sample(idx, rng=rng, include_geometry=True)
        rng = random.Random(int(args.seed) + idx)
        raw_sample = raw_dataset.build_sample(idx, rng=rng, include_geometry=True)
        if str(rgb_sample["sample_id"]) != str(raw_sample["sample_id"]):
            raise RuntimeError(
                f"RGB/RAW sample_id mismatch at index {idx}: {rgb_sample['sample_id']} vs {raw_sample['sample_id']}"
            )
        crop_box = tuple(int(v) for v in rgb_sample["geometry_params"]["crop_box"])
        raw_crop_box = tuple(int(v) for v in raw_sample["geometry_params"]["crop_box"])
        if crop_box != raw_crop_box:
            raise RuntimeError(f"RGB/RAW crop mismatch at index {idx}: {crop_box} vs {raw_crop_box}")

        row = rgb_dataset.rows[idx]
        rgb_dark_hwc = _apply_crop(_load_rgb_u8(row["rgb_dark_path"]), crop_box)
        rgb_normal_hwc = _apply_crop(_load_rgb_u8(row["rgb_normal_path"]), crop_box)
        raw_preram_chw = raw_sample["raw"].detach().cpu().numpy().astype(np.float32)
        dav2s_pred = infer_dav2s_prediction(model, rgb_sample["image"].detach().float(), device)
        pseudo = raw_sample["depth"].detach().cpu().numpy().astype(np.float32)
        valid = raw_sample["valid_mask"].detach().cpu().numpy().astype(bool)
        samples.append(
            {
                "order": order,
                "index": int(idx),
                "sample_id": str(rgb_sample["sample_id"]),
                "normal_id": int(rgb_sample["normal_id"]),
                "dark_id": int(rgb_sample["dark_id"]),
                "crop_box": [int(v) for v in crop_box],
                "rgb_dark_path": str(row["rgb_dark_path"]),
                "raw_dark_path": str(row["raw_dark_path"]),
                "rgb_normal_path": str(row["rgb_normal_path"]),
                "pseudo_depth_path": str(row["target_path"]),
                "rgb_dark_prenorm_hwc": rgb_dark_hwc,
                "rgb_dark_prenorm_chw": hwc_to_chw(rgb_dark_hwc),
                "raw_preram_chw": raw_preram_chw,
                "raw_preram_hwc": chw_to_hwc(raw_preram_chw),
                "dav2s_rgb_dark_pred": dav2s_pred,
                "pseudo": pseudo,
                "valid": valid,
                "rgb_normal_hwc": rgb_normal_hwc,
                "rgb_normal_chw": hwc_to_chw(rgb_normal_hwc),
            }
        )
    return samples


def write_panel(samples: list[dict[str, object]], output_path: Path, *, dpi: int) -> None:
    valid_values = []
    pred_values = []
    for sample in samples:
        pseudo = sample["pseudo"]
        valid = sample["valid"]
        values = np.asarray(pseudo)[np.asarray(valid, dtype=bool) & np.isfinite(pseudo)]
        if values.size:
            valid_values.append(values)
        pred = np.asarray(sample["dav2s_rgb_dark_pred"], dtype=np.float32)
        pred_finite = pred[np.isfinite(pred)]
        if pred_finite.size:
            pred_values.append(pred_finite)
    if valid_values:
        label_values = np.concatenate(valid_values)
        label_vmin, label_vmax = np.percentile(label_values, [1, 99])
    else:
        label_vmin, label_vmax = 0.0, 1.0
    if pred_values:
        all_pred_values = np.concatenate(pred_values)
        pred_vmin, pred_vmax = np.percentile(all_pred_values, [1, 99])
    else:
        pred_vmin, pred_vmax = 0.0, 1.0

    n = len(samples)
    fig, axes = plt.subplots(
        n,
        8,
        figsize=(27, max(3.0 * n, 6.0)),
        gridspec_kw={"width_ratios": [1.8, 1.35, 1.8, 1.35, 1.8, 1.8, 1.8, 1.35]},
        squeeze=False,
    )
    fig.suptitle(
        "True LOD route-input sanity panel: RGB_Dark pre-ImageNet, RAW_Dark pre-RAM, "
        "DAv2-S RGB_Dark prediction, pseudo label Spectral_r, RGB_normal",
        fontsize=13,
        y=0.995,
    )

    for row_idx, sample in enumerate(samples):
        sample_label = (
            f"idx {sample['index']}  pair {sample['sample_id']}\n"
            f"normal {sample['normal_id']} / dark {sample['dark_id']}"
        )

        rgb_dark_hwc = sample["rgb_dark_prenorm_hwc"]
        raw_hwc = sample["raw_preram_hwc"]
        rgb_normal_hwc = sample["rgb_normal_hwc"]
        dav2s_pred = sample["dav2s_rgb_dark_pred"]
        pseudo = sample["pseudo"]
        valid = sample["valid"]

        ax = axes[row_idx, 0]
        ax.imshow(np.clip(rgb_dark_hwc, 0.0, 1.0))
        ax.set_title(f"{sample_label}\nRGB_Dark pre-ImageNet [0,1]", fontsize=8)
        ax.axis("off")

        rgb_stats = finite_stats(sample["rgb_dark_prenorm_chw"])
        draw_channel_hist(
            axes[row_idx, 1],
            sample["rgb_dark_prenorm_chw"],
            title="RGB_Dark distribution\n" + stat_line(rgb_stats),
        )

        ax = axes[row_idx, 2]
        ax.imshow(robust_rgb_preview(raw_hwc))
        ax.set_title("RAW_Dark pre-RAM /65535\npreview p1-p99, values [0,1]", fontsize=8)
        ax.axis("off")

        raw_stats = finite_stats(sample["raw_preram_chw"])
        draw_channel_hist(
            axes[row_idx, 3],
            sample["raw_preram_chw"],
            title="RAW_Dark distribution\n" + stat_line(raw_stats),
        )

        ax = axes[row_idx, 4]
        pred_valid = np.isfinite(dav2s_pred)
        ax.imshow(colorize_label(dav2s_pred, pred_valid, vmin=pred_vmin, vmax=pred_vmax))
        pred_stats = finite_stats(dav2s_pred)
        ax.set_title(
            "DAv2-S pred on RGB_Dark\n"
            f"Spectral_r p1-p99 {pred_vmin:.4g}-{pred_vmax:.4g}\n"
            + stat_line(pred_stats),
            fontsize=8,
        )
        ax.axis("off")

        ax = axes[row_idx, 5]
        ax.imshow(colorize_label(pseudo, valid, vmin=label_vmin, vmax=label_vmax))
        pseudo_stats = finite_stats(np.asarray(pseudo)[np.asarray(valid, dtype=bool)])
        ax.set_title(
            "pseudo inverse-relative\n"
            f"Spectral_r p1-p99 {label_vmin:.4g}-{label_vmax:.4g}\n"
            + stat_line(pseudo_stats),
            fontsize=8,
        )
        ax.axis("off")

        ax = axes[row_idx, 6]
        ax.imshow(np.clip(rgb_normal_hwc, 0.0, 1.0))
        ax.set_title("RGB_normal source [0,1]", fontsize=8)
        ax.axis("off")

        normal_stats = finite_stats(sample["rgb_normal_chw"])
        draw_channel_hist(
            axes[row_idx, 7],
            sample["rgb_normal_chw"],
            title="RGB_normal distribution\n" + stat_line(normal_stats),
        )

    handles = [
        plt.Line2D([0], [0], color=color, lw=1.5, label=label)
        for label, color in zip(CHANNEL_LABELS, CHANNEL_COLORS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.982))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_stats(samples: list[dict[str, object]], output_path: Path, args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lod_root": str(Path(args.lod_root).expanduser()),
        "manifest": str(Path(args.manifest).expanduser()),
        "split": args.split,
        "mode": args.mode,
        "crop_mode": "center" if args.mode == "val" else args.crop_mode,
        "input_size": [int(args.input_height), int(args.input_width)],
        "rgb_route": "RGB_Dark loaded by true-LOD RGB loader before ImageNet normalization; values are [0,1].",
        "raw_route": "RAW_Dark loaded by true-LOD RAW_RGB16 loader before RamCore3; OpenCV BGR PNG reordered to RGB and divided by 65535.",
        "dav2s_prediction": {
            "source": "RGB_Dark center crop, true-LOD RGB loader ImageNet-normalized tensor, DAv2-S vits padded RGB wrapper.",
            "checkpoint": str(Path(args.dav2s_checkpoint).expanduser()),
            "encoder": str(args.dav2s_encoder),
        },
        "pseudo_label": "RGB_normal DAv2-L inverse-relative pseudo label, visualized with Spectral_r.",
        "samples": [],
    }
    for sample in samples:
        payload["samples"].append(
            {
                "index": sample["index"],
                "sample_id": sample["sample_id"],
                "normal_id": sample["normal_id"],
                "dark_id": sample["dark_id"],
                "crop_box": sample["crop_box"],
                "paths": {
                    "rgb_dark": sample["rgb_dark_path"],
                    "raw_dark": sample["raw_dark_path"],
                    "rgb_normal": sample["rgb_normal_path"],
                    "pseudo_depth": sample["pseudo_depth_path"],
                },
                "stats": {
                    "rgb_dark_prenorm_chw": finite_stats(sample["rgb_dark_prenorm_chw"]),
                    "raw_dark_preram_chw": finite_stats(sample["raw_preram_chw"]),
                    "dav2s_rgb_dark_pred": finite_stats(sample["dav2s_rgb_dark_pred"]),
                    "pseudo_valid": finite_stats(np.asarray(sample["pseudo"])[np.asarray(sample["valid"], dtype=bool)]),
                    "rgb_normal_chw": finite_stats(sample["rgb_normal_chw"]),
                },
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    indices = parse_indices(args.indices, args.num_samples)
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir = ROOT / "finetune_stf" / "analysis" / "lod_true_input_panel" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(args.device))
    model = load_dav2s_rgb_model(args, device, sensor_hw=(int(args.input_height), int(args.input_width)))
    samples = build_samples(args, indices, model, device)
    panel_path = output_dir / "lod_true_rgb_raw_input_panel_5samples.png"
    stats_path = output_dir / "lod_true_rgb_raw_input_panel_5samples.stats.json"
    write_panel(samples, panel_path, dpi=int(args.dpi))
    write_stats(samples, stats_path, args)

    print(f"[OK] panel={panel_path}")
    print(f"[OK] stats={stats_path}")
    print(f"[OK] dav2s_checkpoint={Path(args.dav2s_checkpoint).expanduser().resolve()}")
    print(f"[OK] device={device}")


if __name__ == "__main__":
    main()
