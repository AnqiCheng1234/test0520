#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.dataset.eth3d import ETH3DValRaw
from finetune_stf.dataset.robotcar import RobotCarValRaw
from finetune_stf.models.spatial_adapter import build_dav2_padded_rgb_depth_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFICIAL_VITL_CKPT = "/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"
METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "silog", "silog_x100", "d1", "d2", "d3")
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class SplitConfig:
    name: str
    dataset_kind: str
    sensor_hw: tuple[int, int]
    min_depth: float
    max_depth: float
    fast_eval_backend: str
    root: str
    manifest_name: str | None = None


SPLITS: dict[str, SplitConfig] = {
    "eth3d_fast": SplitConfig(
        name="eth3d_fast",
        dataset_kind="eth3d",
        sensor_hw=(640, 960),
        min_depth=0.1,
        max_depth=80.0,
        fast_eval_backend="proxy",
        root="/mnt/drive/3333_raw/eth3d_raw_depth_640960",
    ),
    "robotcar_day_fast": SplitConfig(
        name="robotcar_day_fast",
        dataset_kind="robotcar",
        sensor_hw=(480, 640),
        min_depth=0.1,
        max_depth=50.0,
        fast_eval_backend="sparse",
        root="/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100",
    ),
    "robotcar_night_fast": SplitConfig(
        name="robotcar_night_fast",
        dataset_kind="robotcar",
        sensor_hw=(480, 640),
        min_depth=0.1,
        max_depth=50.0,
        fast_eval_backend="sparse",
        root="/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo",
        manifest_name="robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAW-as-RGB zero-shot baseline: packed raw -> R,(Gr+Gb)/2,B -> official DAv2 RGB wrapper."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(SPLITS.keys()),
        help="Dataset split config.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for summary.json and optional sanity panel.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Head-N samples in manifest order. Omit for full eval.",
    )
    parser.add_argument(
        "--encoder",
        default="vitl",
        choices=sorted(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--pretrained-from",
        default=OFFICIAL_VITL_CKPT,
        help="Official DAv2 checkpoint path.",
    )
    parser.add_argument(
        "--save-sanity-panel",
        action="store_true",
        help="Save one sanity panel (raw preview / pseudo-RGB / GT / pred / error).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP inference.")
    return parser.parse_args()


def unwrap_model_state(state_obj):
    if isinstance(state_obj, dict) and "model" in state_obj and isinstance(state_obj["model"], dict):
        state_obj = state_obj["model"]
    if isinstance(state_obj, dict) and state_obj and all(str(k).startswith("module.") for k in state_obj):
        state_obj = {k[len("module."):]: v for k, v in state_obj.items()}
    return state_obj


def build_eval_dataset(cfg: SplitConfig):
    common = dict(
        depth_mode="fast",
        fast_eval_backend=cfg.fast_eval_backend,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
        norm_mode="sensor_linear",
        channel_mode="rgb_avg_g",
        use_imagenet_norm=True,
        input_mode="raw_naive",
    )
    if cfg.dataset_kind == "eth3d":
        return ETH3DValRaw(eth3d_root=cfg.root, **common)
    if cfg.dataset_kind == "robotcar":
        kwargs = dict(robotcar_root=cfg.root, **common)
        if cfg.manifest_name is not None:
            kwargs["manifest_name"] = cfg.manifest_name
        return RobotCarValRaw(**kwargs)
    raise ValueError(f"Unsupported dataset_kind: {cfg.dataset_kind}")


def build_model(encoder: str, ckpt_path: Path, sensor_hw: tuple[int, int]):
    dav2 = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model = build_dav2_padded_rgb_depth_model(
        dav2,
        sensor_hw=sensor_hw,
        backbone_hw=None,
    )
    state_dict = unwrap_model_state(torch.load(ckpt_path, map_location="cpu"))
    model.load_base_dav2_state_dict(state_dict)
    return model.to(DEVICE).eval()


def infer_batched(model, image_tensor: torch.Tensor, target_hw: tuple[int, int], *, use_amp: bool) -> np.ndarray:
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    amp_enabled = bool(use_amp and DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(image_tensor).float()
    if tuple(pred.shape[-2:]) != tuple(target_hw):
        pred = F.interpolate(pred[:, None], target_hw, mode="bilinear", align_corners=True)[:, 0]
    return pred[0].detach().cpu().numpy()


def summarize_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {key: float("nan") for key in METRIC_KEYS}
    return {
        key: float(sum(record[key] for record in records) / len(records))
        for key in METRIC_KEYS
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def denorm_imagenet(image_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(image_chw, (1, 2, 0))
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def colorize_depth(depth: np.ndarray, valid_mask: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    value = np.clip(depth.astype(np.float32), vmin, vmax)
    denom = max(float(vmax - vmin), 1e-6)
    norm = (value - vmin) / denom
    rgb = cv2.applyColorMap((norm * 255.0).round().astype(np.uint8), cv2.COLORMAP_PLASMA)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~valid_mask.astype(bool)] = 0
    return rgb


def build_raw_preview_from_npz(sample: dict, row: dict, dataset_kind: str) -> np.ndarray:
    if dataset_kind == "eth3d":
        raw_path = Path(row["raw_640_path"]).expanduser().resolve()
        with np.load(raw_path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_4ch"], dtype=np.float32)
    elif dataset_kind == "robotcar":
        from finetune_stf.dataset.robotcar import _canonicalize_raw_channel_order, load_rectified_bayer_npz  # local import

        raw_path = Path(row["raw_eval_path"]).expanduser().resolve()
        raw = load_rectified_bayer_npz(raw_path)
        raw = _canonicalize_raw_channel_order(raw, row.get("pack_order", "[R,Gr,Gb,B]"))
    else:
        raise ValueError(f"Unsupported dataset kind for preview: {dataset_kind}")
    preview = np.stack(
        [raw[..., 0], 0.5 * (raw[..., 1] + raw[..., 2]), raw[..., 3]],
        axis=-1,
    )
    preview = np.clip(preview, 0.0, 1.0) ** (1.0 / 2.2)
    return (preview * 255.0).round().astype(np.uint8)


def save_sanity_panel(
    panel_path: Path,
    sample: dict,
    row: dict,
    dataset_kind: str,
    depth: np.ndarray,
    valid: np.ndarray,
    aligned_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> None:
    raw_preview = build_raw_preview_from_npz(sample, row, dataset_kind)
    pseudo_rgb = denorm_imagenet(np.asarray(sample["image"].cpu(), dtype=np.float32))
    pseudo_rgb = (pseudo_rgb * 255.0).round().astype(np.uint8)

    gt = colorize_depth(depth, valid, vmin=min_depth, vmax=max_depth)
    pred = colorize_depth(aligned_depth, np.isfinite(aligned_depth), vmin=min_depth, vmax=max_depth)
    abs_err = np.abs(aligned_depth - depth).astype(np.float32)
    err_valid = valid & np.isfinite(abs_err)
    err_vmax = float(np.percentile(abs_err[err_valid], 95.0)) if np.any(err_valid) else 5.0
    err_vmax = max(err_vmax, 1e-3)
    err = colorize_depth(abs_err, err_valid, vmin=0.0, vmax=err_vmax)

    target_hw = pseudo_rgb.shape[:2]

    def _resize_tile(tile: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        if tile.shape[:2] == hw:
            return tile
        return cv2.resize(tile, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)

    tiles = [_resize_tile(tile, target_hw) for tile in [raw_preview, pseudo_rgb, gt, pred, err]]
    labels = ["raw_preview", "pseudo_rgb_input", "gt_depth", "pred_depth", "abs_error"]
    h, w, _ = tiles[0].shape
    header_h = 28
    canvas = np.full((h + header_h, w * len(tiles), 3), 255, dtype=np.uint8)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x0 = i * w
        canvas[header_h:, x0 : x0 + w] = tile
        cv2.putText(canvas, label, (x0 + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    Image.fromarray(canvas).save(panel_path)


def main() -> int:
    args = parse_args()
    split_cfg = SPLITS[args.split]
    output_dir = args.output_dir.expanduser().resolve()
    ckpt_path = Path(args.pretrained_from).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0 when provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_eval_dataset(split_cfg)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    model = build_model(args.encoder, ckpt_path, sensor_hw=split_cfg.sensor_hw)

    records = []
    processed = 0
    sanity_saved = False
    print(
        f"[raw_as_rgb_zeroshot] split={split_cfg.name} samples={len(indices)} "
        f"backend={split_cfg.fast_eval_backend} ckpt={ckpt_path}",
        flush=True,
    )

    for count, idx in enumerate(indices, start=1):
        sample = dataset[idx]
        image = sample["image"]
        expected_shape = (3, split_cfg.sensor_hw[0], split_cfg.sensor_hw[1])
        if tuple(image.shape) != expected_shape:
            raise RuntimeError(
                f"{split_cfg.name} sample {sample.get('sample_name', idx)} image shape mismatch: "
                f"got {tuple(image.shape)} expected {expected_shape}"
            )

        depth = np.asarray(sample["depth"], dtype=np.float32)
        valid = np.asarray(sample["valid_mask"]).astype(bool)
        pred_disp = infer_batched(model, image, depth.shape[-2:], use_amp=not args.no_amp)
        aligned_depth, _ = affine_align_disp(depth, pred_disp, valid)
        metrics = compute_metrics(
            depth,
            aligned_depth,
            valid,
            min_depth=split_cfg.min_depth,
            max_depth=split_cfg.max_depth,
        )
        if metrics is None:
            continue
        record = {key: float(metrics[key]) for key in METRIC_KEYS}
        records.append(record)
        processed += 1

        if args.save_sanity_panel and not sanity_saved:
            save_sanity_panel(
                output_dir / "sanity_panel_5up_first_sample.jpg",
                sample=sample,
                row=dataset.rows[idx],
                dataset_kind=split_cfg.dataset_kind,
                depth=depth,
                valid=valid,
                aligned_depth=aligned_depth,
                min_depth=split_cfg.min_depth,
                max_depth=split_cfg.max_depth,
            )
            sanity_saved = True

        if count == 1 or count % 50 == 0 or count == len(indices):
            print(
                f"[raw_as_rgb_zeroshot] processed {count}/{len(indices)} "
                f"(valid_metrics={processed}) sample={sample['sample_name']}",
                flush=True,
            )

    summary = {
        "protocol": "dav2_raw_avg_g_rgb_zeroshot",
        "raw_to_rgb": "R,(Gr+Gb)/2,B",
        "split": split_cfg.name,
        "dataset_size": int(dataset_size),
        "processed_samples": int(processed),
        "min_depth": float(split_cfg.min_depth),
        "max_depth": float(split_cfg.max_depth),
        "fast_eval_backend": split_cfg.fast_eval_backend,
        "pretrained_from": str(ckpt_path),
        "encoder": args.encoder,
        "sensor_hw": [int(split_cfg.sensor_hw[0]), int(split_cfg.sensor_hw[1])],
        "root": str(Path(split_cfg.root).expanduser().resolve()),
        "manifest_name": split_cfg.manifest_name,
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "metrics": summarize_metrics(records),
        "saved_sanity_panel": bool(sanity_saved),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
