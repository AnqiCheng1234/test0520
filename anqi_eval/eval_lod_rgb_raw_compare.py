#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
from matplotlib import colormaps
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_robotcar_rgb_raw_compare import DEVICE, build_raw_model, infer_batched
from finetune_stf.dataset.raw_utils import normalize_raw_4ch
from finetune_stf.models.raw_ram import packed_bayer_to_base_rgb


DEFAULT_LOD_ROOT = Path("/mnt/drive/3333_raw/LOD")
DEFAULT_DAY_MANIFEST = DEFAULT_LOD_ROOT / "pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv"
DEFAULT_NIGHT_MANIFEST = DEFAULT_LOD_ROOT / "pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest.csv"
DEFAULT_HEAVY_ROOT = Path("/mnt/drive/3333_raw/0000_exp_ckpt_186")
LOD_NATIVE_HW = (928, 1440)
ROBOTCAR_VIS_HW = (480, 640)
REQUIRED_COLUMNS = ("split", "sample_name", "rgb_path", "rggb_path", "output_npy")
DEPTH_CMAP = colormaps["Spectral_r"]
DIFF_CMAP = colormaps["inferno"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare existing LOD DAV2 RGB pseudo-depth with a RAW checkpoint on a day/night subset."
    )
    parser.add_argument("exp_dir", type=Path, help="Experiment directory containing config.json.")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help='"best", "last", or a custom .pth path. For heavy-save experiments, best/last are also searched in the heavy checkpoint root.',
    )
    parser.add_argument("--heavy-root", type=Path, default=DEFAULT_HEAVY_ROOT)
    parser.add_argument("--day-manifest", type=Path, default=DEFAULT_DAY_MANIFEST)
    parser.add_argument("--night-manifest", type=Path, default=DEFAULT_NIGHT_MANIFEST)
    parser.add_argument("--split", default="01Valid", help='LOD split to sample from. Use "any" to ignore split.')
    parser.add_argument("--day-count", type=int, default=5)
    parser.add_argument("--night-count", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--eval-height",
        type=int,
        default=ROBOTCAR_VIS_HW[0],
        help="RAW/DAV2 comparison height. Defaults to RobotCar visualization height.",
    )
    parser.add_argument(
        "--eval-width",
        type=int,
        default=ROBOTCAR_VIS_HW[1],
        help="RAW/DAV2 comparison width. Defaults to RobotCar visualization width.",
    )
    parser.add_argument(
        "--spatial-mode",
        default="resize",
        choices=["resize", "center_crop"],
        help="How to map LOD native 928x1440 arrays to eval size. RobotCar-consistent default is resize.",
    )
    parser.add_argument("--norm-mode", default="sensor_linear")
    parser.add_argument(
        "--raw-preview-mode",
        default="direct",
        choices=["direct", "percentile"],
        help=(
            "Display-only RAW panel exposure. direct matches RobotCar/ETH3D linear RAW preview; "
            "percentile rescales each LOD RAW panel for visibility without changing model input."
        ),
    )
    parser.add_argument("--raw-preview-low-pct", type=float, default=0.5)
    parser.add_argument("--raw-preview-high-pct", type=float, default=99.7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--diff-vmax-pct", type=float, default=99.0)
    parser.add_argument(
        "--panel-layout",
        default="5up",
        choices=["4up", "5up", "ram5up", "ram6up"],
        help=(
            "4up saves rgb/raw_base/rgb_pred/raw_pred. 5up also appends relative difference. "
            "ram5up/ram6up insert the post-RAM clamped RGB panel after raw_base."
        ),
    )
    parser.add_argument(
        "--vis-space",
        default="relative_depth",
        choices=["inverse", "relative_depth"],
        help="Colorize predictions in aligned inverse/disparity space, or after converting positive disparity to relative depth.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def resolve_checkpoint(exp_dir: Path, checkpoint_arg: str, heavy_root: Path) -> Path:
    if checkpoint_arg in {"best", "last"}:
        name = "best_model.pth" if checkpoint_arg == "best" else "last_epoch_model.pth"
        candidates = [
            exp_dir / name,
            heavy_root.expanduser().resolve() / exp_dir.name / name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        checked = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"Missing {checkpoint_arg} checkpoint. Checked:\n{checked}")
    path = Path(checkpoint_arg).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    rows = []
    with path.expanduser().open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [name for name in REQUIRED_COLUMNS if name not in set(reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        for row in reader:
            if split != "any" and row["split"] != split:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No LOD rows found in {path} for split={split!r}")
    return rows


def select_rows(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    count = min(int(count), len(rows))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(rows), size=count, replace=False)
    return [rows[int(idx)] for idx in sorted(indices.tolist())]


def center_crop_box(height: int, width: int, target_h: int, target_w: int) -> tuple[int, int, int, int]:
    if target_h > height or target_w > width:
        raise ValueError(f"Requested crop {(target_h, target_w)} exceeds input shape {(height, width)}")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return top, left, target_h, target_w


def apply_crop(array: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, height, width = crop_box
    return np.ascontiguousarray(array[top : top + height, left : left + width, ...])


def resize_array(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    interpolation = cv2.INTER_AREA if target_h <= array.shape[0] and target_w <= array.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(array, (target_w, target_h), interpolation=interpolation).astype(np.float32, copy=False)


def prepare_eval_array(array: np.ndarray, target_hw: tuple[int, int], spatial_mode: str) -> np.ndarray:
    target_h, target_w = target_hw
    if tuple(array.shape[:2]) == (target_h, target_w):
        return np.ascontiguousarray(array)
    if spatial_mode == "resize":
        return np.ascontiguousarray(resize_array(array, target_hw))
    if spatial_mode == "center_crop":
        return apply_crop(array, center_crop_box(array.shape[0], array.shape[1], target_h, target_w))
    raise ValueError(f"Unsupported spatial_mode: {spatial_mode}")


def raw_to_tensor(raw: np.ndarray, norm_mode: str) -> torch.Tensor:
    raw = normalize_raw_4ch(raw.astype(np.float32, copy=False), norm_mode=norm_mode)
    return torch.from_numpy(np.ascontiguousarray(raw.transpose(2, 0, 1)))


def raw_to_base_rgb(raw: np.ndarray) -> np.ndarray:
    rgb = np.stack(
        [raw[..., 0], 0.5 * (raw[..., 1] + raw[..., 2]), raw[..., 3]],
        axis=-1,
    ).astype(np.float32, copy=False)
    return rgb


def build_raw_preview(
    raw: np.ndarray,
    *,
    mode: str,
    low_pct: float,
    high_pct: float,
) -> Image.Image:
    rgb = raw_to_base_rgb(raw)
    if mode == "direct":
        # Match RobotCar/ETH3D visualization: direct linear RAW in [0,1], gamma for display.
        rgb = np.clip(rgb, 0.0, 1.0)
    elif mode == "percentile":
        valid = np.isfinite(rgb)
        if np.any(valid):
            low, high = np.percentile(rgb[valid], [low_pct, high_pct])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = 0.0, max(float(np.nanmax(rgb)), 1e-6)
            rgb = (rgb - float(low)) / max(float(high - low), 1e-6)
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported raw preview mode: {mode}")
    rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8))


def build_rgb_image(rgb: np.ndarray) -> Image.Image:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8))


def infer_ram_rgb(model, image_tensor: torch.Tensor, *, use_amp: bool = True) -> np.ndarray:
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "ram_core"):
        raise ValueError("The loaded model does not expose ram_core, so post-RAM RGB cannot be visualized.")

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE, non_blocking=True).float()
    amp_enabled = use_amp and DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        base_rgb = packed_bayer_to_base_rgb(image_tensor)
        ram_rgb, _ = module.ram_core.forward_with_features(base_rgb)
        ram_rgb = torch.clamp(ram_rgb, min=0.0, max=1.0)
    return ram_rgb[0].float().detach().cpu().permute(1, 2, 0).numpy()


def resize_rgb_to_raw_size(rgb_path: Path, target_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(rgb_path).convert("RGB")
    source_size = image.size
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    return image, source_size


def robust_limits(values: list[np.ndarray], valid: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(value, dtype=np.float32)[valid].reshape(-1) for value in values])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(flat, low_pct))
    vmax = float(np.percentile(flat, high_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(flat)), float(np.max(flat))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def colorize(values: np.ndarray, valid: np.ndarray, *, vmin: float, vmax: float, cmap=DEPTH_CMAP) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    norm = np.clip((values - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    rgb = np.round(cmap(norm)[:, :, :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def affine_align_to_reference(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(ref) & np.isfinite(pred)
    if int(mask.sum()) < 2:
        return pred.astype(np.float32), {"scale": 1.0, "shift": 0.0}
    x = pred[mask].reshape(-1)
    y = ref[mask].reshape(-1)
    A = np.stack([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale = float(coef[0])
    shift = float(coef[1])
    return (pred * scale + shift).astype(np.float32), {"scale": scale, "shift": shift}


def proxy_metrics(reference: np.ndarray, raw_aligned: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float32)
    pred = np.asarray(raw_aligned, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(ref) & np.isfinite(pred)
    diff = pred[mask] - ref[mask]
    ref_values = ref[mask]
    pred_values = pred[mask]
    if diff.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "corr": float("nan")}
    corr = float(np.corrcoef(ref_values, pred_values)[0, 1]) if diff.size > 1 else float("nan")
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "corr": corr,
    }


def disparity_to_relative_depth(disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = 1.0 / disparity[valid]
    return depth, valid


def relative_error_map(reference_depth: np.ndarray, pred_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference_depth, dtype=np.float32)
    pred = np.asarray(pred_depth, dtype=np.float32)
    valid = np.isfinite(ref) & np.isfinite(pred) & (ref > 1e-6)
    err = np.zeros_like(ref, dtype=np.float32)
    err[valid] = np.abs(pred[valid] - ref[valid]) / np.clip(ref[valid], 1e-6, None)
    return err, valid


def build_panel(images: list[Image.Image], labels: list[str], footer: str) -> Image.Image:
    panel_w, panel_h = images[0].size
    header_h = 32
    footer_h = 26
    canvas = Image.new("RGB", (panel_w * len(images), panel_h + header_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(zip(images, labels)):
        x0 = idx * panel_w
        if image.size != (panel_w, panel_h):
            image = image.resize((panel_w, panel_h), Image.Resampling.BILINEAR)
        canvas.paste(image, (x0, header_h))
        draw.text((x0 + 10, 8), label, fill="black")
    draw.text((10, header_h + panel_h + 5), footer, fill="black")
    return canvas


def write_selected_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "order_index",
        "group",
        "split",
        "sample_name",
        "rgb_path",
        "rggb_path",
        "dav2_output_npy",
        "raw_pred_npy",
        "raw_pred_aligned_npy",
        "ram_rgb_npy",
        "panel_path",
        "source_rgb_width",
        "source_rgb_height",
        "eval_width",
        "eval_height",
        "align_scale",
        "align_shift",
        "mae_to_dav2_after_affine",
        "rmse_to_dav2_after_affine",
        "corr_to_dav2_after_affine",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    cfg_path = exp_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    cfg = load_json(cfg_path)
    checkpoint_path = resolve_checkpoint(exp_dir, str(args.checkpoint), args.heavy_root)

    eval_h = int(args.eval_height)
    eval_w = int(args.eval_width)
    if eval_h <= 0 or eval_w <= 0:
        raise ValueError("--eval-height/--eval-width must be positive")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else exp_dir / (
            f"lod_day{args.day_count}_night{args.night_count}_rgb_raw_compare_"
            f"{Path(checkpoint_path).stem}_{args.spatial_mode}_{args.vis_space}_"
            f"rawvis-{args.raw_preview_mode}_{args.panel_layout}_{eval_h}x{eval_w}"
        )
    )
    panel_dir = output_dir / f"panels_{args.panel_layout}"
    raw_pred_dir = output_dir / "raw_predictions"
    raw_pred_aligned_dir = output_dir / "raw_predictions_affine_to_dav2"
    include_ram_rgb = args.panel_layout in {"ram5up", "ram6up"}
    ram_rgb_dir = output_dir / "ram_rgb_clamped"
    for path in (panel_dir, raw_pred_dir, raw_pred_aligned_dir):
        path.mkdir(parents=True, exist_ok=True)
    if include_ram_rgb:
        ram_rgb_dir.mkdir(parents=True, exist_ok=True)

    day_rows = select_rows(read_manifest(args.day_manifest, args.split), args.day_count, args.sample_seed)
    night_rows = select_rows(read_manifest(args.night_manifest, args.split), args.night_count, args.sample_seed + 1000)
    selected = [("day", row) for row in day_rows] + [("night", row) for row in night_rows]
    if not selected:
        raise ValueError("No selected LOD samples.")

    print(
        f"[setup] checkpoint={checkpoint_path} samples={len(selected)} split={args.split} "
        f"eval_hw=({eval_h},{eval_w}) spatial_mode={args.spatial_mode} output_dir={output_dir}",
        flush=True,
    )
    print("[setup] DAV2 RGB pseudo-depth is loaded from existing manifest output_npy files.", flush=True)

    model = build_raw_model(cfg, checkpoint_path)
    records = []
    metric_records = []
    target_size = (eval_w, eval_h)

    for order_index, (group, row) in enumerate(selected, start=1):
        sample_name = row["sample_name"]
        raw_path = Path(row["rggb_path"]).expanduser().resolve()
        dav2_path = Path(row["output_npy"]).expanduser().resolve()
        rgb_path = Path(row["rgb_path"]).expanduser().resolve()
        raw = np.load(raw_path).astype(np.float32, copy=False)
        dav2_pred = np.load(dav2_path).astype(np.float32, copy=False)
        if tuple(raw.shape[:2]) != tuple(LOD_NATIVE_HW) or raw.shape[-1] != 4:
            raise ValueError(f"{sample_name}: expected raw shape {(*LOD_NATIVE_HW, 4)}, got {tuple(raw.shape)}")
        if tuple(dav2_pred.shape) != tuple(LOD_NATIVE_HW):
            raise ValueError(f"{sample_name}: expected DAV2 pseudo shape {LOD_NATIVE_HW}, got {tuple(dav2_pred.shape)}")

        raw_crop = prepare_eval_array(raw, (eval_h, eval_w), args.spatial_mode)
        dav2_crop = prepare_eval_array(dav2_pred, (eval_h, eval_w), args.spatial_mode)
        valid = np.isfinite(dav2_crop) & (dav2_crop > 1e-6)

        image_tensor = raw_to_tensor(raw_crop, norm_mode=args.norm_mode)
        ram_rgb = infer_ram_rgb(model, image_tensor) if include_ram_rgb else None
        raw_pred = infer_batched(model, image_tensor, dav2_crop.shape[-2:]).astype(np.float32, copy=False)
        raw_aligned, align_stats = affine_align_to_reference(dav2_crop, raw_pred, valid)
        metrics = proxy_metrics(dav2_crop, raw_aligned, valid)
        if args.vis_space == "inverse":
            dav2_vis_values = dav2_crop
            raw_vis_values = raw_aligned
            dav2_vis_valid = valid
            raw_vis_valid = valid & np.isfinite(raw_aligned) & (raw_aligned > 1e-6)
            shared_vis_valid = dav2_vis_valid & raw_vis_valid
            vis_label_suffix = "inv"
        else:
            dav2_vis_values, dav2_vis_valid = disparity_to_relative_depth(dav2_crop)
            raw_vis_values, raw_vis_valid = disparity_to_relative_depth(raw_aligned)
            shared_vis_valid = dav2_vis_valid & raw_vis_valid
            vis_label_suffix = "depth"

        safe_name = sample_name.replace("/", "__")
        raw_pred_path = raw_pred_dir / f"{order_index:04d}_{safe_name}_raw_pred.npy"
        raw_aligned_path = raw_pred_aligned_dir / f"{order_index:04d}_{safe_name}_raw_pred_affine_to_dav2.npy"
        np.save(raw_pred_path, raw_pred.astype(np.float32, copy=False))
        np.save(raw_aligned_path, raw_aligned.astype(np.float32, copy=False))
        ram_rgb_path = None
        if ram_rgb is not None:
            ram_rgb_path = ram_rgb_dir / f"{order_index:04d}_{safe_name}_ram_rgb_clamped.npy"
            np.save(ram_rgb_path, ram_rgb.astype(np.float32, copy=False))

        rgb_image, source_rgb_size = resize_rgb_to_raw_size(rgb_path, target_size)
        raw_preview = build_raw_preview(
            raw_crop,
            mode=args.raw_preview_mode,
            low_pct=args.raw_preview_low_pct,
            high_pct=args.raw_preview_high_pct,
        )
        vmin, vmax = robust_limits([dav2_vis_values, raw_vis_values], shared_vis_valid)
        dav2_vis = colorize(dav2_vis_values, dav2_vis_valid, vmin=vmin, vmax=vmax)
        raw_vis = colorize(raw_vis_values, raw_vis_valid, vmin=vmin, vmax=vmax)
        diff, diff_valid = relative_error_map(dav2_vis_values, raw_vis_values)
        diff_vmax = float(np.percentile(diff[diff_valid], args.diff_vmax_pct)) if np.any(diff_valid) else 1.0
        if not np.isfinite(diff_vmax) or diff_vmax <= 0:
            diff_vmax = 1.0
        diff_vis = colorize(diff, diff_valid, vmin=0.0, vmax=diff_vmax, cmap=DIFF_CMAP)

        footer = (
            f"{group} {sample_name} | rgb_src={source_rgb_size[0]}x{source_rgb_size[1]} "
            f"raw/eval={eval_w}x{eval_h} | raw affine-to-DAV2 rmse={metrics['rmse']:.4f} corr={metrics['corr']:.4f}"
        )
        panel_images = [rgb_image, raw_preview]
        panel_labels = ["rgb", "raw_base"]
        if ram_rgb is not None:
            panel_images.append(build_rgb_image(ram_rgb))
            panel_labels.append("ram_rgb")
        panel_images.extend([dav2_vis, raw_vis])
        panel_labels.extend([f"dav2 rgb_pred_{vis_label_suffix}", f"raw_pred_{vis_label_suffix}"])
        if args.panel_layout in {"5up", "ram6up"}:
            panel_images.append(diff_vis)
            panel_labels.append("rel diff")
        panel = build_panel(panel_images, panel_labels, footer)
        panel_path = panel_dir / f"{order_index:04d}_{safe_name}_{args.panel_layout}.jpg"
        panel.save(panel_path, quality=95)

        record = {
            "order_index": order_index,
            "group": group,
            "split": row["split"],
            "sample_name": sample_name,
            "rgb_path": str(rgb_path),
            "rggb_path": str(raw_path),
            "dav2_output_npy": str(dav2_path),
            "raw_pred_npy": str(raw_pred_path),
            "raw_pred_aligned_npy": str(raw_aligned_path),
            "ram_rgb_npy": str(ram_rgb_path) if ram_rgb_path is not None else "",
            "panel_path": str(panel_path),
            "source_rgb_width": int(source_rgb_size[0]),
            "source_rgb_height": int(source_rgb_size[1]),
            "eval_width": eval_w,
            "eval_height": eval_h,
            "align_scale": float(align_stats["scale"]),
            "align_shift": float(align_stats["shift"]),
            "mae_to_dav2_after_affine": float(metrics["mae"]),
            "rmse_to_dav2_after_affine": float(metrics["rmse"]),
            "corr_to_dav2_after_affine": float(metrics["corr"]),
        }
        records.append(record)
        metric_records.append(metrics)
        print(
            f"[sample] {order_index}/{len(selected)} {group} {sample_name} "
            f"rgb_src={source_rgb_size[0]}x{source_rgb_size[1]} raw_eval={eval_w}x{eval_h} "
            f"rmse_to_dav2={metrics['rmse']:.4f} corr={metrics['corr']:.4f}",
            flush=True,
        )

    summary = {
        "exp_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "pretrained_from": cfg.get("pretrained_from"),
        "input_type": cfg.get("input_type"),
        "split": args.split,
        "day_manifest": str(args.day_manifest.expanduser().resolve()),
        "night_manifest": str(args.night_manifest.expanduser().resolve()),
        "day_count": int(args.day_count),
        "night_count": int(args.night_count),
        "sample_seed": int(args.sample_seed),
        "eval_hw": [eval_h, eval_w],
        "lod_native_hw": list(LOD_NATIVE_HW),
        "spatial_mode": args.spatial_mode,
        "vis_space": args.vis_space,
        "raw_preview_mode": args.raw_preview_mode,
        "raw_preview_low_pct": float(args.raw_preview_low_pct),
        "raw_preview_high_pct": float(args.raw_preview_high_pct),
        "dav2_rgb_pseudo_source": "existing_manifest_output_npy",
        "panel_dir": str(panel_dir),
        "panel_layout": panel_labels if records else [],
        "ram_rgb_dir": str(ram_rgb_dir) if include_ram_rgb else None,
        "depth_colormap": "Spectral_r",
        "diff_colormap": "inferno" if args.panel_layout in {"5up", "ram6up"} else None,
        "target_space": (
            "LOD DAV2 pseudo labels are inverse_relative/disparity; raw predictions are affine-aligned to "
            "DAV2 in inverse/disparity space before both predictions are colorized."
            if args.vis_space == "inverse"
            else "LOD DAV2 pseudo labels are inverse_relative/disparity; raw predictions are affine-aligned to "
            "DAV2 in inverse/disparity space, then both positive disparity maps are inverted to relative depth "
            "before colorizing."
        ),
        "raw_preview_policy": (
            "direct: RobotCar/ETH3D-compatible [R,(Gr+Gb)/2,B], clip [0,1], gamma 1/2.2; "
            "percentile: display-only per-panel percentile exposure on the same base RGB before gamma."
        ),
        "ram_rgb_policy": (
            "For ram5up/ram6up, ram_rgb is the model's post-RamCore3, clamp([0,1]) output before ImageNet normalization; "
            "this is the 3-channel tensor that is fed to the DAv2 frontend."
        ),
        "effective_raw_frontend": "raw_ram_rgb_bridge: model receives normalized 4ch packed RGGB, then internally uses [R,(Gr+Gb)/2,B] with RamCore3.",
        "rgb_size_policy": "RGB images are resized to the RAW eval size for display; RGB display/input is never larger than RAW eval size.",
        "raw_input_policy": "RAW model input uses the same eval-size array as the DAV2 pseudo-depth after the selected spatial_mode.",
        "mean_mae_to_dav2_after_affine": float(np.nanmean([m["mae"] for m in metric_records])),
        "mean_rmse_to_dav2_after_affine": float(np.nanmean([m["rmse"] for m in metric_records])),
        "mean_corr_to_dav2_after_affine": float(np.nanmean([m["corr"] for m in metric_records])),
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "panel_manifest.json", records)
    write_selected_manifest(output_dir / "selected_manifest.csv", records)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
