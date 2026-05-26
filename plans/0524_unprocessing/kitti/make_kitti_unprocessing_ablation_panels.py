#!/usr/bin/env python3
"""Create per-sample panels comparing KITTI RAW-like unprocessing configs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from make_kitti_unprocessing_viz_dump import draw_distribution_block, load_raw_packed, load_rgb


DEFAULT_CONFIGS = ["baseline", "ccm_only", "tone_only", "ccm_tone"]
DEFAULT_VARIANTS = ["normal", "dark", "over"]
DAV2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
DEFAULT_DAV2_CHECKPOINT_DIRS = [
    Path("checkpoints"),
    Path("/home/caq/333_cvpr/da_ours/checkpoints"),
    Path("/mnt/drive/3333_raw/checkpoints"),
]
CONFIG_LABELS = {
    "baseline": "baseline",
    "ccm_only": "ccm only",
    "tone_only": "tone only",
    "ccm_tone": "ccm tone",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample-level config ablation panels.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument(
        "--raw-output-base",
        type=Path,
        required=True,
        help="Base path before the config suffix, e.g. .../rawadapter_unproc_image02_s12",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--strategy", choices=["linspace", "first", "random"], default="linspace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--thumb-width", type=int, default=400)
    parser.add_argument("--thumb-height", type=int, default=120)
    parser.add_argument("--include-distribution", action="store_true")
    parser.add_argument("--dist-height", type=int, default=220)
    parser.add_argument("--include-dav2-depth", action="store_true")
    parser.add_argument("--dav2-encoder", choices=sorted(DAV2_MODEL_CONFIGS), default="vits")
    parser.add_argument("--dav2-checkpoint", type=Path, default=None)
    parser.add_argument("--dav2-input-size", type=int, default=518)
    parser.add_argument(
        "--dav2-depth-height",
        type=int,
        default=0,
        help="Use 0 to match --thumb-height.",
    )
    parser.add_argument("--dav2-depth-cmap", default="Spectral_r")
    parser.add_argument(
        "--dav2-raw-rgb-mode",
        choices=["merge_g"],
        default="merge_g",
        help="How to convert packed 4-channel raw to RGB before DAV2 inference.",
    )
    parser.add_argument(
        "--include-gt-depth",
        action="store_true",
        help="Add a top reference row with GT depth from sample_manifest depth_path.",
    )
    parser.add_argument(
        "--gt-depth-height",
        type=int,
        default=0,
        help="Use 0 to match --thumb-height.",
    )
    parser.add_argument(
        "--gt-depth-cmap",
        default=None,
        help="Colormap for GT depth. Defaults to --dav2-depth-cmap.",
    )
    parser.add_argument(
        "--include-baseline-error-map",
        action="store_true",
        help="Add inverse-depth AbsRel error maps for the baseline row's DAV2 depth outputs.",
    )
    parser.add_argument(
        "--error-map-height",
        type=int,
        default=0,
        help="Use 0 to match --thumb-height.",
    )
    parser.add_argument("--error-map-cmap", default="magma")
    parser.add_argument(
        "--error-map-max",
        type=float,
        default=1.0,
        help="AbsRel value mapped to the top of the error colormap.",
    )
    return parser.parse_args()


def sample_name(sample: dict) -> str:
    return sample.get("sample_name") or f"{sample['split']}_{sample['drive']}_{sample['camera']}_{sample['frame']}"


def select_samples(samples: Sequence[dict], count: int, strategy: str, seed: int) -> list[dict]:
    samples = list(samples)
    if count <= 0 or count >= len(samples):
        return samples
    if strategy == "first":
        return samples[:count]
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(samples, count), key=sample_name)
    if strategy == "linspace":
        if count == 1:
            return [samples[len(samples) // 2]]
        last = len(samples) - 1
        indices = sorted({round(i * last / (count - 1)) for i in range(count)})
        return [samples[i] for i in indices]
    raise ValueError(f"Unknown strategy: {strategy}")


def raw_output_dir(raw_output_base: Path, config: str) -> Path:
    return raw_output_base.parent / f"{raw_output_base.name}_{config}"


def preview_path(raw_dir: Path, variant: str, link_path: Path, sample_root: Path) -> Path:
    rel = link_path.relative_to(sample_root).with_suffix(".png")
    return raw_dir / "preview" / variant / rel


def npz_path(raw_dir: Path, variant: str, link_path: Path, sample_root: Path) -> Path:
    rel = link_path.relative_to(sample_root).with_suffix(".npz")
    return raw_dir / variant / rel


def resolve_dav2_checkpoint(checkpoint: Path | None, encoder: str) -> Path:
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing DAV2 checkpoint: {path}")
        return path

    filename = f"depth_anything_v2_{encoder}.pth"
    for root in DEFAULT_DAV2_CHECKPOINT_DIRS:
        candidate = (root / filename).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    candidates = ", ".join(str((root / filename).expanduser()) for root in DEFAULT_DAV2_CHECKPOINT_DIRS)
    raise FileNotFoundError(f"Missing DAV2 checkpoint for encoder={encoder}; checked: {candidates}")


def dav2_label(encoder: str) -> str:
    return {"vits": "dav2-s", "vitb": "dav2-b", "vitl": "dav2-l", "vitg": "dav2-g"}.get(encoder, f"dav2-{encoder}")


def dav2_file_stem(encoder: str) -> str:
    return {"vits": "dav2s", "vitb": "dav2b", "vitl": "dav2l", "vitg": "dav2g"}.get(encoder, f"dav2_{encoder}")


def raw_packed_to_rgb(raw_packed: np.ndarray, mode: str) -> np.ndarray:
    raw = np.asarray(raw_packed, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 4:
        raise ValueError(f"Expected packed raw [4,H,W], got {raw.shape}")
    if mode != "merge_g":
        raise ValueError(f"Unsupported raw RGB mode: {mode}")
    red, green_1, green_2, blue = raw
    green = 0.5 * (green_1 + green_2)
    return np.stack([red, green, blue], axis=-1)


def rgb_float_to_uint8(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image.shape}")
    if image.dtype == np.uint8:
        return image
    image = np.asarray(image, dtype=np.float32)
    return np.asarray(np.clip(image, 0.0, 1.0) * 255.0, dtype=np.uint8)


def load_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def fit_pil_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, size, method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (245, 245, 245))
    left = (size[0] - fitted.width) // 2
    top = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image for panel: {path}")
    image = Image.open(path).convert("RGB")
    return fit_pil_image(image, size)


def draw_pil_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    *,
    title: str,
    x: int,
    y: int,
    size: tuple[int, int],
) -> None:
    fitted = fit_pil_image(image, size)
    sheet.paste(fitted, (x, y))
    draw.rectangle((x, y, x + min(size[0], 230), y + 20), fill=(0, 0, 0))
    draw.text((x + 5, y + 4), title[:42], fill=(255, 255, 255), font=ImageFont.load_default())


def draw_empty_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    size: tuple[int, int],
) -> None:
    draw.rectangle((x, y, x + size[0], y + size[1]), fill=(255, 255, 255), outline=(210, 210, 210))


def draw_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    path: Path,
    *,
    title: str,
    x: int,
    y: int,
    size: tuple[int, int],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image for panel: {path}")
    draw_pil_cell(sheet, draw, Image.open(path).convert("RGB"), title=title, x=x, y=y, size=size)


class Dav2DepthRenderer:
    def __init__(
        self,
        *,
        encoder: str,
        checkpoint: Path,
        input_size: int,
        cmap_name: str,
        raw_rgb_mode: str,
    ) -> None:
        import matplotlib
        import torch

        from depth_anything_v2.dpt import DepthAnythingV2

        self.encoder = encoder
        self.checkpoint = checkpoint
        self.input_size = int(input_size)
        self.raw_rgb_mode = raw_rgb_mode
        self.cmap_name = cmap_name
        self.cmap = matplotlib.colormaps.get_cmap(cmap_name)
        self.cache: dict[str, Image.Image] = {}
        self.depth_cache: dict[str, np.ndarray] = {}

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = DepthAnythingV2(**DAV2_MODEL_CONFIGS[encoder])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = model.to(device).eval()
        self.device = device

    def colorize_depth(self, depth: np.ndarray) -> Image.Image:
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth)
        if not np.any(valid):
            color = np.zeros((*depth.shape, 3), dtype=np.uint8)
            return Image.fromarray(color, mode="RGB")
        lo = float(np.min(depth[valid]))
        hi = float(np.max(depth[valid]))
        denom = max(hi - lo, 1e-12)
        norm = np.clip((depth - lo) / denom, 0.0, 1.0)
        color = np.asarray(self.cmap(norm)[:, :, :3] * 255.0, dtype=np.uint8)
        return Image.fromarray(color, mode="RGB")

    def infer(self, cache_key: str, load_image_rgb: Callable[[], np.ndarray]) -> np.ndarray:
        if cache_key in self.depth_cache:
            return self.depth_cache[cache_key].copy()
        image_rgb = rgb_float_to_uint8(load_image_rgb())
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        depth = self.model.infer_image(image_bgr, self.input_size)
        depth = np.asarray(depth, dtype=np.float32)
        self.depth_cache[cache_key] = depth
        return depth.copy()

    def render(self, cache_key: str, output_path: Path, load_image_rgb: Callable[[], np.ndarray]) -> Image.Image:
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        depth = self.infer(cache_key, load_image_rgb)
        depth_image = self.colorize_depth(depth)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_image.save(output_path)
        self.cache[cache_key] = depth_image
        return depth_image.copy()

    def render_rgb_file(self, path: Path, output_path: Path) -> Image.Image:
        return self.render(f"rgb:{path.resolve()}", output_path, lambda: load_rgb_uint8(path))

    def render_raw_file(self, path: Path, output_path: Path) -> Image.Image:
        cache_key = f"raw:{path.resolve()}:{self.raw_rgb_mode}"
        return self.render(cache_key, output_path, lambda: raw_packed_to_rgb(load_raw_packed(path), self.raw_rgb_mode))

    def infer_rgb_file(self, path: Path) -> np.ndarray:
        return self.infer(f"rgb:{path.resolve()}", lambda: load_rgb_uint8(path))

    def infer_raw_file(self, path: Path) -> np.ndarray:
        cache_key = f"raw:{path.resolve()}:{self.raw_rgb_mode}"
        return self.infer(cache_key, lambda: raw_packed_to_rgb(load_raw_packed(path), self.raw_rgb_mode))


class GtDepthRenderer:
    def __init__(self, *, cmap_name: str) -> None:
        import matplotlib

        self.cmap_name = cmap_name
        self.cmap = matplotlib.colormaps.get_cmap(cmap_name)
        self.cache: dict[str, Image.Image] = {}

    def colorize_depth(self, depth: np.ndarray, valid: np.ndarray) -> Image.Image:
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(depth)
        color = np.zeros((*depth.shape, 3), dtype=np.uint8)
        if not np.any(valid):
            return Image.fromarray(color, mode="RGB")

        lo = float(np.min(depth[valid]))
        hi = float(np.max(depth[valid]))
        denom = max(hi - lo, 1e-12)
        norm = np.clip((depth - lo) / denom, 0.0, 1.0)
        colorized = np.asarray(self.cmap(norm)[:, :, :3] * 255.0, dtype=np.uint8)
        color[valid] = colorized[valid]
        return Image.fromarray(color, mode="RGB")

    def render_file(self, path: Path, output_path: Path) -> Image.Image:
        cache_key = f"gt:{path.resolve()}:{self.cmap_name}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()

        raw = np.asarray(Image.open(path))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        depth = raw.astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if np.issubdtype(raw.dtype, np.integer):
            valid &= raw < np.iinfo(raw.dtype).max
        depth_image = self.colorize_depth(depth, valid)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_image.save(output_path)
        self.cache[cache_key] = depth_image
        return depth_image.copy()


class ErrorMapRenderer:
    def __init__(self, *, cmap_name: str, vmax: float) -> None:
        import matplotlib

        if vmax <= 0:
            raise ValueError(f"error-map-max must be > 0, got {vmax}")
        self.cmap_name = cmap_name
        self.cmap = matplotlib.colormaps.get_cmap(cmap_name)
        self.vmax = float(vmax)
        self.cache: dict[str, Image.Image] = {}

    @staticmethod
    def load_gt_inverse(path: Path) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(Image.open(path))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        valid = np.isfinite(raw) & (raw > 0)
        if np.issubdtype(raw.dtype, np.integer):
            valid &= raw < np.iinfo(raw.dtype).max
        depth = raw.astype(np.float32)
        inverse = np.zeros_like(depth, dtype=np.float32)
        inverse[valid] = 1.0 / depth[valid]
        return inverse, valid

    @staticmethod
    def resize_to(array: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
        if tuple(array.shape[:2]) == tuple(shape_hw):
            return np.asarray(array, dtype=np.float32)
        image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
        resized = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)

    @staticmethod
    def align_to_gt_inverse(pred: np.ndarray, gt_inverse: np.ndarray, valid: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt_inverse, dtype=np.float64)
        mask = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
        if int(mask.sum()) < 10:
            raise ValueError(f"Too few valid pixels for error-map alignment: {int(mask.sum())}")
        x = pred[mask].reshape(-1)
        y = gt[mask].reshape(-1)
        design = np.stack([x, np.ones_like(x)], axis=1)
        scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
        return np.asarray(scale * pred + shift, dtype=np.float32)

    def colorize(self, error: np.ndarray, valid: np.ndarray) -> Image.Image:
        error = np.asarray(error, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(error)
        color = np.zeros((*error.shape, 3), dtype=np.uint8)
        if not np.any(valid):
            return Image.fromarray(color, mode="RGB")
        norm = np.clip(error / self.vmax, 0.0, 1.0)
        colorized = np.asarray(self.cmap(norm)[:, :, :3] * 255.0, dtype=np.uint8)
        color[valid] = colorized[valid]
        return Image.fromarray(color, mode="RGB")

    def render(
        self,
        *,
        cache_key: str,
        output_path: Path,
        pred_inverse_like: np.ndarray,
        gt_depth_path: Path,
    ) -> Image.Image:
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        gt_inverse, valid = self.load_gt_inverse(gt_depth_path)
        pred = self.resize_to(pred_inverse_like, gt_inverse.shape)
        aligned = self.align_to_gt_inverse(pred, gt_inverse, valid)
        safe_valid = valid & np.isfinite(aligned) & (gt_inverse > 0)
        error = np.zeros_like(gt_inverse, dtype=np.float32)
        error[safe_valid] = np.abs(aligned[safe_valid] - gt_inverse[safe_valid]) / gt_inverse[safe_valid]
        error_image = self.colorize(error, safe_valid)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_image.save(output_path)
        self.cache[cache_key] = error_image
        return error_image.copy()


def dav2_depth_path(
    output_dir: Path,
    *,
    sample: str,
    encoder: str,
    config: str | None,
    variant: str | None,
) -> Path:
    stem = dav2_file_stem(encoder)
    if config is None:
        filename = f"{sample}_original_{stem}_depth.png"
    else:
        filename = f"{sample}_{config}_{variant}_{stem}_depth.png"
    return output_dir / "dav2_depth" / filename


def gt_depth_path(output_dir: Path, *, sample: str) -> Path:
    return output_dir / "gt_depth" / f"{sample}_gt_depth.png"


def baseline_error_map_path(output_dir: Path, *, sample: str, column: str) -> Path:
    return output_dir / "baseline_error_maps" / f"{sample}_baseline_{column}_inv_absrel_error.png"


def build_panel(
    *,
    sample: dict,
    sample_root: Path,
    raw_output_base: Path,
    output_dir: Path,
    configs: Sequence[str],
    variants: Sequence[str],
    thumb_size: tuple[int, int],
    include_distribution: bool,
    dist_height: int,
    dav2_renderer: Dav2DepthRenderer | None,
    dav2_depth_size: tuple[int, int] | None,
    gt_depth_renderer: GtDepthRenderer | None,
    gt_depth_size: tuple[int, int] | None,
    error_map_renderer: ErrorMapRenderer | None,
    error_map_size: tuple[int, int] | None,
) -> dict[str, object]:
    name = sample_name(sample)
    link = Path(sample["link_path"])
    columns = ["original", *variants]
    gap = 8
    title_h = 38
    row_label_w = 128
    thumb_w, thumb_h = thumb_size
    dav2_depth_h = dav2_depth_size[1] if dav2_depth_size is not None else 0
    dav2_extra_h = gap + dav2_depth_h if dav2_renderer is not None else 0
    base_cell_h = thumb_h + dav2_extra_h + (gap + dist_height if include_distribution else 0)
    error_map_h = error_map_size[1] if error_map_size is not None else 0
    error_extra_h = gap + error_map_h if error_map_renderer is not None else 0
    gt_row_h = gt_depth_size[1] if gt_depth_size is not None else 0
    gt_extra_h = gt_row_h + gap if gt_depth_renderer is not None else 0
    width = row_label_w + len(columns) * thumb_w + (len(columns) + 2) * gap
    row_heights = [base_cell_h + (error_extra_h if config == "baseline" else 0) for config in configs]
    height = title_h + gt_extra_h + sum(row_heights) + (len(configs) + 1) * gap
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), name, fill=(0, 0, 0), font=ImageFont.load_default())

    distribution_blocks: list[tuple[object, str, int, int, str]] = []
    dav2_depth_paths: list[str] = []
    gt_depth_paths: list[str] = []
    error_map_paths: list[str] = []
    rgb_distribution = load_rgb(link) if include_distribution else None

    rows_y = title_h + gap
    if gt_depth_renderer is not None and gt_depth_size is not None:
        y = rows_y
        draw.rectangle((gap, y, gap + row_label_w, y + gt_row_h), fill=(22, 22, 22))
        draw.text((gap + 8, y + 12), "reference", fill=(255, 255, 255), font=ImageFont.load_default())
        draw.text((gap + 8, y + 32), "gt depth", fill=(190, 190, 190), font=ImageFont.load_default())

        slot_count = 3
        for slot_idx in range(slot_count):
            x = gap + row_label_w + gap + slot_idx * (thumb_w + gap)
            if slot_idx == 0:
                depth_out = gt_depth_path(output_dir, sample=name)
                depth_image = gt_depth_renderer.render_file(Path(sample["depth_path"]), depth_out)
                draw_pil_cell(sheet, draw, depth_image, title="gt depth", x=x, y=y, size=gt_depth_size)
                gt_depth_paths.append(str(depth_out))
            else:
                draw_empty_cell(sheet, draw, x=x, y=y, size=gt_depth_size)
        rows_y += gt_row_h + gap

    for row_idx, config in enumerate(configs):
        raw_dir = raw_output_dir(raw_output_base, config)
        y = rows_y + sum(row_heights[:row_idx]) + row_idx * gap
        row_h = row_heights[row_idx]
        row_error_extra_h = error_extra_h if config == "baseline" else 0
        draw.rectangle((gap, y, gap + row_label_w, y + row_h), fill=(22, 22, 22))
        label = CONFIG_LABELS.get(config, config)
        draw.text((gap + 8, y + 12), label, fill=(255, 255, 255), font=ImageFont.load_default())
        draw.text((gap + 8, y + 32), raw_dir.name[-38:], fill=(190, 190, 190), font=ImageFont.load_default())

        paths = [link, *[preview_path(raw_dir, variant, link, sample_root) for variant in variants]]
        for col_idx, (column, path) in enumerate(zip(columns, paths)):
            x = gap + row_label_w + gap + col_idx * (thumb_w + gap)
            draw_cell(sheet, draw, path, title=column, x=x, y=y, size=thumb_size)
            pred_depth = None
            if dav2_renderer is not None and dav2_depth_size is not None:
                depth_y = y + thumb_h + gap
                if col_idx == 0:
                    depth_out = dav2_depth_path(
                        output_dir,
                        sample=name,
                        encoder=dav2_renderer.encoder,
                        config=None,
                        variant=None,
                    )
                    depth_image = dav2_renderer.render_rgb_file(link, depth_out)
                    if error_map_renderer is not None and config == "baseline":
                        pred_depth = dav2_renderer.infer_rgb_file(link)
                else:
                    variant = variants[col_idx - 1]
                    depth_out = dav2_depth_path(
                        output_dir,
                        sample=name,
                        encoder=dav2_renderer.encoder,
                        config=config,
                        variant=variant,
                    )
                    raw_npz_path = npz_path(raw_dir, variant, link, sample_root)
                    depth_image = dav2_renderer.render_raw_file(raw_npz_path, depth_out)
                    if error_map_renderer is not None and config == "baseline":
                        pred_depth = dav2_renderer.infer_raw_file(raw_npz_path)
                draw_pil_cell(
                    sheet,
                    draw,
                    depth_image,
                    title=f"{dav2_label(dav2_renderer.encoder)} depth",
                    x=x,
                    y=depth_y,
                    size=dav2_depth_size,
                )
                dav2_depth_paths.append(str(depth_out))
            if (
                error_map_renderer is not None
                and error_map_size is not None
                and dav2_renderer is not None
                and config == "baseline"
            ):
                if pred_depth is None:
                    if col_idx == 0:
                        pred_depth = dav2_renderer.infer_rgb_file(link)
                    else:
                        variant = variants[col_idx - 1]
                        pred_depth = dav2_renderer.infer_raw_file(npz_path(raw_dir, variant, link, sample_root))
                error_y = y + thumb_h + dav2_extra_h + gap
                error_out = baseline_error_map_path(output_dir, sample=name, column=column)
                error_image = error_map_renderer.render(
                    cache_key=f"{name}:baseline:{column}",
                    output_path=error_out,
                    pred_inverse_like=pred_depth,
                    gt_depth_path=Path(sample["depth_path"]),
                )
                draw_pil_cell(
                    sheet,
                    draw,
                    error_image,
                    title="inv absrel error",
                    x=x,
                    y=error_y,
                    size=error_map_size,
                )
                error_map_paths.append(str(error_out))
            if include_distribution:
                dist_y = y + thumb_h + dav2_extra_h + row_error_extra_h + gap
                if col_idx == 0:
                    distribution_blocks.append((rgb_distribution, "RGB value distribution", x, dist_y, "unit"))
                else:
                    variant = variants[col_idx - 1]
                    raw = load_raw_packed(npz_path(raw_dir, variant, link, sample_root))
                    distribution_blocks.append((raw, f"{variant} RAW packed distribution", x, dist_y, "auto"))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_ablation_panel.jpg"
    if include_distribution:
        canvas = np.asarray(sheet).copy()
        for array, title, x, y, axis_mode in distribution_blocks:
            draw_distribution_block(
                canvas,
                array,
                title,
                x,
                y,
                thumb_w,
                dist_height,
                axis_mode=axis_mode,
            )
        Image.fromarray(canvas).save(out_path, quality=92)
    else:
        sheet.save(out_path, quality=92)
    return {
        "panel": str(out_path),
        "dav2_depth_images": sorted(set(dav2_depth_paths)),
        "gt_depth_images": sorted(set(gt_depth_paths)),
        "baseline_error_maps": sorted(set(error_map_paths)),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.sample_manifest.expanduser().resolve()
    raw_output_base = args.raw_output_base.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else raw_output_base.parent / "ablation_panels_s5"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_root = Path(manifest["output_dir"]).expanduser().resolve()
    selected = select_samples(manifest["samples"], args.num_samples, args.strategy, args.seed)

    dav2_renderer = None
    dav2_checkpoint = None
    dav2_depth_height = int(args.dav2_depth_height) if int(args.dav2_depth_height) > 0 else int(args.thumb_height)
    if args.include_dav2_depth:
        dav2_checkpoint = resolve_dav2_checkpoint(args.dav2_checkpoint, args.dav2_encoder)
        dav2_renderer = Dav2DepthRenderer(
            encoder=args.dav2_encoder,
            checkpoint=dav2_checkpoint,
            input_size=args.dav2_input_size,
            cmap_name=args.dav2_depth_cmap,
            raw_rgb_mode=args.dav2_raw_rgb_mode,
        )
    dav2_depth_size = (args.thumb_width, dav2_depth_height) if dav2_renderer is not None else None
    gt_depth_renderer = None
    gt_depth_height = int(args.gt_depth_height) if int(args.gt_depth_height) > 0 else int(args.thumb_height)
    gt_depth_cmap = args.gt_depth_cmap or args.dav2_depth_cmap
    if args.include_gt_depth:
        gt_depth_renderer = GtDepthRenderer(cmap_name=gt_depth_cmap)
    gt_depth_size = (args.thumb_width, gt_depth_height) if gt_depth_renderer is not None else None
    if args.include_baseline_error_map and not args.include_dav2_depth:
        raise ValueError("--include-baseline-error-map requires --include-dav2-depth")
    error_map_renderer = None
    error_map_height = int(args.error_map_height) if int(args.error_map_height) > 0 else int(args.thumb_height)
    if args.include_baseline_error_map:
        error_map_renderer = ErrorMapRenderer(cmap_name=args.error_map_cmap, vmax=float(args.error_map_max))
    error_map_size = (args.thumb_width, error_map_height) if error_map_renderer is not None else None

    records = []
    for sample in selected:
        records.append(
            build_panel(
                sample=sample,
                sample_root=sample_root,
                raw_output_base=raw_output_base,
                output_dir=output_dir,
                configs=args.configs,
                variants=args.variants,
                thumb_size=(args.thumb_width, args.thumb_height),
                include_distribution=bool(args.include_distribution),
                dist_height=int(args.dist_height),
                dav2_renderer=dav2_renderer,
                dav2_depth_size=dav2_depth_size,
                gt_depth_renderer=gt_depth_renderer,
                gt_depth_size=gt_depth_size,
                error_map_renderer=error_map_renderer,
                error_map_size=error_map_size,
            )
        )

    panels = [record["panel"] for record in records]
    dav2_depth_images = sorted({path for record in records for path in record["dav2_depth_images"]})
    gt_depth_images = sorted({path for record in records for path in record["gt_depth_images"]})
    baseline_error_maps = sorted({path for record in records for path in record["baseline_error_maps"]})
    summary = {
        "sample_manifest": str(manifest_path),
        "raw_output_base": str(raw_output_base),
        "output_dir": str(output_dir),
        "configs": list(args.configs),
        "variants": list(args.variants),
        "strategy": args.strategy,
        "include_distribution": bool(args.include_distribution),
        "dist_height": int(args.dist_height),
        "num_panels": len(panels),
        "panels": panels,
        "dav2_depth": {
            "enabled": bool(args.include_dav2_depth),
            "encoder": args.dav2_encoder if args.include_dav2_depth else None,
            "label": dav2_label(args.dav2_encoder) if args.include_dav2_depth else None,
            "checkpoint": str(dav2_checkpoint) if dav2_checkpoint is not None else None,
            "input_size": int(args.dav2_input_size) if args.include_dav2_depth else None,
            "depth_cmap": args.dav2_depth_cmap if args.include_dav2_depth else None,
            "depth_height": dav2_depth_height if args.include_dav2_depth else None,
            "raw_rgb_mode": args.dav2_raw_rgb_mode if args.include_dav2_depth else None,
            "device": dav2_renderer.device if dav2_renderer is not None else None,
            "num_depth_images": len(dav2_depth_images),
            "depth_images": dav2_depth_images,
        },
        "gt_depth": {
            "enabled": bool(args.include_gt_depth),
            "depth_cmap": gt_depth_cmap if args.include_gt_depth else None,
            "depth_height": gt_depth_height if args.include_gt_depth else None,
            "num_depth_images": len(gt_depth_images),
            "depth_images": gt_depth_images,
        },
        "baseline_error_map": {
            "enabled": bool(args.include_baseline_error_map),
            "domain": "inverse_depth_absrel_after_per-image_affine_alignment"
            if args.include_baseline_error_map
            else None,
            "error_cmap": args.error_map_cmap if args.include_baseline_error_map else None,
            "error_vmax": float(args.error_map_max) if args.include_baseline_error_map else None,
            "error_height": error_map_height if args.include_baseline_error_map else None,
            "num_error_maps": len(baseline_error_maps),
            "error_maps": baseline_error_maps,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
