#!/usr/bin/env python3
"""
RAW-Adapter-style RGB unprocessing skeleton.

This script is intentionally split into two stages:
1. RGB to raw-RGB backend:
   - analytic: a runnable approximation using inverse sRGB gamma, optional inverse tone, inverse CCM, inverse WB.
   - external_npy: read raw-RGB arrays produced by an external learned InvISP implementation.
2. raw-RGB to packed Bayer:
   - inverse white balance, optional dark or over-exposure synthesis, RGGB packing, saving to .npz plus metadata.

Output target is packed Bayer, shape [4, H/2, W/2], channel order [R, G1, G2, B] for RGGB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **_: x

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class UnprocessMeta:
    input_path: str
    output_npz: str
    backend: str
    cfa: str
    variant: str
    packed_shape: Tuple[int, int, int]
    storage: str
    seed: int
    red_gain: float
    blue_gain: float
    light_scale: float
    shot_noise: float
    read_noise: float
    black_level: float
    white_level: float
    noise_mean_mode: str
    inverse_tone: bool
    ccm_name: str
    note: str


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256((str(base_seed) + "|" + text).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def list_images(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def load_rgb_float(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def inverse_global_tone(x: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """A conservative inverse tone approximation.

    Real InvISP should replace this block. The function slightly expands highlights
    while preserving monotonicity and the [0, 1] range.
    """
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    if strength <= 0:
        return x
    expanded = x / np.maximum(1.0 - strength * (1.0 - x), 1e-6)
    return np.clip(expanded, 0.0, 1.0).astype(np.float32)


def get_inverse_ccm(name: str) -> np.ndarray:
    """Return an RGB to camera-like matrix.

    identity is safest for non-camera-specific training. generic_d65 is a mild
    camera-like transform and should be treated as a heuristic, not calibration.
    """
    name = name.lower()
    if name == "identity":
        mat = np.eye(3, dtype=np.float32)
    elif name == "generic_d65":
        mat = np.array(
            [
                [0.86, 0.08, 0.06],
                [0.05, 0.90, 0.05],
                [0.04, 0.12, 0.84],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown CCM preset: {name}")
    return mat


def apply_ccm(rgb: np.ndarray, mat: np.ndarray) -> np.ndarray:
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError(f"Expected HWC RGB, got shape {rgb.shape}")
    out = rgb.reshape(-1, 3) @ mat.T
    return np.clip(out.reshape(h, w, 3), 0.0, 1.0).astype(np.float32)


def sample_wb_gains(rng: np.random.Generator, red_range: Tuple[float, float], blue_range: Tuple[float, float]) -> Tuple[float, float]:
    red_gain = float(rng.uniform(red_range[0], red_range[1]))
    blue_gain = float(rng.uniform(blue_range[0], blue_range[1]))
    return red_gain, blue_gain


def inverse_white_balance(raw_rgb: np.ndarray, red_gain: float, blue_gain: float) -> np.ndarray:
    gains = np.array([1.0 / red_gain, 1.0, 1.0 / blue_gain], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(raw_rgb.astype(np.float32) * gains, 0.0, 1.0).astype(np.float32)


def analytic_rgb_to_raw_rgb(rgb: np.ndarray, *, inverse_tone: bool, ccm_name: str) -> np.ndarray:
    x = srgb_to_linear(rgb)
    if inverse_tone:
        x = inverse_global_tone(x)
    x = apply_ccm(x, get_inverse_ccm(ccm_name))
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def normalize_external_raw_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"External raw-RGB must be 3D, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"External raw-RGB must have 3 channels, got {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max() > 2.0:
        if arr.max() <= 255.0:
            arr = arr / 255.0
        elif arr.max() <= 65535.0:
            arr = arr / 65535.0
        else:
            arr = arr / float(arr.max())
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def load_external_raw_rgb(input_path: Path, input_dir: Path, external_raw_dir: Path) -> np.ndarray:
    rel = input_path.relative_to(input_dir)
    candidates = [
        external_raw_dir / rel.with_suffix(".npy"),
        external_raw_dir / rel.with_suffix(".npz"),
        external_raw_dir / (rel.stem + ".npy"),
        external_raw_dir / (rel.stem + ".npz"),
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix == ".npy":
                return normalize_external_raw_array(np.load(cand))
            data = np.load(cand)
            for key in ("raw_rgb", "arr_0", "pred_raw", "raw"):
                if key in data:
                    return normalize_external_raw_array(data[key])
            raise KeyError(f"No raw_rgb-like key found in {cand}")
    raise FileNotFoundError(f"No external raw file found for {input_path}; tried {', '.join(str(c) for c in candidates)}")


def crop_even_hw(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[:2]
    return x[: h - (h % 2), : w - (w % 2), ...]


def pack_rggb(raw_rgb: np.ndarray) -> np.ndarray:
    raw_rgb = crop_even_hw(raw_rgb)
    r = raw_rgb[0::2, 0::2, 0]
    g1 = raw_rgb[0::2, 1::2, 1]
    g2 = raw_rgb[1::2, 0::2, 1]
    b = raw_rgb[1::2, 1::2, 2]
    return np.stack([r, g1, g2, b], axis=0).astype(np.float32)


def packed_to_mosaic_rggb(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected packed shape [4,H,W], got {packed.shape}")
    _, h, w = packed.shape
    mosaic = np.zeros((h * 2, w * 2), dtype=np.float32)
    mosaic[0::2, 0::2] = packed[0]
    mosaic[0::2, 1::2] = packed[1]
    mosaic[1::2, 0::2] = packed[2]
    mosaic[1::2, 1::2] = packed[3]
    return mosaic


def preview_from_packed_rggb(packed: np.ndarray, red_gain: float, blue_gain: float) -> np.ndarray:
    _, h, w = packed.shape
    rgb = np.zeros((h * 2, w * 2, 3), dtype=np.float32)
    rgb[0::2, 0::2, 0] = packed[0]
    rgb[0::2, 1::2, 1] = packed[1]
    rgb[1::2, 0::2, 1] = packed[2]
    rgb[1::2, 1::2, 2] = packed[3]

    # Very cheap nearest fill for debug only, not a production demosaicer.
    rgb[0::2, 1::2, 0] = packed[0]
    rgb[1::2, 0::2, 0] = packed[0]
    rgb[1::2, 1::2, 0] = packed[0]
    rgb[0::2, 0::2, 2] = packed[3]
    rgb[0::2, 1::2, 2] = packed[3]
    rgb[1::2, 0::2, 2] = packed[3]
    g_mean = 0.5 * (packed[1] + packed[2])
    rgb[0::2, 0::2, 1] = g_mean
    rgb[1::2, 1::2, 1] = g_mean

    rgb[..., 0] *= red_gain
    rgb[..., 2] *= blue_gain
    return linear_to_srgb(np.clip(rgb, 0.0, 1.0))


def save_preview_png(path: Path, preview: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.round(np.clip(preview, 0, 1) * 255.0).astype(np.uint8))
    img.save(path)


def sample_light_scale(variant: str, rng: np.random.Generator) -> float:
    if variant == "normal":
        return 1.0
    if variant == "dark":
        return float(rng.uniform(0.05, 0.4))
    if variant == "over":
        return float(rng.uniform(1.5, 2.5))
    raise ValueError(f"Unknown variant: {variant}")


def apply_light_synthesis(
    x: np.ndarray,
    light_scale: float,
    rng: np.random.Generator,
    shot_noise: float,
    read_noise: float,
    noise_mean_mode: str = "zero",
    clip: bool = True,
) -> np.ndarray:
    signal = np.clip(light_scale * x.astype(np.float32), 0.0, None)
    if shot_noise > 0 or read_noise > 0:
        variance = (float(read_noise) ** 2) + float(shot_noise) * np.maximum(signal, 0.0)
        if noise_mean_mode == "zero":
            loc = 0.0
        elif noise_mean_mode == "rawadapter_text":
            # RAW-Adapter Eq. (7) is written as x_n ~ N(mu = l*x, ...), y = l*x + x_n.
            # This doubles the expected signal and is kept only for strict text-level reproduction.
            loc = signal
        else:
            raise ValueError(f"Unknown noise_mean_mode: {noise_mean_mode}")
        noise = rng.normal(loc=loc, scale=np.sqrt(variance)).astype(np.float32)
        signal = signal + noise
    if clip:
        signal = np.clip(signal, 0.0, 1.0)
    return signal.astype(np.float32)


def apply_black_white_levels(x: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    if not (0.0 <= black_level < white_level <= 1.0):
        raise ValueError("Require 0 <= black_level < white_level <= 1")
    return np.clip(black_level + x * (white_level - black_level), 0.0, 1.0).astype(np.float32)


def encode_storage(x: np.ndarray, storage: str) -> np.ndarray:
    storage = storage.lower()
    if storage == "float32":
        return x.astype(np.float32)
    if storage == "float16":
        return x.astype(np.float16)
    if storage == "uint16":
        return np.round(np.clip(x, 0.0, 1.0) * 65535.0).astype(np.uint16)
    raise ValueError(f"Unsupported storage: {storage}")


def parse_variants(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            if item:
                out.append(item)
    allowed = {"normal", "dark", "over"}
    bad = [v for v in out if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported variants: {bad}. Allowed: normal,dark,over")
    return out or ["normal"]


def process_one(path: Path, args: argparse.Namespace, input_dir: Path, out_dir: Path) -> List[UnprocessMeta]:
    rel = path.relative_to(input_dir)
    rel_stem = rel.with_suffix("")
    seed = stable_seed(str(rel), args.seed)
    rng_base = np.random.default_rng(seed)

    if args.backend == "analytic":
        rgb = load_rgb_float(path)
        raw_rgb = analytic_rgb_to_raw_rgb(rgb, inverse_tone=args.inverse_tone, ccm_name=args.ccm)
    elif args.backend == "external_npy":
        if args.external_raw_dir is None:
            raise ValueError("--external-raw-dir is required for --backend external_npy")
        raw_rgb = load_external_raw_rgb(path, input_dir, Path(args.external_raw_dir))
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    red_gain, blue_gain = sample_wb_gains(rng_base, tuple(args.red_gain_range), tuple(args.blue_gain_range))
    raw_rgb = inverse_white_balance(raw_rgb, red_gain, blue_gain)
    raw_rgb = crop_even_hw(raw_rgb)
    metas: List[UnprocessMeta] = []

    for variant in args.variants:
        rng_variant = np.random.default_rng(stable_seed(str(rel) + "|" + variant, args.seed))
        light_scale = sample_light_scale(variant, rng_variant)
        raw_variant = apply_light_synthesis(raw_rgb, light_scale, rng_variant, args.shot_noise, args.read_noise, noise_mean_mode=args.noise_mean_mode, clip=not args.no_clip)
        raw_variant = apply_black_white_levels(raw_variant, args.black_level, args.white_level)
        packed = pack_rggb(raw_variant)

        npz_path = out_dir / variant / rel_stem.with_suffix(".npz")
        json_path = out_dir / variant / rel_stem.with_suffix(".json")
        preview_path = out_dir / "preview" / variant / rel_stem.with_suffix(".png")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"raw_packed": encode_storage(packed, args.storage)}
        if args.save_mosaic:
            payload["raw_mosaic"] = encode_storage(packed_to_mosaic_rggb(packed), args.storage)
        np.savez_compressed(npz_path, **payload)

        if args.save_preview:
            preview = preview_from_packed_rggb(packed, red_gain, blue_gain)
            save_preview_png(preview_path, preview)

        meta = UnprocessMeta(
            input_path=str(path),
            output_npz=str(npz_path),
            backend=args.backend,
            cfa="RGGB",
            variant=variant,
            packed_shape=tuple(int(x) for x in packed.shape),
            storage=args.storage,
            seed=int(seed),
            red_gain=float(red_gain),
            blue_gain=float(blue_gain),
            light_scale=float(light_scale),
            shot_noise=float(args.shot_noise),
            read_noise=float(args.read_noise),
            black_level=float(args.black_level),
            white_level=float(args.white_level),
            noise_mean_mode=str(args.noise_mean_mode),
            inverse_tone=bool(args.inverse_tone),
            ccm_name=str(args.ccm),
            note="Packed Bayer channel order is [R,G1,G2,B]. Preview uses a nearest-fill debug demosaic only.",
        )
        json_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")
        metas.append(meta)
    return metas


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RAW-Adapter-style packed RGGB data from RGB images.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing RGB images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for .npz, .json, and optional previews.")
    parser.add_argument("--backend", choices=["analytic", "external_npy"], default="analytic", help="RGB to raw-RGB backend.")
    parser.add_argument("--external-raw-dir", type=Path, default=None, help="Directory containing external raw-RGB .npy/.npz files for backend external_npy.")
    parser.add_argument("--variants", nargs="+", default=["normal"], help="Variants to save: normal,dark,over or space-separated values.")
    parser.add_argument("--max-images", type=int, default=0, help="Process only the first N images. 0 means all.")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed. Per-image randomness is deterministic from path plus this seed.")
    parser.add_argument("--red-gain-range", nargs=2, type=float, default=[1.9, 2.4], help="Forward red WB gain range. Inverse WB uses 1/red_gain.")
    parser.add_argument("--blue-gain-range", nargs=2, type=float, default=[1.5, 1.9], help="Forward blue WB gain range. Inverse WB uses 1/blue_gain.")
    parser.add_argument("--shot-noise", type=float, default=0.001, help="Shot-noise variance coefficient for linear RAW signal.")
    parser.add_argument("--read-noise", type=float, default=0.0005, help="Read-noise std term for linear RAW signal.")
    parser.add_argument("--black-level", type=float, default=0.0, help="Normalized black level to add before saving.")
    parser.add_argument("--white-level", type=float, default=1.0, help="Normalized white level before saturation.")
    parser.add_argument("--storage", choices=["float16", "float32", "uint16"], default="float16", help="Storage dtype inside .npz.")
    parser.add_argument("--noise-mean-mode", choices=["zero", "rawadapter_text"], default="zero", help="zero uses y=l*x+n with zero-mean noise. rawadapter_text follows the printed RAW-Adapter Eq. (7) literally.")
    parser.add_argument("--ccm", choices=["identity", "generic_d65"], default="identity", help="Analytic backend RGB to camera-like CCM.")
    parser.add_argument("--inverse-tone", action="store_true", help="Use a conservative inverse global tone approximation in analytic backend.")
    parser.add_argument("--save-preview", action="store_true", help="Save debug PNG previews.")
    parser.add_argument("--save-mosaic", action="store_true", help="Also save a single-channel RGGB mosaic in each .npz.")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip after exposure synthesis. Not recommended for ordinary training.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.variants = parse_variants(args.variants)

    input_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    all_metas: List[UnprocessMeta] = []
    for path in tqdm(images, desc="unprocessing"):
        all_metas.extend(process_one(path, args, input_dir, out_dir))

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "backend": args.backend,
        "variants": args.variants,
        "num_inputs": len(images),
        "num_outputs": len(all_metas),
        "cfa": "RGGB",
        "packed_channel_order": ["R", "G1", "G2", "B"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
