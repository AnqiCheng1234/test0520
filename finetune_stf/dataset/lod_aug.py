"""True-LOD train-only augmentation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
import warnings
from typing import Any

import cv2
import numpy as np


LODAUG_PRESETS = ("off", "baseline_e10", "geom", "light", "medium", "heavy")


@dataclass(frozen=True)
class LODAugConfig:
    domain: str
    preset_name: str = "off"
    enabled: bool = False
    seed: int = 42
    hflip_prob: float = 0.0
    scale_jitter: tuple[float, float] | None = None
    rotate_deg: float = 0.0
    rgb_brightness: float = 0.0
    rgb_contrast: float = 0.0
    rgb_gamma: tuple[float, float] | None = None
    rgb_color: float = 0.0
    rgb_noise_std: float = 0.0
    rgb_blur_prob: float = 0.0
    raw_gain: tuple[float, float] | None = None
    raw_per_channel_gain: bool = False
    raw_black_offset: float = 0.0
    raw_noise: tuple[float, float] | None = None

    @classmethod
    def from_preset(cls, preset_name: str, *, domain: str, seed: int = 42) -> "LODAugConfig":
        preset = str(preset_name)
        if preset not in LODAUG_PRESETS:
            raise ValueError(f"Unsupported LOD augmentation preset {preset!r}; choices={LODAUG_PRESETS}")
        domain = _normalise_domain(domain)
        cfg = cls(domain=domain, preset_name=preset, seed=int(seed))

        if preset == "off":
            return cfg
        if preset == "baseline_e10":
            return replace(cfg, hflip_prob=0.5)._with_enabled()
        if preset == "geom":
            return replace(cfg, hflip_prob=0.5, scale_jitter=(0.7, 1.4))._with_enabled()
        if preset == "light":
            cfg = replace(cfg, hflip_prob=0.5, scale_jitter=(0.85, 1.2))
            if domain == "rgb":
                cfg = replace(cfg, rgb_noise_std=0.01)
            else:
                cfg = replace(cfg, raw_gain=(0.85, 1.2))
            return cfg._with_enabled()
        if preset == "medium":
            cfg = replace(cfg, hflip_prob=0.5, scale_jitter=(0.7, 1.4))
            if domain == "rgb":
                cfg = replace(cfg, rgb_brightness=0.2, rgb_contrast=0.2, rgb_gamma=(0.8, 1.2), rgb_noise_std=0.02)
            else:
                cfg = replace(cfg, raw_gain=(0.7, 1.4), raw_black_offset=0.002, raw_noise=(0.002, 0.001))
            return cfg._with_enabled()
        if preset == "heavy":
            cfg = replace(cfg, hflip_prob=0.5, scale_jitter=(0.6, 1.6))
            if domain == "rgb":
                cfg = replace(
                    cfg,
                    rgb_brightness=0.3,
                    rgb_contrast=0.3,
                    rgb_gamma=(0.7, 1.3),
                    rgb_color=0.1,
                    rgb_noise_std=0.04,
                    rgb_blur_prob=0.1,
                )
            else:
                cfg = replace(
                    cfg,
                    raw_gain=(0.6, 1.6),
                    raw_per_channel_gain=True,
                    raw_black_offset=0.004,
                    raw_noise=(0.004, 0.002),
                )
            return cfg._with_enabled()
        raise AssertionError(f"Unhandled LOD aug preset: {preset}")

    @classmethod
    def from_args(cls, args: Any, *, domain: str) -> "LODAugConfig":
        explicit = set(getattr(args, "_explicit_cli_args", ()) or ())
        seed = int(getattr(args, "seed", 42))
        cfg = cls.from_preset(str(getattr(args, "aug_preset", "off")), domain=domain, seed=seed)

        overrides = {
            "aug_hflip_prob": ("hflip_prob", float),
            "aug_scale_jitter": ("scale_jitter", _tuple_or_none),
            "aug_rotate_deg": ("rotate_deg", float),
            "aug_rgb_brightness": ("rgb_brightness", float),
            "aug_rgb_contrast": ("rgb_contrast", float),
            "aug_rgb_gamma": ("rgb_gamma", _tuple_or_none),
            "aug_rgb_color": ("rgb_color", float),
            "aug_rgb_noise_std": ("rgb_noise_std", float),
            "aug_rgb_blur_prob": ("rgb_blur_prob", float),
            "aug_raw_gain": ("raw_gain", _tuple_or_none),
            "aug_raw_per_channel_gain": ("raw_per_channel_gain", bool),
            "aug_raw_black_offset": ("raw_black_offset", float),
            "aug_raw_noise": ("raw_noise", _tuple_or_none),
        }
        values = {}
        for arg_name, (field_name, caster) in overrides.items():
            if arg_name not in explicit:
                continue
            values[field_name] = caster(getattr(args, arg_name))
        if values:
            cfg = replace(cfg, **values)
        cfg = cfg._with_enabled()
        cfg.validate()
        return cfg

    def _with_enabled(self) -> "LODAugConfig":
        enabled = (
            self.hflip_prob > 0.0
            or self.scale_jitter is not None
            or self.rotate_deg > 0.0
            or self.rgb_brightness > 0.0
            or self.rgb_contrast > 0.0
            or self.rgb_gamma is not None
            or self.rgb_color > 0.0
            or self.rgb_noise_std > 0.0
            or self.rgb_blur_prob > 0.0
            or self.raw_gain is not None
            or self.raw_per_channel_gain
            or self.raw_black_offset > 0.0
            or self.raw_noise is not None
        )
        return replace(self, enabled=bool(enabled))

    def validate(self) -> None:
        if self.domain not in {"rgb", "raw3"}:
            raise ValueError(f"LOD aug domain must be rgb or raw3, got {self.domain!r}")
        _validate_prob(self.hflip_prob, "aug_hflip_prob")
        _validate_prob(self.rgb_blur_prob, "aug_rgb_blur_prob")
        if self.scale_jitter is not None:
            _validate_range(self.scale_jitter, "aug_scale_jitter", min_value=0.0)
        if self.rgb_gamma is not None:
            _validate_range(self.rgb_gamma, "aug_rgb_gamma", min_value=0.0)
        if self.raw_gain is not None:
            _validate_range(self.raw_gain, "aug_raw_gain", min_value=0.0)
        if self.raw_noise is not None:
            shot, read = self.raw_noise
            if shot < 0.0 or read < 0.0:
                raise ValueError("--aug-raw-noise values must be non-negative")
        if self.rotate_deg > 0.0:
            warnings.warn(
                "--aug-rotate-deg is reserved but not applied in this implementation; keep it at 0 for Plan B.",
                stacklevel=2,
            )
            raise ValueError("--aug-rotate-deg > 0 is not implemented for true-LOD depth/mask boundaries")

        rgb_active = (
            self.rgb_brightness > 0.0
            or self.rgb_contrast > 0.0
            or self.rgb_gamma is not None
            or self.rgb_color > 0.0
            or self.rgb_noise_std > 0.0
            or self.rgb_blur_prob > 0.0
        )
        raw_active = (
            self.raw_gain is not None
            or self.raw_per_channel_gain
            or self.raw_black_offset > 0.0
            or self.raw_noise is not None
        )
        if self.domain == "raw3" and rgb_active:
            raise ValueError("RAW true-LOD augmentation forbids RGB photometric flags")
        if self.domain == "rgb" and raw_active:
            raise ValueError("RGB true-LOD augmentation forbids RAW photometric flags")

    def rng(self, *, epoch: int, sample_id: str, stream: str) -> np.random.Generator:
        token = f"{self.seed}:{int(epoch)}:{sample_id}:{stream}".encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        return np.random.default_rng(seed)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("scale_jitter", "rgb_gamma", "raw_gain", "raw_noise"):
            if payload[key] is not None:
                payload[key] = [float(v) for v in payload[key]]
        return payload


def _normalise_domain(domain: str) -> str:
    domain = str(domain)
    if domain in {"raw", "raw_rgb16"}:
        return "raw3"
    return domain


def _tuple_or_none(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "off"}:
        return None
    if len(value) != 2:
        raise ValueError("Expected two values")
    return float(value[0]), float(value[1])


def _validate_range(value: tuple[float, float], name: str, *, min_value: float) -> None:
    lo, hi = float(value[0]), float(value[1])
    if lo < min_value or hi < min_value or hi < lo:
        raise ValueError(f"--{name.replace('_', '-')} must satisfy {min_value} <= min <= max")


def _validate_prob(value: float, name: str) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")


def _rand_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(float(lo), float(hi)))


def _resize_hw(array: np.ndarray, hw: tuple[int, int], *, interpolation: int) -> np.ndarray:
    height, width = hw
    return cv2.resize(array, (int(width), int(height)), interpolation=interpolation)


def _crop(array: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    h_start, w_start, target_h, target_w = crop_box
    cropped = array[h_start : h_start + target_h, w_start : w_start + target_w]
    return np.ascontiguousarray(cropped)


def apply_geometric(
    input_array: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    cfg: LODAugConfig,
    size: tuple[int, int],
    crop_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply paired scale/crop/hflip to input, depth, and mask."""

    target_h, target_w = int(size[0]), int(size[1])
    height, width = input_array.shape[:2]
    sampled_scale = 1.0
    applied_scale = 1.0

    if cfg.scale_jitter is not None and crop_mode != "center":
        lo, hi = cfg.scale_jitter
        sampled_scale = _rand_uniform(rng, lo, hi)
        min_scale = max(target_h / float(height), target_w / float(width))
        applied_scale = max(sampled_scale, min_scale)
    new_h = max(int(round(height * applied_scale)), target_h)
    new_w = max(int(round(width * applied_scale)), target_w)

    if new_h != height or new_w != width:
        input_array = _resize_hw(input_array, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
        depth = _resize_hw(depth, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
        mask = _resize_hw(mask.astype(np.uint8), (new_h, new_w), interpolation=cv2.INTER_NEAREST).astype(bool)

    if crop_mode == "center":
        h_start = max((new_h - target_h) // 2, 0)
        w_start = max((new_w - target_w) // 2, 0)
    elif crop_mode == "random":
        h_start = int(rng.integers(0, new_h - target_h + 1))
        w_start = int(rng.integers(0, new_w - target_w + 1))
    else:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}")
    crop_box = (h_start, w_start, target_h, target_w)
    input_array = _crop(input_array, crop_box)
    depth = _crop(depth, crop_box)
    mask = _crop(mask, crop_box).astype(bool, copy=False)

    hflip = bool(rng.random() < cfg.hflip_prob)
    if hflip:
        input_array = np.ascontiguousarray(input_array[:, ::-1, ...])
        depth = np.ascontiguousarray(depth[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    valid = mask & np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
    return input_array, depth, valid, {
        "original_hw": [int(height), int(width)],
        "resized_hw": [int(new_h), int(new_w)],
        "sampled_scale": float(sampled_scale),
        "applied_scale": float(applied_scale),
        "crop_box": [int(v) for v in crop_box],
        "crop_box_format": "h_start_w_start_h_w_after_resize",
        "hflip_applied": hflip,
    }


def apply_photometric_rgb(rgb01: np.ndarray, *, rng: np.random.Generator, cfg: LODAugConfig) -> np.ndarray:
    out = rgb01.astype(np.float32, copy=False)
    if cfg.rgb_brightness > 0.0:
        out = out * _rand_uniform(rng, 1.0 - cfg.rgb_brightness, 1.0 + cfg.rgb_brightness)
    if cfg.rgb_contrast > 0.0:
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        out = (out - mean) * _rand_uniform(rng, 1.0 - cfg.rgb_contrast, 1.0 + cfg.rgb_contrast) + mean
    if cfg.rgb_gamma is not None:
        gamma = _rand_uniform(rng, cfg.rgb_gamma[0], cfg.rgb_gamma[1])
        out = np.power(np.clip(out, 0.0, 1.0), gamma)
    if cfg.rgb_color > 0.0:
        gray = np.sum(out * np.array([0.299, 0.587, 0.114], dtype=np.float32), axis=-1, keepdims=True)
        out = gray + (out - gray) * _rand_uniform(rng, 1.0 - cfg.rgb_color, 1.0 + cfg.rgb_color)
    if cfg.rgb_noise_std > 0.0:
        out = out + rng.normal(0.0, cfg.rgb_noise_std, size=out.shape).astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    if cfg.rgb_blur_prob > 0.0 and rng.random() < cfg.rgb_blur_prob:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32, copy=False))


def apply_photometric_raw(raw01: np.ndarray, *, rng: np.random.Generator, cfg: LODAugConfig) -> np.ndarray:
    out = raw01.astype(np.float32, copy=False)
    if cfg.raw_gain is not None:
        if cfg.raw_per_channel_gain:
            gains = rng.uniform(cfg.raw_gain[0], cfg.raw_gain[1], size=(1, 1, out.shape[-1])).astype(np.float32)
        else:
            gains = np.float32(_rand_uniform(rng, cfg.raw_gain[0], cfg.raw_gain[1]))
        out = out * gains
    if cfg.raw_black_offset > 0.0:
        out = out + _rand_uniform(rng, -cfg.raw_black_offset, cfg.raw_black_offset)
    if cfg.raw_noise is not None:
        shot, read = cfg.raw_noise
        variance = np.maximum(out, 0.0) * float(shot) + float(read) ** 2
        out = out + rng.normal(0.0, np.sqrt(variance), size=out.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32, copy=False))
