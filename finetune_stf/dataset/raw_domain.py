from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _as_channel_array(value: Any, *, name: str, channels: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return np.full((channels,), float(value), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full((channels,), float(array), dtype=np.float32)
    if array.shape != (channels,):
        raise ValueError(f"{name} must be a scalar or a {channels}-element sequence, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "black_level" in normalized and "pedestal" not in normalized:
        normalized["pedestal"] = normalized.pop("black_level")
    if "white_level" in normalized and "scale" not in normalized:
        normalized["scale"] = normalized.pop("white_level")
    if "power" in normalized and "gamma" not in normalized:
        normalized["gamma"] = normalized.pop("power")
    return normalized


def _load_config_payload(value: str | Path) -> dict[str, Any]:
    text = str(value).strip()
    if not text or text.lower() in {"none", "identity"}:
        return {}
    path = Path(text).expanduser()
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Raw-domain config file must contain a JSON object: {path}")
        return payload
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "raw_domain_config must be 'identity', a JSON object string, or a path to a JSON file"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("raw_domain_config JSON must be an object")
    return payload


@dataclass(frozen=True)
class RawDomainConfig:
    pedestal: Any = None
    scale: Any = None
    gamma: float | None = None
    oetf: str | None = None
    lut_path: str | None = None
    lut_key: str = "lut"
    quantize_bits: int | None = None
    lowpass_kernel: int | None = None
    clip_min: float = 0.0
    clip_max: float = 1.0
    clip: bool = True

    @classmethod
    def from_any(cls, value: Any) -> "RawDomainConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**_normalise_payload(value))
        if isinstance(value, (str, Path)):
            return cls(**_normalise_payload(_load_config_payload(value)))
        raise TypeError(f"Unsupported raw_domain_config type: {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pedestal": self.pedestal,
            "scale": self.scale,
            "gamma": self.gamma,
            "oetf": self.oetf,
            "lut_path": self.lut_path,
            "lut_key": self.lut_key,
            "quantize_bits": self.quantize_bits,
            "lowpass_kernel": self.lowpass_kernel,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "clip": self.clip,
        }

    def active_dict(self) -> dict[str, Any]:
        active = {key: value for key, value in self.to_dict().items() if value is not None}
        if active.get("clip_min") == 0.0:
            active.pop("clip_min")
        if active.get("clip_max") == 1.0:
            active.pop("clip_max")
        if active.get("clip") is True:
            active.pop("clip")
        return active

    def is_identity(self) -> bool:
        return not self.active_dict()

    def stable_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    def stable_hash(self) -> str:
        return hashlib.sha1(self.stable_json().encode("utf-8")).hexdigest()[:12]

    def describe(self) -> str:
        if self.is_identity():
            return "identity"
        return f"sha1:{self.stable_hash()} {self.stable_json()}"


def parse_raw_domain_config(value: Any) -> RawDomainConfig:
    return RawDomainConfig.from_any(value)


def _srgb_oetf(raw: np.ndarray) -> np.ndarray:
    linear = np.clip(raw, 0.0, None)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def _load_lut(config: RawDomainConfig) -> np.ndarray:
    if config.lut_path is None:
        raise ValueError("lut_path is required")
    path = Path(config.lut_path).expanduser().resolve()
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if config.lut_key not in data.files:
                raise KeyError(f"{path} does not contain LUT key {config.lut_key!r}")
            lut = np.asarray(data[config.lut_key], dtype=np.float32)
    else:
        lut = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if lut.ndim not in {1, 2}:
        raise ValueError(f"LUT must be shape (N,), (N,2), (4,N), or (N,4), got {lut.shape}")
    return lut


def _apply_lut(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    if lut.ndim == 1:
        x = np.linspace(0.0, 1.0, lut.shape[0], dtype=np.float32)
        return np.interp(raw, x, lut).astype(np.float32, copy=False)
    if lut.shape[1] == 2:
        return np.interp(raw, lut[:, 0], lut[:, 1]).astype(np.float32, copy=False)
    if raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError(f"Per-channel LUT requires raw shape (H,W,4), got {raw.shape}")
    if lut.shape[0] == 4:
        per_channel_lut = lut
    elif lut.shape[1] == 4:
        per_channel_lut = lut.T
    else:
        raise ValueError(f"Unsupported per-channel LUT shape: {lut.shape}")
    out = np.empty_like(raw, dtype=np.float32)
    x = np.linspace(0.0, 1.0, per_channel_lut.shape[1], dtype=np.float32)
    for channel in range(4):
        out[..., channel] = np.interp(raw[..., channel], x, per_channel_lut[channel]).astype(np.float32)
    return out


def apply_raw_domain_transform(raw: np.ndarray, config: RawDomainConfig | dict[str, Any] | str | None) -> np.ndarray:
    cfg = parse_raw_domain_config(config)
    image = np.asarray(raw, dtype=np.float32)
    channels = image.shape[-1] if image.ndim >= 3 else 1

    pedestal = _as_channel_array(cfg.pedestal, name="pedestal", channels=channels)
    if pedestal is not None:
        image = image - pedestal.reshape((1,) * (image.ndim - 1) + (channels,))

    scale = _as_channel_array(cfg.scale, name="scale", channels=channels)
    if scale is not None:
        if np.any(scale <= 0):
            raise ValueError(f"scale must be positive, got {scale}")
        image = image / scale.reshape((1,) * (image.ndim - 1) + (channels,))

    if cfg.clip:
        image = np.clip(image, float(cfg.clip_min), float(cfg.clip_max))

    if cfg.oetf:
        oetf = str(cfg.oetf).strip().lower()
        if oetf == "srgb":
            image = _srgb_oetf(image)
        else:
            raise ValueError(f"Unsupported raw-domain oetf: {cfg.oetf}")

    if cfg.gamma is not None:
        gamma = float(cfg.gamma)
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        image = np.power(np.clip(image, 0.0, None), gamma)

    if cfg.lut_path:
        image = _apply_lut(image, _load_lut(cfg))

    if cfg.quantize_bits is not None:
        bits = int(cfg.quantize_bits)
        if bits < 1:
            raise ValueError(f"quantize_bits must be >= 1, got {bits}")
        levels = float((1 << bits) - 1)
        image = np.round(np.clip(image, 0.0, 1.0) * levels) / levels

    if cfg.lowpass_kernel is not None:
        kernel = int(cfg.lowpass_kernel)
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError(f"lowpass_kernel must be a positive odd integer, got {kernel}")
        if kernel > 1:
            image = cv2.blur(image, (kernel, kernel))

    if cfg.clip:
        image = np.clip(image, float(cfg.clip_min), float(cfg.clip_max))
    return np.ascontiguousarray(image.astype(np.float32, copy=False))
