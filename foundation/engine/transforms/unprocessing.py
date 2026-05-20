from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor

RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

# Brooks et al. 2019 official XYZ->camera matrices.
XYZ_TO_CAM_LIBRARY = (
    (
        (1.0234, -0.2969, -0.2266),
        (-0.5625, 1.6328, -0.0469),
        (-0.0703, 0.2188, 0.6406),
    ),
    (
        (0.4913, -0.0541, -0.0202),
        (-0.6130, 1.3513, 0.2906),
        (-0.1564, 0.2151, 0.7183),
    ),
    (
        (0.8380, -0.2630, -0.0639),
        (-0.2887, 1.0725, 0.2496),
        (-0.0627, 0.1427, 0.5438),
    ),
    (
        (0.6596, -0.2079, -0.0562),
        (-0.4782, 1.3016, 0.1933),
        (-0.0970, 0.1581, 0.5181),
    ),
)

CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
CANONICAL_CHANNELS = ("R", "Gr", "Gb", "B")
PATTERN_TO_OFFSETS = {
    "RGGB": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)},
    "BGGR": {"R": (1, 1), "Gr": (1, 0), "Gb": (0, 1), "B": (0, 0)},
    "GRBG": {"R": (0, 1), "Gr": (0, 0), "Gb": (1, 1), "B": (1, 0)},
    "GBRG": {"R": (1, 0), "Gr": (1, 1), "Gb": (0, 0), "B": (0, 1)},
}
GREEN_CHANNEL_INDEX = {"R": 0, "Gr": 1, "Gb": 1, "B": 2}

UNPROCESSING_PRESET_VERSION = "2026-04-24.sensor_linear_v1"


def _build_single_preset(
    *,
    name: str,
    profile_group: str,
    red_gain_range: Tuple[float, float],
    blue_gain_range: Tuple[float, float],
    black_level_range: Tuple[float, float],
    shot_log_gain_range: Tuple[float, float],
    read_noise_std_range: Tuple[float, float],
    exposure_gain_range: Tuple[float, float],
    cfa_patterns: Sequence[str],
    randomize_ccm: bool = True,
    xyz_to_cam_override: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, object]:
    payload = {
        "name": str(name),
        "kind": "single",
        "isp_profile_group": str(profile_group),
        "preset_version": str(UNPROCESSING_PRESET_VERSION),
        "red_gain_range": [float(red_gain_range[0]), float(red_gain_range[1])],
        "blue_gain_range": [float(blue_gain_range[0]), float(blue_gain_range[1])],
        "black_level_range": [float(black_level_range[0]), float(black_level_range[1])],
        "shot_log_gain_range": [float(shot_log_gain_range[0]), float(shot_log_gain_range[1])],
        "read_noise_std_range": [float(read_noise_std_range[0]), float(read_noise_std_range[1])],
        "exposure_gain_range": [float(exposure_gain_range[0]), float(exposure_gain_range[1])],
        "cfa_patterns": [str(pattern) for pattern in cfa_patterns],
    }
    if randomize_ccm is False:
        payload["randomize_ccm"] = False
    if xyz_to_cam_override is not None:
        rows = [[float(value) for value in row] for row in xyz_to_cam_override]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError(f"xyz_to_cam_override must be a 3x3 matrix, got {rows}")
        payload["xyz_to_cam_override"] = rows
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["preset_hash"] = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]
    return payload


def _build_dual_preset(
    *,
    name: str,
    profile_group: str,
    sub_presets: Sequence[str],
    default_mix_weights: Sequence[float],
    default_sub_preset: str,
) -> Dict[str, object]:
    if len(sub_presets) != len(default_mix_weights):
        raise ValueError("sub_presets and default_mix_weights must have the same length")
    payload = {
        "name": str(name),
        "kind": "dual",
        "isp_profile_group": str(profile_group),
        "preset_version": str(UNPROCESSING_PRESET_VERSION),
        "sub_presets": [str(item) for item in sub_presets],
        "default_mix_weights": [float(item) for item in default_mix_weights],
        "default_sub_preset": str(default_sub_preset),
    }
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["preset_hash"] = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()[:16]
    return payload


def _build_preset_registry() -> Dict[str, Dict[str, object]]:
    presets = {
        "stf_legacy": _build_single_preset(
            name="stf_legacy",
            profile_group="stf_legacy",
            red_gain_range=(0.9358269414919648, 1.3501703909326341),
            blue_gain_range=(0.8537964935741034, 1.027460389099097),
            black_level_range=(0.0, 1e-4),
            shot_log_gain_range=(-10.182361082848972, -6.706375051522766),
            read_noise_std_range=(1e-5, 0.008270222169793597),
            exposure_gain_range=(0.34709084358450276, 1.595037682788309),
            cfa_patterns=("GBRG",),
        ),
        "eth3d_sensor_linear": _build_single_preset(
            name="eth3d_sensor_linear",
            profile_group="sensor_linear",
            red_gain_range=(1.4736417770, 2.2989409924),
            blue_gain_range=(1.2264200807, 2.6061923981),
            black_level_range=(0.0, 1e-4),
            shot_log_gain_range=(-11.5129254650, -5.3546184871),
            read_noise_std_range=(1e-5, 0.0146185911),
            exposure_gain_range=(0.2394525580, 0.6099897464),
            cfa_patterns=("RGGB",),
        ),
        "robotcar_subset100_sensor_linear": _build_single_preset(
            name="robotcar_subset100_sensor_linear",
            profile_group="sensor_linear",
            red_gain_range=(0.9761674166, 1.0617214918),
            blue_gain_range=(0.8610724926, 0.9190393567),
            black_level_range=(0.0, 1e-4),
            shot_log_gain_range=(-8.6333074808, -7.2768236395),
            read_noise_std_range=(0.0015373475, 0.0109663301),
            exposure_gain_range=(1.7404851992, 2.4956310581),
            cfa_patterns=("GBRG",),
        ),
        "robotcar_subset100_sensor_linear_fixccm": _build_single_preset(
            name="robotcar_subset100_sensor_linear_fixccm",
            profile_group="sensor_linear",
            red_gain_range=(0.9761674166, 1.0617214918),
            blue_gain_range=(0.8610724926, 0.9190393567),
            black_level_range=(0.0, 1e-4),
            shot_log_gain_range=(-8.6333074808, -7.2768236395),
            read_noise_std_range=(0.0015373475, 0.0109663301),
            exposure_gain_range=(1.7404851992, 2.4956310581),
            cfa_patterns=("GBRG",),
            randomize_ccm=False,
        ),
        "robotcar_night_sensor_linear": _build_single_preset(
            name="robotcar_night_sensor_linear",
            profile_group="sensor_linear",
            red_gain_range=(0.6146325721831747, 0.8074631109936032),
            blue_gain_range=(1.380053190197515, 2.513052384657433),
            black_level_range=(0.0, 0.0001),
            shot_log_gain_range=(-7.2359660421091325, -4.8867059346137545),
            read_noise_std_range=(1e-05, 0.014850541816568075),
            exposure_gain_range=(0.13528898507728163, 1.377187093213195),
            cfa_patterns=("GBRG",),
        ),
        "robotcar_public_gbrg_generic": _build_single_preset(
            name="robotcar_public_gbrg_generic",
            profile_group="public_sensor_generic",
            red_gain_range=(1.2, 2.4),
            blue_gain_range=(1.2, 2.4),
            black_level_range=(0.0, 0.02),
            shot_log_gain_range=(-11.5129254650, -5.3546184871),
            read_noise_std_range=(1e-5, 0.02),
            exposure_gain_range=(0.25, 1.5),
            cfa_patterns=("GBRG",),
            randomize_ccm=True,
        ),
    }
    presets["sensor_linear_dual"] = _build_dual_preset(
        name="sensor_linear_dual",
        profile_group="sensor_linear",
        sub_presets=("eth3d_sensor_linear", "robotcar_subset100_sensor_linear"),
        default_mix_weights=(0.5, 0.5),
        default_sub_preset="eth3d_sensor_linear",
    )
    presets["robotcar_day_night_sensor_linear_dual"] = _build_dual_preset(
        name="robotcar_day_night_sensor_linear_dual",
        profile_group="sensor_linear",
        sub_presets=("robotcar_subset100_sensor_linear", "robotcar_night_sensor_linear"),
        default_mix_weights=(0.5, 0.5),
        default_sub_preset="robotcar_subset100_sensor_linear",
    )
    return presets


UNPROCESSING_PRESETS: Mapping[str, Mapping[str, object]] = MappingProxyType(_build_preset_registry())


def list_unprocessing_presets() -> Tuple[str, ...]:
    return tuple(UNPROCESSING_PRESETS.keys())


def get_unprocessing_preset(name: str) -> Dict[str, object]:
    key = str(name)
    if key not in UNPROCESSING_PRESETS:
        available = ", ".join(sorted(UNPROCESSING_PRESETS))
        raise KeyError(f"Unknown unprocessing preset '{key}'. Available presets: {available}")
    preset = copy.deepcopy(dict(UNPROCESSING_PRESETS[key]))
    if preset.get("kind") == "single":
        preset.setdefault("randomize_ccm", True)
        preset.setdefault("xyz_to_cam_override", None)
    return preset


def _normalize_weight_vector(weights: Sequence[float]) -> Tuple[float, ...]:
    values = [float(item) for item in weights]
    if any(item < 0 for item in values):
        raise ValueError(f"mix weights must be >= 0, got {values}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"mix weights must sum to > 0, got {values}")
    return tuple(item / total for item in values)


def _parse_mix_weights(
    mix_weights: object,
    *,
    sub_presets: Sequence[str],
) -> Tuple[float, ...]:
    if isinstance(mix_weights, Mapping):
        values = [float(mix_weights.get(name, 0.0)) for name in sub_presets]
        return _normalize_weight_vector(values)

    if isinstance(mix_weights, str):
        text = mix_weights.strip()
        if not text:
            raise ValueError("vkitti unprocessing mix weights string is empty")

        if text.startswith("{") or text.startswith("["):
            parsed = json.loads(text)
            return _parse_mix_weights(parsed, sub_presets=sub_presets)

        if "=" in text:
            mapping: Dict[str, float] = {}
            for term in text.split(","):
                chunk = term.strip()
                if not chunk:
                    continue
                if "=" not in chunk:
                    raise ValueError(
                        "mix weights with names must use key=value pairs, "
                        f"got '{mix_weights}'"
                    )
                key, value = chunk.split("=", 1)
                mapping[key.strip()] = float(value.strip())
            return _parse_mix_weights(mapping, sub_presets=sub_presets)

        values = [float(item.strip()) for item in text.split(",") if item.strip()]
        return _parse_mix_weights(values, sub_presets=sub_presets)

    if isinstance(mix_weights, Sequence):
        values = [float(item) for item in mix_weights]
        if len(values) != len(sub_presets):
            raise ValueError(
                f"Expected {len(sub_presets)} mix weights for {list(sub_presets)}, got {values}"
            )
        return _normalize_weight_vector(values)

    raise TypeError(
        "mix_weights must be one of: comma-separated string, key=value string, list/tuple, or mapping"
    )


def resolve_unprocessing_mix_weights(
    preset_name: str,
    mix_weights: Optional[object] = None,
) -> Dict[str, float]:
    preset = get_unprocessing_preset(preset_name)
    if preset["kind"] == "single":
        if mix_weights not in (None, ""):
            raise ValueError(
                f"Preset '{preset_name}' is single-domain and does not accept mix weights, got {mix_weights!r}"
            )
        return {str(preset["name"]): 1.0}

    sub_presets = tuple(str(item) for item in preset["sub_presets"])
    if mix_weights is None:
        weights = _normalize_weight_vector(preset["default_mix_weights"])
    else:
        weights = _parse_mix_weights(mix_weights, sub_presets=sub_presets)
    return {name: float(weight) for name, weight in zip(sub_presets, weights)}


def build_unprocessing_transform_from_preset(
    name: str,
    *,
    randomize: bool = True,
    eps: float = 1e-8,
    sub_preset_name: Optional[str] = None,
) -> "UnprocessingTransform":
    preset = get_unprocessing_preset(name)
    if preset["kind"] == "dual":
        selected_sub = str(sub_preset_name or preset["default_sub_preset"])
        allowed = {str(item) for item in preset["sub_presets"]}
        if selected_sub not in allowed:
            raise ValueError(
                f"sub_preset_name='{selected_sub}' must be one of {sorted(allowed)} for preset '{name}'"
            )
        preset = get_unprocessing_preset(selected_sub)

    return UnprocessingTransform(
        randomize=randomize,
        red_gain_range=tuple(float(v) for v in preset["red_gain_range"]),
        blue_gain_range=tuple(float(v) for v in preset["blue_gain_range"]),
        black_level_range=tuple(float(v) for v in preset["black_level_range"]),
        shot_log_gain_range=tuple(float(v) for v in preset["shot_log_gain_range"]),
        read_noise_std_range=tuple(float(v) for v in preset["read_noise_std_range"]),
        exposure_gain_range=tuple(float(v) for v in preset["exposure_gain_range"]),
        cfa_patterns=tuple(str(v) for v in preset["cfa_patterns"]),
        eps=eps,
        preset_name=str(preset["name"]),
        preset_group=str(preset["isp_profile_group"]),
        preset_version=str(preset["preset_version"]),
        preset_hash=str(preset["preset_hash"]),
        randomize_ccm=bool(preset.get("randomize_ccm", True)),
        xyz_to_cam_override=preset.get("xyz_to_cam_override"),
    )


@dataclass(frozen=True)
class CanonicalParams:
    red_gain: float
    blue_gain: float
    black_level: float
    shot_log_gain: float
    read_noise_std: float
    exposure_gain: float
    cfa_pattern: str


def packed_bayer_to_base_rgb(packed_bayer: Tensor) -> Tensor:
    """Maps packed Bayer [R, Gr, Gb, B] to [R, (Gr+Gb)/2, B]."""
    if packed_bayer.ndim == 3:
        packed_bayer = packed_bayer.unsqueeze(0)
        squeeze = True
    elif packed_bayer.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected packed Bayer with shape (4,H,W) or (N,4,H,W), got {tuple(packed_bayer.shape)}")

    rgb = torch.cat(
        [
            packed_bayer[:, 0:1],
            0.5 * (packed_bayer[:, 1:2] + packed_bayer[:, 2:3]),
            packed_bayer[:, 3:4],
        ],
        dim=1,
    )
    return rgb.squeeze(0) if squeeze else rgb


class UnprocessingTransform(nn.Module):
    """PyTorch re-implementation of Brooks-style unprocessing with extra RAW randomization.

    Prefer constructing this via `build_unprocessing_transform_from_preset(...)`
    to avoid accidentally relying on broad legacy default ranges.
    """

    def __init__(
        self,
        *,
        randomize: bool = True,
        red_gain_range: Tuple[float, float] = (1.2, 2.4),
        blue_gain_range: Tuple[float, float] = (1.2, 2.4),
        black_level_range: Tuple[float, float] = (0.0, 0.02),
        shot_log_gain_range: Tuple[float, float] = (-11.5129254650, -5.3546184871),
        read_noise_std_range: Tuple[float, float] = (1e-5, 0.02),
        exposure_gain_range: Tuple[float, float] = (0.25, 1.5),
        cfa_patterns: Sequence[str] = ("RGGB", "BGGR", "GRBG", "GBRG"),
        eps: float = 1e-8,
        preset_name: Optional[str] = None,
        preset_group: Optional[str] = None,
        preset_version: Optional[str] = None,
        preset_hash: Optional[str] = None,
        randomize_ccm: bool = True,
        xyz_to_cam_override: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        super().__init__()
        if not cfa_patterns:
            raise ValueError("cfa_patterns must not be empty")
        for pattern in cfa_patterns:
            if pattern not in PATTERN_TO_OFFSETS:
                raise ValueError(f"Unsupported CFA pattern: {pattern}")

        self.randomize = bool(randomize)
        self.red_gain_range = tuple(float(v) for v in red_gain_range)
        self.blue_gain_range = tuple(float(v) for v in blue_gain_range)
        self.black_level_range = tuple(float(v) for v in black_level_range)
        self.shot_log_gain_range = tuple(float(v) for v in shot_log_gain_range)
        self.read_noise_std_range = tuple(float(v) for v in read_noise_std_range)
        self.exposure_gain_range = tuple(float(v) for v in exposure_gain_range)
        if self.exposure_gain_range[0] <= 0 or self.exposure_gain_range[0] > self.exposure_gain_range[1]:
            raise ValueError(f"exposure_gain_range must satisfy 0 < low <= high, got {self.exposure_gain_range}")
        self.cfa_patterns = tuple(cfa_patterns)
        self.eps = float(eps)
        self.preset_name = str(preset_name) if preset_name else None
        self.preset_group = str(preset_group) if preset_group else None
        self.preset_version = str(preset_version) if preset_version else None
        self.preset_hash = str(preset_hash) if preset_hash else None
        self.randomize_ccm = bool(randomize_ccm)
        self.canonical_params = CanonicalParams(
            red_gain=0.5 * sum(self.red_gain_range),
            blue_gain=0.5 * sum(self.blue_gain_range),
            black_level=0.5 * sum(self.black_level_range),
            shot_log_gain=0.5 * sum(self.shot_log_gain_range),
            read_noise_std=0.5 * sum(self.read_noise_std_range),
            exposure_gain=1.0,
            cfa_pattern=self.cfa_patterns[0],
        )

        self.register_buffer("rgb_to_xyz", torch.tensor(RGB_TO_XYZ, dtype=torch.float32), persistent=False)
        self.register_buffer("xyz_to_cam_library", torch.tensor(XYZ_TO_CAM_LIBRARY, dtype=torch.float32), persistent=False)
        xyz_to_cam_tensor: Optional[Tensor]
        if xyz_to_cam_override is None:
            xyz_to_cam_tensor = None
        else:
            xyz_to_cam_tensor = torch.tensor(xyz_to_cam_override, dtype=torch.float32)
            if tuple(xyz_to_cam_tensor.shape) != (3, 3):
                raise ValueError(
                    "xyz_to_cam_override must have shape (3, 3), "
                    f"got {tuple(xyz_to_cam_tensor.shape)}"
                )
        self.register_buffer("xyz_to_cam_override", xyz_to_cam_tensor, persistent=False)
        bilinear_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("demosaic_kernel", bilinear_kernel.repeat(3, 1, 1, 1), persistent=False)

    def forward(self, image: Tensor, *, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Dict[str, object]]:
        image, squeeze = self._prepare_input(image)
        batch_size = image.shape[0]
        params = self._sample_params(
            batch_size=batch_size,
            device=image.device,
            dtype=image.dtype,
            generator=generator,
        )

        image = image.clamp(0.0, 1.0)
        image = image.permute(0, 2, 3, 1).contiguous()
        image = self.srgb_to_linear(image)
        image = self.inverse_smoothstep(image)
        image = self.apply_ccm(image, params["rgb2cam"])
        image = self.safe_invert_white_balance(image, params["red_gain"], params["blue_gain"])
        image = image * params["exposure_gain"].view(batch_size, 1, 1, 1)
        packed = self.pack_bayer(image, params["cfa_pattern"])
        packed = packed.clamp_min(0.0)
        packed = packed + params["black_level"].view(batch_size, 1, 1, 1)

        if params["noise_applied"]:
            packed = self.add_poisson_gaussian_noise(
                packed,
                shot_noise_scale=params["shot_noise_scale"],
                read_noise_std=params["read_noise_std"],
                generator=generator,
            )

        packed = packed.clamp(0.0, 1.0)
        metadata = self._finalize_metadata(params, batch_size=batch_size, squeeze=squeeze)
        return (packed.squeeze(0) if squeeze else packed), metadata

    @torch.no_grad()
    def reprocess(
        self,
        packed_bayer: Tensor,
        metadata: Mapping[str, object],
        *,
        apply_tone_mapping: bool = True,
    ) -> Tensor:
        packed_bayer, squeeze = self._prepare_packed_input(packed_bayer)
        metadata = self._expand_metadata_for_batch(metadata, batch_size=packed_bayer.shape[0], device=packed_bayer.device, dtype=packed_bayer.dtype)

        black_level = metadata["black_level"].view(packed_bayer.shape[0], 1, 1, 1)
        packed_bayer = (packed_bayer - black_level).clamp(0.0, 1.0)
        exposure_gain = metadata["exposure_gain"].view(packed_bayer.shape[0], 1, 1, 1)
        packed_bayer = packed_bayer / exposure_gain.clamp_min(self.eps)
        demosaiced = self.demosaic(packed_bayer, metadata["cfa_pattern"])
        demosaiced = self.apply_white_balance(demosaiced, metadata["red_gain"], metadata["blue_gain"])
        demosaiced = self.apply_ccm(demosaiced.permute(0, 2, 3, 1), metadata["cam2rgb"]).permute(0, 3, 1, 2)
        demosaiced = demosaiced.clamp(0.0, 1.0)

        if apply_tone_mapping:
            demosaiced = self.smoothstep(demosaiced)
        demosaiced = self.linear_to_srgb(demosaiced)
        demosaiced = demosaiced.clamp(0.0, 1.0)
        return demosaiced.squeeze(0) if squeeze else demosaiced

    def srgb_to_linear(self, image: Tensor) -> Tensor:
        return image.clamp_min(self.eps).pow(2.2)

    def linear_to_srgb(self, image: Tensor) -> Tensor:
        return image.clamp_min(self.eps).pow(1.0 / 2.2)

    def inverse_smoothstep(self, image: Tensor) -> Tensor:
        image = image.clamp(0.0, 1.0)
        return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)

    def smoothstep(self, image: Tensor) -> Tensor:
        image = image.clamp(0.0, 1.0)
        return image * image * (3.0 - 2.0 * image)

    def apply_ccm(self, image: Tensor, ccm: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Expected image with shape (N,H,W,3), got {tuple(image.shape)}")
        if ccm.ndim != 3 or ccm.shape[1:] != (3, 3):
            raise ValueError(f"Expected CCM with shape (N,3,3), got {tuple(ccm.shape)}")
        return torch.einsum("nhwc,nkc->nhwk", image, ccm)

    def safe_invert_white_balance(self, image: Tensor, red_gain: Tensor, blue_gain: Tensor) -> Tensor:
        gains = torch.stack(
            [
                1.0 / red_gain,
                torch.ones_like(red_gain),
                1.0 / blue_gain,
            ],
            dim=-1,
        ).view(-1, 1, 1, 3)
        gray = image.mean(dim=-1, keepdim=True)
        inflection = 0.9
        mask = ((gray - inflection).clamp_min(0.0) / (1.0 - inflection)) ** 2.0
        safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)
        return image * safe_gains

    def apply_white_balance(self, image: Tensor, red_gain: Tensor, blue_gain: Tensor) -> Tensor:
        gains = torch.stack(
            [
                red_gain,
                torch.ones_like(red_gain),
                blue_gain,
            ],
            dim=1,
        ).view(-1, 3, 1, 1)
        return image * gains

    def add_poisson_gaussian_noise(
        self,
        image: Tensor,
        *,
        shot_noise_scale: Tensor,
        read_noise_std: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        shot_noise_scale = shot_noise_scale.view(-1, 1, 1, 1)
        read_noise_std = read_noise_std.view(-1, 1, 1, 1)
        photon_counts = (image / shot_noise_scale.clamp_min(self.eps)).clamp(0.0, 1e4)
        shot = torch.poisson(photon_counts, generator=generator) * shot_noise_scale
        read = torch.randn(
            image.shape,
            device=image.device,
            dtype=image.dtype,
            generator=generator,
        ) * read_noise_std
        return shot + read

    def pack_bayer(self, image: Tensor, patterns: Sequence[str]) -> Tensor:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Expected image with shape (N,H,W,3), got {tuple(image.shape)}")

        batch_size, height, width, _ = image.shape
        packed_planes: List[Tensor] = []
        for sample_idx in range(batch_size):
            sample = image[sample_idx]
            pattern = patterns[sample_idx]
            offsets = PATTERN_TO_OFFSETS[pattern]
            sample_planes = []
            for channel_name in CANONICAL_CHANNELS:
                row_offset, col_offset = offsets[channel_name]
                color_index = GREEN_CHANNEL_INDEX[channel_name]
                plane = sample[row_offset::2, col_offset::2, color_index]
                sample_planes.append(plane)
            packed_planes.append(torch.stack(sample_planes, dim=0))
        return torch.stack(packed_planes, dim=0)

    def unpack_bayer(self, packed_bayer: Tensor, patterns: Sequence[str]) -> Tuple[Tensor, Tensor]:
        if packed_bayer.ndim != 4 or packed_bayer.shape[1] != 4:
            raise ValueError(f"Expected packed Bayer with shape (N,4,H,W), got {tuple(packed_bayer.shape)}")

        batch_size, _, height, width = packed_bayer.shape
        full_height = height * 2
        full_width = width * 2
        mosaic = packed_bayer.new_zeros((batch_size, 3, full_height, full_width))
        masks = packed_bayer.new_zeros((batch_size, 3, full_height, full_width))

        for sample_idx in range(batch_size):
            offsets = PATTERN_TO_OFFSETS[patterns[sample_idx]]
            for packed_idx, channel_name in enumerate(CANONICAL_CHANNELS):
                row_offset, col_offset = offsets[channel_name]
                color_index = GREEN_CHANNEL_INDEX[channel_name]
                mosaic[sample_idx, color_index, row_offset::2, col_offset::2] = packed_bayer[sample_idx, packed_idx]
                masks[sample_idx, color_index, row_offset::2, col_offset::2] = 1.0
        return mosaic, masks

    def demosaic(self, packed_bayer: Tensor, patterns: Sequence[str]) -> Tensor:
        mosaic, masks = self.unpack_bayer(packed_bayer, patterns)
        kernel = self.demosaic_kernel.to(device=packed_bayer.device, dtype=packed_bayer.dtype)
        numerator = F.conv2d(mosaic, kernel, padding=1, groups=3)
        denominator = F.conv2d(masks, kernel, padding=1, groups=3).clamp_min(self.eps)
        return numerator / denominator

    def _sample_params(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Dict[str, object]:
        rgb2cam = self._sample_rgb2cam(batch_size=batch_size, device=device, dtype=dtype, generator=generator)
        cam2rgb = torch.linalg.inv(rgb2cam)

        if self.randomize:
            red_gain = self._sample_uniform(self.red_gain_range, batch_size, device, dtype, generator)
            blue_gain = self._sample_uniform(self.blue_gain_range, batch_size, device, dtype, generator)
            black_level = self._sample_uniform(self.black_level_range, batch_size, device, dtype, generator)
            shot_log_gain = self._sample_uniform(self.shot_log_gain_range, batch_size, device, dtype, generator)
            read_noise_std = self._sample_uniform(self.read_noise_std_range, batch_size, device, dtype, generator)
            exposure_gain = self._sample_uniform(self.exposure_gain_range, batch_size, device, dtype, generator)
            cfa_indices = torch.randint(len(self.cfa_patterns), (batch_size,), device=device, generator=generator)
            cfa_pattern = [self.cfa_patterns[idx] for idx in cfa_indices.tolist()]
            noise_applied = True
        else:
            red_gain = torch.full((batch_size,), self.canonical_params.red_gain, device=device, dtype=dtype)
            blue_gain = torch.full((batch_size,), self.canonical_params.blue_gain, device=device, dtype=dtype)
            black_level = torch.full((batch_size,), self.canonical_params.black_level, device=device, dtype=dtype)
            shot_log_gain = torch.full((batch_size,), self.canonical_params.shot_log_gain, device=device, dtype=dtype)
            read_noise_std = torch.full((batch_size,), self.canonical_params.read_noise_std, device=device, dtype=dtype)
            exposure_gain = torch.full((batch_size,), self.canonical_params.exposure_gain, device=device, dtype=dtype)
            cfa_pattern = [self.canonical_params.cfa_pattern] * batch_size
            noise_applied = False

        return {
            "rgb2cam": rgb2cam,
            "cam2rgb": cam2rgb,
            "red_gain": red_gain,
            "blue_gain": blue_gain,
            "black_level": black_level,
            "shot_log_gain": shot_log_gain,
            "shot_noise_scale": torch.exp(shot_log_gain),
            "read_noise_std": read_noise_std,
            "exposure_gain": exposure_gain,
            "cfa_pattern": cfa_pattern,
            "noise_applied": noise_applied,
            "randomize": self.randomize,
        }

    def _sample_rgb2cam(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        xyz_to_cam_library = self.xyz_to_cam_library.to(device=device, dtype=dtype)
        rgb_to_xyz = self.rgb_to_xyz.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

        if self.xyz_to_cam_override is not None:
            xyz_to_cam = self.xyz_to_cam_override.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        elif self.randomize_ccm and self.randomize:
            weights = torch.empty((batch_size, xyz_to_cam_library.shape[0]), device=device, dtype=dtype)
            weights.uniform_(1e-6, 1.0, generator=generator)
            weights = weights / weights.sum(dim=1, keepdim=True)
            xyz_to_cam = torch.einsum("bk,kij->bij", weights, xyz_to_cam_library)
        else:
            xyz_to_cam = xyz_to_cam_library.mean(dim=0, keepdim=True).expand(batch_size, -1, -1)

        rgb2cam = torch.matmul(xyz_to_cam, rgb_to_xyz)
        rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return rgb2cam

    def _sample_uniform(
        self,
        value_range: Tuple[float, float],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        values = torch.empty((batch_size,), device=device, dtype=dtype)
        values.uniform_(value_range[0], value_range[1], generator=generator)
        return values

    def _prepare_input(self, image: Tensor) -> Tuple[Tensor, bool]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        elif image.ndim == 4:
            squeeze = False
        else:
            raise ValueError(f"Expected image with shape (3,H,W) or (N,3,H,W), got {tuple(image.shape)}")

        if image.shape[1] != 3:
            raise ValueError(f"Expected image with 3 channels, got {image.shape[1]}")

        height = image.shape[-2] - (image.shape[-2] % 2)
        width = image.shape[-1] - (image.shape[-1] % 2)
        if height <= 0 or width <= 0:
            raise ValueError(f"Input image must have spatial size >= 2, got {tuple(image.shape[-2:])}")
        return image[..., :height, :width].contiguous(), squeeze

    def _prepare_packed_input(self, packed_bayer: Tensor) -> Tuple[Tensor, bool]:
        if packed_bayer.ndim == 3:
            packed_bayer = packed_bayer.unsqueeze(0)
            squeeze = True
        elif packed_bayer.ndim == 4:
            squeeze = False
        else:
            raise ValueError(
                f"Expected packed Bayer with shape (4,H,W) or (N,4,H,W), got {tuple(packed_bayer.shape)}"
            )

        if packed_bayer.shape[1] != 4:
            raise ValueError(f"Expected packed Bayer with 4 channels, got {packed_bayer.shape[1]}")
        return packed_bayer.contiguous(), squeeze

    def _expand_metadata_for_batch(
        self,
        metadata: Mapping[str, object],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, object]:
        expanded: Dict[str, object] = {}
        for key in ("red_gain", "blue_gain", "black_level", "exposure_gain"):
            value = metadata[key]
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=device, dtype=dtype)
            else:
                tensor = torch.tensor(value, device=device, dtype=dtype)
            if tensor.ndim == 0:
                tensor = tensor.repeat(batch_size)
            expanded[key] = tensor

        for key in ("rgb2cam", "cam2rgb"):
            value = metadata[key]
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=device, dtype=dtype)
            else:
                tensor = torch.tensor(value, device=device, dtype=dtype)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1)
            expanded[key] = tensor

        patterns = metadata["cfa_pattern"]
        if isinstance(patterns, str):
            expanded["cfa_pattern"] = [patterns] * batch_size
        else:
            expanded["cfa_pattern"] = list(patterns)
        return expanded

    def _xyz_to_cam_override_as_list(self) -> Optional[List[List[float]]]:
        if self.xyz_to_cam_override is None:
            return None
        return [
            [float(value) for value in row]
            for row in self.xyz_to_cam_override.detach().cpu().tolist()
        ]

    def _finalize_metadata(self, metadata: Dict[str, object], *, batch_size: int, squeeze: bool) -> Dict[str, object]:
        finalized = dict(metadata)
        finalized["packed_channel_order"] = list(CANONICAL_CHANNELS)
        finalized["noise_model"] = "poisson_gaussian"
        finalized["randomize_ccm"] = bool(self.randomize_ccm)
        finalized["xyz_to_cam_override"] = self._xyz_to_cam_override_as_list()
        if self.preset_name:
            finalized.setdefault("isp_profile_name", self.preset_name)
            finalized.setdefault("selected_sub_preset_name", self.preset_name)
        if self.preset_group:
            finalized.setdefault("isp_profile_group", self.preset_group)
        if self.preset_version:
            finalized.setdefault("preset_version", self.preset_version)
        if self.preset_hash:
            finalized.setdefault("preset_hash", self.preset_hash)
        if not squeeze:
            return finalized

        squeezed: Dict[str, object] = {}
        for key, value in finalized.items():
            if key == "xyz_to_cam_override":
                squeezed[key] = value
            elif isinstance(value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] == batch_size:
                    squeezed[key] = value[0]
                else:
                    squeezed[key] = value
            elif isinstance(value, list) and len(value) == batch_size:
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed
