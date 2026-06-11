"""ROD RAW24 to RawPy default RGB rendering helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from finetune_stf.dataset.rod_raw_rgb import RAW_H, RAW_W, ROD_NATIVE_HW, unpack_raw24


RAWPY_RGB_PIPELINE = "rawpy_default_dng_v1"
RAWPY_DNG_PROFILE = "rggb_uint16_srgb_d65_black0_white65535_v1"
RAWPY_POSTPROCESS_PROFILE = "default_kwargs_empty_v1"
RAWPY_POSTPROCESS_KWARGS: dict[str, object] = {}

DNG_CFA_PATTERN = (0, 1, 1, 2)  # RGGB in DNG color IDs: R, G, G, B.
DNG_CFA_PLANE_COLOR = (0, 1, 2)
DNG_COLOR_MATRIX_XYZ_TO_SRGB = (
    3240454,
    1000000,
    -1537138,
    1000000,
    -498531,
    1000000,
    -969266,
    1000000,
    1876011,
    1000000,
    41556,
    1000000,
    55643,
    1000000,
    -204026,
    1000000,
    1057225,
    1000000,
)
DNG_AS_SHOT_NEUTRAL = (1, 1, 1, 1, 1, 1)


def validate_rawpy_profiles(
    *,
    rgb_pipeline: str = RAWPY_RGB_PIPELINE,
    dng_profile: str = RAWPY_DNG_PROFILE,
    postprocess_profile: str = RAWPY_POSTPROCESS_PROFILE,
) -> None:
    if rgb_pipeline != RAWPY_RGB_PIPELINE:
        raise ValueError(f"ROD RawPy RGB requires rgb_pipeline={RAWPY_RGB_PIPELINE}")
    if dng_profile != RAWPY_DNG_PROFILE:
        raise ValueError(f"ROD RawPy RGB requires rawpy_dng_profile={RAWPY_DNG_PROFILE}")
    if postprocess_profile != RAWPY_POSTPROCESS_PROFILE:
        raise ValueError(f"ROD RawPy RGB requires rawpy_postprocess_profile={RAWPY_POSTPROCESS_PROFILE}")


def dng_extratags() -> list[tuple[int, str, int, object, bool]]:
    return [
        (271, "s", 0, "ROD", False),  # Make
        (272, "s", 0, "ROD_RAW24_DNG_RAWPY_DEBUG", False),  # Model
        (274, "H", 1, 1, False),  # Orientation
        (33421, "H", 2, (2, 2), False),  # CFARepeatPatternDim
        (33422, "B", 4, DNG_CFA_PATTERN, False),  # CFAPattern
        (50706, "B", 4, (1, 4, 0, 0), False),  # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), False),  # DNGBackwardVersion
        (50708, "s", 0, "ROD_RAW24_DNG_RAWPY_DEBUG", False),  # UniqueCameraModel
        (50710, "B", 3, DNG_CFA_PLANE_COLOR, False),  # CFAPlaneColor
        (50711, "H", 1, 1, False),  # CFALayout
        (50714, "H", 1, 0, False),  # BlackLevel
        (50717, "H", 1, 65535, False),  # WhiteLevel
        (50721, "2i", 9, DNG_COLOR_MATRIX_XYZ_TO_SRGB, False),  # ColorMatrix1
        (50728, "2I", 3, DNG_AS_SHOT_NEUTRAL, False),  # AsShotNeutral
        (50778, "H", 1, 21, False),  # CalibrationIlluminant1, D65
    ]


def raw_norm_to_mosaic16(raw_norm: np.ndarray) -> np.ndarray:
    if tuple(raw_norm.shape) != (RAW_H, RAW_W):
        raise ValueError(f"Expected full ROD Bayer shape {(RAW_H, RAW_W)}, got {tuple(raw_norm.shape)}")
    if float(np.nanmin(raw_norm)) < -1e-6 or float(np.nanmax(raw_norm)) > 1.0 + 1e-6:
        raise ValueError("ROD raw_norm must already be in [0, 1]; do not pass unnormalized RAW24 values")
    return np.round(np.clip(raw_norm, 0.0, 1.0) * 65535.0).astype(np.uint16)


def write_dng(raw_norm: np.ndarray, dng_path: str | Path) -> np.ndarray:
    import tifffile

    dng_path = Path(dng_path)
    mosaic16 = raw_norm_to_mosaic16(raw_norm)
    dng_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        dng_path,
        mosaic16,
        photometric=32803,
        compression=None,
        metadata=None,
        extratags=dng_extratags(),
    )
    return mosaic16


def rawpy_postprocess_default(dng_path: str | Path) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(dng_path)) as raw:
        return raw.postprocess()


def resize_to_native(rgb_full: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(
        rgb_full,
        (int(ROD_NATIVE_HW[1]), int(ROD_NATIVE_HW[0])),
        interpolation=cv2.INTER_AREA,
    )


def render_rawpy_default_rgb_from_raw_norm(
    raw_norm: np.ndarray,
    *,
    temp_root: str | Path | None = None,
    keep_temp_on_error: bool = False,
) -> np.ndarray:
    temp_dir = Path(temp_root).expanduser() if temp_root is not None else None
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)

    dng_path: Path | None = None
    success = False
    try:
        with tempfile.NamedTemporaryFile(prefix="rod_rawpy_", suffix=".dng", dir=temp_dir, delete=False) as tmp:
            dng_path = Path(tmp.name)
        write_dng(raw_norm, dng_path)
        rgb_full = rawpy_postprocess_default(dng_path)
        rgb_native = resize_to_native(rgb_full)
        success = True
        return rgb_native
    finally:
        if dng_path is not None and (success or not keep_temp_on_error):
            dng_path.unlink(missing_ok=True)


def render_rawpy_default_rgb_from_raw24(
    raw_path: str | Path,
    *,
    temp_root: str | Path | None = None,
    keep_temp_on_error: bool = False,
) -> np.ndarray:
    raw_norm = unpack_raw24(raw_path)
    return render_rawpy_default_rgb_from_raw_norm(
        raw_norm,
        temp_root=temp_root,
        keep_temp_on_error=keep_temp_on_error,
    )


def get_rawpy_runtime_versions() -> dict[str, object]:
    import rawpy
    import tifffile

    libraw_version = getattr(rawpy, "libraw_version", "unknown")
    if isinstance(libraw_version, tuple):
        libraw_version = list(libraw_version)
    return {
        "rawpy_version": getattr(rawpy, "__version__", "unknown"),
        "libraw_version": libraw_version,
        "tifffile_version": getattr(tifffile, "__version__", "unknown"),
    }


__all__ = [
    "RAWPY_RGB_PIPELINE",
    "RAWPY_DNG_PROFILE",
    "RAWPY_POSTPROCESS_PROFILE",
    "RAWPY_POSTPROCESS_KWARGS",
    "DNG_CFA_PATTERN",
    "DNG_CFA_PLANE_COLOR",
    "DNG_COLOR_MATRIX_XYZ_TO_SRGB",
    "DNG_AS_SHOT_NEUTRAL",
    "validate_rawpy_profiles",
    "dng_extratags",
    "raw_norm_to_mosaic16",
    "write_dng",
    "rawpy_postprocess_default",
    "resize_to_native",
    "render_rawpy_default_rgb_from_raw_norm",
    "render_rawpy_default_rgb_from_raw24",
    "get_rawpy_runtime_versions",
]
