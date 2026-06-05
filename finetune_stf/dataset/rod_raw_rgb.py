"""ROD 24-bit RAW to RGB rendering used by ROD-night experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


RAW_H = 1856
RAW_W = 2880
RAW_BYTES = RAW_H * RAW_W * 3
RAW_MAX = float(2**24 - 1)
ROD_NATIVE_HW = (928, 1440)

DEGREEN_GAINS = np.array([1.08, 0.95, 1.10], dtype=np.float32)
STUDENT_WHITE_PERCENTILE = 99.9
STUDENT_GAMMA = 0.9
TEACHER_WHITE_PERCENTILE = 99.5
TEACHER_GAMMA = 0.454545


def unpack_raw24(path: str | Path) -> np.ndarray:
    """Unpack one ROD packed RAW24 file to a normalized RGGB Bayer plane."""

    path = Path(path)
    raw_bytes = np.fromfile(path, dtype=np.uint8)
    if raw_bytes.size != RAW_BYTES:
        raise ValueError(f"{path} has {raw_bytes.size} bytes, expected {RAW_BYTES}")

    raw24 = (
        raw_bytes[0::3].astype(np.uint32)
        + raw_bytes[1::3].astype(np.uint32) * 256
        + raw_bytes[2::3].astype(np.uint32) * 65536
    )
    return raw24.reshape((RAW_H, RAW_W)).astype(np.float32) / RAW_MAX


def raw_rggb_project_to_rgb(raw: np.ndarray) -> np.ndarray:
    """Project true RGGB to R, mean(Gr,Gb), B without demosaic interpolation."""

    if raw.shape != (RAW_H, RAW_W):
        raise ValueError(f"Expected raw shape {(RAW_H, RAW_W)}, got {tuple(raw.shape)}")
    return np.stack(
        (
            raw[0::2, 0::2],
            0.5 * (raw[0::2, 1::2] + raw[1::2, 0::2]),
            raw[1::2, 1::2],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def stretch_white_point(rgb: np.ndarray, white_percentile: float) -> np.ndarray:
    if white_percentile >= 100.0:
        return np.clip(rgb, 0.0, 1.0)

    white_level = float(np.percentile(rgb, white_percentile))
    if white_level <= 1e-6:
        return np.clip(rgb, 0.0, 1.0)
    return np.clip(rgb / white_level, 0.0, 1.0)


def gamma_adjust(rgb: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, 1.0), gamma, dtype=np.float32)


def rawvis_degreen_rgb(
    raw: np.ndarray,
    *,
    white_percentile: float,
    gamma: float,
    gains: np.ndarray | tuple[float, float, float] = DEGREEN_GAINS,
) -> np.ndarray:
    """Render native 928x1440 RGB uint8 from full-frame RAW before any crop."""

    gains_arr = np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    rgb = raw_rggb_project_to_rgb(raw)
    rgb = np.clip(rgb * gains_arr, 0.0, 1.0)
    rgb = stretch_white_point(rgb, white_percentile=white_percentile)
    rgb = gamma_adjust(rgb, gamma)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def student_dark_degreen_v1(raw: np.ndarray) -> np.ndarray:
    return rawvis_degreen_rgb(
        raw,
        white_percentile=STUDENT_WHITE_PERCENTILE,
        gamma=STUDENT_GAMMA,
        gains=DEGREEN_GAINS,
    )


def teacher_bright_degreen_v1(raw: np.ndarray) -> np.ndarray:
    return rawvis_degreen_rgb(
        raw,
        white_percentile=TEACHER_WHITE_PERCENTILE,
        gamma=TEACHER_GAMMA,
        gains=DEGREEN_GAINS,
    )


PIPELINE_PARAMS = {
    "student_dark_degreen_v1": {
        "white_percentile": STUDENT_WHITE_PERCENTILE,
        "gamma": STUDENT_GAMMA,
        "channel_gains": DEGREEN_GAINS.tolist(),
    },
    "teacher_bright_degreen_v1": {
        "white_percentile": TEACHER_WHITE_PERCENTILE,
        "gamma": TEACHER_GAMMA,
        "channel_gains": DEGREEN_GAINS.tolist(),
    },
}


def render_pipeline(raw: np.ndarray, pipeline: str) -> np.ndarray:
    if pipeline == "student_dark_degreen_v1":
        return student_dark_degreen_v1(raw)
    if pipeline == "teacher_bright_degreen_v1":
        return teacher_bright_degreen_v1(raw)
    raise ValueError(f"Unsupported ROD RGB pipeline: {pipeline!r}")


__all__ = [
    "RAW_H",
    "RAW_W",
    "RAW_BYTES",
    "RAW_MAX",
    "ROD_NATIVE_HW",
    "DEGREEN_GAINS",
    "STUDENT_WHITE_PERCENTILE",
    "STUDENT_GAMMA",
    "TEACHER_WHITE_PERCENTILE",
    "TEACHER_GAMMA",
    "PIPELINE_PARAMS",
    "unpack_raw24",
    "raw_rggb_project_to_rgb",
    "stretch_white_point",
    "gamma_adjust",
    "rawvis_degreen_rgb",
    "student_dark_degreen_v1",
    "teacher_bright_degreen_v1",
    "render_pipeline",
]
