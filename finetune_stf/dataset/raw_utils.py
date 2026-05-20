from pathlib import Path

import numpy as np


DEFAULT_RAW_NPZ_ROOT = "/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz"
RECTIFIED_BAYER_KEY = "bayer_rect"
COMPANDED_MAX = 3967.0
DECOMPANDED_MAX = 65535.0
SENSOR_LINEAR_MAX = 1.0
STF_DECOMPANDING_NOTES_PATH = (
    "/home/caq/6666_raw/dav2_raw_0515_vits/plans/0520_final_new/stf/"
    "stf_raw_companding_official_notes.md"
)
STF_DECOMP_KNEEPOINTS = np.array(
    [
        [1023, 1023],
        [2559, 4095],
        [3455, 32767],
        [3967, 65535],
    ],
    dtype=np.float32,
)
STF_RAW_DECODE_MODES = (
    "legacy_companded",
    "legacy_online_decomp16",
    "canonical_decomp16",
)
_STF_DECOMPANDING_LUT = None


def load_rectified_bayer_npz(path, key=RECTIFIED_BAYER_KEY):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} does not contain {key}")
        return np.array(data[key], copy=True)


def bayer_to_3ch(bayer_4ch, channel_mode="rgb_avg_g"):
    bayer_4ch = np.asarray(bayer_4ch, dtype=np.float32)
    if bayer_4ch.ndim != 3 or bayer_4ch.shape[-1] != 4:
        raise ValueError(f"Expected Bayer input with shape (H, W, 4), got {bayer_4ch.shape}")

    if channel_mode == "rgb_avg_g":
        r = bayer_4ch[..., 0]
        g = (bayer_4ch[..., 1] + bayer_4ch[..., 2]) * 0.5
        b = bayer_4ch[..., 3]
        return np.stack([r, g, b], axis=-1)

    if channel_mode == "rggb":
        return bayer_4ch[..., [0, 1, 3]]

    raise ValueError(f"Unsupported channel_mode: {channel_mode}")


def build_stf_decompanding_lut(max_code=int(COMPANDED_MAX)):
    xs = np.array([0, *STF_DECOMP_KNEEPOINTS[:, 0]], dtype=np.float32)
    ys = np.array([0, *STF_DECOMP_KNEEPOINTS[:, 1]], dtype=np.float32)
    codes = np.arange(int(max_code) + 1, dtype=np.float32)
    values = np.interp(codes, xs, ys)
    return np.clip(np.round(values), 0, DECOMPANDED_MAX).astype(np.uint16)


def get_stf_decompanding_lut():
    global _STF_DECOMPANDING_LUT
    if _STF_DECOMPANDING_LUT is None:
        _STF_DECOMPANDING_LUT = build_stf_decompanding_lut()
    return _STF_DECOMPANDING_LUT


def decode_stf_raw_4ch(bayer_4ch, decode_mode="legacy_companded"):
    bayer_4ch = np.asarray(bayer_4ch)
    if bayer_4ch.ndim != 3 or bayer_4ch.shape[-1] != 4:
        raise ValueError(f"Expected STF Bayer input with shape (H, W, 4), got {bayer_4ch.shape}")
    if decode_mode not in STF_RAW_DECODE_MODES:
        raise ValueError(f"Unsupported STF RAW decode mode: {decode_mode}")

    if decode_mode == "legacy_companded":
        return bayer_4ch

    if decode_mode == "legacy_online_decomp16":
        # Old rectified STF NPZ files are stored as [B, G, G, R].
        bayer_4ch = bayer_4ch[..., [3, 1, 2, 0]]
    elif decode_mode == "canonical_decomp16":
        # Future canonical roots are expected to already be [R, Gr, Gb, B].
        bayer_4ch = bayer_4ch

    lut = get_stf_decompanding_lut()
    raw_codes = np.clip(bayer_4ch, 0, len(lut) - 1).astype(np.uint16, copy=False)
    return lut[raw_codes].astype(np.float32) / DECOMPANDED_MAX


def normalize_raw(image_3ch, norm_mode="companded"):
    image_3ch = np.asarray(image_3ch, dtype=np.float32)

    if norm_mode == "companded":
        normalized = image_3ch / COMPANDED_MAX
        return np.clip(normalized, 0.0, 1.0)
    if norm_mode == "sensor_linear":
        return np.clip(image_3ch / SENSOR_LINEAR_MAX, 0.0, 1.0)
    if norm_mode == "passthrough":
        return np.clip(image_3ch, 0.0, 1.0)

    raise ValueError(f"Unsupported norm_mode: {norm_mode}")


def normalize_raw_4ch(bayer_4ch, norm_mode="companded"):
    """Normalize 4-channel packed Bayer to [0, 1] without channel reduction."""
    bayer_4ch = np.asarray(bayer_4ch, dtype=np.float32)

    if norm_mode == "companded":
        normalized = bayer_4ch / COMPANDED_MAX
        return np.clip(normalized, 0.0, 1.0)
    if norm_mode == "sensor_linear":
        return np.clip(bayer_4ch / SENSOR_LINEAR_MAX, 0.0, 1.0)
    if norm_mode == "passthrough":
        return np.clip(bayer_4ch, 0.0, 1.0)

    raise ValueError(f"Unsupported norm_mode: {norm_mode}")


def pseudo_rgb_to_bgr(image_rgb):
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    return np.clip(image_rgb[..., ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
