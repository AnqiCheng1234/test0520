from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawStorageSpec:
    name: str
    storage_channel_order: tuple[str, ...]
    model_channel_order: tuple[str, ...]
    channel_reorder: tuple[int, ...]
    decompand: str
    post_decode_norm: str


STF_LUT_TO_0_1_DECOMPAND = "stf_lut_to_0_1"
RAW_RGB16_DIV_65535 = "uint16_div_65535"
RAW_STORAGE_FORMATS = {
    "legacy_bggR_decomp16": RawStorageSpec(
        name="legacy_bggR_decomp16",
        storage_channel_order=("B", "Gr", "Gb", "R"),
        model_channel_order=("R", "Gr", "Gb", "B"),
        channel_reorder=(3, 1, 2, 0),
        decompand=STF_LUT_TO_0_1_DECOMPAND,
        post_decode_norm="passthrough",
    ),
    "raw_rgb16_png_3ch": RawStorageSpec(
        name="raw_rgb16_png_3ch",
        storage_channel_order=("B", "G", "R"),
        model_channel_order=("R", "G", "B"),
        channel_reorder=(2, 1, 0),
        decompand="none",
        post_decode_norm=RAW_RGB16_DIV_65535,
    ),
}


def get_raw_storage_spec(name: str) -> RawStorageSpec:
    key = str(name)
    try:
        return RAW_STORAGE_FORMATS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw_storage_format: {key!r}") from exc
