from dataclasses import dataclass


@dataclass(frozen=True)
class RawStorageSpec:
    """Descriptor for STF RAW NPZ storage and model-facing conversion."""

    name: str
    storage_channel_order: tuple[str, ...]
    model_channel_order: tuple[str, ...]
    channel_reorder: tuple[int, ...]
    decompand: str
    post_decode_norm: str


_RAW_STORAGE_SPECS = {
    "legacy_bggR_decomp16": RawStorageSpec(
        name="legacy_bggR_decomp16",
        storage_channel_order=("B", "G", "G", "R"),
        model_channel_order=("R", "Gr", "Gb", "B"),
        channel_reorder=(3, 1, 2, 0),
        decompand="stf_lut_to_0_1",
        post_decode_norm="passthrough",
    ),
    "raw_future": RawStorageSpec(
        name="raw_future",
        storage_channel_order=("R", "Gr", "Gb", "B"),
        model_channel_order=("R", "Gr", "Gb", "B"),
        channel_reorder=(0, 1, 2, 3),
        decompand="raw_future",
        post_decode_norm="passthrough",
    ),
}


RAW_STORAGE_FORMAT_CHOICES = tuple(_RAW_STORAGE_SPECS.keys())


def get_raw_storage_spec(spec):
    spec_key = str(spec)
    try:
        return _RAW_STORAGE_SPECS[spec_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw_storage_format={spec_key!r}. Supported formats: {', '.join(RAW_STORAGE_FORMAT_CHOICES)}") from exc

