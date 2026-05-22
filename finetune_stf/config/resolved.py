from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


NONE = "none"
NA = "n_a"
NOT_APPLICABLE = "not_applicable"
SOURCE_EXPLICIT = "explicit"
SOURCE_ALIAS = "alias_from_input_type"
SOURCE_DEFAULT_ENCODER = "default_from_encoder"
SOURCE_DEFAULT_RESOLVER = "default_from_resolver"
SOURCE_DEFAULT_FRONT_END = "default_from_front_end"
SOURCE_DEFAULT_DATASET = "default_from_dataset_family"
SOURCE_DEFAULT_EVAL = "default_from_eval_flag"
SOURCE_INFERRED_FRONT_END = "inferred_from_front_end"

INPUT_DOMAIN_CHOICES = ("rgb", "raw4")
FRONT_END_CHOICES = ("dav2_rgb", "raw_to_rgb_head", "raw_ram4", "raw_to_base_rgb_ram3")
DATASET_FAMILY_CHOICES = ("stf_rgb", "stf_raw")
DATASET_INPUT_MODE_CHOICES = ("rgb", "raw_naive", "raw_ram")
MODEL_INPUT_TENSOR_CHOICES = ("image", "raw")
BRIDGE_CHOICES = (NONE, NA, "raw_feature_bridge")
DECODER_FEATURE_ADAPTER_CHOICES = (NONE, NA, "raw_feature_adapter")
LORA_CHOICES = (NONE, NA, "dav2_lora")
BRIDGE_FEATURE_SOURCE_CHANNEL_CHOICES = (NONE, "x3", "x4")
ADAPTER_FEATURE_SOURCE_CHANNEL_CHOICES = (NONE, "x3", "x4")
RAW_STORAGE_FORMAT_CHOICES = (NONE, NA, "legacy_bggR_decomp16", "raw_future")
KITTI_EVAL_PROTOCOL_CHOICES_RESOLVED = (
    NONE,
    "rgb_pretrained_ref",
    "rgb_checkpoint_decoder",
    "live_raw_model",
)
SOURCE_FIELDS = (
    "input_domain",
    "front_end",
    "dataset_family",
    "dataset_input_mode",
    "model_input_tensor",
    "bridge",
    "decoder_feature_adapter",
    "lora",
    "bridge_feature_source_channels",
    "adapter_feature_source_channels",
    "feature_adapter_keys",
    "bridge_feature_keys",
    "bridge_layers",
    "bridge_source",
    "lora_rank",
    "lora_alpha",
    "lora_lr",
    "lora_tap_layers",
    "raw_storage_format",
    "raw_storage_channel_order",
    "raw_model_channel_order",
    "raw_decompand",
    "raw_post_decode_norm",
    "raw_channel_count",
    "ram_core_type",
    "imagenet_norm_enabled",
    "loss_lambda_grad",
    "loss_grad_scales",
    "loss_mask_downsample",
    "kitti_eval_protocol",
    "kitti_model_source",
    "eval_input_domain",
)

FEATURE_KEYS_BY_SOURCE_CHANNELS = {
    "x3": ("x_cat", "ffm_mid", "x3"),
    "x4": ("x_cat", "ffm_mid", "x4"),
}

DEFAULT_BRIDGE_LAYERS_BY_ENCODER = {
    "vits": (2, 5, 8, 11),
    "vitb": (2, 5, 8, 11),
    "vitl": (4, 11, 17, 23),
    "vitg": (9, 19, 29, 39),
}


@dataclass(frozen=True)
class ResolvedConfig:
    input_domain: str
    front_end: str
    dataset_family: str
    dataset_input_mode: str
    model_input_tensor: str
    bridge: str
    decoder_feature_adapter: str
    lora: str
    bridge_feature_source_channels: str
    adapter_feature_source_channels: str
    feature_adapter_keys: tuple[str, ...]
    bridge_feature_keys: tuple[str, ...]
    bridge_layers: tuple[int, ...]
    lora_tap_layers: tuple[int, ...]
    raw_storage_format: str
    kitti_eval_protocol: str
    input_type_alias: str
    input_type_alias_source: str
    raw_storage_channel_order: str = NOT_APPLICABLE
    raw_model_channel_order: str = NOT_APPLICABLE
    raw_decompand: str = NOT_APPLICABLE
    raw_post_decode_norm: str = NOT_APPLICABLE
    raw_channel_count: int | str = NOT_APPLICABLE
    ram_core_type: str = NOT_APPLICABLE
    imagenet_norm_enabled: bool | str = NOT_APPLICABLE
    bridge_source: str = NOT_APPLICABLE
    lora_rank: int | str = NOT_APPLICABLE
    lora_alpha: float | str = NOT_APPLICABLE
    lora_lr: float | str = NOT_APPLICABLE
    loss_lambda_grad: float | str = NOT_APPLICABLE
    loss_grad_scales: int | str = NOT_APPLICABLE
    loss_mask_downsample: str = NOT_APPLICABLE
    kitti_model_source: str = NOT_APPLICABLE
    eval_input_domain: str = NOT_APPLICABLE
    optimizer_param_groups: tuple[dict[str, Any], ...] = ()
    sources: dict[str, str] = field(default_factory=dict)
    not_applicable: tuple[str, ...] = ()

    def _field_source(self, field_name: str) -> str:
        return self.sources.get(field_name, SOURCE_DEFAULT_RESOLVER)

    def _value_or_not_applicable(self, field_name: str, value: Any) -> Any:
        if self._field_source(field_name) == NOT_APPLICABLE:
            return NOT_APPLICABLE
        return value

    def _tuple_value(self, field_name: str, value: tuple[Any, ...]) -> Any:
        if self._field_source(field_name) == NOT_APPLICABLE:
            return NOT_APPLICABLE
        return list(value) if value else NONE

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "input_domain": self.input_domain,
            "front_end": self.front_end,
            "dataset_family": self.dataset_family,
            "dataset_input_mode": self.dataset_input_mode,
            "model_input_tensor": self.model_input_tensor,
            "bridge": self.bridge,
            "decoder_feature_adapter": self.decoder_feature_adapter,
            "lora": self.lora,
            "bridge_feature_source_channels": self._value_or_not_applicable(
                "bridge_feature_source_channels", self.bridge_feature_source_channels
            ),
            "adapter_feature_source_channels": self._value_or_not_applicable(
                "adapter_feature_source_channels", self.adapter_feature_source_channels
            ),
            "feature_adapter_keys": self._tuple_value("feature_adapter_keys", self.feature_adapter_keys),
            "bridge_feature_keys": self._tuple_value("bridge_feature_keys", self.bridge_feature_keys),
            "bridge_layers": self._tuple_value("bridge_layers", self.bridge_layers),
            "lora_tap_layers": self._tuple_value("lora_tap_layers", self.lora_tap_layers),
            "raw_storage_format": self._value_or_not_applicable("raw_storage_format", self.raw_storage_format),
            "raw_storage_channel_order": self.raw_storage_channel_order,
            "raw_model_channel_order": self.raw_model_channel_order,
            "raw_decompand": self.raw_decompand,
            "raw_post_decode_norm": self.raw_post_decode_norm,
            "raw_channel_count": self.raw_channel_count,
            "ram_core_type": self.ram_core_type,
            "imagenet_norm_enabled": self.imagenet_norm_enabled,
            "bridge_source": self.bridge_source,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_lr": self.lora_lr,
            "loss_lambda_grad": self.loss_lambda_grad,
            "loss_grad_scales": self.loss_grad_scales,
            "loss_mask_downsample": self.loss_mask_downsample,
            "kitti_eval_protocol": self.kitti_eval_protocol,
            "kitti_model_source": self.kitti_model_source,
            "eval_input_domain": self.eval_input_domain,
            "optimizer_param_groups": list(self.optimizer_param_groups),
            "not_applicable": list(self.not_applicable),
            "input_type_alias": self.input_type_alias,
            "input_type_alias_source": self.input_type_alias_source,
        }
        for field_name in SOURCE_FIELDS:
            source_key = f"{field_name}_source"
            if source_key in payload:
                source_key = f"{field_name}_config_source"
            payload[source_key] = self._field_source(field_name)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResolvedConfig":
        data = dict(payload)
        for key in (
            "bridge_feature_source_channels",
            "adapter_feature_source_channels",
            "raw_storage_format",
            "raw_storage_channel_order",
            "raw_model_channel_order",
        ):
            if data.get(key) == NOT_APPLICABLE:
                data[key] = NONE
        for key in ("feature_adapter_keys", "bridge_feature_keys"):
            data[key] = _normalise_str_tuple(data.get(key))
        for key in ("bridge_layers", "lora_tap_layers"):
            data[key] = _normalise_int_tuple(data.get(key))
        source_payload = dict(data.get("sources") or {})
        for key in tuple(data):
            if key.endswith("_source") and key != "input_type_alias_source":
                source_payload[key[: -len("_source")]] = str(data.pop(key))
            elif key.endswith("_config_source"):
                source_payload[key[: -len("_config_source")]] = str(data.pop(key))
        data["sources"] = source_payload
        data["not_applicable"] = tuple(data.get("not_applicable") or ())
        data["optimizer_param_groups"] = tuple(data.get("optimizer_param_groups") or ())
        return cls(**data)

    def with_optimizer_param_groups(self, groups: list[dict[str, Any]]) -> "ResolvedConfig":
        clean_groups = []
        for group in groups:
            clean_groups.append(
                {
                    "group_name": str(group.get("group_name", "unnamed")),
                    "lr": float(group.get("lr", 0.0)),
                    "trainable_param_count": int(group.get("trainable_param_count", 0)),
                    "trainable_tensor_count": int(group.get("trainable_tensor_count", 0)),
                }
            )
        return replace(self, optimizer_param_groups=tuple(clean_groups))


def _base_config(
    *,
    input_domain: str,
    front_end: str,
    dataset_family: str,
    dataset_input_mode: str,
    model_input_tensor: str,
    bridge: str = NONE,
    decoder_feature_adapter: str = NONE,
    lora: str = NONE,
    bridge_feature_source_channels: str = NONE,
    adapter_feature_source_channels: str = NONE,
) -> dict[str, Any]:
    return {
        "input_domain": input_domain,
        "front_end": front_end,
        "dataset_family": dataset_family,
        "dataset_input_mode": dataset_input_mode,
        "model_input_tensor": model_input_tensor,
        "bridge": bridge,
        "decoder_feature_adapter": decoder_feature_adapter,
        "lora": lora,
        "bridge_feature_source_channels": bridge_feature_source_channels,
        "adapter_feature_source_channels": adapter_feature_source_channels,
    }


_RGB = _base_config(
    input_domain="rgb",
    front_end="dav2_rgb",
    dataset_family="stf_rgb",
    dataset_input_mode="rgb",
    model_input_tensor="image",
)
_RAW_NAIVE = _base_config(
    input_domain="rgb",
    front_end="dav2_rgb",
    dataset_family="stf_raw",
    dataset_input_mode="raw_naive",
    model_input_tensor="image",
)
_RAW_PACKED = _base_config(
    input_domain="raw4",
    front_end="raw_to_rgb_head",
    dataset_family="stf_raw",
    dataset_input_mode="raw_ram",
    model_input_tensor="raw",
)
_RAW_RAM4 = _base_config(
    input_domain="raw4",
    front_end="raw_ram4",
    dataset_family="stf_raw",
    dataset_input_mode="raw_ram",
    model_input_tensor="raw",
)
_RAW_RAM3 = _base_config(
    input_domain="raw4",
    front_end="raw_to_base_rgb_ram3",
    dataset_family="stf_raw",
    dataset_input_mode="raw_ram",
    model_input_tensor="raw",
)


def _with(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    merged = dict(base)
    merged.update(updates)
    return merged


INPUT_TYPE_ALIASES: dict[str, dict[str, Any]] = {
    "rgb": _RGB,
    "rgb_lora": _with(_RGB, lora="dav2_lora"),
    "raw": _RAW_NAIVE,
    "raw_packed": _RAW_PACKED,
    "raw_ram": _RAW_RAM4,
    "raw_ram_lora": _with(_RAW_RAM4, lora="dav2_lora"),
    "raw_ram_residual": _RAW_RAM4,
    "raw_ram_rgb": _RAW_RAM3,
    "raw_ram_rgb_lora": _with(_RAW_RAM3, lora="dav2_lora"),
    "raw_ram_bridge": _with(
        _RAW_RAM4,
        bridge="raw_feature_bridge",
        bridge_feature_source_channels="x4",
    ),
    "raw_ram_bridge_lora": _with(
        _RAW_RAM4,
        bridge="raw_feature_bridge",
        lora="dav2_lora",
        bridge_feature_source_channels="x4",
    ),
    "raw_ram_rgb_bridge": _with(
        _RAW_RAM3,
        bridge="raw_feature_bridge",
        bridge_feature_source_channels="x3",
    ),
    "raw_ram_rgb_bridge_lora": _with(
        _RAW_RAM3,
        bridge="raw_feature_bridge",
        lora="dav2_lora",
        bridge_feature_source_channels="x3",
    ),
    "raw_ram_feature_adapter": _with(
        _RAW_RAM4,
        decoder_feature_adapter="raw_feature_adapter",
        adapter_feature_source_channels="x4",
    ),
    "raw_ram_feature_adapter_lora": _with(
        _RAW_RAM4,
        decoder_feature_adapter="raw_feature_adapter",
        lora="dav2_lora",
        adapter_feature_source_channels="x4",
    ),
    "raw_ram_rgb_feature_adapter": _with(
        _RAW_RAM3,
        decoder_feature_adapter="raw_feature_adapter",
        adapter_feature_source_channels="x3",
    ),
    "raw_ram_rgb_feature_adapter_lora": _with(
        _RAW_RAM3,
        decoder_feature_adapter="raw_feature_adapter",
        lora="dav2_lora",
        adapter_feature_source_channels="x3",
    ),
    "raw_ram_bridge_feature_adapter": _with(
        _RAW_RAM4,
        bridge="raw_feature_bridge",
        decoder_feature_adapter="raw_feature_adapter",
        bridge_feature_source_channels="x4",
        adapter_feature_source_channels="x4",
    ),
    "raw_ram_bridge_feature_adapter_lora": _with(
        _RAW_RAM4,
        bridge="raw_feature_bridge",
        decoder_feature_adapter="raw_feature_adapter",
        lora="dav2_lora",
        bridge_feature_source_channels="x4",
        adapter_feature_source_channels="x4",
    ),
    "raw_ram_rgb_bridge_feature_adapter": _with(
        _RAW_RAM3,
        bridge="raw_feature_bridge",
        decoder_feature_adapter="raw_feature_adapter",
        bridge_feature_source_channels="x3",
        adapter_feature_source_channels="x3",
    ),
    "raw_ram_rgb_bridge_feature_adapter_lora": _with(
        _RAW_RAM3,
        bridge="raw_feature_bridge",
        decoder_feature_adapter="raw_feature_adapter",
        lora="dav2_lora",
        bridge_feature_source_channels="x3",
        adapter_feature_source_channels="x3",
    ),
}


def resolve_legacy_input_type(input_type: str) -> dict[str, Any]:
    name = str(input_type)
    if name not in INPUT_TYPE_ALIASES:
        raise ValueError(f"Unsupported legacy input_type={name!r}")
    resolved = dict(INPUT_TYPE_ALIASES[name])
    resolved["input_type_alias"] = name
    resolved["raw_storage_format"] = (
        "legacy_bggR_decomp16" if resolved["dataset_family"] == "stf_raw" else NONE
    )
    bridge_source_channels = resolved.get("bridge_feature_source_channels", NONE)
    adapter_source_channels = resolved.get("adapter_feature_source_channels", NONE)
    resolved["bridge_feature_keys"] = (
        list(_feature_keys_for_source(bridge_source_channels))
        if bridge_source_channels != NONE
        else NOT_APPLICABLE
    )
    resolved["feature_adapter_keys"] = (
        list(_feature_keys_for_source(adapter_source_channels))
        if adapter_source_channels != NONE
        else NOT_APPLICABLE
    )
    return resolved


SCALAR_CONFIG_FIELDS = (
    "input_domain",
    "front_end",
    "dataset_family",
    "dataset_input_mode",
    "model_input_tensor",
    "bridge",
    "decoder_feature_adapter",
    "lora",
    "bridge_feature_source_channels",
    "adapter_feature_source_channels",
    "raw_storage_format",
)

CHOICES_BY_FIELD = {
    "input_domain": INPUT_DOMAIN_CHOICES,
    "front_end": FRONT_END_CHOICES,
    "dataset_family": DATASET_FAMILY_CHOICES,
    "dataset_input_mode": DATASET_INPUT_MODE_CHOICES,
    "model_input_tensor": MODEL_INPUT_TENSOR_CHOICES,
    "bridge": BRIDGE_CHOICES,
    "decoder_feature_adapter": DECODER_FEATURE_ADAPTER_CHOICES,
    "lora": LORA_CHOICES,
    "bridge_feature_source_channels": BRIDGE_FEATURE_SOURCE_CHANNEL_CHOICES,
    "adapter_feature_source_channels": ADAPTER_FEATURE_SOURCE_CHANNEL_CHOICES,
    "raw_storage_format": RAW_STORAGE_FORMAT_CHOICES,
}


def _normalise_str_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, NONE, NOT_APPLICABLE, "", []):
        return ()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in value if str(item).strip()]
    return tuple(dict.fromkeys(values))


def _normalise_int_tuple(value: Any) -> tuple[int, ...]:
    if value in (None, NONE, NOT_APPLICABLE, "", []):
        return ()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = list(value)
    return tuple(dict.fromkeys(int(item) for item in values))


def _feature_keys_for_source(source_channels: str) -> tuple[str, ...]:
    return FEATURE_KEYS_BY_SOURCE_CHANNELS.get(source_channels, ())


def _default_layers_for_encoder(encoder: str) -> tuple[int, ...]:
    try:
        return DEFAULT_BRIDGE_LAYERS_BY_ENCODER[str(encoder)]
    except KeyError as exc:
        raise ValueError(f"Unsupported encoder for resolved config: {encoder}") from exc


def _explicit_arg_names(args: Any) -> set[str]:
    return set(getattr(args, "_explicit_cli_args", ()) or ())


def _arg_was_explicit(args: Any, dest: str) -> bool:
    return dest in _explicit_arg_names(args)


def _set_source(sources: dict[str, str], field_name: str, source: str) -> None:
    sources[field_name] = source


def _source_for_default(explicit_alias: bool) -> str:
    return SOURCE_ALIAS if explicit_alias else SOURCE_DEFAULT_RESOLVER


def _infer_from_front_end(config: dict[str, Any], explicit_fields: set[str], sources: dict[str, str]) -> None:
    front_end = config["front_end"]
    if front_end == "dav2_rgb":
        inferred = {
            "input_domain": "rgb",
            "model_input_tensor": "image",
        }
        if config.get("dataset_family") == "stf_rgb":
            inferred["dataset_input_mode"] = "rgb"
        elif config.get("dataset_family") == "stf_raw":
            inferred["dataset_input_mode"] = "raw_naive"
    elif front_end == "raw_to_rgb_head":
        inferred = {
            "input_domain": "raw4",
            "dataset_family": "stf_raw",
            "dataset_input_mode": "raw_ram",
            "model_input_tensor": "raw",
        }
    elif front_end in {"raw_ram4", "raw_to_base_rgb_ram3"}:
        inferred = {
            "input_domain": "raw4",
            "dataset_family": "stf_raw",
            "dataset_input_mode": "raw_ram",
            "model_input_tensor": "raw",
        }
    else:
        return

    for field, value in inferred.items():
        if field not in explicit_fields:
            config[field] = value
            sources[field] = SOURCE_INFERRED_FRONT_END


def _infer_sources(config: dict[str, Any], explicit_fields: set[str], sources: dict[str, str]) -> None:
    del config, explicit_fields, sources


def _raw_storage_format_from_args(args: Any, dataset_family: str) -> tuple[str, str]:
    explicit = getattr(args, "raw_storage_format", None)
    if explicit is not None:
        value = str(explicit)
        if value == "raw_future":
            raise ValueError(
                "raw_storage_format=raw_future is reserved and not implemented in this project. "
                "Use legacy_bggR_decomp16 or provide a new explicit spec first."
            )
        return (NONE if value == NA else value), SOURCE_EXPLICIT
    if dataset_family != "stf_raw":
        return NONE, NOT_APPLICABLE
    return "legacy_bggR_decomp16", SOURCE_DEFAULT_DATASET


def _kitti_protocol_from_args(args: Any) -> tuple[str, str]:
    if not bool(getattr(args, "eval_kitti", False)):
        return NONE, NOT_APPLICABLE
    return str(getattr(args, "kitti_eval_protocol", "rgb_pretrained_ref")), (
        SOURCE_EXPLICIT if _arg_was_explicit(args, "kitti_eval_protocol") else SOURCE_DEFAULT_EVAL
    )


def _raw_storage_details(config: dict[str, Any], args: Any, sources: dict[str, str]) -> dict[str, Any]:
    if config["raw_storage_format"] == NONE:
        for field_name in (
            "raw_storage_channel_order",
            "raw_model_channel_order",
            "raw_decompand",
            "raw_post_decode_norm",
            "raw_channel_count",
        ):
            sources[field_name] = NOT_APPLICABLE
        return {
            "raw_storage_channel_order": NOT_APPLICABLE,
            "raw_model_channel_order": NOT_APPLICABLE,
            "raw_decompand": NOT_APPLICABLE,
            "raw_post_decode_norm": NOT_APPLICABLE,
            "raw_channel_count": NOT_APPLICABLE,
        }

    storage_spec = get_raw_storage_spec(config["raw_storage_format"])
    for field_name in (
        "raw_storage_channel_order",
        "raw_model_channel_order",
        "raw_decompand",
        "raw_post_decode_norm",
        "raw_channel_count",
    ):
        sources[field_name] = SOURCE_DEFAULT_DATASET
    return {
        "raw_storage_channel_order": storage_spec.storage_channel_order,
        "raw_model_channel_order": storage_spec.model_channel_order,
        "raw_decompand": storage_spec.decompand,
        "raw_post_decode_norm": storage_spec.post_decode_norm,
        "raw_channel_count": len(storage_spec.storage_channel_order),
    }


def _ram_core_type(front_end: str, sources: dict[str, str]) -> str:
    if front_end == "raw_ram4":
        sources["ram_core_type"] = SOURCE_DEFAULT_FRONT_END
        return "RawRamCore"
    if front_end == "raw_to_base_rgb_ram3":
        sources["ram_core_type"] = SOURCE_DEFAULT_FRONT_END
        return "RamCore3"
    sources["ram_core_type"] = NOT_APPLICABLE
    return NOT_APPLICABLE


def _imagenet_norm_enabled(config: dict[str, Any], args: Any, sources: dict[str, str]) -> bool | str:
    if config["front_end"] == "raw_to_base_rgb_ram3":
        sources["imagenet_norm_enabled"] = SOURCE_DEFAULT_FRONT_END
        return False
    if config["front_end"] in {"raw_ram4", "raw_to_rgb_head"}:
        sources["imagenet_norm_enabled"] = SOURCE_DEFAULT_FRONT_END
        return True
    if config["dataset_family"] == "stf_rgb":
        sources["imagenet_norm_enabled"] = SOURCE_DEFAULT_DATASET
        return True
    sources["imagenet_norm_enabled"] = SOURCE_DEFAULT_DATASET
    return bool(getattr(args, "use_imagenet_norm", True))


def _kitti_details(kitti_eval_protocol: str, sources: dict[str, str]) -> tuple[str, str]:
    if kitti_eval_protocol == NONE:
        sources["kitti_model_source"] = NOT_APPLICABLE
        sources["eval_input_domain"] = NOT_APPLICABLE
        return NOT_APPLICABLE, NOT_APPLICABLE
    sources["kitti_model_source"] = SOURCE_DEFAULT_EVAL
    sources["eval_input_domain"] = SOURCE_DEFAULT_EVAL
    if kitti_eval_protocol == "rgb_pretrained_ref":
        return "pretrained_from_rgb_reference", "rgb"
    if kitti_eval_protocol == "rgb_checkpoint_decoder":
        return "live_checkpoint_rgb_decoder", "rgb"
    return "live_raw_model", "raw4"


def _loss_details(args: Any, sources: dict[str, str]) -> tuple[float | str, int | str, str]:
    if str(getattr(args, "loss_type", "")) != "ssi_grad":
        sources["loss_lambda_grad"] = NOT_APPLICABLE
        sources["loss_grad_scales"] = NOT_APPLICABLE
        sources["loss_mask_downsample"] = NOT_APPLICABLE
        return NOT_APPLICABLE, NOT_APPLICABLE, NOT_APPLICABLE
    sources["loss_lambda_grad"] = SOURCE_EXPLICIT if _arg_was_explicit(args, "loss_lambda_grad") else SOURCE_DEFAULT_RESOLVER
    sources["loss_grad_scales"] = SOURCE_EXPLICIT if _arg_was_explicit(args, "loss_grad_scales") else SOURCE_DEFAULT_RESOLVER
    sources["loss_mask_downsample"] = (
        SOURCE_EXPLICIT if _arg_was_explicit(args, "loss_mask_downsample") else SOURCE_DEFAULT_RESOLVER
    )
    return (
        float(getattr(args, "loss_lambda_grad")),
        int(getattr(args, "loss_grad_scales")),
        str(getattr(args, "loss_mask_downsample", "strict")),
    )


def _legacy_alias_from_config(config: dict[str, Any]) -> str:
    front_end = config["front_end"]
    dataset_family = config["dataset_family"]
    dataset_input_mode = config["dataset_input_mode"]
    model_input_tensor = config["model_input_tensor"]
    bridge = config["bridge"] != NONE
    adapter = config["decoder_feature_adapter"] != NONE
    lora = config["lora"] != NONE
    bridge_src = config["bridge_feature_source_channels"]
    adapter_src = config["adapter_feature_source_channels"]

    if dataset_family == "stf_rgb" and front_end == "dav2_rgb" and not bridge and not adapter:
        return "rgb_lora" if lora else "rgb"
    if (
        dataset_family == "stf_raw"
        and front_end == "dav2_rgb"
        and dataset_input_mode == "raw_naive"
        and model_input_tensor == "image"
        and not bridge
        and not adapter
        and not lora
    ):
        return "raw"
    if front_end == "raw_to_rgb_head" and not bridge and not adapter and not lora:
        return "raw_packed"
    if front_end == "raw_ram4":
        if not bridge and not adapter and not lora:
            return "raw_ram"
        if not bridge and not adapter and lora:
            return "raw_ram_lora"
        if bridge and not adapter and bridge_src == "x4":
            return "raw_ram_bridge_lora" if lora else "raw_ram_bridge"
        if adapter and not bridge and adapter_src == "x4":
            return "raw_ram_feature_adapter_lora" if lora else "raw_ram_feature_adapter"
        if bridge and adapter and bridge_src == "x4" and adapter_src == "x4":
            return "raw_ram_bridge_feature_adapter_lora" if lora else "raw_ram_bridge_feature_adapter"
    if front_end == "raw_to_base_rgb_ram3":
        if not bridge and not adapter:
            return "raw_ram_rgb_lora" if lora else "raw_ram_rgb"
        if bridge and not adapter and bridge_src == "x3":
            return "raw_ram_rgb_bridge_lora" if lora else "raw_ram_rgb_bridge"
        if adapter and not bridge and adapter_src == "x3":
            return "raw_ram_rgb_feature_adapter_lora" if lora else "raw_ram_rgb_feature_adapter"
        if bridge and adapter and bridge_src == "x3" and adapter_src == "x3":
            return "raw_ram_rgb_bridge_feature_adapter_lora" if lora else "raw_ram_rgb_bridge_feature_adapter"
    raise ValueError(
        "The resolved orthogonal config is expressible, but this train.py revision has no "
        "legacy model factory alias for it yet: "
        f"front_end={front_end}, bridge={config['bridge']}, "
        f"decoder_feature_adapter={config['decoder_feature_adapter']}, lora={config['lora']}, "
        f"bridge_source={bridge_src}, adapter_source={adapter_src}"
    )


def _validate_choices(config: dict[str, Any]) -> None:
    for field, choices in CHOICES_BY_FIELD.items():
        value = config[field]
        if value not in choices:
            raise ValueError(f"Unsupported {field}={value!r}; valid choices: {', '.join(choices)}")


def _require_explicit_or_alias(
    resolved: ResolvedConfig,
    field_names: tuple[str, ...],
    *,
    context: str,
    allow_encoder_default: bool = False,
) -> None:
    allowed = {SOURCE_EXPLICIT, SOURCE_ALIAS}
    if allow_encoder_default:
        allowed.add(SOURCE_DEFAULT_ENCODER)
    missing = [name for name in field_names if resolved.sources.get(name) not in allowed]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"{context} requires explicit {flags}")


def _reject_explicit_when_disabled(args: Any, field_names: tuple[str, ...], *, context: str) -> None:
    explicit = _explicit_arg_names(args)
    present = sorted(name for name in field_names if name in explicit)
    if present:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in present)
        raise ValueError(f"{context} is disabled; do not pass {flags}")


def validate_applicability(resolved: ResolvedConfig, args: Any | None = None) -> None:
    if args is None:
        return
    lora_block_mode = str(getattr(args, "lora_block_mode", "tap"))

    if resolved.bridge == NONE:
        _reject_explicit_when_disabled(
            args,
            (
                "bridge_feature_source_channels",
                "bridge_feature_keys",
                "bridge_init_from",
            ),
            context="bridge",
        )
        if resolved.decoder_feature_adapter == NONE:
            _reject_explicit_when_disabled(args, ("bridge_source",), context="bridge")
        _reject_explicit_when_disabled(args, ("bridge_layers",), context="bridge")
    else:
        _require_explicit_or_alias(
            resolved,
            (
                "bridge_feature_source_channels",
                "bridge_feature_keys",
            ),
            context="bridge=raw_feature_bridge",
        )
        _require_explicit_or_alias(
            resolved,
            ("bridge_layers",),
            context="bridge=raw_feature_bridge",
            allow_encoder_default=True,
        )
        if not (_arg_was_explicit(args, "bridge_source") or resolved.sources.get("bridge_source") == SOURCE_ALIAS):
            raise ValueError("bridge=raw_feature_bridge requires explicit --bridge-source")

    if resolved.decoder_feature_adapter == NONE:
        _reject_explicit_when_disabled(
            args,
            (
                "adapter_feature_source_channels",
                "feature_adapter_keys",
            ),
            context="decoder_feature_adapter",
        )
    else:
        _require_explicit_or_alias(
            resolved,
            (
                "adapter_feature_source_channels",
                "feature_adapter_keys",
            ),
            context="decoder_feature_adapter=raw_feature_adapter",
        )
        if not (_arg_was_explicit(args, "bridge_source") or resolved.sources.get("bridge_source") == SOURCE_ALIAS):
            raise ValueError("decoder_feature_adapter=raw_feature_adapter requires explicit --bridge-source")

    if resolved.lora == NONE:
        _reject_explicit_when_disabled(
            args,
            (
                "lora_rank",
                "lora_alpha",
                "lora_lr",
                "lora_block_mode",
                "lora_tap_layers",
            ),
            context="lora",
        )
    else:
        _require_explicit_or_alias(
            resolved,
            (
                "lora_rank",
                "lora_alpha",
                "lora_lr",
            ),
            context="lora=dav2_lora",
        )
        if lora_block_mode == "tap":
            _require_explicit_or_alias(
                resolved,
                ("lora_tap_layers",),
                context="lora=dav2_lora with lora_block_mode=tap",
            )
        elif _arg_was_explicit(args, "lora_tap_layers"):
            raise ValueError("lora_tap_layers is only applicable when --lora-block-mode tap")

    if resolved.bridge == NONE and resolved.decoder_feature_adapter == NONE:
        _reject_explicit_when_disabled(args, ("bridge_lr",), context="bridge/decoder feature adapter")
    if resolved.lora == NONE:
        _reject_explicit_when_disabled(args, ("lora_lr",), context="lora")

    if not bool(getattr(args, "eval_kitti", False)):
        _reject_explicit_when_disabled(args, ("kitti_eval_protocol",), context="eval_kitti")
    elif not _arg_was_explicit(args, "kitti_eval_protocol"):
        raise ValueError("--eval-kitti requires explicit --kitti-eval-protocol")

    if str(getattr(args, "loss_type", "")) == "ssi_grad":
        missing = [
            flag
            for flag in ("loss_lambda_grad", "loss_grad_scales")
            if not _arg_was_explicit(args, flag)
        ]
        if missing:
            flags = ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            raise ValueError(f"--loss-type ssi_grad requires explicit {flags}")
    else:
        _reject_explicit_when_disabled(
            args,
            (
                "loss_lambda_grad",
                "loss_grad_scales",
                "loss_mask_downsample",
            ),
            context="grad loss",
        )


def validate_resolved_config(resolved: ResolvedConfig, args: Any | None = None) -> None:
    cfg = resolved
    if cfg.dataset_family == "stf_rgb":
        if cfg.dataset_input_mode != "rgb":
            raise ValueError("dataset_family=stf_rgb requires dataset_input_mode=rgb")
        if cfg.input_domain != "rgb" or cfg.model_input_tensor != "image" or cfg.front_end != "dav2_rgb":
            raise ValueError("dataset_family=stf_rgb currently requires dav2_rgb image input")
    if cfg.dataset_family == "stf_raw" and cfg.dataset_input_mode == "rgb":
        raise ValueError("dataset_family=stf_raw requires dataset_input_mode raw_naive or raw_ram")
    if cfg.front_end == "dav2_rgb":
        if cfg.input_domain != "rgb" or cfg.model_input_tensor != "image":
            raise ValueError("front_end=dav2_rgb requires input_domain=rgb and model_input_tensor=image")
    if cfg.front_end in {"raw_to_rgb_head", "raw_ram4", "raw_to_base_rgb_ram3"}:
        if cfg.input_domain != "raw4" or cfg.model_input_tensor != "raw":
            raise ValueError(f"front_end={cfg.front_end} requires raw4/raw tensor input")
        if cfg.dataset_family != "stf_raw" or cfg.dataset_input_mode != "raw_ram":
            raise ValueError(f"front_end={cfg.front_end} requires stf_raw/raw_ram dataset input")
    if cfg.bridge == NONE:
        if cfg.bridge_feature_source_channels != NONE or cfg.bridge_feature_keys or cfg.bridge_layers:
            raise ValueError("bridge=none requires bridge feature source, keys, and layers to be not applicable")
    else:
        if cfg.bridge_feature_source_channels not in {"x3", "x4"}:
            raise ValueError("bridge=raw_feature_bridge requires bridge_feature_source_channels=x3 or x4")
        if not cfg.bridge_feature_keys or not cfg.bridge_layers:
            raise ValueError("bridge=raw_feature_bridge requires bridge_feature_keys and bridge_layers")
        allowed_keys = set(_feature_keys_for_source(cfg.bridge_feature_source_channels))
        invalid_keys = [key for key in cfg.bridge_feature_keys if key not in allowed_keys]
        if invalid_keys:
            raise ValueError(
                f"bridge_feature_source_channels={cfg.bridge_feature_source_channels} "
                f"does not support bridge_feature_keys={invalid_keys}"
            )
    if cfg.decoder_feature_adapter == NONE:
        if cfg.adapter_feature_source_channels != NONE or cfg.feature_adapter_keys:
            raise ValueError("decoder_feature_adapter=none requires adapter source and keys to be not applicable")
    else:
        if cfg.adapter_feature_source_channels not in {"x3", "x4"}:
            raise ValueError("decoder_feature_adapter=raw_feature_adapter requires adapter_feature_source_channels=x3 or x4")
        if not cfg.feature_adapter_keys:
            raise ValueError("decoder_feature_adapter=raw_feature_adapter requires feature_adapter_keys")
        allowed_keys = set(_feature_keys_for_source(cfg.adapter_feature_source_channels))
        invalid_keys = [key for key in cfg.feature_adapter_keys if key not in allowed_keys]
        if invalid_keys:
            raise ValueError(
                f"adapter_feature_source_channels={cfg.adapter_feature_source_channels} "
                f"does not support feature_adapter_keys={invalid_keys}"
            )
    if cfg.lora == NONE:
        if cfg.lora_tap_layers:
            raise ValueError("lora=none requires lora_tap_layers to be not applicable")
    elif cfg.lora == "dav2_lora" and cfg.sources.get("lora_tap_layers") != NOT_APPLICABLE:
        if not cfg.lora_tap_layers:
            raise ValueError("lora=dav2_lora with lora_block_mode=tap requires lora_tap_layers")
    if cfg.bridge_feature_source_channels == "x3" and cfg.front_end != "raw_to_base_rgb_ram3":
        raise ValueError("bridge_feature_source_channels=x3 requires front_end=raw_to_base_rgb_ram3")
    if cfg.adapter_feature_source_channels == "x3" and cfg.front_end != "raw_to_base_rgb_ram3":
        raise ValueError("adapter_feature_source_channels=x3 requires front_end=raw_to_base_rgb_ram3")
    if cfg.bridge_feature_source_channels == "x4" and cfg.front_end == "raw_to_base_rgb_ram3":
        raise ValueError("front_end=raw_to_base_rgb_ram3 uses x3 bridge features, not x4")
    if cfg.adapter_feature_source_channels == "x4" and cfg.front_end == "raw_to_base_rgb_ram3":
        raise ValueError("front_end=raw_to_base_rgb_ram3 uses x3 adapter features, not x4")
    if cfg.raw_storage_format != NONE and cfg.dataset_family != "stf_raw":
        raise ValueError("raw_storage_format is only applicable when dataset_family=stf_raw")
    if cfg.kitti_eval_protocol not in KITTI_EVAL_PROTOCOL_CHOICES_RESOLVED:
        raise ValueError(f"Unsupported kitti_eval_protocol in resolved config: {cfg.kitti_eval_protocol}")
    if cfg.kitti_eval_protocol == "live_raw_model":
        raise ValueError("kitti_eval_protocol=live_raw_model is reserved in the schema but not implemented in train.py yet")
    validate_applicability(resolved, args)


def resolve_config_from_args(args: Any) -> ResolvedConfig:
    alias = getattr(args, "input_type", None)
    explicit_alias = alias is not None
    explicit_cli_args = _explicit_arg_names(args)
    explicit_fields = {
        field
        for field in SCALAR_CONFIG_FIELDS
        if getattr(args, field, None) is not None
    }
    has_orthogonal_fields = bool(explicit_fields) or any(
        getattr(args, field, None) is not None
        for field in ("feature_adapter_keys", "bridge_feature_keys", "bridge_layers", "lora_tap_layers")
    )

    alias_source = "explicit" if explicit_alias else ("orthogonal" if has_orthogonal_fields else "implicit")
    if alias is None:
        alias = "rgb"
    if alias not in INPUT_TYPE_ALIASES:
        raise ValueError(f"Unsupported input_type alias: {alias}")

    config = dict(INPUT_TYPE_ALIASES[alias])
    sources = {
        field: _source_for_default(explicit_alias)
        for field in (
            "input_domain",
            "front_end",
            "dataset_family",
            "dataset_input_mode",
            "model_input_tensor",
            "bridge",
            "decoder_feature_adapter",
            "lora",
            "bridge_feature_source_channels",
            "adapter_feature_source_channels",
        )
    }

    for field in SCALAR_CONFIG_FIELDS:
        value = getattr(args, field, None)
        if value is None:
            continue
        value = str(value)
        if value == NA:
            value = NONE
        if explicit_alias and field != "raw_storage_format" and config.get(field) != value:
            raise ValueError(
                f"--input-type {alias} expands to {field}={config.get(field)!r}, "
                f"but --{field.replace('_', '-')}={value!r} was also provided"
            )
        config[field] = value
        sources[field] = SOURCE_EXPLICIT

    _infer_from_front_end(config, explicit_fields, sources)
    _infer_sources(config, explicit_fields, sources)
    config["raw_storage_format"], sources["raw_storage_format"] = _raw_storage_format_from_args(
        args, config["dataset_family"]
    )
    _validate_choices(config)

    bridge_feature_keys_arg = _normalise_str_tuple(getattr(args, "bridge_feature_keys", None))
    feature_adapter_keys_arg = _normalise_str_tuple(getattr(args, "feature_adapter_keys", None))
    bridge_layers_arg = _normalise_int_tuple(getattr(args, "bridge_layers", None))
    lora_tap_layers_arg = _normalise_int_tuple(getattr(args, "lora_tap_layers", None))
    encoder = getattr(args, "encoder", "vitl")

    bridge_keys = ()
    adapter_keys = ()
    bridge_layers = ()
    lora_tap_layers = ()
    bridge_feature_keys_source = NOT_APPLICABLE
    adapter_feature_keys_source = NOT_APPLICABLE
    bridge_layers_source = NOT_APPLICABLE
    lora_tap_layers_source = NOT_APPLICABLE

    if config["bridge"] != NONE:
        if bridge_feature_keys_arg:
            bridge_keys = bridge_feature_keys_arg
            bridge_feature_keys_source = SOURCE_EXPLICIT
        elif explicit_alias:
            bridge_keys = _feature_keys_for_source(config["bridge_feature_source_channels"])
            bridge_feature_keys_source = SOURCE_ALIAS

        if bridge_layers_arg:
            bridge_layers = bridge_layers_arg
            bridge_layers_source = SOURCE_EXPLICIT
        else:
            bridge_layers = _default_layers_for_encoder(encoder)
            bridge_layers_source = SOURCE_DEFAULT_ENCODER

    if config["decoder_feature_adapter"] != NONE:
        if feature_adapter_keys_arg:
            adapter_keys = feature_adapter_keys_arg
            adapter_feature_keys_source = SOURCE_EXPLICIT
        elif explicit_alias:
            adapter_keys = _feature_keys_for_source(config["adapter_feature_source_channels"])
            adapter_feature_keys_source = SOURCE_ALIAS

    if config["lora"] != NONE:
        lora_block_mode = str(getattr(args, "lora_block_mode", "tap"))
        if lora_block_mode == "tap" and lora_tap_layers_arg:
            lora_tap_layers = lora_tap_layers_arg
            lora_tap_layers_source = SOURCE_EXPLICIT
        elif lora_block_mode == "tap":
            lora_tap_layers_source = _source_for_default(explicit_alias)
        elif lora_block_mode != "tap":
            lora_tap_layers_source = NOT_APPLICABLE

    sources["bridge_feature_keys"] = bridge_feature_keys_source
    sources["feature_adapter_keys"] = adapter_feature_keys_source
    sources["bridge_layers"] = bridge_layers_source
    sources["lora_tap_layers"] = lora_tap_layers_source

    if config["bridge"] == NONE:
        sources["bridge_feature_source_channels"] = NOT_APPLICABLE
    if config["decoder_feature_adapter"] == NONE:
        sources["adapter_feature_source_channels"] = NOT_APPLICABLE

    bridge_source = NOT_APPLICABLE
    if config["bridge"] != NONE or config["decoder_feature_adapter"] != NONE:
        bridge_source = str(getattr(args, "bridge_source", "ram_core"))
        sources["bridge_source"] = SOURCE_EXPLICIT if "bridge_source" in explicit_cli_args else _source_for_default(explicit_alias)
    else:
        sources["bridge_source"] = NOT_APPLICABLE

    if config["lora"] != NONE:
        lora_rank = int(getattr(args, "lora_rank", 8))
        lora_alpha = float(getattr(args, "lora_alpha", 16.0))
        lora_lr = float(getattr(args, "lora_lr", 5e-5))
        sources["lora_rank"] = SOURCE_EXPLICIT if "lora_rank" in explicit_cli_args else _source_for_default(explicit_alias)
        sources["lora_alpha"] = SOURCE_EXPLICIT if "lora_alpha" in explicit_cli_args else _source_for_default(explicit_alias)
        sources["lora_lr"] = SOURCE_EXPLICIT if "lora_lr" in explicit_cli_args else _source_for_default(explicit_alias)
    else:
        lora_rank = NOT_APPLICABLE
        lora_alpha = NOT_APPLICABLE
        lora_lr = NOT_APPLICABLE
        sources["lora_rank"] = NOT_APPLICABLE
        sources["lora_alpha"] = NOT_APPLICABLE
        sources["lora_lr"] = NOT_APPLICABLE

    config["kitti_eval_protocol"], sources["kitti_eval_protocol"] = _kitti_protocol_from_args(args)
    config["input_type_alias"] = str(alias) if explicit_alias else _legacy_alias_from_config(config)
    config["input_type_alias_source"] = alias_source
    raw_details = _raw_storage_details(config, args, sources)
    ram_core_type = _ram_core_type(config["front_end"], sources)
    imagenet_norm_enabled = _imagenet_norm_enabled(config, args, sources)
    kitti_model_source, eval_input_domain = _kitti_details(config["kitti_eval_protocol"], sources)
    loss_lambda_grad, loss_grad_scales, loss_mask_downsample = _loss_details(args, sources)
    not_applicable = tuple(sorted(field_name for field_name, source in sources.items() if source == NOT_APPLICABLE))

    resolved = ResolvedConfig(
        input_domain=config["input_domain"],
        front_end=config["front_end"],
        dataset_family=config["dataset_family"],
        dataset_input_mode=config["dataset_input_mode"],
        model_input_tensor=config["model_input_tensor"],
        bridge=config["bridge"],
        decoder_feature_adapter=config["decoder_feature_adapter"],
        lora=config["lora"],
        bridge_feature_source_channels=config["bridge_feature_source_channels"],
        adapter_feature_source_channels=config["adapter_feature_source_channels"],
        feature_adapter_keys=adapter_keys,
        bridge_feature_keys=bridge_keys,
        bridge_layers=bridge_layers,
        lora_tap_layers=lora_tap_layers,
        raw_storage_format=config["raw_storage_format"],
        kitti_eval_protocol=config["kitti_eval_protocol"],
        input_type_alias=config["input_type_alias"],
        input_type_alias_source=config["input_type_alias_source"],
        raw_storage_channel_order=raw_details["raw_storage_channel_order"],
        raw_model_channel_order=raw_details["raw_model_channel_order"],
        raw_decompand=raw_details["raw_decompand"],
        raw_post_decode_norm=raw_details["raw_post_decode_norm"],
        raw_channel_count=raw_details["raw_channel_count"],
        ram_core_type=ram_core_type,
        imagenet_norm_enabled=imagenet_norm_enabled,
        bridge_source=bridge_source,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_lr=lora_lr,
        loss_lambda_grad=loss_lambda_grad,
        loss_grad_scales=loss_grad_scales,
        loss_mask_downsample=loss_mask_downsample,
        kitti_model_source=kitti_model_source,
        eval_input_domain=eval_input_domain,
        sources=sources,
        not_applicable=not_applicable,
    )
    validate_resolved_config(resolved, args)
    return resolved


def ensure_resolved_config(args: Any) -> ResolvedConfig:
    resolved = getattr(args, "resolved_config", None)
    if isinstance(resolved, ResolvedConfig):
        return resolved
    if isinstance(resolved, dict):
        resolved = ResolvedConfig.from_dict(resolved)
        validate_resolved_config(resolved, args)
    else:
        resolved = resolve_config_from_args(args)
    setattr(args, "resolved_config", resolved)
    setattr(args, "input_type", resolved.input_type_alias)
    setattr(args, "bridge_feature_keys", list(resolved.bridge_feature_keys) if resolved.bridge_feature_keys else None)
    setattr(args, "feature_adapter_keys", list(resolved.feature_adapter_keys) if resolved.feature_adapter_keys else None)
    setattr(args, "bridge_layers", list(resolved.bridge_layers) if resolved.bridge_layers else None)
    setattr(args, "lora_tap_layers", list(resolved.lora_tap_layers) if resolved.lora_tap_layers else None)
    setattr(args, "raw_storage_format", resolved.raw_storage_format)
    return resolved
from finetune_stf.dataset.raw_storage import get_raw_storage_spec
