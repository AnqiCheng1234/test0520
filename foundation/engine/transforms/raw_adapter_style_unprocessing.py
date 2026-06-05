from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .unprocessing import build_unprocessing_transform_from_preset


Tensor = torch.Tensor

NOT_APPLICABLE = "not_applicable"
UNPROCESSING_METHODS = ("old_brooks_preset", "raw_adapter_style")
RAW_ADAPTER_VERSION = "2026-05-25.raw_adapter_style_v1"
RAW_ADAPTER_CFA_PATTERNS = ("RGGB",)
RAW_ADAPTER_PACKED_CHANNEL_ORDER = "R_Gr_Gb_B"
RAW_ADAPTER_PACKED_CHANNELS = ("R", "Gr", "Gb", "B")
RAW_ADAPTER_ORIGINAL_ORDER = ("R", "G1", "G2", "B")
RAW_ADAPTER_RGB_TRANSFER = "srgb_piecewise"
RAW_ADAPTER_INVERSE_TONE_CHOICES = ("none", "global_0p15")
RAW_ADAPTER_CCM_CHOICES = ("identity", "generic_d65")
RAW_ADAPTER_VARIANTS = ("normal", "dark", "over")
RAW_ADAPTER_VARIANT_POLICIES = (*RAW_ADAPTER_VARIANTS, "mix")
RAW_ADAPTER_NOISE_MEAN_MODES = ("zero", "rawadapter_text")
RAW_ADAPTER_RANDOM_SEED_POLICIES = ("dataloader_generator", "path_hash")

RAW_ADAPTER_GENERIC_D65 = (
    (0.86, 0.08, 0.06),
    (0.05, 0.90, 0.05),
    (0.04, 0.12, 0.84),
)

RAW_ADAPTER_HASH_KEYS = (
    "unprocessing_method",
    "raw_adapter_backend",
    "raw_adapter_cfa_pattern",
    "raw_adapter_packed_channel_order",
    "raw_adapter_rgb_transfer",
    "raw_adapter_inverse_tone",
    "raw_adapter_ccm",
    "raw_adapter_red_gain_range",
    "raw_adapter_blue_gain_range",
    "raw_adapter_fixed_red_gain",
    "raw_adapter_fixed_blue_gain",
    "raw_adapter_variant_policy",
    "raw_adapter_variant_weights",
    "raw_adapter_fixed_light_scale",
    "raw_adapter_dark_light_scale_range",
    "raw_adapter_over_light_scale_range",
    "noise_model",
    "noise_realization_applied",
    "raw_adapter_shot_noise",
    "raw_adapter_read_noise",
    "raw_adapter_noise_mean_mode",
    "raw_adapter_black_level",
    "raw_adapter_white_level",
    "randomize_unprocessing",
    "raw_adapter_random_seed_policy",
    "raw_adapter_external_raw_rgb_root",
    "raw_adapter_external_key",
    "raw_adapter_external_cache_space",
)

RAW_ADAPTER_COMPARE_KEYS = (
    "raw_adapter_inverse_tone",
    "raw_adapter_ccm",
    "fixed_red_gain",
    "fixed_blue_gain",
    "variant_policy",
    "fixed_light_scale",
    "noise_model",
    "noise_realization_applied",
    "shot_noise",
    "read_noise",
    "noise_mean_mode",
    "black_level",
    "white_level",
)

RAW_ADAPTER_ARG_FIELDS = (
    "raw_adapter_backend",
    "raw_adapter_cfa_pattern",
    "raw_adapter_packed_channel_order",
    "raw_adapter_rgb_transfer",
    "raw_adapter_inverse_tone",
    "raw_adapter_ccm",
    "raw_adapter_red_gain_range",
    "raw_adapter_blue_gain_range",
    "raw_adapter_fixed_red_gain",
    "raw_adapter_fixed_blue_gain",
    "raw_adapter_variant_policy",
    "raw_adapter_variant_weights",
    "raw_adapter_fixed_light_scale",
    "raw_adapter_dark_light_scale_range",
    "raw_adapter_over_light_scale_range",
    "raw_adapter_shot_noise",
    "raw_adapter_read_noise",
    "raw_adapter_noise_mean_mode",
    "raw_adapter_black_level",
    "raw_adapter_white_level",
    "raw_adapter_random_seed_policy",
    "raw_adapter_external_raw_rgb_root",
    "raw_adapter_external_key",
    "raw_adapter_external_cache_space",
)


def _is_not_applicable(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in ("", NOT_APPLICABLE)
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value) == 0
    return False


def _clean_string(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is required")
    return str(value).strip()


def _require_choice(value: Any, *, field: str, choices: Sequence[str]) -> str:
    text = _clean_string(value, field=field)
    if text not in choices:
        raise ValueError(f"{field} must be one of {list(choices)}, got {text!r}")
    return text


def _require_float(value: Any, *, field: str) -> float:
    if _is_not_applicable(value):
        raise ValueError(f"{field} is required")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


def _optional_string(value: Any) -> str:
    return NOT_APPLICABLE if _is_not_applicable(value) else str(value).strip()


def _float_pair(value: Any, *, field: str) -> Tuple[float, float]:
    if _is_not_applicable(value):
        raise ValueError(f"{field} is required")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field} is empty")
        chunks = [item.strip() for item in text.replace(",", " ").split() if item.strip()]
    else:
        chunks = list(value)
    if len(chunks) != 2:
        raise ValueError(f"{field} must have exactly two values, got {value!r}")
    lo, hi = float(chunks[0]), float(chunks[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError(f"{field} must satisfy finite low <= high, got {(lo, hi)}")
    return float(lo), float(hi)


def _range_contains(value: float, value_range: Sequence[float], *, field: str, range_field: str) -> None:
    lo, hi = float(value_range[0]), float(value_range[1])
    if not (lo <= float(value) <= hi):
        raise ValueError(f"{field}={value} must fall inside {range_field}={(lo, hi)}")


def parse_raw_adapter_variant_weights(value: Any) -> Dict[str, float]:
    if _is_not_applicable(value):
        return {}
    if isinstance(value, Mapping):
        raw = {str(k).strip(): float(v) for k, v in value.items()}
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{"):
            parsed = json.loads(text)
            if not isinstance(parsed, Mapping):
                raise ValueError("raw_adapter_variant_weights JSON must be an object")
            raw = {str(k).strip(): float(v) for k, v in parsed.items()}
        else:
            raw = {}
            for term in text.split(","):
                term = term.strip()
                if not term:
                    continue
                if "=" not in term:
                    raise ValueError(
                        "raw_adapter_variant_weights must use key=value entries, "
                        f"got {value!r}"
                    )
                key, val = term.split("=", 1)
                raw[key.strip()] = float(val.strip())
    else:
        raise TypeError("raw_adapter_variant_weights must be a mapping or key=value string")

    unknown = sorted(set(raw) - set(RAW_ADAPTER_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown raw adapter variants in weights: {unknown}")
    weights = {name: float(raw.get(name, 0.0)) for name in RAW_ADAPTER_VARIANTS}
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError(f"raw_adapter_variant_weights must be non-negative, got {weights}")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"raw_adapter_variant_weights must sum to > 0, got {weights}")
    return {name: float(weight / total) for name, weight in weights.items()}


def _one_hot_variant_weights(variant: str) -> Dict[str, float]:
    return {name: 1.0 if name == variant else 0.0 for name in RAW_ADAPTER_VARIANTS}


def _validate_one_hot_matches(weights: Mapping[str, float], variant: str) -> None:
    expected = _one_hot_variant_weights(variant)
    for key, expected_value in expected.items():
        if not math.isclose(float(weights.get(key, 0.0)), expected_value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "raw_adapter_variant_weights must be one-hot and match "
                f"raw_adapter_variant_policy={variant!r}, got {dict(weights)}"
            )


def _validate_light_scale_for_variant(
    *,
    variant: str,
    fixed_light_scale: float,
    dark_range: Sequence[float],
    over_range: Sequence[float],
) -> None:
    if variant == "normal":
        if not math.isclose(float(fixed_light_scale), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("raw_adapter_fixed_light_scale must be 1.0 when variant_policy='normal'")
    elif variant == "dark":
        _range_contains(
            fixed_light_scale,
            dark_range,
            field="raw_adapter_fixed_light_scale",
            range_field="raw_adapter_dark_light_scale_range",
        )
    elif variant == "over":
        _range_contains(
            fixed_light_scale,
            over_range,
            field="raw_adapter_fixed_light_scale",
            range_field="raw_adapter_over_light_scale_range",
        )
    else:
        raise ValueError(f"Unsupported fixed raw adapter variant: {variant!r}")


def _raw_adapter_hash(config: Mapping[str, Any]) -> str:
    payload = {key: config[key] for key in RAW_ADAPTER_HASH_KEYS}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _old_brooks_placeholders(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "unprocessing_method": "old_brooks_preset",
        "vkitti_unprocessing_preset": str(config["vkitti_unprocessing_preset"]),
        "vkitti_unprocessing_mix_weights": config.get("vkitti_unprocessing_mix_weights"),
        "randomize_unprocessing": bool(config["randomize_unprocessing"]),
        "noise_model": "poisson_gaussian",
        "noise_realization_applied": bool(config["randomize_unprocessing"]),
        "raw_adapter_backend": NOT_APPLICABLE,
        "raw_adapter_config_hash": NOT_APPLICABLE,
        "raw_adapter_original_order": NOT_APPLICABLE,
        "raw_adapter_variant_weights": {},
        "raw_adapter_external_raw_rgb_root": NOT_APPLICABLE,
        "raw_adapter_external_key": NOT_APPLICABLE,
        "raw_adapter_external_cache_space": NOT_APPLICABLE,
        "raw_adapter_inverse_tone": NOT_APPLICABLE,
        "raw_adapter_ccm": NOT_APPLICABLE,
        "fixed_red_gain": NOT_APPLICABLE,
        "fixed_blue_gain": NOT_APPLICABLE,
        "variant_policy": NOT_APPLICABLE,
        "fixed_light_scale": NOT_APPLICABLE,
        "shot_noise": NOT_APPLICABLE,
        "read_noise": NOT_APPLICABLE,
        "noise_mean_mode": NOT_APPLICABLE,
        "black_level": NOT_APPLICABLE,
        "white_level": NOT_APPLICABLE,
    }


def resolve_unprocessing_config(
    source: Mapping[str, Any],
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    method = str(source.get("unprocessing_method", "old_brooks_preset"))
    if method not in UNPROCESSING_METHODS:
        raise ValueError(f"unprocessing_method must be one of {UNPROCESSING_METHODS}, got {method!r}")

    randomize_value = source.get("randomize_unprocessing")
    if randomize_value is None and validate:
        raise ValueError("randomize_unprocessing must be explicitly resolved before building unprocessing")
    randomize = bool(randomize_value)

    if method == "old_brooks_preset":
        preset = str(source.get("vkitti_unprocessing_preset", NOT_APPLICABLE))
        if preset == NOT_APPLICABLE:
            raise ValueError(
                "vkitti_unprocessing_preset must be an active preset when "
                "unprocessing_method='old_brooks_preset'"
            )
        active_raw_fields = [
            field
            for field in RAW_ADAPTER_ARG_FIELDS
            if not _is_not_applicable(source.get(field))
        ]
        if active_raw_fields:
            raise ValueError(
                "raw-adapter fields are not applicable when unprocessing_method='old_brooks_preset': "
                + ", ".join(active_raw_fields)
            )
        config = {
            "unprocessing_method": method,
            "vkitti_unprocessing_preset": preset,
            "vkitti_unprocessing_mix_weights": source.get("vkitti_unprocessing_mix_weights"),
            "randomize_unprocessing": randomize,
            "noise_model": "poisson_gaussian",
            "noise_realization_applied": randomize,
        }
        config.update(_old_brooks_placeholders(config))
        return config

    preset = source.get("vkitti_unprocessing_preset", NOT_APPLICABLE)
    if not _is_not_applicable(preset):
        raise ValueError(
            "vkitti_unprocessing_preset must be 'not_applicable' when "
            "unprocessing_method='raw_adapter_style'"
        )
    if not _is_not_applicable(source.get("vkitti_unprocessing_mix_weights")):
        raise ValueError(
            "vkitti_unprocessing_mix_weights belongs to old_brooks_preset and must not be active "
            "for raw_adapter_style"
        )

    backend = _require_choice(
        source.get("raw_adapter_backend"),
        field="raw_adapter_backend",
        choices=("analytic", "external_raw_rgb_cache"),
    )
    cfa_pattern = _require_choice(
        source.get("raw_adapter_cfa_pattern"),
        field="raw_adapter_cfa_pattern",
        choices=RAW_ADAPTER_CFA_PATTERNS,
    )
    packed_channel_order = _require_choice(
        source.get("raw_adapter_packed_channel_order"),
        field="raw_adapter_packed_channel_order",
        choices=(RAW_ADAPTER_PACKED_CHANNEL_ORDER,),
    )
    rgb_transfer = _require_choice(
        source.get("raw_adapter_rgb_transfer"),
        field="raw_adapter_rgb_transfer",
        choices=(RAW_ADAPTER_RGB_TRANSFER,),
    )
    inverse_tone = _require_choice(
        source.get("raw_adapter_inverse_tone"),
        field="raw_adapter_inverse_tone",
        choices=RAW_ADAPTER_INVERSE_TONE_CHOICES,
    )
    ccm = _require_choice(source.get("raw_adapter_ccm"), field="raw_adapter_ccm", choices=RAW_ADAPTER_CCM_CHOICES)
    red_gain_range = _float_pair(source.get("raw_adapter_red_gain_range"), field="raw_adapter_red_gain_range")
    blue_gain_range = _float_pair(source.get("raw_adapter_blue_gain_range"), field="raw_adapter_blue_gain_range")
    variant_policy = _require_choice(
        source.get("raw_adapter_variant_policy"),
        field="raw_adapter_variant_policy",
        choices=RAW_ADAPTER_VARIANT_POLICIES,
    )
    variant_weights = parse_raw_adapter_variant_weights(source.get("raw_adapter_variant_weights"))
    if not variant_weights:
        raise ValueError("raw_adapter_variant_weights must be explicitly provided for raw_adapter_style")
    dark_range = _float_pair(
        source.get("raw_adapter_dark_light_scale_range"),
        field="raw_adapter_dark_light_scale_range",
    )
    over_range = _float_pair(
        source.get("raw_adapter_over_light_scale_range"),
        field="raw_adapter_over_light_scale_range",
    )
    shot_noise = _require_float(source.get("raw_adapter_shot_noise"), field="raw_adapter_shot_noise")
    read_noise = _require_float(source.get("raw_adapter_read_noise"), field="raw_adapter_read_noise")
    noise_mean_mode = _require_choice(
        source.get("raw_adapter_noise_mean_mode"),
        field="raw_adapter_noise_mean_mode",
        choices=RAW_ADAPTER_NOISE_MEAN_MODES,
    )
    black_level = _require_float(source.get("raw_adapter_black_level"), field="raw_adapter_black_level")
    white_level = _require_float(source.get("raw_adapter_white_level"), field="raw_adapter_white_level")
    if not (0.0 <= black_level < white_level <= 1.0):
        raise ValueError(
            "Require 0 <= raw_adapter_black_level < raw_adapter_white_level <= 1, "
            f"got {(black_level, white_level)}"
        )
    random_seed_policy = _require_choice(
        source.get("raw_adapter_random_seed_policy"),
        field="raw_adapter_random_seed_policy",
        choices=RAW_ADAPTER_RANDOM_SEED_POLICIES,
    )
    if random_seed_policy == "path_hash":
        raise ValueError("raw_adapter_random_seed_policy='path_hash' is not implemented yet")

    external_root = _optional_string(source.get("raw_adapter_external_raw_rgb_root"))
    external_key = _optional_string(source.get("raw_adapter_external_key"))
    external_cache_space = _optional_string(source.get("raw_adapter_external_cache_space"))
    if backend == "analytic":
        active_external = [
            name
            for name, value in (
                ("raw_adapter_external_raw_rgb_root", external_root),
                ("raw_adapter_external_key", external_key),
                ("raw_adapter_external_cache_space", external_cache_space),
            )
            if value != NOT_APPLICABLE
        ]
        if active_external:
            raise ValueError(
                "external raw-RGB cache fields are not applicable for raw_adapter_backend='analytic': "
                + ", ".join(active_external)
            )
    else:
        if external_root == NOT_APPLICABLE or external_key == NOT_APPLICABLE or external_cache_space == NOT_APPLICABLE:
            raise ValueError(
                "raw_adapter_backend='external_raw_rgb_cache' requires external root, key, and cache space"
            )

    fixed_red_gain: Optional[float]
    fixed_blue_gain: Optional[float]
    fixed_light_scale: Optional[float]
    if randomize:
        if variant_policy != "mix":
            _validate_one_hot_matches(variant_weights, variant_policy)
        fixed_red_gain = None if _is_not_applicable(source.get("raw_adapter_fixed_red_gain")) else float(source["raw_adapter_fixed_red_gain"])
        fixed_blue_gain = None if _is_not_applicable(source.get("raw_adapter_fixed_blue_gain")) else float(source["raw_adapter_fixed_blue_gain"])
        fixed_light_scale = None if _is_not_applicable(source.get("raw_adapter_fixed_light_scale")) else float(source["raw_adapter_fixed_light_scale"])
        noise_model = "raw_adapter_gaussian_signal_dependent"
        noise_realization_applied = True
    else:
        fixed_red_gain = _require_float(source.get("raw_adapter_fixed_red_gain"), field="raw_adapter_fixed_red_gain")
        fixed_blue_gain = _require_float(source.get("raw_adapter_fixed_blue_gain"), field="raw_adapter_fixed_blue_gain")
        fixed_light_scale = _require_float(
            source.get("raw_adapter_fixed_light_scale"),
            field="raw_adapter_fixed_light_scale",
        )
        _range_contains(fixed_red_gain, red_gain_range, field="raw_adapter_fixed_red_gain", range_field="raw_adapter_red_gain_range")
        _range_contains(fixed_blue_gain, blue_gain_range, field="raw_adapter_fixed_blue_gain", range_field="raw_adapter_blue_gain_range")
        if variant_policy == "mix":
            raise ValueError("variant_policy='mix' is not allowed when randomize_unprocessing=False")
        _validate_one_hot_matches(variant_weights, variant_policy)
        _validate_light_scale_for_variant(
            variant=variant_policy,
            fixed_light_scale=fixed_light_scale,
            dark_range=dark_range,
            over_range=over_range,
        )
        noise_model = "none"
        noise_realization_applied = False

    config: Dict[str, Any] = {
        "unprocessing_method": "raw_adapter_style",
        "vkitti_unprocessing_preset": NOT_APPLICABLE,
        "vkitti_unprocessing_mix_weights": None,
        "randomize_unprocessing": randomize,
        "raw_adapter_backend": backend,
        "raw_adapter_cfa_pattern": cfa_pattern,
        "raw_adapter_packed_channel_order": packed_channel_order,
        "raw_adapter_rgb_transfer": rgb_transfer,
        "raw_adapter_inverse_tone": inverse_tone,
        "raw_adapter_ccm": ccm,
        "raw_adapter_red_gain_range": [float(red_gain_range[0]), float(red_gain_range[1])],
        "raw_adapter_blue_gain_range": [float(blue_gain_range[0]), float(blue_gain_range[1])],
        "raw_adapter_fixed_red_gain": None if fixed_red_gain is None else float(fixed_red_gain),
        "raw_adapter_fixed_blue_gain": None if fixed_blue_gain is None else float(fixed_blue_gain),
        "raw_adapter_variant_policy": variant_policy,
        "raw_adapter_variant_weights": {name: float(variant_weights[name]) for name in RAW_ADAPTER_VARIANTS},
        "raw_adapter_fixed_light_scale": None if fixed_light_scale is None else float(fixed_light_scale),
        "raw_adapter_dark_light_scale_range": [float(dark_range[0]), float(dark_range[1])],
        "raw_adapter_over_light_scale_range": [float(over_range[0]), float(over_range[1])],
        "raw_adapter_shot_noise": float(shot_noise),
        "raw_adapter_read_noise": float(read_noise),
        "raw_adapter_noise_mean_mode": noise_mean_mode,
        "raw_adapter_black_level": float(black_level),
        "raw_adapter_white_level": float(white_level),
        "raw_adapter_random_seed_policy": random_seed_policy,
        "raw_adapter_external_raw_rgb_root": external_root,
        "raw_adapter_external_key": external_key,
        "raw_adapter_external_cache_space": external_cache_space,
        "noise_model": noise_model,
        "noise_realization_applied": bool(noise_realization_applied),
    }
    config["raw_adapter_config_hash"] = _raw_adapter_hash(config)
    return config


def raw_adapter_summary_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = resolve_unprocessing_config(config)
    if resolved["unprocessing_method"] != "raw_adapter_style":
        return _old_brooks_placeholders(resolved)
    return {
        "unprocessing_method": "raw_adapter_style",
        "vkitti_unprocessing_preset": NOT_APPLICABLE,
        "vkitti_unprocessing_mix_weights": None,
        "randomize": bool(resolved["randomize_unprocessing"]),
        "randomize_unprocessing": bool(resolved["randomize_unprocessing"]),
        "raw_adapter_backend": resolved["raw_adapter_backend"],
        "raw_adapter_config_hash": resolved["raw_adapter_config_hash"],
        "preset_hash": resolved["raw_adapter_config_hash"],
        "selected_sub_preset_hash": resolved["raw_adapter_config_hash"],
        "raw_adapter_cfa_pattern": resolved["raw_adapter_cfa_pattern"],
        "cfa_pattern": resolved["raw_adapter_cfa_pattern"],
        "raw_adapter_packed_channel_order": resolved["raw_adapter_packed_channel_order"],
        "packed_channel_order": list(RAW_ADAPTER_PACKED_CHANNELS),
        "raw_adapter_original_order": list(RAW_ADAPTER_ORIGINAL_ORDER),
        "raw_adapter_rgb_transfer": resolved["raw_adapter_rgb_transfer"],
        "rgb_transfer": resolved["raw_adapter_rgb_transfer"],
        "raw_adapter_inverse_tone": resolved["raw_adapter_inverse_tone"],
        "inverse_tone": resolved["raw_adapter_inverse_tone"],
        "raw_adapter_ccm": resolved["raw_adapter_ccm"],
        "ccm": resolved["raw_adapter_ccm"],
        "raw_adapter_red_gain_range": list(resolved["raw_adapter_red_gain_range"]),
        "raw_adapter_blue_gain_range": list(resolved["raw_adapter_blue_gain_range"]),
        "fixed_red_gain": resolved["raw_adapter_fixed_red_gain"],
        "fixed_blue_gain": resolved["raw_adapter_fixed_blue_gain"],
        "variant_policy": resolved["raw_adapter_variant_policy"],
        "raw_adapter_variant_policy": resolved["raw_adapter_variant_policy"],
        "raw_adapter_variant_weights": dict(resolved["raw_adapter_variant_weights"]),
        "fixed_light_scale": resolved["raw_adapter_fixed_light_scale"],
        "raw_adapter_dark_light_scale_range": list(resolved["raw_adapter_dark_light_scale_range"]),
        "raw_adapter_over_light_scale_range": list(resolved["raw_adapter_over_light_scale_range"]),
        "noise_model": resolved["noise_model"],
        "noise_realization_applied": bool(resolved["noise_realization_applied"]),
        "shot_noise": resolved["raw_adapter_shot_noise"],
        "read_noise": resolved["raw_adapter_read_noise"],
        "noise_mean_mode": resolved["raw_adapter_noise_mean_mode"],
        "black_level": resolved["raw_adapter_black_level"],
        "white_level": resolved["raw_adapter_white_level"],
        "raw_adapter_random_seed_policy": resolved["raw_adapter_random_seed_policy"],
        "raw_adapter_external_raw_rgb_root": resolved["raw_adapter_external_raw_rgb_root"],
        "raw_adapter_external_key": resolved["raw_adapter_external_key"],
        "raw_adapter_external_cache_space": resolved["raw_adapter_external_cache_space"],
    }


def assert_unprocessing_summaries_compatible(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    context: str,
) -> None:
    ref_method = str(reference.get("unprocessing_method", "old_brooks_preset"))
    cand_method = str(candidate.get("unprocessing_method", "old_brooks_preset"))
    if ref_method != cand_method:
        raise ValueError(f"{context}: unprocessing_method mismatch: {ref_method!r} != {cand_method!r}")
    if ref_method != "raw_adapter_style":
        return
    ref_hash = reference.get("raw_adapter_config_hash")
    cand_hash = candidate.get("raw_adapter_config_hash")
    if ref_hash != cand_hash:
        raise ValueError(f"{context}: raw_adapter_config_hash mismatch: {ref_hash!r} != {cand_hash!r}")
    for key in RAW_ADAPTER_COMPARE_KEYS:
        if reference.get(key) != candidate.get(key):
            raise ValueError(
                f"{context}: raw adapter summary field {key!r} mismatch: "
                f"{reference.get(key)!r} != {candidate.get(key)!r}"
            )


def assert_unprocessing_summary_matches_config(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    expected = raw_adapter_summary_from_config(config)
    assert_unprocessing_summaries_compatible(expected, summary, context=context)


class RawAdapterStyleUnprocessingTransform(nn.Module):
    """RAW-Adapter-style analytic unprocessing for online pseudo-RAW generation."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        resolved = resolve_unprocessing_config(config)
        if resolved["unprocessing_method"] != "raw_adapter_style":
            raise ValueError("RawAdapterStyleUnprocessingTransform requires unprocessing_method='raw_adapter_style'")
        if resolved["raw_adapter_backend"] != "analytic":
            raise NotImplementedError("external_raw_rgb_cache backend is not implemented in the online transform yet")

        self.config = resolved
        self.randomize = bool(resolved["randomize_unprocessing"])
        self.cfa_pattern = str(resolved["raw_adapter_cfa_pattern"])
        self.rgb_transfer = str(resolved["raw_adapter_rgb_transfer"])
        self.inverse_tone = str(resolved["raw_adapter_inverse_tone"])
        self.ccm_name = str(resolved["raw_adapter_ccm"])
        self.red_gain_range = tuple(float(v) for v in resolved["raw_adapter_red_gain_range"])
        self.blue_gain_range = tuple(float(v) for v in resolved["raw_adapter_blue_gain_range"])
        self.fixed_red_gain = resolved["raw_adapter_fixed_red_gain"]
        self.fixed_blue_gain = resolved["raw_adapter_fixed_blue_gain"]
        self.variant_policy = str(resolved["raw_adapter_variant_policy"])
        self.variant_weights = dict(resolved["raw_adapter_variant_weights"])
        self.fixed_light_scale = resolved["raw_adapter_fixed_light_scale"]
        self.dark_light_scale_range = tuple(float(v) for v in resolved["raw_adapter_dark_light_scale_range"])
        self.over_light_scale_range = tuple(float(v) for v in resolved["raw_adapter_over_light_scale_range"])
        self.shot_noise = float(resolved["raw_adapter_shot_noise"])
        self.read_noise = float(resolved["raw_adapter_read_noise"])
        self.noise_mean_mode = str(resolved["raw_adapter_noise_mean_mode"])
        self.black_level = float(resolved["raw_adapter_black_level"])
        self.white_level = float(resolved["raw_adapter_white_level"])
        self.noise_model = str(resolved["noise_model"])
        self.noise_realization_applied = bool(resolved["noise_realization_applied"])
        self.raw_adapter_config_hash = str(resolved["raw_adapter_config_hash"])

        if self.ccm_name == "identity":
            ccm = torch.eye(3, dtype=torch.float32)
        elif self.ccm_name == "generic_d65":
            ccm = torch.tensor(RAW_ADAPTER_GENERIC_D65, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported raw adapter CCM: {self.ccm_name}")
        self.register_buffer("rgb2cam", ccm, persistent=False)
        self.register_buffer("cam2rgb", torch.linalg.inv(ccm), persistent=False)

    def describe_config(self) -> Dict[str, Any]:
        return raw_adapter_summary_from_config(self.config)

    def forward(self, image: Tensor, *, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Dict[str, object]]:
        image, squeeze = self._prepare_input(image)
        batch_size = image.shape[0]
        device = image.device
        dtype = image.dtype

        params = self._sample_params(batch_size=batch_size, device=device, dtype=dtype, generator=generator)
        x = image.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
        x = self.srgb_to_linear_piecewise(x)
        if self.inverse_tone == "global_0p15":
            x = self.inverse_global_tone(x, strength=0.15)
        elif self.inverse_tone != "none":
            raise ValueError(f"Unsupported raw adapter inverse tone: {self.inverse_tone}")

        rgb2cam = self.rgb2cam.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.einsum("nhwc,nkc->nhwk", x, rgb2cam).clamp(0.0, 1.0)
        x = self.inverse_white_balance(x, params["red_gain"], params["blue_gain"])
        x = (x * params["light_scale"].view(batch_size, 1, 1, 1)).clamp_min(0.0)

        if self.noise_realization_applied:
            x = self.add_signal_dependent_gaussian_noise(x, generator=generator)

        x = x.clamp(0.0, 1.0)
        x = (self.black_level + x * (self.white_level - self.black_level)).clamp(0.0, 1.0)
        packed = self.pack_rggb(x).float()
        metadata = self._finalize_metadata(params, batch_size=batch_size, squeeze=squeeze, dtype=dtype, device=device)
        return (packed.squeeze(0) if squeeze else packed), metadata

    def srgb_to_linear_piecewise(self, image: Tensor) -> Tensor:
        image = image.clamp(0.0, 1.0)
        return torch.where(
            image <= 0.04045,
            image / 12.92,
            torch.pow((image + 0.055) / 1.055, 2.4),
        )

    def inverse_global_tone(self, image: Tensor, *, strength: float = 0.15) -> Tensor:
        image = image.clamp(0.0, 1.0)
        denom = torch.clamp(1.0 - float(strength) * (1.0 - image), min=1e-6)
        return (image / denom).clamp(0.0, 1.0)

    def inverse_white_balance(self, image: Tensor, red_gain: Tensor, blue_gain: Tensor) -> Tensor:
        gains = torch.stack(
            [
                1.0 / red_gain,
                torch.ones_like(red_gain),
                1.0 / blue_gain,
            ],
            dim=-1,
        ).view(-1, 1, 1, 3)
        return (image * gains).clamp(0.0, 1.0)

    def add_signal_dependent_gaussian_noise(
        self,
        signal: Tensor,
        *,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        variance = float(self.read_noise) ** 2 + float(self.shot_noise) * signal.clamp_min(0.0)
        loc = signal if self.noise_mean_mode == "rawadapter_text" else torch.zeros_like(signal)
        noise = torch.randn(signal.shape, device=signal.device, dtype=signal.dtype, generator=generator)
        noise = loc + noise * variance.clamp_min(0.0).sqrt()
        return signal + noise

    def pack_rggb(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Expected NHWC RGB/raw-RGB image, got {tuple(image.shape)}")
        height = image.shape[1] - (image.shape[1] % 2)
        width = image.shape[2] - (image.shape[2] % 2)
        if height <= 0 or width <= 0:
            raise ValueError(f"Input image must be at least 2x2, got {tuple(image.shape[1:3])}")
        image = image[:, :height, :width, :]
        return torch.stack(
            [
                image[:, 0::2, 0::2, 0],
                image[:, 0::2, 1::2, 1],
                image[:, 1::2, 0::2, 1],
                image[:, 1::2, 1::2, 2],
            ],
            dim=1,
        )

    def _sample_params(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Dict[str, object]:
        if self.randomize:
            red_gain = self._sample_uniform(self.red_gain_range, batch_size, device, dtype, generator)
            blue_gain = self._sample_uniform(self.blue_gain_range, batch_size, device, dtype, generator)
            variants = self._sample_variants(batch_size, device=device, generator=generator)
            light_scale = self._light_scale_for_variants(variants, device=device, dtype=dtype, generator=generator)
        else:
            red_gain = torch.full((batch_size,), float(self.fixed_red_gain), device=device, dtype=dtype)
            blue_gain = torch.full((batch_size,), float(self.fixed_blue_gain), device=device, dtype=dtype)
            variants = [str(self.variant_policy)] * batch_size
            light_scale = torch.full((batch_size,), float(self.fixed_light_scale), device=device, dtype=dtype)

        return {
            "red_gain": red_gain,
            "blue_gain": blue_gain,
            "variant": variants,
            "light_scale": light_scale,
        }

    def _sample_uniform(
        self,
        value_range: Tuple[float, float],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        values = torch.empty((batch_size,), device=device, dtype=dtype)
        values.uniform_(float(value_range[0]), float(value_range[1]), generator=generator)
        return values

    def _sample_variants(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> list[str]:
        if self.variant_policy != "mix":
            return [self.variant_policy] * batch_size
        probs = torch.tensor(
            [float(self.variant_weights[name]) for name in RAW_ADAPTER_VARIANTS],
            device=device,
            dtype=torch.float32,
        )
        indices = torch.multinomial(probs, batch_size, replacement=True, generator=generator)
        return [RAW_ADAPTER_VARIANTS[int(idx)] for idx in indices.detach().cpu().tolist()]

    def _light_scale_for_variants(
        self,
        variants: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        values = []
        for variant in variants:
            if variant == "normal":
                values.append(torch.tensor(1.0, device=device, dtype=dtype))
            elif variant == "dark":
                values.append(self._sample_uniform(self.dark_light_scale_range, 1, device, dtype, generator)[0])
            elif variant == "over":
                values.append(self._sample_uniform(self.over_light_scale_range, 1, device, dtype, generator)[0])
            else:
                raise ValueError(f"Unsupported raw adapter variant: {variant!r}")
        return torch.stack(values, dim=0)

    def _prepare_input(self, image: Tensor) -> Tuple[Tensor, bool]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze = True
        elif image.ndim == 4:
            squeeze = False
        else:
            raise ValueError(f"Expected image with shape (3,H,W) or (N,3,H,W), got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {image.shape[1]}")
        height = image.shape[-2] - (image.shape[-2] % 2)
        width = image.shape[-1] - (image.shape[-1] % 2)
        if height <= 0 or width <= 0:
            raise ValueError(f"Input image must have spatial size >= 2, got {tuple(image.shape[-2:])}")
        return image[..., :height, :width].contiguous(), squeeze

    def _finalize_metadata(
        self,
        params: Mapping[str, object],
        *,
        batch_size: int,
        squeeze: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, object]:
        rgb2cam = self.rgb2cam.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        cam2rgb = self.cam2rgb.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        metadata: Dict[str, object] = {
            "unprocessing_method": "raw_adapter_style",
            "isp_profile_name": "raw_adapter_style",
            "isp_profile_group": "raw_adapter_style",
            "selected_sub_preset_name": "raw_adapter_style",
            "preset_version": RAW_ADAPTER_VERSION,
            "preset_hash": self.raw_adapter_config_hash,
            "preset_mix_weights": [{"name": "raw_adapter_style", "weight": 1.0}],
            "selected_sub_preset_hash": self.raw_adapter_config_hash,
            "packed_channel_order": list(RAW_ADAPTER_PACKED_CHANNELS),
            "cfa_pattern": [self.cfa_pattern] * batch_size,
            "noise_model": self.noise_model,
            "noise_realization_applied": self.noise_realization_applied,
            "randomize": self.randomize,
            "raw_adapter_backend": "analytic",
            "raw_adapter_config_hash": self.raw_adapter_config_hash,
            "raw_adapter_original_order": list(RAW_ADAPTER_ORIGINAL_ORDER),
            "rgb_transfer": self.rgb_transfer,
            "inverse_tone": self.inverse_tone,
            "ccm": self.ccm_name,
            "red_gain": params["red_gain"],
            "blue_gain": params["blue_gain"],
            "variant": params["variant"],
            "variant_policy": self.variant_policy,
            "variant_weights": dict(self.variant_weights),
            "light_scale": params["light_scale"],
            "shot_noise": torch.full((batch_size,), self.shot_noise, device=device, dtype=dtype),
            "read_noise": torch.full((batch_size,), self.read_noise, device=device, dtype=dtype),
            "noise_mean_mode": self.noise_mean_mode,
            "black_level": torch.full((batch_size,), self.black_level, device=device, dtype=dtype),
            "white_level": torch.full((batch_size,), self.white_level, device=device, dtype=dtype),
            "rgb2cam": rgb2cam,
            "cam2rgb": cam2rgb,
            "raw_adapter_random_seed_policy": self.config["raw_adapter_random_seed_policy"],
            "raw_adapter_external_raw_rgb_root": self.config["raw_adapter_external_raw_rgb_root"],
            "raw_adapter_external_key": self.config["raw_adapter_external_key"],
            "raw_adapter_external_cache_space": self.config["raw_adapter_external_cache_space"],
        }
        if not squeeze:
            return metadata

        batch_keys = {
            "cfa_pattern",
            "red_gain",
            "blue_gain",
            "variant",
            "light_scale",
            "shot_noise",
            "read_noise",
            "black_level",
            "white_level",
            "rgb2cam",
            "cam2rgb",
        }
        squeezed: Dict[str, object] = {}
        for key, value in metadata.items():
            if key not in batch_keys:
                squeezed[key] = value
            elif isinstance(value, torch.Tensor):
                squeezed[key] = value[0]
            elif isinstance(value, list):
                squeezed[key] = value[0]
            else:
                squeezed[key] = value
        return squeezed


def build_unprocessing_transform_from_resolved_config(
    config: Mapping[str, Any],
    *,
    split: str,
):
    resolved = resolve_unprocessing_config(config)
    method = str(resolved["unprocessing_method"])
    if method == "raw_adapter_style":
        transform = RawAdapterStyleUnprocessingTransform(resolved)
        return transform, transform.describe_config()

    randomize = bool(resolved["randomize_unprocessing"]) if split == "train" else False
    transform = build_unprocessing_transform_from_preset(
        str(resolved["vkitti_unprocessing_preset"]),
        randomize=randomize,
    )
    summary = _old_brooks_placeholders({**resolved, "randomize_unprocessing": randomize})
    return transform, summary
