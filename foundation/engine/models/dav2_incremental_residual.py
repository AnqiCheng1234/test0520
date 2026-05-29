from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune_stf.models.raw_ram import RamCore3
from finetune_stf.models.spatial_adapter import BACKBONE_INPUT_HW, SENSOR_INPUT_HW
from foundation.engine.transforms import packed_bayer_to_base_rgb


INCREMENTAL_METHOD_IDS = ("N2", "N3", "N4", "N5", "N7")
INCREMENTAL_FEATURE_SOURCES = ("x3", "ffm_mid", "rgb", "d1")
DELTA_CONDITIONS = ("feature_only", "feature_d1_stopgrad", "d1_only")
GATE_CONDITIONS = ("feature_d1", "d1_only")
RAW_FEATURE_ENCODER_TRAINABLE = ("true", "false", "not_applicable")


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=8, num_channels=int(channels))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _gn(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FeatureEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            _gn(out_ch),
            nn.GELU(),
            ResidualBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallResidualHead(nn.Module):
    def __init__(self, in_ch: int, *, out_kind: str, alpha: float = 0.5) -> None:
        super().__init__()
        if out_kind not in ("delta", "gate"):
            raise ValueError(f"out_kind must be 'delta' or 'gate', got {out_kind!r}")
        self.out_kind = out_kind
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            _gn(64),
            nn.GELU(),
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        last = self.net[-1]
        if not isinstance(last, nn.Conv2d):
            raise TypeError("Unexpected incremental head layout.")
        nn.init.zeros_(last.weight)
        if self.out_kind == "gate":
            nn.init.constant_(last.bias, -4.0)
        else:
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)[:, 0]
        if self.out_kind == "gate":
            return torch.sigmoid(y)
        return self.alpha * torch.tanh(y)


def _feature_raw_channels(feature_source: str) -> int:
    if feature_source == "x3":
        return 3
    if feature_source == "ffm_mid":
        return 64
    if feature_source == "rgb":
        return 3
    if feature_source == "d1":
        return 1
    raise ValueError(f"Unsupported incremental_feature_source={feature_source!r}")


def expected_method_contract(method_id: str) -> dict[str, str]:
    method_id = str(method_id).upper()
    table = {
        "N2": {
            "incremental_feature_source": "x3",
            "delta_condition": "feature_only",
            "gate_condition": "feature_d1",
            "raw_feature_encoder_trainable": "true",
        },
        "N3": {
            "incremental_feature_source": "rgb",
            "delta_condition": "feature_only",
            "gate_condition": "feature_d1",
            "raw_feature_encoder_trainable": "not_applicable",
        },
        "N4": {
            "incremental_feature_source": "ffm_mid",
            "delta_condition": "feature_only",
            "gate_condition": "feature_d1",
            "raw_feature_encoder_trainable": "true",
        },
        "N5": {
            "incremental_feature_source": "d1",
            "delta_condition": "d1_only",
            "gate_condition": "d1_only",
            "raw_feature_encoder_trainable": "not_applicable",
        },
        "N7": {
            "incremental_feature_source": "x3",
            "delta_condition": "feature_d1_stopgrad",
            "gate_condition": "feature_d1",
            "raw_feature_encoder_trainable": "true",
        },
    }
    if method_id not in table:
        raise ValueError(f"Unsupported method_id={method_id!r}; expected {INCREMENTAL_METHOD_IDS}")
    return table[method_id]


def validate_incremental_contract(
    *,
    method_id: str,
    incremental_feature_source: str,
    delta_condition: str,
    gate_condition: str,
    raw_feature_encoder_trainable: str,
    allow_frozen_raw_encoder: bool = True,
) -> None:
    method_id = str(method_id).upper()
    expected = expected_method_contract(method_id)
    actual = {
        "incremental_feature_source": str(incremental_feature_source),
        "delta_condition": str(delta_condition),
        "gate_condition": str(gate_condition),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise ValueError(f"{method_id} requires {key}={expected[key]!r}, got {value!r}")
    rft = str(raw_feature_encoder_trainable)
    if rft not in RAW_FEATURE_ENCODER_TRAINABLE:
        raise ValueError(
            f"raw_feature_encoder_trainable must be one of {RAW_FEATURE_ENCODER_TRAINABLE}, got {rft!r}"
        )
    if expected["raw_feature_encoder_trainable"] == "not_applicable":
        if rft != "not_applicable":
            raise ValueError(f"{method_id} requires raw_feature_encoder_trainable='not_applicable', got {rft!r}")
    elif rft == "not_applicable":
        raise ValueError(f"{method_id} requires active raw_feature_encoder_trainable true/false, got not_applicable")
    elif rft == "false" and not allow_frozen_raw_encoder:
        raise ValueError(f"{method_id} formal default requires raw_feature_encoder_trainable='true'")


class C2FrozenIncrementalResidualDAV2(nn.Module):
    """
    Frozen C2 residual-control baseline with a trainable incremental correction branch.

    The wrapper calls C2 exactly once per forward and reads D0, D0_norm, and D1_norm
    from that output. D1_norm is the frozen C2 base prediction.
    """

    def __init__(
        self,
        c2_model: nn.Module,
        *,
        method_id: str,
        incremental_feature_source: str,
        delta_condition: str,
        gate_condition: str,
        raw_feature_encoder_trainable: str,
        residual_alpha: float = 0.5,
        lambda_lp: float = 0.5,
        lowpass_kernel: int = 31,
        sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
        backbone_hw: tuple[int, int] | None = BACKBONE_INPUT_HW,
    ) -> None:
        super().__init__()
        del sensor_hw, backbone_hw
        validate_incremental_contract(
            method_id=method_id,
            incremental_feature_source=incremental_feature_source,
            delta_condition=delta_condition,
            gate_condition=gate_condition,
            raw_feature_encoder_trainable=raw_feature_encoder_trainable,
        )
        if int(lowpass_kernel) <= 0 or int(lowpass_kernel) % 2 == 0:
            raise ValueError(f"lowpass_kernel must be a positive odd integer, got {lowpass_kernel}")

        self.c2_model = c2_model
        self.method_id = str(method_id).upper()
        self.incremental_feature_source = str(incremental_feature_source)
        self.delta_condition = str(delta_condition)
        self.gate_condition = str(gate_condition)
        self.raw_feature_encoder_trainable = str(raw_feature_encoder_trainable)
        self.residual_alpha = float(residual_alpha)
        self.lambda_lp = float(lambda_lp)
        self.lowpass_kernel = int(lowpass_kernel)

        self.ram_core = RamCore3() if self.incremental_feature_source in ("x3", "ffm_mid") else None
        if self.ram_core is not None and self.raw_feature_encoder_trainable == "false":
            self.ram_core.eval()
            for param in self.ram_core.parameters():
                param.requires_grad = False

        self.feature_encoder = FeatureEncoder(_feature_raw_channels(self.incremental_feature_source), 32)
        self.d1_encoder = FeatureEncoder(1, 32)

        delta_in_ch, gate_in_ch = self._head_channels()
        self.delta_head = SmallResidualHead(delta_in_ch, out_kind="delta", alpha=self.residual_alpha)
        self.gate_head = SmallResidualHead(gate_in_ch, out_kind="gate", alpha=self.residual_alpha)

        self.c2_model.eval()
        for param in self.c2_model.parameters():
            param.requires_grad = False

    def _head_channels(self) -> tuple[int, int]:
        if self.method_id in ("N2", "N3", "N4"):
            delta_in_ch = 32
            gate_in_ch = 64
        elif self.method_id == "N5":
            delta_in_ch = 32
            gate_in_ch = 32
        elif self.method_id == "N7":
            delta_in_ch = 64
            gate_in_ch = 64
        else:
            raise AssertionError(f"Unhandled method_id={self.method_id!r}")
        return delta_in_ch, gate_in_ch

    def train(self, mode: bool = True):
        super().train(mode)
        self.c2_model.eval()
        if self.ram_core is not None and self.raw_feature_encoder_trainable == "false":
            self.ram_core.eval()
        return self

    def _extract_feature(self, *, image: torch.Tensor, raw: torch.Tensor | None, d1_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.incremental_feature_source == "rgb":
            return image, None, None
        if self.incremental_feature_source == "d1":
            return d1_norm.unsqueeze(1), None, None
        if raw is None:
            raise KeyError(f"batch['raw'] is required for incremental_feature_source={self.incremental_feature_source!r}")
        if self.ram_core is None:
            raise AssertionError("ram_core was not created for RAW feature source")
        base_rgb = packed_bayer_to_base_rgb(raw)
        x3, ram_features = self.ram_core.forward_with_features(base_rgb)
        ffm_mid = ram_features["ffm_mid"]
        if self.incremental_feature_source == "x3":
            return x3, x3, ffm_mid
        if self.incremental_feature_source == "ffm_mid":
            return ffm_mid, x3, ffm_mid
        raise AssertionError(f"Unhandled feature source: {self.incremental_feature_source}")

    def _lowpass_delta(self, delta_raw: torch.Tensor) -> torch.Tensor:
        if self.lambda_lp == 0.0:
            return delta_raw
        k = self.lowpass_kernel
        low = F.avg_pool2d(delta_raw.unsqueeze(1), kernel_size=k, stride=1, padding=k // 2)[:, 0]
        return delta_raw - float(self.lambda_lp) * low

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = batch["image"]
        raw = batch.get("raw")
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(image.shape[0], image.shape[-2], image.shape[-1], device=image.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        with torch.no_grad():
            c2_out = self.c2_model({"image": image, "valid_mask": valid_mask})
            d0_raw = c2_out["D0"].detach()
            d0_norm = c2_out["D0_norm"].detach()
            d1_norm = c2_out["pred"].detach()

        feature, x3, ffm_mid = self._extract_feature(image=image, raw=raw, d1_norm=d1_norm)
        feature_enc = self.feature_encoder(feature)
        d1_enc = self.d1_encoder(d1_norm.unsqueeze(1))

        if self.delta_condition == "feature_only":
            delta_input = feature_enc
        elif self.delta_condition == "feature_d1_stopgrad":
            delta_input = torch.cat([feature_enc, d1_enc.detach()], dim=1)
        elif self.delta_condition == "d1_only":
            delta_input = d1_enc
        else:
            raise AssertionError(f"Unhandled delta_condition={self.delta_condition!r}")

        if self.gate_condition == "feature_d1":
            gate_input = torch.cat([feature_enc, d1_enc], dim=1)
        elif self.gate_condition == "d1_only":
            gate_input = d1_enc
        else:
            raise AssertionError(f"Unhandled gate_condition={self.gate_condition!r}")

        expected_delta_ch, expected_gate_ch = self._head_channels()
        if delta_input.shape[1] != expected_delta_ch or gate_input.shape[1] != expected_gate_ch:
            raise RuntimeError(
                "Incremental head channel contract mismatch: "
                f"delta={delta_input.shape[1]} expected={expected_delta_ch}, "
                f"gate={gate_input.shape[1]} expected={expected_gate_ch}"
            )

        delta_raw = self.delta_head(delta_input)
        gate_raw = self.gate_head(gate_input)
        delta_effective = self._lowpass_delta(delta_raw)
        gate_delta = gate_raw * delta_effective
        final_norm = d1_norm + gate_delta
        return {
            "pred": final_norm,
            "D0": d0_raw,
            "D0_norm": d0_norm,
            "D1_norm": d1_norm,
            "base_norm": d1_norm,
            "C2_delta": c2_out.get("delta"),
            "C2_gate": c2_out.get("gate"),
            "delta": delta_raw,
            "delta_effective": delta_effective,
            "gate": gate_raw,
            "gate_delta": gate_delta,
            "x3": x3,
            "ffm_mid": ffm_mid,
            "feature": feature,
        }


def build_c2_frozen_incremental_residual_model(
    c2_model: nn.Module,
    *,
    method_id: str,
    incremental_feature_source: str,
    delta_condition: str,
    gate_condition: str,
    raw_feature_encoder_trainable: str,
    residual_alpha: float = 0.5,
    lambda_lp: float = 0.5,
    lowpass_kernel: int = 31,
    sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
    backbone_hw: tuple[int, int] | None = BACKBONE_INPUT_HW,
) -> C2FrozenIncrementalResidualDAV2:
    return C2FrozenIncrementalResidualDAV2(
        c2_model,
        method_id=method_id,
        incremental_feature_source=incremental_feature_source,
        delta_condition=delta_condition,
        gate_condition=gate_condition,
        raw_feature_encoder_trainable=raw_feature_encoder_trainable,
        residual_alpha=residual_alpha,
        lambda_lp=lambda_lp,
        lowpass_kernel=lowpass_kernel,
        sensor_hw=sensor_hw,
        backbone_hw=backbone_hw,
    )
