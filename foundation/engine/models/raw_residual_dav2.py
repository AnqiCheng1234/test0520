from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune_stf.models.raw_ram import RamCore3
from finetune_stf.models.spatial_adapter import BACKBONE_INPUT_HW, SENSOR_INPUT_HW, CenterPadCropAdapter
from finetune_stf.util.loss import robust_normalize_target_per_sample
from foundation.engine.transforms import packed_bayer_to_base_rgb


RESIDUAL_FEATURE_SOURCES = ("ffm_mid", "x3", "x3_ffm_mid")
RESIDUAL_HEAD_D0_MODES = ("concat", "none")


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


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            _gn(out_ch),
            nn.GELU(),
        )
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            _gn(out_ch),
            nn.GELU(),
        )
        self.res = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.res(self.fuse(torch.cat([x, skip], dim=1)))


class ResidualGateHead(nn.Module):
    def __init__(self, in_ch: int, *, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            _gn(64),
            nn.GELU(),
        )
        self.enc0 = ResidualBlock(64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.delta_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        delta_last = self.delta_head[-1]
        gate_last = self.gate_head[-1]
        if not isinstance(delta_last, nn.Conv2d) or not isinstance(gate_last, nn.Conv2d):
            raise TypeError("Unexpected residual head layout.")
        nn.init.zeros_(delta_last.weight)
        nn.init.zeros_(delta_last.bias)
        nn.init.zeros_(gate_last.weight)
        nn.init.constant_(gate_last.bias, -4.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.enc0(self.stem(features))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x = self.up1(x2, x1)
        x = self.up2(x, x0)
        delta = self.alpha * torch.tanh(self.delta_head(x)[:, 0])
        gate = torch.sigmoid(self.gate_head(x)[:, 0])
        return delta, gate


def _raw_feature_channels(feature_source: str) -> int:
    if feature_source == "ffm_mid":
        return 64
    if feature_source == "x3":
        return 3
    if feature_source == "x3_ffm_mid":
        return 67
    raise ValueError(f"Unsupported residual_feature_source={feature_source!r}; expected {RESIDUAL_FEATURE_SOURCES}")


def _feature_channels(feature_source: str, residual_head_d0_mode: str) -> int:
    if residual_head_d0_mode not in RESIDUAL_HEAD_D0_MODES:
        raise ValueError(
            f"Unsupported residual_head_d0_mode={residual_head_d0_mode!r}; "
            f"expected {RESIDUAL_HEAD_D0_MODES}"
        )
    d0_channels = 1 if residual_head_d0_mode == "concat" else 0
    return d0_channels + _raw_feature_channels(feature_source)


class RawResidualDAV2(nn.Module):
    """
    Frozen RGB DAv2 baseline refined by a trainable RAW RamCore3 residual path.

    Forward input is a dict with:
        image: ImageNet-normalized RGB tensor [B,3,H,W]
        raw: packed Bayer tensor [B,4,H,W]
        valid_mask: optional bool mask [B,H,W]
    """

    def __init__(
        self,
        dav2_model: nn.Module,
        *,
        residual_feature_source: str = "ffm_mid",
        residual_head_d0_mode: str = "concat",
        residual_alpha: float = 0.5,
        d0_sign: int = 1,
        sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
        backbone_hw: tuple[int, int] = BACKBONE_INPUT_HW,
        min_valid_pixels: int = 128,
    ) -> None:
        super().__init__()
        residual_feature_source = str(residual_feature_source)
        if residual_feature_source not in RESIDUAL_FEATURE_SOURCES:
            raise ValueError(
                f"Unsupported residual_feature_source={residual_feature_source!r}; "
                f"expected one of {RESIDUAL_FEATURE_SOURCES}"
            )
        residual_head_d0_mode = str(residual_head_d0_mode)
        if residual_head_d0_mode not in RESIDUAL_HEAD_D0_MODES:
            raise ValueError(
                f"Unsupported residual_head_d0_mode={residual_head_d0_mode!r}; "
                f"expected one of {RESIDUAL_HEAD_D0_MODES}"
            )
        if int(d0_sign) not in (-1, 1):
            raise ValueError(f"d0_sign must be 1 or -1, got {d0_sign}")

        self.dav2 = dav2_model
        self.ram_core = RamCore3()
        self.residual_feature_source = residual_feature_source
        self.residual_head_d0_mode = residual_head_d0_mode
        self.residual_alpha = float(residual_alpha)
        self.d0_sign = int(d0_sign)
        self.min_valid_pixels = int(min_valid_pixels)
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        self.residual_head = ResidualGateHead(
            _feature_channels(residual_feature_source, residual_head_d0_mode),
            alpha=residual_alpha,
        )

        self.dav2.eval()
        for param in self.dav2.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.dav2.eval()
        return self

    def _rgb_dav2(self, image_rgb_norm: torch.Tensor) -> torch.Tensor:
        image_padded = self.spatial_adapter.pad_rgb(image_rgb_norm)
        depth = self.dav2(image_padded)
        return self.spatial_adapter.crop_depth(depth)

    def _normalize_d0(self, d0: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        d0_signed = float(self.d0_sign) * d0
        d0_norm, _ = robust_normalize_target_per_sample(
            d0_signed,
            valid_mask,
            min_valid_pixels=self.min_valid_pixels,
        )
        return d0_norm.detach()

    def _head_input(
        self,
        *,
        d0_norm: torch.Tensor,
        x3: torch.Tensor,
        ffm_mid: torch.Tensor,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.residual_head_d0_mode == "concat":
            features.append(d0_norm.unsqueeze(1))
        elif self.residual_head_d0_mode != "none":
            raise AssertionError(f"Unhandled residual head D0 mode: {self.residual_head_d0_mode}")

        if self.residual_feature_source == "ffm_mid":
            features.append(ffm_mid)
        elif self.residual_feature_source == "x3":
            features.append(x3)
        elif self.residual_feature_source == "x3_ffm_mid":
            features.extend([x3, ffm_mid])
        else:
            raise AssertionError(f"Unhandled residual feature source: {self.residual_feature_source}")
        return torch.cat(features, dim=1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        feature_override: dict[str, torch.Tensor] | None = None,
        feature_ablation_mode: str = "true",
    ) -> dict[str, torch.Tensor]:
        image = batch["image"]
        raw = batch["raw"]
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(raw.shape[0], raw.shape[-2], raw.shape[-1], device=raw.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        with torch.no_grad():
            d0 = self._rgb_dav2(image).detach()
        d0_norm = self._normalize_d0(d0.float(), valid_mask)

        base_rgb = packed_bayer_to_base_rgb(raw)
        x3, ram_features = self.ram_core.forward_with_features(base_rgb)
        ffm_mid = ram_features["ffm_mid"]
        if feature_override:
            if "x3" in feature_override:
                x3 = feature_override["x3"].to(device=x3.device, dtype=x3.dtype)
            if "ffm_mid" in feature_override:
                ffm_mid = feature_override["ffm_mid"].to(device=ffm_mid.device, dtype=ffm_mid.dtype)
        head_input = self._head_input(d0_norm=d0_norm, x3=x3, ffm_mid=ffm_mid)
        delta, gate = self.residual_head(head_input)
        pred = d0_norm + gate * delta
        return {
            "pred": pred,
            "D0": d0,
            "D0_norm": d0_norm,
            "delta": delta,
            "gate": gate,
            "x3": x3,
            "ffm_mid": ffm_mid,
            "ram_out": x3,
            "feature_ablation_mode": feature_ablation_mode,
        }

    def forward_with_feature_override(
        self,
        batch: dict[str, torch.Tensor],
        *,
        feature_override: dict[str, torch.Tensor] | None = None,
        feature_ablation_mode: str = "true",
    ) -> dict[str, torch.Tensor]:
        return self.forward(
            batch,
            feature_override=feature_override,
            feature_ablation_mode=feature_ablation_mode,
        )

    def load_base_dav2_state_dict(self, state_dict: dict[str, torch.Tensor]):
        from finetune_stf.models.lora_bridge import _remap_state_dict_for_lora_modules

        compatible = _remap_state_dict_for_lora_modules(self.dav2, state_dict)
        status = self.dav2.load_state_dict(compatible, strict=False)
        missing = [
            key
            for key in status.missing_keys
            if ".lora_A." not in key and ".lora_B." not in key
        ]
        if missing or status.unexpected_keys:
            raise RuntimeError(
                "Failed to load compatible DAv2 base weights for RawResidualDAV2: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = False):
        return self.load_state_dict(state_dict, strict=strict)


def build_raw_residual_dav2_model(
    dav2_model: nn.Module,
    *,
    residual_feature_source: str = "ffm_mid",
    residual_head_d0_mode: str = "concat",
    residual_alpha: float = 0.5,
    d0_sign: int = 1,
    sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
    backbone_hw: tuple[int, int] = BACKBONE_INPUT_HW,
    min_valid_pixels: int = 128,
) -> RawResidualDAV2:
    return RawResidualDAV2(
        dav2_model,
        residual_feature_source=residual_feature_source,
        residual_head_d0_mode=residual_head_d0_mode,
        residual_alpha=residual_alpha,
        d0_sign=d0_sign,
        sensor_hw=sensor_hw,
        backbone_hw=backbone_hw,
        min_valid_pixels=min_valid_pixels,
    )
