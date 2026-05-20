from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SENSOR_INPUT_HW = (512, 960)
BACKBONE_INPUT_HW = (518, 966)
PATCH_SIZE = 14


def _pair(hw: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(hw) != 2:
        raise ValueError(f"Expected (height, width), got {hw}")
    return int(hw[0]), int(hw[1])


def infer_backbone_hw(
    sensor_hw: tuple[int, int] | list[int],
    *,
    patch_size: int = PATCH_SIZE,
) -> tuple[int, int]:
    sensor_h, sensor_w = _pair(sensor_hw)
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    backbone_h = ((sensor_h + patch_size - 1) // patch_size) * patch_size
    backbone_w = ((sensor_w + patch_size - 1) // patch_size) * patch_size
    return backbone_h, backbone_w


def center_pad_to_hw(
    tensor: torch.Tensor,
    target_hw: tuple[int, int] | list[int] = BACKBONE_INPUT_HW,
    *,
    value: float = 0.0,
) -> torch.Tensor:
    target_h, target_w = _pair(target_hw)
    height, width = tensor.shape[-2:]
    if height > target_h or width > target_w:
        raise ValueError(
            f"Cannot center-pad tensor with spatial size {(height, width)} to smaller target {(target_h, target_w)}"
        )

    pad_h = target_h - height
    pad_w = target_w - width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=value)


def center_crop_to_hw(
    tensor: torch.Tensor,
    target_hw: tuple[int, int] | list[int] = SENSOR_INPUT_HW,
) -> torch.Tensor:
    target_h, target_w = _pair(target_hw)
    height, width = tensor.shape[-2:]
    if height < target_h or width < target_w:
        raise ValueError(
            f"Cannot center-crop tensor with spatial size {(height, width)} to larger target {(target_h, target_w)}"
        )

    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return tensor[..., top : top + target_h, left : left + target_w]


class CenterPadCropAdapter(nn.Module):
    def __init__(
        self,
        *,
        sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
        backbone_hw: tuple[int, int] | None = BACKBONE_INPUT_HW,
        patch_size: int = PATCH_SIZE,
        allow_dynamic_hw: bool = True,
    ) -> None:
        super().__init__()
        self.sensor_hw = _pair(sensor_hw)
        self.backbone_hw = infer_backbone_hw(self.sensor_hw, patch_size=patch_size) if backbone_hw is None else _pair(backbone_hw)
        self.patch_size = int(patch_size)
        self.allow_dynamic_hw = bool(allow_dynamic_hw)
        self._last_sensor_hw = self.sensor_hw
        self._last_backbone_hw = self.backbone_hw

    def pad_rgb(self, x_rgb_or_norm: torch.Tensor) -> torch.Tensor:
        input_hw = tuple(int(v) for v in x_rgb_or_norm.shape[-2:])
        if input_hw == self.sensor_hw:
            target_hw = self.backbone_hw
        elif self.allow_dynamic_hw:
            target_hw = infer_backbone_hw(input_hw, patch_size=self.patch_size)
        else:
            raise ValueError(
                f"Expected sensor-space tensor with spatial size {self.sensor_hw}, got {input_hw}"
            )
        self._last_sensor_hw = input_hw
        self._last_backbone_hw = target_hw
        return center_pad_to_hw(x_rgb_or_norm, target_hw, value=0.0)

    def crop_depth(self, depth: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if depth.ndim == 3:
            depth = depth[:, None]
            squeeze = True
        elif depth.ndim != 4:
            raise ValueError(f"Expected depth tensor with shape (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")

        target_hw = self._last_sensor_hw or self.sensor_hw
        cropped = center_crop_to_hw(depth, target_hw)
        return cropped[:, 0] if squeeze else cropped


class DAV2PaddedRGBDepthModel(nn.Module):
    def __init__(
        self,
        dav2_model: nn.Module,
        *,
        sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
        backbone_hw: tuple[int, int] = BACKBONE_INPUT_HW,
    ) -> None:
        super().__init__()
        self.dav2 = dav2_model
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        x_padded = self.spatial_adapter.pad_rgb(x_rgb_norm)
        depth = self.dav2(x_padded)
        return self.spatial_adapter.crop_depth(depth)

    def load_base_dav2_state_dict(self, state_dict: dict[str, torch.Tensor]):
        return self.dav2.load_state_dict(state_dict, strict=True)

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = False):
        return self.load_state_dict(state_dict, strict=strict)


def build_dav2_padded_rgb_depth_model(
    dav2_model: nn.Module,
    *,
    sensor_hw: tuple[int, int] = SENSOR_INPUT_HW,
    backbone_hw: tuple[int, int] = BACKBONE_INPUT_HW,
) -> DAV2PaddedRGBDepthModel:
    return DAV2PaddedRGBDepthModel(
        dav2_model,
        sensor_hw=sensor_hw,
        backbone_hw=backbone_hw,
    )
