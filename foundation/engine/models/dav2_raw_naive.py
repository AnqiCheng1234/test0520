from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune_stf.models.spatial_adapter import BACKBONE_INPUT_HW, SENSOR_INPUT_HW, CenterPadCropAdapter
from foundation.engine.transforms import packed_bayer_to_base_rgb


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _register_imagenet_stats(module: nn.Module) -> None:
    module.register_buffer(
        "img_mean",
        torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        persistent=False,
    )
    module.register_buffer(
        "img_std",
        torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        persistent=False,
    )


class PackedBayerInputStem(nn.Module):
    """Minimal 4->3 projection with demosaic-like initialization."""

    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(4, 3, kernel_size=1, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[0, 0, 0, 0] = 1.0
            self.proj.weight[1, 1, 0, 0] = 0.5
            self.proj.weight[1, 2, 0, 0] = 0.5
            self.proj.weight[2, 3, 0, 0] = 1.0
            if self.proj.bias is not None:
                self.proj.bias.zero_()

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.proj(x_raw)


class DAV2RawNaiveDepthModel(nn.Module):
    """
    E1-style minimal RAW model:
        packed Bayer raw4 -> 1x1 projection -> 3ch pseudo-RGB
        -> ImageNet norm -> center pad -> DAv2 -> center crop depth
    """

    def __init__(
        self,
        dav2_model: nn.Module,
        *,
        upsample_mode: str = "bilinear",
        clip_rgb: bool = True,
        freeze_backbone: bool = False,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ) -> None:
        super().__init__()
        self.input_stem = PackedBayerInputStem()
        self.dav2 = dav2_model
        self.clip_rgb = bool(clip_rgb)
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        _register_imagenet_stats(self)

        if freeze_backbone:
            self.dav2.pretrained.requires_grad_(False)

    def forward_features(self, x_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        x_rgb = self.input_stem(x_raw)
        if self.clip_rgb:
            x_rgb = x_rgb.clamp(0.0, 1.0)
        x_norm = (x_rgb - self.img_mean.to(dtype=x_rgb.dtype)) / self.img_std.to(dtype=x_rgb.dtype)
        x_norm_padded = self.spatial_adapter.pad_rgb(x_norm)
        depth = self.dav2(x_norm_padded)
        depth = self.spatial_adapter.crop_depth(depth)
        return {
            "raw": x_raw,
            "rgb": x_rgb,
            "rgb_norm": x_norm,
            "rgb_norm_padded": x_norm_padded,
            "depth": depth,
        }

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x_raw)["depth"]

    def stem_as_base_rgb(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Debug helper returning the fixed packed-Bayer-to-RGB initialization output."""
        return packed_bayer_to_base_rgb(x_raw)

    def load_base_dav2_state_dict(self, state_dict: dict[str, torch.Tensor]):
        return self.dav2.load_state_dict(state_dict, strict=True)

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = False):
        return self.load_state_dict(state_dict, strict=strict)


def build_dav2_raw_naive_depth_model(
    dav2_model: nn.Module,
    *,
    upsample_mode: str = "bilinear",
    clip_rgb: bool = True,
    freeze_backbone: bool = False,
    sensor_hw=SENSOR_INPUT_HW,
    backbone_hw=BACKBONE_INPUT_HW,
) -> DAV2RawNaiveDepthModel:
    return DAV2RawNaiveDepthModel(
        dav2_model,
        upsample_mode=upsample_mode,
        clip_rgb=clip_rgb,
        freeze_backbone=freeze_backbone,
        sensor_hw=sensor_hw,
        backbone_hw=backbone_hw,
    )
