"""
E2 RAW-RAM: input-level RAW adapter following the RAM (RAW Adapter Module)
paradigm -- shared RPEncoder, per-ISP RPDecoder heads, parallel ISP
branches, and Feature Fusion Module (FFM).

Reference structure: Beyond RGB paper / RAM module.
Adapted for 4-channel packed Bayer input -> DAv2 3-channel interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune_stf.models.spatial_adapter import BACKBONE_INPUT_HW, SENSOR_INPUT_HW, CenterPadCropAdapter


# ---------------------------------------------------------------------------
# RPEncoder: shared feature encoder that predicts a global parameter vector
# ---------------------------------------------------------------------------

class RPEncoder(nn.Module):
    """Encode a downsampled RAW image into a shared feature vector z (B, 128)."""

    def __init__(self, in_channels=4, img_size=256):
        super().__init__()
        self.img_size = img_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        # x: (B, 4, H, W) -- full resolution
        # Downsample to img_size x img_size for parameter prediction only
        x_down = F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        return self.features(x_down)  # (B, 128)


# ---------------------------------------------------------------------------
# RPDecoder: per-ISP parameter head
# ---------------------------------------------------------------------------

class RPDecoder(nn.Module):
    def __init__(self, in_features=128, out_channels=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, out_channels),
        )

    def forward(self, z):
        return self.head(z)


# ---------------------------------------------------------------------------
# ISP function branches
# ---------------------------------------------------------------------------

class WBBranch(nn.Module):
    """White balance: per-channel multiplicative gain."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 4)

    def forward(self, x_raw, z):
        # z: (B, 128)  x_raw: (B, 4, H, W)
        gain = self.decoder(z).view(-1, 4, 1, 1)  # (B, 4, 1, 1)
        return x_raw * gain


class CCMBranch(nn.Module):
    """Color correction matrix: 4x4 linear mixing in packed Bayer space."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 16)

    def forward(self, x_raw, z):
        ccm = self.decoder(z).view(-1, 4, 4)  # (B, 4, 4)
        # einsum: for each pixel, apply 4x4 matrix to channel dim
        return torch.einsum("bchw,bdc->bdhw", x_raw, ccm)


class GammaBranch(nn.Module):
    """Gamma correction: element-wise power with sigmoid-bounded exponent."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 1)

    def forward(self, x_raw, z):
        gamma = torch.sigmoid(self.decoder(z)).view(-1, 1, 1, 1)  # (0, 1)
        # Clamp input to avoid pow on zero/negative
        return x_raw.clamp(min=1e-6) ** gamma


class BrightnessBranch(nn.Module):
    """Brightness offset: additive shift with sigmoid bound."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 1)

    def forward(self, x_raw, z):
        brightness = torch.sigmoid(self.decoder(z)).view(-1, 1, 1, 1)  # (0, 1)
        return x_raw + brightness


# ---------------------------------------------------------------------------
# Feature Fusion Module (FFM) -- BN_HG style
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FFM(nn.Module):
    """Feature Fusion Module: fuse concatenated ISP branch outputs."""

    def __init__(self, in_ch=16, out_ch=4):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, 16)
        self.conv2 = ConvBNReLU(16, 64)
        self.conv3 = ConvBNReLU(64, 16)
        self.out_conv = nn.Conv2d(16, out_ch, kernel_size=1, padding=0)
        self.out_bn = nn.BatchNorm2d(out_ch, affine=True)

    def forward_with_features(self, x):
        x = self.conv1(x)
        ffm_mid = self.conv2(x)
        x = self.conv3(ffm_mid)
        x = self.out_bn(self.out_conv(x))
        return x, {"ffm_mid": ffm_mid}

    def forward(self, x):
        x, _ = self.forward_with_features(x)
        return x


# ---------------------------------------------------------------------------
# RawRamCore: RPEncoder + ISP branches + FFM
# ---------------------------------------------------------------------------

FUNCTION_ORDER = ["wb", "ccm", "gamma", "brightness"]
RAW_RAM_LORA_INPUT_TYPES = ("raw_ram_lora",)
RAW_RAM_INPUT_TYPES = ("raw_ram", *RAW_RAM_LORA_INPUT_TYPES, "raw_ram_residual")
RAW_RAM_BRIDGE_FEATURE_CHANNELS = {
    "x_cat": 16,
    "ffm_mid": 64,
    "x4": 4,
}
RAW_RAM_RGB_LORA_INPUT_TYPES = ("raw_ram_rgb_lora",)
RAW_RAM_RGB_INPUT_TYPES = ("raw_ram_rgb", *RAW_RAM_RGB_LORA_INPUT_TYPES)
RAW_RGB16_RAM3_INPUT_TYPES = (
    "raw_rgb16_ram3",
    "lod_true_raw_dark_rgb16",
    "lod_true_raw_normal_rgb16",
    "lod_true_raw_dark_normal_pair_rgb16",
)
RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS = {
    "x_cat": 12,
    "ffm_mid": 64,
    "x3": 3,
}
PHASE1B_TANH_ALPHA = 2.5
RAW_RAM_RGB_TAIL_CHOICES = ("identity", "tanh2p5")
RAW_RAM_LOCAL_RESIDUAL_CHOICES = ("none", "noiseaware_v1")
RAW_RAM_LOCAL_GATE_MODE_CHOICES = ("n_a", "scalar", "channel")


def phase1b_tanh_tail_squash(x, alpha=PHASE1B_TANH_ALPHA):
    return float(alpha) * torch.tanh(x / float(alpha))


class RawRamCore(nn.Module):
    """
    RAM core: shared encoder predicts ISP parameters, four parallel ISP
    branches each process the raw input independently, outputs are
    concatenated and fused through FFM.

    Input:  (B, 4, H, W) packed Bayer in [0, 1]
    Output: (B, 4, H, W) fused representation
    """

    def __init__(self):
        super().__init__()
        self.encoder = RPEncoder(in_channels=4, img_size=256)
        self.branches = nn.ModuleDict({
            "wb": WBBranch(),
            "ccm": CCMBranch(),
            "gamma": GammaBranch(),
            "brightness": BrightnessBranch(),
        })
        self.ffm = FFM(in_ch=16, out_ch=4)
        self.function_order = FUNCTION_ORDER

    def forward(self, x_raw):
        x4, _ = self.forward_with_features(x_raw)
        return x4

    def forward_with_features(self, x_raw):
        z = self.encoder(x_raw)  # (B, 128)

        branch_outputs = []
        for name in self.function_order:
            branch_outputs.append(self.branches[name](x_raw, z))

        x_cat = torch.cat(branch_outputs, dim=1)  # (B, 16, H, W)
        x4, ffm_features = self.ffm.forward_with_features(x_cat)
        feature_dict = {"x_cat": x_cat, **ffm_features, "x4": x4}
        return x4, feature_dict


# ---------------------------------------------------------------------------
# 3-channel RAM components (official Beyond-RGB style)
# ---------------------------------------------------------------------------

class WBBranch3(nn.Module):
    """White balance: per-channel multiplicative gain in 3-ch space."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 3)

    def forward(self, x_rgb, z):
        gain = self.decoder(z).view(-1, 3, 1, 1)
        return x_rgb * gain


class CCMBranch3(nn.Module):
    """Color correction matrix: 3x3 linear mixing in RGB-like space."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 9)

    def forward(self, x_rgb, z):
        ccm = self.decoder(z).view(-1, 3, 3)
        return torch.einsum("bchw,bdc->bdhw", x_rgb, ccm)


class GammaBranch3(nn.Module):
    """Gamma correction: element-wise power with sigmoid-bounded exponent."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 1)

    def forward(self, x_rgb, z):
        gamma = torch.sigmoid(self.decoder(z)).view(-1, 1, 1, 1)
        return x_rgb.clamp(min=1e-6) ** gamma


class BrightnessBranch3(nn.Module):
    """Brightness offset: additive shift with sigmoid bound."""

    def __init__(self):
        super().__init__()
        self.decoder = RPDecoder(128, 1)

    def forward(self, x_rgb, z):
        brightness = torch.sigmoid(self.decoder(z)).view(-1, 1, 1, 1)
        return x_rgb + brightness


class FFM3(nn.Module):
    """
    Feature Fusion Module for 3-channel RAM variant.

    Mirrors BN_HG (12 -> 16 -> 64 -> 16 -> 3), with output BN factored out
    into RamCore3.norm_layer to match the official implementation.
    """

    def __init__(self, in_ch=12, out_ch=3):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, 16)
        self.conv2 = ConvBNReLU(16, 64)
        self.conv3 = ConvBNReLU(64, 16)
        self.out_conv = nn.Conv2d(16, out_ch, kernel_size=1, padding=0)

    def forward_with_features(self, x):
        x = self.conv1(x)
        ffm_mid = self.conv2(x)
        x = self.conv3(ffm_mid)
        x = self.out_conv(x)
        return x, {"ffm_mid": ffm_mid}

    def forward(self, x):
        x, _ = self.forward_with_features(x)
        return x


class RamCore3(nn.Module):
    """
    RAM core in 3-channel space (official Beyond-RGB front-end style).

    Input:  (B, 3, H, W)
    Output: (B, 3, H, W)
    """

    def __init__(
        self,
        *,
        local_residual="none",
        local_hidden_ch=16,
        local_residual_scale=0.1,
        local_gate_init=0.03,
        local_gate_mode="channel",
    ):
        super().__init__()
        local_residual = str(local_residual)
        local_gate_mode = str(local_gate_mode)
        if local_residual not in RAW_RAM_LOCAL_RESIDUAL_CHOICES:
            raise ValueError(
                f"Unsupported local_residual={local_residual!r}; "
                f"expected one of {RAW_RAM_LOCAL_RESIDUAL_CHOICES}"
            )
        if local_residual == "none":
            local_gate_mode = "n_a"
        if local_gate_mode not in RAW_RAM_LOCAL_GATE_MODE_CHOICES:
            raise ValueError(
                f"Unsupported local_gate_mode={local_gate_mode!r}; "
                f"expected one of {RAW_RAM_LOCAL_GATE_MODE_CHOICES}"
            )
        if local_residual != "none" and local_gate_mode == "n_a":
            raise ValueError("local_gate_mode must be scalar or channel when local residual is enabled")
        self.encoder = RPEncoder(in_channels=3, img_size=256)
        self.branches = nn.ModuleDict({
            "wb": WBBranch3(),
            "ccm": CCMBranch3(),
            "gamma": GammaBranch3(),
            "brightness": BrightnessBranch3(),
        })
        self.ffm = FFM3(in_ch=12, out_ch=3)
        self.norm_layer = nn.BatchNorm2d(3, affine=True)
        self.function_order = FUNCTION_ORDER
        self.local_residual = local_residual
        self.local_residual_scale = float(local_residual_scale) if local_residual != "none" else 0.0
        self.local_gate_mode = local_gate_mode
        if local_residual == "noiseaware_v1":
            self.local_residual_branch = LocalDenoiseBranch(
                in_ch=6,
                hidden_ch=int(local_hidden_ch),
                out_ch=3,
            )
            gate_shape = (1, 3, 1, 1) if local_gate_mode == "channel" else (1, 1, 1, 1)
            gate_target = float(local_gate_init)
            if not (0.0 < gate_target < 1.0):
                raise ValueError("local_gate_init is tanh(g)'s target and must be in (0, 1)")
            gate_value = torch.atanh(torch.tensor(gate_target, dtype=torch.float32))
            self.local_residual_gate = nn.Parameter(torch.full(gate_shape, float(gate_value.item())))
        else:
            self.local_residual_branch = None
            self.local_residual_gate = None

    def forward(self, x_rgb):
        x3, _ = self.forward_with_features(x_rgb)
        return x3

    def forward_with_features(self, x_rgb):
        z = self.encoder(x_rgb)

        branch_outputs = []
        for name in self.function_order:
            branch_outputs.append(self.branches[name](x_rgb, z))

        x_cat = torch.cat(branch_outputs, dim=1)  # (B, 12, H, W)
        x3_pre_bn, ffm_features = self.ffm.forward_with_features(x_cat)
        x3_ram = self.norm_layer(x3_pre_bn)
        x3 = x3_ram
        feature_dict = {"x_cat": x_cat, **ffm_features, "x3_ram": x3_ram}
        if self.local_residual_branch is not None:
            delta_x3 = self.local_residual_branch(torch.cat([x_rgb, x3_ram], dim=1))
            gate = torch.tanh(self.local_residual_gate)
            x3 = x3_ram + self.local_residual_scale * gate * torch.tanh(delta_x3)
            feature_dict.update(
                {
                    "local_delta_x3": delta_x3,
                    "local_gate_tanh": gate,
                    "x3_out": x3,
                }
            )
        feature_dict["x3"] = x3
        return x3, feature_dict


class LocalDenoiseBranch(nn.Module):
    """Light local residual branch for noise-aware RamCore3 correction."""

    def __init__(self, *, in_ch=6, hidden_ch=16, out_ch=3):
        super().__init__()
        hidden_ch = int(hidden_ch)
        if hidden_ch <= 0:
            raise ValueError("LocalDenoiseBranch hidden_ch must be positive")
        groups = 4 if hidden_ch % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, groups=hidden_ch, bias=False),
            nn.GroupNorm(groups, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.net(x)


def _group_count(channels):
    channels = int(channels)
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class PostRamCleanupResBlock(nn.Module):
    """Small residual block used by PostRamCleanupAdapter."""

    def __init__(self, channels, *, norm="groupnorm"):
        super().__init__()
        channels = int(channels)
        if norm != "groupnorm":
            raise ValueError(f"Unsupported PostRamCleanupResBlock norm={norm!r}")
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=False)


class PostRamCleanupAdapter(nn.Module):
    """
    Residual cleanup adapter in RamCore3 post-BN native domain.

    The module returns x_ram + scale * net(x_ram). With zero-init last conv,
    the initial mapping is exactly identity.
    """

    def __init__(self, *, channels=32, blocks=4, scale=0.1, norm="groupnorm", zero_init=True):
        super().__init__()
        channels = int(channels)
        blocks = int(blocks)
        if channels <= 0:
            raise ValueError("PostRamCleanupAdapter channels must be positive")
        if blocks <= 0:
            raise ValueError("PostRamCleanupAdapter blocks must be positive")
        if norm != "groupnorm":
            raise ValueError(f"Unsupported PostRamCleanupAdapter norm={norm!r}")
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("PostRamCleanupAdapter scale must be positive")
        groups = _group_count(channels)
        layers = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        ]
        layers.extend(PostRamCleanupResBlock(channels, norm=norm) for _ in range(blocks))
        last = nn.Conv2d(channels, 3, kernel_size=3, padding=1, bias=True)
        if bool(zero_init):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def delta(self, x):
        return self.scale * self.net(x)

    def forward(self, x):
        return x + self.delta(x)


# ---------------------------------------------------------------------------
# RGB interface heads: 4ch RAM output -> 3ch DAv2 input
# ---------------------------------------------------------------------------

RGB_INTERFACE_HEAD_MODE_CHOICES = (
    "residual_tanh",
    "linear_clamp",
    "residual_linear",
    "tanh01",
    "sigmoid",
)


def _init_rggb_projection(conv):
    """Initialize a 4ch -> 3ch projection as [R, (Gr+Gb)/2, B]."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[0, 0, 0, 0] = 1.0
        conv.weight[1, 1, 0, 0] = 0.5
        conv.weight[1, 2, 0, 0] = 0.5
        conv.weight[2, 3, 0, 0] = 1.0


class RGBInterfaceHead(nn.Module):
    """
    Map 4-channel RAM features to a 3-channel image for DAv2.

    Modes:
        residual_tanh: clamp(base_rgb + scale * tanh(conv(x4)), 0, 1)
        linear_clamp: clamp(conv(x4), 0, 1), conv initialized as RGGB merge
        residual_linear: clamp(base_rgb + scale * conv(x4), 0, 1)
        tanh01: 0.5 + 0.5 * tanh(conv(x4))
        sigmoid: legacy sigmoid(conv(x4))
    """

    def __init__(self, mode="residual_tanh", residual_scale=0.1):
        super().__init__()
        mode = str(mode)
        if mode not in RGB_INTERFACE_HEAD_MODE_CHOICES:
            raise ValueError(
                f"Unsupported rgb interface mode: {mode}. "
                f"Expected one of {RGB_INTERFACE_HEAD_MODE_CHOICES}"
            )
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.conv = nn.Conv2d(4, 3, kernel_size=1, bias=True)

        if self.mode == "linear_clamp":
            _init_rggb_projection(self.conv)
        elif self.mode in {"residual_tanh", "residual_linear", "tanh01"}:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x4, *, x_raw=None):
        if self.mode == "sigmoid":
            return torch.sigmoid(self.conv(x4))
        if self.mode == "linear_clamp":
            return torch.clamp(self.conv(x4), min=0.0, max=1.0)
        if self.mode == "tanh01":
            return 0.5 + 0.5 * torch.tanh(self.conv(x4))

        if x_raw is None:
            raise ValueError(f"rgb interface mode {self.mode!r} requires x_raw for base RGB")
        base_rgb = packed_bayer_to_base_rgb(x_raw)
        delta_rgb = self.conv(x4)
        if self.mode == "residual_tanh":
            delta_rgb = torch.tanh(delta_rgb)
        return torch.clamp(
            base_rgb + self.residual_scale * delta_rgb,
            min=0.0,
            max=1.0,
        )


class ResidualRGBHead(nn.Module):
    """
    Lightweight residual RGB head used by E2.1:
        RAM core -> delta_rgb
        adapted_rgb = clamp(base_rgb + 0.1 * tanh(delta_rgb), 0, 1)
    """

    def __init__(self, in_ch=4, hidden_ch=16, out_ch=3):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_ch, hidden_ch),
            ConvBNReLU(hidden_ch, hidden_ch),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.head(x)


# ---------------------------------------------------------------------------
# RawRamDepthModel: full pipeline wrapping DAv2
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def packed_bayer_to_base_rgb(x_raw):
    """
    Convert packed Bayer [R, Gr, Gb, B] to a simple 3-channel pseudo-RGB base:
        [R, (Gr + Gb)/2, B]
    """
    r = x_raw[:, 0:1]
    g = 0.5 * (x_raw[:, 1:2] + x_raw[:, 2:3])
    b = x_raw[:, 3:4]
    return torch.cat([r, g, b], dim=1)


def _register_imagenet_stats(module):
    module.register_buffer(
        "img_mean",
        torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        persistent=False,
    )
    module.register_buffer(
        "img_std",
        torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        persistent=False,
    )


class RawRamDepthModel(nn.Module):
    """
    End-to-end model:
        raw4 -> RawRamCore -> RGBInterfaceHead -> ImageNet norm
        -> center pad -> frozen DAv2 -> center crop depth
    """

    def __init__(
        self,
        dav2_model,
        *,
        rgb_interface_mode="residual_tanh",
        rgb_residual_scale=0.1,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        """
        Args:
            dav2_model: a DepthAnythingV2 instance (backbone will be frozen).
        """
        super().__init__()
        self.front_end = "raw_ram4"
        self.ram_core_type = "RawRamCore"
        self.imagenet_norm_enabled = True
        self.uses_base_rgb = True
        self.uses_clamp = rgb_interface_mode in {"residual_tanh", "residual_linear", "linear_clamp"}
        self.raw_ram_rgb_tail = "n_a"
        self.ram_core = RawRamCore()
        self.rgb_head = RGBInterfaceHead(
            mode=rgb_interface_mode,
            residual_scale=rgb_residual_scale,
        )

        # DAv2 backbone + DPT decoder
        self.dav2 = dav2_model

        # Register ImageNet stats as buffers (auto-moved to device)
        _register_imagenet_stats(self)
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)

    def forward(self, x_raw):
        """
        Args:
            x_raw: (B, 4, H, W) packed Bayer in [0, 1], sensor-space native 512x960
        Returns:
            depth: (B, H, W) disparity-like output
        """
        x4 = self.ram_core(x_raw)           # (B, 4, H, W)
        x_rgb = self.rgb_head(x4, x_raw=x_raw)  # (B, 3, H, W) in [0, 1]
        x_norm = (x_rgb - self.img_mean) / self.img_std  # ImageNet normalization
        x_norm = self.spatial_adapter.pad_rgb(x_norm)
        depth = self.dav2(x_norm)
        return self.spatial_adapter.crop_depth(depth)

    def load_base_dav2_state_dict(self, state_dict):
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
                "Failed to load compatible DAv2 base weights for raw_ram4 model: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status


class RawToBaseRgbRam3DepthModel(nn.Module):
    """
    3-channel RAM front-end:
        raw4 -> packed_bayer_to_base_rgb -> RamCore3 BN output
        -> center pad -> frozen DAv2 -> center crop depth
    """

    def __init__(
        self,
        dav2_model,
        *,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
        raw_ram_rgb_tail="tanh2p5",
        raw_ram_local_residual="none",
        raw_ram_local_hidden_ch=16,
        raw_ram_local_residual_scale=0.1,
        raw_ram_local_gate_init=0.03,
        raw_ram_local_gate_mode="channel",
        post_ram_cleanup="none",
        post_ram_cleanup_channels=32,
        post_ram_cleanup_blocks=4,
        post_ram_cleanup_scale=0.1,
        post_ram_cleanup_norm="groupnorm",
        post_ram_cleanup_zero_init=True,
        post_ram_operation_position="n_a",
    ):
        super().__init__()
        raw_ram_rgb_tail = str(raw_ram_rgb_tail)
        if raw_ram_rgb_tail not in RAW_RAM_RGB_TAIL_CHOICES:
            raise ValueError(
                f"Unsupported raw_ram_rgb_tail={raw_ram_rgb_tail!r}; "
                f"expected one of {RAW_RAM_RGB_TAIL_CHOICES}"
            )
        self.front_end = "raw_to_base_rgb_ram3"
        self.ram_core_type = "RamCore3"
        self.imagenet_norm_enabled = False
        self.uses_base_rgb = True
        self.uses_clamp = False
        self.ram_core = RamCore3(
            local_residual=raw_ram_local_residual,
            local_hidden_ch=raw_ram_local_hidden_ch,
            local_residual_scale=raw_ram_local_residual_scale,
            local_gate_init=raw_ram_local_gate_init,
            local_gate_mode=raw_ram_local_gate_mode,
        )
        self.dav2 = dav2_model
        self.raw_ram_rgb_tail = raw_ram_rgb_tail
        self.post_ram_cleanup_mode = str(post_ram_cleanup)
        self.post_ram_operation_position = str(post_ram_operation_position)
        if self.post_ram_cleanup_mode in {"none", "n_a"}:
            self.post_ram_cleanup_mode = "none"
            self.post_ram_cleanup = nn.Identity()
            self.post_ram_cleanup_enabled = False
        elif self.post_ram_cleanup_mode == "cnn":
            if self.post_ram_operation_position != "pre_tail":
                raise ValueError("RawRgb16Ram3DepthModel only implements post_ram_operation_position='pre_tail'")
            self.post_ram_cleanup = PostRamCleanupAdapter(
                channels=post_ram_cleanup_channels,
                blocks=post_ram_cleanup_blocks,
                scale=post_ram_cleanup_scale,
                norm=post_ram_cleanup_norm,
                zero_init=post_ram_cleanup_zero_init,
            )
            self.post_ram_cleanup_enabled = True
        else:
            raise ValueError(f"Unsupported post_ram_cleanup={post_ram_cleanup!r}")
        _register_imagenet_stats(self)
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)

    def forward(self, x_raw):
        x_rgb3_in = packed_bayer_to_base_rgb(x_raw)
        x3 = self.ram_core(x_rgb3_in)
        if self.raw_ram_rgb_tail == "tanh2p5":
            # Phase-1b compatibility path: softly bound heavy tails before DAv2.
            x3 = phase1b_tanh_tail_squash(x3)
        # Previous Phase-1 path:
        # x_norm = x3
        # Pre-Phase-1 path:
        # x_rgb = torch.clamp(x3, 0, 1); x_norm = (x_rgb - self.img_mean) / self.img_std
        x_norm = x3
        x_norm = self.spatial_adapter.pad_rgb(x_norm)
        depth = self.dav2(x_norm)
        return self.spatial_adapter.crop_depth(depth)

    def load_base_dav2_state_dict(self, state_dict):
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
                "Failed to load compatible DAv2 base weights for raw_to_base_rgb_ram3 model: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status


class RawRgb16Ram3DepthModel(nn.Module):
    """
    True-LOD 3-channel uint16 RAW front-end:
        raw_rgb16_3ch -> RamCore3 BN output
        -> optional tail -> center pad -> frozen DAv2 -> center crop depth

    Unlike RawToBaseRgbRam3DepthModel, this wrapper does not project Bayer
    channels. The dataset has already reordered OpenCV BGR PNG input to model
    RGB channel order.
    """

    def __init__(
        self,
        dav2_model,
        *,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
        raw_ram_rgb_tail="identity",
        raw_ram_local_residual="none",
        raw_ram_local_hidden_ch=16,
        raw_ram_local_residual_scale=0.1,
        raw_ram_local_gate_init=0.03,
        raw_ram_local_gate_mode="channel",
        post_ram_cleanup="none",
        post_ram_cleanup_channels=32,
        post_ram_cleanup_blocks=4,
        post_ram_cleanup_scale=0.1,
        post_ram_cleanup_norm="groupnorm",
        post_ram_cleanup_zero_init=True,
        post_ram_operation_position="n_a",
    ):
        super().__init__()
        raw_ram_rgb_tail = str(raw_ram_rgb_tail)
        if raw_ram_rgb_tail not in RAW_RAM_RGB_TAIL_CHOICES:
            raise ValueError(
                f"Unsupported raw_ram_rgb_tail={raw_ram_rgb_tail!r}; "
                f"expected one of {RAW_RAM_RGB_TAIL_CHOICES}"
            )
        self.front_end = "raw_rgb16_ram3"
        self.ram_core_type = "RamCore3"
        self.imagenet_norm_enabled = False
        self.uses_base_rgb = False
        self.uses_clamp = False
        self.ram_core = RamCore3(
            local_residual=raw_ram_local_residual,
            local_hidden_ch=raw_ram_local_hidden_ch,
            local_residual_scale=raw_ram_local_residual_scale,
            local_gate_init=raw_ram_local_gate_init,
            local_gate_mode=raw_ram_local_gate_mode,
        )
        self.dav2 = dav2_model
        self.raw_ram_rgb_tail = raw_ram_rgb_tail
        self.post_ram_cleanup_mode = str(post_ram_cleanup)
        self.post_ram_operation_position = str(post_ram_operation_position)
        if self.post_ram_cleanup_mode in {"none", "n_a"}:
            self.post_ram_cleanup_mode = "none"
            self.post_ram_cleanup = nn.Identity()
            self.post_ram_cleanup_enabled = False
        elif self.post_ram_cleanup_mode == "cnn":
            if self.post_ram_operation_position != "pre_tail":
                raise ValueError("RawRgb16Ram3DepthModel only implements post_ram_operation_position='pre_tail'")
            self.post_ram_cleanup = PostRamCleanupAdapter(
                channels=post_ram_cleanup_channels,
                blocks=post_ram_cleanup_blocks,
                scale=post_ram_cleanup_scale,
                norm=post_ram_cleanup_norm,
                zero_init=post_ram_cleanup_zero_init,
            )
            self.post_ram_cleanup_enabled = True
        else:
            raise ValueError(f"Unsupported post_ram_cleanup={post_ram_cleanup!r}")
        _register_imagenet_stats(self)
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)

    def forward(self, x_raw, *, return_features=False, return_layers=None):
        if x_raw.shape[1] != 3:
            raise ValueError(f"RawRgb16Ram3DepthModel expects 3 input channels, got {x_raw.shape[1]}")
        output = self.forward_with_dav2_features(x_raw, return_layers=return_layers or ())
        if return_features:
            return output
        return output["depth"]

    @staticmethod
    def _tensor_percentiles(tensor):
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            zero = tensor.detach().new_tensor(0.0)
            return zero, zero, zero
        if flat.numel() > 1_000_000:
            stride = max(int(flat.numel() // 1_000_000), 1)
            flat = flat[::stride]
        return torch.quantile(flat, torch.tensor([0.01, 0.50, 0.99], device=flat.device))

    def _prepare_dav2_input(self, x_raw):
        if x_raw.shape[1] != 3:
            raise ValueError(f"RawRgb16Ram3DepthModel expects 3 input channels, got {x_raw.shape[1]}")
        x3, ram_features = self.ram_core.forward_with_features(x_raw)
        x3_ram_native = x3
        if self.post_ram_cleanup_enabled:
            x3 = self.post_ram_cleanup(x3)
        ram_features["post_ram_xram"] = x3_ram_native
        ram_features["post_ram_xclean"] = x3
        if self.raw_ram_rgb_tail == "tanh2p5":
            x3 = phase1b_tanh_tail_squash(x3)
        x_norm = self.spatial_adapter.pad_rgb(x3)
        return x_norm, ram_features

    def forward_with_dav2_features(self, x_raw, *, return_layers):
        return_layers = tuple(int(layer) for layer in (return_layers or ()))
        depth_layers = tuple(int(layer) for layer in self.dav2.intermediate_layer_idx[self.dav2.encoder])
        unsupported = sorted(set(return_layers) - set(depth_layers))
        if unsupported:
            raise ValueError(
                f"RawRgb16Ram3DepthModel v1 only supports feature layers from depth head layers "
                f"{depth_layers}; got unsupported {unsupported}"
            )
        x_norm, ram_features = self._prepare_dav2_input(x_raw)
        patch_h, patch_w = x_norm.shape[-2] // 14, x_norm.shape[-1] // 14
        features = self.dav2.pretrained.get_intermediate_layers(
            x_norm,
            depth_layers,
            norm=True,
            return_class_token=True,
        )
        depth = self.dav2.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth).squeeze(1)
        depth = self.spatial_adapter.crop_depth(depth)
        layer_features = {
            layer: features[idx][0]
            for idx, layer in enumerate(depth_layers)
            if layer in return_layers
        }
        ram_debug = {}
        x3_ram = ram_features.get("x3_ram")
        x3_out = ram_features.get("x3")
        delta_x3 = ram_features.get("local_delta_x3")
        gate = ram_features.get("local_gate_tanh")
        if gate is not None:
            ram_debug["raw_ram_local_gate_value"] = gate.detach()
        if delta_x3 is not None:
            ram_debug["delta_x3_l1"] = delta_x3.detach().abs().mean()
            ram_debug["delta_x3_l2"] = delta_x3.detach().square().mean().sqrt()
        if x3_ram is not None:
            ram_debug["x3_ram_p1"], ram_debug["x3_ram_p50"], ram_debug["x3_ram_p99"] = self._tensor_percentiles(x3_ram)
        if x3_out is not None:
            ram_debug["x3_out_p1"], ram_debug["x3_out_p50"], ram_debug["x3_out_p99"] = self._tensor_percentiles(x3_out)
        xram = ram_features.get("post_ram_xram")
        xclean = ram_features.get("post_ram_xclean")
        if xram is not None and xclean is not None:
            delta = xclean - xram
            mean_abs_xram = xram.detach().abs().mean()
            mean_abs_delta = delta.detach().abs().mean()
            ram_debug["post_ram_enabled"] = xram.detach().new_tensor(float(self.post_ram_cleanup_enabled))
            ram_debug["post_ram_scale"] = xram.detach().new_tensor(
                float(getattr(self.post_ram_cleanup, "scale", 0.0))
            )
            ram_debug["post_ram_mean_abs_xram"] = mean_abs_xram
            ram_debug["post_ram_mean_abs_delta"] = mean_abs_delta
            ram_debug["post_ram_delta_ratio"] = mean_abs_delta / mean_abs_xram.clamp_min(1e-12)
            (
                ram_debug["post_ram_xram_p1"],
                ram_debug["post_ram_xram_p50"],
                ram_debug["post_ram_xram_p99"],
            ) = self._tensor_percentiles(xram)
            (
                ram_debug["post_ram_xclean_p1"],
                ram_debug["post_ram_xclean_p50"],
                ram_debug["post_ram_xclean_p99"],
            ) = self._tensor_percentiles(xclean)
        return {
            "depth": depth,
            "features": layer_features,
            "ram_debug": ram_debug,
        }

    def load_base_dav2_state_dict(self, state_dict):
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
                "Failed to load compatible DAv2 base weights for raw_rgb16_ram3 model: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status


class RawRamRgbDepthModel(RawToBaseRgbRam3DepthModel):
    """Backward-compatible alias. Prefer RawToBaseRgbRam3DepthModel."""


class RawRamResidualDepthModel(nn.Module):
    """
    E2.1 end-to-end model:
        raw4 -> RawRamCore -> ResidualRGBHead -> delta_rgb
        base_rgb = [R, (Gr + Gb)/2, B]
        adapted_rgb = clamp(base_rgb + residual_scale * tanh(delta_rgb), 0, 1)
        adapted_rgb -> ImageNet norm -> frozen DAv2 -> depth
    """

    def __init__(
        self,
        dav2_model,
        residual_scale=0.1,
        *,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        super().__init__()
        self.front_end = "raw_ram4"
        self.ram_core_type = "RawRamCore"
        self.imagenet_norm_enabled = True
        self.uses_base_rgb = True
        self.uses_clamp = True
        self.ram_core = RawRamCore()
        self.residual_head = ResidualRGBHead()
        self.residual_scale = float(residual_scale)
        self.dav2 = dav2_model
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        _register_imagenet_stats(self)

    def forward(self, x_raw):
        x4 = self.ram_core(x_raw)                   # (B, 4, H, W)
        delta_rgb = self.residual_head(x4)         # (B, 3, H, W)
        base_rgb = packed_bayer_to_base_rgb(x_raw) # (B, 3, H, W)
        x_rgb = torch.clamp(
            base_rgb + self.residual_scale * torch.tanh(delta_rgb),
            min=0.0,
            max=1.0,
        )
        x_norm = (x_rgb - self.img_mean) / self.img_std
        x_norm = self.spatial_adapter.pad_rgb(x_norm)
        depth = self.dav2(x_norm)
        return self.spatial_adapter.crop_depth(depth)

    def load_base_dav2_state_dict(self, state_dict):
        return self.dav2.load_state_dict(state_dict, strict=True)


def build_raw_ram_depth_model(
    dav2_model,
    *,
    front_end="raw_ram4",
    input_type="raw_ram",
    residual_scale=0.1,
    rgb_interface_mode="residual_tanh",
    rgb_residual_scale=0.1,
    raw_ram_rgb_tail="tanh2p5",
    raw_ram_local_residual="none",
    raw_ram_local_hidden_ch=16,
    raw_ram_local_residual_scale=0.1,
    raw_ram_local_gate_init=0.03,
    raw_ram_local_gate_mode="channel",
    post_ram_cleanup="none",
    post_ram_cleanup_channels=32,
    post_ram_cleanup_blocks=4,
    post_ram_cleanup_scale=0.1,
    post_ram_cleanup_norm="groupnorm",
    post_ram_cleanup_zero_init=True,
    post_ram_operation_position="n_a",
    sensor_hw=SENSOR_INPUT_HW,
    backbone_hw=BACKBONE_INPUT_HW,
):
    front_end = str(front_end)
    if front_end == "raw_ram4":
        if input_type in ("raw_ram", *RAW_RAM_LORA_INPUT_TYPES):
            return RawRamDepthModel(
                dav2_model,
                rgb_interface_mode=rgb_interface_mode,
                rgb_residual_scale=rgb_residual_scale,
                sensor_hw=sensor_hw,
                backbone_hw=backbone_hw,
            )
        if input_type == "raw_ram_residual":
            return RawRamResidualDepthModel(
                dav2_model,
                residual_scale=residual_scale,
                sensor_hw=sensor_hw,
                backbone_hw=backbone_hw,
            )
        raise ValueError(
            f"front_end={front_end} is only compatible with input_type in {{raw_ram, raw_ram_residual}}; "
            f"got input_type={input_type}"
        )

    if front_end == "raw_to_base_rgb_ram3":
        if input_type not in RAW_RAM_RGB_INPUT_TYPES:
            raise ValueError(
                f"front_end={front_end} requires input_type in {RAW_RAM_RGB_INPUT_TYPES}; "
                f"got input_type={input_type}"
            )
        return RawToBaseRgbRam3DepthModel(
            dav2_model,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
            raw_ram_rgb_tail=raw_ram_rgb_tail,
            raw_ram_local_residual=raw_ram_local_residual,
            raw_ram_local_hidden_ch=raw_ram_local_hidden_ch,
            raw_ram_local_residual_scale=raw_ram_local_residual_scale,
            raw_ram_local_gate_init=raw_ram_local_gate_init,
            raw_ram_local_gate_mode=raw_ram_local_gate_mode,
            post_ram_cleanup=post_ram_cleanup,
            post_ram_cleanup_channels=post_ram_cleanup_channels,
            post_ram_cleanup_blocks=post_ram_cleanup_blocks,
            post_ram_cleanup_scale=post_ram_cleanup_scale,
            post_ram_cleanup_norm=post_ram_cleanup_norm,
            post_ram_cleanup_zero_init=post_ram_cleanup_zero_init,
            post_ram_operation_position=post_ram_operation_position,
        )

    if front_end == "raw_rgb16_ram3":
        if input_type not in RAW_RGB16_RAM3_INPUT_TYPES:
            raise ValueError(
                f"front_end={front_end} requires input_type in {RAW_RGB16_RAM3_INPUT_TYPES}; "
                f"got input_type={input_type}"
            )
        return RawRgb16Ram3DepthModel(
            dav2_model,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
            raw_ram_rgb_tail=raw_ram_rgb_tail,
            raw_ram_local_residual=raw_ram_local_residual,
            raw_ram_local_hidden_ch=raw_ram_local_hidden_ch,
            raw_ram_local_residual_scale=raw_ram_local_residual_scale,
            raw_ram_local_gate_init=raw_ram_local_gate_init,
            raw_ram_local_gate_mode=raw_ram_local_gate_mode,
            post_ram_cleanup=post_ram_cleanup,
            post_ram_cleanup_channels=post_ram_cleanup_channels,
            post_ram_cleanup_blocks=post_ram_cleanup_blocks,
            post_ram_cleanup_scale=post_ram_cleanup_scale,
            post_ram_cleanup_norm=post_ram_cleanup_norm,
            post_ram_cleanup_zero_init=post_ram_cleanup_zero_init,
            post_ram_operation_position=post_ram_operation_position,
        )

    if front_end == "raw_to_rgb_head":
        raise ValueError(
            f"Cannot use build_raw_ram_depth_model for front_end={front_end}; "
            "use build_dav2_raw_naive_depth_model instead"
        )
    raise ValueError(f"Unsupported front_end={front_end} for raw RAM depth model")
