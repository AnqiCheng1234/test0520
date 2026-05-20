import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune_stf.models.raw_ram import (
    RAW_RAM_BRIDGE_FEATURE_CHANNELS,
    RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS,
    RGBInterfaceHead,
    RamCore3,
    RawRamCore,
    _register_imagenet_stats,
    packed_bayer_to_base_rgb,
    phase1b_tanh_tail_squash,
)
from finetune_stf.models.spatial_adapter import BACKBONE_INPUT_HW, SENSOR_INPUT_HW, CenterPadCropAdapter


RAW_RAM_BRIDGE_ONLY_INPUT_TYPES = ("raw_ram_bridge",)
RAW_RAM_BRIDGE_LORA_INPUT_TYPES = ("raw_ram_bridge_lora",)
RAW_RAM_BRIDGE_INPUT_TYPES = RAW_RAM_BRIDGE_ONLY_INPUT_TYPES + RAW_RAM_BRIDGE_LORA_INPUT_TYPES
DEFAULT_BRIDGE_FEATURE_KEYS = ("x_cat", "ffm_mid", "x4")
RAW_RAM_RGB_BRIDGE_ONLY_INPUT_TYPES = ("raw_ram_rgb_bridge",)
RAW_RAM_RGB_BRIDGE_LORA_INPUT_TYPES = ("raw_ram_rgb_bridge_lora",)
RAW_RAM_RGB_BRIDGE_INPUT_TYPES = RAW_RAM_RGB_BRIDGE_ONLY_INPUT_TYPES + RAW_RAM_RGB_BRIDGE_LORA_INPUT_TYPES
DEFAULT_RGB_BRIDGE_FEATURE_KEYS = ("x_cat", "ffm_mid", "x3")
LORA_BLOCK_MODE_CHOICES = ("all", "front", "mid", "back", "tap")
DEFAULT_LORA_BLOCK_MODE = "tap"


class BridgeProjectionHead(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feat, patch_hw):
        pooled = F.adaptive_avg_pool2d(feat, patch_hw)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.norm(self.proj(tokens))


class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, *, rank=8, alpha=16.0):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")

        self.orig = orig_linear
        self.orig.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)

        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        self.lora_A = nn.Linear(in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.orig(x) + self.lora_B(self.lora_A(x)) * self.scale


def _iter_vit_blocks(vit):
    if getattr(vit, "chunked_blocks", False):
        blocks = []
        for block_chunk in vit.blocks:
            for block in block_chunk:
                if isinstance(block, nn.Identity):
                    continue
                blocks.append(block)
        return blocks
    return list(vit.blocks)


def _split_block_indices(num_blocks):
    indices = list(range(num_blocks))
    base = num_blocks // 3
    remainder = num_blocks % 3
    sizes = [base + (1 if idx < remainder else 0) for idx in range(3)]

    splits = []
    start = 0
    for size in sizes:
        splits.append(indices[start : start + size])
        start += size
    return splits


def resolve_lora_block_indices(*, num_blocks, block_mode, tap_layers):
    block_mode = str(block_mode)
    if block_mode not in LORA_BLOCK_MODE_CHOICES:
        raise ValueError(f"Unsupported lora block mode: {block_mode}")

    if block_mode == "all":
        return tuple(range(num_blocks))
    if block_mode == "tap":
        resolved = tuple(sorted({int(layer) for layer in tap_layers}))
    else:
        front, mid, back = _split_block_indices(num_blocks)
        mapping = {
            "front": tuple(front),
            "mid": tuple(mid),
            "back": tuple(back),
        }
        resolved = mapping[block_mode]

    if not resolved:
        raise ValueError(f"Resolved empty LoRA block indices for mode={block_mode}")
    if min(resolved) < 0 or max(resolved) >= num_blocks:
        raise ValueError(
            f"LoRA block indices {resolved} are out of range for num_blocks={num_blocks}"
        )
    return tuple(resolved)


def apply_lora_to_vit(vit, *, block_mode, tap_layers, rank, alpha):
    blocks = _iter_vit_blocks(vit)
    block_indices = resolve_lora_block_indices(
        num_blocks=len(blocks),
        block_mode=block_mode,
        tap_layers=tap_layers,
    )

    for block_idx in block_indices:
        block = blocks[block_idx]
        if not isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
        if not isinstance(block.attn.proj, LoRALinear):
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha)
    return block_indices


def _remap_state_dict_for_lora_modules(module, state_dict):
    compatible = dict(state_dict)
    for module_name, submodule in module.named_modules():
        if not isinstance(submodule, LoRALinear):
            continue
        for suffix in ("weight", "bias"):
            base_key = f"{module_name}.{suffix}"
            orig_key = f"{module_name}.orig.{suffix}"
            if base_key in compatible and orig_key not in compatible:
                compatible[orig_key] = compatible.pop(base_key)
    return compatible


def merge_lora_in_state_dict(state_dict, *, alpha, rank):
    """Fold LoRA delta back into orig weight and rewrite as plain Linear keys.

    For every `…orig.weight`/`…lora_A.weight`/`…lora_B.weight` triple in
    `state_dict`, computes `W_eff = W_orig + (alpha/rank) * (B @ A)` and emits
    the result under `…weight`. Bias (`…orig.bias`) is renamed to `…bias`.
    LoRA-only keys are dropped from the output.

    Used when copying a LoRA-trained state_dict into a non-LoRA-wrapped module
    (e.g. the KITTI rgb-only eval wrapper), so the eval reflects the trained
    attention weights rather than re-introducing the un-merged frozen base.
    """
    scale = float(alpha) / float(rank)
    output = {}
    lora_groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if key.endswith(".orig.weight"):
            base = key[: -len(".orig.weight")]
            lora_groups.setdefault(base, {})["W"] = value
            continue
        if key.endswith(".orig.bias"):
            base = key[: -len(".orig.bias")]
            output[f"{base}.bias"] = value
            continue
        if key.endswith(".lora_A.weight"):
            base = key[: -len(".lora_A.weight")]
            lora_groups.setdefault(base, {})["A"] = value
            continue
        if key.endswith(".lora_B.weight"):
            base = key[: -len(".lora_B.weight")]
            lora_groups.setdefault(base, {})["B"] = value
            continue
        output[key] = value

    for base, parts in lora_groups.items():
        W = parts.get("W")
        A = parts.get("A")
        B = parts.get("B")
        if W is None:
            continue
        if A is not None and B is not None:
            delta = (B.to(dtype=torch.float32) @ A.to(dtype=torch.float32)) * scale
            W_eff = W.to(dtype=torch.float32) + delta
            output[f"{base}.weight"] = W_eff.to(dtype=W.dtype)
        else:
            output[f"{base}.weight"] = W
    return output


class RawFeatureBridgeAdapter(nn.Module):
    def __init__(self, *, feature_channels, feature_keys, target_layers, embed_dim):
        super().__init__()
        if not feature_keys:
            raise ValueError("feature_keys must be non-empty for RawFeatureBridgeAdapter")
        if not target_layers:
            raise ValueError("target_layers must be non-empty for RawFeatureBridgeAdapter")

        self.feature_keys = tuple(feature_keys)
        self.target_layers = tuple(int(layer) for layer in target_layers)
        self.projections = nn.ModuleDict()
        self.gates = nn.ParameterDict()

        for layer in self.target_layers:
            layer_key = str(layer)
            self.projections[layer_key] = nn.ModuleDict(
                {
                    feature_key: BridgeProjectionHead(feature_channels[feature_key], embed_dim)
                    for feature_key in self.feature_keys
                }
            )
            self.gates[layer_key] = nn.Parameter(torch.zeros(1))

    def forward(self, feature_dict, *, patch_hw):
        bridge_injections = {}
        for layer in self.target_layers:
            layer_key = str(layer)
            layer_tokens = None
            for feature_key in self.feature_keys:
                if feature_key not in feature_dict:
                    raise KeyError(f"Missing bridge feature '{feature_key}' in feature_dict")
                tokens = self.projections[layer_key][feature_key](feature_dict[feature_key], patch_hw)
                layer_tokens = tokens if layer_tokens is None else layer_tokens + tokens
            bridge_injections[layer] = torch.tanh(self.gates[layer_key]) * layer_tokens
        return bridge_injections


class RawRamBridgeDepthModel(nn.Module):
    """
    E3.1 bridge-only model:
        raw4 -> RawRamCore -> RGBInterfaceHead -> ImageNet norm -> frozen DAv2 -> depth
                                             \\-> bridge projections -> ViT tap blocks
    """

    def __init__(
        self,
        dav2_model,
        *,
        bridge_feature_keys=DEFAULT_BRIDGE_FEATURE_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        rgb_interface_mode="residual_tanh",
        rgb_residual_scale=0.1,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        super().__init__()
        if bridge_source != "ram_core":
            raise ValueError(f"Unsupported bridge_source for now: {bridge_source}")

        feature_keys = tuple(bridge_feature_keys)
        unknown_keys = [key for key in feature_keys if key not in RAW_RAM_BRIDGE_FEATURE_CHANNELS]
        if unknown_keys:
            raise ValueError(f"Unsupported bridge_feature_keys: {unknown_keys}")

        if bridge_layers is None:
            bridge_layers = dav2_model.intermediate_layer_idx[dav2_model.encoder]

        self.ram_core = RawRamCore()
        self.rgb_head = RGBInterfaceHead(
            mode=rgb_interface_mode,
            residual_scale=rgb_residual_scale,
        )
        self.dav2 = dav2_model
        self.bridge_source = bridge_source
        self.bridge_feature_keys = feature_keys
        self.bridge_layers = tuple(int(layer) for layer in bridge_layers)
        self.bridge_adapter = RawFeatureBridgeAdapter(
            feature_channels=RAW_RAM_BRIDGE_FEATURE_CHANNELS,
            feature_keys=self.bridge_feature_keys,
            target_layers=self.bridge_layers,
            embed_dim=self.dav2.pretrained.embed_dim,
        )
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        _register_imagenet_stats(self)

    def build_bridge_injections(self, x_raw):
        x4, feature_dict = self.ram_core.forward_with_features(x_raw)
        x_rgb = self.rgb_head(x4, x_raw=x_raw)
        x_norm = (x_rgb - self.img_mean) / self.img_std
        x_norm = self.spatial_adapter.pad_rgb(x_norm)
        patch_hw = (
            x_norm.shape[-2] // self.dav2.pretrained.patch_size,
            x_norm.shape[-1] // self.dav2.pretrained.patch_size,
        )
        bridge_injections = self.bridge_adapter(feature_dict, patch_hw=patch_hw)
        return x_norm, bridge_injections

    def forward(self, x_raw):
        x_norm, bridge_injections = self.build_bridge_injections(x_raw)
        depth = self.dav2(x_norm, bridge_injections=bridge_injections)
        return self.spatial_adapter.crop_depth(depth)

    def get_optimizer_param_groups(self, *, base_lr, bridge_lr, lora_lr=None):
        base_params = []
        bridge_params = []
        lora_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".lora_A." in name or ".lora_B." in name:
                lora_params.append(param)
            elif name.startswith("bridge_adapter."):
                bridge_params.append(param)
            else:
                base_params.append(param)

        param_groups = [{"params": base_params, "lr": base_lr, "initial_lr": base_lr}]
        if bridge_params:
            param_groups.append({"params": bridge_params, "lr": bridge_lr, "initial_lr": bridge_lr})
        if lora_params:
            lora_group_lr = bridge_lr if lora_lr is None else lora_lr
            param_groups.append({"params": lora_params, "lr": lora_group_lr, "initial_lr": lora_group_lr})
        return param_groups

    def load_base_dav2_state_dict(self, state_dict):
        compatible = _remap_state_dict_for_lora_modules(self.dav2, state_dict)
        status = self.dav2.load_state_dict(compatible, strict=False)
        missing = [
            key
            for key in status.missing_keys
            if ".lora_A." not in key and ".lora_B." not in key
        ]
        if missing or status.unexpected_keys:
            raise RuntimeError(
                "Failed to load compatible DAv2 base weights for bridge/LoRA model: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status

    def load_compatible_state_dict(self, state_dict, *, strict=False):
        compatible = _remap_state_dict_for_lora_modules(self, state_dict)
        return self.load_state_dict(compatible, strict=strict)


class RawRamBridgeLoRADepthModel(RawRamBridgeDepthModel):
    def __init__(
        self,
        dav2_model,
        *,
        bridge_feature_keys=DEFAULT_BRIDGE_FEATURE_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        rgb_interface_mode="residual_tanh",
        rgb_residual_scale=0.1,
        lora_block_mode=DEFAULT_LORA_BLOCK_MODE,
        lora_rank=8,
        lora_alpha=16.0,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        super().__init__(
            dav2_model,
            bridge_feature_keys=bridge_feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            rgb_interface_mode=rgb_interface_mode,
            rgb_residual_scale=rgb_residual_scale,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
        self.lora_block_mode = str(lora_block_mode)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_block_indices = apply_lora_to_vit(
            self.dav2.pretrained,
            block_mode=self.lora_block_mode,
            tap_layers=self.bridge_layers,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )


class RawRamRgbBridgeDepthModel(nn.Module):
    """
    Bridge-only model with 3-channel RAM front-end:
        raw4 -> [R,(Gr+Gb)/2,B] -> RamCore3 BN output -> frozen DAv2 -> depth
                                                      \\-> bridge projections -> ViT tap blocks
    """

    def __init__(
        self,
        dav2_model,
        *,
        bridge_feature_keys=DEFAULT_RGB_BRIDGE_FEATURE_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        super().__init__()
        if bridge_source != "ram_core":
            raise ValueError(f"Unsupported bridge_source for now: {bridge_source}")

        feature_keys = tuple(bridge_feature_keys)
        unknown_keys = [key for key in feature_keys if key not in RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS]
        if unknown_keys:
            raise ValueError(f"Unsupported bridge_feature_keys: {unknown_keys}")

        if bridge_layers is None:
            bridge_layers = dav2_model.intermediate_layer_idx[dav2_model.encoder]

        self.ram_core = RamCore3()
        self.dav2 = dav2_model
        self.bridge_source = bridge_source
        self.bridge_feature_keys = feature_keys
        self.bridge_layers = tuple(int(layer) for layer in bridge_layers)
        self.bridge_adapter = RawFeatureBridgeAdapter(
            feature_channels=RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS,
            feature_keys=self.bridge_feature_keys,
            target_layers=self.bridge_layers,
            embed_dim=self.dav2.pretrained.embed_dim,
        )
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        _register_imagenet_stats(self)

    def build_bridge_injections(self, x_raw):
        x3_in = packed_bayer_to_base_rgb(x_raw)
        x3, feature_dict = self.ram_core.forward_with_features(x3_in)
        # Phase-1b path shared with raw_ram_rgb: BN output with soft tail squash,
        # still no hard clamp and no ImageNet normalization after RamCore3.
        x3 = phase1b_tanh_tail_squash(x3)
        feature_dict = {**feature_dict, "x3": x3}
        # Pre-Phase-1 path:
        # x_rgb = torch.clamp(x3, 0, 1); x_norm = (x_rgb - self.img_mean) / self.img_std
        x_norm = x3
        x_norm = self.spatial_adapter.pad_rgb(x_norm)
        patch_hw = (
            x_norm.shape[-2] // self.dav2.pretrained.patch_size,
            x_norm.shape[-1] // self.dav2.pretrained.patch_size,
        )
        bridge_injections = self.bridge_adapter(feature_dict, patch_hw=patch_hw)
        return x_norm, bridge_injections

    def forward(self, x_raw):
        x_norm, bridge_injections = self.build_bridge_injections(x_raw)
        depth = self.dav2(x_norm, bridge_injections=bridge_injections)
        return self.spatial_adapter.crop_depth(depth)

    def get_optimizer_param_groups(self, *, base_lr, bridge_lr, lora_lr=None):
        base_params = []
        bridge_params = []
        lora_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".lora_A." in name or ".lora_B." in name:
                lora_params.append(param)
            elif name.startswith("bridge_adapter."):
                bridge_params.append(param)
            else:
                base_params.append(param)

        param_groups = [{"params": base_params, "lr": base_lr, "initial_lr": base_lr}]
        if bridge_params:
            param_groups.append({"params": bridge_params, "lr": bridge_lr, "initial_lr": bridge_lr})
        if lora_params:
            lora_group_lr = bridge_lr if lora_lr is None else lora_lr
            param_groups.append({"params": lora_params, "lr": lora_group_lr, "initial_lr": lora_group_lr})
        return param_groups

    def load_base_dav2_state_dict(self, state_dict):
        compatible = _remap_state_dict_for_lora_modules(self.dav2, state_dict)
        status = self.dav2.load_state_dict(compatible, strict=False)
        missing = [
            key
            for key in status.missing_keys
            if ".lora_A." not in key and ".lora_B." not in key
        ]
        if missing or status.unexpected_keys:
            raise RuntimeError(
                "Failed to load compatible DAv2 base weights for bridge/LoRA model: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status

    def load_compatible_state_dict(self, state_dict, *, strict=False):
        compatible = _remap_state_dict_for_lora_modules(self, state_dict)
        return self.load_state_dict(compatible, strict=strict)


class RawRamRgbBridgeLoRADepthModel(RawRamRgbBridgeDepthModel):
    def __init__(
        self,
        dav2_model,
        *,
        bridge_feature_keys=DEFAULT_RGB_BRIDGE_FEATURE_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        lora_block_mode=DEFAULT_LORA_BLOCK_MODE,
        lora_rank=8,
        lora_alpha=16.0,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        super().__init__(
            dav2_model,
            bridge_feature_keys=bridge_feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
        self.lora_block_mode = str(lora_block_mode)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_block_indices = apply_lora_to_vit(
            self.dav2.pretrained,
            block_mode=self.lora_block_mode,
            tap_layers=self.bridge_layers,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
        )


def build_raw_ram_bridge_depth_model(
    dav2_model,
    *,
    input_type="raw_ram_bridge",
    bridge_source="ram_core",
    bridge_feature_keys=None,
    bridge_layers=None,
    rgb_interface_mode="residual_tanh",
    rgb_residual_scale=0.1,
    lora_block_mode=DEFAULT_LORA_BLOCK_MODE,
    lora_rank=8,
    lora_alpha=16.0,
    sensor_hw=SENSOR_INPUT_HW,
    backbone_hw=BACKBONE_INPUT_HW,
):
    if bridge_feature_keys is None:
        if input_type in RAW_RAM_RGB_BRIDGE_INPUT_TYPES:
            bridge_feature_keys = DEFAULT_RGB_BRIDGE_FEATURE_KEYS
        else:
            bridge_feature_keys = DEFAULT_BRIDGE_FEATURE_KEYS

    if input_type == "raw_ram_bridge":
        return RawRamBridgeDepthModel(
            dav2_model,
            bridge_feature_keys=bridge_feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            rgb_interface_mode=rgb_interface_mode,
            rgb_residual_scale=rgb_residual_scale,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    if input_type == "raw_ram_bridge_lora":
        return RawRamBridgeLoRADepthModel(
            dav2_model,
            bridge_feature_keys=bridge_feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            rgb_interface_mode=rgb_interface_mode,
            rgb_residual_scale=rgb_residual_scale,
            lora_block_mode=lora_block_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    if input_type == "raw_ram_rgb_bridge":
        return RawRamRgbBridgeDepthModel(
            dav2_model,
            bridge_feature_keys=bridge_feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    if input_type == "raw_ram_rgb_bridge_lora":
        return RawRamRgbBridgeLoRADepthModel(
            dav2_model,
            bridge_feature_keys=bridge_feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            lora_block_mode=lora_block_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    raise ValueError(f"Unsupported bridge input_type: {input_type}")


def load_bridge_init_weights(model, state_dict):
    if hasattr(model, "load_compatible_state_dict"):
        return model.load_compatible_state_dict(state_dict, strict=False)
    return model.load_state_dict(state_dict, strict=False)
