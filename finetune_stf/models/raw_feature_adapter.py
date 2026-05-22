import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune_stf.models.lora_bridge import (
    DEFAULT_LORA_BLOCK_MODE,
    DEFAULT_RGB_BRIDGE_FEATURE_KEYS,
    LORA_BLOCK_MODE_CHOICES,
    RawFeatureBridgeAdapter,
    _remap_state_dict_for_lora_modules,
    apply_lora_to_vit,
)
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


RAW_RAM_FEATURE_ADAPTER_ONLY_INPUT_TYPES = ("raw_ram_feature_adapter",)
RAW_RAM_BRIDGE_FEATURE_ADAPTER_ONLY_INPUT_TYPES = ("raw_ram_bridge_feature_adapter",)
RAW_RAM_BRIDGE_FEATURE_ADAPTER_LORA_INPUT_TYPES = ("raw_ram_bridge_feature_adapter_lora",)
RAW_RAM_RGB_BRIDGE_FEATURE_ADAPTER_ONLY_INPUT_TYPES = ("raw_ram_rgb_bridge_feature_adapter",)
RAW_RAM_BRIDGE_FEATURE_ADAPTER_INPUT_TYPES = (
    RAW_RAM_BRIDGE_FEATURE_ADAPTER_ONLY_INPUT_TYPES
    + RAW_RAM_BRIDGE_FEATURE_ADAPTER_LORA_INPUT_TYPES
)
RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES = (
    RAW_RAM_FEATURE_ADAPTER_ONLY_INPUT_TYPES
    + RAW_RAM_BRIDGE_FEATURE_ADAPTER_INPUT_TYPES
    + RAW_RAM_RGB_BRIDGE_FEATURE_ADAPTER_ONLY_INPUT_TYPES
)
DEFAULT_FEATURE_ADAPTER_KEYS = ("x_cat", "ffm_mid", "x4")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, *, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ImageBridgeHeadV2(nn.Module):
    def __init__(
        self,
        in_ch=4,
        out_ch=3,
        *,
        mode="residual_tanh",
        residual_scale=0.1,
    ):
        super().__init__()
        if in_ch != 4 or out_ch != 3:
            raise ValueError("ImageBridgeHeadV2 currently supports only 4ch -> 3ch")
        self.rgb_head = RGBInterfaceHead(mode=mode, residual_scale=residual_scale)

    def forward(self, x, *, x_raw=None):
        return self.rgb_head(x, x_raw=x_raw)


class RAWFeatureProjector(nn.Module):
    def __init__(self, *, feature_channels, feature_keys, adapter_dim=64):
        super().__init__()
        if not feature_keys:
            raise ValueError("feature_keys must be non-empty for RAWFeatureProjector")
        self.feature_keys = tuple(feature_keys)
        self.adapter_dim = int(adapter_dim)
        self.feature_projs = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Conv2d(feature_channels[key], self.adapter_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.adapter_dim),
                    nn.SiLU(inplace=True),
                )
                for key in self.feature_keys
            }
        )
        self.scale_heads = nn.ModuleDict(
            {
                key: nn.Sequential(
                    ConvBNAct(self.adapter_dim, self.adapter_dim),
                    ConvBNAct(self.adapter_dim, self.adapter_dim),
                )
                for key in ("a1", "a2", "a3")
            }
        )

    def forward(self, feature_dict, *, target_sizes):
        fused = None
        for key in self.feature_keys:
            if key not in feature_dict:
                raise KeyError(f"Missing RAW adapter feature '{key}'")
            feat = self.feature_projs[key](feature_dict[key])
            fused = feat if fused is None else fused + feat

        outputs = {}
        for scale_key, size in target_sizes.items():
            scaled = F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            outputs[scale_key] = self.scale_heads[scale_key](scaled)
        return outputs


class DepthMergeBlock(nn.Module):
    def __init__(self, feat_dim, adapter_dim):
        super().__init__()
        hidden_dim = max(feat_dim, adapter_dim)
        self.res_blocks = nn.Sequential(
            ConvBNAct(feat_dim + adapter_dim, hidden_dim),
            ConvBNAct(hidden_dim, hidden_dim),
        )
        self.feat_out = nn.Conv2d(hidden_dim, feat_dim, kernel_size=1, padding=0, bias=True)
        self.adapter_out = nn.Conv2d(hidden_dim, adapter_dim, kernel_size=1, padding=0, bias=True)

    def forward(self, feat, adapter):
        if adapter.shape[-2:] != feat.shape[-2:]:
            adapter = F.interpolate(adapter, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([feat, adapter], dim=1)
        x = self.res_blocks(x)
        fused = feat + self.feat_out(x)
        next_adapter = self.adapter_out(x)
        return fused, next_adapter


class RawRamFeatureAdapterDepthModel(nn.Module):
    """
    E2: raw4 -> RawRamCore -> ImageBridgeHeadV2 -> frozen/decoder-trainable DAv2
        plus decoder-side feature adapters projected from x_cat / ffm_mid / x4.
    """

    def __init__(
        self,
        dav2_model,
        *,
        feature_keys=DEFAULT_FEATURE_ADAPTER_KEYS,
        adapter_dim=64,
        rgb_interface_mode="residual_tanh",
        rgb_residual_scale=0.1,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        super().__init__()
        unknown_keys = [key for key in feature_keys if key not in RAW_RAM_BRIDGE_FEATURE_CHANNELS]
        if unknown_keys:
            raise ValueError(f"Unsupported feature adapter keys: {unknown_keys}")

        self.ram_core = RawRamCore()
        self.image_bridge = ImageBridgeHeadV2(
            mode=rgb_interface_mode,
            residual_scale=rgb_residual_scale,
        )
        self.feature_adapter_keys = tuple(feature_keys)
        self.dav2 = dav2_model
        self.feature_projector = RAWFeatureProjector(
            feature_channels=RAW_RAM_BRIDGE_FEATURE_CHANNELS,
            feature_keys=self.feature_adapter_keys,
            adapter_dim=adapter_dim,
        )
        decoder_feat_dim = int(self.dav2.depth_head.scratch.output_conv1.in_channels)
        self.merge3 = DepthMergeBlock(decoder_feat_dim, adapter_dim)
        self.merge2 = DepthMergeBlock(decoder_feat_dim, adapter_dim)
        self.merge1 = DepthMergeBlock(decoder_feat_dim, adapter_dim)
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        _register_imagenet_stats(self)

    def _build_bridge_injections(self, feature_dict, *, patch_hw):
        return None

    def _extract_backbone_features(self, x_norm, *, bridge_injections=None):
        patch_h = x_norm.shape[-2] // self.dav2.pretrained.patch_size
        patch_w = x_norm.shape[-1] // self.dav2.pretrained.patch_size
        features = self.dav2.pretrained.get_intermediate_layers(
            x_norm,
            self.dav2.intermediate_layer_idx[self.dav2.encoder],
            return_class_token=True,
            bridge_injections=bridge_injections,
        )
        return features, patch_h, patch_w

    def _forward_decoder_with_adapters(self, out_features, patch_h, patch_w, adapters):
        depth_head = self.dav2.depth_head
        out = []
        for i, x in enumerate(out_features):
            if depth_head.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = depth_head.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = depth_head.projects[i](x)
            x = depth_head.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = depth_head.scratch.layer1_rn(layer_1)
        layer_2_rn = depth_head.scratch.layer2_rn(layer_2)
        layer_3_rn = depth_head.scratch.layer3_rn(layer_3)
        layer_4_rn = depth_head.scratch.layer4_rn(layer_4)

        path_4 = depth_head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4, next_a2 = self.merge3(path_4, adapters["a3"])

        a2 = adapters["a2"]
        next_a2 = F.interpolate(next_a2, size=a2.shape[-2:], mode="bilinear", align_corners=False)
        path_3 = depth_head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3, next_a1 = self.merge2(path_3, a2 + next_a2)

        a1 = adapters["a1"]
        next_a1 = F.interpolate(next_a1, size=a1.shape[-2:], mode="bilinear", align_corners=False)
        path_2 = depth_head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_2, _ = self.merge1(path_2, a1 + next_a1)

        path_1 = depth_head.scratch.refinenet1(path_2, layer_1_rn)
        out = depth_head.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * self.dav2.pretrained.patch_size), int(patch_w * self.dav2.pretrained.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        out = depth_head.scratch.output_conv2(out)
        return out

    def forward_features(self, x_raw):
        x4, feature_dict = self.ram_core.forward_with_features(x_raw)
        front_rgb = self.image_bridge(x4, x_raw=x_raw)
        x_norm = (front_rgb - self.img_mean) / self.img_std
        x_norm = self.spatial_adapter.pad_rgb(x_norm)
        patch_hw = (
            x_norm.shape[-2] // self.dav2.pretrained.patch_size,
            x_norm.shape[-1] // self.dav2.pretrained.patch_size,
        )
        bridge_injections = self._build_bridge_injections(feature_dict, patch_hw=patch_hw)
        out_features, patch_h, patch_w = self._extract_backbone_features(
            x_norm,
            bridge_injections=bridge_injections,
        )
        adapters = self.feature_projector(
            feature_dict,
            target_sizes={
                "a3": (patch_h, patch_w),
                "a2": (patch_h * 2, patch_w * 2),
                "a1": (patch_h * 4, patch_w * 4),
            },
        )
        depth = self._forward_decoder_with_adapters(out_features, patch_h, patch_w, adapters)
        depth = F.relu(depth).squeeze(1)
        depth = self.spatial_adapter.crop_depth(depth)
        return {"rgb": front_rgb, "depth": depth}

    def forward(self, x_raw):
        return self.forward_features(x_raw)["depth"]

    def get_optimizer_param_groups(self, *, base_lr, bridge_lr, lora_lr=None):
        base_params = []
        adapter_params = []
        lora_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".lora_A." in name or ".lora_B." in name:
                lora_params.append(param)
            elif name.startswith(
                (
                    "image_bridge.",
                    "feature_projector.",
                    "merge1.",
                    "merge2.",
                    "merge3.",
                    "bridge_adapter.",
                )
            ):
                adapter_params.append(param)
            else:
                base_params.append(param)

        param_groups = [{"params": base_params, "lr": base_lr, "initial_lr": base_lr}]
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": bridge_lr, "initial_lr": bridge_lr})
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
                "Failed to load compatible DAv2 base weights for feature-adapter model: "
                f"missing_non_lora={missing}, unexpected={status.unexpected_keys}"
            )
        return status

    def load_compatible_state_dict(self, state_dict, *, strict=False):
        compatible = _remap_state_dict_for_lora_modules(self, state_dict)
        return self.load_state_dict(compatible, strict=strict)


class RawRamBridgeFeatureAdapterDepthModel(RawRamFeatureAdapterDepthModel):
    """
    raw4 -> RawRamCore -> ImageBridgeHeadV2 -> DAv2 backbone with bridge injections
        plus decoder-side feature adapters from the same RAW-RAM features.
    """

    def __init__(
        self,
        dav2_model,
        *,
        feature_keys=DEFAULT_FEATURE_ADAPTER_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        adapter_dim=64,
        rgb_interface_mode="residual_tanh",
        rgb_residual_scale=0.1,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        if bridge_source != "ram_core":
            raise ValueError(f"Unsupported bridge_source for now: {bridge_source}")
        super().__init__(
            dav2_model,
            feature_keys=feature_keys,
            adapter_dim=adapter_dim,
            rgb_interface_mode=rgb_interface_mode,
            rgb_residual_scale=rgb_residual_scale,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
        if bridge_layers is None:
            bridge_layers = dav2_model.intermediate_layer_idx[dav2_model.encoder]
        self.bridge_source = bridge_source
        self.bridge_feature_keys = tuple(feature_keys)
        self.bridge_layers = tuple(int(layer) for layer in bridge_layers)
        self.bridge_adapter = RawFeatureBridgeAdapter(
            feature_channels=RAW_RAM_BRIDGE_FEATURE_CHANNELS,
            feature_keys=self.bridge_feature_keys,
            target_layers=self.bridge_layers,
            embed_dim=self.dav2.pretrained.embed_dim,
        )

    def _build_bridge_injections(self, feature_dict, *, patch_hw):
        return self.bridge_adapter(feature_dict, patch_hw=patch_hw)


class RawRamRgbBridgeFeatureAdapterDepthModel(RawRamFeatureAdapterDepthModel):
    """
    raw4 -> [R,(Gr+Gb)/2,B] -> RamCore3 BN output -> DAv2 backbone with bridge
        plus decoder-side feature adapters from the same x_cat / ffm_mid / x3 features.
    """

    def __init__(
        self,
        dav2_model,
        *,
        feature_keys=DEFAULT_RGB_BRIDGE_FEATURE_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        adapter_dim=64,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        nn.Module.__init__(self)
        if bridge_source != "ram_core":
            raise ValueError(f"Unsupported bridge_source for now: {bridge_source}")
        unknown_keys = [key for key in feature_keys if key not in RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS]
        if unknown_keys:
            raise ValueError(f"Unsupported feature adapter keys: {unknown_keys}")

        self.ram_core = RamCore3()
        self.feature_adapter_keys = tuple(feature_keys)
        self.dav2 = dav2_model
        self.feature_projector = RAWFeatureProjector(
            feature_channels=RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS,
            feature_keys=self.feature_adapter_keys,
            adapter_dim=adapter_dim,
        )
        decoder_feat_dim = int(self.dav2.depth_head.scratch.output_conv1.in_channels)
        self.merge3 = DepthMergeBlock(decoder_feat_dim, adapter_dim)
        self.merge2 = DepthMergeBlock(decoder_feat_dim, adapter_dim)
        self.merge1 = DepthMergeBlock(decoder_feat_dim, adapter_dim)
        if bridge_layers is None:
            bridge_layers = dav2_model.intermediate_layer_idx[dav2_model.encoder]
        self.bridge_source = bridge_source
        self.bridge_feature_keys = self.feature_adapter_keys
        self.bridge_layers = tuple(int(layer) for layer in bridge_layers)
        self.bridge_adapter = RawFeatureBridgeAdapter(
            feature_channels=RAW_RAM_RGB_BRIDGE_FEATURE_CHANNELS,
            feature_keys=self.bridge_feature_keys,
            target_layers=self.bridge_layers,
            embed_dim=self.dav2.pretrained.embed_dim,
        )
        self.spatial_adapter = CenterPadCropAdapter(sensor_hw=sensor_hw, backbone_hw=backbone_hw)
        _register_imagenet_stats(self)

    def _build_bridge_injections(self, feature_dict, *, patch_hw):
        return self.bridge_adapter(feature_dict, patch_hw=patch_hw)

    def forward_features(self, x_raw):
        x3_in = packed_bayer_to_base_rgb(x_raw)
        x3, feature_dict = self.ram_core.forward_with_features(x3_in)
        x3 = phase1b_tanh_tail_squash(x3)
        feature_dict = {**feature_dict, "x3": x3}
        x_norm = self.spatial_adapter.pad_rgb(x3)
        patch_hw = (
            x_norm.shape[-2] // self.dav2.pretrained.patch_size,
            x_norm.shape[-1] // self.dav2.pretrained.patch_size,
        )
        bridge_injections = self._build_bridge_injections(feature_dict, patch_hw=patch_hw)
        out_features, patch_h, patch_w = self._extract_backbone_features(
            x_norm,
            bridge_injections=bridge_injections,
        )
        adapters = self.feature_projector(
            feature_dict,
            target_sizes={
                "a3": (patch_h, patch_w),
                "a2": (patch_h * 2, patch_w * 2),
                "a1": (patch_h * 4, patch_w * 4),
            },
        )
        depth = self._forward_decoder_with_adapters(out_features, patch_h, patch_w, adapters)
        depth = F.relu(depth).squeeze(1)
        depth = self.spatial_adapter.crop_depth(depth)
        return {"rgb": x3, "depth": depth}


class RawRamBridgeFeatureAdapterLoRADepthModel(RawRamBridgeFeatureAdapterDepthModel):
    def __init__(
        self,
        dav2_model,
        *,
        feature_keys=DEFAULT_FEATURE_ADAPTER_KEYS,
        bridge_layers=None,
        bridge_source="ram_core",
        adapter_dim=64,
        rgb_interface_mode="residual_tanh",
        rgb_residual_scale=0.1,
        lora_block_mode=DEFAULT_LORA_BLOCK_MODE,
        lora_rank=8,
        lora_alpha=16.0,
        sensor_hw=SENSOR_INPUT_HW,
        backbone_hw=BACKBONE_INPUT_HW,
    ):
        if lora_block_mode not in LORA_BLOCK_MODE_CHOICES:
            raise ValueError(f"Unsupported lora_block_mode: {lora_block_mode}")
        super().__init__(
            dav2_model,
            feature_keys=feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            adapter_dim=adapter_dim,
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


def build_raw_ram_feature_adapter_depth_model(
    dav2_model,
    *,
    input_type="raw_ram_feature_adapter",
    feature_keys=DEFAULT_FEATURE_ADAPTER_KEYS,
    bridge_source="ram_core",
    bridge_layers=None,
    adapter_dim=64,
    rgb_interface_mode="residual_tanh",
    rgb_residual_scale=0.1,
    lora_block_mode=DEFAULT_LORA_BLOCK_MODE,
    lora_rank=8,
    lora_alpha=16.0,
    sensor_hw=SENSOR_INPUT_HW,
    backbone_hw=BACKBONE_INPUT_HW,
):
    if input_type not in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
        raise ValueError(f"Unsupported raw feature adapter input_type: {input_type}")
    if input_type == "raw_ram_bridge_feature_adapter":
        return RawRamBridgeFeatureAdapterDepthModel(
            dav2_model,
            feature_keys=feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            adapter_dim=adapter_dim,
            rgb_interface_mode=rgb_interface_mode,
            rgb_residual_scale=rgb_residual_scale,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    if input_type == "raw_ram_rgb_bridge_feature_adapter":
        return RawRamRgbBridgeFeatureAdapterDepthModel(
            dav2_model,
            feature_keys=feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            adapter_dim=adapter_dim,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    if input_type == "raw_ram_bridge_feature_adapter_lora":
        return RawRamBridgeFeatureAdapterLoRADepthModel(
            dav2_model,
            feature_keys=feature_keys,
            bridge_layers=bridge_layers,
            bridge_source=bridge_source,
            adapter_dim=adapter_dim,
            rgb_interface_mode=rgb_interface_mode,
            rgb_residual_scale=rgb_residual_scale,
            lora_block_mode=lora_block_mode,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            sensor_hw=sensor_hw,
            backbone_hw=backbone_hw,
        )
    return RawRamFeatureAdapterDepthModel(
        dav2_model,
        feature_keys=feature_keys,
        adapter_dim=adapter_dim,
        rgb_interface_mode=rgb_interface_mode,
        rgb_residual_scale=rgb_residual_scale,
        sensor_hw=sensor_hw,
        backbone_hw=backbone_hw,
    )
