import argparse
import contextlib
import json
import logging
import math
import os
import pprint
import random
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from depth_anything_v2.dpt import DepthAnythingV2
from finetune_stf.dataset.eth3d import (
    DEFAULT_ETH3D_ROOT,
    ETH3D_FAST_EVAL_BACKENDS,
    ETH3DValRGB,
    ETH3DValRaw,
)
from finetune_stf.dataset.kitti_eval import DEFAULT_KITTI_BASE, DEFAULT_KITTI_VAL_SPLIT, KITTIEval
from finetune_stf.dataset.lod_raw import (
    DEFAULT_LOD_DAY_MANIFEST,
    DEFAULT_LOD_NIGHT_MANIFEST,
    DEFAULT_LOD_ROOT,
    LODRGB,
    LODRaw,
)
from finetune_stf.dataset.nyu_eval import DEFAULT_NYU_DIR, NYUv2Eval
from finetune_stf.dataset.robotcar import (
    DEFAULT_ROBOTCAR_ROOT,
    ROBOTCAR_FAST_EVAL_BACKENDS,
    RobotCarValRGB,
    RobotCarValRaw,
)
from finetune_stf.dataset.raw_utils import DEFAULT_RAW_NPZ_ROOT
from finetune_stf.dataset.stf import DEFAULT_STF_ROOT, STF
from finetune_stf.dataset.stf_raw import STF_RAW, STF_RAW_NATIVE_HW
from finetune_stf.dataset.vkitti2 import DEFAULT_TRAIN_LIST as DEFAULT_VKITTI_TRAIN_LIST, VKITTI2
from finetune_stf.models.lora_bridge import (
    DEFAULT_BRIDGE_FEATURE_KEYS,
    DEFAULT_RGB_BRIDGE_FEATURE_KEYS,
    DEFAULT_LORA_BLOCK_MODE,
    LORA_BLOCK_MODE_CHOICES,
    RAW_RAM_BRIDGE_INPUT_TYPES,
    RAW_RAM_BRIDGE_LORA_INPUT_TYPES,
    RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
    RAW_RAM_RGB_BRIDGE_LORA_INPUT_TYPES,
    _iter_vit_blocks,
    build_raw_ram_bridge_depth_model,
    load_bridge_init_weights,
    merge_lora_in_state_dict,
)
from finetune_stf.models.raw_feature_adapter import (
    DEFAULT_FEATURE_ADAPTER_KEYS,
    RAW_RAM_BRIDGE_FEATURE_ADAPTER_INPUT_TYPES,
    RAW_RAM_BRIDGE_FEATURE_ADAPTER_LORA_INPUT_TYPES,
    RAW_RAM_FEATURE_ADAPTER_ONLY_INPUT_TYPES,
    RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    build_raw_ram_feature_adapter_depth_model,
)
from finetune_stf.models.spatial_adapter import (
    BACKBONE_INPUT_HW,
    SENSOR_INPUT_HW,
    build_dav2_padded_rgb_depth_model,
)
from finetune_stf.util.dist_helper import setup_distributed
from finetune_stf.models.raw_ram import (
    FUNCTION_ORDER,
    RAW_RAM_INPUT_TYPES,
    RAW_RAM_RGB_INPUT_TYPES,
    RGB_INTERFACE_HEAD_MODE_CHOICES,
    build_raw_ram_depth_model,
)
from finetune_stf.util.loss import (
    AlignedInverseSigLoss,
    DAv2RelativeLoss,
    ScaleShiftInvariantLoss,
)
from finetune_stf.util.utils import init_log
from finetune_stf.util.viz_dump import (
    collect_fixed_samples,
    collect_fixed_train_source_samples,
    dump_fixed_samples,
    dump_train_source_samples,
)
from foundation.engine.datasets import DEFAULT_HYPERSIM_PROCESSED_BASE, CachedVKITTI2Raw, HypersimProcessedRaw, VKITTI2Raw
from foundation.engine.models import build_dav2_raw_naive_depth_model


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}
DEFAULT_BRIDGE_LAYERS_BY_ENCODER = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
    "vitg": [9, 19, 29, 39],
}

METRIC_KEYS = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "log10",
    "silog",
    "silog_x100",
    "d1",
    "d2",
    "d3",
    "edge_sobel_l1",
    "edge_overlap_iou",
)
RAW_PACKED_INPUT_TYPES = ("raw_packed",)
BRIDGE_FEATURE_KEY_CHOICES = tuple(dict.fromkeys((*DEFAULT_BRIDGE_FEATURE_KEYS, *DEFAULT_RGB_BRIDGE_FEATURE_KEYS)))
RAW_MODEL_INPUT_TYPES = (
    "raw",
    *RAW_PACKED_INPUT_TYPES,
    *RAW_RAM_INPUT_TYPES,
    *RAW_RAM_BRIDGE_INPUT_TYPES,
    *RAW_RAM_RGB_INPUT_TYPES,
    *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
    *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
)
RGB_EVAL_INPUT_TYPES = (
    "rgb",
    *RAW_PACKED_INPUT_TYPES,
    *RAW_RAM_INPUT_TYPES,
    *RAW_RAM_BRIDGE_INPUT_TYPES,
    *RAW_RAM_RGB_INPUT_TYPES,
    *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
    *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
)
PHASE1_BNCLEAN_GUARDED_INPUT_TYPES = (
    *RAW_RAM_RGB_INPUT_TYPES,
    *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
)
ETH3D_EVAL_MODE_CHOICES = ("fast", "full", "both")
ETH3D_FAST_EVAL_BACKEND_CHOICES = ETH3D_FAST_EVAL_BACKENDS
ROBOTCAR_EVAL_MODE_CHOICES = ("fast", "full", "both")
ROBOTCAR_FAST_EVAL_BACKEND_CHOICES = ROBOTCAR_FAST_EVAL_BACKENDS
DEFAULT_ROBOTCAR_NIGHT_ROOT = "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo"
DEFAULT_ROBOTCAR_NIGHT_MANIFEST_NAME = "robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv"
KITTI_EVAL_PROTOCOL_CHOICES = ("rgb_pretrained_ref", "rgb_checkpoint_decoder")
BEST_METRIC_CHOICES = ("stf", "kitti", "eth3d", "robotcar", "robotcar_day", "robotcar_night", "avg4")
RAW_MIX_SOURCE_CHOICES = ("vkitti", "hypersim", "lod", "lod_day", "lod_night")
DEFAULT_HEAVY_SAVE_ROOT = "/mnt/drive/3333_raw/0000_exp_ckpt"


def resolve_heavy_save_path(save_path, heavy_save_root):
    if not heavy_save_root:
        return save_path
    exp_name = os.path.basename(os.path.normpath(save_path))
    if not exp_name:
        raise ValueError(f"Could not derive experiment name from save_path={save_path!r}")
    return os.path.join(heavy_save_root, exp_name)


def parse_csv_list(value, *, field_name):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_float_csv(value, *, field_name):
    try:
        return [float(item) for item in parse_csv_list(value, field_name=field_name)]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain comma-separated floats: {value!r}") from exc


def _validate_dav2_train_mode(mode):
    mode = str(mode)
    if mode in {"none", "decoder", "full"}:
        return
    if re.fullmatch(r"last:\d+", mode):
        return
    if re.fullmatch(r"first:\d+", mode):
        return
    if re.fullmatch(r"range:\d+-\d+", mode):
        return
    raise ValueError(
        f"Unsupported dav2_train_mode={mode!r}. "
        "Expected one of: none, decoder, full, last:N, first:N, range:a-b"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DAv2 relative depth on STF")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument(
        "--stage",
        default="stf_only",
        choices=["stf_only", "mixed", "vkitti_only", "lod_only", "vkitti_lod", "raw_mix"],
    )
    parser.add_argument(
        "--input-type",
        default="rgb",
        choices=[
            "rgb",
            "raw",
            *RAW_PACKED_INPUT_TYPES,
            *RAW_RAM_INPUT_TYPES,
            *RAW_RAM_RGB_INPUT_TYPES,
            *RAW_RAM_BRIDGE_INPUT_TYPES,
            *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
            *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
        ],
    )
    parser.add_argument("--stf-root", default=DEFAULT_STF_ROOT)
    parser.add_argument("--raw-npz-root", default=DEFAULT_RAW_NPZ_ROOT)
    parser.add_argument("--vkitti-train-list", default=str(DEFAULT_VKITTI_TRAIN_LIST))
    parser.add_argument("--hypersim-processed-base", default=str(DEFAULT_HYPERSIM_PROCESSED_BASE))
    parser.add_argument("--hypersim-train-root", default=None)
    parser.add_argument("--hypersim-train-list", default=None)
    parser.add_argument("--hypersim-train-meta", default=None)
    parser.add_argument("--hypersim-min-depth", default=0.1, type=float)
    parser.add_argument("--hypersim-max-depth", default=80.0, type=float)
    parser.add_argument("--lod-root", default=DEFAULT_LOD_ROOT)
    parser.add_argument("--lod-day-manifest", default=DEFAULT_LOD_DAY_MANIFEST)
    parser.add_argument(
        "--lod-night-manifest",
        default=None,
        help=(
            "Optional night-split LOD manifest; when provided, concatenated with --lod-day-manifest. "
            f"Example: {DEFAULT_LOD_NIGHT_MANIFEST}"
        ),
    )
    parser.add_argument("--pretrained-from", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--bridge-init-from", type=str, default=None)
    parser.add_argument("--input-height", default=512, type=int)
    parser.add_argument("--input-width", default=960, type=int)
    parser.add_argument(
        "--dav2-train-mode",
        default="decoder",
        type=str,
        help="One of: none, decoder, full, last:N, first:N, range:a-b",
    )
    parser.add_argument("--loss-type", default="aligned_sig", choices=["aligned_sig", "ssi", "ssi_grad"])
    parser.add_argument("--loss-lambda-grad", default=2.0, type=float)
    parser.add_argument("--loss-grad-scales", default=4, type=int)
    parser.add_argument("--loss-mask-downsample", default="strict", choices=["strict", "loose"])
    parser.add_argument(
        "--loss-target-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-image target trim-normalization before SSI / grad alignment.",
    )
    parser.add_argument("--loss-norm-min-scale", default=1e-3, type=float)
    parser.add_argument("--min-depth", default=1.0, type=float)
    parser.add_argument("--max-depth", default=80.0, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--bs", default=4, type=int)
    parser.add_argument("--accum-steps", default=1, type=int, help="Gradient accumulation steps; effective bs = bs * accum_steps")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--bridge-lr", default=5e-5, type=float)
    parser.add_argument("--lora-lr", default=5e-5, type=float)
    parser.add_argument(
        "--backbone-layer-decay",
        default=1.0,
        type=float,
        help="Layer-wise lr decay for DAv2 backbone params; 1.0 disables decay.",
    )
    parser.add_argument("--stf-repeat", default=7, type=int)
    parser.add_argument(
        "--lod-per-vkitti",
        default=1,
        type=int,
        help="[vkitti_lod stage] LOD batches between two VKITTI batches",
    )
    parser.add_argument(
        "--lod-fraction",
        default=None,
        type=float,
        help=(
            "[vkitti_lod stage] Optional LOD sampling fraction in (0, 1). "
            "When set, keeps the baseline total steps from --lod-per-vkitti and changes only the LOD/VKITTI mix."
        ),
    )
    parser.add_argument(
        "--train-sources",
        default="",
        help="[raw_mix stage] comma-separated sources, e.g. lod_day,lod_night,vkitti,hypersim",
    )
    parser.add_argument(
        "--train-source-ratios",
        default="",
        help="[raw_mix stage] comma-separated non-negative source weights matching --train-sources",
    )
    parser.add_argument(
        "--train-steps-per-epoch",
        default=6696,
        type=int,
        help="[raw_mix stage] fixed micro-step budget per epoch; 6696 matches existing vkitti_lod baselines",
    )
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--log-interval", default=500, type=int)
    parser.add_argument("--norm-mode", default="companded")
    parser.add_argument(
        "--lod-raw-domain-config",
        default="identity",
        help="LOD raw-domain transform: identity, JSON object string, or JSON file path.",
    )
    parser.add_argument("--channel-mode", default="rgb_avg_g")
    parser.add_argument(
        "--rgb-interface-mode",
        default="residual_tanh",
        choices=RGB_INTERFACE_HEAD_MODE_CHOICES,
        help=(
            "4ch->3ch head for raw_ram/raw_ram_bridge/raw_ram_feature_adapter variants. "
            "residual_tanh is the distribution-preserving default; sigmoid keeps the legacy behavior."
        ),
    )
    parser.add_argument(
        "--rgb-residual-scale",
        default=0.1,
        type=float,
        help="Residual scale for residual_tanh / residual_linear RGB interface modes.",
    )
    parser.add_argument("--bridge-source", default="ram_core", choices=["ram_core"])
    parser.add_argument(
        "--bridge-feature-keys",
        nargs="+",
        default=None,
        choices=list(BRIDGE_FEATURE_KEY_CHOICES),
    )
    parser.add_argument("--bridge-layers", nargs="+", type=int, default=None)
    parser.add_argument("--lora-block-mode", default=DEFAULT_LORA_BLOCK_MODE, choices=LORA_BLOCK_MODE_CHOICES)
    parser.add_argument("--lora-rank", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16.0, type=float)
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument(
        "--heavy-save-root",
        default=DEFAULT_HEAVY_SAVE_ROOT,
        help=(
            "Root for large experiment artifacts (.pth and TensorBoard events). "
            "The experiment directory name is derived from --save-path."
        ),
    )
    parser.add_argument(
        "--enable-fixed-viz-dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dump fixed eval samples after each epoch for qualitative edge comparison.",
    )
    parser.add_argument(
        "--enable-train-source-viz-dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dump fixed train-source samples after each epoch for per-source qualitative/loss tracking.",
    )
    parser.add_argument(
        "--train-viz-sources",
        default="auto",
        help="Comma-separated train sources to dump, or auto to use the active training sources.",
    )
    parser.add_argument("--train-viz-n-per-source", default=8, type=int)
    parser.add_argument("--train-viz-seed", default=None, type=int)
    parser.add_argument(
        "--train-viz-rgb-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add an independent RGB DAv2 baseline panel/loss to train_viz. "
            "Uses --pretrained-from unless --train-viz-rgb-baseline-checkpoint is set."
        ),
    )
    parser.add_argument(
        "--train-viz-rgb-baseline-checkpoint",
        default=None,
        help="Optional train.py checkpoint for an online RGB baseline panel/loss.",
    )
    parser.add_argument(
        "--train-viz-rgb-baseline-label",
        default=None,
        help="Label used for the train-viz RGB baseline. Defaults to checkpoint parent directory name.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--debug-max-train-steps", default=None, type=int)
    parser.add_argument("--debug-max-val-samples", default=None, type=int)
    parser.add_argument("--debug-max-kitti-samples", default=None, type=int)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--vkitti-cache-root",
        default=None,
        help="Optional offline VKITTI pseudo-RAW cache root generated by cache_vkitti2_pseudoraw.py.",
    )
    parser.add_argument("--vkitti-hflip-prob", default=0.5, type=float)
    parser.add_argument(
        "--lod-crop-mode",
        default="random",
        choices=("random", "center"),
        help="LOD train crop mode. random=per-sample random crop; center=fixed center crop (legacy).",
    )
    parser.add_argument("--vkitti-randomize-unprocessing", action="store_true", default=True)
    parser.add_argument("--no-vkitti-randomize-unprocessing", action="store_false", dest="vkitti_randomize_unprocessing")
    parser.add_argument(
        "--vkitti-unprocessing-preset",
        default="sensor_linear_dual",
        help=(
            "Pseudo-raw preset for VKITTI2Raw. "
            "Supported presets include: stf_legacy, eth3d_sensor_linear, "
            "robotcar_public_gbrg_generic (public GBRG + generic ranges), "
            "robotcar_subset100_sensor_linear, robotcar_subset100_sensor_linear_fixccm, "
            "robotcar_night_sensor_linear, sensor_linear_dual, "
            "robotcar_day_night_sensor_linear_dual. The robotcar_subset100*, "
            "robotcar_night_sensor_linear, sensor_linear_dual, and "
            "robotcar_day_night_sensor_linear_dual presets use RobotCar statistics "
            "and are not public-only."
        ),
    )
    parser.add_argument(
        "--vkitti-unprocessing-mix-weights",
        default=None,
        help=(
            "Optional mix weights for dual preset. "
            "Examples: '0.3,0.7' or "
            "'eth3d_sensor_linear=0.3,robotcar_subset100_sensor_linear=0.7'."
        ),
    )
    parser.add_argument("--eval-kitti", action="store_true")
    parser.add_argument("--kitti-base", default=DEFAULT_KITTI_BASE)
    parser.add_argument("--kitti-val-split", default=str(DEFAULT_KITTI_VAL_SPLIT))
    parser.add_argument("--kitti-min-depth", default=0.1, type=float)
    parser.add_argument("--kitti-max-depth", default=80.0, type=float)
    parser.add_argument("--eval-nyu", action="store_true")
    parser.add_argument("--nyu-dir", default=DEFAULT_NYU_DIR)
    parser.add_argument("--nyu-min-depth", default=0.001, type=float)
    parser.add_argument("--nyu-max-depth", default=10.0, type=float)
    parser.add_argument("--nyu-max-samples", default=None, type=int)
    parser.add_argument("--eval-eth3d", action="store_true")
    parser.add_argument(
        "--eval-stf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eth3d-root", default=DEFAULT_ETH3D_ROOT)
    parser.add_argument("--eth3d-eval-mode", default="fast", choices=ETH3D_EVAL_MODE_CHOICES)
    parser.add_argument("--eth3d-min-depth", default=0.1, type=float)
    parser.add_argument("--eth3d-max-depth", default=80.0, type=float)
    parser.add_argument("--eth3d-max-samples", default=None, type=int)
    parser.add_argument("--eth3d-norm-mode", default="sensor_linear")
    parser.add_argument("--eth3d-fast-eval-backend", default="proxy", choices=ETH3D_FAST_EVAL_BACKEND_CHOICES)
    parser.add_argument("--eval-robotcar", action="store_true")
    parser.add_argument("--robotcar-root", default=DEFAULT_ROBOTCAR_ROOT)
    parser.add_argument("--robotcar-eval-mode", default="fast", choices=ROBOTCAR_EVAL_MODE_CHOICES)
    parser.add_argument("--robotcar-min-depth", default=0.1, type=float)
    parser.add_argument("--robotcar-max-depth", default=50.0, type=float)
    parser.add_argument("--robotcar-max-samples", default=None, type=int)
    parser.add_argument("--robotcar-norm-mode", default="sensor_linear")
    parser.add_argument(
        "--robotcar-raw-domain-config",
        default="identity",
        help="RobotCar day raw-domain transform: identity, JSON object string, or JSON file path.",
    )
    parser.add_argument("--robotcar-fast-eval-backend", default="sparse", choices=ROBOTCAR_FAST_EVAL_BACKEND_CHOICES)
    parser.add_argument("--eval-robotcar-night", action="store_true")
    parser.add_argument("--robotcar-night-root", default=DEFAULT_ROBOTCAR_NIGHT_ROOT)
    parser.add_argument("--robotcar-night-manifest-name", default=DEFAULT_ROBOTCAR_NIGHT_MANIFEST_NAME)
    parser.add_argument("--robotcar-night-min-depth", default=0.1, type=float)
    parser.add_argument("--robotcar-night-max-depth", default=50.0, type=float)
    parser.add_argument("--robotcar-night-max-samples", default=None, type=int)
    parser.add_argument("--robotcar-night-norm-mode", default="sensor_linear")
    parser.add_argument(
        "--robotcar-night-raw-domain-config",
        default="identity",
        help="RobotCar night raw-domain transform: identity, JSON object string, or JSON file path.",
    )
    parser.add_argument("--robotcar-night-fast-eval-backend", default="sparse", choices=ROBOTCAR_FAST_EVAL_BACKEND_CHOICES)
    parser.add_argument(
        "--kitti-eval-protocol",
        default="rgb_pretrained_ref",
        choices=KITTI_EVAL_PROTOCOL_CHOICES,
        help=(
            "rgb_pretrained_ref: load --pretrained-from RGB DAv2 weights once, never updated (existing behavior). "
            "rgb_checkpoint_decoder: build a separate RGB wrapper and sync DAv2/spatial_adapter weights from the live "
            "training model before each KITTI eval."
        ),
    )
    parser.add_argument(
        "--save-best-stf-only",
        action="store_true",
        default=False,
        help="Legacy no-op option kept for backward compatibility. Use --save-best-checkpoint for best checkpoint writing.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        default=False,
        help="Write best_model.pth whenever --best-metric improves.",
    )
    parser.add_argument("--best-metric", default="stf", choices=BEST_METRIC_CHOICES)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.set_defaults(use_imagenet_norm=True)
    args = parser.parse_args()
    try:
        _validate_dav2_train_mode(args.dav2_train_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.accum_steps < 1:
        parser.error("--accum-steps must be >= 1")
    if args.lod_per_vkitti < 1:
        parser.error("--lod-per-vkitti must be >= 1")
    if args.lod_fraction is not None:
        if args.stage != "vkitti_lod":
            parser.error("--lod-fraction is only valid with --stage vkitti_lod")
        if not (0.0 < args.lod_fraction < 1.0):
            parser.error("--lod-fraction must be in (0, 1)")
    if args.train_steps_per_epoch < 1:
        parser.error("--train-steps-per-epoch must be >= 1")
    try:
        args.train_sources = tuple(parse_csv_list(args.train_sources, field_name="--train-sources"))
        args.train_source_ratios = tuple(parse_float_csv(args.train_source_ratios, field_name="--train-source-ratios"))
    except ValueError as exc:
        parser.error(str(exc))
    if args.stage == "raw_mix":
        if not args.train_sources:
            parser.error("--stage raw_mix requires --train-sources")
        unknown_sources = [source for source in args.train_sources if source not in RAW_MIX_SOURCE_CHOICES]
        if unknown_sources:
            parser.error(f"Unknown raw_mix source(s): {unknown_sources}; choices={RAW_MIX_SOURCE_CHOICES}")
        if len(args.train_source_ratios) != len(args.train_sources):
            parser.error("--train-source-ratios must have the same length as --train-sources")
        if any(weight < 0 for weight in args.train_source_ratios) or sum(args.train_source_ratios) <= 0:
            parser.error("--train-source-ratios must be non-negative and sum to > 0")
        if len(set(args.train_sources)) != len(args.train_sources):
            parser.error(f"--train-sources must not contain duplicates: {args.train_sources}")
    elif args.train_sources or args.train_source_ratios:
        parser.error("--train-sources / --train-source-ratios are only valid with --stage raw_mix")
    if args.lora_rank < 1:
        parser.error("--lora-rank must be >= 1")
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be > 0")
    if not (0.0 < args.backbone_layer_decay <= 1.0):
        parser.error("--backbone-layer-decay must be in (0, 1]")
    if args.rgb_residual_scale < 0:
        parser.error("--rgb-residual-scale must be >= 0")
    if args.train_viz_n_per_source < 0:
        parser.error("--train-viz-n-per-source must be >= 0")
    if args.train_viz_seed is None:
        args.train_viz_seed = args.seed
    if args.train_viz_rgb_baseline_checkpoint:
        baseline_path = Path(args.train_viz_rgb_baseline_checkpoint).expanduser()
        if not baseline_path.is_file():
            parser.error(f"--train-viz-rgb-baseline-checkpoint does not exist: {baseline_path}")
        args.train_viz_rgb_baseline_checkpoint = str(baseline_path.resolve())
        if not args.train_viz_rgb_baseline_label:
            args.train_viz_rgb_baseline_label = baseline_path.parent.name or baseline_path.stem
    if not (0.0 <= args.vkitti_hflip_prob <= 1.0):
        parser.error("--vkitti-hflip-prob must be in [0, 1]")
    if args.vkitti_cache_root and args.input_type not in RAW_MODEL_INPUT_TYPES:
        parser.error("--vkitti-cache-root is only valid with raw-like --input-type")
    if args.kitti_min_depth <= 0 or args.kitti_max_depth <= args.kitti_min_depth:
        parser.error("--kitti-max-depth must be greater than --kitti-min-depth > 0")
    if args.nyu_min_depth <= 0 or args.nyu_max_depth <= args.nyu_min_depth:
        parser.error("--nyu-max-depth must be greater than --nyu-min-depth > 0")
    if args.nyu_max_samples is not None and args.nyu_max_samples < 1:
        parser.error("--nyu-max-samples must be >= 1")
    if args.eth3d_min_depth <= 0 or args.eth3d_max_depth <= args.eth3d_min_depth:
        parser.error("--eth3d-max-depth must be greater than --eth3d-min-depth > 0")
    if args.eth3d_max_samples is not None and args.eth3d_max_samples < 1:
        parser.error("--eth3d-max-samples must be >= 1")
    if args.robotcar_min_depth <= 0 or args.robotcar_max_depth <= args.robotcar_min_depth:
        parser.error("--robotcar-max-depth must be greater than --robotcar-min-depth > 0")
    if args.robotcar_max_samples is not None and args.robotcar_max_samples < 1:
        parser.error("--robotcar-max-samples must be >= 1")
    if args.robotcar_night_min_depth <= 0 or args.robotcar_night_max_depth <= args.robotcar_night_min_depth:
        parser.error("--robotcar-night-max-depth must be greater than --robotcar-night-min-depth > 0")
    if args.robotcar_night_max_samples is not None and args.robotcar_night_max_samples < 1:
        parser.error("--robotcar-night-max-samples must be >= 1")
    if args.hypersim_min_depth <= 0 or args.hypersim_max_depth <= args.hypersim_min_depth:
        parser.error("--hypersim-max-depth must be greater than --hypersim-min-depth > 0")
    if args.stage in ("lod_only", "vkitti_lod") and args.input_type not in (
        "rgb",
        *RAW_PACKED_INPUT_TYPES,
        *RAW_RAM_INPUT_TYPES,
        *RAW_RAM_BRIDGE_INPUT_TYPES,
        *RAW_RAM_RGB_INPUT_TYPES,
        *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
        *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    ):
        parser.error("--stage lod_only / vkitti_lod requires --input-type rgb or a 4-channel packed-raw input type")
    if args.stage == "raw_mix" and args.input_type not in (
        *RAW_PACKED_INPUT_TYPES,
        *RAW_RAM_INPUT_TYPES,
        *RAW_RAM_BRIDGE_INPUT_TYPES,
        *RAW_RAM_RGB_INPUT_TYPES,
        *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
        *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    ):
        parser.error("--stage raw_mix requires a 4-channel packed-raw --input-type")
    if args.input_type in PHASE1_BNCLEAN_GUARDED_INPUT_TYPES and os.environ.get("PHASE1_BNCLEAN_REVIEWED") != "1":
        parser.error(
            "Phase-1 BN-clean guard: raw_ram_rgb/raw_ram_rgb_bridge/raw_ram_rgb_bridge_lora "
            "now feed RamCore3 BN+tanh2.5 output to DAv2, without hard clamp or ImageNet norm. "
            "Re-audit finetune_stf/models/raw_ram.py, finetune_stf/models/lora_bridge.py, and "
            "plans/0519_log_night_only/phase1_lod_night_only_plan.md, then rerun with "
            "PHASE1_BNCLEAN_REVIEWED=1."
        )
    if args.save_best_checkpoint:
        if args.best_metric == "eth3d" and not args.eval_eth3d:
            parser.error("--best-metric eth3d requires --eval-eth3d when --save-best-checkpoint is enabled")
        if args.best_metric == "kitti" and not args.eval_kitti:
            parser.error("--best-metric kitti requires --eval-kitti when --save-best-checkpoint is enabled")
        if args.best_metric in {"robotcar", "robotcar_day"} and not args.eval_robotcar:
            parser.error(
                "--best-metric robotcar/robotcar_day requires --eval-robotcar when --save-best-checkpoint is enabled"
            )
        if args.best_metric == "robotcar_night" and not args.eval_robotcar_night:
            parser.error(
                "--best-metric robotcar_night requires --eval-robotcar-night when --save-best-checkpoint is enabled"
            )
        if args.best_metric == "avg4" and not (
            args.eval_kitti and args.eval_eth3d and args.eval_robotcar and args.eval_robotcar_night
        ):
            parser.error(
                "--best-metric avg4 requires --eval-kitti --eval-eth3d --eval-robotcar "
                "--eval-robotcar-night when --save-best-checkpoint is enabled"
            )
        if args.best_metric == "stf" and not args.eval_stf:
            parser.error("--best-metric stf requires --eval-stf when --save-best-checkpoint is enabled")
    if args.eval_nyu and args.eval_kitti and args.kitti_eval_protocol != "rgb_checkpoint_decoder":
        parser.error("--eval-nyu with --eval-kitti requires --kitti-eval-protocol rgb_checkpoint_decoder")
    if args.bridge_feature_keys is None:
        if args.input_type in RAW_RAM_RGB_BRIDGE_INPUT_TYPES:
            args.bridge_feature_keys = list(DEFAULT_RGB_BRIDGE_FEATURE_KEYS)
        elif args.input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
            args.bridge_feature_keys = list(DEFAULT_FEATURE_ADAPTER_KEYS)
        else:
            args.bridge_feature_keys = list(DEFAULT_BRIDGE_FEATURE_KEYS)
    args.bridge_feature_keys = list(dict.fromkeys(args.bridge_feature_keys))
    if args.input_type in RAW_RAM_RGB_BRIDGE_INPUT_TYPES:
        invalid_keys = [key for key in args.bridge_feature_keys if key == "x4"]
        if invalid_keys:
            parser.error(
                f"--input-type {args.input_type} does not support bridge feature key(s): {invalid_keys}. "
                f"Use {list(DEFAULT_RGB_BRIDGE_FEATURE_KEYS)}"
            )
    elif args.input_type in (*RAW_RAM_BRIDGE_INPUT_TYPES, *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES):
        invalid_keys = [key for key in args.bridge_feature_keys if key == "x3"]
        if invalid_keys:
            parser.error(
                f"--input-type {args.input_type} does not support bridge feature key(s): {invalid_keys}. "
                f"Use {list(DEFAULT_BRIDGE_FEATURE_KEYS)}"
            )
    if args.bridge_layers is not None:
        args.bridge_layers = list(dict.fromkeys(args.bridge_layers))
    elif args.input_type in (
        *RAW_RAM_BRIDGE_INPUT_TYPES,
        *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
        *RAW_RAM_BRIDGE_FEATURE_ADAPTER_INPUT_TYPES,
    ):
        args.bridge_layers = list(DEFAULT_BRIDGE_LAYERS_BY_ENCODER[args.encoder])
    args.heavy_save_path = resolve_heavy_save_path(args.save_path, args.heavy_save_root)
    return args


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def build_checkpoint_payload(model, optimizer, epoch, best_metrics, best_metric):
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return {
        "model": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_metric": str(best_metric),
        "best_metrics": {name: float(value) for name, value in best_metrics.items()},
        "best_abs_rel": float(best_metrics.get("stf", float("inf"))),
        "best_kitti_abs_rel": float(best_metrics.get("kitti", float("inf"))),
        "best_eth3d_abs_rel": float(best_metrics.get("eth3d", float("inf"))),
        "best_robotcar_abs_rel": float(best_metrics.get("robotcar", float("inf"))),
        "best_robotcar_day_abs_rel": float(best_metrics.get("robotcar_day", best_metrics.get("robotcar", float("inf")))),
        "best_robotcar_night_abs_rel": float(best_metrics.get("robotcar_night", float("inf"))),
        "best_avg4_abs_rel": float(best_metrics.get("avg4", float("inf"))),
    }


def save_checkpoint(path, model, optimizer, epoch, best_metrics, best_metric):
    torch.save(build_checkpoint_payload(model, optimizer, epoch, best_metrics, best_metric), path)


def get_best_metrics_from_resume(resume):
    best_metrics = {name: float("inf") for name in BEST_METRIC_CHOICES}
    if not isinstance(resume, dict):
        return best_metrics

    resume_best_metrics = resume.get("best_metrics")
    if isinstance(resume_best_metrics, dict):
        for name in BEST_METRIC_CHOICES:
            if name in resume_best_metrics:
                best_metrics[name] = float(resume_best_metrics[name])

    if "best_abs_rel" in resume:
        best_metrics["stf"] = min(best_metrics["stf"], float(resume["best_abs_rel"]))
    if "best_kitti_abs_rel" in resume:
        best_metrics["kitti"] = min(best_metrics["kitti"], float(resume["best_kitti_abs_rel"]))
    if "best_eth3d_abs_rel" in resume:
        best_metrics["eth3d"] = min(best_metrics["eth3d"], float(resume["best_eth3d_abs_rel"]))
    if "best_robotcar_abs_rel" in resume:
        best_metrics["robotcar"] = min(best_metrics["robotcar"], float(resume["best_robotcar_abs_rel"]))
        best_metrics["robotcar_day"] = min(best_metrics["robotcar_day"], float(resume["best_robotcar_abs_rel"]))
    if "best_robotcar_day_abs_rel" in resume:
        best_metrics["robotcar_day"] = min(best_metrics["robotcar_day"], float(resume["best_robotcar_day_abs_rel"]))
        best_metrics["robotcar"] = min(best_metrics["robotcar"], float(resume["best_robotcar_day_abs_rel"]))
    if "best_robotcar_night_abs_rel" in resume:
        best_metrics["robotcar_night"] = min(best_metrics["robotcar_night"], float(resume["best_robotcar_night_abs_rel"]))
    if "best_avg4_abs_rel" in resume:
        best_metrics["avg4"] = min(best_metrics["avg4"], float(resume["best_avg4_abs_rel"]))
    return best_metrics


def resolve_stf_raw_input_mode(input_type):
    if input_type in (
        *RAW_PACKED_INPUT_TYPES,
        *RAW_RAM_INPUT_TYPES,
        *RAW_RAM_BRIDGE_INPUT_TYPES,
        *RAW_RAM_RGB_INPUT_TYPES,
        *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
        *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
    ):
        return "raw_ram"
    return "raw_naive"


def iter_requested_eth3d_modes(args):
    if args.eth3d_eval_mode == "both":
        return ("fast", "full")
    return (args.eth3d_eval_mode,)


def iter_requested_robotcar_modes(args):
    if args.robotcar_eval_mode == "both":
        return ("fast", "full")
    return (args.robotcar_eval_mode,)


def build_model(args):
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    sensor_hw = (args.input_height, args.input_width)
    if args.input_type in ("rgb", "raw"):
        model = build_dav2_padded_rgb_depth_model(model, sensor_hw=sensor_hw, backbone_hw=None)
    elif args.input_type in RAW_PACKED_INPUT_TYPES:
        model = build_dav2_raw_naive_depth_model(
            model,
            freeze_backbone=False,
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif args.input_type in (*RAW_RAM_INPUT_TYPES, *RAW_RAM_RGB_INPUT_TYPES):
        model = build_raw_ram_depth_model(
            model,
            input_type=args.input_type,
            rgb_interface_mode=args.rgb_interface_mode,
            rgb_residual_scale=args.rgb_residual_scale,
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif args.input_type in (*RAW_RAM_BRIDGE_INPUT_TYPES, *RAW_RAM_RGB_BRIDGE_INPUT_TYPES):
        model = build_raw_ram_bridge_depth_model(
            model,
            input_type=args.input_type,
            bridge_source=args.bridge_source,
            bridge_feature_keys=args.bridge_feature_keys,
            bridge_layers=args.bridge_layers,
            rgb_interface_mode=args.rgb_interface_mode,
            rgb_residual_scale=args.rgb_residual_scale,
            lora_block_mode=args.lora_block_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    elif args.input_type in RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES:
        model = build_raw_ram_feature_adapter_depth_model(
            model,
            input_type=args.input_type,
            feature_keys=args.bridge_feature_keys,
            bridge_source=args.bridge_source,
            bridge_layers=args.bridge_layers,
            rgb_interface_mode=args.rgb_interface_mode,
            rgb_residual_scale=args.rgb_residual_scale,
            lora_block_mode=args.lora_block_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            sensor_hw=sensor_hw,
            backbone_hw=None,
        )
    configure_dav2_train_mode(model, args.dav2_train_mode)
    if (
        args.input_type
        in (
            *RAW_RAM_BRIDGE_LORA_INPUT_TYPES,
            *RAW_RAM_RGB_BRIDGE_LORA_INPUT_TYPES,
            *RAW_RAM_BRIDGE_FEATURE_ADAPTER_LORA_INPUT_TYPES,
        )
    ):
        enable_lora_params(model)
    return model


def _parse_train_mode(mode, num_blocks):
    mode = str(mode)
    if mode in ("none", "decoder", "full"):
        return {"kind": mode, "block_indices": (), "train_embeddings": False, "train_norm": False}

    m = re.fullmatch(r"last:(\d+)", mode)
    if m:
        n = int(m.group(1))
        if not (1 <= n <= num_blocks):
            raise ValueError(f"Invalid {mode}: N must be in [1, {num_blocks}]")
        return {
            "kind": "partial",
            "block_indices": tuple(range(num_blocks - n, num_blocks)),
            "train_embeddings": False,
            "train_norm": False,
        }

    m = re.fullmatch(r"first:(\d+)", mode)
    if m:
        n = int(m.group(1))
        if not (1 <= n <= num_blocks):
            raise ValueError(f"Invalid {mode}: N must be in [1, {num_blocks}]")
        return {
            "kind": "partial",
            "block_indices": tuple(range(0, n)),
            "train_embeddings": True,
            "train_norm": False,
        }

    m = re.fullmatch(r"range:(\d+)-(\d+)", mode)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if not (0 <= start <= end < num_blocks):
            raise ValueError(f"Invalid {mode}: require 0 <= a <= b < {num_blocks}")
        return {
            "kind": "partial",
            "block_indices": tuple(range(start, end + 1)),
            "train_embeddings": False,
            "train_norm": False,
        }

    raise ValueError(f"Unsupported dav2_train_mode: {mode!r}")


def enable_lora_params(model):
    dav2_module = model.dav2 if hasattr(model, "dav2") else model
    if not hasattr(dav2_module, "pretrained"):
        return
    for name, param in dav2_module.pretrained.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)


def configure_dav2_train_mode(model, dav2_train_mode):
    dav2_module = model.dav2 if hasattr(model, "dav2") else model
    if not hasattr(dav2_module, "pretrained"):
        raise ValueError("configure_dav2_train_mode expects a module with a DAv2 pretrained backbone")

    pretrained = dav2_module.pretrained
    blocks = list(_iter_vit_blocks(pretrained))
    spec = _parse_train_mode(dav2_train_mode, num_blocks=len(blocks))

    dav2_module.requires_grad_(False)

    if spec["kind"] == "none":
        return
    if spec["kind"] == "decoder":
        dav2_module.depth_head.requires_grad_(True)
        return
    if spec["kind"] == "full":
        dav2_module.requires_grad_(True)
        return

    dav2_module.depth_head.requires_grad_(True)
    for idx in spec["block_indices"]:
        blocks[idx].requires_grad_(True)
    if spec["train_embeddings"]:
        pretrained.patch_embed.requires_grad_(True)
        if hasattr(pretrained, "cls_token"):
            pretrained.cls_token.requires_grad_(True)
        if hasattr(pretrained, "pos_embed"):
            pretrained.pos_embed.requires_grad_(True)
        if hasattr(pretrained, "mask_token"):
            pretrained.mask_token.requires_grad_(True)
        if hasattr(pretrained, "register_tokens") and pretrained.register_tokens is not None:
            pretrained.register_tokens.requires_grad_(True)
    if spec["train_norm"] and hasattr(pretrained, "norm"):
        pretrained.norm.requires_grad_(True)


def build_rgb_reference_eval_model(args):
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model = build_dav2_padded_rgb_depth_model(model)
    configure_dav2_train_mode(model, "none")
    load_initial_weights(model, args.pretrained_from, input_type="rgb")
    return model


def build_rgb_decoder_eval_model(args):
    """Build the RGB wrapper used by checkpoint-decoder RGB eval datasets.

    Weights are NOT loaded here; sync_rgb_decoder_eval_model must be called
    once per eval stage to copy the live training model's compatible parameters
    into the wrapper before all RGB checkpoint-decoder eval datasets run.
    """
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model = build_dav2_padded_rgb_depth_model(model)
    configure_dav2_train_mode(model, "none")
    return model


def build_train_viz_rgb_baseline_model(args):
    model = build_rgb_decoder_eval_model(args)
    ckpt_obj = torch.load(args.train_viz_rgb_baseline_checkpoint, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    if any(key.startswith("dav2.") or key.startswith("spatial_adapter.") for key in state_dict):
        status = model.load_compatible_state_dict(state_dict, strict=False)
        missing_dav2 = [key for key in status.missing_keys if key.startswith("dav2.")]
        if missing_dav2:
            raise RuntimeError(
                "RGB baseline checkpoint is incompatible with the RGB DAv2 wrapper; "
                f"missing dav2 keys include: {missing_dav2[:10]}"
            )
        return model, status

    status = model.load_base_dav2_state_dict(state_dict)
    return model, status


_LEGACY_FFM_REMAP = {
    "ram_core.ffm.fuse.0.": "ram_core.ffm.conv1.",
    "ram_core.ffm.fuse.1.": "ram_core.ffm.conv2.",
    "ram_core.ffm.fuse.2.": "ram_core.ffm.conv3.",
    "ram_core.ffm.fuse.3.": "ram_core.ffm.out_conv.",
    "ram_core.ffm.fuse.4.": "ram_core.ffm.out_bn.",
}


def remap_legacy_ffm_keys(state_dict):
    output = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _LEGACY_FFM_REMAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        output[new_key] = value
    return output


def sync_rgb_decoder_eval_model(rgb_decoder_eval_model, live_model, *, logger, rank, sync_tag):
    """Copy compatible (DAv2 + spatial_adapter) parameters from the live model into the RGB wrapper.

    If the live model wraps any attention layers in `LoRALinear`, fold the LoRA delta
    `(alpha/rank) * B @ A` into the orig weight before loading, so the eval wrapper
    (plain DAv2, no LoRA) receives the *effective* trained attention weights rather
    than re-introducing the un-merged frozen base. Without this merge,
    `…attn.qkv.orig.weight` would be reported as an unexpected key while the eval
    model's `…attn.qkv.weight` would stay at random init for every LoRA-wrapped
    block — which made every kitti_val number in 0426_1311_lora_decoder bogus.
    """
    source = live_model.module if hasattr(live_model, "module") else live_model
    state_dict = strip_module_prefix(source.state_dict())
    if any(key.startswith("ram_core.ffm.fuse.") for key in state_dict):
        state_dict = remap_legacy_ffm_keys(state_dict)
    if any(key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight") for key in state_dict):
        state_dict = merge_lora_in_state_dict(
            state_dict,
            alpha=getattr(source, "lora_alpha", 16.0),
            rank=getattr(source, "lora_rank", 8),
        )
    status = rgb_decoder_eval_model.load_compatible_state_dict(state_dict, strict=False)
    if rank == 0 and logger is not None:
        logger.info(
            "[EVAL][rgb_decoder] sync tag=%s missing_keys=%d unexpected_keys=%d",
            sync_tag,
            len(status.missing_keys),
            len(status.unexpected_keys),
        )
    return status


def load_initial_weights(model, path, *, input_type="rgb"):
    ckpt_obj = torch.load(path, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    if hasattr(model, "load_base_dav2_state_dict"):
        model.load_base_dav2_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict, strict=True)


def load_optional_bridge_init_weights(model, path):
    ckpt_obj = torch.load(path, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt_obj))
    return load_bridge_init_weights(model, state_dict)


def save_args(args):
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def get_stf_eval_size(args):
    if args.input_type in RAW_MODEL_INPUT_TYPES:
        return STF_RAW_NATIVE_HW
    return (args.input_height, args.input_width)


def resolve_lod_manifest_paths(args):
    manifests = [args.lod_day_manifest]
    if args.lod_night_manifest:
        manifests.append(args.lod_night_manifest)
    return manifests


def describe_rgb_interface(mode, residual_scale):
    if mode == "residual_tanh":
        return f"base_rgb+{residual_scale:g}*tanh(1x1_conv(x4))"
    if mode == "linear_clamp":
        return "clamp(1x1_conv(x4),0,1), rggb_init"
    if mode == "residual_linear":
        return f"base_rgb+{residual_scale:g}*1x1_conv(x4)"
    if mode == "tanh01":
        return "0.5+0.5*tanh(1x1_conv(x4))"
    if mode == "sigmoid":
        return "1x1_conv+sigmoid"
    return str(mode)


def build_datasets(args):
    size = (args.input_height, args.input_width)
    stf_eval_size = get_stf_eval_size(args)
    raw_mix_sources = set(args.train_sources) if args.stage == "raw_mix" else set()
    stf_dataset_cls = STF if args.input_type == "rgb" else STF_RAW
    stf_val_kwargs = {
        "stf_root": args.stf_root,
        "size": stf_eval_size,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
    }
    if args.input_type in RAW_MODEL_INPUT_TYPES:
        stf_val_kwargs.update(
            {
                "raw_npz_root": args.raw_npz_root,
                "norm_mode": args.norm_mode,
                "channel_mode": args.channel_mode,
                "use_imagenet_norm": args.use_imagenet_norm,
                "input_mode": resolve_stf_raw_input_mode(args.input_type),
            }
        )

    datasets = {}
    if args.eval_stf:
        stf_val = stf_dataset_cls("val", merge_test_into_train=False, **stf_val_kwargs)
        datasets["val"] = stf_val
    if args.stage in ("stf_only", "mixed", "vkitti_only"):
        stf_train_kwargs = dict(stf_val_kwargs)
        stf_train_kwargs["size"] = size
        datasets["stf_train"] = stf_dataset_cls("train", merge_test_into_train=True, **stf_train_kwargs)
    if args.stage in ("lod_only", "vkitti_lod") or "lod" in raw_mix_sources:
        lod_manifests = resolve_lod_manifest_paths(args)
        if args.input_type == "rgb":
            datasets["lod_train"] = LODRGB(
                lod_root=args.lod_root,
                manifest_path=lod_manifests,
                size=size,
                mode="train",
                crop_mode=args.lod_crop_mode,
            )
        else:
            datasets["lod_train"] = LODRaw(
                lod_root=args.lod_root,
                manifest_path=lod_manifests,
                size=size,
                norm_mode=args.norm_mode,
                raw_domain_config=args.lod_raw_domain_config,
                mode="train",
                crop_mode=args.lod_crop_mode,
            )
    if "lod_day" in raw_mix_sources:
        datasets["lod_day_train"] = LODRaw(
            lod_root=args.lod_root,
            manifest_path=args.lod_day_manifest,
            size=size,
            norm_mode=args.norm_mode,
            raw_domain_config=args.lod_raw_domain_config,
            mode="train",
            crop_mode=args.lod_crop_mode,
        )
    if "lod_night" in raw_mix_sources:
        if not args.lod_night_manifest:
            raise ValueError("--train-sources includes lod_night but --lod-night-manifest was not provided")
        datasets["lod_night_train"] = LODRaw(
            lod_root=args.lod_root,
            manifest_path=args.lod_night_manifest,
            size=size,
            norm_mode=args.norm_mode,
            raw_domain_config=args.lod_raw_domain_config,
            mode="train",
            crop_mode=args.lod_crop_mode,
        )
    if args.eval_kitti:
        if args.input_type not in RGB_EVAL_INPUT_TYPES:
            raise ValueError(f"--eval-kitti is only supported for rgb/raw-like input types right now, got {args.input_type}")
        datasets["kitti_val"] = KITTIEval(
            filelist_path=args.kitti_val_split,
            kitti_base=args.kitti_base,
            size=size,
            min_depth=args.kitti_min_depth,
            max_depth=args.kitti_max_depth,
            input_type="rgb",
        )
    if args.eval_nyu:
        if args.input_type not in RGB_EVAL_INPUT_TYPES:
            raise ValueError(f"--eval-nyu is only supported for rgb/raw-like input types right now, got {args.input_type}")
        datasets["nyu_val"] = NYUv2Eval(
            nyu_dir=args.nyu_dir,
            size=size,
            min_depth=args.nyu_min_depth,
            max_depth=args.nyu_max_depth,
        )
    if args.eval_eth3d:
        if args.input_type == "rgb":
            eth3d_dataset_cls = ETH3DValRGB
            eth3d_kwargs = {
                "fast_eval_backend": args.eth3d_fast_eval_backend,
            }
        else:
            eth3d_dataset_cls = ETH3DValRaw
            eth3d_kwargs = {
                "norm_mode": args.eth3d_norm_mode,
                "channel_mode": args.channel_mode,
                "use_imagenet_norm": args.use_imagenet_norm,
                "input_mode": resolve_stf_raw_input_mode(args.input_type),
                "fast_eval_backend": args.eth3d_fast_eval_backend,
            }
        requested_modes = iter_requested_eth3d_modes(args) if args.eval_only else ("fast",)
        for depth_mode in requested_modes:
            datasets[f"eth3d_val_{depth_mode}"] = eth3d_dataset_cls(
                eth3d_root=args.eth3d_root,
                depth_mode=depth_mode,
                min_depth=args.eth3d_min_depth,
                max_depth=args.eth3d_max_depth,
                **eth3d_kwargs,
            )
    if args.eval_robotcar:
        if args.input_type == "rgb":
            robotcar_dataset_cls = RobotCarValRGB
            robotcar_kwargs = {
                "fast_eval_backend": args.robotcar_fast_eval_backend,
            }
        else:
            robotcar_dataset_cls = RobotCarValRaw
            robotcar_kwargs = {
                "norm_mode": args.robotcar_norm_mode,
                "channel_mode": args.channel_mode,
                "use_imagenet_norm": args.use_imagenet_norm,
                "input_mode": resolve_stf_raw_input_mode(args.input_type),
                "fast_eval_backend": args.robotcar_fast_eval_backend,
                "raw_domain_config": args.robotcar_raw_domain_config,
            }
        requested_modes = iter_requested_robotcar_modes(args) if args.eval_only else ("fast",)
        for depth_mode in requested_modes:
            datasets[f"robotcar_val_{depth_mode}"] = robotcar_dataset_cls(
                robotcar_root=args.robotcar_root,
                depth_mode=depth_mode,
                min_depth=args.robotcar_min_depth,
                max_depth=args.robotcar_max_depth,
                **robotcar_kwargs,
            )
    if args.eval_robotcar_night:
        if args.input_type == "rgb":
            robotcar_night_dataset_cls = RobotCarValRGB
            robotcar_night_kwargs = {
                "fast_eval_backend": args.robotcar_night_fast_eval_backend,
            }
        else:
            robotcar_night_dataset_cls = RobotCarValRaw
            robotcar_night_kwargs = {
                "norm_mode": args.robotcar_night_norm_mode,
                "channel_mode": args.channel_mode,
                "use_imagenet_norm": args.use_imagenet_norm,
                "input_mode": resolve_stf_raw_input_mode(args.input_type),
                "fast_eval_backend": args.robotcar_night_fast_eval_backend,
                "raw_domain_config": args.robotcar_night_raw_domain_config,
            }
        requested_night_modes = iter_requested_robotcar_modes(args) if args.eval_only else ("fast",)
        for depth_mode in requested_night_modes:
            datasets[f"robotcar_night_val_{depth_mode}"] = robotcar_night_dataset_cls(
                robotcar_root=args.robotcar_night_root,
                manifest_name=args.robotcar_night_manifest_name,
                depth_mode=depth_mode,
                min_depth=args.robotcar_night_min_depth,
                max_depth=args.robotcar_night_max_depth,
                **robotcar_night_kwargs,
            )
    if args.stage in ("mixed", "vkitti_only", "vkitti_lod") or "vkitti" in raw_mix_sources:
        if args.input_type in RAW_MODEL_INPUT_TYPES:
            if args.vkitti_cache_root:
                expected_vkitti = VKITTI2Raw(
                    args.vkitti_train_list,
                    mode="train",
                    size=size,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    randomize_unprocessing=args.vkitti_randomize_unprocessing,
                    unprocessing_preset=args.vkitti_unprocessing_preset,
                    unprocessing_mix_weights=args.vkitti_unprocessing_mix_weights,
                    hflip_prob=args.vkitti_hflip_prob,
                )
                expected_desc = expected_vkitti.describe_unprocessing()
                datasets["vkitti_train"] = CachedVKITTI2Raw(
                    args.vkitti_cache_root,
                    filelist_path=args.vkitti_train_list,
                    size=size,
                    expected_num_samples=len(expected_vkitti),
                    expected_preset_hash=expected_desc["preset_hash"],
                )
            else:
                datasets["vkitti_train"] = VKITTI2Raw(
                    args.vkitti_train_list,
                    mode="train",
                    size=size,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    randomize_unprocessing=args.vkitti_randomize_unprocessing,
                    unprocessing_preset=args.vkitti_unprocessing_preset,
                    unprocessing_mix_weights=args.vkitti_unprocessing_mix_weights,
                    hflip_prob=args.vkitti_hflip_prob,
                )
        else:
            datasets["vkitti_train"] = VKITTI2(
                args.vkitti_train_list,
                mode="train",
                size=size,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
            )
    if "hypersim" in raw_mix_sources:
        datasets["hypersim_train"] = HypersimProcessedRaw(
            filelist_path=args.hypersim_train_list,
            processed_base=args.hypersim_processed_base,
            split="train",
            split_root=args.hypersim_train_root,
            metadata_path=args.hypersim_train_meta,
            mode="train",
            size=size,
            min_depth=args.hypersim_min_depth,
            max_depth=args.hypersim_max_depth,
            randomize_unprocessing=args.vkitti_randomize_unprocessing,
            unprocessing_preset=args.vkitti_unprocessing_preset,
            unprocessing_mix_weights=args.vkitti_unprocessing_mix_weights,
            hflip_prob=args.vkitti_hflip_prob,
        )
    return datasets


def build_loader(dataset, sampler, batch_size, num_workers, loader_kwargs, *, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        **loader_kwargs,
    )


def build_mixed_schedule(num_stf_steps, stf_per_vkitti):
    if num_stf_steps <= 0:
        return []
    if stf_per_vkitti < 1:
        raise ValueError(f"stf_per_vkitti must be >= 1, got {stf_per_vkitti}")

    num_vkitti_steps = max(1, num_stf_steps // stf_per_vkitti)
    target_positions = [
        min(max(((idx + 1) * num_stf_steps) // (num_vkitti_steps + 1), 1), num_stf_steps)
        for idx in range(num_vkitti_steps)
    ]

    schedule = []
    position_iter = iter(target_positions)
    next_position = next(position_iter, None)
    for stf_idx in range(1, num_stf_steps + 1):
        schedule.append("stf")
        if next_position is not None and stf_idx >= next_position:
            schedule.append("vkitti")
            next_position = next(position_iter, None)
    return schedule


def build_lod_vkitti_schedule(num_lod_steps, lod_per_vkitti):
    if num_lod_steps <= 0:
        return []
    if lod_per_vkitti < 1:
        raise ValueError(f"lod_per_vkitti must be >= 1, got {lod_per_vkitti}")

    num_vkitti_steps = max(1, num_lod_steps // lod_per_vkitti)
    target_positions = [
        min(max(((idx + 1) * num_lod_steps) // (num_vkitti_steps + 1), 1), num_lod_steps)
        for idx in range(num_vkitti_steps)
    ]

    schedule = []
    position_iter = iter(target_positions)
    next_position = next(position_iter, None)
    for lod_idx in range(1, num_lod_steps + 1):
        schedule.append("lod")
        if next_position is not None and lod_idx >= next_position:
            schedule.append("vkitti")
            next_position = next(position_iter, None)
    return schedule


def build_raw_mix_schedule(sources, weights, steps_per_epoch):
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    if len(sources) != len(weights):
        raise ValueError("sources and weights must have the same length")
    if not sources:
        raise ValueError("raw_mix schedule requires at least one source")
    weights = [float(weight) for weight in weights]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError(f"raw_mix weights must be non-negative and sum to > 0, got {weights}")

    total = float(sum(weights))
    counts = {source: 0 for source in sources}
    schedule = []
    for step_idx in range(int(steps_per_epoch)):
        step_number = step_idx + 1
        best_source = None
        best_deficit = None
        for source, weight in zip(sources, weights):
            desired = step_number * weight / total
            deficit = desired - counts[source]
            if best_deficit is None or deficit > best_deficit:
                best_source = source
                best_deficit = deficit
        schedule.append(best_source)
        counts[best_source] += 1
    return schedule


def raw_mix_dataset_key(source):
    return f"{source}_train"


def build_dataloaders(args, datasets):
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    state = {"samplers": {}}
    if args.stage == "raw_mix":
        source_loaders = {}
        source_batches = {}
        for source in args.train_sources:
            dataset_key = raw_mix_dataset_key(source)
            if dataset_key not in datasets:
                raise KeyError(f"Missing dataset for raw_mix source {source!r}: expected key {dataset_key!r}")
            sampler = DistributedSampler(datasets[dataset_key], shuffle=True)
            loader = build_loader(
                datasets[dataset_key],
                sampler,
                args.bs,
                args.num_workers,
                loader_kwargs,
                drop_last=True,
            )
            source_loaders[source] = loader
            source_batches[source] = len(loader)
            state["samplers"][source] = sampler
        schedule = build_raw_mix_schedule(args.train_sources, args.train_source_ratios, args.train_steps_per_epoch)
        state["mode"] = "raw_mix"
        state["source_loaders"] = source_loaders
        state["source_batches_per_epoch"] = source_batches
        state["schedule"] = schedule
        state["steps_per_epoch"] = len(schedule)
        state["train_sources"] = tuple(args.train_sources)
        state["train_source_ratios"] = tuple(float(item) for item in args.train_source_ratios)
    elif args.stage == "stf_only":
        stf_sampler = DistributedSampler(datasets["stf_train"], shuffle=True)
        stf_loader = build_loader(
            datasets["stf_train"],
            stf_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        state["mode"] = "single"
        state["train_loader"] = stf_loader
        state["samplers"]["stf"] = stf_sampler
        state["steps_per_epoch"] = len(stf_loader)
        state["single_source"] = "stf"
        state["train_sources"] = ("stf",)
    elif args.stage == "lod_only":
        lod_sampler = DistributedSampler(datasets["lod_train"], shuffle=True)
        lod_loader = build_loader(
            datasets["lod_train"],
            lod_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        state["mode"] = "single"
        state["train_loader"] = lod_loader
        state["samplers"]["lod"] = lod_sampler
        state["steps_per_epoch"] = len(lod_loader)
        state["single_source"] = "lod"
        state["train_sources"] = ("lod",)
    elif args.stage == "vkitti_lod":
        lod_sampler = DistributedSampler(datasets["lod_train"], shuffle=True)
        vkitti_sampler = DistributedSampler(datasets["vkitti_train"], shuffle=True)
        lod_loader = build_loader(
            datasets["lod_train"],
            lod_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        vkitti_loader = build_loader(
            datasets["vkitti_train"],
            vkitti_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        baseline_schedule = build_lod_vkitti_schedule(len(lod_loader), args.lod_per_vkitti)
        if args.lod_fraction is None:
            schedule = baseline_schedule
        else:
            schedule = build_raw_mix_schedule(
                ("lod", "vkitti"),
                (args.lod_fraction, 1.0 - args.lod_fraction),
                len(baseline_schedule),
            )
        state["mode"] = "lod_vkitti_mixed"
        state["lod_loader"] = lod_loader
        state["vkitti_loader"] = vkitti_loader
        state["schedule"] = schedule
        state["steps_per_epoch"] = len(schedule)
        state["lod_batches_per_epoch"] = schedule.count("lod")
        state["vkitti_batches_per_epoch"] = schedule.count("vkitti")
        state["samplers"]["lod"] = lod_sampler
        state["samplers"]["vkitti"] = vkitti_sampler
        state["train_sources"] = ("lod", "vkitti")
        state["lod_fraction"] = args.lod_fraction
    elif args.stage == "vkitti_only":
        vkitti_sampler = DistributedSampler(datasets["vkitti_train"], shuffle=True)
        vkitti_loader = build_loader(
            datasets["vkitti_train"],
            vkitti_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        state["mode"] = "single"
        state["train_loader"] = vkitti_loader
        state["samplers"]["vkitti"] = vkitti_sampler
        state["steps_per_epoch"] = len(vkitti_loader)
        state["single_source"] = "vkitti"
        state["train_sources"] = ("vkitti",)
    else:
        stf_sampler = DistributedSampler(datasets["stf_train"], shuffle=True)
        vkitti_sampler = DistributedSampler(datasets["vkitti_train"], shuffle=True)
        stf_loader = build_loader(
            datasets["stf_train"],
            stf_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        vkitti_loader = build_loader(
            datasets["vkitti_train"],
            vkitti_sampler,
            args.bs,
            args.num_workers,
            loader_kwargs,
            drop_last=True,
        )
        schedule = build_mixed_schedule(len(stf_loader), args.stf_repeat)
        state["mode"] = "mixed"
        state["stf_loader"] = stf_loader
        state["vkitti_loader"] = vkitti_loader
        state["schedule"] = schedule
        state["steps_per_epoch"] = len(schedule)
        state["stf_batches_per_epoch"] = len(stf_loader)
        state["vkitti_batches_per_epoch"] = schedule.count("vkitti")
        state["samplers"]["stf"] = stf_sampler
        state["samplers"]["vkitti"] = vkitti_sampler
        state["train_sources"] = ("stf", "vkitti")

    if "val" in datasets:
        valsampler = DistributedSampler(datasets["val"], shuffle=False)
        state["val_loader"] = build_loader(
            datasets["val"],
            valsampler,
            1,
            args.num_workers,
            loader_kwargs,
            drop_last=False,
        )
        state["samplers"]["val"] = valsampler
    if "kitti_val" in datasets:
        kitti_valsampler = DistributedSampler(datasets["kitti_val"], shuffle=False)
        state["kitti_val_loader"] = build_loader(
            datasets["kitti_val"],
            kitti_valsampler,
            1,
            args.num_workers,
            loader_kwargs,
            drop_last=False,
        )
        state["samplers"]["kitti_val"] = kitti_valsampler
    if "nyu_val" in datasets:
        nyu_valsampler = DistributedSampler(datasets["nyu_val"], shuffle=False)
        state["nyu_val_loader"] = build_loader(
            datasets["nyu_val"],
            nyu_valsampler,
            1,
            args.num_workers,
            loader_kwargs,
            drop_last=False,
        )
        state["samplers"]["nyu_val"] = nyu_valsampler
    for depth_mode in ("fast", "full"):
        dataset_key = f"eth3d_val_{depth_mode}"
        if dataset_key not in datasets:
            continue
        sampler = DistributedSampler(datasets[dataset_key], shuffle=False)
        loader_num_workers = args.num_workers
        loader_kwargs_for_dataset = dict(loader_kwargs)
        if depth_mode == "full":
            loader_num_workers = 0
            loader_kwargs_for_dataset = {}
        state[f"{dataset_key}_loader"] = build_loader(
            datasets[dataset_key],
            sampler,
            1,
            loader_num_workers,
            loader_kwargs_for_dataset,
            drop_last=False,
        )
        state["samplers"][dataset_key] = sampler
        state[f"{dataset_key}_num_workers"] = loader_num_workers
    for depth_mode in ("fast", "full"):
        dataset_key = f"robotcar_val_{depth_mode}"
        if dataset_key not in datasets:
            continue
        sampler = DistributedSampler(datasets[dataset_key], shuffle=False)
        loader_num_workers = args.num_workers
        loader_kwargs_for_dataset = dict(loader_kwargs)
        if depth_mode == "full":
            loader_num_workers = 0
            loader_kwargs_for_dataset = {}
        state[f"{dataset_key}_loader"] = build_loader(
            datasets[dataset_key],
            sampler,
            1,
            loader_num_workers,
            loader_kwargs_for_dataset,
            drop_last=False,
        )
        state["samplers"][dataset_key] = sampler
        state[f"{dataset_key}_num_workers"] = loader_num_workers
    for depth_mode in ("fast", "full"):
        dataset_key = f"robotcar_night_val_{depth_mode}"
        if dataset_key not in datasets:
            continue
        sampler = DistributedSampler(datasets[dataset_key], shuffle=False)
        loader_num_workers = args.num_workers
        loader_kwargs_for_dataset = dict(loader_kwargs)
        if depth_mode == "full":
            loader_num_workers = 0
            loader_kwargs_for_dataset = {}
        state[f"{dataset_key}_loader"] = build_loader(
            datasets[dataset_key],
            sampler,
            1,
            loader_num_workers,
            loader_kwargs_for_dataset,
            drop_last=False,
        )
        state["samplers"][dataset_key] = sampler
        state[f"{dataset_key}_num_workers"] = loader_num_workers
    return state


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def format_seconds(seconds):
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def preview_batch_ids(sample, limit=2):
    if "sample_name" in sample:
        values = sample["sample_name"]
    elif "image_path" in sample:
        values = sample["image_path"]
    else:
        return "n/a"

    if isinstance(values, (list, tuple)):
        preview = [str(item) for item in values[:limit]]
        return ", ".join(preview)

    return str(values)


def get_single_sample_meta(sample, key, default=None):
    value = sample.get(key, default)
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]
    return value


def resolve_batch_target_space(sample, default="metric_depth"):
    value = sample.get("target_space", default)
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        unique_values = {str(item) for item in value}
        if len(unique_values) != 1:
            raise ValueError(f"Mixed target_space values in one batch are not supported yet: {sorted(unique_values)}")
        return next(iter(unique_values))
    return str(value)


def summarize_tensor(tensor, quantile=0.99, max_quantile_elements=1_000_000):
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "p99": 0.0, "max": 0.0}
    quantile_flat = flat
    if flat.numel() > max_quantile_elements:
        stride = max(math.ceil(flat.numel() / max_quantile_elements), 1)
        quantile_flat = flat[::stride]
    return {
        "mean": float(flat.mean().item()),
        "p99": float(torch.quantile(quantile_flat, quantile).item()),
        "max": float(flat.max().item()),
    }


def format_source_running_avgs(source_stats, source_names):
    parts = []
    for source_name in source_names:
        stats = source_stats[source_name]
        if stats["steps"] == 0:
            parts.append(f"{source_name}_avg=n/a (0)")
            continue
        avg = stats["loss_sum"] / max(stats["steps"], 1)
        parts.append(f"{source_name}_avg={avg:.6e} ({stats['steps']})")
    return " ".join(parts)


def format_optional_metric(value, spec):
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    return format(value, spec)


def format_summary_metric(summary, key, spec=".4f"):
    if summary is None:
        return "n/a"
    return format_optional_metric(summary.get(key), spec)


def write_summary_scalars(writer, prefix, summary, epoch):
    if writer is None or summary is None:
        return
    for key, value in summary.items():
        value = float(value)
        if math.isfinite(value):
            writer.add_scalar(f"{prefix}/{key}", value, epoch)


def compute_avg4_abs_rel(metric_values):
    keys = ("kitti", "eth3d", "robotcar_day", "robotcar_night")
    values = [float(metric_values.get(key, float("inf"))) for key in keys]
    if any(not math.isfinite(value) for value in values):
        return float("inf")
    return sum(values) / len(values)


def make_loss_term_accumulator():
    return {
        "steps": 0,
        "total_loss_sum": 0.0,
        "loss_ssi_sum": 0.0,
        "loss_grad_sum": 0.0,
        "loss_grad_weighted_sum": 0.0,
    }


def summarize_loss_terms(loss_info, total_loss, lambda_grad):
    metrics = {}
    if "loss_ssi" in loss_info:
        metrics["loss_ssi"] = float(loss_info["loss_ssi"])
    if "loss_grad" in loss_info:
        loss_grad = float(loss_info["loss_grad"])
        metrics["loss_grad"] = loss_grad
        metrics["loss_grad_weighted"] = float(lambda_grad) * loss_grad
    total_loss = float(total_loss)
    if "loss_grad_weighted" in metrics and abs(total_loss) > 1e-12:
        metrics["loss_grad_share"] = metrics["loss_grad_weighted"] / total_loss
    return metrics


def update_loss_term_accumulator(stats, term_metrics, total_loss):
    if not term_metrics:
        return
    stats["steps"] += 1
    stats["total_loss_sum"] += float(total_loss)
    if "loss_ssi" in term_metrics:
        stats["loss_ssi_sum"] += float(term_metrics["loss_ssi"])
    if "loss_grad" in term_metrics:
        stats["loss_grad_sum"] += float(term_metrics["loss_grad"])
    if "loss_grad_weighted" in term_metrics:
        stats["loss_grad_weighted_sum"] += float(term_metrics["loss_grad_weighted"])


def format_loss_term_summary(term_metrics, running_stats):
    if not term_metrics or running_stats["steps"] == 0:
        return "terms=n/a"

    steps = max(running_stats["steps"], 1)
    avg_total_loss = running_stats["total_loss_sum"] / steps
    avg_loss_ssi = running_stats["loss_ssi_sum"] / steps
    avg_loss_grad = running_stats["loss_grad_sum"] / steps
    avg_loss_grad_weighted = running_stats["loss_grad_weighted_sum"] / steps
    avg_loss_grad_share = None
    if abs(avg_total_loss) > 1e-12:
        avg_loss_grad_share = avg_loss_grad_weighted / avg_total_loss

    return (
        "terms=ssi={ssi} grad={grad} grad_w={grad_w} grad_share={grad_share} "
        "terms_avg=ssi={ssi_avg} grad={grad_avg} grad_w={grad_w_avg} grad_share={grad_share_avg}"
    ).format(
        ssi=format_optional_metric(term_metrics.get("loss_ssi"), ".6e"),
        grad=format_optional_metric(term_metrics.get("loss_grad"), ".6e"),
        grad_w=format_optional_metric(term_metrics.get("loss_grad_weighted"), ".6e"),
        grad_share=format_optional_metric(term_metrics.get("loss_grad_share"), ".3f"),
        ssi_avg=format_optional_metric(avg_loss_ssi, ".6e"),
        grad_avg=format_optional_metric(avg_loss_grad, ".6e"),
        grad_w_avg=format_optional_metric(avg_loss_grad_weighted, ".6e"),
        grad_share_avg=format_optional_metric(avg_loss_grad_share, ".3f"),
    )


def affine_align_disp_1d(gt_depth, pred_disp):
    gt_depth = np.asarray(gt_depth, dtype=np.float64).reshape(-1)
    pred_disp = np.asarray(pred_disp, dtype=np.float64).reshape(-1)
    valid = np.isfinite(gt_depth) & (gt_depth > 0) & np.isfinite(pred_disp)
    if valid.sum() < 2:
        aligned_depth = np.full(gt_depth.shape, np.nan, dtype=np.float64)
        return aligned_depth, {"scale": 0.0, "shift": 0.0, "invalid_aligned_pixels": int(valid.size), "invalid_aligned_ratio": 1.0}

    gt_disp = 1.0 / np.clip(gt_depth[valid], a_min=1e-9, a_max=None)
    x = pred_disp[valid]
    A = np.stack([x, np.ones_like(x)], axis=-1)
    coef, *_ = np.linalg.lstsq(A, gt_disp, rcond=None)
    scale, shift = float(coef[0]), float(coef[1])

    aligned_disp = pred_disp * scale + shift
    aligned_depth = np.full(pred_disp.shape, np.nan, dtype=np.float64)
    pos = np.isfinite(aligned_disp) & (aligned_disp > 0)
    aligned_depth[pos] = 1.0 / aligned_disp[pos]
    invalid_count = int(valid.sum() - np.count_nonzero(valid & pos))
    return aligned_depth, {
        "scale": scale,
        "shift": shift,
        "invalid_aligned_pixels": invalid_count,
        "invalid_aligned_ratio": float(invalid_count / max(int(valid.sum()), 1)),
    }


def sample_bilinear_disparity_at_mask(pred_disp, valid_mask, full_hw):
    if pred_disp.ndim != 2:
        raise ValueError(f"Expected 2D disparity map, got shape {tuple(pred_disp.shape)}")
    coords = torch.nonzero(valid_mask, as_tuple=False)
    if coords.numel() == 0:
        return coords, pred_disp.new_zeros((0,), dtype=pred_disp.dtype)

    src_h, src_w = pred_disp.shape
    full_h, full_w = int(full_hw[0]), int(full_hw[1])
    ys = coords[:, 0].to(dtype=torch.float32)
    xs = coords[:, 1].to(dtype=torch.float32)
    if full_h > 1:
        ys = ys * float(src_h - 1) / float(full_h - 1)
    else:
        ys.zero_()
    if full_w > 1:
        xs = xs * float(src_w - 1) / float(full_w - 1)
    else:
        xs.zero_()

    y0 = torch.floor(ys).to(dtype=torch.long)
    x0 = torch.floor(xs).to(dtype=torch.long)
    y1 = torch.clamp(y0 + 1, max=src_h - 1)
    x1 = torch.clamp(x0 + 1, max=src_w - 1)

    wy = ys - y0.to(dtype=torch.float32)
    wx = xs - x0.to(dtype=torch.float32)
    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    flat = pred_disp.reshape(-1)
    idx00 = y0 * src_w + x0
    idx01 = y0 * src_w + x1
    idx10 = y1 * src_w + x0
    idx11 = y1 * src_w + x1
    samples = (
        flat[idx00] * w00
        + flat[idx01] * w01
        + flat[idx10] * w10
        + flat[idx11] * w11
    )
    return coords, samples


def log_setup(logger, args, datasets, train_state, model):
    total_params, trainable_params = count_parameters(model)
    model_ref = model.module if hasattr(model, "module") else model
    effective_bs = args.bs * args.accum_steps
    optimizer_steps_per_epoch = math.ceil(train_state["steps_per_epoch"] / args.accum_steps)
    logger.info(
        "[SETUP] stage=%s input_type=%s encoder=%s dav2_train_mode=%s epochs=%d bs=%d accum_steps=%d effective_bs=%d lr=%.2e num_workers=%d",
        args.stage,
        args.input_type,
        args.encoder,
        args.dav2_train_mode,
        args.epochs,
        args.bs,
        args.accum_steps,
        effective_bs,
        args.lr,
        args.num_workers,
    )
    logger.info(
        "[SETUP] optimizer_steps_per_epoch=%d micro_steps_per_epoch=%d",
        optimizer_steps_per_epoch,
        train_state["steps_per_epoch"],
    )
    spatial_adapter = getattr(model_ref, "spatial_adapter", None)
    if spatial_adapter is not None:
        logger.info(
            "[SETUP] default_sensor_hw=%s default_backbone_hw=%s allow_dynamic_hw=%s patch_size=%s",
            getattr(spatial_adapter, "sensor_hw", SENSOR_INPUT_HW),
            getattr(spatial_adapter, "backbone_hw", BACKBONE_INPUT_HW),
            getattr(spatial_adapter, "allow_dynamic_hw", False),
            getattr(spatial_adapter, "patch_size", "n/a"),
        )
    else:
        logger.info(
            "[SETUP] sensor_input_hw=%s backbone_input_hw=%s",
            SENSOR_INPUT_HW,
            BACKBONE_INPUT_HW,
        )
    logger.info(
        "[LOSS] type=%s lambda_grad=%.2f grad_scales=%d mask_downsample=%s target_norm=%s norm_min_scale=%.2e",
        args.loss_type,
        args.loss_lambda_grad,
        args.loss_grad_scales,
        args.loss_mask_downsample,
        args.loss_target_normalization,
        args.loss_norm_min_scale,
    )
    lod_manifests = resolve_lod_manifest_paths(args)
    if "stf_train" in datasets:
        if "val" in datasets:
            logger.info(
                "[DATASET] stf_train=%d val=%d merge_test_into_train=%s",
                len(datasets["stf_train"]),
                len(datasets["val"]),
                True,
            )
        else:
            logger.info(
                "[DATASET] stf_train=%d merge_test_into_train=%s",
                len(datasets["stf_train"]),
                True,
            )
    elif "lod_train" in datasets:
        if "val" in datasets:
            logger.info(
                "[DATASET] lod_train=%d lod_manifests=%s val=%d stf_val_input_hw=%s",
                len(datasets["lod_train"]),
                lod_manifests,
                len(datasets["val"]),
                get_stf_eval_size(args),
            )
        else:
            logger.info(
                "[DATASET] lod_train=%d lod_manifests=%s",
                len(datasets["lod_train"]),
                lod_manifests,
            )
    else:
        if "val" in datasets:
            logger.info(
                "[DATASET] val=%d stf_val_input_hw=%s",
                len(datasets["val"]),
                get_stf_eval_size(args),
            )
    if "kitti_val" in datasets:
        logger.info(
            "[DATASET] kitti_val=%d min_depth=%.1f max_depth=%.1f protocol=%s",
            len(datasets["kitti_val"]),
            args.kitti_min_depth,
            args.kitti_max_depth,
            args.kitti_eval_protocol,
        )
    if "nyu_val" in datasets:
        logger.info(
            "[DATASET] nyu_val=%d root=%s min_depth=%.3f max_depth=%.1f protocol=rgb_checkpoint_decoder input_hw=%dx%d native_hw=480x640",
            len(datasets["nyu_val"]),
            args.nyu_dir,
            args.nyu_min_depth,
            args.nyu_max_depth,
            args.input_height,
            args.input_width,
        )
    for depth_mode in ("fast", "full"):
        dataset_key = f"eth3d_val_{depth_mode}"
        if dataset_key not in datasets:
            continue
        logger.info(
            "[DATASET] %s=%d root=%s min_depth=%.1f max_depth=%.1f input_type=%s norm_mode=%s fast_eval_backend=%s loader_workers=%s",
            dataset_key,
            len(datasets[dataset_key]),
            args.eth3d_root,
            args.eth3d_min_depth,
            args.eth3d_max_depth,
            args.input_type,
            args.eth3d_norm_mode if args.input_type != "rgb" else "n/a",
            args.eth3d_fast_eval_backend,
            train_state.get(f"{dataset_key}_num_workers", "n/a"),
        )
    for depth_mode in ("fast", "full"):
        dataset_key = f"robotcar_val_{depth_mode}"
        if dataset_key not in datasets:
            continue
        logger.info(
            "[DATASET] %s=%d root=%s min_depth=%.1f max_depth=%.1f input_type=%s norm_mode=%s raw_domain=%s fast_eval_backend=%s loader_workers=%s",
            dataset_key,
            len(datasets[dataset_key]),
            args.robotcar_root,
            args.robotcar_min_depth,
            args.robotcar_max_depth,
            args.input_type,
            args.robotcar_norm_mode if args.input_type != "rgb" else "n/a",
            getattr(getattr(datasets[dataset_key], "raw_domain_config", None), "describe", lambda: "n/a")(),
            args.robotcar_fast_eval_backend,
            train_state.get(f"{dataset_key}_num_workers", "n/a"),
        )
    for depth_mode in ("fast", "full"):
        dataset_key = f"robotcar_night_val_{depth_mode}"
        if dataset_key not in datasets:
            continue
        logger.info(
            "[DATASET] %s=%d root=%s manifest=%s min_depth=%.1f max_depth=%.1f input_type=%s norm_mode=%s raw_domain=%s fast_eval_backend=%s loader_workers=%s",
            dataset_key,
            len(datasets[dataset_key]),
            args.robotcar_night_root,
            args.robotcar_night_manifest_name,
            args.robotcar_night_min_depth,
            args.robotcar_night_max_depth,
            args.input_type,
            args.robotcar_night_norm_mode if args.input_type != "rgb" else "n/a",
            getattr(getattr(datasets[dataset_key], "raw_domain_config", None), "describe", lambda: "n/a")(),
            args.robotcar_night_fast_eval_backend,
            train_state.get(f"{dataset_key}_num_workers", "n/a"),
        )
    if args.eval_eth3d:
        logger.info(
            "[DATASET] ETH3D eval images use runtime input_hw=(640, 960); spatial_adapter switches dynamically from the configured STF default"
        )
    if args.eval_robotcar:
        logger.info(
            "[DATASET] RobotCar eval images use dataset-native runtime input_hw from the manifest; spatial_adapter switches dynamically from the configured STF default"
        )
    if args.eval_robotcar_night:
        logger.info(
            "[DATASET] RobotCar-night eval uses balanced250 manifest %s under %s; spatial_adapter switches dynamically from the configured STF default",
            args.robotcar_night_manifest_name,
            args.robotcar_night_root,
        )
    if args.input_type in RAW_MODEL_INPUT_TYPES:
        logger.info(
            "[DATASET] raw_npz_root=%s norm_mode=%s channel_mode=%s imagenet_norm=%s rgb_interface_mode=%s rgb_residual_scale=%.3g",
            args.raw_npz_root,
            args.norm_mode,
            args.channel_mode,
            args.use_imagenet_norm,
            args.rgb_interface_mode,
            args.rgb_residual_scale,
        )
    if "lod_train" in datasets:
        logger.info(
            "[DATASET] lod_train_crop_hw=%s lod_crop_mode=%s lod_input=%s lod_root=%s raw_domain=%s target_space=inverse_relative",
            (args.input_height, args.input_width),
            args.lod_crop_mode,
            "rgb" if args.input_type == "rgb" else "raw",
            args.lod_root,
            getattr(getattr(datasets["lod_train"], "raw_domain_config", None), "describe", lambda: "n/a")(),
        )
    for source_name in ("lod_day", "lod_night"):
        dataset_key = f"{source_name}_train"
        if dataset_key in datasets:
            logger.info(
                "[DATASET] %s=%d crop_hw=%s lod_crop_mode=%s lod_root=%s raw_domain=%s target_space=inverse_relative",
                dataset_key,
                len(datasets[dataset_key]),
                (args.input_height, args.input_width),
                args.lod_crop_mode,
                args.lod_root,
                getattr(getattr(datasets[dataset_key], "raw_domain_config", None), "describe", lambda: "n/a")(),
            )
    if "hypersim_train" in datasets:
        hypersim_unproc_desc = datasets["hypersim_train"].describe_unprocessing()
        logger.info(
            "[DATASET] hypersim_train=%d mode=marigold_processed pseudo_raw_online split_root=%s "
            "min_depth=%.2f max_depth=%.1f randomize_unprocessing=%s hflip_prob=%.2f "
            "unprocessing_preset=%s selected_default_sub=%s mix_weights=%s preset_version=%s preset_hash=%s",
            len(datasets["hypersim_train"]),
            datasets["hypersim_train"].split_root,
            args.hypersim_min_depth,
            args.hypersim_max_depth,
            args.vkitti_randomize_unprocessing,
            args.vkitti_hflip_prob,
            hypersim_unproc_desc["unprocessing_preset"],
            hypersim_unproc_desc["default_sub_preset"],
            hypersim_unproc_desc["mix_weights"],
            hypersim_unproc_desc["preset_version"],
            hypersim_unproc_desc["preset_hash"],
        )
    if args.input_type in RAW_PACKED_INPUT_TYPES:
        logger.info(
            "[MODEL] %s packed_bayer4_native -> 1x1 stem -> imagenet norm -> center_pad -> DAv2 -> center_crop",
            args.input_type,
        )
    if args.input_type in RAW_RAM_INPUT_TYPES:
        head_desc = (
            describe_rgb_interface(args.rgb_interface_mode, args.rgb_residual_scale)
            if args.input_type == "raw_ram"
            else "base_rgb+0.1*tanh(residual_head)"
        )
        logger.info(
            "[MODEL] %s functions=%s ram_core_out_channels=4 rgb_interface_head=%s",
            args.input_type,
            FUNCTION_ORDER,
            head_desc,
        )
    if args.input_type in RAW_RAM_RGB_INPUT_TYPES:
        logger.info(
            "[MODEL] %s functions=%s ram_core_out_channels=3 dav2_input=ramcore_bn_tanh25_no_clamp_no_imagenet_norm",
            args.input_type,
            FUNCTION_ORDER,
        )
    if args.input_type in (*RAW_RAM_BRIDGE_INPUT_TYPES, *RAW_RAM_RGB_BRIDGE_INPUT_TYPES):
        bridge_head_desc = (
            "ramcore_bn_tanh25_no_clamp_no_imagenet_norm"
            if args.input_type in RAW_RAM_RGB_BRIDGE_INPUT_TYPES
            else describe_rgb_interface(args.rgb_interface_mode, args.rgb_residual_scale)
        )
        if args.input_type in (*RAW_RAM_BRIDGE_LORA_INPUT_TYPES, *RAW_RAM_RGB_BRIDGE_LORA_INPUT_TYPES):
            logger.info(
                "[MODEL] %s bridge_source=%s bridge_feature_keys=%s bridge_layers=%s rgb_interface_head=%s "
                "dav2_train_mode=%s base_lr=%.2e bridge_lr=%.2e "
                "lora_block_mode=%s lora_blocks=%s lora_rank=%d lora_alpha=%.1f lora_lr=%.2e",
                args.input_type,
                args.bridge_source,
                args.bridge_feature_keys,
                args.bridge_layers,
                bridge_head_desc,
                args.dav2_train_mode,
                args.lr,
                args.bridge_lr,
                args.lora_block_mode,
                getattr(model_ref, "lora_block_indices", ()),
                args.lora_rank,
                args.lora_alpha,
                args.lora_lr,
            )
        else:
            logger.info(
                "[MODEL] %s bridge_source=%s bridge_feature_keys=%s bridge_layers=%s rgb_interface_head=%s "
                "dav2_train_mode=%s base_lr=%.2e bridge_lr=%.2e",
                args.input_type,
                args.bridge_source,
                args.bridge_feature_keys,
                args.bridge_layers,
                bridge_head_desc,
                args.dav2_train_mode,
                args.lr,
                args.bridge_lr,
            )
    if args.input_type in RAW_RAM_BRIDGE_FEATURE_ADAPTER_INPUT_TYPES:
        if args.input_type in RAW_RAM_BRIDGE_FEATURE_ADAPTER_LORA_INPUT_TYPES:
            logger.info(
                "[MODEL] %s bridge_source=%s feature_keys=%s bridge_layers=%s "
                "dav2_train_mode=%s base_lr=%.2e adapter_lr=%.2e "
                "decoder_fusion=path_4,path_3,path_2 image_bridge=%s "
                "lora_block_mode=%s lora_blocks=%s lora_rank=%d lora_alpha=%.1f lora_lr=%.2e",
                args.input_type,
                args.bridge_source,
                args.bridge_feature_keys,
                args.bridge_layers,
                args.dav2_train_mode,
                args.lr,
                args.bridge_lr,
                describe_rgb_interface(args.rgb_interface_mode, args.rgb_residual_scale),
                args.lora_block_mode,
                getattr(model_ref, "lora_block_indices", ()),
                args.lora_rank,
                args.lora_alpha,
                args.lora_lr,
            )
        else:
            logger.info(
                "[MODEL] %s bridge_source=%s feature_keys=%s bridge_layers=%s "
                "dav2_train_mode=%s base_lr=%.2e adapter_lr=%.2e "
                "decoder_fusion=path_4,path_3,path_2 image_bridge=%s",
                args.input_type,
                args.bridge_source,
                args.bridge_feature_keys,
                args.bridge_layers,
                args.dav2_train_mode,
                args.lr,
                args.bridge_lr,
                describe_rgb_interface(args.rgb_interface_mode, args.rgb_residual_scale),
            )
    if args.input_type in RAW_RAM_FEATURE_ADAPTER_ONLY_INPUT_TYPES:
        logger.info(
            "[MODEL] %s feature_keys=%s dav2_train_mode=%s base_lr=%.2e adapter_lr=%.2e "
            "decoder_fusion=path_4,path_3,path_2 image_bridge=%s",
            args.input_type,
            args.bridge_feature_keys,
            args.dav2_train_mode,
            args.lr,
            args.bridge_lr,
            describe_rgb_interface(args.rgb_interface_mode, args.rgb_residual_scale),
        )
    if "vkitti_train" in datasets:
        if args.input_type in RAW_MODEL_INPUT_TYPES:
            vkitti_dataset = datasets["vkitti_train"]
            vkitti_unproc_desc = vkitti_dataset.describe_unprocessing()
            if isinstance(vkitti_dataset, CachedVKITTI2Raw):
                logger.info(
                    "[DATASET] vkitti_train=%d mode=pseudo_raw_cached cache_root=%s k=%d randomized_once seed=%s "
                    "raw_dtype=%s depth_dtype=%s valid_mask_dtype=%s partial=%s tag=%s "
                    "unprocessing_preset=%s selected_default_sub=%s mix_weights=%s preset_version=%s preset_hash=%s",
                    len(vkitti_dataset),
                    vkitti_dataset.cache_root,
                    vkitti_dataset.num_variants,
                    vkitti_dataset.seed,
                    vkitti_dataset.storage.get("raw_dtype"),
                    vkitti_dataset.storage.get("depth_dtype"),
                    vkitti_dataset.storage.get("valid_mask_dtype"),
                    vkitti_dataset.allow_partial,
                    vkitti_dataset.cache_tag,
                    vkitti_unproc_desc["unprocessing_preset"],
                    vkitti_unproc_desc["default_sub_preset"],
                    vkitti_unproc_desc["mix_weights"],
                    vkitti_unproc_desc["preset_version"],
                    vkitti_unproc_desc["preset_hash"],
                )
            else:
                logger.info(
                    "[DATASET] vkitti_train=%d mode=pseudo_raw_online randomize_unprocessing=%s hflip_prob=%.2f "
                    "unprocessing_preset=%s selected_default_sub=%s mix_weights=%s preset_version=%s preset_hash=%s",
                    len(vkitti_dataset),
                    args.vkitti_randomize_unprocessing,
                    args.vkitti_hflip_prob,
                    vkitti_unproc_desc["unprocessing_preset"],
                    vkitti_unproc_desc["default_sub_preset"],
                    vkitti_unproc_desc["mix_weights"],
                    vkitti_unproc_desc["preset_version"],
                    vkitti_unproc_desc["preset_hash"],
                )
        else:
            logger.info(
                "[DATASET] vkitti_train=%d mode=rgb",
                len(datasets["vkitti_train"]),
            )
    if train_state["mode"] == "raw_mix":
        logger.info(
            "[DATALOADER] mode=raw_mix sources=%s ratios=%s source_batches=%s total_steps_per_epoch=%d",
            train_state["train_sources"],
            train_state["train_source_ratios"],
            train_state["source_batches_per_epoch"],
            train_state["steps_per_epoch"],
        )
        logger.info(
            "[DATALOADER] raw_mix schedule_counts=%s",
            {source: train_state["schedule"].count(source) for source in train_state["train_sources"]},
        )
    elif train_state["mode"] == "mixed":
        logger.info(
            "[DATASET] vkitti_train=%d stf_repeat=%d stf_batches=%d vkitti_batches=%d total_steps_per_epoch=%d",
            len(datasets["vkitti_train"]),
            args.stf_repeat,
            train_state["stf_batches_per_epoch"],
            train_state["vkitti_batches_per_epoch"],
            train_state["steps_per_epoch"],
        )
    elif train_state["mode"] == "lod_vkitti_mixed":
        if train_state.get("lod_fraction") is None:
            logger.info(
                "[DATASET] lod_batches=%d vkitti_batches=%d lod_per_vkitti=%d total_steps_per_epoch=%d",
                train_state["lod_batches_per_epoch"],
                train_state["vkitti_batches_per_epoch"],
                args.lod_per_vkitti,
                train_state["steps_per_epoch"],
            )
        else:
            logger.info(
                "[DATASET] lod_batches=%d vkitti_batches=%d lod_fraction=%.4f baseline_lod_per_vkitti=%d total_steps_per_epoch=%d",
                train_state["lod_batches_per_epoch"],
                train_state["vkitti_batches_per_epoch"],
                train_state["lod_fraction"],
                args.lod_per_vkitti,
                train_state["steps_per_epoch"],
            )
    else:
        logger.info(
            "[DATALOADER] mode=single source=%s train_steps_per_epoch=%d val_steps=%s kitti_val_steps=%s",
            train_state.get("single_source", "stf"),
            train_state["steps_per_epoch"],
            len(train_state["val_loader"]) if "val_loader" in train_state else "n/a",
            len(train_state["kitti_val_loader"]) if "kitti_val_loader" in train_state else "n/a",
        )
    logger.info(
        "[MODEL] total_params=%d trainable_params=%d frozen_params=%d",
        total_params,
        trainable_params,
        total_params - trainable_params,
    )


def prepare_model_input(sample, args, *, input_type_override=None):
    input_type = args.input_type if input_type_override is None else str(input_type_override)
    if input_type in RAW_MODEL_INPUT_TYPES:
        tensor = sample["raw"] if "raw" in sample else sample["image"]
    else:
        tensor = sample["image"]

    tensor = tensor.cuda(non_blocking=True).float()
    return tensor


def evaluate(
    model,
    valloader,
    args,
    rank,
    writer=None,
    epoch=None,
    logger=None,
    tag="val",
    *,
    min_depth=None,
    max_depth=None,
    max_samples=None,
    writer_prefix="eval",
    model_input_type=None,
):
    model.eval()
    amp_dtype = torch.float16 if getattr(args, "amp_dtype", "bf16") == "fp16" else torch.bfloat16
    min_depth = args.min_depth if min_depth is None else float(min_depth)
    max_depth = args.max_depth if max_depth is None else float(max_depth)
    max_samples = args.debug_max_val_samples if max_samples is None else max_samples
    results = {key: torch.tensor([0.0], device="cuda", dtype=torch.float64) for key in METRIC_KEYS}
    metric_counts = {key: torch.tensor([0.0], device="cuda", dtype=torch.float64) for key in METRIC_KEYS}
    nsamples = torch.tensor([0.0], device="cuda")
    eval_start = time.time()

    if rank == 0 and logger is not None:
        logger.info(
            "[EVAL][%s] start epoch=%s max_samples=%s min_depth=%g max_depth=%g",
            tag,
            "init" if epoch is None else epoch,
            max_samples,
            min_depth,
            max_depth,
        )

    processed = 0
    for sample in valloader:
        if max_samples is not None and processed >= max_samples:
            break

        img = prepare_model_input(sample, args, input_type_override=model_input_type)
        depth = sample["depth"][0].cuda(non_blocking=True).float()
        valid_mask = sample["valid_mask"][0].cuda(non_blocking=True).bool()
        depth_mode = str(get_single_sample_meta(sample, "depth_mode", "full"))
        fast_eval_backend = str(get_single_sample_meta(sample, "fast_eval_backend", "proxy"))

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
            pred_disp = model(img)
            pred_disp = pred_disp.float()

        valid_mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)
        if int(valid_mask.sum().item()) < 10:
            continue

        use_sparse_fast_eval = depth_mode == "fast" and fast_eval_backend == "sparse"
        if use_sparse_fast_eval:
            _, pred_samples = sample_bilinear_disparity_at_mask(pred_disp[0], valid_mask, depth.shape[-2:])
            depth_samples = depth[valid_mask]
            pred_np = pred_samples.detach().cpu().numpy()
            depth_np = depth_samples.detach().cpu().numpy()
            valid_np = np.ones_like(depth_np, dtype=bool)
            aligned_depth, _ = affine_align_disp_1d(depth_np, pred_np)
        else:
            pred_disp = F.interpolate(
                pred_disp[:, None],
                depth.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            pred_np = pred_disp.detach().cpu().numpy()
            depth_np = depth.detach().cpu().numpy()
            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            aligned_depth, _ = affine_align_disp(depth_np, pred_np, valid_np)
        metrics = compute_metrics(
            depth_np,
            aligned_depth,
            valid_np,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if metrics is None:
            continue

        for key in METRIC_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            results[key] += value
            metric_counts[key] += 1.0
        nsamples += 1
        processed += 1

    dist.barrier()
    for key in METRIC_KEYS:
        dist.reduce(results[key], dst=0)
        dist.reduce(metric_counts[key], dst=0)
    dist.reduce(nsamples, dst=0)

    summary = None
    if rank == 0:
        if nsamples.item() == 0:
            raise RuntimeError("Validation produced zero valid samples.")

        summary = {}
        for key in METRIC_KEYS:
            count = float(metric_counts[key].item())
            summary[key] = (results[key] / metric_counts[key]).item() if count > 0 else float("nan")
        if writer is not None and epoch is not None:
            write_summary_scalars(writer, writer_prefix, summary, epoch)
        if logger is not None:
            logger.info(
                "[EVAL][%s] done epoch=%s samples=%d abs_rel=%.4f rmse=%.4f silog=%.4f d1=%.4f "
                "edge_l1=%s edge_iou=%s elapsed=%s",
                tag,
                "init" if epoch is None else epoch,
                int(nsamples.item()),
                summary["abs_rel"],
                summary["rmse"],
                summary["silog"],
                summary["d1"],
                format_summary_metric(summary, "edge_sobel_l1"),
                format_summary_metric(summary, "edge_overlap_iou"),
                format_seconds(time.time() - eval_start),
            )

    return summary


def iter_eth3d_eval_loaders(train_state, *, include_full: bool):
    for depth_mode in ("fast", "full"):
        if depth_mode == "full" and not include_full:
            continue
        loader_key = f"eth3d_val_{depth_mode}_loader"
        loader = train_state.get(loader_key)
        if loader is not None:
            yield depth_mode, loader


def iter_robotcar_eval_loaders(train_state, *, include_full: bool):
    for depth_mode in ("fast", "full"):
        if depth_mode == "full" and not include_full:
            continue
        loader_key = f"robotcar_val_{depth_mode}_loader"
        loader = train_state.get(loader_key)
        if loader is not None:
            yield depth_mode, loader


def iter_robotcar_night_eval_loaders(train_state, *, include_full: bool):
    for depth_mode in ("fast", "full"):
        if depth_mode == "full" and not include_full:
            continue
        loader_key = f"robotcar_night_val_{depth_mode}_loader"
        loader = train_state.get(loader_key)
        if loader is not None:
            yield depth_mode, loader


def make_source_accumulator():
    return {
        "loss_sum": 0.0,
        "steps": 0,
        "valid_pred_sum": 0.0,
        "valid_pred_pixels": 0,
        "valid_pred_max": float("-inf"),
    }


def update_source_accumulator(stats, source, loss_value, pred_disp, valid_mask):
    bucket = stats[source]
    bucket["loss_sum"] += float(loss_value)
    bucket["steps"] += 1

    valid_pred = pred_disp.detach()[valid_mask]
    if valid_pred.numel() == 0:
        return

    bucket["valid_pred_sum"] += float(valid_pred.sum().item())
    bucket["valid_pred_pixels"] += int(valid_pred.numel())
    bucket["valid_pred_max"] = max(bucket["valid_pred_max"], float(valid_pred.max().item()))


def reduce_epoch_stats(running_loss, used_steps, source_stats, device):
    if dist.is_initialized():
        dist.barrier()

    source_names = tuple(source_stats.keys())
    payload_sum_values = [running_loss, float(used_steps)]
    payload_max_values = []
    for source_name in source_names:
        stats = source_stats[source_name]
        payload_sum_values.extend(
            [
                stats["loss_sum"],
                float(stats["steps"]),
                stats["valid_pred_sum"],
                float(stats["valid_pred_pixels"]),
            ]
        )
        payload_max_values.append(stats["valid_pred_max"])

    payload_sum = torch.tensor(payload_sum_values, device=device, dtype=torch.float64)
    payload_max = torch.tensor(payload_max_values, device=device, dtype=torch.float64)

    if dist.is_initialized():
        dist.all_reduce(payload_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(payload_max, op=dist.ReduceOp.MAX)

    def unpack(offset):
        loss_sum = payload_sum[offset + 0].item()
        steps = int(payload_sum[offset + 1].item())
        valid_pred_sum = payload_sum[offset + 2].item()
        valid_pred_pixels = int(payload_sum[offset + 3].item())
        return {
            "loss_sum": loss_sum,
            "steps": steps,
            "valid_pred_sum": valid_pred_sum,
            "valid_pred_pixels": valid_pred_pixels,
        }

    source_summary = {}
    offset = 2
    for idx, source_name in enumerate(source_names):
        stats = unpack(offset)
        stats["valid_pred_max"] = payload_max[idx].item()
        source_summary[source_name] = stats
        offset += 4

    return {
        "running_loss": payload_sum[0].item(),
        "used_steps": int(payload_sum[1].item()),
        "source": source_summary,
    }


def fetch_train_sample(train_state):
    mode = train_state["mode"]
    if mode == "single":
        return next(train_state["train_iter"]), train_state.get("single_source", "stf")

    source = train_state["schedule"][train_state["schedule_index"]]
    train_state["schedule_index"] += 1

    if mode == "raw_mix":
        try:
            return next(train_state["source_iters"][source]), source
        except StopIteration:
            train_state["source_iters"][source] = iter(train_state["source_loaders"][source])
            return next(train_state["source_iters"][source]), source

    if mode == "mixed":
        if source == "stf":
            return next(train_state["stf_iter"]), source
        iterator_key = "vkitti_iter"
        loader_key = "vkitti_loader"
    elif mode == "lod_vkitti_mixed":
        if source == "lod":
            iterator_key = "lod_iter"
            loader_key = "lod_loader"
        elif source == "vkitti":
            iterator_key = "vkitti_iter"
            loader_key = "vkitti_loader"
        else:
            raise ValueError(f"Unsupported source in lod_vkitti_mixed schedule: {source!r}")
    else:
        raise ValueError(f"Unsupported train mode: {mode!r}")

    try:
        return next(train_state[iterator_key]), source
    except StopIteration:
        train_state[iterator_key] = iter(train_state[loader_key])
        return next(train_state[iterator_key]), source


def attach_file_logger(logger, log_path):
    log_path = os.path.abspath(log_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == log_path:
            return

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logger.level)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def log_eval_summary(logger, tag, summary):
    logger.info(
        "[EVAL][%s] abs_rel=%.4f rmse=%.4f silog=%.4f d1=%.4f d2=%.4f d3=%.4f edge_l1=%s edge_iou=%s",
        tag,
        summary["abs_rel"],
        summary["rmse"],
        summary["silog"],
        summary["d1"],
        summary["d2"],
        summary["d3"],
        format_summary_metric(summary, "edge_sobel_l1"),
        format_summary_metric(summary, "edge_overlap_iou"),
    )


def _build_backbone_layer_map(model):
    dav2_module = model.dav2 if hasattr(model, "dav2") else model
    if not hasattr(dav2_module, "pretrained"):
        return {}, 0

    pretrained = dav2_module.pretrained
    blocks = list(_iter_vit_blocks(pretrained))
    num_blocks = len(blocks)
    layer_map = {}

    def _mark_params(params, layer_id):
        for param in params:
            if param.requires_grad:
                layer_map[id(param)] = layer_id

    if hasattr(pretrained, "patch_embed"):
        _mark_params(pretrained.patch_embed.parameters(), 0)
    for attr_name in ("cls_token", "pos_embed", "mask_token", "register_tokens"):
        param = getattr(pretrained, attr_name, None)
        if isinstance(param, torch.nn.Parameter) and param.requires_grad:
            layer_map[id(param)] = 0
    for idx, block in enumerate(blocks):
        _mark_params(block.parameters(), idx + 1)
    if hasattr(pretrained, "norm"):
        _mark_params(pretrained.norm.parameters(), num_blocks + 1)
    return layer_map, num_blocks


def _build_layer_decay_param_groups(args, model):
    if args.dav2_train_mode != "full":
        raise ValueError("--backbone-layer-decay < 1 is only supported with --dav2-train-mode full")

    layer_map, num_blocks = _build_backbone_layer_map(model)
    if not layer_map:
        raise ValueError("Failed to resolve DAv2 backbone params for layer decay")

    max_layer_id = num_blocks + 2
    lora_group_lr = args.bridge_lr if args.lora_lr is None else args.lora_lr
    adapter_prefixes = (
        "bridge_adapter.",
        "image_bridge.",
        "feature_projector.",
        "merge1.",
        "merge2.",
        "merge3.",
    )

    groups = []
    key_to_index = {}

    def _append_param(key, lr, param):
        group_index = key_to_index.get(key)
        if group_index is None:
            key_to_index[key] = len(groups)
            groups.append({"params": [param], "lr": lr, "initial_lr": lr})
        else:
            groups[group_index]["params"].append(param)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if ".lora_A." in name or ".lora_B." in name:
            _append_param(("lora",), lora_group_lr, param)
            continue
        if name.startswith(adapter_prefixes):
            _append_param(("adapter",), args.bridge_lr, param)
            continue

        layer_id = layer_map.get(id(param))
        if layer_id is not None:
            lr = args.lr * (args.backbone_layer_decay ** (max_layer_id - layer_id))
            _append_param(("backbone", layer_id), lr, param)
            continue

        _append_param(("base",), args.lr, param)

    return groups


def build_optimizer(args, model):
    if args.backbone_layer_decay < 1.0:
        param_groups = _build_layer_decay_param_groups(args, model)
    elif (
        args.input_type in (
            *RAW_RAM_BRIDGE_INPUT_TYPES,
            *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
            *RAW_RAM_FEATURE_ADAPTER_INPUT_TYPES,
        )
        and hasattr(model, "get_optimizer_param_groups")
    ):
        param_groups = model.get_optimizer_param_groups(
            base_lr=args.lr,
            bridge_lr=args.bridge_lr,
            lora_lr=args.lora_lr,
        )
    else:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        param_groups = [{"params": trainable_params, "lr": args.lr, "initial_lr": args.lr}]

    for group in param_groups:
        group.setdefault("initial_lr", group["lr"])

    return AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        foreach=True,
    )


def build_training_criterion(args, device):
    kwargs = {"min_valid_pixels_per_sample": 128}
    if args.loss_type == "aligned_sig":
        criterion = AlignedInverseSigLoss(**kwargs)
    elif args.loss_type == "ssi":
        criterion = ScaleShiftInvariantLoss(
            use_target_normalization=args.loss_target_normalization,
            norm_min_scale=args.loss_norm_min_scale,
            **kwargs,
        )
    elif args.loss_type == "ssi_grad":
        criterion = DAv2RelativeLoss(
            lambda_grad=args.loss_lambda_grad,
            n_scales=args.loss_grad_scales,
            use_ssi=True,
            use_grad=True,
            mask_downsample=args.loss_mask_downsample,
            use_target_normalization=args.loss_target_normalization,
            norm_min_scale=args.loss_norm_min_scale,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported loss_type={args.loss_type!r}")
    return criterion.cuda(device)


def update_optimizer_lrs(optimizer, scale):
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def set_dav2_train_eval_from_requires_grad(dav2_module):
    dav2_module.eval()
    if any(param.requires_grad for param in dav2_module.depth_head.parameters()):
        dav2_module.depth_head.train()

    pretrained = dav2_module.pretrained
    if hasattr(pretrained, "patch_embed") and any(param.requires_grad for param in pretrained.patch_embed.parameters()):
        pretrained.patch_embed.train()
    for block in _iter_vit_blocks(pretrained):
        if any(param.requires_grad for param in block.parameters()):
            block.train()
    if hasattr(pretrained, "norm") and any(param.requires_grad for param in pretrained.norm.parameters()):
        pretrained.norm.train()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for finetune_stf/train.py")

    warnings.simplefilter("ignore", np.RankWarning)
    set_random_seed(args.seed)

    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ["LOCAL_RANK"])

    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    writer = None

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.heavy_save_path, exist_ok=True)
        attach_file_logger(logger, os.path.join(args.save_path, "train.log"))
        save_args(args)
        all_args = {**vars(args), "ngpus": world_size}
        logger.info("%s\n", pprint.pformat(all_args))
        logger.info("[OUTPUT] save_path=%s heavy_save_path=%s", args.save_path, args.heavy_save_path)
        writer = SummaryWriter(args.heavy_save_path)

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    datasets = build_datasets(args)
    train_state = build_dataloaders(args, datasets)
    if rank == 0 and args.enable_fixed_viz_dump:
        train_state["fixed_viz_samples"] = collect_fixed_samples(train_state)
    else:
        train_state["fixed_viz_samples"] = {}
    if rank == 0 and args.enable_train_source_viz_dump:
        train_state["train_source_viz_samples"] = collect_fixed_train_source_samples(
            train_state,
            datasets,
            args,
            logger=logger,
        )
    else:
        train_state["train_source_viz_samples"] = {}
    valloader = train_state.get("val_loader")
    kitti_valloader = train_state.get("kitti_val_loader")
    nyu_valloader = train_state.get("nyu_val_loader")

    model = build_model(args)
    rgb_decoder_eval_model = None
    kitti_reference_eval_model = None
    fixed_viz_rgb_baseline_model = None
    train_viz_rgb_baseline_model = None
    train_viz_rgb_baseline_status = None
    start_epoch = 0
    best_metrics = {name: float("inf") for name in BEST_METRIC_CHOICES}
    bridge_init_status = None

    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(strip_module_prefix(resolve_model_state(resume)), strict=True)
        start_epoch = int(resume.get("epoch", -1)) + 1
        best_metrics = get_best_metrics_from_resume(resume)
    else:
        load_initial_weights(model, args.pretrained_from, input_type=args.input_type)
        if (
            args.input_type
            in (
                *RAW_RAM_BRIDGE_INPUT_TYPES,
                *RAW_RAM_RGB_BRIDGE_INPUT_TYPES,
                *RAW_RAM_BRIDGE_FEATURE_ADAPTER_INPUT_TYPES,
            )
            and args.bridge_init_from
        ):
            bridge_init_status = load_optional_bridge_init_weights(model, args.bridge_init_from)

    if kitti_valloader is not None and args.kitti_eval_protocol == "rgb_pretrained_ref":
        kitti_reference_eval_model = build_rgb_reference_eval_model(args)
    if nyu_valloader is not None or (
        kitti_valloader is not None and args.kitti_eval_protocol == "rgb_checkpoint_decoder"
    ):
        rgb_decoder_eval_model = build_rgb_decoder_eval_model(args)
    needs_fixed_viz_rgb_baseline = (
        rank == 0
        and args.enable_fixed_viz_dump
        and any(
            split_name in train_state.get("fixed_viz_samples", {})
            for split_name in ("robotcar", "robotcar_night")
        )
    )
    if needs_fixed_viz_rgb_baseline:
        fixed_viz_rgb_baseline_model = kitti_reference_eval_model or build_rgb_reference_eval_model(args)
    train_viz_has_rgb_baseline_inputs = any(
        record.get("rgb_baseline_input") is not None
        for records in train_state.get("train_source_viz_samples", {}).values()
        for record in records
    )
    needs_train_viz_rgb_baseline = (
        rank == 0
        and args.enable_train_source_viz_dump
        and (getattr(args, "train_viz_rgb_baseline", True) or args.train_viz_rgb_baseline_checkpoint)
        and train_state.get("train_source_viz_samples")
        and train_viz_has_rgb_baseline_inputs
    )
    if needs_train_viz_rgb_baseline:
        if args.train_viz_rgb_baseline_checkpoint:
            train_viz_rgb_baseline_model, train_viz_rgb_baseline_status = build_train_viz_rgb_baseline_model(args)
        else:
            train_viz_rgb_baseline_model = (
                kitti_reference_eval_model
                or fixed_viz_rgb_baseline_model
                or build_rgb_reference_eval_model(args)
            )

    model.cuda(local_rank)
    if rgb_decoder_eval_model is not None:
        rgb_decoder_eval_model.cuda(local_rank)
    if kitti_reference_eval_model is not None:
        kitti_reference_eval_model.cuda(local_rank)
    if fixed_viz_rgb_baseline_model is not None and fixed_viz_rgb_baseline_model is not kitti_reference_eval_model:
        fixed_viz_rgb_baseline_model.cuda(local_rank)
    if train_viz_rgb_baseline_model is not None:
        train_viz_rgb_baseline_model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    criterion = build_training_criterion(args, local_rank)
    optimizer = build_optimizer(args, model.module)

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")

    if rank == 0:
        log_setup(logger, args, datasets, train_state, model)
        if bridge_init_status is not None:
            logger.info(
                "[INIT][bridge] from=%s missing=%d unexpected=%d",
                args.bridge_init_from,
                len(bridge_init_status.missing_keys),
                len(bridge_init_status.unexpected_keys),
            )
            if bridge_init_status.unexpected_keys:
                logger.info("[INIT][bridge] unexpected_keys=%s", bridge_init_status.unexpected_keys)
        if kitti_reference_eval_model is not None:
            logger.info(
                "[EVAL][kitti_val] protocol=rgb_pretrained_ref weights=%s (frozen, never updated)",
                args.pretrained_from,
            )
        if rgb_decoder_eval_model is not None:
            logger.info(
                "[EVAL][rgb_decoder] protocol=rgb_checkpoint_decoder weights synced once before each RGB eval stage"
            )
        if fixed_viz_rgb_baseline_model is not None:
            logger.info(
                "[VIZ] fixed RobotCar RGB DAv2 baseline weights=%s",
                args.pretrained_from,
            )
        if train_viz_rgb_baseline_model is not None:
            if args.train_viz_rgb_baseline_checkpoint:
                logger.info(
                    "[TRAIN_VIZ] rgb_baseline checkpoint=%s label=%s missing=%d unexpected=%d",
                    args.train_viz_rgb_baseline_checkpoint,
                    args.train_viz_rgb_baseline_label,
                    len(getattr(train_viz_rgb_baseline_status, "missing_keys", ())),
                    len(getattr(train_viz_rgb_baseline_status, "unexpected_keys", ())),
                )
                unexpected_keys = list(getattr(train_viz_rgb_baseline_status, "unexpected_keys", ()))
                if unexpected_keys:
                    logger.info("[TRAIN_VIZ] rgb_baseline unexpected_keys=%s", unexpected_keys[:20])
            else:
                logger.info(
                    "[TRAIN_VIZ] rgb_baseline weights=%s label=%s",
                    args.pretrained_from,
                    args.train_viz_rgb_baseline_label or f"RGB DAv2 {args.encoder}",
                )

    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

    if args.eval_only:
        if valloader is not None:
            summary = evaluate(model, valloader, args, rank, writer=None, epoch=None, logger=logger, tag="eval_only_stf")
            if rank == 0:
                logger.info("[EVAL][eval_only_stf] summary=%s", summary)
        if rgb_decoder_eval_model is not None:
            sync_rgb_decoder_eval_model(
                rgb_decoder_eval_model, model, logger=logger, rank=rank, sync_tag="eval_only"
            )
        if kitti_valloader is not None:
            kitti_eval_model = (
                rgb_decoder_eval_model
                if args.kitti_eval_protocol == "rgb_checkpoint_decoder"
                else kitti_reference_eval_model
            )
            kitti_eval_tag = f"eval_only_kitti_{args.kitti_eval_protocol}"
            kitti_summary = evaluate(
                kitti_eval_model,
                kitti_valloader,
                args,
                rank,
                writer=None,
                epoch=None,
                logger=logger,
                tag=kitti_eval_tag,
                min_depth=args.kitti_min_depth,
                max_depth=args.kitti_max_depth,
                max_samples=args.debug_max_kitti_samples,
                writer_prefix="eval_kitti",
                model_input_type="rgb",
            )
            if rank == 0:
                logger.info("[EVAL][%s] summary=%s", kitti_eval_tag, kitti_summary)
        if nyu_valloader is not None:
            nyu_eval_tag = "eval_only_nyu_rgb_checkpoint_decoder"
            nyu_summary = evaluate(
                rgb_decoder_eval_model,
                nyu_valloader,
                args,
                rank,
                writer=None,
                epoch=None,
                logger=logger,
                tag=nyu_eval_tag,
                min_depth=args.nyu_min_depth,
                max_depth=args.nyu_max_depth,
                max_samples=args.nyu_max_samples,
                writer_prefix="eval_nyu",
                model_input_type="rgb",
            )
            if rank == 0:
                logger.info("[EVAL][%s] summary=%s", nyu_eval_tag, nyu_summary)
        for depth_mode, eth3d_loader in iter_eth3d_eval_loaders(train_state, include_full=True):
            eth3d_summary = evaluate(
                model,
                eth3d_loader,
                args,
                rank,
                writer=None,
                epoch=None,
                logger=logger,
                tag=f"eval_only_eth3d_{depth_mode}",
                min_depth=args.eth3d_min_depth,
                max_depth=args.eth3d_max_depth,
                max_samples=args.eth3d_max_samples,
                writer_prefix=f"eval_eth3d_{depth_mode}",
                model_input_type=args.input_type,
            )
            if rank == 0:
                logger.info("[EVAL][eval_only_eth3d_%s] summary=%s", depth_mode, eth3d_summary)
        for depth_mode, robotcar_loader in iter_robotcar_eval_loaders(train_state, include_full=True):
            robotcar_summary = evaluate(
                model,
                robotcar_loader,
                args,
                rank,
                writer=None,
                epoch=None,
                logger=logger,
                tag=f"eval_only_robotcar_{depth_mode}",
                min_depth=args.robotcar_min_depth,
                max_depth=args.robotcar_max_depth,
                max_samples=args.robotcar_max_samples,
                writer_prefix=f"eval_robotcar_{depth_mode}",
                model_input_type=args.input_type,
            )
            if rank == 0:
                logger.info("[EVAL][eval_only_robotcar_%s] summary=%s", depth_mode, robotcar_summary)
        for depth_mode, robotcar_night_loader in iter_robotcar_night_eval_loaders(train_state, include_full=True):
            robotcar_night_summary = evaluate(
                model,
                robotcar_night_loader,
                args,
                rank,
                writer=None,
                epoch=None,
                logger=logger,
                tag=f"eval_only_robotcar_night_{depth_mode}",
                min_depth=args.robotcar_night_min_depth,
                max_depth=args.robotcar_night_max_depth,
                max_samples=args.robotcar_night_max_samples,
                writer_prefix=f"eval_robotcar_night_{depth_mode}",
                model_input_type=args.input_type,
            )
            if rank == 0:
                logger.info("[EVAL][eval_only_robotcar_night_%s] summary=%s", depth_mode, robotcar_night_summary)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    steps_per_epoch = train_state["steps_per_epoch"]
    if args.debug_max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, args.debug_max_train_steps)
    optimizer_steps_per_epoch = max(math.ceil(steps_per_epoch / args.accum_steps), 1)
    total_iters = max(args.epochs * optimizer_steps_per_epoch, 1)

    pretrain_summary = None
    if valloader is not None:
        pretrain_summary = evaluate(model, valloader, args, rank, writer=None, epoch=None, logger=logger, tag="pretrain_stf")
    pretrain_kitti_summary = None
    pretrain_nyu_summary = None
    pretrain_eth3d_summaries = {}
    pretrain_robotcar_summaries = {}
    pretrain_robotcar_night_summaries = {}
    if rgb_decoder_eval_model is not None:
        sync_rgb_decoder_eval_model(
            rgb_decoder_eval_model, model, logger=logger, rank=rank, sync_tag="pretrain"
        )
    if kitti_valloader is not None:
        kitti_eval_model = (
            rgb_decoder_eval_model
            if args.kitti_eval_protocol == "rgb_checkpoint_decoder"
            else kitti_reference_eval_model
        )
        pretrain_kitti_tag = f"pretrain_kitti_{args.kitti_eval_protocol}"
        pretrain_kitti_summary = evaluate(
            kitti_eval_model,
            kitti_valloader,
            args,
            rank,
            writer=None,
            epoch=None,
            logger=logger,
            tag=pretrain_kitti_tag,
            min_depth=args.kitti_min_depth,
            max_depth=args.kitti_max_depth,
            max_samples=args.debug_max_kitti_samples,
            writer_prefix="eval_kitti",
            model_input_type="rgb",
        )
    if nyu_valloader is not None:
        pretrain_nyu_summary = evaluate(
            rgb_decoder_eval_model,
            nyu_valloader,
            args,
            rank,
            writer=None,
            epoch=None,
            logger=logger,
            tag="pretrain_nyu_rgb_checkpoint_decoder",
            min_depth=args.nyu_min_depth,
            max_depth=args.nyu_max_depth,
            max_samples=args.nyu_max_samples,
            writer_prefix="eval_nyu",
            model_input_type="rgb",
        )
    for depth_mode, eth3d_loader in iter_eth3d_eval_loaders(train_state, include_full=False):
        pretrain_eth3d_summaries[depth_mode] = evaluate(
            model,
            eth3d_loader,
            args,
            rank,
            writer=None,
            epoch=None,
            logger=logger,
            tag=f"pretrain_eth3d_{depth_mode}",
            min_depth=args.eth3d_min_depth,
            max_depth=args.eth3d_max_depth,
            max_samples=args.eth3d_max_samples,
            writer_prefix=f"eval_eth3d_{depth_mode}",
            model_input_type=args.input_type,
        )
    for depth_mode, robotcar_loader in iter_robotcar_eval_loaders(train_state, include_full=False):
        pretrain_robotcar_summaries[depth_mode] = evaluate(
            model,
            robotcar_loader,
            args,
            rank,
            writer=None,
            epoch=None,
            logger=logger,
            tag=f"pretrain_robotcar_{depth_mode}",
            min_depth=args.robotcar_min_depth,
            max_depth=args.robotcar_max_depth,
            max_samples=args.robotcar_max_samples,
            writer_prefix=f"eval_robotcar_{depth_mode}",
            model_input_type=args.input_type,
        )
    for depth_mode, robotcar_night_loader in iter_robotcar_night_eval_loaders(train_state, include_full=False):
        pretrain_robotcar_night_summaries[depth_mode] = evaluate(
            model,
            robotcar_night_loader,
            args,
            rank,
            writer=None,
            epoch=None,
            logger=logger,
            tag=f"pretrain_robotcar_night_{depth_mode}",
            min_depth=args.robotcar_night_min_depth,
            max_depth=args.robotcar_night_max_depth,
            max_samples=args.robotcar_night_max_samples,
            writer_prefix=f"eval_robotcar_night_{depth_mode}",
            model_input_type=args.input_type,
        )
    if rank == 0:
        if pretrain_summary is not None:
            log_eval_summary(logger, "pretrain_stf", pretrain_summary)
            save_json(
                os.path.join(args.save_path, "pretrain_eval.json"),
                {
                    "checkpoint_source": args.resume_from or args.pretrained_from,
                    "stage": args.stage,
                    "split": "stf_val",
                    "metrics": pretrain_summary,
                },
            )
            if writer is not None:
                for key, value in pretrain_summary.items():
                    writer.add_scalar(f"eval_init/{key}", value, start_epoch)
        if pretrain_kitti_summary is not None:
            log_eval_summary(logger, pretrain_kitti_tag, pretrain_kitti_summary)
            kitti_checkpoint_source = (
                args.pretrained_from
                if args.kitti_eval_protocol == "rgb_pretrained_ref"
                else (args.resume_from or args.pretrained_from)
            )
            save_json(
                os.path.join(args.save_path, "pretrain_eval_kitti.json"),
                {
                    "checkpoint_source": kitti_checkpoint_source,
                    "stage": args.stage,
                    "split": "kitti_val",
                    "protocol": args.kitti_eval_protocol,
                    "metrics": pretrain_kitti_summary,
                },
            )
            if writer is not None:
                for key, value in pretrain_kitti_summary.items():
                    writer.add_scalar(f"eval_init_kitti/{key}", value, start_epoch)
        if pretrain_nyu_summary is not None:
            log_eval_summary(logger, "pretrain_nyu_rgb_checkpoint_decoder", pretrain_nyu_summary)
            save_json(
                os.path.join(args.save_path, "pretrain_eval_nyu.json"),
                {
                    "checkpoint_source": args.resume_from or args.pretrained_from,
                    "stage": args.stage,
                    "split": "nyu_val",
                    "protocol": "rgb_checkpoint_decoder",
                    "metrics": pretrain_nyu_summary,
                },
            )
            if writer is not None:
                for key, value in pretrain_nyu_summary.items():
                    writer.add_scalar(f"eval_init_nyu/{key}", value, start_epoch)
        for depth_mode, eth3d_summary in pretrain_eth3d_summaries.items():
            log_eval_summary(logger, f"pretrain_eth3d_{depth_mode}", eth3d_summary)
            save_json(
                os.path.join(args.save_path, f"pretrain_eval_eth3d_{depth_mode}.json"),
                {
                    "checkpoint_source": args.resume_from or args.pretrained_from,
                    "stage": args.stage,
                    "split": f"eth3d_val_{depth_mode}",
                    "metrics": eth3d_summary,
                },
            )
            if writer is not None:
                for key, value in eth3d_summary.items():
                    writer.add_scalar(f"eval_init_eth3d_{depth_mode}/{key}", value, start_epoch)
        for depth_mode, robotcar_summary in pretrain_robotcar_summaries.items():
            log_eval_summary(logger, f"pretrain_robotcar_{depth_mode}", robotcar_summary)
            save_json(
                os.path.join(args.save_path, f"pretrain_eval_robotcar_{depth_mode}.json"),
                {
                    "checkpoint_source": args.resume_from or args.pretrained_from,
                    "stage": args.stage,
                    "split": f"robotcar_val_{depth_mode}",
                    "metrics": robotcar_summary,
                },
            )
            if writer is not None:
                for key, value in robotcar_summary.items():
                    writer.add_scalar(f"eval_init_robotcar_{depth_mode}/{key}", value, start_epoch)
        for depth_mode, robotcar_night_summary in pretrain_robotcar_night_summaries.items():
            log_eval_summary(logger, f"pretrain_robotcar_night_{depth_mode}", robotcar_night_summary)
            save_json(
                os.path.join(args.save_path, f"pretrain_eval_robotcar_night_{depth_mode}.json"),
                {
                    "checkpoint_source": args.resume_from or args.pretrained_from,
                    "stage": args.stage,
                    "split": f"robotcar_night_val_{depth_mode}",
                    "manifest_name": args.robotcar_night_manifest_name,
                    "metrics": robotcar_night_summary,
                },
            )
            if writer is not None:
                for key, value in robotcar_night_summary.items():
                    writer.add_scalar(f"eval_init_robotcar_night_{depth_mode}/{key}", value, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        for sampler_name, sampler in train_state["samplers"].items():
            if sampler_name in {"val", "kitti_val", "nyu_val"}:
                continue
            sampler.set_epoch(epoch + 1)
        if train_state["mode"] == "raw_mix":
            train_state["source_iters"] = {
                source: iter(loader)
                for source, loader in train_state["source_loaders"].items()
            }
            train_state["schedule_index"] = 0
        elif train_state["mode"] == "mixed":
            train_state["stf_iter"] = iter(train_state["stf_loader"])
            train_state["vkitti_iter"] = iter(train_state["vkitti_loader"])
            train_state["schedule_index"] = 0
        elif train_state["mode"] == "lod_vkitti_mixed":
            train_state["lod_iter"] = iter(train_state["lod_loader"])
            train_state["vkitti_iter"] = iter(train_state["vkitti_loader"])
            train_state["schedule_index"] = 0
        else:
            train_state["train_iter"] = iter(train_state["train_loader"])
        model.train()
        dav2_ref = model.module.dav2 if hasattr(model.module, "dav2") else model.module
        if hasattr(dav2_ref, "pretrained") and hasattr(dav2_ref, "depth_head"):
            set_dav2_train_eval_from_requires_grad(dav2_ref)
        epoch_start_time = time.time()
        last_log_time = epoch_start_time

        if rank == 0:
            logger.info(
                "[EPOCH] start epoch=%d/%d best_%s_abs_rel=%.4f",
                epoch,
                args.epochs,
                args.best_metric,
                best_metrics[args.best_metric],
            )
            if train_state["mode"] == "raw_mix":
                logger.info(
                    "[EPOCH] raw_mix schedule: counts=%s ratios=%s",
                    {source: train_state["schedule"].count(source) for source in train_state["train_sources"]},
                    train_state["train_source_ratios"],
                )
            elif train_state["mode"] == "mixed":
                logger.info(
                    "[EPOCH] mixed schedule: %d STF batches + %d VKITTI2 batches per epoch (target ratio %d:1)",
                    train_state["stf_batches_per_epoch"],
                    train_state["vkitti_batches_per_epoch"],
                    args.stf_repeat,
                )
            elif train_state["mode"] == "lod_vkitti_mixed":
                if train_state.get("lod_fraction") is None:
                    logger.info(
                        "[EPOCH] lod-vkitti schedule: %d LOD batches + %d VKITTI2 batches per epoch (target ratio %d:1)",
                        train_state["lod_batches_per_epoch"],
                        train_state["vkitti_batches_per_epoch"],
                        args.lod_per_vkitti,
                    )
                else:
                    logger.info(
                        "[EPOCH] lod-vkitti schedule: %d LOD batches + %d VKITTI2 batches per epoch (lod_fraction %.4f)",
                        train_state["lod_batches_per_epoch"],
                        train_state["vkitti_batches_per_epoch"],
                        train_state["lod_fraction"],
                    )

        running_loss = 0.0
        used_steps = 0
        optimizer_steps_done = 0
        loss_term_stats = make_loss_term_accumulator()
        source_stats = {source_name: make_source_accumulator() for source_name in train_state["train_sources"]}
        logged_source_stats = set()

        accum_steps = args.accum_steps
        pending_gradients = False
        optimizer.zero_grad(set_to_none=True)

        for step_idx in range(steps_per_epoch):
            window_start_idx = (step_idx // accum_steps) * accum_steps
            window_size = min(accum_steps, steps_per_epoch - window_start_idx)

            sample, source = fetch_train_sample(train_state)

            img = prepare_model_input(sample, args)
            depth = sample["depth"].cuda(non_blocking=True).float()
            valid_mask = sample["valid_mask"].cuda(non_blocking=True).bool()
            target_space = resolve_batch_target_space(sample)
            if rank == 0 and source not in logged_source_stats:
                logger.info(
                    "[BATCH] source=%s target_space=%s input_shape=%s depth_shape=%s valid_shape=%s sample_preview=%s",
                    source,
                    target_space,
                    tuple(img.shape),
                    tuple(depth.shape),
                    tuple(valid_mask.shape),
                    preview_batch_ids(sample),
                )
                if source in {"lod", "lod_day", "lod_night", "vkitti", "hypersim"}:
                    source_tag = source.upper()
                    raw_stats = summarize_tensor(img)
                    valid_count = int(valid_mask.sum().item())
                    target_stats = summarize_tensor(depth[valid_mask]) if valid_count > 0 else {"mean": 0.0, "p99": 0.0, "max": 0.0}
                    logger.info(
                        "[%s][STATS] raw_mean=%.6f raw_p99=%.6f raw_max=%.6f target_mean=%.3f target_p99=%.3f target_max=%.3f valid_pixels=%d",
                        source_tag,
                        raw_stats["mean"],
                        raw_stats["p99"],
                        raw_stats["max"],
                        target_stats["mean"],
                        target_stats["p99"],
                        target_stats["max"],
                        valid_count,
                    )
                logged_source_stats.add(source)

            apply_runtime_hflip = args.input_type not in RAW_MODEL_INPUT_TYPES
            if apply_runtime_hflip and random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            # Determine if this micro-step is the accumulation boundary
            is_accum_boundary = ((step_idx + 1) % accum_steps == 0) or (step_idx + 1 >= steps_per_epoch)

            # Use no_sync for non-boundary steps to skip DDP all-reduce
            sync_ctx = contextlib.nullcontext() if is_accum_boundary else model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=args.amp):
                    pred_disp = model(img)
                loss, loss_info = criterion(pred_disp.float(), depth, valid_mask, target_space=target_space)
                if loss_info["used_samples"] > 0:
                    loss_scaled = loss / window_size
                    if scaler.is_enabled():
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    pending_gradients = True

            if loss_info["used_samples"] > 0:
                loss_value = float(loss.item())
                loss_term_metrics = summarize_loss_terms(loss_info, loss_value, args.loss_lambda_grad)
                running_loss += loss_value
                used_steps += 1
                update_source_accumulator(source_stats, source, loss_value, pred_disp, valid_mask)
                update_loss_term_accumulator(loss_term_stats, loss_term_metrics, loss_value)
            else:
                loss_value = None
                loss_term_metrics = {}

            # Step optimizer and update LR at accumulation boundary
            if is_accum_boundary and pending_gradients:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer_steps_done += 1
                current_iter = epoch * optimizer_steps_per_epoch + optimizer_steps_done - 1
                scale = (1 - min(current_iter, total_iters - 1) / total_iters) ** 0.9
                update_optimizer_lrs(optimizer, scale)
                optimizer.zero_grad(set_to_none=True)
                pending_gradients = False
            elif is_accum_boundary:
                optimizer.zero_grad(set_to_none=True)

            if rank == 0 and writer is not None and loss_info["used_samples"] > 0:
                current_iter_log = epoch * steps_per_epoch + used_steps - 1
                writer.add_scalar("train/loss", loss_value, current_iter_log)
                writer.add_scalar("train/running_avg_loss", running_loss / max(used_steps, 1), current_iter_log)
                writer.add_scalar("train/used_samples", loss_info["used_samples"], current_iter_log)
                writer.add_scalar("train/skipped_samples", loss_info["skipped_samples"], current_iter_log)
                if "loss_ssi" in loss_info:
                    writer.add_scalar("train/loss_ssi", loss_info["loss_ssi"], current_iter_log)
                if "loss_grad" in loss_info:
                    writer.add_scalar("train/loss_grad", loss_info["loss_grad"], current_iter_log)
                if "loss_grad_weighted" in loss_term_metrics:
                    writer.add_scalar("train/loss_grad_weighted", loss_term_metrics["loss_grad_weighted"], current_iter_log)
                if "loss_grad_share" in loss_term_metrics:
                    writer.add_scalar("train/loss_grad_share", loss_term_metrics["loss_grad_share"], current_iter_log)
                if "norm_scale_mean" in loss_info:
                    writer.add_scalar("train/norm_scale_mean", loss_info["norm_scale_mean"], current_iter_log)
                if "normalized_samples" in loss_info:
                    writer.add_scalar("train/normalized_samples", loss_info["normalized_samples"], current_iter_log)
                writer.add_scalar(f"train/source_loss/{source}", loss_value, current_iter_log)

            micro_steps_done = step_idx + 1
            if rank == 0 and micro_steps_done % args.log_interval == 0:
                now = time.time()
                log_elapsed = max(now - last_log_time, 1e-6)
                epoch_elapsed = now - epoch_start_time
                steps_per_sec = args.log_interval / log_elapsed
                eta_seconds = (steps_per_epoch - micro_steps_done) / max(steps_per_sec, 1e-6)
                logger.info(
                    "[TRAIN] epoch=%d micro_step=%d/%d opt_step=%d/%d lr=%.7f loss=%.6e running_avg=%.6e "
                    "%s %s used=%d skipped=%d step_per_sec=%.2f "
                    "elapsed=%s eta=%s max_mem_mb=%.0f",
                    epoch,
                    micro_steps_done,
                    steps_per_epoch,
                    optimizer_steps_done,
                    optimizer_steps_per_epoch,
                    optimizer.param_groups[0]["lr"],
                    float(loss.item()),
                    running_loss / max(used_steps, 1),
                    format_source_running_avgs(source_stats, train_state["train_sources"]),
                    format_loss_term_summary(loss_term_metrics, loss_term_stats),
                    loss_info["used_samples"],
                    loss_info["skipped_samples"],
                    steps_per_sec,
                    format_seconds(epoch_elapsed),
                    format_seconds(eta_seconds),
                    torch.cuda.max_memory_allocated(device=img.device) / (1024 ** 2),
                )
                last_log_time = now

        epoch_stats = reduce_epoch_stats(running_loss, used_steps, source_stats, img.device)
        summary = None
        if valloader is not None:
            summary = evaluate(model, valloader, args, rank, writer=writer, epoch=epoch, logger=logger, tag="val", writer_prefix="eval")
        kitti_summary = None
        nyu_summary = None
        eth3d_summaries = {}
        robotcar_summaries = {}
        robotcar_night_summaries = {}
        if rgb_decoder_eval_model is not None:
            sync_rgb_decoder_eval_model(
                rgb_decoder_eval_model, model, logger=logger, rank=rank, sync_tag=f"epoch_{epoch}"
            )
        if kitti_valloader is not None:
            if args.kitti_eval_protocol == "rgb_checkpoint_decoder":
                kitti_summary = evaluate(
                    rgb_decoder_eval_model,
                    kitti_valloader,
                    args,
                    rank,
                    writer=writer,
                    epoch=epoch,
                    logger=logger,
                    tag=f"kitti_{args.kitti_eval_protocol}",
                    min_depth=args.kitti_min_depth,
                    max_depth=args.kitti_max_depth,
                    max_samples=args.debug_max_kitti_samples,
                    writer_prefix="eval_kitti",
                    model_input_type="rgb",
                )
            else:
                kitti_summary = pretrain_kitti_summary
        if nyu_valloader is not None:
            nyu_summary = evaluate(
                rgb_decoder_eval_model,
                nyu_valloader,
                args,
                rank,
                writer=writer,
                epoch=epoch,
                logger=logger,
                tag="nyu_rgb_checkpoint_decoder",
                min_depth=args.nyu_min_depth,
                max_depth=args.nyu_max_depth,
                max_samples=args.nyu_max_samples,
                writer_prefix="eval_nyu",
                model_input_type="rgb",
            )
        for depth_mode, eth3d_loader in iter_eth3d_eval_loaders(train_state, include_full=False):
            eth3d_summaries[depth_mode] = evaluate(
                model,
                eth3d_loader,
                args,
                rank,
                writer=writer,
                epoch=epoch,
                logger=logger,
                tag=f"eth3d_{depth_mode}",
                min_depth=args.eth3d_min_depth,
                max_depth=args.eth3d_max_depth,
                max_samples=args.eth3d_max_samples,
                writer_prefix=f"eval_eth3d_{depth_mode}",
                model_input_type=args.input_type,
            )
        for depth_mode, robotcar_loader in iter_robotcar_eval_loaders(train_state, include_full=False):
            robotcar_summaries[depth_mode] = evaluate(
                model,
                robotcar_loader,
                args,
                rank,
                writer=writer,
                epoch=epoch,
                logger=logger,
                tag=f"robotcar_{depth_mode}",
                min_depth=args.robotcar_min_depth,
                max_depth=args.robotcar_max_depth,
                max_samples=args.robotcar_max_samples,
                writer_prefix=f"eval_robotcar_{depth_mode}",
                model_input_type=args.input_type,
            )
        for depth_mode, robotcar_night_loader in iter_robotcar_night_eval_loaders(train_state, include_full=False):
            robotcar_night_summaries[depth_mode] = evaluate(
                model,
                robotcar_night_loader,
                args,
                rank,
                writer=writer,
                epoch=epoch,
                logger=logger,
                tag=f"robotcar_night_{depth_mode}",
                min_depth=args.robotcar_night_min_depth,
                max_depth=args.robotcar_night_max_depth,
                max_samples=args.robotcar_night_max_samples,
                writer_prefix=f"eval_robotcar_night_{depth_mode}",
                model_input_type=args.input_type,
            )

        if rank == 0:
            avg_loss = epoch_stats["running_loss"] / max(epoch_stats["used_steps"], 1)
            logger.info(
                "[EPOCH] done epoch=%d avg_loss=%.6e used_steps=%d elapsed=%s",
                epoch,
                avg_loss,
                epoch_stats["used_steps"],
                format_seconds(time.time() - epoch_start_time),
            )
            for source_name in train_state["train_sources"]:
                stats = epoch_stats["source"][source_name]
                if stats["steps"] == 0:
                    logger.info("[EPOCH][%s] avg_loss=n/a raw_pred_valid_mean_max=n/a", source_name)
                    continue

                source_avg = stats["loss_sum"] / max(stats["steps"], 1)
                pred_mean = stats["valid_pred_sum"] / max(stats["valid_pred_pixels"], 1)
                pred_max = stats["valid_pred_max"]
                logger.info(
                    "[EPOCH][%s] avg_loss=%.4f steps=%d raw_pred_valid_mean=%.4f raw_pred_valid_max=%.4f",
                    source_name,
                    source_avg,
                    stats["steps"],
                    pred_mean,
                    pred_max,
                )
                if writer is not None:
                    writer.add_scalar(f"train_epoch/source_avg_loss/{source_name}", source_avg, epoch)
                    writer.add_scalar(f"train_epoch/raw_pred_mean/{source_name}", pred_mean, epoch)
                    writer.add_scalar(f"train_epoch/raw_pred_max/{source_name}", pred_max, epoch)
            if summary is not None:
                log_eval_summary(logger, "val", summary)
            if kitti_summary is not None:
                log_eval_summary(logger, f"kitti_val_{args.kitti_eval_protocol}", kitti_summary)
                write_summary_scalars(writer, "eval_kitti", kitti_summary, epoch)
            if nyu_summary is not None:
                log_eval_summary(logger, "nyu_rgb_checkpoint_decoder", nyu_summary)
                write_summary_scalars(writer, "eval_nyu", nyu_summary, epoch)
            for depth_mode, eth3d_summary in eth3d_summaries.items():
                log_eval_summary(logger, f"eth3d_{depth_mode}", eth3d_summary)
                write_summary_scalars(writer, f"eval_eth3d_{depth_mode}", eth3d_summary, epoch)
            for depth_mode, robotcar_summary in robotcar_summaries.items():
                log_eval_summary(logger, f"robotcar_{depth_mode}", robotcar_summary)
                write_summary_scalars(writer, f"eval_robotcar_{depth_mode}", robotcar_summary, epoch)
            for depth_mode, robotcar_night_summary in robotcar_night_summaries.items():
                log_eval_summary(logger, f"robotcar_night_{depth_mode}", robotcar_night_summary)
                write_summary_scalars(writer, f"eval_robotcar_night_{depth_mode}", robotcar_night_summary, epoch)

            prev_best_metrics = dict(best_metrics)
            fast_eth3d_summary = eth3d_summaries.get("fast")
            fast_robotcar_summary = robotcar_summaries.get("fast")
            fast_robotcar_night_summary = robotcar_night_summaries.get("fast")
            metric_values = {
                "stf": summary["abs_rel"] if summary is not None else float("inf"),
                "kitti": kitti_summary["abs_rel"] if kitti_summary is not None else float("inf"),
                "eth3d": fast_eth3d_summary["abs_rel"] if fast_eth3d_summary is not None else float("inf"),
                "robotcar": fast_robotcar_summary["abs_rel"] if fast_robotcar_summary is not None else float("inf"),
                "robotcar_day": fast_robotcar_summary["abs_rel"] if fast_robotcar_summary is not None else float("inf"),
                "robotcar_night": (
                    fast_robotcar_night_summary["abs_rel"]
                    if fast_robotcar_night_summary is not None
                    else float("inf")
                ),
            }
            metric_values["avg4"] = compute_avg4_abs_rel(metric_values)
            updated_metrics = dict(prev_best_metrics)
            best_metric_improved = False
            best_metric_value = float("inf")
            for metric_name in BEST_METRIC_CHOICES:
                metric_value = metric_values.get(metric_name, float("inf"))
                if metric_value < prev_best_metrics[metric_name]:
                    updated_metrics[metric_name] = metric_value
                    if metric_name == args.best_metric:
                        best_metric_improved = True
                        best_metric_value = metric_value
                    logger.info(
                        "[CHECKPOINT] best_%s improved to %.4f",
                        metric_name,
                        metric_value,
                    )
            best_metrics = dict(updated_metrics)
            if args.save_best_checkpoint and best_metric_improved:
                best_ckpt_path = os.path.join(args.heavy_save_path, "best_model.pth")
                save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_metrics, args.best_metric)
                logger.info(
                    "[CHECKPOINT] saved best=%s metric=%s value=%.4f epoch=%d",
                    best_ckpt_path,
                    args.best_metric,
                    best_metric_value,
                    epoch,
                )
            current_ckpt_path = os.path.join(args.heavy_save_path, "current_model.pth")
            save_checkpoint(current_ckpt_path, model, optimizer, epoch, best_metrics, args.best_metric)
            logger.info("[CHECKPOINT] saved current=%s epoch=%d", current_ckpt_path, epoch)
            if epoch == args.epochs - 1:
                last_epoch_ckpt_path = os.path.join(args.heavy_save_path, "last_epoch_model.pth")
                save_checkpoint(last_epoch_ckpt_path, model, optimizer, epoch, best_metrics, args.best_metric)
                logger.info("[CHECKPOINT] saved last_epoch=%s epoch=%d", last_epoch_ckpt_path, epoch)
            if args.enable_train_source_viz_dump and train_state.get("train_source_viz_samples"):
                dump_train_source_samples(
                    model,
                    train_state["train_source_viz_samples"],
                    args,
                    epoch,
                    args.save_path,
                    writer=writer,
                    baseline_model=train_viz_rgb_baseline_model,
                    baseline_label=args.train_viz_rgb_baseline_label or f"RGB DAv2 {args.encoder}",
                    logger=logger,
                )
            if args.enable_fixed_viz_dump and train_state.get("fixed_viz_samples"):
                model_overrides = {}
                input_type_overrides = {}
                if kitti_valloader is not None:
                    kitti_dump_model = (
                        rgb_decoder_eval_model
                        if args.kitti_eval_protocol == "rgb_checkpoint_decoder"
                        else kitti_reference_eval_model
                    )
                    if kitti_dump_model is not None:
                        model_overrides["kitti"] = kitti_dump_model
                        input_type_overrides["kitti"] = "rgb"
                dump_outputs = dump_fixed_samples(
                    model,
                    train_state["fixed_viz_samples"],
                    args,
                    epoch,
                    args.save_path,
                    model_overrides=model_overrides,
                    input_type_overrides=input_type_overrides,
                    rgb_baseline_model=fixed_viz_rgb_baseline_model,
                    rgb_baseline_splits=("robotcar", "robotcar_night"),
                    rgb_baseline_label="RGB DAv2",
                )
                logger.info(
                    "[VIZ] dumped fixed samples epoch=%d splits=%s root=%s",
                    epoch,
                    {split: len(paths) for split, paths in dump_outputs.items()},
                    os.path.join(args.save_path, "viz_fixed", f"epoch_{epoch:02d}"),
                )

        dist.barrier()

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
