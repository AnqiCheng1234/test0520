from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from foundation.engine.models import build_dav2_raw_naive_depth_model


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for Step 3 naive RAW model.")
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument(
        "--pretrained-from",
        default="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth",
    )
    parser.add_argument("--height", type=int, default=259)
    parser.add_argument("--width", type=int, default=483)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def resolve_model_state(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    return ckpt_obj


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    base_model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder]).to(device)
    wrapper = build_dav2_raw_naive_depth_model(base_model, freeze_backbone=args.freeze_backbone).to(device)

    ckpt = torch.load(args.pretrained_from, map_location="cpu")
    state_dict = strip_module_prefix(resolve_model_state(ckpt))
    wrapper.load_base_dav2_state_dict(state_dict)

    x = torch.rand(args.batch_size, 4, args.height, args.width, device=device)
    with torch.no_grad():
        outputs = wrapper.forward_features(x)

    total_params = sum(p.numel() for p in wrapper.parameters())
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    print(f"device={device}")
    print(f"encoder={args.encoder}")
    print(f"input_shape={tuple(x.shape)}")
    print(f"rgb_shape={tuple(outputs['rgb'].shape)}")
    print(f"depth_shape={tuple(outputs['depth'].shape)}")
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")
    print(f"freeze_backbone={args.freeze_backbone}")
    print(f"stem_r={wrapper.input_stem.proj.weight[0, :, 0, 0].tolist()}")
    print(f"stem_g={wrapper.input_stem.proj.weight[1, :, 0, 0].tolist()}")
    print(f"stem_b={wrapper.input_stem.proj.weight[2, :, 0, 0].tolist()}")


if __name__ == "__main__":
    main()
