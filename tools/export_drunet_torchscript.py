#!/usr/bin/env python3
"""Export the DPIR DRUNet color checkpoint to a TorchScript denoiser."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CKPT = Path("/mnt/drive/3333_raw/0000_exp_ckpt/external_denoisers/DPIR/drunet_color.pth")
DEFAULT_OUTPUT = Path("/mnt/drive/3333_raw/0000_exp_ckpt/external_denoisers/DPIR/drunet_color_pad8.ts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


def sequential(*args: nn.Module) -> nn.Module:
    modules: list[nn.Module] = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    if len(modules) == 1:
        return modules[0]
    return nn.Sequential(*modules)


def conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    mode: str = "C",
) -> nn.Module:
    layers: list[nn.Module] = []
    current_in = in_channels
    for token in mode:
        if token == "C":
            layers.append(
                nn.Conv2d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
            current_in = out_channels
        elif token == "T":
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
            current_in = out_channels
        elif token == "R":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError(f"Unsupported DPIR conv mode token: {token}")
    return sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, bias: bool = False, mode: str = "CRC"):
        super().__init__()
        self.res = conv(channels, channels, bias=bias, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)


def downsample_strideconv(in_channels: int, out_channels: int, *, bias: bool = False, mode: str = "2") -> nn.Module:
    scale = int(mode[0])
    return conv(in_channels, out_channels, kernel_size=scale, stride=scale, padding=0, bias=bias, mode="C")


def upsample_convtranspose(in_channels: int, out_channels: int, *, bias: bool = False, mode: str = "2") -> nn.Module:
    scale = int(mode[0])
    return conv(in_channels, out_channels, kernel_size=scale, stride=scale, padding=0, bias=bias, mode="T")


class UNetRes(nn.Module):
    def __init__(
        self,
        *,
        in_nc: int = 4,
        out_nc: int = 3,
        nc: tuple[int, int, int, int] = (64, 128, 256, 512),
        nb: int = 4,
    ):
        super().__init__()
        self.m_head = conv(in_nc, nc[0], bias=False, mode="C")

        self.m_down1 = sequential(
            *[ResBlock(nc[0], bias=False, mode="CRC") for _ in range(nb)],
            downsample_strideconv(nc[0], nc[1], bias=False, mode="2"),
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], bias=False, mode="CRC") for _ in range(nb)],
            downsample_strideconv(nc[1], nc[2], bias=False, mode="2"),
        )
        self.m_down3 = sequential(
            *[ResBlock(nc[2], bias=False, mode="CRC") for _ in range(nb)],
            downsample_strideconv(nc[2], nc[3], bias=False, mode="2"),
        )
        self.m_body = sequential(*[ResBlock(nc[3], bias=False, mode="CRC") for _ in range(nb)])

        self.m_up3 = sequential(
            upsample_convtranspose(nc[3], nc[2], bias=False, mode="2"),
            *[ResBlock(nc[2], bias=False, mode="CRC") for _ in range(nb)],
        )
        self.m_up2 = sequential(
            upsample_convtranspose(nc[2], nc[1], bias=False, mode="2"),
            *[ResBlock(nc[1], bias=False, mode="CRC") for _ in range(nb)],
        )
        self.m_up1 = sequential(
            upsample_convtranspose(nc[1], nc[0], bias=False, mode="2"),
            *[ResBlock(nc[0], bias=False, mode="CRC") for _ in range(nb)],
        )
        self.m_tail = conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


class PadToMultipleDenoiser(nn.Module):
    def __init__(self, denoiser: nn.Module, multiple: int = 8):
        super().__init__()
        self.denoiser = denoiser
        self.multiple = int(multiple)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2]
        w = x.shape[-1]
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = self.denoiser(x)
        return y[..., :h, :w]


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected state_dict at {path}, got {type(payload)!r}")
    return payload


def validate(scripted: torch.jit.ScriptModule, eager: nn.Module) -> None:
    torch.manual_seed(0)
    for height, width in ((64, 64), (65, 67)):
        x = torch.rand(1, 4, height, width)
        with torch.no_grad():
            y_eager = eager(x)
            y_scripted = scripted(x)
        max_abs = float((y_eager - y_scripted).abs().max().item())
        if y_scripted.shape != (1, 3, height, width):
            raise RuntimeError(f"Unexpected scripted output shape: {tuple(y_scripted.shape)}")
        if max_abs > 1e-5:
            raise RuntimeError(f"Scripted validation failed for {height}x{width}: max_abs={max_abs}")
        print(f"[DRUNET_EXPORT] validate {height}x{width} max_abs={max_abs:.3e}", flush=True)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()

    model = UNetRes(in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4)
    state_dict = load_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    wrapped = PadToMultipleDenoiser(model, multiple=8).eval()
    for param in wrapped.parameters():
        param.requires_grad_(False)

    scripted = torch.jit.script(wrapped)
    scripted = torch.jit.freeze(scripted)
    if not args.skip_validate:
        validate(scripted, wrapped)

    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"[DRUNET_EXPORT] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
