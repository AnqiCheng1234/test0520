from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_stf.dataset.raw_utils import bayer_to_3ch, normalize_raw, normalize_raw_4ch
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_ETH3D_ROOT = "/mnt/drive/3333_raw/eth3d_raw_depth_640960"
DEFAULT_ETH3D_MANIFEST = "eth3d_raw_depth_v2_val.csv"
DEPTH_PROXY_KEY = "depth"
VALID_MASK_KEY = "valid_mask"
RAW_4CH_KEY = "raw_4ch"
DEPTH_FULL_HW = (4032, 6048)
ETH3D_EVAL_HW = (640, 960)
ETH3D_FAST_EVAL_BACKENDS = ("proxy", "sparse")


def _resolve_manifest_path(eth3d_root: Path, manifest_name: str) -> Path:
    manifest_path = Path(manifest_name).expanduser()
    if manifest_path.is_absolute():
        return manifest_path.resolve()
    return (eth3d_root / "manifests" / manifest_name).resolve()


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "scene",
            "sample_name",
            "rgb_src_path",
            "rgb_640_path",
            "raw_native_path",
            "raw_640_path",
            "depth_src_path",
            "depth_proxy_path",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{manifest_path} is missing required ETH3D columns: {', '.join(missing)}")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No ETH3D rows found in {manifest_path}")
    return rows


def _load_depth_full(depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    depth = np.fromfile(depth_path, dtype=np.float32)
    if depth.size != DEPTH_FULL_HW[0] * DEPTH_FULL_HW[1]:
        raise RuntimeError(f"Unexpected ETH3D full depth size for {depth_path}: {depth.size}")
    depth = depth.reshape(DEPTH_FULL_HW)
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
    return depth, valid


def _load_depth_proxy(depth_proxy_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(depth_proxy_path, allow_pickle=False) as data:
        if DEPTH_PROXY_KEY not in data.files or VALID_MASK_KEY not in data.files:
            raise KeyError(f"{depth_proxy_path} must contain '{DEPTH_PROXY_KEY}' and '{VALID_MASK_KEY}'")
        depth = np.asarray(data[DEPTH_PROXY_KEY], dtype=np.float32)
        valid = np.asarray(data[VALID_MASK_KEY]).astype(bool)
    return depth, valid


class _ETH3DValBase(Dataset):
    def __init__(
        self,
        *,
        eth3d_root: str | Path = DEFAULT_ETH3D_ROOT,
        manifest_name: str = DEFAULT_ETH3D_MANIFEST,
        depth_mode: str = "fast",
        fast_eval_backend: str = "proxy",
        min_depth: float = 0.1,
        max_depth: float = 80.0,
    ) -> None:
        self.eth3d_root = Path(eth3d_root).expanduser().resolve()
        self.manifest_path = _resolve_manifest_path(self.eth3d_root, manifest_name)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing ETH3D manifest: {self.manifest_path}")
        if depth_mode not in {"fast", "full"}:
            raise ValueError(f"Unsupported ETH3D depth_mode: {depth_mode}")
        if fast_eval_backend not in ETH3D_FAST_EVAL_BACKENDS:
            raise ValueError(f"Unsupported ETH3D fast_eval_backend: {fast_eval_backend}")
        self.depth_mode = str(depth_mode)
        self.fast_eval_backend = str(fast_eval_backend)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.rows = _load_manifest_rows(self.manifest_path)
        self.rgb_normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.prepare_for_net = PrepareForNet()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_depth_and_mask(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, str]:
        if self.depth_mode == "fast" and self.fast_eval_backend == "proxy":
            depth_path = Path(row["depth_proxy_path"]).expanduser().resolve()
            depth, valid = _load_depth_proxy(depth_path)
        else:
            depth_path = Path(row["depth_src_path"]).expanduser().resolve()
            depth, valid = _load_depth_full(depth_path)
        valid = valid & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
        return depth, valid.astype(bool, copy=False), str(depth_path)

    def _build_common_output(
        self,
        *,
        row: dict[str, str],
        image: torch.Tensor,
        image_path: str,
    ) -> dict[str, torch.Tensor | str]:
        depth, valid_mask, depth_path = self._load_depth_and_mask(row)
        sample_stem = row["sample_name"]
        sample_name = f"{row['scene']}/{sample_stem}"
        output = {
            "image": image,
            "depth": torch.from_numpy(np.ascontiguousarray(depth)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)),
            "scene": row["scene"],
            "sample_stem": sample_stem,
            "sample_name": sample_name,
            "image_path": image_path,
            "depth_path": depth_path,
            "depth_mode": self.depth_mode,
            "fast_eval_backend": self.fast_eval_backend,
            "rgb_src_path": row["rgb_src_path"],
            "rgb_640_path": row["rgb_640_path"],
            "raw_native_path": row["raw_native_path"],
        }
        return output


class ETH3DValRaw(_ETH3DValBase):
    def __init__(
        self,
        *,
        eth3d_root: str | Path = DEFAULT_ETH3D_ROOT,
        manifest_name: str = DEFAULT_ETH3D_MANIFEST,
        depth_mode: str = "fast",
        fast_eval_backend: str = "proxy",
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        norm_mode: str = "sensor_linear",
        channel_mode: str = "rgb_avg_g",
        use_imagenet_norm: bool = True,
        input_mode: str = "raw_ram",
    ) -> None:
        super().__init__(
            eth3d_root=eth3d_root,
            manifest_name=manifest_name,
            depth_mode=depth_mode,
            fast_eval_backend=fast_eval_backend,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        self.norm_mode = str(norm_mode)
        self.channel_mode = str(channel_mode)
        self.use_imagenet_norm = bool(use_imagenet_norm)
        self.input_mode = str(input_mode)
        if self.input_mode not in {"raw_ram", "raw_naive"}:
            raise ValueError(f"Unsupported ETH3D raw input_mode: {self.input_mode}")

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        raw_path = Path(row["raw_640_path"]).expanduser().resolve()
        with np.load(raw_path, allow_pickle=False) as data:
            if RAW_4CH_KEY not in data.files:
                raise KeyError(f"{raw_path} does not contain '{RAW_4CH_KEY}'")
            raw = np.asarray(data[RAW_4CH_KEY], dtype=np.float32)
        if raw.shape[:2] != ETH3D_EVAL_HW:
            raise RuntimeError(f"Unexpected ETH3D raw eval shape {raw.shape[:2]} for {raw_path}")
        raw = normalize_raw_4ch(raw, norm_mode=self.norm_mode)
        if self.input_mode == "raw_ram":
            image_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(raw, (2, 0, 1))))
            output = self._build_common_output(row=row, image=image_tensor, image_path=str(raw_path))
            output["raw"] = image_tensor
            return output

        raw_rgb = bayer_to_3ch(raw, channel_mode=self.channel_mode)
        raw_rgb = normalize_raw(raw_rgb, norm_mode=self.norm_mode)
        sample = {"image": raw_rgb}
        if self.use_imagenet_norm:
            sample = self.rgb_normalize(sample)
        sample = self.prepare_for_net(sample)
        image_tensor = torch.from_numpy(np.ascontiguousarray(sample["image"]))
        return self._build_common_output(row=row, image=image_tensor, image_path=str(raw_path))


class ETH3DValRGB(_ETH3DValBase):
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rgb_path = Path(row["rgb_640_path"]).expanduser().resolve()
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read ETH3D RGB image: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if image.shape[:2] != ETH3D_EVAL_HW:
            raise RuntimeError(f"Unexpected ETH3D RGB eval shape {image.shape[:2]} for {rgb_path}")
        sample = self.rgb_normalize({"image": image})
        sample = self.prepare_for_net(sample)
        image_chw = torch.from_numpy(np.ascontiguousarray(sample["image"]))
        return self._build_common_output(row=row, image=image_chw, image_path=str(rgb_path))
