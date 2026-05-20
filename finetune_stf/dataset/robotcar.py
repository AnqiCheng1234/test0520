from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_stf.dataset.raw_domain import apply_raw_domain_transform, parse_raw_domain_config
from finetune_stf.dataset.raw_utils import (
    bayer_to_3ch,
    load_rectified_bayer_npz,
    normalize_raw,
    normalize_raw_4ch,
)
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_ROBOTCAR_ROOT = "/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100"
DEFAULT_ROBOTCAR_MANIFEST = "robotcar_raw_depth_v1_val.csv"
DEPTH_PROXY_KEY = "depth"
VALID_MASK_KEY = "valid_mask"
ROBOTCAR_FAST_EVAL_BACKENDS = ("proxy", "sparse")


def _resolve_manifest_path(robotcar_root: Path, manifest_name: str) -> Path:
    manifest_path = Path(manifest_name).expanduser()
    if manifest_path.is_absolute():
        return manifest_path.resolve()
    return (robotcar_root / "manifests" / manifest_name).resolve()


def _parse_hw(hw_str: str) -> tuple[int, int]:
    parts = str(hw_str).lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid hw string: {hw_str}")
    return int(parts[0]), int(parts[1])


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "scene",
            "sample_name",
            "rgb_src_path",
            "rgb_eval_path",
            "raw_src_path",
            "raw_native_path",
            "raw_eval_path",
            "depth_src_path",
            "depth_proxy_path",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{manifest_path} is missing required RobotCar columns: {', '.join(missing)}")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No RobotCar rows found in {manifest_path}")
    return rows


def _load_depth_full(depth_path: Path) -> tuple[np.ndarray, np.ndarray]:
    depth = np.load(depth_path).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected RobotCar full depth shape for {depth_path}: {depth.shape}")
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


def _canonicalize_raw_channel_order(raw: np.ndarray, pack_order: str) -> np.ndarray:
    if pack_order == "[R,Gr,Gb,B]":
        return raw
    if pack_order == "[R,Gb,Gr,B]":
        return raw[..., [0, 2, 1, 3]]
    raise ValueError(f"Unsupported RobotCar pack_order: {pack_order}")


class _RobotCarValBase(Dataset):
    def __init__(
        self,
        *,
        robotcar_root: str | Path = DEFAULT_ROBOTCAR_ROOT,
        manifest_name: str = DEFAULT_ROBOTCAR_MANIFEST,
        depth_mode: str = "fast",
        fast_eval_backend: str = "sparse",
        min_depth: float = 0.1,
        max_depth: float = 80.0,
    ) -> None:
        self.robotcar_root = Path(robotcar_root).expanduser().resolve()
        self.manifest_path = _resolve_manifest_path(self.robotcar_root, manifest_name)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Missing RobotCar manifest: {self.manifest_path}")
        if depth_mode not in {"fast", "full"}:
            raise ValueError(f"Unsupported RobotCar depth_mode: {depth_mode}")
        if fast_eval_backend not in ROBOTCAR_FAST_EVAL_BACKENDS:
            raise ValueError(f"Unsupported RobotCar fast_eval_backend: {fast_eval_backend}")
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
        return {
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
            "rgb_eval_path": row["rgb_eval_path"],
            "raw_native_path": row["raw_native_path"],
            "raw_src_path": row["raw_src_path"],
        }


class RobotCarValRaw(_RobotCarValBase):
    def __init__(
        self,
        *,
        robotcar_root: str | Path = DEFAULT_ROBOTCAR_ROOT,
        manifest_name: str = DEFAULT_ROBOTCAR_MANIFEST,
        depth_mode: str = "fast",
        fast_eval_backend: str = "sparse",
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        norm_mode: str = "sensor_linear",
        channel_mode: str = "rgb_avg_g",
        use_imagenet_norm: bool = True,
        input_mode: str = "raw_ram",
        raw_domain_config=None,
    ) -> None:
        super().__init__(
            robotcar_root=robotcar_root,
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
        self.raw_domain_config = parse_raw_domain_config(raw_domain_config)
        if self.input_mode not in {"raw_ram", "raw_naive"}:
            raise ValueError(f"Unsupported RobotCar raw input_mode: {self.input_mode}")

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        raw_path = Path(row["raw_eval_path"]).expanduser().resolve()
        raw = load_rectified_bayer_npz(raw_path)
        raw = _canonicalize_raw_channel_order(raw, row.get("pack_order", "[R,Gr,Gb,B]"))
        expected_hw = _parse_hw(row.get("raw_eval_hw", f"{raw.shape[0]}x{raw.shape[1]}"))
        if tuple(raw.shape[:2]) != expected_hw:
            raise RuntimeError(f"Unexpected RobotCar raw eval shape {raw.shape[:2]} for {raw_path}")
        raw = normalize_raw_4ch(raw, norm_mode=self.norm_mode)
        raw = apply_raw_domain_transform(raw, self.raw_domain_config)
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


class RobotCarValRGB(_RobotCarValBase):
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rgb_path = Path(row["rgb_eval_path"]).expanduser().resolve()
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read RobotCar RGB image: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        expected_hw = _parse_hw(row.get("rgb_eval_hw", f"{image.shape[0]}x{image.shape[1]}"))
        if tuple(image.shape[:2]) != expected_hw:
            raise RuntimeError(f"Unexpected RobotCar RGB eval shape {image.shape[:2]} for {rgb_path}")
        sample = self.rgb_normalize({"image": image})
        sample = self.prepare_for_net(sample)
        image_chw = torch.from_numpy(np.ascontiguousarray(sample["image"]))
        return self._build_common_output(row=row, image=image_chw, image_path=str(rgb_path))
