import csv
import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torchvision.transforms import Compose
except ImportError:
    class Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms)

        def __call__(self, sample):
            for transform in self.transforms:
                sample = transform(sample)
            return sample

from finetune_stf.dataset.raw_domain import apply_raw_domain_transform, parse_raw_domain_config
from finetune_stf.dataset.raw_utils import normalize_raw_4ch
from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_LOD_ROOT = "/mnt/drive/3333_raw/LOD"
DEFAULT_LOD_DAY_MANIFEST = (
    "/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv"
)
DEFAULT_LOD_NIGHT_MANIFEST = (
    "/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest.csv"
)
LOD_NATIVE_HW = (928, 1440)
LOD_REQUIRED_COLUMNS = ("split", "sample_name", "rggb_path", "output_npy")
LOD_RGB_REQUIRED_COLUMNS = ("split", "sample_name", "rgb_path", "output_npy")
LOD_SAMPLE_PREFIXES = ("day-", "night-")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resolve_data_path(root, path_str):
    path = Path(path_str.strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(root).expanduser().resolve() / path).resolve()


def _load_manifest_rows(manifest_path, lod_root):
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in LOD_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required LOD columns: {', '.join(missing)}"
            )

        for row in reader:
            sample_name = row["sample_name"].strip()
            if not sample_name.startswith(LOD_SAMPLE_PREFIXES):
                raise ValueError(f"LOD manifest contains unexpected sample {sample_name!r}")
            rows.append(
                {
                    "split": row["split"].strip(),
                    "sample_name": sample_name,
                    "raw_path": _resolve_data_path(lod_root, row["rggb_path"]),
                    "rgb_path": (
                        _resolve_data_path(lod_root, row["rgb_path"])
                        if row.get("rgb_path", "").strip()
                        else None
                    ),
                    "target_path": _resolve_data_path(lod_root, row["output_npy"]),
                }
            )
    return rows


def _load_rgb_manifest_rows(manifest_path, lod_root):
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in LOD_RGB_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required LOD RGB columns: {', '.join(missing)}"
            )

        for row in reader:
            sample_name = row["sample_name"].strip()
            if not sample_name.startswith(LOD_SAMPLE_PREFIXES):
                raise ValueError(f"LOD manifest contains unexpected sample {sample_name!r}")
            rows.append(
                {
                    "split": row["split"].strip(),
                    "sample_name": sample_name,
                    "image_path": _resolve_data_path(lod_root, row["rgb_path"]),
                    "target_path": _resolve_data_path(lod_root, row["output_npy"]),
                }
            )
    return rows


def _sample_crop_box(height, width, size, crop_mode, rng=random):
    target_h, target_w = size
    if height < target_h or width < target_w:
        raise ValueError(f"Requested crop {size} exceeds input shape {(height, width)}")
    if crop_mode == "center":
        h_start = max((height - target_h) // 2, 0)
        w_start = max((width - target_w) // 2, 0)
    elif crop_mode == "random":
        h_start = rng.randint(0, height - target_h)
        w_start = rng.randint(0, width - target_w)
    else:
        raise ValueError(f"Unsupported LOD crop_mode {crop_mode!r}")
    return h_start, w_start, target_h, target_w


def _apply_crop(array, crop_box):
    h_start, w_start, target_h, target_w = crop_box
    h_end = h_start + target_h
    w_end = w_start + target_w
    if array.ndim == 2:
        return np.ascontiguousarray(array[h_start:h_end, w_start:w_end])
    return np.ascontiguousarray(array[h_start:h_end, w_start:w_end, ...])


def _center_crop(array, size):
    height, width = array.shape[:2]
    return _apply_crop(array, _sample_crop_box(height, width, size, "center"))


def _rgb_preview_tensor(image_rgb, target_hw):
    resized = cv2.resize(image_rgb, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.transpose(np.ascontiguousarray(resized), (2, 0, 1)).astype(np.float32, copy=False))


def _imagenet_normalize_rgb_tensor(image_rgb, target_hw):
    resized = cv2.resize(image_rgb, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32, copy=False) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(np.ascontiguousarray(normalized), (2, 0, 1)).astype(np.float32, copy=False))


class LODRaw(Dataset):
    def __init__(
        self,
        *,
        lod_root=DEFAULT_LOD_ROOT,
        manifest_path=DEFAULT_LOD_DAY_MANIFEST,
        size=(644, 1008),
        norm_mode="sensor_linear",
        raw_domain_config=None,
        mode="train",
        crop_mode="center",
    ):
        self.lod_root = Path(lod_root).expanduser().resolve()
        if isinstance(manifest_path, (str, Path)):
            resolved_manifest_paths = [Path(manifest_path).expanduser().resolve()]
        elif isinstance(manifest_path, Sequence):
            resolved_manifest_paths = [Path(path).expanduser().resolve() for path in manifest_path]
        else:
            raise TypeError(
                "manifest_path must be a path or a sequence of paths, "
                f"got {type(manifest_path).__name__}"
            )
        if not resolved_manifest_paths:
            raise ValueError("manifest_path sequence is empty")
        self.manifest_paths = resolved_manifest_paths
        self.manifest_path = resolved_manifest_paths[0]
        self.size = tuple(size)
        self.norm_mode = str(norm_mode)
        self.raw_domain_config = parse_raw_domain_config(raw_domain_config)
        self.mode = str(mode)
        if crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be one of {'center', 'random'}")
        self.crop_mode = "center" if self.mode == "val" else crop_mode
        self.rows = []
        for path in self.manifest_paths:
            self.rows.extend(_load_manifest_rows(path, self.lod_root))
        if not self.rows:
            raise ValueError(f"No LOD samples found in manifests: {self.manifest_paths}")

        self.transform = Compose([PrepareForNet()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.build_sample(idx)

    def build_sample(self, idx, *, rng=random, include_geometry=False):
        row = self.rows[idx]
        raw_path = row["raw_path"]
        target_path = row["target_path"]

        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing LOD raw file: {raw_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing LOD pseudo depth file: {target_path}")

        raw = np.load(raw_path).astype(np.float32, copy=False)
        if raw.shape != (*LOD_NATIVE_HW, 4):
            raise ValueError(
                f"Expected LOD raw shape {(LOD_NATIVE_HW[0], LOD_NATIVE_HW[1], 4)}, got {tuple(raw.shape)}"
            )

        target = np.load(target_path).astype(np.float32, copy=False)
        if target.shape != LOD_NATIVE_HW:
            raise ValueError(f"Expected LOD target shape {LOD_NATIVE_HW}, got {tuple(target.shape)}")

        crop_box = _sample_crop_box(raw.shape[0], raw.shape[1], self.size, self.crop_mode, rng)
        raw = _apply_crop(raw, crop_box)
        target = _apply_crop(target, crop_box)

        raw = normalize_raw_4ch(raw, norm_mode=self.norm_mode)
        raw = apply_raw_domain_transform(raw, self.raw_domain_config)
        valid_mask = np.isfinite(target) & (target > 0)
        target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)

        sample = self.transform({"image": raw, "depth": target, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["sample_name"] = row["sample_name"]
        sample["image_path"] = str(raw_path)
        sample["raw_path"] = str(raw_path)
        if row.get("rgb_path") is not None:
            sample["rgb_path"] = str(row["rgb_path"])
        sample["depth_path"] = str(target_path)
        sample["split"] = row["split"]
        sample["target_space"] = "inverse_relative"
        if include_geometry:
            sample["geometry_params"] = {
                "original_hw": [int(LOD_NATIVE_HW[0]), int(LOD_NATIVE_HW[1])],
                "crop_box": [int(v) for v in crop_box],
                "crop_box_format": "h_start_w_start_h_w",
                "hflip_applied": False,
            }
            if row.get("rgb_path") is not None:
                rgb_path = row["rgb_path"]
                if not rgb_path.is_file():
                    raise FileNotFoundError(f"Missing LOD RGB file for preview: {rgb_path}")
                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to read LOD RGB image for preview: {rgb_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                h_start, w_start, crop_h, crop_w = [int(v) for v in crop_box]
                scale_h = float(image.shape[0]) / float(LOD_NATIVE_HW[0])
                scale_w = float(image.shape[1]) / float(LOD_NATIVE_HW[1])
                rgb_h_start = int(round(h_start * scale_h))
                rgb_w_start = int(round(w_start * scale_w))
                rgb_h_end = int(round((h_start + crop_h) * scale_h))
                rgb_w_end = int(round((w_start + crop_w) * scale_w))
                image = np.ascontiguousarray(image[rgb_h_start:rgb_h_end, rgb_w_start:rgb_w_end, ...])
                if image.size == 0:
                    raise ValueError(
                        f"LOD RGB preview crop is empty for idx={idx}: "
                        f"crop_box={crop_box} rgb_shape={image.shape}"
                    )
                sample["rgb_preview"] = _rgb_preview_tensor(image, self.size)
        return sample

    def build_rgb_baseline_input(self, idx, geometry, *, target_hw):
        row = self.rows[idx]
        rgb_path = row.get("rgb_path")
        if rgb_path is None:
            raise ValueError(
                "LOD RGB baseline requires an rgb_path column in the manifest. "
                f"manifest={self.manifest_path} idx={idx} sample={row.get('sample_name')}"
            )
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing LOD RGB file for baseline: {rgb_path}")

        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read LOD RGB image for baseline: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        crop_box = geometry.get("crop_box")
        if crop_box is None:
            raise ValueError(f"Missing LOD geometry crop_box for idx={idx}")
        h_start, w_start, crop_h, crop_w = [int(v) for v in crop_box]
        scale_h = float(image.shape[0]) / float(LOD_NATIVE_HW[0])
        scale_w = float(image.shape[1]) / float(LOD_NATIVE_HW[1])
        rgb_h_start = int(round(h_start * scale_h))
        rgb_w_start = int(round(w_start * scale_w))
        rgb_h_end = int(round((h_start + crop_h) * scale_h))
        rgb_w_end = int(round((w_start + crop_w) * scale_w))
        image = np.ascontiguousarray(image[rgb_h_start:rgb_h_end, rgb_w_start:rgb_w_end, ...])
        if image.size == 0:
            raise ValueError(
                f"LOD RGB baseline crop is empty for idx={idx}: crop_box={crop_box} rgb_shape={image.shape}"
            )
        return _imagenet_normalize_rgb_tensor(image, target_hw), _rgb_preview_tensor(image, target_hw)


class LODRGB(Dataset):
    def __init__(
        self,
        *,
        lod_root=DEFAULT_LOD_ROOT,
        manifest_path=DEFAULT_LOD_DAY_MANIFEST,
        size=(644, 1008),
        mode="train",
        crop_mode="center",
    ):
        self.lod_root = Path(lod_root).expanduser().resolve()
        if isinstance(manifest_path, (str, Path)):
            resolved_manifest_paths = [Path(manifest_path).expanduser().resolve()]
        elif isinstance(manifest_path, Sequence):
            resolved_manifest_paths = [Path(path).expanduser().resolve() for path in manifest_path]
        else:
            raise TypeError(
                "manifest_path must be a path or a sequence of paths, "
                f"got {type(manifest_path).__name__}"
            )
        if not resolved_manifest_paths:
            raise ValueError("manifest_path sequence is empty")
        self.manifest_paths = resolved_manifest_paths
        self.manifest_path = resolved_manifest_paths[0]
        self.size = tuple(size)
        self.mode = str(mode)
        if crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be one of {'center', 'random'}")
        self.crop_mode = "center" if self.mode == "val" else crop_mode
        self.rows = []
        for path in self.manifest_paths:
            self.rows.extend(_load_rgb_manifest_rows(path, self.lod_root))
        if not self.rows:
            raise ValueError(f"No LOD RGB samples found in manifests: {self.manifest_paths}")

        self.transform = Compose(
            [
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        target_path = row["target_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing LOD RGB file: {image_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing LOD pseudo depth file: {target_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read LOD RGB image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        target = np.load(target_path).astype(np.float32, copy=False)
        if target.shape != LOD_NATIVE_HW:
            raise ValueError(f"Expected LOD target shape {LOD_NATIVE_HW}, got {tuple(target.shape)}")
        if image.shape[:2] != target.shape:
            image = cv2.resize(
                image,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        crop_box = _sample_crop_box(image.shape[0], image.shape[1], self.size, self.crop_mode, random)
        image = _apply_crop(image, crop_box)
        target = _apply_crop(target, crop_box)

        valid_mask = np.isfinite(target) & (target > 0)
        target = np.where(valid_mask, target, 0.0).astype(np.float32, copy=False)

        sample = self.transform({"image": image, "depth": target, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["sample_name"] = row["sample_name"]
        sample["image_path"] = str(image_path)
        sample["depth_path"] = str(target_path)
        sample["split"] = row["split"]
        sample["target_space"] = "inverse_relative"
        return sample
