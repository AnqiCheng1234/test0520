from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet
from foundation.engine.transforms import build_unprocessing_transform_from_preset

RAW_LIKE_INPUT_TYPES = {
    "raw_packed",
    "raw_ram",
    "raw_ram_residual",
    "raw_ram_feature_adapter",
    "raw_ram_bridge_feature_adapter",
    "raw_ram_bridge_feature_adapter_lora",
    "raw_ram_bridge",
    "raw_ram_bridge_lora",
}

DEFAULT_KITTI_BASE = "/mnt/drive/kitti"
DEFAULT_KITTI_VAL_SPLIT = (
    Path(__file__).resolve().parents[2] / "metric_depth" / "dataset" / "splits" / "kitti" / "val.txt"
)


def _remap_kitti_path(path_str: str, kitti_base: Path) -> Path:
    path = path_str.strip()
    if path.startswith("/mnt/bn/liheyang/Kitti/raw_data/"):
        rel = path[len("/mnt/bn/liheyang/Kitti/raw_data/") :]
        rel_path = Path(rel)
        if rel_path.suffix.lower() == ".png":
            rel_path = rel_path.with_suffix(".jpg")
        return kitti_base / rel_path
    if path.startswith("/mnt/bn/liheyang/Kitti/data_depth_annotated/"):
        rel = path[len("/mnt/bn/liheyang/Kitti/data_depth_annotated/") :]
        return kitti_base / "annotated_depth" / rel
    return Path(path)


class KITTIEval(Dataset):
    def __init__(
        self,
        filelist_path: str | Path | None = None,
        *,
        kitti_base: str | Path = DEFAULT_KITTI_BASE,
        size=(512, 960),
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        input_type: str = "rgb",
    ) -> None:
        self.kitti_base = Path(kitti_base).expanduser().resolve()
        self.filelist_path = Path(filelist_path or DEFAULT_KITTI_VAL_SPLIT).expanduser().resolve()
        self.size = tuple(size)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.input_type = str(input_type)
        if self.input_type not in {"rgb", *RAW_LIKE_INPUT_TYPES}:
            raise ValueError(f"KITTIEval only supports rgb/raw-like input types, got {self.input_type}")

        if not self.kitti_base.is_dir():
            raise FileNotFoundError(f"Missing KITTI base directory: {self.kitti_base}")
        if not self.filelist_path.is_file():
            raise FileNotFoundError(f"Missing KITTI split file: {self.filelist_path}")

        self.rows = []
        with self.filelist_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path_str, depth_path_str = line.split()
                image_path = _remap_kitti_path(image_path_str, self.kitti_base)
                depth_path = _remap_kitti_path(depth_path_str, self.kitti_base)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing KITTI image: {image_path}")
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing KITTI depth: {depth_path}")
                self.rows.append(
                    {
                        "image_path": image_path,
                        "depth_path": depth_path,
                    }
                )

        if not self.rows:
            raise ValueError(f"No KITTI eval samples found in {self.filelist_path}")

        self.raw_fullres_size = (self.size[0] * 2, self.size[1] * 2)
        self.rgb_transforms = [
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
        self.unprocessing = (
            build_unprocessing_transform_from_preset("stf_legacy", randomize=False)
            if self.input_type in RAW_LIKE_INPUT_TYPES
            else None
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read KITTI image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(str(row["depth_path"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read KITTI depth: {row['depth_path']}")
        depth = depth.astype(np.float32) / 256.0

        valid_mask = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        output = {
            "depth": torch.from_numpy(np.ascontiguousarray(depth)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)),
            "image_path": str(row["image_path"]),
            "depth_path": str(row["depth_path"]),
            "sample_name": row["image_path"].stem,
        }

        if self.input_type in RAW_LIKE_INPUT_TYPES:
            image_fullres = cv2.resize(
                image,
                (self.raw_fullres_size[1], self.raw_fullres_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            image_chw = np.transpose(np.ascontiguousarray(image_fullres), (2, 0, 1)).astype(np.float32, copy=False)
            raw, isp_params = self.unprocessing(torch.from_numpy(image_chw))
            output["raw"] = raw.float()
            output["isp_params"] = isp_params
            return output

        image_resized = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
        sample = {
            "image": image_resized,
            "depth": depth,
            "mask": valid_mask.astype(np.float32),
        }
        for transform in self.rgb_transforms:
            sample = transform(sample)

        mask = sample.pop("mask")
        output["image"] = torch.from_numpy(sample["image"])
        output["depth"] = torch.from_numpy(sample["depth"])
        output["valid_mask"] = torch.from_numpy(mask > 0.5)
        return output
