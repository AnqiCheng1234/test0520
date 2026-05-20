from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet


DEFAULT_NYU_DIR = "/mnt/drive/nyu/nyu_test"


class NYUv2Eval(Dataset):
    def __init__(
        self,
        nyu_dir: str | Path = DEFAULT_NYU_DIR,
        *,
        size=(512, 960),
        min_depth: float = 0.001,
        max_depth: float = 10.0,
    ) -> None:
        self.nyu_dir = Path(nyu_dir).expanduser().resolve()
        self.size = tuple(size)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        if not self.nyu_dir.is_dir():
            raise FileNotFoundError(f"Missing NYUv2 directory: {self.nyu_dir}")
        self.files = sorted(self.nyu_dir.glob("*.h5"))
        if not self.files:
            raise ValueError(f"No NYUv2 .h5 files found in {self.nyu_dir}")

        self.rgb_transforms = [
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        with h5py.File(h5_path, "r") as hf:
            rgb = hf["rgb"][()]
            depth = hf["depth"][()].astype(np.float32)
            mask = hf["mask"][()]

        if rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError(f"Expected NYUv2 rgb shape (3,H,W), got {rgb.shape} in {h5_path}")
        if depth.ndim != 2:
            raise ValueError(f"Expected NYUv2 depth shape (H,W), got {depth.shape} in {h5_path}")
        if mask.shape != depth.shape:
            raise ValueError(f"NYUv2 mask/depth shape mismatch in {h5_path}: {mask.shape} vs {depth.shape}")

        image = np.transpose(rgb, (1, 2, 0)).astype(np.float32) / 255.0
        valid_mask = (
            (mask > 0.5)
            & np.isfinite(depth)
            & (depth > self.min_depth)
            & (depth < self.max_depth)
        )
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        image_resized = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
        sample = {
            "image": image_resized,
            "depth": depth,
            "mask": valid_mask.astype(np.float32),
        }
        for transform in self.rgb_transforms:
            sample = transform(sample)

        return {
            "image": torch.from_numpy(sample["image"]),
            "depth": torch.from_numpy(sample["depth"]),
            "valid_mask": torch.from_numpy(sample["mask"] > 0.5),
            "image_path": str(h5_path),
            "depth_path": str(h5_path),
            "sample_name": h5_path.stem,
        }
