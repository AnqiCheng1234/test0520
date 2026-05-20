from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from finetune_stf.dataset.transform import Crop, NormalizeImage, PrepareForNet, Resize


DEFAULT_TRAIN_LIST = Path(__file__).resolve().parent / "splits" / "vkitti2" / "train.txt"


class VKITTI2(Dataset):
    def __init__(
        self,
        filelist_path=None,
        *,
        mode="train",
        size=(518, 966),
        min_depth=1.0,
        max_depth=80.0,
    ):
        self.mode = mode
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = tuple(size)
        self.filelist_path = Path(filelist_path or DEFAULT_TRAIN_LIST).expanduser().resolve()

        if not self.filelist_path.is_file():
            raise FileNotFoundError(
                "Missing VKITTI2 split file. Generate it with scripts/generate_vkitti2_split.py "
                f"or pass --vkitti-train-list. Expected: {self.filelist_path}"
            )

        with self.filelist_path.open("r", encoding="utf-8") as f:
            self.filelist = [line.strip() for line in f if line.strip()]

        if not self.filelist:
            raise ValueError(f"No VKITTI2 samples found in {self.filelist_path}")

        height, width = self.size
        transforms = [
            Resize(
                width=width,
                height=height,
                resize_target=self.mode == "train",
                keep_aspect_ratio=self.mode == "train",
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
        if self.mode == "train":
            transforms.append(Crop((height, width)))
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_path_str, depth_path_str = self.filelist[idx].split()
        img_path = Path(img_path_str)
        depth_path = Path(depth_path_str)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read VKITTI2 image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to read VKITTI2 depth: {depth_path}")
        depth = depth.astype(np.float32) / 100.0

        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        sample = self.transform({"image": image, "depth": depth, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["image_path"] = str(img_path)
        sample["depth_path"] = str(depth_path)
        sample["sample_name"] = f"{img_path.parent.name}_{img_path.stem}"
        sample["target_space"] = "metric_depth"
        return sample
