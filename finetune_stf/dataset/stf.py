import csv
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

from finetune_stf.dataset.transform import NormalizeImage, PrepareForNet, Resize


DEFAULT_STF_ROOT = "/home/caq/6666_raw/seeingthroughfog"
REQUIRED_COLUMNS = ("filename_stem", "lut_preview", "lidar_proj_left")


def _resolve_data_path(root, path_str):
    path = Path(path_str.strip()).expanduser()
    if path.is_absolute():
        return path
    return (Path(root).expanduser().resolve() / path).resolve()


def _load_manifest_rows(manifest_path, stf_root):
    rows = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required STF columns: {', '.join(missing)}"
            )

        for row in reader:
            rows.append(
                {
                    "sample_name": row["filename_stem"],
                    "image_path": _resolve_data_path(stf_root, row["lut_preview"]),
                    "depth_path": _resolve_data_path(stf_root, row["lidar_proj_left"]),
                }
            )
    return rows


def _load_depth_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "arr_0" not in data.files:
            raise KeyError(f"{path} does not contain arr_0")
        return np.array(data["arr_0"], dtype=np.float32, copy=True)


class STF(Dataset):
    def __init__(
        self,
        split,
        *,
        stf_root=DEFAULT_STF_ROOT,
        size=(512, 960),
        min_depth=1.0,
        max_depth=80.0,
        merge_test_into_train=True,
    ):
        self.split = split
        self.mode = "train" if split == "train" else "val"
        self.stf_root = Path(stf_root).expanduser().resolve()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.size = tuple(size)

        manifest_dir = self.stf_root / "manifests"
        if split == "train" and merge_test_into_train:
            manifest_paths = [
                manifest_dir / "stf_raw_depth_v1_train.csv",
                manifest_dir / "stf_raw_depth_v1_test.csv",
            ]
        else:
            manifest_paths = [manifest_dir / f"stf_raw_depth_v1_{split}.csv"]

        self.rows = []
        for manifest_path in manifest_paths:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing STF manifest: {manifest_path}")
            self.rows.extend(_load_manifest_rows(manifest_path, self.stf_root))

        if not self.rows:
            raise ValueError(f"No STF samples found for split={split}")

        height, width = self.size
        self.transform = Compose(
            [
                Resize(
                    width=width,
                    height=height,
                    resize_target=self.mode == "train",
                    keep_aspect_ratio=False,
                    ensure_multiple_of=1,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        depth_path = row["depth_path"]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read STF image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = _load_depth_npz(depth_path)
        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        sample = self.transform({"image": image, "depth": depth, "mask": valid_mask.astype(np.float32)})
        mask = sample.pop("mask")

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(mask > 0.5)
        sample["image_path"] = str(image_path)
        sample["depth_path"] = str(depth_path)
        sample["sample_name"] = row["sample_name"]
        sample["target_space"] = "metric_depth"
        return sample
