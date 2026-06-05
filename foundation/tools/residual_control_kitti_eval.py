from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from anqi_eval.eval_rel_depth_strict import affine_align_disp, compute_metrics
from foundation.tools.residual_training_common import METRIC_KEYS, format_seconds, save_json, to_jsonable


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_EVEN_HW = (374, 1242)
TARGET_MODEL_HW = (187, 621)
GEOMETRY_POLICY = "canonical_even_pad_crop"
MIN_VALID_PIXELS = 128
CONTROL_KITTI_EVAL_PROTOCOL = "halfres_rgb_canonical_even_pad_crop_affine_disp"


class KittiGeometryError(ValueError):
    pass


def remap_kitti_path(path_str: str, kitti_base: Path) -> Path:
    path = path_str.strip()
    raw_prefix = "/mnt/bn/liheyang/Kitti/raw_data/"
    depth_prefix = "/mnt/bn/liheyang/Kitti/data_depth_annotated/"
    if path.startswith(raw_prefix):
        rel_path = Path(path[len(raw_prefix) :])
        if rel_path.suffix.lower() == ".png":
            rel_path = rel_path.with_suffix(".jpg")
        return kitti_base / rel_path
    if path.startswith(depth_prefix):
        return kitti_base / "annotated_depth" / Path(path[len(depth_prefix) :])
    return Path(path)


def numpy_to_torch(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array))


def imagenet_normalize_rgb_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    normalized = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return numpy_to_torch(np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False))


def rgb_preview_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image_rgb = np.clip(image_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    return numpy_to_torch(np.transpose(image_rgb, (2, 0, 1)).astype(np.float32, copy=False))


def downsample_rgb_2x2_area(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise KittiGeometryError(f"Expected even fullres RGB shape, got {(h, w)}")
    return image_rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)).astype(np.float32, copy=False)


def downsample_depth_valid_mean_2x2(depth: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise KittiGeometryError(f"Expected even fullres depth shape, got {(h, w)}")
    depth_blocks = np.ascontiguousarray(depth).reshape(h // 2, 2, w // 2, 2)
    valid_blocks = np.ascontiguousarray(valid_mask).reshape(h // 2, 2, w // 2, 2)
    counts = valid_blocks.sum(axis=(1, 3)).astype(np.float32, copy=False)
    sums = (depth_blocks * valid_blocks.astype(np.float32, copy=False)).sum(axis=(1, 3))
    valid_half = counts > 0.0
    depth_half = np.zeros((h // 2, w // 2), dtype=np.float32)
    depth_half[valid_half] = sums[valid_half] / counts[valid_half]
    return depth_half, valid_half


def sample_name_from_image_path(image_path: Path) -> str:
    if image_path.parent.name == "data" and image_path.parent.parent.name.startswith("image_"):
        camera = image_path.parent.parent.name
        drive = image_path.parent.parent.parent.name
        return f"{drive}_{camera}_{image_path.stem}"
    return image_path.stem


def center_crop_pad_params(source_hw: tuple[int, int], target_hw: tuple[int, int]) -> dict[str, int]:
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if source_h <= 0 or source_w <= 0:
        raise KittiGeometryError(f"Invalid source shape for canonicalization: {source_hw}")
    h_crop = max(source_h - target_h, 0)
    w_crop = max(source_w - target_w, 0)
    top_crop = h_crop // 2
    bottom_crop = h_crop - top_crop
    left_crop = w_crop // 2
    right_crop = w_crop - left_crop
    cropped_h = source_h - top_crop - bottom_crop
    cropped_w = source_w - left_crop - right_crop
    h_pad = max(target_h - cropped_h, 0)
    w_pad = max(target_w - cropped_w, 0)
    top_pad = h_pad // 2
    bottom_pad = h_pad - top_pad
    left_pad = w_pad // 2
    right_pad = w_pad - left_pad
    return {
        "top_pad": int(top_pad),
        "bottom_pad": int(bottom_pad),
        "left_pad": int(left_pad),
        "right_pad": int(right_pad),
        "top_crop": int(top_crop),
        "bottom_crop": int(bottom_crop),
        "left_crop": int(left_crop),
        "right_crop": int(right_crop),
    }


def apply_canonical_even_pad_crop(
    image: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    target_hw: tuple[int, int] = TARGET_EVEN_HW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    source_hw = tuple(int(v) for v in image.shape[:2])
    params = center_crop_pad_params(source_hw, target_hw)
    h, w = source_hw
    top_crop = params["top_crop"]
    bottom_crop = params["bottom_crop"]
    left_crop = params["left_crop"]
    right_crop = params["right_crop"]
    h_end = h - bottom_crop
    w_end = w - right_crop
    image = np.ascontiguousarray(image[top_crop:h_end, left_crop:w_end, :])
    depth = np.ascontiguousarray(depth[top_crop:h_end, left_crop:w_end])
    valid_mask = np.ascontiguousarray(valid_mask[top_crop:h_end, left_crop:w_end])

    pad_hw = (
        (params["top_pad"], params["bottom_pad"]),
        (params["left_pad"], params["right_pad"]),
    )
    if any(v > 0 for key, v in params.items() if key.endswith("_pad")):
        image = np.pad(image, (*pad_hw, (0, 0)), mode="edge")
        depth = np.pad(depth, pad_hw, mode="constant", constant_values=0.0)
        valid_mask = np.pad(valid_mask, pad_hw, mode="constant", constant_values=False)

    if tuple(image.shape[:2]) != tuple(target_hw):
        raise KittiGeometryError(f"Canonical KITTI shape mismatch: got={tuple(image.shape[:2])} target={target_hw}")
    return image, depth.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False), params


class KittiHalfresRGBDepthDataset(Dataset):
    def __init__(
        self,
        *,
        filelist_path: str | Path,
        kitti_base: str | Path,
        min_depth: float,
        max_depth: float,
    ) -> None:
        self.filelist_path = Path(filelist_path).expanduser().resolve()
        self.kitti_base = Path(kitti_base).expanduser().resolve()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

        if not self.filelist_path.is_file():
            raise FileNotFoundError(f"Missing KITTI split file: {self.filelist_path}")
        if not self.kitti_base.is_dir():
            raise FileNotFoundError(f"Missing KITTI base directory: {self.kitti_base}")

        self.rows: list[dict[str, Any]] = []
        with self.filelist_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                image_path_str, depth_path_str = line.split()
                image_path = remap_kitti_path(image_path_str, self.kitti_base)
                depth_path = remap_kitti_path(depth_path_str, self.kitti_base)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing KITTI image on split row {line_idx}: {image_path}")
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing KITTI depth on split row {line_idx}: {depth_path}")
                self.rows.append(
                    {
                        "image_path": image_path,
                        "depth_path": depth_path,
                        "raw_image_path": image_path_str,
                        "raw_depth_path": depth_path_str,
                    }
                )
        if not self.rows:
            raise ValueError(f"No KITTI samples found in {self.filelist_path}")

        self.source_shape_counts = self._scan_source_shape_counts()

    def __len__(self) -> int:
        return len(self.rows)

    def _scan_source_shape_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in self.rows:
            image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read KITTI image while scanning shapes: {row['image_path']}")
            h, w = image.shape[:2]
            counts[f"{int(h)}x{int(w)}"] += 1
        return dict(sorted(counts.items()))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        try:
            sample = self.build_sample(idx)
            sample["status"] = "ok"
            return sample
        except FileNotFoundError as exc:
            return self._error_sample(idx, "skipped_io_error", exc)
        except KittiGeometryError as exc:
            return self._error_sample(idx, "skipped_geometry_error", exc)
        except (OSError, ValueError, cv2.error) as exc:
            return self._error_sample(idx, "skipped_io_error", exc)

    def _error_sample(self, idx: int, status: str, exc: BaseException) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "dataset_index": int(idx),
            "sample_name": sample_name_from_image_path(row["image_path"]),
            "image_path": str(row["image_path"]),
            "depth_path": str(row["depth_path"]),
            "status": status,
            "error": str(exc),
        }

    def build_sample(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = row["image_path"]
        depth_path = row["depth_path"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read KITTI image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        source_hw = [int(image.shape[0]), int(image.shape[1])]

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Failed to read KITTI depth: {depth_path}")
        depth = depth_raw.astype(np.float32) / 256.0
        if tuple(depth.shape[:2]) != tuple(image.shape[:2]):
            raise KittiGeometryError(
                f"KITTI RGB/depth shape mismatch for idx={idx}: image={image.shape[:2]} depth={depth.shape[:2]}"
            )

        valid_mask = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.where(valid_mask, depth, 0.0).astype(np.float32, copy=False)

        cropped_bottom_rows = 0
        if image.shape[0] % 2 != 0:
            image = np.ascontiguousarray(image[:-1, :, :])
            depth = np.ascontiguousarray(depth[:-1, :])
            valid_mask = np.ascontiguousarray(valid_mask[:-1, :])
            cropped_bottom_rows = 1
        after_even_policy_hw = [int(image.shape[0]), int(image.shape[1])]

        image, depth, valid_mask, pad_crop = apply_canonical_even_pad_crop(image, depth, valid_mask)
        canonical_even_hw = [int(image.shape[0]), int(image.shape[1])]
        if tuple(canonical_even_hw) != TARGET_EVEN_HW:
            raise KittiGeometryError(f"Expected canonical KITTI HW {TARGET_EVEN_HW}, got {canonical_even_hw}")

        rgb_half = downsample_rgb_2x2_area(image)
        depth_half, valid_half = downsample_depth_valid_mean_2x2(depth, valid_mask)
        if tuple(rgb_half.shape[:2]) != TARGET_MODEL_HW:
            raise KittiGeometryError(f"Halfres RGB shape mismatch: got={rgb_half.shape[:2]}")
        if tuple(depth_half.shape) != TARGET_MODEL_HW or tuple(valid_half.shape) != TARGET_MODEL_HW:
            raise KittiGeometryError(
                f"Halfres depth/mask shape mismatch: depth={depth_half.shape} valid={valid_half.shape}"
            )

        geometry = {
            "source_hw": source_hw,
            "after_even_policy_hw": after_even_policy_hw,
            "canonical_even_hw": canonical_even_hw,
            "target_even_fullres_hw": list(TARGET_EVEN_HW),
            "target_model_hw": list(TARGET_MODEL_HW),
            "geometry_policy": GEOMETRY_POLICY,
            "fullres_even_policy": "crop_bottom_to_even",
            "cropped_bottom_rows": int(cropped_bottom_rows),
            "pad_crop": pad_crop,
        }
        return {
            "dataset_index": int(idx),
            "image": imagenet_normalize_rgb_tensor(rgb_half),
            "rgb_preview": rgb_preview_tensor(rgb_half),
            "depth": numpy_to_torch(depth_half.astype(np.float32, copy=False)),
            "valid_mask": numpy_to_torch(valid_half.astype(np.uint8)).bool(),
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "sample_name": sample_name_from_image_path(image_path),
            "geometry_params": geometry,
            "target_space": "metric_depth",
        }

    def describe_geometry(self) -> dict[str, Any]:
        return {
            "name": GEOMETRY_POLICY,
            "target_even_fullres_hw": list(TARGET_EVEN_HW),
            "target_model_hw": list(TARGET_MODEL_HW),
            "fullres_even_policy": "crop_bottom_to_even",
            "canonicalization": "center pad/crop to fixed even fullres canvas; RGB edge pad; depth zero pad; valid false pad",
            "source_shape_counts": self.source_shape_counts,
            "input_domain": "rgb",
            "raw_storage_format": "not_applicable",
        }


def collate_single_sample(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError(f"Expected batch_size=1 for KITTI eval, got {len(batch)}")
    return batch[0]


def build_kitti_val_loader(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[KittiHalfresRGBDepthDataset | None, DataLoader | None]:
    if not args.eval_kitti:
        return None, None

    dataset = KittiHalfresRGBDepthDataset(
        filelist_path=args.kitti_val_split,
        kitti_base=args.kitti_base,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    if args.kitti_expected_val_samples is not None and len(dataset) != int(args.kitti_expected_val_samples):
        raise RuntimeError(f"Expected KITTI val length {int(args.kitti_expected_val_samples)}, got {len(dataset)}")
    workers = int(args.kitti_num_workers) if args.kitti_num_workers is not None else max(min(args.num_workers, 2), 0)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_single_sample,
        persistent_workers=workers > 0,
    )
    return dataset, loader


def filter_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    if metrics is None:
        return {key: None for key in METRIC_KEYS}
    return {key: to_jsonable(metrics.get(key)) for key in METRIC_KEYS}


def mean_finite(values: list[Any]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(finite))


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {key: mean_finite([row.get(key) for row in rows]) for key in METRIC_KEYS}


def make_base_row(sample: dict[str, Any]) -> dict[str, Any]:
    geometry = sample.get("geometry_params", {})
    return {
        "dataset_index": int(sample["dataset_index"]),
        "sample_name": str(sample["sample_name"]),
        "image_path": str(sample["image_path"]),
        "depth_path": str(sample["depth_path"]),
        "status": str(sample.get("status", "ok")),
        "source_hw": geometry.get("source_hw"),
        "after_even_policy_hw": geometry.get("after_even_policy_hw"),
        "canonical_even_hw": geometry.get("canonical_even_hw"),
        "target_model_hw": geometry.get("target_model_hw"),
        "geometry_policy": geometry.get("geometry_policy", GEOMETRY_POLICY),
        "pad_crop": geometry.get("pad_crop"),
        "valid_pixels": None,
        "final": filter_metrics(None),
        "D0": filter_metrics(None),
        "diagnostics": {
            "mean_gate": None,
            "max_gate": None,
            "mean_abs_delta": None,
            "mean_abs_gate_delta": None,
        },
    }


def collect_eval_for_sample(
    *,
    sample: dict[str, Any],
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    row = make_base_row(sample)
    if sample.get("status") != "ok":
        row["error"] = sample.get("error")
        return row

    image = sample["image"].unsqueeze(0).to(device, non_blocking=True).float()
    depth_t = sample["depth"].unsqueeze(0).to(device, non_blocking=True).float()
    valid_t = sample["valid_mask"].unsqueeze(0).to(device, non_blocking=True).bool()
    valid_t = valid_t & (depth_t >= float(config["min_depth"])) & (depth_t <= float(config["max_depth"]))
    valid_pixels = int(valid_t[0].sum().item())
    row["valid_pixels"] = valid_pixels
    if valid_pixels < MIN_VALID_PIXELS:
        row["status"] = "skipped_invalid_pixels"
        row["error"] = f"valid_pixels={valid_pixels} < {MIN_VALID_PIXELS}"
        return row

    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=amp_enabled and device.type == "cuda",
        ):
            out = model({"image": image, "valid_mask": valid_t})

        depth_np = depth_t[0].detach().cpu().numpy().astype(np.float32)
        valid_np = valid_t[0].detach().cpu().numpy().astype(bool)
        pred_np = out["pred"][0].float().detach().cpu().numpy().astype(np.float32)
        d0_np = (float(config["d0_sign"]) * out["D0"][0].float()).detach().cpu().numpy().astype(np.float32)
        aligned_final, _ = affine_align_disp(depth_np, pred_np, valid_np)
        aligned_d0, _ = affine_align_disp(depth_np, d0_np, valid_np)
        metrics_final = compute_metrics(
            depth_np,
            aligned_final,
            valid_np,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )
        metrics_d0 = compute_metrics(
            depth_np,
            aligned_d0,
            valid_np,
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
        )
        if metrics_final is None or metrics_d0 is None:
            row["status"] = "skipped_metric_failure"
            row["error"] = "compute_metrics returned None"
            return row

        gate = out["gate"].float()
        delta = out["delta"].float()
        gate_delta = gate * delta
        row["status"] = "ok"
        row["final"] = filter_metrics(metrics_final)
        row["D0"] = filter_metrics(metrics_d0)
        row["diagnostics"] = {
            "mean_gate": float(gate[valid_t].mean().detach().item()),
            "max_gate": float(gate[valid_t].max().detach().item()),
            "mean_abs_delta": float(delta[valid_t].abs().mean().detach().item()),
            "mean_abs_gate_delta": float(gate_delta[valid_t].abs().mean().detach().item()),
        }
        return row
    except Exception as exc:  # noqa: BLE001 - per-sample eval should produce status rows.
        row["status"] = "skipped_metric_failure"
        row["error"] = str(exc)
        return row


def evaluate_control_kitti_model(
    model: torch.nn.Module,
    dataset: KittiHalfresRGBDepthDataset,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epoch: int,
    amp_dtype: torch.dtype,
    logger: Any,
    output_dir: Path,
) -> dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_visit = len(dataset) if args.max_kitti_val_samples is None else min(int(args.max_kitti_val_samples), len(dataset))
    amp_enabled = bool(args.amp) and device.type == "cuda"
    config = dict(vars(args))
    rows: list[dict[str, Any]] = []
    ok_metric_rows: list[dict[str, Any]] = []
    start = time.time()
    per_sample_path = output_dir / "per_sample.jsonl"

    logger.info(
        "[EVAL][KITTI] start epoch=%d dataset_samples=%d max_visit=%d max_kitti_val_samples=%s",
        epoch,
        len(dataset),
        max_visit,
        args.max_kitti_val_samples,
    )
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for visited, sample in enumerate(dataloader):
            if visited >= max_visit:
                break
            row = collect_eval_for_sample(
                sample=sample,
                model=model,
                config=config,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            row["epoch"] = int(epoch)
            rows.append(row)
            if row["status"] == "ok":
                ok_metric_rows.append(row)
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
            if (visited + 1) % 50 == 0 or visited + 1 == max_visit:
                logger.info(
                    "[EVAL][KITTI] processed=%d/%d ok=%d elapsed=%s",
                    visited + 1,
                    max_visit,
                    len(ok_metric_rows),
                    format_seconds(time.time() - start),
                )

    if not ok_metric_rows:
        raise RuntimeError("KITTI eval produced zero ok samples.")

    final_metrics = [row["final"] for row in ok_metric_rows]
    d0_metrics = [row["D0"] for row in ok_metric_rows]
    overall_final = average_metrics(final_metrics)
    overall_d0 = average_metrics(d0_metrics)
    delta = {
        "final_abs_rel_minus_D0_abs_rel": (
            None
            if overall_final["abs_rel"] is None or overall_d0["abs_rel"] is None
            else overall_final["abs_rel"] - overall_d0["abs_rel"]
        ),
        "final_d1_minus_D0_d1": (
            None
            if overall_final["d1"] is None or overall_d0["d1"] is None
            else overall_final["d1"] - overall_d0["d1"]
        ),
    }
    status_counts = Counter(str(row["status"]) for row in rows)
    elapsed_seconds = time.time() - start
    summary = {
        "dataset": "kitti_val_halfres_rgb_control",
        "epoch": int(epoch),
        "dataset_samples": int(len(dataset)),
        "visited_samples": int(len(rows)),
        "samples": int(len(ok_metric_rows)),
        "max_val_samples": args.max_kitti_val_samples,
        "kitti_val_split": str(Path(args.kitti_val_split).expanduser().resolve()),
        "kitti_base": str(Path(args.kitti_base).expanduser().resolve()),
        "eval_protocol": args.kitti_eval_protocol,
        "note": (
            "KITTI val is evaluated with canonical_even_pad_crop to match the VKITTI-trained "
            "fixed 187x621 RGB/D0 control residual model; scores are not KITTI public benchmark settings."
        ),
        "geometry_policy": dataset.describe_geometry(),
        "status_counts": dict(status_counts),
        "overall": {"final": overall_final, "D0": overall_d0, "delta": delta},
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_visited_sample": float(elapsed_seconds / max(len(rows), 1)),
        "per_sample_path": str(per_sample_path),
    }
    save_json(output_dir / "metrics.json", summary)
    logger.info(
        "[EVAL][KITTI] done epoch=%d visited=%d ok=%d final_abs_rel=%.5f D0_abs_rel=%.5f "
        "delta_abs_rel=%.5f final_d1=%.5f D0_d1=%.5f elapsed=%s",
        epoch,
        len(rows),
        len(ok_metric_rows),
        float(overall_final["abs_rel"]),
        float(overall_d0["abs_rel"]),
        float(delta["final_abs_rel_minus_D0_abs_rel"]),
        float(overall_final["d1"]),
        float(overall_d0["d1"]),
        format_seconds(elapsed_seconds),
    )
    return summary
