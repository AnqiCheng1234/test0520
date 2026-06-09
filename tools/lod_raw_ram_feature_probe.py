#!/usr/bin/env python3
"""Probe RAW-RAM features for LOD RAW ablation inputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from finetune_stf.dataset.lod_true import LODTrueRawDarkRGB16  # noqa: E402
from finetune_stf.models.raw_ram import phase1b_tanh_tail_squash  # noqa: E402
from finetune_stf.train import build_model  # noqa: E402
from finetune_stf.util.model_input import select_model_input  # noqa: E402
from lod_raw_per_sample_eval import (  # noqa: E402
    amp_dtype_from_args,
    load_checkpoint_into_model,
    load_run_payloads,
    planned_indices,
    train_args_from_eval_args,
    validate_semantics,
)


@dataclass(frozen=True)
class ManifestSpec:
    label: str
    manifest: Path
    dataset_family: str
    dataset_input_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--dataset-family", required=True)
    parser.add_argument("--dataset-input-mode", required=True)
    parser.add_argument("--input-domain", required=True)
    parser.add_argument("--front-end", required=True)
    parser.add_argument("--model-input-tensor", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--decoder-feature-adapter", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--lora-block-mode", default="tap")
    parser.add_argument("--lora-tap-layers", nargs="+", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--raw-front-end-lr", type=float, default=None)
    parser.add_argument("--lora-lr", type=float, default=None)
    parser.add_argument("--raw-storage-format", default="raw_rgb16_png_3ch")
    parser.add_argument("--lod-raw-norm-mode", default="uint16_div_65535")
    parser.add_argument("--raw-ram-rgb-tail", default="identity")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--pretrained-from", default=None)
    parser.add_argument("--dav2-train-mode", default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--lod-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-spec",
        action="append",
        required=True,
        help=(
            "Repeatable input spec: label=PATH[,dataset_family=VALUE,dataset_input_mode=VALUE]. "
            "The dataset fields are explicit per input and may differ from the checkpoint run "
            "for same-model oracle sanity probes."
        ),
    )
    parser.add_argument("--lod-label-space", default="inverse_relative")
    parser.add_argument("--split", default="01Valid")
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--include-dav2-tokens", action="store_true")
    parser.add_argument("--dav2-layers", default="2,5,8,11")
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_manifest_spec(value: str, args: argparse.Namespace) -> ManifestSpec:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts or "=" not in parts[0]:
        raise ValueError(
            f"Invalid --manifest-spec {value!r}. Expected label=PATH[,dataset_family=...,...]"
        )
    label, raw_path = parts[0].split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid --manifest-spec {value!r}: empty label")
    extras: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Invalid manifest spec option {part!r} in {value!r}")
        key, item_value = part.split("=", 1)
        extras[key.strip()] = item_value.strip()
    allowed = {"dataset_family", "dataset_input_mode"}
    unknown = sorted(set(extras) - allowed)
    if unknown:
        raise ValueError(f"Unknown manifest spec fields for {label}: {unknown}")
    return ManifestSpec(
        label=label,
        manifest=Path(raw_path).expanduser().resolve(),
        dataset_family=extras.get("dataset_family", str(args.dataset_family)),
        dataset_input_mode=extras.get("dataset_input_mode", str(args.dataset_input_mode)),
    )


def parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            layers.append(int(item))
    if not layers:
        raise ValueError("--dav2-layers must contain at least one layer index")
    return layers


def train_args_for_manifest(
    base_args: argparse.Namespace,
    config_payload: dict[str, Any],
    manifest: Path,
) -> argparse.Namespace:
    local_args = copy.copy(base_args)
    local_args.lod_manifest = manifest
    return train_args_from_eval_args(local_args, config_payload)


def make_dataset(args: argparse.Namespace, train_args: argparse.Namespace, spec: ManifestSpec) -> LODTrueRawDarkRGB16:
    if not spec.manifest.is_file():
        raise FileNotFoundError(f"Missing manifest for {spec.label}: {spec.manifest}")
    if spec.dataset_family not in {"lod_true_raw_dark_rgb16", "lod_true_raw_normal_rgb16"}:
        raise ValueError(
            f"Unsupported dataset_family for {spec.label}: {spec.dataset_family!r}. "
            "Expected lod_true_raw_dark_rgb16 or lod_true_raw_normal_rgb16."
        )
    return LODTrueRawDarkRGB16(
        lod_root=train_args.lod_root,
        manifest_path=spec.manifest,
        split=str(args.split),
        size=(int(train_args.input_height), int(train_args.input_width)),
        mode="val",
        label_space=train_args.lod_label_space,
        crop_mode=train_args.lod_val_crop_mode,
        raw_storage_format=train_args.raw_storage_format,
        lod_raw_norm_mode=train_args.lod_raw_norm_mode,
        raw_input_mode=str(spec.dataset_input_mode),
    )


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    arr = tensor.detach().float().cpu().numpy()
    flat = arr.reshape(-1)
    stats = {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std": float(np.std(flat)),
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_max": float(np.max(flat)),
        f"{prefix}_p1": float(np.percentile(flat, 1.0)),
        f"{prefix}_p50": float(np.percentile(flat, 50.0)),
        f"{prefix}_p99": float(np.percentile(flat, 99.0)),
        f"{prefix}_l2_rms": float(np.sqrt(np.mean(np.square(flat)))),
    }
    return stats


def channel_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"Expected {prefix} tensor shape (1,C,H,W), got {tuple(arr.shape)}")
    arr = arr[0]
    rows: dict[str, float] = {}
    for channel in range(arr.shape[0]):
        channel_values = arr[channel].reshape(-1)
        rows[f"{prefix}_c{channel}_mean"] = float(np.mean(channel_values))
        rows[f"{prefix}_c{channel}_std"] = float(np.std(channel_values))
    if arr.shape[0] >= 2:
        reshaped = arr.reshape(arr.shape[0], -1)
        corr = np.corrcoef(reshaped)
        off_diag = corr[np.triu_indices(arr.shape[0], k=1)]
        rows[f"{prefix}_channel_corr_mean"] = float(np.nanmean(off_diag))
        rows[f"{prefix}_channel_corr_min"] = float(np.nanmin(off_diag))
        rows[f"{prefix}_channel_corr_max"] = float(np.nanmax(off_diag))
    return rows


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().cpu().reshape(-1)
    b_flat = b.detach().float().cpu().reshape(-1)
    denom = torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat)
    if float(denom) <= 0.0:
        return float("nan")
    return float(1.0 - torch.dot(a_flat, b_flat) / denom)


def l1_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a_cpu = a.detach().float().cpu()
    b_cpu = b.detach().float().cpu()
    return float(torch.mean(torch.abs(a_cpu - b_cpu)))


def l2_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    delta = a.detach().float().cpu() - b.detach().float().cpu()
    return float(torch.sqrt(torch.mean(delta * delta)))


def summarize_rows(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for label, group_rows in sorted(grouped.items()):
        out: dict[str, Any] = {group_key: label, "rows": len(group_rows)}
        keys = sorted({key for row in group_rows for key in row})
        for key in keys:
            values: list[float] = []
            for row in group_rows:
                try:
                    value = float(row[key])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values))
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]], distance_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    interesting = [
        "ram_x3_tail_p1_mean",
        "ram_x3_tail_p50_mean",
        "ram_x3_tail_p99_mean",
        "ram_x3_tail_std_mean",
        "ram_x3_tail_abs_gt_2p5_mean",
        "dav2_layer2_tokens_std_mean",
        "dav2_layer5_tokens_std_mean",
    ]
    lines = ["# LOD RAW RAM feature probe", ""]
    lines.append("## Summary by label")
    lines.append("")
    header = ["label", "rows", *interesting]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in summary_rows:
        values = [str(row.get("label", "")), str(row.get("rows", ""))]
        for key in interesting:
            value = row.get(key)
            values.append("" if value is None else f"{float(value):.6g}")
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Pairwise distance to reference")
    lines.append("")
    distance_keys = [
        "ram_x3_raw_l1_mean_mean",
        "ram_x3_tail_l1_mean_mean",
        "ram_x3_tail_cosine_distance_mean",
        "dav2_layer2_tokens_cosine_distance_mean",
        "dav2_layer5_tokens_cosine_distance_mean",
        "dav2_layer8_tokens_cosine_distance_mean",
        "dav2_layer11_tokens_cosine_distance_mean",
    ]
    header = ["label", "reference_label", "rows", *distance_keys]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    distance_summary = summarize_rows(distance_rows, "label")
    reference_by_label = {
        str(row["label"]): str(row.get("reference_label", ""))
        for row in distance_rows
    }
    for row in distance_summary:
        values = [
            str(row.get("label", "")),
            reference_by_label.get(str(row.get("label", "")), ""),
            str(row.get("rows", "")),
        ]
        for key in distance_keys:
            value = row.get(key)
            values.append("" if value is None else f"{float(value):.6g}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_ram_features(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    *,
    train_args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    include_dav2_tokens: bool,
    dav2_layers: list[int],
) -> dict[str, torch.Tensor]:
    model_input = model_input.to(device=device, non_blocking=True).float()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        x3_raw, ram_features = model.ram_core.forward_with_features(model_input)
        x3_tail = x3_raw
        if str(getattr(train_args, "raw_ram_rgb_tail", "identity")) == "tanh2p5":
            x3_tail = phase1b_tanh_tail_squash(x3_tail)
        out = {
            "ram_x3_raw": x3_raw.detach(),
            "ram_x3_tail": x3_tail.detach(),
            "ram_x_cat": ram_features["x_cat"].detach(),
            "ram_ffm_mid": ram_features["ffm_mid"].detach(),
        }
        if include_dav2_tokens:
            x_norm = model.spatial_adapter.pad_rgb(x3_tail)
            features = model.dav2.pretrained.get_intermediate_layers(
                x_norm,
                dav2_layers,
                return_class_token=True,
            )
            for layer_idx, (patch_tokens, cls_token) in zip(dav2_layers, features):
                out[f"dav2_layer{layer_idx}_tokens"] = patch_tokens.detach()
                out[f"dav2_layer{layer_idx}_cls"] = cls_token.detach()
    return out


def stats_for_feature_set(features: dict[str, torch.Tensor], *, include_dav2_tokens: bool, dav2_layers: list[int]) -> dict[str, float]:
    row: dict[str, float] = {}
    for name in ("ram_x3_raw", "ram_x3_tail", "ram_x_cat", "ram_ffm_mid"):
        row.update(tensor_stats(name, features[name]))
    row.update(channel_stats("ram_x3_tail", features["ram_x3_tail"]))
    row["ram_x3_raw_abs_gt_2p5"] = float(torch.mean((torch.abs(features["ram_x3_raw"]) > 2.5).float()))
    row["ram_x3_tail_abs_gt_2p5"] = float(torch.mean((torch.abs(features["ram_x3_tail"]) > 2.5).float()))
    if include_dav2_tokens:
        for layer_idx in dav2_layers:
            token_name = f"dav2_layer{layer_idx}_tokens"
            cls_name = f"dav2_layer{layer_idx}_cls"
            row.update(tensor_stats(token_name, features[token_name]))
            row.update(tensor_stats(cls_name, features[cls_name]))
    return row


def distance_to_reference(
    label: str,
    reference_label: str,
    features: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    include_dav2_tokens: bool,
    dav2_layers: list[int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "reference_label": reference_label,
    }
    for name in ("ram_x3_raw", "ram_x3_tail", "ram_x_cat", "ram_ffm_mid"):
        row[f"{name}_l1_mean"] = l1_mean(features[name], reference[name])
        row[f"{name}_l2_rms"] = l2_rms(features[name], reference[name])
        row[f"{name}_cosine_distance"] = cosine_distance(features[name], reference[name])
    if include_dav2_tokens:
        for layer_idx in dav2_layers:
            for suffix in ("tokens", "cls"):
                name = f"dav2_layer{layer_idx}_{suffix}"
                row[f"{name}_l1_mean"] = l1_mean(features[name], reference[name])
                row[f"{name}_l2_rms"] = l2_rms(features[name], reference[name])
                row[f"{name}_cosine_distance"] = cosine_distance(features[name], reference[name])
    return row


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    resolved_payload, config_payload, run_config_paths = load_run_payloads(run_dir)
    validate_semantics(args, resolved_payload)
    manifest_specs = [parse_manifest_spec(value, args) for value in args.manifest_spec]
    labels = [spec.label for spec in manifest_specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate manifest spec labels: {labels}")
    if args.reference_label not in labels:
        raise ValueError(f"--reference-label {args.reference_label!r} not in manifest labels {labels}")
    base_train_args = train_args_for_manifest(args, config_payload, manifest_specs[0].manifest)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = build_model(base_train_args)
    ckpt_meta = load_checkpoint_into_model(model, checkpoint)
    if not hasattr(model, "ram_core"):
        raise ValueError(f"Model built from {args.front_end!r} does not expose ram_core")
    model.to(device)
    model.eval()
    use_amp = device.type == "cuda" and bool(getattr(base_train_args, "amp", False))
    amp_dtype = amp_dtype_from_args(base_train_args)
    dav2_layers = parse_layers(args.dav2_layers)
    datasets: dict[str, LODTrueRawDarkRGB16] = {}
    train_args_by_label: dict[str, argparse.Namespace] = {}
    for spec in manifest_specs:
        train_args = train_args_for_manifest(args, config_payload, spec.manifest)
        datasets[spec.label] = make_dataset(args, train_args, spec)
        train_args_by_label[spec.label] = train_args
    lengths = {label: len(dataset) for label, dataset in datasets.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All manifest specs must have equal split length, got {lengths}")
    dataset_len = next(iter(lengths.values()))
    indices = planned_indices(args, dataset_len)
    print(
        f"[RAM_PROBE] model_label={args.label or checkpoint.parent.name} specs={labels} "
        f"split={args.split} samples={len(indices)}/{dataset_len} device={device} "
        f"include_dav2_tokens={bool(args.include_dav2_tokens)}",
        flush=True,
    )
    per_sample_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    for ordinal, idx in enumerate(indices):
        sample_features: dict[str, dict[str, torch.Tensor]] = {}
        sample_meta: dict[str, dict[str, Any]] = {}
        for spec in manifest_specs:
            train_args = train_args_by_label[spec.label]
            sample = datasets[spec.label].build_sample(int(idx), include_geometry=False)
            model_input = select_model_input(
                sample,
                train_args.resolved_config.model_input_tensor,
                dataset_family=train_args.resolved_config.dataset_family,
                sample_source=f"lod_{args.split}_{spec.label}",
                add_batch_dim=True,
            )
            features = extract_ram_features(
                model,
                model_input,
                train_args=base_train_args,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                include_dav2_tokens=bool(args.include_dav2_tokens),
                dav2_layers=dav2_layers,
            )
            sample_features[spec.label] = features
            sample_meta[spec.label] = sample
            row: dict[str, Any] = {
                "label": spec.label,
                "sample_index": int(idx),
                "sample_id": str(sample["sample_id"]),
                "student_input_path": str(sample["student_input_path"]),
                "manifest": str(spec.manifest),
                "dataset_family": spec.dataset_family,
                "dataset_input_mode": spec.dataset_input_mode,
                "checkpoint": str(checkpoint),
                "checkpoint_label": str(args.label or checkpoint.parent.name),
            }
            row.update(stats_for_feature_set(features, include_dav2_tokens=bool(args.include_dav2_tokens), dav2_layers=dav2_layers))
            per_sample_rows.append(row)
        reference = sample_features[args.reference_label]
        reference_id = str(sample_meta[args.reference_label]["sample_id"])
        for spec in manifest_specs:
            if spec.label == args.reference_label:
                continue
            sample_id = str(sample_meta[spec.label]["sample_id"])
            if sample_id != reference_id:
                raise ValueError(
                    f"Sample id mismatch at index {idx}: {spec.label} has {sample_id}, "
                    f"reference has {reference_id}"
                )
            row = distance_to_reference(
                spec.label,
                args.reference_label,
                sample_features[spec.label],
                reference,
                include_dav2_tokens=bool(args.include_dav2_tokens),
                dav2_layers=dav2_layers,
            )
            row.update(
                {
                    "sample_index": int(idx),
                    "sample_id": sample_id,
                    "label_input_path": str(sample_meta[spec.label]["student_input_path"]),
                    "reference_input_path": str(sample_meta[args.reference_label]["student_input_path"]),
                }
            )
            distance_rows.append(row)
        if (ordinal + 1) % max(int(args.progress_interval), 1) == 0 or ordinal + 1 == len(indices):
            print(f"[RAM_PROBE] processed={ordinal + 1}/{len(indices)}", flush=True)
    output_dir = args.output_dir.expanduser().resolve()
    per_sample_csv = output_dir / "ram_feature_per_sample.csv"
    distance_csv = output_dir / "ram_feature_distance_to_reference.csv"
    summary_json = output_dir / "ram_feature_summary.json"
    summary_md = output_dir / "ram_feature_summary.md"
    summary_rows = summarize_rows(per_sample_rows, "label")
    distance_summary_rows = summarize_rows(distance_rows, "label")
    payload = {
        "model_label": str(args.label or checkpoint.parent.name),
        "checkpoint": str(checkpoint),
        "checkpoint_meta": ckpt_meta,
        "run_config_paths": run_config_paths,
        "split": str(args.split),
        "samples": len(indices),
        "dataset_len": dataset_len,
        "reference_label": str(args.reference_label),
        "include_dav2_tokens": bool(args.include_dav2_tokens),
        "dav2_layers": dav2_layers,
        "manifest_specs": [
            {
                "label": spec.label,
                "manifest": str(spec.manifest),
                "dataset_family": spec.dataset_family,
                "dataset_input_mode": spec.dataset_input_mode,
            }
            for spec in manifest_specs
        ],
        "summary_by_label": summary_rows,
        "distance_to_reference_summary": distance_summary_rows,
    }
    write_csv(per_sample_csv, per_sample_rows)
    write_csv(distance_csv, distance_rows)
    write_json(summary_json, payload)
    write_markdown_summary(summary_md, summary_rows, distance_rows)
    print(
        f"[RAM_PROBE] wrote output_dir={output_dir} rows={len(per_sample_rows)} "
        f"distance_rows={len(distance_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
