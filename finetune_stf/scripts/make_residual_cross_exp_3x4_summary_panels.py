#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack existing residual 3x4 panel manifests into per-sample cross-experiment summaries."
    )
    parser.add_argument("--split", required=True, choices=("kitti", "vkitti"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="KEY|LABEL|MANIFEST",
        help="Experiment source. Repeat in display order.",
    )
    parser.add_argument("--header-height", type=int, default=48)
    parser.add_argument("--gap", type=int, default=1)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_source(spec: str) -> tuple[str, str, Path]:
    parts = spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"--source must be KEY|LABEL|MANIFEST, got: {spec}")
    key, label, manifest = parts
    if not key.strip() or not label.strip() or not manifest.strip():
        raise ValueError(f"--source has an empty field: {spec}")
    return key.strip(), label.strip(), Path(manifest).expanduser().resolve()


def sample_slug(split: str, record: dict[str, Any]) -> str:
    order = int(record["order"])
    idx = int(record["dataset_index"])
    sample = str(record["sample_name"]).replace("/", "_")
    if split == "vkitti":
        variant = str(record.get("variant") or "unknown")
        return f"{order:02d}_vkitti_idx{idx:04d}_{variant}_{sample}_cross_exp_3x4_summary.jpg"
    return f"{order:02d}_kitti_idx{idx:04d}_{sample}_cross_exp_3x4_summary.jpg"


def metric_line(record: dict[str, Any]) -> str:
    epoch = record.get("epoch")
    final = record.get("final") or {}
    d0 = record.get("D0") or {}
    final_abs = final.get("abs_rel")
    d0_abs = d0.get("abs_rel")
    final_d1 = final.get("d1")
    pieces = []
    if epoch is not None:
        pieces.append(f"panel epoch {int(epoch):02d}")
    if final_abs is not None:
        pieces.append(f"ours abs_rel {float(final_abs):.4f}")
    if d0_abs is not None:
        pieces.append(f"D0 abs_rel {float(d0_abs):.4f}")
    if final_d1 is not None:
        pieces.append(f"ours d1 {float(final_d1):.4f}")
    return " | ".join(pieces)


def draw_header(
    canvas: Image.Image,
    y: int,
    width: int,
    height: int,
    label: str,
    record: dict[str, Any],
    title_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, y, width, y + height), fill=(15, 15, 15))
    draw.text((14, y + 7), label, font=title_font, fill=(240, 240, 240))
    draw.text((14, y + 29), metric_line(record), font=small_font, fill=(185, 185, 185))


def validate_alignment(source_payloads: list[dict[str, Any]]) -> None:
    lengths = {len(payload.get("records", [])) for payload in source_payloads}
    if len(lengths) != 1:
        raise ValueError(f"Source manifests have different record counts: {sorted(lengths)}")
    for i in range(next(iter(lengths))):
        keys = {
            (
                int(payload["records"][i]["dataset_index"]),
                str(payload["records"][i]["sample_name"]),
                str(payload["records"][i].get("variant")),
            )
            for payload in source_payloads
        }
        if len(keys) != 1:
            raise ValueError(f"Source manifests are not aligned at record {i + 1}: {sorted(keys)}")


def main() -> None:
    args = parse_args()
    sources = [parse_source(spec) for spec in args.source]
    payloads = [load_json(path) for _, _, path in sources]
    validate_alignment(payloads)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    title_font = load_font(15, bold=True)
    small_font = load_font(11)
    header_h = int(args.header_height)
    gap = int(args.gap)

    experiments = [{"key": key, "label": label} for key, label, _ in sources]
    output_records: list[dict[str, Any]] = []
    count = len(payloads[0]["records"])
    for rec_idx in range(count):
        source_records = [payload["records"][rec_idx] for payload in payloads]
        panels = [Image.open(record["panel_path"]).convert("RGB") for record in source_records]
        width = max(panel.width for panel in panels)
        height = sum(panel.height + header_h for panel in panels) + gap * (len(panels) - 1)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))

        y = 0
        for (_, label, _), record, panel in zip(sources, source_records, panels, strict=True):
            draw_header(canvas, y, width, header_h, label, record, title_font, small_font)
            y += header_h
            canvas.paste(panel, (0, y))
            y += panel.height + gap

        out_path = output_dir / sample_slug(args.split, source_records[0])
        canvas.save(out_path, quality=95)
        output_records.append(
            {
                "order": int(source_records[0]["order"]),
                "dataset_index": int(source_records[0]["dataset_index"]),
                "variant": source_records[0].get("variant"),
                "sample_name": source_records[0]["sample_name"],
                "panel_path": str(out_path),
                "source_records": source_records,
            }
        )
        print(f"[write] {out_path}")

    manifest = {
        "split": args.split,
        "output_dir": str(output_dir),
        "layout": "three experiment 3x4 panels stacked vertically with per-experiment headers",
        "selected_indices": [int(record["dataset_index"]) for record in payloads[0]["records"]],
        "experiments": experiments,
        "source_manifests": [str(path) for _, _, path in sources],
        "records": output_records,
    }
    save_json(output_dir / "manifest.json", manifest)
    print(f"[done] wrote {count} summaries to {output_dir}")


if __name__ == "__main__":
    main()
