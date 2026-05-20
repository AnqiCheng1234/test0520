from __future__ import annotations

import argparse
import json
from pathlib import Path


CHANNEL_NAMES = ("R", "Gr", "Gb", "B")
ABLATIONS = ("A0", "A1", "A2", "A3", "A4")


def parse_args():
    parser = argparse.ArgumentParser(description="Build raw-domain config files for LoD -> RobotCar ablations.")
    parser.add_argument("--stats-json", required=True, help="Output JSON from quantify_lod_robotcar_raw_gap.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lod-group", default="lod_day", choices=("lod_day", "lod_night"))
    parser.add_argument("--robotcar-day-group", default="robotcar_day")
    parser.add_argument("--robotcar-night-group", default="robotcar_night")
    parser.add_argument("--pedestal-percentile", default="p1")
    parser.add_argument("--scale-percentile", default="p99.9")
    parser.add_argument(
        "--robotcar-night-pedestal-mode",
        default="fixed_zero",
        choices=("fixed_zero", "same_percentile", "p0.1"),
        help=(
            "RobotCar-night pedestal override. fixed_zero is the default because night has many true/near-zero "
            "pixels and should not blindly inherit the day-oriented pedestal percentile."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.4545454545)
    parser.add_argument("--lut-path", default=None)
    parser.add_argument("--quantize-bits", type=int, default=8)
    parser.add_argument("--lowpass-kernel", type=int, default=3)
    return parser.parse_args()


def read_json(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def channel_percentile(summary: dict, group: str, percentile: str) -> list[float]:
    channels = summary["groups"][group]["distribution"]["channels"]
    return [float(channels[name]["percentiles"][percentile]) for name in CHANNEL_NAMES]


def pedestal_scale_config(
    summary: dict,
    group: str,
    *,
    pedestal_percentile: str,
    scale_percentile: str,
    pedestal_override: list[float] | None = None,
) -> dict:
    pedestal = pedestal_override if pedestal_override is not None else channel_percentile(summary, group, pedestal_percentile)
    high = channel_percentile(summary, group, scale_percentile)
    scale = [max(h - p, 1e-8) for p, h in zip(pedestal, high)]
    return {"pedestal": pedestal, "scale": scale}


def robotcar_night_pedestal_override(summary: dict, group: str, mode: str) -> list[float] | None:
    if mode == "same_percentile":
        return None
    if mode == "fixed_zero":
        return [0.0 for _ in CHANNEL_NAMES]
    if mode == "p0.1":
        return channel_percentile(summary, group, "p0.1")
    raise ValueError(f"Unsupported robotcar night pedestal mode: {mode}")


def group_rows_selected(summary: dict, group: str) -> int | None:
    try:
        return int(summary["groups"][group]["rows_selected"])
    except KeyError:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_path(output_dir: Path, ablation: str, split: str) -> Path:
    return output_dir / f"{ablation}_{split}_raw_domain.json"


def main():
    args = parse_args()
    summary = read_json(args.stats_json)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_lod = pedestal_scale_config(
        summary,
        args.lod_group,
        pedestal_percentile=args.pedestal_percentile,
        scale_percentile=args.scale_percentile,
    )
    base_robotcar_day = pedestal_scale_config(
        summary,
        args.robotcar_day_group,
        pedestal_percentile=args.pedestal_percentile,
        scale_percentile=args.scale_percentile,
    )
    base_robotcar_night = pedestal_scale_config(
        summary,
        args.robotcar_night_group,
        pedestal_percentile=args.pedestal_percentile,
        scale_percentile=args.scale_percentile,
        pedestal_override=robotcar_night_pedestal_override(
            summary,
            args.robotcar_night_group,
            args.robotcar_night_pedestal_mode,
        ),
    )

    matrix = {
        "A0": {
            "description": "current baseline / identity raw-domain transform",
            "lod": {},
            "robotcar_day": {},
            "robotcar_night": {},
        },
        "A1": {
            "description": (
                f"per-dataset pedestal + scale using {args.pedestal_percentile}/{args.scale_percentile}; "
                f"RobotCar-night pedestal mode={args.robotcar_night_pedestal_mode}"
            ),
            "lod": base_lod,
            "robotcar_day": base_robotcar_day,
            "robotcar_night": base_robotcar_night,
        },
        "A2": {
            "description": "A1 + LoD power-law companding toward display-referred space",
            "lod": {**base_lod, "gamma": args.gamma},
            "robotcar_day": base_robotcar_day,
            "robotcar_night": base_robotcar_night,
        },
        "A3": {
            "description": "CDF/LUT mapping for LoD; RobotCar kept identity unless an explicit eval config is desired",
            "lod": {"lut_path": args.lut_path} if args.lut_path else {},
            "robotcar_day": {},
            "robotcar_night": {},
        },
        "A4": {
            "description": (
                "A3 + RobotCar-style 8-bit quantization and light low-pass degradation on LoD; "
                "the low-pass is a lower-bound approximation, not cv2.remap's spatially varying rectification"
            ),
            "lod": {
                **({"lut_path": args.lut_path} if args.lut_path else {}),
                "quantize_bits": args.quantize_bits,
                "lowpass_kernel": args.lowpass_kernel,
            },
            "robotcar_day": {},
            "robotcar_night": {},
        },
    }

    written: dict[str, dict[str, str]] = {}
    for ablation in ABLATIONS:
        written[ablation] = {}
        for split in ("lod", "robotcar_day", "robotcar_night"):
            path = config_path(output_dir, ablation, split)
            write_json(path, matrix[ablation][split])
            written[ablation][split] = str(path)

    matrix_path = output_dir / "ablation_raw_domain_matrix.json"
    stats_rows = {
        args.lod_group: group_rows_selected(summary, args.lod_group),
        args.robotcar_day_group: group_rows_selected(summary, args.robotcar_day_group),
        args.robotcar_night_group: group_rows_selected(summary, args.robotcar_night_group),
    }
    write_json(
        matrix_path,
        {
            "stats_json": str(Path(args.stats_json).expanduser().resolve()),
            "stats_rows_selected": stats_rows,
            "robotcar_night_pedestal_mode": args.robotcar_night_pedestal_mode,
            "matrix": matrix,
            "files": written,
        },
    )

    lines = [
        "# LoD -> RobotCar raw-domain ablation config matrix",
        "",
        f"- stats_json: `{Path(args.stats_json).expanduser().resolve()}`",
        f"- lod_group: `{args.lod_group}`",
        f"- stats rows selected: `{stats_rows}`",
        f"- pedestal/scale percentiles: `{args.pedestal_percentile}` / `{args.scale_percentile}`",
        f"- RobotCar-night pedestal mode: `{args.robotcar_night_pedestal_mode}`",
        "- Hard rule: configs from partial LoD stats are for smoke/trend checks only; formal ablation configs must be rebuilt from the full-stats JSON.",
        "",
        "| Ablation | Meaning | Train flags | RobotCar day flags | RobotCar night flags |",
        "|---|---|---|---|---|",
    ]
    for ablation in ABLATIONS:
        files = written[ablation]
        lines.append(
            "| {ablation} | {desc} | `--lod-raw-domain-config {lod}` | "
            "`--robotcar-raw-domain-config {day}` | `--robotcar-night-raw-domain-config {night}` |".format(
                ablation=ablation,
                desc=matrix[ablation]["description"],
                lod=files["lod"],
                day=files["robotcar_day"],
                night=files["robotcar_night"],
            )
        )
    md_path = output_dir / "ablation_raw_domain_matrix.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {matrix_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
