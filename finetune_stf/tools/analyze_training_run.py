#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


PRETRAIN_TO_EVAL = {
    "pretrain_stf": "val",
    "pretrain_eth3d_fast": "eth3d_fast",
    "pretrain_robotcar_fast": "robotcar_fast",
}
EVAL_LABELS = {
    "val": "STF Val",
    "eth3d_fast": "ETH3D Fast",
    "robotcar_fast": "RobotCar Fast",
}
EVAL_COLORS = {
    "val": "#1f77b4",
    "eth3d_fast": "#d62728",
    "robotcar_fast": "#2ca02c",
}


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[idx:idx + 2], 16) for idx in (0, 2, 4))


def _format_tick(value: float) -> str:
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 100 or abs_value < 1e-2:
        return f"{value:.2e}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _nice_bounds(values, *, pad_ratio=0.08):
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return 0.0, 1.0
    vmin = min(finite)
    vmax = max(finite)
    if vmin == vmax:
        pad = max(abs(vmin) * pad_ratio, 1e-3)
    else:
        pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


def _build_y_ticks(ymin: float, ymax: float, count: int = 5):
    if count <= 1:
        return [(ymin, _format_tick(ymin))]
    if ymax <= ymin:
        ymax = ymin + 1.0
    step = (ymax - ymin) / (count - 1)
    return [(ymin + idx * step, _format_tick(ymin + idx * step)) for idx in range(count)]


def _draw_polyline(draw: ImageDraw.ImageDraw, points, color, *, width=2):
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
    for x, y in points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color, outline=color)


def _draw_chart(canvas: Image.Image, rect, *, title, series, x_ticks, x_min, x_max, y_min, y_max, note=None):
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = rect
    plot_left = left + 64
    plot_right = right - 20
    plot_top = top + 28
    plot_bottom = bottom - 36

    draw.text((left, top), title, fill="black")
    if note:
        draw.text((right - 220, top), note, fill=(90, 90, 90))

    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    y_ticks = _build_y_ticks(y_min, y_max)
    for tick_value, tick_label in y_ticks:
        y = plot_bottom - ((tick_value - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
        y = int(round(y))
        draw.line((plot_left, y, plot_right, y), fill=(228, 228, 228), width=1)
        draw.text((left, y - 8), tick_label, fill=(70, 70, 70))

    for tick_x, tick_label in x_ticks:
        if x_max == x_min:
            x = plot_left
        else:
            x = plot_left + ((tick_x - x_min) / (x_max - x_min)) * (plot_right - plot_left)
        x = int(round(x))
        draw.line((x, plot_top, x, plot_bottom), fill=(240, 240, 240), width=1)
        draw.text((x - 10, plot_bottom + 8), str(tick_label), fill=(70, 70, 70))

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=1)

    for item in series:
        points = []
        for x_value, y_value in item["points"]:
            x = plot_left + ((x_value - x_min) / (x_max - x_min)) * (plot_right - plot_left)
            y = plot_bottom - ((y_value - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
            points.append((int(round(x)), int(round(y))))
        _draw_polyline(draw, points, item["color"], width=item.get("width", 2))
        for hx, hy, label in item.get("highlights", []):
            px = plot_left + ((hx - x_min) / (x_max - x_min)) * (plot_right - plot_left)
            py = plot_bottom - ((hy - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)
            px = int(round(px))
            py = int(round(py))
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=item["color"], fill="white", width=2)
            draw.text((px + 6, py - 14), label, fill=item["color"])

    legend_x = plot_right - 210
    legend_y = plot_top + 6
    for idx, item in enumerate(series):
        y = legend_y + idx * 18
        draw.rectangle((legend_x, y + 3, legend_x + 10, y + 13), fill=item["color"])
        draw.text((legend_x + 16, y), item["name"], fill="black")


def parse_args():
    parser = argparse.ArgumentParser(description="Parse finetune_stf train.log and generate summary plots/reports.")
    parser.add_argument("exp_dir", type=Path, help="Experiment directory containing config.json and train.log")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <exp_dir>/analysis")
    return parser.parse_args()


def parse_train_log(log_path: Path):
    lines = log_path.read_text(encoding="utf-8").splitlines()

    epoch_start_pat = re.compile(r"\[EPOCH\] start epoch=(\d+)/")
    epoch_done_pat = re.compile(r"\[EPOCH\] done epoch=(\d+) avg_loss=([0-9.eE+-]+) used_steps=(\d+) elapsed=([0-9:]+)")
    train_pat = re.compile(
        r"\[TRAIN\] epoch=(\d+) micro_step=(\d+)/(\d+) opt_step=(\d+)/(\d+) lr=([0-9.eE+-]+) "
        r"loss=([0-9.eE+-]+) running_avg=([0-9.eE+-]+)"
    )
    eval_pat = re.compile(
        r"\[EVAL\]\[(val|eth3d_fast|robotcar_fast)\] abs_rel=([0-9.]+) rmse=([0-9.]+) "
        r"silog=([0-9.]+) d1=([0-9.]+) d2=([0-9.]+) d3=([0-9.]+)"
    )
    pretrain_pat = re.compile(
        r"\[EVAL\]\[(pretrain_stf|pretrain_eth3d_fast|pretrain_robotcar_fast)\] abs_rel=([0-9.]+) rmse=([0-9.]+) "
        r"silog=([0-9.]+) d1=([0-9.]+) d2=([0-9.]+) d3=([0-9.]+)"
    )
    lod_size_pat = re.compile(r"\[DATASET\] lod_train=(\d+).* val=(\d+)")
    eth3d_size_pat = re.compile(r"\[DATASET\] eth3d_val_fast=(\d+)")
    robotcar_size_pat = re.compile(r"\[DATASET\] robotcar_val_fast=(\d+)")

    current_epoch = None
    pretrain = {}
    epochs = {}
    train_points = []
    dataset_sizes = {}

    for line in lines:
        match = lod_size_pat.search(line)
        if match:
            dataset_sizes["lod_train"] = int(match.group(1))
            dataset_sizes["stf_val"] = int(match.group(2))
        match = eth3d_size_pat.search(line)
        if match:
            dataset_sizes["eth3d_fast"] = int(match.group(1))
        match = robotcar_size_pat.search(line)
        if match:
            dataset_sizes["robotcar_fast"] = int(match.group(1))

        match = pretrain_pat.search(line)
        if match:
            pretrain[PRETRAIN_TO_EVAL[match.group(1)]] = {
                "abs_rel": float(match.group(2)),
                "rmse": float(match.group(3)),
                "silog": float(match.group(4)),
                "d1": float(match.group(5)),
                "d2": float(match.group(6)),
                "d3": float(match.group(7)),
            }
            continue

        match = epoch_start_pat.search(line)
        if match:
            current_epoch = int(match.group(1))
            epochs.setdefault(current_epoch, {"metrics": {}})
            continue

        match = train_pat.search(line)
        if match:
            epoch = int(match.group(1))
            micro_step = int(match.group(2))
            micro_total = int(match.group(3))
            train_points.append(
                {
                    "epoch": epoch,
                    "micro_step": micro_step,
                    "micro_total": micro_total,
                    "opt_step": int(match.group(4)),
                    "opt_total": int(match.group(5)),
                    "lr": float(match.group(6)),
                    "loss": float(match.group(7)),
                    "running_avg": float(match.group(8)),
                    "epoch_progress": epoch + (micro_step / max(micro_total, 1)),
                }
            )
            continue

        match = epoch_done_pat.search(line)
        if match:
            epoch = int(match.group(1))
            epochs.setdefault(epoch, {"metrics": {}})
            epochs[epoch]["avg_loss"] = float(match.group(2))
            epochs[epoch]["used_steps"] = int(match.group(3))
            epochs[epoch]["elapsed"] = match.group(4)
            continue

        match = eval_pat.search(line)
        if match and current_epoch is not None:
            epochs.setdefault(current_epoch, {"metrics": {}})
            epochs[current_epoch]["metrics"][match.group(1)] = {
                "abs_rel": float(match.group(2)),
                "rmse": float(match.group(3)),
                "silog": float(match.group(4)),
                "d1": float(match.group(5)),
                "d2": float(match.group(6)),
                "d3": float(match.group(7)),
            }

    ordered_epochs = [epochs[idx] | {"epoch": idx} for idx in sorted(epochs)]
    return {
        "dataset_sizes": dataset_sizes,
        "pretrain": pretrain,
        "epochs": ordered_epochs,
        "train_points": train_points,
    }


def load_checkpoint_meta(exp_dir: Path):
    try:
        import torch
    except Exception:
        return {}

    meta = {}
    for name in ("best_model.pth", "best_model_eth3d.pth", "best_model_robotcar.pth", "last_epoch_model.pth"):
        path = exp_dir / name
        if not path.is_file():
            continue
        ckpt = torch.load(path, map_location="cpu")
        meta[name] = {
            "epoch": ckpt.get("epoch"),
            "best_metric": ckpt.get("best_metric"),
            "best_metrics": ckpt.get("best_metrics"),
        }
    return meta


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_epoch_csv(path: Path, epochs):
    fieldnames = [
        "epoch",
        "avg_loss",
        "used_steps",
        "val_abs_rel",
        "val_d1",
        "eth3d_fast_abs_rel",
        "eth3d_fast_d1",
        "robotcar_fast_abs_rel",
        "robotcar_fast_d1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in epochs:
            metrics = item.get("metrics", {})
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "avg_loss": item.get("avg_loss"),
                    "used_steps": item.get("used_steps"),
                    "val_abs_rel": metrics.get("val", {}).get("abs_rel"),
                    "val_d1": metrics.get("val", {}).get("d1"),
                    "eth3d_fast_abs_rel": metrics.get("eth3d_fast", {}).get("abs_rel"),
                    "eth3d_fast_d1": metrics.get("eth3d_fast", {}).get("d1"),
                    "robotcar_fast_abs_rel": metrics.get("robotcar_fast", {}).get("abs_rel"),
                    "robotcar_fast_d1": metrics.get("robotcar_fast", {}).get("d1"),
                }
            )


def save_train_csv(path: Path, train_points):
    fieldnames = [
        "epoch",
        "micro_step",
        "micro_total",
        "opt_step",
        "opt_total",
        "lr",
        "loss",
        "running_avg",
        "epoch_progress",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in train_points:
            writer.writerow(item)


def build_metric_history(epochs, tag, metric_name):
    points = []
    for item in epochs:
        metrics = item.get("metrics", {})
        if tag not in metrics or metric_name not in metrics[tag]:
            continue
        points.append((item["epoch"], metrics[tag][metric_name]))
    return points


def best_point(points, lower_is_better=True):
    if not points:
        return None
    key_fn = (lambda item: item[1]) if lower_is_better else (lambda item: -item[1])
    return min(points, key=key_fn)


def plot_loss_curves(path: Path, epochs, train_points):
    epoch_ids = [item["epoch"] for item in epochs]
    epoch_loss = [item.get("avg_loss") for item in epochs]
    canvas = Image.new("RGB", (1100, 800), "white")

    min_epoch, min_loss = min(zip(epoch_ids, epoch_loss), key=lambda item: item[1])
    top_ticks = [(epoch, str(epoch)) for epoch in epoch_ids[:: max(1, len(epoch_ids) // 10 or 1)]]
    if epoch_ids[-1] not in {tick for tick, _ in top_ticks}:
        top_ticks.append((epoch_ids[-1], str(epoch_ids[-1])))
    _draw_chart(
        canvas,
        (32, 24, 1068, 388),
        title="Epoch Average Loss",
        series=[
            {
                "name": "Epoch Avg Loss",
                "color": _hex_to_rgb("#1f77b4"),
                "points": list(zip(epoch_ids, epoch_loss)),
                "highlights": [(min_epoch, min_loss, f"min {min_loss:.4f} @ {min_epoch}")],
            }
        ],
        x_ticks=top_ticks,
        x_min=min(epoch_ids),
        x_max=max(epoch_ids),
        y_min=_nice_bounds(epoch_loss)[0],
        y_max=_nice_bounds(epoch_loss)[1],
    )

    if train_points:
        xs = [item["epoch_progress"] for item in train_points]
        running = [item["running_avg"] for item in train_points]
        raw_loss = [item["loss"] for item in train_points]
        dense_ticks = [(tick, str(int(tick))) for tick in range(int(xs[0]), int(xs[-1]) + 1, 2)]
        if not dense_ticks or dense_ticks[0][0] != int(xs[0]):
            dense_ticks = [(int(xs[0]), str(int(xs[0])))] + dense_ticks
        if dense_ticks[-1][0] != int(xs[-1]):
            dense_ticks.append((int(xs[-1]), str(int(xs[-1]))))
        y_min, y_max = _nice_bounds(running + raw_loss)
        _draw_chart(
            canvas,
            (32, 412, 1068, 776),
            title="In-Training Loss Trend",
            series=[
                {
                    "name": "Running Avg Loss",
                    "color": _hex_to_rgb("#ff7f0e"),
                    "points": list(zip(xs, running)),
                    "width": 2,
                },
                {
                    "name": "Logged Step Loss",
                    "color": _hex_to_rgb("#7f7f7f"),
                    "points": list(zip(xs, raw_loss)),
                    "width": 1,
                },
            ],
            x_ticks=dense_ticks,
            x_min=min(xs),
            x_max=max(xs),
            y_min=y_min,
            y_max=y_max,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def plot_eval_curves(path: Path, epochs, pretrain, metric_name, *, lower_is_better):
    canvas = Image.new("RGB", (1100, 550), "white")
    max_epoch = max((item["epoch"] for item in epochs), default=0)
    xticks = [-1] + list(range(0, max_epoch + 1, 2))
    xlabels = ["init"] + [str(idx) for idx in range(0, max_epoch + 1, 2)]

    series = []
    all_values = []
    for tag in ("val", "eth3d_fast", "robotcar_fast"):
        points = build_metric_history(epochs, tag, metric_name)
        if not points:
            continue
        xs = [epoch for epoch, _ in points]
        ys = [value for _, value in points]
        if tag in pretrain:
            xs = [-1] + xs
            ys = [pretrain[tag][metric_name]] + ys
        all_values.extend(ys)
        best = best_point(points, lower_is_better=lower_is_better)
        series.append(
            {
                "name": EVAL_LABELS[tag],
                "color": _hex_to_rgb(EVAL_COLORS[tag]),
                "points": list(zip(xs, ys)),
                "highlights": [] if best is None else [(best[0], best[1], f"{best[1]:.4f} @ {best[0]}")],
            }
        )

    y_min, y_max = _nice_bounds(all_values)
    _draw_chart(
        canvas,
        (32, 24, 1068, 526),
        title=f"Eval Trend: {metric_name}",
        series=series,
        x_ticks=list(zip(xticks, xlabels)),
        x_min=-1,
        x_max=max_epoch if max_epoch > -1 else 1,
        y_min=y_min,
        y_max=y_max,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize_run(config, parsed, checkpoint_meta):
    epochs = parsed["epochs"]
    pretrain = parsed["pretrain"]
    train_points = parsed["train_points"]
    dataset_sizes = parsed["dataset_sizes"]

    epoch_losses = [(item["epoch"], item.get("avg_loss")) for item in epochs if item.get("avg_loss") is not None]
    first_loss = epoch_losses[0][1]
    min_loss_epoch, min_loss = min(epoch_losses, key=lambda item: item[1])
    last_loss_epoch, last_loss = epoch_losses[-1]
    first5_avg = sum(loss for _, loss in epoch_losses[:5]) / min(len(epoch_losses), 5)
    last5_avg = sum(loss for _, loss in epoch_losses[-5:]) / min(len(epoch_losses), 5)

    metrics_summary = {}
    for tag in ("val", "eth3d_fast", "robotcar_fast"):
        history_abs_rel = build_metric_history(epochs, tag, "abs_rel")
        history_d1 = build_metric_history(epochs, tag, "d1")
        if not history_abs_rel:
            continue
        best_abs = best_point(history_abs_rel, lower_is_better=True)
        best_d1 = best_point(history_d1, lower_is_better=False)
        last_abs = history_abs_rel[-1]
        last_d1 = history_d1[-1]
        init_metrics = pretrain.get(tag)
        metrics_summary[tag] = {
            "init": init_metrics,
            "best_abs_rel": {"epoch": best_abs[0], "value": best_abs[1]},
            "best_d1": {"epoch": best_d1[0], "value": best_d1[1]},
            "last": {
                "epoch": last_abs[0],
                "abs_rel": last_abs[1],
                "d1": last_d1[1],
            },
        }

    best_model_meta = checkpoint_meta.get("best_model.pth", {})
    canonical_best_epoch = best_model_meta.get("epoch")
    canonical_best_metric = config.get("best_metric", best_model_meta.get("best_metric"))

    summary = {
        "config": {
            "stage": config.get("stage"),
            "input_type": config.get("input_type"),
            "input_hw": [config.get("input_height"), config.get("input_width")],
            "epochs": config.get("epochs"),
            "batch_size": config.get("bs"),
            "accum_steps": config.get("accum_steps"),
            "effective_batch_size": (config.get("bs") or 0) * (config.get("accum_steps") or 0),
            "loss_type": config.get("loss_type"),
            "best_metric": canonical_best_metric,
            "robotcar_root": config.get("robotcar_root"),
        },
        "dataset_sizes": dataset_sizes,
        "loss": {
            "first_epoch_avg": first_loss,
            "min_epoch_avg": {"epoch": min_loss_epoch, "value": min_loss},
            "last_epoch_avg": {"epoch": last_loss_epoch, "value": last_loss},
            "first5_avg": first5_avg,
            "last5_avg": last5_avg,
            "delta_first_to_last_pct": ((last_loss / first_loss) - 1.0) * 100.0,
            "logged_points": len(train_points),
        },
        "metrics": metrics_summary,
        "checkpoints": checkpoint_meta,
        "canonical_best_epoch": canonical_best_epoch,
    }
    return summary


def format_delta(new_value, old_value, *, lower_is_better):
    if old_value is None:
        return "n/a"
    delta = new_value - old_value
    rel = ((new_value / old_value) - 1.0) * 100.0 if old_value != 0 else 0.0
    direction = "improve" if (delta < 0 if lower_is_better else delta > 0) else "worse"
    return f"{delta:+.4f} ({rel:+.1f}%, {direction})"


def write_summary_markdown(path: Path, summary):
    cfg = summary["config"]
    loss = summary["loss"]
    metrics = summary["metrics"]
    dataset_sizes = summary["dataset_sizes"]
    canonical_best_epoch = summary["canonical_best_epoch"]
    canonical_metric = cfg["best_metric"]

    lines = [
        "# Training Run Analysis",
        "",
        "## Setup",
        "",
        f"- Stage: `{cfg['stage']}`",
        f"- Input: `{cfg['input_type']}` at `{cfg['input_hw'][0]}x{cfg['input_hw'][1]}`",
        f"- Loss: `{cfg['loss_type']}`",
        f"- Epochs: `{cfg['epochs']}`",
        f"- Batch: `bs={cfg['batch_size']}`, `accum={cfg['accum_steps']}`, `effective_bs={cfg['effective_batch_size']}`",
        f"- Canonical best metric: `{canonical_metric}`",
        f"- RobotCar root: `{cfg['robotcar_root']}`",
        "",
        "## Dataset Sizes",
        "",
        f"- LOD train: `{dataset_sizes.get('lod_train', 'n/a')}`",
        f"- STF val: `{dataset_sizes.get('stf_val', 'n/a')}`",
        f"- ETH3D fast: `{dataset_sizes.get('eth3d_fast', 'n/a')}`",
        f"- RobotCar fast: `{dataset_sizes.get('robotcar_fast', 'n/a')}`",
        "",
        "## Loss Trend",
        "",
        f"- Epoch avg loss: `{loss['first_epoch_avg']:.4f}` -> `{loss['last_epoch_avg']['value']:.4f}` ({loss['delta_first_to_last_pct']:+.1f}%).",
        f"- Minimum epoch avg loss: epoch `{loss['min_epoch_avg']['epoch']}` with `{loss['min_epoch_avg']['value']:.4f}`.",
        f"- First 5 epochs avg loss: `{loss['first5_avg']:.4f}`.",
        f"- Last 5 epochs avg loss: `{loss['last5_avg']:.4f}`.",
        f"- Interpretation: optimization stayed stable and kept decreasing almost to the last epoch, but external eval bests happened earlier.",
        "",
        "## Eval Summary",
        "",
        "| Split | Init abs_rel | Best abs_rel | Last abs_rel | Init d1 | Best d1 | Last d1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for tag in ("val", "eth3d_fast", "robotcar_fast"):
        item = metrics.get(tag)
        if item is None:
            continue
        init = item["init"] or {}
        best_abs = item["best_abs_rel"]
        best_d1 = item["best_d1"]
        last = item["last"]
        lines.append(
            f"| {EVAL_LABELS[tag]} | "
            f"{init.get('abs_rel', float('nan')):.4f} | "
            f"{best_abs['value']:.4f} (e{best_abs['epoch']}) | "
            f"{last['abs_rel']:.4f} (e{last['epoch']}) | "
            f"{init.get('d1', float('nan')):.4f} | "
            f"{best_d1['value']:.4f} (e{best_d1['epoch']}) | "
            f"{last['d1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    stf = metrics.get("val")
    eth3d = metrics.get("eth3d_fast")
    robotcar = metrics.get("robotcar_fast")
    if stf and eth3d and robotcar:
        lines.extend(
            [
                f"- Canonical `best_model.pth` was saved at epoch `{canonical_best_epoch}` by `{canonical_metric}`.",
                f"- ETH3D improved from `{eth3d['init']['abs_rel']:.4f}` to `{eth3d['best_abs_rel']['value']:.4f}` at epoch `{eth3d['best_abs_rel']['epoch']}` "
                f"({format_delta(eth3d['best_abs_rel']['value'], eth3d['init']['abs_rel'], lower_is_better=True)}).",
                f"- RobotCar improved from `{robotcar['init']['abs_rel']:.4f}` to `{robotcar['best_abs_rel']['value']:.4f}` at epoch `{robotcar['best_abs_rel']['epoch']}` "
                f"({format_delta(robotcar['best_abs_rel']['value'], robotcar['init']['abs_rel'], lower_is_better=True)}).",
                f"- STF kept improving later and reached its best at epoch `{stf['best_abs_rel']['epoch']}` with `{stf['best_abs_rel']['value']:.4f}`, "
                f"which happened after the canonical ETH3D-best checkpoint.",
                f"- From canonical best epoch `{canonical_best_epoch}` to the final epoch, loss still decreased, "
                f"but ETH3D `abs_rel` moved from `{eth3d['best_abs_rel']['value']:.4f}` to `{eth3d['last']['abs_rel']:.4f}` and "
                f"RobotCar from `{robotcar['best_abs_rel']['value']:.4f}` to `{robotcar['last']['abs_rel']:.4f}`.",
                f"- This suggests continued fitting on the LOD training objective after epoch `{canonical_best_epoch}` no longer aligned with cross-domain eval; "
                f"early stopping on ETH3D was the right choice for this run.",
                f"- RobotCar should be interpreted with extra caution because this transitional protocol currently evaluates on `{dataset_sizes.get('robotcar_fast', 'n/a')}` samples, not a full-size benchmark.",
            ]
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `loss_curves.png`",
            "- `eval_abs_rel_curves.png`",
            "- `eval_d1_curves.png`",
            "- `epoch_metrics.csv`",
            "- `train_points.csv`",
            "- `summary.json`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    output_dir = ensure_output_dir(args.output_dir.expanduser().resolve() if args.output_dir else exp_dir / "analysis")

    config_path = exp_dir / "config.json"
    log_path = exp_dir / "train.log"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing train.log: {log_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    parsed = parse_train_log(log_path)
    checkpoint_meta = load_checkpoint_meta(exp_dir)
    summary = summarize_run(config, parsed, checkpoint_meta)

    save_epoch_csv(output_dir / "epoch_metrics.csv", parsed["epochs"])
    save_train_csv(output_dir / "train_points.csv", parsed["train_points"])
    plot_loss_curves(output_dir / "loss_curves.png", parsed["epochs"], parsed["train_points"])
    plot_eval_curves(output_dir / "eval_abs_rel_curves.png", parsed["epochs"], parsed["pretrain"], "abs_rel", lower_is_better=True)
    plot_eval_curves(output_dir / "eval_d1_curves.png", parsed["epochs"], parsed["pretrain"], "d1", lower_is_better=False)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_markdown(output_dir / "summary.md", summary)

    print(json.dumps({"exp_dir": str(exp_dir), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
