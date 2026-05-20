#!/usr/bin/env python3
"""Build multi-split analysis artifacts for finetune_stf experiments.

The older shared analyzer only covered STF/ETH3D/RobotCar-day. Recent
vkitti_lod runs evaluate these splits instead:

  - KITTI val through the RGB checkpoint-decoder path
  - NYUv2 val through the RGB checkpoint-decoder path
  - ETH3D fast
  - RobotCar day fast
  - RobotCar night fast

This script parses train.log and writes the same full-analysis artifacts used
by the 0426/0427 experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("kitti", "nyu", "eth3d_fast", "robotcar_fast", "robotcar_night_fast")
SPLIT_LABELS = {
    "kitti": "KITTI val (RGB checkpoint-decoder)",
    "nyu": "NYUv2 val (RGB checkpoint-decoder)",
    "eth3d_fast": "ETH3D Fast",
    "robotcar_fast": "RobotCar Day Fast",
    "robotcar_night_fast": "RobotCar Night Fast",
}
SPLIT_COLORS = {
    "kitti": "#1f77b4",
    "nyu": "#ff7f0e",
    "eth3d_fast": "#d62728",
    "robotcar_fast": "#2ca02c",
    "robotcar_night_fast": "#9467bd",
}

# Offline DAv2 RGB zero-shot baselines used in the existing full analysis.
# Splits without an entry, such as NYUv2, use the pretrain eval parsed from train.log.
DAV2_RGB_ZS = {
    "kitti": {"abs_rel": 0.0680, "d1": 0.9512},
    "eth3d_fast": {"abs_rel": 0.0524, "d1": 0.9760},
    "robotcar_fast": {"abs_rel": 0.2882, "d1": 0.4097},
    "robotcar_night_fast": {"abs_rel": 0.2526, "d1": 0.4844},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-split analysis plots and summaries.")
    parser.add_argument("exp_dir", type=Path, help="Experiment directory containing config.json and train.log.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <exp_dir>/analysis.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/mnt/drive/3333_raw/0000_exp_ckpt"),
        help="Root where checkpoint/event files were moved. Stored in the summary for traceability.",
    )
    return parser.parse_args()


def _metric_dict(groups: tuple[str, ...], offset: int = 0) -> dict[str, float]:
    return {
        "abs_rel": float(groups[offset + 0]),
        "rmse": float(groups[offset + 1]),
        "silog": float(groups[offset + 2]),
        "d1": float(groups[offset + 3]),
        "d2": float(groups[offset + 4]) if len(groups) > offset + 4 and groups[offset + 4] is not None else None,
        "d3": float(groups[offset + 5]) if len(groups) > offset + 5 and groups[offset + 5] is not None else None,
    }


def parse_train_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    rename = {
        "kitti_rgb_checkpoint_decoder": "kitti",
        "kitti_val_rgb_checkpoint_decoder": "kitti",
        "nyu_rgb_checkpoint_decoder": "nyu",
        "eth3d_fast": "eth3d_fast",
        "robotcar_fast": "robotcar_fast",
        "robotcar_night_fast": "robotcar_night_fast",
    }

    epoch_start_pat = re.compile(r"\[EPOCH\] start epoch=(\d+)/")
    epoch_done_pat = re.compile(
        r"\[EPOCH\] done epoch=(\d+) avg_loss=([0-9.eE+-]+) used_steps=(\d+) elapsed=([0-9:]+)"
    )
    compact_eval_pat = re.compile(
        r"\[EVAL\]\[(kitti_val_rgb_checkpoint_decoder|kitti_rgb_checkpoint_decoder|nyu_rgb_checkpoint_decoder|eth3d_fast|robotcar_fast|robotcar_night_fast)\] "
        r"abs_rel=([0-9.]+) rmse=([0-9.]+) silog=([0-9.]+) "
        r"d1=([0-9.]+) d2=([0-9.]+) d3=([0-9.]+)"
    )
    compact_pretrain_pat = re.compile(
        r"\[EVAL\]\[pretrain_(kitti_rgb_checkpoint_decoder|nyu_rgb_checkpoint_decoder|eth3d_fast|robotcar_fast|robotcar_night_fast)\] "
        r"abs_rel=([0-9.]+) rmse=([0-9.]+) silog=([0-9.]+) "
        r"d1=([0-9.]+) d2=([0-9.]+) d3=([0-9.]+)"
    )
    done_eval_pat = re.compile(
        r"\[EVAL\]\[(?:pretrain_)?(kitti_rgb_checkpoint_decoder|nyu_rgb_checkpoint_decoder|eth3d_fast|robotcar_fast|robotcar_night_fast)\] "
        r"done epoch=(init|\d+) samples=(\d+) abs_rel=([0-9.]+) rmse=([0-9.]+) "
        r"silog=([0-9.]+) d1=([0-9.]+) elapsed=([0-9:]+)"
    )
    train_pat = re.compile(
        r"\[TRAIN\] epoch=(\d+) micro_step=(\d+)/(\d+) opt_step=(\d+)/(\d+) lr=([0-9.eE+-]+) "
        r"loss=([0-9.eE+-]+) running_avg=([0-9.eE+-]+)"
    )
    dataset_pat = re.compile(r"\[DATASET\] (kitti_val|nyu_val|eth3d_val_fast|robotcar_val_fast|robotcar_night_val_fast)=(\d+)")

    epochs: dict[int, dict] = {}
    pretrain: dict[str, dict] = {}
    dataset_sizes: dict[str, int] = {}
    train_points: list[dict] = []
    current_epoch = -1

    for line in text.splitlines():
        match = dataset_pat.search(line)
        if match:
            dataset_sizes[match.group(1)] = int(match.group(2))

        match = compact_pretrain_pat.search(line)
        if match:
            split = rename[match.group(1)]
            pretrain[split] = _metric_dict(match.groups(), offset=1)
            pretrain[split]["_source"] = "train.log compact pretrain line"
            continue

        match = done_eval_pat.search(line)
        if match:
            split = rename[match.group(1)]
            epoch_text = match.group(2)
            metrics = {
                "abs_rel": float(match.group(4)),
                "rmse": float(match.group(5)),
                "silog": float(match.group(6)),
                "d1": float(match.group(7)),
                "samples": int(match.group(3)),
            }
            if epoch_text == "init":
                pretrain.setdefault(split, {}).update(metrics)
                pretrain[split]["_source"] = "train.log pretrain done line"
            else:
                epoch = int(epoch_text)
                epochs.setdefault(epoch, {"metrics": {}})
                epochs[epoch]["metrics"].setdefault(split, metrics)
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

        match = compact_eval_pat.search(line)
        if match:
            if current_epoch < 0:
                continue
            split = rename[match.group(1)]
            epochs.setdefault(current_epoch, {"metrics": {}})
            epochs[current_epoch]["metrics"][split] = _metric_dict(match.groups(), offset=1)

    ordered_epochs = []
    for epoch in sorted(epochs):
        item = epochs[epoch]
        item["epoch"] = epoch
        ordered_epochs.append(item)

    return {
        "dataset_sizes": dataset_sizes,
        "pretrain": pretrain,
        "epochs": ordered_epochs,
        "train_points": train_points,
    }


def init_metric(parsed: dict, split: str, metric: str) -> float | None:
    fixed = DAV2_RGB_ZS.get(split, {}).get(metric)
    if fixed is not None:
        return fixed
    return parsed["pretrain"].get(split, {}).get(metric)


def save_epoch_csv(path: Path, epochs: list[dict]) -> None:
    fields = ["epoch", "avg_loss", "used_steps"]
    for split in SPLITS:
        for metric in ("abs_rel", "rmse", "silog", "d1", "d2", "d3"):
            fields.append(f"{split}_{metric}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in epochs:
            row = {
                "epoch": item["epoch"],
                "avg_loss": item.get("avg_loss"),
                "used_steps": item.get("used_steps"),
            }
            for split in SPLITS:
                metrics = item.get("metrics", {}).get(split, {})
                for metric in ("abs_rel", "rmse", "silog", "d1", "d2", "d3"):
                    row[f"{split}_{metric}"] = metrics.get(metric)
            writer.writerow(row)


def save_train_csv(path: Path, train_points: list[dict]) -> None:
    fields = ["epoch", "micro_step", "micro_total", "opt_step", "opt_total", "lr", "loss", "running_avg", "epoch_progress"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_points)


def best_per_split(parsed: dict) -> dict:
    out = {}
    for split in SPLITS:
        best_abs = best_d1 = None
        best_abs_epoch = best_d1_epoch = None
        last_metrics = {}
        for item in parsed["epochs"]:
            metrics = item.get("metrics", {}).get(split)
            if not metrics:
                continue
            last_metrics = metrics
            if metrics.get("abs_rel") is not None and (best_abs is None or metrics["abs_rel"] < best_abs):
                best_abs = metrics["abs_rel"]
                best_abs_epoch = item["epoch"]
            if metrics.get("d1") is not None and (best_d1 is None or metrics["d1"] > best_d1):
                best_d1 = metrics["d1"]
                best_d1_epoch = item["epoch"]
        out[split] = {
            "best_abs_rel": best_abs,
            "best_abs_rel_epoch": best_abs_epoch,
            "best_d1": best_d1,
            "best_d1_epoch": best_d1_epoch,
            "last_abs_rel": last_metrics.get("abs_rel"),
            "last_d1": last_metrics.get("d1"),
            "log_pretrain_abs_rel": parsed["pretrain"].get(split, {}).get("abs_rel"),
            "log_pretrain_d1": parsed["pretrain"].get(split, {}).get("d1"),
            "dav2_rgb_zs_abs_rel": init_metric(parsed, split, "abs_rel"),
            "dav2_rgb_zs_d1": init_metric(parsed, split, "d1"),
        }
    return out


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def plot_metric(parsed: dict, *, metric: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
    for split in SPLITS:
        xs = []
        ys = []
        init_value = init_metric(parsed, split, metric)
        if init_value is not None:
            xs.append(-1)
            ys.append(init_value)
        for item in parsed["epochs"]:
            metrics = item.get("metrics", {}).get(split)
            if not metrics or metrics.get(metric) is None:
                continue
            xs.append(item["epoch"])
            ys.append(metrics[metric])
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=SPLIT_COLORS[split],
            label=SPLIT_LABELS[split],
        )
        if init_value is not None:
            ax.annotate("init", (xs[0], ys[0]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=SPLIT_COLORS[split])
    ax.set_xlabel("epoch (-1 = eval init)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss(parsed: dict, path: Path, loss_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
    xs = [item["epoch"] for item in parsed["epochs"] if "avg_loss" in item]
    ys = [item["avg_loss"] for item in parsed["epochs"] if "avg_loss" in item]
    ax.plot(xs, ys, marker="o", linewidth=1.5, color="#444")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"avg train loss ({loss_label})")
    ax.set_title("Training loss per epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_summary(config: dict, parsed: dict, exp_dir: Path, checkpoint_root: Path) -> dict:
    cfg = {
        "stage": config.get("stage"),
        "input_type": config.get("input_type"),
        "dav2_train_mode": config.get("dav2_train_mode"),
        "input_hw": [config.get("input_height"), config.get("input_width")],
        "epochs": config.get("epochs"),
        "batch_size": config.get("bs"),
        "accum_steps": config.get("accum_steps"),
        "effective_batch_size": (config.get("bs") or 0) * (config.get("accum_steps") or 0),
        "loss_type": config.get("loss_type"),
        "loss_lambda_grad": config.get("loss_lambda_grad"),
        "loss_grad_scales": config.get("loss_grad_scales"),
        "loss_mask_downsample": config.get("loss_mask_downsample"),
        "loss_target_normalization": config.get("loss_target_normalization"),
        "best_metric": config.get("best_metric"),
        "bridge_feature_keys": config.get("bridge_feature_keys"),
        "bridge_layers": config.get("bridge_layers"),
        "lora_rank": config.get("lora_rank"),
        "lora_alpha": config.get("lora_alpha"),
    }
    return {
        "exp_dir": str(exp_dir),
        "checkpoint_dir": str((checkpoint_root / exp_dir.name).resolve()),
        "config": cfg,
        "dataset_sizes": parsed["dataset_sizes"],
        "log_pretrain_init": parsed["pretrain"],
        "dav2_rgb_zero_shot": DAV2_RGB_ZS,
        "eval_init_baseline": {
            split: {"abs_rel": init_metric(parsed, split, "abs_rel"), "d1": init_metric(parsed, split, "d1")}
            for split in SPLITS
        },
        "best_per_split": best_per_split(parsed),
        "num_epochs_parsed": len(parsed["epochs"]),
        "num_train_points": len(parsed["train_points"]),
    }


def write_summary_md(path: Path, exp_name: str, summary: dict, parsed: dict) -> None:
    cfg = summary["config"]
    loss_label = cfg["loss_type"]
    if cfg.get("loss_lambda_grad") is not None:
        loss_label = f"{loss_label}+lambda_grad={cfg['loss_lambda_grad']}"

    lines = [
        f"# Training Run Analysis - `{exp_name}`",
        "",
        "## Setup",
        "",
        f"- Stage: `{cfg['stage']}`",
        f"- Input: `{cfg['input_type']}` at `{cfg['input_hw'][0]}x{cfg['input_hw'][1]}`",
        f"- DAv2 train mode: `{cfg['dav2_train_mode']}`",
        f"- Loss: `{loss_label}`",
        f"- Epochs parsed: `{summary['num_epochs_parsed']}`",
        f"- Batch: `bs={cfg['batch_size']}`, `accum={cfg['accum_steps']}`, `effective_bs={cfg['effective_batch_size']}`",
        f"- Canonical best metric: `{cfg['best_metric']}`",
        f"- Checkpoint dir: `{summary['checkpoint_dir']}`",
        "",
        "## Eval Summary",
        "",
        "The init columns below use fixed offline DAv2 RGB zero-shot baselines where available; "
        "splits without a fixed baseline use the pretrain eval parsed from `train.log`.",
        "",
        "| Split | Init abs_rel | Best abs_rel (epoch) | Last abs_rel | Init d1 | Best d1 (epoch) | Last d1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    best = summary["best_per_split"]
    for split in SPLITS:
        item = best[split]
        best_abs_epoch = "" if item["best_abs_rel_epoch"] is None else f" (e{item['best_abs_rel_epoch']})"
        best_d1_epoch = "" if item["best_d1_epoch"] is None else f" (e{item['best_d1_epoch']})"
        lines.append(
            f"| {SPLIT_LABELS[split]} | "
            f"{_fmt(item['dav2_rgb_zs_abs_rel'])} | "
            f"{_fmt(item['best_abs_rel'])}{best_abs_epoch} | "
            f"{_fmt(item['last_abs_rel'])} | "
            f"{_fmt(item['dav2_rgb_zs_d1'])} | "
            f"{_fmt(item['best_d1'])}{best_d1_epoch} | "
            f"{_fmt(item['last_d1'])} |"
        )

    first_loss = next((item.get("avg_loss") for item in parsed["epochs"] if item.get("avg_loss") is not None), None)
    last_loss = next((item.get("avg_loss") for item in reversed(parsed["epochs"]) if item.get("avg_loss") is not None), None)
    lines.extend(
        [
            "",
            "## Loss Trend",
            "",
            f"- Epoch avg loss: `{_fmt(first_loss, 6)}` -> `{_fmt(last_loss, 6)}`.",
            "- See `loss_curves_full.png` for the full curve.",
            "",
            "## Generated Files",
            "",
            "- `loss_curves_full.png`",
            "- `eval_abs_rel_curves_full.png`",
            "- `eval_d1_curves_full.png`",
            "- `epoch_metrics_full.csv`",
            "- `train_points.csv`",
            "- `summary_full.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notes_zh(path: Path, exp_name: str, summary: dict) -> None:
    lines = [
        f"# 多 split analysis 记录 - `{exp_name}`",
        "",
        "## 输出",
        "",
        "- 曲线：`loss_curves_full.png`、`eval_abs_rel_curves_full.png`、`eval_d1_curves_full.png`",
        "- 表格：`epoch_metrics_full.csv`、`train_points.csv`",
        "- 汇总：`summary.md`、`summary_full.json`",
        "",
        "## 口径",
        "",
        "- 五个 split：KITTI val、NYUv2 val、ETH3D fast、RobotCar day fast、RobotCar night fast。",
        "- 曲线里的 `epoch=-1` 表示 eval init；有离线 DAv2 RGB zero-shot baseline 的 split 使用固定值，NYUv2 使用 train.log 中的 pretrain eval。",
        f"- checkpoint/event 根目录记录为：`{summary['checkpoint_dir']}`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else exp_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_dir / "config.json"
    log_path = exp_dir / "train.log"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing train.log: {log_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    parsed = parse_train_log(log_path)
    summary = build_summary(config, parsed, exp_dir, args.checkpoint_root.expanduser())

    loss_type = config.get("loss_type", "loss")
    loss_lambda = config.get("loss_lambda_grad")
    loss_label = f"{loss_type} + {loss_lambda} * grad" if loss_lambda is not None else str(loss_type)

    save_epoch_csv(output_dir / "epoch_metrics_full.csv", parsed["epochs"])
    save_train_csv(output_dir / "train_points.csv", parsed["train_points"])
    (output_dir / "summary_full.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_loss(parsed, output_dir / "loss_curves_full.png", loss_label)
    plot_metric(
        parsed,
        metric="abs_rel",
        ylabel="abs_rel",
        title="abs_rel per epoch (multi-split, x=-1 is eval init)",
        path=output_dir / "eval_abs_rel_curves_full.png",
    )
    plot_metric(
        parsed,
        metric="d1",
        ylabel="d1",
        title="d1 per epoch (multi-split, x=-1 is eval init)",
        path=output_dir / "eval_d1_curves_full.png",
    )
    write_summary_md(output_dir / "summary.md", exp_dir.name, summary, parsed)
    write_notes_zh(output_dir / "notes_zh.md", exp_dir.name, summary)
    print(json.dumps({"exp_dir": str(exp_dir), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
