# KITTI Offline Eval And Epoch09 Visualization Plan

## Goal

Run offline evaluation only, no training, for the current VKITTI-trained RAW residual experiment on KITTI val.

Target run:

- Run dir: `finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10`
- Checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/epoch_09.pth`
- KITTI split: `/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt`
- KITTI base: `/mnt/drive/kitti`
- Environment: `dav3`

The output must include:

- KITTI val metrics comparing epoch09 model vs frozen DAv2 baseline `D0`.
- 10 qualitative visualization panels.
- Each panel must include RGB, pseudo-RAW, RGB distribution, RAW distribution, and the same comparison content as the existing VKITTI reference panel:
  `finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/qual_compare_epoch09_3x3_residual_sharedscale/08_validx1955_Camera_0_rgb_00391_compare_epoch09_3x3_sharedscale.jpg`

## Confirmed Constraints

- Do not modify the training queue right now.
- Do not copy the KITTI split into this repo right now.
- Do not launch training.
- Use KITTI only as offline val.
- Match current VKITTI halfres RAW experiment semantics exactly:
  - `raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - `fullres_even_policy=crop_bottom_to_even`
  - `rgb_input_space=halfres_2x2_area`
  - `depth_target_space=halfres_2x2_valid_mean`
  - `unprocessing_preset=sensor_linear_dual`
  - eval-time `randomize_unprocessing=False`
  - `input_height=187`
  - `input_width=621`
  - `min_depth=1.0`
  - `max_depth=80.0`

KITTI sample geometry checked locally:

- Example RGB: `/mnt/drive/kitti/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.jpg`
- Example depth: `/mnt/drive/kitti/annotated_depth/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png`
- Original image/depth shape: `375x1242`
- Halfres policy: crop bottom to `374x1242`, then pack/downsample to `187x621`.

Full KITTI val image-size scan:

| Source RGB shape | Count |
| --- | ---: |
| `375x1242` | 487 |
| `370x1226` | 71 |
| `376x1241` | 48 |
| `374x1238` | 23 |
| `370x1224` | 23 |

This means a strict VKITTI-style `even_fullres_hw == (374, 1242)` assertion would only accept the `375x1242` subset. The offline KITTI eval must explicitly handle this before implementation.

Chosen strategy: `canonical_even_pad_crop`.

- Keep all 652 KITTI val samples.
- First apply the experiment's even-height policy: crop one bottom row only when source height is odd.
- Then canonicalize RGB/depth/valid to the fixed even fullres canvas `(374, 1242)` required by the trained model:
  - center-pad smaller dimensions;
  - center-crop larger dimensions;
  - use edge padding for RGB to avoid artificial black borders;
  - use depth `0.0` and `valid=False` for padded pixels.
- Record `source_hw`, `after_even_policy_hw`, `canonical_even_hw`, and per-side pad/crop parameters in every sample and in `per_sample.jsonl`.
- Metrics are computed on all valid pixels after canonicalization. `metrics.json` must include a note that KITTI was evaluated with fixed-canvas pad/crop to match the trained model input size.

## Implementation Plan

### 1. Add A KITTI Halfres RAW Dataset Helper

Add this inside the offline eval script unless it becomes too large. If it becomes reusable, move it to `foundation/engine/datasets/kitti_raw.py`.

Responsibilities:

- Read two-column KITTI split rows.
- Implement KITTI path remapping with the same behavior as `finetune_stf/dataset/kitti_eval.py`, but do not import its private `_remap_kitti_path` helper:
  - `/mnt/bn/liheyang/Kitti/raw_data/.../*.png` maps to `/mnt/drive/kitti/.../*.jpg`
  - `/mnt/bn/liheyang/Kitti/data_depth_annotated/...` maps to `/mnt/drive/kitti/annotated_depth/...`
- Load RGB as float RGB in `[0, 1]`.
- Load KITTI annotated depth as `uint16 / 256.0`.
- Apply valid mask with `min_depth <= depth <= max_depth`.
  - Use the closed interval intentionally, to match `vkitti2_raw.py` and `train_vkitti2_raw_residual.py` evaluate behavior.
  - This differs from `KITTIEval`, which uses open interval bounds.
- Apply `canonical_even_pad_crop` to every source sample before packing:
  - crop bottom one row when height is odd;
  - pad/crop to `(374, 1242)`;
  - do not resize KITTI images or depths.
- Generate pseudo-RAW with `build_unprocessing_transform_from_preset(config["vkitti_unprocessing_preset"], randomize=False)`.
  - Log this as `unprocessing_preset (from VKITTI training config) = sensor_linear_dual, randomize=False`.
- Build:
  - `raw`: packed tensor `[4, 187, 621]`
  - `image`: ImageNet-normalized halfres RGB `[3, 187, 621]`, using 2x2 area average
  - `rgb_preview`: unnormalized halfres RGB `[3, 187, 621]`
  - `depth`: 2x2 valid mean depth `[187, 621]`
  - `valid_mask`: halfres valid mask `[187, 621]`
  - `image_path`, `depth_path`, `sample_name`, `geometry_params`

Acceptance checks:

- Dataset length is `652`.
- Source shape counts match the table above.
- All samples return:
  - `raw.shape == (4, 187, 621)`
  - `image.shape == (3, 187, 621)`
  - `rgb_preview.shape == (3, 187, 621)`
  - `depth.shape == (187, 621)`
  - `valid_mask.shape == (187, 621)`
- At least one non-`375x1242` sample is checked to confirm pad/crop metadata and output shape.

### 2. Add Offline Eval Script

Create:

`foundation/tools/eval_raw_residual_kitti.py`

CLI:

```bash
python foundation/tools/eval_raw_residual_kitti.py \
  --run-dir <run_dir> \
  --checkpoint <checkpoint> \
  --kitti-val-split <split> \
  --kitti-base <base> \
  --output-dir <output_dir> \
  --max-samples <optional> \
  --num-workers <optional> \
  --max-panels 10 \
  --panel-selection uniform \
  --sample-indices <optional comma-separated original KITTI dataset indices> \
  --device cuda
```

Core behavior:

- Load `config.json` from `--run-dir`.
- Validate that the config has the confirmed halfres RAW semantic parameters.
- Require explicit `--kitti-val-split`. Do not provide a default, because this repo does not contain `metric_depth/dataset/splits/kitti/val.txt`.
- Build `DepthAnythingV2` and `build_raw_residual_dav2_model(...)` from run config.
- Load `epoch_09.pth` strictly.
- Iterate KITTI val with `batch_size=1`.
- For every valid sample:
  - Run model with `{"image": image, "raw": raw, "valid_mask": valid_mask}`.
  - Evaluate `final = out["pred"]`.
  - Evaluate baseline `D0 = d0_sign * out["D0"]`.
  - Use `affine_align_disp(depth, pred_disp, valid_mask)`.
  - Use `compute_metrics(..., min_depth=1.0, max_depth=80.0)`.
- Save:
  - `metrics.json`
  - `per_sample.jsonl`
  - `eval.log`

Panel selection behavior:

- `--sample-indices` follows the existing VKITTI visualization script convention and refers to original KITTI dataset indices.
- If `--sample-indices` is omitted, `--panel-selection uniform` selects uniformly over the successfully evaluated `status="ok"` rows after `--max-samples` is applied.
- Enforce `max_panels <= ok_sample_count` for uniform selection. If fewer valid rows exist, fail with a clear error instead of silently producing fewer panels.
- If explicit sample indices are provided, evaluate those original indices for panels and record any skipped samples in `panels/manifest.json`.

Metric JSON shape:

```json
{
  "dataset": "kitti_val_halfres_raw",
  "samples": 652,
  "checkpoint": ".../epoch_09.pth",
  "note": "KITTI val is evaluated with min_depth=1.0 and canonical_even_pad_crop to match the VKITTI-trained fixed 187x621 RAW residual model; scores are not directly comparable to KITTI public benchmark settings.",
  "geometry_policy": {
    "name": "canonical_even_pad_crop",
    "target_even_fullres_hw": [374, 1242],
    "target_model_hw": [187, 621]
  },
  "overall": {
    "final": {"abs_rel": 0.0, "rmse": 0.0, "silog": 0.0, "d1": 0.0},
    "D0": {"abs_rel": 0.0, "rmse": 0.0, "silog": 0.0, "d1": 0.0},
    "delta": {
      "final_abs_rel_minus_D0_abs_rel": 0.0,
      "final_d1_minus_D0_d1": 0.0
    }
  }
}
```

`per_sample.jsonl` schema:

```json
{
  "dataset_index": 0,
  "sample_name": "2011_09_26_drive_0002_sync_image_02_0000000069",
  "image_path": "/mnt/drive/kitti/...",
  "depth_path": "/mnt/drive/kitti/annotated_depth/...",
  "status": "ok",
  "source_hw": [375, 1242],
  "after_even_policy_hw": [374, 1242],
  "canonical_even_hw": [374, 1242],
  "raw_hw": [187, 621],
  "raw_shape": [4, 187, 621],
  "geometry_policy": "canonical_even_pad_crop",
  "pad_crop": {
    "top_pad": 0,
    "bottom_pad": 0,
    "left_pad": 0,
    "right_pad": 0,
    "top_crop": 0,
    "bottom_crop": 0,
    "left_crop": 0,
    "right_crop": 0
  },
  "valid_pixels": 0,
  "final": {
    "abs_rel": 0.0,
    "sq_rel": 0.0,
    "rmse": 0.0,
    "rmse_log": 0.0,
    "log10": 0.0,
    "silog": 0.0,
    "d1": 0.0,
    "d2": 0.0,
    "d3": 0.0
  },
  "D0": {
    "abs_rel": 0.0,
    "sq_rel": 0.0,
    "rmse": 0.0,
    "rmse_log": 0.0,
    "log10": 0.0,
    "silog": 0.0,
    "d1": 0.0,
    "d2": 0.0,
    "d3": 0.0
  },
  "diagnostics": {
    "mean_gate": 0.0,
    "max_gate": 0.0,
    "mean_abs_delta": 0.0,
    "mean_abs_gate_delta": 0.0
  }
}
```

Allowed `status` values:

- `ok`
- `skipped_invalid_pixels`
- `skipped_metric_failure`
- `skipped_io_error`
- `skipped_geometry_error`

### 3. Generate 10 Visualization Panels

The script should generate panels during the same pass to avoid a second inference pass. Store the necessary arrays for selected samples only.

Default panel output:

`finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/kitti_eval_epoch09/panels/`

Panel count:

- Exactly 10 panels by default.
- Default selection: uniform over successfully evaluated `status="ok"` rows after `--max-samples` is applied.
- Also support explicit indices:

```bash
--sample-indices 0,72,144,216,288,360,432,504,576,651
```

Panel layout:

Use a larger layout than the existing 3x3 VKITTI panel because KITTI requires RGB/RAW distributions too.

Recommended layout: `4 columns x 3 rows`, 12 tiles per sample.

Tiles:

1. `RGB input`
2. `Pseudo-RAW preview`
   - Convert packed RAW to display RGB as `[R, 0.5 * (Gr + Gb), B]`.
   - Use percentile stretch for visualization only, e.g. p1..p99.
3. `RGB distribution`
   - Per-channel histogram over halfres RGB values.
   - Channels: R/G/B.
4. `RAW distribution`
   - Per-channel histogram over packed RAW values.
   - Channels: R/Gr/Gb/B.
5. `GT depth`
6. `DAV2-S depth`
7. `Ours epoch09`
8. `DAV2 error`
9. `Ours error`
10. `Err improve +green`
11. `Residual gate*delta`
12. `Gate`

Visualization scale rules:

- Depth tiles share the same per-panel depth scale:
  - p1..p99 of valid GT depth, clipped to `[1.0, 80.0]`.
- Error tiles share `0..error_max_abs_rel`, default `0.75`.
- Improvement tile:
  - green means `D0 error - final error > 0`, final improved.
  - red means final worse.
  - range `[-error_max_abs_rel, +error_max_abs_rel]`.
- Residual `gate*delta`:
  - use a global symmetric scale over the selected 10 panel samples, p99 abs.
- Gate:
  - fixed `0..1`.
- RGB/RAW histograms:
  - x-axis fixed `0..1`.
  - y-axis normalized per channel.
  - Include text stats: min, p50, p99, max.

Reusable code:

- Import and reuse depth/error/gate/residual colorization from:
  `foundation/tools/make_vkitti_raw_residual_qual_panels.py`
- Do not import histogram code from `plans/` in production code.
- Extract distribution drawing helpers into a stable module:
  `foundation/tools/_viz_distribution.py`
- Reuse the stable distribution helpers from `eval_raw_residual_kitti.py`. If later needed, update temporary scripts under `plans/0524_unprocessing/` to import the same helper.

Panel manifest:

Write `panels/manifest.json` with:

- selected dataset indices
- sample names
- image/depth paths
- per-sample final metrics
- per-sample D0 metrics
- raw stats
- RGB stats
- panel path
- `visualization.rgb_distribution`:
  - `channels=["R", "G", "B"]`
  - `x_range=[0.0, 1.0]`
  - `bins=128`
  - `normalization="per_channel_max"`
  - `stats=["min", "p50", "p99", "max"]`
- `visualization.raw_distribution`:
  - `channels=["R", "Gr", "Gb", "B"]`
  - `x_range=[0.0, 1.0]`
  - `bins=128`
  - `normalization="per_channel_max"`
  - `stats=["min", "p50", "p99", "max"]`

### 4. Output Paths

Use this default output root:

`finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/kitti_eval_epoch09`

Expected tree:

```text
kitti_eval_epoch09/
  eval.log
  metrics.json
  per_sample.jsonl
  panels/
    manifest.json
    01_kitti_<sample>_epoch09_panel.jpg
    ...
    10_kitti_<sample>_epoch09_panel.jpg
```

This is a formal analysis output, not a smoke output, so do not auto-delete it.

### 5. Smoke Test Before Full KITTI Eval

Use a clearly temporary path and delete it only after success:

```bash
SMOKE_OUT=plans/0524_new/codex_smoke_kitti_offline_eval_epoch09

conda run --live-stream -n dav3 python foundation/tools/eval_raw_residual_kitti.py \
  --run-dir finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10 \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/epoch_09.pth \
  --kitti-val-split /home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt \
  --kitti-base /mnt/drive/kitti \
  --output-dir "${SMOKE_OUT}" \
  --max-samples 4 \
  --max-panels 2 \
  --num-workers 0 \
  --device cuda
```

Smoke acceptance:

- `metrics.json` exists.
- `per_sample.jsonl` has at least 1 row.
- `panels/manifest.json` exists.
- Exactly 2 smoke panels exist.
- Each smoke panel is non-empty and has expected tile labels:
  - `RGB input`
  - `Pseudo-RAW preview`
  - `RGB distribution`
  - `RAW distribution`
  - `DAV2-S depth`
  - `Ours epoch09`
  - `Residual gate*delta`
  - `Gate`

After successful smoke:

```bash
rm -rf plans/0524_new/codex_smoke_kitti_offline_eval_epoch09
```

If smoke fails, keep the smoke directory and report it.

### 6. Full Offline Eval Command

```bash
OUT=finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/kitti_eval_epoch09

conda run --live-stream -n dav3 python foundation/tools/eval_raw_residual_kitti.py \
  --run-dir finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10 \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/epoch_09.pth \
  --kitti-val-split /home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt \
  --kitti-base /mnt/drive/kitti \
  --output-dir "${OUT}" \
  --max-panels 10 \
  --panel-selection uniform \
  --num-workers 2 \
  --device cuda
```

Run the smoke first and estimate full runtime from `elapsed_seconds / processed_samples * 652`.

- If the estimate is `<= 30` minutes, run the full eval in the foreground and keep stdout.
- If the estimate is `> 1` hour, run the full eval in tmux.
- If the estimate is between 30 minutes and 1 hour, prefer foreground unless the machine is unstable or the shell session is likely to disconnect.

tmux command if the estimate requires it:

```bash
SESSION="$(date +%m%d_%H%M)_kitti_offline_eval_epoch09"
LOG="finetune_stf/logs/${SESSION}.log"
tmux new-session -d -s "${SESSION}" "cd /home/caq/6666_raw/dav2_raw_0522 && OUT='${OUT}' conda run --live-stream -n dav3 python foundation/tools/eval_raw_residual_kitti.py --run-dir finetune_stf/exp/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10 --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0524_2141_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10/epoch_09.pth --kitti-val-split /home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt --kitti-base /mnt/drive/kitti --output-dir \"${OUT}\" --max-panels 10 --panel-selection uniform --num-workers 2 --device cuda 2>&1 | tee -a '${LOG}'"
```

Report after launching tmux:

```text
Started tmux session: <SESSION>
Log: <LOG>
Attach: tmux attach -t <SESSION>
Monitor: tail -f <LOG>
```

### 7. Validation Checklist

- `metrics.json` contains both `overall.final` and `overall.D0`.
- `overall.delta.final_abs_rel_minus_D0_abs_rel` is present.
- `metrics.json.note` records `min_depth=1.0`, non-benchmark KITTI comparability, and `canonical_even_pad_crop`.
- `per_sample.jsonl` row count equals visited KITTI samples, and `status="ok"` count equals the metrics sample count.
- `per_sample.jsonl` includes at least one non-`375x1242` source shape with fixed `raw_hw=[187, 621]`.
- `panels/` contains exactly 10 final panel JPGs.
- `panels/manifest.json` lists exactly 10 records.
- `panels/manifest.json` records RGB and RAW distribution bin count, axis range, and normalization.
- All 10 panels visibly contain RGB, pseudo-RAW, RGB distribution, RAW distribution, GT, D0, Ours, errors, improvement, residual, and gate.
- `eval.log` records the config semantic parameters and the checkpoint path.

## Notes

- The current `finetune_stf/dataset/kitti_eval.py` is useful as a reference for path remapping, but its raw-like branch is not directly suitable for this eval because it does not apply the same halfres depth target semantics as the current VKITTI residual experiment.
- Do not import private `_remap_kitti_path` from `KITTIEval`; copy the remap behavior into the new helper or expose a public helper in a future cleanup.
- This plan intentionally does not add `--val-dataset kitti` to the training entrypoint. It keeps the work scoped to offline analysis.
- No split copy is needed for this pass.
- `min_depth=1.0` is chosen to match the current training/eval semantics. This is not the KITTI public benchmark protocol, so the output is only for internal D0-vs-epoch09 comparison.
