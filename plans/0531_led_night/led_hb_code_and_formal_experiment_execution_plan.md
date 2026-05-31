# LED-HB code integration and formal experiment execution plan

Date: 2026-05-31

Revision (2026-05-31, post-review):
- Geometry changed to **aspect-preserving isotropic** `1080x1920 -> resize x0.7 -> 756x1344 -> 2x2 -> 378x672` (exact 16:9, no anisotropic squish). Previous draft used `700x1260 / 350x630` which silently distorted the aspect ratio.
- Batch size now tries `bs=8 accum=1` first (matching VKITTI), with a verified `bs=4 accum=2` OOM fallback.
- D0 sign is pinned once on the full val set and passed explicitly; no per-script re-derivation.
- Eval-time feature ablation reuses the existing incremental-tool path instead of a second implementation.

Scope: implement the project integration, smoke tests, formal launch scripts, evaluation, and visualization needed to run the first-stage LED-HB main experiment. Dataset download/extraction is treated as already complete and is not repeated here.

This plan starts from:

```text
/mnt/drive/3333_raw/led_night/extracted/HB/
/mnt/drive/3333_raw/led_night/manifests/led_hb_extracted_inventory_summary.json
/mnt/drive/3333_raw/led_night/manifests/hb_train_extracted_inventory.jsonl
/mnt/drive/3333_raw/led_night/manifests/hb_val_extracted_inventory.jsonl
```

Observed prepared data facts:

```text
HB/train rgb_ldr_color:      14,997
HB/train depth_image_plane:  14,997
HB/val rgb_ldr_color:         4,999
HB/val depth_image_plane:     4,999

train maps: china, herrenberg, ottosuhrallee
val map:    hamburg
RGB:        1080 x 1920 PNG
depth GT:   distance_to_image_plane EXR float32, meters, may contain inf
```

---

## 0. Execution rules

Use the project root:

```bash
cd /home/caq/6666_raw/dav2_raw_0522
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3
```

All long-running training/eval queues must run in tmux and write logs under:

```text
finetune_stf/logs/
```

Formal experiment names must start with the launch timestamp in `MMDD_HHMM` format from the local machine running the job.

Smoke test outputs must contain one of:

```text
smoke
debug
tmp
codex_smoke
```

Delete successful smoke artifacts only if the path clearly contains one of those markers. Keep failed smoke artifacts and report the path.

---

## 1. Fixed experiment-semantics for LED-HB stage 1

These are experiment-semantic parameters. They must be explicit in code, config JSON, and formal launch scripts.

```text
dataset_name = led_hb
illumination = HB
train_split = hb_train_all
val_split = hb_val_stride5_n1000_seed42
train_maps = china,herrenberg,ottosuhrallee
val_maps = hamburg
depth_label = distance_to_image_plane
depth_unit = meter
eval_protocol = per_image_affine_disp_depth_anything_v2
main_min_depth = 1.0
main_max_depth = 200.0
secondary_max_depth = 80.0
low_light_stress = off
randomize_unprocessing = false
raw_adapter_fixed_light_scale = 1.0
raw_storage_format = synthetic_packed_bayer_4ch_halfres
```

Use this geometry for the first formal run:

```text
led_geometry_mode = resize_fullres_756x1344_then_halfres_378x672
source RGB/depth: 1080 x 1920
resized fullres:  756 x 1344
model sensor hw:  378 x 672
DAV2 backbone pad: already dynamic, both 378 and 672 are multiples of 14
```

Reason: LED source is exactly 16:9 (1080 x 1920). `756 x 1344` is an isotropic `x0.7` resize that preserves the aspect ratio exactly (`756/1080 == 1344/1920 == 0.7`) with no anisotropic distortion, no padding, and no cropping. Both `756 x 1344` and the 2x2-halved `378 x 672` are multiples of 14, so they are DAV2-compatible. `378 x 672` keeps detail close to the original `350 x 630` intent (254,016 px, ~2.19x the VKITTI N-series 187 x 621) while feeding the frozen DAV2 backbone undistorted, in-distribution images, which also improves the frozen D0/D1 baseline quality.

`describe_geometry()` must record the per-axis resize scale and assert isotropy (`abs(scale_h - scale_w) < 1e-6`); it must not claim aspect preservation while silently distorting.

Do not use path-name inference for any of these semantics. The loader and scripts must take explicit args and validate them centrally.

---

## 2. Target code changes

Add these files:

```text
foundation/engine/datasets/led_hb.py
foundation/tools/build_led_hb_filelists.py
foundation/tools/smoke_led_hb_dataset.py
foundation/tools/eval_led_hb_d0.py
foundation/tools/train_led_hb_residual_control.py
foundation/tools/train_led_hb_incremental_residual.py
foundation/tools/eval_led_hb_formal.py
foundation/tools/make_led_hb_residual_panels.py
foundation/tools/summarize_led_hb_formal.py
finetune_stf/scripts/smoke/0531_smoke_led_hb_stack.sh
finetune_stf/scripts/formal/0531_run_led_hb_c2_queue.sh
finetune_stf/scripts/formal/0531_run_led_hb_nseries_queue.sh
finetune_stf/scripts/formal/0531_run_led_hb_posteval_panels.sh
```

Modify these files:

```text
foundation/engine/datasets/__init__.py
```

Do not change existing VKITTI behavior while adding LED. Prefer LED-specific tools over broad refactors.

---

## 3. Preflight checks

Run:

```bash
cd /home/caq/6666_raw/dav2_raw_0522
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3

python -V
python - <<'PY'
import cv2
import torch
print("cv2", cv2.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
PY

test -d /mnt/drive/3333_raw/led_night/extracted/HB
test -f /mnt/drive/3333_raw/led_night/manifests/led_hb_extracted_inventory_summary.json
cat /mnt/drive/3333_raw/led_night/manifests/led_hb_extracted_inventory_summary.json
```

Check EXR reading:

```bash
OPENCV_IO_ENABLE_OPENEXR=1 python - <<'PY'
from pathlib import Path
import cv2
import numpy as np

root = Path("/mnt/drive/3333_raw/led_night/extracted/HB")
rgb_path = root / "train/china/ldr_color/0.png"
depth_path = root / "train/china/distance_to_image_plane/0.exr"
image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
print("rgb", rgb_path, None if image is None else image.shape, None if image is None else image.dtype)
print("depth", depth_path, None if depth is None else depth.shape, None if depth is None else depth.dtype)
if image is None or depth is None:
    raise SystemExit("failed to read RGB or EXR depth")
finite = np.isfinite(depth)
print("finite_ratio", float(finite.mean()))
print("finite_minmax", float(np.nanmin(depth[finite])), float(np.nanmax(depth[finite])))
PY
```

If EXR reading fails, set `OPENCV_IO_ENABLE_OPENEXR=1` before importing cv2 inside all new LED tools and dataset modules.

---

## 4. Build LED-HB project filelists

Implement:

```text
foundation/tools/build_led_hb_filelists.py
```

Required args:

```text
--led-root
--illum HB
--train-maps china,herrenberg,ottosuhrallee
--val-maps hamburg
--val-stride 5
--val-n 1000
--seed 42
--out-dir
```

Behavior:

1. Read from `${LED_ROOT}/extracted/HB/{train,val}/{map}`.
2. Pair by integer frame stem.
3. Require these paths for every selected sample:

```text
ldr_color/{frame}.png
distance_to_image_plane/{frame}.exr
camera_params/{frame}.json
transforms/{frame}.p
```

Note: only `ldr_color` (RGB) and `distance_to_image_plane` (GT depth) are consumed by the depth pipeline. `camera_params` and `transforms` are required-present and carried as path metadata only; they are not used to compute depth. Verified counts are equal (14,997 each), so requiring all four does not currently drop any sample.

4. Sort frames numerically within each map.
5. Train uses all frames from `china,herrenberg,ottosuhrallee`.
6. Val uses `hamburg` with fixed stride 5 and at most 1000 frames. Stride is applied over the numerically-sorted frame **index positions** (keep every 5th sorted frame: positions 0,5,10,...), not over raw frame-number values, so gaps in stems do not change the count. With 4,999 sorted hamburg frames this yields exactly 1000. Use seed only for tie-breaking / future offset recording; do not random-sample adjacent video frames. The `seed42` tag is for provenance only — the selection is deterministic.
7. Write 4-column filelists:

```text
absolute_rgb_path absolute_depth_path absolute_camera_params_path absolute_transforms_path
```

8. Also write a summary JSON with counts, maps, stride, n, and first/last 10 samples.

Expected output:

```text
finetune_stf/dataset/splits/led_hb/hb_train_all.txt
finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt
finetune_stf/dataset/splits/led_hb/hb_split_summary.json
```

Run:

```bash
python foundation/tools/build_led_hb_filelists.py \
  --led-root /mnt/drive/3333_raw/led_night \
  --illum HB \
  --train-maps china,herrenberg,ottosuhrallee \
  --val-maps hamburg \
  --val-stride 5 \
  --val-n 1000 \
  --seed 42 \
  --out-dir finetune_stf/dataset/splits/led_hb

wc -l finetune_stf/dataset/splits/led_hb/hb_train_all.txt
wc -l finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt
cat finetune_stf/dataset/splits/led_hb/hb_split_summary.json
```

Expected:

```text
train: 14997
val:   1000
```

Stop if any count differs unexpectedly.

---

## 5. Add LED-HB dataset module

Implement:

```text
foundation/engine/datasets/led_hb.py
```

Classes:

```text
LEDHBHalfresRGBDepth
LEDHBRaw
```

Constants:

```text
LED_GEOMETRY_MODE_CHOICES = ("resize_fullres_756x1344_then_halfres_378x672",)
LED_RAW_STORAGE_FORMAT_CHOICES = ("synthetic_packed_bayer_4ch_halfres",)
LED_RGB_INPUT_SPACE_CHOICES = ("resize_area_756x1344_then_2x2_area",)
LED_DEPTH_TARGET_SPACE_CHOICES = ("resize_nearest_756x1344_then_2x2_valid_mean",)
LED_DEPTH_LABEL_CHOICES = ("distance_to_image_plane",)
LED_DEPTH_UNIT_CHOICES = ("meter",)
```

Central validators:

```text
validate_led_hb_rgb_depth_semantics(...)
validate_led_hb_raw_semantics(...)
```

The validators must enforce:

```text
dataset_name == led_hb
illumination == HB
led_geometry_mode == resize_fullres_756x1344_then_halfres_378x672
input_height == 378
input_width == 672
depth_label == distance_to_image_plane
depth_unit == meter
min_depth == 1.0
max_depth in {200.0, 80.0}
```

For RGB/D1 methods:

```text
input_domain == rgb
model_input_tensor == image
raw_storage_format == not_applicable
unprocessing_method == not_applicable
raw_adapter_* == not_applicable
```

For RAW/x3 methods:

```text
input_domain == raw4
model_input_tensor == raw
raw_storage_format == synthetic_packed_bayer_4ch_halfres
unprocessing_method == raw_adapter_style
raw_adapter_backend == analytic
randomize_unprocessing == false
raw_adapter_fixed_light_scale == 1.0
raw_adapter_variant_policy == normal
```

Reading and geometry:

1. Set `OPENCV_IO_ENABLE_OPENEXR=1` before importing cv2 in this module.
2. Read RGB PNG as RGB float32 `[0, 1]`.
3. Read EXR depth as float32 meters. If cv2 returns multiple channels, use the first channel and record that in metadata.
4. Valid depth mask:

```text
finite(depth) & depth >= min_depth & depth <= max_depth
```

5. Resize RGB fullres to `756x1344` with area interpolation.
6. Resize depth fullres to `756x1344` with nearest-neighbor interpolation.
7. Resize valid mask to `756x1344` with nearest-neighbor interpolation.
8. Downsample RGB 2x2 area to `378x672`.
9. Downsample depth 2x2 valid mean to `378x672`.
10. For `LEDHBRaw`, unprocess the resized fullres RGB tensor `3x756x1344` to packed Bayer `4x378x672`.
11. For horizontal flip during train, apply flip before RAW packing and before 2x2 RGB/depth downsampling so RGB, raw, and GT remain aligned.

Sample keys for `LEDHBHalfresRGBDepth`:

```text
image              torch.float32 [3,378,672], ImageNet-normalized
rgb_preview        torch.float32 [3,378,672], [0,1]
depth              torch.float32 [378,672], meters
valid_mask         torch.bool [378,672]
image_path
depth_path
camera_params_path
transforms_path
sample_name        led_hb_{split}_{map}_{frame}
target_space       metric_depth
dataset_index
geometry_params
```

Additional keys for `LEDHBRaw`:

```text
raw                torch.float32 [4,378,672]
isp_params
```

`describe_geometry()` must return:

```text
dataset_name
illumination
source_original_hw
resized_fullres_hw
input_hw
raw_storage_format
led_geometry_mode
rgb_input_space
depth_target_space
depth_label
depth_unit
```

Update:

```text
foundation/engine/datasets/__init__.py
```

to export the LED classes and validators.

---

## 6. Add LED dataset smoke tool

Implement:

```text
foundation/tools/smoke_led_hb_dataset.py
```

Required args:

```text
--led-train-list
--led-val-list
--input-height 378
--input-width 672
--min-depth 1.0
--max-depth 200.0
--led-geometry-mode resize_fullres_756x1344_then_halfres_378x672
--raw-storage-format synthetic_packed_bayer_4ch_halfres
--output
```

Checks:

```text
RGB dataset train/val lengths
RAW dataset train/val lengths
image shape == [3,378,672]
rgb_preview shape == [3,378,672]
depth shape == [378,672]
raw shape == [4,378,672]
per-image valid pixels > 0 and finite GT present (light sanity only)
per-image valid ratio recorded in the summary (expect ~0.3-0.4 at max_depth=200; not asserted)
raw min/max finite
depth finite/valid ratio
raw and image are aligned after optional hflip disabled in val
```

Run:

```bash
SMOKE_ROOT=plans/0531_led_night/codex_smoke_led_hb_dataset_$(date +%m%d_%H%M)
mkdir -p "${SMOKE_ROOT}"
python foundation/tools/smoke_led_hb_dataset.py \
  --led-train-list finetune_stf/dataset/splits/led_hb/hb_train_all.txt \
  --led-val-list finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt \
  --input-height 378 \
  --input-width 672 \
  --min-depth 1.0 \
  --max-depth 200.0 \
  --led-geometry-mode resize_fullres_756x1344_then_halfres_378x672 \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --output "${SMOKE_ROOT}/dataset_smoke_summary.json"
cat "${SMOKE_ROOT}/dataset_smoke_summary.json"
rm -rf "${SMOKE_ROOT}"
```

If it fails, keep `${SMOKE_ROOT}`.

---

## 7. Add L0 D0 evaluator

Implement:

```text
foundation/tools/eval_led_hb_d0.py
```

Purpose: produce the no-training LED-HB frozen DAV2-S baseline.

Required args:

```text
--encoder vits
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--led-val-list finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt
--input-height 378
--input-width 672
--min-depth 1.0
--max-depth 200.0
--max-val-samples optional
--output-dir
--device cuda
```

Behavior:

1. Build `LEDHBHalfresRGBDepth` with val mode.
2. Load frozen DAV2-S.
3. Use `CenterPadCropAdapter(sensor_hw=(378,672), backbone_hw=None)`.
4. Evaluate both D0 signs with per-image affine disparity alignment.
5. Write `recommended_d0_sign`.
6. Write metrics for sign `+1`, sign `-1`, and selected D0.
7. Include LED region metrics.

Run:

```bash
OUT=plans/0531_led_night/led_hb_l0_d0_eval_$(date +%m%d_%H%M)
CUDA_VISIBLE_DEVICES=0 python foundation/tools/eval_led_hb_d0.py \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --led-val-list finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt \
  --input-height 378 \
  --input-width 672 \
  --min-depth 1.0 \
  --max-depth 200.0 \
  --output-dir "${OUT}" \
  --device cuda
cat "${OUT}/d0_summary.json"
```

Expected artifact:

```text
plans/0531_led_night/led_hb_l0_d0_eval_<timestamp>/d0_summary.json
```

Pin `D0_SIGN` here, on the **full** val set (all 1000 frames), and treat it as a fixed experiment-semantic parameter: copy `recommended_d0_sign` from `d0_summary.json` and pass it **explicitly** as `D0_SIGN=` into the C2 queue. Do not let any downstream script re-derive the sign from a different (e.g. 64-sample) subset — re-derivation could disagree with this full-set result and silently change experiment semantics. The N-series queue then reads the sign back from the C2 `config.json` (the sign C2 actually trained with), which keeps the whole chain consistent.

---

## 8. Add LED-HB C2 training tool

Implement:

```text
foundation/tools/train_led_hb_residual_control.py
```

Base it on:

```text
foundation/tools/train_vkitti2_residual_control.py
```

Main differences:

1. Use `LEDHBHalfresRGBDepth`.
2. Use `--led-train-list` and `--led-val-list`.
3. Remove KITTI eval from this first LED-HB tool.
4. Replace VKITTI hardcoded validation with LED semantics:

```text
dataset_name == led_hb
dataset_geometry_mode == resize_fullres_756x1344_then_halfres_378x672
input_height,input_width == 378,672
min_depth,max_depth == 1.0,200.0 for formal training
```

5. Save config keys. The tool must dump the full resolved arg namespace to `config.json` (so the N-series validator can match it) and additionally record these LED-specific keys explicitly:

```text
dataset_name
illumination
led_train_list
led_val_list
train_split
val_split
led_geometry_mode
depth_label
depth_unit
rgb_input_space
depth_target_space
eval_protocol = per_image_affine_disp_depth_anything_v2
input_domain
model_input_tensor
raw_storage_format
input_height
input_width
min_depth
max_depth
encoder
pretrained_from
residual_feature_source
residual_alpha
d0_sign
```

   The new C2 tool must expose argparse entries for the full LED arg surface used by the §13 launch command, including: `--dataset-name`, `--illumination`, `--depth-label`, `--depth-unit`, `--rgb-input-space`, `--depth-target-space`, `--led-train-list`, `--led-val-list`, `--train-split`, `--val-split`, and `--dataset-geometry-mode` (LED choices). It must **not** carry over VKITTI-only args such as `--fullres-even-policy`.

6. `config.json` must contain enough metadata for the N-series tool to verify C2 compatibility (every key the N-series C2-compat validator checks in section 9 must be present and match).
7. Evaluate and save both:

```text
overall.final
overall.D0
overall.delta.final_minus_D0
region.final
region.D0
region.delta.final_minus_D0
```

8. Use LED-specific region keys:

```text
boundary_abs_rel
d0_high_error_abs_rel
near_1_20_abs_rel
mid_20_50_abs_rel
far50_abs_rel
far100_abs_rel
dark_q20_abs_rel
saturated_abs_rel
```

`dark_q20_abs_rel` must be per-image luma quantile, not a fixed `luma < 0.15`.

Formal C2 hyperparameters:

```text
experiment_id=C2
residual_feature_source=d0
front_end=dav2_rgb_frozen
input_domain=rgb
model_input_tensor=image
raw_storage_format=not_applicable
residual_alpha=0.5
epochs=20
bs=8
accum_steps=1
lr=1e-4
weight_decay=1e-4
hflip_prob=0.5
amp=bf16
seed=42
```

Batch-size policy (matches VKITTI first): try `bs=8 accum_steps=1` first, exactly like the VKITTI C2/N-series runs. On CUDA OOM, fall back to `bs=4 accum_steps=2` (same effective batch 8) and record the actual values in the launch script and `config.json`.

Memory context: at `378x672` (254,016 px), `bs=8` is ~2.19x the per-step activation memory of VKITTI's `bs=8` at 187x621, so OOM is possible — especially for the N2/N7 RAW path with a trainable RAW encoder. The `bs=4 accum_steps=2` fallback is ~1.09x VKITTI `bs=8`, which fits. Before committing the formal queue, run a `bs=8 max_train_steps=2` probe (and the same at `bs=4` if `bs=8` OOMs) so the chosen size is verified, not assumed. Because the run name encodes `bs`/`acc`, restarting at the fallback size produces a correctly-labeled run.

---

## 9. Add LED-HB N-series training tool

Implement:

```text
foundation/tools/train_led_hb_incremental_residual.py
```

Base it on:

```text
foundation/tools/train_vkitti2_incremental_residual.py
```

Main differences:

1. Use `LEDHBRaw` for RAW/x3 methods.
2. Use `LEDHBHalfresRGBDepth` for RGB/D1 methods.
3. Use `--led-train-list` and `--led-val-list`.
4. Remove KITTI eval from first LED-HB tool.
5. Accept `--eval-only` for feature ablation and checkpoint re-eval.
6. Validate C2 metadata against LED keys, not VKITTI keys.
7. Validate all experiment-semantic parameters centrally before building datasets.
8. Rewrite the three hardcoded VKITTI guards copied from `train_vkitti2_incremental_residual.py` (these will otherwise reject all LED runs):
   - the `dataset_geometry_mode != "vkitti2_even_fullres_halfres_2x2"` check (around `validate_args`) -> accept only the LED geometry mode `resize_fullres_756x1344_then_halfres_378x672`.
   - the `(input_height, input_width) != (187, 621)` assertion -> require `(378, 672)`.
   - inside `validate_c2_metadata`: the `fullres_even_policy` key does not exist for LED -> replace it with `led_geometry_mode`; and replace the `vkitti_train_list` / `vkitti_val_list` matched keys with the LED list keys (`led_train_list` / `led_val_list`).

C2 compatibility validation must require:

```text
dataset_name == led_hb
illumination == HB
led_train_list == current led_train_list
led_val_list == current led_val_list
led_geometry_mode == current led_geometry_mode
input_height == 378
input_width == 672
min_depth == 1.0
max_depth == 200.0
depth_label == distance_to_image_plane
depth_unit == meter
encoder == vits
d0_sign == current d0_sign
pretrained_from == current pretrained_from
residual_alpha == current residual_alpha
```

N-series method contracts:

```text
N5:
  input_domain=rgb
  model_input_tensor=image
  front_end=c2_frozen_d1_incremental
  incremental_feature_source=d1
  delta_condition=d1_only
  gate_condition=d1_only
  raw_feature_encoder_trainable=not_applicable
  raw_storage_format=not_applicable

N3:
  input_domain=rgb
  model_input_tensor=image
  front_end=c2_frozen_rgb_incremental
  incremental_feature_source=rgb
  delta_condition=feature_only
  gate_condition=feature_d1
  raw_feature_encoder_trainable=not_applicable
  raw_storage_format=not_applicable

N2:
  input_domain=raw4
  model_input_tensor=raw
  front_end=c2_frozen_raw_ram_incremental
  incremental_feature_source=x3
  delta_condition=feature_only
  gate_condition=feature_d1
  raw_feature_encoder_trainable=true
  raw_storage_format=synthetic_packed_bayer_4ch_halfres

N7:
  input_domain=raw4
  model_input_tensor=raw
  front_end=c2_frozen_raw_ram_incremental
  incremental_feature_source=x3
  delta_condition=feature_d1_stopgrad
  gate_condition=feature_d1
  raw_feature_encoder_trainable=true
  raw_storage_format=synthetic_packed_bayer_4ch_halfres
```

Clean RA0 RAW args for N2/N7:

```text
--unprocessing-method raw_adapter_style
--vkitti-unprocessing-preset not_applicable
--no-randomize-unprocessing
--raw-adapter-backend analytic
--raw-adapter-cfa-pattern RGGB
--raw-adapter-packed-channel-order R_Gr_Gb_B
--raw-adapter-rgb-transfer srgb_piecewise
--raw-adapter-inverse-tone global_0p15
--raw-adapter-ccm identity
--raw-adapter-red-gain-range 1.9 2.4
--raw-adapter-blue-gain-range 1.5 1.9
--raw-adapter-fixed-red-gain 2.15
--raw-adapter-fixed-blue-gain 1.70
--raw-adapter-fixed-light-scale 1.0
--raw-adapter-dark-light-scale-range 0.05 0.4
--raw-adapter-over-light-scale-range 1.5 2.5
--raw-adapter-shot-noise 0.001
--raw-adapter-read-noise 0.0005
--raw-adapter-noise-mean-mode zero
--raw-adapter-black-level 0.0
--raw-adapter-white-level 1.0
--raw-adapter-random-seed-policy dataloader_generator
--raw-adapter-variant-policy normal
--raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0
```

Note: with `--no-randomize-unprocessing`, shot/read noise parameters are required by the resolver but no noise realization should be applied. Verify this in the saved unprocessing summary.

RGB/D1 NA args:

```text
--unprocessing-method not_applicable
--vkitti-unprocessing-preset not_applicable
--raw-adapter-backend not_applicable
--raw-adapter-cfa-pattern not_applicable
--raw-adapter-packed-channel-order not_applicable
--raw-adapter-rgb-transfer not_applicable
--raw-adapter-inverse-tone not_applicable
--raw-adapter-ccm not_applicable
--raw-adapter-red-gain-range not_applicable
--raw-adapter-blue-gain-range not_applicable
--raw-adapter-fixed-red-gain not_applicable
--raw-adapter-fixed-blue-gain not_applicable
--raw-adapter-fixed-light-scale not_applicable
--raw-adapter-dark-light-scale-range not_applicable
--raw-adapter-over-light-scale-range not_applicable
--raw-adapter-shot-noise not_applicable
--raw-adapter-read-noise not_applicable
--raw-adapter-noise-mean-mode not_applicable
--raw-adapter-black-level not_applicable
--raw-adapter-white-level not_applicable
--raw-adapter-random-seed-policy not_applicable
--raw-adapter-variant-policy not_applicable
--raw-adapter-variant-weights not_applicable
```

Formal N-series hyperparameters (same bs policy as C2: try `bs=8 accum_steps=1` first, fall back to `bs=4 accum_steps=2` on OOM — most likely needed for N2/N7 with a trainable RAW encoder):

```text
epochs=10
bs=8
accum_steps=1
lr=1e-4
weight_decay=1e-4
hflip_prob=0.5
amp=bf16
seed=42
lambda_final=1.0
lambda_boundary=2.0
lambda_grad=0.5
lambda_keep_good_d1=0.2
lambda_gate_sparse=0.05
lambda_invalid_keep=0.1
lambda_lowfreq_loss=0.0
lowpass_kernel=31
```

Method-specific:

```text
N5: lambda_lp=0.5, q_good=0.3
N3: lambda_lp=0.5, q_good=0.3
N2: lambda_lp=0.8, q_good=0.3
N7: lambda_lp=0.5, q_good=0.3
```

Evaluation output must include:

```text
overall.final
overall.D1
overall.D0
overall.delta_final_minus_D1
overall.delta_D1_minus_D0
region.final
region.D1
region.D0
region.delta_final_minus_D1
region.delta_D1_minus_D0
diagnostics
```

Use LED region keys:

```text
boundary_abs_rel
d0_high_error_abs_rel
d1_high_error_abs_rel
near_1_20_abs_rel
mid_20_50_abs_rel
far50_abs_rel
far100_abs_rel
dark_q20_abs_rel
saturated_abs_rel
```

---

## 10. Add secondary max-depth re-eval

Implement:

```text
foundation/tools/eval_led_hb_formal.py
```

Purpose: evaluate a completed C2/N-series run under either `max_depth=200` or `max_depth=80` without retraining.

Reuse, do not re-implement: the genuinely new capability here is the eval-time `max_depth` override (200 vs 80) without retraining. The N-series feature-ablation logic already exists in `train_vkitti2_incremental_residual.py` via `--eval-only` + `--eval-feature-ablation-mode {true,zero,mean,shuffle}` + `--feature-ablation-scope` + `--feature-ablation-donor-offset`. This tool must call into that same eval/ablation code path (import and reuse it) rather than writing a second ablation implementation, so the two cannot drift. Keep this tool LED-specific only in dataset construction and the `max_depth` override.

Required args:

```text
--run-dir
--checkpoint
--run-kind d0|c2|nseries
--max-depth 200.0 or 80.0
--output-dir
--device cuda
```

Behavior:

1. Load `config.json`.
2. Override only evaluation `max_depth`; do not mutate training config.
3. Rebuild LED dataset from `led_val_list`.
4. Rebuild model from config and checkpoint.
5. Save:

```text
metrics.json
per_sample.jsonl
```

6. For N-series x3 ablation, support:

```text
--feature-ablation-mode true|zero|mean|shuffle
--feature-ablation-scope both|delta|gate
--feature-ablation-key x3
--feature-ablation-donor-offset 1
```

---

## 11. Add panels

Implement:

```text
foundation/tools/make_led_hb_residual_panels.py
```

Build from `foundation/tools/make_residual_vs_c2_panels.py`, but use LED loaders and add RAW/x3 views.

For each chosen val sample, save at least:

```text
RGB / ldr_color
raw4 visualization
x3 visualization
GT depth
D0 depth
D1 depth
Final depth
D0 abs-rel error
D1 abs-rel error
Final abs-rel error
Final - D1 improvement map
gate map
gate * delta map
```

Also support comparison panels:

```text
N7 vs N3 improvement-over-D1
N7 vs N5 improvement-over-D1
N7 vs N2 improvement-over-D1
```

Selection modes:

```text
fixed
linspace
scan_topk_better_worse_by_final_minus_D1_absrel
```

Default panel command should create 8 mixed panels per method.

---

## 12. Add full-stack smoke script

Implement:

```text
finetune_stf/scripts/smoke/0531_smoke_led_hb_stack.sh
```

It must:

1. Use `dav3`.
2. Use `plans/0531_led_night/codex_smoke_led_hb_stack_<timestamp>` as root.
3. Run `py_compile` for all new Python files.
4. Run `build_led_hb_filelists.py` into a smoke output copy or verify existing split files.
5. Run `smoke_led_hb_dataset.py`.
6. Run `eval_led_hb_d0.py --max-val-samples 8`.
7. Run C2 tiny train:

```text
epochs=1
bs=1
accum_steps=1
max_train_steps=2
max_val_samples=4
num_workers=0
```

8. Run N5 tiny train using the smoke C2 checkpoint.
9. Run N3 tiny train using the smoke C2 checkpoint.
10. Run N2 tiny train using the smoke C2 checkpoint.
11. Run N7 tiny train using the smoke C2 checkpoint.
12. Run N7 feature ablation eval for:

```text
true
zero
mean
shuffle
```

13. Run tiny panel generation for N7.
14. Check required artifacts exist:

```text
config.json
val_metrics.json
run_summary.json
best_abs_rel.pth or latest.pth
feature ablation metrics
at least one panel jpg/png
```

15. If all pass and `KEEP_SMOKE != 1`, delete smoke root only after verifying it matches:

```text
*/codex_smoke* | */smoke* | */debug* | */tmp*
```

Run:

```bash
KEEP_SMOKE=0 CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/smoke/0531_smoke_led_hb_stack.sh
```

Do not launch formal runs until this passes.

---

## 13. Formal C2 queue script

Implement:

```text
finetune_stf/scripts/formal/0531_run_led_hb_c2_queue.sh
```

Pattern: follow the existing tmux wrapper style in:

```text
finetune_stf/scripts/formal/0524_run_vkitti_cseries_residual_controls_queue.sh
```

Defaults:

```bash
ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0522}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-led_hb_c2}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LED_TRAIN_LIST="${LED_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_train_all.txt}"
LED_VAL_LIST="${LED_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt}"
D0_SIGN="${D0_SIGN:?set D0_SIGN to recommended_d0_sign from the section 7 full-val L0 D0 eval}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
```

Derived (after computing `RUN_TIMESTAMP` and the formal run name `RUN_NAME`):

```bash
SAVE="${EXP_ROOT}/${RUN_NAME}"
HEAVY="${HEAVY_ROOT}/${RUN_NAME}"
```

Wrapper behavior:

1. If not `--run-internal`, create tmux session:

```text
<MMDD_HHMM>_led_hb_c2
```

2. Print:

```text
tmux session
queue log path
attach command
tail command
```

3. Internal mode:
   - run preflight file checks
   - run smoke unless `RUN_SMOKE=0`
   - require `D0_SIGN` to be set explicitly (it is `?`-guarded above); refuse to start otherwise. Do **not** re-derive the sign from a 64-sample subset here — it must be the value pinned in section 7 on the full val set, so C2 records the same sign the N-series will later read back from `config.json`.
   - launch formal C2

Formal run name:

```text
${RUN_TIMESTAMP}_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}
```

Formal launch core command:

```bash
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
  python foundation/tools/train_led_hb_residual_control.py \
  --experiment-id C2 \
  --dataset-name led_hb \
  --illumination HB \
  --input-domain rgb \
  --model-input-tensor image \
  --dataset-geometry-mode resize_fullres_756x1344_then_halfres_378x672 \
  --raw-storage-format not_applicable \
  --rgb-input-space resize_area_756x1344_then_2x2_area \
  --depth-target-space resize_nearest_756x1344_then_2x2_valid_mean \
  --depth-label distance_to_image_plane \
  --depth-unit meter \
  --front-end dav2_rgb_frozen \
  --encoder vits \
  --pretrained-from "${PRETRAINED}" \
  --led-train-list "${LED_TRAIN_LIST}" \
  --led-val-list "${LED_VAL_LIST}" \
  --train-split hb_train_all \
  --val-split hb_val_stride5_n1000_seed42 \
  --input-height 378 \
  --input-width 672 \
  --min-depth 1.0 \
  --max-depth 200.0 \
  --residual-feature-source d0 \
  --residual-alpha 0.5 \
  --d0-sign "${D0_SIGN}" \
  --hflip-prob 0.5 \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --log-interval 100 \
  --save-interval 1 \
  --eval-interval 1 \
  --save-best-checkpoint \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --save-path "${SAVE}" \
  --heavy-save-path "${HEAVY}"
```

Launch:

```bash
CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/formal/0531_run_led_hb_c2_queue.sh
```

After completion, record:

```bash
C2_RUN_DIR=<printed save path>
C2_CHECKPOINT=<printed heavy path>/best_abs_rel.pth
test -d "${C2_RUN_DIR}"
test -f "${C2_CHECKPOINT}"
cat "${C2_RUN_DIR}/best_val_metrics.json"
```

---

## 14. Formal N-series queue script

Implement:

```text
finetune_stf/scripts/formal/0531_run_led_hb_nseries_queue.sh
```

Pattern: follow:

```text
finetune_stf/scripts/formal/0527_run_vkitti_nseries_incremental_queue.sh
```

Defaults:

```bash
ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0522}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-led_hb_nseries}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LED_TRAIN_LIST="${LED_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_train_all.txt}"
LED_VAL_LIST="${LED_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt}"
C2_RUN_DIR="${C2_RUN_DIR:?set completed LED-HB C2 run dir}"
C2_CHECKPOINT="${C2_CHECKPOINT:?set completed LED-HB C2 checkpoint}"
D0_SIGN="${D0_SIGN:-}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
RUN_N5_D1="${RUN_N5_D1:-1}"
RUN_N3_RGB="${RUN_N3_RGB:-1}"
RUN_N2_X3="${RUN_N2_X3:-1}"
RUN_N7_STOPGRAD="${RUN_N7_STOPGRAD:-1}"
```

Wrapper behavior:

1. Refuse to start if `C2_RUN_DIR` or `C2_CHECKPOINT` is missing.
2. If `D0_SIGN` empty, parse it from `${C2_RUN_DIR}/config.json`.
3. Run full-stack smoke unless `RUN_SMOKE=0`.
4. Launch enabled methods sequentially in the same tmux session.
5. Refuse to overwrite existing `SAVE` or `HEAVY` paths.

Common args must explicitly include:

```text
--dataset-name led_hb
--illumination HB
--dataset-geometry-mode resize_fullres_756x1344_then_halfres_378x672
--rgb-input-space resize_area_756x1344_then_2x2_area
--depth-target-space resize_nearest_756x1344_then_2x2_valid_mean
--depth-label distance_to_image_plane
--depth-unit meter
--train-split hb_train_all
--val-split hb_val_stride5_n1000_seed42
--input-height 378
--input-width 672
--min-depth 1.0
--max-depth 200.0
--eval-protocol per_image_affine_disp_depth_anything_v2
```

Run names:

```text
${ts}_led_hb_n5_d1_lp0p5_q0p3_lfl0p0_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}
${ts}_led_hb_n3_rgb_lp0p5_q0p3_lfl0p0_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}
${ts}_led_hb_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}
${ts}_led_hb_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}
```

The `bs`/`acc` tags must be interpolated from the actual `BS`/`ACCUM_STEPS` (primary `bs8_acc1`, OOM fallback `bs4_acc2`), so the run name always reflects the size that actually ran.

Launch after C2 completes:

```bash
export C2_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_C2_RUN_NAME>
export C2_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_C2_RUN_NAME>/best_abs_rel.pth
CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/formal/0531_run_led_hb_nseries_queue.sh
```

When the script starts, it must print:

```text
tmux session: <session>
queue log: <path>
attach: tmux attach -t <session>
monitor: tail -f <path>
```

---

## 15. Feature ablation formal eval

After N2 and N7 complete, run feature ablation eval for both.

Use:

```text
feature_ablation_modes = true,zero,mean,shuffle
feature_ablation_scope = both
feature_ablation_key = x3
feature_ablation_donor_offset = 1
```

Implement this either inside:

```text
finetune_stf/scripts/formal/0531_run_led_hb_posteval_panels.sh
```

or as a separate section in the N-series queue after training if runtime is acceptable.

Example command for one mode:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --live-stream -n dav3 \
  python foundation/tools/eval_led_hb_formal.py \
  --run-kind nseries \
  --run-dir "${N7_RUN_DIR}" \
  --checkpoint "${N7_CHECKPOINT}" \
  --max-depth 200.0 \
  --feature-ablation-mode shuffle \
  --feature-ablation-scope both \
  --feature-ablation-key x3 \
  --feature-ablation-donor-offset 1 \
  --output-dir "plans/0531_led_night/diagnostics/${POST_TS}/n7_x3_shuffle_max200"
```

Run the same for:

```text
N2 true/zero/mean/shuffle
N7 true/zero/mean/shuffle
```

Repeat selected `true` and `shuffle` for `max_depth=80.0`.

---

## 16. Post-eval and panel queue script

Implement:

```text
finetune_stf/scripts/formal/0531_run_led_hb_posteval_panels.sh
```

Inputs:

```bash
C2_RUN_DIR
C2_CHECKPOINT
N5_RUN_DIR
N5_CHECKPOINT
N3_RUN_DIR
N3_CHECKPOINT
N2_RUN_DIR
N2_CHECKPOINT
N7_RUN_DIR
N7_CHECKPOINT
```

It must run in tmux and write:

```text
plans/0531_led_night/diagnostics/<MMDD_HHMM>_led_hb_posteval/
```

Steps:

1. Re-eval L0 D0 at `max_depth=200`.
2. Re-eval C2/N5/N3/N2/N7 at `max_depth=200`.
3. Re-eval C2/N5/N3/N2/N7 at `max_depth=80`.
4. Run x3 ablation for N2/N7.
5. Generate per-method panels.
6. Generate N7-vs-control comparison panels.
7. Summarize all metrics into Markdown/CSV/JSON.

Summary outputs:

```text
led_hb_formal_summary.md
led_hb_formal_summary.json
led_hb_formal_records.csv
```

The Markdown summary must contain:

```text
overall main [1,200] table
overall secondary [1,80] table
region main [1,200] table
final-D1 table for N-series
N7 vs C2/N5/N3/N2 deltas
x3 ablation table
panel manifest
```

Launch:

```bash
CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/formal/0531_run_led_hb_posteval_panels.sh
```

---

## 17. Minimal manual command sequence

Run these in order after implementation.

### 17.1 Build filelists

```bash
cd /home/caq/6666_raw/dav2_raw_0522
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3

python foundation/tools/build_led_hb_filelists.py \
  --led-root /mnt/drive/3333_raw/led_night \
  --illum HB \
  --train-maps china,herrenberg,ottosuhrallee \
  --val-maps hamburg \
  --val-stride 5 \
  --val-n 1000 \
  --seed 42 \
  --out-dir finetune_stf/dataset/splits/led_hb
```

### 17.2 Full-stack smoke

```bash
KEEP_SMOKE=0 CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/smoke/0531_smoke_led_hb_stack.sh
```

### 17.3 L0 D0

```bash
OUT=plans/0531_led_night/led_hb_l0_d0_eval_$(date +%m%d_%H%M)
CUDA_VISIBLE_DEVICES=0 python foundation/tools/eval_led_hb_d0.py \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --led-val-list finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt \
  --input-height 378 \
  --input-width 672 \
  --min-depth 1.0 \
  --max-depth 200.0 \
  --output-dir "${OUT}" \
  --device cuda
cat "${OUT}/d0_summary.json"
```

### 17.4 C2 formal

```bash
CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/formal/0531_run_led_hb_c2_queue.sh
```

Record the printed run paths when it completes.

### 17.5 N-series formal

```bash
export C2_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_C2_RUN_NAME>
export C2_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_C2_RUN_NAME>/best_abs_rel.pth
CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/formal/0531_run_led_hb_nseries_queue.sh
```

### 17.6 Post-eval and panels

```bash
export C2_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_C2_RUN_NAME>
export C2_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_C2_RUN_NAME>/best_abs_rel.pth
export N5_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_N5_RUN_NAME>
export N5_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_N5_RUN_NAME>/best_abs_rel.pth
export N3_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_N3_RUN_NAME>
export N3_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_N3_RUN_NAME>/best_abs_rel.pth
export N2_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_N2_RUN_NAME>
export N2_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_N2_RUN_NAME>/best_abs_rel.pth
export N7_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/<LED_HB_N7_RUN_NAME>
export N7_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/<LED_HB_N7_RUN_NAME>/best_abs_rel.pth

CUDA_VISIBLE_DEVICES=0 bash finetune_stf/scripts/formal/0531_run_led_hb_posteval_panels.sh
```

---

## 18. Completion criteria

Code integration is complete when:

- `python -m py_compile` passes for all new and modified Python files.
- `build_led_hb_filelists.py` creates 14,997 train lines and 1,000 val lines.
- `smoke_led_hb_dataset.py` verifies RGB/depth/raw shape and valid masks.
- Full-stack smoke script passes and removes successful smoke artifacts.
- L0 D0 eval writes `d0_summary.json`.
- C2 formal run writes `config.json`, `val_metrics.json`, `best_val_metrics.json`, `run_summary.json`, and `best_abs_rel.pth`.
- N5/N3/N2/N7 formal runs each write `config.json`, `val_metrics.json`, `best_val_metrics.json`, `run_summary.json`, and `best_abs_rel.pth`.
- N-series configs all point to the same C2 checkpoint and same LED train/val filelists.
- RAW/x3 configs show `randomize_unprocessing=false`, `raw_adapter_fixed_light_scale=1.0`, `raw_adapter_variant_policy=normal`, and no applied noise realization.
- Post-eval writes main `[1,200]` and secondary `[1,80]` summaries.
- N2/N7 x3 zero/mean/shuffle evals exist and are summarized.
- Panels exist for N7 and controls.

Stop and report exact log paths if any check fails.

---

## 19. Result interpretation checklist

Strong success:

```text
N7 overall AbsRel < C2 by >= 0.002
N7 boundary AbsRel < C2 by >= 0.015
N7 dark_q20/saturated better than N3
N7 better than N5
x3 zero/mean/shuffle degrades N7 clearly
```

Medium success:

```text
N7 overall near C2
N7 boundary or hard-region metrics clearly better
N7 not consistently better than N3
```

Failure:

```text
N7 not better than N5
N7 not better than N3
x3 ablation does not degrade
```

Report conclusions separately for:

```text
overall
boundary
dark_q20
saturated
d0_high_error
d1_high_error
far50/far100
```

Do not describe the LED experiment as real RAW. Use:

```text
true nighttime RGB-D + inverse-ISP synthetic RAW-like packed Bayer
```
