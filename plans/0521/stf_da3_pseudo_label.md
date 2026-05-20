# STF DA3 Pseudo Label 生成计划

## Summary
生成与现有 DAV2 目录一致的 STF RGB-LUT 6216 张 pseudo depth：

- 输入清单沿用 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
- 输出新目录：`/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_${MMDD_HHMM}`
- 每个样本输出一对扁平文件：`<sample_name>.npy` 和 `<sample_name>.png`
- 使用 `dav3` conda 环境、GPU 0、模型 `/home/caq/dav3/DA3MONO-LARGE`
- `process_res=1008`，`process_res_method=upper_bound_resize`
- 可视化参数固化为 DA3 当前 `visualize_depth` 逻辑：`vis_cmap="Spectral"`、`vis_percentile=2`、`vis_inverse=True`。即仅对 `depth > 0` 取 `1/depth`，再按 p2/p98 做 percentile clip，避免 DA3 后续默认值变化导致历史结果不可复现。
- 注意：DA3MONO 的 `.npy` 数值不是 DAV2 pseudo label 的相对反深度。旧 DAV2 数值是 disparity-like relative inverse depth，越远值越小；DA3MONO 输出是 affine-invariant depth，越远值越大。这里只保持文件组织、shape、dtype 兼容，不声明数值口径兼容。

关键原因：`cam_stereo_left_lut` 实际有 12997 张图，而旧 DAV2 pseudo label 只覆盖 manifest 中 6216 张；并且 `da3 images` 会一次性处理整个目录，不适合直接跑 6216 张。因此需要写一个薄封装脚本按 manifest 逐张或小 batch 跑 DA3。

## Key Changes
创建脚本 `scripts/generate_stf_da3_pseudo.py`：

- 读取旧 DAV2 manifest，固定处理其中 6216 行，不扫描整个 RGB 目录。
- 写新 manifest 时只复用 `sample_name,split,rgb_path,sparse_depth_path`；必须显式重写 `pseudo_depth_npy` 和 `pseudo_vis_png` 为新 `output_root / f"{sample_name}.npy"`、`output_root / f"{sample_name}.png"`，禁止把旧 DAV2 manifest 中这两列原样复制到新 manifest。
- `stf_rgb_lut_inputs_6216.txt` 原样复刻旧 DAV2 文件格式：无 header，每行一个 RGB 绝对路径，内容来自 manifest 的 `rgb_path` 列，顺序与 manifest 行顺序完全一致。旧文件前三行已确认是：

```text
/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_lut/2018-02-03_20-57-26_00010.png
/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_lut/2018-02-05_12-04-44_00100.png
/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_lut/2018-02-05_12-04-44_00600.png
```

- 加载 `DepthAnything3.from_pretrained("/home/caq/dav3/DA3MONO-LARGE").to("cuda")` 一次。
- 默认 `batch_size=1`，保证每张 STF 图像独立推理，避免 DA3 多图目录模式把无关样本当作同一 scene。
- DA3 在 `process_res=1008` + `upper_bound_resize` 下不会直接输出原图分辨率；保存前必须将 raw prediction depth resize 回每张输入图像的原图分辨率。STF RGB-LUT 当前预期为 `1024x1920`，但脚本应从实际输入图读取 `original_hw`，不要硬编码。连续 depth 的上采样插值使用 `cv2.INTER_LINEAR`。
- 保存 depth-only 可视化 `.png`，不保存 DA3 默认拼接的 input+depth `depth_vis/*.jpg`。
- 可视化实现必须显式传入或等价实现 `vis_cmap="Spectral"`、`vis_percentile=2`、`vis_inverse=True`，并写入 `run_config.json`；不要依赖 DA3 包的默认参数。
- `.npy` 和 `.png` 都使用 atomic write：先写同目录临时文件，再 `tmp_path.replace(path)`，沿用 `scripts/generate_lod_day_dav2_pseudo.py` 的 `atomic_save_npy/atomic_save_png` 模式，避免中断后留下半写文件被 resume 误判为完成。
- 默认支持 resume：当目标 `<sample_name>.npy` 和 `<sample_name>.png` 都已存在且未传 `--overwrite` 时跳过该样本；增加 `--overwrite` flag 供强制重跑。若只存在其中一个文件，则重新生成这一对输出。
- 进度日志参考 LOD 脚本每 25 张左右打印一次，但每行同时包含 `imgs_per_sec`、`eta_sec`/`eta_hms` 和当前 `torch.cuda.max_memory_allocated()` 峰值显存，便于中途判断是否需要停止后调整 `process_res`。
- 写出：
  - `stf_rgb_lut_inputs_6216.txt`
  - `stf_rgb_lut_manifest_6216.csv`
  - `run_config.json`
  - `run_summary.json`
  - `failed_samples.json` 仅在有失败样本时生成
  - `tmux_stdout.log`

输出 manifest schema：

```text
sample_name,split,rgb_path,sparse_depth_path,pseudo_depth_npy,pseudo_vis_png
```

Manifest path policy:

- `pseudo_depth_npy` 必须等于 `<output_root>/<sample_name>.npy`
- `pseudo_vis_png` 必须等于 `<output_root>/<sample_name>.png`
- 旧 manifest 的 `pseudo_depth_npy/pseudo_vis_png` 只用于确认旧输出来源，不参与新 manifest 写出

Depth value policy:

- `run_config.json` 和 `run_summary.json` 必须写入：
  - `depth_value_units.value = "affine_invariant_depth_from_da3mono"`
  - `depth_value_units.direction = "larger_is_farther"`
  - `depth_value_units.compatibility_note = "Shape/dtype/file schema match DAV2 pseudo labels, but values are not DAV2 relative inverse depth/disparity."`
- 下游不得把 DA3 `.npy` 当作 DAV2 反深度直接使用；需要按 affine-invariant depth 或重新适配 alignment/normalization 逻辑处理。

Visualization metadata policy:

- `run_config.json` 必须写入：
  - `vis_cmap = "Spectral"`
  - `vis_percentile = 2`
  - `vis_inverse = true`
  - `vis_clip_percentiles = [2, 98]`
- `run_summary.json` 也保留同一组可视化参数，方便只看 summary 时复核输出口径。

Runtime metadata policy:

- `run_config.json` 和 `run_summary.json` 都记录 `process_res`、`process_res_method`、`align_to_input_ext_scale`、`use_ray_pose`、`ref_view_strategy`、`model_dir`、`model_identifier`。
- 尽量记录 DA3 代码版本和模型版本：`da3_repo_commit` 使用 `/home/caq/dav3` 下的 `git rev-parse HEAD`；`model_config_sha256` 记录 `/home/caq/dav3/DA3MONO-LARGE/config.json` 的 sha256；如能可靠定位权重文件，也记录 `model_weight_sha256` 或权重文件名与大小。
- `run_summary.json` 额外记录 `peak_cuda_memory_allocated_bytes`、`peak_cuda_memory_reserved_bytes`。

Resume / overwrite policy:

- 默认跳过已完整存在的 `(npy, png)` 输出对。
- `--overwrite` 会重写已存在输出。
- 计数写入 `run_summary.json`：`generated`、`skipped`、`failed`、`sample_count`。
- `run_config.json` 必须记录 `overwrite`、`target_resolution_policy="original_image_resolution"`、`resize_interpolation="cv2.INTER_LINEAR"`。

## Execution
先做 smoke test：

```bash
TS=$(date +%m%d_%H%M)
SMOKE_OUT=/mnt/drive/3333_raw/seeing_through_fog/codex_smoke_da3_stf_${TS}

source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3

CUDA_VISIBLE_DEVICES=0 python scripts/generate_stf_da3_pseudo.py \
  --manifest /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv \
  --output-root "$SMOKE_OUT" \
  --model-dir /home/caq/dav3/DA3MONO-LARGE \
  --process-res 1008 \
  --process-res-method upper_bound_resize \
  --align-to-input-ext-scale \
  --device cuda \
  --batch-size 1 \
  --vis-cmap Spectral \
  --vis-percentile 2 \
  --vis-inverse \
  --max-samples 20
```

Smoke 至少跑 20 张。成功后校验 `.npy/.png` 数量、shape、finite ratio、可视化可读性、新 manifest 的 `pseudo_depth_npy/pseudo_vis_png` 是否都指向 `$SMOKE_OUT`，以及 `run_config.json/run_summary.json` 是否写明 DA3 depth value units、可视化参数和峰值显存，然后只删除 `$SMOKE_OUT`。

正式任务用 tmux：

```bash
TS=$(date +%m%d_%H%M)
OUT=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_${TS}
LOG=${OUT}/tmux_stdout.log
SESSION=da3_stf_pseudo_${TS}

tmux new-session -d -s "$SESSION" "bash -lc '
set -euo pipefail
cd /home/caq/6666_raw/dav2_raw_0520
mkdir -p \"$OUT\"
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3
CUDA_VISIBLE_DEVICES=0 python scripts/generate_stf_da3_pseudo.py \
  --manifest /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv \
  --output-root \"$OUT\" \
  --model-dir /home/caq/dav3/DA3MONO-LARGE \
  --process-res 1008 \
  --process-res-method upper_bound_resize \
  --align-to-input-ext-scale \
  --device cuda \
  --batch-size 1 \
  --vis-cmap Spectral \
  --vis-percentile 2 \
  --vis-inverse \
  2>&1 | tee \"$LOG\"
'"
```

启动后报告：

```bash
tmux attach -t "$SESSION"
tail -f "$LOG"
```

## Test Plan
完成后执行验收：

- `run_summary.json` 中 `status=completed`，`sample_count=6216`，`failed=0`
- `run_config.json` 和 `run_summary.json` 都包含 `depth_value_units.value="affine_invariant_depth_from_da3mono"`、`direction="larger_is_farther"`，并明确不是 DAV2 relative inverse depth/disparity
- `run_config.json` 包含 `target_resolution_policy="original_image_resolution"`、`resize_interpolation="cv2.INTER_LINEAR"`、`vis_cmap="Spectral"`、`vis_percentile=2`、`vis_inverse=true`
- `run_config.json/run_summary.json` 包含 `process_res_method`、`align_to_input_ext_scale`、`use_ray_pose`、`model_config_sha256`、`da3_repo_commit`；若权重 hash 无法可靠计算，至少记录原因、权重文件名和大小
- 日志中每 25 张左右有进度行，包含 `imgs_per_sec`、`eta` 和 GPU peak memory；`run_summary.json` 也包含峰值显存
- 输出目录中 `.npy` 数量为 6216，`.png` 数量为 6216
- 输出 stem 与旧 manifest 的 `sample_name` 完全一致
- 新 manifest 的每一行 `pseudo_depth_npy/pseudo_vis_png` 都指向新输出目录 `$OUT` 下的 `<sample_name>.npy/.png`，不出现旧 DAV2 输出目录 `pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417`
- `stf_rgb_lut_inputs_6216.txt` 无 header，每行一个 RGB 绝对路径，内容与新 manifest 的 `rgb_path` 列逐行一致
- 随机抽查若干 `.npy`：
  - shape 等于对应输入 RGB 的原图分辨率；当前 STF RGB-LUT 预期为 `(1024, 1920)`
  - dtype 为 `float32`
  - 无 NaN/Inf
  - 非全零
- 中断后重跑 smoke：不加 `--overwrite` 时完整 `(npy, png)` 输出对会被 skip；加 `--overwrite` 时会重新生成
- 随机抽查若干 `.png`，确认 colormap 正常、没有空图或尺寸错误
- 如 smoke test 失败，保留 `codex_smoke` 目录用于排查；如正式任务失败，保留完整输出目录和日志

## Assumptions
- 目标是复刻旧 DAV2 pseudo label 的 6216 样本集合，而不是处理 `cam_stereo_left_lut` 里的全部 12997 张图。
- 本任务不替换旧 DAV2 pseudo label，新 DA3MONO pseudo label 写入独立输出目录。
- 下游希望 `.npy` 保持对应输入 RGB 的原图分辨率，所以 DA3 processed depth 会用 `cv2.INTER_LINEAR` resize 回原图尺寸；当前 STF RGB-LUT 预期为 `1024x1920`。
- 下游契约只在 manifest schema、文件命名、shape 和 dtype 层面兼容旧 DAV2 pseudo label；数值物理意义不兼容。DAV2 是 relative inverse depth / disparity-like，DA3MONO 是 affine-invariant depth。
- 使用 GPU 0；如 batch size 1 仍 OOM，则停止并报告，不擅自降低 `process_res=1008`。
