# VKITTI2 RAW-Adapter-style unprocessing 执行计划

本文覆盖把本机 VKITTI2 RGB 按 RAW-Adapter-style 解析近似路线生成 packed Bayer RAW-like 小样本，并做 contact sheet / per-image panel / 2x2 ablation panel 检查。这里不接入训练入口，也不复用当前工程已有的在线 VKITTI2 pseudo-RAW cache。

## 1. 本机 VKITTI2 情况

工作目录：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
conda activate dav3
```

VKITTI2 根目录：

```text
/mnt/drive/1111_new_works/VKITTI2
```

### 1.1 完整 VKITTI2 数据集统计（双相机全集，仅作参考）

```text
rgb:   42520 张 .jpg
depth: 42520 张 .png

scenes (Camera_0 + Camera_1 合计):
  Scene01:  8940
  Scene02:  4660
  Scene06:  5400
  Scene18:  6780
  Scene20: 16740

conditions (Camera_0 + Camera_1 合计):
  15-deg-left / 15-deg-right / 30-deg-left / 30-deg-right
  clone / fog / morning / overcast / rain / sunset
  每个 condition 4252 张  (= 42520 / 10)

cameras:
  Camera_0: 21260
  Camera_1: 21260
```

以上是 `/mnt/drive/1111_new_works/VKITTI2` 整目录下的完整文件数，仅作 Sec 9 容量估算和后续是否引入 Camera_1 的参考；本次样本检查不直接基于这个口径。

### 1.2 当前工程实际使用的 VKITTI2 split

```text
finetune_stf/dataset/splits/vkitti2/train.txt
```

已检查到的 split 情况：

```text
entries:       19559
missing RGB:   0
missing depth: 0
camera:        只有 Camera_0
```

也就是说，本计划全程基于这 19559 条 Camera_0 条目。在这个 split 下，按 condition 拆分大致是：

```text
每个 condition ≈ 1956 张 (= 19559 / 10)
每个 scene ≈ 8940/2 … 16740/2 之间，按 condition 平均分布
```

后面 Sec 3 `prepare_vkitti_unprocessing_samples.py` 在 `--cameras Camera_0` 过滤后，`counts.conditions` 打印出来就是 1956 量级，不是 4252。两者口径不同，不要混用。

当前工程没有对应的 `val.txt`。因此本计划默认只基于这个已有 `train.txt` 做 Camera_0 样本检查，不暗中构造 train/val 语义。如果后面需要验证泛化，可以单独确认按 scene 或 condition 划一个 diagnostic split。

本机已有一个 VKITTI2 cached pseudo-RAW：

```text
/mnt/drive/1111_new_works/VKITTI2/cache_raw_sensor_linear_dual_644x1008_k1rand_fp32_seed20260516
```

它是 `VKITTI2Raw` 在线 unprocessing 路线缓存下来的 19559 条样本，包含 resize/crop/flip/unprocessing preset 等训练语义。本计划暂时不用它，避免把“RAW-Adapter-style 离线 RGB unprocessing 检查”和“工程已有 pseudo-RAW 训练缓存”混在一起。

## 2. 本次检查目标

先默认只看当前 split 里的 `Camera_0`，从 19559 条中抽 12 张，生成：

```text
normal RAW-like
dark RAW-like
over-exposure RAW-like
preview PNG
contact sheet
per-image viz_dump panel
2x2 ablation panel
```

输出格式和 KITTI 计划保持一致：

```text
raw_packed: [4, H/2, W/2]
CFA: RGGB
channel order: [R, G1, G2, B]
storage: float16
```

VKITTI2 RGB 尺寸为 `1242x375`，高度是奇数，脚本会裁掉最后一行后 pack。因此典型 packed shape 是：

```text
(4, 187, 621)
```

本轮实际输出目录放在：

```text
plans/0524_unprocessing/vkitti/vkitti_rawadapter_camera0_s12_flat_viz/
```

### 2.1 `--ccm` 和 `--inverse-tone` 的 2x2 ablation

沿用 KITTI 的 4 组配置：

```text
baseline:   --ccm identity     无 --inverse-tone     仅 inverse-gamma + inverse-WB
ccm_only:   --ccm generic_d65  无 --inverse-tone     加一个轻量 camera-like CCM
tone_only:  --ccm identity     有 --inverse-tone     加一个保守的 inverse global tone
ccm_tone:   --ccm generic_d65  有 --inverse-tone     两个都加，最接近 RAW-Adapter 完整解析近似路线
```

`--noise-mean-mode zero` 在 4 组里保持一致，先不使用论文公式文字版的双倍期望。

### 2.2 VKITTI2 RGB 的特殊注意点

本机 VKITTI2 RGB 是 8-bit `.jpg`，是合成渲染后的 RGB 图像，并不是真实相机 RAW。这里生成的是：

```text
synthetic RAW-like packed Bayer derived from VKITTI2 rendered RGB JPEG
```

不要标成真实 VKITTI RAW。判读时注意：

```text
1. clone/fog/rain/sunset 是渲染 condition，不是传感器曝光条件；
   dark/over variants 是额外在线性 RAW-like 域合成出来的曝光退化。
2. 雾、雨、日落本身已经改变了图像分布，过曝/低光效果要和 condition 一起看，
   不要把 weather/domain shift 误判为 unprocessing bug。
3. JPEG 压缩纹理会进入 inverse-gamma 后的 RAW-like 数据，
   contact sheet 上优先看通道对齐、亮度合成和颜色倾向。
4. analytic backend 默认对输入跑 srgb_to_linear，本质是假定 VKITTI 渲染器输出
   遵循 sRGB OETF。Unity-based VKITTI2 的实际 OETF 并未明确文档化，
   这一步可能多做了一次 gamma 反演。判读时如果发现 contact sheet / panel
   出现系统性偏暗或偏亮，回到这条假设，考虑改 backend 或跳过 linearize。
```

### 2.3 smoke 路径保留策略

本计划的 s12 / s100 输出目录命名是 `*_s12_flat_viz` / `*_s100_flat_viz`，不含 `smoke|debug|tmp|codex_smoke` 这几个标记字。按项目约定，它们属于「诊断产物」而不是「smoke 测试产物」，跑完后不会被自动清理脚本删掉，需要人工评估后再处理。如果某次实验明确只是 smoke，应把目录名改成例如 `*_s12_flat_viz_smoke` 让自动清理可识别。

### 2.4 flat sample/output 和 per-image panel

样本链接使用 flat layout，把 split、scene、condition、camera、frame 写进文件名：

```text
train_Scene20_sunset_Camera_0_rgb_00560.jpg
train_Scene06_overcast_Camera_0_rgb_00125.jpg
```

对应 `.npz`、preview、per-image panel 和 ablation panel 也使用同一个 stem，便于按图像名检索。

本计划新增了 VKITTI2 样本准备脚本：

```text
plans/0524_unprocessing/vkitti/prepare_vkitti_unprocessing_samples.py
```

VKITTI2 的 contact sheet / viz_dump / ablation panel 入口在：

```text
plans/0524_unprocessing/vkitti/make_vkitti_unprocessing_contact_sheet.py
plans/0524_unprocessing/vkitti/make_vkitti_unprocessing_viz_dump.py
plans/0524_unprocessing/vkitti/make_vkitti_unprocessing_ablation_panels.py
```

这三个入口复用现有 manifest-compatible 绘图实现，输出格式和 KITTI 保持一致。

## 3. 准备 Camera_0 RGB 样本目录

不要把 VKITTI2 根目录直接传给 unprocessing 脚本，因为里面包含 RGB、depth、cache、多个相机和非目标文件。

先建立只含抽样 RGB 的 symlink 目录：

```bash
EXP_ROOT=plans/0524_unprocessing/vkitti/vkitti_rawadapter_camera0_s12_flat_viz
SAMPLE_DIR="$EXP_ROOT/rgb_samples_camera0_s12_flat"

mkdir -p "$EXP_ROOT"

python plans/0524_unprocessing/vkitti/prepare_vkitti_unprocessing_samples.py \
  --vkitti-root /mnt/drive/1111_new_works/VKITTI2 \
  --filelist-path finetune_stf/dataset/splits/vkitti2/train.txt \
  --output-dir "$SAMPLE_DIR" \
  --split-name train \
  --samples 12 \
  --strategy linspace \
  --seed 2026 \
  --cameras Camera_0 \
  --layout flat \
  --overwrite-symlinks
```

检查 manifest（不写死 `"train"` key，避免改 `--split-name` 时 KeyError）：

```bash
python - <<PY
import json
from pathlib import Path

p = Path("$SAMPLE_DIR/sample_manifest.json")
d = json.loads(p.read_text())
split, c = next(iter(d["counts"].items()))
print("split:", split)
print("available_with_rgb_depth:", c["available_with_rgb_depth"])
print("missing_paths:", c["missing_paths"])
print("selected:", c["selected"])
print("conditions:", json.dumps(c["conditions"], ensure_ascii=False))
print("num_samples:", len(d["samples"]))
print("first_sample:", d["samples"][0]["sample_name"])
PY
```

预期：

```text
split = train
available_with_rgb_depth = 19559
missing_paths = 0
selected = 12
num_samples = 12
conditions ≈ {15-deg-left: 1956, ..., sunset: 1956}
```

## 4. 执行 RAW-Adapter-style unprocessing

公共参数含义：

```text
--backend analytic:       使用解析近似版，不使用当前工程已有 VKITTI2Raw pseudo-RAW transform。
--noise-mean-mode zero:   y = l*x + zero-mean noise。
--variants normal,dark,over: 同时保存正常、低光、过曝三种 RAW-like。
--save-preview:           保存 debug preview PNG。
--save-mosaic:            debug 阶段同时保存 H*W 单通道 mosaic。
--storage float16:        小样本检查阶段足够。
```

4 组 ablation 共享同一份 sample dir：

```bash
OUT_BASE="$EXP_ROOT/rawadapter_unproc_camera0_s12"

for cfg in baseline ccm_only tone_only ccm_tone; do
  case $cfg in
    baseline)  CCM_ARG="--ccm identity";    TONE_ARG="" ;;
    ccm_only)  CCM_ARG="--ccm generic_d65"; TONE_ARG="" ;;
    tone_only) CCM_ARG="--ccm identity";    TONE_ARG="--inverse-tone" ;;
    ccm_tone)  CCM_ARG="--ccm generic_d65"; TONE_ARG="--inverse-tone" ;;
  esac

  python plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py \
    --input-dir "$SAMPLE_DIR" \
    --output-dir "${OUT_BASE}_${cfg}" \
    --backend analytic \
    --variants normal,dark,over \
    --max-images 0 \
    --save-preview \
    --save-mosaic \
    --storage float16 \
    --noise-mean-mode zero \
    $CCM_ARG $TONE_ARG
done
```

执行后目录结构：

```text
plans/0524_unprocessing/vkitti/vkitti_rawadapter_camera0_s12_flat_viz/
  rgb_samples_camera0_s12_flat/
    sample_manifest.json
    train_Scene20_sunset_Camera_0_rgb_00560.jpg -> symlink 到 /mnt/drive/1111_new_works/VKITTI2
    ...

  rawadapter_unproc_camera0_s12_baseline/
  rawadapter_unproc_camera0_s12_ccm_only/
  rawadapter_unproc_camera0_s12_tone_only/
  rawadapter_unproc_camera0_s12_ccm_tone/
```

## 5. 检查数值和文件结构

先看一个 config 的 manifest：

```bash
cat "${OUT_BASE}_baseline/manifest.json"
```

抽 baseline 的一个 normal `.npz` 看 shape / dtype / 值域：

```bash
python - <<PY
import numpy as np
from pathlib import Path

p = next(Path("${OUT_BASE}_baseline/normal").rglob("*.npz"))
d = np.load(p)
x = d["raw_packed"]
print("file:", p)
print("raw_packed:", x.shape, x.dtype, float(x.min()), float(x.max()))
if "raw_mosaic" in d:
    m = d["raw_mosaic"]
    print("raw_mosaic:", m.shape, m.dtype, float(m.min()), float(m.max()))
PY
```

预期：

```text
raw_packed: (4, 187, 621) float16
数值大体在 [0, 1]
```

对 4 个 config 做统一 sanity check：

```bash
for cfg in baseline ccm_only tone_only ccm_tone; do
  echo "==== $cfg ===="
  python - <<PY
import numpy as np
from pathlib import Path

p = next(Path("${OUT_BASE}_${cfg}/normal").rglob("*.npz"))
d = np.load(p)
x = d["raw_packed"]
print(p.name, x.shape, x.dtype, "[", float(x.min()), float(x.max()), "]")
PY
done
```

## 6. 生成 contact sheet、viz_dump 和 ablation panel

VKITTI2 原图宽高比和 KITTI 接近，继续用 `400x120` 缩略图。

每个 config 各生成一张 contact sheet。每一行：

```text
original RGB | normal preview | dark preview | over preview
```

```bash
for cfg in baseline ccm_only tone_only ccm_tone; do
  python plans/0524_unprocessing/vkitti/make_vkitti_unprocessing_contact_sheet.py \
    --sample-manifest "$SAMPLE_DIR/sample_manifest.json" \
    --raw-output-dir "${OUT_BASE}_${cfg}" \
    --thumb-width 400 \
    --thumb-height 120 \
    --group-by all
done
```

输出：

```text
${OUT_BASE}_${cfg}/contact_sheets/all_contact_sheet.jpg
```

如果想看某一个 config 的每图分布面板，先建议看 `ccm_only`：

```bash
python plans/0524_unprocessing/vkitti/make_vkitti_unprocessing_viz_dump.py \
  --sample-manifest "$SAMPLE_DIR/sample_manifest.json" \
  --raw-output-dir "${OUT_BASE}_ccm_only" \
  --tile-width 400 \
  --tile-height 120 \
  --dist-height 220 \
  --max-images 0
```

输出：

```text
${OUT_BASE}_ccm_only/viz_dump/*_panel.jpg
${OUT_BASE}_ccm_only/viz_dump/summary.json
${OUT_BASE}_ccm_only/viz_dump/manifest.jsonl
```

生成 5 张跨 config 的 ablation panel。和 KITTI 参考产物（`plans/0524_unprocessing/kitti_rawadapter_image02_s12_flat_viz/ablation_panels_s5/*.jpg`，`include_dav2_depth=true, encoder=vits`）保持视觉一致，本计划默认也开启 DAV2 vits 深度行：

```bash
python plans/0524_unprocessing/vkitti/make_vkitti_unprocessing_ablation_panels.py \
  --sample-manifest "$SAMPLE_DIR/sample_manifest.json" \
  --raw-output-base "$OUT_BASE" \
  --output-dir "$EXP_ROOT/ablation_panels_s5" \
  --num-samples 5 \
  --strategy linspace \
  --seed 2026 \
  --thumb-width 400 \
  --thumb-height 120 \
  --include-distribution \
  --dist-height 220 \
  --include-dav2-depth \
  --dav2-encoder vits
```

输出类似：

```text
$EXP_ROOT/ablation_panels_s5/train_Scene20_sunset_Camera_0_rgb_00560_ablation_panel.jpg
$EXP_ROOT/ablation_panels_s5/summary.json
$EXP_ROOT/ablation_panels_s5/dav2_depth/*.png
```

说明：

```text
1. --strategy linspace 在按 image_path 排序的 19559 条上等距取 5 张，
   由于 Scene20 占 43% 左右，5 张里大概率有 2~3 张落在 Scene20，
   想要确认 scene 分布更均的话，可以再跑一次 --strategy random，
   把输出目录换成 ablation_panels_s5_random，便于横向对比。
2. dav2 深度行会在 ablation panel 每一格下面附加 DAV2-vits 的彩色深度，
   既能看 unprocessing 对 DAV2 推理的影响，也是 KITTI 参考图同款。
   首次运行需要本地存在 depth_anything_v2_vits.pth，
   脚本默认在 checkpoints/、/home/caq/333_cvpr/da_ours/checkpoints/、
   /mnt/drive/3333_raw/checkpoints/ 顺序查找。
```

判读重点：

```text
1. 单个 config 内部
   - normal preview 没有明显颜色通道错位。
   - dark preview 明显变暗但结构仍可辨。
   - over preview 有合理高光饱和，而不是整张全白。

2. 4 个 config 横向对比
   - baseline vs ccm_only:  CCM 是否让色调更 camera-like，有无颜色塌缩。
   - baseline vs tone_only: inverse-tone 是否合理拉开高光，没有破坏性反向曲线。
   - baseline vs ccm_tone:  叠加后颜色/亮度是否仍在合理范围。

3. VKITTI2 condition
   - fog/rain/sunset 的输入分布本身不同，判断时要先看 original RGB。
   - 不要把渲染 domain shift 当成 RAW 合成问题。
```

## 7. 可选：按 condition 做专项样本

如果 s12 看起来整体正常，但想单独检查 fog/rain/sunset，**不要**直接 `--conditions fog rain sunset --samples 12 --strategy linspace`：那样按 image_path 排序后 linspace 抽样，12 张里很容易扎堆某一个 condition（fog 全占满或 sunset 全占满），起不到横向对比的目的。

推荐两种正确做法。

### 7.1 三种 condition 各跑一次（推荐，分布最可控）

每次只过滤一个 condition + 抽 4 张，sample dir 各自独立、命名清晰：

```bash
WEATHER_ROOT=plans/0524_unprocessing/vkitti/vkitti_rawadapter_camera0_weather_per_condition_s4_flat_viz
mkdir -p "$WEATHER_ROOT"

for cond in fog rain sunset; do
  COND_EXP="$WEATHER_ROOT/$cond"
  COND_SAMPLE_DIR="$COND_EXP/rgb_samples_camera0_${cond}_s4_flat"
  mkdir -p "$COND_EXP"

  python plans/0524_unprocessing/vkitti/prepare_vkitti_unprocessing_samples.py \
    --vkitti-root /mnt/drive/1111_new_works/VKITTI2 \
    --filelist-path finetune_stf/dataset/splits/vkitti2/train.txt \
    --output-dir "$COND_SAMPLE_DIR" \
    --split-name train \
    --samples 4 \
    --strategy linspace \
    --seed 2026 \
    --cameras Camera_0 \
    --conditions $cond \
    --layout flat \
    --overwrite-symlinks
done
```

然后对每个 `$WEATHER_ROOT/$cond/` 复用 Sec 4 和 Sec 6 的 unprocessing / panel 命令（把 `EXP_ROOT`、`SAMPLE_DIR`、`OUT_BASE` 替换成对应 condition 的路径）。

### 7.2 一次跑三 condition 但用 random 策略

如果只想要一份 12 张混合样本而不是三份小目录，至少把 `--strategy` 改成 `random`，让抽样不再被路径排序绑定：

```bash
EXP_ROOT=plans/0524_unprocessing/vkitti/vkitti_rawadapter_camera0_weather_s12_flat_viz
SAMPLE_DIR="$EXP_ROOT/rgb_samples_camera0_weather_s12_flat"

mkdir -p "$EXP_ROOT"

python plans/0524_unprocessing/vkitti/prepare_vkitti_unprocessing_samples.py \
  --vkitti-root /mnt/drive/1111_new_works/VKITTI2 \
  --filelist-path finetune_stf/dataset/splits/vkitti2/train.txt \
  --output-dir "$SAMPLE_DIR" \
  --split-name train \
  --samples 12 \
  --strategy random \
  --seed 2026 \
  --cameras Camera_0 \
  --conditions fog rain sunset \
  --layout flat \
  --overwrite-symlinks
```

random 不保证严格 4/4/4，但在 seed=2026 下大概率会拿到三种 condition 都有。跑完后用 `sample_manifest.json` 里的 `counts.conditions` 核对一下实际分布。

之后复用 Sec 4 和 Sec 6 的 unprocessing / panel 命令。

## 8. 如果样本效果通过，再跑更多样本

s100 不需要 tmux，仍然写在 `plans/0524_unprocessing/vkitti` 下：

```bash
EXP_ROOT=plans/0524_unprocessing/vkitti/vkitti_rawadapter_camera0_s100_flat_viz
SAMPLE_DIR="$EXP_ROOT/rgb_samples_camera0_s100_flat"
OUT_BASE="$EXP_ROOT/rawadapter_unproc_camera0_s100"

mkdir -p "$EXP_ROOT"

python plans/0524_unprocessing/vkitti/prepare_vkitti_unprocessing_samples.py \
  --vkitti-root /mnt/drive/1111_new_works/VKITTI2 \
  --filelist-path finetune_stf/dataset/splits/vkitti2/train.txt \
  --output-dir "$SAMPLE_DIR" \
  --split-name train \
  --samples 100 \
  --strategy linspace \
  --seed 2026 \
  --cameras Camera_0 \
  --layout flat \
  --overwrite-symlinks

for cfg in baseline ccm_only tone_only ccm_tone; do
  case $cfg in
    baseline)  CCM_ARG="--ccm identity";    TONE_ARG="" ;;
    ccm_only)  CCM_ARG="--ccm generic_d65"; TONE_ARG="" ;;
    tone_only) CCM_ARG="--ccm identity";    TONE_ARG="--inverse-tone" ;;
    ccm_tone)  CCM_ARG="--ccm generic_d65"; TONE_ARG="--inverse-tone" ;;
  esac

  python plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py \
    --input-dir "$SAMPLE_DIR" \
    --output-dir "${OUT_BASE}_${cfg}" \
    --backend analytic \
    --variants normal,dark,over \
    --max-images 0 \
    --save-preview \
    --save-mosaic \
    --storage float16 \
    --noise-mean-mode zero \
    $CCM_ARG $TONE_ARG
done
```

## 9. 全量 Camera_0 的注意事项

当前 split 是 19559 张 Camera_0。单张 packed float16 大约 0.9 MB；如果保存 mosaic，大约再翻一倍。粗略量级：

```text
1 config，3 variants，不含 mosaic: 约 50-60 GB
1 config，3 variants，含 mosaic:   约 100-120 GB
4 configs，3 variants，不含 mosaic: 约 200-240 GB
4 configs，3 variants，含 mosaic:   约 400-500 GB，不建议
```

全量阶段约定（与 KITTI 计划口径一致，但全量保留 4 个 config 用于横向对比）：

```text
1. 先用 s12 / s100 在 4 个 config 上做完 contact sheet / ablation panel 判读。
2. 全量保留 baseline / ccm_only / tone_only / ccm_tone 4 个 config，不在这一步收敛。
3. 去掉 --save-mosaic；--save-preview 默认也去掉，确实需要 preview 再单独跑小样本。
4. 输出放到数据盘 /mnt/drive/1111_new_works/VKITTI2/ 下，不放工程 plans 目录。
5. 用 tmux，session 名 + 日志路径都要清晰。
6. 跑之前 df -h /mnt/drive 确认数据盘剩余 ≥ 280 GB 余量（200 GB 数据 + 80 GB buffer）。
```

跑之前先确认磁盘空间：

```bash
df -h /mnt/drive
```

全量命令模板（4 configs 在一个 tmux 里串行跑，sample dir 只准备一次）：

```bash
RUN_TAG=$(date +%m%d_%H%M)
SESSION="vkitti_unproc_full_camera0_${RUN_TAG}"
LOG_DIR=plans/0524_unprocessing/logs
LOG_PATH="$LOG_DIR/${SESSION}.log"
RUN_SCRIPT="$LOG_DIR/${SESSION}.sh"
FULL_SAMPLE_DIR="/mnt/drive/1111_new_works/VKITTI2/rawadapter_camera0_full_samples_${RUN_TAG}"
FULL_OUT_BASE="/mnt/drive/1111_new_works/VKITTI2/rawadapter_unproc_camera0_full_${RUN_TAG}"

mkdir -p "$LOG_DIR"

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /home/caq/6666_raw/dav2_raw_0522
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3

python plans/0524_unprocessing/vkitti/prepare_vkitti_unprocessing_samples.py \\
  --vkitti-root /mnt/drive/1111_new_works/VKITTI2 \\
  --filelist-path finetune_stf/dataset/splits/vkitti2/train.txt \\
  --output-dir "$FULL_SAMPLE_DIR" \\
  --split-name train \\
  --samples 0 \\
  --strategy linspace \\
  --seed 2026 \\
  --cameras Camera_0 \\
  --layout flat \\
  --overwrite-symlinks

for cfg in baseline ccm_only tone_only ccm_tone; do
  case \$cfg in
    baseline)  CCM_ARG="--ccm identity";    TONE_ARG="" ;;
    ccm_only)  CCM_ARG="--ccm generic_d65"; TONE_ARG="" ;;
    tone_only) CCM_ARG="--ccm identity";    TONE_ARG="--inverse-tone" ;;
    ccm_tone)  CCM_ARG="--ccm generic_d65"; TONE_ARG="--inverse-tone" ;;
  esac

  python plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py \\
    --input-dir "$FULL_SAMPLE_DIR" \\
    --output-dir "${FULL_OUT_BASE}_\${cfg}" \\
    --backend analytic \\
    --variants normal,dark,over \\
    --max-images 0 \\
    --storage float16 \\
    --noise-mean-mode zero \\
    \$CCM_ARG \$TONE_ARG
done
EOF

chmod +x "$RUN_SCRIPT"
tmux new -d -s "$SESSION" "bash '$RUN_SCRIPT' > '$LOG_PATH' 2>&1"
```

启动后记录：

```bash
tmux attach -t "$SESSION"
tail -f "$LOG_PATH"
```

跑完产物分布：

```text
/mnt/drive/1111_new_works/VKITTI2/rawadapter_camera0_full_samples_${RUN_TAG}/   # 19559 个 symlink + manifest
/mnt/drive/1111_new_works/VKITTI2/rawadapter_unproc_camera0_full_${RUN_TAG}_baseline/
/mnt/drive/1111_new_works/VKITTI2/rawadapter_unproc_camera0_full_${RUN_TAG}_ccm_only/
/mnt/drive/1111_new_works/VKITTI2/rawadapter_unproc_camera0_full_${RUN_TAG}_tone_only/
/mnt/drive/1111_new_works/VKITTI2/rawadapter_unproc_camera0_full_${RUN_TAG}_ccm_tone/
```

注意：4 个 config 的 `FULL_OUT_BASE` 后缀和 ablation_panels 脚本期望的 `_<config>` 命名一致，跑完后可以直接拿同一个 `sample_manifest.json` 配 `--raw-output-base $FULL_OUT_BASE` 做全量 ablation 抽样可视化。

## 10. 需要确认后再做的选择

下面这些事会改变实验语义或视觉判读口径，本计划默认按当前选项执行，需要变动时由用户显式确认：

```text
1. 是否引入 Camera_1。
   当前工程 train.txt 只有 Camera_0；如果要用完整 VKITTI2，两相机会从 19559 扩到更多样本，
   并且要确认 stereo/right-view 是否应和当前训练协议混用。

2. 是否构造 VKITTI2 val/diagnostic split。
   可以按 held-out scene 或 held-out condition 划分，但这会改变评估协议，
   需要显式命名并写进后续正式 launch config。

3. 是否在 ablation panel 上启用 DAV2 vits 深度行。
   当前 Sec 6 默认启用，和 KITTI 参考产物对齐；如果只想看 unprocessing 本身、
   不想引入 DAV2 推理（例如想加速、或回避 DAV2 checkpoint 依赖），
   可以去掉 --include-dav2-depth --dav2-encoder vits，
   panel 会回退到只有 RGB + 3 variants preview + 可选 distribution 的版本。
```
