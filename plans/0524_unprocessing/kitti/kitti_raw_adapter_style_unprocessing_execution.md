# KITTI RAW-Adapter-style unprocessing 执行计划

本文只覆盖你刚提供的 RAW-Adapter-style unprocessing 方法在本机 KITTI train/val 上的样本生成和效果检查。这里不讨论、也不接入当前工程已有的在线 unprocessing 或网络训练/评估代码。

## 1. 本机 KITTI 情况

工作目录：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
conda activate dav3
```

KITTI 根目录：

```text
/mnt/drive/kitti
```

本机 `annotated_depth` 可用于定位 train/val 对应 RGB：

```text
/mnt/drive/kitti/annotated_depth/train
/mnt/drive/kitti/annotated_depth/val
```

已检查到的 `image_02` 左目可用情况：

```text
train: 38421 张 RGB 可用，4528 条 annotated depth 找不到对应 RGB
val:    3347 张 RGB 可用，79 条 annotated depth 找不到对应 RGB
```

样本准备脚本会跳过缺 RGB 的条目，并把前 100 个缺失例子写入 manifest。

## 2. 本次检查目标

先默认只看 KITTI 左目 `image_02`，从 train 和 val 各抽 12 张，生成：

```text
normal RAW-like
dark RAW-like
over-exposure RAW-like
preview PNG
contact sheet
```

输出格式来自你提供的脚本：

```text
raw_packed: [4, H/2, W/2]
CFA: RGGB
channel order: [R, G1, G2, B]
storage: float16
```

注意：KITTI 原图高度/宽度可能不是偶数，脚本会裁掉最后一行或列。因此 preview 对应的 packed 尺寸通常类似：

```text
(4, 187, 621) 或 (4, 188, 620)
```

这是正常现象。

### 2.1 关于 `--ccm` 和 `--inverse-tone` 的 2×2 ablation

本次同时跑 4 组配置，方便对比不同复杂度的 unprocessing 视觉效果：

```text
baseline:   --ccm identity     无 --inverse-tone     仅 inverse-gamma + inverse-WB
ccm_only:   --ccm generic_d65  无 --inverse-tone     加一个轻量 camera-like CCM
tone_only:  --ccm identity     有 --inverse-tone     加一个保守的 inverse global tone
ccm_tone:   --ccm generic_d65  有 --inverse-tone     两个都加，最接近 RAW-Adapter 完整路线
```

`baseline` 的解析管线退化为 `sRGB → linear → inverse-WB → pack → 曝光/噪声合成`，是最容易判断 inverse WB 和曝光合成是否正确的口径。其他 3 组用来观察 CCM 和 inverse-tone 各自对色调和高光的影响。`--noise-mean-mode zero` 在 4 组里保持一致，先不引入论文公式文字版的双倍期望。

### 2.2 KITTI RGB 是 JPEG 压缩 sRGB

本机 KITTI `image_02 / image_03` 是 8-bit `.jpg`，已经经过 demosaic、white balance、color correction、tone mapping、gamma 和 JPEG 压缩。这里的 unprocessing 只能近似生成 RAW-like，不要把输出当成真实传感器 RAW。注意：

```text
1. JPEG 8x8 块状量化会被 sRGB-to-linear 放大后写进 packed RAW；
   preview 上看到的轻微块状或振铃多半是 JPEG artifact，不是 unprocessing 的 bug。
2. 看 contact sheet 时优先判断通道对齐、曝光合成和颜色倾向，而不是绝对噪声水平。
3. 不要把这个 packed RAW 标注成 "真实 KITTI RAW"，更稳妥的叫法是
   "synthetic RAW-like packed Bayer derived from KITTI sRGB JPEG"。
```

### 2.3 本轮修订：flat sample/output 和 per-image viz dump

当前检查阶段不再在输出文件夹层面拆 `train/val` 或 drive 场景。样本链接使用单层 flat layout，并把 split、drive、camera、frame 写入 RGB 对应文件名，例如：

```text
train_2011_09_26_drive_0001_sync_image_02_0000000005.jpg
val_2011_09_30_drive_0016_sync_image_02_0000000187.jpg
```

对应的 `.npz`、preview 和 per-image panel 也使用同一个 stem，便于直接按图像名检索。新增每图一张 `viz_dump/*_panel.jpg`，面板内容为：

```text
original RGB | normal preview | dark preview | over preview
RGB value distribution | normal RAW packed distribution | dark RAW packed distribution | over RAW packed distribution
```

本轮实际输出目录为：

```text
plans/0524_unprocessing/kitti_rawadapter_image02_s12_flat_viz/
```

## 3. 准备 train/val RGB 样本目录

不要把 `/mnt/drive/kitti` 直接传给 unprocessing 脚本，因为里面包含多个相机、depth、timestamps 等非目标文件。

先建立一个只含抽样 RGB 的 symlink 目录。所有本次 s12 检查产物统一放到同一个实验根目录下，避免和 KITTI 原数据集或其他 debug 输出混在一起：

```bash
EXP_ROOT=plans/0524_unprocessing/kitti_rawadapter_image02_s12_ablation
SAMPLE_DIR="$EXP_ROOT/rgb_samples_image02_s12"

mkdir -p "$EXP_ROOT"

python plans/0524_unprocessing/kitti/prepare_kitti_unprocessing_samples.py \
  --kitti-root /mnt/drive/kitti \
  --output-dir "$SAMPLE_DIR" \
  --splits train val \
  --cameras image_02 \
  --samples-per-split 12 \
  --strategy linspace \
  --seed 2026 \
  --overwrite-symlinks
```

检查 manifest（避开 `head -n 80` 会截掉 `missing_rgb_examples`）：

```bash
python - <<PY
import json
from pathlib import Path

p = Path("$SAMPLE_DIR/sample_manifest.json")
d = json.loads(p.read_text())
print("counts:", json.dumps(d["counts"], ensure_ascii=False))
print("num_samples:", len(d["samples"]))
print("num_missing_rgb_examples:", len(d["missing_rgb_examples"]))
PY
```

预期：

```text
num_links = 24
train selected = 12
val selected = 12
```

## 4. 执行 RAW-Adapter-style unprocessing

公共参数含义：

```text
--backend analytic:    使用文档里的解析近似版，不使用本机旧 unprocessing。
--noise-mean-mode zero: y = l*x + zero-mean noise，不用论文公式打印版的双倍期望。
--variants normal,dark,over: 同时保存正常、低光、过曝三种 RAW-like。
--save-mosaic:         同时保存 H*W 单通道 mosaic，方便肉眼对照 CFA。
--storage float16:     debug 阶段足够。
```

`--ccm` 和 `--inverse-tone` 4 组配置共享同一份 sample dir，各自写到独立的输出目录：

```bash
OUT_BASE="$EXP_ROOT/rawadapter_unproc_image02_s12"

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

执行完后会有 4 个目录：

```text
${OUT_BASE}_baseline
${OUT_BASE}_ccm_only
${OUT_BASE}_tone_only
${OUT_BASE}_ccm_tone
```

也就是本次 s12 检查最终集中在：

```text
plans/0524_unprocessing/kitti_rawadapter_image02_s12_ablation/
  rgb_samples_image02_s12/
    sample_manifest.json
    train/.../*.jpg  -> symlink 到 /mnt/drive/kitti
    val/.../*.jpg    -> symlink 到 /mnt/drive/kitti

  rawadapter_unproc_image02_s12_baseline/
  rawadapter_unproc_image02_s12_ccm_only/
  rawadapter_unproc_image02_s12_tone_only/
  rawadapter_unproc_image02_s12_ccm_tone/
```

每个 `rawadapter_unproc_image02_s12_*` 目录内部都按 `normal/dark/over/preview/contact_sheets` 组织，便于和 contact sheet 对接。s12 的总输出（4 组 × 3 variants × 24 张 × 含 mosaic）量级在 ~250 MB 左右，本地盘够用。

## 5. 检查数值和文件结构

先看一个 config 的 manifest，确认 num_inputs / num_outputs 和 variants：

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
raw_packed 是 4 通道
dtype 是 float16
数值大体在 [0, 1]
```

如果想对 4 个 config 都过一遍 sanity check：

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

## 6. 生成对比 contact sheet

每个 config 各生成 train / val 两张 contact sheet。每一行：

```text
original RGB | normal preview | dark preview | over preview
```

KITTI 原图宽高比接近 3.31:1，把 thumb 调成 `400x120` 比脚本默认 `320x120` 更贴近原始比例，更利于肉眼判断通道对齐：

```bash
for cfg in baseline ccm_only tone_only ccm_tone; do
  python plans/0524_unprocessing/kitti/make_kitti_unprocessing_contact_sheet.py \
    --sample-manifest "$SAMPLE_DIR/sample_manifest.json" \
    --raw-output-dir "${OUT_BASE}_${cfg}" \
    --thumb-width 400 \
    --thumb-height 120
done
```

每个 config 的输出：

```text
${OUT_BASE}_${cfg}/contact_sheets/train_contact_sheet.jpg
${OUT_BASE}_${cfg}/contact_sheets/val_contact_sheet.jpg
```

判读重点：

```text
1. 单个 config 内部
   - normal preview 没有明显颜色通道错位（不应出现 R/B 偏移、网格状色块）。
   - dark preview 明显变暗但结构仍然可辨。
   - over preview 出现合理饱和，而不是整张全白。
   - train 和 val 视觉一致。

2. 4 个 config 横向对比
   - baseline vs ccm_only:  CCM 是否让色调更"相机化"，有无颜色塌缩。
   - baseline vs tone_only: inverse-tone 是否合理地把高光拉开，没有出现破坏性的反向 S 曲线。
   - baseline vs ccm_tone:  两个改动叠加后整体亮度/色彩是否仍然落在 [0, 1] 合理范围。

3. JPEG artifact
   - 局部 8x8 块状或环状纹理多半是输入 JPEG 的，不要误判为 unprocessing 缺陷。
```

## 7. 暂时先不需要：如果还想看右目 image_03

本步骤暂时先不执行。当前优先完成 `image_02` 左目的样本生成、数值检查和 contact sheet 判读。

重新准备一个独立实验目录，避免和 image_02 混在一起，然后复用 section 4 / 6 的循环：

```bash
EXP_ROOT=plans/0524_unprocessing/kitti_rawadapter_image03_s12_ablation
SAMPLE_DIR="$EXP_ROOT/rgb_samples_image03_s12"
OUT_BASE="$EXP_ROOT/rawadapter_unproc_image03_s12"

mkdir -p "$EXP_ROOT"

python plans/0524_unprocessing/kitti/prepare_kitti_unprocessing_samples.py \
  --kitti-root /mnt/drive/kitti \
  --output-dir "$SAMPLE_DIR" \
  --splits train val \
  --cameras image_03 \
  --samples-per-split 12 \
  --strategy linspace \
  --seed 2026 \
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

  python plans/0524_unprocessing/kitti/make_kitti_unprocessing_contact_sheet.py \
    --sample-manifest "$SAMPLE_DIR/sample_manifest.json" \
    --raw-output-dir "${OUT_BASE}_${cfg}" \
    --thumb-width 400 \
    --thumb-height 120
done
```

## 8. 如果样本效果通过，再跑更多样本

s100（train/val 各 100 张，4 个 config × 3 variants × 200 张 × 含 mosaic ≈ 4 ~ 5 GB），不需要 tmux：

```bash
EXP_ROOT=plans/0524_unprocessing/kitti_rawadapter_image02_s100_ablation
SAMPLE_DIR="$EXP_ROOT/rgb_samples_image02_s100"
OUT_BASE="$EXP_ROOT/rawadapter_unproc_image02_s100"

mkdir -p "$EXP_ROOT"

python plans/0524_unprocessing/kitti/prepare_kitti_unprocessing_samples.py \
  --kitti-root /mnt/drive/kitti \
  --output-dir "$SAMPLE_DIR" \
  --splits train val \
  --cameras image_02 \
  --samples-per-split 100 \
  --strategy linspace \
  --seed 2026 \
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

### 8.1 全量 train+val 的注意事项

train 38421 + val 3347 ≈ 41768 张，乘以 3 variants 再乘以 config 数，量级估算（单张 packed float16 ≈ 0.9 MB，含 mosaic 翻倍）：

```text
1 config，含 mosaic:     ≈ 230 GB
1 config，不含 mosaic:   ≈ 115 GB
4 configs，含 mosaic:    ≈ 920 GB   (不建议)
4 configs，不含 mosaic:  ≈ 460 GB   (慎重)
```

因此跑全量时建议：

```text
1. 先用 s12 / s100 在 4 个 config 上做对比，挑出最满意的一组（多半是 baseline 或 ccm_tone）。
2. 全量阶段只跑那一个 config，并去掉 --save-mosaic，把 --save-preview 也限制到抽样输出。
3. 用 tmux，session 名 + 日志路径都要清晰。建议命名：
   tmux new -s kitti_unproc_full_image02_$(date +%m%d_%H%M)
   日志写到 plans/0524_unprocessing/logs/kitti_unproc_full_image02_$(date +%m%d_%H%M).log
4. 跑之前 df -h 确认输出目录所在盘剩余空间至少留 150 GB 余量。
5. 注意：当前 sample 目录都放在 plans/ 下，全量输出最好放到数据盘而不是工程目录，
   避免 git 状态被几十万个 .npz 污染（.gitignore 也要相应加上）。
```
