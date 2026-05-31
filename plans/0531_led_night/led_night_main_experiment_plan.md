# LED-night 主实验后续执行计划

日期：2026-05-31
目标：在 **LED Nighttime Synthetic Drive Dataset** 上做单数据集 in-domain 夜间实验，验证当前 `C2/D1 + RAW-like x3 incremental correction` 路线是否能在真夜间外观分布下继续成立。

---

## 0. 当前决策摘要

### 0.1 主数据集

使用 **LED Nighttime Synthetic Drive Dataset**，不再使用 VKITTI day→night 作为主夜间实验。

理由：LED 本身就是夜间 synthetic driving RGB-D，避免 VKITTI 中同时叠加：

```text
day RGB -> night RGB
night RGB -> synthetic RAW-like
```

这样 LED 的数据流只剩：

```text
true nighttime LDR RGB -> inverse ISP / unprocessing -> synthetic RAW-like
```

claim 仍然要写成：

```text
night RGB-D + synthetic RAW-like representation
```

不要写成 real RAW。

### 0.2 主照明模式

第一阶段只选 **HB / High Beam**。

不先选 Pattern 的原因：

1. HB 更接近普通夜间车灯照明，少一个 structured-light / checkerboard pattern confound。
2. Pattern 本身会把显式几何纹理投射到画面中，RGB 和 RAW 都会看到这种 pattern；如果直接用 Pattern 做主实验，提升可能被解释成“主动结构光 cue”，而不是 RAW-like residual cue。
3. Pattern 很适合后续作为第二阶段：测试 active illumination 是否进一步放大 RAW-like / x3 的局部修正优势。

第一阶段命名：

```text
LED-HB-main
```

第二阶段再做：

```text
LED-Pattern-followup
```

### 0.3 训练 / 验证范围

只做 in-domain eval：

```text
train: LED / HB / train
val:   LED / HB / val
test:  暂时不使用
```

第一阶段主 split：

```text
train: all official HB/train frames, ~14,997 frames
val:    1,000 frames from official HB/val, stride=5 within hamburg
seed:   42, only for val phase offset / tie-breaking and future subsets
```

采样原则：

- train 使用 `china / herrenberg / ottosuhrallee` 三个官方 train map 的全量帧。
- val 从 `hamburg` 官方 val map 中采样。
- train 如果官方 HB/train 只有约 14,997 帧，则直接用全量，不再抽 12,000；这样不人为丢掉训练数据，也避免把 train split 做成一个 dense random subset。
- val 每个 map 内先按 frame id / timestamp 排序，再做固定间隔抽样；不要对连续序列直接 random sample。
- val 从 hamburg 每约 5 帧取 1 帧，得到约 1,000 帧，避免验证集被几乎重复的相邻帧主导。
- 不要从 test / wuppertal 取样。
- 不要 random frame split 混合 train/val。

### 0.4 low-light stress

第一阶段不额外加 low-light stress。

也就是说：

```text
raw_adapter_fixed_light_scale = 1.0
randomize_unprocessing = false
no extra exposure reduction
no extra Poisson-Gaussian noise
```

因为 LED 图像本身已经是夜间渲染。第一阶段要先验证“真夜间 LDR RGB-D + synthetic RAW-like”是否有用，不要再额外引入 exposure/noise 变量。

---

## 1. 数据集空间估算

### 1.1 官方全量大小

官方 GitHub README 写明：完整 LED 数据集大小约为：

```text
483 Go ≈ 483 GB
```

数据按如下维度拆成多个 zip：

```text
illumination: Pattern / HB
split: train / val / test
annotation type: ldr_color / distance_to_image_plane / normals / semantic_segmentation / ...
```

官方项目页说明 LED 总共有：

```text
49,990 images
24,995 Pattern images
24,995 HB images
```

因此粗略平均：

```text
483 GB / 49,990 frames ≈ 9.66 MB / frame
```

这个平均值包含所有 annotation 类型，不是只包含 RGB + depth。

### 1.2 按 illumination / split 估算

官方结构每个 illumination 下有 5 个 map：

```text
train: china, herrenberg, ottosuhrallee
val:   hamburg
test:  wuppertal
```

因为每个 illumination 总数是 24,995，且 24,995 = 4,999 × 5，所以第一阶段可以按每个 map 约 4,999 frames 估算。

| 范围 | 帧数估计 | 全 annotation 空间估计 |
|---|---:|---:|
| Full LED, HB + Pattern, train + val + test | 49,990 | 483 GB |
| HB only, all splits | 24,995 | 241.5 GB |
| HB train only | 14,997 | 144.9 GB |
| HB val only | 4,999 | 48.3 GB |
| HB train + val | 19,996 | 193.2 GB |
| HB test only | 4,999 | 48.3 GB |

第一阶段不下载 Pattern、不下载 test，因此如果下载 **HB train + val 全 annotation**，预计约：

```text
193 GB
```

### 1.3 只下载必要 annotation 的估算

主实验实际只需要：

```text
ldr_color
距深标签: distance_to_image_plane
camera_params
transforms
```

可选但第一阶段不必下载：

```text
normals
semantic_segmentation
instance_segmentation
object detection labels
occlusion
dynamics
```

因为官方 zips 按 annotation type 拆分，建议只下载必要 annotation。这样实际空间应低于 193 GB。

保守估算。这里按当前 VKITTI N-series 实现修正：D0/D1/raw4/x3 默认都是训练/验证时在线生成，不作为第一阶段必需的持久 cache 计入磁盘预算。

| 下载内容 | 估算空间 |
|---|---:|
| HB train+val 全 annotation | 约 193 GB |
| HB train+val 只含 ldr_color + distance_to_image_plane + camera/transforms | 约 80–140 GB |
| interval-aware 抽样后的软链接/manifest 本身 | < 1 GB |
| D0 / D1 / raw4 / x3 持久 cache | 第一阶段默认不生成，0 GB |
| checkpoints/logs/panels | 10–30 GB |

当前 VKITTI 代码路径对应关系：

- `VKITTI2Raw.__getitem__()` 从 RGB/depth filelist 读图，并在线 unprocess 出 `raw`。
- `C2FrozenIncrementalResidualDAV2.forward()` 每个 batch 在线调用 frozen C2，得到 `D0` / `D1_norm`。
- 同一个 forward 内通过 `RamCore3.forward_with_features()` 从 `raw4` 在线得到 `x3` / `ffm_mid`。
- 仓库里有旧的 `cache_vkitti2_pseudoraw.py` / `CachedVKITTI2Raw`，但当前 N-series formal 脚本没有使用这些 cache 入口；LED-HB 第一阶段应沿用在线生成语义。

建议磁盘预算：

```text
最低可运行预算: 200 GB
推荐预算:       300 GB
保守预算:       500 GB
```

如果要同时保留 zip 和解压目录：

```text
推荐准备 500 GB 以上给 HB train+val
```

如果下载完整 LED 并同时保留 zip + extracted：

```text
建议准备 1 TB 以上
```

---

## 2. linear / HDR / renderer buffer 检查结论

### 2.1 官方公开目录

官方 README 中每个 map 下列出的 annotation 目录是：

```text
bounding_box_2d_loose
bounding_box_2d_tight
bounding_box_3d
camera_params
distance_to_camera
distance_to_image_plane
dynamics
instance_segmentation
ldr_color
normals
occlusion
semantic_segmentation
transforms
```

这里没有明确的：

```text
linear_color
hdr_color
radiance
raw
sensor
albedo
exposure buffer
renderer HDR buffer
```

### 2.2 官方 loader 使用方式

官方 `DriveSimDepthDataset` 使用：

```text
input_dir = ldr_color
label_dir = distance_to_image_plane
```

depth 读取方式是 OpenEXR 单通道 `Y` float。

这说明当前公开 release 至少在官方 README / dataset loader 层面，是：

```text
input: LDR RGB image
label: EXR depth map
```

不是 linear/HDR/RGBE/sensor RAW release。

### 2.3 对我们实验的影响

第一阶段计划应明确写：

```text
LED does not appear to provide public linear/HDR/sensor buffers in the documented release.
We therefore generate synthetic RAW-like inputs from LED ldr_color using the same analytic inverse-ISP / RAW-Adapter-style unprocessing pipeline.
```

中文论文表述：

```text
由于 LED 官方公开 release 中未发现 linear/HDR/sensor RAW buffer，我们从其夜间 LDR RGB 图像反推 synthetic RAW-like packed Bayer 表示。该实验验证的是 true-night RGB-D 场景下的 inverse-ISP RAW-like residual cue，而不是真实 RAW sensor cue。
```

如果下载后发现 Google Drive 中存在 README 没列出的 hidden HDR / linear buffer，则优先改用：

```text
linear/HDR buffer -> packed RAW-like
```

但在当前计划里，默认没有。

---

## 3. 数据准备流程

### 3.1 下载范围

第一阶段下载：

```text
LED/HB/train/ldr_color
LED/HB/train/distance_to_image_plane
LED/HB/train/camera_params
LED/HB/train/transforms

LED/HB/val/ldr_color
LED/HB/val/distance_to_image_plane
LED/HB/val/camera_params
LED/HB/val/transforms
```

暂时不下载：

```text
Pattern/*
HB/test/*
object detection labels
semantic labels
instance labels
normals
occlusion
dynamics
```

如果下载脚本不支持细粒度 annotation zip 选择，则第二优先方案是下载：

```text
HB/train all annotations
HB/val all annotations
```

不要下载：

```text
HB/test
Pattern/train
Pattern/val
Pattern/test
```

### 3.2 本地目录建议

```text
/data/LED_NSDD/
  HB/
    train/
      china/
        ldr_color/
        distance_to_image_plane/
        camera_params/
        transforms/
      herrenberg/
      ottosuhrallee/
    val/
      hamburg/
        ldr_color/
        distance_to_image_plane/
        camera_params/
        transforms/
  filelists/
    hb_train_all.txt
    hb_val_stride5_n1000_seed42.txt
  exp/
```

### 3.3 文件计数检查

解压后先跑：

```bash
find /data/LED_NSDD/HB/train -path '*/ldr_color/*' -type f | wc -l
find /data/LED_NSDD/HB/train -path '*/distance_to_image_plane/*' -type f | wc -l
find /data/LED_NSDD/HB/val   -path '*/ldr_color/*' -type f | wc -l
find /data/LED_NSDD/HB/val   -path '*/distance_to_image_plane/*' -type f | wc -l
```

预期约：

```text
HB/train: ~14,997 RGB-depth pairs
HB/val:   ~4,999 RGB-depth pairs
```

如果数量不一致，使用实际数量，不要强行假设。

### 3.4 构建 interval-aware split

建议实现：

```bash
python foundation/tools/build_led_filelists.py \
  --root /data/LED_NSDD \
  --illum HB \
  --train_maps china,herrenberg,ottosuhrallee \
  --val_maps hamburg \
  --train_mode all \
  --val_stride 5 \
  --val_n 1000 \
  --seed 42 \
  --out_dir /data/LED_NSDD/filelists
```

split 构建规则：

- 每个 map 独立排序；不能把不同 map 合并后随机抽。
- frame id 优先从文件名解析；如果文件名不是纯 frame id，则从 metadata / transforms 中取稳定时间戳排序。
- train 使用 official HB/train 全量：`china / herrenberg / ottosuhrallee` 约 14,997 帧。
- val 使用 stride=5 时，从 hamburg 约 4,999 帧中取约 1,000 帧。
- 输出 filelist 保持排序后的稳定顺序；训练时 DataLoader 再 shuffle。
- 如果后续为了速度做 train subset，必须输出独立文件名，例如 `hb_train_stride2_debug_seed42.txt` 或 `hb_train_dense12000_seed42.txt`，并在实验表中明确标注 subset / dense split，不能和全量主 split 混用。

输出格式每行：

```text
relative_rgb_path relative_depth_path relative_camera_params_path relative_transforms_path
```

例如：

```text
HB/train/china/ldr_color/000001.png HB/train/china/distance_to_image_plane/000001.exr HB/train/china/camera_params/000001.json HB/train/china/transforms/000001.json
```

具体扩展名以实际下载结果为准。

---

## 4. 输入、RAW-like 生成与公平性

### 4.1 主路径输入

所有方法的 DAV2 主路径都使用：

```text
LED HB ldr_color
```

即：

```text
D0 = frozen DAV2(ldr_color)
```

不能让 RAW 方法的 DAV2 看不同版本图像。

### 4.2 RGB control 输入

RGB incremental control 使用：

```text
feature = LED HB ldr_color
```

即：

```text
N3-LED-HB: D1 + RGB incremental branch
```

### 4.3 RAW/x3 输入

RAW/x3 方法使用：

```text
raw4 = unprocess(LED HB ldr_color)
x3 = RamCore3(raw4).x3
```

第一阶段沿用 clean RA0：

```text
unprocessing_method=raw_adapter_style
raw_adapter_backend=analytic
raw_adapter_ccm=identity
raw_adapter_inverse_tone=global_0p15
raw_adapter_rgb_transfer=srgb_piecewise
raw_adapter_cfa_pattern=RGGB
raw_adapter_packed_channel_order=R_Gr_Gb_B
randomize_unprocessing=false
raw_adapter_variant_policy=normal
raw_adapter_fixed_red_gain=2.15
raw_adapter_fixed_blue_gain=1.7
raw_adapter_fixed_light_scale=1.0
```

不要额外变暗，不要额外加噪声。

### 4.4 公平性原则

公平比较只允许 incremental branch 输入不同：

```text
D0 path: same LED ldr_color
D1 path: same C2-HB calibrator
GT:      same LED distance_to_image_plane
```

对比：

```text
N3: branch input = ldr_color
N2/N7: branch input = synthetic raw4 -> x3
N5: branch input = D1 only
```

---

## 5. Depth label 与 evaluation protocol

### 5.1 深度标签

使用：

```text
distance_to_image_plane
```

理由：官方 depth loader 默认把这个作为 label。

读取：

```text
OpenEXR, channel Y, float32
```

### 5.2 valid mask

第一阶段建议：

```text
min_depth = 1.0
max_depth = 200.0
invalid if depth <= 1.0 or depth > 200.0 or NaN/Inf
```

原因：官方 loader 默认 `max_depth=200`，并把超过最大深度的值替换为 placeholder。

同时建议额外报告一个 secondary protocol：

```text
max_depth = 80.0
```

用于和 VKITTI/KITTI 风格结果比较，但主结果用 `[1, 200]`。

### 5.3 DAV2 relative protocol

第一阶段沿用当前 relative/DAV2 protocol：

```text
per-image affine disparity alignment
clip predicted depth to [1.0, 200.0]
compute AbsRel, RMSE, SILog, delta1
```

报告时要同时给：

```text
D0
D1
Final
Final - D1
```

不要只报 Final。

---

## 6. 模型与最小实验矩阵

### 6.1 第一阶段固定设置

```text
dataset = LED-HB
train_split = hb_train_all, ~14997 frames
val_split = hb_val_stride5_n1000_seed42, 1000 frames
test = not used
encoder = DAV2-S / vits
input = ldr_color resized/padded to DAV2-compatible resolution
unprocessing = clean RA0, light_scale=1.0
low-light stress = off
```

### 6.2 推荐分辨率

先读取第一张图，确认 LED 原始分辨率。

建议实现两种模式：

```text
mode A: aspect-preserving resize + center pad to multiples of 14
mode B: fixed resize to 350x630 if aspect ratio is close to 16:9
```

第一阶段优先：

```text
aspect-preserving resize, long side <= 700, H/W rounded to multiple of 14
```

如果代码需要固定 H/W，先用：

```text
350 x 630
```

显存不够再降到：

```text
280 x 504
```

### 6.3 最小实验矩阵

| ID | Method | 输入 | 是否训练 | 目的 |
|---|---|---|---|---|
| L0 | Frozen DAV2-S D0 | ldr_color | no | 夜间 RGB foundation baseline |
| L1 | C2-LED-HB | D0 only | yes | D0-only calibrator，必须作为强 baseline |
| L2 | N5-LED-HB | D1 only | yes | extra head / regularization / D1-only control |
| L3 | N3-LED-HB | ldr_color + D1 | yes | matched RGB incremental control |
| L4 | N2-LED-HB | x3 + D1 gate, delta feature-only | yes | 更干净的 RAW/x3 proof-of-concept |
| L5 | N7-LED-HB | x3 + stopgrad D1 in delta | yes | 当前最强候选主方法 |
| L6 | x3 shuffle eval | load L4 or L5 | eval only | 判断模型是否真的使用 x3；可选但强烈建议 |

如果只能先跑 4 个：

```text
L0, L1, L3, L5
```

但正式结果至少需要：

```text
L0, L1, L2, L3, L4, L5
```

### 6.4 训练顺序

```text
Step 1: 实现并 smoke LED-HB online loader，确认 RGB/depth/raw4 几何对齐
Step 2: 跑 L0 frozen DAV2-S D0 eval，作为夜间 RGB foundation baseline
Step 3: 训练 C2-LED-HB，得到 D1 calibrator
Step 4: 训练 N5 D1-only
Step 5: 训练 N3 RGB incremental
Step 6: 训练 N2 x3 feature-only
Step 7: 训练 N7 x3 + stopgrad D1 delta
Step 8: 对 N2/N7 做 x3 shuffle / zero / mean eval
```

---

## 7. 具体 run 配置

### 7.1 L1: C2-LED-HB

```text
method_id=C2_LED_HB
front_end=dav2_rgb_frozen
input_domain=rgb
model_input_tensor=image
residual_feature_source=d0
trainable=ResidualGateHead only
```

loss 沿用 C2：

```text
L = L_depth
  + 0.5 * L_grad
  + 0.1 * L_keep
  + 0.01 * L_res
  + 0.005 * L_gate
  + 0.05 * L_gate_sup
```

### 7.2 L2: N5-LED-HB

```text
method_id=N5_LED_HB
front_end=c2_frozen_d1_incremental
incremental_feature_source=d1
delta_condition=d1_only
gate_condition=d1_only
lambda_lp=0.5
q_good=0.3
lambda_lowfreq_loss=0.0
```

### 7.3 L3: N3-LED-HB

```text
method_id=N3_LED_HB
front_end=c2_frozen_rgb_incremental
incremental_feature_source=rgb
delta_condition=feature_only
gate_condition=feature_d1
lambda_lp=0.5
q_good=0.3
lambda_lowfreq_loss=0.0
```

### 7.4 L4: N2-LED-HB

```text
method_id=N2_LED_HB
front_end=c2_frozen_raw_ram_incremental
input_domain=raw4
raw_storage_format=synthetic_packed_bayer_4ch_halfres
incremental_feature_source=x3
delta_condition=feature_only
gate_condition=feature_d1
lambda_lp=0.5 或 0.8
q_good=0.3
lambda_lowfreq_loss=0.0
```

建议第一轮：

```text
lambda_lp=0.8, q_good=0.3
```

因为 VKITTI 上这个 N2 设置 boundary / overall 较强。

如果只想更稳：

```text
lambda_lp=0.5, q_good=0.3
```

### 7.5 L5: N7-LED-HB

```text
method_id=N7_LED_HB
front_end=c2_frozen_raw_ram_incremental
input_domain=raw4
raw_storage_format=synthetic_packed_bayer_4ch_halfres
incremental_feature_source=x3
delta_condition=feature_d1_stopgrad
gate_condition=feature_d1
lambda_lp=0.5
q_good=0.3
lambda_lowfreq_loss=0.0
```

这是第一阶段主方法。

---

## 8. Metrics 与可视化

### 8.1 Overall metrics

主表：

```text
AbsRel
RMSE
SILog
delta1
final-D1 AbsRel
```

需要报告两套 depth range：

```text
main:      [1, 200]
secondary: [1, 80]
```

### 8.2 Region metrics

LED-HB 第一阶段建议：

```text
boundary: GT depth gradient top 10%
D0 high-error: abs(D0_norm - y_norm) top 20%
D1 high-error: abs(D1_norm - y_norm) top 20%
near: 1-20m
mid: 20-50m
far50: >50m
far100: >100m
clean_dark_q20: ldr_color luma lowest 20%
saturated: max RGB > 0.95
```

注意：LED 全部是夜间，因此普通 `luma < 0.15` 可能覆盖太多像素。建议用 quantile mask：

```text
dark_q20 = luma <= percentile(luma, 20)
```

### 8.3 可视化 panel

每个方法至少保存：

```text
RGB / ldr_color
raw4 visualization
x3 visualization
GT depth
D0 depth
D1 depth
Final depth
D0 error
D1 error
Final error
Final - D1 improvement map
gate map
gate * delta map
```

特别要增加：

```text
N7 vs N3 RGB improvement-over-D1
N7 vs N5 D1-only improvement-over-D1
N7 vs N2 improvement-over-D1
```

---

## 9. 成功标准

### 9.1 强成功

满足：

```text
N7-HB overall AbsRel < C2-HB by >= 0.002
N7-HB boundary AbsRel < C2-HB by >= 0.015
N7-HB saturated/dark_q20 region 优于 N3-RGB-HB
N7-HB 优于 N5-D1-only
x3 shuffle/zero eval 明显退化
```

结论：

```text
RAW-like x3 provides useful incremental correction in true nighttime synthetic driving.
```

### 9.2 中等成功

满足：

```text
N7-HB overall ≈ C2-HB 或略优
N7-HB boundary 明显优于 C2-HB / N5-HB
N7-HB 不稳定超过 N3-RGB-HB
```

结论：

```text
RAW-like x3 mainly provides region-specific boundary/detail benefit, not broad overall improvement.
```

### 9.3 失败

出现：

```text
N7-HB 不优于 N5-D1-only
N7-HB 不优于 N3-RGB-HB
x3 shuffle/zero 不退化
```

结论：

```text
在 LED-HB + inverse-ISP synthetic RAW-like 设置下，x3 没有显示出稳定边际贡献。
```

这时再考虑：

```text
Pattern split
更物理的 unprocessing
使用 normals/semantic 做辅助 mask
换真实夜间验证集
```

---

## 10. Pattern split 后续策略

第一阶段不做 Pattern。

如果 HB 结果成立，再做：

```text
LED-Pattern-main-ablation
```

最小矩阵只需要：

```text
D0-Pattern
C2-Pattern
N3-RGB-Pattern
N5-D1-Pattern
N7-x3-Pattern
x3 shuffle eval
```

Pattern 的论文表述应与 HB 分开：

```text
HB tests standard high-beam nighttime driving.
Pattern tests active structured-light illumination.
```

不要把 Pattern 结果直接混入 HB 主表求平均，否则解释会混乱。

---

## 11. 实现 checklist

### 11.1 数据

- [ ] 下载并解压 HB/train 必要 annotation。
- [ ] 下载并解压 HB/val 必要 annotation。
- [ ] 检查 `ldr_color` 与 `distance_to_image_plane` 数量一致。
- [ ] 检查 EXR depth 是否能读取。
- [ ] 统计 depth min/max/invalid ratio。
- [ ] 构建 `hb_train_all.txt`，并检查包含 `china / herrenberg / ottosuhrallee` 官方 HB/train 全量约 14,997 帧。
- [ ] 构建 `hb_val_stride5_n1000_seed42.txt`，并检查 hamburg 内相邻入选 frame id gap >= 5。

### 11.2 模型输入

- [ ] 确认 LED 原始 RGB 分辨率。
- [ ] 确认 resize/pad 后 H/W 是 14 的倍数。
- [ ] 确认 RGB 和 depth resize 后对齐。
- [ ] 确认 raw4 与 RGB 对齐。
- [ ] 确认 x3 可视化没有 collapse。

### 11.3 在线中间量 / 可选缓存

- [ ] 确认 LED-HB loader 沿用 VKITTI N-series 语义：raw4 在线 unprocess 生成。
- [ ] 确认 L1/L2/L3/L4/L5 的 D0/D1 均由模型 forward 在线生成，不落盘。
- [ ] 确认 x3/ffm_mid 由 RamCore3 在线生成，不落盘。
- [ ] 第一阶段不生成 D0/D1/raw4/x3 持久 cache。
- [ ] 只有当 profiling 证明在线 raw4 生成成为主要瓶颈时，再新增显式 cache 方案；该方案必须单独记录 raw_storage_format、unprocessing config hash、几何策略和 train/val 对齐校验。

### 11.4 实验

- [ ] L0 D0 eval。
- [ ] L1 C2-HB train/eval。
- [ ] L2 N5-HB train/eval。
- [ ] L3 N3-RGB-HB train/eval。
- [ ] L4 N2-x3-HB train/eval。
- [ ] L5 N7-x3-HB train/eval。
- [ ] L6 x3 shuffle / zero / mean eval。

### 11.5 报告

- [ ] overall table。
- [ ] region metrics table。
- [ ] final-D1 improvement table。
- [ ] panel 可视化。
- [ ] x3 shuffle 诊断。
- [ ] 结论按“overall / boundary / dark / saturated / high-error”分开写。

---

## 12. 论文表述边界

可以写：

```text
We evaluate the proposed D1-conditioned RAW-like incremental correction on LED Nighttime Synthetic Drive Dataset, a true nighttime synthetic driving RGB-D dataset.
```

可以写：

```text
Since the documented LED release provides LDR color images and rendered depth but no public linear/HDR/sensor RAW buffers, we generate inverse-ISP synthetic RAW-like packed Bayer inputs from LED LDR images.
```

不要写：

```text
LED provides real RAW.
```

不要写：

```text
RAW contains additional sensor information beyond LED RGB.
```

更稳的 claim：

```text
Synthetic RAW-like representations derived from true nighttime RGB can serve as residual cues for refining frozen RGB depth foundation model predictions, particularly in boundary and difficult nighttime regions.
```

---

## 13. 最终执行建议

第一阶段直接执行：

```text
illumination = HB
train = all official HB/train, ~14,997
val = interval-aware stride5 from official val, 1k
test = unused
low_light_stress = off
unprocessing = clean RA0, light_scale=1.0
main model = N7-LED-HB
core controls = D0, C2, N5, N3, N2
```

第一阶段完成后再决定是否做：

```text
Pattern split
full HB train/val
low-light stress
external real night validation
```
