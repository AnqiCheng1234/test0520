# M2 / C1 / C2 residual 实验分析

日期：2026-05-25

本文记录 `/home/caq/6666_raw/dav2_raw_0522` 当前 M2、C1、C2 三组 residual 实验的定义、网络结构差异、训练参数差异、当前结果、原因分析和下一步建议。

> **当前问题 / 注意**
>
> 本文里的 M2 RAW4 路径当前看起来仍然使用工程遗留的在线 unprocessing：`foundation/engine/transforms/unprocessing.py`，formal run 中对应 `sensor_linear_dual` preset。这个路径更接近旧的 Brooks/2019-style unprocessing，并通过 ETH3D / RobotCar 等 sensor preset 参数做模拟；它不是我希望验证的 `plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py` 这套 RAW-Adapter-style / InvISP unprocessing。
>
> 因此，下面所有 M2 结果应先标记为“旧 online Brooks-style preset unprocessing + RAM residual”的结果，不能直接当作 `plans/0524_unprocessing` 的 InvISP 方法结论。如果目标是验证 0524 的 InvISP 路线，需要把 `unprocessing_method=raw_adapter_style_analytic`、`cfa_pattern`、`packed_channel_order`、`inverse_tone`、`ccm_mode`、`exposure_variant` 等实验语义参数显式接入训练/评估配置，并重新 launch formal runs。

## 1. 速查结论

三组实验的共同高层形式都是：

```text
frozen DAv2-S -> D0 -> D0_norm -> residual path -> pred = D0_norm + gate * delta
```

DAv2-S 主干冻结，不参与训练。训练的核心是 residual correction path。

当前结论：

- C2 是这一轮里最强、也最稳定的 control baseline。
- M2 明显优于 frozen DAv2-S 的 D0，但没有超过 C2。
- C1 也优于 D0，但泛化不如 C2，且 last epoch 明显回退。
- 当前 M2 不能作为“RAW/RAM cue 相比 D0-only post-processing 有额外收益”的强正证据。
- 结合当前 panel 可视化看，三者 residual 的大尺度形状高度相似，主要都在修 D0 几何；M2 的额外 RAW/RAM feature 更像带来局部 photometric/纹理化变化，而不是稳定的新几何信息。

实际解释：

- residual formulation 本身是有效的。
- 在当前 per-image affine disparity eval 协议下，D0-only residual control 是很强的 baseline。
- 后续如果想证明 RAW-like cue 有价值，M 系列必须超过 C2，而不是只超过 D0。

## 2. 实验定义

本文对照的 formal runs：

| Method | Run | Stage | Input domain | Residual source |
|---|---|---|---|---|
| C1 | `0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | `vkitti_residual_control` | RGB | `rgb` |
| C2 | `0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | `vkitti_residual_control` | RGB | `d0` |
| M2 | `0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | `vkitti_raw_residual` | RAW4 | `ffm_mid` |

数据和评估协议：

- VKITTI validation：Scene20 holdout，1000 samples。
- KITTI validation：652 samples，halfres canonical eval，用作 cross-domain sanity check；这不是 KITTI public benchmark setting。
- VKITTI eval protocol：`per_image_affine_disp_depth_anything_v2`。
- KITTI eval protocol：
  - C1/C2：`halfres_rgb_canonical_even_pad_crop_affine_disp`
  - M2：`halfres_raw_canonical_even_pad_crop_affine_disp`
- 输入尺寸：`187x621`。
- frozen D0 baseline 都来自同一个 frozen DAv2-S checkpoint。

## 3. 网络结构

### 3.1 共同 residual 形式

三组实验都遵循：

```text
D0 = frozen_DAv2(image_rgb_norm)
D0_norm = robust_normalize(D0)
delta, gate = residual_head(head_input)
pred = D0_norm + gate * delta
```

其中 `delta` 被 `residual_alpha` 限制幅度：

```text
delta = residual_alpha * tanh(delta_head(x))
gate  = sigmoid(gate_head(x))
```

当前 formal runs 中：

- `residual_alpha = 0.5`
- `d0_sign = 1`

### 3.2 Residual head 骨架

M2、C1、C2 的 residual head 骨架是一样的。head 内部唯一结构差别是第一层卷积的输入通道数。

```text
head_input
  -> Conv2d(in_ch -> 64, 3x3) + GroupNorm + GELU
  -> ResidualBlock(64)
  -> DownBlock(64 -> 128)
  -> DownBlock(128 -> 256)
  -> UpBlock(256 + 128 -> 128)
  -> UpBlock(128 + 64 -> 64)
  -> delta_head: Conv2d(64 -> 32, 3x3) + GELU + Conv2d(32 -> 1, 1x1)
  -> gate_head:  Conv2d(64 -> 32, 3x3) + GELU + Conv2d(32 -> 1, 1x1)
```

初始化也相同：

- `delta_head` 最后一层 conv 初始化为 0。
- `gate_head` 最后一层 conv weight 初始化为 0。
- `gate_head` 最后一层 bias 初始化为 `-4.0`。
- 因此模型初始状态接近 `D0_norm`，gate 很小。

代码位置：

- C1/C2 residual head：`foundation/engine/models/dav2_residual_control.py`
- M2 residual head：`foundation/engine/models/raw_residual_dav2.py`

### 3.3 Head input 差异

| Method | Head input | Residual head 输入通道 | 语义 |
|---|---|---:|---|
| C2 | `D0_norm.unsqueeze(1)` | 1 | D0-only post-processing |
| C1 | `concat(D0_norm, image_rgb_norm)` | 4 | RGB residual refinement |
| M2 | `concat(D0_norm, ffm_mid)` | 65 | RAW/RAM middle-feature residual |

解释：

- C2 测试 residual branch 是否只靠 D0 自身就能改善。
- C1 测试普通 RGB appearance 是否能在 D0 之外继续提供收益。
- M2 测试 RAW-like / RAM feature 是否能在 D0 之外继续提供收益。

### 3.4 M2 的 RAW/RAM 路径

M2 在 residual head 前面多了一个可训练 RAW/RAM 路径：

```text
halfres RGB
  -> online unprocessing: sensor_linear_dual
  -> synthetic packed Bayer RAW4 [R, Gr, Gb, B]
  -> packed_bayer_to_base_rgb: [R, (Gr + Gb) / 2, B]
  -> RamCore3
  -> ffm_mid, 64 channels
  -> concat(D0_norm, ffm_mid)
  -> residual head
```

RamCore3 结构：

```text
input RGB-like 3ch
  -> RPEncoder
  -> four branches: WB, CCM, gamma, brightness
  -> concatenate branch outputs: 12 channels
  -> FFM3: 12 -> 16 -> 64 -> 16 -> 3
  -> ffm_mid 是 FFM 中间的 64-channel feature
```

重要细节：当前实现中，M2 的 RamCore3 是可训练的，不是 frozen feature extractor。

## 4. 可训练参数和训练设置

三组实验的 DAv2-S frozen 参数数相同：

```text
frozen DAv2-S params = 24,785,089
```

实际可训练参数：

| Method | Frozen params | Trainable params | Trainable modules |
|---|---:|---:|---|
| C2 | 24,785,089 | 2,881,858 | residual head |
| C1 | 24,785,089 | 2,883,586 | residual head |
| M2 | 24,785,089 | 3,059,625 | RamCore3 + residual head |

C1 和 C2 的参数差异很小，主要来自 residual head 第一层卷积：

```text
C2: Conv2d(1 -> 64)
C1: Conv2d(4 -> 64)
M2: Conv2d(65 -> 64)
```

M2 参数更多，因为它包含 RamCore3，并且 residual head 第一层输入通道更宽。

三组 formal run 主要训练超参一致：

| Parameter | Value |
|---|---|
| encoder | `vits` |
| epochs | 20 |
| batch size | 8 |
| accumulation steps | 1 |
| lr | `1e-4` |
| weight decay | `1e-4` |
| hflip prob | `0.5` |
| AMP | enabled |
| AMP dtype | `bf16` |
| input size | `187x621` |
| residual alpha | `0.5` |
| d0 sign | `1` |

Loss：

```text
loss = L_depth
     + 0.5   * L_grad
     + 0.1   * L_keep
     + 0.01  * L_res
     + 0.005 * L_gate
     + 0.05  * L_gate_sup
```

其中 gate supervision 使用 D0-vs-target normalized error 生成 high-error mask。这会让当前任务天然更偏向 D0-based post-processing。

## 5. 当前结果

用户提供的结果表：

| Experiment | Split | Best abs_rel | Last abs_rel | Best d1 | Last d1 |
|---|---|---:|---:|---:|---:|
| `c1_rgb_0525_0203` | KITTI | 0.09843 @06 | 0.09989 | 0.89620 @17 | 0.89371 |
| `c2_d0only_0525_0203` | KITTI | 0.09503 @06 | 0.09597 | 0.90004 @10 | 0.89690 |
| `m2_ffm_mid_0525_0204` | KITTI | 0.09558 @12 | 0.09660 | 0.90058 @15 | 0.89719 |
| `c1_rgb_0525_0203` | VKITTI | 0.12574 @03 | 0.13702 | 0.85146 @03 | 0.83280 |
| `c2_d0only_0525_0203` | VKITTI | 0.12100 @11 | 0.12134 | 0.86215 @17 | 0.85531 |
| `m2_ffm_mid_0525_0204` | VKITTI | 0.12621 @08 | 0.13076 | 0.85406 @09 | 0.84346 |

按 best abs_rel 排序：

```text
VKITTI: C2 (0.12100) < C1 (0.12574) < M2 (0.12621)
KITTI:  C2 (0.09503) < M2 (0.09558) < C1 (0.09843)
```

从 best 到 last 的稳定性：

| Method | VKITTI best -> last | KITTI best -> last | 稳定性 |
|---|---:|---:|---|
| C2 | 0.12100 -> 0.12134 | 0.09503 -> 0.09597 | 很稳定 |
| C1 | 0.12574 -> 0.13702 | 0.09843 -> 0.09989 | VKITTI 明显回退 |
| M2 | 0.12621 -> 0.13076 | 0.09558 -> 0.09660 | VKITTI 中等回退 |

VKITTI best abs_rel epoch 的 region metrics：

| Method | Epoch | Boundary | DAv2 high-error | Far50 | Dark | Saturated | Mean gate | Mean abs delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 3 | 0.5169 | 0.3279 | 0.9089 | 0.1160 | 0.1747 | 0.3213 | 0.4671 |
| C2 | 11 | 0.4748 | 0.2703 | 0.5908 | 0.1143 | 0.1382 | 0.3358 | 0.4537 |
| M2 | 8 | 0.5723 | 0.3366 | 0.6057 | 0.1214 | 0.1347 | 0.3122 | 0.4665 |

观察：

- C2 在 VKITTI overall、boundary、DAv2 high-error、far50、dark 上最好。
- M2 在 saturated region 上略好。
- M2 的 KITTI best d1 略高于 C2，但 abs_rel 仍略差。

### 5.1 当前 panel 可视化观察

当前已有 panel 输出位置：

- C1 VKITTI：`finetune_stf/exp/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/vkitti_best_val_epoch03_3x4_condition_panels`
- C2 VKITTI：`finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/vkitti_best_val_epoch11_3x4_condition_panels`
- M2 VKITTI：`finetune_stf/exp/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/vkitti_best_val_epoch08_3x4_condition_panels`
- 对应 KITTI panel 在各 run 的 `kitti_best_val_epoch*_3x4_panels` 目录。

Panel 布局一致：

```text
RGB input | input preview | RGB distribution | model/raw input distribution
DAV2-S depth | Ours | Residual gate*delta | Gate
DAV2 error | Ours error | Err improve +green | GT depth
```

可视化上的主要现象：

- C1、C2、M2 的 `Residual gate*delta` 大尺度结构非常接近，通常都是沿着天空/远处、车辆、道路坡面、树墙边界等 D0 已有几何结构做推拉。
- C2 的 gate 和 residual 更像 D0-driven 的几何校准，形态较平滑，泛化稳定。
- C1 和 M2 会额外出现更明显的 appearance/photometric 纹理。某些样本这些纹理能帮助 error 下降，但也更容易在 Scene20 holdout 上形成误修。
- M2 的 RAW/RAM cue 在 fog、sunset、部分 saturated/高亮区域有局部正信号；但在 overcast、rain、部分 KITTI 样本上没有稳定压过 C2。
- 这些 panel 支持“RAW/RAM 可能有 region-specific benefit”，但不支持“ffm_mid RAW/RAM cue 已经稳定提供 C2 之外的新收益”。

这里说“M2 更多是局部 photometric/纹理化变化”，具体含义不是 M2 不修几何，而是：C2 和 M2 都在修类似的 D0 几何大结构；M2 相比 C2 多出来的差异，更常出现在雾、夕阳、高亮、树墙/路面纹理、局部边缘等外观条件上。

几个具体例子：

- VKITTI fog `rgb_00502`：M2 final abs_rel `0.0810`，优于 C2 的 `0.0957` 和 C1 的 `0.0939`。Panel 里 M2 的 gate/residual 在雾天低对比区域、车辆/路牌边缘附近更活跃，`Err improve +green` 也更集中地出现在这些外观退化区域。这是 M2 可能利用 photometric cue 的正例。
- VKITTI sunset `rgb_00427`：M2 final abs_rel `0.1073`，略优于 C2 的 `0.1097`。改进主要来自强光照/高亮背景附近的局部修正，属于很小但方向一致的 photometric-region gain。
- VKITTI overcast `rgb_00423`：M2 final abs_rel `0.1225`，明显不如 C2 的 `0.1069`；C1 甚至从 D0 的 `0.1245` 变差到 `0.1409`。这类样本说明 appearance/photometric 额外输入会引入误修，尤其是在树墙、道路坡面和远处背景纹理上。
- VKITTI rain `rgb_00416`：M2 `0.1347` 和 C2 `0.1336` 非常接近但仍略差。即使是天气退化样本，M2 的局部 photometric 变化也没有稳定转化成 overall gain。
- KITTI panel order 07 `2011_09_26_drive_0101_sync_image_02_0000000556`：M2 final abs_rel `0.1200`，不仅差于 C2 的 `0.0920`，也差于 D0 的 `0.1114`。这是跨域上 M2 额外纹理化修正可能过拟合 synthetic RAW/RAM cue 的反例。

VKITTI 10 张 condition panel 的 per-sample best abs_rel：

| Order | Condition / sample | D0 | C1 | C2 | M2 | Best |
|---:|---|---:|---:|---:|---:|---|
| 01 | clone `rgb_00460` | 0.1238 | 0.1086 | 0.1143 | 0.1191 | C1 |
| 02 | fog `rgb_00502` | 0.1188 | 0.0939 | 0.0957 | 0.0810 | M2 |
| 03 | rain `rgb_00416` | 0.1695 | 0.1511 | 0.1336 | 0.1347 | C2 |
| 04 | overcast `rgb_00423` | 0.1245 | 0.1409 | 0.1069 | 0.1225 | C2 |
| 05 | morning `rgb_00407` | 0.1017 | 0.0988 | 0.0884 | 0.0890 | C2 |
| 06 | sunset `rgb_00427` | 0.1511 | 0.1091 | 0.1097 | 0.1073 | M2 |
| 07 | 15-deg-left `rgb_00413` | 0.1621 | 0.1221 | 0.1320 | 0.1305 | C1 |
| 08 | 15-deg-right `rgb_00478` | 0.1581 | 0.1179 | 0.1039 | 0.1160 | C2 |
| 09 | 30-deg-left `rgb_00509` | 0.2024 | 0.1900 | 0.1668 | 0.1665 | M2 |
| 10 | 30-deg-right `rgb_00453` | 0.1653 | 0.1173 | 0.1222 | 0.1187 | C1 |

这 10 张 curated panel 的 winner count 是：

```text
VKITTI panels: C1 3 / C2 4 / M2 3
KITTI panels:  C1 3 / C2 4 / M2 3
```

注意：panel 是诊断样本，不是 full validation。它们说明三种 residual 都能在局部样本上赢，但不能推翻 full validation 上 C2 最稳定的结论。

KITTI full validation 有 per-sample jsonl，可以做一个初步 paired sanity check：

| Setting | Comparison | Mean abs_rel diff | Median diff | Bootstrap 95% CI | M2 wins |
|---|---|---:|---:|---:|---:|
| 用 VKITTI best panel 对应 checkpoint：C2@11 vs M2@08 | M2 - C2 | -0.000286 | +0.000028 | [-0.000939, +0.000371] | 323 / 652 |
| 各自 KITTI best abs_rel checkpoint：C2@06 vs M2@12 | M2 - C2 | +0.000546 | +0.000697 | [-0.000046, +0.001153] | 299 / 652 |

这个结果基本是 noise-level 接近打平。M2 在 KITTI 上没有显著优于 C2；如果看各自 KITTI best abs_rel checkpoint，方向反而略偏 C2。

## 6. 原因分析

### 6.1 为什么 C2 很强

C2 只看 `D0_norm`，但这正好贴合当前任务。

在 per-image affine disparity eval 下，全局 scale 和 shift 已经通过每张图的 affine alignment 去掉。剩下主要是局部结构修正。D0 本身已经包含 DAv2-S 的强深度形状先验，所以 D0-only residual head 可以有效学习系统性局部修正。

此外，loss 也让 C2 非常有竞争力：

- high-error gate supervision 来自 `abs(D0_norm - y_norm)`。
- 模型被鼓励在 D0 已经出错的位置打开 gate。
- 这本质上很接近显式的 D0 post-processing。

因此 C2 不是弱 baseline，而是当前 objective 下很强的 calibration / post-processing baseline。

Panel 也支持这个判断：C2 只输入 `D0_norm`，但它的 gate 能沿着 D0 的物体边界、天空/远处区域、道路坡面和遮挡边缘打开；`Residual gate*delta` 的形状已经覆盖了多数可见修正区域。这说明当前任务下很多收益来自 D0 几何自身的局部重标定，而不是新传感器 cue。

### 6.2 为什么 C1 不如 C2

C1 在 `D0_norm` 外额外加入 ImageNet-normalized RGB，但 VKITTI 上 early best 后明显回退。

证据：

- VKITTI best abs_rel 在 epoch 3：`0.12574`。
- VKITTI last abs_rel 在 epoch 19：`0.13702`。
- 同时 train loss 还在持续下降。

可能原因：

- RGB 给 residual head 提供了 appearance shortcut。
- 这些 shortcut 可以降低训练 loss，但对 Scene20 holdout 泛化不好。
- C1 只比 C2 多很少参数，所以主要问题更可能是输入信号导致的 shortcut，而不是参数量。

Panel 里的典型例子是 overcast condition：C1 的 final abs_rel 从 D0 的 `0.1245` 变差到 `0.1409`，而 C2 改善到 `0.1069`。这类样本说明 RGB appearance 确实可能驱动 residual 走向不泛化的局部修正。

### 6.3 为什么 M2 没超过 C2

M2 确实比 D0 有 residual improvement，但没有超过 C2。

可能原因：

1. RAW signal 是 synthetic 的，不是独立真实信号。

   M2 的 RAW4 来自 RGB online unprocessing，不是真实传感器 RAW。因此 RAW-like cue 未必包含足够独立于 RGB/D0 的信息。

2. `ffm_mid` 可能太宽或太 noisy。

   M2 使用 64-channel FFM middle feature，自由度更高，可能让 residual head 更容易受 synthetic RAW distribution artifact 影响。Panel 中 M2 的 gate/residual 往往比 C2 更纹理化，说明 `ffm_mid` 可能把 photometric distribution 细节带入了 residual path；这些细节在 fog/sunset/saturated region 有时有用，但没有稳定转化成 overall 改善。

3. RamCore3 是可训练的，并且内部使用 BN。

   batch size 8 加 randomized unprocessing 时，BN statistics 和 RAM feature 可能不稳定。M2 train loss 持续下降，但 VKITTI 从 epoch 8 best 到 epoch 19 明显回退。

4. 当前任务更奖励 D0-local correction，而不是 photometric correction。

   由于 eval 是 per-image affine disparity，D0 geometry 已经是强信号。M2 的 photometric feature 可能帮助某些区域，例如 saturated region，但没有足够改善主导 overall 的 geometry-sensitive errors。

因此，当前更合理的解释不是“M2 没学到东西”，而是“M2 学到的额外信息与 C2 的 D0-local correction 大量重叠，且新增 photometric 细节的收益和误修互相抵消”。

### 6.4 Saturated-region 信号怎么理解

M2 在 VKITTI best epoch 的 saturated-region abs_rel 最好：

```text
M2 saturated: 0.1347
C2 saturated: 0.1382
C1 saturated: 0.1747
```

这是一个弱正信号，说明 RAW/RAM path 可能在某些 photometric region 捕捉到有用信息。但优势很小，且没有转化成 overall 优势。

在用于论文 claim 前，需要做 paired per-sample 或 bootstrap significance analysis。

## 7. 下一步建议

### Step 1：先做显著性分析，再做 claim

优先做 C2 vs M2 的 per-sample 对照：

- VKITTI abs_rel 差值：`0.12621 - 0.12100 = 0.00521`。
- KITTI abs_rel 差值：`0.09558 - 0.09503 = 0.00055`。
- M2 saturated-region 相比 C2：`0.1347 - 0.1382 = -0.0035`。

建议检查：

- paired per-sample abs_rel delta
- bootstrap confidence interval
- sign test / paired ranking count
- saturated、dark、boundary、high-error regions 的同类分析
- direct differential panels：`M2 error - C2 error`、`M2 gate - C2 gate`、`M2 residual - C2 residual`

这一步用于判断 KITTI 小差距和 saturated-region 优势是否有意义，还是只是噪声。

当前已有的 KITTI paired sanity check 显示，M2 和 C2 在 KITTI 上基本接近打平；因此真正需要补的是 VKITTI full validation 的 paired per-sample / region-level 显著性，以及更直接的 C2-vs-M2 differential visualization。

### Step 2：下一条 M-series 优先跑 M1 (`x3`)

M1 使用 RamCore3 final 3-channel output，而不是 64-channel `ffm_mid`。

价值：

- 检查 `ffm_mid` 是否太宽 / 太 noisy。
- residual head 输入从 65 channels 降到 4 channels。
- M1 在 residual head 形态上更接近 C1，但仍使用 RAM output。

如果 M1 优于 M2，说明问题更可能在 feature choice，即 `ffm_mid`，而不是 RAW/RAM 整体无效。

### Step 3：跑 M2 no-randomize-unprocessing

显式设置：

```text
randomize_unprocessing = false
vkitti_unprocessing_preset = sensor_linear_dual
```

价值：

- 检查 distribution randomization 是否让 RAM feature 不稳定。
- 如果 no-randomize M2 明显改善，说明当前 RAW augmentation distribution 太宽或太 noisy。

### Step 4：必要时做 RAM normalization control

如果 M1 和 M2 no-randomize 仍然不能超过 C2，可以测试 RamCore3 normalization：

```text
ram_norm_mode = bn | frozen_bn | gn
```

这个参数是 experiment-semantic parameter，正式 launch script 必须显式设置。

优先级：

1. `frozen_bn`
2. `gn`

目的：

- 检查 small-batch BN 加 randomized unprocessing 是否伤害 M2。

### Step 5：C1 暂时不优先继续

C1 已经回答了控制问题：

- RGB residual refinement 相比 D0 有用。
- RGB residual refinement 不如 C2 稳定。
- C1 不是当前最强 control。

继续训练 C1 优先级较低。后续比较应使用 best checkpoint，而不是 last checkpoint。

## 8. 后续结果解释规则

建议把 C2 作为必须超过的 control baseline。

解释规则：

| Future result | Interpretation |
|---|---|
| M1/M2/M3 在 VKITTI 超过 C2，且 KITTI 不变差 | RAW/RAM cue 有额外价值证据 |
| M-series 只超过 D0，但不超过 C2 | residual formulation 有效，但 RAW/RAM 额外价值未证明 |
| M-series 只在 saturated/dark regions 超过 C2 | 可能存在 region-specific RAW/RAM benefit；需要显著性分析 |
| M-series train 改善但 val 回退 | 可能过拟合或 RAW/RAM feature 不稳定 |
| C2 持续最好 | 主线应转向 D0 residual calibration / post-processing |

当前 0525 runs 的 safest conclusion：

```text
Residual correction 有效。
C2 D0-only residual 是当前最强 control。
M2 RAW/RAM ffm_mid 相比 frozen D0 有稳定收益，
但没有超过 C2，因此还不能作为 RAW-like cue 额外收益的正证据。
```

## 9. 相关源码和输出

模型代码：

- `foundation/engine/models/dav2_residual_control.py`
- `foundation/engine/models/raw_residual_dav2.py`
- `finetune_stf/models/raw_ram.py`
- `foundation/tools/residual_training_common.py`

Formal run 输出：

- `finetune_stf/exp/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
- `finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
- `finetune_stf/exp/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`

已有相关 summary：

- `plans/result/rgb_raw_baseline_fairness_summary.md`
