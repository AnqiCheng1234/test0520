# RAMCore3 + Frozen DAV2 的 RAW-like Residual Correction 执行计划

## 0. 本文档目的

本文档用于把当前的 `RAMCore3 -> DAV2` 网络设置，改造成一个更适合论文主线的实验方案。

当前已有数据流：

```text
raw4 [R, Gr, Gb, B]
  -> base_rgb = [R, (Gr+Gb)/2, B]
  -> RamCore3
  -> x3
  -> identity tail: 不 clamp，不 ImageNet norm
  -> center pad
  -> DAV2
  -> center crop depth
```

当前 `RamCore3` 内部：

```text
RPEncoder(3ch)
  Conv 3->16 k7 + BN + LeakyReLU + MaxPool
  Conv 16->32 k5 + BN + LeakyReLU + MaxPool
  Conv 32->128 k3 + BN + LeakyReLU + MaxPool
  AdaptiveAvgPool + Flatten
  => z: (B,128)

四个并行 ISP branch:
  wb:         预测 3ch gain
  ccm:        预测 3x3 color matrix
  gamma:      预测 1 个 gamma
  brightness: 预测 1 个 brightness

concat:
  x_cat: (B,12,H,W)

FFM3:
  ConvBNLeakyReLU 12->16
  ConvBNLeakyReLU 16->64   # ffm_mid
  ConvBNLeakyReLU 64->16
  Conv1x1 16->3
  BatchNorm2d(3)
  => x3: (B,3,H,W)
```

本文档给出明确建议：

1. 这套结构**可以用**，而且有文献动机。
2. 但它不建议作为最终主方法的完整数据流，即不建议把 `x3` 直接作为 DAV2 输入。
3. 更推荐把 `RamCore3` 作为 RAW-like encoder，用它提取 RAW-like cues，然后在 **frozen RGB-DAV2 输出层**做 gated residual correction。
4. 当前 `RAMCore3 -> DAV2` 路线应作为重要 baseline，用来证明 input replacement / RAW-to-DAV2 adaptation 的问题。

---

## 1. 当前架构和之前建议的对齐情况

### 1.1 当前架构确实是 RAM-like 的

当前结构与 RAM / Beyond RGB 类方法的核心思想是对齐的：

- 输入不是标准 sRGB，而是 RAW-like representation。
- 先由一个 encoder 提取全局图像状态 `z`。
- 根据 `z` 预测多个 ISP-like 分支参数。
- 多个 ISP branch 并行产生不同处理结果。
- FFM 对这些并行表示进行融合。
- 输出一个 3-channel representation `x3`。

因此，从方法动机上可以说：

> 当前 RamCore3 是一个 RAM-inspired / task-driven ISP-like RAW front-end。

它比普通 U-Net 更容易解释为“从 RAW-like input 中提取 ISP-sensitive cues”。

---

### 1.2 当前架构和之前推荐方案的主要差异

之前推荐的主线是：

```text
sRGB -> frozen DAV2 -> D0
RAW-like -> RAM-like RAW encoder -> RAW features
[D0, RAW features] -> residual/gate head -> D_final
```

而当前数据流是：

```text
RAW-like -> RamCore3 -> x3 -> DAV2 -> depth
```

核心差别在于 **RAW-derived representation 和 DAV2 的交互位置不同**。

| 设计 | 当前 RAMCore3->DAV2 | 推荐主方法 |
|---|---|---|
| DAV2 输入 | RAM 生成的 `x3` | 原始 sRGB |
| DAV2 是否 frozen | 可 frozen / 可训练，但通常作为主干 | 必须 frozen |
| RAW 的作用 | 替代 RGB 输入 | 修正 DAV2 输出 |
| 主要风险 | 破坏 DAV2 的 RGB 输入分布 | correction branch 过修 |
| 可解释性 | 可解释为 RAW front-end | 可解释为 RAW residual cue |
| 和 RAM 的关系 | 更接近原始 RAM 用法 | 借用 RAM encoder，但改变 interaction level |
| 和论文主假设是否一致 | 不完全一致 | 高度一致 |

结论：

> 当前架构适合作为 RAM-style input replacement baseline，也适合作为 proposed residual method 的 RAW encoder；但不建议把 `RAMCore3 -> DAV2` 作为最终主方法。

---

## 2. 对当前架构的具体判断

### 2.1 可以保留的部分

以下部分建议保留：

```text
raw4 -> base_rgb = [R, (Gr+Gb)/2, B]
base_rgb -> RPEncoder -> z
z -> wb / ccm / gamma / brightness
parallel ISP outputs -> x_cat
x_cat -> FFM3 -> ffm_mid, x3
```

理由：

1. 它有 RAM-like 文献依据。
2. 它比普通 CNN 更适合讲 RAW-specific encoding。
3. `wb / ccm / gamma / brightness` 分支能对应 RAW 到 RGB 处理中常见的 ISP 操作。
4. `FFM3` 可以解释为融合不同 ISP-like views。
5. 你已经有代码，实验成本低。

---

### 2.2 需要谨慎的部分

#### 问题 1：`base_rgb = [R, (Gr+Gb)/2, B]` 会丢掉 Gr/Gb 差异

这个设计和部分 RAM-style 代码保持一致，便于复现和解释。

但它也会牺牲一部分 Bayer-level 信息，尤其是：

- Gr/Gb 差异；
- Bayer pattern 中可能和边缘方向有关的局部差异；
- packed RAW 的真实 4-channel 结构。

建议：

第一版主实验可以继续使用 `base_rgb`，因为它和当前代码一致，风险最低。

但后续必须加一个 ablation：

```text
Ablation-Raw4:
raw4 [R,Gr,Gb,B] -> RPEncoder first conv 改为 4->16
```

再加一个可选 ablation：

```text
Ablation-Raw5:
[R, Gr, Gb, B, Gr-Gb]
```

这样可以回答 reviewer：

> 合并两个 green channel 是否损失了 RAW-specific cues？

---

#### 问题 2：RPEncoder 只预测全局 ISP 参数

当前 `wb / ccm / gamma / brightness` 都是由全局向量 `z` 预测出来的。

这意味着它更擅长建模：

- 全局白平衡；
- 全局颜色变换；
- 全局 gamma；
- 全局 brightness。

但它不一定擅长建模：

- 局部 depth boundary；
- 小物体；
- 反光区域局部异常；
- sky/far region 的语义性错误。

因此，不建议只依赖 `x3 -> DAV2`。

更合理的做法是使用 `ffm_mid` 或 `x3` 作为 RAW cue，然后让 residual decoder 利用 `D0` 上下文做局部修正。

---

#### 问题 3：`x3` 直接进入 DAV2 有强 distribution shift 风险

当前 direct pipeline 是：

```text
RAW-like -> RamCore3 -> x3 -> DAV2
```

这和你的核心观察一致：training loss 可以下降，但边界、天空、语义区域容易变糊或错判。

可能原因：

1. DAV2 是强 RGB depth foundation model，对输入分布敏感。
2. `x3` 虽然是 3-channel，但不等于真实 sRGB。
3. `x3` 经过最后 `BatchNorm2d(3)` 后可能出现负值或非自然图像统计。
4. 如果后面没有使用 DAV2 官方 normalization，输入分布会进一步偏离。
5. 即使 DAV2 frozen，前端也会学习一种“骗过 DAV2”的 representation，而不是稳定的 RGB-like image。

因此：

> `RamCore3 -> DAV2` 应该保留为 baseline，不应作为最终主方法。

---

#### 问题 4：identity tail 不 clamp、不 ImageNet norm 需要检查

需要确认 DAV2 wrapper 内部是否已经做了官方 preprocessing。

执行者必须检查：

```text
Case A: DAV2 wrapper 内部已经 normalize
  -> 外部不要重复 ImageNet/DAV2 normalize。

Case B: DAV2 wrapper 内部没有 normalize
  -> 当前 identity tail 会导致严重分布错位。
```

必须做一个 sanity check：

```text
同一张原始 sRGB 图像：
1. 走官方 DAV2 preprocessing + DAV2
2. 走当前代码路径 identity tail + center pad + DAV2

比较两者输出是否几乎一致。
```

如果两者差很多，说明当前 DAV2 输入预处理不一致。

---

## 3. 最推荐的主方法：RAMCore3 as RAW encoder + gated residual correction

### 3.1 总体数据流

推荐主方法不再让 `x3` 进入 DAV2。

新的主方法：

```text
sRGB
  -> frozen DAV2
  -> D0

raw4 [R,Gr,Gb,B]
  -> base_rgb = [R, (Gr+Gb)/2, B]
  -> RamCore3
  -> x3, ffm_mid

[D0, x3, ffm_mid]
  -> ResidualGateHead
  -> delta, gate

D_final = D0 + gate * delta
```

其中：

```text
DAV2 receives original sRGB only.
RAMCore3 never replaces DAV2 input in the main method.
RAMCore3 provides RAW-like correction cues only.
```

---

### 3.2 为什么这个设计更适合作为论文主线

它同时满足三个要求：

1. **和 RAM 有关系**
   RAW branch 使用 RAM-like parallel ISP processing，而不是普通 U-Net。

2. **和你的核心假设一致**
   RAW 不替代 RGB，只提供 complementary residual cue。

3. **失败可诊断**
   如果失败，可以通过 ablation 判断是 RAW cue 无效、RAM encoder 无效、residual formulation 无效，还是 synthetic RAW 本身无效。

论文里可以这样写：

> We adopt a RAM-inspired RAW encoder to extract task-relevant RAW-like representations. Unlike RAM-style input replacement, we preserve the frozen RGB depth foundation model and apply RAW-like cues only through bounded, gated residual correction.

---

## 4. 需要实现的模型版本

### 4.1 Baseline B0：Frozen DAV2 RGB

```text
sRGB -> frozen DAV2 -> D0
```

作用：主 baseline。

---

### 4.2 Baseline B1：当前 RAMCore3 direct path

```text
raw4
  -> base_rgb
  -> RamCore3
  -> x3
  -> DAV2
  -> depth
```

作用：证明 RAM-style input replacement 对 DAV2 可能不稳定。

注意：

这个 baseline 一定要保留，因为它是论文 motivation 的关键。

---

### 4.3 Main M1：RAMCore3-x3 residual

```text
sRGB -> frozen DAV2 -> D0
raw4 -> base_rgb -> RamCore3 -> x3
[D0, x3] -> ResidualGateHead -> delta, gate
D_final = D0 + gate * delta
```

作用：最简单的 RAM residual 版本。

优点：实现简单。

缺点：只使用 3-channel `x3`，可能损失 `FFM3` 中间特征。

---

### 4.4 Main M2：RAMCore3-ffm_mid residual

```text
sRGB -> frozen DAV2 -> D0
raw4 -> base_rgb -> RamCore3 -> ffm_mid
[D0, ffm_mid] -> ResidualGateHead -> delta, gate
D_final = D0 + gate * delta
```

作用：推荐主版本。

理由：

1. `ffm_mid` 是 64-channel，信息量比 `x3` 大。
2. 不强迫 RAM 输出 image-like representation。
3. 更适合作为 correction feature。
4. 避免 `x3` 末尾 `BatchNorm2d(3)` 带来的 image distribution 问题。

---

### 4.5 Main M3：RAMCore3-x3 + ffm_mid residual

```text
sRGB -> frozen DAV2 -> D0
raw4 -> base_rgb -> RamCore3 -> x3, ffm_mid
[D0, x3, ffm_mid] -> ResidualGateHead -> delta, gate
D_final = D0 + gate * delta
```

作用：最终候选主方法。

如果 M3 明显优于 M1/M2，可以用 M3 作为 main model。

如果 M3 和 M2 接近，优先用 M2，因为结构更干净。

---

## 5. ResidualGateHead 具体结构

### 5.1 输入尺寸

假设训练分辨率为：

```text
H x W
```

输入：

```text
D0:      [B, 1, H, W]
x3:      [B, 3, H, W]
ffm_mid: [B, 64, H, W]
```

如果 DAV2 因 center pad 输出后 crop，必须保证：

```text
D0, x3, ffm_mid, GT depth 全部对齐到同一 H x W。
```

---

### 5.2 推荐主输入

第一优先级：

```text
Input = concat(ffm_mid, D0_norm)
通道数 = 64 + 1 = 65
```

第二优先级：

```text
Input = concat(ffm_mid, x3, D0_norm)
通道数 = 64 + 3 + 1 = 68
```

建议先实现第二种，因为信息最完整。

---

### 5.3 Head 结构

推荐结构：

```text
Input: [B, 68, H, W]

Stem:
  Conv 3x3, 68 -> 64
  GroupNorm
  GELU
  ResBlock 64

Encoder:
  DownBlock 64 -> 128, H/2, W/2
  ResBlock 128
  DownBlock 128 -> 256, H/4, W/4
  ResBlock 256

Decoder:
  UpBlock 256 -> 128, H/2, W/2
  skip with encoder 128
  ResBlock 128
  UpBlock 128 -> 64, H, W
  skip with stem 64
  ResBlock 64

Heads:
  delta_head: Conv 3x3 64->32 + GELU + Conv 1x1 32->1
  gate_head:  Conv 3x3 64->32 + GELU + Conv 1x1 32->1
```

不要在 ResidualGateHead 里用 BatchNorm。

建议使用：

```text
GroupNorm(num_groups=8)
```

原因：

- depth 训练 batch size 通常较小；
- BatchNorm 对 small batch 不稳定；
- RAW-like input 的 batch statistics 可能变化较大；
- GroupNorm 更稳定。

---

### 5.4 输出定义

```text
delta_raw = delta_head(feature)
gate_logit = gate_head(feature)
```

定义：

```text
delta = alpha * tanh(delta_raw)
gate = sigmoid(gate_logit)
D_final = D0_norm + gate * delta
```

推荐初始值：

```text
alpha = 0.5
```

---

### 5.5 Identity initialization

必须做。

```text
delta_head 最后一层 Conv weight = 0
delta_head 最后一层 Conv bias = 0

gate_head 最后一层 Conv weight = 0
gate_head 最后一层 Conv bias = -4
```

这样初始化时：

```text
delta ≈ 0
gate ≈ 0.018
D_final ≈ D0_norm
```

作用：

- 防止训练初期破坏 DAV2；
- 让模型从“保持 DAV2”开始学习；
- 让 residual branch 只在确实有用的地方逐渐开启。

---

## 6. 训练时的 depth 表示

第一版建议做 relative depth refinement，不做 metric depth。

### 6.1 Ground truth 转 inverse depth

```text
Y = 1 / (D_gt + eps)
```

推荐：

```text
eps = 1e-6
max_depth = 80m 或 100m
```

---

### 6.2 DAV2 输出方向检查

必须检查 DAV2 输出和 inverse depth 的相关性。

执行：

```python
corr = correlation(D0[valid], Y[valid])
```

如果整个 dataset 上平均 `corr < 0`，则统一使用：

```python
D0 = -D0
```

不要每张图单独翻转。

---

### 6.3 Robust normalization

对每张图分别做：

```text
D0_norm = (D0 - median(D0_valid)) / (MAD(D0_valid) + eps)
Y_norm  = (Y  - median(Y_valid))  / (MAD(Y_valid)  + eps)
```

其中：

```text
MAD(x) = median(abs(x - median(x)))
```

训练目标：

```text
D_final -> Y_norm
```

---

## 7. Loss 设计

总 loss：

```text
L = L_depth
  + lambda_grad * L_grad
  + lambda_keep * L_keep
  + lambda_res  * L_res
  + lambda_gate * L_gate
  + lambda_gate_sup * L_gate_sup
```

推荐初始权重：

```text
lambda_grad = 0.5
lambda_keep = 0.1
lambda_res = 0.01
lambda_gate = 0.005
lambda_gate_sup = 0.05
```

---

### 7.1 Depth loss

```text
L_depth = mean_valid(abs(D_final - Y_norm))
```

第一版用 L1 即可。

---

### 7.2 Gradient loss

```text
L_grad = mean_valid(abs(grad_x(D_final) - grad_x(Y_norm)))
       + mean_valid(abs(grad_y(D_final) - grad_y(Y_norm)))
```

作用：保护边界，避免 depth blur。

---

### 7.3 DAV2 error mask

先算 baseline error：

```text
E0 = abs(D0_norm - Y_norm)
```

用每张图的 valid pixels 生成 soft error mask：

```python
q80 = quantile(E0_valid, 0.80)
q95 = quantile(E0_valid, 0.95)
M_error = clamp((E0 - q80) / (q95 - q80 + eps), 0, 1)
```

含义：

- DAV2 error top 20% 区域允许 correction；
- low-error 区域尽量保持 DAV2；
- top error 区域给 gate 更强监督。

---

### 7.4 Keep loss

```text
L_keep = mean_valid((1 - M_error) * abs(gate * delta))
```

作用：

- 防止 correction branch 在 DAV2 已经正确的地方乱改；
- 保留 DAV2 的 RGB depth prior；
- 避免全图 residual refinement 退化。

---

### 7.5 Residual magnitude loss

```text
L_res = mean_valid(abs(gate * delta))
```

作用：限制 correction 幅度。

---

### 7.6 Gate sparsity loss

```text
L_gate = mean_valid(gate)
```

作用：避免 gate 全图开启。

---

### 7.7 Gate supervision loss

```text
L_gate_sup = BCE(gate, M_error)
```

注意：

`lambda_gate_sup` 不要太大，否则 gate 会过拟合训练集误差分布。

---

## 8. 训练设置

### 8.1 Optimizer

```text
AdamW
lr = 1e-4
weight_decay = 1e-4
```

---

### 8.2 Batch size

推荐：

```text
batch_size = 4 或 8
```

如果 batch size 小于 4，更不建议使用 BatchNorm。

---

### 8.3 Epoch

第一轮：

```text
20 epochs
```

先不要长时间训练。第一轮目标是判断方向。

---

### 8.4 冻结策略

必须保证：

```text
DAV2 完全 frozen
DAV2 eval mode
DAV2 output stop-gradient
```

主方法中：

```text
RamCore3 trainable
ResidualGateHead trainable
DAV2 frozen
```

Baseline B1 中，如果测试 RAMCore3 direct path：

```text
DAV2 frozen
RamCore3 trainable
```

不要一开始 fine-tune DAV2，否则会混淆问题。

---

## 9. 必做实验矩阵

### 9.1 最小必做版本

| ID | Method | 目的 |
|---|---|---|
| B0 | sRGB -> frozen DAV2 | 主 baseline |
| B1 | raw4 -> base_rgb -> RamCore3 -> x3 -> frozen DAV2 | RAM-style input replacement baseline |
| M1 | frozen DAV2 + x3 residual | 检查 `x3` 作为 correction cue 是否有用 |
| M2 | frozen DAV2 + ffm_mid residual | 推荐主版本 |
| M3 | frozen DAV2 + x3 + ffm_mid residual | 最强候选主版本 |
| C1 | frozen DAV2 + RGB residual branch | 判断是否只是 RGB refinement 有效 |
| C2 | frozen DAV2 + D0-only residual branch | 判断是否只是 depth post-processing 有效 |
| C3 | frozen DAV2 + parameter-matched plain U-Net RAW residual | 判断 RAM-like encoder 是否必要 |

---

### 9.2 运行顺序

严格按下面顺序执行：

#### Step 1：B0

确认 frozen DAV2 在 VKITTI RGB 上的 baseline。

输出：

```text
overall metrics
boundary metrics
far-region metrics
DAV2 error map visualization
```

---

#### Step 2：B1

跑当前 `RAMCore3 -> DAV2` direct baseline。

目标：

1. 复现你之前观察到的问题；
2. 检查是否边界变糊；
3. 检查 sky/far 是否变糊；
4. 检查是否 loss 下降但 visual quality 不稳定。

---

#### Step 3：M2

先跑推荐主版本：

```text
frozen DAV2 + ffm_mid residual
```

如果 M2 都没有超过 B0，则先不要做 M3，优先检查 loss、normalization 和 gate。

---

#### Step 4：M1 和 M3

比较：

```text
x3 cue vs ffm_mid cue vs x3+ffm_mid cue
```

判断用哪个作为最终主方法。

---

#### Step 5：C1/C2/C3

确认提升不是来自：

- 多加参数；
- RGB refinement；
- D0 post-processing；
- 普通 U-Net refinement。

---

## 10. Evaluation metrics

### 10.1 整体指标

必须报告：

```text
AbsRel
RMSE
SILog
δ1
SqRel
log10
```

对于 relative depth，所有方法都必须使用同样的 scale-and-shift / affine alignment。

---

### 10.2 Region metrics

至少报告：

```text
Boundary AbsRel
DAV2 high-error region AbsRel
Far-region AbsRel
Dark-region AbsRel
Saturated-region AbsRel
```

推荐 mask 定义：

#### Boundary region

```python
edge = abs(grad_x(depth_gt)) + abs(grad_y(depth_gt))
boundary_mask = edge > percentile(edge[valid], 90)
```

#### DAV2 high-error region

```python
E0 = abs(D0_norm - Y_norm)
high_error_mask = E0 > quantile(E0_valid, 0.80)
```

#### Far region

```python
far_mask = depth_gt > 50m
```

如果 VKITTI 中远处较多，可以再报告：

```python
far80_mask = depth_gt > 80m
```

#### Dark region

```python
luma = 0.299 * R + 0.587 * G + 0.114 * B
dark_mask = luma < 0.15
```

#### Saturated region

```python
saturated_mask = max(R,G,B) > 0.95
```

---

## 11. Visualization 必做项

每个方法保存同一组样例：

```text
RGB
RAW visualization / base_rgb
x3 visualization
GT depth
DAV2 depth D0
Final depth
DAV2 error map
Final error map
Improvement map = abs(D0-GT) - abs(Final-GT)
gate map
residual map
boundary mask
far mask
```

B1 必须特别保存：

```text
x3 image
DAV2(B1) depth
B1 vs B0 error map
```

M2/M3 必须特别保存：

```text
ffm_mid feature PCA 或 channel mean visualization
gate map
residual map
```

---

## 12. 成功/失败判据

### 12.1 强成功

如果满足：

```text
M2/M3 overall > B0
M2/M3 high-error region > B0
M2/M3 boundary region > B0
M2/M3 low-error region 不明显变差
M2/M3 > C1 或至少在 RAW-sensitive region > C1
M2/M3 > C2
M2/M3 > C3
```

则论文主线成立：

> RAM-like RAW encoding is more effective when used as bounded residual correction for a frozen RGB depth foundation model than as input replacement.

---

### 12.2 可接受成功

如果：

```text
M2/M3 overall 小幅提升
high-error region 明显提升
boundary 或 dark/saturated region 有提升
但不超过 RGB fine-tuning
```

仍然可以继续。

论文 claim 应写成：

> RAW-like representation provides useful residual cues under a frozen foundation model setting, but does not replace full RGB fine-tuning.

---

### 12.3 危险结果

如果：

```text
M2/M3 ≈ C1
M2/M3 ≈ C2
M2/M3 只训练集提升，验证集不提升
gate 全图开启
boundary 继续变糊
sky/far region 明显变差
```

说明：

1. synthetic RAW-like cue 可能没有稳定额外信息；
2. RamCore3 可能只是在学习 dataset-specific correction；
3. residual branch 可能退化为普通后处理；
4. 需要降低 claim 或补真实 RAW-depth / 更物理的 synthetic RAW。

---

## 13. 代码改动清单

### 13.1 修改 RamCore3 forward

当前可能只返回：

```python
return x3
```

建议改成：

```python
return {
    "x3": x3,
    "ffm_mid": ffm_mid,
    "x_cat": x_cat,
    "wb_gain": wb_gain,
    "ccm": ccm,
    "gamma": gamma,
    "brightness": brightness,
}
```

这样方便：

- 主方法使用 `ffm_mid`；
- ablation 使用 `x3`；
- 可视化 ISP 参数；
- 诊断 RAM 分支是否退化。

---

### 13.2 新增模型 RawResidualDAV2

伪代码：

```python
class RawResidualDAV2(nn.Module):
    def __init__(self, dav2, ramcore3, residual_head):
        super().__init__()
        self.dav2 = dav2
        self.ramcore3 = ramcore3
        self.residual_head = residual_head

        for p in self.dav2.parameters():
            p.requires_grad = False
        self.dav2.eval()

    def forward(self, rgb, raw4, valid_mask=None):
        # 1. frozen RGB DAV2 path
        with torch.no_grad():
            D0 = self.dav2(rgb)              # [B,1,H,W]
        D0 = D0.detach()

        # 2. RAW-like RAM path
        base_rgb = torch.cat([
            raw4[:, 0:1],
            0.5 * (raw4[:, 1:2] + raw4[:, 2:3]),
            raw4[:, 3:4],
        ], dim=1)

        ram_out = self.ramcore3(base_rgb)
        x3 = ram_out["x3"]
        ffm_mid = ram_out["ffm_mid"]

        # 3. normalize D0 outside or inside training loop
        D0_norm = normalize_depth_like_training(D0, valid_mask)

        # 4. residual correction
        delta_raw, gate_logit = self.residual_head(
            D0_norm=D0_norm,
            x3=x3,
            ffm_mid=ffm_mid,
        )

        delta = self.alpha * torch.tanh(delta_raw)
        gate = torch.sigmoid(gate_logit)
        D_final = D0_norm + gate * delta

        return {
            "D0": D0,
            "D0_norm": D0_norm,
            "D_final": D_final,
            "delta": delta,
            "gate": gate,
            "x3": x3,
            "ffm_mid": ffm_mid,
            "ram_out": ram_out,
        }
```

---

### 13.3 新增 direct baseline model

保留当前路线：

```python
class RamToDAV2Baseline(nn.Module):
    def __init__(self, dav2, ramcore3):
        ...

    def forward(self, raw4):
        base_rgb = make_base_rgb(raw4)
        ram_out = self.ramcore3(base_rgb)
        x3 = ram_out["x3"]
        depth = self.dav2(x3)
        return depth, ram_out
```

注意：

这个模型只作为 baseline。

---

## 14. 需要记录的 training log

每个 iteration / epoch 记录：

```text
L_depth
L_grad
L_keep
L_res
L_gate
L_gate_sup
mean(gate)
max(gate)
mean(abs(delta))
mean(abs(gate * delta))
mean(abs(D_final - D0_norm))
```

对 RamCore3 额外记录：

```text
wb_gain mean/std
gamma mean/std
brightness mean/std
ccm deviation from identity
x3 mean/std/min/max
ffm_mid mean/std
```

如果出现：

```text
mean(gate) -> 1
mean(abs(delta)) -> alpha
x3 min/max 爆炸
ccm 数值异常
gamma 接近 0 或特别大
```

说明训练不稳定，需要调小 learning rate 或加强 regularization。

---

## 15. 关于当前 direct path 的特别检查

必须做以下 sanity checks。

### 15.1 DAV2 preprocessing check

输入同一张 RGB：

```text
Path A: 官方 DAV2 preprocessing -> DAV2
Path B: 当前代码 identity tail -> center pad -> DAV2
```

如果输出差异明显，说明当前 DAV2 输入 pipeline 不等价于官方推理流程。

---

### 15.2 x3 range check

统计：

```text
x3 min / max / mean / std
x3 per-channel histogram
```

如果：

```text
x3 有大量负值
x3 数值范围远超 [0,1]
x3 每通道统计和自然 RGB 差异很大
```

则 `x3 -> DAV2` 的失败很可能来自输入分布错位，而不是 RAW cue 本身没用。

---

### 15.3 x3 visualization

保存 `x3` 的可视化。

如果 `x3` 看起来不是 natural image，不要期待 DAV2 稳定。

---

## 16. 论文中应该如何描述当前架构

建议表述：

```text
We first instantiate a RAM-style input replacement baseline, where a RAW-like image is processed by a RAM-inspired front-end and the resulting three-channel representation is fed into the frozen DAV2 model. Although this design follows prior RAW object detection pipelines, we observe that it can disrupt the RGB input prior of DAV2 and produce unstable depth structures.
```

然后引出主方法：

```text
To preserve the RGB depth prior, we reuse the RAM-inspired module only as a RAW-like feature encoder. The frozen DAV2 still receives the original sRGB image, and the RAW branch predicts a bounded, gated residual over the DAV2 prediction.
```

---

## 17. 最终推荐

最终主方法不要用：

```text
raw4 -> base_rgb -> RamCore3 -> x3 -> DAV2
```

而应该用：

```text
sRGB -> frozen DAV2 -> D0
raw4 -> base_rgb -> RamCore3 -> x3, ffm_mid
[D0, x3, ffm_mid] -> ResidualGateHead -> delta, gate
D_final = D0 + gate * delta
```

其中最推荐的主版本是：

```text
M2: frozen DAV2 + ffm_mid residual
```

或者：

```text
M3: frozen DAV2 + x3 + ffm_mid residual
```

选择规则：

```text
如果 M3 明显优于 M2，用 M3。
如果 M3 和 M2 接近，用 M2。
如果 M2/M3 不超过 C1/C2，说明 RAW-like cue 贡献不足，需要降低 claim 或更换 synthetic RAW 生成方式。
```

---

## 18. 执行者最终 checklist

按顺序完成：

```text
[ ] 1. 检查 DAV2 官方 preprocessing 是否和当前代码一致。
[ ] 2. 修改 RamCore3，使其返回 x3、ffm_mid、x_cat 和 ISP 参数。
[ ] 3. 跑 B0: sRGB -> frozen DAV2。
[ ] 4. 跑 B1: raw4 -> base_rgb -> RamCore3 -> x3 -> frozen DAV2。
[ ] 5. 实现 ResidualGateHead。
[ ] 6. 跑 M2: frozen DAV2 + ffm_mid residual。
[ ] 7. 跑 M1: frozen DAV2 + x3 residual。
[ ] 8. 跑 M3: frozen DAV2 + x3 + ffm_mid residual。
[ ] 9. 跑 C1: RGB residual branch。
[ ] 10. 跑 C2: D0-only residual branch。
[ ] 11. 跑 C3: parameter-matched plain U-Net RAW residual。
[ ] 12. 输出 overall metrics。
[ ] 13. 输出 boundary / high-error / far / dark / saturated region metrics。
[ ] 14. 输出 gate / residual / improvement map visualization。
[ ] 15. 根据 M2/M3 vs B0/C1/C2/C3 判断是否继续。
```

---

## 19. 一句话结论

当前 RAMCore3 架构可以用，而且比普通 U-Net 更有 RAW 文献依据；但它最适合作为 **RAW-like encoder** 和 **input replacement baseline**，不适合作为最终主方法中直接喂给 DAV2 的唯一输入。最终主线应该是：

```text
RAM-inspired RAW encoding
+
frozen sRGB-DAV2 prior
+
bounded gated residual correction
```
