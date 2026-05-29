# RAW-like Residual Correction 当前实验复盘与下一步计划

日期：2026-05-27  
项目：VKITTI / KITTI 上的 frozen DAV2-S + RAW-like residual correction 方向  
当前重点：重新分析 M / C 系列结果，并制定下一步实验计划

---

## 0. 当前上下文

你已经完成了 VKITTI 训练与评测，并加入了 KITTI sanity eval。当前实验包括：

- `C1`: RGB residual branch；
- `C2`: D0-only residual branch；
- `M1`: RAW/RAM `x3 + D0` residual；
- `M2`: RAW/RAM `ffm_mid + D0` residual；
- `M1/M2 no-D0`: 只用 RAW/RAM feature，不输入 D0；
- D0 已确认是 `sRGB -> frozen DAV2-S` 的输出，而不是 RAW/x3 输入 DAV2 后的输出。

新版 region metrics 已经修正了之前不公平的问题：

- region mask 定义保持不变；
- VKITTI GT valid range 使用 `[1.0, 80.0]`；
- per-image affine disparity 对齐后，aligned D0 和 aligned final 都先 clip 到 `[1.0, 80.0]` 再计算 region `abs_rel`；
- 7 个 run 的 D0 consistency check 完全一致：`abs_rel=0.1531`，`d1=0.8184`。

这个修正会改变之前对“D0 是否主要靠天空/远处区域获益”的判断。

---

## 1. 更新后的核心判断

### 1.1 不能再简单说 RAW 没用

旧判断偏向：

> residual 提升主要来自 D0，RAW/RGB feature 没有明显贡献。

现在需要修正为：

> D0-only residual 仍然是最强 overall baseline，但 RAW/x3 在局部区域，尤其是 boundary 和 fog / low-contrast 场景中有正信号。当前问题不是 RAW 完全没用，而是 RAW 的区域性收益还没有转化成超过 C2 D0-only 的 overall 收益。

### 1.2 C2 D0-only 仍然是必须击败的强 baseline

Overall formal eval 中，C2 依然最强。它说明：

- D0 本身包含非常强的 depth layout；
- 一个轻量 residual head 可以学到 DAV2 输出空间到 VKITTI / driving depth 分布的后处理校准；
- 当前 M1/M2 直接 concat RAW/RAM feature 并没有稳定超过这个 D0-only calibrator。

因此，后续所有 RAW claim 都必须以 C2 为主 baseline，而不是只和 frozen DAV2 D0 对比。

### 1.3 D0 修天空/远处的可视化现象不应再解释为主要指标收益

你观察到 C2 会修很多天空或远处区域。但新版 metrics 使用 valid mask 和 clipping 后，C2 在 `far50` 并不是最强，甚至比 D0 baseline 略差：

| Method | far50 abs_rel |
|---|---:|
| Frozen DAv2-S D0 | 0.2481 |
| C2 D0-only | 0.2734 |
| M2 RA0 ffm_mid + D0 | **0.2254** |
| M1 x3 + D0 | 0.2547 |

所以更合理的解释是：

> C2 panel 里大面积天空/无效区域 residual 很显眼，但这些位置大多不进入 valid-depth 指标。它们更可能是未监督区域中的自由漂移，或者是低频 residual 的外溢，而不是 C2 指标提升的主要来源。

因此下一步需要额外统计 residual/gate 在 valid 和 invalid 区域的分布，并加 invalid keep regularization。

---

## 2. 新 region metrics 的关键信息

新版 clipped region metrics：

| Experiment | epoch | boundary | high-error | far50 | dark | saturated | mean_gate | mean_abs_delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen DAv2-S D0 | n/a | 0.3497 | 0.3484 | 0.2481 | 0.1317 | 0.2333 | n/a | n/a |
| C1 RGB residual | 3 | 0.2689 | 0.2306 | 0.2436 | 0.1049 | 0.1616 | 0.3213 | 0.4671 |
| C2 D0-only residual | 11 | 0.2692 | **0.2198** | 0.2734 | **0.1026** | **0.1357** | 0.3358 | 0.4537 |
| M2 caveated ffm_mid | 8 | 0.2653 | 0.2262 | 0.2345 | 0.1080 | **0.1308** | 0.3122 | 0.4665 |
| M2 RA0 ffm_mid | 9 | 0.2714 | 0.2316 | **0.2254** | 0.1080 | 0.1469 | 0.3175 | 0.4707 |
| M1 RA0 x3 + D0 | 14 | **0.2592** | 0.2252 | 0.2547 | 0.1038 | 0.1413 | 0.3233 | 0.4698 |
| M2 no-D0 ffm_mid | 6 | 0.3064 | 0.2764 | 0.2445 | 0.1297 | 0.1966 | 0.2473 | 0.4365 |
| M1 no-D0 x3 | 14 | 0.3156 | 0.2920 | 0.2470 | 0.1239 | 0.1891 | 0.2335 | 0.4445 |

boundary mask 的定义必须固定并随结果一起记录：

```text
boundary = top-10% GT depth gradient magnitude within valid GT pixels
```

不要用 D0 gradient 或 final prediction gradient 来定义 boundary。否则这个 region 会偏向模型自身的边界 artefact，而不是 GT depth 里的真实结构边界，`M1 x3 + D0` 的 boundary claim 也会变得不可比较。

这个表可以拆成三类 correction mode：

```text
C2 D0-only      -> high-error / dark / saturated / overall 校准最强
M1 x3 + D0      -> boundary 局部结构修正最强
M2 ffm_mid + D0 -> far50 区域最好
```

这说明不同输入确实可能学到不同 correction behavior。当前问题是这些 behavior 被单个 residual head 混在一起，RAW/x3 的局部收益没有变成 overall 主导收益。

---

## 3. 对 no-D0 实验的正确解释

no-D0 的结果明显弱于 D0-conditioned 方法：

- `M2 no-D0 ffm_mid`: overall 明显退步；
- `M1 no-D0 x3`: overall 也明显退步。

但这不应直接解释为 “RAW 没信息”。更准确的解释是：

> residual correction 是相对于 D0 的误差修正。没有 D0 时，RAW/RAM feature 不知道当前 base prediction 错在哪里，也不知道 residual 应该往哪个方向修。因此 no-D0 弱是预期内的。

no-D0 的作用是证明：

> RAW/RAM feature 不能独立替代 D0 作为 depth predictor。

但它不是判断 RAW 是否能辅助 D0 的核心证据。真正关键的问题应该是：

```text
C2 D0-only
vs
C2 frozen + RAW incremental correction
```

而不是：

```text
D0-only
vs
RAW-only
```

---

## 4. 对四组 panel 的重新分析

### 4.1 KITTI idx0000

这个样本中：

- D0 已经有合理整体结构；
- C2 D0-only residual 明显最强；
- M1 x3 + D0 和 M2 ffm_mid + D0 也有改善，但没有超过 C2。

解释：

> 这是 D0-only calibrator 的典型成功样本。主要收益来自 D0 输出空间的系统性校准，而不是 RAW/x3 的局部信息。

这个样本支持 C2 是强 baseline。

### 4.2 KITTI idx0072

这个样本是强光照变化、局部低光/过曝/树影区域。你给出的 panel 中：

- M1 x3 + D0 的表现最好；
- M2 no-D0 和 M1 no-D0 也不是完全崩；
- x3 在局部结构上确实给出了更好的修正。

解释：

> x3 可能对 appearance-degraded regions 有帮助，尤其是局部边界、强光照变化、低对比区域。这类样本是 RAW/x3 方向最有希望的证据之一。

### 4.3 VKITTI clone idx0740

这个样本中：

- C1、C2、M2、M1 都有改善；
- 不同方法差距不大；
- M1 x3 + D0 不一定最强。

解释：

> 当 D0 本身已经给出稳定结构时，RAW/x3 的增量空间有限。RAW/x3 的收益不是普遍全图型，而是条件性和区域性的。

### 4.4 VKITTI fog idx0505

这个样本中：

- M1 x3 + D0 明显强于 C1、C2、M2、no-D0；
- fog / low contrast 场景里，x3 对局部结构、桥、车辆、标牌、道路附近 depth 修正更明显。

解释：

> 这是当前最支持 RAW/x3 的 qualitative evidence。它说明 x3 可能在 fog / low-contrast / boundary-heavy 场景下提供 D0-only 难以获得的局部 cue。

但这仍然需要统计验证，不能只靠几张 panel。

额外注意：

> fog 场景里的 depth 误差可能有相当一部分是低频整体偏移，而不是纯高频边界错误。因此后续 high-pass residual / `L_lowfreq` 不能只固定一个强度后直接下结论。尤其是 `lambda_lp=0.5` 只能作为中间点，必须 sweep `0.0 / 0.3 / 0.5 / 0.8`，并单独报告 fog / low-contrast subset。

---

## 5. 关于 D0 与 feature concat 的判断

你的疑问是合理的：`D0 + ffm_mid/x3 concat` 会不会弄乱学习？

我的判断是：

> concat 本身不是错误，但它会造成非常强的 D0 shortcut。

在 depth completion / refinement 中，把 depth map 和 image feature concat 是常见操作。问题不在于形式，而在于：

```text
D0 已经包含大部分 depth layout；
residual head 很容易只利用 D0；
RAW/RGB feature 的边际贡献会被 D0 shortcut 掩盖。
```

因此当前单头结构：

```text
[D0, raw_feature] -> one residual head
```

不是最适合验证 RAW usefulness 的结构。它会自然学成：

```text
residual ≈ f(D0)
```

而不是：

```text
residual ≈ f(D0) + RAW-specific correction
```

下一步应该显式拆分：

```text
D0 branch -> calibration
RAW/x3 branch -> incremental detail correction
```

---

## 6. 当前阶段可以写出的中立结论

比较稳妥的阶段性结论：

> A lightweight residual head can significantly improve frozen DAV2-S on VKITTI, but a D0-only residual calibrator is a strong control and currently achieves the best overall AbsRel. Directly concatenating RAW/RAM features with D0 does not yet outperform D0-only overall. However, the x3 + D0 variant achieves the best clipped boundary-region error and shows qualitative advantages in fog / low-contrast examples. This suggests that RAW-like features may be more useful as incremental local-detail cues after D0 calibration, rather than as a general residual predictor competing with D0-only post-processing.

中文论文动机可以写成：

> 初步实验显示，直接将 RAW/RAM feature 与 DAV2 输出拼接训练 residual head，整体上容易被 D0 shortcut 主导。D0-only residual 已经是强基线。但 RAW/x3 在边界和雾天等局部困难区域有正信号，因此后续方法应从“RAW 直接做全局 residual”改为“D0 校准后，RAW 做局部增量修正”。

---

## 7. 下一步总体方向

不要继续盲目扩大 RAM / adapter / fusion 网络。当前最需要的是：

1. 诊断现有模型是否真的使用 RAW/x3；
2. 将 D0-only calibration 与 RAW local-detail correction 解耦；
3. 重新定义成功标准：不要求 RAW 在所有区域 overall 大幅超过 C2，但必须在 boundary / fog / low-contrast / saturated 等目标区域超过 C2 和 RGB control。

推荐下一版方法：

```text
C2-frozen + x3 incremental correction
```

核心思想：

```text
sRGB -> frozen DAV2-S -> D0
D0 -> C2 D0-only calibrator -> D1, frozen
raw4 -> base_rgb -> RamCore3 -> x3
x3 -> RAW detail branch -> Δraw
[D1 feature, x3 feature] -> gate branch -> graw
D_final = D1 + graw * Δraw
```

关键设计原则：

- D1 可以进入 gate head，因为 gate 需要知道哪里可能需要修；
- D1 不要直接进入 delta head，避免 delta 再次退化成 D0-only correction；
- RAW/x3 branch 主要负责 boundary / local detail；
- 使用 invalid keep、gate sparsity、low-frequency regularization，避免 RAW branch 在无效天空或大面积低频区域乱修；
- low-frequency 约束强度必须作为实验语义参数显式 sweep，避免在 fog / far / large-object 场景里把 RAW branch 最可能有用的低频修正压死；
- 所有新方法沿用当前同一套 eval protocol：VKITTI formal eval + KITTI sanity eval。KITTI sanity 不作为单独新实验，但必须报告 N2 相对 C2 的差值，用来排查 frozen C2 calibrator 是否只继承了 VKITTI-specific calibration。

---

## 8. 无需重训的优先诊断实验

### 8.1 RAW / x3 shuffle test

对现有 `M1 x3 + D0` 和 `M2 ffm_mid + D0` checkpoint 做 eval-time ablation：

```text
A. true RAW / x3
B. zero RAW / x3
C. mean RAW / x3
D. shuffled RAW / x3 from another image
```

记录：

```text
overall abs_rel
boundary abs_rel
high-error abs_rel
far50 abs_rel
dark abs_rel
saturated abs_rel
mean_gate
mean_abs_delta
```

判断：

- 如果 `true RAW ≈ shuffled RAW`，说明当前模型基本没有真实使用 RAW/x3；
- 如果 true RAW 在 boundary / fog / saturated 上明显更好，说明 RAW/x3 有真实条件性贡献。

### 8.2 Improvement-over-C2 map

当前 panel 主要展示 `method vs D0`。下一步应该生成：

```text
M1 x3+D0 error - C2 error
M2 ffm+D0 error - C2 error
```

建议可视化：

```text
green: method better than C2
red: method worse than C2
```

这比继续看 `method vs D0` 更关键，因为现在真正要证明的是 RAW 是否超过 C2。

### 8.3 Gate / residual energy 分布

对 C2、M1、M2 统计：

```text
sum(M * gate) / sum(gate)
sum(M * abs(gate * delta)) / sum(abs(gate * delta))
```

其中 M 包括：

```text
valid
invalid
boundary
high-error
far50
dark
saturated
```

重点回答：

- C2 是否大量 residual energy 在 invalid 区域；
- M1 x3+D0 是否更集中在 boundary；
- M2 ffm_mid 是否更集中在 far50；
- no-D0 是否 gate 更低、更分散。

### 8.4 Residual frequency analysis

对 `gate * delta` 做低频/高频分解：

```text
low = avgpool(residual, k=31)
high = residual - low
high_ratio = mean(abs(high)) / mean(abs(residual))
```

预期：

```text
C2: low-frequency correction 更多
M1 x3+D0: high-frequency / boundary correction 更多
```

如果成立，可以支撑新的方法设计：D0 branch 做低频校准，RAW/x3 branch 做高频结构修正。

### 8.5 已完成结果：0527_2127 diagnostics / panels

本节结果只引用以下第 8 节产物，不额外引用其他 run 目录的 metrics：

```text
plans/0527/diagnostics/0527_2127_raw_feature_ablation/
plans/0527/diagnostics/0527_2127_residual_energy_frequency/
plans/0527/panels/0527_2127_vs_c2_m1_x3_d0/
plans/0527/panels/0527_2127_vs_c2_m2_ffm_mid_d0/
```

#### 8.5.1 RAW / feature ablation summary

M1 `x3 + D0`:

| mode | abs_rel | d1 | boundary | high_error | far50 | dark | saturated | mean_gate | delta abs_rel vs true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| true | 0.125409 | 0.853737 | 0.259211 | 0.225184 | 0.254741 | 0.103809 | 0.141340 | 0.323347 | 0.000000 |
| zero | 0.123814 | 0.844285 | 0.278526 | 0.217913 | 0.301410 | 0.103421 | 0.162436 | 0.390350 | -0.001595 |
| mean | 0.123728 | 0.844327 | 0.278229 | 0.217459 | 0.301961 | 0.103450 | 0.162299 | 0.391297 | -0.001681 |
| shuffle | 0.128737 | 0.841927 | 0.273410 | 0.227611 | 0.267289 | 0.109633 | 0.156998 | 0.341801 | +0.003328 |

解读：

- M1 的 `shuffle - true = +0.003328 abs_rel`，说明 x3 不是完全没被使用。
- 但 `zero` / `mean` 的 overall abs_rel 反而低于 `true`，说明 true x3 当前不是稳定全局收益。
- true x3 在 `boundary`、`far50`、`saturated` 上优于 zero/mean；但在 `high_error`、`dark` 上不稳定。
- 因此 M1 更像是有条件性局部贡献，同时带来整体副作用。

M2 `ffm_mid + D0`:

| mode | abs_rel | d1 | boundary | high_error | far50 | dark | saturated | mean_gate | delta abs_rel vs true |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| true | 0.126223 | 0.853583 | 0.271365 | 0.231648 | 0.225408 | 0.108030 | 0.146856 | 0.317451 | 0.000000 |
| zero | 0.135370 | 0.830287 | 0.294970 | 0.231058 | 0.279085 | 0.116158 | 0.157877 | 0.358377 | +0.009147 |
| mean | 0.129829 | 0.846759 | 0.278215 | 0.226192 | 0.254593 | 0.109630 | 0.159670 | 0.385729 | +0.003606 |
| shuffle | 0.132491 | 0.839116 | 0.292307 | 0.238218 | 0.245365 | 0.113200 | 0.170412 | 0.347306 | +0.006267 |

解读：

- M2 的 zero/mean/shuffle 都比 true 差，说明 `ffm_mid` 对当前 M2 是明确正贡献。
- `shuffle - true = +0.006267 abs_rel`，比 M1 的 shuffle 退化更大，feature dependency 更干净。
- M2 true 在 `far50` 上明显优于 zero/mean/shuffle，但该结论只是在 M2 内部 ablation 中成立，不等价于相对 C2 的 far50 优势。

#### 8.5.2 Gate / residual energy 与 frequency summary

三个 summary 都是 1000 samples。ratio 的分母按脚本定义为 valid 区域内的 `sum(gate)` 或 `sum(abs(gate * delta))`；因此 `invalid` 行表示 invalid 区相对 valid 分母的大小，不是全图百分比。

Frequency:

| model | low_ratio | high_ratio |
|---|---:|---:|
| C2 | 0.848403 | 0.302289 |
| M1 x3+D0 | 0.855406 | 0.314007 |
| M2 ffm_mid+D0 | 0.845304 | 0.315196 |

解读：

- M1/M2 的 high_ratio 略高于 C2，但差距只有约 0.012-0.013。
- 只能作为 RAW/ffm residual 更偏高频的弱趋势，不能强证明当前分支已经清晰承担高频结构修正。

Residual energy ratio:

| mask | C2 | M1 x3+D0 | M2 ffm_mid+D0 |
|---|---:|---:|---:|
| invalid | 0.948398 | 0.993192 | 0.833008 |
| boundary | 0.111192 | 0.118353 | 0.104438 |
| high_error | 0.341587 | 0.321368 | 0.341553 |
| far50 | 0.065654 | 0.065337 | 0.056724 |
| dark | 0.391312 | 0.388766 | 0.389744 |
| saturated | 0.001230 | 0.001106 | 0.001138 |

解读：

- invalid 区 residual energy：M1 最高，C2 次之，M2 最低。M1 在 invalid 区的 residual 活动偏重，是一个风险信号。
- boundary：M1 高于 C2 和 M2，支持“M1 有一点 boundary 集中倾向”；但绝对差距不大。
- far50：M2 没有更集中，反而低于 C2/M1。因此“ffm_mid 更集中在 far50”的预期不成立。
- high_error：C2 与 M2 基本相同，M1 稍低。

Mean gate:

| mask | C2 | M1 x3+D0 | M2 ffm_mid+D0 |
|---|---:|---:|---:|
| valid | 0.335817 | 0.323347 | 0.317451 |
| invalid | 0.748235 | 0.745550 | 0.612299 |
| boundary | 0.370113 | 0.380530 | 0.328393 |
| far50 | 0.399770 | 0.387541 | 0.308133 |

解读：

- M2 的 gate 总体更克制，尤其 invalid / far50。
- M1 boundary mean_gate 略高于 C2，但 invalid mean_gate 仍很高。

#### 8.5.3 Improvement-over-C2 panels

Panel manifest 只包含两个 selected samples：

```text
dataset_index = 0, sample_name = Camera_0_rgb_00618
dataset_index = 72, sample_name = Camera_0_rgb_00623
```

M1 `x3 + D0`:

| sample | C2 abs_rel | method abs_rel | method - C2 |
|---|---:|---:|---:|
| Camera_0_rgb_00618 | 0.102325 | 0.094314 | -0.008011 |
| Camera_0_rgb_00623 | 0.124138 | 0.105727 | -0.018411 |

M2 `ffm_mid + D0`:

| sample | C2 abs_rel | method abs_rel | method - C2 |
|---|---:|---:|---:|
| Camera_0_rgb_00618 | 0.102325 | 0.101969 | -0.000356 |
| Camera_0_rgb_00623 | 0.124138 | 0.113171 | -0.010967 |

解读：

- 这两个 panel sample 上，M1/M2 都优于 C2，且 M1 改善更明显。
- 但 panels 只覆盖两个样本，不能外推为全验证集相对 C2 的结论。
- `improvement over C2` 图应主要用于看局部空间分布：绿色区域是 method 比 C2 好，红色区域是 method 比 C2 差。现有图像显示改善和退化是混合分布，适合分析“哪里改善/哪里变坏”，不适合单独作为 overall 结论。

#### 8.5.4 仅基于第 8 节产物的结论

- M1 的 x3 确实被模型使用，但 true x3 的 overall 收益不稳定；zero/mean 比 true 略好，说明 x3 当前可能带来局部结构收益和全局副作用的混合。
- M2 的 ffm_mid feature dependency 更明确；true 明显优于 zero/mean/shuffle。
- Energy/frequency 支持 M1 有弱 boundary / 高频倾向，但 invalid residual energy 偏重。
- M2 gate/residual 更克制，尤其 invalid 区；但没有体现 far50 residual energy 更集中的预期。
- Panels 证明两个 selected samples 上 method 相对 C2 有改善，但不能作为全量 C2 对比结论。

---

## 9. 新一轮重训实验设计

### 9.1 主方法 N2：C2 frozen + x3 incremental correction

数据流：

```text
D0 = frozen_DAV2(sRGB)
D1 = frozen_C2_calibrator(D0)
x3 = RamCore3(raw4).x3

F_x3 = RawDetailEncoder(x3)
F_d1 = D1Encoder(D1)

delta_raw = DeltaHead(F_x3)
gate_raw = GateHead(concat(F_x3, F_d1))

D_final = D1 + gate_raw * delta_raw
```

推荐约束：

```text
D1 可以进入 GateHead
D1 不直接进入 DeltaHead
```

这样可以减少 D0 shortcut。

但这只是 soft cutoff，不是完全消除 D0 shortcut：

```text
gate_raw 仍然会乘到 delta_raw 上
D1 仍然可以通过 gate_raw 调制 RAW delta 的幅度和空间位置
```

因此需要额外加一个 D1-conditioned DeltaHead 的 stop-gradient 变体，作为更严格的 shortcut ablation：

```text
F_d1_sg = stop_gradient(F_d1)
delta_raw = DeltaHead(concat(F_x3, F_d1_sg))
gate_raw = GateHead(concat(F_x3, F_d1))
```

这个变体不作为默认主方法，而是用来判断：

```text
如果加入 stop-gradient F_d1 后明显提升，说明 D1-conditioned delta 仍有价值；
如果它退化成接近 D1-only extra head 或 shuffled x3 也不掉，说明 RAW/x3 贡献仍被 D1 shortcut 掩盖。
```

### 9.2 可选 high-pass residual

为了防止 RAW branch 重新学习 C2 的低频 calibration，可以使用 soft high-pass：

```text
low = avgpool(delta_raw, k=31)
delta_raw_hp = delta_raw - lambda_lp * low
D_final = D1 + gate_raw * delta_raw_hp
```

最低 sweep：

```text
lambda_lp sweep = {0.0, 0.3, 0.5, 0.8}
```

`lambda_lp` 是实验语义参数，formal launch script 里必须显式设置，不能依赖默认值。`0.5` 只能作为中间强度，不足以支撑最终结论。

不要一开始完全去除低频，因为 fog / far / large object 可能仍然需要部分低频修正。每个 `lambda_lp` 都需要报告 overall、boundary、far50、fog / low-contrast subset，以及 residual low-frequency energy。

---

## 10. 新 loss 设计

Stage-2 RAW incremental branch 推荐 loss：

```text
L = L_final
  + 2.0 * L_boundary
  + 0.5 * L_grad
  + 0.2 * L_keep_good_D1
  + 0.05 * L_gate_sparse
  + 0.1 * L_lowfreq
  + 0.1 * L_invalid_keep
```

### 10.1 L_final

valid pixels 上的普通 depth regression loss：

```text
L_final = |D_final - Y|
```

### 10.2 L_boundary

boundary mask 上加权：

```text
G_gt = depth_gradient(Y)
threshold = percentile(G_gt[valid_gt], 90)   # per-image
boundary = valid_gt & (G_gt > threshold)
L_boundary = boundary * |D_final - Y|
```

定义固定为 GT depth gradient top-10%，并且 percentile 在每张图的 valid GT pixels 内计算。不要用 D0 gradient、D1 gradient 或 prediction gradient 定义 boundary。

用途：突出 x3 在真实 depth boundary 上的潜在优势，同时保证 §2 的 boundary region claim 和训练 loss 使用同一类语义。

### 10.3 L_grad

保护局部结构：

```text
L_grad = |grad(D_final) - grad(Y)|
```

### 10.4 L_keep_good_D1

在 C2 已经好的地方，不允许 RAW branch 乱改：

```text
E1 = |D1 - Y|
q_good = {0.3, 0.5, 0.7}
threshold_i = quantile(E1_i[valid_gt_i], q_good)   # per-image
M_good_i = valid_gt_i & (E1_i < threshold_i)
L_keep_good_D1_i = M_good_i * |gate_raw_i * delta_raw_i|
```

quantile 粒度固定为 per-image。不要用 per-batch 或 global quantile：

- per-image 会让每张图都有明确比例的 good-D1 区域，用来约束 RAW branch；
- per-batch 会让简单图几乎全是 good、困难图几乎全是 bad；
- global quantile 会受到数据分布和采样顺序影响，和单张图的局部修正目标不一致。

`q_good` 也是实验语义参数，需要 sweep `0.3 / 0.5 / 0.7`。`0.5` 不是默认结论，只是中间强度。

### 10.5 L_gate_sparse

避免 gate 全图打开：

```text
L_gate_sparse = mean(gate_raw)
```

### 10.6 L_lowfreq

避免 RAW branch 学低频 D0 calibrator：

```text
L_lowfreq = |avgpool(gate_raw * delta_raw, k=31)|
```

这个项和 fog / far / large-object 场景存在直接 tension：这些场景的错误可能包含低频整体偏移，而 `L_lowfreq` 会压制 RAW branch 的低频输出。因此不能只报告单一低频约束设置。

最低要求：

```text
lambda_lp sweep = {0.0, 0.3, 0.5, 0.8}
fog / low-contrast subset metrics per lambda_lp
residual low/high frequency energy per lambda_lp
```

如果 `lambda_lp` 越大导致 fog subset 明显变差，即使 overall 或 boundary 改善，也要把 claim 写成“高频结构受益、低频 fog correction 被抑制”，不能泛化为 RAW/x3 全场景有效。

### 10.7 L_invalid_keep

抑制 sky / invalid 区域的无监督漂移：

```text
L_invalid_keep = invalid_mask * |gate_raw * delta_raw|
```

这个项对可视化很重要，也能减少无效区域 residual 外溢。

---

## 11. 新实验矩阵

优先实验分成核心对照和必做 sweep / ablation。核心对照不要太多，但下面这些实验语义必须显式记录：

| ID | Method | 目的 |
|---|---|---|
| N0 | frozen DAV2-S D0 | 原始 baseline |
| N1 | C2 D0-only residual | 当前最强 baseline |
| N2 | C2 frozen + x3 incremental correction | 主实验，`lambda_lp` 和 `q_good` 必须显式设置 |
| N3 | C2 frozen + RGB incremental correction | 判断 RAW/x3 是否优于 RGB detail |
| N4 | C2 frozen + ffm_mid incremental correction | 判断 ffm_mid 是否主要适合 far50 |
| N5 | C2 frozen + D1-only extra head | 判断是否只是多加参数 |
| N6 | N2 checkpoint eval-time shuffled x3 | 判断模型是否真的使用 x3 |
| N7 | N2 stop-gradient F_d1 into DeltaHead | 判断 D1-conditioned delta 是否仍在形成 shortcut |

注意：N6 不需要单独训练，是 N2 checkpoint 的 eval-time ablation。

所有训练型 N 系列实验都保持同样评测：

```text
VKITTI formal eval
KITTI sanity eval
report: N2 - C2 on KITTI sanity
```

这里不需要为 KITTI 单独增加新方法编号；它是统一 eval protocol 的一部分。如果 N2 在 VKITTI target regions 提升，但在 KITTI sanity 上相对 C2 明显退化，需要把结果解释为可能继承或放大了 VKITTI-specific calibrator，而不是稳健的 RAW local correction。

N2 的最低 sweep：

```text
lambda_lp: 0.0 / 0.3 / 0.5 / 0.8
q_good: 0.3 / 0.5 / 0.7
```

如果训练预算有限，先固定 `q_good=0.5` 跑完整 `lambda_lp` sweep；再在最有希望的 `lambda_lp` 上跑 `q_good` sweep。正式结论不能只来自一个隐含默认值。

---

## 12. 成功标准

### 12.1 强成功

满足：

```text
N2 overall abs_rel < C2
N2 boundary abs_rel < C2
N2 fog / low-contrast subset < C2
N2 > N3 RGB control
N2 true x3 > shuffled x3
```

这时可以继续推进主论文，claim 可以相对积极。

同时必须满足：

```text
boundary 使用 GT gradient top-10% 固定定义
lambda_lp sweep 已报告，fog / low-contrast subset 没有只靠单一默认值支撑
q_good sweep 或至少预声明的 q_good 设置已报告
KITTI sanity 上已报告 N2 - C2，且没有明显额外退化
```

### 12.2 中等成功

满足：

```text
N2 overall ≈ C2
N2 boundary 明显优于 C2
N2 fog / saturated / low-contrast subset 优于 C2
N2 不一定超过 RGB control
```

这仍有价值，但 claim 要写成区域性：

> RAW-like x3 provides region-specific local-detail refinement, not universal overall improvement.

### 12.3 失败

出现：

```text
N2 overall > C2
N2 boundary 不优于 C2
N2 true x3 ≈ shuffled x3
```

这说明当前 synthetic RAW/x3 没有稳定边际贡献。应考虑降低 claim，或换 RAW 生成方式 / 真实 RAW-depth 数据。

---

## 13. 后续 paper positioning 建议

当前阶段不要写：

> RAW improves monocular depth estimation.

更稳妥的写法：

> Direct RAW/RAM residual fusion is dominated by the D0 shortcut. However, RAW-like x3 features show localized benefits in boundary and adverse-appearance regions. We therefore reformulate RAW-guided refinement as incremental local-detail correction after D0-only calibration.

如果 N2 成功，可以写：

> Instead of replacing RGB inputs or directly competing with a D0-only calibrator, RAW-like features are most useful as constrained local correction cues on top of a strong frozen RGB depth prior.

如果 N2 失败，则写成 negative finding 也有价值：

> Under inverse-ISP synthetic RAW on VKITTI, most residual gains can be explained by D0-conditioned calibration; RAW-like representations do not yet provide robust marginal gains over strong D0-only and RGB residual controls.

---

## 14. 最终建议

当前不要放弃 RAW 路线，但也不要继续沿着 `D0 + raw_feature -> single residual head` 直接加复杂模块。

下一步最合理路线是：

```text
C2 frozen + x3 incremental local correction
```

同时做：

```text
1. true / zero / mean / shuffled x3 eval
2. improvement-over-C2 visualization
3. valid / invalid residual energy analysis
4. residual frequency analysis
5. RGB incremental correction control
6. D1-only extra head control
7. lambda_lp sweep: 0.0 / 0.3 / 0.5 / 0.8
8. per-image q_good sweep: 0.3 / 0.5 / 0.7
9. stop-gradient F_d1 DeltaHead ablation
```

如果这组实验能证明 x3 在 boundary / fog / low-contrast 上稳定超过 C2 和 RGB control，方向仍然有论文价值。否则，应降低 claim，并把结果定位为 synthetic RAW-like representation 在当前 inverse-ISP 设定下边际增益不足。
