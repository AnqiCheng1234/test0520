# LOD RAW Post-RAM 去噪实验执行计划

生成日期：2026-06-09  
目标：先用现成 denoiser 快速验证 **RamCore3 输出后、DAv2 输入前的 post-BN pseudo RGB 是否存在可修复噪声/伪影**，再训练自己的 **Post-RAM Cleanup Adapter**。  
当前约束：RAW 输入分支没有 DAv2 ImageNet norm；`RamCore3` 末尾有可学习 `BatchNorm2d(3, affine=True)`，其输出经 optional tail / pad 后直接作为 DAv2 backbone 输入。

---

## 结果位置索引

下次快速查看本轮结果时，优先看这些路径：

```text
Stage A DRUNet external denoise summary：
finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/summary.md

Stage A DRUNet external denoise CSV：
finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/summary.csv

Stage A DRUNet 最好 cell 可视化：
finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D3/viz/EXT_D3/panel_grid.png

Stage A DRUNet TorchScript：
/mnt/drive/3333_raw/0000_exp_ckpt/external_denoisers/DPIR/drunet_color_pad8.ts

strict 总结：
finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/summary.md

strict CSV：
finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/stageB_strict_summary.csv
finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/stageB_paired_deltas.csv

Stage B cleanup 可视化总览图：
finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz/viz/T_seed123_cleanup/panel_grid.png

Stage B cleanup 可视化目录：
finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz

Stage B cleanup 单样本 panel：
finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz/viz/T_seed123_cleanup/per_sample/
```

对应执行记录见：

```text
0610_1132 Stage A DRUNet external denoise minimal probe 完成
0610_0213 strict 复核完成，本轮计划完成
0610_1122 Stage B cleanup 可视化补充
```

---

## 0. 当前判断

### 0.1 已确认事实

1. `RAW_dark` 和 `RAW_normal` 的差异是真实瓶颈之一。  
   cross-eval 中，canonical LoRA recipe 下：
   - `C_dark + I_dark`: D1 ≈ `0.8453`
   - `C_dark + I_normal`: D1 ≈ `0.8805`
   - `C_normal + I_normal`: D1 ≈ `0.8989`
   - `C_normal + I_dark`: D1 ≈ `0.8170`

2. 这说明：
   - `C_dark` 可以吃 `RAW_normal`，input swap 有明显提升；
   - 但 `C_dark(I_normal)` 没有达到 `C_normal(I_normal)`；
   - `C_normal` 直接吃 `RAW_dark` 不鲁棒；
   - 输入退化和训练域 / 参数差异都重要。

3. 之前的 internal local residual + feature distillation 路线没有突破已观测 seed 上沿：
   - 第二轮最强 `AB_s05_sp`: best D1 `0.8469`
   - seed-only baseline 上沿 `L0_seed123`: best D1 `0.8502`
   - 因此当前 internal residual 设计不建议继续加码。

   但这里的 `0.8502` 是训练过程里的 `best_lod_d1` 口径，不能直接和 cross-eval strict 的 `0.8453` 放在同一张阈值表里。仅 seed `42 -> 123` 已观测到 D1 `0.8456 -> 0.8502`，波动 `+0.0046`，大于原计划中若干判据档位宽度。

4. 归因实验显示，LOD `RAW_dark` 的主要退化更接近 **noise / low SNR / 结构破坏**，不是单纯曝光不足：
   - exposure-only 几乎不伤 D1；
   - noise-only 可解释大部分 `RAW_normal -> RAW_dark` drop；
   - hand median / bilateral / tone / oracle tone+denoise 都没有稳定提升。

### 0.2 对下一步的定位

下一步不再继续当前 RAM 内部 local residual 设计，而改成：

```text
RAW_dark
  -> RamCore3
  -> x_ram         # RamCore3 post-BN pseudo RGB；canonical tail=identity
  -> Post-RAM denoise / cleanup
  -> optional raw_ram_rgb_tail
  -> pad
  -> DAv2 backbone
  -> DPT decoder
```

第一阶段先用现成 denoiser 做 quick probe；第二阶段训练自己的 cleanup adapter。

---

## 1. 关键修正：RAW 分支没有 ImageNet norm

之前方案里如果默认 “pseudo RGB 经过 ImageNet norm 后进入 backbone”，这个假设需要修正。

当前 true-LOD RAW RGB16 输入链路是：

```text
x_raw3: 3ch RAW RGB16 tensor   # 不是 Bayer packed 4ch
  -> RamCore3.forward_with_features
  -> x_ram: 3ch post-BN pseudo RGB
  -> optional raw_ram_rgb_tail
  -> CenterPadCropAdapter.pad_rgb
  -> DAv2 backbone
```

`RamCore3` 内部真实顺序是：

```text
FFM3 output x3_pre_bn
  -> norm_layer = BatchNorm2d(3, affine=True)
  -> x3_ram
  -> optional internal local_residual
  -> x3
```

canonical `C_dark(0608_2026)` 使用 `raw_ram_rgb_tail=identity`、`raw_ram_local_residual=none`，所以当前 `x_ram = x3 = x3_ram`。它不是自然图像域，也不是未知 `[0,1]` pseudo RGB，而是 **post-BN native domain**：可能有负值，均值和尺度受 BN running stats 以及 affine `gamma/beta` 影响。

因此后续的 denoise / cleanup 模块应该处理 **RamCore3 post-BN native domain**，而不是处理 ImageNet-normalized RGB。

这会影响两个设计点：

### 1.1 现成 denoiser 不能直接当最终方案

现成 denoiser 多数是在 natural sRGB / `[0,1]` 图像域上训练的，而 `x_ram` 是 RAM 学出来的 post-BN pseudo RGB。它可能：

- 数值范围不是标准 `[0,1]`，并且可能有负值；
- 通道统计不是自然 RGB；
- 通道均值 / 方差受 BN running stats 和 affine 参数影响；
- 噪声形态不是普通 sRGB camera noise；
- DAv2 对它的读取方式也不等同于 ImageNet RGB。

所以现成 denoiser 只能作为 **zero-training probe**：

> 如果它能提升，说明 post-RAM input-space cleanup 方向值得做；  
> 如果它失败，不能直接否定训练式 post-RAM cleanup。

### 1.2 自己训练的 cleanup adapter 应该直接工作在 `x_ram` native domain

训练自己的模块时，不做 sRGB 化，不做 ImageNet norm，不做外部图像重建目标；模块直接工作在 post-BN native domain。

```text
x_clean = x_ram + scale * Adapter(x_ram)
```

训练目标先只用原来的 depth loss。

---

## 2. 阶段 A：现成 denoiser quick probe

### 2.1 目的

快速回答一个问题：

> 在不训练新模块的情况下，如果直接清理 `RamCore3` 输出的 post-BN pseudo RGB，D1 是否有上升信号？

这一步只做 zero-eval，不更新 RAM、LoRA、decoder、DAv2 backbone，也不更新 denoiser。

---

## 3. 阶段 A 的推荐 denoiser 候选

优先顺序如下。

### A1. DRUNet / DPIR

推荐优先试。原因：

- 它是通用 deep denoiser prior；
- 可以通过 noise level 控制去噪强度；
- 比固定 real-image denoiser 更适合做强弱扫描；
- 如果失败，可以较明确地判断“简单 post-BN pseudo-RGB denoise 不够”。

建议先扫：

```text
sigma ∈ {10, 25, 50} / 255
alpha ∈ {0.25, 0.5, 1.0}
```

其中 `alpha` 是 denoiser residual blend 系数，定义见 §4。

### A2. Restormer real-image denoising

作为第二候选。原因：

- 它有 real image denoising 任务；
- 更接近真实相机噪声；
- 但它假设输入是自然 sRGB，和 `x_ram` domain mismatch 更强。

建议只试少量配置：

```text
alpha ∈ {0.25, 0.5}
```

### A3. NAFNet SIDD denoising

作为第三候选。原因：

- 模型较轻，集成成本通常低；
- SIDD denoise 目标仍然是 sRGB real-noise；
- 与 `x_ram` 仍有 domain mismatch。

建议只试：

```text
alpha ∈ {0.25, 0.5}
```

### A4. 不建议作为主线的现成方法

不要优先重复：

```text
median3
bilateral
tone/gamma/percentile only
```

这些在前面的 attribution / ablation 里已经没有稳定收益。

---

## 4. 阶段 A 的 pseudo-RGB wrapper

由于现成 denoiser 通常要求 `[0,1]` 输入，而 `x_ram` 是 RamCore3 post-BN native domain，必须加一个 wrapper。  
这一阶段的主要风险不是“denoiser 强弱”，而是 wrapper 自己把 post-BN 尾部信息截掉，导致 D1 变化无法归因。

### 4.1 先收集 `x_ram` 统计

用 canonical `C_dark` checkpoint 在 `00Train` 上跑一遍，记录 RamCore3 输出 `x_ram` 的通道统计。统计必须按 post-BN 输出解释，而不是按自然 RGB 解释：

```text
per-channel:
  q001, q01, q05, q50, q95, q99, q999
  mean, std
  min, max
```

推荐默认使用：

```text
q001 / q999 per-channel affine
```

同时记录 `q01 / q99` 作为对照，但不要默认用它做主 probe，除非 identity round-trip 证明尾部截断对 D1 几乎无影响。

每个 affine 设置都必须报告：

```text
clamp_low_ratio
clamp_high_ratio
clamp_total_ratio
per-channel clamp ratio
```

### 4.2 必跑 null / identity sanity cell

在任何外部 denoiser 之前，先跑两个 sanity cell：

| ID | denoiser | alpha | affine | 目的 |
|---|---|---:|---|---|
| EXT_NOOP_A0 | none / bypass | 0 | n_a | 检查插入点和 eval 路径，`x_out == x_ram`，strict D1 必须复现 baseline |
| EXT_ID_Q001 | identity | 1.0 | q001/q999 | 量化 affine+clamp round-trip 的信息损失 |
| EXT_ID_Q01 | identity，可选 | 1.0 | q01/q99 | 量化更强 clamp 的信息损失，不作为默认主线 |

其中 `EXT_NOOP_A0` 不应经过 clamp；它只验证 optional wrapper 打开后仍能完全退化成原模型。  
`EXT_ID_Q001/Q01` 则故意经过：

```text
u = clamp(...)
x_dn = inverse_affine(u)
x_out = x_dn
```

如果 `EXT_NOOP_A0` 不能复现 `C_dark(I_dark)` strict baseline，先修实现，不进入 denoiser probe。  
如果 `EXT_ID_Q001` 已明显拉低 D1，外部 denoiser probe 的结果必须标注“含 clamp round-trip 损失”，不能直接归因到 denoiser。

### 4.3 标准 wrapper

对每个通道独立做：

```text
u = clamp((x_ram - q_lo) / (q_hi - q_lo + eps), 0, 1)
u_dn = Denoiser(u)
x_dn = u_dn * (q_hi - q_lo + eps) + q_lo
x_out = x_ram + alpha * (x_dn - x_ram)
```

其中：

```text
默认 q_lo/q_hi = q001/q999
alpha ∈ {0.25, 0.5, 1.0}
```

解释：

- `u` 是给现成 denoiser 的标准输入；
- `x_dn` 是映射回 post-BN native domain 的输出；
- `x_out` 是最终送给 DAv2 backbone 的输入；
- `alpha` 防止现成 denoiser 过度改变 DAv2 已经适应的 post-BN pseudo RGB。

### 4.4 不推荐直接做的 wrapper

暂时不要做：

```text
直接 clamp x_ram 到 [0,1] 后送 denoiser
denoiser 输出直接送 DAv2，不映射回 x_ram domain
per-image min-max normalization
强制 color correction / white balance
```

原因：这些会显著改变 DAv2 已经适应的 post-BN pseudo-RGB 数值空间，容易把实验变成 domain shift，而不是 denoise probe。

---

## 5. 阶段 A 实验矩阵

不重新训练 P0 baseline。  
但阶段 A 必须在同一 strict eval 路径里包含 `EXT_NOOP_A0`，保证 no-op wrapper 复现已有 strict baseline。

### 5.1 canonical LoRA checkpoint

使用：

```text
C_dark = 0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
```

冻结：

```text
RAM
LoRA
DAv2 backbone
DPT decoder
external denoiser
```

只改变：

```text
x_ram -> external denoiser wrapper -> x_out
```

### 5.2 推荐最小矩阵

| ID | denoiser | sigma | alpha | affine | 备注 |
|---|---|---:|---:|---|---|
| EXT_NOOP_A0 | bypass | - | 0 | n_a | 必跑 no-op sanity |
| EXT_ID_Q001 | identity | - | 1.0 | q001/q999 | 必跑 clamp round-trip |
| EXT_D1 | DRUNet/DPIR | 10/255 | 0.25 | q001/q999 | 弱去噪 |
| EXT_D2 | DRUNet/DPIR | 10/255 | 0.5 | q001/q999 | 弱去噪，中等 blend |
| EXT_D3 | DRUNet/DPIR | 25/255 | 0.25 | q001/q999 | 中等去噪 |
| EXT_D4 | DRUNet/DPIR | 25/255 | 0.5 | q001/q999 | 中等去噪，中等 blend |
| EXT_D5 | DRUNet/DPIR | 50/255 | 0.25 | q001/q999 | 强去噪，保守 blend |
| EXT_R1 | Restormer real denoise | - | 0.25 | q001/q999 | real denoise |
| EXT_R2 | Restormer real denoise | - | 0.5 | q001/q999 | real denoise |
| EXT_N1 | NAFNet SIDD | - | 0.25 | q001/q999 | real denoise |
| EXT_N2 | NAFNet SIDD | - | 0.5 | q001/q999 | real denoise |

如果资源紧张，先只跑：

```text
EXT_NOOP_A0, EXT_ID_Q001, EXT_D1, EXT_D3, EXT_D4, EXT_R1
```

`EXT_ID_Q01` 不列入默认最小矩阵。仅在以下情况补跑：

```text
1. EXT_ID_Q001 已显示 q001/q999 仍有不可忽略的 clamp round-trip 损失，需要量化 q01/q99 会恶化到什么程度；
2. 后续有人想把 q01/q99 作为主 affine，此时必须先用 EXT_ID_Q01 证明 round-trip 损失可接受。
```

### 5.3 每个 EXT 实验后必须接可视化

阶段 A 每个 denoiser / sanity cell 跑完 eval 后，必须立即生成对应可视化，再进入下一项汇总判断。  
可视化不是可选诊断；如果某个 EXT 没有可视化产物，该实验结果标记为 `metrics_only_incomplete`，不得单独用于 go/no-go。

固定一组可视化样本，保证不同 denoiser 可肉眼横向比较：

```text
fixed_viz_samples:
  8-16 张固定样本
  覆盖高噪声 quartile / 低噪声 quartile
  覆盖 D1 改善样本 / D1 变差样本
  所有 EXT 使用同一批样本、同一裁剪、同一 colormap、同一数值范围
```

每个 EXT 至少输出：

```text
viz/<EXT_ID>/panel_grid.png
viz/<EXT_ID>/per_sample/*.png
viz/<EXT_ID>/manifest.json
```

`manifest.json` 记录 run、checkpoint、eval 口径、BN 协议、denoiser 参数、affine 分位数、clamp ratio、样本 ID、指标变化和可视化文件路径。

---

## 6. 评估口径与判定

### 6.0 口径原则

不要把以下三类数字放在同一把尺子上：

| 数字 | 来源 run | epoch | eval 口径 | 用途 |
|---:|---|---:|---|---|
| 0.845279 | `C_dark 0608_2026` | e40 | cross-eval strict `M_DD` | 阶段 A 和最终 strict baseline |
| 0.842198 | 同 checkpoint，不同 BN 处理 | e40 | 非 strict / BN 协议差异示例 | 说明 BN 协议可带来约 `0.003` 漂移，不作阈值 |
| 0.8502 | `L0_seed123 0609_2045` | e10 | 训练 `best_lod_d1` | 阶段 B seed-only 上沿的弱估计 |
| 0.855 | `RGB_dark RL 0608_2246` | e22 | 训练 `best_lod_d1` | RGB_dark 对照的弱参考，最终也应回到 strict |

阶段 A 是 zero-eval，主判据使用 **cross-eval strict**。  
阶段 B 是训练实验，训练期只用 `best_lod_d1` 做早期筛选；新实验的最终比较必须统一回到 **cross-eval strict，固定 BN 协议**。不得用训练 `best_lod_d1` 直接比较 strict baseline。

### 6.1 主指标

```text
D1
AbsRel
RMSE
```

阶段 A strict 参考数：

```text
C_dark(I_dark) strict D1 ≈ 0.8453
C_normal(I_normal) strict D1 ≈ 0.899
```

阶段 B training-best 参考数：

```text
seed42 L0 best_lod_d1 ≈ 0.8456
seed123 L0 best_lod_d1 ≈ 0.8502
当前 2-seed seed-only 波动 ≈ 0.0046
RGB_dark training best_lod_d1 ≈ 0.855
```

### 6.2 阶段 A strict 成功分级

| 结果 | 判断 |
|---|---|
| `EXT_NOOP_A0` 不能复现 strict baseline | 实现或 eval 路径有问题，停止 denoiser probe |
| `EXT_ID_Q001` 明显低于 no-op | clamp round-trip 已损伤信息，denoiser 结果必须谨慎解释 |
| EXT D1 ≤ no-op strict D1 | 现成 denoiser 无效或破坏 post-BN domain |
| no-op strict D1 < EXT D1 < no-op strict D1 + 0.003 | 弱 strict 信号，只能作为诊断 |
| EXT D1 ≥ no-op strict D1 + 0.003 且 AbsRel/RMSE 不恶化 | 有 strict 信号，可支持进入阶段 B |
| EXT 接近后续 strict RGB_dark 对照 | 强信号；只有 strict RGB_dark 对照补齐后才能这样宣称 |

`0.8502` 不能作为阶段 A strict 阈值；它来自训练 `best_lod_d1`。

### 6.3 阶段 B training-best 初筛分级

| 结果 | 判断 |
|---|---|
| D1 ≤ matched-seed baseline 分布 | 无效 |
| D1 高于 seed42 但未超过已观测 seed 上沿 | 不能分辨，仍可能是 seed 噪声 |
| D1 ≥ 0.8502 | 暂定超过当前 2-seed 上沿；必须用 ≥2 cleanup seeds 或 repeated strict eval 复核 |
| D1 ≥ 0.855 | 接近 RGB_dark training-best 对照；仍需 strict cross-eval 统一比较 |
| AbsRel 明显恶化 | 即使 D1 小涨，也需谨慎 |

### 6.4 辅助诊断

每个 denoiser 设置保存：

```text
x_ram
x_out
delta = x_out - x_ram
pred depth
error map
D1 map
```

每个 denoiser 设置还必须生成可视化 panel，至少包含：

```text
RAW_dark / display RGB
x_ram normalized view
x_out normalized view
delta = x_out - x_ram heatmap
abs(delta) heatmap
pred depth
depth error map
D1 hit/miss map
per-sample metrics: D1 / AbsRel / RMSE / clamp ratio / delta ratio
```

可视化规则：

```text
x_ram 与 x_out 使用同一个 per-channel affine 和同一个显示范围
delta heatmap 使用所有 EXT 共享的 symmetric range
depth / error / D1 map 使用所有 EXT 共享的 colormap 和 range
panel 标题必须写 EXT_ID、sigma、alpha、affine、strict D1、AbsRel、clamp_total_ratio
```

并记录：

```text
mean|delta| / mean|x_ram|
delta p50 / p95 / p99
x_ram 和 x_out 的 p1/p50/p99
clamp_low_ratio / clamp_high_ratio / clamp_total_ratio
DAv2 layer5/8/11 token distance to C_normal(RAW_normal)
高噪声 quartile D1
低噪声 quartile D1
```

### 6.5 如何解释结果

| 观察 | 解释 | 下一步 |
|---|---|---|
| EXT 有明显提升 | post-RAM input-space cleanup 有效 | 进入阶段 B，训练自己的 adapter |
| EXT 提升但视觉上过平滑 | denoiser有方向，但现成模型过强 | 阶段 B 用小残差 adapter |
| EXT feature distance 下降但 D1 不涨 | decoder / depth loss 不匹配 | 阶段 B 先只用 depth loss |
| EXT D1 下降 | 现成 sRGB denoiser domain mismatch | 仍可继续阶段 B，但不再投入外部 denoiser |
| 所有 EXT 都失败 | post-BN pseudo-RGB image-space cleanup 可能不够 | 阶段 B 仍可跑一版；若也失败，fallback 需重新确认，不自动滑回 side branch |

---

## 7. 阶段 B：训练自己的 Post-RAM Cleanup Adapter

### 7.1 目的

回答：

> 在 RamCore3 post-BN 输出后、DAv2 backbone 前，给模型一个可学习的 pseudo-RGB cleanup capacity，是否能在 depth loss 下学出有效修复？

这一阶段不加 feature loss，不加 noise augmentation，不加 RAM consistency，不加 edge loss。

核心实验只比较：

```text
with LoRA
without LoRA
```

---

## 8. Post-RAM Cleanup Adapter 设计

### 8.1 插入位置

```text
x_raw3 = RAW_dark_RGB16
x_ram, ram_features = self.ram_core.forward_with_features(x_raw3)
x_clean = PostRamCleanupAdapter(x_ram)
if raw_ram_rgb_tail == "tanh2p5":
    x_clean = phase1b_tanh_tail_squash(x_clean)
x_pad = self.spatial_adapter.pad_rgb(x_clean)
depth = self.dav2(x_pad)
```

注意：

```text
x_clean 经过 pad 后进入 DAv2 backbone
没有 ImageNet norm
```

canonical `C_dark(0608_2026)` 是 `raw_ram_rgb_tail=identity`，所以 cleanup 前后没有 tail 歧义。  
若未来使用 `tanh2p5`，默认插入点为 **RamCore3 输出 x3 之后、tail/pad 之前**。如果要测 tail 后 cleanup / denoiser，必须新增显式 experiment-semantic 参数，例如 `post_ram_operation_position=post_tail`，不能靠路径名或隐含默认决定。

### 8.2 第一版模块

推荐小残差 CNN：

```text
Adapter:
  Conv3x3(3 -> C)
  GroupNorm + ReLU
  ResBlock x N   # Conv/GN/ReLU/Conv/GN residual
  Conv3x3(C -> 3), zero-init

x_clean = x_ram + scale * Adapter(x_ram)
```

建议默认：

```text
C = 32
N = 4
scale = 0.1
last conv zero-init
```

模块内部使用 `GroupNorm`，不使用 `BatchNorm`。原因是当前链路已经有 RamCore3 末尾 BN，并且 eval 依赖固定 BN recalib / strict 协议；post-RAM cleanup 再引入一组 running stats 会增加新的协议耦合。`GroupNorm` 也与既有 `LocalDenoiseBranch` 的规范化方式一致。

可选轻量版本：

```text
C = 16
N = 3
scale = 0.1
```

可选放大版本：

```text
C = 32
N = 4
scale = 0.25
```

第一版不要使用 learnable scalar gate。  
原因不是证明 gate 设计错误，而是第一轮需要减少不可解释自由度。上一版 local residual 的 gate 基本贴着初始化，既可能是 gate 约束问题，也可能是任务本身没有收益；这里先用 fixed residual scale 做更直接的 capacity probe。

### 8.3 输出约束

第一版不做 hard clamp。

只做：

```text
last conv zero-init
fixed residual scale
```

如果出现数值爆炸，再考虑：

```text
x_clean = x_ram + scale * tanh(Adapter(x_ram))
```

但第一版先不要上 `tanh`，避免限制表达。

### 8.4 与 LocalDenoiseBranch 的关系

新模块和既有 `LocalDenoiseBranch` 都是小 CNN，确实有同构风险。计划中的合理动机不是“换一个相同模块再试一次”，而是：

```text
LocalDenoiseBranch:
  位于 RamCore3 内部，基于 x_rgb 与 post-BN x3_ram 做 gated/tanh internal residual。

PostRamCleanupAdapter:
  位于 RamCore3 外部，直接作用在 DAv2 实际看到的 post-BN input space，
  不再参与 RamCore3 内部 gate/tanh/local residual 机制。
```

因此阶段 B 必须把问题表述为“backbone input-space cleanup 是否有效”，而不是泛泛声称“denoise CNN 是否有效”。

---

## 9. 阶段 B loss 设计

第一版只用原始 depth loss：

```text
L_total = L_depth
```

明确不加：

```text
L_feat_middeep
L_ram
L_edge
L_smooth
L_delta
teacher depth loss
noise augmentation
external denoiser supervision
RAW_normal reconstruction loss
```

这样可以回答最干净的问题：

> 只增加 post-RAM cleanup capacity，在原 depth supervision 下是否有效？

如果这一版有效，再追加 feature loss；如果这一版无效，feature loss 才有必要重新讨论。

---

## 10. 阶段 B 训练矩阵

### 10.1 LoRA 主线

使用当前 canonical LoRA recipe。

初始化：

```text
C_dark LoRA checkpoint:
0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
```

训练参数：

```text
trainable:
  RAM
  PostRamCleanupAdapter
  LoRA
  DPT decoder

frozen:
  DAv2 backbone base weights
```

实验：

| ID | init | trainable | loss | epoch | 备注 |
|---|---|---|---|---:|---|
| T_LORA_s01 | C_dark LoRA | RAM + cleanup + LoRA + decoder | depth only | 10 | scale=0.1 |
| T_LORA_s025 | C_dark LoRA | RAM + cleanup + LoRA + decoder | depth only | 10 | scale=0.25，可选 |

如果只想先跑一个，跑：

```text
T_LORA_s01
```

### 10.2 no-LoRA 对照

不要直接从 LoRA checkpoint 禁用 LoRA。  
应使用 matching no-LoRA / decoder-only RAW_dark checkpoint 初始化。

初始化：

```text
C_dark_W0:
0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly
```

训练参数：

```text
trainable:
  RAM
  PostRamCleanupAdapter
  DPT decoder

frozen:
  DAv2 backbone
  no LoRA
```

实验：

| ID | init | trainable | loss | epoch | 备注 |
|---|---|---|---|---:|---|
| T_W0_s01 | C_dark_W0 | RAM + cleanup + decoder | depth only | 10 | scale=0.1 |
| T_W0_s025 | C_dark_W0 | RAM + cleanup + decoder | depth only | 10 | scale=0.25，可选 |

如果只想先跑一个，跑：

```text
T_W0_s01
```

### 10.3 seed-only baseline 前置

不能等 cleanup “看起来有效”后再补 matched-seed baseline。  
在进入阶段 B go/no-go 之前，必须先补或并行启动 1-2 个 no-cleanup matched-seed baseline，用同一 recipe、epoch、数据、eval 脚本量化 seed-only 分布。

`B0_LORA` 与 `T_LORA` 必须共享：

```text
init = C_dark
recipe
epoch
data / split
eval script and eval protocol
BN protocol
seed set
```

两者唯一实验语义差异应是：

```text
B0_LORA: post_ram_cleanup=none
T_LORA:  post_ram_cleanup=cnn
```

由于适用性校验需要，部分 inactive 字段会随开关变成 `n_a`，例如 `B0_LORA` 的 `post_ram_operation_position=n_a` 与 `T_LORA` 的 `pre_tail`。这类差异不算实验语义差异，但必须由 resolved config 明确记录。

`B0_W0` 与 `T_W0` 同理，必须共享 `init=C_dark_W0` 和同一 seed set。  
已有 `L0 seed42/123` 只有在 init、recipe、epoch、data、eval、BN 协议都与本轮 `T_LORA` 完全一致时，才能纳入 `B_lora_upper`；否则只能作为历史弱参考，必须重跑本轮 `B0_LORA`。

已有临时参考：

```text
LoRA L0 seed42  best_lod_d1 ≈ 0.8456
LoRA L0 seed123 best_lod_d1 ≈ 0.8502
seed-only 已观测波动 ≈ 0.0046

no-LoRA W0 baseline:
  C_dark_W0(I_dark) ≈ 0.8297
```

新增 baseline 建议：

| ID | 内容 | 目的 |
|---|---|---|
| B0_LORA_seedX | LoRA recipe，无 cleanup，新增 seed | 扩展 LoRA seed-only 分布 |
| B0_LORA_seedY | LoRA recipe，无 cleanup，新增 seed，可选 | 估计上沿是否高于 `0.8502` |
| B0_W0_seedX | W0 recipe，无 cleanup，新增 seed，可选 | no-LoRA cleanup 判据用 |

阶段 B 训练可以和这些 baseline 并行启动，但解释 `T_LORA/T_W0` 结果时必须等 baseline 分布补齐。单个 cleanup seed 超过 `0.8502` 只能叫 candidate，不能叫有效。

---

## 11. 阶段 B 成功判据

### 11.1 LoRA recipe

定义：

```text
B_lora_upper = max(best_lod_d1 of matched no-cleanup LoRA seeds)
B_lora_mean/std = matched no-cleanup LoRA seed distribution
```

| 结果 | 判断 |
|---|---|
| T_LORA D1 ≤ B_lora_upper | 不可分辨，不能声称有效 |
| T_LORA 单 seed > B_lora_upper | candidate；必须补 cleanup seed 或 strict 复核 |
| ≥2 个 T_LORA cleanup seeds 超过 B_lora_upper，且 AbsRel/RMSE 不恶化 | 训练口径有信号 |
| winner checkpoint 在 cross-eval strict 下超过 `C_dark(I_dark)` strict no-op | 最终可比较信号 |
| strict 结果接近/超过 strict RGB_dark 对照 | 强有效；需先补 strict RGB_dark 对照 |

当前 `0.8502` 只是 2-seed 弱估计上沿，不是固定真上沿。

### 11.2 no-LoRA recipe

不要和 LoRA recipe 直接横向比较。  
只看相对 matched no-LoRA W0 seed distribution 是否提升。

| 结果 | 判断 |
|---|---|
| T_W0 D1 ≤ B_w0_upper | 不可分辨 |
| T_W0 单 seed > B_w0_upper | candidate，需要复核 |
| ≥2 个 T_W0 cleanup seeds 超过 B_w0_upper | no-LoRA 口径有信号 |
| T_W0 strict 结果接近 LoRA strict baseline | 说明 cleanup adapter 在 frozen backbone 下可能有较强作用 |

### 11.3 诊断日志

每个训练实验必须记录：

```text
mean|adapter_delta| / mean|x_ram|
adapter_delta p50 / p95 / p99
x_ram p1/p50/p99
x_clean p1/p50/p99
train/val D1 gap
AbsRel/RMSE
best_lod_d1 的 seed / epoch / eval 口径
winner checkpoint 的 cross-eval strict D1/AbsRel/RMSE
高噪声 quartile D1
低噪声 quartile D1
```

建议保存 panel：

```text
RAW_dark visualization
x_ram
x_clean
delta heatmap
pred depth
error map
D1 map
```

---

## 12. 如果阶段 B 有效，后续补充实验

如果满足以下任一条件：

```text
1. ≥2 个 cleanup seeds 超过 matched no-cleanup seed 上沿
2. winner checkpoint 在 cross-eval strict 下超过 strict no-op baseline，且 AbsRel/RMSE 不恶化
```

则进入 formal ablation。`F0` 不再放在 go 之后才做；它应已在 §10.3 前置或并行完成。

| ID | 内容 | 目的 |
|---|---|---|
| F0 | matched-seed baseline，无 cleanup | 正式排除 seed；应前置或并行 |
| F1 | cleanup only，depth loss | 证明模块有效 |
| F2 | cleanup + feat_middeep | 检查 feature loss 是否额外有效 |
| F3 | cleanup + denoiser-as-init / pseudo target，可选 | 检查现成 denoiser 是否可作训练期初始化或伪标签来源 |
| F4 | cleanup + LoRA off/on | 正式评估 LoRA 交互 |

注意：`F3` 不是阶段 A 的 `post_ram_external_denoiser` 推理路径 wrapper。  
阶段 A 的 external denoiser 会改 DAv2 实际输入，必须与 trainable cleanup 互斥；`F3` 的 denoiser 只作为训练期初始化 / pseudo target 来源，不在本轮 schema 内。若进入 `F3`，需要另加显式参数，例如：

```text
--post-ram-cleanup-init-source none | external_denoiser
--post-ram-cleanup-pseudo-target none | external_denoiser
```

这些参数属于 go 之后扩展，不能复用 `--post-ram-external-denoiser`。

这时再考虑加：

```text
feature loss
noise augmentation
reliability gate
edge regularization
```

不要在第一轮就加。

---

## 13. 如果阶段 B 无效，停止条件

如果出现：

```text
T_LORA_s01 没有超过 matched LoRA seed-only 分布
T_W0_s01 没有超过 matched W0 seed-only 分布
winner strict cross-eval 也没有超过 strict no-op baseline
```

则不要继续扩大 post-RAM image-space cleanup。

下一步不能自动滑回 side branch。  
主线诉求是 input-level / post-RAM input-space cleanup；decoder-side feature adapter 或 RAW side branch 会回到“旁路中间特征”路线，和这个诉求有张力。因此若阶段 A/B 失败，应先停下来重新确认 fallback 方向，再决定是否转向以下方案。

### 13.1 Token-space denoise / adapter

```text
x_ram
  -> patch embedding
  -> token cleanup adapter
  -> DAv2 mid/deep layers
```

动机：之前 probe 显示，RAM/layer2 稍微接近 RAW_normal 并不能保证 layer5/8/11 接近。

### 13.2 Decoder-side RAW feature adapter

不再要求 backbone input 被清理干净，而是让 decoder 额外看到 RAW/RAM features。

```text
DAv2 backbone features
  + RAW/RAM side features
  -> DPT decoder
```

### 13.3 RAW side branch

让 RAM post-BN pseudo RGB 继续作为 DAv2 主输入，同时保留 RAW_dark 原始证据进入 decoder 或中间层。

```text
RAW_dark -> RamCore3 -> DAv2
RAW_dark -> raw side encoder -> fusion
```

---

## 14. 实现 checklist

### 14.1 新增配置参数

建议新增：

```text
--post-ram-cleanup none | cnn
--post-ram-cleanup-channels n_a | 16 | 32
--post-ram-cleanup-blocks n_a | 3 | 4
--post-ram-cleanup-scale n_a | 0.1 | 0.25
--post-ram-cleanup-norm n_a | groupnorm
--post-ram-cleanup-zero-init n_a | true
--post-ram-cleanup-lr n_a | 1e-5 | 5e-5
--post-ram-operation-position n_a | pre_tail | post_tail
```

外部 denoiser probe 可新增：

```text
--post-ram-external-denoiser none | identity | drunet | restormer | nafnet
--post-ram-denoiser-sigma n_a | 10 | 25 | 50
--post-ram-denoiser-alpha n_a | 0 | 0.25 | 0.5 | 1.0
--post-ram-denoiser-affine n_a | q001q999 | q01q99
--post-ram-denoiser-frozen n_a | true
```

### 14.2 代码插入位置

优先在 `RawRgb16Ram3DepthModel._prepare_dav2_input` 中插入，按真实命名对齐：

```text
x3, ram_features = self.ram_core.forward_with_features(x_raw)
x3 = self.post_ram_cleanup(x3)  # optional; default identity
if self.raw_ram_rgb_tail == "tanh2p5":
    x3 = phase1b_tanh_tail_squash(x3)
x_norm = self.spatial_adapter.pad_rgb(x3)
depth = self.dav2(x_norm)
```

`raw_to_base_rgb_ram3` 分支不是本轮 true-LOD 主线。它的代码结构和 `raw_rgb16_ram3` 不同：当前在 `forward` 中调用 `self.ram_core(...)`，没有 `_prepare_dav2_input` 插入点；若后续确实要支持，应在 `packed_bayer_to_base_rgb(x_raw)` 之后、`self.ram_core(...)` 输出 x3 之后改 `forward`。本轮优先只实现 `raw_rgb16_ram3`。

确保：

```text
RGB branch 不受影响
raw_rgb16_ram3 branch 可开关
checkpoint 兼容旧权重
post_ram_cleanup 默认 identity
canonical tail=identity 时无 tail 歧义
future tail=tanh2p5 时必须显式记录 post_ram_operation_position
```

### 14.3 初始化

```text
last conv weight = 0
last conv bias = 0
```

初始时：

```text
x_clean == x_ram
```

### 14.4 日志

在 train/eval 时写入：

```text
post_ram/enabled
post_ram/scale
post_ram/mean_abs_delta
post_ram/mean_abs_xram
post_ram/delta_ratio
post_ram/xram_p01_p50_p99
post_ram/xclean_p01_p50_p99
post_ram/eval_protocol
post_ram/bn_protocol
post_ram_external/clamp_low_ratio
post_ram_external/clamp_high_ratio
post_ram_external/clamp_total_ratio
```

### 14.5 resolved config 校验

新增参数属于 experiment-semantic 参数，必须在 resolved config 阶段中心化校验，而不是只依赖默认 identity：

```text
post_ram_cleanup=none:
  cleanup_channels/blocks/scale/norm/zero_init/lr 必须是 n_a

post_ram_cleanup=cnn:
  本轮 front_end 必须是 raw_rgb16_ram3
  cleanup_channels/blocks/scale/norm/zero_init/lr 必须显式有效
  cleanup_norm 必须是 groupnorm
  cleanup_scale > 0
  cleanup_lr > 0
  post_ram_external_denoiser 必须是 none

raw_to_base_rgb_ram3:
  非本轮主线；只有实际实现它自己的 forward 插入点后，才能扩展 schema 允许。

post_ram_external_denoiser=none:
  denoiser_sigma/alpha/affine/frozen 必须是 n_a

post_ram_external_denoiser=identity:
  只允许阶段 A sanity
  denoiser_sigma 必须是 n_a
  denoiser_alpha 和 denoiser_affine 必须显式设置
  post_ram_cleanup 必须是 none

post_ram_external_denoiser in {drunet, restormer, nafnet}:
  只允许阶段 A zero-eval
  post_ram_cleanup 必须是 none
  denoiser_alpha/affine/frozen 必须显式设置
  drunet 必须显式设置 denoiser_sigma
  restormer/nafnet 的 denoiser_sigma 必须是 n_a

本轮 schema 中，`post_ram_external_denoiser` 只表示 **阶段 A 的 denoiser-as-inference-wrapper**，也就是推理路径中的：

```text
x_ram -> external denoiser wrapper -> x_out -> DAv2
```

因此它必须与 `post_ram_cleanup=cnn` 互斥。

`F3` 的 denoiser-as-init / pseudo target 是另一种语义，不在本轮 schema 内；进入 go 之后扩展时应新增 `post_ram_cleanup_init_source` / `post_ram_cleanup_pseudo_target` 等参数，不能复用 `post_ram_external_denoiser`。

post_ram_cleanup 与 post_ram_external_denoiser 默认互斥。
post_ram_cleanup=none 且 post_ram_external_denoiser=none 时，post_ram_operation_position 必须是 n_a。
只要启用 cleanup 或 external denoiser，post_ram_operation_position 必须显式设置并写入 resolved_config。
raw_ram_rgb_tail != identity 时，必须明确说明 pre_tail / post_tail 的实验语义。
```

formal launch scripts 必须显式传入这些语义参数；不允许靠路径名、字符串匹配或“未传就是 identity”来决定实验含义。

---

## 15. 推荐执行顺序

### Step 0：前置 matched-seed baseline

在阶段 A 运行或等待外部 denoiser 集成时，并行补 no-cleanup baseline seed：

```text
B0_LORA_seedX
B0_LORA_seedY，可选
B0_W0_seedX，可选
```

目的：在训练 cleanup 前先知道 seed-only 分布，不再用 2-seed 弱上沿做 go/no-go。

`至少 3 个 seed` 的含义：

```text
如果历史 seed42/123 与本轮 init=C_dark、recipe、epoch、data、eval、BN 协议完全一致：
  可计入已有 2 个，再补 1 个新 seed。

如果任一项不一致：
  历史 seed 只作背景参考，本轮重新跑 3 个 B0_LORA seeds。
```

### Step 1：收集 post-BN 统计 + wrapper sanity

先跑：

```text
x_ram post-BN stats
EXT_NOOP_A0
EXT_ID_Q001
```

如果 no-op 不复现 strict baseline，停止修实现。  
如果 identity clamp 已明显损伤 D1，外部 denoiser 结果只能作为带 clamp 损失的诊断。

`EXT_NOOP_A0` 和 `EXT_ID_Q001` 也必须生成可视化。  
`EXT_NOOP_A0` 的 panel 用来确认 x_ram/x_out 完全一致；`EXT_ID_Q001` 的 panel 用来肉眼检查 clamp round-trip 是否截掉边缘、高光或局部高对比结构。

### Step 2：外部 denoiser quick probe

先跑最小集合：

```text
EXT_D1: DRUNet sigma=10, alpha=0.25, affine=q001/q999
EXT_D3: DRUNet sigma=25, alpha=0.25, affine=q001/q999
EXT_D4: DRUNet sigma=25, alpha=0.5, affine=q001/q999
EXT_R1: Restormer real denoise, alpha=0.25, affine=q001/q999
```

每个 EXT 跑完后立即执行对应 visualization job：

```text
EXT_D1 eval -> EXT_D1 viz
EXT_D3 eval -> EXT_D3 viz
EXT_D4 eval -> EXT_D4 viz
EXT_R1 eval -> EXT_R1 viz
```

只有 metric 和 viz 都完成，才把该 EXT 记入阶段 A 汇总表。

如果全部低于 no-op strict，不再扩大 EXT 矩阵。  
如果有任意一项 strict D1 ≥ no-op strict D1 + 0.003，且 AbsRel/RMSE 不恶化，再补完整 EXT 矩阵。

### Step 3：训练自己的 cleanup adapter

等 `B0_LORA` 至少有 3 个可比 seed 样本，或确认同 seed set 的 baseline 任务已在并行运行并会在解释前完成，再跑：

```text
T_LORA_s01
T_W0_s01
```

如果 `delta_ratio < 0.2%` 且 D1 不涨，补：

```text
T_LORA_s025
T_W0_s025
```

如果 `delta_ratio` 已经明显但 D1 不涨，不继续放大 scale。

### Step 4：统一 strict 复核

对训练 winner checkpoint 做 cross-eval strict：

```text
strict no-op baseline
strict cleanup winner
strict RGB_dark 对照，如要使用 0.855 附近结论
```

只有 strict 复核和 matched-seed 分布都支持时，才进入 formal ablation：

```text
cleanup + feature loss
cleanup with/without LoRA
long training
```

---

## 16. 当前最终决策

本轮执行目标不是马上证明完整方法，而是做一个清晰的架构判断：

```text
问题：
  DAv2 直接看到的 RamCore3 post-BN pseudo-RGB 是否可以通过 post-RAM cleanup 改善？

先用现成 denoiser：
  快速 zero-eval probe。

再训练自己的 adapter：
  只用 depth loss。
  同时做 LoRA / no-LoRA。
  不加 feature loss、不加 noise aug、不加其它正则。
```

如果有效，再进入正式 ablation；如果无效，停止 post-BN pseudo-RGB image-space cleanup，并重新确认是否转向 token-space / decoder-side / RAW side branch。

---

## 17. 具体实现步骤

这一节作为工程落地 checklist。前面 §0-§16 保持实验逻辑和判据；真正改代码、写脚本、跑 smoke / formal 时按本节执行。

### 17.0 执行约束

默认使用 conda 环境：

```text
conda activate dav3
```

所有 smoke 输出路径必须包含：

```text
codex_smoke / smoke / debug / tmp
```

smoke 成功后只删除这些明确临时产物；失败则保留并报告路径。  
formal 实验名必须以远端本地启动时间 `MMDD_HHMM` 开头。预计运行数小时以上的 baseline / cleanup train / external denoiser sweep 必须用 tmux，并报告 session、log、attach、tail。

### 17.1 第一步：配置 schema 与 resolved 校验

修改文件：

```text
finetune_stf/train.py
finetune_stf/config/resolved.py
```

实现内容：

```text
1. 在 argparse 增加 §14.1 的所有 post-ram 参数。
2. 在 ResolvedConfig 增加对应字段和 sources。
3. 在 resolved_config.json / config.json 写出所有 post-ram 字段。
4. 在 validate_resolved_config 里实现 §14.5 的互斥/依赖校验。
5. formal launch scripts 必须显式传入这些语义参数。
```

最低 smoke：

```text
cleanup=none + cleanup_scale=0.1 必须报错
external_denoiser=none + denoiser_alpha=0.25 必须报错
cleanup=cnn + external_denoiser=drunet 必须报错
cleanup=cnn + cleanup_lr=n_a 必须报错
front_end=raw_rgb16_ram3 + cleanup=cnn + groupnorm/pre_tail/scale/lr 全显式时通过
front_end=raw_to_base_rgb_ram3 + cleanup=cnn 本轮必须报错，除非已实现该分支 forward 插入点
```

### 17.2 第二步：模型插入点与 trainable cleanup

修改文件：

```text
finetune_stf/models/raw_ram.py
finetune_stf/train.py
```

实现内容：

```text
1. 在 raw_ram.py 增加 PostRamCleanupAdapter。
2. 模块结构使用 Conv + GroupNorm + ReLU + ResBlock + zero-init last conv。
3. RawRgb16Ram3DepthModel.__init__ 创建 self.post_ram_cleanup。
4. RawRgb16Ram3DepthModel._prepare_dav2_input 顺序改为：
   ram_core.forward_with_features -> post_ram_cleanup -> optional tail -> pad_rgb。
5. cleanup=none 时使用 nn.Identity，保证旧 checkpoint 兼容。
6. raw_to_base_rgb_ram3 非本轮主线；若按需支持，需改它自己的 forward，在 RamCore3 输出 x3 后插入，不要照搬 RawRgb16Ram3DepthModel._prepare_dav2_input。
```

optimizer 修改：

```text
1. 新增 optimizer group: post_ram_cleanup。
2. _optimizer_group_name_for_param 识别 post_ram_cleanup 参数。
3. _optimizer_lr_for_group 使用 --post-ram-cleanup-lr。
4. _required_optimizer_groups 在 cleanup=cnn 时要求 post_ram_cleanup 非空。
5. resolved_config 记录 optimizer group 的 trainable count。
```

最低 smoke：

```text
1. cleanup=none: old checkpoint strict=False load 不出现非预期 fatal missing。
2. cleanup=cnn + zero-init: 同一 batch 上 x_clean == x_ram，max_abs_diff 应接近 0。
3. cleanup=cnn: optimizer groups 包含 raw_front_end / post_ram_cleanup / decoder / lora，按 recipe 区分。
4. cleanup=cnn: post_ram_cleanup 参数 requires_grad=True，frozen DAv2 base 仍 frozen。
```

### 17.3 第三步：阶段 A 统计、外部 denoiser 与可视化工具

建议新增工具：

```text
tools/lod_postram_collect_stats.py
tools/lod_postram_external_eval.py
tools/lod_postram_visualize.py
tools/lod_postram_ext_summary.py
```

`lod_postram_collect_stats.py`：

```text
输入：run_dir, checkpoint, split=00Train, max_samples, output_json
输出：post-BN x_ram 的 per-channel q001/q01/q05/q50/q95/q99/q999/mean/std/min/max
要求：读取的是 ram_core.forward_with_features 的 x3，canonical tail=identity 时即 x3_ram
```

`lod_postram_external_eval.py`：

```text
输入：C_dark checkpoint、EXT_ID、denoiser 参数、affine stats、strict eval 配置
输出：metrics.json、per_sample.csv、artifacts/*.npz
要求：strict BN 协议对齐 tools/lod_raw_cross_eval.py
要求：EXT_NOOP_A0 完全 bypass clamp
要求：EXT_ID_Q001 经过 affine/clamp/inverse affine
```

`lod_postram_visualize.py`：

```text
输入：artifacts/*.npz、per_sample.csv、fixed_viz_samples.json
输出：viz/<EXT_ID>/panel_grid.png, per_sample/*.png, manifest.json
要求：所有 EXT 共用 fixed_viz_samples、显示范围和 colormap
```

`lod_postram_ext_summary.py`：

```text
输入：多个 EXT 的 metrics.json / manifest.json
输出：阶段 A summary.md / matrix.csv
要求：缺少 viz 的 EXT 标记 metrics_only_incomplete
```

最低 smoke：

```text
1. 用 max_samples=2 跑 collect_stats，输出 JSON 字段完整。
2. 用 max_samples=2 跑 EXT_NOOP_A0，x_out 与 x_ram diff 为 0，生成 artifacts。
3. 用 max_samples=2 跑 EXT_ID_Q001，输出 clamp ratio。
4. 用同两个样本生成 panel_grid.png / manifest.json。
5. smoke 成功后删除 /tmp/codex_smoke_lod_postram_*。
```

### 17.4 第四步：阶段 B 训练脚本

建议新增 formal queue scripts：

```text
finetune_stf/scripts/formal/<MMDD>_run_lod_postram_b0_seed_queue.sh
finetune_stf/scripts/formal/<MMDD>_run_lod_postram_cleanup_e10_queue.sh
```

`b0_seed_queue` 必须显式传入：

```text
--seed <seed>
--resume-from <C_dark 或 C_dark_W0 init checkpoint>
--post-ram-cleanup none
--post-ram-cleanup-channels n_a
--post-ram-cleanup-blocks n_a
--post-ram-cleanup-scale n_a
--post-ram-cleanup-norm n_a
--post-ram-cleanup-zero-init n_a
--post-ram-cleanup-lr n_a
--post-ram-external-denoiser none
--post-ram-denoiser-sigma n_a
--post-ram-denoiser-alpha n_a
--post-ram-denoiser-affine n_a
--post-ram-denoiser-frozen n_a
--post-ram-operation-position n_a
```

`cleanup_e10_queue` 必须显式传入：

```text
--seed <seed>
--resume-from <与对应 B0 相同的 init checkpoint>
--post-ram-cleanup cnn
--post-ram-cleanup-channels 32
--post-ram-cleanup-blocks 4
--post-ram-cleanup-scale 0.1
--post-ram-cleanup-norm groupnorm
--post-ram-cleanup-zero-init true
--post-ram-cleanup-lr 5e-5
--post-ram-external-denoiser none
--post-ram-denoiser-sigma n_a
--post-ram-denoiser-alpha n_a
--post-ram-denoiser-affine n_a
--post-ram-denoiser-frozen n_a
--post-ram-operation-position pre_tail
```

`b0_seed_queue` 与 `cleanup_e10_queue` 必须使用同一 seed 集，例如：

```text
SEEDS=(42 123 <new_seed>)
```

对每个 seed，`B0_LORA_seedS` 与 `T_LORA_seedS` 除 `post_ram_cleanup` 开关外，其它语义参数必须一致。W0 口径同理。

脚本必须写：

```text
resolved_config.json
optimizer group summary
train/eval log
train_viz 或 post_ram panel 输出路径
```

### 17.5 第五步：strict 复核和汇总

winner checkpoint 训练结束后，不直接读 training best 下结论。必须执行：

```text
1. strict no-op baseline
2. strict cleanup winner
3. strict RGB_dark 对照，如要使用 RGB_dark 结论
4. 阶段 B summary.md，显式区分 training-best 和 strict
```

汇总表至少包含：

```text
run_id
seed
epoch
eval_protocol
bn_protocol
best_lod_d1
strict_d1
AbsRel
RMSE
delta_ratio
clamp_ratio 或 n_a
viz_path
resolved_config_path
```

### 17.6 推荐落地顺序

按以下顺序实现，避免把实验时间浪费在未校验的配置上：

```text
1. resolved 参数 + 负例校验
2. PostRamCleanupAdapter + zero-init identity smoke
3. optimizer group + trainable 参数 smoke
4. collect_stats 工具
5. EXT_NOOP_A0 / EXT_ID_Q001 eval + viz smoke
6. EXT_D1 小样本 smoke
7. b0_seed_queue 脚本 audit
8. cleanup_e10_queue 脚本 audit
9. 正式 tmux 启动 B0 seeds / EXT probe / cleanup train
10. strict 复核 + summary
```

---

## 18. 执行记录

### 0610_0025 启动

- 当前目标：按本文档执行 post-RAM denoise / cleanup 计划，并把关键节点、命令、路径、结果持续记录回本节。
- 当前目录：`/home/caq/6666_raw/dav2_raw_0603`
- 默认环境：按项目指令使用 conda env `dav3`；预计超过数小时的正式 eval / train 使用 tmux。
- 已确认：当前 git 工作树已有未提交修改，后续只在现有状态上增量改动，不回滚用户已有修改。
- 初始实现顺序：resolved 参数与校验 -> `PostRamCleanupAdapter` 插入点 -> optimizer group -> 阶段 A 统计/eval/viz 工具 -> smoke -> tmux 正式实验。

### 0610_0037 Schema / 模型插入完成

- 修改文件：`finetune_stf/config/resolved.py`、`finetune_stf/config/__init__.py`、`finetune_stf/models/raw_ram.py`、`finetune_stf/train.py`。
- 已新增 resolved 参数：`post_ram_cleanup`、cleanup channels/blocks/scale/norm/zero_init/lr、`post_ram_external_denoiser`、denoiser sigma/alpha/affine/frozen、`post_ram_operation_position`。
- 已实现中心化校验：cleanup 与 external denoiser 互斥；inactive 字段必须为 `n_a/not_applicable`；`cleanup=cnn` 仅允许 `front_end=raw_rgb16_ram3`、`position=pre_tail`、`norm=groupnorm`、`zero_init=true`、正数 scale/lr。
- 已实现模型插入：`RawRgb16Ram3DepthModel._prepare_dav2_input` 中 `RamCore3.forward_with_features -> PostRamCleanupAdapter -> optional tail -> pad_rgb`；`cleanup=none` 为无参数 `Identity`，旧 checkpoint strict load 不新增 key。
- 已实现 optimizer group：新增 `post_ram_cleanup` group，使用 `--post-ram-cleanup-lr`；`cleanup=cnn` 时要求该 group 非空。
- Smoke 结果：
  - `python -m py_compile finetune_stf/config/resolved.py finetune_stf/config/__init__.py finetune_stf/models/raw_ram.py finetune_stf/train.py` 通过。
  - CLI schema smoke 通过：`cleanup=none + cleanup_scale=0.1`、`external=none + alpha=0.25`、`cleanup+external`、`cleanup=cnn + lr=n_a`、`raw_to_base + cleanup=cnn` 均按预期报错；`raw_rgb16_ram3 + cleanup=cnn + groupnorm/pre_tail/scale/lr` 通过。
  - 模型 smoke 通过：zero-init `max_abs(x_clean-x_ram)=0.000e+00`；optimizer groups 为 `raw_front_end=140903`、`post_ram_cleanup=76035`、`lora=73728`、`dav2_decoder=2728513`。
- 临时 smoke 路径：`/tmp/codex_smoke_postram_schema`、`/tmp/codex_smoke_postram_model` 已在成功后清理。

### 0610_0046 阶段 A 工具链完成

- 新增工具：
  - `tools/lod_postram_collect_stats.py`：收集 `RamCore3.forward_with_features` 的 post-BN `x3_ram` per-channel 统计；mean/std/min/max 为流式统计，分位数为 per-batch pixel sampling 估计。
  - `tools/lod_postram_external_eval.py`：strict eval 路径中插入 `x_ram -> wrapper -> DAv2`；支持 `EXT_NOOP_A0` bypass 和 `EXT_ID_Q001/Q01` affine clamp round-trip；`drunet/restormer/nafnet` 当前要求显式 `--denoiser-torchscript`，不会 fallback 到 median/blur 等手工滤波。
  - `tools/lod_postram_visualize.py`：从 artifact npz 生成 `panel_grid.png`、`per_sample/*.png`、`manifest.json`。
  - `tools/lod_postram_ext_summary.py`：汇总多个 EXT 的 `metrics.json` 为 CSV/Markdown。
- Smoke 路径：`finetune_stf/analysis/lod_postram_external/codex_smoke_stageA`。
- Smoke 内容：`max_samples=2` 串起 `collect_stats -> EXT_NOOP_A0 eval -> EXT_ID_Q001 eval -> 两组 viz -> summary`。
- Smoke 结果：通过；该 2-sample D1 只用于验证路径，不作实验结论。成功后已删除 `codex_smoke_stageA`。
- 当前外部 denoiser 约束：仓库内没有现成 DRUNet/Restormer/NAFNet 实现或权重，后续若要跑 `EXT_D*/EXT_R*/EXT_N*`，必须先提供/生成对应 TorchScript denoiser，或明确新增第三方模型接入步骤；工具不会把其它滤波器伪装成这些 denoiser。

### 0610_0051 阶段 A sanity 正式队列启动

- 启动命令：`GPU=0 EXT_SET=sanity bash finetune_stf/scripts/formal/0610_run_lod_postram_ext_sanity_queue.sh`
- tmux session：`0610_0050_lod_postram_ext_sanity_queue`
- 队列日志：`finetune_stf/logs/0610_0050_lod_postram_ext_sanity_queue.queue.log`
- attach：`tmux attach -t 0610_0050_lod_postram_ext_sanity_queue`
- monitor：`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_0050_lod_postram_ext_sanity_queue.queue.log`
- 输出目录：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity`
- 当前进度：已完成 00Train post-RAM stats，样本数 `1958`；已完成 `EXT_NOOP_A0` strict eval，`D1=0.845279`、`AbsRel=3.666699`、`samples=112`，与文档中的 strict no-op 基线预期一致；已完成 `EXT_ID_Q001` strict eval，`D1=0.845080`、`AbsRel=3.691696`、`samples=112`，相对 no-op 的 D1 变化约 `-0.000199`。

### 0610_0052 阶段 A sanity 完成

- tmux 结果：`0610_0050_lod_postram_ext_sanity_queue` 已正常退出，日志尾部为 `[DONE] 2026-06-10T00:51:37+08:00`。
- Summary：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/summary.md`
- Stats：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/xram_stats_00Train.json`
- `EXT_NOOP_A0`：
  - metrics：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/EXT_NOOP_A0/metrics.json`
  - strict eval：`D1=0.845279`、`AbsRel=3.666699`、`RMSE=24.878314`、`samples=112`
  - checkpoint meta：`best_lod_d1=0.8450678996860131`、`epoch=37`
  - viz manifest：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/EXT_NOOP_A0/viz/EXT_NOOP_A0/manifest.json`
- `EXT_ID_Q001`：
  - metrics：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/EXT_ID_Q001/metrics.json`
  - strict eval：`D1=0.845080`、`AbsRel=3.691696`、`RMSE=24.874620`、`samples=112`
  - 相对 no-op：`delta D1=-0.000200`
  - clamp / delta：`clamp_total_ratio=0.001919`、`delta_ratio=0.000936`、`delta_p95=0.005674`、`delta_p99=0.017932`
  - viz manifest：`finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/EXT_ID_Q001/viz/EXT_ID_Q001/manifest.json`
- 结论：strict no-op 基线与训练 checkpoint meta 一致，说明阶段 A strict eval 路径没有明显协议偏差；`EXT_ID_Q001` 的量化/clamp 损失很小，可继续进入阶段 B cleanup/baseline 训练。深度外部 denoiser 组仍阻塞于缺少 DRUNet/Restormer/NAFNet TorchScript/pretrained 接入，不能用手工滤波替代。

### 0610_0052 阶段 B LoRA e10 正式队列启动

- 启动前检查：GPU 0 为 `NVIDIA GeForce RTX 4090`，显存 `28 / 24564 MiB`，利用率 `0%`；`0610_run_lod_postram_cleanup_e10_queue.sh --audit` 对 6 个 LoRA run 全部通过。
- 启动命令：`GPU=0 RUN_MATRIX=lora SEEDS="42 123 777" bash finetune_stf/scripts/formal/0610_run_lod_postram_cleanup_e10_queue.sh`
- tmux session：`0610_0052_lod_postram_cleanup_e10_queue`
- 队列日志：`finetune_stf/logs/0610_0052_lod_postram_cleanup_e10_queue.queue.log`
- attach：`tmux attach -t 0610_0052_lod_postram_cleanup_e10_queue`
- monitor：`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_0052_lod_postram_cleanup_e10_queue.queue.log`
- 队列矩阵：`B0_LORA_s01` 与 `T_LORA_s01`，seeds `42 123 777`，每个 run `10` epochs。
- 语义参数：baseline 使用 `post_ram_cleanup=none` 且 cleanup 相关字段均为 `not_applicable`；cleanup 使用 `post_ram_cleanup=cnn`、`channels=32`、`blocks=4`、`scale=0.1`、`norm=groupnorm`、`zero_init=true`、`position=pre_tail`、`post_ram_cleanup_lr=5e-5`；两组均从 C_dark LoRA checkpoint 初始化并使用 `compatible` load。

### 0610_0054 阶段 B 队列重启

- `0610_0052_lod_postram_cleanup_e10_queue` 未进入训练：queue log 报错 `[ERROR] refusing to overwrite .../0610_0052_B0_LORA_s01_lod_postram_cleanup_e10_seed42`。
- 原因：脚本的 `--audit` 路径在 `run_one()` 中先执行了 `mkdir -p "${save}"` 并写 run log，再判断 `AUDIT_ONLY`，导致 audit 在 `finetune_stf/exp/0610_0052_*lod_postram_cleanup_e10*` 下创建了 6 个正式命名的空输出目录。按项目规则不删除这些 formal 命名目录，保留并在此记录其来源。
- 修复：`finetune_stf/scripts/formal/0610_run_lod_postram_cleanup_e10_queue.sh` 已调整为 audit 分支在创建目录和写 log 之前返回；`bash -n` 通过；复核 audit 未创建 `0610_0054_*` 的 exp/heavy/log 文件。
- 重启命令：`GPU=0 RUN_MATRIX=lora SEEDS="42 123 777" bash finetune_stf/scripts/formal/0610_run_lod_postram_cleanup_e10_queue.sh`
- 新 tmux session：`0610_0054_lod_postram_cleanup_e10_queue`
- 新队列日志：`finetune_stf/logs/0610_0054_lod_postram_cleanup_e10_queue.queue.log`
- attach：`tmux attach -t 0610_0054_lod_postram_cleanup_e10_queue`
- monitor：`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_0054_lod_postram_cleanup_e10_queue.queue.log`

### 0610_0054 阶段 B 第一个 run 健康检查

- 当前有效 run：`0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42`
- save：`finetune_stf/exp/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42`
- heavy：`/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42`
- run log：`finetune_stf/logs/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42.tmux.log`
- 配置健康检查：
  - `front_end=raw_rgb16_ram3`、`model_input_tensor=raw`、`raw_storage_format=raw_rgb16_png_3ch`
  - `post_ram_cleanup=none`，cleanup/denoiser 相关 inactive 字段均为 `not_applicable`
  - optimizer groups：`raw_front_end lr=5e-5 params=140903`、`lora lr=5e-5 params=73728`、`dav2_decoder lr=1e-5 params=2728513`
  - checkpoint load：`strict=compatible missing=0 unexpected=0 ram_keys=65 lora_keys=16 decoder_keys=64`
- init eval：
  - `pretrain_lod_val`：`D1=0.8451`、`AbsRel=3.6800`、`RMSE=24.8838`、`samples=112`
  - `pretrain_lod_train_proxy`：`D1=0.8667`、`AbsRel=2.5364`、`RMSE=23.1440`、`samples=112`
- 状态：已开始 `epoch=0/10`，说明 0054 队列已真正进入训练。

### 0610_0055 阶段 B 第一个 run epoch 0 完成

- run：`0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42`
- epoch：`0/10`
- train：`avg_loss=7.776771e-02`、`used_steps=244`、`elapsed=00:01:13`
- `lod_val`：`D1=0.8422`、`AbsRel=3.7576`、`RMSE=25.0127`、`samples=112`
- `lod_train_proxy`：`D1=0.8602`、`AbsRel=2.5295`、`RMSE=23.7176`、`samples=112`
- gap：`train_proxy_d1 - val_d1 = 0.0180`
- checkpoint：已保存 `best_model.pth` 与 `current_model.pth` 到 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/`，当前 `best_lod_d1=0.8422`。

### 0610_0059 阶段 B 第一个 run 中点状态

- run：`0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42`
- 已完成：epoch `0` 到 `4`，epoch `5` 已启动。
- 当前 best：`best_lod_d1=0.8444`，来自 epoch `1`，best checkpoint 路径仍为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/best_model.pth`。
- 近期指标：
  - epoch `2`：`lod_val D1=0.8436`、`AbsRel=3.4475`、`RMSE=24.7603`
  - epoch `3`：`lod_val D1=0.8434`、`AbsRel=3.5901`、`RMSE=24.6091`
  - epoch `4`：`lod_val D1=0.8400`、`AbsRel=3.5292`、`RMSE=24.8681`
- 状态：无错误行，current checkpoint 每个 epoch 正常保存，tmux 队列继续运行。

### 0610_0103 阶段 B 第一个 run 完成，cleanup run 启动

- 完成 run：`0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42`
- 类型：`B0_LORA_s01`，`post_ram_cleanup=none`
- 最终 best：epoch `9`，`best_lod_d1=0.8476`
- epoch `9` 指标：
  - `lod_val`：`D1=0.8476`、`AbsRel=3.7326`、`RMSE=24.3683`
  - `lod_train_proxy`：`D1=0.8711`、`AbsRel=2.4610`、`RMSE=22.5661`
  - gap：`0.0234`
  - train：`avg_loss=6.806563e-02`、`used_steps=244`、`elapsed=00:00:48`
- checkpoint 产物：
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/best_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/current_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/last_epoch_model.pth`
- 新 run：`0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42`
- 类型：`T_LORA_s01`，`post_ram_cleanup=cnn`
- cleanup run 健康检查：
  - optimizer groups 包含 `post_ram_cleanup lr=5e-5 params=76035 tensors=29`
  - resolved post-RAM：`cleanup=cnn channels=32 blocks=4 scale=0.1 norm=groupnorm zero_init=true cleanup_lr=5e-5 external=none position=pre_tail`
  - checkpoint load：`strict=compatible missing=29 unexpected=0`；29 个 missing keys 均为新增 `post_ram_cleanup.*` 参数，符合预期。
  - init eval 与 baseline 一致：`pretrain_lod_val D1=0.8451`、`AbsRel=3.6800`、`RMSE=24.8838`；`pretrain_lod_train_proxy D1=0.8667`、`AbsRel=2.5364`、`RMSE=23.1440`。
  - 状态：已开始 `epoch=0/10`。

### 0610_0105 cleanup seed42 epoch 0 完成

- run：`0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42`
- epoch：`0/10`
- train：`avg_loss=7.767078e-02`、`used_steps=244`、`elapsed=00:02:01`
- `lod_val`：`D1=0.8430`、`AbsRel=3.7580`、`RMSE=24.9878`
- `lod_train_proxy`：`D1=0.8589`、`AbsRel=2.5378`、`RMSE=23.9554`
- gap：`0.0159`
- checkpoint：已保存 best/current 到 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/`，当前 `best_lod_d1=0.8430`。
- 对照：baseline seed42 epoch 0 为 `D1=0.8422`，cleanup 首轮略高；后续仍需看 10 epoch best 与 strict eval。

### 0610_0112 cleanup seed42 中点状态

- run：`0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42`
- 已完成：epoch `0` 到 `4`，epoch `5` 已启动。
- 当前 best：`best_lod_d1=0.8436`，来自 epoch `1`。
- 近期指标：
  - epoch `1`：`lod_val D1=0.8436`、`AbsRel=3.4625`、`RMSE=24.8043`
  - epoch `2`：`lod_val D1=0.8418`、`AbsRel=3.4263`、`RMSE=24.6734`
  - epoch `3`：`lod_val D1=0.8423`、`AbsRel=3.4842`、`RMSE=24.6464`
  - epoch `4`：`lod_val D1=0.8409`、`AbsRel=3.4853`、`RMSE=24.6777`
- 状态：无错误行，current checkpoint 每个 epoch 正常保存；与 seed42 baseline 当前 best `0.8476` 相比，training-best 暂时落后，仍需等后半段和 strict eval。

### 0610_0118 cleanup seed42 后半程状态

- run：`0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42`
- 已完成：epoch `0` 到 `8`，epoch `9` 已启动。
- 当前 best：`best_lod_d1=0.8471`，来自 epoch `7`，best checkpoint 路径为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/best_model.pth`。
- 后半程指标：
  - epoch `5`：`lod_val D1=0.8446`、`AbsRel=3.7413`、`RMSE=24.6232`；`lod_train_proxy D1=0.8676`
  - epoch `6`：`lod_val D1=0.8468`、`AbsRel=3.6829`、`RMSE=24.4524`；`lod_train_proxy D1=0.8682`
  - epoch `7`：`lod_val D1=0.8471`、`AbsRel=3.8236`、`RMSE=24.4779`；`lod_train_proxy D1=0.8697`
  - epoch `8`：`lod_val D1=0.8468`、`AbsRel=3.7895`、`RMSE=24.4141`；`lod_train_proxy D1=0.8699`
- 状态：无错误行，current checkpoint 正常保存；cleanup seed42 已接近 baseline seed42 best `0.8476`，最终对照仍以 epoch9 收尾和后续 strict eval 为准。

### 0610_0119 cleanup seed42 完成，seed123 baseline 启动

- 完成 run：`0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42`
- 类型：`T_LORA_s01`，`post_ram_cleanup=cnn`
- 最终 best：epoch `9`，`best_lod_d1=0.8485`
- epoch `9` 指标：
  - `lod_val`：`D1=0.8485`、`AbsRel=3.7640`、`RMSE=24.2617`
  - `lod_train_proxy`：`D1=0.8705`、`AbsRel=2.4108`、`RMSE=22.6338`
  - gap：`0.0220`
  - train：`avg_loss=6.750007e-02`、`used_steps=244`、`elapsed=00:01:33`
- checkpoint 产物：
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/best_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/current_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/last_epoch_model.pth`
- seed42 training-best 对照：cleanup `0.8485` vs baseline `0.8476`，cleanup 高 `+0.0009`；最终结论仍需看所有 seeds 和 strict eval。
- 新 run：`0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123`
- 类型：`B0_LORA_s01`，`post_ram_cleanup=none`
- save：`finetune_stf/exp/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123`
- heavy：`/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123`
- 健康检查：
  - optimizer groups：`raw_front_end lr=5e-5 params=140903`、`lora lr=5e-5 params=73728`、`dav2_decoder lr=1e-5 params=2728513`
  - resolved post-RAM：`cleanup=none`，cleanup/denoiser 相关字段均为 `not_applicable`
  - checkpoint load：`strict=compatible missing=0 unexpected=0 ram_keys=65 lora_keys=16 decoder_keys=64`
  - init eval：`pretrain_lod_val D1=0.8451`、`AbsRel=3.6800`、`RMSE=24.8838`；`pretrain_lod_train_proxy D1=0.8550`、`AbsRel=2.4543`、`RMSE=22.9028`
  - 状态：已开始 `epoch=0/10`。

### 0610_0120 seed123 baseline epoch 0 完成

- run：`0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123`
- epoch：`0/10`
- train：`avg_loss=7.719521e-02`、`used_steps=244`、`elapsed=00:01:13`
- `lod_val`：`D1=0.8448`、`AbsRel=3.9497`、`RMSE=25.1204`
- `lod_train_proxy`：`D1=0.8525`、`AbsRel=2.4495`、`RMSE=22.9171`
- gap：`0.0077`
- checkpoint：已保存 best/current 到 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/`，当前 `best_lod_d1=0.8448`。
- 状态：无错误行，epoch `1` 已启动。

### 0610_0124 seed123 baseline 中点状态

- run：`0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123`
- 已完成：epoch `0` 到 `4`，epoch `5` 已启动。
- 当前 best：`best_lod_d1=0.8466`，来自 epoch `4`，best checkpoint 路径为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth`。
- 近期指标：
  - epoch `1`：`lod_val D1=0.8458`、`AbsRel=3.7368`、`RMSE=24.9509`
  - epoch `2`：`lod_val D1=0.8453`、`AbsRel=3.7667`、`RMSE=24.7786`
  - epoch `3`：`lod_val D1=0.8453`、`AbsRel=3.7930`、`RMSE=24.8911`
  - epoch `4`：`lod_val D1=0.8466`、`AbsRel=3.6379`、`RMSE=24.6483`；`lod_train_proxy D1=0.8603`
- 状态：无错误行，current checkpoint 每个 epoch 正常保存。

### 0610_0128 seed123 baseline 完成，cleanup seed123 启动

- 完成 run：`0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123`
- 类型：`B0_LORA_s01`，`post_ram_cleanup=none`
- 最终 best：epoch `9`，`best_lod_d1=0.8486`
- epoch `9` 指标：
  - `lod_val`：`D1=0.8486`、`AbsRel=3.8619`、`RMSE=24.3454`
  - `lod_train_proxy`：`D1=0.8595`、`AbsRel=2.3699`、`RMSE=22.1586`
  - gap：`0.0109`
  - train：`avg_loss=7.067684e-02`、`used_steps=244`、`elapsed=00:00:48`
- checkpoint 产物：
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/current_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/last_epoch_model.pth`
- 新 run：`0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- 类型：`T_LORA_s01`，`post_ram_cleanup=cnn`
- save：`finetune_stf/exp/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- heavy：`/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- cleanup run 健康检查：
  - optimizer groups 包含 `post_ram_cleanup lr=5e-5 params=76035 tensors=29`
  - resolved post-RAM：`cleanup=cnn channels=32 blocks=4 scale=0.1 norm=groupnorm zero_init=true cleanup_lr=5e-5 external=none position=pre_tail`
  - checkpoint load：`strict=compatible missing=29 unexpected=0`；29 个 missing keys 均为新增 `post_ram_cleanup.*` 参数，符合预期。
  - init eval：`pretrain_lod_val D1=0.8451`、`AbsRel=3.6777`、`RMSE=24.8838`；`pretrain_lod_train_proxy D1=0.8550`、`AbsRel=2.4556`、`RMSE=22.9028`
  - 状态：已开始 `epoch=0/10`。

### 0610_0130 cleanup seed123 epoch 0 完成

- run：`0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- epoch：`0/10`
- train：`avg_loss=7.705152e-02`、`used_steps=244`、`elapsed=00:02:01`
- `lod_val`：`D1=0.8439`、`AbsRel=3.9699`、`RMSE=25.1188`
- `lod_train_proxy`：`D1=0.8533`、`AbsRel=2.3753`、`RMSE=22.9356`
- gap：`0.0094`
- checkpoint：已保存 best/current 到 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/`，当前 `best_lod_d1=0.8439`。
- 对照：baseline seed123 epoch 0 为 `D1=0.8448`，cleanup 首轮略低；后续仍需看 10 epoch best 与 strict eval。

### 0610_0137 cleanup seed123 中点状态

- run：`0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- 已完成：epoch `0` 到 `4`，epoch `5` 已启动。
- 当前 best：`best_lod_d1=0.8502`，来自 epoch `4`，best checkpoint 路径为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth`。
- 近期指标：
  - epoch `1`：`lod_val D1=0.8462`、`AbsRel=3.7760`、`RMSE=24.9340`
  - epoch `2`：`lod_val D1=0.8453`、`AbsRel=3.8092`、`RMSE=24.6925`
  - epoch `3`：`lod_val D1=0.8455`、`AbsRel=3.7260`、`RMSE=24.8180`
  - epoch `4`：`lod_val D1=0.8502`、`AbsRel=3.8821`、`RMSE=24.4135`；`lod_train_proxy D1=0.8625`
- 对照：cleanup seed123 当前 training-best `0.8502` 已高于 baseline seed123 final `0.8486`，差值 `+0.0016`；最终结论仍以 epoch9 收尾和 strict eval 为准。
- 状态：无错误行，current checkpoint 每个 epoch 正常保存。

### 0610_0143 cleanup seed123 后半程状态

- run：`0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- 已完成：epoch `0` 到 `8`，epoch `9` 已启动。
- 当前 best：`best_lod_d1=0.8504`，来自 epoch `6`，best checkpoint 路径为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth`。
- 后半程指标：
  - epoch `5`：`lod_val D1=0.8494`、`AbsRel=3.8537`、`RMSE=24.3500`；`lod_train_proxy D1=0.8593`
  - epoch `6`：`lod_val D1=0.8504`、`AbsRel=3.6696`、`RMSE=24.2705`；`lod_train_proxy D1=0.8621`
  - epoch `7`：`lod_val D1=0.8501`、`AbsRel=3.7621`、`RMSE=24.3734`；`lod_train_proxy D1=0.8624`
  - epoch `8`：`lod_val D1=0.8502`、`AbsRel=3.7870`、`RMSE=24.2952`；`lod_train_proxy D1=0.8638`
- 对照：cleanup seed123 当前 training-best `0.8504` 高于 baseline seed123 final `0.8486`，差值 `+0.0018`；最终仍需等待 epoch9 收尾和 strict eval。
- 状态：无错误行，current checkpoint 正常保存；tmux 队列继续运行。

### 0610_0144 cleanup seed123 完成，seed777 baseline 启动

- 完成 run：`0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`
- 类型：`T_LORA_s01`，`post_ram_cleanup=cnn`
- 最终 best：epoch `9`，`best_lod_d1=0.8508`
- epoch `9` 指标：
  - `lod_val`：`D1=0.8508`、`AbsRel=3.7581`、`RMSE=24.1613`
  - `lod_train_proxy`：`D1=0.8636`、`AbsRel=2.3639`、`RMSE=22.0417`
  - gap：`0.0128`
  - train：`avg_loss=6.988147e-02`、`used_steps=244`、`elapsed=00:01:33`
- checkpoint 产物：
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/current_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/last_epoch_model.pth`
- seed123 training-best 对照：cleanup `0.8508` vs baseline `0.8486`，cleanup 高 `+0.0022`；最终结论仍需看 seed777 和 strict eval。
- 新 run：`0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777`
- 类型：`B0_LORA_s01`，`post_ram_cleanup=none`
- save：`finetune_stf/exp/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777`
- heavy：`/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777`
- 健康检查：
  - optimizer groups：`raw_front_end lr=5e-5 params=140903`、`lora lr=5e-5 params=73728`、`dav2_decoder lr=1e-5 params=2728513`
  - resolved post-RAM：`cleanup=none`，cleanup/denoiser 相关字段均为 `not_applicable`
  - checkpoint load：`strict=compatible missing=0 unexpected=0 ram_keys=65 lora_keys=16 decoder_keys=64`
  - init eval：`pretrain_lod_val D1=0.8451`、`AbsRel=3.6800`、`RMSE=24.8838`；`pretrain_lod_train_proxy D1=0.8522`、`AbsRel=3.3047`、`RMSE=23.7058`
  - 状态：已开始 `epoch=0/10`。

### 0610_0146 seed777 baseline epoch 0 完成

- run：`0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777`
- epoch：`0/10`
- train：`avg_loss=7.456960e-02`、`used_steps=244`、`elapsed=00:01:13`
- `lod_val`：`D1=0.8389`、`AbsRel=4.1363`、`RMSE=25.4037`
- `lod_train_proxy`：`D1=0.8469`、`AbsRel=3.6591`、`RMSE=24.1157`
- gap：`0.0080`
- checkpoint：已保存 best/current 到 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/`，当前 `best_lod_d1=0.8389`。
- 状态：无错误行，epoch `1` 已启动。

### 0610_0149 seed777 baseline 中点状态

- run：`0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777`
- 已完成：epoch `0` 到 `4`，epoch `5` 已启动。
- 当前 best：`best_lod_d1=0.8442`，来自 epoch `1`，best checkpoint 路径为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/best_model.pth`。
- 近期指标：
  - epoch `1`：`lod_val D1=0.8442`、`AbsRel=4.2675`、`RMSE=25.0593`；`lod_train_proxy D1=0.8538`
  - epoch `2`：`lod_val D1=0.8439`、`AbsRel=3.8814`、`RMSE=24.8661`；`lod_train_proxy D1=0.8508`
  - epoch `3`：`lod_val D1=0.8439`、`AbsRel=3.8136`、`RMSE=24.8666`；`lod_train_proxy D1=0.8515`
  - epoch `4`：`lod_val D1=0.8433`、`AbsRel=3.5862`、`RMSE=24.9496`；`lod_train_proxy D1=0.8543`
- 状态：无错误行，current checkpoint 每个 epoch 正常保存；baseline seed777 中点 best 低于 seed42/seed123 baseline 最终 best，后半程继续观察。

### 0610_0154 seed777 baseline 完成，cleanup seed777 启动

- 完成 run：`0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777`
- 类型：`B0_LORA_s01`，`post_ram_cleanup=none`
- 最终 best：epoch `9`，`best_lod_d1=0.8452`
- epoch `9` 指标：
  - `lod_val`：`D1=0.8452`、`AbsRel=3.8084`、`RMSE=24.5169`
  - `lod_train_proxy`：`D1=0.8591`、`AbsRel=3.4850`、`RMSE=23.1047`
  - gap：`0.0140`
  - train：`avg_loss=7.327370e-02`、`used_steps=244`、`elapsed=00:00:48`
- checkpoint 产物：
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/best_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/current_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/last_epoch_model.pth`
- 新 run：`0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777`
- 类型：`T_LORA_s01`，`post_ram_cleanup=cnn`
- save：`finetune_stf/exp/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777`
- heavy：`/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777`
- cleanup run 健康检查：
  - optimizer groups 包含 `post_ram_cleanup lr=5e-5 params=76035 tensors=29`
  - resolved post-RAM：`cleanup=cnn channels=32 blocks=4 scale=0.1 norm=groupnorm zero_init=true cleanup_lr=5e-5 external=none position=pre_tail`
  - checkpoint load：`strict=compatible missing=29 unexpected=0`；29 个 missing keys 均为新增 `post_ram_cleanup.*` 参数，符合预期。
  - init eval：`pretrain_lod_val D1=0.8451`、`AbsRel=3.6800`、`RMSE=24.8838`；`pretrain_lod_train_proxy D1=0.8522`、`AbsRel=3.3047`、`RMSE=23.7058`
  - 状态：已开始 `epoch=0/10`。

### 0610_0156 cleanup seed777 epoch 0 完成

- run：`0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777`
- epoch：`0/10`
- train：`avg_loss=7.448039e-02`、`used_steps=244`、`elapsed=00:02:01`
- `lod_val`：`D1=0.8405`、`AbsRel=3.8818`、`RMSE=25.2981`
- `lod_train_proxy`：`D1=0.8462`、`AbsRel=3.9791`、`RMSE=24.2741`
- gap：`0.0057`
- checkpoint：已保存 best/current 到 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/`，当前 `best_lod_d1=0.8405`。
- 对照：baseline seed777 epoch 0 为 `D1=0.8389`，cleanup 首轮略高；baseline seed777 final best 为 `0.8452`，后续仍需看 10 epoch best 与 strict eval。

### 0610_0202 cleanup seed777 中点状态

- run：`0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777`
- 已完成：epoch `0` 到 `4`，epoch `5` 已启动。
- 当前 best：`best_lod_d1=0.8463`，来自 epoch `1`，best checkpoint 路径为 `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/best_model.pth`。
- 近期指标：
  - epoch `1`：`lod_val D1=0.8463`、`AbsRel=4.3222`、`RMSE=25.0381`；`lod_train_proxy D1=0.8487`
  - epoch `2`：`lod_val D1=0.8415`、`AbsRel=4.0075`、`RMSE=24.9559`；`lod_train_proxy D1=0.8491`
  - epoch `3`：`lod_val D1=0.8451`、`AbsRel=3.8525`、`RMSE=24.7029`；`lod_train_proxy D1=0.8496`
  - epoch `4`：`lod_val D1=0.8461`、`AbsRel=3.5203`、`RMSE=24.5887`；`lod_train_proxy D1=0.8514`
- 对照：cleanup seed777 当前 training-best `0.8463` 高于 baseline seed777 final `0.8452`，差值 `+0.0011`；最终仍以 epoch9 收尾和 strict eval 为准。
- 状态：无错误行，current checkpoint 每个 epoch 正常保存。

### 0610_0209 cleanup seed777 完成，阶段 B 队列完成

- 完成 run：`0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777`
- 类型：`T_LORA_s01`，`post_ram_cleanup=cnn`
- 最终 best：epoch `1`，`best_lod_d1=0.8463`
- epoch `9` 指标：
  - `lod_val`：`D1=0.8454`、`AbsRel=3.7258`、`RMSE=24.3988`
  - `lod_train_proxy`：`D1=0.8582`、`AbsRel=3.4159`、`RMSE=23.1217`
  - gap：`0.0128`
  - train：`avg_loss=7.315483e-02`、`used_steps=244`、`elapsed=00:01:33`
- checkpoint 产物：
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/best_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/current_model.pth`
  - `/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/last_epoch_model.pth`
- seed777 training-best 对照：cleanup `0.8463` vs baseline `0.8452`，cleanup 高 `+0.0011`。
- 阶段 B 队列状态：`0610_0054_lod_postram_cleanup_e10_queue` 已于 `2026-06-10T02:09:37+08:00` 完成并退出；队列日志 `finetune_stf/logs/0610_0054_lod_postram_cleanup_e10_queue.queue.log` 无 `[ERROR]`。
- 当前解释边界：以上仍是训练期 `best_lod_d1` 口径，最终比较必须进入 cross-eval strict 复核。

### 0610_0212 strict 复核启动

- strict eval 脚本：`finetune_stf/scripts/formal/0610_run_lod_postram_stageB_strict_eval.sh`
- 输出目录：`finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict`
- audit：已通过，所有 run 的 `config.json`、`resolved_config.json`、`best_model.pth` 以及 Stage A no-op metrics 均存在。
- eval cell：`M_DD strict`，固定 `I_dark`，`strict_no_bn_recalib`，不使用 BN recalib。
- 待跑 checkpoint：
  - `REF_C_DARK_0608`
  - `B0_seed42` / `T_seed42`
  - `B0_seed123` / `T_seed123`
  - `B0_seed777` / `T_seed777`
- 工具补充：`tools/lod_raw_cross_eval.py` 已补齐 post-RAM 参数转发，并在 strict row 中记录 `post_ram_delta_ratio` 等 debug 均值；2-sample cleanup smoke 已通过并清理 `/tmp/codex_smoke_postram_cross_eval_cleanup`。

### 0610_0213 strict 复核完成，本轮计划完成

- strict 输出目录：`finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict`
- summary：
  - `finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/summary.md`
  - `finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/stageB_strict_summary.csv`
  - `finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/stageB_paired_deltas.csv`
- Stage A strict no-op reference：`D1=0.845279`、`AbsRel=3.666699`、`RMSE=24.878314`
- strict rows：
  - `B0_seed42`：`D1=0.847592`、`AbsRel=3.666920`、`RMSE=24.367533`
  - `T_seed42`：`D1=0.848504`、`AbsRel=3.754150`、`RMSE=24.267365`、`delta_ratio=0.091457`
  - `B0_seed123`：`D1=0.848470`、`AbsRel=3.860734`、`RMSE=24.346629`
  - `T_seed123`：`D1=0.850708`、`AbsRel=3.746498`、`RMSE=24.166887`、`delta_ratio=0.102271`
  - `B0_seed777`：`D1=0.845085`、`AbsRel=3.778942`、`RMSE=24.514426`
  - `T_seed777`：`D1=0.846413`、`AbsRel=4.324509`、`RMSE=25.026583`、`delta_ratio=0.039808`
- matched strict deltas：
  - seed `42`：`+0.000912`
  - seed `123`：`+0.002238`
  - seed `777`：`+0.001329`
  - mean：`+0.001493`
- winner：`0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`，strict `D1=0.850708`，training-best `0.850784`。
- 判定：
  - 有 matched-seed D1 正信号：3/3 cleanup strict D1 均高于对应 baseline。
  - 但 go-gate 未完整通过：只有 seed123 cleanup 超过 matched baseline training-best 上沿 `0.848650`，且 winner 相对 Stage A strict no-op reference 的 AbsRel 从 `3.666699` 变为 `3.746498`，不满足 “D1 提升且 AbsRel/RMSE 不恶化” 的完整强条件。
  - 因此本轮不自动启动 formal ablation；结论是 post-RAM cleanup 是可复核的弱正 D1 candidate，但还不是可直接扩展的强结论。
- 本轮文档范围内任务状态：schema / model / optimizer / smoke / Stage A sanity / Stage B matched-seed train / strict eval / summary 均完成。

### 0610_1122 Stage B cleanup 可视化补充

- 可视化对象：winner checkpoint `0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth`
- 输出目录：`finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz`
- 总览图：`finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz/viz/T_seed123_cleanup/panel_grid.png`
- 单样本 panel：`finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz/viz/T_seed123_cleanup/per_sample/`
- artifact dump 工具：`tools/lod_postram_cleanup_artifacts.py`
- 注意：本可视化只取前 `16` 个 valid 样本并保存前 `8` 个 panel，用于诊断 `x_ram -> x_clean` 的形态变化；其中局部 `D1=0.789076` 不代表全量 strict，winner 全量 strict 仍以 `0.850708` 为准。
- 观察：`x_clean` 相比 `x_ram` 的变化不是简单随机噪声抹平；`mean delta` / `abs(delta)` 显示出沿物体边界、局部亮暗结构和竖向条纹的结构化修正，更像 post-RAM pseudo-RGB 分布/纹理校正，而不是传统图像去噪。

### 0610_1132 Stage A DRUNet external denoise minimal probe 完成

- 目的：补齐此前未实际尝试的 external denoise。选择 `DRUNet / DPIR`，因为它支持显式 `sigma` 控制，最匹配本计划的 `sigma={10,25,50}/255` 快速强弱扫描。
- 模型来源：
  - DPIR 官方下载脚本指向 `cszn/KAIR` release 的 `drunet_color.pth`。
  - 下载权重：`/mnt/drive/3333_raw/0000_exp_ckpt/external_denoisers/DPIR/drunet_color.pth`
  - sha256：`479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`
  - TorchScript：`/mnt/drive/3333_raw/0000_exp_ckpt/external_denoisers/DPIR/drunet_color_pad8.ts`
- 新增导出工具：`tools/export_drunet_torchscript.py`
  - 结构：`UNetRes(in_nc=4, out_nc=3, nc=[64,128,256,512], nb=4)`。
  - wrapper：内部对输入 pad 到 `8` 的倍数，输出裁回原尺寸，适配 LOD crop 非 8 倍尺寸。
  - 校验：`64x64` 和 `65x67` eager/TorchScript max_abs 均为 `0.000e+00`。
- DRUNet eval smoke：
  - 临时路径：`finetune_stf/analysis/lod_postram_external/codex_smoke_drunet_ext`
  - 内容：`EXT_D1` 单样本，`sigma=10`、`alpha=0.25`、`q001q999`
  - 结果：成功串起 TorchScript、sigma channel、affine wrapper 和 strict eval；成功后已删除该 `codex_smoke` 临时目录。
- 正式队列：
  - tmux session：`0610_1132_lod_postram_ext_drunet_queue`
  - 队列日志：`finetune_stf/logs/0610_1132_lod_postram_ext_drunet_queue.queue.log`
  - attach：`tmux attach -t 0610_1132_lod_postram_ext_drunet_queue`
  - monitor：`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_1132_lod_postram_ext_drunet_queue.queue.log`
  - 状态：已正常退出；`tmux has-session` 返回 session 不存在。
- 输出目录：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal`
- summary：
  - `finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/summary.md`
  - `finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/summary.csv`
- 可视化：
  - `EXT_NOOP_A0`：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_NOOP_A0/viz/EXT_NOOP_A0/panel_grid.png`
  - `EXT_ID_Q001`：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_ID_Q001/viz/EXT_ID_Q001/panel_grid.png`
  - `EXT_D1`：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D1/viz/EXT_D1/panel_grid.png`
  - `EXT_D3`：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D3/viz/EXT_D3/panel_grid.png`
  - `EXT_D4`：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D4/viz/EXT_D4/panel_grid.png`
- strict no-op / identity 复现：
  - `EXT_NOOP_A0`：`D1=0.845279`、`AbsRel=3.666699`、`RMSE=24.878314`
  - `EXT_ID_Q001`：`D1=0.845080`、`AbsRel=3.691696`、`RMSE=24.874620`、`delta D1=-0.000200`
  - `q001/q999` clamp ratio：`0.001919`，round-trip 损伤仍很小，外部 denoise 结果可解释。
- DRUNet cells：
  - `EXT_D1`：`sigma=10`、`alpha=0.25`、`D1=0.841947`、`AbsRel=3.789210`、`RMSE=25.035134`、`delta D1=-0.003332`、`delta_ratio=0.034174`
  - `EXT_D3`：`sigma=25`、`alpha=0.25`、`D1=0.846295`、`AbsRel=3.593061`、`RMSE=24.742452`、`delta D1=+0.001016`、`delta_ratio=0.044219`
  - `EXT_D4`：`sigma=25`、`alpha=0.5`、`D1=0.844203`、`AbsRel=3.689165`、`RMSE=24.803842`、`delta D1=-0.001076`、`delta_ratio=0.088437`
- 判定：
  - `EXT_D3` 是唯一正向 cell，同时 D1、AbsRel、RMSE 都优于 no-op，但提升只有 `+0.001016 D1`，低于本计划阶段 A 强信号门槛 `+0.003`。
  - `EXT_D1` 和 `EXT_D4` 为负信号，说明现成 sRGB/高斯先验 DRUNet 对 post-BN pseudo RGB 的作用很敏感，alpha 增大后容易破坏 DAv2 已适应的输入域。
  - 当前结论不是“外部 denoise 成功”，而是“DRUNet 有一个弱正诊断 cell，但整体不构成强 go-gate”；它支持继续把 cleanup 做成可训练、小残差、受 depth loss 约束的 post-RAM adapter，而不支持把现成 DRUNet 作为正式推理模块。
