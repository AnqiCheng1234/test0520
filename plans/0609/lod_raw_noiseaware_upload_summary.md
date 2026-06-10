# LOD RAW Noise-Aware RAM + Feature Distillation 简要报告

生成日期：2026-06-10  
来源计划：`plans/0609/lod_raw_noiseaware_ram_featdistill_plan.md`

本文用于独立讨论本轮 `Noise-aware RAM + mid/deep feature distillation` 实验。它不是只列 run_id 的 summary，而是说明每组实验在验证什么、结果如何、为什么最终停止这条路线。

## 1. 实验问题

前序 cross-eval 已经说明，`RAW_dark` 和 `RAW_normal` 之间存在明显输入质量 gap：

| 矩阵 | 含义 | D1 |
|---|---|---:|
| `C_dark + RAW_dark` | 当前 RAW_dark baseline | 0.845279 |
| `C_dark + RAW_normal` | 同一个 dark checkpoint 改吃 normal input | 0.880533 |
| `C_normal + RAW_normal` | RAW_normal expert / oracle | 0.898946 |
| `C_normal + RAW_dark` | normal checkpoint 直接吃 dark input | 0.816972 |

这说明输入质量很重要，但不能简单把 `C_normal` 直接迁到 `RAW_dark`。因此本计划验证的是：

> 能不能在 `RAW_dark -> RamCore3 -> DAv2` 这条链路里，让 student 学到更接近 `RAW_normal` expert 的中后层特征，同时给 RamCore3 一个轻量的可学习去噪 / restoration 分支，从而恢复一部分 RAW_dark 到 RAW_normal 的性能 gap？

## 2. 本轮具体做了什么

本轮分三层实验。

| 层级 | 实验 | 具体含义 | 目的 |
|---|---|---|---|
| 第一轮 LoRA short e10 | `L0/L1/L2/L3/L0_seed123` | 在 canonical LoRA recipe 下拆分 feature distillation 和 local residual | 判断 feature loss、local residual、二者组合是否超过 seed 噪声 |
| no-LoRA 诊断 | `W0/W3/W0_seed123` | 去掉 LoRA，只保留 decoder/RAM 训练，对比 baseline 与 feature+local | 排除“LoRA 阻碍 local/feature 生效”的解释 |
| 第二轮上限探针 | `AB_s05_sp` | 放大 local residual 预算，并把 teacher 换成 same-param clean teacher `C_dark(RAW_normal)` | 验证“第一轮失败是否只是 local 预算太小或 teacher 选错” |

这里所有 D1 都是 e10 短训的 `best_lod_d1` 口径，用于路线筛选；它们不能直接当作最终 strict cross-eval 结论。主比较方式是看 matched baseline、seed-only 上沿和预设 go-gate。

## 3. 第一轮 LoRA short e10

第一轮用 canonical LoRA recipe，从 `C_dark` 初始化 student，teacher 先用 `C_normal(RAW_normal)`。核心变量如下：

| 组 | 实验含义 | local branch | feature distill | seed | D1 | AbsRel | RMSE |
|---|---|---|---|---:|---:|---:|---:|
| `L0` | baseline，不加新机制 | none | none | 42 | 0.845600 | 3.5956 | 24.3999 |
| `L0_seed123` | 只换 seed 的 baseline | none | none | 123 | 0.850200 | 3.6286 | 24.2723 |
| `L1` | 只加 mid/deep feature distill | none | `C_normal` teacher | 42 | 0.848400 | 3.6448 | 24.3138 |
| `L2` | 只加 noise-aware local residual | `scale=0.1, gate_init=0.03` | none | 42 | 0.846000 | 3.6692 | 24.3968 |
| `L3` | feature distill + local residual | `scale=0.1, gate_init=0.03` | `C_normal` teacher | 42 | 0.848500 | 3.7276 | 24.3634 |

第一轮表面上 `L1/L3` 高于 seed42 baseline `L0`，但关键问题是：

- `L0_seed123=0.850200` 高于 `L1=0.848400` 和 `L3=0.848500`。
- seed42 到 seed123 的 baseline 波动约 `+0.0046 D1`，大于 L3 相对 L0 的 `+0.0029`。
- `L3` 没过 full gate：原判据要求 `L3 > L0 + 0.010 = 0.8556`，实际只有 `0.8485`。
- `AbsRel` 全部变差，`L3` 从 L0 的 `3.5956` 变到 `3.7276`，恶化约 `3.7%`。

feature probe 还显示，`L1/L3` 的 layer5/8/11 token cosine distance 确实比 L0 更接近 teacher，但 D1 没有超过 seed-only 上沿。这说明“中后层 token 更像 teacher”没有稳定转化成验证集深度指标收益。

## 4. 第一轮 local residual 诊断

`L3` 的 local residual 分支几乎没有真正参与修复。诊断结果：

- `tanh(local_residual_gate)=[0.03134, 0.03290, 0.03090]`，基本停在初始化附近。
- `mean|local|=0.000842`，只占 `x3_ram` 平均幅度的 `0.124%`。
- `mean|RAW_dark - RAW_normal|≈0.0777`，约为 local contribution 的 `92x`。
- 可视化上 `L3 RAM pre-local` 和 `L3 RAM out` 肉眼几乎一致。

因此 `L3` 相对 `L1` 的极小 D1 差异不能解释为 local residual 学到了有效去噪；更像是训练噪声或其它权重更新带来的波动。

## 5. no-LoRA 诊断

为了排除“LoRA 干扰了 local residual / feature distill”的解释，又做了 no-LoRA short 诊断。这里 student/teacher 使用 matching no-LoRA decoder-only checkpoints。

| 组 | 实验含义 | LoRA | local branch | feature distill | D1 | AbsRel | RMSE |
|---|---|---|---|---|---:|---:|---:|
| `W0` | no-LoRA baseline | none | none | none | 0.831200 | 4.0555 | 27.3274 |
| `W0_seed123` | no-LoRA baseline 换 seed | none | none | none | 0.830800 | 4.0279 | 27.1004 |
| `W3` | no-LoRA 下加 local + feature distill | none | `noiseaware_v1` | `C_normal_W0` teacher | 0.831000 | 3.9351 | 27.0682 |

结果：

- `W3=0.831000` 没有超过 `W0=0.831200`。
- `W0_seed123=0.830800` 与 W0 同量级，说明 no-LoRA 的 seed 波动没有掩盖一个明显正效应。
- `W3` 的 token distance 略低于 W0，但 D1 仍更低。

结论：失败不是因为 LoRA 挡住了这条路线；在 no-LoRA setting 下，feature matching 也没有转化为有效验证集增益。

## 6. 第二轮上限探针：放大 local + same-param teacher

第一轮失败后，第二轮只跑最强组合 `AB_s05_sp`，作为上限探针。它同时改两个第一轮最可疑的问题：

- local residual 预算放大：`scale 0.1 -> 0.5`，`gate_init 0.03 -> 0.1`。
- teacher 从 `C_normal(RAW_normal)` 改成 same-param clean teacher：`C_dark(RAW_normal)`。

这样做的含义是：

- 如果第一轮只是 local 预算太小，AB 应该能明显超过第一轮。
- 如果第一轮主要是 `C_normal` teacher 参数空间不匹配，same-param teacher 应该能改善。
- 如果连 AB 这个最强组合都不过 seed 上沿，就不再回补 A/B 单轴和额外 seed。

`AB_s05_sp` 结果：

| 组 | 实验含义 | D1 | AbsRel | RMSE | train proxy D1 | best epoch |
|---|---|---:|---:|---:|---:|---:|
| `AB_s05_sp` | 放大 local + same-param teacher | 0.846900 | 3.5605 | 24.3927 | 0.8721 | 9 |

主判据：

| 对照 | 数值 |
|---|---:|
| seed-only 上沿 `L0_seed123` | 0.850200 |
| `AB_s05_sp` | 0.846900 |
| 差值 | -0.003300 |

因此 AB 未通过上限探针。它不仅低于 `L0_seed123`，也低于第一轮 `L1=0.848400` 和 `L3=0.848500`。

## 7. 第二轮 gate / local contribution 诊断

AB 的 local 幅度确实比第一轮大了，但仍没有转化为 D1。

- `tanh(local_residual_gate)=[0.0983, 0.1004, 0.1000]`，基本贴着初始化 `0.1`。
- `mean|local|=0.01283`。
- `mean|x3_ram|=0.79344`。
- `local/x3_ram=1.617%`。
- `mean|RAW_dark - RAW_normal|=0.07767`。
- `local/RAW_gap≈16.5%`。

解释：

- 相比第一轮 `local/x3_ram=0.124%`，第二轮 local contribution 放大到约 `13x`。
- 但 gate 仍没有主动长大，说明优化没有把这条 branch 当作主要下降方向。
- same-param teacher 确实让 AbsRel 不再恶化，`3.5605` 还略优于 L0 的 `3.5956`。
- 但主目标 D1 仍失败，因此不能把它判为有效去噪。

## 8. 总体结论

本轮最重要的结论是：

> `noiseaware_v1` internal local residual + mid/deep feature distillation 这条路线没有通过短训筛选，不建议继续补 A/B 单轴、更多 seed 或 formal e40。

更具体地说：

- 第一轮 LoRA 下，`L1/L3` 的 apparent gain 被 `L0_seed123` 直接覆盖，落在 seed-only 波动范围内。
- 第一轮 local residual 基本空转，贡献只有 `x3_ram` 幅度的 `0.124%`。
- no-LoRA 诊断中 `W3` 仍不超过 `W0`，排除“LoRA 挡路”的解释。
- 第二轮把 local 预算放大、teacher 换成 same-param clean 后，最强 AB 仍只有 `D1=0.8469`，低于 seed-only 上沿 `0.8502`。
- same-param teacher 对 AbsRel 有帮助，但没有让 D1 过关。

因此当前证据支持停止这条 internal residual 去噪路线。后续如果还要推进 RAW_dark restoration，应另立 v2 方案，而不是继续 sweep `lambda_feat/scale/gate`。

## 9. 后续讨论方向

可以讨论的下一步不是“再把这组参数扫大一点”，而是结构或监督目标的变化：

1. 把 restoration 放到更早、更大容量的位置，例如 RAW input 级或 RamCore3 之前。
2. 改变监督信号，例如直接 RAW restoration target、teacher-depth consistency，或更明确的 clean-input consistency。
3. 从 internal local residual 转向 DAv2 实际输入空间的 post-RAM cleanup，或者 token-space / decoder-side adapter。

这些都已经超出本计划的第一版边界，应作为新计划单独定义实验语义和判据。

## 10. 主要产物路径

- 原计划与执行记录：`plans/0609/lod_raw_noiseaware_ram_featdistill_plan.md`
- 第一轮 LoRA summary：`plans/0609/lod_raw_noiseaware_featdistill_short_0609_2045_summary/ablation_summary.md`
- 第一轮 LoRA matrix：`plans/0609/lod_raw_noiseaware_featdistill_short_0609_2045_summary/matrix.csv`
- no-LoRA summary：`plans/0609/lod_raw_noiseaware_featdistill_w0_w3_short_0609_2153_summary/ablation_summary.md`
- no-LoRA matrix：`plans/0609/lod_raw_noiseaware_featdistill_w0_w3_short_0609_2153_summary/matrix.csv`
- 第二轮 AB 队列日志：`finetune_stf/logs/0609_2255_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue.queue.log`
- 第二轮 AB 实验目录：`finetune_stf/exp/0609_2255_AB_s05sp_lod_raw_pair_lora_decoder_e10_featmiddeep_sameparam_localnoiseaware_s05_g01`
- 第二轮 AB best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0609_2255_AB_s05sp_lod_raw_pair_lora_decoder_e10_featmiddeep_sameparam_localnoiseaware_s05_g01/best_model.pth`
