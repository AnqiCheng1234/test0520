# LOD Post-RAM Denoise / Cleanup 简要报告

生成日期：2026-06-10  
来源计划：`plans/0609/lod_postram_denoise_cleanup_plan.md`

本文用于独立讨论本轮 post-RAM denoise / cleanup 实验，不只是列 run_id。核心问题是：`RAW_dark -> RamCore3` 之后、DAv2 backbone 之前的 post-BN pseudo RGB，是否还能通过输入空间 cleanup 得到可复核的深度指标收益。

## 1. 实验问题

本轮实验没有继续做 RAW 图像域的 median / bilateral / tone preprocessing，也没有继续加码 RamCore3 内部 local residual。原因是前序结果已经显示：

- `RAW_dark` 的主要问题更接近 noise / low SNR / 结构破坏，不是单纯曝光不足。
- 手工图像域 denoise / tone 处理没有稳定提升。
- 之前 RamCore3 内部 residual + feature distillation 没有突破 seed-only 上沿。

因此这次把操作点移到 DAv2 实际看到的输入上：

```text
RAW_dark
  -> RamCore3
  -> x_ram  # RamCore3 post-BN pseudo RGB，非 ImageNet norm，非自然 RGB
  -> post-RAM denoise / cleanup
  -> pad
  -> DAv2 backbone
  -> DPT decoder
```

这里的 `x_ram` 是 RamCore3 post-BN native domain，可能有负值，通道统计也不是自然 sRGB。因此外部 denoiser 只能作为诊断 probe，最终更可信的是在这个 native domain 上训练一个小的 cleanup adapter。

## 2. 做了哪些实验

| 实验 | 具体含义 | 目的 |
|---|---|---|
| Stage A no-op sanity | 打开 post-RAM eval 插入点，但完全 bypass，不改变 `x_ram` | 验证 strict eval 路径能复现原始 `C_dark(I_dark)` baseline |
| Stage A identity clamp | 用 `q001/q999` per-channel affine 把 `x_ram` 映射到 `[0,1]`，再映回 native domain | 判断 wrapper/clamp 本身是否损伤信息 |
| Stage A DRUNet external denoise | 使用 DPIR/DRUNet TorchScript，扫少量 `sigma/alpha`，不训练任何参数 | 看现成 denoiser 是否能直接清理 post-BN pseudo RGB |
| Stage B matched baseline | `post_ram_cleanup=none`，seeds `42/123/777`，从同一个 C_dark LoRA checkpoint 继续 e10 训练 | 给 cleanup 提供同 recipe、同 seed set 的 seed-only 对照 |
| Stage B trainable cleanup | `post_ram_cleanup=cnn`，32 channels、4 blocks、GroupNorm、zero-init、scale=0.1、lr=5e-5，只用 depth loss | 判断可学习的小残差 cleanup adapter 是否有效 |
| Stage B strict 复核 | 对 B0/T 的 best checkpoints 统一跑 `M_DD strict`，固定 `I_dark`，不做 BN recalib | 避免只用训练期 best D1 下结论 |

实现层面也补齐了相关工程约束：新增 post-RAM 参数 schema、resolved config 中心化校验、cleanup 与 external denoiser 互斥校验、optimizer group、外部 denoiser eval/visualize/summary 工具，以及 strict eval 的 post-RAM 参数转发。

## 3. 评估口径

主指标是 `D1`、`AbsRel`、`RMSE`。`D1` 越高越好，`AbsRel/RMSE` 越低越好。

本报告区分两种口径：

- `training-best`：训练过程中按 `lod_val D1` 保存的 best，用于筛选。
- `strict`：统一 cross-eval strict 复核，固定 BN 协议，是本轮最终比较口径。

Stage A strict no-op reference 为：

| reference | D1 | AbsRel | RMSE |
|---|---:|---:|---:|
| `C_dark(I_dark)` strict no-op | 0.845279 | 3.666699 | 24.878314 |

## 4. Stage A：外部 DRUNet probe 结果

Stage A 先验证 wrapper，再试 DRUNet。`q001/q999` clamp ratio 为 `0.001919`，identity round-trip 的 D1 只下降 `0.000200`，说明 wrapper 本身损伤很小，后续 DRUNet 结果可以解释。

| cell | 实验含义 | D1 | ΔD1 vs no-op | AbsRel | RMSE | 判断 |
|---|---|---:|---:|---:|---:|---|
| `EXT_NOOP_A0` | 不改变 post-RAM 输入 | 0.845279 | 0.000000 | 3.666699 | 24.878314 | strict 路径复现成功 |
| `EXT_ID_Q001` | affine/clamp/inverse affine round-trip | 0.845080 | -0.000200 | 3.691696 | 24.874620 | wrapper 损伤很小 |
| `EXT_D1` | DRUNet `sigma=10`、`alpha=0.25` | 0.841947 | -0.003332 | 3.789210 | 25.035134 | 负信号 |
| `EXT_D3` | DRUNet `sigma=25`、`alpha=0.25` | 0.846295 | +0.001016 | 3.593061 | 24.742452 | 唯一弱正 cell |
| `EXT_D4` | DRUNet `sigma=25`、`alpha=0.5` | 0.844203 | -0.001076 | 3.689165 | 24.803842 | blend 变强后转负 |

结论：DRUNet 不是一个可以直接接入推理的成功方案。`EXT_D3` 有弱正信号，而且 D1、AbsRel、RMSE 都优于 no-op，但 D1 提升只有 `+0.001016`，低于计划里设定的 Stage A 强信号门槛 `+0.003`。更合理的解释是：post-RAM input-space cleanup 方向可能有一点信号，但现成 sRGB/高斯先验 denoiser 对 post-BN pseudo RGB 很敏感，不能作为正式模块。

## 5. Stage B：可训练 cleanup adapter 结果

Stage B 的核心比较不是和历史 run 横比，而是同 seed matched baseline：

```text
B0_LORA_s01：无 cleanup，继续 e10 训练
T_LORA_s01：加入 post-RAM cleanup CNN，继续 e10 训练
```

两者从同一个 C_dark LoRA checkpoint 初始化，语义差异集中在 `post_ram_cleanup=none` vs `post_ram_cleanup=cnn`。

| seed | training-best ΔD1 | strict D1: B0 -> T | strict ΔD1 | AbsRel: B0 -> T | RMSE: B0 -> T | 解释 |
|---:|---:|---:|---:|---:|---:|---|
| 42 | +0.000853 | 0.847592 -> 0.848504 | +0.000912 | 3.666920 -> 3.754150 | 24.367533 -> 24.267365 | D1 小涨，RMSE 改善，AbsRel 变差 |
| 123 | +0.002134 | 0.848470 -> 0.850708 | +0.002238 | 3.860734 -> 3.746498 | 24.346629 -> 24.166887 | 最强正向 seed，三项指标都优于 matched baseline |
| 777 | +0.001145 | 0.845085 -> 0.846413 | +0.001329 | 3.778942 -> 4.324509 | 24.514426 -> 25.026583 | D1 小涨，但 AbsRel/RMSE 明显变差 |

汇总：

- 3/3 个 seed 的 strict D1 都高于对应 matched baseline。
- mean matched strict ΔD1 为 `+0.001493`。
- winner 是 `0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123`，strict D1 为 `0.850708`，training-best D1 为 `0.850784`。
- cleanup adapter 的 `delta_ratio` 在三个 seed 上分别约为 `0.091457`、`0.102271`、`0.039808`，说明模块确实在改 DAv2 输入，不是完全 idle。

但这还不是强结论：

- matched baseline training-best 上沿是 `0.848650`，只有 seed123 cleanup 明确超过它。
- seed42 / seed777 的 D1 提升幅度较小。
- seed777 的 AbsRel/RMSE 恶化明显。
- winner 相对原始 Stage A strict no-op reference，AbsRel 从 `3.666699` 到 `3.746498`，没有满足“D1 提升且 AbsRel/RMSE 不恶化”的强 go-gate。

## 6. 可视化观察

对 winner `T_seed123` 额外生成了 cleanup 可视化。该可视化只取前 `16` 个 valid 样本并保存前 `8` 个 panel，用于看 `x_ram -> x_clean` 的形态变化，不代表全量 strict 指标。

观察结论：`x_clean` 相比 `x_ram` 的变化不是简单随机噪声抹平。`mean delta` / `abs(delta)` 更像沿物体边界、局部亮暗结构和竖向条纹做结构化修正，因此它更接近 post-RAM pseudo-RGB 分布/纹理校正，而不是传统图像去噪。

## 7. 总体结论

本轮结果支持一个谨慎判断：

> post-RAM cleanup 有可复核的弱正 D1 信号，但目前还不是可以直接扩大 formal ablation 的强结论。

更具体地说：

- 外部 DRUNet 只给出一个弱正诊断 cell，不支持把现成 denoiser 当正式推理模块。
- 训练式 cleanup adapter 在 3 个 matched seeds 上 strict D1 全部为正，说明方向不是完全无效。
- 但收益幅度小，且 AbsRel/RMSE 不稳定，尤其 seed777 风险明显。
- 因此本轮不建议自动启动更大规模 formal ablation，也不建议把结论表述为“post-RAM cleanup 已经有效”。更准确的说法是：它是一个弱正 candidate，需要围绕稳定性和误差副作用继续讨论。

## 8. 建议讨论点

1. 是否先做更小规模的复核，而不是直接 formal ablation：例如补 2 到 3 个 seed，或只重复 seed123/777 看稳定性。
2. 是否给 cleanup 加更明确的约束：例如控制 delta、约束 AbsRel/RMSE 副作用，或降低 scale/lr，避免 D1 小涨但整体误差变差。
3. 如果目标是稳定超过 `0.8507`，当前 image-space post-RAM cleanup 可能不够强，需要讨论 token-space adapter 或 decoder-side RAW/RAM feature fusion，但这已经是下一条路线，不应和本轮结论混在一起。

## 9. 主要产物路径

- 原计划与执行记录：`plans/0609/lod_postram_denoise_cleanup_plan.md`
- Stage A DRUNet summary：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/summary.md`
- Stage A DRUNet CSV：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/summary.csv`
- Stage A 最好 cell 可视化：`finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D3/viz/EXT_D3/panel_grid.png`
- Stage B strict summary：`finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/summary.md`
- Stage B strict CSV：`finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/stageB_strict_summary.csv`
- Stage B matched deltas：`finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/stageB_paired_deltas.csv`
- Stage B cleanup 可视化：`finetune_stf/analysis/lod_postram_stageB_visual/0610_1122_T_seed123_cleanup_viz/viz/T_seed123_cleanup/panel_grid.png`
