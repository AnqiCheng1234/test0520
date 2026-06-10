# LOD RAW_dark 退化归因简要报告

生成日期：2026-06-10  
来源计划：`plans/0609/lod_raw_dark_degradation_diagnosis_plan.md`

本文由原 `plans/0609/report_v1.md` 整理而来，用作独立讨论材料。目标不是只列实验编号，而是说明：`RAW_dark` 相比 `RAW_normal` 到底差在哪里，哪些解释被数据支持，哪些方向已经不值得继续投入。

## 1. 核心问题

当前 LOD RAW 分支存在明显 `RAW_dark -> RAW_normal` 性能 gap：

| 输入 / 模型状态 | D1 | 含义 |
|---|---:|---|
| `RAW_dark` baseline | 0.845067 | 当前 RAW_dark zero-eval identity |
| `RAW_normal` oracle | 0.899011 | 同任务下 RAW_normal identity |
| zero-train gap | 0.053944 | `RAW_normal - RAW_dark` 的可解释空间 |

本轮诊断要回答三个问题：

1. `RAW_dark` 的主要退化是曝光不足、噪声/低 SNR、结构损坏，还是颜色/统计偏移？
2. 简单 test-time preprocessing，如 tone、median、bilateral，能不能恢复 D1？
3. 如果图像域修复看起来改善了浅层统计，是否真的把 DAv2 中后层特征拉向 `RAW_normal`？

## 2. 一句话结论

当前证据支持：

> `RAW_dark` 的主导退化是噪声 / 低 SNR / 结构破坏，而不是单纯曝光不足；简单手工 preprocessing 和 oracle tone/denoise 都没有恢复 D1；后续应转向训练式 restoration / feature alignment，而不是继续做 median、bilateral、tone 这类 test-time 修复。

这个结论由四条证据共同支撑：数据审计、per-sample 相关性、synthetic degrade 反事实、以及 RAM/DAv2 feature probe。

## 3. 数据层审计：RAW_dark 不是只“更暗”

数据级 audit 使用与 val 对齐的 `center512x960` crop。核心统计如下：

| split | samples | reg outliers | ev_gap_luma p50 | snr_ratio_lowbin p50 | edge_recall p50 | chroma_delta_l1 p50 |
|---|---:|---:|---:|---:|---:|---:|
| `00Train` | 512 | 6 | -0.07278 | 0.3591 | 0.5514 | 0.3215 |
| `01Valid` | 112 | 0 | -0.08100 | 0.3142 | 0.6016 | 0.2797 |

读法：

- `RAW_dark` 的 flat 区 luma noise proxy 明显更大，01Valid 的 `noise_sigma_ratio_dark_over_normal` median 约为 `6.045`。
- `snr_ratio_lowbin` median 为 `0.3142`，说明低亮度区域 SNR 下降明显。
- `edge_recall` median 为 `0.6016`，说明不是只有亮度偏移，也有结构/边缘保留问题。
- 01Valid 没有 registration outlier，因此 val 结论不主要由配准异常污染。

这一步的作用是排除“只是整体暗一点”的过度简化解释：RAW_dark 同时表现出噪声、SNR 和结构层面的退化。

## 4. Per-sample 相关性：D1 gap 跟噪声和结构最相关

对 01Valid 112 个样本，定义：

```text
delta_d1 = d1_raw_normal - d1_raw_dark
```

也就是同一样本上 RAW_normal 比 RAW_dark 多出来的 D1。相关性结果：

| variable | Spearman rho with delta_d1 | p-value | 解释 |
|---|---:|---:|---|
| `snr_ratio_highbin` | -0.5214 | 3.77e-09 | 高亮区域 SNR 越差，D1 gap 越大 |
| `noise_sigma_dark_luma_flat` | 0.4656 | 2.31e-07 | RAW_dark flat noise 越大，gap 越大 |
| `edge_recall` | -0.4479 | 7.33e-07 | edge 保留越差，gap 越大 |
| `gradient_corr` | -0.4464 | 8.06e-07 | 结构相关性越差，gap 越大 |
| `noise_sigma_ratio_dark_over_normal` | 0.4118 | 6.45e-06 | dark/normal 噪声比越大，gap 越大 |
| `chromaticity_delta_l1` | 0.3933 | 1.79e-05 | 颜色统计也参与，但不是单独主因 |
| `ev_gap_luma` | -0.2121 | 0.02477 | 曝光相关性存在但弱于噪声/结构 |
| `snr_ratio_lowbin` | 0.03325 | 0.7278 | 单独 lowbin SNR 与 D1 gap 不稳定 |

读法：

- 最强相关项集中在 SNR、噪声和结构保留。
- 曝光项 `ev_gap_luma` 有相关，但显著弱于噪声/结构项。
- `d1_raw_normal` 与这些 RAW_dark audit 指标相关性弱，说明这些指标主要解释 RAW_dark 侧退化，而不是样本本身“难”。

因此，数据相关性支持“噪声/结构是主导因素”，不支持“曝光是唯一主因”。

## 5. Synthetic degrade：只降曝光几乎不伤，噪声能复现大部分 drop

反事实实验从 `RAW_normal` 出发，合成不同退化，再看 D1 如何下降。

### 5.1 Zero-train 快筛

| degrade | zero-train D1 | drop vs RAW_normal identity | 含义 |
|---|---:|---:|---|
| `raw_normal_identity` | 0.899011 | 0.000000 | RAW_normal oracle |
| exposure-only | 0.897730 | -0.001281 | 只降曝光几乎不伤 |
| noise-only | 0.849621 | -0.049390 | 噪声可复现大部分下降 |
| exposure + noise | 0.848701 | -0.050310 | 比 noise-only 只略差 |

这说明，只把 RAW_normal 变暗并不会造成 RAW_dark 那样的 D1 drop；加入暗光噪声后，D1 才接近 RAW_dark 区间。

### 5.2 Short-train 复核

zero-train 有 test-time domain shift，因此又做同预算短训，用同 seed anchors 归一化：

| degrade | seed42 D1 | seed123 D1 | mean D1 | drop_ratio_st |
|---|---:|---:|---:|---:|
| exposure-only | 0.8504 | 0.8283 | 0.8394 | 0.066 |
| noise-only | 0.7913 | 0.7596 | 0.7755 | 0.668 |

读法：

- exposure-only 的 short-train drop ratio 只有 `0.066`，不能解释主 gap。
- noise-only 的 short-train drop ratio 为 `0.668`，能解释约三分之二 gap。
- 但不是 100%，说明真实 RAW_dark 还有 synthetic noise 没覆盖的因素，例如非高斯噪声、局部结构损失、颜色/ISP 偏移等。

## 6. 手工修复失败：tone / median / bilateral 没有稳定恢复 D1

对 RAW_dark 做 test-time preprocessing，固定当前 RAW_dark checkpoint，不训练新参数。相对 `raw_dark_identity` 的 D1 `0.845067`：

| transform | 类型 | zero-train D1 | Δ vs identity | 解释 |
|---|---|---:|---:|---|
| `raw_dark_identity` | baseline | 0.845067 | 0.000000 | RAW_dark sanity |
| weak bilateral | deployable | 0.845271 | +0.000204 | 基本持平 |
| medium bilateral | deployable | 0.845138 | +0.000071 | 基本持平 |
| luma percentile tone | deployable | 0.845238 | +0.000171 | 基本持平 |
| tone then denoise | deployable | 0.844387 | -0.000680 | 略差 |
| percentile tone | deployable | 0.843358 | -0.001709 | 略差 |
| gamma tone | deployable | 0.839717 | -0.005350 | 变差 |
| median3 denoise | deployable | 0.826092 | -0.018975 | 明显变差 |
| E10 oracle tone | oracle | 0.832221 | -0.012846 | oracle tone 也变差 |
| E11 oracle tone+denoise | oracle | 0.829316 | -0.015751 | oracle tone+denoise 也变差 |

这里 E10/E11 读取 paired RAW_normal 信息，因此只作为诊断上界，不是 deployable 方法。关键结论是：即使用 oracle 级 tone/denoise，当前 checkpoint 的 D1 也没有恢复，反而下降。

这说明问题不是“调一个更好的全局 tone 或普通滤波器”就能解决。手工图像域修复容易改变 RamCore3 / DAv2 已适应的输入分布，D1 不涨。

## 7. Feature probe：浅层稍微接近，DAv2 中后层没有接近

固定 RAW_dark L0 best checkpoint，比较这些输入：

- `raw_dark_identity`
- E10 `raw_dark_tone_percentile_oracle`
- E11 `raw_dark_tone_denoise_oracle`
- `raw_normal_identity`

抓取 RAM `x3/x_cat/ffm_mid` 和 DAv2 layers `2/5/8/11` token。相对 `raw_normal_identity` 的平均距离如下：

| 输入 | RAM x3 L1 | RAM x3 cosine | DAv2 L2 cosine | DAv2 L5 cosine | DAv2 L8 cosine | DAv2 L11 cosine |
|---|---:|---:|---:|---:|---:|---:|
| `raw_dark_identity` | 0.319836 | 0.079426 | 0.109136 | 0.083064 | 0.078499 | 0.090494 |
| E10 oracle tone | 0.294110 | 0.066013 | 0.106929 | 0.084506 | 0.080184 | 0.097011 |
| E11 oracle tone+denoise | 0.289149 | 0.062889 | 0.098686 | 0.084674 | 0.080932 | 0.099004 |

相对 `raw_dark_identity -> raw_normal_identity` 的距离变化：

| metric | E10 | E11 | 读法 |
|---|---:|---:|---|
| RAM x3 L1 | closer 8.0% | closer 9.6% | RAM 输出轻微接近 RAW_normal |
| RAM x3 cosine | closer 16.9% | closer 20.8% | RAM 方向轻微接近 |
| DAv2 layer2 token cosine | closer 2.0% | closer 9.6% | early token 轻微接近 |
| DAv2 layer5 token cosine | farther 1.7% | farther 1.9% | 中层没有接近 |
| DAv2 layer8 token cosine | farther 2.1% | farther 3.1% | 中后层没有接近 |
| DAv2 layer11 token cosine | farther 7.2% | farther 9.4% | 深层更远 |

结论：E10/E11 确实让 RAM output 和 DAv2 early token 稍微接近 `RAW_normal`，但没有形成 DAv2 layer5/8/11 的有效 alignment。E11 比 E10 更接近 RAM x3 / layer2，但 D1 更差，这说明浅层统计改善不等于深度任务收益。

## 8. 总体判断

本轮归因结论可以概括为：

1. 噪声 / 低 SNR 是 `RAW_dark` 退化的主导因素，但不能解释全部 gap。
2. 单纯曝光 / dynamic range 不足不是主因。
3. 简单 test-time median、bilateral、tone 不值得继续作为主线。
4. 即使 oracle tone/denoise 能让浅层 RAM 表示稍微接近 `RAW_normal`，也没有让 DAv2 中后层 token 对齐，更没有提升 D1。
5. 下一步应该做训练式 restoration / denoise 或 feature-level alignment，重点看 DAv2 layer5/8/11，而不是只看 RAM x3 或图像亮度统计。

这份归因报告后续实际引出了两条实验路线：

- `Noise-aware RAM + feature distillation`：验证 internal local residual 和 mid/deep distillation，结果未通过短训筛选。
- `Post-RAM cleanup`：把 cleanup 移到 DAv2 实际输入空间，得到弱正 D1 candidate，但仍需复核稳定性。

## 9. 可用可视化素材

如果要做讨论页，可以优先用这些图：

- 数据分布直方图：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/histograms/`
- `RAW_dark` vs `RAW_normal` percentile：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/histograms/raw_dark_vs_normal_percentiles.png`
- 噪声 proxy 分布：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/histograms/noise_proxy_distribution.png`
- edge retention 分布：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/histograms/edge_retention_distribution.png`
- attribution scatter plots：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/scatter_plots/`

这些图适合说明“为什么判断噪声/结构是主因”。E10/E11 feature probe 目前主要是表格数据，不是最好看的对外图。

## 10. 主要产物路径

- 原计划与执行记录：`plans/0609/lod_raw_dark_degradation_diagnosis_plan.md`
- data audit summary：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/summary_by_split.md`
- per-sample raw stats：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/per_sample_raw_stats.csv`
- attribution report：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/correlation_report.md`
- zero-eval summary：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary.csv`
- oracle zero-eval summary：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary_oracle.csv`
- short-train anchors：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train/short_train_summary_anchors_e5_s80.csv`
- short-train synthetic degrade：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train/short_train_summary_degrade_e5_s80.csv`
- feature probe against RAW_normal reference：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/feature_probe/0609_1727_raw_dark_L0_e10_e11_ref_normal_tokens/ram_feature_summary.md`
