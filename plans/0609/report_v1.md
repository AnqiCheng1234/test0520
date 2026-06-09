# LOD RAW_Dark Degradation Attribution Report v1

生成时间：2026-06-09 17:40 CST

## 1. Summary

- RAW_dark baseline：D1 = **0.8451**（zero-eval identity `0.845067`；per-sample eval mean `0.845308`）。
- RAW_normal oracle：D1 = **0.8990**（zero-eval identity `0.899011`；per-sample eval mean `0.898944`）。
- zero-train gap：`G_zero = 0.899011 - 0.845067 = 0.053944`。
- short-train anchors：RAW_dark floor = **0.7402**，RAW_normal ceiling = **0.8464**，gap = **0.1062**。
- 主归因：**噪声 / 低 SNR 是当前证据下的主导因素，但不能解释全部 gap**。
- 不支持的主因：单纯曝光 / 动态范围不足。E7 exposure-only 短训 drop_ratio 只有 **0.066**。
- 额外结论：E10/E11 oracle tone/denoise 没有提升 D1；feature probe 显示它们只轻微改善 RAM/early token，对 DAv2 中后层 token 没有形成有效 RAW_normal alignment。

## 2. Data-Level Audit

数据级 audit 使用 `center512x960`，与 val center crop 对齐。

| split | samples | reg outliers | ev_gap_luma p50 | snr_ratio_lowbin p50 | edge_recall p50 | chroma_delta_l1 p50 |
|---|---:|---:|---:|---:|---:|---:|
| 00Train | 512 | 6 | -0.07278 | 0.3591 | 0.5514 | 0.3215 |
| 01Valid | 112 | 0 | -0.08100 | 0.3142 | 0.6016 | 0.2797 |

关键读法：

- RAW_dark flat 区 luma noise proxy 约为 RAW_normal 的 **6x**：01Valid median `noise_sigma_ratio_dark_over_normal = 6.045`。
- 01Valid `snr_ratio_lowbin` median = **0.3142**，说明低亮度 bin 的 SNR 明显下降。
- 01Valid `edge_recall` median = **0.6016**，存在结构/边缘保留问题。
- 01Valid 无 registration outlier，当前 val 结论不主要受配准异常污染。

来源：

- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/summary_by_split.md`
- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/per_sample_raw_stats.csv`

## 3. Per-Sample Correlation

`delta_d1 = d1_raw_normal - d1_raw_dark`，01Valid 112 samples，registration outlier 已排除。

| variable | Spearman rho with delta_d1 | p-value | interpretation |
|---|---:|---:|---|
| `snr_ratio_highbin` | -0.5214 | 3.77e-09 | SNR 越差，RAW_dark/RAW_normal gap 越大 |
| `noise_sigma_dark_luma_flat` | 0.4656 | 2.31e-07 | RAW_dark flat noise 越大，gap 越大 |
| `edge_recall` | -0.4479 | 7.33e-07 | edge 保留越差，gap 越大 |
| `gradient_corr` | -0.4464 | 8.06e-07 | 结构相关性越差，gap 越大 |
| `noise_sigma_ratio_dark_over_normal` | 0.4118 | 6.45e-06 | dark/normal 噪声比越大，gap 越大 |
| `linear_residual_sigma` | 0.4101 | 7.11e-06 | 简单线性拟合越难，gap 越大 |
| `chromaticity_delta_l1` | 0.3933 | 1.79e-05 | 颜色统计偏移也参与，但不是单独主因 |
| `ev_gap_luma` | -0.2121 | 0.02477 | 曝光相关性弱于噪声/结构项 |
| `snr_ratio_lowbin` | 0.03325 | 0.7278 | 单独 lowbin SNR 与 delta_d1 不稳定 |

辅助因变量方向一致：对 `d1_raw_dark`，`noise_sigma_dark_luma_flat` 为负相关，`gradient_corr` 为正相关。`d1_raw_normal` 与这些 RAW_dark audit 指标相关性弱，说明这些指标主要解释 RAW_dark 侧退化，而不是 normal 模型本身质量。

来源：

- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/correlation_report.md`
- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_dark_val.summary.json`
- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_normal_val.summary.json`

## 4. Ablation Eval

### 4.1 Dark-Side Zero-Train

相对 `raw_dark_identity` D1 `0.845067`，所有 hand preprocessing / oracle preprocessing 都没有稳定恢复 D1。

| transform | type | zero-train D1 | Δ vs identity | interpretation |
|---|---|---:|---:|---|
| `raw_dark_identity` | baseline | 0.845067 | 0.000000 | RAW_dark zero-eval sanity |
| `raw_dark_bilateral_weak_trainfit_sigma` | deployable | 0.845271 | +0.000204 | 基本持平 |
| `raw_dark_bilateral_medium_trainfit_sigma` | deployable | 0.845138 | +0.000071 | 基本持平 |
| `raw_dark_tone_percentile_luma_trainfit` | deployable | 0.845238 | +0.000171 | 基本持平 |
| `raw_dark_tone_then_denoise_trainfit` | deployable | 0.844387 | -0.000680 | 基本持平略差 |
| `raw_dark_tone_percentile_trainfit` | deployable | 0.843358 | -0.001709 | 略差 |
| `raw_dark_tone_gamma_trainfit` | deployable | 0.839717 | -0.005350 | 变差 |
| `raw_dark_denoise_median3` | deployable | 0.826092 | -0.018975 | 明显变差 |
| E10 `raw_dark_tone_percentile_oracle` | oracle | 0.832221 | -0.012846 | oracle tone 也变差 |
| E11 `raw_dark_tone_denoise_oracle` | oracle | 0.829316 | -0.015751 | oracle tone+denoise 也变差 |

读法：简单 test-time 图像域修复不能直接转化为当前 RAW_dark checkpoint 的 D1 收益。E10/E11 读取 paired RAW_normal，因此只能作为 diagnostic，不能作为 deployable baseline。

### 4.2 Short-Train Anchors

| anchor | seed42 D1 | seed123 D1 | mean D1 |
|---|---:|---:|---:|
| `raw_dark_identity` floor_st | 0.7452 | 0.7352 | 0.7402 |
| `raw_normal_identity` ceiling_st | 0.8548 | 0.8379 | 0.8464 |

## 5. Synthetic Degrade Eval

### 5.1 Zero-Train Normal-to-Dark

| degrade | zero-train D1 | drop vs raw_normal_identity | interpretation |
|---|---:|---:|---|
| `raw_normal_identity` | 0.899011 | 0.000000 | RAW_normal oracle |
| `raw_normal_to_dark_exposure_trainfit` | 0.897730 | -0.001281 | 仅曝光几乎不伤 |
| `raw_normal_to_dark_noise_only_trainfit` | 0.849621 | -0.049390 | 噪声可复现大部分 RAW_dark 区间下降 |
| `raw_normal_to_dark_exposure_noise_trainfit` | 0.848701 | -0.050310 | 曝光+噪声只比噪声略差 |

### 5.2 Short-Train Synthetic Degrade

归一化使用同 seed anchors。

| degrade | seed42 D1 | seed123 D1 | mean D1 | drop_ratio_st |
|---|---:|---:|---:|---:|
| E7 `raw_normal_to_dark_exposure` | 0.8504 | 0.8283 | 0.8394 | 0.066 |
| E8 `raw_normal_to_dark_noise_only` | 0.7913 | 0.7596 | 0.7755 | 0.668 |

结论：exposure-only 不成立为主因；noise-only 在短训下解释约 **66.8%** 的 short-train gap，但没有达到 100%，说明真实 RAW_dark 还有未被当前 synthetic noise model 覆盖的因素。

## 6. RAM / DAv2 Feature Probe

固定 RAW_dark L0 best checkpoint，比较 `raw_dark_identity`、E10、E11、`raw_normal_identity`。每个输入 112 val samples，抓 RAM `x3/x_cat/ffm_mid` 和 DAv2 layers `2/5/8/11` token。

相对 `raw_normal_identity` 的平均距离：

| 输入 | RAM x3 L1 | RAM x3 cosine | DAv2 L2 cosine | DAv2 L5 cosine | DAv2 L8 cosine | DAv2 L11 cosine |
|---|---:|---:|---:|---:|---:|---:|
| `raw_dark_identity` | 0.319836 | 0.079426 | 0.109136 | 0.083064 | 0.078499 | 0.090494 |
| E10 `raw_dark_tone_percentile_oracle` | 0.294110 | 0.066013 | 0.106929 | 0.084506 | 0.080184 | 0.097011 |
| E11 `raw_dark_tone_denoise_oracle` | 0.289149 | 0.062889 | 0.098686 | 0.084674 | 0.080932 | 0.099004 |

相对 `raw_dark_identity -> raw_normal_identity` 的距离，E10/E11 的变化：

| metric | E10 | E11 | read |
|---|---:|---:|---|
| RAM x3 L1 | closer 8.0% | closer 9.6% | RAM 输出轻微接近 RAW_normal |
| RAM x3 cosine | closer 16.9% | closer 20.8% | RAM 方向轻微接近 |
| DAv2 layer2 token cosine | closer 2.0% | closer 9.6% | early token 轻微接近 |
| DAv2 layer5 token cosine | farther 1.7% | farther 1.9% | 中层没有接近 |
| DAv2 layer8 token cosine | farther 2.1% | farther 3.1% | 中后层没有接近 |
| DAv2 layer11 token cosine | farther 7.2% | farther 9.4% | 深层更远 |

结论：E10/E11 虽然轻微改善 RAM/early token，但没有把 DAv2 中后层 token 拉向 RAW_normal。E11 比 E10 更接近 RAM x3/layer2，但 zero-eval D1 更差，因此当前 hand/oracle preprocessing 不是有效的 feature alignment。

来源：

- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/feature_probe/0609_1718_raw_dark_L0_e10_e11_normal_tokens`
- `/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/feature_probe/0609_1727_raw_dark_L0_e10_e11_ref_normal_tokens`

## 7. Decision

Next main direction：

- 优先考虑 **RAW_dark restoration / denoise 的训练式方案**，但不要再押注简单 median/bilateral/tone preprocessing。
- 如果做 restoration，应把目标定义成训练中可学习的恢复或特征蒸馏，而不是 test-time 手工滤波。

Secondary direction：

- 补 RAW_normal checkpoint 侧 feature probe：`raw_normal_identity` vs `raw_normal_to_dark_noise_only/exposure`，确认 synthetic noise 是否同样主要扰动 DAv2 中后层 token。
- 考虑 feature-level alignment / distillation，重点关注 DAv2 layer5/8/11，而不是只对齐 RAM x3。

Do not prioritize yet：

- 单纯曝光校正 / dynamic range 校正。
- 无训练的 median/bilateral denoise。
- 只看 RAM x3 统计的 shallow alignment。

## 8. Failure Galleries

本轮还没有整理人工精选 failure gallery。已有可用素材：

- audit histograms / panels：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/histograms` 和 `audit_v1/panels`
- attribution join：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/attribution_join.csv`
- feature probe per-sample：`/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/feature_probe/*/ram_feature_per_sample.csv`

建议下一版按这些类型各挑 8-12 个样本：高噪声、高 edge loss、高 chromaticity shift、高 feature mismatch。
