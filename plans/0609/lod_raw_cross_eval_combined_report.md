# LOD RAW Dark/Normal Cross-Eval Combined Report

生成时间：2026-06-09 CST

## 1. 报告定位

本文件合并两组 2x2 cross-eval 结果。

- **主报告**：`0609_1906_lod_raw_dark_normal_lora_cross_eval`
  - 使用当前 canonical LoRA recipe：`lora_tap_r8a16 + hflip0.5 + decoder + ssi`
  - 端点匹配当前讨论前提：`M_DD ~= 0.845`，`M_NN ~= 0.899`
  - 因此作为后续 teacher / student 设计的主依据。

- **补充报告**：`0609_1903_lod_raw_dark_normal_cross_eval`
  - 使用 earlier W0 decoder-only recipe：无 LoRA，端点为 `M_DD ~= 0.830`，`M_NN ~= 0.887`
  - 这组不是当前 canonical endpoint，但已经完成，作为 recipe sensitivity / sanity control 保留。

两个报告都使用：

- eval split：`01Valid`，112 samples
- BN recalibration split：`00Train`，1958 samples
- BN recalibration 定义：reset front-end/RAM BN running stats，用 train split center crop 累计更新 running mean/var；不反传，不更新任何权重。

## 2. 主报告：Canonical LoRA Pair

来源：

- report：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1906_lod_raw_dark_normal_lora_cross_eval/report.md`
- matrix：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1906_lod_raw_dark_normal_lora_cross_eval/matrix.csv`
- recovery：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1906_lod_raw_dark_normal_lora_cross_eval/recovery.csv`

Checkpoint：

| label | run | best D1 |
|---|---|---:|
| C_dark | `0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly` | 0.845068 |
| C_normal | `0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly` | 0.899010 |

### 2.1 Matrix

| variant | matrix | checkpoint | input | D1 | AbsRel | RMSE |
|---|---|---|---|---:|---:|---:|
| strict | M_DD | C_dark | I_dark | 0.845279 | 3.666699 | 24.878314 |
| strict | M_DN | C_dark | I_normal | 0.880533 | 3.052718 | 20.466344 |
| strict | M_NN | C_normal | I_normal | 0.898946 | 2.469963 | 18.394070 |
| strict | M_ND | C_normal | I_dark | 0.816972 | 3.237003 | 28.824663 |
| bn_recalib | M_DD | C_dark | I_dark | 0.842198 | 3.764893 | 25.189090 |
| bn_recalib | M_DN | C_dark | I_normal | 0.878907 | 3.179445 | 20.650424 |
| bn_recalib | M_NN | C_normal | I_normal | 0.898185 | 2.572419 | 18.362419 |
| bn_recalib | M_ND | C_normal | I_dark | 0.820890 | 3.422002 | 28.445251 |

### 2.2 Recovery

| variant | G = M_NN - M_DD | R_DN | M_DN - M_DD | M_ND - M_DD |
|---|---:|---:|---:|---:|
| strict | 0.053667 | 0.656898 | 0.035253 | -0.028307 |
| bn_recalib | 0.055987 | 0.655682 | 0.036710 | -0.021308 |

### 2.3 主报告结论

- `M_DN = 0.880533`，相对 `M_DD = 0.845279` 恢复 `0.035253` D1。
- `R_DN = 0.656898`，落在 `0.4 - 0.7` 区间：**输入退化和训练域 / 参数差异都重要**。
- `M_DN` 没有接近 `M_NN = 0.898946`，所以不能把 `C_dark(RAW_normal)` 当作完整 oracle。
- `M_ND = 0.816972 < M_DD = 0.845279`，说明 `C_normal` 直接吃 `RAW_dark` 不鲁棒。
- BN recalibration 对 `M_DN` 没有提升，`0.880533 -> 0.878907`，不支持“主要是 front-end BN running stats 错位”。

主报告支持的 teacher 语义：

- `C_dark(RAW_normal)` 可以作为 same-parameter clean-view teacher，但只能解释约 66% 的 gap。
- `C_normal(RAW_normal)` 仍应作为 normal-domain expert teacher。
- 如果做 feature matching，建议采用双 teacher 或保守权重；不能只依赖 same-parameter teacher。

## 3. 补充报告：W0 Decoder-Only Pair

来源：

- report：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1903_lod_raw_dark_normal_cross_eval/report.md`
- matrix：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1903_lod_raw_dark_normal_cross_eval/matrix.csv`
- recovery：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1903_lod_raw_dark_normal_cross_eval/recovery.csv`

Checkpoint：

| label | run | best D1 |
|---|---|---:|
| C_dark_W0 | `0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly` | 0.829851 |
| C_normal_W0 | `0609_0010_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly` | 0.887057 |

注意：这组补充报告用的是 RAW W0 pair，不是 `R0`。`R0` 指的是 `0608_1509_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R0_aug-baseline_e10_poly`，是 `RGB_dark + dav2_rgb + lora=none` 的 decoder-only run，best D1 为 0.8351。它不是 RAW_normal checkpoint，因此没有 `C_normal + I_normal` 这一格。

### 3.1 Matrix

| variant | matrix | checkpoint | input | D1 | AbsRel | RMSE |
|---|---|---|---|---:|---:|---:|
| strict | M_DD | C_dark_W0 | I_dark | 0.829694 | 3.904160 | 27.016630 |
| strict | M_DN | C_dark_W0 | I_normal | 0.875950 | 2.811440 | 20.942291 |
| strict | M_NN | C_normal_W0 | I_normal | 0.887073 | 2.380199 | 19.731537 |
| strict | M_ND | C_normal_W0 | I_dark | 0.792805 | 3.684180 | 31.994471 |
| bn_recalib | M_DD | C_dark_W0 | I_dark | 0.828288 | 4.000188 | 27.199405 |
| bn_recalib | M_DN | C_dark_W0 | I_normal | 0.874958 | 3.191405 | 21.184416 |
| bn_recalib | M_NN | C_normal_W0 | I_normal | 0.887237 | 2.767326 | 19.900387 |
| bn_recalib | M_ND | C_normal_W0 | I_dark | 0.801819 | 3.762086 | 30.455739 |

### 3.2 Recovery

| variant | G = M_NN - M_DD | R_DN | M_DN - M_DD | M_ND - M_DD |
|---|---:|---:|---:|---:|
| strict | 0.057380 | 0.806138 | 0.046256 | -0.036889 |
| bn_recalib | 0.058949 | 0.791702 | 0.046670 | -0.026468 |

### 3.3 补充报告结论

- W0 下 `R_DN = 0.806138`，比主报告高，说明在较弱 recipe 中 input swap 能恢复更多比例。
- 但 W0 端点本身低于 canonical LoRA endpoint，不能替代主报告作为当前方法设计依据。
- `M_ND` 仍低于 `M_DD`，方向与主报告一致：normal checkpoint 不天然鲁棒于 RAW_dark。
- BN recalibration 仍没有实质提升 `M_DN`，方向与主报告一致。

补充报告的用途：

- 支持“input quality 是真实强因素”这一点。
- 同时提示：`R_DN` 会随 recipe / checkpoint 变化，不能只用 W0 的 `>0.8` 结论推出 same-parameter teacher 已足够。

## 4. 合并判断

两组报告共同支持：

1. `RAW_dark -> RAW_normal` input swap 明显提升 D1，因此输入退化是重要瓶颈。
2. `C_normal(RAW_dark)` 明显低于 `C_dark(RAW_dark)`，因此 normal-domain checkpoint 对 RAW_dark 不鲁棒。
3. front-end BN recalibration 没有带来 M_DN 的明显提升，因此当前 gap 不是主要由 BN running stats 错位造成。

主报告决定最终判断：

- 当前 canonical LoRA recipe 下，`R_DN ~= 0.657`。
- 这不是 `<0.4`，说明 same-parameter clean input teacher 有价值。
- 也不是 `>0.7`，说明 `C_dark(RAW_normal)` 不能单独作为完整 oracle。

因此下一步建议：

- Teacher A：`C_dark(RAW_normal)`，用于 same-parameter clean-view feature target。
- Teacher B：`C_normal(RAW_normal)`，用于 normal-domain expert target。
- Student：优先从 `C_dark` 或 current RAW_dark recipe 初始化，输入 `RAW_dark`。
- Feature loss 权重保守设置，特别是使用 `C_normal` teacher 时要避免把 checkpoint 参数差异过强地压给 student。

不建议优先做：

- 只做 BN/domain-stat recalibration。
- 直接用 `C_normal` 初始化然后裸吃 `RAW_dark`，因为 `M_ND` 低于 `M_DD`。
