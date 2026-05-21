# STF boundary diagnostics: `0521_0306` vs `0521_0835`

日期：2026-05-21

本文单独记录针对 `rgb_raw_baseline_fairness_summary.md` 中两个关键实验的边界清晰度诊断：

- `0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5`
- `0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10`

结果文件：

- summary CSV：`/home/caq/6666_raw/dav2_raw_0520/plans/result/stf_boundary_diagnostics_key_0521_1528/summary.csv`
- per-sample CSV：`/home/caq/6666_raw/dav2_raw_0520/plans/result/stf_boundary_diagnostics_key_0521_1528/per_sample.csv`
- script：`/home/caq/6666_raw/dav2_raw_0520/anqi_eval/eval_stf_boundary_diagnostics.py`

## 1. 评估口径

split：STF val，`808` samples。

checkpoint：两行均使用各自 `best_model.pth`。

常规 sparse depth 指标：

- 使用 STF sparse LiDAR GT。
- 与现有 eval 一致，先做 per-image disparity-space affine alignment。
- depth range：`min_depth=1.0 / max_depth=80.0`。

### 1.1 PDBE-style pseudo boundary route

这一路线用固定 dense pseudo target 衡量预测深度边界是否贴近 pseudo depth 边界。

固定 pseudo target：

`/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`

参数：

- depth edge source：`log(depth)` Sobel magnitude
- edge threshold：每张图 95 percentile
- DBE truncation：`10 px`
- boundary F1 tolerance：`3 px`

指标：

- `pseudo_dbe_acc`：预测 edge 到 pseudo target edge 的平均截断距离，越低越好。
- `pseudo_dbe_comp`：pseudo target edge 到预测 edge 的平均截断距离，越低越好。
- `pseudo_edge_f1`：3 px tolerance 下的 boundary F1，越高越好。

注意：这条路线衡量的是“是否接近固定 DAv2-ViTL RGB pseudo depth 的边界”，不是直接衡量 sparse LiDAR GT 边界。

### 1.2 Image-edge-band sparse route

这一路线用 RGB 图像边缘定义边界区域，然后只在该区域内的 sparse LiDAR GT 点上计算误差。

参数：

- edge source：STF RGB LUT image
- image edge threshold：Sobel magnitude 90 percentile
- edge band：edge mask dilation `3 px`
- 最少 edge-band sparse points：`10`

指标：

- `image_edge_band_abs_rel`：只在图像边缘带内 sparse GT 点计算的 `abs_rel`，越低越好。
- `image_edge_band_d1`：只在图像边缘带内 sparse GT 点计算的 `d1`，越高越好。
- `image_edge_band_points`：每张图平均参与该指标的 sparse GT 点数。
- `image_edge_band_coverage`：edge-band sparse 点占全图有效 sparse GT 点比例。

注意：这条路线更适配 STF sparse GT，但图像边缘不一定等于真实 depth discontinuity。

## 2. 结果

| experiment | n | abs_rel ↓ | d1 ↑ | PDBE acc ↓ | PDBE comp ↓ | pseudo edge F1 ↑ | image-edge abs_rel ↓ | image-edge d1 ↑ | image-edge points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0521_0306` RGB LoRA decoder | 808 | 0.1274 | 0.8588 | 1.500 | 2.332 | 0.7977 | 0.2535 | 0.7805 | 2714.9 |
| `0521_0835` RAW bridge LoRA | 808 | 0.1297 | 0.8480 | 2.490 | 3.594 | 0.6687 | 0.2173 | 0.7684 | 2715.0 |

Additional values:

| experiment | pseudo precision ↑ | pseudo recall ↑ | image-edge coverage |
|---|---:|---:|---:|
| `0521_0306` RGB LoRA decoder | 0.8557 | 0.7483 | 0.4016 |
| `0521_0835` RAW bridge LoRA | 0.7385 | 0.6124 | 0.4016 |

## 3. 结论

按 PDBE-style pseudo boundary route，`0521_0306` 明显优于 `0521_0835`：

- `pseudo_dbe_acc`: `1.500` vs `2.490`
- `pseudo_dbe_comp`: `2.332` vs `3.594`
- `pseudo_edge_f1`: `0.7977` vs `0.6687`

也就是说，如果把固定 DAv2-ViTL RGB pseudo depth 的 dense boundary 当作参考，RGB LoRA decoder 的预测边界更贴近 pseudo target。

按 image-edge-band sparse route，结果是 mixed：

- `0521_0835` 的 edge-band `abs_rel` 更低：`0.2173` vs `0.2535`
- 但 `0521_0835` 的 edge-band `d1` 更低：`0.7684` vs `0.7805`

因此目前不能简单说 `0521_0835` 的边界“整体更好”。更准确的表述是：

`0521_0835` 在 RGB 图像边缘带内的 sparse LiDAR 点平均相对误差更小，但在 dense pseudo boundary 的边界位置/完整性上明显弱于 `0521_0306`。

## 4. 建议

如果后续论文或报告中需要支撑“RAW 输入视觉上更 sharp”，建议不要只报告 PDBE，因为 PDBE 会偏向固定 RGB pseudo target。更合理的组合是：

1. 主边界指标：`pseudo_dbe_acc / pseudo_dbe_comp / pseudo_edge_f1`
2. sparse STF 辅助指标：`image_edge_band_abs_rel / image_edge_band_d1`
3. qualitative panels：对两者分歧较大的样本做可视化，尤其看 `0521_0835` 是否有更锐但不贴 pseudo target 的结构。

当前两条路线的分歧本身是有信息量的：它提示 RAW bridge 可能在图像边缘附近改善了部分 sparse 点误差，但这种改善没有转化为与 RGB pseudo-depth 边界更一致的 dense boundary。
