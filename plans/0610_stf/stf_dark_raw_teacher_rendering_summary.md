# STF Dark RAW Teacher RGB 渲染调参汇总

更新时间：2026-06-11 02:40（Asia/Shanghai）

## 1. 背景

目标是在 STF dark 样本上效仿 ROD 的 `teacher_bright_degreen_v1`，从 RAW 生成更适合 DAv2-L 的 `teacher_rgb`，再观察 DAv2-L prediction 是否优于 dataset 官方 RGB。

这次主要结论：

1. 直接增强 dataset RGB 不稳定；
2. STF RAW 直接套 ROD-like bright teacher 会有明显白幕；
3. 在 RAW 渲染阶段做 black percentile 去白幕 + 较弱 gamma + 轻量锐化，能得到更清晰的图和更好的 sparse holdout depth 指标。

## 2. 数据与评估口径

样本：

- 只看 dark；
- 使用 10 张 STF 官方 RGB 亮度最低且 sparse LiDAR 点数足够的样本；
- 样本列表来自：
  - `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_dark_enhance_dav2_compare/selected_dark_overexp_10_metrics.csv`

RAW：

- 路径：`/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`
- key：`bayer_rect`
- channel order：`[R, Gr, Gb, B]`
- RAW RGB 投影：`[R, 0.5*(Gr+Gb), B]`

评估：

- DAv2-L checkpoint：`/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth`
- input size：`924`
- 对每张图做 deterministic sparse LiDAR holdout；
- prior points 拟合 DAv2 relative inverse prediction 到 `1/depth`；
- holdout points 上计算 `AbsRel` 和 `D1`；
- depth range：`1m..80m`。

## 3. RGB 增强尝试

脚本：

- `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/scripts/make_stf_teacher_rgb_dav2_dark_overexp_compare.py`

输出：

- `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_dark_enhance_dav2_compare`

增强配方：

- source：dataset official RGB；
- recipe：LAB L-channel CLAHE + white percentile stretch + RGB gamma brighten；
- `clahe_clip=2.0`
- `clahe_grid=8`
- `white_percentile=99.5`
- `gamma=0.65`

结果：

| 输入 | mean AbsRel | mean D1 |
|---|---:|---:|
| dataset RGB | 0.1078800833 | 0.8733744041 |
| raw teacher_rgb, ROD-like | 0.1014532213 | 0.8764321446 |
| enhanced RGB | 0.1137088127 | 0.8591592196 |

结论：

- dataset RGB 后处理增强能拉亮暗部，但会同步放大车灯、路面高亮和暗部噪声；
- 平均指标低于原始 dataset RGB 和 raw teacher_rgb；
- 不建议沿这条线直接继续全量。

## 4. RAW 渲染粗 sweep

脚本：

- `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/scripts/sweep_stf_raw_teacher_renderings_dark.py`

输出：

- `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_raw_render_sweep_dark10`

主要可视化：

- `stf_raw_render_variants_10dark.png`
- `stf_raw_render_dav2l_depth_variants_10dark_spectral_r.png`

结果：

| branch | mean AbsRel | mean D1 | AbsRel wins |
|---|---:|---:|---:|
| `dataset_rgb` | 0.1078800833 | 0.8733744041 | 2 |
| `raw_rod_like_g045` | 0.1014532213 | 0.8764321446 | 3 |
| `raw_black01_g065` | 0.1046477852 | 0.8679006235 | 0 |
| `raw_black05_g075` | 0.1093539176 | 0.8588743203 | 0 |
| `raw_black10_g085` | 0.1287355525 | 0.8325058779 | 0 |
| `raw_black05_g075_unsharp` | **0.0986005595** | **0.8768454861** | **5** |
| `raw_decomp_black05_g075` | 0.1090977519 | 0.8626748254 | 0 |

观察：

- `raw_rod_like_g045` 视觉上白幕最明显，但指标已经优于 dataset RGB；
- black percentile 可以压白幕，但压得过强会让图变暗，DAV2-L prediction 变差；
- `raw_black05_g075_unsharp` 是粗 sweep 里最好的折中；
- decompanded raw 这组没有收益，当前 companded 渲染更适合 DAv2-L 输入。

## 5. RAW 渲染 focus sweep

命令：

```bash
conda run -n dav3 python finetune_stf/scripts/sweep_stf_raw_teacher_renderings_dark.py \
  --recipe-set focus \
  --output-dir finetune_stf/analysis/0611_stf_raw_render_focus_sweep_dark10 \
  --tile-width 220 \
  --overwrite-preds
```

输出：

- `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_raw_render_focus_sweep_dark10`

主要可视化：

- `stf_raw_render_variants_10dark.png`
- `stf_raw_render_dav2l_depth_variants_10dark_spectral_r.png`

结果：

| branch | mean AbsRel | mean D1 | AbsRel wins |
|---|---:|---:|---:|
| `dataset_rgb` | 0.1078800833 | 0.8733744041 | 1 |
| `raw_rod_like_g045` | 0.1014532213 | 0.8764321446 | 2 |
| `raw_b03_g065_us06` | 0.1001018426 | **0.8778080676** | 1 |
| `raw_b03_g070_us06` | 0.1002160119 | 0.8774594842 | 0 |
| `raw_b03_g075_us06` | 0.1000391144 | 0.8775817792 | 0 |
| `raw_b05_g070_us06` | 0.1007410036 | 0.8712151348 | 0 |
| `raw_b05_g075_us06` | 0.1000737620 | 0.8730334016 | 0 |
| `raw_b05_g075_us085` | **0.0986005595** | 0.8768454861 | **4** |
| `raw_b05_g080_us06` | 0.0997829596 | 0.8743575656 | 1 |
| `raw_b08_g080_us06` | 0.1110091689 | 0.8555436791 | 1 |

## 6. 当前推荐配方

推荐候选：

```text
raw_b05_g075_us085
```

参数：

```text
raw_normalization = companded
black_percentile = 5.0
white_percentile = 99.8
gamma = 0.75
gains = [1.08, 0.95, 1.10]
unsharp_sigma = 1.1
unsharp_amount = 0.85
```

理由：

- 10 张 dark 上 mean AbsRel 最低；
- AbsRel wins 最多；
- 视觉上比 ROD-like 配方明显少白幕，边界更清晰；
- 没有像 `black=10%` 那样压得过暗。

单分支对比图：

- `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_raw_render_focus_sweep_dark10/stf_raw_b05_g075_us085_compare_10dark_spectral_r.png`

备选：

- `raw_b03_g065_us06`
- mean D1 最高，但 mean AbsRel 不如推荐候选。

## 7. 后续建议

1. 用 `raw_b05_g075_us085` 跑 100 张 dark-only 统计；
2. 100 张仍稳定后，再生成 dark subset 的 teacher_rgb DAv2-L pseudo manifest；
3. 不建议直接全量替换，因为白天 / 过曝样本可能不适合同一套 black percentile；
4. 若后续要训练 RAW 模型，应把 teacher recipe 作为实验语义参数显式写入 config，不要靠路径名或默认值隐式决定。
