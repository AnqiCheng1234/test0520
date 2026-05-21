# DA3 vs DAv2-vitL pseudo target 对比分析（0521_1137 vs 0521_0306）

## 背景

两个 STF RGB LoRA decoder 实验，唯一意图差异是把 pseudo depth teacher 从 DAv2 official vitL
换成 DA3 mono large（+ sparse-LiDAR affine 对齐）。但结果与预期相反：DA3 路线 abs_rel 单调劣化。

| | 0521_0306 (DAv2 vitL) | 0521_1137 (DA3 mono large + sparse_metric) |
|---|---|---|
| `stf_pseudo_manifest` | `pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/...` | `pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/...` |
| `stf_train_target_mode` | `dav2_pseudo` | `da3_pseudo_sparse_metric` |
| 走入 loss 的 `target_space` | `inverse_relative`（target 原样为 disparity） | `metric_depth`（loss 内部 1/depth） |
| target_source | `dense_pseudo`（全密集 vitL 视差） | `dense_aligned`（每帧 affine 拟合 LiDAR 后米制深度） |
| target 数值量级 | mean≈210, p99≈600（vitL 视差原值） | mean≈25 m, p99≈74 m（米制深度） |

其他超参完全相同：encoder=vits, lr=1e-5, bridge_lr=lora_lr=5e-5, loss_type=ssi,
lambda_grad=2.0, target_norm=True, epochs=5, bs=8。

## 实证 1：训练日志中的 eval / pred 演化

DAv2 pretrained init 评测（两个实验一致，证明起点相同）：
abs_rel=0.1286, rmse=7.9181, d1=0.8577

| epoch | 0306 abs_rel | 0306 raw_pred_mean / max | 1137 abs_rel | 1137 raw_pred_mean / max |
|---|---|---|---|---|
| init | 0.1286 | — | 0.1286 | — |
| 0 | **0.1271 ↓** | 2.9695 / 11.8125 | 0.1327 ↑ | 2.7414 / 12.2500 |
| 1 | 0.1274 | 2.9794 / 11.8125 | 0.1343 ↑ | 2.4197 / 10.3125 |
| 2 | (停于 epoch 1+) | — | 0.1355 ↑ | 2.3087 / 10.0000 |
| 3 | — | — | 0.1371 ↑ | 2.2473 / 9.8125 |
| 4 | — | — | 0.1364 | 2.2162 / 9.6250 |

观察：
- 0306 训完一个 epoch 即微幅改善（vits 向 vitL 同家族蒸馏）。
- 1137 abs_rel 单调劣化，且模型预测的有效视差 max 从 12.25 → 9.6、mean 从 2.74 → 2.22，
  显著被「向小处压」，与训练目标分布偏向远场塌缩的特征一致。

train loss 量级（信息性，跨 target_space 不可直接比较）：

| epoch | 0306 avg_loss | 1137 avg_loss |
|---|---|---|
| 0 | 0.0071 | 0.0301 |
| 1 | 0.0056 | 0.0251 |
| 2 | — | 0.0232 |
| 3 | — | 0.0220 |
| 4 | — | 0.0213 |

## 实证 2：DA3 与 DAv2 vitL teacher 在 LiDAR 上的对齐质量

随机抽 60 张 STF 训练样本，仅在 LiDAR 有效像素（`1 m ≤ d ≤ 80 m`）处比较三种候选 target
对真视差 `1/d_lidar` 的回归能力：

- `1/da3_raw` —— 直接对 DA3 affine-invariant 输出取倒数
- `1/(s·da3 + b)` —— 1137 当前路线：每张图 affine 拟合 LiDAR 得米制，再取倒数
- `vitL_raw` —— 0306 路线：vitL 输出直接当 disparity

为去除 affine 自由度的影响，对每个候选 target 做 SSI 对齐（lstsq `α·t + β ≈ 1/d`）
然后计算 truth-MAD 归一化后的 RMSE。

| target | SSI-aligned RMSE↓ | Spearman ρ↑ |
|---|---|---|
| `1/da3_raw` | **0.658** | 0.908 |
| `1/(s·da3+b)` | 0.560 | 0.908 |
| `vitL_raw` | **0.494** | 0.924 |

每张图 DA3 → LiDAR 米制的 affine 拟合统计：

```
scale: mean=15.21  std=3.46   range=[7.11, 21.53]    (3× 帧间方差)
shift: mean=+6.04  std=2.22   range=[+1.95, +12.12]  (6× 帧间方差)
shift/scale (= DA3 单位下"真 0 点"): mean=+0.448  std=0.286  range=[+0.118, +1.519]
```

## 根因诊断

### (1) target_space 反演不是问题来源，affine 才是

`finetune_stf/util/loss.py:40-47` 的 `build_training_target` 在 `metric_depth` 模式下做 1/depth，
而 `inverse_relative` 直接用原值。这两条路线**给到 SSI 归一化前的张量数学等价**：

- 1137 走 `metric_depth`：先 `_fit_da3_to_sparse_metric` 得到 `metric = s·da3 + b`，loss 里 `1/metric`
- 假想的「离线转 disparity 再喂 `inverse_relative`」：先 `metric = s·da3 + b`，再 `1/metric`

两者得到的是同一个 target 张量，只是 1/x 算在管线里的位置不同。所以**不能通过切 target_space 解决**。
（前一轮分析中的 suggestion #1 在此修正：那条建议是空话。）

### (2) DA3 的 affine 偏移项 b 在数值上不可忽略

测得 `shift / scale = -b̂/â ≈ +0.45 ± 0.29`，量级显著高于 DA3 raw 的近场最小值（0.05–0.15）。
这意味着 DA3 raw 在数值上把"无穷远"映到了 DA3≈0.4–0.5 附近的位置，再小才是真正的近处。
直接 `1/da3_raw` 把这一段错当作"近处大视差"，**远场结构被严重压扭**——SSI-RMSE 比正确 affine 大 17%。

### (3) per-image affine 引入帧间漂移噪声

`scale` 帧间 std/mean ≈ 23%，`shift` ≈ 37%。STF 是雾雪雨场景，LiDAR 本身在恶劣天气下稀疏带噪，
每张图 lstsq 拟出来的 (s, b) 不稳定 → 同一段距离在不同帧被映到不同米制 → 损失看到的"教师答案"
逐帧抖动。即使 fit 不掉到 sparse_fallback（日志里全部 `target_source=dense_aligned`），
拟合「通过」也不等于拟合「准确」。

### (4) Spearman 0.908 vs 0.924：DA3 ranking 信号本身是好的

排序质量 DA3 略低于 vitL，但差距不大。意味着 DA3 mono 的几何 prior 没坏，坏的是把它强行打回米制
再做 MSE 监督的这一步——尺度/偏移噪声把好的 ranking 信号污染了。

### (5) 训练动态侧证

`raw_pred_valid_max` 从 12.25 → 9.6 单调下降，说明模型在被 target 的远场塌缩 + 帧间漂移
共同拉扯下逐渐输出更窄的视差范围。这种"输出范围被压窄"恰恰会破坏 STF eval 用的
`rgb_pretrained_ref` 校准协议（按 DAv2 视差分布定标）。

## 结论

- **0306（vitL pseudo）能 work**，是因为 teacher 与 student 同在 DAv2 disparity 表达空间，
  全密集、无 per-frame affine 噪声，本质上是同家族蒸馏。
- **1137（DA3 + sparse_metric）变差**，主要不是 target_space 选择，而是
  「per-image LiDAR affine 拟合」这一步把好的 DA3 ranking 信号污染成尺度噪声 + 偏移噪声，
  再被 SSI loss 在反演空间放大。
- 想"绕开 1/depth"是无效的——`metric_depth` 与等价的「离线 1/(s·da3+b) + inverse_relative」
  是同一张 target 张量。

## 可试方案（按改动量从小到大）

| 方案 | 思路 | 预期 | 代价 |
|---|---|---|---|
| A | 继续用 vitL pseudo（0306 路线） | 已 work | 0 |
| B | DA3 metric + **log-depth 监督**（不再走 1/depth） | 避免远场塌缩，affine 噪声仍在 | loss 一处 |
| C | DA3 → **全局** affine 而非 per-image | 消除帧间漂移；但单 (s, b) 表达力弱 | 离线脚本 |
| D | DA3 只做 **ranking loss**（pairwise / spearman 风格） | Spearman 0.908 是真信号，绕开尺度/偏移 | loss 大改 |
| E | **DA3 + vitL 双教师融合**（vitL 给远场尺度，DA3 给结构细节） | 兼顾两边 | 数据管线改 |

最低成本是 B 或 D。要最稳的评测，回 A。

## 关联文件

- 配置：
  - `finetune_stf/exp/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/config.json`
  - `finetune_stf/exp/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/config.json`
- 训练日志：`finetune_stf/exp/<run>/train.log`
- 关键源码：
  - `finetune_stf/util/loss.py:40-47`  build_training_target（target_space 路由）
  - `finetune_stf/util/loss.py:50-96`  robust_normalize_target_per_sample（MAD 归一化）
  - `finetune_stf/util/loss.py:185-240` solve_scale_shift_per_sample（SSI 对齐）
  - `finetune_stf/dataset/stf.py:170-245` _fit_da3_to_sparse_metric + build_da3_sparse_metric_target
  - `finetune_stf/dataset/stf.py:367-411` STF.__getitem__（target_kind 分发与 target_space 设置）
- pseudo manifest 元数据：
  - `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/run_summary.json`
    （`depth_value_units.value = "affine_invariant_depth_from_da3mono"`, direction=larger_is_farther）
