# LOD `RAW_Dark` 退化来源诊断执行计划

> 目标：在已经确认 `RAW_normal` 与 `RAW_dark` 是主要差异来源之后，进一步拆解 `RAW_dark` 退化到底主要来自 **曝光/动态范围不足**、**噪声/SNR 降低**、**局部结构损失**、**通道/颜色统计偏移**，还是 **RAM/DAv2 feature-domain mismatch**。  
> 当前阶段不直接设计最终去噪网络，也不直接进入 feature-level alignment 训练；先做可量化归因。

---

## 0. 当前前提与判定目标

### 0.1 已知前提

你已经完成第一步确认：

```text
RAW_dark  -> D1 ≈ 0.845
RGB_dark  -> D1 ≈ 0.855
RAW_normal -> D1 ≈ 0.899
```

因此当前总 gap 定义为：

```text
G = D1(RAW_normal) - D1(RAW_dark) = 0.899 - 0.845 = 0.054
```

后续所有诊断实验都用这个 gap 归一化：

```text
recovered_ratio(transform) = [D1(transform) - D1(RAW_dark)] / G

drop_ratio(degrade_normal) = [D1(RAW_normal) - D1(degraded_RAW_normal)] / G
```

上式是用全训 D1 / 全训 gap `G` 的**概念定义**。但短训达不到全训 best，所以 §9 决策闸实际用的是**同配方、同步数**的短训变体 `recovered_ratio_st`（§6.3）与 `drop_ratio_st`（§5.7），分母换成短训 floor/ceiling。zero-train eval 只作快筛，不进闸。

解释标准（同一套阈值适用于 `_st` 变体）：

| ratio | 解释 |
|---:|---|
| `< 0.10` | 该因素不是主要瓶颈，至少不是单独瓶颈 |
| `0.10 - 0.30` | 有影响，但不是主因 |
| `0.30 - 0.50` | 重要因素，值得进入下一步设计 |
| `> 0.50` | 主导因素，优先围绕它设计下一阶段方法 |

### 0.2 核心问题

本阶段要回答四个问题：

1. `RAW_dark` 相对 `RAW_normal` 的信息损失主要是 **可逆的曝光/动态范围问题**，还是 **不可简单恢复的噪声/结构破坏问题**？
2. 简单的非学习型 preprocessing 能否恢复一部分 D1 gap？
3. `RAW_normal -> synthetic RAW_dark` 的受控退化能否复现 `RAW_dark` 的性能下降？
4. 如果图像域修复有效但模型不受益，是否说明下一步应转向 RAM output / DAv2 feature-level alignment？

### 0.3 配方锁定（Recipe Lock）—— 全链路统一，强制项

> 本阶段所有 eval / short-train 必须复用定义 gap 的那套配方，否则 `recovered_ratio` 的分子分母串味，且 `--resume-from` 会因架构不匹配在 `strict=True` 下报错。

**前提数字的真实来源（已钉死）：**

| 角色 | run 目录 | checkpoint | best lod_d1 |
|---|---|---:|---:|
| RAW_dark baseline | `0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly` | `/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_.../best_model.pth` | `0.8451 @e37` |
| RAW_normal oracle ref | `0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly` | `/mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_.../best_model.pth` | `0.8990 @e36` |
| RGB_dark ref | `0608_2246_lod_true_rgb_dark_block8excl10_dav2s_lora_tap_r8a16_decoder_e40_flip05_poly` | `/mnt/drive/3333_raw/0000_exp_ckpt/0608_2246_.../best_model.pth` | `0.8555 @e22` |

因此 `G = 0.8990 - 0.8451 = 0.0539`（≈ 0.054）。

**这三个 run 都用同一套配方 `lora_tap_r8a16 + hflip0.5 + decoder + ssi`，且用的是 block8 split manifest，不是通用 manifest。** 关键点：

- **canonical manifest（必须用这个，不是通用版）：**
  `/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv`
  split：`00Train`(1958) / `01Valid`(112) / `02HoldoutBuffer`(160)。**val split 名是 `01Valid`，没有 `00Val`。**
- LoRA tap 层 `[2,5,8,11]` 是 L0 显式传入的（`_lora` alias 不会自动补 tap 层），因此每条命令都必须显式带上，strict-load 才能对齐。
- **不要把 `--input-type` alias 当正式实验语义。** 当前 alias 能展开到正确 resolved config，但正式 eval / short-train / full-train 必须显式写出 `dataset_family`、`dataset_input_mode`、`input_domain`、`front_end`、`model_input_tensor`、`bridge`、`decoder_feature_adapter` 等实验语义参数，避免隐式 alias 或未来默认值改变导致配方漂移。
- **所有 `finetune_stf/train.py` 调用必须用 `torchrun`。** 当前训练入口会读取 `RANK/WORLD_SIZE/LOCAL_RANK`，不能直接 `python finetune_stf/train.py`。默认用 `dav3` 环境：

```bash
conda run --live-stream -n dav3 torchrun --nproc_per_node=1 --master_port <PORT> finetune_stf/train.py ...
```

长时间的 short-train / full-train 队列按远程开发规范放进 tmux；zero-train eval 如果只是 112 张全量验证、运行时间很短，可以直接跑，但仍建议写清 log 路径。

**CANONICAL_COMMON_FLAGS（dark 与 normal 共用，下文命令引用此块）：**

```text
--encoder vits
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--lod-root /home/caq/6666_raw/0000_dataset/LOD
--lod-manifest <ABLATION_OR_BASE_MANIFEST_block8_val112_excl10_uniform.csv>
--lod-label-space inverse_relative
--lod-train-crop-mode random
--lod-val-crop-mode center
--input-domain raw3
--front-end raw_rgb16_ram3
--model-input-tensor raw
--raw-storage-format raw_rgb16_png_3ch
--lod-raw-norm-mode uint16_div_65535
--raw-ram-rgb-tail identity
--bridge none
--decoder-feature-adapter none
--lora dav2_lora
--lora-block-mode tap
--lora-tap-layers 2 5 8 11
--lora-rank 8
--lora-alpha 16
--lora-lr 5e-5
--raw-front-end-lr 5e-5
--dav2-train-mode decoder
--backbone-layer-decay 1.0
--lr 1e-5
--lr-schedule poly
--warmup-steps 0
--loss-type ssi
--loss-target-normalization
--loss-norm-min-scale 1e-3
--bs 8
--accum-steps 1
--input-height 512
--input-width 960
--aug-preset off
--aug-hflip-prob 0.5
--eval-lod
--no-eval-stf
--eval-lod-train-proxy
--lod-train-proxy-count 112
--amp
--amp-dtype bf16
--num-workers 4
--log-interval 500
--no-enable-fixed-viz-dump
--no-enable-train-source-viz-dump
--seed 42
```

输入语义只换以下显式 resolved flags。`--input-type lod_true_raw_*_rgb16_lora` 只能作为人工检查时的兼容缩写，正式命令不要只靠它。

```text
dark:
  --dataset-family lod_true_raw_dark_rgb16
  --dataset-input-mode raw_rgb16_dark      # 读 raw_dark_path 列

normal:
  --dataset-family lod_true_raw_normal_rgb16
  --dataset-input-mode raw_rgb16_normal    # 读 raw_normal_path 列
```

**全量评估口径：不要传 `--debug-max-val-samples`（默认 None = 全量）。传 `-1` 会让评估在第一个样本前就 break，结果是 0 个样本、NaN。**

**输出路径口径：** `train.py` 会把 `--save-path` 作为轻量实验目录，并用 `basename(save_path)` 拼到 `--heavy-save-root` 下生成 checkpoint / TensorBoard 目录。因此诊断脚本和训练模板必须显式区分：

```text
light save-path : /home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/lod_raw_diag/<phase>/<RUN_NAME>
heavy-save-root : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/<phase>
heavy artifacts : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/<phase>/<RUN_NAME>/
```

不要把 `--save-path` 直接当作 heavy checkpoint 目录来理解。

---

## 1. 代码环境与最小侵入原则

### 1.1 不建议改训练主线代码

本阶段建议只新增 `tools/` 下的诊断脚本，不修改以下训练主线：

```text
finetune_stf/dataset/lod_true.py
finetune_stf/models/raw_ram.py
finetune_stf/train.py
finetune_stf/config/resolved.py
```

原因：当前代码已经支持 `RAW_Dark` / `RAW_normal` 两种 LOD RAW 输入，诊断阶段没有必要先引入新 dataset class。

### 1.2 利用现有数据入口

当前 LOD true RAW dataset 已经有两个 RAW mode：

```text
raw_rgb16_dark   -> raw_dark_path   -> RAW_Dark
raw_rgb16_normal -> raw_normal_path -> RAW_normal
```

新增 transformed input 时，生成新的 PNG16 文件，并写一份新的 manifest。

**注意：manifest 不是只有两列。** `lod_true.py` 要求完整 13 列并带硬校验（`dark_id==normal_id+1`、`height/width==(800,1200)`、`label_space=='inverse_relative'`、`teacher_source`、`pseudo_depth_path` 等，见 `LOD_TRUE_REQUIRED_COLUMNS`）。因此 ablation manifest 必须**从 canonical block8 manifest 复制每一行的全部 13 列，只覆盖目标路径列的值**：

```text
基准（canonical block8 manifest，全部 13 列原样保留）:
pair_id,split,normal_id,dark_id,rgb_normal_path,rgb_dark_path,
raw_normal_path,raw_dark_path,pseudo_depth_path,label_space,height,width,teacher_source

ablation（dark 类 transform，只改 raw_dark_path 这一列）:
raw_dark_path = /.../generated/raw_dark_tone_match/xxx.png   # 其余 12 列不动

ablation（normal->dark 合成退化，只改 raw_normal_path 这一列）:
raw_normal_path = /.../generated/raw_normal_to_dark_exposure/xxx.png  # 其余 12 列不动
```

这样仍然走 §0.3 的 `CANONICAL_COMMON_FLAGS`，只是把 `--lod-manifest` 指向 ablation manifest（仍是 block8 split），并切换显式输入语义：

```text
dark transform:
  --dataset-family lod_true_raw_dark_rgb16
  --dataset-input-mode raw_rgb16_dark

normal degrade:
  --dataset-family lod_true_raw_normal_rgb16
  --dataset-input-mode raw_rgb16_normal
```

不用改 `LODTrueRawDarkRGB16`。

### 1.3 注意 PNG16 的通道顺序

读取逻辑是：

```python
raw_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
raw_rgb = raw_bgr[..., ::-1].astype(np.float32) / 65535.0
```

因此生成 transformed PNG16 时，如果内部数组是 RGB 顺序，写盘前要转回 BGR：

```python
cv2.imwrite(str(out_path), np.round(np.clip(raw_rgb, 0, 1) * 65535).astype(np.uint16)[..., ::-1])
```

**额外硬约束（否则 dataset 直接抛错）：**

- 生成的 PNG 必须是**完整原生分辨率 `800x1200x3` 且 dtype `uint16`**。`_load_raw_rgb16_png` 会精确校验 `shape==(800,1200,3)` 且 `dtype==uint16`。**transform 要在原生 800×1200 上做，不要在 audit 用的 512×960 center crop 上做**——eval 时 dataset 会自己对 val 做 center crop（见 §2.2）。
- OpenCV dtype 限制：`cv2.medianBlur` 在 float32 上只支持 ksize 3/5；`cv2.bilateralFilter` 需要 8U 或 32F。raw 已是 `[0,1]` float32，所以 `bilateralFilter` 的 `sigmaColor` 要用 `[0,1]` 量纲（约 `0.005~0.05`），不是像素值量纲——§5.3 的 `1x/2x/4x sigma_noise` 缩放只有在 `sigma_noise` 也在归一化域上估计时才成立（§2.4C 正是如此）。
- 溯源：每条 ablation eval 记录的 `input_domain`/`student_input_kind` 标签都一样，transform 身份只活在 save-path / manifest 名里。join 时务必额外写一列 `transform_name`，否则 per-sample CSV 自身无法区分是哪个 transform。

---

## 2. 新增脚本 A：`tools/lod_raw_degradation_audit.py`

### 2.1 目的

对 paired `RAW_Dark` / `RAW_normal` 做纯数据层面的统计，不跑模型。输出每个 sample 的退化指标，并生成 summary 和可视化 panel。

### 2.2 命令模板

```bash
python tools/lod_raw_degradation_audit.py \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --splits 00Train,01Valid \
  --crop-mode center512x960 \
  --max-samples-per-split -1 \
  --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1
```

**crop 对齐（决定 per-sample join 是否有效）：** audit 的 `center512x960` 必须复现 dataset 对 val 的 center crop 偏移，否则 §3/§8 的 `delta_d1` 与 audit 指标算的不是同一批像素。从原生 `(800,1200)` 到 `(512,960)` 的 center crop 是 `top=(800-512)//2=144`、`left=(1200-960)//2=120`（见 `lod_raw.py:_sample_crop_box` 的 center 分支）。audit 脚本必须用同样的 `[144:144+512, 120:120+960]`。

建议第一轮：

```text
01Valid 全量(112) + 00Train 抽样 512 或 1024 张
```

如果运行时间可以接受，再跑 train 全量。注：`--max-samples-per-split -1` 是本 audit 脚本自定义语义（=全量），与 train.py 的 `--debug-max-val-samples` 不同——后者传 `-1` 会评估 0 个样本（见 §0.3 / §6.2），不要混用。

### 2.3 输出文件

```text
lod_raw_diag/audit_v1/
  per_sample_raw_stats.csv
  summary_by_split.json
  summary_by_split.md
  histograms/
    raw_dark_vs_normal_percentiles.png
    ev_gap_distribution.png
    noise_proxy_distribution.png
    edge_retention_distribution.png
  panels/
    high_noise_top20/*.png
    high_ev_gap_top20/*.png
    low_edge_retention_top20/*.png
    random_val/*.png
```

### 2.4 每个 sample 必须计算的指标

#### A. 全局亮度 / 动态范围

对 RGB 三通道和 luma 都计算：

```python
luma = 0.299 * R + 0.587 * G + 0.114 * B
```

指标：

```text
p0.1, p1, p5, p10, p50, p90, p95, p99, p99.9, max
mean, std
black_ratio      = mean(x <= 1 / 65535 或 x <= p0.1_threshold)
sat_ratio        = mean(x >= 65534 / 65535)
dynamic_range_98 = p99 - p1
```

曝光 gap：

```text
gain_p95_c = p95(raw_normal_c) / max(p95(raw_dark_c), eps)
ev_gap_c   = log2(gain_p95_c)

gain_p95_luma = p95(luma_normal) / max(p95(luma_dark), eps)
ev_gap_luma   = log2(gain_p95_luma)
```

解释：

| 现象 | 解释 |
|---|---|
| `ev_gap_luma` 很大，且 D1 gap 与它强相关 | 曝光/动态范围是主因 |
| `raw_dark` p99 很低但 edge retention 尚可 | 可能用 tone mapping / learned illumination 恢复 |
| `raw_dark` p99 低且 edge retention 也低 | 仅提亮可能不够，需要 denoise/restoration |

#### B. 通道统计 / 颜色偏移

计算：

```text
mean_R, mean_G, mean_B
p95_R, p95_G, p95_B
chromaticity = [R, G, B] / (R + G + B + eps)
Δchromaticity = L1(chromaticity_dark - chromaticity_normal)
channel_gain_ratio = gain_p95_R : gain_p95_G : gain_p95_B
```

解释：

| 现象 | 解释 |
|---|---|
| 三通道 gain 很不一致 | RAM 可能被迫同时做 WB/CCM/曝光补偿 |
| `Δchromaticity` 与 D1 gap 强相关 | 需要 RAW color/stat normalization，而不只是去噪 |

#### C. 噪声 proxy / SNR proxy

不能直接把高频都当噪声，因为场景纹理也是高频。建议只在 flat patch 上估计噪声。

步骤：

```python
# 1. 用 RAW_normal luma 找 flat 区域
G_normal = sobel_magnitude(luma_normal)
flat_mask = G_normal < percentile(G_normal, 20)

# 2. 对 raw_dark / raw_normal 做 median blur 残差
residual_dark   = raw_dark   - median_blur(raw_dark, k=5)
residual_normal = raw_normal - median_blur(raw_normal, k=5)

# 3. 在 flat_mask 内用 MAD 估计 sigma
sigma_mad = 1.4826 * median(abs(residual - median(residual)))
```

按亮度 bin 计算：

```text
bin0: luma_normal p0-p20
bin1: luma_normal p20-p40
bin2: luma_normal p40-p60
bin3: luma_normal p60-p80
bin4: luma_normal p80-p100

snr_proxy_dark_bin = mean(raw_dark_luma_bin) / sigma_dark_bin
snr_proxy_normal_bin = mean(raw_normal_luma_bin) / sigma_normal_bin
snr_ratio_bin = snr_proxy_dark_bin / snr_proxy_normal_bin
```

关键字段：

```text
noise_sigma_dark_luma_flat
noise_sigma_normal_luma_flat
noise_sigma_ratio_dark_over_normal
snr_proxy_dark_lowbin
snr_proxy_normal_lowbin
snr_ratio_lowbin
snr_ratio_midbin
snr_ratio_highbin
```

解释：

| 现象 | 解释 |
|---|---|
| lowbin SNR 明显下降，且与 D1 gap 强相关 | dark RAW 的低照度噪声是主因 |
| highbin SNR 尚可但 lowbin 极差 | 需要 spatially adaptive denoise / reliability gate |
| 全部 bin 都差 | RAW_dark 整体退化严重，简单局部去噪可能不够 |

#### D. 局部结构 / 边缘保真

先做一个非 oracle 的 robust normalization，避免仅因亮度尺度不同导致 edge 比较失真：

```python
x_norm = clip((x - p1) / (p99 - p1 + eps), 0, 1)
```

计算：

```text
sobel_mean_dark
sobel_mean_normal
sobel_p90_dark
sobel_p90_normal
gradient_ratio = sobel_p90_dark / sobel_p90_normal
gradient_corr  = corr(sobel_dark, sobel_normal)
```

边缘 recall/F1：

```text
edge_normal = sobel_normal > percentile(sobel_normal, 90)
edge_dark   = sobel_dark   > percentile(sobel_dark, 90)

edge_precision = |edge_dark ∩ edge_normal| / |edge_dark|
edge_recall    = |edge_dark ∩ edge_normal| / |edge_normal|
edge_f1        = 2PR / (P + R)
```

解释：

| 现象 | 解释 |
|---|---|
| `gradient_ratio` 低、`edge_recall` 低 | RAW_dark 丢了几何边界，单纯 feature alignment 风险大 |
| edge 保留好但 D1 差 | 更像 RAM/DAv2 分布 mismatch 或 tone/stat mismatch |

#### E. RAW_dark → RAW_normal 可拟合性

对每个 sample 拟合两个简单映射：

```text
linear: raw_normal ≈ a * raw_dark + b
gamma:  raw_normal ≈ a * raw_dark^gamma + b
```

输出：

```text
linear_residual_sigma
linear_residual_mad
gamma_residual_sigma
gamma_residual_mad
gamma_value
gamma_fit_r2
```

解释：

| 现象 | 解释 |
|---|---|
| gamma/linear residual 小 | 主要是可逆 tone/exposure 问题 |
| residual 大且集中在暗部 | 噪声或局部结构不可逆损失更严重 |
| residual 大且边缘附近更大 | 可能存在运动/配准误差或局部非线性 ISP 差异 |

#### F. 配准 sanity check

虽然 LOD 理论上 pair 对齐，但诊断必须防止误把 misalignment 当成噪声/结构损失。

建议计算：

```text
phase_corr_shift_yx
phase_corr_response
ECC score 或 gradient NCC
```

处理规则：

```text
abs(shift_y) > 2 px 或 abs(shift_x) > 2 px 的 sample 单独标记。
这些 sample 不参与 edge retention 统计的主结论，只放入异常列表。
```

---

## 3. 新增脚本 B：`tools/lod_raw_per_sample_eval.py`

### 3.1 目的

输出每个 val sample 的 depth proxy 指标，尤其是：

```text
d1
abs_rel
edge_sobel_l1
edge_overlap_iou
```

然后把：

```text
RAW_dark checkpoint 的 per-sample metrics
RAW_normal checkpoint 的 per-sample metrics
```

与 `per_sample_raw_stats.csv` join，得到：

```text
delta_d1 = d1_raw_normal - d1_raw_dark
```

这是本阶段最关键的 dependent variable，但它不是纯粹的数据退化量：它同时混入了 `RAW_dark` checkpoint 和 `RAW_normal` checkpoint 两个独立训练模型的差异。因此相关性报告至少同时输出以下辅助因变量，防止把训练随机性或模型容量差异误判成图像退化归因：

```text
d1_raw_dark
1 - d1_raw_dark
d1_raw_normal
delta_d1 = d1_raw_normal - d1_raw_dark
```

如时间允许，再补一个同模型 cross-eval sanity：

```text
dark checkpoint   on raw_dark / raw_normal
normal checkpoint on raw_normal / raw_dark
```

只有当 `delta_d1`、`1-d1_raw_dark`、以及同模型 cross-eval 的趋势一致时，才把 per-sample 相关性作为强归因证据。

### 3.2 命令模板

该脚本须按 §0.3 配方搭建带 LoRA 的模型（显式 raw3/RamCore3/LoRA tap `[2,5,8,11]` + rank8/alpha16），否则 `--checkpoint` 在 strict-load 下因架构或 LoRA 权重不匹配报错。实现上建议直接读取 run 目录中的 `resolved_config.json` 作为默认值，并允许命令行显式覆盖；如果命令行传入的语义字段与 checkpoint 的 resolved config 不一致，直接报错。

```bash
DARK_RUN=0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
NORMAL_RUN=0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
MANIFEST=/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv

python tools/lod_raw_per_sample_eval.py \
  --run-dir /mnt/drive/3333_raw/0000_exp_ckpt/$DARK_RUN \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/$DARK_RUN/best_model.pth \
  --dataset-family lod_true_raw_dark_rgb16 \
  --dataset-input-mode raw_rgb16_dark \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --lora dav2_lora \
  --lora-block-mode tap \
  --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest $MANIFEST \
  --split 01Valid \
  --output-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval_raw_dark.csv

python tools/lod_raw_per_sample_eval.py \
  --run-dir /mnt/drive/3333_raw/0000_exp_ckpt/$NORMAL_RUN \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/$NORMAL_RUN/best_model.pth \
  --dataset-family lod_true_raw_normal_rgb16 \
  --dataset-input-mode raw_rgb16_normal \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --lora dav2_lora \
  --lora-block-mode tap \
  --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest $MANIFEST \
  --split 01Valid \
  --output-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval_raw_normal.csv
```

### 3.3 Join 与相关性分析

新增脚本 C：

```text
tools/lod_raw_attribution_join.py
```

输入：

```text
per_sample_raw_stats.csv
per_sample_eval_raw_dark.csv
per_sample_eval_raw_normal.csv
```

输出：

```text
attribution_join.csv
correlation_report.md
correlation_report.json
scatter_plots/
  delta_d1_vs_ev_gap_luma.png
  delta_d1_vs_snr_ratio_lowbin.png
  delta_d1_vs_edge_recall.png
  delta_d1_vs_chromaticity_delta.png
```

相关性使用：

```text
Spearman rho：主报告
Pearson r：辅助报告
```

**统计口径与陷阱（必须在报告里写明）：**

- val 只有 `n=112`。功效本身够（`rho=0.35` 距 0 约 3.6 个 SE，`p<0.001`），但单变量 rho 不足以归因，因为 `ev_gap_luma / p99_dark / dynamic_range_98 / snr_ratio_lowbin` 彼此强共线——它们会同时越过 §9 的 `|rho|>0.35` 闸，导致多条决策规则一起命中、无法区分主因。**在宣布单一主因前，至少补一个偏相关（控制 `ev_gap_luma` 后看 `snr_ratio_lowbin` 的残差相关）或一个小型多元回归（delta_d1 ~ 几个去共线后的主因子）。**
- `delta_d1 = d1_normal - d1_dark` 存在天花板/零膨胀：很多简单样本 d1≈1、delta≈0。报告 Spearman 之外，也看 delta_d1 的分布直方图，必要时只在 `delta_d1 > 0` 的子集上复算相关。

主判断：

| 最强相关变量 | 优先解释 |
|---|---|
| `ev_gap_luma`, `p99_dark`, `dynamic_range_98_dark` | 曝光 / 动态范围 |
| `snr_ratio_lowbin`, `noise_sigma_ratio` | 噪声 / 低 SNR |
| `edge_recall`, `gradient_corr`, `local_contrast_ratio` | 结构 / 边界保真 |
| `Δchromaticity`, `channel_gain_ratio` | 通道统计 / WB/CCM 问题 |
| 以上都弱，但 RAM feature gap 强 | feature-level mismatch |

---

## 4. 新增脚本 D：`tools/lod_raw_make_ablation_inputs.py`

### 4.1 目的

生成一组 controlled transformed RAW 输入，用于判断单因素修复或单因素退化能解释多少 D1 gap。

### 4.2 输出目录结构

```text
lod_raw_diag/ablation_inputs/
  manifests/
    raw_dark_identity.csv
    raw_dark_denoise_median3.csv
    raw_dark_denoise_bilateral.csv
    raw_dark_tone_percentile_trainfit.csv
    raw_dark_tone_gamma_trainfit.csv
    raw_dark_tone_denoise_trainfit.csv
    raw_normal_to_dark_exposure_trainfit.csv
    raw_normal_to_dark_noise_only_trainfit.csv
    raw_normal_to_dark_exposure_noise_trainfit.csv
  png16/
    raw_dark_denoise_median3/...
    raw_dark_denoise_bilateral/...
    raw_dark_tone_percentile_trainfit/...
    ...
  transform_params/
    trainfit_percentile_params.json
    trainfit_gamma_params.json
    trainfit_noise_model.json
```

### 4.3 两类 transform 必须分开标记

#### 类型 1：oracle diagnostic transform

使用同一个 val sample 的 `RAW_normal` 来拟合 transform 参数。

用途：诊断 upper bound。  
不能作为公平模型结果汇报。

例子：

```text
raw_dark_tone_percentile_oracle
raw_dark_gamma_oracle
raw_dark_tone_denoise_oracle
```

#### 类型 2：deployable proxy transform

只允许使用：

```text
训练集统计参数
或 RAW_dark 本身可估计的参数
```

不能访问 val sample 的 `RAW_normal`。

用途：可以作为下一步方法选择依据。

例子：

```text
raw_dark_tone_percentile_trainfit
raw_dark_gamma_trainfit
raw_dark_denoise_bilateral
raw_dark_tone_denoise_trainfit
```

本阶段报告中必须把 oracle 与 deployable 分栏。

---

## 5. Ablation transform 设计

### 5.1 `raw_dark_identity`

目的：确认 manifest 重写、PNG16 写盘和 eval 流程不改变结果。

处理：直接复制 `RAW_Dark`。

预期：

```text
D1 应与原 RAW_dark run 基本一致。
```

如果不一致，先修复数据生成/读取流程，不进入后续分析。

---

### 5.2 `raw_dark_denoise_median3`

目的：粗测 salt-like / high-frequency 噪声是否影响 depth。

处理：

```python
out_c = medianBlur(raw_dark_c, k=3)
```

注意：median 容易损伤细边界，只作为诊断，不作为最终方案。

解释：

| 结果 | 判断 |
|---|---|
| D1 明显上升 | 高频噪声是有效因素 |
| D1 下降 | 普通去噪抹掉了 depth-useful edge，后续要做保边/restoration |
| D1 基本不变 | 简单高频噪声不是主因 |

---

### 5.3 `raw_dark_denoise_bilateral`

目的：测试 edge-preserving denoise 是否比 median 更适合。

处理：

```python
out = bilateralFilter(raw_dark, d=5, sigmaColor=?, sigmaSpace=?)
```

参数建议用 audit 中的 `noise_sigma_dark_luma_flat` 估计，先设三档：

```text
weak   : sigmaColor = 1.0 * sigma_noise
medium : sigmaColor = 2.0 * sigma_noise
strong : sigmaColor = 4.0 * sigma_noise
```

注意 oracle 泄漏边界：如果 `sigma_noise` 来自 val sample 的 paired `RAW_normal` flat mask 或 paired residual，它只能标为 `oracle`。deployable bilateral 只能使用 `00Train` 拟合出的全局/分亮度 bin 参数，或只从当前 `RAW_dark` 自身估计的参数。报告中必须拆成：

```text
raw_dark_bilateral_*_oracle_sigma
raw_dark_bilateral_*_trainfit_sigma
raw_dark_bilateral_*_darkonly_sigma
```

输出三个 manifest：

```text
raw_dark_bilateral_weak.csv
raw_dark_bilateral_medium.csv
raw_dark_bilateral_strong.csv
```

---

### 5.4 `raw_dark_tone_percentile_trainfit`

目的：单独测试曝光/动态范围是否是主因。

训练集拟合每通道 mapping：

```text
for channel c:
  p1_dark_train_c, p99_dark_train_c
  p1_normal_train_c, p99_normal_train_c

x' = (x - p1_dark_train_c) / (p99_dark_train_c - p1_dark_train_c + eps)
y  = x' * (p99_normal_train_c - p1_normal_train_c) + p1_normal_train_c
y  = clip(y, 0, 1)
```

同时生成 luma-only 版本：

```text
raw_dark_tone_percentile_luma_trainfit
```

用途：区分“整体亮度尺度”与“三通道颜色统计”。

解释：

| 结果 | 判断 |
|---|---|
| tone mapping 单独恢复大量 gap | 优先做 illumination/dynamic-range normalization |
| luma 版有效、RGB per-channel 版更有效 | 通道统计也重要 |
| tone mapping 视觉变亮但 D1 不变 | 可能不是曝光主因，或 RAM/DAv2 不适应 transform |

---

### 5.5 `raw_dark_tone_gamma_trainfit`

目的：测试 RAW_dark → RAW_normal 是否主要是非线性 tone curve 问题。

训练集拟合：

```text
raw_normal ≈ a_c * raw_dark ^ gamma_c + b_c
```

保存：

```text
gamma_c, a_c, b_c, residual_sigma_c
```

建议拟合时只用中间亮度范围，避免黑电平和高光支配：

```text
mask = raw_dark_luma between p5 and p95
```

解释：

| 结果 | 判断 |
|---|---|
| gamma mapping 明显优于 linear/percentile | 低光非线性响应是重要因素 |
| gamma residual 仍很大 | 噪声/结构损失更重要 |

---

### 5.6 `raw_dark_tone_denoise_trainfit`

目的：测试 tone 和 denoise 是否互补。

顺序要做两版：

```text
A: denoise -> tone
B: tone -> denoise
```

因为提亮后噪声也会被放大，两个顺序可能差异明显。

解释：

| 结果 | 判断 |
|---|---|
| tone + denoise 明显高于单独 tone/denoise | joint restoration 是主线 |
| denoise -> tone 优于 tone -> denoise | 低照度噪声需要先处理 |
| tone -> denoise 优于 denoise -> tone | 提亮后更容易估计/抑制噪声 |

---

> **合成退化组(E7–E9)统一说明（实验语义，必须显式）：**
> - **zero-train 快筛：固定用 RAW_normal 模型 NL**（`0609_0044/best_model.pth`，显式 `--dataset-family lod_true_raw_normal_rgb16 --dataset-input-mode raw_rgb16_normal`），把退化图写进 `raw_normal_path` 列，看 D1 从 `0.899` 往 `0.845` 掉多少。这只是 test-time domain shift 下的方向性快筛（§13.2），不作准。
> - **必做同配方短训确认（本计划采用此方案，不靠纯 eval 下结论）：** 对每个退化用 §6.3 的 `CANONICAL_COMMON_FLAGS`（normal 显式输入语义，manifest 的 `raw_normal_path` 指向退化 PNG）从 pretrained 起短训同一 `SHORT_EPOCHS/MAX_STEPS_PER_EPOCH`、≥2 seed。这样 `drop_ratio` 与 dark 侧 `recovered_ratio` 处在同一训练预算、同一配方下，可等价对待。
> - **短训 drop_ratio 归一化**（与 §6.3 共用 floor/ceiling）：
>   ```text
>   drop_ratio_st(degrade) = [ceiling_st - D1_st(degrade)] / (ceiling_st - floor_st)
>   floor_st   = 短训 raw_dark_identity 的 best lod_d1
>   ceiling_st = 短训 raw_normal_identity 的 best lod_d1
>   ```
>   `drop_ratio_st ≈ 1.0` 表示该退化（如 exposure+noise）已把 normal 打到真实 dark 的水平，即退化模型基本解释 gap。

### 5.7 `raw_normal_to_dark_exposure_trainfit`

目的：从反方向验证曝光/动态范围是否足以造成 D1 drop。

处理：用训练集拟合 normal → dark 的 inverse tone mapping：

```text
RAW_normal -> synthetic dark exposure only
```

不加噪声。

解释：

| 结果 | 判断 |
|---|---|
| D1 从 0.899 大幅掉向 0.845 | 曝光/动态范围本身能解释大部分 gap |
| D1 仍接近 0.899 | 单纯变暗不够，真实 RAW_dark 的噪声/结构破坏更关键 |

---

### 5.8 `raw_normal_to_dark_noise_only_trainfit`

目的：单独测试噪声模型的影响。

处理：保持 RAW_normal 的亮度分布，加入从训练集估计的 noise model。

噪声模型建议：

```text
sigma^2(x) = a * x + b
```

其中 `a` 近似 shot noise，`b` 近似 read noise。用 `RAW_dark` 与 `RAW_normal -> dark exposure` 的 residual 拟合。

拟合口径必须保守，否则会把结构残差当噪声：

```text
split       : 只用 00Train
mask        : flat mask / low-gradient 区域，排除 edge 与 registration outlier
tone source : exposure model 只能用 trainfit 参数
randomness  : 固定并记录 noise_seed，至少 seed 42 / 123 复算一次
report      : 保存 a,b、bin-wise residual sigma、以及 synthetic noise 前后 p1/p50/p99/SNR summary
```

解释：

| 结果 | 判断 |
|---|---|
| noise-only 导致明显 D1 drop | 噪声本身是主因 |
| noise-only drop 小 | 噪声不是单独主因，可能与低曝光耦合 |

---

### 5.9 `raw_normal_to_dark_exposure_noise_trainfit`

目的：测试“曝光 + 噪声”能否复现真实 `RAW_dark` 的性能。

```text
RAW_normal -> exposure matched dark -> add fitted noise
```

解释：

| 结果 | 判断 |
|---|---|
| D1 接近真实 RAW_dark | 当前退化模型基本解释 gap |
| D1 明显高于真实 RAW_dark | 真实 RAW_dark 还有未建模因素，例如局部 ISP、配准、非高斯噪声、颜色偏移 |
| D1 明显低于真实 RAW_dark | synthetic noise 过强或破坏了真实结构分布 |

---

## 6. 快速模型评估流程

### 6.1 Level-0：流程一致性检查

先跑：

```text
raw_dark_identity.csv
```

目标：

```text
D1(identity) 与原始 RAW_dark D1 差异 < 0.002
```

如果不满足，检查：

```text
PNG16 写盘是否反了 BGR/RGB
manifest path 是否写错
crop/eval split 是否一致
checkpoint 是否一致
raw norm mode 是否仍是 uint16_div_65535
```

### 6.2 Level-1：zero-training eval

目的：快速看 test-time transform 对已训练 RAW_dark checkpoint 是否有方向性收益。

命令模板：

命令（`CANONICAL_COMMON_FLAGS` + dark 显式输入语义 + eval_only；tap 层是 strict-load L0 的硬条件；**不传 `--debug-max-val-samples`** = 全量 112）：

```bash
DARK_RUN=0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
TRANSFORM_NAME=<TRANSFORM_NAME>
PORT=29651

conda run --live-stream -n dav3 torchrun --nproc_per_node=1 --master_port ${PORT} finetune_stf/train.py \
  --stage eval_only \
  --encoder vits \
  --dataset-family lod_true_raw_dark_rgb16 \
  --dataset-input-mode raw_rgb16_dark \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --resume-from /mnt/drive/3333_raw/0000_exp_ckpt/$DARK_RUN/best_model.pth \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest <ABLATION_MANIFEST_block8.csv> \
  --lod-label-space inverse_relative \
  --lod-train-crop-mode random \
  --lod-val-crop-mode center \
  --raw-storage-format raw_rgb16_png_3ch \
  --lod-raw-norm-mode uint16_div_65535 \
  --raw-ram-rgb-tail identity \
  --lora dav2_lora --lora-block-mode tap --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 --lora-alpha 16 --lora-lr 5e-5 --raw-front-end-lr 5e-5 \
  --dav2-train-mode decoder --backbone-layer-decay 1.0 \
  --lr 1e-5 --lr-schedule poly --warmup-steps 0 --loss-type ssi \
  --loss-target-normalization --loss-norm-min-scale 1e-3 \
  --bs 8 --accum-steps 1 \
  --input-height 512 --input-width 960 \
  --aug-preset off --aug-hflip-prob 0.5 \
  --eval-lod \
  --no-eval-stf \
  --eval-lod-train-proxy \
  --lod-train-proxy-count 112 \
  --amp --amp-dtype bf16 \
  --num-workers 4 --log-interval 500 \
  --no-enable-fixed-viz-dump \
  --no-enable-train-source-viz-dump \
  --save-path /home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/lod_raw_diag/eval_zero/${TRANSFORM_NAME} \
  --heavy-save-root /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero
```

注意：zero-training eval 有 domain shift，不作为最终结论，只作为筛选。合成退化组(E7–E9)的 zero-train 把输入语义换成 `--dataset-family lod_true_raw_normal_rgb16 --dataset-input-mode raw_rgb16_normal`、`--resume-from` 换成 NL 的 `0609_0044/best_model.pth`（见 §5.7 callout）。

### 6.3 Level-2：短训确认

对 Level-1 中最有希望的 3-5 个 transform 做短训。

建议：

```text
训练预算：正式训练的 10%-20%，或 3-5 个完整 epoch
可选加速：若限制每 epoch step，必须写成 SHORT_EPOCHS × MAX_STEPS_PER_EPOCH
重复 seed：至少 2 个 seed（如 42 / 123）
保存 best by lod_d1
```

**短训归一化口径（关键，别用全训 gap 当分母）：** 短训 3–5 epoch 达不到全训的 best（L0 是 e37=0.8451），所以 `recovered_ratio` 的 floor/ceiling 必须用**同配方、同 `SHORT_EPOCHS/MAX_STEPS_PER_EPOCH`** 的两个锚点：

```text
floor   = 短训 raw_dark_identity（E0，identity manifest）的 best lod_d1
ceiling = 短训 raw_normal_identity（同步预算、显式 normal 输入语义、读 raw_normal_path）的 best lod_d1
recovered_ratio_st(transform) = [D1_st(transform) - floor] / (ceiling - floor)
```

不要用 `0.054` 这个全训 gap 归一化短训 delta。也不要把 `--debug-max-train-steps` 理解成“总步数”：当前 `train.py` 中它限制的是**每个 epoch 的 step 数**，实际总训练步数约为 `--epochs * --debug-max-train-steps / accum_steps`。

命令模板（`CANONICAL_COMMON_FLAGS` + dark 显式输入语义 + lod_only；short-train 从 pretrained 起训，不 resume L0）。short-train 如果预期超过几小时，必须用 tmux 包一层并记录 log：

```bash
MANIFEST=<ABLATION_MANIFEST_block8.csv>   # identity / 各 transform 各自一份
TRANSFORM_NAME=<TRANSFORM_NAME>
SHORT_EPOCHS=5
MAX_STEPS_PER_EPOCH=80
SEED=42
PORT=29661
RUN_NAME=${TRANSFORM_NAME}_e${SHORT_EPOCHS}_s${MAX_STEPS_PER_EPOCH}_seed${SEED}

conda run --live-stream -n dav3 torchrun --nproc_per_node=1 --master_port ${PORT} finetune_stf/train.py \
  --stage lod_only \
  --encoder vits \
  --dataset-family lod_true_raw_dark_rgb16 \
  --dataset-input-mode raw_rgb16_dark \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest $MANIFEST \
  --lod-label-space inverse_relative \
  --lod-train-crop-mode random \
  --lod-val-crop-mode center \
  --raw-storage-format raw_rgb16_png_3ch \
  --lod-raw-norm-mode uint16_div_65535 \
  --raw-ram-rgb-tail identity \
  --lora dav2_lora --lora-block-mode tap --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 --lora-alpha 16 --lora-lr 5e-5 --raw-front-end-lr 5e-5 \
  --dav2-train-mode decoder --backbone-layer-decay 1.0 \
  --lr 1e-5 --lr-schedule poly --warmup-steps 0 --loss-type ssi \
  --loss-target-normalization --loss-norm-min-scale 1e-3 \
  --bs 8 --accum-steps 1 --input-height 512 --input-width 960 \
  --aug-preset off --aug-hflip-prob 0.5 \
  --eval-lod --no-eval-stf \
  --eval-lod-train-proxy --lod-train-proxy-count 112 \
  --amp --amp-dtype bf16 \
  --num-workers 4 --log-interval 500 \
  --no-enable-fixed-viz-dump \
  --no-enable-train-source-viz-dump \
  --save-best-checkpoint --best-metric lod_d1 \
  --epochs ${SHORT_EPOCHS} \
  --debug-max-train-steps ${MAX_STEPS_PER_EPOCH} \
  --save-path /home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/lod_raw_diag/short_train/${RUN_NAME} \
  --heavy-save-root /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train \
  --seed ${SEED}
```

### 6.4 Level-3：正式复现实验

只对满足以下条件之一的 transform 做完整训练：

```text
zero-training recovered_ratio > 0.20
或 short-train recovered_ratio_st > 0.30
或 synthetic degrade short-train drop_ratio_st > 0.50
```

正式实验至少包括（全部用 §0.3 同配方和显式输入语义；reference 列直接复用已有 run，不必重训）：

```text
RAW_dark baseline      -> 复用 L0 (0608_2026, 0.8451@e37)
best transform #1      -> 新训（CANONICAL_COMMON_FLAGS + dark 显式输入语义 + 该 transform manifest，全 40 epoch）
best transform #2      -> 新训
RAW_normal oracle ref  -> 复用 NL (0609_0044, 0.8990@e36)
RGB_dark reference     -> 复用 RL (0608_2246, 0.8555@e22)
```

正式 run 名按规范以 `MMDD_HHMM` 起头，并必须用 tmux 启动，报告：

```text
tmux session:
queue/train log:
attach command:
tail command:
light save-path:
heavy save-root:
```

---

## 7. RAM output / feature-domain probe

如果图像域 transform 改善了 audit 指标，但 D1 不明显提升，需要判断是否进入 feature-level alignment。

新增脚本：

```text
tools/lod_raw_ram_feature_probe.py
```

### 7.1 RAM output probe

加载模型 checkpoint，手动调用：

```python
x3, features = model.ram_core.forward_with_features(raw)
```

保存：

```text
ram_x3_p1/p50/p99/min/max/std
ram_x3_abs_tail_ratio = mean(abs(x3) > 2.5)
ram_x3_channel_corr
ram_x3_histogram_distance_to_raw_normal
ffm_mid_mean/std/norm
x_cat_mean/std/norm
```

比较对象：

```text
RAW_dark through RAW_dark model
RAW_normal through RAW_normal model
RAW_dark transformed through RAW_dark model
```

注意模型差异混杂：上面三项可以作为实际系统状态对比，但 feature attribution 还应补同模型输入对比，至少包含：

```text
RAW_dark model   : raw_dark vs transformed_raw_dark
RAW_dark model   : raw_dark vs raw_normal（oracle sanity）
RAW_normal model : raw_normal vs synthetic_degraded_normal
```

否则 RAM/DAv2 feature distance 可能反映的是两个 checkpoint 的训练差异，而不是输入域差异。

解释：

| 现象 | 判断 |
|---|---|
| transformed RAW 让 RAM x3 更接近 RAW_normal，但 D1 不升 | backbone/feature mismatch 更可能 |
| RAM x3 仍与 RAW_normal 差异大 | preprocessing 不够，需要 learned restoration 或改 RAM |
| RAM x3 已接近但 edge error 仍大 | DPT/feature-level alignment 或 decoder-side fusion 更重要 |

### 7.2 DAv2 token / feature probe

建议先不要改 DAv2 forward，可用 forward hook 抓取几个层的输出：

```text
early block: 2 / 3
middle block: 5 / 6
late block: 9 / 11 for ViT-S
```

每个 sample 输出：

```text
cosine_distance(raw_dark_feature, raw_normal_feature)
L2_distance_normalized
token_std
token_norm
CLS/token mean norm
```

关联：

```text
feature_distance vs delta_d1
feature_distance vs image-level audit metrics
```

判断：

| 结果 | 下一步 |
|---|---|
| feature distance 与 delta_d1 强相关，且 image metrics 解释弱 | feature-level alignment 优先 |
| image metrics 与 delta_d1 强相关，feature distance 只是中介 | RAW restoration / denoise 优先 |
| 两者都强 | 做 restoration + feature distillation，而不是二选一 |

---

## 8. 最终归因报告格式

建议每轮输出一个固定格式 markdown：

```text
lod_raw_diag/report_v1.md
```

结构如下：

```markdown
# LOD RAW_Dark Degradation Attribution Report

## 1. Summary
- Baseline D1 RAW_dark:
- Oracle D1 RAW_normal:
- Gap G:
- Main attributed factor:

## 2. Data-level audit
- exposure gap summary
- noise/SNR summary
- edge retention summary
- chromaticity summary
- registration outliers

## 3. Per-sample correlation
| variable | Spearman rho with delta_d1 | p-value | interpretation |

同时报告 `d1_raw_dark`、`1-d1_raw_dark`、`d1_raw_normal` 的相关性，以及同模型 cross-eval sanity 是否支持同一方向。

## 4. Ablation eval
| transform | zero-train D1 | short-train D1 | recovered_ratio_st | type: oracle/deployable |

（floor_st = A1 raw_dark_identity 短训；ceiling_st = A2 raw_normal_identity 短训）

## 5. Synthetic degrade eval
| degrade | zero-train D1 | short-train D1 | drop_ratio_st | interpretation |

## 6. Decision
- Next main direction:
- Secondary direction:
- Do not prioritize yet:

## 7. Failure galleries
- high noise samples
- high EV gap samples
- low edge retention samples
- feature mismatch samples
```

---

## 9. 决策规则：下一步到底做 denoise 还是 feature-level alignment

### 9.1 优先做 RAW 去噪 / restoration 的条件

满足任意两条即可：

```text
1. snr_ratio_lowbin 与 delta_d1 的 Spearman |rho| > 0.35
2. denoise 或 tone+denoise 的 short-train recovered_ratio_st > 0.30
3. raw_normal_to_dark_noise 或 exposure+noise 的 short-train drop_ratio_st > 0.40
4. edge_recall / gradient_corr 与 delta_d1 强相关
5. failure gallery 显示错误集中在噪声重、边界被污染区域
```

下一步方法方向：

```text
depth-oriented RAW restoration
RAW_dark -> restored RAW / restored RAM input
loss = depth distillation + RAW_normal structural target + optional feature target
```

### 9.2 优先做 exposure / illumination normalization 的条件

满足任意两条即可：

```text
1. ev_gap_luma 或 p99_dark 与 delta_d1 的 Spearman |rho| > 0.35
2. tone_percentile 或 tone_gamma short-train recovered_ratio_st > 0.30
3. raw_normal_to_dark_exposure short-train drop_ratio_st > 0.40
4. denoise 单独无效，但 tone 有效
```

下一步方法方向：

```text
learned illumination / dynamic range normalization
不要先做复杂 denoiser
可以把 RAW_normal teacher 当作 exposure-normal target
```

### 9.3 优先做 feature-level alignment 的条件

满足任意两条即可：

```text
1. image-level transform 明显改善 histogram/SNR/edge，但 D1 不升
2. RAM x3 / DAv2 token distance 与 delta_d1 强相关
3. denoise/tone short-train recovered_ratio_st < 0.15
4. RAW_dark 和 RAW_normal 的图像统计差异不大，但模型差异大
5. transformed RAW 的 RAM x3 分布仍显著偏离 RAW_normal/RGB teacher feature manifold
```

下一步方法方向：

```text
RAW_dark student 对齐 RAW_normal teacher feature
或 RGB_normal teacher feature
但需要加 reliability mask，避免在低 SNR 区域强行 hallucinate
```

### 9.4 做 restoration + feature distillation 的条件

如果：

```text
tone/denoise short-train recovered_ratio_st 在 0.20-0.40
且 feature distance 仍明显预测 delta_d1
```

则不要二选一，下一步直接做：

```text
RAW_dark restoration front-end
+ RAW_normal teacher feature distillation
+ depth pseudo-label loss
```

这是最可能的最终组合，但不建议在本诊断阶段提前实现。

---

## 10. 推荐执行顺序

### Day 1：数据审计脚本

```text
实现 tools/lod_raw_degradation_audit.py
跑 01Valid 全量(112)
跑 00Train 抽样
输出 per_sample_raw_stats.csv + summary.md
```

检查：

```text
RAW_dark / RAW_normal 是否严格 paired
是否有明显 misalignment outlier
是否存在极端黑电平 / 饱和 / channel 异常
```

### Day 2：per-sample eval + join

```text
实现 tools/lod_raw_per_sample_eval.py
分别评估 RAW_dark checkpoint 和 RAW_normal checkpoint
实现 tools/lod_raw_attribution_join.py
输出 delta_d1 correlation report
```

第一轮判断：

```text
delta_d1 更像由 exposure、noise、edge、channel 中哪一类解释？
```

### Day 3：生成 ablation manifest

```text
实现 tools/lod_raw_make_ablation_inputs.py
生成 identity / denoise / tone / synthetic degrade manifests
先跑 identity 确认流程无偏差
```

### Day 4：zero-training eval

```text
对所有 ablation manifest 跑 eval-only
输出 ablation_zero_train_table.csv
筛选 top 3-5 个 transform
```

### Day 5-7：短训确认

```text
先短训两个归一化锚点：A1 raw_dark_identity(floor_st) 与 A2 raw_normal_identity(ceiling_st)
对 top dark transforms 做 short train
对合成退化 E7-E9 也做 short train（显式 normal 输入语义，退化图写 raw_normal_path）
每个 transform / 退化 / 锚点 至少 2 seeds（42/123）
输出 short_train_table.csv：dark 侧记 recovered_ratio_st，degrade 侧记 drop_ratio_st
```

### Day 8：RAM/feature probe

只有当 image-level 结果解释不足时执行。

```text
实现 lod_raw_ram_feature_probe.py
比较 RAW_dark / RAW_normal / transformed RAW 的 RAM x3 与 DAv2 token 差异
```

### Day 9：归因报告

```text
汇总 report_v1.md
给出下一阶段主线：
- denoise/restoration
- illumination normalization
- feature-level alignment
- restoration + feature distillation
```

---

## 11. 最小实验矩阵

第一轮不要过大，建议如下：

| 编号 | 输入/处理 | 类型 | 是否训练 | 目的 |
|---:|---|---|---|---|
| E0 | RAW_dark identity | sanity | no | 验证 manifest/PNG 流程 |
| A1 | RAW_dark identity（短训 floor） | anchor | short | 短训归一化下界 floor_st |
| A2 | RAW_normal identity（短训 ceiling） | anchor | short | 短训归一化上界 ceiling_st |
| E1 | RAW_dark median3 | deployable | no + short | 粗测高频噪声 |
| E2 | RAW_dark bilateral weak/medium/strong | deployable | no + short | 测试保边去噪 |
| E3 | RAW_dark percentile tone trainfit | deployable | no + short | 测试曝光/动态范围 |
| E4 | RAW_dark gamma tone trainfit | deployable | no + short | 测试非线性 tone |
| E5 | RAW_dark denoise -> tone | deployable | no + short | 测试联合恢复 |
| E6 | RAW_dark tone -> denoise | deployable | no + short | 测试联合恢复顺序 |
| E7 | RAW_normal -> dark exposure | synthetic degrade | no + short | 反向验证曝光因素 |
| E8 | RAW_normal -> noise only | synthetic degrade | no + short | 反向验证噪声因素 |
| E9 | RAW_normal -> exposure + noise | synthetic degrade | no + short | 验证退化模型完整性 |
| E10 | RAW_dark tone oracle | oracle | no | 估计可恢复上界 |
| E11 | RAW_dark tone+denoise oracle | oracle | no | 估计 restoration 上界 |

---

## 12. 可能的结果模式与对应结论

### 模式 A：tone mapping 有效，denoise 无效

表现：

```text
E3/E4 recovered_ratio 高
E1/E2 recovered_ratio 低
E7 drop_ratio 高
```

结论：

```text
RAW_dark 的主问题是曝光/动态范围，不是传统噪声。
下一步优先做 illumination normalization 或 RAW_dark -> RAW_normal tone restoration。
feature alignment 暂时作为辅助。
```

### 模式 B：denoise 有效，tone 无效

表现：

```text
E1/E2 recovered_ratio 高
E3/E4 recovered_ratio 低
E8 drop_ratio 高
snr_ratio_lowbin 与 delta_d1 强相关
```

结论：

```text
RAW_dark 的主问题是噪声/SNR。
下一步优先做 depth-oriented RAW denoise/restoration。
注意必须保边，不建议普通 L1 denoiser。
```

### 模式 C：tone + denoise 有效，单独都一般

表现：

```text
E5/E6 明显高于 E1/E2/E3/E4
E9 能复现真实 RAW_dark drop
```

结论：

```text
主问题是低曝光与噪声耦合。
下一步做 joint restoration，而不是单独 denoise 或单独 tone。
```

### 模式 D：图像域 transform 改善统计，但 D1 不升

表现：

```text
audit 显示 transformed RAW 更接近 RAW_normal
但 zero-train/short-train recovered_ratio 都低
RAM x3 或 DAv2 token gap 仍大
```

结论：

```text
优先转向 feature-level alignment 或 RAM redesign。
图像域 preprocessing 不是足够接口。
```

### 模式 E：oracle transform 有效，trainfit transform 无效

表现：

```text
E10/E11 recovered_ratio 高
E3/E4/E5/E6 recovered_ratio 低
```

结论：

```text
RAW_dark 中确实存在可恢复信息，但简单全局参数不够。
下一步应做 learned restoration，并用 RAW_normal 作为 teacher/target。
```

---

## 13. 报告中不要混淆的两件事

### 13.1 诊断 upper bound 不等于公平方法

只要 transform 使用了同一个 val sample 的 `RAW_normal`，它就是 oracle diagnostic。它只能回答：

```text
如果知道 normal 视图，最多能恢复多少？
```

不能作为最终实验与 RGB_dark / RAW_dark baseline 对比。

### 13.2 zero-training eval 不等于最终模型能力

test-time transform 会改变训练分布。即使 zero-training eval 不升，也不能立刻否定该因素；必须看 short-train 结果。

推荐最终排序依据：

```text
short-train recovered_ratio > zero-training recovered_ratio > pure image audit
```

---

## 14. 交付物清单

完成本阶段后，应该有以下文件：

```text
lod_raw_diag/
  audit_v1/
    per_sample_raw_stats.csv
    summary_by_split.md
    panels/
  per_sample_eval/
    raw_dark_val.csv
    raw_normal_val.csv
  attribution/
    attribution_join.csv
    correlation_report.md
    scatter_plots/
  ablation_inputs/
    manifests/*.csv
    transform_params/*.json
  eval_zero/
    ablation_zero_train_table.csv
  short_train/
    short_train_table.csv
  report_v1.md
```

最终 `report_v1.md` 必须明确给出：

```text
主瓶颈：exposure / noise / structure / channel / feature mismatch 中的哪一个或哪几个
下一阶段主线：denoise/restoration、illumination normalization、feature alignment、joint restoration+feature distillation
不优先做什么，以及理由
```

---

## 15. 推荐结论模板

```markdown
# Conclusion

RAW_dark -> RAW_normal 的 0.054 D1 gap 中：

- exposure/dynamic range 可解释约 XX%
- noise/SNR 可解释约 XX%
- edge/structure loss 可解释约 XX%
- channel/stat shift 可解释约 XX%
- RAM/DAv2 feature mismatch 仍解释约 XX%

因此下一阶段优先：

1. <main direction>
2. <secondary direction>

暂不优先：

- <direction>: <reason>
```

---

## 16. 执行日志

### 2026-06-09 13:33 CST - 开始

状态：进行中。

环境确认：

```text
host: caq
user: caq
cwd: /home/caq/6666_raw/dav2_raw_0603
默认 conda 环境：dav3
```

初始执行范围：

1. 实现 `tools/lod_raw_degradation_audit.py`。
2. 使用临时 `codex_smoke` 输出 smoke-test audit 脚本。
3. 在 `01Valid` 全量和 `00Train` 抽样上运行 `audit_v1`。
4. 在这里记录 audit 命令、输出路径和关键汇总指标。

### 2026-06-09 13:37 CST - audit 脚本 smoke 通过

已实现：

```text
tools/lod_raw_degradation_audit.py
```

已运行检查：

```bash
conda run -n dav3 python -m py_compile tools/lod_raw_degradation_audit.py
conda run -n dav3 python tools/lod_raw_degradation_audit.py --help
conda run -n dav3 python tools/lod_raw_degradation_audit.py \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --splits 01Valid \
  --crop-mode center512x960 \
  --max-samples-per-split 2 \
  --panel-count 2 \
  --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_audit_v1
```

Smoke 结果：

```text
选择样本：01Valid 中 2/112
已检查输出：per_sample_raw_stats.csv、summary_by_split.{json,md}、histograms、panels
smoke 中配准异常样本：0
已清理临时 smoke 输出：/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_audit_v1
```

### 2026-06-09 13:42 CST - audit_v1 完成

命令：

```bash
conda run -n dav3 python tools/lod_raw_degradation_audit.py \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --splits 01Valid,00Train \
  --crop-mode center512x960 \
  --max-samples-per-split 512 \
  --sample-seed 42 \
  --panel-count 20 \
  --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1
```

输出：

```text
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/per_sample_raw_stats.csv
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/summary_by_split.json
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/summary_by_split.md
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/histograms/*.png
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/panels/*/*.png
```

输出检查：

```text
行数：624
CSV 字段数：221
audit_v1 下文件数：88
```

关键汇总：

| split | 样本数 | 配准异常 | ev_gap_luma p50 | noise sigma ratio p50 | snr_ratio_lowbin p50 | edge_recall p50 | chroma_delta_l1 p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01Valid` | 112 | 0 | -0.0810 | 6.045 | 0.3142 | 0.6016 | 0.2797 |
| `00Train` 抽样 | 512 | 6 | -0.0728 | 5.936 | 0.3591 | 0.5514 | 0.3215 |

即时解读：

```text
按 >2 px 规则，01Valid 没有 phase-correlation 配准异常样本。
两个 split 上 RAW_dark flat 区 luma 噪声 proxy 约为 RAW_normal 的 6 倍。
RAW_dark low-bin SNR proxy 约为 RAW_normal 的 0.31x-0.36x。
p95-luma EV gap 中位数为负，因此仅凭该统计不支持“纯全局欠曝光”解释。
边缘重叠程度中等：val edge_recall 中位数 0.60，train 抽样中位数 0.55。
通道/色度偏移不可忽略：chromaticity_delta_l1 在 val 上中位数 0.28，在 train 抽样上中位数 0.32。
```

### 2026-06-09 13:49 CST - per-sample eval 和 join smoke 通过

已实现：

```text
tools/lod_raw_per_sample_eval.py
tools/lod_raw_attribution_join.py
```

已运行检查：

```bash
conda run -n dav3 python -m py_compile \
  tools/lod_raw_per_sample_eval.py \
  tools/lod_raw_attribution_join.py
```

Smoke 命令：

```bash
conda run -n dav3 python tools/lod_raw_per_sample_eval.py \
  --run-dir /mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth \
  --label codex_smoke_raw_dark \
  --dataset-family lod_true_raw_dark_rgb16 \
  --dataset-input-mode raw_rgb16_dark \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --lora dav2_lora \
  --lora-block-mode tap \
  --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --split 01Valid \
  --max-samples 1 \
  --output-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_per_sample_raw_dark.csv

conda run -n dav3 python tools/lod_raw_per_sample_eval.py \
  --run-dir /mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth \
  --label codex_smoke_raw_normal \
  --dataset-family lod_true_raw_normal_rgb16 \
  --dataset-input-mode raw_rgb16_normal \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --lora dav2_lora \
  --lora-block-mode tap \
  --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --split 01Valid \
  --max-samples 1 \
  --output-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_per_sample_raw_normal.csv

conda run -n dav3 python tools/lod_raw_attribution_join.py \
  --raw-stats-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/per_sample_raw_stats.csv \
  --dark-eval-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_per_sample_raw_dark.csv \
  --normal-eval-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_per_sample_raw_normal.csv \
  --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_attribution
```

Smoke 结果：

```text
RAW_dark 1-sample eval：strict-load 通过，CUDA inference 通过，mean_d1=0.963984
RAW_normal 1-sample eval：strict-load 通过，CUDA inference 通过，mean_d1=0.936444
join smoke：rows=1，report/scatter 生成通过
临时 smoke 输出已清理
```

### 2026-06-09 13:52 CST - per-sample eval 和 attribution join 完成

命令：

```bash
conda run -n dav3 python tools/lod_raw_per_sample_eval.py \
  --run-dir /mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth \
  --label raw_dark_L0_best \
  --dataset-family lod_true_raw_dark_rgb16 \
  --dataset-input-mode raw_rgb16_dark \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --lora dav2_lora \
  --lora-block-mode tap \
  --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --split 01Valid \
  --output-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_dark_val.csv

conda run -n dav3 python tools/lod_raw_per_sample_eval.py \
  --run-dir /mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly \
  --checkpoint /mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth \
  --label raw_normal_NL_best \
  --dataset-family lod_true_raw_normal_rgb16 \
  --dataset-input-mode raw_rgb16_normal \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --lora dav2_lora \
  --lora-block-mode tap \
  --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --split 01Valid \
  --output-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_normal_val.csv

conda run -n dav3 python tools/lod_raw_attribution_join.py \
  --raw-stats-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/audit_v1/per_sample_raw_stats.csv \
  --dark-eval-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_dark_val.csv \
  --normal-eval-csv /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_normal_val.csv \
  --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution
```

输出：

```text
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_dark_val.csv
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_dark_val.summary.json
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_normal_val.csv
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/per_sample_eval/raw_normal_val.summary.json
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/attribution_join.csv
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/correlation_report.{json,md}
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/attribution/scatter_plots/*.png
```

Eval sanity 检查：

| eval | 行数 | mean D1 | mean abs_rel | edge_overlap_iou |
|---|---:|---:|---:|---:|
| RAW_dark L0 best on `raw_rgb16_dark` | 112 | 0.845308 | 3.665038 | 0.253733 |
| RAW_normal NL best on `raw_rgb16_normal` | 112 | 0.898944 | 2.471021 | 0.288983 |

Join 后的因变量：

```text
行数：112
delta_d1 均值：0.053636
delta_d1 p50: 0.025933
delta_d1 p90: 0.147220
delta_d1 min/max: -0.073806 / 0.445921
delta_d1 为正的样本数：97 / 112
```

与 `delta_d1` 的 Top Spearman 相关性：

| variable | rho | p-value | 方向 |
|---|---:|---:|---|
| `snr_ratio_highbin` | -0.5214 | 3.77e-09 | high-bin SNR 越低，gap 越大 |
| `noise_sigma_dark_luma_flat` | 0.4656 | 2.31e-07 | dark flat 区噪声越强，gap 越大 |
| `edge_recall` | -0.4479 | 7.33e-07 | 边缘保留越弱，gap 越大 |
| `gradient_corr` | -0.4464 | 8.06e-07 | gradient match 越弱，gap 越大 |
| `gradient_ratio` | 0.4226 | 3.45e-06 | dark 梯度/噪声边缘过强，gap 越大 |
| `noise_sigma_ratio_dark_over_normal` | 0.4118 | 6.45e-06 | 相对噪声越高，gap 越大 |
| `linear_residual_sigma` | 0.4101 | 7.11e-06 | RAW_dark->RAW_normal 越难线性拟合，gap 越大 |
| `chromaticity_delta_l1` | 0.3933 | 1.79e-05 | 颜色/统计偏移越大，gap 越大 |

控制 `ev_gap_luma` 后的 Partial Spearman：

| x | partial rho with `delta_d1` | p-value |
|---|---:|---:|
| `snr_ratio_lowbin` | 0.0530 | 0.579 |
| `edge_recall` | -0.4108 | 6.84e-06 |
| `chromaticity_delta_l1` | 0.3781 | 3.95e-05 |
| `noise_sigma_ratio_dark_over_normal` | 0.3641 | 7.92e-05 |

即时解读：

```text
第一轮 per-sample 归因指向噪声、结构、颜色统计因素，而不是全局 p95 曝光。
`ev_gap_luma` 本身与 delta_d1 的相关性较弱（rho=-0.212, p=0.0248），明显弱于 SNR/edge/color 指标。
辅助因变量 `1 - d1_raw_dark` 对 dark 噪声、边缘保留、残差拟合和色度偏移给出相同大方向，但 rho 较小（约 0.25-0.33）。
纯 SNR 字段里最强的是 high-bin SNR；low-bin SNR 单独在该 val set 上没有预测力。
```

### 2026-06-09 13:55 CST - ablation 输入生成器 smoke 通过

已实现：

```text
tools/lod_raw_make_ablation_inputs.py
```

已运行检查：

```bash
conda run -n dav3 python -m py_compile tools/lod_raw_make_ablation_inputs.py

conda run -n dav3 python tools/lod_raw_make_ablation_inputs.py \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
  --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_ablation_inputs \
  --transforms raw_dark_identity,raw_dark_denoise_median3,raw_dark_tone_percentile_trainfit,raw_normal_to_dark_exposure_trainfit \
  --fit-max-samples 4 \
  --fit-pixels-per-sample 512 \
  --max-rows 3 \
  --overwrite
```

Smoke 结果：

```text
manifest 行数：2230
manifest 字段数：13
生成的 smoke PNG shape/dtype：800x1200x3 uint16
raw_dark_identity max_abs_diff_u16 vs source: 0
临时 smoke 输出已清理：/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/codex_smoke_ablation_inputs
```

### 2026-06-09 13:56 CST - 全量 ablation 输入生成已启动

这是数据转换任务，因此放在 tmux 中运行。

启动命令：

```bash
tmux new-session -d -s lod_ablation_inputs_0609_1355 \
  "cd /home/caq/6666_raw/dav2_raw_0603 && conda run --live-stream -n dav3 python tools/lod_raw_make_ablation_inputs.py \
    --lod-root /home/caq/6666_raw/0000_dataset/LOD \
    --manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv \
    --output-dir /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/ablation_inputs \
    --fit-max-samples -1 \
    --fit-pixels-per-sample 4096 \
    --max-rows -1 \
    --sample-seed 42 \
    --noise-seed 42 \
    --png-compression 3 \
    > /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1355_ablation_inputs.log 2>&1"
```

tmux / log：

```text
tmux session：lod_ablation_inputs_0609_1355
log path：/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1355_ablation_inputs.log
attach: tmux attach -t lod_ablation_inputs_0609_1355
monitor: tail -f /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1355_ablation_inputs.log
output dir：/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/ablation_inputs
```

初始日志：

```text
[FIT] rows=1958 pixels_per_sample=4096
```

进度检查：

```text
2026-06-09 13:58 CST：fit 已处理 1100/1958，暂无错误输出。
2026-06-09 14:06 CST：fit 完成 1958/1958。14 个 transform 的 PNG 生成开始，总计 2230 行。`raw_dark_identity` 完成 2230/2230；`raw_dark_denoise_median3` 到 100/2230。输出目录大小：13G。tmux 仍在运行。
2026-06-09 14:07 CST：`raw_dark_denoise_median3` 到 300/2230。输出目录大小：14G。tmux 仍在运行。
2026-06-09 14:08 CST：`raw_dark_denoise_median3` 到 600/2230。输出目录大小：15G。`/mnt/drive` 剩余：828G。tmux 仍在运行。
2026-06-09 14:09 CST：`raw_dark_denoise_median3` 到 900/2230。输出目录大小：16G。tmux 仍在运行。
2026-06-09 14:10 CST：`raw_dark_denoise_median3` 到 1100/2230。输出目录大小：17G。tmux 仍在运行。
2026-06-09 14:11 CST：`raw_dark_denoise_median3` 到 1400/2230。输出目录大小：18G。tmux 仍在运行。
2026-06-09 14:11 CST：`raw_dark_denoise_median3` 到 1600/2230。输出目录大小：19G。tmux 仍在运行。
2026-06-09 14:12 CST：`raw_dark_denoise_median3` 到 1900/2230。输出目录大小：20G。tmux 仍在运行。
2026-06-09 14:13 CST：`raw_dark_denoise_median3` 到 2100/2230。输出目录大小：22G。tmux 仍在运行。
2026-06-09 14:14 CST：`raw_dark_denoise_median3` 完成 2230/2230。`raw_dark_bilateral_weak_trainfit_sigma` 到 200/2230。输出目录大小：23G。tmux 仍在运行。
2026-06-09 14:15 CST：`raw_dark_bilateral_weak_trainfit_sigma` 到 500/2230。输出目录大小：24G。tmux 仍在运行。
2026-06-09 14:15 CST：`raw_dark_bilateral_weak_trainfit_sigma` 到 800/2230。输出目录大小：26G。tmux 仍在运行。
2026-06-09 14:16 CST：`raw_dark_bilateral_weak_trainfit_sigma` 到 1100/2230。输出目录大小：28G。tmux 仍在运行。
2026-06-09 14:17 CST：`raw_dark_bilateral_weak_trainfit_sigma` 到 1400/2230。输出目录大小：29G。tmux 仍在运行。
2026-06-09 14:18 CST：`raw_dark_bilateral_weak_trainfit_sigma` 到 1700/2230。输出目录大小：31G。tmux 仍在运行。
2026-06-09 14:19 CST：`raw_dark_bilateral_weak_trainfit_sigma` 到 2000/2230。输出目录大小：33G。tmux 仍在运行。
2026-06-09 14:20 CST：`raw_dark_bilateral_weak_trainfit_sigma` 完成 2230/2230。`raw_dark_bilateral_medium_trainfit_sigma` 到 100/2230。输出目录大小：34G。tmux 仍在运行。
2026-06-09 14:20 CST：`raw_dark_bilateral_medium_trainfit_sigma` 到 400/2230。输出目录大小：35G。tmux 仍在运行。
2026-06-09 14:21 CST：`raw_dark_bilateral_medium_trainfit_sigma` 到 700/2230。输出目录大小：37G。tmux 仍在运行。
2026-06-09 14:22 CST：`raw_dark_bilateral_medium_trainfit_sigma` 到 1000/2230。输出目录大小：39G。tmux 仍在运行。
2026-06-09 14:23 CST：`raw_dark_bilateral_medium_trainfit_sigma` 到 1300/2230。输出目录大小：40G。tmux 仍在运行。
2026-06-09 14:24 CST：`raw_dark_bilateral_medium_trainfit_sigma` 到 1600/2230。输出目录大小：42G。tmux 仍在运行。
2026-06-09 14:25 CST：`raw_dark_bilateral_medium_trainfit_sigma` 到 2000/2230。输出目录大小：44G。tmux 仍在运行。一次长 `tail -n 500` 轮询通过工具没有返回内容，但后续短 tail 确认 tmux 任务健康。
2026-06-09 14:26 CST：`raw_dark_bilateral_medium_trainfit_sigma` 完成 2230/2230。`raw_dark_bilateral_strong_trainfit_sigma` 到 100/2230。输出目录大小：45G。tmux 仍在运行。
2026-06-09 14:26 CST：`raw_dark_bilateral_strong_trainfit_sigma` 到 400/2230。输出目录大小：47G。tmux 仍在运行。
2026-06-09 14:27 CST：`raw_dark_bilateral_strong_trainfit_sigma` 到 700/2230。输出目录大小：48G。`/mnt/drive` 剩余：795G。tmux 仍在运行。
2026-06-09 14:28 CST：`raw_dark_bilateral_strong_trainfit_sigma` 到 1000/2230。输出目录大小：50G。tmux 仍在运行。
2026-06-09 14:29 CST：`raw_dark_bilateral_strong_trainfit_sigma` 到 1400/2230。输出目录大小：52G。`/mnt/drive` 剩余：791G。tmux 仍在运行。
2026-06-09 14:31 CST：`raw_dark_bilateral_strong_trainfit_sigma` 到 1900/2230。输出目录大小：54G。`/mnt/drive` 剩余：789G。tmux 仍在运行。
2026-06-09 14:32 CST：`raw_dark_bilateral_strong_trainfit_sigma` 完成 2230/2230。输出目录大小：56G。`/mnt/drive` 剩余：787G。tmux 仍在运行。
2026-06-09 14:33 CST：`raw_dark_tone_percentile_trainfit` 启动并到 400/2230。输出目录大小：58G。`/mnt/drive` 剩余：785G。tmux 仍在运行。
2026-06-09 14:34 CST：`raw_dark_tone_percentile_trainfit` 到 800/2230。输出目录大小：60G。`/mnt/drive` 剩余：783G。tmux 仍在运行。
2026-06-09 14:35 CST：`raw_dark_tone_percentile_trainfit` 到 1100/2230。输出目录大小：62G。`/mnt/drive` 剩余：781G。tmux 仍在运行。
2026-06-09 14:36 CST：`raw_dark_tone_percentile_trainfit` 到 1500/2230。输出目录大小：64G。`/mnt/drive` 剩余：779G。tmux 仍在运行。
2026-06-09 14:37 CST：`raw_dark_tone_percentile_trainfit` 到 1900/2230。输出目录大小：66G。`/mnt/drive` 剩余：777G。tmux 仍在运行。
2026-06-09 14:38 CST：`raw_dark_tone_percentile_trainfit` 完成 2230/2230。输出目录大小：67G。`/mnt/drive` 剩余：776G。tmux 仍在运行；下一个 transform 尚未输出进度。
2026-06-09 14:39 CST：`raw_dark_tone_percentile_luma_trainfit` 启动并到 400/2230。输出目录大小：69G。`/mnt/drive` 剩余：774G。tmux 仍在运行。
2026-06-09 14:40 CST：`raw_dark_tone_percentile_luma_trainfit` 到 800/2230。输出目录大小：71G。`/mnt/drive` 剩余：772G。tmux 仍在运行。
2026-06-09 14:41 CST：`raw_dark_tone_percentile_luma_trainfit` 到 1200/2230。输出目录大小：73G。`/mnt/drive` 剩余：770G。tmux 仍在运行。
2026-06-09 14:42 CST：`raw_dark_tone_percentile_luma_trainfit` 到 1600/2230。输出目录大小：75G。`/mnt/drive` 剩余：768G。tmux 仍在运行。
2026-06-09 14:43 CST：`raw_dark_tone_percentile_luma_trainfit` 到 2000/2230。输出目录大小：78G。`/mnt/drive` 剩余：765G。tmux 仍在运行。
2026-06-09 14:44 CST：`raw_dark_tone_percentile_luma_trainfit` 完成 2230/2230。`raw_dark_tone_gamma_trainfit` 启动并到 100/2230。输出目录大小：79G。`/mnt/drive` 剩余：764G。tmux 仍在运行。
2026-06-09 14:45 CST：`raw_dark_tone_gamma_trainfit` 到 500/2230。输出目录大小：81G。`/mnt/drive` 剩余：762G。tmux 仍在运行。
2026-06-09 14:46 CST：`raw_dark_tone_gamma_trainfit` 到 900/2230。输出目录大小：83G。`/mnt/drive` 剩余：760G。tmux 仍在运行。
2026-06-09 14:47 CST：`raw_dark_tone_gamma_trainfit` 到 1200/2230。输出目录大小：85G。`/mnt/drive` 剩余：758G。tmux 仍在运行。
2026-06-09 14:48 CST：`raw_dark_tone_gamma_trainfit` 到 1500/2230。输出目录大小：86G。`/mnt/drive` 剩余：756G。tmux 仍在运行。
2026-06-09 14:49 CST：`raw_dark_tone_gamma_trainfit` 到 1900/2230。输出目录大小：88G。`/mnt/drive` 剩余：755G。tmux 仍在运行。
2026-06-09 14:51 CST：`raw_dark_tone_gamma_trainfit` 完成 2230/2230。`raw_dark_denoise_then_tone_trainfit` 启动并到 100/2230。输出目录大小：90G。`/mnt/drive` 剩余：753G。tmux 仍在运行。
2026-06-09 14:52 CST：`raw_dark_denoise_then_tone_trainfit` 到 400/2230。输出目录大小：92G。`/mnt/drive` 剩余：751G。tmux 仍在运行。
2026-06-09 14:53 CST：`raw_dark_denoise_then_tone_trainfit` 到 800/2230。输出目录大小：94G。`/mnt/drive` 剩余：749G。tmux 仍在运行。
2026-06-09 14:54 CST：`raw_dark_denoise_then_tone_trainfit` 到 1100/2230。输出目录大小：95G。`/mnt/drive` 剩余：748G。tmux 仍在运行。
2026-06-09 14:55 CST：`raw_dark_denoise_then_tone_trainfit` 到 1500/2230。输出目录大小：97G。`/mnt/drive` 剩余：746G。tmux 仍在运行。
2026-06-09 14:56 CST：`raw_dark_denoise_then_tone_trainfit` 到 1900/2230。输出目录大小：100G。`/mnt/drive` 剩余：743G。tmux 仍在运行。
2026-06-09 14:57 CST：`raw_dark_denoise_then_tone_trainfit` 完成 2230/2230。`raw_dark_tone_then_denoise_trainfit` 启动并到 100/2230。输出目录大小：101G。`/mnt/drive` 剩余：741G。tmux 仍在运行。
2026-06-09 14:58 CST：`raw_dark_tone_then_denoise_trainfit` 到 400/2230。输出目录大小：103G。`/mnt/drive` 剩余：740G。tmux 仍在运行。
2026-06-09 14:59 CST：`raw_dark_tone_then_denoise_trainfit` 到 800/2230。输出目录大小：105G。`/mnt/drive` 剩余：738G。tmux 仍在运行。
2026-06-09 15:01 CST：`raw_dark_tone_then_denoise_trainfit` 到 1300/2230。输出目录大小：107G。`/mnt/drive` 剩余：735G，使用率 94%。tmux 仍在运行。
2026-06-09 15:07 CST：`raw_dark_tone_then_denoise_trainfit` 完成 2230/2230。`raw_normal_identity` 启动并到 1300/2230。输出目录大小：118G。`/mnt/drive` 剩余：725G，使用率 94%。tmux 仍在运行。
2026-06-09 15:08 CST：`raw_normal_identity` 到 1600/2230。输出目录大小：120G。`/mnt/drive` 剩余：723G，使用率 94%。tmux 仍在运行。
2026-06-09 15:09 CST：`raw_normal_identity` 到 2000/2230。输出目录大小：121G。`/mnt/drive` 剩余：722G，使用率 94%。tmux 仍在运行。
2026-06-09 15:10 CST：`raw_normal_identity` 完成 2230/2230。`raw_normal_to_dark_exposure_trainfit` 启动并到 100/2230。输出目录大小：123G。`/mnt/drive` 剩余：720G，使用率 94%。tmux 仍在运行。
2026-06-09 15:12 CST：`raw_normal_to_dark_exposure_trainfit` 到 400/2230。输出目录大小：125G。`/mnt/drive` 剩余：718G，使用率 94%。tmux 仍在运行。
2026-06-09 15:13 CST：`raw_normal_to_dark_exposure_trainfit` 到 700/2230。输出目录大小：126G。`/mnt/drive` 剩余：717G，使用率 94%。tmux 仍在运行。
2026-06-09 15:14 CST：`raw_normal_to_dark_exposure_trainfit` 到 1100/2230。输出目录大小：128G。`/mnt/drive` 剩余：715G，使用率 94%。tmux 仍在运行。
2026-06-09 15:15 CST：`raw_normal_to_dark_exposure_trainfit` 到 1500/2230。输出目录大小：130G。`/mnt/drive` 剩余：713G，使用率 94%。tmux 仍在运行。
2026-06-09 15:16 CST：`raw_normal_to_dark_exposure_trainfit` 到 2000/2230。输出目录大小：132G。`/mnt/drive` 剩余：711G，使用率 94%。tmux 仍在运行。
2026-06-09 15:17 CST：`raw_normal_to_dark_exposure_trainfit` 完成 2230/2230。输出目录大小：133G。`/mnt/drive` 剩余：710G，使用率 94%。tmux 仍在运行；下一个 transform 尚未输出进度。
2026-06-09 15:18 CST：`raw_normal_to_dark_noise_only_trainfit` 启动并到 300/2230。输出目录大小：134G。`/mnt/drive` 剩余：708G，使用率 94%。tmux 仍在运行。
2026-06-09 15:20 CST：`raw_normal_to_dark_noise_only_trainfit` 到 600/2230。输出目录大小：136G。`/mnt/drive` 剩余：707G，使用率 94%。tmux 仍在运行。
2026-06-09 15:21 CST：`raw_normal_to_dark_noise_only_trainfit` 到 900/2230。输出目录大小：137G。`/mnt/drive` 剩余：705G，使用率 94%。tmux 仍在运行。
2026-06-09 15:22 CST：`raw_normal_to_dark_noise_only_trainfit` 到 1300/2230。输出目录大小：139G。`/mnt/drive` 剩余：704G，使用率 94%。tmux 仍在运行。
2026-06-09 15:23 CST：`raw_normal_to_dark_noise_only_trainfit` 到 1600/2230。输出目录大小：141G。`/mnt/drive` 剩余：702G，使用率 94%。tmux 仍在运行。
2026-06-09 15:24 CST：`raw_normal_to_dark_noise_only_trainfit` 到 1900/2230。输出目录大小：142G。`/mnt/drive` 剩余：701G，使用率 94%。tmux 仍在运行。
2026-06-09 15:25 CST：`raw_normal_to_dark_noise_only_trainfit` 完成 2230/2230。输出目录大小：144G。`/mnt/drive` 剩余：699G，使用率 94%。tmux 仍在运行；最后一个 transform 尚未输出进度。
2026-06-09 15:26 CST：`raw_normal_to_dark_exposure_noise_trainfit` 启动并到 300/2230。输出目录大小：145G。`/mnt/drive` 剩余：698G，使用率 94%。tmux 仍在运行。
2026-06-09 15:27 CST：`raw_normal_to_dark_exposure_noise_trainfit` 到 600/2230。输出目录大小：147G。`/mnt/drive` 剩余：696G，使用率 94%。tmux 仍在运行。
2026-06-09 15:28 CST：`raw_normal_to_dark_exposure_noise_trainfit` 到 1000/2230。输出目录大小：148G。`/mnt/drive` 剩余：694G，使用率 94%。tmux 仍在运行。
2026-06-09 15:30 CST：`raw_normal_to_dark_exposure_noise_trainfit` 到 1300/2230。输出目录大小：150G。`/mnt/drive` 剩余：693G，使用率 94%。tmux 仍在运行。
2026-06-09 15:31 CST：`raw_normal_to_dark_exposure_noise_trainfit` 到 1600/2230。输出目录大小：152G。`/mnt/drive` 剩余：691G，使用率 94%。tmux 仍在运行。
2026-06-09 15:32 CST：`raw_normal_to_dark_exposure_noise_trainfit` 到 1900/2230。输出目录大小：153G。`/mnt/drive` 剩余：690G，使用率 94%。tmux 仍在运行。
2026-06-09 15:33 CST: `raw_normal_to_dark_exposure_noise_trainfit` 完成 2230/2230。日志报告 `[GEN] completed transforms=14`。tmux session `lod_ablation_inputs_0609_1355` 已退出。最终输出目录大小：154G。`/mnt/drive` 剩余：688G，使用率 94%。
2026-06-09 15:34 CST: integrity check 通过。发现 14 个 manifest CSV 文件和 14 个 `png16` transform 目录。每个 manifest 都有 2230 行数据；每个 PNG 目录都有 2230 个 `.png` 文件。`run_metadata.json` 报告 `processed_rows=2230` 且 transform 数为 14。抽查 `raw_dark_identity` 图像 shape 为 `(800, 1200, 3)`，dtype 为 `uint16`，相对原始 `/home/caq/6666_raw/0000_dataset/LOD/RAW_Dark/2.png` 的 `max_abs_diff_u16=0`。
```

### 2026-06-09 15:40 CST - `raw_dark_identity` zero-training eval sanity 通过

命令：

```bash
conda run --live-stream -n dav3 torchrun --nproc_per_node=1 --master_port 49877 finetune_stf/train.py \
  --stage eval_only \
  --encoder vits \
  --dataset-family lod_true_raw_dark_rgb16 \
  --dataset-input-mode raw_rgb16_dark \
  --input-domain raw3 \
  --front-end raw_rgb16_ram3 \
  --model-input-tensor raw \
  --bridge none \
  --decoder-feature-adapter none \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --resume-from /mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth \
  --lod-root /home/caq/6666_raw/0000_dataset/LOD \
  --lod-manifest /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/ablation_inputs/manifests/raw_dark_identity.csv \
  --lod-label-space inverse_relative \
  --lod-train-crop-mode random \
  --lod-val-crop-mode center \
  --raw-storage-format raw_rgb16_png_3ch \
  --lod-raw-norm-mode uint16_div_65535 \
  --raw-ram-rgb-tail identity \
  --lora dav2_lora --lora-block-mode tap --lora-tap-layers 2 5 8 11 \
  --lora-rank 8 --lora-alpha 16 --lora-lr 5e-5 --raw-front-end-lr 5e-5 \
  --dav2-train-mode decoder --backbone-layer-decay 1.0 \
  --lr 1e-5 --lr-schedule poly --warmup-steps 0 --loss-type ssi \
  --loss-target-normalization --loss-norm-min-scale 1e-3 \
  --bs 8 --accum-steps 1 \
  --input-height 512 --input-width 960 \
  --aug-preset off --aug-hflip-prob 0.5 \
  --eval-lod --no-eval-stf --eval-lod-train-proxy --lod-train-proxy-count 112 \
  --amp --amp-dtype bf16 \
  --num-workers 4 --log-interval 500 \
  --no-enable-fixed-viz-dump --no-enable-train-source-viz-dump \
  --save-path /home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/lod_raw_diag/eval_zero/raw_dark_identity \
  --heavy-save-root /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero
```

日志：

```text
/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1534_eval_zero_raw_dark_identity.log
```

结果：

```text
val 样本数：112
val d1: 0.8450669757120602
val abs_rel: 3.67772195375844
train_proxy 样本数：112
train_proxy d1: 0.8667
gap train_proxy-val: 0.0216
checkpoint metadata 中的 baseline L0 best lod_d1：0.8450678996860131
相对 baseline 的绝对 D1 差异：约 0.000000924
sanity 状态：PASS (< 0.002)
```

注：该 `eval_only` 路径记录了 metrics，但没有生成 `pretrain_eval.json`；上面的结果来自命令日志。

### 2026-06-09 15:49 CST - 剩余 zero-training ablation eval 完成

批量命令：

```bash
conda run --live-stream -n dav3 python tools/lod_raw_eval_zero_batch.py \
  --skip-transforms raw_dark_identity \
  --run-prefix 0609_1540
```

输出：

```text
summary csv : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary.csv
summary json: /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary.json
logs        : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1540_eval_zero_<transform>.log
status      : 剩余 13/13 个 transform 完成，全部 returncode=0
```

Dark 侧 zero-eval 排名，相对 `raw_dark_identity` D1 `0.8450669757`：

```text
transform                                 val_d1       delta_vs_identity
raw_dark_bilateral_weak_trainfit_sigma    0.845271278  +0.000204302
raw_dark_tone_percentile_luma_trainfit    0.845238107  +0.000171131
raw_dark_bilateral_medium_trainfit_sigma  0.845138166  +0.000071190
raw_dark_tone_then_denoise_trainfit       0.844387005  -0.000679971
raw_dark_bilateral_strong_trainfit_sigma  0.843924815  -0.001142161
raw_dark_denoise_then_tone_trainfit       0.843888197  -0.001178779
raw_dark_tone_percentile_trainfit         0.843357853  -0.001709123
raw_dark_tone_gamma_trainfit              0.839716713  -0.005350263
raw_dark_denoise_median3                  0.826091504  -0.018975471
```

Normal/synthetic zero-eval 排名，相对 `raw_normal_identity` D1 `0.8990109310`：

```text
transform                                      val_d1       delta_vs_normal_identity
raw_normal_identity                            0.899010931  +0.000000000
raw_normal_to_dark_exposure_trainfit           0.897730425  -0.001280506
raw_normal_to_dark_noise_only_trainfit         0.849621079  -0.049389852
raw_normal_to_dark_exposure_noise_trainfit     0.848701075  -0.050309856
```

即时解读：

```text
Dark 侧 zero-eval 没有显示 tone 或简单 denoise 带来明显恢复。weak bilateral 和 luma-only percentile tone 基本与 identity 持平（+0.0002 / +0.00017 D1），median denoise 和 gamma tone 反而下降。

Normal-to-dark 反事实显示 exposure-only 几乎不伤害 normal 模型（-0.0013 D1），而加入 dark noise 会让 D1 下降约 0.049，接近 RAW_dark 区间。这支持前面的归因：主导退化因素是噪声/低 SNR 结构，而不是单纯全局曝光。
```

---

## 17. 阶段性归因小结与下一步决策 — 2026-06-09 16:05 CST

> 本节把 §16 跑到的 zero-train 结果、归因结论、以及据此锁定的短训决策固化下来，方便后续跟进。**重要口径：§17.1 全部是 zero-train（test-time domain shift），按 §13.2 只作快筛，不作最终结论；最终归因以下面正在跑的短训 `drop_ratio_st` 为准。**

### 17.1 Zero-train ablation 结果汇总（仅快筛）

zero-train gap 分母 `G_zero = D1(raw_normal_identity) − D1(raw_dark_identity) = 0.89901 − 0.84507 = 0.05394`。

**Dark 侧（相对 `raw_dark_identity` 0.84507）—— 无任何有效恢复项：**

| transform | val_d1 | Δ vs identity | 占 G_zero |
|---|---:|---:|---:|
| raw_dark_bilateral_weak_trainfit_sigma | 0.84527 | +0.00020 | +0.4% |
| raw_dark_tone_percentile_luma_trainfit | 0.84524 | +0.00017 | +0.3% |
| raw_dark_bilateral_medium_trainfit_sigma | 0.84514 | +0.00007 | +0.1% |
| raw_dark_tone_then_denoise_trainfit | 0.84439 | −0.00068 | — |
| raw_dark_bilateral_strong_trainfit_sigma | 0.84392 | −0.00114 | — |
| raw_dark_denoise_then_tone_trainfit | 0.84389 | −0.00118 | — |
| raw_dark_tone_percentile_trainfit | 0.84336 | −0.00171 | — |
| raw_dark_tone_gamma_trainfit | 0.83972 | −0.00535 | — |
| raw_dark_denoise_median3 | 0.82609 | −0.01898 | — |

**Normal 反事实（相对 `raw_normal_identity` 0.89901）—— 噪声主导、曝光几乎无害：**

| degrade | val_d1 | drop | 占 G_zero |
|---|---:|---:|---:|
| raw_normal_to_dark_exposure_trainfit | 0.89773 | −0.00128 | **2.4%** |
| raw_normal_to_dark_noise_only_trainfit | 0.84962 | −0.04939 | **91.6%** |
| raw_normal_to_dark_exposure_noise_trainfit | 0.84870 | −0.05031 | 93.3% |

`noise_only` 把强 normal 模型一路打到 0.84962，离真实 RAW_dark 0.84507 仅差 0.0045；`exposure_only` 几乎不掉；加曝光只比纯噪声多解释 ~1.7%。

### 17.2 归因结论（当前证据）

- **曝光/动态范围：排除（非主因）。** audit `ev_gap_luma` 中位 −0.08（dark 在 RAW16 域反而略亮）、与 delta_d1 rho=−0.21 且 Pearson≈−0.03；`p99_dark`/`dynamic_range` rho≈−0.04；exposure-only 反事实只掉 2.4%。
- **噪声/低 SNR：主导。** noise-only 反事实掉 91.6%（落到 0.84962，离真实 dark 仅 0.0045）；audit `snr_ratio_highbin` rho=−0.52（最强）、`noise_sigma_dark` rho=+0.47、`noise_sigma_ratio` rho=+0.41。dark flat 区噪声 ≈ normal 的 6×，且通道不均（噪声模型 a_G=0.02 ≪ a_R/a_B≈0.05）。
- **残差 ~7%**（noise-only 与真实 dark 的 0.0045 差 + 曝光增量 ~1.7%）归于 edge/结构（`edge_recall` rho=−0.45、`gradient_corr` rho=−0.45）、通道色彩（`chromaticity_delta_l1` rho=+0.39）、以及模型/feature mismatch。
- **对应计划 §12 的模式 B（噪声/低 SNR 主导）**，曝光不足被反事实明确否掉。

### 17.3 方法论提醒（写 report 时不要踩）

1. **zero-train ≠ 结论。** §17.1 有 domain shift。noise-only 崩，可能只是"normal 模型从没见过噪声"，不等于"噪声本质破坏了深度信息"。**必须用短训 E8（在噪声输入上重训仍回不到 ceiling）排除该混淆**——这正是 §17.4 短训的目的。
2. **§9.1 闸① 判据需修正。** 原文用 `snr_ratio_lowbin`，但实测 lowbin rho=+0.03（控曝光后偏相关 0.05），信号在 **highbin**（rho=−0.52）。**建议把 §9.1 条件① 改为 `snr_ratio_highbin` 或 `noise_sigma_ratio`。** 为避免静默改实验语义，此处只记录修正建议，不直接改 §9.1 正文；噪声结论由 highbin SNR + noise_sigma + noise-only 反事实三条独立证据支撑，不依赖 lowbin 那一条。
3. **delta_d1 零膨胀**（97/112 为正、p50=0.026、长尾到 0.45），单变量相关性部分由难样本尾部驱动；偏相关（控 `ev_gap_luma`）已做，edge_recall(−0.41)/chromaticity(0.38)/noise_ratio(0.36) 存活。

### 17.4 下一步决策（已锁定范围）

**做：** noise-vs-exposure dissociation 短训集（cold-start，from pretrained，§6.3 配方，≥2 seed 42/123）：

| run | 角色 | 期望 |
|---|---|---|
| A1 `raw_dark_identity` | floor_st | 短训下界 |
| A2 `raw_normal_identity` | ceiling_st | 短训上界 |
| E7 `raw_normal_to_dark_exposure` | 对照 | ≈ ceiling（曝光无害）|
| E8 `raw_normal_to_dark_noise_only` | **关键** | ≈ floor（噪声因果）|

**跳过：** 7 个 dark restoration 重训（zero-train 全平到变差；"图像域是否够"由 oracle 上界比重训 deployable 更能回答）；E9 `exposure+noise`（对因果判断与 E8 冗余）。

**判据：** `drop_ratio_st(E8) = (ceiling_st − D1_st(E8)) / (ceiling_st − floor_st)`。
- E8 ≈ 1.0（落到 floor）→ 噪声在同等训练预算下仍打到 dark 水平 → 噪声因果坐实 → 点亮 §9.1。
- E7 ≈ 0（停在 ceiling）→ §9.2 闸③（exposure drop_ratio_st>0.40）不亮，二次确认曝光非主因。

### 17.5 短训设计与 budget 注意

- **入口已验证：** 短训 path smoke 通过（dark_identity 1ep×5step，rc=0，按 lod_d1 存 best），成功后清理 codex_smoke 产物。
- **launcher：** `tools/lod_raw_short_train_batch.sh`（`SET=anchors|degrade`，`EPOCHS`/`MAXSTEPS`/`SEEDS` 可覆盖；默认 5/80/"42 123"）。
- **budget caveat：** cold-start init d1≈0.37（从 `depth_anything_v2_vits.pth` 起训，非 resume L0）。默认 5ep×80step=400 步达不到全训 best，这是设计内（ratio 用短训 floor/ceiling 归一化，不用全训 0.845/0.899）。**先只跑 anchors 确认 `ceiling_st−floor_st` ≥ ~0.03 且 seed spread 小，再放 E7/E8；若 separation 被 seed 噪声淹没，则升到 5 full epoch（不限 step）重跑——此为实验语义旋钮，会显式记录，不静默改。**

### 17.6 执行记录

- **2026-06-09 16:00 CST：** 短训 path smoke 通过并清理。
- **2026-06-09 16:05 CST：** anchors 短训队列启动（tmux，单卡 RTX 4090 顺序跑）：
  ```text
  tmux session : lod_short_train_anchors_0609_1605
  queue log    : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1605_short_train_anchors_queue.log
  per-run logs : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_short_train_<run>.log
  summary csv  : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train/short_train_summary_anchors_e5_s80.csv
  attach       : tmux attach -t lod_short_train_anchors_0609_1605
  tail         : tail -f /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1605_short_train_anchors_queue.log
  ```
- **2026-06-09 16:17 CST：** anchors（锚点）短训队列完成，4/4 returncode=0，tmux session `lod_short_train_anchors_0609_1605` 已退出。汇总文件：
  ```text
  /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train/short_train_summary_anchors_e5_s80.csv
  ```

  | 锚点 | seed42 best_lod_d1 | seed123 best_lod_d1 | 均值 | seed 差值 |
  |---|---:|---:|---:|---:|
  | `raw_dark_identity` floor_st | 0.7452 | 0.7352 | 0.7402 | 0.0100 |
  | `raw_normal_identity` ceiling_st | 0.8548 | 0.8379 | 0.8464 | 0.0169 |

  短训上下界分离度足够，可以继续使用当前计划预算：
  ```text
  ceiling_st - floor_st:
    seed42 = 0.1096
    seed123 = 0.1027
    均值 = 0.1062
  ```
  该值明显高于 §17.5 设定的约 0.03 最小门槛，因此不需要静默修改 `EPOCHS/MAXSTEPS`。
- **2026-06-09 16:18 CST：** 退化短训队列启动（E7 仅曝光退化 + E8 仅噪声退化，5ep×80step，seed 42/123，tmux，单卡 RTX 4090 顺序跑）：
  ```text
  tmux session : lod_short_train_degrade_0609_1618
  queue log    : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1618_short_train_degrade_queue.log
  per-run logs : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_short_train_raw_normal_to_dark_<exposure|noise_only>_e5_s80_seed<42|123>.log
  summary csv  : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train/short_train_summary_degrade_e5_s80.csv
  attach       : tmux attach -t lod_short_train_degrade_0609_1618
  tail         : tail -f /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1618_short_train_degrade_queue.log
  ```
  初始检查：队列已启动 `raw_normal_to_dark_exposure_e5_s80_seed42`；tmux session 存活，GPU 有占用。
- **2026-06-09 16:22 CST：** E7 仅曝光退化 seed42 完成，returncode=0，best_lod_d1=**0.8504**。
  - 同 seed 的 ceiling_st（`raw_normal_identity`, seed42）：0.8548。
  - 同 seed 的 floor_st（`raw_dark_identity`, seed42）：0.7452。
  - 临时同 seed `drop_ratio_st(E7, seed42) = (0.8548 - 0.8504) / (0.8548 - 0.7452) = 0.040`（约 4% drop），符合“曝光不是主因”的假设。E7 最终判断等待 seed123。
  - 队列已推进到 `raw_normal_to_dark_exposure_e5_s80_seed123`。
- **2026-06-09 16:24 CST：** E7 仅曝光退化两个 seed 均已完成。

  | 实验 | seed42 best_lod_d1 | seed123 best_lod_d1 | 均值 |
  |---|---:|---:|---:|
  | E7 `raw_normal_to_dark_exposure` | 0.8504 | 0.8283 | 0.8394 |

  drop_ratio 计算：
  ```text
  seed42: (0.8548 - 0.8504) / (0.8548 - 0.7452) = 0.040
  seed123: (0.8379 - 0.8283) / (0.8379 - 0.7352) = 0.094
  按均值: (0.8464 - 0.8394) / (0.8464 - 0.7402) = 0.066
  ```
  解读：仅曝光退化只造成很小的短训下降（约 4-9%，均值约 6.6%），远低于 §9.2 中 `drop_ratio_st > 0.40` 的曝光主因门槛。因此，短训确认后曝光/动态范围仍不是主导因素。队列已推进到 E8 `raw_normal_to_dark_noise_only_e5_s80_seed42`。
- **2026-06-09 16:28 CST：** E8 仅噪声退化 seed42 完成，returncode=0，best_lod_d1=**0.7913**。
  - 同 seed 的 ceiling_st（`raw_normal_identity`, seed42）：0.8548。
  - 同 seed 的 floor_st（`raw_dark_identity`, seed42）：0.7452。
  - 临时同 seed `drop_ratio_st(E8, seed42) = (0.8548 - 0.7913) / (0.8548 - 0.7452) = 0.579`。
  - 解读：仅噪声退化超过 §6.4 的 synthetic-degrade 闸值（`drop_ratio_st > 0.50`），但 seed42 没有完全跌到 floor。E8 最终判断等待 seed123 和按均值计算的 ratio。
  - 队列已推进到 `raw_normal_to_dark_noise_only_e5_s80_seed123`。
- **2026-06-09 16:31 CST：** 退化短训队列完成，4/4 returncode=0，tmux session `lod_short_train_degrade_0609_1618` 已退出。汇总文件：
  ```text
  /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train/short_train_summary_degrade_e5_s80.csv
  ```

  最终短训 ratio 如下，归一化使用 16:17 得到的同 seed anchors：

  | 实验 | seed42 best_lod_d1 | seed123 best_lod_d1 | D1 均值 | drop_ratio seed42 | drop_ratio seed123 | 按均值 drop_ratio |
  |---|---:|---:|---:|---:|---:|---:|
  | E7 `raw_normal_to_dark_exposure` | 0.8504 | 0.8283 | 0.8394 | 0.040 | 0.094 | 0.066 |
  | E8 `raw_normal_to_dark_noise_only` | 0.7913 | 0.7596 | 0.7755 | 0.579 | 0.762 | 0.668 |

  判定：
  ```text
  E7 仅曝光退化：
    mean drop_ratio_st = 0.066  -> 低于 0.10，也远低于 §9.2 的 0.40 曝光主因门槛。
    短训确认后，曝光/动态范围仍不是主导因素。

  E8 仅噪声退化：
    mean drop_ratio_st = 0.668  -> 高于 §6.4 / §9.1 风格的 >0.50 synthetic-degrade 闸值。
    噪声/低 SNR 被确认为当前已测试单因素中的主导因素。
    但 E8 没有达到 drop_ratio_st ~= 1.0，说明当前拟合的 synthetic noise model 不能完全复现真实 RAW_dark。
    剩余 gap（约短训 gap 的 33%）在 oracle/probe 证据出来前，应暂归于结构/边缘损失、颜色/统计偏移、真实噪声模型不匹配或 feature-domain mismatch。
  ```

  更新后的阶段结论：zero-train 的方向性结论通过了 short-train 确认，但措辞应改成 **噪声/低 SNR 是主导因素，但不是全部解释**。仅曝光退化已同时被 zero-train 和 short-train 排除为主因。
- **2026-06-09 16:47 CST：** oracle E10/E11 输入生成启动（tmux，单独输出目录，避免覆盖已有 14 个 deployable/synthetic transform 的 metadata）：
  ```text
  tmux session : lod_oracle_inputs_0609_1647
  log path     : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1647_oracle_inputs.log
  output dir   : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/ablation_inputs_oracle
  transforms   : raw_dark_tone_percentile_oracle, raw_dark_tone_denoise_oracle
  attach       : tmux attach -t lod_oracle_inputs_0609_1647
  tail         : tail -f /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1647_oracle_inputs.log
  ```
  口径说明：这两个 oracle transform 使用同一个 pair 的 `RAW_normal`，因此 trainfit 参数不参与语义；命令仍传了小型 fit 设置（`fit-max-samples=4`, `fit-pixels-per-sample=512`）只是满足当前生成器固定流程，不影响 E10/E11 输出。

  E10/E11 含义补充（避免后续误读）：
  ```text
  E10 raw_dark_tone_percentile_oracle:
    对每个 pair 单独读取 RAW_dark 和对应 RAW_normal。
    用该 pair 自己的 RAW_normal p1/p99 统计，把 RAW_dark 的每通道动态范围映射到 RAW_normal。
    它回答的是：如果样本级曝光/颜色统计可被 oracle 精准校正，D1 最多能恢复多少。

  E11 raw_dark_tone_denoise_oracle:
    先执行 E10 的 per-sample oracle tone mapping。
    再用对应 RAW_normal 的低梯度 flat 区估计 tone 后残差噪声强度，并做一次保边 bilateral denoise。
    它回答的是：如果 tone + paired-normal 辅助去噪都给到，图像域 restoration 上界是否明显高于 E10。

  重要限制：
    E10/E11 都读取同一 val/test pair 的 RAW_normal，因此是 oracle diagnostic，不是 deployable 方法。
    它们只能用于判断“图像域是否还有可恢复上界”，不能作为最终公平实验结果和 RAW_dark/RGB_dark baseline 对比。
    若 E10/E11 有效而 trainfit/deployable transform 无效，说明简单全局参数不够，下一步更偏 learned restoration。
    若 E10/E11 也无效，说明问题更可能来自真实结构损失或 RAM/DAv2 feature-domain mismatch。
  ```
- **2026-06-09 17:07 CST：** oracle E10/E11 输入生成完成，tmux session `lod_oracle_inputs_0609_1647` 已退出。
  ```text
  manifest:
    raw_dark_tone_percentile_oracle.csv 2230 data rows
    raw_dark_tone_denoise_oracle.csv    2230 data rows
  png16:
    raw_dark_tone_percentile_oracle 2230 PNGs
    raw_dark_tone_denoise_oracle    2230 PNGs
  output dir size: 23G
  /mnt/drive free: 663G, use 94%
  ```
  抽样 PNG 检查通过：shape `(800, 1200, 3)`，dtype `uint16`。
- **2026-06-09 17:08 CST：** oracle E10/E11 zero-training eval 完成，2/2 returncode=0。
  ```text
  summary csv : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary_oracle.csv
  summary json: /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/eval_zero/zero_eval_summary_oracle.json
  logs        : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1708_eval_zero_raw_dark_tone_*_oracle.log
  ```

  相对 `raw_dark_identity` zero-eval D1 `0.8450669757`：

  | oracle | val_d1 | Δ vs identity | recovered_ratio_zero |
  |---|---:|---:|---:|
  | E10 `raw_dark_tone_percentile_oracle` | 0.832221 | -0.012846 | -0.238 |
  | E11 `raw_dark_tone_denoise_oracle` | 0.829316 | -0.015751 | -0.292 |

  解读：
  ```text
  E10/E11 作为 test-time oracle preprocessing 并没有恢复 D1，反而低于 identity。
  这说明当前 RAW_dark checkpoint/RAM+DAv2 管线不直接受益于“向 RAW_normal 统计靠近”的图像域 oracle transform。
  结合 deployable dark-side zero-eval 全部接近持平或变差，下一步不宜只靠手工 preprocessing。
  但这仍是 zero-train 证据，不能单独否定 learned restoration；更强的判断需要 feature/RAM probe 或短训 restoration 方案。
  ```

### 17.7 RAM / DAv2 feature probe 第一版完成

- **2026-06-09 17:18-17:26 CST：** 实现并运行 `tools/lod_raw_ram_feature_probe.py`。该 probe 固定 `RAW_dark` L0 best checkpoint，分别喂入：
  ```text
  raw_dark_identity                  = 原始 RAW_dark，reference
  raw_dark_tone_percentile_oracle    = E10
  raw_dark_tone_denoise_oracle       = E11
  raw_normal_identity                = 同一个 RAW_dark 模型下喂 RAW_normal，用作 oracle sanity，不是公平 eval
  ```
  probe 抓两层信息：
  ```text
  RAM:  ram_core.forward_with_features(raw) 的 x3 / x_cat / ffm_mid
  DAv2: layers 2/5/8/11 的 patch token 和 CLS token
  ```
  记录每个输入的 p1/p50/p99/std/tail ratio，以及相对 reference 的 L1/L2/cosine distance。

  smoke 结果：
  ```text
  py_compile 通过。
  2-sample RAM-only smoke 通过。
  1-sample DAv2 token smoke 通过。
  成功 smoke 输出已清理；未保留 codex_smoke_ram_probe* 目录。
  ```
  修复过一个实现细节：GPU 上直接做大张量 pairwise reduction 会触发进程级 FPE，已改成先转 CPU float 再计算距离；脚本重新 `py_compile` 通过。

- **2026-06-09 17:26 CST：** 第一轮以 `raw_dark_identity` 为 reference 跑完整 112 val samples：
  ```text
  output dir : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/feature_probe/0609_1718_raw_dark_L0_e10_e11_normal_tokens
  log path   : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1718_ram_feature_probe_tokens.log
  rows       : per_sample 448, distance_to_reference 336
  files      : ram_feature_per_sample.csv, ram_feature_distance_to_reference.csv, ram_feature_summary.{json,md}
  ```

  相对 `raw_dark_identity` 的平均距离：

  | 输入 | RAM x3 L1 | RAM x3 cosine | DAv2 L2 cosine | DAv2 L5 cosine | DAv2 L8 cosine | DAv2 L11 cosine |
  |---|---:|---:|---:|---:|---:|---:|
  | E10 `raw_dark_tone_percentile_oracle` | 0.177992 | 0.065852 | 0.017373 | 0.013156 | 0.012668 | 0.015982 |
  | E11 `raw_dark_tone_denoise_oracle` | 0.188378 | 0.073949 | 0.020349 | 0.016075 | 0.016180 | 0.020592 |
  | `raw_normal_identity` sanity | 0.319836 | 0.079426 | 0.109136 | 0.083064 | 0.078499 | 0.090494 |

  读法：
  ```text
  E10/E11 确实改变了 RAW_dark 模型内部表示，但 DAv2 token 的改变远小于真正喂 RAW_normal 的改变。
  E10/E11 到 identity 的 DAv2 cosine 距离约为 RAW_normal sanity 的 15%-25%，说明它们不是在 DAv2 token 空间里大幅变成 RAW_normal。
  ```

- **2026-06-09 17:27-17:35 CST：** 第二轮把 reference 换成 `raw_normal_identity`，直接判断 E10/E11 是否更接近 RAW_normal：
  ```text
  output dir : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/feature_probe/0609_1727_raw_dark_L0_e10_e11_ref_normal_tokens
  log path   : /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs/0609_1727_ram_feature_probe_tokens_ref_normal.log
  rows       : per_sample 448, distance_to_reference 336
  files      : ram_feature_per_sample.csv, ram_feature_distance_to_reference.csv, ram_feature_summary.{json,md}
  ```

  相对 `raw_normal_identity` 的平均距离：

  | 输入 | RAM x3 L1 | RAM x3 cosine | DAv2 L2 cosine | DAv2 L5 cosine | DAv2 L8 cosine | DAv2 L11 cosine |
  |---|---:|---:|---:|---:|---:|---:|
  | `raw_dark_identity` | 0.319836 | 0.079426 | 0.109136 | 0.083064 | 0.078499 | 0.090494 |
  | E10 `raw_dark_tone_percentile_oracle` | 0.294110 | 0.066013 | 0.106929 | 0.084506 | 0.080184 | 0.097011 |
  | E11 `raw_dark_tone_denoise_oracle` | 0.289149 | 0.062889 | 0.098686 | 0.084674 | 0.080932 | 0.099004 |

  相对 `raw_dark_identity` 到 `raw_normal_identity` 的距离，E10/E11 的变化比例：
  ```text
  RAM x3 L1:
    E10 closer by 8.0%
    E11 closer by 9.6%
  RAM x3 cosine:
    E10 closer by 16.9%
    E11 closer by 20.8%
  DAv2 layer2 token cosine:
    E10 closer by 2.0%
    E11 closer by 9.6%
  DAv2 layer5/8/11 token cosine:
    E10 farther by 1.7% / 2.1% / 7.2%
    E11 farther by 1.9% / 3.1% / 9.4%
  ```

  解读：
  ```text
  E10/E11 只在 RAM x3 和 DAv2 early layer2 上轻微拉近 RAW_normal。
  中后层 token（layer5/8/11）没有拉近，反而略远。
  E11 比 E10 更拉近 RAM x3 和 layer2，但 zero-eval D1 更差（0.8293 < 0.8322 < identity 0.8451）。
  因此“把输入统计做 oracle tone/denoise”没有稳定变成 DAv2 可用的 feature alignment。
  当前证据更支持：问题不只是图像域亮度/简单去噪；噪声/低 SNR 主导仍成立，同时存在 RAM/DAv2 中后层 token mismatch 或训练域适配问题。
  ```

### 17.8 `report_v1.md` 已生成

- **2026-06-09 17:40 CST：** 已按 §8 格式生成正式归因报告：
  ```text
  /mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/report_v1.md
  ```
  报告已汇总：
  ```text
  §17.6 short-train anchors/degrade
  E10/E11 oracle zero-eval
  §17.7 RAM/DAv2 feature probe
  audit_v1 data-level summary
  attribution per-sample correlation
  zero-train ablation ranking
  decision / next direction
  ```
  当前报告结论：
  ```text
  主瓶颈 = 噪声 / 低 SNR（主导但非全部解释）。
  单纯曝光 / 动态范围不足不成立为主因。
  E10/E11 zero-train oracle preprocessing 无效。
  E10/E11 只轻微改善 RAM/early token，对 DAv2 中后层 token 不构成有效 RAW_normal alignment。
  下一步不宜继续押注手工 preprocessing；更合理的是训练式 restoration / denoise，或面向 DAv2 layer5/8/11 的 feature-level alignment / distillation。
  ```

### 17.9 待办（后续跟进）

1. 如需补强 §7 的系统状态对比，可再跑 RAW_normal checkpoint 侧 feature probe：`RAW_normal model: raw_normal_identity vs raw_normal_to_dark_noise_only/exposure`，用于确认 normal 模型中 synthetic degrade 是否也主要扰动 DAv2 中后层 token。
2. 整理 failure galleries：从 `audit_v1/panels`、`attribution_join.csv`、feature-probe per-sample CSV 中各挑 8-12 个高噪声、高 edge loss、高 chromaticity shift、高 feature mismatch 样本。
3. 若进入下一阶段实验，优先设计训练式 RAW_dark restoration / denoise 或 DAv2 layer5/8/11 feature distillation；不要再优先做无训练 median/bilateral/tone preprocessing。
