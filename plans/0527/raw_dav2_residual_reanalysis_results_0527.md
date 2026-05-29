# RAW-like Residual Reanalysis 结果汇总

日期：2026-05-28

本文用于持续记录 `plans/0527/raw_dav2_residual_reanalysis_plan_0527.md` 对应的实验结果与阶段性结论。当前版本已记录 N2 `lambda_lp` sweep、N2 `q_good` sweep，以及 N3 RGB incremental correction control；所有结果都和前一轮核心 baseline 对齐比较。

写作规约：主指标表只放 formal eval 或同口径 eval json/log 中的数字；只用于 smoke、诊断或可视化的 sample loss 不混入主指标表。除非特别说明，本文所有主表都使用 **VKITTI overall abs_rel best checkpoint**，避免 overall best 和 boundary/target-region best checkpoint 混用。

指标标注规约：`abs_rel` 越低越好，`d1` 越高越好。`D0` 指 frozen DAv2-S 在同一 halfres RGB 输入上得到的初始深度，经同一 per-image affine disparity protocol 对齐后计算。`D1` 指 N2 / N3 中 frozen C2 calibrator 对 `D0` 的输出。

## 0. 协议速查 / 数据源规则

当前纳入比较的 run：

- Frozen DAv2-S `D0`
  - baseline，不是单独训练 run。
  - VKITTI Scene20 holdout：`abs_rel ~= 0.1531, d1 ~= 0.8184`。
  - KITTI halfres canonical sanity：`abs_rel ~= 0.1184, d1 ~= 0.8665`。
- `0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - label：C2 D0-only residual calibrator。
  - path：halfres RGB -> frozen DAv2-S -> `D0`；ResidualGateHead 输入为 `D0_norm`。
  - VKITTI overall-best checkpoint：epoch 11。
  - 该 checkpoint 是 N2 的 frozen C2 checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_11.pth`。
- `0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - label：M2-RA0 direct RAW/RAM residual，`raw4 + ffm_mid`。
  - path：halfres RGB -> RAW-Adapter-style analytic unprocessing -> packed RAW4 -> RamCore3 / RAW front-end -> residual correction。
  - VKITTI overall-best checkpoint：epoch 9。
- `0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - label：M1-RA0 direct RAW/RAM residual，`raw4 + x3 + D0_norm concat`。
  - path：halfres RGB -> RAW-Adapter-style analytic unprocessing -> packed RAW4 -> RamCore3 / RAW front-end -> residual correction。
  - VKITTI overall-best checkpoint：epoch 14。
- `0527_2144_vkitti_n2_x3_lp0p0_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - label：N2 `lambda_lp=0.0`。
  - 注意：该 run 没有 `run_summary.json`，日志停在 epoch 13 eval start，当前未检测到对应训练进程；本文只引用其已有 `best_val_metrics.json`。
- `0527_2300_vkitti_n2_x3_lp0p3_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10`
  - label：N2 `lambda_lp=0.3`。
- `0527_2354_vkitti_n2_x3_lp0p5_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10`
  - label：N2 `lambda_lp=0.5`。
- `0528_0049_vkitti_n2_x3_lp0p8_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10`
  - label：N2 `lambda_lp=0.8`。
- `0529_1752_vkitti_n3_rgb_lp0p5_q0p3_lfl0p0_rftna_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10`
  - label：N3 RGB incremental correction control，`lambda_lp=0.5, q_good=0.3`。
  - path：halfres RGB -> frozen DAv2-S -> frozen C2 `D1` -> RGB incremental branch correction。
  - VKITTI overall-best checkpoint：epoch 2。

N2 固定设置：

```text
method_id=N2
front_end=c2_frozen_raw_ram_incremental
input_domain=raw4
model_input_tensor=raw
raw_storage_format=synthetic_packed_bayer_4ch_halfres
incremental_feature_source=x3
delta_condition=feature_only
gate_condition=feature_d1
q_good=0.5
lambda_lowfreq_loss=0.0
lambda_lp in 0.0 / 0.3 / 0.5 / 0.8
```

N2 数据流：

```text
D0 = frozen_DAV2(sRGB)
D1 = frozen_C2_calibrator(D0)
x3 = RamCore3(raw4).x3

delta_raw = DeltaHead(F_x3)
gate_raw = GateHead(concat(F_x3, F_d1))
delta_effective = delta_raw - lambda_lp * lowpass(delta_raw)
D_final = D1 + gate_raw * delta_effective
```

N3 RGB control 固定设置：

```text
method_id=N3
front_end=c2_frozen_rgb_incremental
input_domain=rgb
model_input_tensor=image
raw_storage_format=not_applicable
incremental_feature_source=rgb
delta_condition=feature_only
gate_condition=feature_d1
q_good=0.3
lambda_lp=0.5
lambda_lowfreq_loss=0.0
```

数据源优先级：

1. N2 / N3 主结果：各 run 的 `best_val_metrics.json`，其中 VKITTI / KITTI / region / diagnostics 均来自同一个 VKITTI overall-best checkpoint。
2. C2 / M2-RA0 / M1-RA0 VKITTI overall：各 run 的 `best_val_metrics.json`。
3. C2 / M2-RA0 / M1-RA0 KITTI sanity：各 run 的 `kitti_val_metrics.json` 中与 VKITTI overall-best epoch 相同的条目。
4. C2 / M2-RA0 / M1-RA0 clipped region metrics：`plans/result/0527_vkitti_region_clip_recalc_section_1_2_all.json`。

## 1. 实验参数设置

### 1.1 字段速查

| 字段 | 说明 |
|---|---|
| `eff_bs` | `bs * accum_steps`。当前所有纳入比较 run 均为 `8 * 1 = 8`。 |
| `D0` | frozen DAv2-S 在 halfres RGB 输入上的初始预测。 |
| `D1` | N2 中 frozen C2 calibrator 对 `D0` 的输出；N2 的增量分支在 `D1` 上继续修正。 |
| `lambda_lp` | N-series architectural high-pass subtract strength：`delta_effective = delta_raw - lambda_lp * lowpass(delta_raw)`。 |
| `q_good` | N-series per-image good-D1 mask quantile；当前 N2 sweep 固定或显式扫描该值。 |
| `lambda_lowfreq_loss` | N-series loss-side low-frequency regularizer 权重；当前 sweep 固定 `0.0`，避免和 `lambda_lp` 混淆。 |
| `correction scale` | 旧 residual run 表示 `mean_abs_delta`；N2 / N3 表示实际施加项 `mean_abs_gate_delta`。 |
| `raw_adapter_config_hash=5fd8f0d2345f9683` | 当前 clean RA0 / N2 synthetic RAW 设置的配置 hash。 |

### 1.2 共同设置

| 项 | 值 |
|---|---|
| 默认 conda env | `dav3` |
| 项目根目录 | `/home/caq/6666_raw/dav2_raw_0522` |
| C2 代码入口 | `foundation/tools/train_vkitti2_residual_control.py` |
| M1/M2 代码入口 | `foundation/tools/train_vkitti2_raw_residual.py` |
| N2/N3 代码入口 | `foundation/tools/train_vkitti2_incremental_residual.py` |
| encoder / pretrained | `vits` / `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth` |
| split | VKITTI train `finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt`；VKITTI val `finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt` |
| split size | train `11870`，VKITTI val `1000`，KITTI val `652` |
| geometry | original `375x1242` -> crop bottom row -> even fullres `374x1242` -> model input `187x621` |
| RGB / depth space | `rgb_input_space=halfres_2x2_area`，`depth_target_space=halfres_2x2_valid_mean` |
| depth range | `min_depth=1.0`，`max_depth=80.0` |
| optimizer / LR | AdamW，`lr=1e-4`，`weight_decay=1e-4` |
| AMP / seed | bf16 AMP，`seed=42` |
| augmentation | `hflip_prob=0.5` |
| eval cadence | `eval_interval=1`，`save_interval=1`，`save_best_checkpoint=true` |
| VKITTI eval protocol | `per_image_affine_disp_depth_anything_v2` |
| KITTI eval protocol | C2/N3: `halfres_rgb_canonical_even_pad_crop_affine_disp`；RAW/N2: `halfres_raw_canonical_even_pad_crop_affine_disp` |
| heavy artifact root | `/mnt/drive/3333_raw/0000_exp_ckpt/<run>/` |

### 1.3 核心 run 参数矩阵

| Run | Method | input / interface | train scope | loss semantic params | epochs | bs/acc/eff | LR / wd | trainable params | notes |
|---|---|---|---|---|---:|---|---|---:|---|
| `0525_0203...c2_d0only...` | C2 D0-only residual | `input_domain=rgb`；`model_input_tensor=image`；`front_end=dav2_rgb_frozen`；`raw_storage_format=not_applicable`；`residual_feature_source=d0` | frozen DAv2-S；ResidualGateHead trainable | old residual loss: `L_depth + 0.5*L_grad + 0.1*L_keep + 0.01*L_res + 0.005*L_gate + 0.05*L_gate_sup` | 20 | `8/1/8` | `1e-4 / 1e-4` | 2,881,858 | total params `27,666,947`；frozen `24,785,089`；VKITTI best e11 |
| `0525_1425...m2_ra0_rawadapter...` | M2-RA0 direct RAW/RAM residual | `input_domain=raw4`；`model_input_tensor=raw`；`front_end=raw_to_base_rgb_ram3`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`residual_feature_source=ffm_mid` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | old residual loss, same weights as C2/M-series | 20 | `8/1/8` | `1e-4 / 1e-4` | 3,059,625 | clean RA0；VKITTI best e9；KITTI best e12 |
| `0526_0040...m1_ra0_x3_d0concat...` | M1-RA0 direct x3 residual | `input_domain=raw4`；`model_input_tensor=raw`；`front_end=raw_to_base_rgb_ram3`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`residual_feature_source=x3`；`residual_head_d0_mode=concat` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | old residual loss, same weights as C2/M-series | 20 | `8/1/8` | `1e-4 / 1e-4` | 3,024,489 | clean RA0；VKITTI best e14；KITTI best e9 |
| `0527_2144...n2_x3_lp0p0...` | N2 C2-frozen x3 incremental | `input_domain=raw4`；`model_input_tensor=raw`；`front_end=c2_frozen_raw_ram_incremental`；`incremental_feature_source=x3`；`delta_condition=feature_only`；`gate_condition=feature_d1` | frozen DAv2-S + frozen C2；RAW detail incremental branch trainable | N2 loss, `lambda_lp=0.0`，`q_good=0.5`，`lambda_lowfreq_loss=0.0` | 20 | `8/1/8` | `1e-4 / 1e-4` | 420,393 | incomplete: no `run_summary.json`; best json available |
| `0527_2300...n2_x3_lp0p3...` | N2 C2-frozen x3 incremental | same as N2 `lp0.0` | same as N2 `lp0.0` | N2 loss, `lambda_lp=0.3`，`q_good=0.5`，`lambda_lowfreq_loss=0.0` | 10 | `8/1/8` | `1e-4 / 1e-4` | 420,393 | completed; VKITTI best e8 |
| `0527_2354...n2_x3_lp0p5...` | N2 C2-frozen x3 incremental | same as N2 `lp0.0` | same as N2 `lp0.0` | N2 loss, `lambda_lp=0.5`，`q_good=0.5`，`lambda_lowfreq_loss=0.0` | 10 | `8/1/8` | `1e-4 / 1e-4` | 420,393 | completed; VKITTI best e7 |
| `0528_0049...n2_x3_lp0p8...` | N2 C2-frozen x3 incremental | same as N2 `lp0.0` | same as N2 `lp0.0` | N2 loss, `lambda_lp=0.8`，`q_good=0.5`，`lambda_lowfreq_loss=0.0` | 10 | `8/1/8` | `1e-4 / 1e-4` | 420,393 | completed; VKITTI best e5 |
| `0529_1752...n3_rgb_lp0p5_q0p3...` | N3 C2-frozen RGB incremental control | `input_domain=rgb`；`model_input_tensor=image`；`front_end=c2_frozen_rgb_incremental`；`raw_storage_format=not_applicable`；`incremental_feature_source=rgb`；`delta_condition=feature_only`；`gate_condition=feature_d1` | frozen DAv2-S + frozen C2；RGB incremental branch trainable | N-series loss, `lambda_lp=0.5`，`q_good=0.3`，`lambda_lowfreq_loss=0.0` | 10 | `8/1/8` | `1e-4 / 1e-4` | 279,490 | total params `27,946,437`；frozen `27,666,947`；completed; VKITTI best e2 |

Clean RA0 / N2 synthetic RAW settings are shared unless a row says otherwise:

```text
unprocessing_method=raw_adapter_style
raw_adapter_backend=analytic
raw_adapter_ccm=identity
raw_adapter_inverse_tone=global_0p15
raw_adapter_rgb_transfer=srgb_piecewise
raw_adapter_cfa_pattern=RGGB
raw_adapter_packed_channel_order=R_Gr_Gb_B
randomize_unprocessing=false
raw_adapter_variant_policy=normal
raw_adapter_fixed_red_gain=2.15
raw_adapter_fixed_blue_gain=1.7
raw_adapter_fixed_light_scale=1.0
raw_adapter_config_hash=5fd8f0d2345f9683
```

### 1.4 N-series incremental loss 参数

N2 / N3 Stage-2 incremental branch 使用：

```text
L = 1.0 * L_final
  + 2.0 * L_boundary
  + 0.5 * L_grad
  + 0.2 * L_keep_good_D1
  + 0.05 * L_gate_sparse
  + 0.0 * L_lowfreq
  + 0.1 * L_invalid_keep
```

其中：

```text
L_boundary: GT depth gradient top-10% boundary pixels
L_keep_good_D1: per-image q_good=0.5 mask where D1 is already good
L_lowfreq: loss-side low-frequency penalty; current sweep explicitly disabled
L_invalid_keep: invalid pixels suppress gate * delta_effective drift
lowpass_kernel=31
raw_feature_encoder_trainable=true for N2; not_applicable for N3 RGB control
```

## 2. VKITTI Scene20 holdout val

### 2.1 Overall metrics at VKITTI overall-best checkpoint

下表中，C2 / M2-RA0 / M1-RA0 的 delta 以 `D0` 为 baseline；N2 / N3 的 delta 以 frozen C2 output `D1` 为 baseline。KITTI 列也使用同一个 VKITTI overall-best checkpoint，而不是 KITTI-best checkpoint。

| Method | selected ckpt | VK abs_rel | VK d1 | delta target | KITTI same ckpt abs_rel | KITTI same ckpt d1 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen DAv2-S `D0` | n/a | 0.1531 | 0.8184 | n/a | 0.1184 | 0.8665 | frozen DAV2-S baseline |
| C2 D0-only | e11 | 0.1210 | 0.8558 | -0.0321 vs D0 | 0.0965 | 0.8960 | frozen C2 calibrator for N2 |
| M2 RA0 `ffm_mid + D0` | e9 | 0.1262 | 0.8536 | -0.0269 vs D0 | 0.0970 | 0.8977 | previous direct RAW/RAM residual |
| M1 RA0 `x3 + D0 concat` | e14 | 0.1254 | 0.8537 | -0.0277 vs D0 | 0.0958 | 0.8961 | previous best direct x3 residual |
| N2 x3 `lp0.0` | e3 | 0.1203 | 0.8557 | -0.0007 vs D1 | 0.0965 | 0.8957 | incomplete run; best json available |
| N2 x3 `lp0.3` | e8 | 0.1203 | 0.8569 | -0.0007 vs D1 | 0.0969 | 0.8952 | overall-best checkpoint |
| N2 x3 `lp0.5` | e7 | 0.1190 | **0.8578** | -0.0020 vs D1 | 0.0966 | 0.8953 | overall-best checkpoint |
| N2 x3 `lp0.8` | e5 | **0.1186** | 0.8575 | **-0.0024 vs D1** | 0.0975 | 0.8932 | best VKITTI overall; KITTI regression largest |
| N3 RGB `lp0.5 q0.3` | e2 | 0.1194 | 0.8567 | -0.0016 vs D1 | 0.0971 | 0.8944 | RGB incremental control; below matched N2 `lp0.5 q0.3` |

### 2.2 Region metrics at the same checkpoint

Region mask 定义沿用 0527 clipped region 口径：

```text
boundary = top-10% GT depth gradient magnitude within valid GT pixels
DAv2 high-error = top-20% abs(D0_norm - y_norm) within valid GT pixels
far50 = valid GT depth > 50m
dark = RGB preview luma < 0.15
saturated = max(R,G,B) > 0.95
```

Region `abs_rel` 均使用 per-image affine disparity 对齐后的 depth，并在计算前 clip 到 `[1.0, 80.0]`。`correction scale` 对旧 residual run 表示 `mean_abs_delta`；对 N2 / N3 表示 `mean_abs_gate_delta`，因为 incremental branch 真正施加到 D1 上的是 `gate * delta_effective`。

| Method | epoch | boundary | high-error | far50 | dark | saturated | mean_gate | correction scale | low_ratio | high_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen DAv2-S `D0` | n/a | 0.3497 | 0.3484 | 0.2481 | 0.1317 | 0.2333 | n/a | n/a | n/a | n/a |
| C2 D0-only | 11 | 0.2692 | **0.2198** | 0.2734 | **0.1026** | 0.1357 | 0.3358 | 0.4537 | n/a | n/a |
| M2 RA0 `ffm_mid + D0` | 9 | 0.2714 | 0.2316 | **0.2254** | 0.1080 | 0.1469 | 0.3175 | 0.4707 | n/a | n/a |
| M1 RA0 `x3 + D0 concat` | 14 | 0.2592 | 0.2252 | 0.2547 | 0.1038 | 0.1413 | 0.3233 | 0.4698 | n/a | n/a |
| N2 x3 `lp0.0` | 3 | 0.2636 | 0.2660 | 0.2732 | 0.1033 | 0.1297 | 0.0306 | 0.0153 | 0.9361 | 0.5878 |
| N2 x3 `lp0.3` | 8 | 0.2619 | 0.2665 | 0.2664 | 0.1040 | 0.1307 | 0.0338 | 0.0122 | 0.9069 | 0.7373 |
| N2 x3 `lp0.5` | 7 | 0.2545 | 0.2612 | 0.2697 | 0.1028 | 0.1266 | 0.0309 | 0.0101 | 0.7592 | 1.0629 |
| N2 x3 `lp0.8` | 5 | **0.2505** | 0.2589 | 0.2687 | 0.1032 | **0.1245** | 0.0264 | 0.0097 | 0.7303 | 1.2294 |
| N3 RGB `lp0.5 q0.3` | 2 | 0.2542 | 0.2615 | 0.2667 | 0.1036 | 0.1227 | 0.0336 | 0.0107 | 0.7475 | 1.0674 |

## 3. N2 lambda_lp sweep 细表

本表只展示 N2，且每行均为该 run 的 VKITTI overall-best checkpoint。`final-D1`、`boundary-D1`、`far50-D1` 为 N2 final 相对 frozen C2 `D1` 的差值，负数表示 N2 更好。

| lambda_lp | epoch | VK final | D1 | final-D1 | boundary | boundary-D1 | far50 | far50-D1 | low_ratio | high_ratio | KITTI final | KITTI final-D1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 3 | 0.120302 | 0.120994 | -0.000692 | 0.263600 | -0.005597 | 0.273200 | -0.000202 | 0.936052 | 0.587753 | 0.096546 | +0.000074 |
| 0.3 | 8 | 0.120285 | 0.120994 | -0.000709 | 0.261923 | -0.007274 | 0.266377 | -0.007026 | 0.906853 | 0.737286 | 0.096870 | +0.000397 |
| 0.5 | 7 | 0.118974 | 0.120994 | -0.002020 | 0.254521 | -0.014673 | 0.269733 | -0.003667 | 0.759182 | 1.062943 | 0.096603 | +0.000131 |
| 0.8 | 5 | **0.118635** | 0.120994 | **-0.002359** | **0.250456** | **-0.018743** | 0.268708 | -0.004693 | 0.730285 | 1.229382 | 0.097497 | +0.001025 |

## 4. N2 q_good sweep 预登记 / 阶段结果

本节记录 2026-05-29 启动的 N2 `q_good` sweep。该 sweep 以第 3 节的 `lambda_lp` sweep 结果为基础：保留 `lambda_lp=0.5` 作为 KITTI sanity 较稳候选，同时追加 `lambda_lp=0.8` 作为 VKITTI / boundary 更激进候选。`q_good=0.5` 行来自第 3 节已经完成的 run，作为本 sweep 的中间点 anchor；`q_good=0.3 / 0.7` 为新增 formal queue。

当前队列：

```text
tmux session: 0529_1333_vkitti_n2_qgood_lp05_lp08
queue log: /home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0529_1333_vkitti_n2_qgood_lp05_lp08.queue.log
launch time: 2026-05-29 13:33 CST
```

共同参数除下表列出的 `lambda_lp / q_good / epochs` 外，与第 1 节 N2 设置一致：`method_id=N2`，`front_end=c2_frozen_raw_ram_incremental`，`input_domain=raw4`，`model_input_tensor=raw`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`，`incremental_feature_source=x3`，`delta_condition=feature_only`，`gate_condition=feature_d1`，`raw_feature_encoder_trainable=true`，`lambda_lowfreq_loss=0.0`，`lowpass_kernel=31`，`bs=8`，`accum_steps=1`，`lr=1e-4`，`weight_decay=1e-4`，`seed=42`。

### 4.1 q_good sweep run 状态

| Run | lambda_lp | q_good | epochs | status | source / note |
|---|---:|---:|---:|---|---|
| `0529_1334_vkitti_n2_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10` | 0.5 | 0.3 | 10 | completed | new q_good sweep; best e8 |
| `0527_2354_vkitti_n2_x3_lp0p5_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10` | 0.5 | 0.5 | 10 | completed | anchor from lambda_lp sweep; best e7 |
| `0529_1429_vkitti_n2_x3_lp0p5_q0p7_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10` | 0.5 | 0.7 | 10 | completed | new q_good sweep; best e3 |
| `0529_1523_vkitti_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10` | 0.8 | 0.3 | 10 | completed | new q_good sweep; VKITTI best e3; KITTI-best e9 |
| `0528_0049_vkitti_n2_x3_lp0p8_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10` | 0.8 | 0.5 | 10 | completed | anchor from lambda_lp sweep; best e5 |
| `0529_1648_vkitti_n2_x3_lp0p8_q0p7_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10` | 0.8 | 0.7 | 10 | completed | new q_good sweep; VKITTI best e5; KITTI-best e7 |

### 4.2 q_good sweep overall metrics at VKITTI overall-best checkpoint

本表风格对齐第 2.1 节：仍使用 VKITTI overall-best checkpoint；KITTI 列使用同一个 checkpoint，不使用 KITTI-best checkpoint。未完成或未启动的 run 暂不填中间日志数字。这里的 delta target 统一以 frozen C2 output `D1` 为 baseline。

| Method | selected ckpt | VK abs_rel | VK d1 | delta target | KITTI same ckpt abs_rel | KITTI same ckpt d1 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| N2 x3 `lp0.5 q0.3` | e8 | 0.118629 | **0.858525** | -0.002369 vs D1 | 0.096933 | 0.894446 | best VKITTI d1 among current q_good sweep |
| N2 x3 `lp0.5 q0.5` | e7 | 0.118974 | 0.857796 | -0.002020 vs D1 | **0.096603** | **0.895309** | KITTI sanity best among lp0.5 q sweep |
| N2 x3 `lp0.5 q0.7` | e3 | 0.118736 | 0.857355 | -0.002258 vs D1 | 0.096672 | 0.894960 | middle ground; VK close to q0.3, KITTI close to q0.5 |
| N2 x3 `lp0.8 q0.3` | e3 | **0.118624** | 0.857758 | **-0.002374 vs D1** | 0.096738 | 0.895021 | best VKITTI overall among current q_good sweep; KITTI same-ckpt improves over lp0.8 q0.5 |
| N2 x3 `lp0.8 q0.5` | e5 | 0.118635 | 0.857467 | -0.002359 vs D1 | 0.097497 | 0.893181 | anchor from lambda sweep |
| N2 x3 `lp0.8 q0.7` | e5 | 0.118658 | 0.857217 | -0.002337 vs D1 | 0.097738 | 0.892737 | VK close to q0.3/q0.5, but KITTI same-ckpt sanity is weakest |

### 4.3 q_good sweep region metrics at the same checkpoint

本表风格对齐第 2.2 节，所有 region metrics 都来自第 4.2 节同一个 VKITTI overall-best checkpoint。Region metric 口径与第 2.2 节一致；`correction scale` 表示 N2 实际施加项 `mean_abs_gate_delta`。

| Method | epoch | boundary | high-error | far50 | dark | saturated | mean_gate | correction scale | low_ratio | high_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N2 x3 `lp0.5 q0.3` | 8 | 0.250046 | **0.258165** | 0.264208 | 0.104519 | **0.121147** | 0.047703 | 0.014935 | 0.772791 | 0.857397 |
| N2 x3 `lp0.5 q0.5` | 7 | 0.254521 | 0.261179 | 0.269733 | **0.102792** | 0.126631 | 0.030890 | 0.010059 | 0.759182 | 1.062943 |
| N2 x3 `lp0.5 q0.7` | 3 | 0.251217 | 0.258940 | 0.268822 | 0.104010 | 0.122850 | 0.042501 | 0.013347 | 0.785908 | 0.914558 |
| N2 x3 `lp0.8 q0.3` | 3 | **0.249490** | 0.259498 | 0.267145 | 0.102838 | 0.123042 | 0.028699 | 0.010370 | 0.714824 | 1.190763 |
| N2 x3 `lp0.8 q0.5` | 5 | 0.250456 | 0.258877 | 0.268708 | 0.103195 | 0.124461 | 0.026367 | 0.009731 | 0.730285 | 1.229382 |
| N2 x3 `lp0.8 q0.7` | 5 | 0.249673 | 0.260573 | **0.261204** | 0.104816 | 0.125410 | 0.027156 | 0.009789 | 0.735162 | 1.219245 |

### 4.4 阶段性观察

- 在 `lambda_lp=0.5` 下，`q_good=0.3` 的 VKITTI overall best 为 `0.118629`，略好于既有 `q_good=0.5` 的 `0.118974`，boundary 也更低：`0.250046` vs `0.254521`。
- `q_good=0.3` 的 KITTI same-checkpoint sanity 为 `0.096933`，比 `q_good=0.5` 的 `0.096603` 更差；因此当前只能写成 VKITTI / boundary 改善更强，跨域 sanity 不如 `q_good=0.5` 稳。
- `q_good=0.7` 的 VKITTI overall 为 `0.118736`，介于 `q_good=0.3` 和 `q_good=0.5` 之间；boundary `0.251217` 也介于二者之间。
- `q_good=0.7` 的 KITTI same-checkpoint sanity 为 `0.096672`，比 `q_good=0.3` 稳，但仍弱于 `q_good=0.5`。
- 在 `lambda_lp=0.8` 下，`q_good=0.3` 成为当前 q_good sweep 的 VKITTI overall best：`0.118624`，同时 boundary 达到 `0.249490`，略好于 `lp0.8/q0.5` 的 `0.250456` 和 `lp0.5/q0.3` 的 `0.250046`。
- `lp0.8/q0.3` 的 KITTI same-checkpoint sanity 为 `0.096738`，明显好于 `lp0.8/q0.5` 的 `0.097497`，但仍弱于 `lp0.5/q0.5` 的 `0.096603`。
- `lp0.8/q0.7` 的 VKITTI overall 为 `0.118658`，接近 `lp0.8/q0.3/q0.5`，且 far50 为当前 q_good sweep 最低的 `0.261204`；但 KITTI same-checkpoint sanity 为 `0.097738`，是当前 sweep 最弱。
- 两个 `lp0.8` 新 run 均已完成；本节现在只引用各 run 的 VKITTI overall-best checkpoint，不再依赖未完成 run 的中间日志。

### 4.5 N3 RGB incremental control

N3 是 plan 第 11 节定义的 RGB incremental correction control，用来判断 N2 的 x3/RAW-like detail cue 是否优于普通 RGB detail cue。本节把 N3 和最接近的 N2 设置放在同一口径比较：所有行都使用 VKITTI overall-best checkpoint，KITTI 列使用同一 checkpoint。

| Method | feature source | lambda_lp | q_good | epoch | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | saturated | KITTI same ckpt abs_rel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 D0-only `D1` | D0-only calibrator | n/a | n/a | 11 | 0.120998 | 0.855789 | n/a | 0.269197 | 0.271879 | 0.273403 | 0.135660 | 0.096472 |
| N3 RGB `lp0.5 q0.3` | RGB | 0.5 | 0.3 | 2 | 0.119356 | 0.856687 | -0.001639 | 0.254240 | 0.261532 | 0.266733 | 0.122711 | 0.097086 |
| N2 x3 `lp0.5 q0.3` | RAW/RAM x3 | 0.5 | 0.3 | 8 | 0.118629 | 0.858525 | -0.002369 | 0.250046 | **0.258165** | 0.264208 | **0.121147** | 0.096933 |
| N2 x3 `lp0.8 q0.3` | RAW/RAM x3 | 0.8 | 0.3 | 3 | **0.118624** | 0.857758 | **-0.002374** | **0.249490** | 0.259498 | 0.267145 | 0.123042 | **0.096738** |

阶段判断：

- N3 RGB control 本身有效：相对 C2 `D1`，VKITTI overall 改善 `-0.001639`，boundary 改善 `-0.014957`，说明 extra incremental head / RGB detail cue 也能解释一部分收益。
- 在 matched `lambda_lp=0.5, q_good=0.3` 下，N2 x3 优于 N3 RGB：VKITTI overall `0.118629` vs `0.119356`，boundary `0.250046` vs `0.254240`，high-error `0.258165` vs `0.261532`，KITTI same-checkpoint `0.096933` vs `0.097086`。
- 当前最好 N2 `lp0.8/q0.3` 也优于 N3 RGB：VKITTI overall `0.118624` vs `0.119356`，boundary `0.249490` vs `0.254240`，KITTI same-checkpoint `0.096738` vs `0.097086`。
- 因此，plan 中的 `N2 > N3 RGB control` 在当前 single-run formal eval 上成立；但 margin 是小幅优势，仍需 true/zero/mean/shuffled x3 eval 和 D1-only extra head control 来排除 extra head / loss / regularization 的解释。

## 5. 当前结论

- N2 相比 previous direct RAW/RAM residual 有实质改善：`lp0.5/q0.3` 和 `lp0.8/q0.3` 的 VKITTI overall abs_rel 分别为 `0.118629` / `0.118624`，均优于 C2 `0.1210`、M1 RA0 `0.1254` 和 M2 RA0 `0.1262`。
- N2 的收益主要来自 C2 之后的增量修正。`lp0.8/q0.3` 相对 D1 的 overall 改善为 `-0.002374`，boundary 改善为 `-0.019708`，说明 x3 incremental branch 在局部结构区域确实有正信号。
- N3 RGB incremental control 已完成。它也能改善 C2，但 matched N2 x3 `lp0.5/q0.3` 在 VKITTI overall、boundary、high-error、saturated 和 KITTI same-checkpoint 上均小幅优于 N3，支持 x3/RAW-like cue 有边际贡献。
- `lambda_lp` 越大，低频比例下降、高频比例上升：`low_ratio 0.936 -> 0.730`，`high_ratio 0.588 -> 1.229`。这和 N2 设计目标一致：RAW/x3 branch 更偏 local/detail correction，而不是重新学习 C2 的低频 calibration。
- `lp0.8/q0.3` 是当前 VKITTI overall / boundary 最强；但 KITTI same-checkpoint sanity 仍不如 `lp0.5/q0.5` 稳：`0.096738` vs `0.096603`。`lp0.8/q0.7` 的 KITTI same-checkpoint regression 最大，`KITTI final-D1=+0.001266`。
- `far50` 仍不是 N2 的优势项。M2 RA0 `ffm_mid + D0` 在 far50 上仍最好，`0.2254`；N2 虽相对 D1 略有改善，但 far50 仍显著弱于 M2 RA0 和 D0。
- high-error region 仍由 C2 最强，`0.2198`。N2 的 high-error 指标比 C2 差，说明 N2 当前更像 boundary/saturated/local-detail correction，而不是全面替代 C2 的 high-error correction。

## 6. 后续使用建议

- 后续若选择主候选，`lp0.8/q0.3` 可作为 VKITTI overall / boundary 最强的激进候选；`lp0.5/q0.5` 仍应作为 KITTI same-checkpoint sanity 更稳的保守候选。
- 结论不能写成 RAW/x3 已经全面优于所有控制项。更准确的是：N2 证明 `C2 frozen + x3 incremental correction` 能在 VKITTI overall 和 boundary 上超过 C2，但还没有证明跨域 KITTI 更稳，也没有证明 far50 / high-error 全面优于旧方法。
- 还缺两组关键对照：N2 checkpoint 的 true/zero/mean/shuffled x3 eval，N5 D1-only extra head。N3 RGB incremental correction 已完成；没有剩余两组对照前，仍不能完全排除一部分收益来自 extra head / training loss / regularization，而不是 x3 本身。
