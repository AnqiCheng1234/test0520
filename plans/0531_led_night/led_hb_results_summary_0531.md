# LED-HB 结果汇总

日期：2026-06-01

本文用于持续记录 `plans/0531_led_night/led_night_main_experiment_plan.md` 和 `plans/0531_led_night/led_hb_code_and_formal_experiment_execution_plan.md` 对应的 LED-HB 实验参数与结果。当前版本已纳入 C2 D0-only residual run，以及 2026-06-01 完成的 N5 / N3 / N2 / N7 四条 N-series run；frozen DAv2-S `D0` standalone eval 只作为 fp32 sanity/reference 记录。

写作规约：主指标表只放 formal eval、run 内 best metrics 或同口径 eval json/log 中的数字；smoke、probe、诊断 sample loss 不混入主指标表。`abs_rel` / `rmse` / `silog_x100` 越低越好，`d1/d2/d3` 越高越好。

## 0. 协议速查 / 数据源规则

当前纳入比较的结果：

- Frozen DAv2-S `D0` standalone eval reference
  - path：`plans/0531_led_night/led_hb_l0_d0_eval_0531_2339/d0_summary.json`
  - val：`hb_val_stride5_n1000_seed42`，1000 samples。
  - recommended `d0_sign=1`。
  - `pretrained_from` in this eval points to the backup mirror path `/mnt/drive/9999_backup_from_home_0514/.../depth_anything_v2_vits.pth`；training runs use `/home/caq/333_cvpr/.../depth_anything_v2_vits.pth`。
  - fp32 overall reference：`abs_rel=0.55523, d1=0.37196`。
- `0531_2341_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e20`
  - path：`finetune_stf/exp/0531_2341_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e20`
  - heavy checkpoint root：`/mnt/drive/3333_raw/0000_exp_ckpt/0531_2341_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e20`
  - source：`best_val_metrics.json` / `run_summary.json`。
  - LED-HB overall-best checkpoint：epoch 19。
  - same-pass AMP/bf16 D0 baseline：`abs_rel=0.55660, d1=0.37099`。
- `0601_0355_led_hb_n5_d1_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - label：N5 D1-only incremental control。
  - path：`finetune_stf/exp/0601_0355_led_hb_n5_d1_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - LED-HB overall-best checkpoint：epoch 8。
- `0601_0550_led_hb_n3_rgb_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - label：N3 RGB incremental control。
  - path：`finetune_stf/exp/0601_0550_led_hb_n3_rgb_lp0p5_q0p3_lfl0p0_rftnot_applicable_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - LED-HB overall-best checkpoint：epoch 8。
- `0601_0751_led_hb_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - label：N2 x3 incremental，`lambda_lp=0.8, q_good=0.3`。
  - path：`finetune_stf/exp/0601_0751_led_hb_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - LED-HB overall-best checkpoint：epoch 6。
- `0601_1005_led_hb_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - label：N7 x3 stop-gradient D1 delta control，`lambda_lp=0.5, q_good=0.3`。
  - path：`finetune_stf/exp/0601_1005_led_hb_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half378x672_trainall_valstride5n1000_seed42_bs8_acc1_e10`
  - LED-HB overall-best checkpoint：epoch 4。

数据源优先级：

1. LED formal training run 主结果：各 run 的 `best_val_metrics.json`。
2. 主表 baseline / delta 使用同一次 eval pass 内的 D0/D1/final，保持 same precision、same code path、same valid mask。
3. D0 standalone baseline：`plans/0531_led_night/led_hb_l0_d0_eval_0531_2339/d0_summary.json`，只作为 fp32 sanity/reference。
4. 后续统一 post-eval：`foundation/tools/eval_led_hb_formal.py` 输出的 `metrics.json` 和 `summarize_led_hb_formal.py` 汇总。
5. 只用于 smoke / probe / panel 的输出不进主表。

注意：当前 standalone D0 eval 的 overall D0 `abs_rel=0.55523`，C2/N-series run 内同一 val 口径记录的 D0 约 `0.55660`，差异主要来自 standalone fp32 vs train eval AMP/bf16 precision。为保证公平，主表使用各 run 自己 `best_val_metrics.json` 里的 same-pass D0/D1/final；standalone fp32 D0 只作为 sanity/reference。

## 1. 实验参数设置

### 1.1 字段速查

| 字段 | 说明 |
|---|---|
| `eff_bs` | `bs * accum_steps`。当前 C2 formal 为 `8 * 1 = 8`。 |
| `D0` | frozen DAv2-S 在 LED-HB halfres RGB 输入上的初始预测，经 `per_image_affine_disp_depth_anything_v2` 对齐后计算指标。 |
| `C2` | D0-only residual calibrator：frozen DAv2-S + trainable ResidualGateHead。 |
| `D1` | N-series 中 frozen C2 calibrator 输出；本批 N-series 都加载 `0531_2341.../best_abs_rel.pth`。 |
| `correction scale` | 主表中统一记录 `mean_abs_gate_delta`；C2 的该值接近 `mean_abs_final_minus_d0_norm`，N-series 表示实际施加项 `gate * delta_effective` 的平均幅度。 |
| `selected ckpt` | 以 LED-HB val overall `abs_rel` 最优 checkpoint 为准。 |

### 1.2 共同设置

| 项 | 值 |
|---|---|
| 默认 conda env | `dav3` |
| 项目根目录 | `/home/caq/6666_raw/dav2_raw_0522` |
| 数据集 | LED Nighttime Synthetic Drive Dataset, `illumination=HB` |
| train split | `finetune_stf/dataset/splits/led_hb/hb_train_all.txt`，14,997 frames |
| val split | `finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt`，1000 frames |
| train maps / val map | `china,herrenberg,ottosuhrallee` / `hamburg` |
| geometry | source `1080x1920` -> isotropic resize `756x1344` -> 2x2 halfres `378x672` |
| RGB / depth space | `rgb_input_space=resize_area_756x1344_then_2x2_area`；`depth_target_space=resize_nearest_756x1344_then_2x2_valid_mean` |
| depth label / unit | `distance_to_image_plane` / meter |
| depth range | `min_depth=1.0`，`max_depth=200.0` |
| eval protocol | `per_image_affine_disp_depth_anything_v2` |
| encoder / pretrained | `vits` / `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth` |
| optimizer / LR / wd | AdamW，`lr=1e-4`，`weight_decay=1e-4` |
| AMP / seed | bf16 AMP，`seed=42` |
| augmentation | `hflip_prob=0.5` |
| eval cadence | `eval_interval=1`，`save_interval=1`，`save_best_checkpoint=true` |

### 1.3 当前 run 参数矩阵

| Run | Method | input / interface | train scope | experiment-semantic params | epochs | bs/acc/eff | LR / wd | trainable params | notes |
|---|---|---|---|---|---:|---|---|---:|---|
| `led_hb_l0_d0_eval_0531_2339` | Frozen DAv2-S `D0` | `input_domain=rgb`；`model_input_tensor=image`；`raw_storage_format=not_applicable` | eval-only；frozen DAv2-S | `d0_sign=+1` selected；same LED-HB val split / geometry / depth range | n/a | eval only | n/a | 0 | standalone fp32 sanity/reference，不用于主表 delta |
| `0531_2341...c2_d0only...` | C2 D0-only residual | `input_domain=rgb`；`model_input_tensor=image`；`front_end=dav2_rgb_frozen`；`raw_storage_format=not_applicable`；`residual_feature_source=d0` | frozen DAv2-S；ResidualGateHead trainable | `residual_alpha=0.5`；`d0_sign=1`；old residual loss: `L_depth + 0.5*L_grad + 0.1*L_keep + 0.01*L_res + 0.005*L_gate + 0.05*L_gate_sup` | 20 | `8/1/8` | `1e-4 / 1e-4` | 2,881,858 | total params `27,666,947`；frozen `24,785,089`；best e19；结果可能还有提升空间 |
| `0601_0355...n5_d1...` | N5 D1-only extra head | `input_domain=rgb`；`model_input_tensor=image`；`front_end=c2_frozen_d1_incremental`；`raw_storage_format=not_applicable`；`incremental_feature_source=d1` | frozen DAv2-S + frozen C2；D1-only branch trainable | `delta_condition=d1_only`；`gate_condition=d1_only`；`lambda_lp=0.5`；`q_good=0.3`；`lambda_lowfreq_loss=0.0` | 10 | `8/1/8` | `1e-4 / 1e-4` | 260,482 | total params `27,927,429`；best e8 |
| `0601_0550...n3_rgb...` | N3 RGB incremental control | `input_domain=rgb`；`model_input_tensor=image`；`front_end=c2_frozen_rgb_incremental`；`raw_storage_format=not_applicable`；`incremental_feature_source=rgb` | frozen DAv2-S + frozen C2；RGB incremental branch trainable | `delta_condition=feature_only`；`gate_condition=feature_d1`；`lambda_lp=0.5`；`q_good=0.3`；`lambda_lowfreq_loss=0.0` | 10 | `8/1/8` | `1e-4 / 1e-4` | 279,490 | total params `27,946,437`；best e8 |
| `0601_0751...n2_x3...` | N2 x3 incremental | `input_domain=raw4`；`model_input_tensor=raw`；`front_end=c2_frozen_raw_ram_incremental`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`incremental_feature_source=x3` | frozen DAv2-S + frozen C2；RAW/RAM x3 branch trainable | `delta_condition=feature_only`；`gate_condition=feature_d1`；`lambda_lp=0.8`；`q_good=0.3`；`lambda_lowfreq_loss=0.0`；`raw_adapter_fixed_light_scale=1.0`；normal-only unprocessing | 10 | `8/1/8` | `1e-4 / 1e-4` | 420,393 | total params `28,087,340`；best e6 |
| `0601_1005...n7_x3...` | N7 x3 stop-gradient D1 delta | `input_domain=raw4`；`model_input_tensor=raw`；`front_end=c2_frozen_raw_ram_incremental`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`incremental_feature_source=x3` | frozen DAv2-S + frozen C2；RAW/RAM x3 branch trainable | `delta_condition=feature_d1_stopgrad`；`gate_condition=feature_d1`；`lambda_lp=0.5`；`q_good=0.3`；`lambda_lowfreq_loss=0.0`；`raw_adapter_fixed_light_scale=1.0`；normal-only unprocessing | 10 | `8/1/8` | `1e-4 / 1e-4` | 438,825 | total params `28,105,772`；best e4 |

0601 N-series 参数比对结论：四条 run 均使用同一个 C2 checkpoint `0531_2341.../best_abs_rel.pth`、同一个 LED-HB split / geometry / depth range；实际 config 与 formal queue 的 N5/N3/N2/N7 语义设置一致，未发现混用 split、checkpoint 或 input interface。

## 2. LED-HB hamburg val

### 2.1 Overall metrics

| Method | selected ckpt | source | abs_rel | d1 | d2 | d3 | rmse | silog_x100 | delta target | note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| Frozen DAv2-S `D0` | same pass as C2 e19 | C2 `best_val_metrics.json` | 0.55660 | 0.37099 | 0.66965 | 0.78354 | 26.734 | 64.90 | n/a | AMP/bf16 same-pass baseline；standalone fp32 reference is `0.55523` |
| C2 D0-only residual / `D1` | e19 | `best_val_metrics.json` | 0.45750 | 0.46079 | **0.69864** | **0.81474** | 22.768 | 55.96 | `-0.09910 vs same-pass D0` | same-pass D0 abs_rel 改善约 `17.80%` |
| N5 D1-only `lp0.5 q0.3` | e8 | `best_val_metrics.json` | 0.42353 | 0.47622 | 0.69659 | 0.81191 | 20.881 | 54.90 | `-0.03396 vs same-pass D1` | same-pass D0 abs_rel 改善约 `23.91%` |
| N3 RGB `lp0.5 q0.3` | e8 | `best_val_metrics.json` | 0.42070 | 0.47420 | 0.69377 | 0.81097 | 20.739 | 54.78 | `-0.03678 vs same-pass D1` | best RGB incremental；same-pass D0 改善约 `24.42%` |
| N2 x3 `lp0.8 q0.3` | e6 | `best_val_metrics.json` | 0.42223 | **0.47745** | 0.69634 | 0.81240 | 21.305 | 54.98 | `-0.03525 vs same-pass D1` | best `d1`；same-pass D0 改善约 `24.14%` |
| N7 x3 stopgrad `lp0.5 q0.3` | e4 | `best_val_metrics.json` | **0.41137** | 0.47382 | 0.69457 | 0.81079 | **20.109** | **54.29** | **`-0.04611 vs same-pass D1`** | best overall abs_rel / rmse / silog；same-pass D0 改善约 `26.09%` |

### 2.2 Region metrics at selected checkpoint

下表使用各 run 的 `best_val_metrics.json`，region delta 来自同一次 eval pass 和同一套 region mask。C2 的 target 是 run-internal `D0`；N-series 的 target 是各自 run 内的 frozen C2 `D1`。括号内为 `final - target`，负数表示 final 优于 target。加粗表示同列 performance metric 的最优值；`mean_gate` 和 `correction scale` 只是诊断量，不按好坏加粗。后续如果用 `eval_led_hb_formal.py` 统一 post-eval，应以 post-eval 输出重算并覆盖本节。

| Method | epoch | target | boundary | d0_high_error | d1_high_error | far50 | far100 | dark_q20 | saturated | mid_20_50 | mean_gate | correction scale |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 D0-only | 19 | D0 | 0.72692 (-0.09762) | 0.92413 (-0.15899) | n/a | 0.67011 (-0.09285) | **0.66024 (-0.03267)** | 0.47907 (-0.05514) | 0.31924 (-0.09071) | 0.58194 (-0.23985) | 0.47257 | 0.23443 |
| N5 D1-only | 8 | D1 | 0.64159 (-0.08556) | 0.81673 (-0.10739) | 0.90918 (-0.06537) | 0.65755 (-0.01263) | 0.70054 (+0.04022) | **0.46598 (-0.01307)** | 0.28249 (-0.03674) | 0.50049 (-0.08153) | 0.11635 | 0.04137 |
| N3 RGB | 8 | D1 | 0.62149 (-0.10567) | 0.79824 (-0.12585) | 0.90025 (-0.07427) | 0.65555 (-0.01462) | 0.69832 (+0.03802) | 0.47066 (-0.00838) | 0.26094 (-0.05831) | 0.49544 (-0.08656) | 0.10408 | 0.03540 |
| N2 x3 | 6 | D1 | 0.62916 (-0.09800) | 0.78415 (-0.13994) | **0.88216 (-0.09236)** | 0.66315 (-0.00703) | 0.69525 (+0.03495) | 0.47703 (-0.00201) | 0.29908 (-0.02017) | 0.51580 (-0.06620) | 0.09059 | 0.03445 |
| N7 x3 stopgrad | 4 | D1 | **0.60355 (-0.12361)** | **0.76285 (-0.16123)** | 0.88592 (-0.08860) | **0.65343 (-0.01675)** | 0.71623 (+0.05593) | 0.47123 (-0.00781) | **0.25002 (-0.06923)** | **0.45660 (-0.12540)** | 0.20463 | 0.06795 |

### 2.3 Training / eval trend snapshot

本表只对 eval `abs_rel` 相关指标加粗；train loss 和 epoch 不是跨方法可直接比较的性能指标。

| Method | train loss first -> last | eval e0 final abs_rel | best epoch | best abs_rel | last epoch abs_rel | note |
|---|---:|---:|---:|---:|---:|---|
| C2 D0-only | `0.43691 -> 0.36176` | 0.49602 | 19 | 0.45750 | 0.45750 | best at last epoch；可能还有提升空间 |
| N5 D1-only | `0.76350 -> 0.71823` | 0.43955 | 8 | 0.42353 | 0.43910 | last epoch 回落 |
| N3 RGB | `0.76131 -> 0.71272` | 0.43111 | 8 | 0.42070 | **0.42441** | RGB incremental 最好 |
| N2 x3 | `0.74740 -> 0.70474` | **0.42856** | 6 | 0.42223 | 0.42778 | best d1；overall 略弱于 N3 |
| N7 x3 stopgrad | `0.73604 -> 0.66622` | 0.43609 | 4 | **0.41137** | 0.42798 | overall / key regions 最好，后期震荡回落 |

观察：

- C2 明显优于 frozen D0，N-series 又明显优于 C2/D1：四条 N-series 相对 D1 的 overall `abs_rel` 改善在 `0.03396` 到 `0.04611` 之间。
- N7 x3 stopgrad 是当前最佳：overall `abs_rel=0.41137`，相对 same-pass D1 改善约 `10.08%`，相对 same-pass D0 改善约 `26.09%`；boundary、D0 high-error、saturated、mid-depth region 也最好。
- N3 RGB control 已经很强：`abs_rel=0.42070`，优于 N2 x3 `0.42223`，说明本轮 LED-HB 的整体收益不能简单归因于 RAW/x3 cue。
- N2 x3 的 `d1=0.47745` 是当前最高，且 `d1_high_error` region 最好；但 overall / boundary / saturated 不如 N7。
- N-series 在 `far100` 上相对 D1 都变差，N7 变差最大（`+0.05593`）。如果后续写结论，远距离极深区需要单独检查或报告为 tradeoff。
- 多个 run 的 best epoch 不在最后，eval 明显震荡；后续仍应按 overall-best checkpoint 记录，不用 last checkpoint。

## 3. 当前排序

| Rank | Method | best epoch | abs_rel | delta vs D1 | target_region_score | short read |
|---:|---|---:|---:|---:|---:|---|
| 1 | N7 x3 stopgrad `lp0.5 q0.3` | 4 | **0.41137** | **-0.04611** | **-0.06120** | 当前 LED-HB best overall 和 best region row |
| 2 | N3 RGB `lp0.5 q0.3` | 8 | 0.42070 | -0.03678 | -0.05225 | 最强 RGB incremental control |
| 3 | N2 x3 `lp0.8 q0.3` | 6 | 0.42223 | -0.03525 | -0.04391 | best d1 / d1-high-error，overall 略弱 |
| 4 | N5 D1-only `lp0.5 q0.3` | 8 | 0.42353 | -0.03396 | -0.04267 | D1-only extra head 也能明显改善 C2 |
| 5 | C2 D0-only / D1 | 19 | 0.45750 | n/a | n/a | N-series baseline |
| 6 | Frozen DAv2-S D0 | same pass as C2 e19 | 0.55660 | n/a | n/a | AMP/bf16 same-pass baseline；fp32 reference `0.55523` |
