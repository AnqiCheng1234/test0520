# VKITTI RGB / RAW residual baseline 公平性汇总

日期：2026-05-26

本文记录 `/home/caq/6666_raw/dav2_raw_0522` 当前 VKITTI residual formal validation 结果。

写作规约：指标表只放 formal eval 或同口径 eval log / json 中的数字；只用于 smoke 或可视化的 sample loss 不混入主指标表。

指标表标注规约：`abs_rel` 越低越好，`d1` 越高越好。`D0` 指 frozen DAv2-S 在同一 halfres RGB 输入上得到的初始深度，经同一 per-image affine disparity protocol 对齐后计算。

## 0. 协议速查 / 数据源规则

当前 formal runs：

- `0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_0203_vkitti_cseries_residual_controls.queue.log`
  - stage: `vkitti_residual_control`
  - input: `rgb`，`model_input_tensor=image`，`raw_storage_format=not_applicable`
  - path：halfres RGB `[3,187,621]` -> frozen DAv2-S -> `D0`；ResidualGateHead 输入为 `concat(D0_norm, image_rgb_norm)`，`residual_feature_source=rgb`
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_rgb_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 3；`latest.pth` 为 epoch 19
- `0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_0203_vkitti_cseries_residual_controls.queue.log`
  - stage: `vkitti_residual_control`
  - input: `rgb`，`model_input_tensor=image`，`raw_storage_format=not_applicable`
  - path：halfres RGB `[3,187,621]` -> frozen DAv2-S -> `D0`；ResidualGateHead 输入仅为 `D0_norm`，`residual_feature_source=d0`
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_rgb_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 11；`latest.pth` 为 epoch 19
- `0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_0204_vkitti_mseries_residual.queue.log`
  - stage: `vkitti_raw_residual`
  - input: `raw4`，`model_input_tensor=raw`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - path：halfres RGB -> Brooks 2019-style online unprocessing `sensor_linear_dual` -> packed RAW4 `[4,187,621]` -> RamCore3 / RAW front-end -> frozen DAv2-S；residual uses `ffm_mid`
  - known caveat：该 run 的 unprocessing 配置后续确认存在问题；`randomize_unprocessing=true` 且 `sensor_linear_dual` 每个样本随机抽到 `eth3d_sensor_linear` / `robotcar_subset100_sensor_linear` 两套参数之一，默认等权混合。该 run 指标保留用于追溯，但不应作为干净 RAW/RAM 公平对比的正证据。
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_raw_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 8；`latest.pth` 为 epoch 19
- `0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0525_1425_vkitti_mseries_rawadapter.queue.log`
  - label：formal M2-RA0 rawadapter analytic identity normal
  - input: `raw4`，`model_input_tensor=raw`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - path：halfres RGB -> RAW-Adapter-style analytic unprocessing -> packed RAW4 `[4,187,621]` -> RamCore3 / RAW front-end -> frozen DAv2-S；residual uses `ffm_mid`
  - unprocessing：`unprocessing_method=raw_adapter_style`，`raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_inverse_tone=global_0p15`，`raw_adapter_variant_policy=normal`，`randomize_unprocessing=false`，`vkitti_unprocessing_preset=not_applicable`；不使用 RobotCar/ETH3D dual preset 随机混合。
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_raw_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 9；`latest.pth` 为 epoch 19
- `0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0526_0039_vkitti_rawadapter_feature_d0_ablation.queue.log`
  - label：formal M1 x3 residual with D0_norm
  - input: `raw4`，`model_input_tensor=raw`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - path：halfres RGB -> RAW-Adapter-style analytic unprocessing -> packed RAW4 `[4,187,621]` -> RamCore3 / RAW front-end -> frozen DAv2-S；residual uses `x3` with `residual_head_d0_mode=concat`
  - unprocessing：`unprocessing_method=raw_adapter_style`，`raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_inverse_tone=global_0p15`，`raw_adapter_variant_policy=normal`，`randomize_unprocessing=false`，`vkitti_unprocessing_preset=not_applicable`。
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_raw_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 14；`latest.pth` 为 epoch 19
- `0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0526_0039_vkitti_rawadapter_feature_d0_ablation.queue.log`
  - label：formal ffm_mid-only residual head, no D0_norm input
  - input: `raw4`，`model_input_tensor=raw`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - path：halfres RGB -> RAW-Adapter-style analytic unprocessing -> packed RAW4 `[4,187,621]` -> RamCore3 / RAW front-end -> frozen DAv2-S；residual uses `ffm_mid` with `residual_head_d0_mode=none`
  - unprocessing：`unprocessing_method=raw_adapter_style`，`raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_inverse_tone=global_0p15`，`raw_adapter_variant_policy=normal`，`randomize_unprocessing=false`，`vkitti_unprocessing_preset=not_applicable`。
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_raw_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 6；`latest.pth` 为 epoch 19
- `0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20`
  - source log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0522/finetune_stf/logs/0526_0039_vkitti_rawadapter_feature_d0_ablation.queue.log`
  - label：formal x3-only residual head, no D0_norm input
  - input: `raw4`，`model_input_tensor=raw`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - path：halfres RGB -> RAW-Adapter-style analytic unprocessing -> packed RAW4 `[4,187,621]` -> RamCore3 / RAW front-end -> frozen DAv2-S；residual uses `x3` with `residual_head_d0_mode=none`
  - unprocessing：`unprocessing_method=raw_adapter_style`，`raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_inverse_tone=global_0p15`，`raw_adapter_variant_policy=normal`，`randomize_unprocessing=false`，`vkitti_unprocessing_preset=not_applicable`。
  - train/eval split：VKITTI train `11870` samples，VKITTI val `1000` samples，Scene20 holdout n1000
  - eval protocol：VKITTI `per_image_affine_disp_depth_anything_v2`；KITTI val `halfres_raw_canonical_even_pad_crop_affine_disp`，`652` samples，非 KITTI public benchmark 口径
  - checkpoint：`best_abs_rel.pth` 按 VKITTI val final `abs_rel` 保存，best epoch 为 epoch 14；`latest.pth` 为 epoch 19

baseline/control 行：

- Frozen DAv2-S `D0`：所有 run 都以同一 frozen DAv2-S checkpoint 产生 `D0`；在 VKITTI val 上约 `abs_rel=0.1531, d1=0.8184`，在 KITTI halfres canonical val 上约 `abs_rel=0.1184, d1=0.8665`。它不是单独训练 run，但作为 residual 改善幅度的对照。

run-row 数据源优先级：

1. `finetune_stf/exp/<run>/run_summary.json`、`best_val_metrics.json`、`best_kitti_val_metrics.json`。
2. `finetune_stf/exp/<run>/train.log`。
3. `finetune_stf/exp/<run>/config.json` 只用于确认配置。

## 1. VKITTI Scene20 holdout val

### 1.1 Overall metrics

| Experiment | Method / input | D0 abs_rel / d1 | final abs_rel best (epoch) | final abs_rel last | final d1 best (epoch) | final d1 last | best abs_rel delta vs D0 | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `0525_0203...c1_rgb_residual...` | C1 RGB residual control；`residual_feature_source=rgb` | 0.1531 / 0.8184 | 0.1257 (e3) | 0.1370 | 0.8515 (e3) | 0.8328 | -0.0274 | RGB residual 可以明显优于 D0，但 best 出现在早期；epoch 19 已回退。 |
| `0525_0203...c2_d0only_residual...` | C2 D0-only residual control；`residual_feature_source=d0` | 0.1531 / 0.8184 | **0.1210 (e11)** | 0.1213 | **0.8622 (e17)** | 0.8553 | **-0.0321** | 当前所有 run 里 VKITTI abs_rel 最好；只用 D0_norm 已能超过 RGB/RAW residual。 |
| `0525_0204...m2_ffm_mid_residual...` | M2 RAW/RAM residual；`raw4 + ffm_mid`；Brooks 2019-style unprocessing | 0.1531 / 0.8184 | 0.1262 (e8) | 0.1308 | 0.8541 (e9) | 0.8435 | -0.0269 | unprocessing caveat：该 run 随机混用 `eth3d_sensor_linear` / `robotcar_subset100_sensor_linear` 两套参数；指标只作追溯，不作为干净公平对比。 |
| `0525_1425...m2_ra0_rawadapter...` | M2-RA0 RAW-Adapter analytic identity normal；`raw4 + ffm_mid` | 0.1531 / 0.8184 | 0.1262 (e9) | 0.1328 | 0.8549 (e16) | 0.8427 | -0.0269 | 修正为 `raw_adapter_style` analytic / identity / normal，`randomize_unprocessing=false`，不混用 RobotCar/ETH3D；VKITTI best 接近 0525_0204，但仍未超过 C1/C2。 |
| `0526_0040...m1_ra0_x3_d0concat...` | M1-RA0 RAW-Adapter；`raw4 + x3 + D0_norm concat` | 0.1531 / 0.8184 | 0.1254 (e14) | 0.1360 | 0.8537 (e14) | 0.8362 | -0.0277 | clean RA0 feature/D0 ablation；x3 加 D0_norm 后略优于 0525_1425，接近 C1，但仍弱于 C2。 |
| `0526_0213...m2nod0_ra0_ffm_mid_only...` | M2-noD0 RA0；`raw4 + ffm_mid only` | 0.1531 / 0.8184 | 0.1428 (e6) | 0.1497 | 0.8271 (e7) | 0.8106 | -0.0103 | 去掉 D0_norm 后明显退化；ffm_mid-only 只能小幅改善 abs_rel，d1 不稳定。 |
| `0526_0344...m1nod0_ra0_x3_only...` | M1-noD0 RA0；`raw4 + x3 only` | 0.1531 / 0.8184 | 0.1441 (e14) | 0.1502 | 0.8300 (e2) | 0.8121 | -0.0090 | x3-only 同样明显弱于带 D0_norm 的 residual head；best d1 出现在早期。 |

### 1.2 Region metrics at best VKITTI abs_rel

`0525_0204` caveated M2 的 region metrics 同样受上述 Brooks 2019-style random dual unprocessing caveat 影响，仅保留为 run 追溯记录；`0525_1425` M2-RA0 不受该 dual-preset caveat 影响。

当前生效口径：region mask 定义保持不变；GT valid mask 使用 VKITTI depth range `[1.0, 80.0]`；per-image affine disparity 对齐后，`aligned D0` 和 `aligned final` 都先 clip 到 `[1.0, 80.0]` 再计算 region `abs_rel`，与 1.1 overall `compute_metrics` 口径一致。重算结果来自 `plans/result/0527_vkitti_region_clip_recalc_section_1_2_all.json`，每行均为 1000 个 Scene20 holdout val samples。D0 consistency check：7 个 run 的 D0 overall 完全一致，`abs_rel=0.1531`、`d1=0.8184`；下表第一行给出同一 D0 的 clipped region 指标。

| Experiment | epoch | boundary abs_rel | DAv2 high-error abs_rel | far50 abs_rel | dark abs_rel | saturated abs_rel | mean_gate | mean_abs_delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen DAv2-S `D0` baseline | n/a | 0.3497 | 0.3484 | 0.2481 | 0.1317 | 0.2333 | n/a | n/a |
| `0525_0203...c1_rgb_residual...` | 3 | 0.2689 | 0.2306 | 0.2436 | 0.1049 | 0.1616 | 0.3213 | 0.4671 |
| `0525_0203...c2_d0only_residual...` | 11 | 0.2692 | 0.2198 | 0.2734 | 0.1026 | 0.1357 | 0.3358 | 0.4537 |
| `0525_0204...m2_ffm_mid_residual...` | 8 | 0.2653 | 0.2262 | 0.2345 | 0.1080 | 0.1308 | 0.3122 | 0.4665 |
| `0525_1425...m2_ra0_rawadapter...` | 9 | 0.2714 | 0.2316 | 0.2254 | 0.1080 | 0.1469 | 0.3175 | 0.4707 |
| `0526_0040...m1_ra0_x3_d0concat...` | 14 | 0.2592 | 0.2252 | 0.2547 | 0.1038 | 0.1413 | 0.3233 | 0.4698 |
| `0526_0213...m2nod0_ra0_ffm_mid_only...` | 6 | 0.3064 | 0.2764 | 0.2445 | 0.1297 | 0.1966 | 0.2473 | 0.4365 |
| `0526_0344...m1nod0_ra0_x3_only...` | 14 | 0.3156 | 0.2920 | 0.2470 | 0.1239 | 0.1891 | 0.2335 | 0.4445 |

旧 1.2 表作废：这些数值没有统一纳入 D0 baseline / D0 consistency check，且 region `abs_rel` 使用未 clip 的 aligned prediction，和 1.1 overall 指标口径不一致。

注：`boundary abs_rel` 的 boundary 区域按每张样本 GT depth 梯度幅值 `sqrt(dx^2 + dy^2)` 在 valid pixels 中取 top 10% 得到；`DAv2 high-error abs_rel` 的 high-error 区域按每张样本 normalized inverse-depth 空间里的 `abs(D0_norm - y_norm)` 在 valid pixels 中取 top 20% 得到；`far50 abs_rel` 的 far50 区域为 GT depth `> 50m` 的 valid pixels；`dark abs_rel` 的 dark 区域为 RGB preview luma `< 0.15` 的 valid pixels；`saturated abs_rel` 的 saturated 区域为 RGB preview `max(R,G,B) > 0.95` 的 valid pixels。上述生效 region `abs_rel` 均在对应 mask 内对 per-image affine disparity 对齐后的 depth 先执行 `[1.0, 80.0]` clip，再计算 `mean(abs(pred - gt) / gt)`，最后对样本平均。`mean_gate` 为 residual gate 在 valid pixels 上的平均开启程度；`mean_abs_delta` 为 residual head 候选修正量 `delta` 在 valid pixels 上的平均绝对值，统计空间为 normalized inverse-depth / `D0_norm` 空间。

## 2. KITTI val halfres canonical eval

KITTI val 结果用于跨域 sanity check。该口径为了匹配 VKITTI 训练得到的固定 `187x621` residual model，使用 canonical even pad/crop；不是 KITTI public benchmark setting。

| Experiment | KITTI protocol | D0 abs_rel / d1 | final abs_rel best (epoch) | final abs_rel last | final d1 best (epoch) | final d1 last | best abs_rel delta vs D0 | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `0525_0203...c1_rgb_residual...` | `halfres_rgb_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8665 | 0.0984 (e6) | 0.0999 | 0.8962 (e17) | 0.8937 | -0.0200 | RGB control 在 KITTI 上也稳定优于 D0。 |
| `0525_0203...c2_d0only_residual...` | `halfres_rgb_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8665 | **0.0950 (e6)** | 0.0960 | 0.9000 (e10) | 0.8969 | **-0.0234** | 当前所有 run 里 KITTI abs_rel 最好。 |
| `0525_0204...m2_ffm_mid_residual...` | `halfres_raw_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8665 | 0.0956 (e12) | 0.0966 | **0.9006 (e15)** | 0.8972 | -0.0228 | unprocessing caveat 同上：Brooks 2019-style `sensor_linear_dual` 随机抽 `eth3d` / `robotcar` 参数；不作为干净跨域公平对比。 |
| `0525_1425...m2_ra0_rawadapter...` | `halfres_raw_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8665 | 0.0955 (e12) | 0.0969 | 0.8995 (e13) | 0.8983 | -0.0229 | clean RA0 RAW-Adapter 设置；KITTI abs_rel 接近 caveated M2，但仍略弱于 C2。 |
| `0526_0040...m1_ra0_x3_d0concat...` | `halfres_raw_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8665 | 0.0955 (e9) | 0.0970 | 0.9004 (e6) | 0.8999 | -0.0229 | KITTI abs_rel 与 clean RA0 M2 基本持平，d1 接近当前最高；VKITTI 仍未超过 C2。 |
| `0526_0213...m2nod0_ra0_ffm_mid_only...` | `halfres_raw_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8664 | 0.1109 (e1) | 0.1140 | 0.8761 (e1) | 0.8723 | -0.0075 | no-D0 的跨域收益很弱，明显落后于带 D0_norm 的 RAW/RGB residual。 |
| `0526_0344...m1nod0_ra0_x3_only...` | `halfres_raw_canonical_even_pad_crop_affine_disp` | 0.1184 / 0.8664 | 0.1106 (e1) | 0.1149 | 0.8788 (e0) | 0.8710 | -0.0078 | x3-only no-D0 与 ffm_mid-only no-D0 接近，仍远弱于 0526_0040。 |

## 3. 当前结论

- C1/C2 两条 control run 显著优于 frozen DAv2-S `D0`，说明 residual formulation 本身有效；clean RA0 系列中 `0526_0040` 的 `x3 + D0_norm concat` 最强，但仍没有超过 C2。
- `0525_0204` M2 受 Brooks 2019-style dual unprocessing 问题影响，只能作为 caveated run 记录；`0525_1425` 是修正后的 RAW-Adapter analytic / identity / normal 对照，不再随机混用 RobotCar/ETH3D 参数。
- 在 VKITTI Scene20 holdout 上，C2 D0-only residual 是当前最强控制项：`0.1210` abs_rel，优于 `0526_0040` M1-RA0 x3+D0 `0.1254`、C1 `0.1257`、clean M2-RA0 `0.1262` 和 caveated M2 `0.1262`。
- 在 KITTI halfres canonical eval 上，C2 仍略优：`0.0950` abs_rel；`0526_0040` 和 clean M2-RA0 均约 `0.0955`，caveated M2 为 `0.0956`。
- `0526_0213` / `0526_0344` no-D0 ablation 明确显示：仅用 `ffm_mid` 或 `x3` 的 RAW/RAM feature head 明显弱于带 D0_norm 的 residual head。当前 clean RA0 仍不能作为“RAW-like cue 带来额外收益”的正证据。
- C1、caveated M2、clean RA0 以及 0526 ablation 的 last epoch 多数弱于 best epoch；后续比较应优先使用 `best_abs_rel.pth` 对应的 epoch 指标，而不是 epoch 19 的 latest 指标。
