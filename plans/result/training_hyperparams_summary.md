# 训练超参数汇总

日期：2026-05-26

本文记录 `/home/caq/6666_raw/dav2_raw_0522` 当前 VKITTI residual formal runs 的训练设置。结果数值见 [`rgb_raw_baseline_fairness_summary.md`](rgb_raw_baseline_fairness_summary.md)。

## 字段速查

| 字段 | 说明 |
|---|---|
| `eff_bs` | `bs * accum_steps`。 |
| `D0` | frozen DAv2-S 在 halfres RGB 输入上的初始预测；训练和评估中作为 residual base。 |
| `C1` | RGB residual control：ResidualGateHead 输入 `concat(D0_norm, image_rgb_norm)`；不使用 RAW / RamCore3 / `x3` / `ffm_mid`。 |
| `C2` | D0-only residual control：ResidualGateHead 输入仅为 `D0_norm`；用于判断是否只是 D0 post-processing。 |
| `M1` | RAW/RAM residual：`raw4 -> raw_to_base_rgb_ram3`，residual feature 使用 `x3`。 |
| `M2` | RAW/RAM residual：`raw4 -> raw_to_base_rgb_ram3`，residual feature 使用 `ffm_mid`。 |
| `M2-RA0` | M2 的 RAW-Adapter-style analytic identity normal 对照：不使用 Brooks 2019-style dual preset，不随机混用 RobotCar/ETH3D 参数。 |
| `residual_head_d0_mode=concat` | ResidualGateHead 输入显式拼接 `D0_norm`，即 `concat(D0_norm, feature)`；旧 M-series 默认语义。 |
| `residual_head_d0_mode=none` | ResidualGateHead 输入不拼接 `D0_norm`，只输入 RAW/RAM feature；D0 仍作为 residual base 和评估对照。 |
| `raw_adapter_style` | RGB -> synthetic RAW 的 analytic unprocessing 路线；`0525_1425` 使用 `raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_variant_policy=normal`。 |
| Brooks 2019-style unprocessing | RGB -> synthetic RAW 的 online unprocessing 路线；`0525_0204` 使用该路线，但后续确认 unprocessing 设定有问题，需要在结果表中显式 caveat。 |
| `synthetic_packed_bayer_4ch_halfres` | VKITTI halfres synthetic packed Bayer 4ch，shape `[4,187,621]`。 |
| `not_applicable` | 控制实验中 RAW storage 不适用；不能用 active RAW value 代替。 |
| `residual_alpha=0.5` | 模型残差幅度缩放参数。 |
| `d0_sign=1` | D0 inverse-depth 方向检查得到 / 显式使用的符号。 |
| `per_image_affine_disp_depth_anything_v2` | VKITTI val 主评估口径，per-image affine disparity 对齐后算 depth metrics。 |
| `halfres_*_canonical_even_pad_crop_affine_disp` | KITTI val sanity-check 口径；为匹配 `187x621` fixed residual model，不是 KITTI public benchmark setting。 |
| `best_abs_rel.pth` | 按 VKITTI val final `abs_rel` 最低保存的 checkpoint。 |

## 当前共同设置

| 项 | 值 |
|---|---|
| 默认 conda env | `dav3` |
| 项目根目录 | `/home/caq/6666_raw/dav2_raw_0522` |
| C1/C2 代码入口 | `foundation/tools/train_vkitti2_residual_control.py` |
| RAW/M-series 代码入口 | `foundation/tools/train_vkitti2_raw_residual.py` |
| encoder / checkpoint | `vits` / `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth` |
| DAv2 scope | frozen + eval + no_grad；只训练 residual / RAW-RAM front-end 侧参数 |
| VKITTI split | train：`finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt`；val：`finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt` |
| split size | train `11870`，VKITTI val `1000`，Scene20 holdout n1000；`missing_rgb=0`，`missing_depth=0`，`overlap_count=0` |
| geometry | original `375x1242` -> crop bottom row -> even fullres `374x1242` -> model input `187x621` |
| RGB / depth space | `rgb_input_space=halfres_2x2_area`，`depth_target_space=halfres_2x2_valid_mean` |
| depth range | `min_depth=1.0`，`max_depth=80.0` |
| optimizer / LR | AdamW，`lr=1e-4`，`weight_decay=1e-4`，betas `(0.9,0.999)` |
| AMP / seed | `bf16 AMP`，`seed=42` |
| loss | `L_depth + 0.5*L_grad + 0.1*L_keep + 0.01*L_res + 0.005*L_gate + 0.05*L_gate_sup` |
| eval cadence | `eval_interval=1`，`save_interval=1`，`save_best_checkpoint=true` |
| heavy artifact root | `/mnt/drive/3333_raw/0000_exp_ckpt/<run>/` |

## 当前 formal runs

| Run | Status | Training data | input / RAM / interface | DAv2 train scope | loss | epochs | bs/acc/eff | LR | trainable params | notes |
|---|---|---|---|---|---|---:|---|---|---:|---|
| `0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0525_0203_vkitti_cseries_residual_controls` | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=rgb`；`model_input_tensor=image`；`front_end=dav2_rgb_frozen`；`raw_storage_format=not_applicable`；`residual_feature_source=rgb` | frozen DAv2-S；ResidualGateHead trainable | residual loss, metric-depth target normalized internally | 20 | `8/1/8` | `1e-4` | 2,883,586 | total params `27,668,675`；frozen `24,785,089`；best VKITTI checkpoint epoch 3；best KITTI epoch 6 |
| `0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0525_0203_vkitti_cseries_residual_controls` | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=rgb`；`model_input_tensor=image`；`front_end=dav2_rgb_frozen`；`raw_storage_format=not_applicable`；`residual_feature_source=d0` | frozen DAv2-S；ResidualGateHead trainable | same residual loss | 20 | `8/1/8` | `1e-4` | 2,881,858 | total params `27,666,947`；frozen `24,785,089`；best VKITTI checkpoint epoch 11；best KITTI epoch 6 |
| `0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0525_0204_vkitti_mseries_residual`；unprocessing caveat | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=raw4`；`model_input_tensor=raw`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`front_end=raw_to_base_rgb_ram3`；Brooks 2019-style `vkitti_unprocessing_preset=sensor_linear_dual`；`randomize_unprocessing=true`；sub-presets `eth3d_sensor_linear` / `robotcar_subset100_sensor_linear` 等权随机；`residual_feature_source=ffm_mid` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | same residual loss | 20 | `8/1/8` | `1e-4` | 3,059,625 | caveat：该 run 的 unprocessing 设定后续确认有问题，指标只作追溯，不作为干净 RAW/RAM 公平对比；total params `27,844,714`；frozen `24,785,089`；best VKITTI checkpoint epoch 8；best KITTI epoch 12 |
| `0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0525_1425_vkitti_mseries_rawadapter`；clean RA0 | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=raw4`；`model_input_tensor=raw`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`front_end=raw_to_base_rgb_ram3`；`unprocessing_method=raw_adapter_style`；`raw_adapter_backend=analytic`；`raw_adapter_ccm=identity`；`raw_adapter_inverse_tone=global_0p15`；`raw_adapter_variant_policy=normal`；`randomize_unprocessing=false`；`vkitti_unprocessing_preset=not_applicable`；`residual_feature_source=ffm_mid` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | same residual loss | 20 | `8/1/8` | `1e-4` | 3,059,625 | total params `27,844,714`；frozen `24,785,089`；best VKITTI checkpoint epoch 9；best KITTI epoch 12；best VKITTI d1 epoch 16；best KITTI d1 epoch 13 |
| `0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0526_0039_vkitti_rawadapter_feature_d0_ablation`；clean RA0 | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=raw4`；`model_input_tensor=raw`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`front_end=raw_to_base_rgb_ram3`；`unprocessing_method=raw_adapter_style`；`raw_adapter_backend=analytic`；`raw_adapter_ccm=identity`；`raw_adapter_inverse_tone=global_0p15`；`raw_adapter_variant_policy=normal`；`randomize_unprocessing=false`；`vkitti_unprocessing_preset=not_applicable`；`residual_feature_source=x3`；`residual_head_d0_mode=concat` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | same residual loss | 20 | `8/1/8` | `1e-4` | 3,024,489 | total params `27,809,578`；frozen `24,785,089`；best VKITTI checkpoint epoch 14；best KITTI epoch 9 |
| `0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0526_0039_vkitti_rawadapter_feature_d0_ablation`；clean RA0 | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=raw4`；`model_input_tensor=raw`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`front_end=raw_to_base_rgb_ram3`；`unprocessing_method=raw_adapter_style`；`raw_adapter_backend=analytic`；`raw_adapter_ccm=identity`；`raw_adapter_inverse_tone=global_0p15`；`raw_adapter_variant_policy=normal`；`randomize_unprocessing=false`；`vkitti_unprocessing_preset=not_applicable`；`residual_feature_source=ffm_mid`；`residual_head_d0_mode=none` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | same residual loss | 20 | `8/1/8` | `1e-4` | 3,059,049 | total params `27,844,138`；frozen `24,785,089`；best VKITTI checkpoint epoch 6；best KITTI epoch 1 |
| `0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20` | completed；tmux `0526_0039_vkitti_rawadapter_feature_d0_ablation`；clean RA0 | VKITTI Scene20 holdout；train `11870`，val `1000`；KITTI val `652` | `input_domain=raw4`；`model_input_tensor=raw`；`raw_storage_format=synthetic_packed_bayer_4ch_halfres`；`front_end=raw_to_base_rgb_ram3`；`unprocessing_method=raw_adapter_style`；`raw_adapter_backend=analytic`；`raw_adapter_ccm=identity`；`raw_adapter_inverse_tone=global_0p15`；`raw_adapter_variant_policy=normal`；`randomize_unprocessing=false`；`vkitti_unprocessing_preset=not_applicable`；`residual_feature_source=x3`；`residual_head_d0_mode=none` | frozen DAv2-S；RAW/RAM front-end + ResidualGateHead trainable | same residual loss | 20 | `8/1/8` | `1e-4` | 3,023,913 | total params `27,809,002`；frozen `24,785,089`；best VKITTI checkpoint epoch 14；best KITTI epoch 1 |

## Eval 设置记录

| Split | Enabled | Backend / samples | Depth range | Notes |
|---|---|---|---|---|
| VKITTI Scene20 holdout | yes | `1000` samples | `1-80m` | 主指标；protocol `per_image_affine_disp_depth_anything_v2`；`best_abs_rel.pth` 按该 split 的 final abs_rel 保存。 |
| KITTI val | yes | `652` samples | `1-80m` | sanity-check；C1/C2 使用 `halfres_rgb_canonical_even_pad_crop_affine_disp`，RAW/M-series 使用 `halfres_raw_canonical_even_pad_crop_affine_disp`；不是 public benchmark setting。 |

## 训练过程摘录

### `0525_0203...c1_rgb_residual...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.28055 | 0.13744 | 0.83029 | 0.10254 | 0.88923 | 0.23801 | 1484 | 00:06:06 |
| 3 | 0.24027 | **0.12574** | **0.85146** | 0.09915 | 0.89164 | 0.32127 | 1484 | 00:06:11 |
| 6 | 0.23072 | 0.13234 | 0.83634 | **0.09843** | 0.89606 | 0.33568 | 1484 | 00:06:49 |
| 19 | 0.21237 | 0.13702 | 0.83280 | 0.09989 | 0.89371 | 0.36252 | 1484 | 00:06:36 |

运行摘录：

- setup: `experiment_id=C1`，`input_domain=rgb`，`model_input_tensor=image`，`front_end=dav2_rgb_frozen`，`residual_feature_source=rgb`，`d0_sign=1`。
- batch shape: `image=(8,3,187,621)`，`depth=(8,187,621)`，`valid=(8,187,621)`。
- model: total params `27,668,675`，trainable `2,883,586`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.12574`，epoch `3`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

### `0525_0203...c2_d0only_residual...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.28461 | 0.12862 | 0.84874 | 0.09819 | 0.89517 | 0.22614 | 1484 | 00:07:19 |
| 6 | 0.24080 | 0.12399 | 0.85249 | **0.09503** | 0.89939 | 0.31655 | 1484 | 00:03:21 |
| 11 | 0.23090 | **0.12100** | 0.85578 | 0.09648 | 0.89602 | 0.33200 | 1484 | 00:03:21 |
| 19 | 0.22182 | 0.12134 | 0.85531 | 0.09597 | 0.89690 | 0.34469 | 1484 | 00:03:21 |

运行摘录：

- setup: `experiment_id=C2`，`input_domain=rgb`，`model_input_tensor=image`，`front_end=dav2_rgb_frozen`，`residual_feature_source=d0`，`d0_sign=1`。
- batch shape: `image=(8,3,187,621)`，`depth=(8,187,621)`，`valid=(8,187,621)`。
- model: total params `27,666,947`，trainable `2,881,858`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.12100`，epoch `11`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

### `0525_0204...m2_ffm_mid_residual...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.28662 | 0.13755 | 0.83872 | 0.10071 | 0.89258 | 0.22817 | 1484 | 00:07:11 |
| 8 | 0.23492 | **0.12621** | 0.85069 | 0.09619 | 0.89890 | 0.32831 | 1484 | 00:07:37 |
| 12 | 0.22879 | 0.12778 | 0.84919 | **0.09558** | 0.89861 | 0.33806 | 1484 | 00:07:42 |
| 19 | 0.22050 | 0.13076 | 0.84346 | 0.09660 | 0.89719 | 0.35002 | 1484 | 00:06:58 |

运行摘录：

- setup: `input_domain=raw4`，`model_input_tensor=raw`，`front_end=raw_to_base_rgb_ram3`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`，`residual_feature_source=ffm_mid`，`d0_sign=1`。
- RAW/data path: Brooks 2019-style online unprocessing；`vkitti_unprocessing_preset=sensor_linear_dual`，`randomize_unprocessing=true`，sub-presets `eth3d_sensor_linear` 和 `robotcar_subset100_sensor_linear` 等权随机混合。
- caveat: 该 unprocessing 设定后续确认有问题；本 run 的指标需要在结果表中显式标注，不应作为干净 RAW/RAM 公平对比或“RAW-like cue 带来额外收益”的正证据。
- batch shape: `image=(8,3,187,621)`，`raw=(8,4,187,621)`，`depth=(8,187,621)`，`valid=(8,187,621)`。
- model: total params `27,844,714`，trainable `3,059,625`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.12621`，epoch `8`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

### `0525_1425...m2_ra0_rawadapter...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.28292 | 0.13400 | 0.84293 | 0.09759 | 0.89570 | 0.27795 | 1484 | 00:04:19 |
| 9 | 0.22728 | **0.12622** | 0.85360 | 0.09699 | 0.89770 | 0.31744 | 1484 | 00:03:48 |
| 12 | 0.22280 | 0.12797 | 0.84798 | **0.09554** | 0.89778 | 0.32394 | 1484 | 00:03:48 |
| 16 | 0.21781 | 0.12747 | **0.85486** | 0.09705 | 0.89659 | 0.29600 | 1484 | 00:03:48 |
| 19 | 0.21472 | 0.13285 | 0.84271 | 0.09688 | 0.89833 | 0.33284 | 1484 | 00:03:48 |

运行摘录：

- setup: formal M2-RA0 rawadapter analytic identity normal；`input_domain=raw4`，`model_input_tensor=raw`，`front_end=raw_to_base_rgb_ram3`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`，`residual_feature_source=ffm_mid`，`d0_sign=1`。
- RAW/data path: `unprocessing_method=raw_adapter_style`，`raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_inverse_tone=global_0p15`，`raw_adapter_rgb_transfer=srgb_piecewise`，`raw_adapter_cfa_pattern=RGGB`，`raw_adapter_packed_channel_order=R_Gr_Gb_B`。
- RAW-Adapter fixed values: `randomize_unprocessing=false`，`vkitti_unprocessing_preset=not_applicable`，`fixed_red_gain=2.15`，`fixed_blue_gain=1.7`，`fixed_light_scale=1.0`，`variant_policy=normal`，`noise_model=none`，config hash `5fd8f0d2345f9683`。
- batch shape: `image=(8,3,187,621)`，`raw=(8,4,187,621)`，`depth=(8,187,621)`，`valid=(8,187,621)`。
- model: total params `27,844,714`，trainable `3,059,625`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.12622`，epoch `9`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

### `0526_0040...m1_ra0_x3_d0concat...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.28205 | 0.13427 | 0.83851 | 0.10226 | 0.88595 | 0.25006 | 1484 | 00:04:21 |
| 9 | 0.22761 | 0.13367 | 0.84154 | **0.09551** | **0.89996** | 0.30696 | 1484 | 00:03:47 |
| 14 | 0.22071 | **0.12541** | **0.85373** | 0.09575 | 0.89610 | 0.32336 | 1484 | 00:03:47 |
| 19 | 0.21536 | 0.13604 | 0.83617 | 0.09701 | 0.89986 | 0.36590 | 1484 | 00:03:47 |

运行摘录：

- setup: formal M1-RA0 rawadapter analytic identity normal；`input_domain=raw4`，`model_input_tensor=raw`，`front_end=raw_to_base_rgb_ram3`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`，`residual_feature_source=x3`，`residual_head_d0_mode=concat`，`d0_sign=1`。
- RAW/data path: 与 `0525_1425` clean RA0 相同；`unprocessing_method=raw_adapter_style`，`raw_adapter_backend=analytic`，`raw_adapter_ccm=identity`，`raw_adapter_inverse_tone=global_0p15`，`variant_policy=normal`，config hash `5fd8f0d2345f9683`。
- model: total params `27,809,578`，trainable `3,024,489`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.12541`，epoch `14`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

### `0526_0213...m2nod0_ra0_ffm_mid_only...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.30825 | 0.15057 | 0.82051 | 0.11463 | 0.87148 | 0.17098 | 1484 | 00:04:14 |
| 1 | 0.28933 | 0.14670 | **0.82702** | **0.11092** | **0.87605** | 0.16378 | 1484 | 00:03:44 |
| 6 | 0.25880 | **0.14279** | 0.82597 | 0.11699 | 0.86434 | 0.24730 | 1484 | 00:03:44 |
| 19 | 0.23716 | 0.14974 | 0.81059 | 0.11395 | 0.87230 | 0.31602 | 1484 | 00:03:44 |

运行摘录：

- setup: formal ffm_mid-only residual head；`input_domain=raw4`，`model_input_tensor=raw`，`front_end=raw_to_base_rgb_ram3`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`，`residual_feature_source=ffm_mid`，`residual_head_d0_mode=none`，`d0_sign=1`。
- 语义：ResidualGateHead 输入只包含 `ffm_mid`，不拼接 `D0_norm`；D0 仍作为 `D_final = D0_norm + gate * delta` 的 residual base 和评估对照。
- RAW/data path: 与 `0525_1425` clean RA0 相同；config hash `5fd8f0d2345f9683`。
- model: total params `27,844,138`，trainable `3,059,049`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.14279`，epoch `6`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

### `0526_0344...m1nod0_ra0_x3_only...`

| epoch | avg_loss | VKITTI abs_rel | VKITTI d1 | KITTI abs_rel | KITTI d1 | mean_gate | used steps | elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.30985 | 0.15218 | 0.81917 | 0.11065 | **0.87883** | 0.17993 | 1484 | 00:04:21 |
| 1 | 0.29315 | 0.14575 | **0.82817** | **0.11058** | 0.87854 | 0.18278 | 1484 | 00:03:46 |
| 14 | 0.24528 | **0.14408** | 0.82541 | 0.11469 | 0.87301 | 0.23351 | 1484 | 00:03:46 |
| 19 | 0.23933 | 0.15016 | 0.81213 | 0.11488 | 0.87098 | 0.30385 | 1484 | 00:03:46 |

运行摘录：

- setup: formal x3-only residual head；`input_domain=raw4`，`model_input_tensor=raw`，`front_end=raw_to_base_rgb_ram3`，`raw_storage_format=synthetic_packed_bayer_4ch_halfres`，`residual_feature_source=x3`，`residual_head_d0_mode=none`，`d0_sign=1`。
- 语义：ResidualGateHead 输入只包含 `x3`，不拼接 `D0_norm`；D0 仍作为 residual base 和评估对照。
- RAW/data path: 与 `0525_1425` clean RA0 相同；config hash `5fd8f0d2345f9683`。
- model: total params `27,809,002`，trainable `3,023,913`，frozen `24,785,089`。
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth`，VKITTI final abs_rel `0.14408`，epoch `14`。
- latest checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/latest.pth`，epoch `19`。
- light artifacts: `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/`。

## Queue / smoke 记录

| Queue | Session | Log | Notes |
|---|---|---|---|
| C-series | `0525_0203_vkitti_cseries_residual_controls` | `finetune_stf/logs/0525_0203_vkitti_cseries_residual_controls.queue.log` | `SKIP_M2_GATE=1`，显式 `D0_SIGN=1`，依次跑 C1/C2，最终 `status=0`。 |
| M-series | `0525_0204_vkitti_mseries_residual` | `finetune_stf/logs/0525_0204_vkitti_mseries_residual.queue.log` | formal M2 前跑 smoke 与 D0 sign check；sign check 推荐 `D0_SIGN=1`；successful smoke artifacts 已删除，最终 `status=0`。 |
| RawAdapter M-series | `0525_1425_vkitti_mseries_rawadapter` | `finetune_stf/logs/0525_1425_vkitti_mseries_rawadapter.queue.log` | formal M2-RA0 前跑 transform parity 与 dataset/training smoke；successful smoke artifacts `plans/0524_unprocessing/codex_smoke_0525_rawadapter_queue_0525_1425` 已删除，最终 `status=0`。 |
| RawAdapter feature/D0 ablation | `0526_0039_vkitti_rawadapter_feature_d0_ablation` | `finetune_stf/logs/0526_0039_vkitti_rawadapter_feature_d0_ablation.queue.log` | 依次跑 `x3 + D0_norm`、`ffm_mid only`、`x3 only`；每个正式 run 使用真实启动时间戳；successful smoke artifacts `plans/0524_unprocessing/codex_smoke_0526_feature_d0_ablation_queue_0526_0039` 已删除，最终 `status=0`。 |
