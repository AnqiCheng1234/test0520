# 训练超参数汇总

日期：2026-05-21

本文记录 `/home/caq/6666_raw/dav2_raw_0520` 当前 formal runs 的训练设置。结果数值见 [`rgb_raw_baseline_fairness_summary.md`](rgb_raw_baseline_fairness_summary.md)。

## 字段速查

| 字段 | 说明 |
|---|---|
| `eff_bs` | `bs * accum_steps`. |
| `raw_ram_rgb` | packed RAW 先合成为 3ch `[R,(Gr+Gb)/2,B]`，再经过 RamCore3 接 DAv2。 |
| `raw_ram_rgb_tail=identity` | 当前 BN-clean path 保留 RamCore3 BN output；实际 DAv2 输入以 model log 中 `dav2_input=...` 为准。 |
| `dav2_train_mode=none` | DAv2 冻住，只训练 RAW/RAM front-end 参数。 |
| `stf_only` | STF-only training/eval path，使用 STF pseudo depth targets。 |
| `dav2_pseudo` | 训练 target 来自 DAv2 pseudo labels，而不是 sparse GT depth。 |
| `ssi` | Scale-shift invariant loss，带 gradient regularization 和 target normalization。 |

## 当前共同设置

| 项 | 值 |
|---|---|
| 代码入口 | `finetune_stf/train.py` |
| 默认 conda env | `dav3` |
| 项目根目录 | `/home/caq/6666_raw/dav2_raw_0520` |
| encoder / checkpoint | `vits` / `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth` |
| sensor input size | `512x960` |
| backbone padded size | `518x966` |
| STF depth train/eval range | `min_depth=1.0`，`max_depth=80.0` |
| Loss | `ssi`, `lambda_grad=2.0`, `loss_grad_scales=4`, `loss_mask_downsample=strict`, `loss_target_normalization=true`, `loss_norm_min_scale=1e-3` |
| optimizer / LR | train code 中 AdamW；base `lr=1e-5`；optimizer step 后 poly decay |
| AMP / seed | `bf16 AMP`, `seed=42` |
| heavy artifact root | `/mnt/drive/3333_raw/0000_exp_ckpt/<run>/` |

## 预留 baseline/control 行

| 预留行 | 状态 | 目标协议 |
|---|---|---|
| DAv2-S RGB 直推 | 待补 | Official RGB DAv2-S checkpoint，不训练，STF RGB input direct eval。 |
| DAv2-S RAW-preview 直推 | 待补 | Official DAv2-S checkpoint，不训练，STF RAW preview / pseudo-RGB direct eval。 |
| RGB input, same STF setting | 待补 | 与 `0521_0012...` 使用同一 train/eval protocol，但 `input_type=rgb`，作为 fair RGB-input control。 |

## 当前 formal runs

| Run | Status | Training data | input / RAM / interface | DAv2 train scope | loss | epochs | bs/acc/eff | LR | trainable params | notes |
|---|---|---|---|---|---|---:|---|---|---:|---|
| `0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5` | completed | STF train+test pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；model log `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`；functions `wb,ccm,gamma,brightness` | `none` | `ssi`, target norm, `lambda_grad=2.0` | 5 | `8/1/8` | base `1e-5` | 140,903 | total params `24,925,992`；frozen `24,785,089`；best STF checkpoint epoch 4；current/last checkpoint epoch 4 |

## Eval 设置记录

| Split | Enabled | Backend / samples | Depth range | Notes |
|---|---|---|---|---|
| STF val | yes | `sparse`, `808` samples | `1-80m` | train-time per-epoch formal eval；`best_metric=stf`。 |
| ETH3D fast | no | configured `proxy`，未运行 | `0.1-80m` | config 保留，但本 run 关闭。 |
| RobotCar Day fast | no | configured `sparse`，未运行 | `0.1-50m` | 本 run 关闭。 |
| RobotCar Night fast | no | configured `sparse`，未运行 | `0.1-50m` | 本 run 关闭。 |
| KITTI val | no | configured `rgb_pretrained_ref`，未运行 | `0.1-80m` | 本 run 关闭。 |
| NYUv2 | no | 未运行 | `0.001-10m` | 本 run 关闭。 |

## 训练过程摘录

### `0521_0012...raw_ram_rgb_bnclean_identity_e5`

| epoch | avg_loss | STF loss | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.040066 | 0.0401 | 676 | 2.6566 | 11.6250 | 00:09:01 |
| 1 | 0.026609 | 0.0266 | 676 | 2.7175 | 11.0000 | 00:08:25 |
| 2 | 0.024256 | 0.0243 | 676 | 2.7383 | 10.7500 | 00:08:57 |
| 3 | 0.022976 | 0.0230 | 676 | 2.7587 | 10.5000 | 00:07:00 |
| 4 | 0.022226 | 0.0222 | 676 | 2.7503 | 10.8750 | 00:07:35 |

运行摘录：

- setup: `stage=stf_only`, `input_type=raw_ram_rgb`, `encoder=vits`, `dav2_train_mode=none`, `epochs=5`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/train_viz`.
- train log 记录的 max GPU memory：`7765 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/best_model.pth`, metric `stf`, value `0.1388`, epoch `4`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/last_epoch_model.pth`, epoch `4`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/`.
