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
| `dav2_train_mode=decoder` | 冻住 DINOv2 backbone，训练 DAv2 depth head / decoder；RAW/RAM run 还会训练 RAW front-end。 |
| `dav2_train_mode=full` | 训练 DINOv2 backbone + DAv2 depth head；`backbone_layer_decay=0.9` 的 run 使用 backbone layer-wise LR decay。 |
| `*_lora` | 在 DAv2 tap blocks `(2,5,8,11)` 加 LoRA；当前 `lora_rank=8`, `lora_alpha=16`, `lora_lr=5e-5`, `lora_block_mode=tap`。 |
| `*_bridge*` | RAW/RAM feature bridge 接入 DAv2 tap blocks；当前 `bridge_source=ram_core`, `bridge_layers=[2,5,8,11]`, `bridge_feature_keys=[x_cat,ffm_mid,x3]`, `bridge_lr=5e-5`。 |
| `*_feature_adapter*` | RAW/RAM features 额外接入 decoder-side feature adapter；当前 `bridge_feature_keys=[x_cat,ffm_mid,x4]`，decoder fusion 为 `path_4,path_3,path_2`。 |
| `stf_only` | STF-only training/eval path，使用 STF pseudo depth targets。 |
| `dav2_pseudo` | 训练 target 来自 DAv2 pseudo labels，而不是 sparse GT depth。 |
| `da3_pseudo_sparse_metric` | 训练 target 来自 DA3 Mono large RGB LUT pseudo labels；loader 对 sparse metric target 做 aligned/dense target 训练。 |
| `ssi` | Scale-shift invariant loss；除特别注明外只用 target normalization；`0521_1308` config/log 额外启用 `loss_lambda_grad=2`, `loss_grad_scales=4`。 |

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
| Loss | `ssi`, `loss_mask_downsample=strict`, `loss_target_normalization=true`, `loss_norm_min_scale=1e-3` |
| optimizer / LR | train code 中 AdamW；base `lr=1e-5`；optimizer step 后 poly decay |
| AMP / seed | `bf16 AMP`, `seed=42` |
| heavy artifact root | `/mnt/drive/3333_raw/0000_exp_ckpt/<run>/` |

## 预留 baseline/control 行

| 预留行 | 状态 | 目标协议 |
|---|---|---|
| DAv2-S RGB 直推 | 待补 | Official RGB DAv2-S checkpoint，不训练，STF RGB input direct eval。 |
| DAv2-S RAW-preview 直推 | 待补 | Official DAv2-S checkpoint，不训练，STF RAW preview / pseudo-RGB direct eval。 |
| RGB input, decoder control | 已启动：`0521_0133...rgb_decoder_e5` | 与 `0521_0112...raw_ram_rgb...decoder_e5` 使用同一 STF pseudo-label train/eval protocol，但 `input_type=rgb`。 |
| RGB input, frozen control | 待补 | 与 `0521_0012...` 使用同一 frozen-DAv2 train/eval protocol，但 `input_type=rgb`。 |

## 当前 formal runs

| Run | Status | Training data | input / RAM / interface | DAv2 train scope | loss | epochs | bs/acc/eff | LR | trainable params | notes |
|---|---|---|---|---|---|---:|---|---|---:|---|
| `0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5` | completed | STF train+test pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；model log `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`；functions `wb,ccm,gamma,brightness` | `none` | `ssi`, target norm | 5 | `8/1/8` | base `1e-5` | 140,903 | total params `24,925,992`；frozen `24,785,089`；best STF checkpoint epoch 4；current/last checkpoint epoch 4 |
| `0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5` | running；tmux `stf_pseudovitl_bnclean_decoder_0521_0112`；last observed epoch 4 started | STF train+test pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；model log `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`；functions `wb,ccm,gamma,brightness`；`rgb_interface_mode=residual_tanh`，`rgb_residual_scale=0.1` | `decoder` | `ssi`, target norm | 5 | `8/1/8` | base `1e-5` | 2,869,416 | total params `24,925,992`；frozen `22,056,576`；best observed STF checkpoint epoch 3, value `0.1346`；current checkpoint epoch 3 |
| `0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5` | running；tmux `stf_rgb_decoder_0521_0133`；last observed epoch 0 step `500/676` | STF train+test pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `rgb` direct input；no RAW/RAM front-end active；model log input shape `(8, 3, 512, 960)`；config still carries inactive RAW defaults `stf_raw_decode_mode=legacy_companded`, `raw_ram_rgb_tail=tanh2p5` | `decoder` | `ssi`, target norm | 5 | `8/1/8` | base `1e-5` | 2,728,513 | total params `24,785,089`；frozen `22,056,576`；pretrain STF abs_rel `0.1286`；no best checkpoint observed yet |
| `0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5` | incomplete；no active tmux observed；last log `2026-05-21 03:51:48` epoch 2 val start | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `rgb_lora` direct RGB；LoRA tap blocks `(2,5,8,11)`；rank `8`，alpha `16`；inactive RAW defaults `stf_raw_decode_mode=legacy_companded`, `raw_ram_rgb_tail=tanh2p5` | `decoder` + LoRA | `ssi`, target norm | 5 | `8/1/8` | base `1e-5`；LoRA `5e-5` | 2,802,241 | total params `24,858,817`；frozen `22,056,576`；best STF checkpoint epoch 0, value `0.1271`；current checkpoint epoch 1 |
| `0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `rgb` direct input；no RAW/RAM front-end active | `full`; `backbone_layer_decay=0.9` | `ssi`, target norm | 5 | `8/1/8` | base `1e-5` | 24,785,089 | total params `24,785,089`；frozen `0`；best STF checkpoint epoch 2, value `0.1281`；last checkpoint epoch 4 |
| `0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb_lora`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；model log `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`；functions `wb,ccm,gamma,brightness`；`rgb_interface_mode=residual_tanh`，`rgb_residual_scale=0.1` | `decoder` + LoRA；RAW/RAM front-end trainable | `ssi`, target norm | 10 | `8/1/8` | base `1e-5`；LoRA `5e-5` | 2,943,144 | total params `24,999,720`；frozen `22,056,576`；best STF checkpoint epoch 8, value `0.1303`；last checkpoint epoch 9 |
| `0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；model log `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`；functions `wb,ccm,gamma,brightness`；`rgb_interface_mode=residual_tanh`，`rgb_residual_scale=0.1` | `full`; RAW/RAM front-end trainable；`backbone_layer_decay=0.9` | `ssi`, target norm | 10 | `8/1/8` | base `1e-5` | 24,925,992 | total params `24,925,992`；frozen `0`；best STF checkpoint epoch 4, value `0.1280`；last checkpoint epoch 9 |
| `0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb_bridge_lora`；RAW path同上；bridge `source=ram_core`, layers `[2,5,8,11]`, keys `[x_cat,ffm_mid,x3]`；`rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm` | `decoder` + LoRA + bridge；RAW/RAM front-end trainable | `ssi`, target norm | 10 | `8/1/8` | base `1e-5`；LoRA `5e-5`；bridge `5e-5` | 3,078,316 | total params `25,134,892`；frozen `22,056,576`；best STF checkpoint epoch 1, value `0.1294`；last checkpoint epoch 9 |
| `0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835` | completed on 186；launched with `WAIT_FOR_GPU_IDLE=0` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_bridge_feature_adapter_lora`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；bridge/features `source=ram_core`, layers `[2,5,8,11]`, keys `[x_cat,ffm_mid,x4]`；decoder fusion `path_4,path_3,path_2`；image bridge `base_rgb+0.1*tanh(1x1_conv(x4))` | `decoder` + LoRA + bridge + decoder-side feature adapter；RAW/RAM front-end trainable | `ssi`, target norm | 10 | `8/1/8` | base `1e-5`；adapter/bridge `5e-5`；LoRA `5e-5` | 3,673,638 | total params `25,730,214`；frozen `22,056,576`；best STF checkpoint epoch 2, value `0.1293`；last checkpoint epoch 9；follows `0521_0835` setting, no `resume_from`/`bridge_init_from` |
| `0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb_bridge`；RAW path同上；bridge `source=ram_core`, layers `[2,5,8,11]`, keys `[x_cat,ffm_mid,x3]`；`rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm` | `full` + bridge；RAW/RAM front-end trainable；`backbone_layer_decay=0.9` | `ssi`, target norm | 10 | `8/1/8` | base `1e-5`；bridge `5e-5` | 25,061,164 | total params `25,061,164`；frozen `0`；best STF checkpoint epoch 4, value `0.1282`；last checkpoint epoch 9 |
| `0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004` | completed on 186；launched with `WAIT_FOR_GPU_IDLE=0` | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_bridge_feature_adapter`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；bridge/features `source=ram_core`, layers `[2,5,8,11]`, keys `[x_cat,ffm_mid,x4]`；decoder fusion `path_4,path_3,path_2`；image bridge `base_rgb+0.1*tanh(1x1_conv(x4))` | `full` + bridge + decoder-side feature adapter；RAW/RAM front-end trainable；`backbone_layer_decay=0.9` | `ssi`, target norm | 10 | `8/1/8` | base `1e-5`；adapter/bridge `5e-5` | 25,656,486 | total params `25,656,486`；frozen `0`；best STF checkpoint epoch 5, value `0.1279`；last checkpoint epoch 9；follows `0521_1004` setting, no `resume_from`/`bridge_init_from` |
| `0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DA3 sparse metric pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `rgb_lora` direct RGB；LoRA tap blocks `(2,5,8,11)`；rank `8`，alpha `16` | `decoder` + LoRA | `ssi`, target norm | 5 | `8/1/8` | base `1e-5`；LoRA `5e-5` | 2,802,241 | total params `24,858,817`；frozen `22,056,576`；best STF checkpoint epoch 0, value `0.1327`；last checkpoint epoch 4 |
| `0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10` | completed；queue tmux `0521_0402_stf_0521_exp2_to_exp8_seq` | STF train+test DA3 sparse metric pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb_lora`；`legacy_online_decomp16`；`norm_mode=passthrough`；`channel_mode=rgb_avg_g`；`raw_ram_rgb_tail=identity`；model log `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`；functions `wb,ccm,gamma,brightness`；`rgb_interface_mode=residual_tanh`，`rgb_residual_scale=0.1` | `decoder` + LoRA；RAW/RAM front-end trainable | `ssi + grad`, target norm；`lambda_grad=2`, `grad_scales=4` | 10 | `8/1/8` | base `1e-5`；LoRA `5e-5` | 2,943,144 | total params `24,999,720`；frozen `22,056,576`；best STF checkpoint epoch 0, value `0.1359`；last checkpoint epoch 9 |
| `0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10` | completed | STF train+test DAv2 pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | `raw_ram_rgb_bridge`；RAW path同上；bridge `source=ram_core`, layers `[2,5,8,11]`, keys `[x_cat,ffm_mid,x3]`；`rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`；`rgb_interface_mode=residual_tanh`，`rgb_residual_scale=0.1` | `none` + bridge；DAv2 frozen；RAW/RAM front-end + bridge trainable | `ssi`, target norm | 10 | `8/1/8` | base `1e-5`；bridge `5e-5` | 276,075 | total params `25,061,164`；frozen `24,785,089`；best STF checkpoint epoch 7, value `0.1327`；last checkpoint epoch 9 |

## Eval 设置记录

| Split | Enabled | Backend / samples | Depth range | Notes |
|---|---|---|---|---|
| STF val | yes | `sparse`, `808` samples | `1-80m` | train-time per-epoch formal eval；`best_metric=stf`。 |
| ETH3D fast | no | configured `proxy`，未运行 | `0.1-80m` | 当前这些 STF-only runs 关闭。 |
| RobotCar Day fast | no | configured `sparse`，未运行 | `0.1-50m` | 当前这些 STF-only runs 关闭。 |
| RobotCar Night fast | no | configured `sparse`，未运行 | `0.1-50m` | 当前这些 STF-only runs 关闭。 |
| KITTI val | no | configured `rgb_pretrained_ref`，未运行 | `0.1-80m` | 当前这些 STF-only runs 关闭。 |
| NYUv2 | no | 未运行 | `0.001-10m` | 当前这些 STF-only runs 关闭。 |

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

### `0521_0112...raw_ram_rgb_bnclean_identity_decoder_e5`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.025270 | 0.1383 | 676 | 2.7000 | 11.1250 | 00:08:32 |
| 1 | 0.017229 | 0.1364 | 676 | 2.7198 | 10.6250 | 00:08:54 |
| 2 | 0.015428 | 0.1351 | 676 | 2.7026 | 10.5000 | 00:12:09 |
| 3 | 0.014435 | 0.1346 | 676 | 2.6758 | 9.6875 | 00:12:17 |

运行摘录：

- status: still running in tmux `stf_pseudovitl_bnclean_decoder_0521_0112`; latest exp log line observed `2026-05-21 01:57:44`, epoch `4/5` started.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=5`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3202`, silog `0.4303`, d1 `0.5218`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/train_viz`.
- train log 记录的 max GPU memory：`9265 MB`.
- best observed checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/best_model.pth`, metric `stf`, value `0.1346`, epoch `3`.
- current observed checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/current_model.pth`, epoch `3`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/`.

### `0521_0133...rgb_decoder_e5`

| checkpoint / stage | abs_rel | rmse | silog | d1 | elapsed |
|---|---:|---:|---:|---:|---:|
| pretrain STF eval | 0.1286 | 7.9180 | 0.2576 | 0.8577 | 00:03:43 |
| epoch 0 train step `500/676` | n/a | n/a | n/a | n/a | 00:23:07 |

运行摘录：

- status: still running in tmux `stf_rgb_decoder_0521_0133`; latest tmux/exp log line observed `2026-05-21 02:00:22`, epoch `0`, micro step `500/676`.
- setup: `stage=stf_only`, `input_type=rgb`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=5`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`.
- RGB/data path: STF RGB input direct；training target mode `dav2_pseudo`；pseudo manifest `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- model: total params `24,785,089`, trainable `2,728,513`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/train_viz`.
- train log 记录的 max GPU memory so far：`4838 MB`.
- heavy artifacts so far: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/` currently only has TensorBoard event file; no best/current checkpoint observed yet.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/`.

### `0521_0306...rgb_lora_decoder_e5`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.007057 | 0.1271 | 676 | 2.9695 | 11.8125 | 00:12:44 |
| 1 | 0.005566 | 0.1274 | 676 | 2.9794 | 11.8125 | 00:15:07 |

运行摘录：

- status: incomplete；未观察到 active tmux；latest exp log line `2026-05-21 03:51:48`，epoch `2/5` val start；无 `last_epoch_model.pth`。
- setup: `stage=stf_only`, `input_type=rgb_lora`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=5`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- LoRA: `lora_block_mode=tap`, `lora_blocks=(2,5,8,11)`, `lora_rank=8`, `lora_alpha=16.0`, `lora_lr=5e-5`; base `lr=1e-5`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `dav2_pseudo`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.1286`, RMSE `7.9181`, silog `0.2576`, d1 `0.8577`.
- model: total params `24,858,817`, trainable `2,802,241`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/train_viz`.
- train log 记录的 max GPU memory：`9245 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/best_model.pth`, metric `stf`, value `0.1271`, epoch `0`.
- current checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/current_model.pth`, epoch `1`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/`.

### `0521_0402...rgb_full_lrd09_e5`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.007863 | 0.1288 | 676 | 3.1622 | 12.1875 | 00:15:32 |
| 1 | 0.005018 | 0.1292 | 676 | 3.2949 | 13.8125 | 00:13:20 |
| 2 | 0.003840 | 0.1281 | 676 | 3.4498 | 13.8125 | 00:16:37 |
| 3 | 0.003248 | 0.1291 | 676 | 3.5197 | 14.1875 | 00:15:32 |
| 4 | 0.002778 | 0.1284 | 676 | 3.5541 | 14.1250 | 00:15:57 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=rgb`, `encoder=vits`, `dav2_train_mode=full`, `backbone_layer_decay=0.9`, `epochs=5`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `dav2_pseudo`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.1286`, RMSE `7.9181`, silog `0.2576`, d1 `0.8577`.
- model: total params `24,785,089`, trainable `24,785,089`, frozen `0`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/train_viz`.
- train log 记录的 max GPU memory：`9498 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/best_model.pth`, metric `stf`, value `0.1281`, epoch `2`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/last_epoch_model.pth`, epoch `4`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/`.

### `0521_0522...raw_ram_rgb_bnclean_identity_lora_decoder_e10`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.018639 | 0.1315 | 676 | 2.6923 | 10.6875 | 00:10:35 |
| 1 | 0.011700 | 0.1305 | 676 | 2.7007 | 10.5000 | 00:09:51 |
| 2 | 0.010023 | 0.1311 | 676 | 2.7059 | 10.6250 | 00:08:53 |
| 3 | 0.009088 | 0.1321 | 676 | 2.7054 | 10.5000 | 00:09:01 |
| 4 | 0.008418 | 0.1315 | 676 | 2.7102 | 10.7500 | 00:08:33 |
| 5 | 0.007968 | 0.1315 | 676 | 2.7068 | 10.6875 | 00:08:23 |
| 6 | 0.007631 | 0.1312 | 676 | 2.7237 | 10.4375 | 00:08:41 |
| 7 | 0.007360 | 0.1309 | 676 | 2.7500 | 10.5000 | 00:08:26 |
| 8 | 0.007214 | 0.1303 | 676 | 2.7332 | 10.4375 | 00:09:25 |
| 9 | 0.007062 | 0.1305 | 676 | 2.7456 | 10.5625 | 00:07:57 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb_lora`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- LoRA: `lora_block_mode=tap`, `lora_blocks=(2,5,8,11)`, `lora_rank=8`, `lora_alpha=16.0`, `lora_lr=5e-5`; base `lr=1e-5`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- model log: `functions=['wb','ccm','gamma','brightness']`, `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`, `rgb_interface_mode=residual_tanh`, `rgb_residual_scale=0.1`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3202`, silog `0.4303`, d1 `0.5218`.
- model: total params `24,999,720`, trainable `2,943,144`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/train_viz`.
- train log 记录的 max GPU memory：`9267 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/best_model.pth`, metric `stf`, value `0.1303`, epoch `8`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/`.

### `0521_0656...raw_ram_rgb_bnclean_identity_full_lrd09_e10`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.013359 | 0.1301 | 676 | 3.1149 | 11.9375 | 00:09:40 |
| 1 | 0.007687 | 0.1310 | 676 | 3.1733 | 13.6250 | 00:08:48 |
| 2 | 0.006111 | 0.1300 | 676 | 3.3258 | 14.0625 | 00:09:00 |
| 3 | 0.005140 | 0.1293 | 676 | 3.4242 | 13.7500 | 00:09:17 |
| 4 | 0.004559 | 0.1280 | 676 | 3.5408 | 14.5000 | 00:09:13 |
| 5 | 0.003982 | 0.1294 | 676 | 3.6126 | 14.9375 | 00:10:08 |
| 6 | 0.003540 | 0.1293 | 676 | 3.6885 | 15.0625 | 00:08:59 |
| 7 | 0.003185 | 0.1284 | 676 | 3.7304 | 15.4375 | 00:09:14 |
| 8 | 0.002923 | 0.1291 | 676 | 3.7889 | 15.7500 | 00:09:08 |
| 9 | 0.002715 | 0.1287 | 676 | 3.8302 | 15.8750 | 00:09:16 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb`, `encoder=vits`, `dav2_train_mode=full`, `backbone_layer_decay=0.9`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- model log: `functions=['wb','ccm','gamma','brightness']`, `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`, `rgb_interface_mode=residual_tanh`, `rgb_residual_scale=0.1`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3202`, silog `0.4303`, d1 `0.5218`.
- model: total params `24,925,992`, trainable `24,925,992`, frozen `0`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/train_viz`.
- train log 记录的 max GPU memory：`10065 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/best_model.pth`, metric `stf`, value `0.1280`, epoch `4`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/`.

### `0521_0835...raw_ram_rgb_bnclean_bridge_lora_decoder_e10`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.016640 | 0.1308 | 676 | 2.8776 | 10.7500 | 00:09:22 |
| 1 | 0.010969 | 0.1294 | 676 | 2.8783 | 11.0625 | 00:09:30 |
| 2 | 0.009587 | 0.1300 | 676 | 2.8526 | 11.2500 | 00:08:21 |
| 3 | 0.008769 | 0.1316 | 676 | 2.8556 | 10.7500 | 00:07:55 |
| 4 | 0.008220 | 0.1306 | 676 | 2.8737 | 11.0625 | 00:08:15 |
| 5 | 0.007770 | 0.1308 | 676 | 2.8929 | 11.0625 | 00:08:19 |
| 6 | 0.007418 | 0.1304 | 676 | 2.8985 | 11.0000 | 00:08:03 |
| 7 | 0.007151 | 0.1304 | 676 | 2.9236 | 11.2500 | 00:09:33 |
| 8 | 0.006996 | 0.1298 | 676 | 2.9132 | 11.0625 | 00:08:31 |
| 9 | 0.006881 | 0.1298 | 676 | 2.9194 | 11.0625 | 00:07:48 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb_bridge_lora`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- bridge/LoRA: `bridge_source=ram_core`, `bridge_layers=[2,5,8,11]`, `bridge_feature_keys=[x_cat,ffm_mid,x3]`, `bridge_lr=5e-5`; `lora_rank=8`, `lora_alpha=16.0`, `lora_lr=5e-5`.
- model log: `rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3243`, silog `0.4305`, d1 `0.5216`.
- model: total params `25,134,892`, trainable `3,078,316`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/train_viz`.
- train log 记录的 max GPU memory：`9327 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/best_model.pth`, metric `stf`, value `0.1294`, epoch `1`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/`.

### `0521_1606...raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.016143 | 0.1315 | 676 | 2.8707 | 10.3125 | 00:18:54 |
| 1 | 0.009660 | 0.1302 | 676 | 2.8413 | 10.5625 | 00:18:57 |
| 2 | 0.008296 | 0.1293 | 676 | 2.8485 | 11.1250 | 00:18:32 |
| 3 | 0.007532 | 0.1312 | 676 | 2.8883 | 10.9375 | 00:18:50 |
| 4 | 0.007071 | 0.1295 | 676 | 2.9074 | 11.3125 | 00:18:55 |
| 5 | 0.006682 | 0.1309 | 676 | 2.9082 | 11.5625 | 00:18:53 |
| 6 | 0.006363 | 0.1299 | 676 | 2.9321 | 11.5625 | 00:18:51 |
| 7 | 0.006191 | 0.1299 | 676 | 2.9528 | 11.5625 | 00:18:47 |
| 8 | 0.006039 | 0.1296 | 676 | 2.9315 | 11.6250 | 00:19:01 |
| 9 | 0.005886 | 0.1294 | 676 | 2.9416 | 11.5000 | 00:18:59 |

运行摘录：

- status: completed on 186；launched with `WAIT_FOR_GPU_IDLE=0`; final tmux log ended `2026-05-21T19:23:56+08:00 status=0`.
- setup: `stage=stf_only`, `input_type=raw_ram_bridge_feature_adapter_lora`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- LoRA: `lora_block_mode=tap`, `lora_blocks=(2,5,8,11)`, `lora_rank=8`, `lora_alpha=16.0`, `lora_lr=5e-5`; base `lr=1e-5`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- feature adapter / bridge: `bridge_source=ram_core`, `bridge_layers=[2,5,8,11]`, `bridge_feature_keys=[x_cat,ffm_mid,x4]`, `bridge_lr=5e-5`; decoder fusion `path_4,path_3,path_2`; image bridge `base_rgb+0.1*tanh(1x1_conv(x4))`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `dav2_pseudo`; batch log `target_space=inverse_relative`, `target_source=dense_pseudo`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrained init: `/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth`; no `resume_from` / no `bridge_init_from`.
- pretrain eval: STF val abs_rel `0.1458`, RMSE `8.3949`, silog `0.2647`, d1 `0.8126`.
- model: total params `25,730,214`, trainable `3,673,638`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/train_viz`.
- train log 记录的 max GPU memory：`13205 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/best_model.pth`, metric `stf`, value `0.1293`, epoch `2`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/`.

### `0521_1004...raw_ram_rgb_bnclean_bridge_full_lrd09_e10`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.012605 | 0.1318 | 676 | 3.2092 | 12.4375 | 00:09:31 |
| 1 | 0.007477 | 0.1309 | 676 | 3.3072 | 13.7500 | 00:08:31 |
| 2 | 0.005900 | 0.1296 | 676 | 3.4304 | 14.8125 | 00:08:42 |
| 3 | 0.005044 | 0.1311 | 676 | 3.5258 | 14.5625 | 00:09:00 |
| 4 | 0.004384 | 0.1282 | 676 | 3.6385 | 15.0625 | 00:09:40 |
| 5 | 0.003831 | 0.1286 | 676 | 3.6984 | 15.1875 | 00:08:42 |
| 6 | 0.003429 | 0.1293 | 676 | 3.7556 | 15.6250 | 00:08:33 |
| 7 | 0.003090 | 0.1289 | 676 | 3.8008 | 15.6250 | 00:08:10 |
| 8 | 0.002850 | 0.1293 | 676 | 3.8451 | 15.8750 | 00:08:18 |
| 9 | 0.002647 | 0.1288 | 676 | 3.8908 | 16.1250 | 00:08:37 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb_bridge`, `encoder=vits`, `dav2_train_mode=full`, `backbone_layer_decay=0.9`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- bridge: `bridge_source=ram_core`, `bridge_layers=[2,5,8,11]`, `bridge_feature_keys=[x_cat,ffm_mid,x3]`, `bridge_lr=5e-5`; model log `rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3243`, silog `0.4305`, d1 `0.5216`.
- model: total params `25,061,164`, trainable `25,061,164`, frozen `0`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/train_viz`.
- train log 记录的 max GPU memory：`10786 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/best_model.pth`, metric `stf`, value `0.1282`, epoch `4`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/`.

### `0521_1606...raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.012427 | 0.1301 | 676 | 2.9076 | 11.2500 | 00:18:54 |
| 1 | 0.006853 | 0.1311 | 676 | 3.0671 | 12.5000 | 00:18:53 |
| 2 | 0.005434 | 0.1294 | 676 | 3.2190 | 12.6875 | 00:18:32 |
| 3 | 0.004542 | 0.1303 | 676 | 3.3070 | 13.0625 | 00:18:48 |
| 4 | 0.003969 | 0.1291 | 676 | 3.4089 | 13.6250 | 00:18:49 |
| 5 | 0.003597 | 0.1279 | 676 | 3.4943 | 14.1250 | 00:18:50 |
| 6 | 0.003150 | 0.1291 | 676 | 3.6035 | 14.3750 | 00:18:48 |
| 7 | 0.002793 | 0.1280 | 676 | 3.6227 | 14.3125 | 00:18:46 |
| 8 | 0.002534 | 0.1291 | 676 | 3.6652 | 14.9375 | 00:19:01 |
| 9 | 0.002333 | 0.1285 | 676 | 3.6893 | 14.6875 | 00:18:55 |

运行摘录：

- status: completed on 186；launched with `WAIT_FOR_GPU_IDLE=0`; final tmux log ended `2026-05-21T19:23:56+08:00 status=0`.
- setup: `stage=stf_only`, `input_type=raw_ram_bridge_feature_adapter`, `encoder=vits`, `dav2_train_mode=full`, `backbone_layer_decay=0.9`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- feature adapter / bridge: `bridge_source=ram_core`, `bridge_layers=[2,5,8,11]`, `bridge_feature_keys=[x_cat,ffm_mid,x4]`, `bridge_lr=5e-5`; decoder fusion `path_4,path_3,path_2`; image bridge `base_rgb+0.1*tanh(1x1_conv(x4))`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `dav2_pseudo`; batch log `target_space=inverse_relative`, `target_source=dense_pseudo`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrained init: `/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth`; no `resume_from` / no `bridge_init_from`.
- pretrain eval: STF val abs_rel `0.1458`, RMSE `8.3948`, silog `0.2647`, d1 `0.8126`.
- model: total params `25,656,486`, trainable `25,656,486`, frozen `0`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/train_viz`.
- train log 记录的 max GPU memory：`14667 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/best_model.pth`, metric `stf`, value `0.1279`, epoch `5`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/`.

### `0521_1137...pseudoda3_sparse_metric_rgb_lora_decoder_e5`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.030097 | 0.1327 | 676 | 2.7414 | 12.2500 | 00:19:38 |
| 1 | 0.025137 | 0.1343 | 676 | 2.4197 | 10.3125 | 00:18:23 |
| 2 | 0.023223 | 0.1355 | 676 | 2.3087 | 10.0000 | 00:18:02 |
| 3 | 0.022021 | 0.1371 | 676 | 2.2473 | 9.8125 | 00:16:12 |
| 4 | 0.021338 | 0.1364 | 676 | 2.2162 | 9.6250 | 00:15:23 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=rgb_lora`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=5`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- LoRA: `lora_block_mode=tap`, `lora_blocks=(2,5,8,11)`, `lora_rank=8`, `lora_alpha=16.0`, `lora_lr=5e-5`; base `lr=1e-5`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `da3_pseudo_sparse_metric`; batch log `target_space=metric_depth`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.1286`, RMSE `7.9181`, silog `0.2576`, d1 `0.8577`.
- model: total params `24,858,817`, trainable `2,802,241`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/train_viz`.
- train log 记录的 max GPU memory：`9245 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/best_model.pth`, metric `stf`, value `0.1327`, epoch `0`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/last_epoch_model.pth`, epoch `4`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/`.

### `0521_1308...pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.044095 | 0.1359 | 676 | 2.6083 | 10.6250 | 00:11:06 |
| 1 | 0.033459 | 0.1360 | 676 | 2.5191 | 9.4375 | 00:10:21 |
| 2 | 0.030348 | 0.1388 | 676 | 2.3745 | 9.2500 | 00:09:59 |
| 3 | 0.028238 | 0.1404 | 676 | 2.2307 | 8.6250 | 00:09:46 |
| 4 | 0.026982 | 0.1398 | 676 | 2.1643 | 8.3125 | 00:10:30 |
| 5 | 0.026033 | 0.1392 | 676 | 2.1047 | 8.3125 | 00:09:59 |
| 6 | 0.025179 | 0.1388 | 676 | 2.0474 | 8.1875 | 00:09:00 |
| 7 | 0.024646 | 0.1392 | 676 | 2.0173 | 7.8125 | 00:09:44 |
| 8 | 0.024158 | 0.1394 | 676 | 1.9974 | 8.1250 | 00:12:09 |
| 9 | 0.023859 | 0.1396 | 676 | 1.9761 | 7.9375 | 00:12:17 |

运行摘录：

- status: completed via tmux queue `0521_0402_stf_0521_exp2_to_exp8_seq`.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb_lora`, `encoder=vits`, `dav2_train_mode=decoder`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- loss: `ssi`, `loss_target_normalization=true`, `loss_lambda_grad=2`, `loss_grad_scales=4`, `loss_mask_downsample=strict`, `loss_norm_min_scale=1e-3`.
- LoRA: `lora_block_mode=tap`, `lora_blocks=(2,5,8,11)`, `lora_rank=8`, `lora_alpha=16.0`, `lora_lr=5e-5`; base `lr=1e-5`.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- model log: `functions=['wb','ccm','gamma','brightness']`, `dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`, `rgb_interface_mode=residual_tanh`, `rgb_residual_scale=0.1`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `da3_pseudo_sparse_metric`; batch log `target_space=metric_depth`, `target_source=dense_aligned`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3202`, silog `0.4303`, d1 `0.5218`.
- model: total params `24,999,720`, trainable `2,943,144`, frozen `22,056,576`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/train_viz`.
- train log 记录的 max GPU memory：`9267 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/best_model.pth`, metric `stf`, value `0.1359`, epoch `0`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/`.

### `0521_1542...pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10`

| epoch | avg_loss | STF val abs_rel | used steps | raw_pred_valid_mean | raw_pred_valid_max | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.029993 | 0.1390 | 676 | 2.6887 | 10.5625 | 00:09:13 |
| 1 | 0.019055 | 0.1358 | 676 | 2.7284 | 9.8125 | 00:09:10 |
| 2 | 0.016186 | 0.1345 | 676 | 2.7282 | 10.1875 | 00:10:23 |
| 3 | 0.014349 | 0.1335 | 676 | 2.7263 | 9.9375 | 00:08:38 |
| 4 | 0.013473 | 0.1334 | 676 | 2.7308 | 10.0000 | 00:09:49 |
| 5 | 0.012845 | 0.1331 | 676 | 2.7369 | 9.8750 | 00:09:44 |
| 6 | 0.012291 | 0.1329 | 676 | 2.7439 | 9.6250 | 00:09:50 |
| 7 | 0.012061 | 0.1327 | 676 | 2.7419 | 10.0000 | 00:08:35 |
| 8 | 0.011836 | 0.1328 | 676 | 2.7438 | 9.8750 | 00:10:33 |
| 9 | 0.011633 | 0.1328 | 676 | 2.7479 | 10.0000 | 00:06:52 |

运行摘录：

- status: completed.
- setup: `stage=stf_only`, `input_type=raw_ram_rgb_bridge`, `encoder=vits`, `dav2_train_mode=none`, `epochs=10`, `bs=8`, `accum_steps=1`, `effective_bs=8`, `num_workers=4`.
- scheduler counters: `optimizer_steps_per_epoch=676`, `micro_steps_per_epoch=676`.
- loss: `ssi`, `loss_target_normalization=true`, `loss_mask_downsample=strict`, `loss_norm_min_scale=1e-3`; no gradient regularization in config.
- RAW/data path: `raw_npz_root=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`, `stf_raw_decode_mode=legacy_online_decomp16`, `norm_mode=passthrough`, `channel_mode=rgb_avg_g`, `raw_ram_rgb_tail=identity`.
- bridge: `bridge_source=ram_core`, `bridge_layers=[2,5,8,11]`, `bridge_feature_keys=[x_cat,ffm_mid,x3]`, `bridge_lr=5e-5`; model log `rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`.
- dataset: `stf_train=5408`, `val=808`, `merge_test_into_train=True`; target mode `dav2_pseudo`; batch log `target_space=inverse_relative`, `target_source=dense_pseudo`.
- pseudo target manifest: `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`.
- pretrain eval: STF val abs_rel `0.2898`, RMSE `12.3243`, silog `0.4305`, d1 `0.5216`.
- model: total params `25,061,164`, trainable `276,075`, frozen `24,785,089`.
- train-viz: fixed samples `{'stf': 8}`, RGB baseline panel enabled, root `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/train_viz`.
- train log 记录的 max GPU memory：`9237 MB`.
- best checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/best_model.pth`, metric `stf`, value `0.1327`, epoch `7`.
- last checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/last_epoch_model.pth`, epoch `9`.
- light artifacts: `/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/`.
