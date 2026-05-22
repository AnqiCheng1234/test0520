# RGB / RAW baseline 公平性汇总

日期：2026-05-22

本文记录 `/home/caq/6666_raw/dav2_raw_0520` 当前 STF formal validation 结果。

写作规约：指标表只放 train-time formal eval 或同口径 eval log 中的数字；只用于可视化的 sample loss 不混入主指标表。

指标表标注规约：`abs_rel` 越低越好，`d1` 越高越好。若后续新增尚未补跑的行，保留为 `待补`。

## 0. 协议速查 / 数据源规则

当前 formal runs：

- `0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/train.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` 合成 `[R,(Gr+Gb)/2,B]` -> RamCore3 BN-clean path -> DAv2
  - model log：`functions=['wb','ccm','gamma','brightness']`，`ram_core_out_channels=3`，`dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 4
- `0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb_bridge`
  - RAW path：同 `0521_0012` 的 STF legacy online decomp16 RAW -> `channel_mode=rgb_avg_g` -> RamCore3 BN-clean path；额外启用 `bridge_source=ram_core`
  - model log：`bridge_feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，`rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`，`dav2_train_mode=none`，`bridge_lr=5.00e-05`，`trainable_params=276075`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 7；last epoch 为 epoch 9
- `0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0522_0137_stf_ram_fa_bridge_from_0521_1542.queue.log`
  - stage: `stf_only`
  - input: `raw_ram_bridge_feature_adapter`
  - RAW path：同 `0521_1542` 的 STF legacy online decomp16 RAW -> `channel_mode=rgb_avg_g` -> RamCore3；额外启用 bridge + decoder-side feature adapter
  - model log：`feature_keys=['x_cat','ffm_mid','x4']`，`bridge_layers=[2,5,8,11]`，decoder fusion `path_4,path_3,path_2`，`image_bridge=base_rgb+0.1*tanh(1x1_conv(x4))`，`dav2_train_mode=none`，`trainable_params=871397`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 9；last epoch 为 epoch 9
- `0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0522_1423_stf_ram_rgb_fa_bridge_from_0521_1542.queue.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb_bridge_feature_adapter`
  - RAW path：同 `0521_1542` 的 STF legacy online decomp16 RAW -> `channel_mode=rgb_avg_g` -> RamCore3；额外启用 bridge + decoder-side feature adapter
  - model log：`feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，decoder fusion `path_4,path_3,path_2`，`image_bridge=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`，`dav2_train_mode=none`，`trainable_params=860971`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 9；last epoch 为 epoch 9
- `0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/train.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb`
  - RAW path：同 `0521_0012`，STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` 合成 `[R,(Gr+Gb)/2,B]` -> RamCore3 BN-clean path -> DAv2
  - model log：`functions=['wb','ccm','gamma','brightness']`，`ram_core_out_channels=3`，`dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`，`dav2_train_mode=decoder`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 4

baseline/control 行：

- DAv2-L RGB 直推：official RGB DAv2-L checkpoint，不训练；已补跑。
  - source log：`/home/caq/6666_raw/dav2_raw_0520/logs/0521_2347_stf_dav2l_rgb_direct_baseline.log`
  - eval report：`/home/caq/6666_raw/dav2_raw_0520/anqi_eval/results/0521_2347_stf_val_dav2l_rgb_direct/eval_stf_rel_depth_val_2026-05-21.txt`
  - eval split：STF val，`808` samples，sparse backend equivalent，`min_depth=1 / max_depth=80`，input size `512x960`
- DAv2-S RGB 直推：official RGB DAv2-S checkpoint，不训练；已补跑。
  - source log：`/home/caq/6666_raw/dav2_raw_0520/logs/0521_0104_stf_dav2s_direct_baselines.log`
  - eval report：`/home/caq/6666_raw/dav2_raw_0520/anqi_eval/results/0521_0104_stf_val_dav2s_rgb_direct/eval_stf_rel_depth_val_2026-05-21.txt`
  - eval split：STF val，`808` samples，sparse backend equivalent，`min_depth=1 / max_depth=80`，input size `512x960`
- DAv2-S RAW-preview 直推：official DAv2-S 在 RAW preview / pseudo-RGB 输入上直推，不训练；已补跑。
  - source log：`/home/caq/6666_raw/dav2_raw_0520/logs/0521_0104_stf_dav2s_direct_baselines.log`
  - eval report：`/home/caq/6666_raw/dav2_raw_0520/anqi_eval/results/0521_0104_stf_val_dav2s_raw_preview_direct/eval_stf_rel_depth_val_2026-05-21.txt`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` 合成 `[R,(Gr+Gb)/2,B]` -> ImageNet norm -> official DAv2-S
  - eval split：STF val，`808` samples，sparse backend equivalent，`min_depth=1 / max_depth=80`，input size `512x960`
- 当前同设定 RGB 输入：沿用当前 STF pseudo-label training protocol，但 `input_type=rgb`，没有 RAW/RAM front end，只训练 DAv2 decoder；run 已完成。
  - run：`0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5.tmux.log`
  - stage: `stf_only`
  - input: `rgb`
  - model log：`dav2_train_mode=decoder`，`trainable_params=2728513`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 0
- `0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5.tmux.log`
  - stage: `stf_only`
  - input: `rgb_lora`
  - model log：`dav2_train_mode=decoder`，`lora_block_mode=tap`，`lora_blocks=(2,5,8,11)`，`lora_rank=8`，`lora_alpha=16.0`，`trainable_params=2802241`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 0；该 run 日志停在 epoch 2 eval start，完整 val 只有 epoch 0/1，current checkpoint 为 epoch 1，无 `last_epoch_model.pth`
- `0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5.tmux.log`
  - stage: `stf_only`
  - input: `rgb`
  - model log：`dav2_train_mode=full`，`backbone_layer_decay=0.9`，`trainable_params=24785089`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 2；last epoch 为 epoch 4
- `0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb_lora`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` 合成 `[R,(Gr+Gb)/2,B]` -> RamCore3 BN-clean path -> DAv2
  - model log：`functions=['wb','ccm','gamma','brightness']`，`ram_core_out_channels=3`，`dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`，`dav2_train_mode=decoder`，LoRA tap blocks `(2,5,8,11)`，`trainable_params=2943144`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 8；last epoch 为 epoch 9
- `0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb`
  - RAW path：同 `0521_0522` identity BN-clean RAW/RAM path
  - model log：`functions=['wb','ccm','gamma','brightness']`，`ram_core_out_channels=3`，`dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，`trainable_params=24925992`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 4；last epoch 为 epoch 9
- `0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb_bridge_lora`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` -> RamCore3 BN-clean path；bridge 使用 `bridge_source=ram_core`
  - model log：`bridge_feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，`rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`，`dav2_train_mode=decoder`，LoRA tap blocks `(2,5,8,11)`，`trainable_params=3078316`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 1；last epoch 为 epoch 9
- `0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835`
  - source log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/train.log`
  - tmux log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_bridge_feature_adapter_lora`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` -> RamCore3 -> image bridge；额外启用 bridge + decoder-side feature adapter
  - model log：`feature_keys=['x_cat','ffm_mid','x4']`，`bridge_layers=[2,5,8,11]`，decoder fusion `path_4,path_3,path_2`，`image_bridge=base_rgb+0.1*tanh(1x1_conv(x4))`，`dav2_train_mode=decoder`，LoRA tap blocks `(2,5,8,11)`，`trainable_params=3673638`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 2；last epoch 为 epoch 9
- `0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb_bridge`
  - RAW path：同 `0521_0835` bridge RAW/RAM path
  - model log：`bridge_feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，`rgb_interface_head=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，`trainable_params=25061164`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 4；last epoch 为 epoch 9
- `0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004`
  - source log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/train.log`
  - tmux log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_bridge_feature_adapter`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` -> RamCore3 -> image bridge；额外启用 bridge + decoder-side feature adapter
  - model log：`feature_keys=['x_cat','ffm_mid','x4']`，`bridge_layers=[2,5,8,11]`，decoder fusion `path_4,path_3,path_2`，`image_bridge=base_rgb+0.1*tanh(1x1_conv(x4))`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，`trainable_params=25656486`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 5；last epoch 为 epoch 9
- `0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_bridge_feature_adapter`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` -> RamCore3 -> image bridge；额外启用 bridge + decoder-side feature adapter
  - model log：`feature_keys=['x_cat','ffm_mid','x4']`，`bridge_layers=[2,5,8,11]`，decoder fusion `path_4,path_3,path_2`，`image_bridge=base_rgb+0.1*tanh(1x1_conv(x4))`，`dav2_train_mode=full`，`backbone_layer_decay=1.0` / no LRD，`trainable_params=25656486`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 7；last epoch 为 epoch 9
- `0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5.tmux.log`
  - stage: `stf_only`
  - input: `rgb_lora`
  - model log：`dav2_train_mode=decoder`，`lora_block_mode=tap`，`lora_blocks=(2,5,8,11)`，`lora_rank=8`，`lora_alpha=16.0`，`trainable_params=2802241`
  - train target：DA3 mono large sparse metric pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 0；last epoch 为 epoch 4
- `0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10.tmux.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb_lora`
  - RAW path：同 `0521_0522` identity BN-clean RAW/RAM path
  - model log：`functions=['wb','ccm','gamma','brightness']`，`ram_core_out_channels=3`，`dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`，`dav2_train_mode=decoder`，LoRA tap blocks `(2,5,8,11)`，`trainable_params=2943144`
  - train target：DA3 mono large sparse metric pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 0；last epoch 为 epoch 9

run-row 数据源优先级：

1. 后续若生成 `finetune_stf/exp/<run>/analysis/*.csv` 或 `summary*.json`，优先使用。
2. `finetune_stf/exp/<run>/train.log`.
3. `finetune_stf/exp/<run>/config.json` 只用于确认配置。

## 1. STF val

### 1.1 DAv2 direct zero-shot baselines

| Experiment | Training data / schedule | abs_rel best (epoch) | abs_rel last/current | d1 best (epoch) | d1 last/current | notes |
|---|---|---:|---:|---:|---:|---|
| DAv2-L RGB 直推 | baseline，不训练；`0521_2347_stf_val_dav2l_rgb_direct` | 0.1292 (zero-shot) | 0.1292 | **0.8601 (zero-shot)** | 0.8601 | official RGB DAv2-L 在 STF RGB LUT preview 输入上 zero-shot；`rmse=7.9835`，`silog=0.2613`。 |
| DAv2-S RGB 直推 | baseline，不训练；`0521_0104_stf_val_dav2s_rgb_direct` | **0.1287 (zero-shot)** | 0.1287 | 0.8575 (zero-shot) | 0.8575 | official RGB DAv2-S 在 STF RGB LUT preview 输入上 zero-shot；`rmse=7.9067`，`silog=0.2560`。 |
| DAv2-S RAW-preview 直推 | baseline，不训练；`0521_0104_stf_val_dav2s_raw_preview_direct` | 0.1649 (zero-shot) | 0.1649 | 0.7774 (zero-shot) | 0.7774 | official DAv2-S 在 STF RAW preview / pseudo-RGB path 上 zero-shot；`legacy_online_decomp16 + passthrough + rgb_avg_g`；`rmse=9.0814`，`silog=0.2897`。 |

### 1.2 DAv2 pseudo RGB finetune runs

| Experiment | Training data / schedule | abs_rel best (epoch) | abs_rel last/current | d1 best (epoch) | d1 last/current | notes |
|---|---|---:|---:|---:|---:|---|
| `0521_0133...rgb_decoder_e5` | STF train+test pseudo-label training；`input_type=rgb`，val `808`，`stf_repeat=7`，每 epoch `676` steps；只训练 DAv2-S decoder | 0.1278 (e0) | 0.1287 | 0.8576 (e0) | 0.8562 | fair RGB-input training control；best checkpoint 按 abs_rel 在 epoch 0 保存，last 为 epoch 4；last `rmse=7.9351`，`silog=0.2591`。 |
| `0521_0306...rgb_lora_decoder_e5` | STF train+test pseudo-label training；`input_type=rgb_lora`，val `808`，`stf_repeat=7`，每 epoch `676` steps；DAv2-S decoder + LoRA tap blocks `(2,5,8,11)` | **0.1271 (e0)** | 0.1274 | **0.8590 (e0)** | 0.8578 | RGB LoRA/decoder control；run incomplete，完整 val 只有 e0/e1，日志停在 e2 eval start；current checkpoint 为 e1，无 `last_epoch_model.pth`。 |
| `0521_0402...rgb_full_lrd09_e5` | STF train+test pseudo-label training；`input_type=rgb`，val `808`，`stf_repeat=7`，每 epoch `676` steps；DAv2-S full finetune，`backbone_layer_decay=0.9` | 0.1281 (e2) | 0.1284 | 0.8581 (e2/e4) | 0.8581 | RGB full-finetune control；best checkpoint 按 abs_rel 在 epoch 2 保存，last 为 epoch 4；last `rmse=7.9202`，`silog=0.2591`。 |

### 1.3 DAv2 pseudo RAW/RAM training runs

| Experiment | Training data / schedule | abs_rel best (epoch) | abs_rel last/current | d1 best (epoch) | d1 last/current | notes |
|---|---|---:|---:|---:|---:|---|
| `0521_0012...raw_ram_rgb_bnclean_identity_e5` | STF train+test pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | 0.1388 (e4) | 0.1388 | 0.8319 (e4) | 0.8319 | 当前项目摘要中的第一条 formal run。DAv2-S 冻住（`dav2_train_mode=none`），只训练 RAW/RAM front end；best 和 last 均为 epoch 4。 |
| `0521_1542...bridge_ram_e10` | STF train+test pseudo-label training；同 `0521_0012` 为 DAv2-S frozen RAW/RAM adapter 口径；`input_type=raw_ram_rgb_bridge`，bridge layers `[2,5,8,11]`，keys `x_cat/ffm_mid/x3`，每 epoch `676` steps | 0.1327 (e7) | 0.1328 | 0.8453 (e9) | 0.8453 | 与 `0521_0012` 主要对照 bridge adapter 和训练轮数；loss 同为 SSI。best checkpoint 按 abs_rel 在 epoch 7 保存，last 为 epoch 9；last `rmse=8.0426`，`silog=0.2621`。 |
| `0522_0137...feature_adapter_ram_e10_from_0521_1542`（4ch RAW） | STF train+test pseudo-label training；同 `0521_1542` 为 DAv2-S frozen RAW/RAM adapter 口径；`input_type=raw_ram_bridge_feature_adapter`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`，每 epoch `676` steps | 0.1303 (e9) | 0.1303 | 0.8498 (e9) | 0.8498 | 基于 `0521_1542` 加 decoder-side feature adapter；best checkpoint 按 abs_rel 在 epoch 9 保存，last 为 epoch 9；last `rmse=7.9462`，`silog=0.2599`。 |
| `0522_1423...raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542` | STF train+test pseudo-label training；同 `0521_1542` 为 DAv2-S frozen RAW/RAM adapter 口径；`input_type=raw_ram_rgb_bridge_feature_adapter`，feature keys `x_cat/ffm_mid/x3`，decoder fusion `path_4/path_3/path_2`，每 epoch `676` steps | 0.1313 (e9) | 0.1313 | 0.8468 (e9) | 0.8468 | 基于 `0521_1542` 加 RamCore3/x3 decoder-side feature adapter；best checkpoint 按 abs_rel 在 epoch 9 保存，last 为 epoch 9；last `rmse=8.0112`，`silog=0.2608`。 |
| `0521_0112...raw_ram_rgb_bnclean_identity_decoder_e5` | STF train+test pseudo-label training；同 `0521_0012`，但 `dav2_train_mode=decoder`；每 epoch `676` steps | 0.1346 (e4) | 0.1346 | 0.8394 (e3) | 0.8392 | RAW/RAM path 相同，额外解冻 DAv2-S decoder；best checkpoint 按 abs_rel 在 epoch 4 保存，last 为 epoch 4；last `rmse=8.1073`，`silog=0.2642`。 |
| `0521_0522...identity_lora_decoder_e10` | STF train+test pseudo-label training；`input_type=raw_ram_rgb_lora`，identity BN-clean RAW/RAM path；val `808`，`stf_repeat=7`，每 epoch `676` steps；DAv2-S decoder + LoRA | 0.1303 (e8) | 0.1305 | 0.8507 (e9) | 0.8507 | identity RAW/RAM + LoRA decoder；best checkpoint 按 abs_rel 在 epoch 8 保存，last 为 epoch 9；last `rmse=7.9874`，`silog=0.2608`。 |
| `0521_0656...identity_full_lrd09_e10` | STF train+test pseudo-label training；`input_type=raw_ram_rgb`，identity BN-clean RAW/RAM path；val `808`，`stf_repeat=7`，每 epoch `676` steps；full finetune，`backbone_layer_decay=0.9` | **0.1280 (e4)** | 0.1287 | **0.8574 (e7)** | 0.8561 | identity RAW/RAM + full finetune；best checkpoint 按 abs_rel 在 epoch 4 保存，last 为 epoch 9；last `rmse=7.9307`，`silog=0.2592`。 |
| `0521_0835...bridge_lora_decoder_e10` | STF train+test pseudo-label training；`input_type=raw_ram_rgb_bridge_lora`，bridge layers `[2,5,8,11]`，keys `x_cat/ffm_mid/x3`；val `808`，每 epoch `676` steps；DAv2-S decoder + LoRA | 0.1294 (e1) | 0.1298 | 0.8512 (e9) | 0.8512 | bridge RAW/RAM + LoRA decoder；best checkpoint 按 abs_rel 在 epoch 1 保存，last 为 epoch 9；last `rmse=7.9542`，`silog=0.2600`。 |
| `0521_1606...feature_adapter_lora_decoder_e10_from_0521_0835`（4ch RAW） | STF train+test pseudo-label training；`input_type=raw_ram_bridge_feature_adapter_lora`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`；val `808`，每 epoch `676` steps；DAv2-S decoder + LoRA + bridge + decoder-side feature adapter | 0.1293 (e2) | 0.1294 | 0.8526 (e9) | 0.8526 | 186 run；follows `0521_0835` setting but no `resume_from`/`bridge_init_from`；best checkpoint 按 abs_rel 在 epoch 2 保存，last 为 epoch 9；last `rmse=7.9246`，`silog=0.2596`。 |
| `0521_1004...bridge_full_lrd09_e10` | STF train+test pseudo-label training；`input_type=raw_ram_rgb_bridge`，bridge layers `[2,5,8,11]`，keys `x_cat/ffm_mid/x3`；val `808`，每 epoch `676` steps；full finetune，`backbone_layer_decay=0.9` | 0.1282 (e4) | 0.1288 | 0.8561 (e7) | 0.8557 | bridge RAW/RAM + full finetune；best checkpoint 按 abs_rel 在 epoch 4 保存，last 为 epoch 9；last `rmse=7.9349`，`silog=0.2593`。 |
| `0521_1606...feature_adapter_full_lrd09_e10_from_0521_1004`（4ch RAW） | STF train+test pseudo-label training；`input_type=raw_ram_bridge_feature_adapter`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`；val `808`，每 epoch `676` steps；full finetune，`backbone_layer_decay=0.9`，bridge + decoder-side feature adapter | **0.1279 (e5)** | 0.1285 | **0.8581 (e7)** | 0.8570 | 186 run；follows `0521_1004` setting but no `resume_from`/`bridge_init_from`；best checkpoint 按 abs_rel 在 epoch 5 保存，last 为 epoch 9；last `rmse=7.9135`，`silog=0.2589`。 |
| `0521_2217...feature_adapter_full_no_lrd_e10_from_0521_1606`（4ch RAW） | STF train+test pseudo-label training；`input_type=raw_ram_bridge_feature_adapter`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`；val `808`，每 epoch `676` steps；full finetune，`backbone_layer_decay=1.0` / no LRD，bridge + decoder-side feature adapter | 0.1280 (e7) | 0.1286 | 0.8576 (e7) | 0.8565 | local no-LRD ablation of the `0521_1606` full feature-adapter setting；best checkpoint 按 abs_rel 在 epoch 7 保存，last 为 epoch 9；last `rmse=7.9177`，`silog=0.2588`。 |

### 1.4 DA3 pseudo training runs

| Experiment | Training data / schedule | abs_rel best (epoch) | abs_rel last/current | d1 best (epoch) | d1 last/current | notes |
|---|---|---:|---:|---:|---:|---|
| `0521_1137...pseudoda3_sparse_metric_rgb_lora_decoder_e5` | STF train+test DA3 sparse metric pseudo-label training；`input_type=rgb_lora`，val `808`，`stf_repeat=7`，每 epoch `676` steps；DAv2-S decoder + LoRA | **0.1327 (e0)** | 0.1364 | **0.8498 (e0)** | 0.8458 | train target 换成 DA3 mono large sparse metric pseudo；best checkpoint 按 abs_rel 在 epoch 0 保存，last 为 epoch 4；last `rmse=8.2620`，`silog=0.2751`。 |
| `0521_1308...pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10` | STF train+test DA3 sparse metric pseudo-label training；`input_type=raw_ram_rgb_lora`，identity BN-clean RAW/RAM path；val `808`，`stf_repeat=7`，每 epoch `676` steps；DAv2-S decoder + LoRA | 0.1359 (e0) | 0.1396 | 0.8391 (e6) | 0.8376 | DA3 target + identity RAW/RAM LoRA decoder；best checkpoint 按 abs_rel 在 epoch 0 保存，last 为 epoch 9；last `rmse=8.3460`，`silog=0.2779`。 |

### `0521_0133...rgb_decoder_e5` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1286 | 7.9180 | 0.2576 | 0.8577 | 0.9327 | 0.9606 |
| 0 | **0.1278** | **7.8987** | **0.2576** | **0.8576** | **0.9329** | **0.9612** |
| 1 | 0.1281 | 7.9042 | 0.2580 | 0.8565 | 0.9328 | **0.9612** |
| 2 | 0.1280 | 7.9159 | 0.2584 | 0.8567 | 0.9327 | 0.9611 |
| 3 | 0.1289 | 7.9419 | 0.2592 | 0.8562 | 0.9324 | 0.9609 |
| 4 | 0.1287 | 7.9351 | 0.2591 | 0.8562 | 0.9324 | 0.9609 |

说明：

- `init/pretrain` 行来自 `pretrain_eval.json`，checkpoint source 为 `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth`；指标表 best 只按训练 epoch 0-4 统计。
- 与 DAv2-S RGB 直推 baseline 相比，decoder-only RGB control 的 best abs_rel 从 `0.1287` 到 `0.1278`，但 last 回到 `0.1287`；d1 未超过同一 train pipeline 的 `init/pretrain=0.8577`。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0133_stf_train_test_pseudovitl_rgb_decoder_e5/last_epoch_model.pth`。

### `0521_0012...raw_ram_rgb_bnclean_identity_e5` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3202 | 0.4303 | 0.5218 | 0.7651 | 0.8800 |
| 0 | 0.1437 | 8.4243 | 0.2704 | 0.8219 | 0.9176 | 0.9560 |
| 1 | 0.1407 | 8.3619 | 0.2686 | 0.8281 | 0.9203 | 0.9567 |
| 2 | 0.1401 | 8.3148 | 0.2673 | 0.8294 | 0.9215 | 0.9573 |
| 3 | 0.1392 | 8.2893 | **0.2672** | 0.8317 | 0.9222 | 0.9575 |
| 4 | **0.1388** | **8.2819** | 0.2673 | **0.8319** | **0.9225** | **0.9578** |

说明：

- `init/pretrain` 行是当前 RAW/RAM path 在 epoch 0 前的未训练前端评估，不替代预留的 RGB 或 RAW-preview direct baseline。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/last_epoch_model.pth`。

### `0521_1542...raw_ram_rgb_bnclean_bridge_ram_e10` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3243 | 0.4305 | 0.5216 | 0.7650 | 0.8799 |
| 0 | 0.1390 | 8.2374 | 0.2656 | 0.8323 | 0.9231 | 0.9582 |
| 1 | 0.1358 | 8.1511 | 0.2639 | 0.8384 | 0.9259 | 0.9592 |
| 2 | 0.1345 | 8.1055 | 0.2633 | 0.8413 | 0.9272 | 0.9595 |
| 3 | 0.1335 | 8.0785 | 0.2626 | 0.8426 | 0.9277 | 0.9599 |
| 4 | 0.1334 | 8.0735 | 0.2628 | 0.8430 | 0.9277 | 0.9598 |
| 5 | 0.1331 | 8.0625 | 0.2627 | 0.8442 | 0.9279 | 0.9598 |
| 6 | 0.1329 | 8.0516 | 0.2622 | 0.8445 | 0.9281 | 0.9600 |
| 7 | **0.1327** | **8.0364** | **0.2619** | 0.8448 | 0.9282 | **0.9602** |
| 8 | 0.1328 | 8.0441 | **0.2619** | 0.8450 | 0.9282 | 0.9601 |
| 9 | 0.1328 | 8.0426 | 0.2621 | **0.8453** | **0.9284** | **0.9602** |

说明：

- 与 `0521_0012` 同属 DAv2-S frozen RAW/RAM adapter 口径：STF pseudo target、STF val split、sparse backend、`raw_ram_rgb_tail=identity`、DAv2-S frozen 均一致。
- 主要对照差异是引入 bridge interface 并训练 bridge/RAM adapter：`bridge_feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，log 中 `trainable_params=276075`；另一个差异是训练从 e5 延长到 e10。两者实际 loss 口径同为 SSI。
- best checkpoint 按 abs_rel 在 epoch 7 保存；last 为 epoch 9。d1 best 出现在 epoch 9。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10/last_epoch_model.pth`。

### `0522_0137...raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1458 | 8.3949 | 0.2647 | 0.8125 | 0.9126 | 0.9551 |
| 0 | 0.1337 | 8.0829 | 0.2620 | 0.8416 | 0.9256 | 0.9594 |
| 1 | 0.1329 | 8.0397 | 0.2620 | 0.8448 | 0.9281 | 0.9600 |
| 2 | 0.1308 | 7.9675 | 0.2597 | 0.8481 | 0.9296 | 0.9609 |
| 3 | 0.1313 | 7.9865 | 0.2610 | 0.8484 | 0.9296 | 0.9608 |
| 4 | 0.1304 | 7.9591 | 0.2604 | 0.8494 | 0.9300 | 0.9609 |
| 5 | 0.1313 | 7.9676 | 0.2602 | 0.8477 | 0.9298 | 0.9610 |
| 6 | 0.1304 | 7.9523 | 0.2605 | 0.8495 | **0.9304** | 0.9609 |
| 7 | 0.1308 | 7.9606 | 0.2608 | 0.8486 | 0.9300 | 0.9610 |
| 8 | 0.1303 | 7.9511 | 0.2601 | 0.8497 | 0.9302 | 0.9609 |
| 9 | **0.1303** | **7.9462** | **0.2599** | **0.8498** | **0.9304** | **0.9611** |

说明：

- 该 run 放在 `0521_1542` 下面：两者同为 DAv2-S frozen RAW/RAM adapter 口径，训练/验证数据、STF pseudo target、sparse eval backend、`raw_ram_rgb_tail=identity` 均一致。
- 主要差异是把 bridge feature 从 `x3` 改为 `x4` 并加 decoder-side feature adapter：`bridge_feature_keys=['x_cat','ffm_mid','x4']`，decoder fusion `path_4,path_3,path_2`，`image_bridge=base_rgb+0.1*tanh(1x1_conv(x4))`，log 中 `trainable_params=871397`。
- best checkpoint 按 abs_rel 在 epoch 9 保存；last 为 epoch 9。rounded 后 epoch 8/9 的 abs_rel 都是 `0.1303`，但 checkpoint log 在 epoch 9 记录 `best_stf improved`。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting/train.log`。
- tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting.tmux.log`。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0522_0137_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting/last_epoch_model.pth`。

### `0522_1423...raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2902 | 12.3261 | 0.4308 | 0.5204 | 0.7648 | 0.8798 |
| 0 | 0.1358 | 8.1368 | 0.2629 | 0.8360 | 0.9256 | 0.9597 |
| 1 | 0.1333 | 8.0728 | 0.2620 | 0.8407 | 0.9272 | 0.9601 |
| 2 | 0.1322 | 8.0349 | **0.2605** | 0.8436 | 0.9284 | 0.9602 |
| 3 | 0.1327 | 8.0577 | 0.2621 | 0.8435 | 0.9278 | 0.9600 |
| 4 | 0.1325 | 8.0521 | 0.2623 | 0.8457 | 0.9286 | 0.9600 |
| 5 | 0.1320 | 8.0223 | 0.2607 | 0.8445 | 0.9284 | **0.9605** |
| 6 | 0.1317 | 8.0248 | 0.2618 | 0.8457 | 0.9287 | 0.9602 |
| 7 | 0.1316 | 8.0207 | 0.2612 | 0.8459 | 0.9286 | 0.9603 |
| 8 | 0.1317 | 8.0352 | 0.2610 | 0.8464 | 0.9288 | 0.9603 |
| 9 | **0.1313** | **8.0112** | 0.2608 | **0.8468** | **0.9290** | 0.9603 |

说明：

- 该 run 放在 `0521_1542` 和 `0522_0137` 下面：同为 DAv2-S frozen RAW/RAM adapter 口径，训练/验证数据、STF pseudo target、sparse eval backend、`raw_ram_rgb_tail=identity` 均一致。
- 主要差异是使用 RamCore3/x3 decoder-side feature adapter：`input_type=raw_ram_rgb_bridge_feature_adapter`，`bridge_feature_keys=['x_cat','ffm_mid','x3']`，decoder fusion `path_4,path_3,path_2`，`image_bridge=ramcore_bn_tanh25_no_clamp_no_imagenet_norm`，log 中 `trainable_params=860971`。
- best checkpoint 按 abs_rel 在 epoch 9 保存；last 为 epoch 9。d1/d2 best 也出现在 epoch 9，silog best 出现在 epoch 2，d3 best 出现在 epoch 5。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting/train.log`。
- tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting.tmux.log`。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0522_1423_stf_train_test_pseudovitl_raw_ram_rgb_bridge_feature_adapter_ram_e10_from_0521_1542_setting/last_epoch_model.pth`。

### `0521_0112...raw_ram_rgb_bnclean_identity_decoder_e5` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3202 | 0.4303 | 0.5218 | 0.7651 | 0.8800 |
| 0 | 0.1383 | 8.2229 | 0.2666 | 0.8317 | 0.9237 | 0.9585 |
| 1 | 0.1364 | 8.1587 | 0.2658 | 0.8366 | 0.9252 | 0.9589 |
| 2 | 0.1351 | 8.1284 | 0.2645 | 0.8391 | 0.9263 | 0.9592 |
| 3 | 0.1346 | **8.0942** | **0.2633** | **0.8394** | **0.9269** | **0.9597** |
| 4 | **0.1346** | 8.1073 | 0.2642 | 0.8392 | 0.9266 | 0.9594 |

说明：

- 与 `0521_0012` 使用同一 RAW/RAM 输入、STF pseudo-label target、STF val split 和 `best_metric=stf` 规则；差异是 `dav2_train_mode=decoder`，log 中 `trainable_params=2869416`。
- `train.log` 中 epoch 3/4 的 `abs_rel` rounded 后都为 `0.1346`，但 checkpoint log 在 epoch 4 记录 `best_stf improved` 并保存 best，因此指标表按 epoch 4 标注 best abs_rel。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0112_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_decoder_e5/last_epoch_model.pth`。

### `0521_0306...rgb_lora_decoder_e5` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1286 | 7.9181 | 0.2576 | 0.8577 | 0.9327 | 0.9606 |
| 0 | **0.1271** | **7.8926** | **0.2567** | **0.8590** | **0.9336** | **0.9616** |
| 1 | 0.1274 | 7.8961 | 0.2573 | 0.8578 | 0.9333 | **0.9616** |

说明：

- `rgb_lora` control：RGB input，DAv2-S decoder + LoRA tap blocks `(2,5,8,11)`；STF pseudo-label target、STF val split 和 `best_metric=stf` 规则同 `0521_0133`。
- 该 run 未完整跑完：`train.log` 和 tmux log 都停在 epoch 2 的 `[EVAL][val] start`，没有 epoch 2 完整指标，也没有 `last_epoch_model.pth`。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/best_model.pth`。
- current checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0306_stf_train_test_pseudovitl_rgb_lora_decoder_e5/current_model.pth`，对应 epoch 1。

### `0521_0402...rgb_full_lrd09_e5` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1286 | 7.9181 | 0.2576 | 0.8577 | 0.9327 | 0.9606 |
| 0 | 0.1288 | 7.9178 | 0.2591 | 0.8548 | 0.9330 | 0.9612 |
| 1 | 0.1292 | 7.9622 | 0.2605 | 0.8528 | 0.9316 | 0.9612 |
| 2 | **0.1281** | **7.9056** | **0.2579** | **0.8581** | **0.9336** | **0.9616** |
| 3 | 0.1291 | 7.9547 | 0.2602 | 0.8574 | 0.9327 | 0.9610 |
| 4 | 0.1284 | 7.9202 | 0.2591 | **0.8581** | 0.9331 | 0.9613 |

说明：

- RGB full-finetune control：`input_type=rgb`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，`trainable_params=24785089`。
- best checkpoint 按 abs_rel 在 epoch 2 保存；last 为 epoch 4。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0402_stf_train_test_pseudovitl_rgb_full_lrd09_e5/last_epoch_model.pth`。

### `0521_0522...raw_ram_rgb_bnclean_identity_lora_decoder_e10` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3202 | 0.4303 | 0.5218 | 0.7651 | 0.8800 |
| 0 | 0.1315 | **7.9732** | 0.2606 | 0.8444 | 0.9291 | **0.9606** |
| 1 | 0.1305 | 7.9828 | **0.2605** | 0.8461 | 0.9295 | **0.9606** |
| 2 | 0.1311 | 8.0075 | 0.2612 | 0.8458 | 0.9290 | 0.9601 |
| 3 | 0.1321 | 8.0415 | 0.2633 | 0.8465 | 0.9289 | 0.9600 |
| 4 | 0.1315 | 8.0311 | 0.2625 | 0.8490 | 0.9293 | 0.9599 |
| 5 | 0.1315 | 7.9963 | 0.2619 | 0.8479 | 0.9293 | 0.9603 |
| 6 | 0.1312 | 8.0135 | 0.2616 | 0.8488 | 0.9293 | 0.9603 |
| 7 | 0.1309 | 7.9911 | 0.2611 | 0.8496 | 0.9297 | 0.9604 |
| 8 | **0.1303** | 7.9816 | **0.2605** | 0.8502 | **0.9301** | 0.9604 |
| 9 | 0.1305 | 7.9874 | 0.2608 | **0.8507** | 0.9300 | 0.9604 |

说明：

- 与 `0521_0112` 使用同一 identity BN-clean RAW/RAM path，差异是加入 LoRA tap blocks `(2,5,8,11)`，训练 DAv2-S decoder + LoRA，log 中 `trainable_params=2943144`。
- best checkpoint 按 abs_rel 在 epoch 8 保存；last 为 epoch 9。d1 best 出现在 epoch 9，因此指标表的 d1 best 与 abs_rel checkpoint epoch 不同。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0522_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10/last_epoch_model.pth`。

### `0521_0656...raw_ram_rgb_bnclean_identity_full_lrd09_e10` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3202 | 0.4303 | 0.5218 | 0.7651 | 0.8800 |
| 0 | 0.1301 | 7.9958 | 0.2606 | 0.8481 | 0.9301 | 0.9607 |
| 1 | 0.1310 | 8.0439 | 0.2618 | 0.8451 | 0.9285 | 0.9606 |
| 2 | 0.1300 | 7.9672 | 0.2592 | 0.8525 | 0.9316 | 0.9611 |
| 3 | 0.1293 | 7.9324 | 0.2596 | 0.8546 | 0.9321 | 0.9612 |
| 4 | **0.1280** | **7.8925** | 0.2594 | 0.8560 | 0.9321 | 0.9611 |
| 5 | 0.1294 | 7.9357 | 0.2588 | 0.8546 | 0.9320 | **0.9613** |
| 6 | 0.1293 | 7.9490 | 0.2596 | 0.8546 | 0.9320 | 0.9611 |
| 7 | 0.1284 | 7.9215 | **0.2582** | **0.8574** | **0.9327** | **0.9613** |
| 8 | 0.1291 | 7.9444 | 0.2601 | 0.8550 | 0.9316 | 0.9610 |
| 9 | 0.1287 | 7.9307 | 0.2592 | 0.8561 | 0.9321 | **0.9613** |

说明：

- identity BN-clean RAW/RAM + full finetune：`input_type=raw_ram_rgb`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，log 中 `trainable_params=24925992`。
- best checkpoint 按 abs_rel 在 epoch 4 保存；last 为 epoch 9。d1 best 出现在 epoch 7。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0656_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10/last_epoch_model.pth`。

### `0521_0835...raw_ram_rgb_bnclean_bridge_lora_decoder_e10` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3243 | 0.4305 | 0.5216 | 0.7650 | 0.8799 |
| 0 | 0.1308 | 7.9586 | 0.2598 | 0.8448 | 0.9295 | 0.9606 |
| 1 | **0.1294** | **7.9319** | **0.2585** | 0.8476 | 0.9304 | **0.9611** |
| 2 | 0.1300 | 7.9576 | 0.2593 | 0.8486 | 0.9302 | 0.9608 |
| 3 | 0.1316 | 8.0101 | 0.2627 | 0.8469 | 0.9292 | 0.9603 |
| 4 | 0.1306 | 7.9920 | 0.2609 | 0.8500 | 0.9302 | 0.9604 |
| 5 | 0.1308 | 7.9719 | 0.2609 | 0.8486 | 0.9300 | 0.9606 |
| 6 | 0.1304 | 7.9881 | 0.2609 | 0.8497 | 0.9302 | 0.9604 |
| 7 | 0.1304 | 7.9718 | 0.2606 | 0.8500 | 0.9302 | 0.9606 |
| 8 | 0.1298 | 7.9584 | 0.2598 | 0.8507 | **0.9307** | 0.9606 |
| 9 | 0.1298 | 7.9542 | 0.2600 | **0.8512** | 0.9306 | 0.9606 |

说明：

- bridge RAW/RAM + LoRA decoder：`bridge_feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，`dav2_train_mode=decoder`，log 中 `trainable_params=3078316`。
- best checkpoint 按 abs_rel 在 epoch 1 保存；last 为 epoch 9。d1 best 出现在 epoch 9。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_0835_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10/last_epoch_model.pth`。

### `0521_1606...raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1458 | 8.3949 | 0.2647 | 0.8126 | 0.9126 | 0.9551 |
| 0 | 0.1315 | 8.0118 | 0.2615 | 0.8446 | 0.9275 | 0.9601 |
| 1 | 0.1302 | 7.9473 | 0.2596 | 0.8491 | 0.9304 | 0.9608 |
| 2 | **0.1293** | 7.9320 | **0.2584** | 0.8493 | 0.9300 | 0.9607 |
| 3 | 0.1312 | 7.9864 | 0.2621 | 0.8486 | 0.9295 | 0.9605 |
| 4 | 0.1295 | 7.9446 | 0.2604 | 0.8523 | 0.9305 | 0.9609 |
| 5 | 0.1309 | 7.9469 | 0.2605 | 0.8493 | 0.9299 | 0.9609 |
| 6 | 0.1299 | 7.9480 | 0.2603 | 0.8515 | 0.9308 | 0.9608 |
| 7 | 0.1299 | 7.9280 | 0.2603 | 0.8515 | 0.9306 | 0.9609 |
| 8 | 0.1296 | 7.9348 | 0.2601 | 0.8523 | 0.9309 | 0.9610 |
| 9 | 0.1294 | **7.9246** | 0.2596 | **0.8526** | **0.9311** | **0.9611** |

说明：

- feature-adapter LoRA run：`input_type=raw_ram_bridge_feature_adapter_lora`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`，`dav2_train_mode=decoder`，LoRA tap blocks `(2,5,8,11)`，log 中 `trainable_params=3673638`。
- 该 run 只沿用 `0521_0835` 的设定，不从本机 checkpoint 继续训练；`resume_from=None`，没有 `bridge_init_from`。
- best checkpoint 按 abs_rel 在 epoch 2 保存；last 为 epoch 9。d1/d2/d3 best 出现在 epoch 9。
- source log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/train.log`。
- tmux log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1.tmux.log`。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting_retry1/last_epoch_model.pth`。

### `0521_1004...raw_ram_rgb_bnclean_bridge_full_lrd09_e10` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3243 | 0.4305 | 0.5216 | 0.7650 | 0.8799 |
| 0 | 0.1318 | 8.0400 | 0.2614 | 0.8445 | 0.9295 | 0.9605 |
| 1 | 0.1309 | 8.0395 | 0.2622 | 0.8470 | 0.9288 | 0.9604 |
| 2 | 0.1296 | 7.9343 | 0.2587 | 0.8534 | 0.9319 | 0.9611 |
| 3 | 0.1311 | 7.9935 | 0.2616 | 0.8527 | 0.9311 | 0.9607 |
| 4 | **0.1282** | 7.9101 | 0.2595 | 0.8552 | 0.9318 | 0.9609 |
| 5 | 0.1286 | **7.9057** | **0.2583** | 0.8551 | 0.9319 | **0.9615** |
| 6 | 0.1293 | 7.9493 | 0.2600 | 0.8538 | 0.9317 | 0.9611 |
| 7 | 0.1289 | 7.9335 | 0.2589 | **0.8561** | **0.9322** | 0.9611 |
| 8 | 0.1293 | 7.9457 | 0.2602 | 0.8545 | 0.9313 | 0.9609 |
| 9 | 0.1288 | 7.9349 | 0.2593 | 0.8557 | 0.9321 | 0.9611 |

说明：

- bridge RAW/RAM + full finetune：`bridge_feature_keys=['x_cat','ffm_mid','x3']`，`bridge_layers=[2,5,8,11]`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，log 中 `trainable_params=25061164`。
- best checkpoint 按 abs_rel 在 epoch 4 保存；last 为 epoch 9。d1 best 出现在 epoch 7。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1004_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10/last_epoch_model.pth`。

### `0521_1606...raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1458 | 8.3948 | 0.2647 | 0.8126 | 0.9126 | 0.9551 |
| 0 | 0.1301 | 7.9156 | 0.2597 | 0.8489 | 0.9301 | 0.9610 |
| 1 | 0.1311 | 8.0348 | 0.2621 | 0.8457 | 0.9290 | 0.9606 |
| 2 | 0.1294 | 7.9356 | 0.2592 | 0.8535 | 0.9317 | 0.9610 |
| 3 | 0.1303 | 7.9611 | 0.2615 | 0.8539 | 0.9318 | 0.9608 |
| 4 | 0.1291 | 7.8926 | 0.2590 | 0.8559 | 0.9326 | 0.9613 |
| 5 | **0.1279** | **7.8756** | **0.2574** | 0.8573 | **0.9332** | **0.9615** |
| 6 | 0.1291 | 7.9210 | 0.2595 | 0.8560 | 0.9323 | 0.9611 |
| 7 | 0.1280 | 7.9049 | 0.2586 | **0.8581** | 0.9327 | 0.9611 |
| 8 | 0.1291 | 7.9289 | 0.2598 | 0.8558 | 0.9318 | 0.9610 |
| 9 | 0.1285 | 7.9135 | 0.2589 | 0.8570 | 0.9325 | 0.9612 |

说明：

- feature-adapter full finetune run：`input_type=raw_ram_bridge_feature_adapter`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`，`dav2_train_mode=full`，`backbone_layer_decay=0.9`，log 中 `trainable_params=25656486`。
- 该 run 只沿用 `0521_1004` 的设定，不从本机 checkpoint 继续训练；`resume_from=None`，没有 `bridge_init_from`。
- best checkpoint 按 abs_rel 在 epoch 5 保存；last 为 epoch 9。d1 best 出现在 epoch 7。
- source log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/train.log`。
- tmux log on 186：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1.tmux.log`。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1606_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting_retry1/last_epoch_model.pth`。

### `0521_2217...raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1458 | 8.3949 | 0.2647 | 0.8125 | 0.9126 | 0.9551 |
| 0 | 0.1327 | 8.0224 | 0.2618 | 0.8419 | 0.9284 | 0.9601 |
| 1 | 0.1313 | 8.0052 | 0.2621 | 0.8439 | 0.9281 | 0.9608 |
| 2 | 0.1309 | 7.9433 | 0.2599 | 0.8506 | 0.9312 | 0.9612 |
| 3 | 0.1315 | 8.0455 | 0.2634 | 0.8465 | 0.9293 | 0.9605 |
| 4 | 0.1290 | 7.9060 | 0.2588 | 0.8530 | 0.9322 | 0.9613 |
| 5 | 0.1285 | **7.8782** | **0.2581** | 0.8530 | 0.9318 | **0.9614** |
| 6 | 0.1288 | 7.9016 | 0.2587 | 0.8542 | 0.9321 | 0.9612 |
| 7 | **0.1280** | 7.8954 | **0.2581** | **0.8576** | **0.9330** | 0.9612 |
| 8 | 0.1291 | 7.9332 | 0.2599 | 0.8544 | 0.9316 | 0.9610 |
| 9 | 0.1286 | 7.9177 | 0.2588 | 0.8565 | 0.9326 | 0.9611 |

说明：

- feature-adapter full no-LRD ablation：`input_type=raw_ram_bridge_feature_adapter`，feature keys `x_cat/ffm_mid/x4`，decoder fusion `path_4/path_3/path_2`，`dav2_train_mode=full`，`backbone_layer_decay=1.0`，log 中 `trainable_params=25656486`。
- 该 run 与 `0521_1606` full feature-adapter setting 同口径，但去掉 layer-wise LR decay；`resume_from=None`，没有 `bridge_init_from`。
- best checkpoint 按 abs_rel 在 epoch 7 保存；last 为 epoch 9。d1 best 同样出现在 epoch 7。
- source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting/train.log`。
- tmux log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting.tmux.log`。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_2217_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_no_lrd_e10_from_0521_1606_setting/last_epoch_model.pth`。

### `0521_1137...pseudoda3_sparse_metric_rgb_lora_decoder_e5` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.1286 | 7.9181 | 0.2576 | 0.8577 | 0.9327 | 0.9606 |
| 0 | **0.1327** | **8.1292** | **0.2713** | **0.8498** | 0.9278 | 0.9581 |
| 1 | 0.1343 | 8.1857 | 0.2728 | 0.8493 | **0.9286** | **0.9584** |
| 2 | 0.1355 | 8.2376 | 0.2741 | 0.8462 | 0.9271 | 0.9580 |
| 3 | 0.1371 | 8.3141 | 0.2766 | 0.8451 | 0.9267 | 0.9573 |
| 4 | 0.1364 | 8.2620 | 0.2751 | 0.8458 | 0.9273 | 0.9579 |

说明：

- 与 `0521_0306` 相同 RGB LoRA/decoder training surface，但 train target 换成 `da3_pseudo_sparse_metric`，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`。
- 指标表仍记录 STF val ground-truth eval 结果，因此可作为同一 eval 口径下的 target-ablation 行；该 target 下 epoch 0 后 val 指标整体变差。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1137_stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5/last_epoch_model.pth`。

### `0521_1308...pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10` per-epoch pointer

| epoch | abs_rel | rmse | silog | d1 | d2 | d3 |
|---:|---:|---:|---:|---:|---:|---:|
| init/pretrain | 0.2898 | 12.3202 | 0.4303 | 0.5218 | 0.7651 | 0.8800 |
| 0 | **0.1359** | **8.1910** | **0.2735** | 0.8372 | 0.9232 | 0.9571 |
| 1 | 0.1360 | 8.2067 | 0.2736 | 0.8386 | 0.9243 | **0.9575** |
| 2 | 0.1388 | 8.3408 | 0.2781 | 0.8345 | 0.9212 | 0.9560 |
| 3 | 0.1404 | 8.3732 | 0.2793 | 0.8344 | 0.9216 | 0.9558 |
| 4 | 0.1398 | 8.3869 | 0.2797 | 0.8375 | 0.9234 | 0.9557 |
| 5 | 0.1392 | 8.3224 | 0.2777 | 0.8369 | 0.9230 | 0.9565 |
| 6 | 0.1388 | 8.3028 | 0.2766 | **0.8391** | **0.9249** | 0.9570 |
| 7 | 0.1392 | 8.3084 | 0.2763 | 0.8387 | 0.9246 | 0.9572 |
| 8 | 0.1394 | 8.3633 | 0.2779 | 0.8374 | 0.9235 | 0.9563 |
| 9 | 0.1396 | 8.3460 | 0.2779 | 0.8376 | 0.9234 | 0.9566 |

说明：

- 与 `0521_0522` 相同 identity BN-clean RAW/RAM + LoRA decoder training surface，差异是 train target 换成 `da3_pseudo_sparse_metric`，并启用 `lambda_grad=2.0`、`grad_scales=4`，log 中 `trainable_params=2943144`。
- 指标表仍记录 STF val ground-truth eval 结果，因此可作为 RAW/RAM 输入下的 target-ablation 行；该 target 下 best abs_rel 出现在 epoch 0，后续 epoch 整体回退。
- `pretrain_eval.json` 记录了 split `stf_val` 的 full metrics。
- best checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/best_model.pth`。
- last checkpoint：`/mnt/drive/3333_raw/0000_exp_ckpt/0521_1308_stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10/last_epoch_model.pth`。
