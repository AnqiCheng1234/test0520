# ROD Night RGB / RAW baseline 公平性汇总

日期：2026-06-08

本文记录 `/home/caq/6666_raw/dav2_raw_0603` 当前 ROD-night formal validation 结果。当前主表覆盖本机 9 个 run 与 186 机器上 2 个复刻 run，共 11 个 run，均已完成。

写作规约：指标表只放 formal eval log / json 中的数字；仍在运行的 run 不把 partial epoch 指标混入主指标表。

指标表标注规约：`abs_rel` 越低越好，`d1` 越高越好。本文的 ROD 指标是相对 DAv2-L pseudo inverse label 的 `inverse_relative` proxy，一律不是 metric depth benchmark 指标；数值只能在同一 ROD-night eval protocol 下比较。

## 0. 协议速查 / 数据源规则

当前 formal runs：

- `0604_0752_rod_night_studentrgb_dav2s_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0604_0752_rod_night_studentrgb_dav2s_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0604_0752_rod_night_studentrgb_dav2s_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0604_0752_rod_night_studentrgb_decoder_e10.queue.log`
  - label：Student RGB decoder-only baseline
  - stage：`rod_only`
  - input：`rgb`，`dataset_family=rod_raw_student_rgb`，`dataset_input_mode=raw24_student_rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`
  - path：ROD raw24 full frame -> `student_dark_degreen_v1` RGB (`wp=99.9`, `gamma=0.9`, gains `(1.08,0.95,1.10)`) -> native `928x1440` -> crop `[3,512,960]` -> DAv2-S；teacher label 为 `teacher_bright_degreen_v1` -> DAv2-L pseudo inverse-relative。
  - trainable：DAv2-S decoder only，`lr=1e-5`，无 LoRA
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val`，对 pseudo inverse target 做 affine align 后在 inverse space 计算 `abs_rel/rmse/silog/d1/d2/d3`
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 2；`last_epoch_model.pth` 为 epoch 9
- `0605_0139_rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0605_0139_rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_0139_rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_0139_rod_night_studentrgb_encoder_budget_e10.queue.log`
  - label：B LoRA tap r8/a16 + decoder
  - stage：`rod_only`
  - input：`rgb`，`dataset_family=rod_raw_student_rgb`，`dataset_input_mode=raw24_student_rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`
  - path：ROD raw24 full frame -> `student_dark_degreen_v1` RGB (`wp=99.9`, `gamma=0.9`, gains `(1.08,0.95,1.10)`) -> native `928x1440` -> crop `[3,512,960]` -> DAv2-S；teacher label 为 `teacher_bright_degreen_v1` -> DAv2-L pseudo inverse-relative。
  - trainable：DAv2-S decoder + LoRA tap layers `[2,5,8,11]`，`rank=8`，`alpha=16`，`lr=1e-5`，`lora_lr=5e-5`
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val`，对 pseudo inverse target 做 affine align 后在 inverse space 计算 `abs_rel/rmse/silog/d1/d2/d3`
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 2；`last_epoch_model.pth` 为 epoch 9
- `0605_0730_rod_night_studentrgb_dav2s_backbone_ld09_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0605_0730_rod_night_studentrgb_dav2s_backbone_ld09_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_0730_rod_night_studentrgb_dav2s_backbone_ld09_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_0139_rod_night_studentrgb_encoder_budget_e10.queue.log`
  - label：C full backbone LLRD 0.9 + decoder
  - stage：`rod_only`
  - input：`rgb`，`dataset_family=rod_raw_student_rgb`，`dataset_input_mode=raw24_student_rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`
  - path：同 `0605_0139`，ROD raw24 -> `student_dark_degreen_v1` RGB -> DAv2-S
  - trainable：DAv2-S full backbone + decoder，`backbone_layer_decay=0.9`，`lr=1e-5`，无 LoRA
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 3；`last_epoch_model.pth` 为 epoch 9
- `0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e5.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_1332_rod_night_raw_ram_v1_e5.queue.log`
  - label：R1 raw RAM identity + decoder
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：ROD raw24 -> packed RGGB RAW4 `[R,Gr,Gb,B]` -> crop `[4,512,960]` -> `[R,(Gr+Gb)/2,B]` preview / RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S
  - trainable：raw front-end + DAv2-S decoder，`lr=1e-5`，`raw_front_end_lr=5e-5`，无 LoRA
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 2；`last_epoch_model.pth` 为 epoch 4
- `0606_1348_repl0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e10` on 186
  - source log：`186:/home/a5000/6666_raw/dav2_raw_0603/finetune_stf/exp/0606_1348_repl0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e10/train.log`
  - tmux log：`186:/home/a5000/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_1348_repl0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e10.tmux.log`
  - queue log：`186:/home/a5000/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_1348_repl0605_1332_rod_night_rawram3_identity_dav2s_ram_decoder_e10_gpu0.queue.log`
  - queue status：`[END] 2026-06-07T07:39:08+08:00 status=0`
  - label：R1-repl raw RAM identity + decoder, e10
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：同 `0605_1332`，ROD raw24 -> packed RGGB RAW4 -> RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S
  - trainable：raw front-end + DAv2-S decoder，`lr=1e-5`，`raw_front_end_lr=5e-5`，无 LoRA，epochs=10
  - relation to local run：对应本机 `0605_1332`；实验语义参数一致，显式差异为 `epochs=10` vs `epochs=5`。远端 `pretrained_from` 路径不同，但 checkpoint SHA256 与本机一致。
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 2；`last_epoch_model.pth` 为 epoch 9
- `0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e5.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_1332_rod_night_raw_ram_v1_e5.queue.log`
  - label：R2 raw RAM identity + LoRA tap r8/a16 + decoder
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：同 `0605_1332`，ROD raw24 -> packed RGGB RAW4 -> RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S
  - trainable：raw front-end + DAv2-S decoder + LoRA tap layers `[2,5,8,11]`，`rank=8`，`alpha=16`，`lr=1e-5`，`raw_front_end_lr=5e-5`，`lora_lr=5e-5`
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 1；`last_epoch_model.pth` 为 epoch 4
- `0606_1348_repl0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e10` on 186
  - source log：`186:/home/a5000/6666_raw/dav2_raw_0603/finetune_stf/exp/0606_1348_repl0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e10/train.log`
  - tmux log：`186:/home/a5000/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_1348_repl0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e10.tmux.log`
  - queue log：`186:/home/a5000/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_1348_repl0605_1844_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e10_gpu1.queue.log`
  - queue status：`[END] 2026-06-07T07:39:08+08:00 status=0`
  - label：R2-repl raw RAM identity + LoRA tap r8/a16 + decoder, e10
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：同 `0605_1844`，ROD raw24 -> packed RGGB RAW4 -> RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S
  - trainable：raw front-end + DAv2-S decoder + LoRA tap layers `[2,5,8,11]`，`rank=8`，`alpha=16`，`lr=1e-5`，`raw_front_end_lr=5e-5`，`lora_lr=5e-5`，epochs=10
  - relation to local run：对应本机 `0605_1844`；实验语义参数一致，显式差异为 `epochs=10` vs `epochs=5`。远端 `pretrained_from` 路径不同，但 checkpoint SHA256 与本机一致。
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 3；`last_epoch_model.pth` 为 epoch 9
- `0605_2346_rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0605_2346_rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_2346_rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_1332_rod_night_raw_ram_v1_e5.queue.log`
  - queue status：`[END] 2026-06-06T02:50:27+08:00 status=0`
  - label：RGB-ref student RGB low-lr backbone LLRD 0.9
  - stage：`rod_only`
  - input：`rgb`，`dataset_family=rod_raw_student_rgb`，`dataset_input_mode=raw24_student_rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`
  - path：同 `0605_0139`，ROD raw24 -> `student_dark_degreen_v1` RGB -> DAv2-S
  - trainable：DAv2-S full backbone + decoder，`backbone_layer_decay=0.9`，`lr=1e-6`，无 LoRA，epochs=5
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 1；`last_epoch_model.pth` 为 epoch 4
- `0606_0330_rod_night_rawram3_identity_dav2s_ram_backbone_ld09_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0606_0330_rod_night_rawram3_identity_dav2s_ram_backbone_ld09_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_0330_rod_night_rawram3_identity_dav2s_ram_backbone_ld09_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_0230_rod_night_rawram3_backbone_e10.queue.log`
  - queue status：`[END] 2026-06-06T14:31:54+08:00 status=0`
  - label：R3 raw RAM identity + full backbone LLRD 0.9 + decoder
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：同 `0605_1332`，ROD raw24 -> packed RGGB RAW4 -> RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S
  - trainable：raw front-end + DAv2-S full backbone + decoder，`backbone_layer_decay=0.9`，`lr=1e-5`，`raw_front_end_lr=5e-5`，无 LoRA，epochs=10
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 3；`last_epoch_model.pth` 为 epoch 9
- `0606_1431_rod_night_rawram3_identity_dav2s_ram_backbone_lowlr_ld09_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0606_1431_rod_night_rawram3_identity_dav2s_ram_backbone_lowlr_ld09_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_1431_rod_night_rawram3_identity_dav2s_ram_backbone_lowlr_ld09_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_0230_rod_night_rawram3_backbone_e10.queue.log`
  - queue status：`[END] 2026-06-07T01:17:46+08:00 status=0`
  - label：R4 raw RAM identity + low-lr full backbone LLRD 0.9 + decoder
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：同 `0605_1332`，ROD raw24 -> packed RGGB RAW4 -> RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S
  - trainable：raw front-end + DAv2-S full backbone + decoder，`backbone_layer_decay=0.9`，backbone / decoder `lr=1e-6`，`raw_front_end_lr=5e-5`，无 LoRA，epochs=10
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` abs_rel 保存，best epoch 为 epoch 7；`last_epoch_model.pth` 为 epoch 9
- `0608_1444_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0608_1444_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_1444_rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_1444_rod_night_rawram3_lora_bridge_fa_e10.queue.log`
  - queue status：`[END] 2026-06-08T17:31:09+08:00 status=0`
  - label：R5 raw RAM identity + LoRA tap r8/a16 + bridge + decoder feature adapter
  - stage：`rod_only`
  - input：`raw4`，`dataset_family=rod_raw`，`dataset_input_mode=raw_ram`，`front_end=raw_to_base_rgb_ram3`，`model_input_tensor=raw`，`raw_storage_format=none`
  - path：同 `0605_1844`，ROD raw24 -> packed RGGB RAW4 -> RamCore3 -> `raw_ram_rgb_tail=identity` -> raw feature bridge / decoder feature adapter -> DAv2-S
  - trainable：raw front-end + bridge + decoder feature adapter + DAv2-S decoder + LoRA tap layers `[2,5,8,11]`，`rank=8`，`alpha=16`，`lr=1e-5`，`raw_front_end_lr=5e-5`，`bridge_lr=5e-5`，adapter lr `5e-5`，`lora_lr=5e-5`，epochs=10
  - bridge/adapter：`bridge=raw_feature_bridge`，`decoder_feature_adapter=raw_feature_adapter`，`bridge_source=ram_core`，`bridge_feature_source_channels=x3`，`adapter_feature_source_channels=x3`，`bridge_feature_keys=[x_cat,ffm_mid,x3]`，`feature_adapter_keys=[x_cat,ffm_mid,x3]`
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：`best_model.pth` 按 `rod` metric 保存，best epoch 为 epoch 4；`last_epoch_model.pth` 为 epoch 9

baseline/control 行：

- RGB-path official DAv2-S init：`pretrain_eval.json` 中 ROD night val 约 `abs_rel=13.574-13.579, d1=0.7617`。这是 raw24 -> `student_dark_degreen_v1` RGB 输入下的初始化模型表现。
- RAW-RAM-path init：`pretrain_eval.json` 中 ROD night val 约 `abs_rel=130.38-130.71, d1=0.2697-0.2722`。该数值包含 raw-RAM front-end 初始状态；`0608_1444` 还包含 bridge / feature adapter 初始状态，不是和 RGB-path DAv2-S init 同义的公共 D0 baseline。

run-row 数据源优先级：

1. `finetune_stf/exp/<run>/pretrain_eval.json`。
2. `finetune_stf/exp/<run>/train.log`。
3. `finetune_stf/logs/<run>.tmux.log` 与 queue log 只用于完成状态和队列上下文。
4. `finetune_stf/exp/<run>/config.json`、`resolved_config.json` 只用于确认配置。

## 1. ROD Night Val

### 1.1 Overall metrics

| Experiment | Method / input | init abs_rel / d1 | final abs_rel best (epoch) | final abs_rel last | final d1 best (epoch) | final d1 last | best abs_rel delta vs init | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `0604_0752...studentrgb...decoder_e10` | Student RGB；DAv2-S decoder-only baseline | 13.5764 / 0.7617 | 9.1288 (e2) | 10.3904 | 0.8015 (e7) | 0.7995 | -4.4476 | 0604 baseline；decoder-only 可以优于 init，但弱于后续 LoRA / full-backbone 设置。 |
| `0605_0139...lora_tap...decoder_e10` | Student RGB；DAv2-S decoder + LoRA tap r8/a16 | 13.5764 / 0.7617 | 8.3252 (e2) | 10.7638 | 0.8251 (e7) | 0.8216 | -5.2512 | RGB LoRA 明显优于 init，但 abs_rel best 出现在早期；last 已回退。 |
| `0605_0730...backbone_ld09_decoder_e10` | Student RGB；full backbone LLRD 0.9 + decoder | 13.5786 / 0.7617 | 5.8880 (e3) | 8.0998 | 0.8483 (e8) | 0.8471 | -7.6906 | RGB-path 中 abs_rel / d1 最强；best abs_rel 与 best d1 不在同一 epoch。 |
| `0605_2346...studentrgb...backbone_lowlr_ld09_e5` | Student RGB；full backbone LLRD 0.9 + decoder，`lr=1e-6` | 13.5739 / 0.7617 | 8.4721 (e1) | 9.9882 | 0.8274 (e1) | 0.8231 | -5.1018 | low-lr full-backbone 优于 decoder-only baseline，但 abs_rel 略弱于 RGB LoRA，明显弱于 `0605_0730`。 |
| `0605_1332...rawram3_identity...decoder_e5` | RAW4 -> RamCore3 identity；raw front-end + decoder | 130.3805 / 0.2722 | 5.9413 (e2) | 6.7567 | 0.8303 (e4) | 0.8303 | -124.4392 | abs_rel 接近 `0605_0730`，但 init 不是公共 RGB D0；last 优于 R2 raw LoRA。 |
| `0606_1348@186...repl0605_1332...decoder_e10` | RAW4 -> RamCore3 identity；raw front-end + decoder，e10 repl | 130.3882 / 0.2721 | 5.5653 (e2) | 6.6496 | 0.8412 (e6) | 0.8397 | -124.8229 | 186 上 `0605_1332` 的 10-epoch 复刻；语义配置一致，除 epochs / 路径外未发现差异；best / last 均优于本机 e5 R1。 |
| `0605_1844...rawram3_identity...lora_tap...decoder_e5` | RAW4 -> RamCore3 identity；raw front-end + decoder + LoRA tap | 130.3823 / 0.2722 | 7.3929 (e1) | 9.4693 | 0.8362 (e3) | 0.8287 | -122.9894 | raw-RAM + LoRA 的 abs_rel 弱于 raw-RAM decoder-only；d1 best 略高于 R1，但 last 回退。 |
| `0606_1348@186...repl0605_1844...lora_tap...decoder_e10` | RAW4 -> RamCore3 identity；raw front-end + decoder + LoRA tap，e10 repl | 130.3907 / 0.2721 | 6.8383 (e3) | 8.9307 | 0.8437 (e3) | 0.8342 | -123.5524 | 186 上 `0605_1844` 的 10-epoch 复刻；best 优于本机 e5 R2，但 last 仍回退，且弱于 e10 decoder-only repl。 |
| `0606_0330...rawram3_identity...backbone_ld09_decoder_e10` | RAW4 -> RamCore3 identity；raw front-end + full backbone LLRD 0.9 + decoder | 130.3823 / 0.2722 | **4.2575 (e3)** | **5.5728** | **0.8700 (e9)** | **0.8700** | -126.1248 | 当前 ROD-night best；best abs_rel 在 e3，d1 / last 在 e9，raw init 仍不等同 RGB D0。 |
| `0606_1431...rawram3_identity...backbone_lowlr_ld09_e10` | RAW4 -> RamCore3 identity；raw front-end + low-lr full backbone LLRD 0.9 + decoder | 130.3823 / 0.2722 | 6.4195 (e7) | 6.4649 | 0.8574 (e6) | 0.8562 | -123.9628 | backbone/decoder `lr=1e-6`，raw front-end 仍为 `5e-5`；弱于 `0606_0330`，last 接近 best。 |
| `0608_1444...rawram3_identity...lora_tap...bridge_feature_adapter...decoder_e10` | RAW4 -> RamCore3 identity；raw front-end + bridge + decoder feature adapter + decoder + LoRA tap | 130.7106 / 0.2697 | 5.6001 (e4) | 7.6183 | 0.8517 (e4) | 0.8477 | -125.1105 | 在 R2 LoRA 基础上加 raw feature bridge / decoder feature adapter；abs_rel 明显优于 raw LoRA e10 repl，接近但略弱于 e10 decoder-only repl，d1 高于两者；last 回退。 |

注：epoch 为训练代码的 0-based epoch 号。`0605_2346` 已在 2026-06-06 02:50:27 CST 正常结束；186 上两个 `0606_1348` 复刻 run 已在 2026-06-07 07:39:08 CST 正常结束；`0606_0330` 已在 2026-06-06 14:31:54 CST 正常结束；`0606_1431` 已在 2026-06-07 01:17:46 CST 正常结束；`0608_1444` 已在 2026-06-08 17:31:09 CST 正常结束。

### 1.2 其他 eval / region metrics

当前 11 个 run 未启用 KITTI / NYU / STF cross-dataset eval，也没有对应 region metric 重算表。本文暂不新增空的跨域指标表，避免把不同协议的后续结果误读为已完成 formal eval。

## 2. 当前结论

- 当前 11 个 run 中，`0606_0330` RAW-RAM identity + full-backbone LLRD 0.9 + decoder 是 ROD-night inverse-relative val 最强项：`abs_rel=4.2575`，`d1=0.8700`；但它的 init baseline 仍是 RAW-RAM-path init，不能和 RGB-path DAv2-S init 的 delta 直接比较。
- RGB 路径上，`0604_0752` decoder-only baseline 的 best abs_rel 为 `9.1288`；`0605_0139` RGB LoRA 为 `8.3252`；`0605_2346` low-lr full-backbone 为 `8.4721`。低学习率 full-backbone 优于 decoder-only，但未复现 `0605_0730` 的 full-backbone 收益。
- RAW-RAM decoder-only 路径上，`0605_1332` e5 best abs_rel 为 `5.9413`；186 上 e10 复刻 `0606_1348_repl0605_1332` 为 `5.5653`，best / last 均优于本机 e5 R1。
- RAW-RAM LoRA 路径上，`0605_1844` e5 best abs_rel 为 `7.3929`；186 上 e10 复刻 `0606_1348_repl0605_1844` 为 `6.8383`，best 有改善但 last 仍回退到 `8.9307`，且仍弱于 e10 decoder-only 复刻。`0608_1444` 在 LoRA 基础上加入 raw feature bridge / decoder feature adapter 后降到 `5.6001`，明显强于 raw LoRA e10 repl，abs_rel 接近 e10 decoder-only repl，d1 提高到 `0.8517`。
- RAW-RAM full-backbone 路径上，`0606_0330` full-backbone 进一步降到 `4.2575`；`0606_1431` low-lr full-backbone 为 `6.4195`，弱于正常 lr full-backbone，也略弱于 186 e10 RAW-RAM decoder-only 复刻。
- LoRA-only 在这批设置里没有成为更强项：RGB LoRA `0605_0139` 弱于 RGB full-backbone `0605_0730`；RAW-RAM LoRA `0605_1844` / `0606_1348_repl0605_1844` 的 abs_rel 均弱于对应 RAW-RAM decoder-only。新增的 `0608_1444` 说明 bridge + feature adapter 可以显著修复 RAW-RAM LoRA 路径，但还没有超过 `0606_0330` full-backbone。
- 参数对照：186 的两个 `0606_1348` 复刻 run 与本机 `0605_1332` / `0605_1844` 的 inspected experiment-semantic 参数一致；显式差异为 `epochs=10` vs `epochs=5`，以及远端 `save_path` / `heavy_save_root` / `pretrained_from` 路径。`pretrained_from` 的 checkpoint SHA256 与本机一致。
- 已完成 run 的 best abs_rel 多数早于 last epoch；`0606_0330` best abs_rel 在 epoch 3，但 last epoch 仍保持当前最强的 last 指标。`0608_1444` best abs_rel / d1 均在 epoch 4，last epoch 回退到 `abs_rel=7.6183`。比较时应优先使用 `best_model.pth` 对应 epoch，同时保留 last epoch 作为稳定性参考。
