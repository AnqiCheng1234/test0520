# ROD Night RGB / RAW baseline 公平性汇总

日期：2026-06-06

本文记录 `/home/caq/6666_raw/dav2_raw_0603` 当前 ROD-night formal validation 结果。当前主表覆盖 2026-06-04 / 2026-06-05 启动的 6 个 run；`0605_2346...` 仍在运行，结果栏暂留空。

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
- `0605_2346_rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0605_2346_rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_2346_rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0605_1332_rod_night_raw_ram_v1_e5.queue.log`
  - running tmux session：`0605_1332_rod_night_raw_ram_v1_e5`
  - label：RGB-ref student RGB low-lr backbone LLRD 0.9
  - stage：`rod_only`
  - input：`rgb`，`dataset_family=rod_raw_student_rgb`，`dataset_input_mode=raw24_student_rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`
  - path：同 `0605_0139`，ROD raw24 -> `student_dark_degreen_v1` RGB -> DAv2-S
  - trainable：DAv2-S full backbone + decoder，`backbone_layer_decay=0.9`，`lr=1e-6`，无 LoRA，epochs=5
  - train/eval split：ROD train `12036` samples，ROD night val `2000` samples
  - eval protocol：`rod_night_val` inverse-relative pseudo label protocol
  - checkpoint：仍在运行；主表结果暂留空

baseline/control 行：

- RGB-path official DAv2-S init：`pretrain_eval.json` 中 ROD night val 约 `abs_rel=13.574-13.579, d1=0.7617`。这是 raw24 -> `student_dark_degreen_v1` RGB 输入下的初始化模型表现。
- RAW-RAM-path init：`pretrain_eval.json` 中 ROD night val 约 `abs_rel=130.38, d1=0.2722`。该数值包含 raw-RAM front-end 初始状态，不是和 RGB-path DAv2-S init 同义的公共 D0 baseline。

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
| `0605_0730...backbone_ld09_decoder_e10` | Student RGB；full backbone LLRD 0.9 + decoder | 13.5786 / 0.7617 | **5.8880 (e3)** | **8.0998** | **0.8483 (e8)** | **0.8471** | **-7.6906** | 当前已完成 run 中 abs_rel / d1 最强；best abs_rel 与 best d1 不在同一 epoch。 |
| `0605_1332...rawram3_identity...decoder_e5` | RAW4 -> RamCore3 identity；raw front-end + decoder | 130.3805 / 0.2722 | 5.9413 (e2) | 6.7567 | 0.8303 (e4) | 0.8303 | -124.4392 | abs_rel 接近 `0605_0730`，但 init 不是公共 RGB D0；last 优于 R2 raw LoRA。 |
| `0605_1844...rawram3_identity...lora_tap...decoder_e5` | RAW4 -> RamCore3 identity；raw front-end + decoder + LoRA tap | 130.3823 / 0.2722 | 7.3929 (e1) | 9.4693 | 0.8362 (e3) | 0.8287 | -122.9894 | raw-RAM + LoRA 的 abs_rel 弱于 raw-RAM decoder-only；d1 best 略高于 R1，但 last 回退。 |
| `0605_2346...studentrgb...backbone_lowlr_ld09_e5` | Student RGB；full backbone LLRD 0.9 + decoder，`lr=1e-6` | 13.5739 / 0.7617 | pending | pending | pending | pending | pending | 仍在运行；partial epoch 指标不进入主表。 |

注：epoch 为训练代码的 0-based epoch 号。`0605_2346` 在 2026-06-06 00:40 CST 仍由 tmux session `0605_1332_rod_night_raw_ram_v1_e5` 持续运行；表内未使用其 epoch 0 partial result。

### 1.2 其他 eval / region metrics

当前 6 个 run 未启用 KITTI / NYU / STF cross-dataset eval，也没有对应 region metric 重算表。本文暂不新增空的跨域指标表，避免把不同协议的后续结果误读为已完成 formal eval。

## 2. 当前结论

- 当前已完成 run 中，`0605_0730` Student RGB full-backbone LLRD 0.9 + decoder 是 ROD-night inverse-relative val 最强项：`abs_rel=5.8880`，`d1=0.8483`。
- `0604_0752` Student RGB decoder-only baseline 的 best abs_rel 为 `9.1288`；`0605_0139` RGB LoRA 将 best abs_rel 推到 `8.3252`，但仍明显弱于 `0605_0730` full-backbone。
- `0605_1332` RAW-RAM identity decoder-only 的 best abs_rel 为 `5.9413`，非常接近 `0605_0730`，但其 init baseline 与 RGB-path init 不同，不能用 raw init 的巨大 delta 直接证明 RAW cue 优于 RGB cue。
- LoRA 在这批设置里没有成为更强项：RGB LoRA `0605_0139` 弱于 RGB full-backbone `0605_0730`；RAW-RAM LoRA `0605_1844` 的 abs_rel 弱于 RAW-RAM decoder-only `0605_1332`。
- 已完成 run 的 best abs_rel 多数早于 last epoch；比较时应优先使用 `best_model.pth` 对应 epoch，而不是 `last_epoch_model.pth`。
- `0605_2346` 是低学习率 RGB-ref，仍在运行；等其 queue log 出现 `[END] ... status=0` 后再补主表结果和结论。
