# True LOD RGB_Dark / RAW_RGB16 baseline 公平性汇总

日期：2026-06-08

本文记录 `/home/caq/6666_raw/dav2_raw_0603` 当前本机 True LOD formal validation 结果。当前主表覆盖 2 个本机 run，均已完成。

写作规约：指标表只放 formal eval log / json 中的数字；仍在运行的 run 不把 partial epoch 指标混入主指标表。当前这 2 个 LOD run 均已正常结束。

指标表标注规约：`abs_rel` 越低越好，`d1` 越高越好。本文的 LOD 指标是相对 DAv2-L pseudo inverse label 的 `inverse_relative` proxy，一律不是 metric depth benchmark 指标；数值只能在同一 True LOD eval protocol 下比较。True LOD formal run 的 best checkpoint 按 `lod_d1` 最大化保存，不按 `abs_rel` 最小化保存。

注意：本文中的 LOD 是 `/home/caq/6666_raw/0000_dataset/LOD` 下的真实 LOD 四目录数据集，不是早期被误命名为 LOD 的 ROD 数据。

## 0. 协议速查 / 数据源规则

当前 formal runs：

- `0608_0109_lod_true_rgb_dark_dav2s_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_formal_rgb_then_raw_e10.queue.log`
  - queue status：`[END] 2026-06-08T01:13:06+08:00 status=0`
  - label：True LOD RGB_Dark decoder-only baseline
  - stage：`lod_only`
  - input：`rgb`，`dataset_family=lod_true_rgb_dark`，`dataset_input_mode=rgb_dark`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`
  - path：LOD `RGB_normal` -> frozen DAv2-L pseudo inverse-relative label；student input 为 LOD `RGB_Dark` JPG -> native `800x1200` -> crop `[3,512,960]` -> DAv2-S。
  - trainable：DAv2-S decoder only，`lr=1e-5`，无 LoRA，epochs=10
  - train/eval split：LOD train `2118` pairs，LOD val `112` pairs
  - eval protocol：`lod_val`，center crop，对 pseudo inverse target 做 affine align 后在 inverse space 计算 `abs_rel/rmse/silog/d1/d2/d3`
  - checkpoint：`best_model.pth` 按 `lod_d1` 保存，best epoch 为 epoch 3；`last_epoch_model.pth` 为 epoch 9
- `0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10`
  - source log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10/train.log`
  - tmux log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10.tmux.log`
  - queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_formal_rgb_then_raw_e10.queue.log`
  - queue status：`[END] 2026-06-08T01:21:47+08:00 status=0`
  - label：True LOD RAW_Dark RGB16 RamCore3 identity + decoder baseline
  - stage：`lod_only`
  - input：`raw3`，`dataset_family=lod_true_raw_dark_rgb16`，`dataset_input_mode=raw_rgb16_dark`，`front_end=raw_rgb16_ram3`，`model_input_tensor=raw`，`raw_storage_format=raw_rgb16_png_3ch`
  - path：LOD `RGB_normal` -> frozen DAv2-L pseudo inverse-relative label；student input 为 LOD `RAW_Dark` 3-channel 16-bit PNG，storage order `BGR` -> model order `RGB` -> `uint16_div_65535` -> native `800x1200` -> crop `[3,512,960]` -> RamCore3 -> `raw_ram_rgb_tail=identity` -> DAv2-S。
  - trainable：raw front-end + DAv2-S decoder，`lr=1e-5`，`raw_front_end_lr=5e-5`，无 LoRA，epochs=10
  - train/eval split：LOD train `2118` pairs，LOD val `112` pairs
  - eval protocol：`lod_val`，center crop，对 pseudo inverse target 做 affine align 后在 inverse space 计算 `abs_rel/rmse/silog/d1/d2/d3`
  - checkpoint：`best_model.pth` 按 `lod_d1` 保存，best epoch 为 epoch 9；`last_epoch_model.pth` 为 epoch 9

LOD pseudo label / split source：

- LOD root：`/home/caq/6666_raw/0000_dataset/LOD`
- pair manifest：`/home/caq/6666_raw/0000_dataset/LOD/manifests/lod_true_pairs_2118_112_seed42.csv`
- pseudo manifest：`/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest.csv`
- teacher：`RGB_normal` -> DAv2-L `vitl`，`input_size=924`，target shape `[800,1200]`，label space `inverse_relative`
- split policy：seed42 pair-random split，`00Train=2118`，`01Valid=112`；该 split 不是 BMVC paper 的 `1830/400` detection split

baseline/control 行：

- RGB-path official DAv2-S init：`pretrain_eval.json` 中 LOD val 为 `abs_rel=3.1468, d1=0.8154`。这是 LOD `RGB_Dark` 输入下的初始化模型表现。
- RAW-RGB16-path init：`pretrain_eval.json` 中 LOD val 为 `abs_rel=31.2529, d1=0.3279`。该数值包含 raw-RGB16 RamCore3 front-end 初始状态，不是和 RGB-path DAv2-S init 同义的公共 D0 baseline。

run-row 数据源优先级：

1. `finetune_stf/exp/<run>/pretrain_eval.json`。
2. `finetune_stf/exp/<run>/train.log`。
3. `finetune_stf/logs/<run>.tmux.log` 与 queue log 只用于完成状态和队列上下文。
4. `finetune_stf/exp/<run>/config.json`、`resolved_config.json` 只用于确认配置。
5. Pseudo label `run_config.json` 与 pair manifest `.meta.json` 只用于确认 teacher / split / target shape。

## 1. True LOD Val

### 1.1 Overall metrics

| Experiment | Method / input | init abs_rel / d1 | checkpoint d1 best (epoch) | abs_rel at d1-best | last abs_rel / d1 | lowest abs_rel (epoch) | best d1 delta vs init | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `0608_0109...rgb_dark...decoder_e10` | LOD `RGB_Dark`；DAv2-S decoder-only baseline | 3.1468 / 0.8154 | **0.8210 (e3)** | **2.5893** | 2.5126 / 0.8196 | **2.4589 (e6)** | +0.0056 | d1 只小幅提升；lowest abs_rel 在 e6，不是保存 best checkpoint 的 e3。 |
| `0608_0113...raw_rgb16_ram3...decoder_e10` | LOD `RAW_Dark` RGB16 -> RamCore3 identity；raw front-end + decoder | 31.2529 / 0.3279 | 0.8205 (e9) | 2.7649 | 2.7649 / 0.8205 | 2.6725 (e7) | +0.4926 | RAW init 很弱但训练后追近 RGB；best d1 / last 同为 e9，d1 与 RGB best 只差 0.0005。 |

注：epoch 为训练代码的 0-based epoch 号。两个 run 由同一个 sequential queue 在本机单 GPU 上依次执行；`0608_0109` RGB_Dark 于 2026-06-08 01:13:06 CST 正常结束，随后 `0608_0113` RAW_RGB16 于 2026-06-08 01:21:47 CST 正常结束。

### 1.2 Epoch trace

| Epoch | RGB_Dark abs_rel / d1 | RAW_RGB16 abs_rel / d1 |
|---:|---:|---:|
| init | 3.1468 / 0.8154 | 31.2529 / 0.3279 |
| 0 | 2.7431 / 0.8164 | 3.4118 / 0.7749 |
| 1 | 2.7494 / 0.8186 | 3.1657 / 0.7972 |
| 2 | 2.8013 / 0.8204 | 2.8407 / 0.8024 |
| 3 | 2.5893 / **0.8210** | 2.7120 / 0.8071 |
| 4 | 2.6371 / 0.8186 | 2.7135 / 0.8129 |
| 5 | 2.6250 / 0.8190 | 2.7686 / 0.8124 |
| 6 | **2.4589** / 0.8188 | 2.6938 / 0.8155 |
| 7 | 2.5211 / 0.8198 | **2.6725** / 0.8190 |
| 8 | 2.5395 / 0.8197 | 2.7725 / 0.8178 |
| 9 | 2.5126 / 0.8196 | 2.7649 / **0.8205** |

### 1.3 其他 eval / region metrics

当前 2 个 True LOD run 未启用 KITTI / NYU / STF / ROD cross-dataset eval，也没有对应 region metric 重算表。本文暂不新增空的跨域指标表，避免把不同协议的后续结果误读为已完成 formal eval。

## 2. 当前结论

- 当前 2 个 True LOD decoder baseline 的 `lod_d1` 非常接近：RGB_Dark best `0.8210`，RAW_RGB16 best `0.8205`。按 formal checkpoint 规则，RGB_Dark 略高 `0.0005`。
- RGB_Dark 的 `abs_rel` 整体更低：checkpoint-selected epoch 为 `2.5893`，最低 epoch 为 `2.4589`；RAW_RGB16 checkpoint-selected epoch 为 `2.7649`，最低 epoch 为 `2.6725`。
- RAW_RGB16 的 init baseline 不能和 RGB init 直接作为同义 D0 比较：RAW init 包含未训练 raw front-end，初始 `d1=0.3279`，训练后到 `0.8205`，说明 raw front-end + decoder 已经基本追平 RGB_Dark decoder-only 的 d1 水平。
- 两个 run 都只训练 decoder 级别能力：RGB 为 DAv2-S decoder-only，RAW 为 raw front-end + DAv2-S decoder；目前还没有 True LOD 上的 LoRA / full-backbone 对照。
- 因为 LOD formal best metric 是 `lod_d1`，后续比较 best checkpoint 时应优先看 d1-best epoch，同时保留最低 abs_rel 作为补充稳定性参考。
