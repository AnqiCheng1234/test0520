# RGB / RAW baseline 公平性汇总

日期：2026-05-21

本文记录 `/home/caq/6666_raw/dav2_raw_0520` 当前 STF formal validation 结果。

写作规约：主表只放 train-time formal eval 或同口径 eval log 中的数字；只用于可视化的 sample loss 不混入主指标表。

主表标注规约：`abs_rel` 越低越好，`d1` 越高越好。尚未补跑的行保留为 `待补`。

## 0. 协议速查 / 数据源规则

当前 formal run：

- `0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5`
  - source log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/0521_0012_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5/train.log`
  - stage: `stf_only`
  - input: `raw_ram_rgb`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` 合成 `[R,(Gr+Gb)/2,B]` -> RamCore3 BN-clean path -> DAv2
  - model log：`functions=['wb','ccm','gamma','brightness']`，`ram_core_out_channels=3`，`dav2_input=ramcore_bn_no_clamp_no_imagenet_norm`
  - train target：DAv2 pseudo depth，manifest 为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - eval split：STF val，`808` samples，sparse backend，`min_depth=1 / max_depth=80`
  - checkpoint：`best_model.pth` 按 `best_metric=stf` 保存，best epoch 为 epoch 4

baseline/control 行：

- DAv2-S RGB 直推：official RGB DAv2-S checkpoint，不训练；已补跑。
  - source log：`/home/caq/6666_raw/dav2_raw_0520/logs/0521_0104_stf_dav2s_direct_baselines.log`
  - eval report：`/home/caq/6666_raw/dav2_raw_0520/anqi_eval/results/0521_0104_stf_val_dav2s_rgb_direct/eval_stf_rel_depth_val_2026-05-21.txt`
  - eval split：STF val，`808` samples，sparse backend equivalent，`min_depth=1 / max_depth=80`，input size `512x960`
- DAv2-S RAW-preview 直推：official DAv2-S 在 RAW preview / pseudo-RGB 输入上直推，不训练；已补跑。
  - source log：`/home/caq/6666_raw/dav2_raw_0520/logs/0521_0104_stf_dav2s_direct_baselines.log`
  - eval report：`/home/caq/6666_raw/dav2_raw_0520/anqi_eval/results/0521_0104_stf_val_dav2s_raw_preview_direct/eval_stf_rel_depth_val_2026-05-21.txt`
  - RAW path：STF legacy online decomp16 RAW -> packed Bayer -> `channel_mode=rgb_avg_g` 合成 `[R,(Gr+Gb)/2,B]` -> ImageNet norm -> official DAv2-S
  - eval split：STF val，`808` samples，sparse backend equivalent，`min_depth=1 / max_depth=80`，input size `512x960`
- 当前同设定 RGB 输入：沿用当前 STF pseudo-label training protocol，但 `input_type=rgb`；待补。

run-row 数据源优先级：

1. 后续若生成 `finetune_stf/exp/<run>/analysis/*.csv` 或 `summary*.json`，优先使用。
2. `finetune_stf/exp/<run>/train.log`.
3. `finetune_stf/exp/<run>/config.json` 只用于确认配置。

## 1. STF val

| Experiment | Training data / schedule | abs_rel best (epoch) | abs_rel last/current | d1 best (epoch) | d1 last/current | notes |
|---|---|---:|---:|---:|---:|---|
| DAv2-S RGB 直推 | baseline，不训练；`0521_0104_stf_val_dav2s_rgb_direct` | **0.1287 (zero-shot)** | 0.1287 | **0.8575 (zero-shot)** | 0.8575 | official RGB DAv2-S 在 STF RGB LUT preview 输入上 zero-shot；`rmse=7.9067`，`silog=0.2560`。 |
| DAv2-S RAW-preview 直推 | baseline，不训练；`0521_0104_stf_val_dav2s_raw_preview_direct` | **0.1649 (zero-shot)** | 0.1649 | **0.7774 (zero-shot)** | 0.7774 | official DAv2-S 在 STF RAW preview / pseudo-RGB path 上 zero-shot；`legacy_online_decomp16 + passthrough + rgb_avg_g`；`rmse=9.0814`，`silog=0.2897`。 |
| RGB input, same STF setting | STF pseudo-label training；同当前 run 协议但 `input_type=rgb` | 待补 | 待补 | 待补 | 待补 | 预留 fair RGB-input training control。 |
| `0521_0012...raw_ram_rgb_bnclean_identity_e5` | STF train+test pseudo-label training；`stf_train=5408`，val `808`，`stf_repeat=7`，每 epoch `676` steps | **0.1388 (e4)** | 0.1388 | **0.8319 (e4)** | 0.8319 | 当前项目摘要中的第一条 formal run。DAv2-S 冻住（`dav2_train_mode=none`），只训练 RAW/RAM front end；best 和 last 均为 epoch 4。 |

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
