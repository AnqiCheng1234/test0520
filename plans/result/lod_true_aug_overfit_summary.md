# True LOD Plan B Augmentation / Overfit Summary

日期：2026-06-08

本文记录 `/home/caq/6666_raw/dav2_raw_0603` 当前 True LOD Plan B block-split e40 matrix 的结果。六个 formal run 均已完成，全部 `status=0`。

指标表标注规约：`abs_rel` 越低越好，`d1` 越高越好。本文的 LOD 指标是相对 DAv2-L pseudo inverse label 的 `inverse_relative` proxy，一律不是 metric depth benchmark 指标；数值只能在同一 True LOD eval protocol 下比较。True LOD formal run 的 best checkpoint 按 `lod_d1` 最大化保存，不按 `abs_rel` 最小化保存。

## 0. 协议速查

Matrix source：

- queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_1509_lod_true_aug_e40_block8excl10_full_matrix.queue.log`
- queue completion：`[QUEUE] all RAW_RGB16 aug e40 runs done 2026-06-08T19:18:12+08:00`
- split manifest：`/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv`
- split policy：`8` uniform validation blocks x `14` pairs；validation neighborhoods `±10` pairs moved to `02HoldoutBuffer`
- split counts：`00Train=1958`，`01Valid=112`，`02HoldoutBuffer=160`
- nearest val-to-train distance：min `11` pairs，max `24` pairs
- train-proxy：fixed `112` train pairs，`seed=42`，center crop，no aug
- epochs：`40`
- LR：`poly`，`warmup_steps=0`
- best metric：`lod_d1` maximize
- min effective delta for RGB-vs-RAW d1：`0.003`

Run list：

| Run | Full run name | Link | Aug | End status |
|---|---|---|---|---:|
| R0 | `0608_1509_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R0_aug-baseline_e10_poly` | RGB_Dark | `baseline_e10` | `0` |
| Rg | `0608_1605_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_Rg_aug-geom_poly` | RGB_Dark | `geom` | `0` |
| R1 | `0608_1659_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R1_aug-medium_poly` | RGB_Dark | `medium` | `0` |
| W0 | `0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly` | RAW_RGB16 | `baseline_e10` | `0` |
| Wg | `0608_1811_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_Wg_aug-geom_poly` | RAW_RGB16 | `geom` | `0` |
| W1 | `0608_1844_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W1_aug-medium_poly` | RAW_RGB16 | `medium` | `0` |

Configuration invariants：

- RGB runs：`dataset_family=lod_true_rgb_dark`，`input_domain=rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`；trainable path is DAv2-S decoder-only.
- RAW runs：`dataset_family=lod_true_raw_dark_rgb16`，`input_domain=raw3`，`front_end=raw_rgb16_ram3`，`model_input_tensor=raw`，`raw_storage_format=raw_rgb16_png_3ch`；trainable path is raw front-end + DAv2-S decoder.
- All runs：same block split manifest, same train/val crop policy, same pseudo label source, same `loss-type=ssi`, same `best_metric=lod_d1`, same `lr_schedule=poly`, same `warmup_steps=0`.

Run-row 数据源优先级：

1. `finetune_stf/exp/<run>/pretrain_eval.json` and `pretrain_eval_lod_train_proxy.json` for init metrics.
2. `finetune_stf/exp/<run>/train.log` and matching `finetune_stf/logs/<run>.tmux.log` for epoch metrics.
3. Queue log only for completion status and run order.
4. `config.json` / `resolved_config.json` only for configuration confirmation.

## 1. Main Metrics

| Run | Method / input | init val d1 | epoch 9 val d1 | best val d1 (epoch) | val abs_rel @best | train_proxy d1 @best | gap @best | last val d1 | last gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | RGB_Dark, `baseline_e10` | `0.8245` | `0.8323` | **`0.8351` (e30)** | `3.8012` | `0.8204` | `-0.0146` | `0.8337` | `-0.0139` |
| Rg | RGB_Dark, `geom` | `0.8245` | `0.8315` | `0.8338` (e30) | `3.6853` | `0.8193` | `-0.0145` | `0.8332` | `-0.0144` |
| R1 | RGB_Dark, `medium` | `0.8245` | `0.8283` | `0.8332` (e27) | `3.8724` | `0.8137` | `-0.0195` | `0.8319` | `-0.0196` |
| W0 | RAW_RGB16, `baseline_e10` | `0.3704` | `0.8160` | **`0.8299` (e36)** | `3.9027` | `0.8320` | `0.0021` | `0.8295` | `0.0039` |
| Wg | RAW_RGB16, `geom` | `0.3704` | `0.8122` | `0.8258` (e33) | `3.6027` | `0.8320` | `0.0062` | `0.8257` | `0.0044` |
| W1 | RAW_RGB16, `medium` | `0.3704` | `0.8150` | `0.8260` (e39) | `3.8018` | `0.8257` | `-0.0004` | `0.8260` | `-0.0004` |

Notes：

- Epoch numbers are 0-based training epoch indices.
- `gap = d1(train_proxy) - d1(lod_val)`.
- RGB init is the DAv2-S RGB path on RGB_Dark. RAW init includes the raw-RGB16 RamCore3 front-end initialization and is not a shared D0 baseline with RGB.

## 2. Fair RGB-vs-RAW Comparison

Headline comparison must use `geom`, because `geom` uses the same hflip/scale-crop augmentation in both domains and no domain-specific photometric transform.

| Aug axis | RGB best d1 | RAW best d1 | RGB - RAW | Judgment |
|---|---:|---:|---:|---|
| `baseline_e10` | `0.8351` | `0.8299` | `+0.0052` | RGB higher, above threshold |
| `geom` headline | `0.8338` | `0.8258` | `+0.0080` | RGB pipeline wins |
| `medium` | `0.8332` | `0.8260` | `+0.0072` | RGB higher; used only as robustness because photometric is domain-specific |

Conclusion：under the block8excl10 True LOD pseudo-val protocol, RGB_Dark decoder pipeline beats RAW_RGB16 RamCore3 pipeline. The fair `geom` delta is `+0.0080 d1`, above `min_effective_d1_delta=0.003`, and the direction is consistent in all three augmentation axes.

## 3. Augmentation Effect

Within-domain best d1 comparison：

| Domain | Baseline | Geom delta vs baseline | Medium delta vs baseline | Interpretation |
|---|---:|---:|---:|---|
| RGB_Dark | `0.8351` | `-0.0013` | `-0.0019` | No material gain; differences are below `0.003` |
| RAW_RGB16 | `0.8299` | `-0.0041` | `-0.0039` | Augmentation hurts RAW at the planned threshold |

Conclusion：`baseline_e10` is the strongest setting in both domains. Plan B does not support enabling `geom` or `medium` by default for this True LOD protocol.

## 4. Overfit / Under-Training Readout

RGB overfit readout：

- R0 best gap is `-0.0146`; last gap is `-0.0139`.
- Rg best gap is `-0.0145`; last gap is `-0.0144`.
- R1 best gap is `-0.0195`; last gap is `-0.0196`.
- RGB train-proxy d1 is not higher than val d1. By the defined Plan B criterion, there is no positive train-only memorization signal.
- R0 best-to-last val d1 drop is `0.0014`, below the `0.003` noise threshold. This looks like plateau/noise, not strong overfitting.

RAW under-training readout：

- W0 improves from epoch 9 `d1=0.8160` to best `0.8299@e36`.
- Wg improves from epoch 9 `0.8122` to best `0.8258@e33`.
- W1 improves from epoch 9 `0.8150` to best `0.8260@e39`.
- This confirms the e10 RAW setting was under-trained. However, the best RAW setting is still W0 `baseline_e10`, so the next default should not be medium augmentation.

Residual caution：

- RAW W0/Wg show small positive train-proxy gaps at best / last, especially Wg (`0.0062@best`). This is a weak train-vs-val separation signal, but it is not paired with a strong val collapse. Given the fixed 112-sample train proxy and 112-sample val, treat it as a diagnostic signal rather than proof of memorization.

## 5. Actionable Conclusions

1. Use the block8excl10 manifest for formal True LOD conclusions. The old seed42 pair-random split remains debug-only because every validation pair had a nearest train pair at distance `1`.
2. For current Plan B checkpoints, the best RGB run is `R0` (`0608_1509...R0...`) with `lod_val d1=0.8351@e30`.
3. The best RAW run is `W0` (`0608_1739...W0...`) with `lod_val d1=0.8299@e36`.
4. The fair RGB-vs-RAW conclusion is that RGB_Dark decoder pipeline wins over RAW_RGB16 RamCore3 pipeline on this pseudo-val protocol.
5. Augmentation is not the lever that improved this matrix. If more work is needed, a cleaner next test is RAW `baseline_e10` longer training (`e60`) or a schedule-only twin, not `geom` / `medium` as default.
6. Do not describe these numbers as metric-depth accuracy or paper-level LOD benchmark performance; they are pseudo-label proxy results over the block8excl10 split.
