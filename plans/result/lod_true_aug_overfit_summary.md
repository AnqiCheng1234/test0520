# True LOD Plan B Augmentation / LoRA / RAW_Normal Follow-up / Overfit Summary

日期：2026-06-09

本文记录 `/home/caq/6666_raw/dav2_raw_0603` 当前 True LOD Plan B block-split e40 matrix、RAW_Dark_RGB16 / RGB_Dark LoRA follow-up，以及 0609 RAW_Normal_RGB16 matched follow-up 的结果。原六个 augmentation matrix formal run、三个 0608 follow-up run 和三个 0609 RAW_Normal matched run 均已完成，全部 `status=0`。

指标表标注规约：`silog` 越低越好，`d1` 越高越好。本文的 LOD 指标是相对 DAv2-L pseudo inverse label 的 `inverse_relative` proxy，一律不是 metric depth benchmark 指标；数值只能在同一 True LOD eval protocol 下比较。error 列采用 `silog` 而非 `abs_rel`：在 `inverse_relative` proxy 下，`abs_rel = mean(|pred-target|/target)` 的分母是 disparity，远景（视差→0）像素被极小分母放大并主导指标，数值膨胀到 ~2-4，不可与 metric-depth 论文的 AbsRel 比较；`silog`/`d1` 对 depth↔inverse 反转不变，是该 proxy 下更可信的 error/accuracy 口径。True LOD formal run 的 best checkpoint 按 `lod_d1` 最大化保存，不按 error 指标最小化保存。`silog` 取与 best `lod_d1` 同一 epoch 的 `lod_val` 值。

## 0. 协议速查

Matrix source：

- queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_1509_lod_true_aug_e40_block8excl10_full_matrix.queue.log`
- queue completion：`[QUEUE] all RAW_RGB16 aug e40 runs done 2026-06-08T19:18:12+08:00`
- follow-up queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_2026_lod_true_raw_rgb16_ram3_lora_bridge_fa_e40_flip05_queue.queue.log`
- follow-up queue completion：`[QUEUE_END] 2026-06-08T22:00:11+08:00 status=0`
- RGB LoRA follow-up log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_2246_lod_true_rgb_dark_block8excl10_dav2s_lora_tap_r8a16_decoder_e40_flip05_poly.tmux.log`
- RGB LoRA completion：`[END] 2026-06-08T23:07:20+08:00 status=0`
- RAW_Normal matched queue log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0609_0010_lod_true_raw_normal_rgb16_match_0608_dark_e40_queue.queue.log`
- RAW_Normal matched queue completion：`[QUEUE_END] 2026-06-09T02:17:49+08:00 status=0`
- split manifest：`/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv`
- split policy：`8` uniform validation blocks x `14` pairs；validation neighborhoods `±10` pairs moved to `02HoldoutBuffer`
- split counts：`00Train=1958`，`01Valid=112`，`02HoldoutBuffer=160`
- nearest val-to-train distance：min `11` pairs，max `24` pairs
- train-proxy：fixed `112` train pairs，`seed=42`，center crop，no aug
- epochs：`40`
- LR：`poly`，`warmup_steps=0`
- best metric：`lod_d1` maximize
- min effective delta for RGB_Dark-vs-RAW_Dark d1：`0.003`

Run list：

| Run | Full run name | Link | Aug | End status |
|---|---|---|---|---:|
| R0 | `0608_1509_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R0_aug-baseline_e10_poly` | RGB_Dark | `baseline_e10` | `0` |
| Rg | `0608_1605_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_Rg_aug-geom_poly` | RGB_Dark | `geom` | `0` |
| R1 | `0608_1659_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_R1_aug-medium_poly` | RGB_Dark | `medium` | `0` |
| W0 | `0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly` | RAW_Dark_RGB16 | `baseline_e10` | `0` |
| Wg | `0608_1811_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_Wg_aug-geom_poly` | RAW_Dark_RGB16 | `geom` | `0` |
| W1 | `0608_1844_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W1_aug-medium_poly` | RAW_Dark_RGB16 | `medium` | `0` |
| L0 | `0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly` | RAW_Dark_RGB16 + LoRA tap | `hflip0.5_only` | `0` |
| LB | `0608_2059_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e40_flip05_poly` | RAW_Dark_RGB16 + LoRA + bridge/FA | `hflip0.5_only` | `0` |
| RL | `0608_2246_lod_true_rgb_dark_block8excl10_dav2s_lora_tap_r8a16_decoder_e40_flip05_poly` | RGB_Dark + LoRA tap | `hflip0.5_only` | `0` |
| N0 | `0609_0010_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly` | RAW_Normal_RGB16 | `baseline_e10` | `0` |
| NL | `0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly` | RAW_Normal_RGB16 + LoRA tap | `hflip0.5_only` | `0` |
| NB | `0609_0117_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e40_flip05_poly` | RAW_Normal_RGB16 + LoRA + bridge/FA | `hflip0.5_only` | `0` |

Configuration invariants：

- RGB runs：`dataset_family=lod_true_rgb_dark`，`input_domain=rgb`，`front_end=dav2_rgb`，`model_input_tensor=image`，`raw_storage_format=none`；trainable path is DAv2-S decoder-only.
- RAW_Dark runs：`dataset_family=lod_true_raw_dark_rgb16`，`input_domain=raw3`，`front_end=raw_rgb16_ram3`，`model_input_tensor=raw`，`raw_storage_format=raw_rgb16_png_3ch`；trainable path is raw front-end + DAv2-S decoder.
- RAW_Normal runs：`dataset_family=lod_true_raw_normal_rgb16`，`dataset_input_mode=raw_rgb16_normal`，`input_domain=raw3`，`front_end=raw_rgb16_ram3`，`model_input_tensor=raw`，`raw_storage_format=raw_rgb16_png_3ch`；matched to the 0608 RAW_Dark decoder / LoRA / bridge-FA settings but using `RAW_normal_RGB16` input.
- L0 follow-up：same RAW_Dark_RGB16 data path，`aug_preset=off` + `aug_hflip_prob=0.5`，`lora=dav2_lora`，`lora_block_mode=tap`，`lora_rank=8`，`lora_alpha=16`，`lora_tap_layers=(2,5,8,11)`，`bridge=none`，`decoder_feature_adapter=none`.
- LB follow-up：same L0 base plus `bridge=raw_feature_bridge`，`decoder_feature_adapter=raw_feature_adapter`，`bridge_source=ram_core`，`bridge_layers=(2,5,8,11)`，`bridge_feature_keys=(x_cat,ffm_mid,x3)`，`feature_adapter_keys=(x_cat,ffm_mid,x3)`，`bridge_feature_source_channels=x3`，`adapter_feature_source_channels=x3`.
- RL follow-up：same RGB_Dark data path as R0，`aug_preset=off` + `aug_hflip_prob=0.5`，`lora=dav2_lora`，`lora_block_mode=tap`，`lora_rank=8`，`lora_alpha=16`，`lora_tap_layers=(2,5,8,11)`，`bridge=none`，`decoder_feature_adapter=none`.
- NL / NB follow-up：same L0 / LB capacity settings respectively, but on the RAW_Normal_RGB16 input path.
- All runs：same block split manifest, same train/val crop policy, same pseudo label source, same `loss-type=ssi`, same `best_metric=lod_d1`, same `lr_schedule=poly`, same `warmup_steps=0`.

Run-row 数据源优先级：

1. `finetune_stf/exp/<run>/pretrain_eval.json` and `pretrain_eval_lod_train_proxy.json` for init metrics.
2. `finetune_stf/exp/<run>/train.log` and matching `finetune_stf/logs/<run>.tmux.log` for epoch metrics.
3. Queue log only for completion status and run order.
4. `config.json` / `resolved_config.json` only for configuration confirmation.

## 1. Main Metrics

| Run | Method / input | init val d1 | epoch 9 val d1 | best val d1 (epoch) | val silog @best | train_proxy d1 @best | gap @best | last val d1 | last gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | RGB_Dark, `baseline_e10` | `0.8245` | `0.8323` | **`0.8351` (e30)** | `0.6062` | `0.8204` | `-0.0146` | `0.8337` | `-0.0139` |
| Rg | RGB_Dark, `geom` | `0.8245` | `0.8315` | `0.8338` (e30) | `0.5589` | `0.8193` | `-0.0145` | `0.8332` | `-0.0144` |
| R1 | RGB_Dark, `medium` | `0.8245` | `0.8283` | `0.8332` (e27) | `0.5596` | `0.8137` | `-0.0195` | `0.8319` | `-0.0196` |
| W0 | RAW_Dark_RGB16, `baseline_e10` | `0.3704` | `0.8160` | **`0.8299` (e36)** | `0.5397` | `0.8320` | `0.0021` | `0.8295` | `0.0039` |
| Wg | RAW_Dark_RGB16, `geom` | `0.3704` | `0.8122` | `0.8258` (e33) | `0.5256` | `0.8320` | `0.0062` | `0.8257` | `0.0044` |
| W1 | RAW_Dark_RGB16, `medium` | `0.3704` | `0.8150` | `0.8260` (e39) | `0.5362` | `0.8257` | `-0.0004` | `0.8260` | `-0.0004` |
| L0 | RAW_Dark_RGB16, `lora_tap_r8a16_hflip0.5` | `0.3704` | `0.8269` | `0.8451` (e37) | `0.5523` | `0.8667` | `0.0216` | `0.8429` | `0.0219` |
| LB | RAW_Dark_RGB16, `lora_tap_r8a16_bridge_fa_hflip0.5` | `0.3701` | `0.8290` | **`0.8510` (e27)** | `0.5690` | `0.8647` | `0.0138` | `0.8483` | `0.0190` |
| RL | RGB_Dark, `lora_tap_r8a16_hflip0.5` | `0.8245` | `0.8510` | **`0.8555` (e22)** | `0.5851` | `0.8614` | `0.0059` | `0.8546` | `0.0172` |
| N0 | RAW_Normal_RGB16, `baseline_e10` | `0.3703` | `0.8714` | `0.8871` (e35) | `0.5271` | `0.9087` | `0.0216` | `0.8858` | `0.0246` |
| NL | RAW_Normal_RGB16, `lora_tap_r8a16_hflip0.5` | `0.3703` | `0.8908` | `0.8990` (e36) | `0.4912` | `0.9186` | `0.0196` | `0.8988` | `0.0195` |
| NB | RAW_Normal_RGB16, `lora_tap_r8a16_bridge_fa_hflip0.5` | `0.3699` | `0.8878` | **`0.8997` (e34)** | `0.5417` | `0.9201` | `0.0204` | `0.8970` | `0.0254` |

Notes：

- Epoch numbers are 0-based training epoch indices.
- Bold values mark category landmarks: R0 is best original RGB augmentation-matrix run, W0 is best original RAW_Dark augmentation-matrix run, LB is best RAW_Dark follow-up, RL is best dark-input follow-up, and NB is best RAW_Normal matched run / highest score in the expanded table.
- `gap = d1(train_proxy) - d1(lod_val)`.
- RGB init is the DAv2-S RGB path on RGB_Dark. RAW init includes the raw-RGB16 RamCore3 front-end initialization and is not a shared D0 baseline with RGB.
- L0 / LB / RL / NL / NB are follow-up capacity-adapter runs with hflip-only augmentation; they should not be folded into the original three-axis augmentation comparison.
- N0 / NL / NB use RAW_Normal_RGB16 input. They share the same split, label source, and eval protocol, but they are not dark-input runs and should not revise the fair RGB_Dark-vs-RAW_Dark conclusion.

## 2. Fair RGB_Dark-vs-RAW_Dark Comparison

For the original six-run augmentation matrix, headline comparison must use `geom`, because `geom` uses the same hflip/scale-crop augmentation in both domains and no domain-specific photometric transform. L0 / LB / RL are excluded from this table because they change trainable capacity and architecture, not only input domain or augmentation. N0 / NL / NB are excluded because they use RAW_Normal_RGB16 input rather than dark input.

| Aug axis | RGB_Dark best d1 | RAW_Dark best d1 | RGB_Dark - RAW_Dark | Judgment |
|---|---:|---:|---:|---|
| `baseline_e10` | `0.8351` | `0.8299` | `+0.0052` | RGB higher, above threshold |
| `geom` headline | `0.8338` | `0.8258` | `+0.0080` | RGB pipeline wins |
| `medium` | `0.8332` | `0.8260` | `+0.0072` | RGB higher; used only as robustness because photometric is domain-specific |

Conclusion for the original matrix：under the block8excl10 True LOD pseudo-val protocol, RGB_Dark decoder pipeline beats RAW_Dark_RGB16 RamCore3 pipeline. The fair `geom` delta is `+0.0080 d1`, above `min_effective_d1_delta=0.003`, and the direction is consistent in all three augmentation axes.

Dark-input follow-up capacity comparison：

| Comparison | Delta in best d1 | Readout |
|---|---:|---|
| L0 vs W0 | `+0.0152` | LoRA tap is a strong improvement over the original RAW_Dark baseline |
| LB vs L0 | `+0.0059` | bridge + feature adapter adds another material gain over LoRA-only |
| LB vs R0 | `+0.0159` | strongest RAW_Dark follow-up beats the original best RGB matrix score |
| RL vs R0 | `+0.0204` | RGB_Dark also benefits strongly from LoRA tap + hflip-only |
| RL vs L0 | `+0.0104` | under the LoRA tap-only hflip setup, RGB_Dark remains higher than RAW_Dark_RGB16 |
| RL vs LB | `+0.0045` | RGB_Dark LoRA tap-only is above the strongest RAW_Dark follow-up |

Conclusion for dark-input follow-up：RGB_Dark with LoRA tap is the best dark-input run (`0.8555@e22`). The earlier RAW_Dark follow-up conclusion still holds internally: RAW_Dark_RGB16 RamCore3 benefits from LoRA tap and bridge/feature adapter. But the RGB_Dark LoRA twin is stronger than both RAW_Dark follow-ups, so the dark-input best-score ranking favors RGB_Dark again.

RAW_Normal matched comparison：

| Comparison | Delta in best d1 | Readout |
|---|---:|---|
| N0 vs W0 | `+0.0572` | RAW_Normal input is much higher than matched RAW_Dark decoder baseline |
| NL vs L0 | `+0.0539` | RAW_Normal remains much higher under LoRA tap |
| NB vs LB | `+0.0487` | RAW_Normal remains much higher under LoRA + bridge/FA |
| NL vs N0 | `+0.0119` | LoRA tap materially improves RAW_Normal baseline |
| NB vs NL | `+0.0007` | bridge + feature adapter adds no material gain beyond RAW_Normal LoRA at the `0.003` threshold |
| NB vs RL | `+0.0442` | NB is highest in the expanded table, but this is not a fair dark-input comparison |

Conclusion for RAW_Normal matched runs：the 0609 results show that input exposure/source is a dominant lever under this pseudo-label protocol. RAW_Normal_RGB16 is roughly `+0.05 d1` above the matched RAW_Dark_RGB16 settings. NB is the highest-score run in this expanded summary (`0.8997@e34`), but because it uses RAW_Normal input, it should be treated as a separate upper/input-source ablation, not as a replacement for the dark-input RGB_Dark-vs-RAW_Dark conclusion.

## 3. Augmentation Effect

Within-domain best d1 comparison：

| Domain | Baseline | Geom delta vs baseline | Medium delta vs baseline | Interpretation |
|---|---:|---:|---:|---|
| RGB_Dark | `0.8351` | `-0.0013` | `-0.0019` | No material gain; differences are below `0.003` |
| RAW_Dark_RGB16 | `0.8299` | `-0.0041` | `-0.0039` | Augmentation hurts RAW_Dark at the planned threshold |

Conclusion：`baseline_e10` is the strongest setting in both domains. Plan B does not support enabling `geom` or `medium` by default for this True LOD protocol.

L0 / LB / RL / NL / NB use `hflip0.5_only` and added trainable modules, so they are not evidence for changing the default augmentation preset. They are evidence that RAW_Dark, RAW_Normal, and RGB_Dark paths benefit from extra adaptation capacity.

## 4. Overfit / Under-Training Readout

RGB overfit readout：

- R0 best gap is `-0.0146`; last gap is `-0.0139`.
- Rg best gap is `-0.0145`; last gap is `-0.0144`.
- R1 best gap is `-0.0195`; last gap is `-0.0196`.
- RGB train-proxy d1 is not higher than val d1. By the defined Plan B criterion, there is no positive train-only memorization signal.
- R0 best-to-last val d1 drop is `0.0014`, below the `0.003` noise threshold. This looks like plateau/noise, not strong overfitting.

RAW_Dark under-training readout：

- W0 improves from epoch 9 `d1=0.8160` to best `0.8299@e36`.
- Wg improves from epoch 9 `0.8122` to best `0.8258@e33`.
- W1 improves from epoch 9 `0.8150` to best `0.8260@e39`.
- This confirms the e10 RAW_Dark setting was under-trained. Within the original RAW_Dark augmentation matrix, the best setting remains W0 `baseline_e10`, so the next default should not be medium augmentation.

Residual caution：

- RAW_Dark W0/Wg show small positive train-proxy gaps at best / last, especially Wg (`0.0062@best`). This is a weak train-vs-val separation signal, but it is not paired with a strong val collapse. Given the fixed 112-sample train proxy and 112-sample val, treat it as a diagnostic signal rather than proof of memorization.

LoRA / bridge follow-up readout：

- L0 improves from epoch 9 `d1=0.8269` to best `0.8451@e37`; last is `0.8429`, a `0.0022` best-to-last drop below the `0.003` threshold.
- LB improves from epoch 9 `d1=0.8290` to best `0.8510@e27`; last is `0.8483`, a `0.0027` best-to-last drop below but close to the `0.003` threshold.
- RL improves from epoch 9 `d1=0.8510` to best `0.8555@e22`; last is `0.8546`, a `0.0009` best-to-last drop below the `0.003` threshold.
- L0 has a larger positive train-proxy gap (`0.0216@best`, `0.0219@last`) than the original RAW_Dark matrix runs. LB has a smaller best gap (`0.0138`) but rises to `0.0190` at last.
- RL has a modest best gap (`0.0059`) but rises to `0.0172` at last; the val curve does not collapse, but the train-proxy separation increases late.
- These gaps are real train-vs-val separation signals. Because val d1 remains materially higher than the matching non-LoRA baselines and does not collapse, treat this as mild overfit pressure / capacity signal, not as evidence to reject the follow-up.

RAW_Normal matched readout：

- N0 improves from epoch 9 `d1=0.8714` to best `0.8871@e35`; last is `0.8858`, a `0.0013` best-to-last drop below the `0.003` threshold.
- NL improves from epoch 9 `d1=0.8908` to best `0.8990@e36`; last is `0.8988`, a `0.0002` best-to-last drop below the threshold.
- NB improves from epoch 9 `d1=0.8878` to best `0.8997@e34`; last is `0.8970`, a `0.0027` best-to-last drop below but close to the threshold.
- N0 / NL / NB all show positive train-proxy gaps around `0.02` at best and last. NB rises from `0.0204@best` to `0.0254@last`, so it has the strongest late train-vs-val separation signal among the RAW_Normal runs.
- Val d1 does not collapse in any RAW_Normal run, so this reads as mild capacity / train-proxy separation pressure rather than a reason to reject the RAW_Normal result.

## 5. Actionable Conclusions

1. Use the block8excl10 manifest for formal True LOD conclusions. The old seed42 pair-random split remains debug-only because every validation pair had a nearest train pair at distance `1`.
2. For the original augmentation matrix, the best RGB run is `R0` (`0608_1509...R0...`) with `lod_val d1=0.8351@e30`.
3. The best original RAW_Dark augmentation-matrix run is `W0` (`0608_1739...W0...`) with `lod_val d1=0.8299@e36`.
4. The best RAW_Dark follow-up is `LB` (`0608_2059...bridge_feature_adapter...`) with `lod_val d1=0.8510@e27`.
5. The best dark-input run remains `RL` (`0608_2246...rgb_dark...lora_tap...`) with `lod_val d1=0.8555@e22`.
6. The best expanded-table run is `NB` (`0609_0117...raw_normal...bridge_feature_adapter...`) with `lod_val d1=0.8997@e34`, but it uses RAW_Normal input and should not be used as a fair dark-input result.
7. The fair original matrix RGB_Dark-vs-RAW_Dark conclusion is that RGB_Dark decoder pipeline wins over RAW_Dark_RGB16 RamCore3 decoder pipeline on this pseudo-val protocol. The dark-input follow-up conclusion is consistent with that direction: under comparable LoRA tap-only / hflip-only settings, RGB_Dark beats RAW_Dark_RGB16 by `+0.0104 d1`, and also beats the stronger RAW_Dark bridge/FA follow-up by `+0.0045 d1`.
8. The 0609 RAW_Normal matched result says input exposure/source is a much stronger lever than bridge/FA: NL beats N0 by `+0.0119`, while NB beats NL by only `+0.0007`.
9. Augmentation is not the lever that improved the original matrix. The stronger levers in the follow-ups are adapter capacity and, separately, RAW_Normal input source.
10. Do not describe these numbers as metric-depth accuracy or paper-level LOD benchmark performance; they are pseudo-label proxy results over the block8excl10 split.
