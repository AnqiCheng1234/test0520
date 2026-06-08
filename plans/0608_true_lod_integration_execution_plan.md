# 0608 True LOD Integration Execution Plan

## Execution Log

- `0608_0032` started execution in `/home/caq/6666_raw/dav2_raw_0603`.
- `0608_0032` git status before edits: this plan file is untracked; unrelated modified file `plans/result/rod_night_rgb_raw_baseline_fairness_summary.md` is present and will not be touched.
- `0608_0032` Phase 0 status: in progress. Implementing canonical pair manifest builder first.
- `0608_0036` Phase 0 smoke passed: `scripts/build_lod_true_pair_manifest.py --max-pairs 16` wrote `/tmp/codex_smoke_lod_true_manifest/lod_true_pairs_smoke.csv`, split `15/1`, then the successful smoke directory was deleted.
- `0608_0039` Phase 0 completed: full manifest written to `/home/caq/6666_raw/0000_dataset/LOD/manifests/lod_true_pairs_2118_112_seed42.csv`; metadata written to `/home/caq/6666_raw/0000_dataset/LOD/manifests/lod_true_pairs_2118_112_seed42.meta.json`; split counts `00Train=2118`, `01Valid=112`, total `2230`.
- `0608_0039` Phase 0b status: in progress. Implementing read-only RAW characterization, alignment, and channel-order audit.
- `0608_0040` Phase 0b smoke passed: `finetune_stf/scripts/audit_lod_true_ingest.py` with `raw-max-images=8`, `alignment-samples=8`, `channel-samples=8` wrote `/tmp/codex_smoke_lod_true_audit/codex_smoke_lod_true_audit`, gates `pass/pass/pass`, then the successful smoke directory was deleted.
- `0608_0040` Phase 0b completed: full audit output `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_ingest_audit/0608_0037_lod_true_audit/`; gates `normalization=pass`, `alignment=pass`, `channel_order=pass`.
- `0608_0040` Phase 0b RAW stats: RAW_normal mean `9282.56`, p50 `5386`, p99 `50532`, clip@65535 `0.00317509`; RAW_Dark mean `11746.50`, p50 `8491`, p99 `54250`, clip@65535 `0.00286339`; dark/normal mean ratio `1.2654`. Decision: keep explicit `lod_raw_norm_mode=uint16_div_65535`.
- `0608_0040` Phase 0b alignment stats: paired RGB NCC p10/p50/p90 `0.5017/0.8765/0.9628`; unrelated RGB-dark NCC p90 `0.2154`; downsample phase-shift abs p90 `0.3236` pixels at 128x192 audit scale. Decision: normal-teacher labels are geometrically usable for dark inputs without crop/shift correction.
- `0608_0040` Phase 0b channel order: best mapping `cv2_channel_0->B`, `cv2_channel_1->G`, `cv2_channel_2->R`; decision: OpenCV reads RAW PNG as BGR, use `channel_reorder=(2,1,0)` and feed model `[R,G,B]`.
- `0608_0040` Phase 1 status: in progress. Implementing RGB_normal DAv2-L pseudo label builder with input-size preview support.
- `0608_0042` Phase 1 input-size preview completed: `/tmp/codex_smoke_lod_true_pseudo_size_sweep_0608_004156` generated `20` samples for each of `812` and `924`; both shapes are `800x1200`, mean positive coverage is `0.99561` for `812` and `0.99512` for `924`.
- `0608_0042` Phase 1 input-size decision: choose formal `input_size=924`. Quantitative preview: mean relative abs difference between `812` and `924` is `0.0144`; `924` mean edge energy is `1.6814` vs `812` `1.6167` (`924/812=1.0418`). Visual spot-checks showed slightly sharper thin structures without obvious extra artifacts.
- `0608_0043` Phase 1 pseudo generation smoke passed: `max_samples=4`, `input_size=924`, output `/tmp/codex_smoke_lod_true_pseudo_0608_004312`; successful smoke output was deleted by `--cleanup-success`. Preview smoke directory `/tmp/codex_smoke_lod_true_pseudo_size_sweep_0608_004156` was also deleted after recording the decision.
- `0608_0043` Phase 1 full pseudo label generation launched in tmux. Session: `lod_true_pseudo_0608_0043`; log: `/home/caq/6666_raw/dav2_raw_0603/logs/0608_0043_lod_true_pseudo_dav2l_rgb_normal_input924.log`; attach: `tmux attach -t lod_true_pseudo_0608_0043`; monitor: `tail -f /home/caq/6666_raw/dav2_raw_0603/logs/0608_0043_lod_true_pseudo_dav2l_rgb_normal_input924.log`; output root: `/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800`.
- `0608_0050` Phase 1 tmux correction: first tmux command produced partial outputs (`00Train=889`, `01Valid=52`) but no manifest/sidecars and an empty log because `conda run` captured output. Original session was not killed or reused.
- `0608_0050` Phase 1 resume launched in new tmux with unbuffered output. Session: `lod_true_pseudo_resume_0608_0050`; log: `/home/caq/6666_raw/dav2_raw_0603/logs/0608_0050_lod_true_pseudo_dav2l_rgb_normal_input924_resume.log`; attach: `tmux attach -t lod_true_pseudo_resume_0608_0050`; monitor: `tail -f /home/caq/6666_raw/dav2_raw_0603/logs/0608_0050_lod_true_pseudo_dav2l_rgb_normal_input924_resume.log`.
- `0608_0043` Phase 2/3/4 status: in progress while tmux pseudo generation runs. Implementing LOD datasets, raw storage spec, resolved config, RAW-RGB16 RamCore3 wrapper, and train/eval wiring.
- `0608_0058` user reconnect/status: execution continued. Full pseudo generation still running in `lod_true_pseudo_resume_0608_0050`; latest observed progress `1437/2230`; current file counts were increasing and final manifest/sidecars are expected only after the script finishes.
- `0608_0058` Phase 2/3/4 code status: implemented and syntax-checked. Added `finetune_stf/dataset/lod_true.py`, `raw_rgb16_png_3ch` storage spec, `raw3/raw_rgb16_ram3` resolved-config schema, `RawRgb16Ram3DepthModel`, `lod_only`, `--eval-lod`, `lod_train/lod_val`, and `best_metric=lod_d1` higher-better checkpoint direction.
- `0608_0058` Phase 6 draft status: formal scripts are present and parse-aligned: `finetune_stf/scripts/formal/0608_run_lod_true_rgb_dark_decoder_e10_queue.sh` and `finetune_stf/scripts/formal/0608_run_lod_true_raw_rgb16_ram3_decoder_e10_queue.sh`. They will be smoke/audit-checked after the full pseudo manifest exists.
- `0608_0054` Phase 2/3/4 implementation checkpoint: added `finetune_stf/dataset/lod_true.py`, `raw_rgb16_png_3ch` raw storage spec, `raw3/raw_rgb16_ram3` resolved-config schema, `RawRgb16Ram3DepthModel`, `lod_only`/`--eval-lod`, `lod_train` loader, `lod_val` eval tag, and `lod_d1` higher-better checkpoint metric logic.
- `0608_0054` Code validation: `python -m py_compile` passed for new/modified Python files; resolved-config quick check passed for both `lod_true_rgb_dark` and `lod_true_raw_dark_rgb16`; formal scripts `bash -n` passed.
- `0608_0054` Phase 1 resume progress: realtime log reached about `1252/2230`; output file counts `00Train=1182`, `01Valid=71`. Waiting for manifest/sidecars before dataset/train smoke.
- `0608_0100` Phase 1 resume progress: realtime log reached about `1797/2230`; output file counts `00Train=1700`, `01Valid=97`. Still running normally in `lod_true_pseudo_resume_0608_0050`.
- `0608_0105` Phase 1 full pseudo label generation completed: output root `/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800`; manifest rows `2230`; split counts `00Train=2118`, `01Valid=112`; `run_config.json` records `input_size=924`; target shape `[800,1200]`; mean positive coverage `0.97158`; log reports elapsed `903.6s`.
- `0608_0105` Phase 5 status: in progress. Running dataset loading smoke for `LODTrueRGBDark` and `LODTrueRawDarkRGB16`, then train smoke for both paths.
- `0608_0106` Phase 5 dataset smoke passed: `LODTrueRGBDark` and `LODTrueRawDarkRGB16` both load `00Train=2118`, `01Valid=112`; train and val crops produce `image/raw [3,512,960]`, `depth [512,960]`, `valid_mask [512,960]`, `target_space=inverse_relative`.
- `0608_0107` Phase 5 RGB_Dark train smoke passed: `torchrun` 1 train step + 2 val samples succeeded with `--stage lod_only`, `--dataset-family lod_true_rgb_dark`, `--eval-lod`, `--best-metric lod_d1`; `best_lod_d1` improved and best/current/last checkpoints were written under `/tmp/codex_smoke_lod_true_rgb_heavy` then deleted.
- `0608_0108` Phase 5 RAW_Dark train smoke passed: `torchrun` 1 train step + 2 val samples succeeded with `--dataset-family lod_true_raw_dark_rgb16`, `--input-domain raw3`, `--front-end raw_rgb16_ram3`, `--raw-storage-format raw_rgb16_png_3ch`, `--lod-raw-norm-mode uint16_div_65535`, `--raw-ram-rgb-tail identity`; log confirms `[MODEL] raw_rgb16_png_3ch -> RamCore3 -> identity -> DAv2`; `best_lod_d1` improved and temporary smoke outputs were deleted.
- `0608_0108` Phase 6 completed: added executable formal scripts `finetune_stf/scripts/formal/0608_run_lod_true_rgb_dark_decoder_e10_queue.sh` and `finetune_stf/scripts/formal/0608_run_lod_true_raw_rgb16_ram3_decoder_e10_queue.sh`; both passed `bash -n`.
- `0608_0109` Formal launch: one-GPU sequential tmux queue started to avoid RGB/RAW contention. Session: `0608_0109_lod_true_formal_rgb_then_raw_e10`; attach: `tmux attach -t 0608_0109_lod_true_formal_rgb_then_raw_e10`; queue log: `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_formal_rgb_then_raw_e10.queue.log`; monitor queue: `tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_formal_rgb_then_raw_e10.queue.log`.
- `0608_0109` Formal RGB_Dark run started first: run name `0608_0109_lod_true_rgb_dark_dav2s_decoder_e10`; run log `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10.tmux.log`; monitor RGB: `tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10.tmux.log`; save path `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10`; heavy path `/mnt/drive/3333_raw/0000_exp_ckpt/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10`.
- `0608_0109` Formal RGB_Dark initial status: pretrain `lod_val` over 112 samples completed with `d1=0.8154`; epoch `0/10` started. RAW_RGB16 formal is queued to start only after RGB_Dark script exits successfully; its run timestamp will be taken at actual start time.
- `0608_0109` Formal RGB_Dark epoch `0/10` completed: avg loss `0.1182`; `lod_val d1=0.8164`; best checkpoint improved and was saved to `/mnt/drive/3333_raw/0000_exp_ckpt/0608_0109_lod_true_rgb_dark_dav2s_decoder_e10/best_model.pth`; epoch `1/10` started.
- `0608_0110` Formal RGB_Dark epoch `1/10` completed: avg loss `0.1169`; `lod_val d1=0.8186`; best checkpoint improved and was saved; epoch `2/10` started.
- `0608_0110` Formal RGB_Dark progress: epochs `2/10` and `3/10` completed; best `lod_val d1` improved to `0.8210` at epoch `3`; epoch `4/10` started.
- `0608_0111` Formal RGB_Dark progress: epochs `4/10` and `5/10` completed with `lod_val d1=0.8186/0.8190`; best remains epoch `3` with `d1=0.8210`; epoch `6/10` started.
- `0608_0112` Formal RGB_Dark progress: epochs `6/10`, `7/10`, and `8/10` completed with `lod_val d1=0.8188/0.8198/0.8197`; best remains epoch `3` with `d1=0.8210`; epoch `9/10` eval started.
- `0608_0113` Formal RGB_Dark completed successfully: final epoch `9/10` `lod_val d1=0.8196`; best remains epoch `3` with `lod_val d1=0.8210`; `last_epoch_model.pth` saved; process exited `status=0`.
- `0608_0113` Formal RAW_RGB16 run started after RGB_Dark success: run name `0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10`; run log `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10.tmux.log`; monitor RAW: `tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10.tmux.log`; save path `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10`; heavy path `/mnt/drive/3333_raw/0000_exp_ckpt/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10`.
- `0608_0113` Formal RAW_RGB16 initial status: resolved config confirmed `input_domain=raw3`, `front_end=raw_rgb16_ram3`, `model_input_tensor=raw`, `dataset_family=lod_true_raw_dark_rgb16`, `raw_storage_format=raw_rgb16_png_3ch`, storage order `BGR`, model order `RGB`, norm `uint16_div_65535`; optimizer groups `raw_front_end lr=5e-5` and `dav2_decoder lr=1e-5`.
- `0608_0113` Input sanity panels generated for visual audit: script `finetune_stf/scripts/dump_lod_true_input_panel.py`; outputs `finetune_stf/analysis/lod_true_input_panel/0608_011309_prenorm_preram/` and `finetune_stf/analysis/lod_true_input_panel/0608_011309_prenorm_preram_dav2s_pred/`.
- `0608_0113` Formal RAW_RGB16 pretrain `lod_val` over 112 samples completed with `d1=0.3279`; epoch `0/10` started. First RAW batch shape `[8,3,512,960]`; RAW stats `mean=0.183120`, `p99=0.678019`, `max=1.0`, confirming uint16 `/65535` normalization is active.
- `0608_0114` Formal RAW_RGB16 epoch `0/10` completed: avg loss `0.2058`; `lod_val d1=0.7749`; best checkpoint saved to `/mnt/drive/3333_raw/0000_exp_ckpt/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10/best_model.pth`; epoch `1/10` started.
- `0608_0116` Formal RAW_RGB16 progress: epochs `1/10` and `2/10` completed; avg loss improved `0.1406 -> 0.1291`; best `lod_val d1` improved `0.7972 -> 0.8024`; epoch `3/10` is starting.
- `0608_0117` Formal RAW_RGB16 progress: epochs `3/10` and `4/10` completed; avg loss improved `0.1257 -> 0.1193`; best `lod_val d1` improved `0.8071 -> 0.8129`; epoch `5/10` started.
- `0608_0120` Formal RAW_RGB16 progress: epochs `5/10` through `8/10` completed; best `lod_val d1` improved to `0.8155` at epoch `6` and `0.8190` at epoch `7`; epoch `8` finished with `d1=0.8178`; final epoch `9/10` started.
- `0608_0121` Formal RAW_RGB16 completed successfully: final epoch `9/10` avg loss `0.1120`, `lod_val d1=0.8205`, `abs_rel=2.7649`, `rmse=28.8037`, `silog=0.7391`; best checkpoint improved at epoch `9` and was saved to `/mnt/drive/3333_raw/0000_exp_ckpt/0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10/best_model.pth`; `current_model.pth` and `last_epoch_model.pth` also saved; process exited `status=0`.
- `0608_0121` Formal sequential queue completed: queue log reports `[QUEUE] all done 2026-06-08T01:21:47+08:00`; tmux session `0608_0109_lod_true_formal_rgb_then_raw_e10` has exited.
- `0608_0110` Formal RGB_Dark latest status: epoch `0` reached `lod_val` evaluation. Active tmux session remains `0608_0109_lod_true_formal_rgb_then_raw_e10`.
- `0608_1435` Plan B B0.2/B0.3 status: implemented `finetune_stf/scripts/eval_lod_train_val_gap.py` and ran best/last train-proxy + val diagnostics for existing True LOD RGB/RAW e10 checkpoints. Smoke checks for RGB and RAW used `/tmp/codex_smoke_lod_gap_*` paths and were deleted after success.
- `0608_1435` Plan B RGB gap report: `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_overfit_diag/0608_1434_0608_0109_lod_true_rgb_dark_dav2s_decoder_e10/gap_report.json`. Best checkpoint: train_proxy `d1=0.8104`, val `d1=0.8210`, gap `-0.0106`; last checkpoint: train_proxy `d1=0.8120`, val `d1=0.8196`, gap `-0.0076`.
- `0608_1435` Plan B RAW gap report: `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_overfit_diag/0608_1435_0608_0113_lod_true_raw_rgb16_ram3_dav2s_decoder_e10/gap_report.json`. Best/last checkpoint are both epoch 9: train_proxy `d1=0.8170`, val `d1=0.8205`, gap `-0.0035`.
- `0608_1435` Plan B split neighbor audit: `00Train=2118`, `01Valid=112`; every val pair has nearest train pair distance exactly `1` pair (`val_within_1_pair=100%`). Interpretation: current pair-random split has strong neighbor leakage, so fixed-split train↔val gap is only a lower-bound signal and cannot prove cross-scene generalization; B0 does not show a positive train-only memorization gap, while RAW e10 remains consistent with under-training from the epoch trace.

## Execution Checklist

- [x] Phase 0 canonical pair manifest
- [x] Phase 0b RAW characterization and alignment audit
- [x] Phase 1 pseudo label builder and input-size preview
- [x] Phase 1 full pseudo label generation in tmux
- [x] Phase 2 LOD dataset classes
- [x] Phase 3 config/train wiring
- [x] Phase 4 RAW-RGB16 front-end
- [x] Phase 5 smoke tests
- [x] Phase 6 formal launch scripts

## Confirmed Decisions

用户已确认：

1. Split 使用按 pair 随机 `95/5`，`seed=42`，并定死精确计数为 `2118 train / 112 val`。不强行对齐论文 1830/400 计数。
2. `RGB_Dark` 和 `RAW_Dark` 两条训练输入链路同时修改接入。
3. DAv2-L teacher input_size 主用 `812`（≈LOD 原生短边 800，近原生处理、插值最小，与 ROD「近原生」teacher 口径一致）；生成前做 `812` vs `924` preview，确认轻度上采样是否还能多挤出细节，取更优者写入 `run_config.json`。
4. Formal best checkpoint 使用 LOD validation `d1` 衡量，而不是 `abs_rel`。

## Goal

把真实 LOD 数据集接入当前训练，而不是继续使用之前被误称为 LOD 的 ROD 数据。

数据位置：

```text
/home/caq/6666_raw/0000_dataset/LOD
```

语义映射：

```text
teacher source: LOD RGB_normal -> frozen DAv2-L -> offline pseudo inverse-relative label
student RGB:    LOD RGB_Dark    -> DAv2-S training input
student RAW:    LOD RAW_Dark    -> RAW-RGB16 3ch front-end -> DAv2-S training input
```

当前训练/验证协议应和 ROD pseudo-label 训练保持可比：训练拟合 dense inverse-relative pseudo label，验证也在 pseudo inverse-relative label 空间做 proxy 指标，不解释为真实 metric depth benchmark。LOD formal 的 best checkpoint 按 `d1` 最大化保存。

## Confirmed Dataset Facts

本地 LOD 四目录：

```text
LOD/
  RGB_normal/*.JPG
  RGB_Dark/*.JPG
  RAW_normal/*.png
  RAW_Dark/*.png
```

已核对：

- `RGB_normal`: 2230 files，odd ids `1..4459`，JPG。
- `RGB_Dark`: 2230 files，even ids `2..4460`，JPG。
- `RAW_normal`: 2230 files，same odd ids as `RGB_normal`，PNG。
- `RAW_Dark`: 2230 files，same even ids as `RGB_Dark`，PNG。
- 每个 normal id 都有完整 dark pair：`dark_id = normal_id + 1`。
- 本地根目录未发现 split/annotation 文件。
- RGB 文件为 `800x1200x3 uint8` JPEG。
- RAW 文件为 `800x1200x3 uint16` PNG，注意不是当前 ROD RAW4 Bayer cache。

RAW 输入关键约束：

- 当前 `rod_raw` 路径使用 RAW24 unpack 后的 4-channel packed Bayer `[R, Gr, Gb, B]`。
- 当前真实 LOD RAW 是 3-channel 16-bit PNG。
- 因此 LOD RAW 不能复用 Bayer RAW4 路径；必须新增显式 `raw_rgb16_png_3ch` / `raw_rgb16_dark` 语义和兼容前端。

## External Split Notes

官方 GitHub：

- https://github.com/ying-fu/LODDataset
- README 说明 RGB-normal、RGB-dark、RAW-normal、RAW-dark 四类数据。
- README 说明 short-exposure 文件名对应 long-exposure 文件名加 1。
- README 说明 RAW/RGB 同名数据来自相同场景和照明。
- 未看到公开 split list。

论文/BMVC 资料：

- https://bmva-archive.org.uk/bmvc/2021/conference/papers/paper_0085.html
- 搜索索引到的 BMVC PDF 文本说明 2230 image pairs 被随机分成 1830 train pairs 和 400 test pairs。

因此本项目采用可复现随机 split（按 pair 95/5，精确计数写死为 2118/112）：

```text
split unit: pair_id = normal odd id
ratio: 95/5
train: 2118 pairs
val:   112 pairs
seed:  42
```

注：官方无公开 split 清单，本 split 为复现随机划分，**不等于**论文 1830/400 那批 train/test，结果不可与论文 detection split 直接对号；split summary 里需写死此声明。不要在实现中用浮点比例临时 round/floor/ceil，manifest builder 直接接收或内置 `train_count=2118, val_count=112`。

## Current Codebase Context

当前 ROD 低光训练链路：

- `finetune_stf/dataset/rod_raw_student_rgb.py`
  - 在线生成 student dark RGB。
  - 读取离线 DAv2-L teacher pseudo label。
  - 返回 `target_space=inverse_relative`。
- `finetune_stf/dataset/rod_raw.py`
  - 读取 ROD RAW24 / RGGB cache。
  - 返回 4ch Bayer `raw`。
- `finetune_stf/train.py`
  - 当前 stage 包含 `stf_only / rod_only / eval_only`。
  - ROD eval 已经按 inverse-relative pseudo label 做 proxy 指标。
- `finetune_stf/config/resolved.py`
  - 当前支持 `rod_raw_student_rgb` 和 `rod_raw`。
  - 尚不支持真实 LOD family。

旧 LOD 代码不可直接复用：

- `finetune_stf/dataset/lod_raw.py` 假设旧 manifest 和 `(928,1440,4)` RGGB `.npy`。
- `scripts/generate_lod_day_dav2_pseudo.py` 假设旧 `00Train/01Valid`、day/night、SDR RGB、RGGB `.npy` 布局。
- 当前真实 LOD 是四目录 `JPG/PNG` 布局，需要新 manifest 和新 dataset。

## Phase 0: Canonical Pair Manifest

新增脚本：

```text
scripts/build_lod_true_pair_manifest.py
```

职责：

- 验证四个目录存在。
- 验证每个 pair 完整：
  - `RGB_normal/<normal_id>.JPG`
  - `RGB_Dark/<normal_id+1>.JPG`
  - `RAW_normal/<normal_id>.png`
  - `RAW_Dark/<normal_id+1>.png`
- 验证 shape/dtype：
  - RGB: `800x1200x3 uint8`
  - RAW: `800x1200x3 uint16`
- 用 `seed=42` 按 pair 生成精确 `2118/112` split。
- 写 canonical manifest 和 sidecar metadata。

输出：

```text
/home/caq/6666_raw/0000_dataset/LOD/manifests/lod_true_pairs_2118_112_seed42.csv
/home/caq/6666_raw/0000_dataset/LOD/manifests/lod_true_pairs_2118_112_seed42.meta.json
```

Manifest 字段：

```text
pair_id
split
normal_id
dark_id
rgb_normal_path
rgb_dark_path
raw_normal_path
raw_dark_path
label_space
height
width
```

Split names 建议沿用当前训练代码风格：

```text
00Train
01Valid
```

## Phase 0b: RAW Characterization & Pair Alignment Audit

只读数据出报告，不改训练码；作为 `/65535` 归一化与几何对齐口径的**硬闸**，通过后才进入实现。

输出：`finetune_stf/analysis/lod_ingest_audit/<MMDD_HHMM>_lod_true_audit/`；可另维护一个 `latest` 软链接或摘要文件，但每次审计必须写入 run-specific 子目录，避免覆盖历史审计结果。

1. RAW 强度 / 黑电平刻画
   - 统计 `RAW_normal` / `RAW_Dark` 的 p0/p1/p50/p99/max、是否 clip 到 65535、逐通道分布。
   - ⚠️ 已知异常待解释：实测 `RAW_Dark` 均值(≈15344) **高于** `RAW_normal`(≈7706)，p99 也更高，疑似 dark raw 已做增益放大。需确认 `/65535` 归一化是否失真；必要时改黑电平减除 / 缩放，并写成显式 norm 常量。

2. dark↔normal 空间对齐审计
   - 对随机 ~100 对算下采样 NCC 分布（已抽验 normal#1 vs dark#2 NCC=0.67，vs 无关帧 −0.17）。
   - 若存在系统性平移 / 错位，需决定裁剪 / 对齐，否则 teacher(来自 normal) 的深度赋给 dark 输入的几何前提不成立。

3. 通道序确认
   - 确认 `RAW` PNG 三通道的实际 RGB 顺序，定死与 ROD `base_rgb` `[R, G, B]` 一致的读取口径（见 Phase 2 / Phase 4）。

仅当三项审计通过（归一化常量与对齐 / 通道序口径确定）才进入 Phase 1 之后的代码实现。

## Phase 1: Offline RGB_normal Pseudo Labels

新增脚本：

```text
finetune_stf/scripts/build_lod_true_rgb_normal_teacher_labels.py
```

职责：

- 读取 Phase 0 pair manifest。
- 用 `rgb_normal_path` 作为 teacher input。
- 加载 frozen DAv2-L：
  - `encoder=vitl`
  - checkpoint `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth`
- 主用 `input_size=812`（近原生，插值最小，质量最高）；做 `812` vs `924` preview，确认轻度上采样是否还能多挤出细节。
- 正式 input size 根据 preview 结果固定并写入 `run_config.json`。
- 输出 dense inverse-relative `.npy`，shape `800x1200`。
- 输出 pseudo 可视化 PNG。
- 写训练 manifest、`run_config.json`、`run_summary.json`。

建议输出：

```text
/home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/
  00Train/*.npy
  01Valid/*.npy
  vis/...
  lod_true_rgb_normal_dav2l_rel_manifest.csv
  run_config.json
  run_summary.json
```

Pseudo manifest 字段：

```text
pair_id
split
normal_id
dark_id
rgb_normal_path
rgb_dark_path
raw_normal_path
raw_dark_path
pseudo_depth_path
label_space
height
width
teacher_source
```

长任务规则：

- 正式 2230 样本生成必须放入 tmux。
- session/log 名称清晰。
- 启动后报告：
  - tmux session
  - log path
  - attach command
  - tail command

## Phase 2: LOD Dataset Classes

新增模块：

```text
finetune_stf/dataset/lod_true.py
```

同时实现两条数据链路。

### RGB_Dark Dataset

类名建议：

```text
LODTrueRGBDark
```

语义：

```text
student input: RGB_Dark/<dark_id>.JPG
teacher label: RGB_normal -> DAv2-L pseudo label
model_input_tensor: image
input_domain: rgb
front_end: dav2_rgb
```

处理：

- 读取 dark JPG，BGR->RGB，转 float `[0,1]`。
- 如果输入与 pseudo label shape 不一致则显式报错或按受控策略 resize；当前预期均为 `800x1200`，优先报错。
- 在 full native `800x1200` 上和 pseudo label 共享 crop box。
- train crop random，val crop center。
- first formal crop size 沿用 ROD：`512x960`。
- ImageNet normalize + `PrepareForNet`。

### RAW_Dark RGB16 Dataset

类名建议：

```text
LODTrueRawDarkRGB16
```

语义：

```text
student input: RAW_Dark/<dark_id>.png
raw_storage_format: raw_rgb16_png_3ch
model_input_tensor: raw
input_domain: raw3 or raw_rgb16
front_end: raw_rgb16_ram3
```

处理：

- 读取 `uint16` PNG，shape `800x1200x3`。
- ⚠️ 通道序：`cv2.IMREAD_UNCHANGED` 读出为 **BGR**，必须显式 BGR→RGB，使进 `RamCore3` 的通道序与 ROD `packed_bayer_to_base_rgb` 输出的 `[R, meanG, B]` 一致，否则破坏前端语义 / 权重迁移。
- 显式 normalization，第一版建议：
  - `lod_raw_norm_mode=uint16_div_65535`
  - raw = `raw_uint16.astype(float32) / 65535.0`
  - （最终归一化常量由 Phase 0b 刻画确认，尤其 dark>normal 增益异常）
- 不做 Bayer pack，不做 `[R,mean(G),B]` 投影。
- 和 pseudo label 共享 crop box。
- 返回：
  - `raw`: `[3,512,960]`
  - `image`: preview `[3,512,960]`，用于日志/可视化
  - `depth`
  - `valid_mask`
  - `target_space=inverse_relative`
  - source paths

## Phase 3: Config and Train Wiring

在 `finetune_stf/config/resolved.py` 中新增显式 choices。

Dataset family：

```text
lod_true_rgb_dark
lod_true_raw_dark_rgb16
```

Dataset input mode：

```text
rgb_dark
raw_rgb16_dark
```

Raw storage format：

```text
raw_rgb16_png_3ch
```

同时必须更新：

- `finetune_stf/dataset/raw_storage.py`
  - 新增 `RawStorageSpec(name="raw_rgb16_png_3ch", ...)`。
  - `storage_channel_order` 先按 OpenCV 读取口径写为 `("B", "G", "R")`，最终由 Phase 0b 通道序审计确认；如审计推翻，必须更新 spec 与 manifest metadata。
  - `model_channel_order=("R", "G", "B")`。
  - `channel_reorder=(2, 1, 0)`。
  - `decompand="none"` 或明确命名为 `uint16_linear`。
  - `post_decode_norm="uint16_div_65535"`，如果 Phase 0b 决定改黑电平/缩放，则这里改成对应显式名称。
- `finetune_stf/config/resolved.py`
  - `RAW_STORAGE_FORMAT_CHOICES` 增加 `raw_rgb16_png_3ch`。
  - `_raw_storage_details()` 能正常解析该 spec，不能只改 parser choices。

Front-end：

```text
raw_rgb16_ram3
```

`raw_rgb16_ram3` 不是单个 wrapper 名称，必须完整接入 resolved config 和 train pipeline：

- `INPUT_DOMAIN_CHOICES` 增加 `raw3` 或 `raw_rgb16`，正式采用哪一个要统一；本计划后续命令暂用 `raw3`。
- `FRONT_END_CHOICES` 增加 `raw_rgb16_ram3`。
- `_infer_from_front_end()`：`raw_rgb16_ram3` 推导为 `input_domain=raw3`、`dataset_input_mode=raw_rgb16_dark`、`model_input_tensor=raw`。
- `_ram_core_type()`：`raw_rgb16_ram3 -> RamCore3`。
- `_imagenet_norm_enabled()`：`raw_rgb16_ram3 -> False`，沿用 RamCore3 直接送 DAv2 的语义。
- `build_model()`：新增分支构建 `RawRgb16Ram3DepthModel`。
- `_required_optimizer_groups()`：把 `raw_rgb16_ram3` 加入需要 `raw_front_end` 参数组的 front-end 集合。
- `log_setup()`：打印 `raw_rgb16_png_3ch -> RamCore3 -> identity -> DAv2`，不要落入旧 ROD RAW4 日志。

Stage / eval：

- 新增 `lod_only` stage。
- 新增 `--eval-lod`。
- 新增 `best_metric=lod_d1`。
- 训练 loader 需要从硬编码 `rod_train/stf_train` 泛化，支持 `lod_train`。
- primary eval tag 建议：
  - `lod_val`
  - writer prefix `eval_lod`

Best checkpoint 逻辑必须同步修改：

- 当前 `train.py` 的 best metric 逻辑按 `abs_rel` 越小越好更新。
- LOD formal 要按 `d1` 越大越好，因此不能复用 `lod -> abs_rel` 的隐式语义。
- 建议新增 `BEST_METRIC_CHOICES += ("lod_d1",)`。
- 新增 metric direction map，例如：
  - minimize: `stf`, `rod`, `kitti`, `eth3d`, `robotcar`, `robotcar_day`, `robotcar_night`, `avg4`
  - maximize: `lod_d1`
- `metric_values["lod_d1"] = lod_summary["d1"]`。
- checkpoint payload 增加 `best_lod_d1`，resume 时也按 maximize 恢复。
- 初始化时 maximize 指标不能用 `inf`，应使用 `-inf` 或通过 direction map 提供 identity value。
- 更新比较不能写死 `<`，必须按 direction 调用统一 helper，例如 `metric_improved(name, new, old)`。
- `summary` 在非 ROD 时当前容易被当作 STF `abs_rel`，LOD 接入必须新增 `uses_lod_dataset()` / `is_lod_primary_eval` 分支，避免 LOD val 同时污染 `stf` best。
- epoch-start 日志不能继续硬写 `best_%s_abs_rel`；应按 metric 名和方向打印，例如 `best_lod_d1=... (higher better)`。
- 日志应写清：`best_lod_d1` 越高越好。

必须集中校验：

`lod_true_rgb_dark`：

- `input_domain=rgb`
- `front_end=dav2_rgb`
- `model_input_tensor=image`
- `dataset_input_mode=rgb_dark`
- `raw_storage_format=n_a` 或 `none`
- `bridge=none`
- `decoder_feature_adapter=none`

`lod_true_raw_dark_rgb16`：

- `dataset_input_mode=raw_rgb16_dark`
- `raw_storage_format=raw_rgb16_png_3ch`
- `model_input_tensor=raw`
- `front_end=raw_rgb16_ram3`
- `lod_raw_norm_mode` 必须显式设置
- 禁止使用 ROD/STF Bayer RAW4 storage format

## Phase 4: RAW-RGB16 Front-End

新增模型 wrapper，复用现有 `RamCore3`，但不复用 Bayer 投影。

建议位置：

```text
finetune_stf/models/raw_ram.py
```

新增 wrapper：

```text
RawRgb16Ram3DepthModel
```

语义：

```text
raw_rgb16_3ch -> RamCore3 -> optional tail -> DAv2
```

与当前 `RawToBaseRgbRam3DepthModel` 的区别：

- 当前 ROD path:
  - `raw4 -> packed_bayer_to_base_rgb(raw4) -> RamCore3 -> DAv2`
- 新 LOD RAW path:
  - `raw3 -> RamCore3 -> DAv2`

空间适配器：

- `CenterPadCropAdapter(SENSOR_INPUT_HW=512x960, BACKBONE_INPUT_HW)` 沿用不改（LOD crop 与 ROD 同为 `512x960`，pad/crop 逻辑不变）。
- 实现时不要漏掉 `pad_rgb` / `crop_depth`，否则尺寸对不上。

第一版 tail：

- 沿用可比较设置：`raw_ram_rgb_tail=identity`。
- 如稳定性差，再显式测试 `tanh2p5`，不能隐藏默认改变实验语义。

日志必须写清：

```text
[MODEL] raw_rgb16_png_3ch -> RamCore3 -> identity -> DAv2
```

## Phase 5: Smoke Tests

全部 smoke 输出路径必须含 `codex_smoke`。

Smoke 1: pair manifest

```text
--max-pairs 16
```

检查：

- pair 完整。
- split metadata 正确。
- shape/dtype 正确。

Smoke 2: pseudo label input-size preview

```text
input_size: 812, 924
max_samples: 20
output: /tmp/codex_smoke_lod_true_pseudo_size_sweep_*
```

检查：

- `.npy` shape `800x1200`。
- finite positive coverage。
- 可视化无明显空图/严重伪影。

Smoke 3: pseudo label generation

```text
max_samples: 4
output: /tmp/codex_smoke_lod_true_pseudo_*
```

成功后删除临时 smoke 输出。

Smoke 4: dataset loading

同时验证：

- `LODTrueRGBDark`
- `LODTrueRawDarkRGB16`

检查：

- train random crop shape：
  - RGB image `[3,512,960]`
  - RAW raw `[3,512,960]`
  - depth `[512,960]`
- val center crop shape。
- `target_space=inverse_relative`。

Smoke 5: train smoke

RGB_Dark:

```text
--stage lod_only
--dataset-family lod_true_rgb_dark
--debug-max-train-steps 1
--debug-max-val-samples 2
--save-path /tmp/codex_smoke_lod_true_rgb_train
--heavy-save-root /tmp/codex_smoke_lod_true_rgb_heavy
```

RAW_Dark:

```text
--stage lod_only
--dataset-family lod_true_raw_dark_rgb16
--debug-max-train-steps 1
--debug-max-val-samples 2
--save-path /tmp/codex_smoke_lod_true_raw_train
--heavy-save-root /tmp/codex_smoke_lod_true_raw_heavy
```

成功后只删除明确含 `codex_smoke` 的临时目录。失败则保留并报告路径。

## Phase 6: Formal Scripts

新增 formal launch scripts：

```text
finetune_stf/scripts/formal/0608_run_lod_true_rgb_dark_decoder_e10_queue.sh
finetune_stf/scripts/formal/0608_run_lod_true_raw_rgb16_ram3_decoder_e10_queue.sh
```

正式实验名必须在 launch 时以当前时间 `MMDD_HHMM` 开头。

长任务必须 tmux，且不复用已有 session。

### RGB_Dark First Formal

核心参数：

```bash
--encoder vits
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--stage lod_only
--lod-root /home/caq/6666_raw/0000_dataset/LOD
--lod-manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest.csv
--dataset-family lod_true_rgb_dark
--dataset-input-mode rgb_dark
--input-domain rgb
--front-end dav2_rgb
--model-input-tensor image
--raw-storage-format n_a
--bridge none
--decoder-feature-adapter none
--lora none
--dav2-train-mode decoder
--input-height 512
--input-width 960
--lod-train-crop-mode random
--lod-val-crop-mode center
--loss-type ssi
--loss-target-normalization
--loss-norm-min-scale 1e-3
--epochs 10
--bs 8
--accum-steps 1
--lr 1e-5
--amp
--amp-dtype bf16
--seed 42
--num-workers 4
--no-eval-stf
--eval-lod
--best-metric lod_d1
--save-best-checkpoint
```

### RAW_Dark First Formal

核心参数：

```bash
--encoder vits
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--stage lod_only
--lod-root /home/caq/6666_raw/0000_dataset/LOD
--lod-manifest /home/caq/6666_raw/0000_dataset/LOD/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest.csv
--dataset-family lod_true_raw_dark_rgb16
--dataset-input-mode raw_rgb16_dark
--input-domain raw3
--front-end raw_rgb16_ram3
--model-input-tensor raw
--raw-storage-format raw_rgb16_png_3ch
--lod-raw-norm-mode uint16_div_65535
--bridge none
--decoder-feature-adapter none
--lora none
--dav2-train-mode decoder
--raw-front-end-lr 5e-5
--raw-ram-rgb-tail identity
--input-height 512
--input-width 960
--lod-train-crop-mode random
--lod-val-crop-mode center
--loss-type ssi
--loss-target-normalization
--loss-norm-min-scale 1e-3
--epochs 10
--bs 8
--accum-steps 1
--lr 1e-5
--amp
--amp-dtype bf16
--seed 42
--num-workers 4
--no-eval-stf
--eval-lod
--best-metric lod_d1
--save-best-checkpoint
```

## Implementation Order

1. Add pair manifest builder and run it with exact `train_count=2118, val_count=112, seed=42`.
2. Run Phase 0b RAW characterization + pair alignment audit; lock normalization / alignment / channel-order before any training-code change.
3. Add pseudo label builder and run `812/924` smoke preview (`812`≈native primary).
4. Generate full RGB_normal DAv2-L pseudo labels in tmux.
5. Add `LODTrueRGBDark` and `LODTrueRawDarkRGB16`.
6. Add `raw_rgb16_png_3ch` raw storage spec, config choices, and central validation.
7. Add `raw_rgb16_ram3` model wrapper and all resolved-config / optimizer / logging branches.
8. Wire `lod_only`, `eval_lod`, `uses_lod_dataset()`, `best_metric=lod_d1`, and maximize-based best checkpoint logic.
9. Run dataset and train smoke for both RGB_Dark and RAW_Dark.
10. Add formal scripts.
11. Launch formal experiments only after smoke passes.

## Risks / Things To Watch

- RAW_Dark PNG dynamic range may not behave like display RGB or Bayer RAW; first RAW formal must be treated as a new front-end experiment, not a ROD RAW clone.
- `input_domain=raw3` may require expanding current config choices beyond `rgb/raw4`; this is desired to avoid hiding a semantic change.
- Pseudo label quality depends on `RGB_normal` DAv2-L preprocessing size; the `812/924` preview is a required gate（`812`≈native primary）。
- LOD val is pseudo-label proxy validation, not metric-depth validation.
- Split 为按 pair 随机 `2118/112`（seed42），仅复现性划分，非论文 1830/400，不与论文 detection split 对号。
- RAW 通道序（BGR→RGB）与 dark↔normal 几何对齐、`/65535` 归一化合理性，统一由 Phase 0b 审计闸把关；未通过不得进入实现。
- If annotation XMLs are later added, the random split can be replaced by class-stratified split, but that must be a new explicit split manifest.

## Follow-up Note: Training Speed / Overfitting Risk

LOD formal 训练非常快，主要原因之一是数据量显著小于 ROD night：

- LOD true split: train `2118`, val `112`, batch size `8`, 每个 epoch `264` optimizer steps。
- ROD night split: train `12036`, val `2000`, batch size `8`, 每个 epoch `1504` optimizer steps。
- Step budget 对照：ROD 每个 epoch 约为 LOD 的 `1504 / 264 ~= 5.7x`；LOD `10` epochs 约 `2640` steps，只相当于 ROD 约 `1.75` epochs 的 step 数。
- 实测耗时：LOD RGB_Dark 约 `20s/epoch`，LOD RAW_RGB16 约 `47-50s/epoch`；ROD RGB decoder-only epoch 0 约 `33min`，ROD RAW full-backbone epoch 0 约 `69min`。
- Val 耗时也差异很大：LOD val `112` samples，约 `4s/eval`；ROD val `2000` samples，约 `3-5min/eval`。

由于 LOD 样本量和验证集都较小，后续更长训练可能更容易出现过拟合或 val proxy 噪声放大。后续实验可以考虑在保持实验语义显式的前提下引入 data augmentation，例如轻量 photometric jitter、random horizontal flip、crop/scale jitter，尤其优先给 RAW_RGB16 长训对照使用。

---

## 0608 Plan B: Data Augmentation + Anti-Overfitting / Under-training

> 本节是 0608 True LOD 接入完成后的后续计划，针对两个独立但相关的问题：
> 1. **疑似过拟合（RGB 链路）**：可视化发现 RGB_Dark 输入虽然极暗，但对应 pred 仍能看出大量结构。怀疑模型在 2118 个固定场景上记忆 scene→pseudo-depth，而非学到泛化的低光深度。
> 2. **训练不充分（RAW 链路）**：RAW_RGB16 在 e10 时 best 出现在最后一个 epoch（`d1=0.8205@ep9`），val 仍在爬升，明显欠训练。

### B0. 问题诊断（先量化，再动手）

在引入 augmentation 之前，必须先用数据确认"过拟合"假设，避免把欠训练误判为过拟合、或反之。

#### B0.1 train↔val proxy gap 诊断

当前只在 `lod_val`（112 样本）上报指标，无法判断 train 是否被记忆。新增一个**固定 train 子集**的 proxy 评估（center crop，no aug），与 val 同口径比较：

- 从 `00Train` 取固定子集（建议 `seed=42` 取 `112` 个，和 val 同量级，便于直接比较 `d1`）。
- 评估 tag 建议 `lod_train_proxy`，writer prefix `eval_lod_train`。
- 关键指标：`gap = d1(train_proxy) - d1(val)`。
- 重要限制：当前 `00Train/01Valid` 是 seed42 pair-random split，val 与 train 存在相邻帧/相邻 pair 的高相似性。因此 train↔val gap 只能诊断**当前固定 split 上的拟合/记忆程度**，不能单独证明跨场景泛化。
- 判据：
  - `gap` 小（≈0）且 val 仍在升 → 当前 split 上没有明显 train-only 记忆信号，**欠训练**优先。
  - `gap` 明显为正且随 epoch 拉大、val 转平/回落 → 当前 split 上的**过拟合/记忆**确认，augmentation 优先。

#### B0.2 直接复用现有 best + last checkpoint 做一次性诊断

无需重训：用已存的两个 formal run 同时评估 **best checkpoint** 与 **last_epoch checkpoint**（RGB `0608_0109...`、RAW `0608_0113...`），各跑一次 train-proxy + val 评估，得到当前 gap 基线和 late-epoch gap。这是开发 augmentation 前的硬性前置数据点，写入本计划 Execution Log。

```text
新增脚本: finetune_stf/scripts/eval_lod_train_val_gap.py
输入: --checkpoint, --checkpoint-label {best,last}, --dataset-family, manifest, --train-proxy-count 112 --seed 42
输出: finetune_stf/analysis/lod_overfit_diag/<MMDD_HHMM>_<run>/gap_report.json + 可视化对照；报告中按 checkpoint_label 分组
```

> 注：若 B0.2 显示 RGB last gap 明显大于 RGB best，且 RAW best/last gap 仍小或 val 仍升，则印证"RGB 更可能过拟合 / RAW 更可能欠训练"。两条线的后续配方应区别对待（RGB 重正则 + 适度 epoch；RAW 加 epoch 为主 + 中等正则）。

#### B0.3 split 邻近性审计 + block/neighbor-excluded holdout 预案

**B0.3 是 B0.1/B0.2 的前置硬闸**：必须先跑、其结果决定 gap 用哪个 split 解释，不能反过来。先新增只读 split audit，量化每个 `01Valid` pair 到最近 `00Train` pair 的 ID/时间邻近距离，并写入 `gap_report.json`。

- 若 val 大量被相邻 train pair 包围（强邻近泄漏）：当前 `val d1≈0.82` 本身可能虚高，且 train/val 都"见过"相邻场景会**压小** gap、反而掩盖过拟合。此时 B0.1/B0.2 的 fixed-split gap 只能当下限信号，过拟合的主判读必须改到下面的 neighbor-excluded holdout 上重算。
- 若邻近性弱：fixed-split gap 可直接作为过拟合主证据。

`block/neighbor-excluded` 诊断 split：按连续 ID block 或时间段选 val，并从 train 中排除 val 邻域 `±K` pairs。它既是邻近泄漏成立时的 gap 重算口径，也是跨场景/跨序列泛化结论的唯一合法依据。该 split 单独成 follow-up，不混入当前 e40 augmentation 主矩阵（主矩阵仍用 seed42 split 以与现有 baseline 对照）。

### B1. Augmentation 设计原则

1. **实验语义显式**（遵守全局 CLAUDE.md 第 6 条）：所有 augmentation 开关与强度都是 experiment-semantic 参数，必须在 formal 脚本显式设置，集中在 `resolved.py` 校验并写入 resolved config + `log_setup()`。当前 `train.py` 中隐藏的 runtime hflip=0.5 也必须迁入显式 aug config；实现后 LOD 训练不允许再保留独立 runtime hflip。
2. **默认关闭，但 baseline 显式复现**：代码默认 `aug-preset=off` 表示真正无增广（包括 `hflip=0.0`）。现有 e10 baseline 实际包含隐藏 runtime hflip=0.5（[train.py:3625-3629](../finetune_stf/train.py#L3625)），因此复现实有 baseline 时必须显式使用 `aug-preset=baseline_e10` 或 `--aug-hflip-prob 0.5`，并在 resolved config 中记录。
   - ⚠️ 语义差：旧 runtime hflip 是 **batch 级**（整个 batch 一起翻），迁入 aug config 后是 **sample 级独立 50%**。两者期望相同、统计等价，但**不是 bit-exact**。因此 `baseline_e10` 是与历史 e10 的**统计等价**复现，不是逐像素复现；sample 级多样性略高，属可接受改进，但 fairness summary 必须写明此差异。
3. **只作用于 train**：`mode=="val"` 一律不增广（保持 center crop、scale=1.0、no photometric），val proxy 口径不被污染，结果与现有 baseline 可比。
4. **几何增广 input 与 label 必须严格联动**：flip / resize / crop 同一套参数同时作用于 student input、pseudo depth、valid_mask；mask 用最近邻、depth 用 bilinear、recompute `valid = finite & >0`。
5. **SSI 损失的便利**：当前 `loss-type=ssi` + target-normalization 对全局尺度/平移不变，因此 **全局亮度 / 增益 / gamma** 类增广不破坏 label 语义（深度值本身不随图像 resize / 亮度变化而改变），可以较大胆使用。
6. **保深度先验**：用 horizontal flip，**禁止 vertical flip**（破坏地面/天空重力先验）；旋转默认关闭（边界 invalid + 破坏深度边界），仅作为可选小角度实验。
7. **域适配的 photometric**：RGB 与 RAW 共享同一套几何增广（公平），但 photometric 分域：
   - RGB 走显示域 jitter（brightness/contrast/gamma/color/noise）。
   - RAW 走**线性域**物理增广（per-channel gain、black-level offset、Poisson-Gaussian 噪声），**禁止 hue/saturation jitter**，必须保持送入 `RamCore3 -> identity -> DAv2` 的通道语义和 `[R,G,B]` 顺序不变。
8. **公平性边界（几何 vs 光度）**：几何增广（hflip/scale/crop）对两域是同一算子、同强度、且同时作用于 label，是**真正单变量**扰动，作为 RGB-vs-RAW pipeline 主对照的公平基础。分域 photometric 是不同算子且强度不可标定互比（`RGB brightness=0.2` 与 `RAW gain=0.7,1.4` 无等正则对应），会给每条链路各引入一个自由超参，**因此不得作为 headline pipeline 对照的一部分**：主公平对照固定在纯几何配方（B2.4 `geom` 预设、B5 矩阵），photometric 仅作分域 best-effort 次要轴 + 排序稳定性鲁棒检查。

### B2. Augmentation 菜单（含默认强度，全部默认 OFF）

#### B2.1 几何（RGB / RAW 共享）

| 增广 | flag | 默认 | 建议强度 |
| --- | --- | --- | --- |
| 水平翻转 | `--aug-hflip-prob` | `0.0` | `0.5` |
| 尺度抖动 + 裁剪 | `--aug-scale-jitter`（min,max） | `none`（=固定 1.0 原生裁剪） | `0.7,1.4` |
| 旋转（可选） | `--aug-rotate-deg` | `0.0` | 默认不用；如试 `±3~5°` |

- 尺度抖动实现：对原生 `800x1200` 按 `s∈[min,max]` resize 后随机裁 `512x960`；若 `s<crop/native` 导致短边不足，则 reflect-pad 或将 `s` 下限 clamp 到可裁尺寸（实现时取后者，避免引入 padding 边界 invalid）。`s=1.0` 时必须**逐像素等价于现有 `_sample_crop_box`+`_apply_crop`**，保证关闭时 baseline 不变。
- flip 命中时更新 `geometry_params["hflip_applied"]=True`（该字段已是占位符）。

#### B2.2 Photometric — RGB（作用于 `[0,1]` RGB，ImageNet normalize 之前）

| 增广 | flag | 默认 | 建议强度 |
| --- | --- | --- | --- |
| 亮度 | `--aug-rgb-brightness` | `0.0` | `0.2` |
| 对比度 | `--aug-rgb-contrast` | `0.0` | `0.2` |
| gamma | `--aug-rgb-gamma`（min,max） | `none` | `0.8,1.2` |
| 色彩 hue/sat | `--aug-rgb-color` | `0.0` | `0.1`（轻） |
| 高斯噪声 std | `--aug-rgb-noise-std` | `0.0` | `0.02` |
| 高斯模糊概率 | `--aug-rgb-blur-prob` | `0.0` | 可选 `0.1` |

> 低光图本就暗，亮度/对比度抖动取偏小值，避免把信息抖没；噪声是低光最贴合实际、最有效的正则项之一。

#### B2.3 Photometric — RAW（作用于 `/65535` 归一化后的线性空间，RamCore3 之前）

| 增广 | flag | 默认 | 建议强度 |
| --- | --- | --- | --- |
| 全局/逐通道增益 | `--aug-raw-gain`（min,max） | `none` | `0.7,1.4` |
| 逐通道增益独立性 | `--aug-raw-per-channel-gain` | `false` | 可选 `true`（轻微白平衡扰动） |
| 黑电平偏移 | `--aug-raw-black-offset` | `0.0` | `0.002`（归一化单位） |
| Poisson-Gaussian 噪声 | `--aug-raw-noise` | `none` | shot+read，轻量 |

- 所有 RAW photometric 后 clamp 到 `[0,1]` 并保留现有 finite 检查。
- **不做** hue/saturation/gamma-display 类操作；增益/噪声是 RAW 域物理可解释的扰动。
- per-channel gain 若开启需保持三通道顺序不变，仅扰动幅度。

#### B2.4 强度分级预设（便于 A/B，少调 flag）

为减少 formal 脚本里散落十几个 flag，建议提供 `--aug-preset {off,baseline_e10,geom,light,medium,heavy}`，preset 展开为上面各 flag 的具体值，并在 resolved config 里**展开记录每一项实际值**（不能只记 preset 名，避免隐藏语义）。preset 仅是便捷入口，任何单 flag 显式覆盖优先。

- `off`：真正无 augmentation，`hflip=0.0`、无 scale jitter、无 photometric；这是新增的严格 no-aug ablation，不等同于现有 e10 baseline。
- `baseline_e10`：hflip0.5 + 无 scale jitter + 无 photometric；用于显式复现实有 e10 baseline 中隐藏的 runtime hflip。
- `geom`（**纯几何，跨域逐项相同**）：hflip0.5 + scale 0.7,1.4 + **无 photometric**；几何强度与 `medium` 对齐，使 `Rg→R1` / `Wg→W1` 恰好隔离分域 photometric 的增量。这是 RGB-vs-RAW pipeline 主公平对照的标准配方（两域同算子同强度、同时作用于 label；待比较变量是输入域 + front-end / trainable path），也是针对"场景记忆"过拟合最直接的杠杆（scale-jitter 改变取景/尺度签名）。
- `light`：hflip0.5 + scale 0.85,1.2 +（RGB 噪声0.01 / RAW gain 0.85,1.2）。
- `medium`：hflip0.5 + scale 0.7,1.4 +（RGB brightness/contrast0.2+gamma+noise0.02 / RAW gain0.7,1.4+noise+black-offset）。
- `heavy`：medium 基础上加更宽 scale、color、更强噪声（仅在 medium 仍过拟合时用）。

> 公平性提醒：`off` / `baseline_e10` / `geom` 三档对两域**逐项相同**（无 photometric 或仅几何），是合法的单变量跨域 pipeline 对照；`light` / `medium` / `heavy` = 几何 + **分域 photometric**，跨域不再单变量，只能作为各域 best-effort，不能直接当 pipeline 对照结论（见 B1.8、B5）。

### B3. 代码改动点

#### B3.1 新增增广模块

```text
finetune_stf/dataset/lod_aug.py
```

- `@dataclass LODAugConfig`：显式持有上面所有字段 + `domain∈{rgb,raw3}` + `enabled` + `preset_name`。
- `LODAugConfig.from_args(args)` / `from_preset(name)`：集中解析与 preset 展开。
- `apply_geometric(input_hw_arrays, depth, mask, rng, cfg) -> (...)`：flip + scale-crop + (rotate)，input/depth/mask 联动。
- 几何 RNG 必须可配对复现：对 LOD formal 主矩阵，hflip/scale/crop 的随机参数由 `(global_seed, epoch, sample_id 或 manifest_index)` 派生，而不是依赖 dataloader 到达顺序。这样 `Rg/Wg` 对同一 pair 使用同一组几何参数，避免 realized augmentation noise 混入 RGB-vs-RAW 主对照；若实现上无法严格配对，必须至少把每个 sample 的 geometry params 记录到 debug/audit 输出，并在 summary 中声明未配对风险。
- `apply_photometric_rgb(rgb01, rng, cfg)` / `apply_photometric_raw(raw01, rng, cfg)`。
- 必须有 `cfg.enabled is False` 时的 **bit-exact passthrough**（几何退回到现有 crop 行为）。`baseline_e10` 不走旧 train-loop hflip，而是在 dataset/aug 模块内按 `aug_hflip_prob=0.5` 显式执行。

#### B3.2 `finetune_stf/dataset/lod_true.py`

- `_LODTrueBase.__init__` 增加 `aug_config: LODAugConfig | None = None`，仅在 `mode=="train"` 且 `aug_config.enabled` 时生效。
- `LODTrueRGBDark.build_sample`：crop 阶段改走 `apply_geometric`，normalize 前插入 `apply_photometric_rgb`。
- `LODTrueRawDarkRGB16.build_sample`：crop 阶段改走 `apply_geometric`，`RamCore3` 前（即 `_chw_tensor` 前、归一化后）插入 `apply_photometric_raw`。
- 两条链路用**同一份几何 cfg**，photometric 走各自分支，保证公平对照。

#### B3.3 `finetune_stf/config/resolved.py`

- 新增全部 `--aug-*` flag 与 `--aug-preset`（experiment-semantic）。
- `--aug-preset` choices 至少为 `{off,baseline_e10,geom,light,medium,heavy}`；resolved config 必须展开记录最终 `aug_hflip_prob`、scale-jitter、各 photometric 实际值，避免把现有 hidden hflip 或 preset 差异藏回 preset 名。
- 新增 LR schedule experiment-semantic flags：`--lr-schedule {poly,constant,cosine}`、`--warmup-steps`。默认/基线为当前实现的 `poly`（`(1 - iter/total_iters)^0.9`）与 `warmup_steps=0`；cosine + warmup 只能作为单独 schedule axis，不和 aug 主效应混在一起解释。
- 集中校验：
  - `aug-preset=off` 与任何显式 `aug-*>0` 同时出现时，以显式值为准并 warn；或要求二选一（实现时取"显式覆盖 preset"且记录最终展开值）。
  - RAW 链路禁止出现 RGB-only 的 color/gamma flag 为非零（`raw3` 域传 `--aug-rgb-color>0` 应 raise，避免跨域误用）。
  - `aug-rotate-deg>0` 时打印强警告（深度边界风险）。
- resolved config dump **展开记录每一项 aug 与 LR schedule 实际值**；`log_setup()` 打印 `[AUG] domain=... preset=... hflip=... scale=... photometric=...` 与 `[LR] schedule=... warmup_steps=...`。

#### B3.4 `finetune_stf/train.py`

- 仅给 **train dataset** 注入 `aug_config`；val dataset 永远 `enabled=False`。
- `build_datasets()` / dataset wrapper 必须把 formal seed、epoch、sample id/index 传给 `LODAugConfig` 的几何 RNG 派生逻辑，保证 `geom` 档 RGB/RAW 两条链路对同一 sample 使用同一 hflip/scale/crop 参数。DistributedSampler shuffle 或 dataloader worker 数变化不得改变单样本几何参数。
- 迁移现有 runtime hflip：LOD 训练循环不得再执行独立 `random.random()<0.5` flip；所有 LOD hflip 只由 `LODAugConfig.aug_hflip_prob` 控制，避免双重 flip 或 hidden baseline。
  - ⚠️ 当前 [train.py:3625](../finetune_stf/train.py#L3625) 的 `apply_runtime_hflip = not uses_stf_raw_dataset(args)` 同时覆盖 ROD/STF（非 raw）路径。迁移时**只能摘掉 LOD 分支**，ROD/STF 的 runtime hflip 行为必须保持原样，不得顺手改动，否则会改变其它实验语义。
- ⚠️ RAW 链路 model 输入 tensor 一致性：现有 [train.py:3639](../finetune_stf/train.py#L3639) 喂的是 `img`，而 LOD RAW dataset 里 `image` 是 `raw` 的拷贝；加 RAW photometric 后 `raw`(加噪) 与 `image`(preview) 会分叉。必须保证 **flip 与 photometric 都作用在真正进 model 的那个 tensor 上**（即送 `RamCore3` 的 `raw`），`rgb_preview`/`image` 仅供日志可视化，绝不能出现 input 加了噪/翻了、label 或喂入 model 的 tensor 没同步的几何/强度错位。
- 将现有 polynomial LR decay 改成 `--lr-schedule poly` 的显式分支；新增 `constant` 与 `cosine` 分支，`warmup_steps>0` 只在配置显式要求时启用。
- 新增可选 **train-proxy 评估**（B0.1）：`--eval-lod-train-proxy --lod-train-proxy-count 112`，复用 `lod_val` 评估路径但数据源是固定 train 子集、center crop、no aug；输出 `eval_lod_train/*` 指标与 `gap`。
- best checkpoint 仍只看 `lod_val d1`（maximize），train-proxy 只用于诊断，不参与 best 选择。

### B4. Smoke（路径含 `codex_smoke`）

1. `lod_aug.py` 单测式 smoke：对一张固定样本，分别跑 `off/baseline_e10/geom/medium`，断言：
   - `off` 与现有 build_sample 输出 bit-exact。
   - `baseline_e10`：固定 rng 下单样本 flip 操作本身正确（翻转后 input/depth/mask 一致、`hflip_applied=True`），且**训练循环不再产生第二次 hflip**（断言 LOD 路径 runtime flip 已移除）。注意旧实现是 batch 级、新实现是 sample 级，整 batch 不要求逐像素一致，只验证单样本 flip 正确性与"无双重翻转"。
   - `geom`：RGB/RAW 对同一 `sample_id`、同一 epoch、同一 seed 派生出完全一致的 hflip/scale/crop 参数；不同 dataloader worker 数或 shuffle 顺序不改变单样本几何参数。
   - 几何增广下 input/depth/mask shape 一致、flip 时 `hflip_applied=True`、valid 比例合理。
   - RAW photometric 后仍 finite 且 `[0,1]`。
2. train smoke：RGB + RAW 各跑 `--aug-preset geom --debug-max-train-steps 1 --debug-max-val-samples 2` 与 `--aug-preset medium --debug-max-train-steps 1 --debug-max-val-samples 2`，确认 `[AUG]` 日志、resolved config 展开、1 步训练通过。
3. train-proxy smoke：`--eval-lod-train-proxy --lod-train-proxy-count 8`，确认 `eval_lod_train` 指标与 `gap` 正常打印。
4. LR schedule smoke：`--lr-schedule poly --warmup-steps 0` 跑 1 步确认等价当前 baseline 分支；`--lr-schedule cosine --warmup-steps 1` 跑 1 步确认 `[LR]` 日志与 lr 更新正常。

### B5. Formal 实验矩阵

> 目标是**同时回答**两个问题：过拟合是否真实、augmentation 是否有效、欠训练加 epoch 是否够。所有 run 共享同一 split / val 协议 / crop size，**仅** epoch 与 aug 显式不同，记入 fairness summary。

建议 epoch 预算：当前 e10≈2640 steps≈ROD 1.75 epoch，明显偏少。长训对照取 **e40**（≈10560 steps≈ROD ~7 epoch）；如显存/时间允许 RAW 可再加 e60 点。主矩阵沿用当前 baseline 的显式 `lr_schedule=poly, warmup_steps=0`，避免把 LR schedule 与 epoch/aug 混成同一个变量；cosine + warmup 只作为 paired schedule axis。

**公平对照分层**（回应 RGB-vs-RAW 单变量要求）：
- **主公平对照 = 纯几何 `geom` 档**：`Rg`(RGB) vs `Wg`(RAW)，两域同算子、同强度、同时作用于 label；结论表述为 **RGB pipeline vs RAW pipeline**（输入域 + front-end / trainable path），而不是只称为纯 input-domain 结论。RAW 侧包含 raw front-end 与独立 optimizer group，这是 pipeline 变量的一部分，必须在 summary 列明。
- **次要 = `medium` 档**：各域 best-effort，含分域 photometric，跨域**不再单变量**，只用于各域上限与排序稳定性，不单独下 pipeline 结论。
- **pipeline 结论鲁棒判据**：若 RGB-vs-RAW 的 d1 排序在 `baseline_e10 / geom / medium` 三档一致，且每档差值超过最小有效差值阈值，则 pipeline 结论对增广选择鲁棒，photometric confound 不影响定性结论。考虑 112 样本 val 的噪声，默认 `min_effective_d1_delta=0.003`；`abs(d1_rgb - d1_raw) <= 0.003` 一律记为 tie / inconclusive，不给 winner。

| Run | 链路 | epochs | aug preset | lr schedule | 目的 |
| --- | --- | --- | --- | --- | --- |
| R0 | RGB | 40 | baseline_e10 | poly / warmup0 | 隔离"只是欠训练"——若 val 随 epoch 升则非过拟合；若转平/回落则过拟合 |
| Rg | RGB | 40 | geom | poly / warmup0 | **pipeline 主对照(RGB 侧)**；并看共享 scale-jitter 单独的正则收益(R0→Rg) |
| R1 | RGB | 40 | medium | poly / warmup0 | RGB best-effort；分域 photometric 额外收益(Rg→R1，跨域不可比) |
| W0 | RAW | 40 | baseline_e10 | poly / warmup0 | RAW 加 epoch 的纯收益（已知 e10 欠训练） |
| Wg | RAW | 40 | geom | poly / warmup0 | **pipeline 主对照(RAW 侧)**；与 Rg 构成单变量 pipeline 对照 |
| W1 | RAW | 40 | medium | poly / warmup0 | RAW best-effort + aug（follow-up note 指定优先） |
| (可选) W2 | RAW | 60 | medium | poly / warmup0 | 进一步确认 RAW 饱和点 |

Schedule axis（单独解释，不混入主矩阵主效应）：
- 优先对 headline 主对照 `Rg/Wg` 增加 paired twin：仅把 `--lr-schedule poly --warmup-steps 0` 改成 `--lr-schedule cosine --warmup-steps <N>`，其余完全不变，用于确认 LR schedule 不改变 pipeline 主结论。
- 对 R0/R1/W0/W1 中需要复核的 run 可追加 paired twin，但不得替代 `Rg/Wg` 的 schedule 稳定性检查。
- schedule twin 的结论只回答 LR schedule 是否改善长训稳定性；不能和 augmentation 主效应混写。

判读：
- **pipeline 对照（主）**：`Rg` vs `Wg` 的 `lod_val d1` 给 RGB pipeline vs RAW pipeline 的 headline 结论；若 `abs(d1_Rg - d1_Wg) <= 0.003`，结论记为 tie / inconclusive，不给 winner。再核对 `baseline_e10`(R0/W0) 与 `medium`(R1/W1) 两档排序是否一致，确认结论对增广鲁棒。
- R0 val 在 e10→e40 持续下降而 Rg/R1 不降 → 过拟合确认且几何 aug 有效；R0→Rg 收益归"共享 scale-jitter"，Rg→R1 增量归"RGB 分域 photometric"（不跨域比较）。
- W0/Wg/W1 在 e40 前 val 仍升 → RAW 主要矛盾是欠训练，aug 为锦上添花。
- 全程对比 `lod_val d1` 与 `lod_train_proxy d1` 的 gap 曲线作为 fixed-split 过拟合主证据；跨场景泛化结论需引用 B0.3 的 block/neighbor-excluded holdout 后续结果。

#### Formal 脚本

```text
finetune_stf/scripts/formal/0608_run_lod_true_rgb_aug_e40_queue.sh        # R0/Rg/R1
finetune_stf/scripts/formal/0608_run_lod_true_raw_rgb16_aug_e40_queue.sh  # W0/Wg/W1(/W2)
```

- 命名 `MMDD_HHMM` 开头取启动时刻；长任务进 tmux，不复用 session；单 GPU 顺序排队，避免 RGB/RAW 争用（沿用 0608_0109 队列模式）。
- 主矩阵在现有 baseline 参数基础上仅追加：`--epochs 40`、`--aug-preset {baseline_e10|geom|medium}`、`--lr-schedule poly --warmup-steps 0`、`--eval-lod-train-proxy --lod-train-proxy-count 112`。其余（split / manifest / crop / loss / best-metric=lod_d1）保持不变以确保对照公平。
- schedule twin 只把 LR 参数改成 `--lr-schedule cosine --warmup-steps <N>`，不得同时改 aug/epoch/bs/loss。

### B6. 记录与公平性

- 结果写入新的 `plans/result/lod_true_aug_overfit_summary.md`，与现有 `lod_true_rgb_raw_baseline_fairness_summary.md` 交叉引用。
- summary 必须写死：主矩阵相对 baseline 的**唯一变量**是 `{epochs, aug-preset}`，并显式记录 `lr_schedule=poly, warmup_steps=0`；schedule axis 单独成表，paired twin 的唯一变量是 `{lr-schedule, warmup-steps}`。其余口径不变；并附 B0.2 的 best/last gap 基线、各 run 的 gap 曲线、val d1 曲线。
- **RGB-vs-RAW pipeline 对照的公平边界必须写死**：headline 结论只引用 `geom` 档（`Rg` vs `Wg`，跨域逐项相同增广，单变量），并按 `min_effective_d1_delta=0.003` 判定 winner/tie；`medium` 档跨域存在**分域 photometric confound**，只能作为各域 best-effort 与排序稳定性证据，不得单独支撑 pipeline 结论。summary 附 `{baseline_e10, geom, medium}` 三档 RGB/RAW d1 排序一致性表，并明确列出每个 run 的 trainable groups / optimizer groups（RGB decoder-only；RAW raw_front_end + decoder）。
- 所有 aug 实际展开值从 resolved config dump 摘录进 summary，不允许只写 preset 名。

### B7. Implementation Order

1. 先做 B0.2 + B0.3：用现有 best/last checkpoint 跑 train-proxy/val gap 诊断，并完成当前 split 邻近性 audit；结论写入 Execution Log。
2. 实现 `lod_aug.py` + `LODAugConfig`，保证 `off` bit-exact passthrough。
3. 接入 `lod_true.py` 两条链路的几何 + 分域 photometric。
4. `resolved.py` 增 flag / preset 展开 / 集中校验 / log；`train.py` 注入 train-only aug、迁移 runtime hflip、train-proxy 评估、显式 poly/constant/cosine schedule。
5. 跑 B4 全部 smoke。
6. 加 formal 脚本，跑 B5 矩阵：先跑 `geom` 主对照（`Rg`/`Wg`）确立 pipeline 结论，再补 `baseline_e10`(R0/W0) 与 `medium`(R1/W1) best-effort；RAW 长训按 follow-up note 优先。
7. 汇总 B6 summary 并判读过拟合/欠训练结论。

### B8. Risks / Watch

- 几何增广 `off` 必须 bit-exact 等于现有 crop，否则破坏与 e10 baseline 的对照——smoke 1 是硬闸。
- RAW photometric 只能在线性归一化空间做增益/噪声，**严禁** display-域 color/gamma，且通道序不能动，否则破坏 `RamCore3 -> identity -> DAv2` 权重迁移语义。
- val 永不增广；train-proxy 评估也用 center crop / no aug，保证 gap 口径干净。
- 112 样本 val 本身噪声较大，单点 d1 波动可能 ~±0.003；判读以多 epoch 趋势和 train-proxy gap 为准，不抠单 epoch 小数。
- `Rg/Wg` headline pipeline 结论必须使用最小有效差值阈值：`abs(d1_Rg - d1_Wg) <= 0.003` 时只能写 tie / inconclusive；不能因为 0.000x 级差异给出 winner。
- 过强 aug（尤其低光下的强亮度/噪声）可能把本就稀薄的信号淹没，导致欠拟合；preset 从 light/medium 起步，heavy 仅在确认 medium 仍过拟合后再用。
- RGB-vs-RAW pipeline 结论只在 `geom`（纯几何、跨域逐项相同）档成立；用 `medium` 跨域下结论会把分域 photometric 的自由超参混进 pipeline 变量，属无效对照（见 B1.8 / B5 / B6）。
