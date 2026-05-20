# STF-Only RAW-RAM-RGB 实验计划

## Summary
- 目标项目：`/home/caq/6666_raw/dav2_raw_0520`
- 训练：STF `train + test`，输入用 STF RAW，target 用 DAv2 ViT-L LUT RGB pseudo depth。
- 验证：STF `val`，target 用 sparse LiDAR GT，按 RobotCar sparse eval 风格评估。
- 输入尺寸：本轮 STF train / val / pseudo target 全部固定原生 `512x960`；正式命令显式传 `--input-height 512 --input-width 960`，并在 `stf_only + STF_RAW` 路径 fail-fast 防止误用其他尺寸。
- 模型：`raw_ram_rgb`，4ch RAW 先合并双 G 成 3ch，进 `RamCore3`，BN 后直接给 DAv2 backbone；不 clamp、不 ImageNet norm、不 tanh；只训练 RAM，DAv2 冻结。
- 可视化：每 epoch dump `train_viz` 和 `viz_fixed/stf`，panel 延续当前布局，但 STF sparse GT 点做可视化 dilation 加粗。
- 通道修复影响：旧 STF rectified NPZ 按 `[B,G,G,R]` 解释，新 decode 会显式转成 `[R,Gr,Gb,B]`；历史 `raw_ram_rgb` STF raw 实验的 base RGB 很可能红蓝反，本轮等同于重置基线，结果不要和历史 dashboard 直接横向比较。

## Key Changes
- 增加 STF RAW decode 接口：
  - 新 CLI：`--stf-raw-decode-mode {legacy_companded,legacy_online_decomp16,canonical_decomp16}`，默认 `legacy_companded` 保持兼容。
  - 新增 `--norm-mode passthrough`，并让 `normalize_raw()` / `normalize_raw_4ch()` 支持该分支。
  - `legacy_companded` 保持历史行为：读取旧 NPZ 后再按 `--norm-mode companded` 做 `/3967`。
  - knee LUT 来源固定记录为 `/home/caq/6666_raw/dav2_raw_0515_vits/plans/0520_final_new/stf/stf_raw_companding_official_notes.md`，其中整理了 STF 官方 `tools/Raw2LUTImages/conversion_lib/process.py` 的 `decomp_kneepoints = [[1023,1023], [2559,4095], [3455,32767], [3967,65535]]`。
  - 本实验使用 `legacy_online_decomp16`：旧 NPZ `[B,G,G,R] -> [R,Gr,Gb,B]`，再用上述官方 knee points 构造 LUT 并 decompand 到 16-bit，最后 `/65535`，输出已经是 `[0,1]`。
  - 当 `--stf-raw-decode-mode != legacy_companded` 时，强制有效 `norm_mode=passthrough`；若用户传 `companded` 则 fail-fast，避免 decode 后再次 `/3967` 的二次缩放。
  - 对 `raw_npz_root` 做 fail-fast：旧 root 只允许 `legacy_companded/legacy_online_decomp16`；未来 canonical root 只允许 `canonical_decomp16`。

- 增加 STF pseudo-depth 训练 target：
  - 新 CLI：`--stf-train-target-mode {gt_sparse,dav2_pseudo}`，默认 `gt_sparse`。
  - 新 CLI：`--stf-pseudo-manifest`，本实验固定为 `/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`。
  - `dav2_pseudo` 训练端完全用 pseudo manifest 取代 `stf_raw_depth_v1_{train,test}.csv`：只取 `split in {train,test}`，raw NPZ 用 `raw_npz_root/{sample_name}.npz` 拼出并 fail-fast，pseudo target 用 `pseudo_depth_npy`。
  - 已核实 pseudo manifest 与 v1 manifest 的样本名集合一致：train `3526`、test `1882`、val `808`，三组 symmetric diff 都是 `0`；训练期望样本数 `3526 + 1882 = 5408`。
  - 训练 sample `target_space="inverse_relative"`，与 LoD pseudo-depth loss 路径一致。
  - 验证仍走 v1 `STF_RAW("val")` sparse GT，而不是 pseudo manifest 的 `split==val`；样本数 `808`，`target_space="metric_depth"`。

- 恢复 BN-clean raw_ram_rgb 路径：
  - 新 CLI：`--raw-ram-rgb-tail {identity,tanh2p5}`，默认 `tanh2p5` 兼容当前 0520 代码。
  - 本实验使用 `identity`：`RamCore3 -> BN -> DAv2`，不做 tail tanh、不 clamp、不 ImageNet norm。
  - `--dav2-train-mode none`，只训练 RAM 相关参数。

- 更新 eval/viz：
  - best checkpoint 保留当前 CLI：`--best-metric stf`。这里的 `stf` 明确表示 STF val sparse GT 的 `summary["abs_rel"]`，越低越好。
  - `collect_fixed_samples()` 增加 `("stf", "val_loader")`，让 `viz_fixed/epoch_xx/stf` 每 epoch 输出 STF val panel。
  - `collect_fixed_train_source_samples()` 增加 `stf -> stf_train` 映射，保证 `stf_only` 时确实输出 `train_viz/epoch_xx/stf`。
  - 新 CLI：`--stf-fast-eval-backend {sparse,proxy}`，默认 `sparse`。`STF_RAW("val")` 自动写入 `depth_mode="fast"`、`fast_eval_backend=args.stf_fast_eval_backend`，复用 sparse mask 上采样/对齐逻辑。
  - `STF_RAW` sample 携带 `lut_preview`/`rgb_preview`，保证 train_viz 和 fixed_viz 有 RGB 参考图。
  - depth panel 对 STF sparse GT 做显示 dilation，默认 kernel `7`；只影响可视化，不影响 metric/loss。
  - STF preview 的当前 DAv2-S 预测使用 `512x960` 输入生成；模型内部 pad 到 patch 对齐尺寸后 crop 回 `512x960`，`train_viz` 还会用 `_ensure_pred_hw(..., depth.shape[-2:])` 再兜底到 target 尺寸。

## Test Plan
- 静态/loader smoke：
  - 用 `conda run -n dav3` 实例化 STF train/val dataset。
  - 检查 train pseudo 样本数 `5408`、val sparse GT 样本数 `808`。
  - 检查 pseudo train/test/val sample_name 与 v1 filename_stem 集合 symmetric diff 为 `0`。
  - 检查输入 tensor 为 `4x512x960`，数值 finite 且在 `[0,1]`。
  - 校验 decode 端点：`new[...,0] == lut[old[...,3]] / 65535`，`new[...,3] == lut[old[...,0]] / 65535`。
  - 校验 LUT knee points 来自 `/home/caq/6666_raw/dav2_raw_0515_vits/plans/0520_final_new/stf/stf_raw_companding_official_notes.md` 记录的官方 decompanding 点。
  - 校验 `legacy_online_decomp16/canonical_decomp16` 下有效 `norm_mode=passthrough`，没有二次 `/3967`。
  - 检查 STF val sample 带 `depth_mode="fast"`、`fast_eval_backend="sparse"`。

- 训练 smoke：
  - 输出路径必须包含 `codex_smoke`。
  - 跑 `--epochs 1 --debug-max-train-steps 1 --debug-max-val-samples 4`。
  - 成功后删除明确带 `codex_smoke` 的临时输出；失败保留并报告路径。

- 可视化 smoke：
  - 确认 smoke 输出包含 `train_viz/epoch_00/stf/*_panel.jpg` 和 `viz_fixed/epoch_00/stf/*_panel.png`。
  - 确认 `train_viz` / `viz_fixed/stf` 保存的 current DAv2-S pred shape 为 `512x960`。
  - 人工检查 RGB/RAW/RAM/target/current panel：RAW 不再红蓝反，GT sparse 点明显可见。

## Formal Launch
- 正式实验名使用启动时本机时间 `MMDD_HHMM` 前缀，例如：
  - `0520_HHMM_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_e5`
- 建议命令参数：
  - `--stage stf_only`
  - `--input-height 512 --input-width 960`
  - `--input-type raw_ram_rgb`
  - `--encoder vits`
  - `--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth`
  - `--raw-npz-root /mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz`
  - `--stf-raw-decode-mode legacy_online_decomp16`
  - `--norm-mode passthrough`
  - `--stf-train-target-mode dav2_pseudo`
  - `--stf-pseudo-manifest /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv`
  - `--stf-fast-eval-backend sparse`
  - `--eval-stf --best-metric stf --save-best-checkpoint`
  - `--dav2-train-mode none`
  - `--raw-ram-rgb-tail identity`
  - 其他训练参数沿用 `0519_2256_phase1_lodnight_raw_ram_rgb_bnclean_e5`：`epochs=5, bs=8, accum_steps=1, lr=1e-5, loss_type=ssi, loss_lambda_grad=2, loss_grad_scales=4, loss_target_normalization=true, amp_dtype=bf16`。

- 正式训练用 tmux：
  - session：`stf_pseudovitl_bnclean_$(date +%m%d_%H%M)`
  - log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/<session>.log`
  - 启动后记录：
    - `tmux attach -t <session>`
    - `tail -f <log>`

## Assumptions
- 你说的 “dav2-t pseudo label” 按实际路径解释为 `official_vitl` 这份 ViT-L teacher pseudo depth。
- 本轮不生成新的 canonical STF RAW NPZ；先使用旧 rectified NPZ 在线修正通道和 decompanding。
- STF decompanding LUT 的依据是 `/home/caq/6666_raw/dav2_raw_0515_vits/plans/0520_final_new/stf/stf_raw_companding_official_notes.md`，不再在本计划里把 knee points 当作无来源常量。
- sparse GT 加粗只用于 panel 显示，不改变训练 loss、eval metric 或保存的 GT 数值。
- 旧实验兼容性优先：新增 CLI 都给默认值，历史命令不改变行为。
- 由于本轮修正 STF RAW 红蓝通道和 decompanding/归一化语义，历史 STF raw baseline 只能作背景参考，不能当作同一输入定义下的直接对照。
