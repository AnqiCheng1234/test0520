# STF DA3 Pseudo Depth 训练适配计划

## Summary
目标是让 STF 训练可以用已生成的 DA3MONO-LARGE pseudo depth：
`/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`

核心策略采用已确认的“在线 Dataset 对齐”方案：训练读取 DA3 affine-invariant depth 后，用同一样本的 sparse LiDAR metric depth 拟合 `metric_depth = scale * da3_depth + shift`，再把样本标记为 `target_space="metric_depth"`，复用现有 loss 里的 `1 / metric_depth` 路径。旧 `dav2_pseudo` 行为保持不变。

## Key Changes
- 新增 STF 训练 target mode：
  - `--stf-train-target-mode da3_pseudo_sparse_metric`
  - 仍使用现有 `--stf-pseudo-manifest` 传入 DA3 manifest。
  - `dav2_pseudo` 继续表示“文件本身已经是 relative inverse depth”。

- 修改 STF RGB/RAW dataset 读取逻辑：
  - pseudo manifest loader 需要把 `target_kind` 设置为当前 mode，而不是固定 `dav2_pseudo`。
  - `dav2_pseudo` 分支保持原样：加载 `.npy`，`target_space="inverse_relative"`。
  - `da3_pseudo_sparse_metric` 分支：
    - 加载 DA3 `.npy` 作为 affine depth。
    - 加载 `sparse_depth_path` 中的 STF sparse metric depth。
    - 在原始分辨率上用 sparse 有效点拟合 `metric = scale * da3 + shift`。
    - dense metric target 只保留 `min_depth <= metric <= max_depth` 的有效像素。
    - 输出 `sample["target_space"] = "metric_depth"`。

- Robust sparse alignment 规则固定为：
  - sparse 有效点：`finite(da3) & da3 > 0 & finite(sparse) & min_depth <= sparse <= max_depth`。
  - 最少有效点：`128`。
  - 先做全点 least squares，再用 residual MAD 做一次 inlier 筛选。
  - inlier threshold：`max(3 * 1.4826 * MAD, 2.0 meters)`。
  - inlier 点数仍需 `>=128`，再 refit。
  - 接受条件：`scale`、`shift` finite，且 `scale > 1e-6`。
  - 若拟合失败，fallback 到 sparse metric target：只用 sparse GT mask，仍设 `target_space="metric_depth"`，不让训练崩掉。

- 增加误用保护：
  - 如果 `--stf-train-target-mode dav2_pseudo` 传入的 manifest sibling `run_config.json` 或 `run_summary.json` 标明 `depth_value_units.value="affine_invariant_depth_from_da3mono"`，直接报错，提示改用 `da3_pseudo_sparse_metric`。
  - 如果 `da3_pseudo_sparse_metric` 使用的 manifest 有 metadata 且明显不是 DA3 affine depth，也报错；metadata 缺失时允许继续。

- 训练主逻辑基本不改：
  - 不新增 loss 类型。
  - 不改 `build_training_target()`，因为 `metric_depth` 已经会走 `1 / depth`。
  - 只扩展 CLI choices、manifest 校验、首个 batch 日志，让日志能看到 STF DA3 样本的 `target_space=metric_depth` 和 target 统计。

## Implementation Steps
- 在 `finetune_stf/dataset/stf.py` 中新增共享 helper：
  - `STF_TRAIN_TARGET_MODES = ("gt_sparse", "dav2_pseudo", "da3_pseudo_sparse_metric")`
  - `STF_PSEUDO_TRAIN_TARGET_MODES = ("dav2_pseudo", "da3_pseudo_sparse_metric")`
  - helper 负责加载 sparse depth、DA3-to-metric robust alignment、metadata 检查。
- 在 `finetune_stf/dataset/stf_raw.py` 复用这些 helper，避免 RGB/RAW 两套逻辑漂移。
- 修改两个 dataset 的 pseudo manifest loader：
  - 增加 `target_kind` 参数。
  - train split 下 mode 为 `dav2_pseudo` 或 `da3_pseudo_sparse_metric` 时，都走 pseudo manifest。
- 修改两个 dataset 的 `__getitem__` / `build_sample`：
  - `dav2_pseudo` 分支保持当前反深度逻辑。
  - 新增 `da3_pseudo_sparse_metric` 分支，返回 metric dense target 或 sparse fallback target。
  - 对 RAW 训练，最终 target 必须 resize 到 `(512, 960)`，dense 用 linear，mask 用 nearest；sparse fallback 用 nearest，保持现有 sparse GT 行为。
- 修改 `finetune_stf/train.py`：
  - CLI choices 自动包含新 mode。
  - manifest 存在性校验从 `dav2_pseudo` 扩展到所有 pseudo mode。
  - 首个 batch 日志中加入 STF target stats，确认新模式下 target 是 metric depth 而不是 inverse relative。

## Smoke Test
所有命令默认使用 `dav3` conda 环境。

1. 静态检查：
```bash
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3
python -m compileall finetune_stf/dataset/stf.py finetune_stf/dataset/stf_raw.py finetune_stf/train.py finetune_stf/util/loss.py
```

2. Dataset 只读 smoke：
- 构造 `STF_RAW("train", stf_train_target_mode="da3_pseudo_sparse_metric", stf_pseudo_manifest=<DA3 manifest>)`。
- 抽查 20 个样本：
  - `depth.shape == (512, 960)`
  - `target_space == "metric_depth"`
  - `valid_mask.sum() >= 128`
  - `depth[valid_mask]` finite 且在 `[1, 80]`
  - 至少多数样本走 dense aligned target，不是 fallback sparse。

3. 误用保护 smoke：
- 用 DA3 manifest 跑：
```bash
--stf-train-target-mode dav2_pseudo
```
应 fail-fast，报错提示 DA3 affine depth 不能当 inverse relative 使用。

4. 训练 smoke，输出路径必须带 `codex_smoke`：
```bash
TS=$(date +%m%d_%H%M)
SAVE=/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp/codex_smoke_da3_sparse_metric_${TS}
DA3_MANIFEST=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv

PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29591 \
  finetune_stf/train.py \
  --encoder vits \
  --stage stf_only \
  --input-type raw_ram_rgb \
  --raw-npz-root /mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz \
  --stf-raw-decode-mode legacy_online_decomp16 \
  --norm-mode passthrough \
  --channel-mode rgb_avg_g \
  --raw-ram-rgb-tail identity \
  --rgb-interface-mode residual_tanh \
  --stf-train-target-mode da3_pseudo_sparse_metric \
  --stf-pseudo-manifest "$DA3_MANIFEST" \
  --dav2-train-mode none \
  --input-height 512 \
  --input-width 960 \
  --bs 2 \
  --accum-steps 1 \
  --epochs 1 \
  --lr 1e-5 \
  --loss-type ssi \
  --loss-target-normalization \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --num-workers 0 \
  --log-interval 1 \
  --debug-max-train-steps 2 \
  --debug-max-val-samples 8 \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --heavy-save-root /mnt/drive/3333_raw/0000_exp_ckpt \
  --save-path "$SAVE"
```
成功后只删除 `codex_smoke` local/heavy 输出目录；失败则保留并报告路径。

## Formal Run Plan
- 正式实验仍沿用当前 STF pseudo-label baseline 参数，只改：
  - `--stf-train-target-mode da3_pseudo_sparse_metric`
  - `--stf-pseudo-manifest /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv`
- 实验名必须用远端/本机当前时间 `MMDD_HHMM` 开头，例如：
  - `${TS}_stf_train_test_da3_sparse_metric_raw_ram_rgb_bnclean_identity_e5`
- 训练可能较长，正式 run 用 tmux：
  - session：`stf_da3_sparse_metric_${TS}`
  - log：`/home/caq/6666_raw/dav2_raw_0520/finetune_stf/logs/${RUN_NAME}.tmux.log`
  - 启动后报告：
    - `tmux attach -t stf_da3_sparse_metric_${TS}`
    - `tail -f <log_path>`

## Acceptance Criteria
- `dav2_pseudo` 旧训练路径行为不变，旧 manifest 仍可跑。
- DA3 manifest 不能被 `dav2_pseudo` 静默误用。
- 新模式首个 batch 日志显示 `target_space=metric_depth`。
- Smoke train 能完成 2 个 train step 和少量 STF val。
- DA3 mode 下 loss 不需要新增分支，仍复用 `metric_depth -> inverse depth -> SSI/SSI+grad`。
- 正式训练 validation 仍走 STF val sparse GT，不使用 DA3 val pseudo target。

## Assumptions
- 使用已完成的 DA3 输出目录 `pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051`。
- DA3 `.npy` 值为 affine-invariant depth，越远越大。
- STF sparse LiDAR 在 train/test 样本上足够用于每图 affine metric alignment；少数失败样本 fallback 到 sparse metric 监督。
- 当前选择是在线 Dataset 对齐，不新增离线转换目录。
