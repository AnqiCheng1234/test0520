**迁移 0520 STF 项目到 186**

**审核结论**
- 认可审核里的阻塞项 1-3，计划必须先补全并显式处理 `STF_ROOT` 和 checkpoint 兜底。
- 认可改进项 4-10，并已并入执行步骤。
- 唯一边界：不做全局 `*.npy` 排除。理由是项目后续可能放真实 `.npy` 清单或小数据；执行前先枚举项目内 `.npy`，只排除确认是生成缓存的具体目录或文件。

**目标**
- 远端：`a5000@10.97.8.186`。
- 远端项目目录：`/home/a5000/6666_raw/dav2_raw_0520`。
- 默认环境：`/home/a5000/anaconda3/bin/conda run -n dav3`。
- 小 STF 根目录：`/home/caq/6666_raw/seeingthroughfog` -> `/home/a5000/6666_raw/seeingthroughfog`，本机约 542M。
- 大 STF 镜像：`/mnt/drive/3333_raw/seeing_through_fog` -> 186 同路径，约 860G。
- 远端实验输出隔离：
  - 轻量输出：`/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp_186`
  - 大 checkpoint/TensorBoard：`/mnt/drive/3333_raw/0000_exp_ckpt_186`
  - `exp_186` 与本地 `finetune_stf/exp/` 完全隔离，不混用 resume、best 软链接或历史实验目录。

**必须先修**
- 计划文件已补全，本文件不再停在 `HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ck`。
- `finetune_stf/scripts/0521_run_stf_lora_full_da3_queue.sh` 必须在同步前支持 `STF_ROOT` 覆盖：
  - 在路径变量区加入：
    `STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"`
  - 在 `common_args` 里加入：
    `--stf-root "${STF_ROOT}"`
  - 保留现有 `RAW_NPZ_ROOT` 覆盖。
  - 远端正式启动时必须设置：
    `STF_ROOT=/home/a5000/6666_raw/seeingthroughfog`
- checkpoint 兜底：
  - 本地实际文件是 `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth`，约 95M。
  - 如果 186 缺 `/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth`，先执行：
    ```bash
    ssh a5000@10.97.8.186 'mkdir -p /mnt/drive/3333_raw/checkpoints'
    scp /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
      a5000@10.97.8.186:/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth
    ```

**同步策略**
- 长时间同步必须在本机 tmux 中跑，不复用已有 session。
- 建议 session：`0521_sync_stf_186`。
- 建议日志：`/home/caq/6666_raw/dav2_raw_0520/logs/0521_sync_stf_186.log`。
- 启动后报告：
  - `tmux attach -t 0521_sync_stf_186`
  - `tail -f /home/caq/6666_raw/dav2_raw_0520/logs/0521_sync_stf_186.log`
- 860G 走 rsync over SSH，千兆口理论约 2 小时，实际按 2.5 小时以上预估；网络、磁盘和 SSH 加密都会拉长。
- 使用 `--partial-dir=.rsync-partial` 代替普通 `--partial`，便于中断恢复。
- 对大数据同步加 `nice -n 10 ionice -c 2 -n 7`，必要时加 `--bwlimit=<KB/s>` 避免影响本机其它训练。

**预检**
- 本机：
  ```bash
  test -f /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
  du -sh /home/caq/6666_raw/seeingthroughfog /mnt/drive/3333_raw/seeing_through_fog /home/caq/6666_raw/dav2_raw_0520
  find /home/caq/6666_raw/dav2_raw_0520 -path '*/.git' -prune -o -type f -name '*.npy' -print
  ```
- 远端：
  ```bash
  ssh a5000@10.97.8.186 '
    hostname
    whoami
    pwd
    /home/a5000/anaconda3/bin/conda env list
    df -h /mnt/drive /home/a5000
    nvidia-smi -L
    test -f /mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth || true
  '
  ```
- 空间要求：
  - `/mnt/drive` 至少留出 900G 给完整 STF 镜像和 checkpoint 输出。
  - `/home/a5000` 需要覆盖项目源码和 542M 小 STF 根目录，仍需显式 `df -h /home/a5000`。

**同步代码**
- 保留 `.git/`，用于保留历史、未提交 diff 和远端诊断上下文；当前体量可接受。
- 排除运行产物和 IDE/环境目录：
  - `finetune_stf/exp/`
  - `finetune_stf/exp_186/`，源端现在可能不存在，保留排除项是为了防止未来同名目录被误传。
  - `finetune_stf/logs/`
  - `anqi_eval/results/`
  - `logs/`
  - `wandb/`
  - `tensorboard/`
  - `.venv/`
  - `.idea/`
  - `.vscode/`
  - `codex_debug/`
  - `codex_smoke/`
  - `__pycache__/`
  - `*.pyc`
  - 已确认是生成缓存的具体 `.npy` 路径，不使用全局 `*.npy`
- 命令：
  ```bash
  ssh a5000@10.97.8.186 'mkdir -p /home/a5000/6666_raw/dav2_raw_0520'
  rsync -aH --info=progress2 --partial-dir=.rsync-partial \
    --exclude 'finetune_stf/exp/' \
    --exclude 'finetune_stf/exp_186/' \
    --exclude 'finetune_stf/logs/' \
    --exclude 'anqi_eval/results/' \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'tensorboard/' \
    --exclude '.venv/' \
    --exclude '.idea/' \
    --exclude '.vscode/' \
    --exclude 'codex_debug/' \
    --exclude 'codex_smoke/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    /home/caq/6666_raw/dav2_raw_0520/ \
    a5000@10.97.8.186:/home/a5000/6666_raw/dav2_raw_0520/
  ```

**同步 STF 数据**
- 小 STF 根目录：
  ```bash
  ssh a5000@10.97.8.186 'mkdir -p /home/a5000/6666_raw/seeingthroughfog'
  rsync -aH --info=progress2 --partial-dir=.rsync-partial \
    /home/caq/6666_raw/seeingthroughfog/ \
    a5000@10.97.8.186:/home/a5000/6666_raw/seeingthroughfog/
  ```
- 大 STF 镜像：
  ```bash
  ssh a5000@10.97.8.186 'mkdir -p /mnt/drive/3333_raw/seeing_through_fog'
  nice -n 10 ionice -c 2 -n 7 rsync -aH --info=progress2 \
    --partial-dir=.rsync-partial --append-verify \
    /mnt/drive/3333_raw/seeing_through_fog/ \
    a5000@10.97.8.186:/mnt/drive/3333_raw/seeing_through_fog/
  ```

**同步后校验**
- 代码 dry-run 比对，期望 0 个待传输文件：
  ```bash
  rsync -aHn --delete --info=stats2 --partial-dir=.rsync-partial \
    --exclude 'finetune_stf/exp/' \
    --exclude 'finetune_stf/exp_186/' \
    --exclude 'finetune_stf/logs/' \
    --exclude 'anqi_eval/results/' \
    --exclude 'logs/' \
    --exclude 'wandb/' \
    --exclude 'tensorboard/' \
    --exclude '.venv/' \
    --exclude '.idea/' \
    --exclude '.vscode/' \
    --exclude 'codex_debug/' \
    --exclude 'codex_smoke/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    /home/caq/6666_raw/dav2_raw_0520/ \
    a5000@10.97.8.186:/home/a5000/6666_raw/dav2_raw_0520/
  ```
- 数据 dry-run 比对，期望 0 个待传输文件：
  ```bash
  rsync -aHn --delete --info=stats2 --partial-dir=.rsync-partial \
    /home/caq/6666_raw/seeingthroughfog/ \
    a5000@10.97.8.186:/home/a5000/6666_raw/seeingthroughfog/
  rsync -aHn --delete --info=stats2 --partial-dir=.rsync-partial \
    /mnt/drive/3333_raw/seeing_through_fog/ \
    a5000@10.97.8.186:/mnt/drive/3333_raw/seeing_through_fog/
  ```
- `--delete` 只用于 dry-run 检查远端额外文件；不要在没有明确确认前执行真实删除。

**远端启动前验证**
- 连接 186 后先确认 host、user、pwd、conda 环境：
  ```bash
  ssh a5000@10.97.8.186
  hostname
  whoami
  pwd
  /home/a5000/anaconda3/bin/conda env list
  ```
- 在远端确认关键路径：
  ```bash
  cd /home/a5000/6666_raw/dav2_raw_0520
  test -f /mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth
  test -d /home/a5000/6666_raw/seeingthroughfog/manifests
  test -d /mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz
  test -f /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv
  test -f /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv
  ```

**正式队列启动**
- 在 186 项目根目录执行。脚本会先跑 `codex_smoke`，成功后只清理明确临时 smoke 产物；正式实验名由脚本用远端本地时间生成，格式为 `MMDD_HHMM_*`。
- 命令：
  ```bash
  cd /home/a5000/6666_raw/dav2_raw_0520
  ROOT=/home/a5000/6666_raw/dav2_raw_0520 \
  STF_ROOT=/home/a5000/6666_raw/seeingthroughfog \
  EXP_ROOT=/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/exp_186 \
  LOG_ROOT=/home/a5000/6666_raw/dav2_raw_0520/finetune_stf/logs \
  HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ckpt_186 \
  PRETRAINED=/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth \
  RAW_NPZ_ROOT=/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz \
  DAV2_MANIFEST=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv \
  DA3_MANIFEST=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv \
  CONDA_BIN=/home/a5000/anaconda3/bin/conda \
  CONDA_ENV=dav3 \
  GPU=0 \
  bash finetune_stf/scripts/0521_run_stf_lora_full_da3_queue.sh
  ```
- 启动后脚本会打印 tmux session、queue log、attach 和 tail 命令，按脚本输出记录。

**非阻塞维护项**
- `finetune_stf/dataset/raw_utils.py` 里的 `STF_DECOMPANDING_NOTES_PATH` 指向旧项目 `/home/caq/6666_raw/dav2_raw_0515_vits/...`，当前 grep 结果显示未被引用，不会阻塞 186 训练。
- 建议后续清理：删除该未用常量，或改成项目内相对路径，避免误导维护者。

**训练产物回流**
- 正式实验完成后，只回流 metrics、配置摘要和关键 checkpoint，不回流中间 epoch。
- 建议本机目标：
  `/mnt/drive/3333_raw/0000_exp_ckpt/from_186/<run>/`
- 示例：
  ```bash
  mkdir -p /mnt/drive/3333_raw/0000_exp_ckpt/from_186
  rsync -aH --info=progress2 --partial-dir=.rsync-partial \
    --include '*/' \
    --include 'metrics*' \
    --include '*summary*' \
    --include '*config*' \
    --include '*best*.pth' \
    --exclude '*' \
    a5000@10.97.8.186:/mnt/drive/3333_raw/0000_exp_ckpt_186/ \
    /mnt/drive/3333_raw/0000_exp_ckpt/from_186/
  ```
