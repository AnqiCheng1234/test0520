# 0606 迁移 `dav2_raw_0603` 到 186 的 ROD 复现实验准备计划

## 目标

在 186 机器上准备当前本机项目 `/home/caq/6666_raw/dav2_raw_0603` 的可运行副本，用于后续复刻当前 ROD-night 训练/评估相关实验。

本计划只覆盖迁移、路径对齐、依赖检查和小 batch smoke test。不启动 formal 实验，不运行完整训练队列。

## 当前执行状态（2026-06-06 13:13 CST）

已完成：

- 已连接 186 并确认基础环境：
  - host：`A5000`
  - user：`a5000`
  - pwd：`/home/a5000`
  - conda 环境 `dav3` 存在
  - `/mnt/drive` 可用空间约 `4.9T`
- 已完整同步本机项目到 186：
  - 远端路径：`/home/a5000/6666_raw/dav2_raw_0603`
  - 传输量约 `120M`
  - 关键文件验证通过：`finetune_stf/train.py` 和 `finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh`
  - 当前本机/远端 git 状态均显示计划文件 `plans/0606_186_dav2_raw_0603_rod_migration_plan.md` 为 untracked
- 已在 186 上通过保护检查后将旧名目录改名：
  - `/mnt/drive/3333_raw/LOD` -> `/mnt/drive/3333_raw/ROD`
  - 改名后 `/mnt/drive/3333_raw/ROD` 存在，`/mnt/drive/3333_raw/LOD` 不存在
- 已完成 186 基础 ROD 数据计数校验：
  - train raw：`16089`
  - valid raw：`4000`
  - train rggb：`16089`
  - valid rggb：`4000`
- teacherbright pseudo 数据 rsync 已按长任务规则放入本地 tmux，并已完成：
  - session：`0606_rod_teacherbright_rsync_to_186`
  - log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_rod_teacherbright_rsync_to_186.log`
  - attach：`tmux attach -t 0606_rod_teacherbright_rsync_to_186`
  - monitor：`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_rod_teacherbright_rsync_to_186.log`
  - 2026-06-06 13:08 CST 检查：tmux session 已退出，日志尾部显示 `75,031,025,294 100%`，`xfr#14038`，`to-chk=0/14041`
- 已完成 186 teacherbright pseudo 数据校验：
  - manifest：`/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv` 存在
  - run config：`/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/run_config.json` 存在
  - train pseudo：`12036`
  - valid pseudo：`2000`
  - manifest 行数：`14037`
  - 目录体积：约 `70G`
- 已完成 student RGB smoke：
  - 命令使用 `dav3` 环境、`CUDA_VISIBLE_DEVICES=0`、`--debug-max-train-steps 1`、`--debug-max-val-samples 2`
  - 数据链路：`rod_raw_student_rgb` / `raw24_student_rgb`
  - 解析到 `rod_train=12036`、`rod_night_val=2000`
  - init eval 成功：`samples=2`
  - 训练成功：`used_steps=1`
  - epoch 0 eval 成功：`samples=2`
  - 临时输出曾写入 `/tmp/codex_smoke_186_rod_studentrgb` 和 `/tmp/codex_smoke_186_heavy_studentrgb`
- 已完成 raw RAM smoke：
  - 命令使用 `dav3` 环境、`CUDA_VISIBLE_DEVICES=0`、`PHASE1_BNCLEAN_REVIEWED=1`、`--debug-max-train-steps 1`、`--debug-max-val-samples 2`
  - 数据链路：`rod_raw` / `raw_ram`
  - 解析到 `rod_train=12036`、`rod_night_val=2000`
  - init eval 成功：`samples=2`
  - 训练成功：`used_steps=1`
  - epoch 0 eval 成功：`samples=2`
  - 临时输出曾写入 `/tmp/codex_smoke_186_rod_rawram` 和 `/tmp/codex_smoke_186_heavy_rawram`
- 已按 smoke 清理规则删除成功 smoke 临时目录：
  - `/tmp/codex_smoke_186_rod_studentrgb`
  - `/tmp/codex_smoke_186_heavy_studentrgb`
  - `/tmp/codex_smoke_186_rod_rawram`
  - `/tmp/codex_smoke_186_heavy_rawram`
  - 删除后再次 `ls` 确认这些路径不存在

待继续：

- 本迁移、路径对齐、依赖检查和小 batch smoke test 计划已完成。
- 本轮没有启动 formal 实验。
- 后续如启动 formal 实验，仍需按计划使用远端本地时间生成 `MMDD_HHMM` 前缀，并将长时间训练放入 tmux。

## 已确认信息

### 本机项目

- 项目路径：`/home/caq/6666_raw/dav2_raw_0603`
- 代码体积：约 `118M`
- git 状态：干净，无未跟踪文件
- 按用户要求：代码迁移不排除任何文件，包含 `.git`、`finetune_stf/exp`、`finetune_stf/logs`、`codex_tmp` 等

### 186 机器

- host：`A5000`
- user：`a5000`
- 默认登录目录：`/home/a5000`
- conda：`/home/a5000/anaconda3/bin/conda`
- 默认环境：`dav3`
- `dav3` 可用，CUDA 可用，GPU 为 2 张 NVIDIA RTX A5000
- 既有迁移项目目录：`/home/a5000/6666_raw`
- 计划目标代码目录：`/home/a5000/6666_raw/dav2_raw_0603`
- 目标代码目录当前不存在

### 当前 ROD formal 入口范围

当前最新 ROD-night 相关 formal 脚本为：

- `finetune_stf/scripts/formal/0604_run_rod_night_student_rgb_decoder_e10_queue.sh`
- `finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh`
- `finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh`

这些脚本覆盖：

- student RGB decoder baseline
- student RGB LoRA tap + decoder
- student RGB full backbone LLRD + decoder
- raw RAM identity + decoder
- raw RAM identity + LoRA tap + decoder
- RGB-ref low-lr backbone reference

注意：本迁移计划不会启动这些 formal 脚本的正式运行模式。后续只做等价参数的小 batch smoke。

## 数据现状

### 本机 ROD

规范目录：

```bash
/mnt/drive/3333_raw/ROD
```

关键说明：

- `README_name_note.md` 明确说明早期误命名为 `LOD`，后续应按 `ROD` 理解。
- 当前最新 ROD-night 脚本默认使用 `ROD_ROOT=/mnt/drive/3333_raw/ROD`。

训练硬依赖：

- `00Train-raws/00Train/*.raw`
- `01Valid-raws/01Valid/*.raw`
- `00Train-rggb/00Train/*.npy`
- `01Valid-rggb/01Valid/*.npy`
- `pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv`
- `pseudo_depth_dav2l_night_teacherbright_rel_1440x928/00Train/*.npy`
- `pseudo_depth_dav2l_night_teacherbright_rel_1440x928/01Valid/*.npy`

本机 teacherbright pseudo 统计：

- manifest 行数：`14037`，即样本 `14036`
- train：`12036`
- valid：`2000`
- 目录体积：约 `70G`

### 186 数据

当前已有旧名目录：

```bash
/mnt/drive/3333_raw/LOD
```

它实际对应本项目所说的 ROD，已有：

- train raw：`16089`
- valid raw：`4000`
- train rggb：`16089`
- valid rggb：`4000`

186 当前缺失：

- `/mnt/drive/3333_raw/ROD`
- `/mnt/drive/3333_raw/LOD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928`

因为用户后续可能会下载真实 LOD，所以本轮不做 symlink。直接将旧名 `LOD` 改名为 `ROD`。

## 迁移执行计划

### 1. 远端基础检查

登录 186 后先确认：

```bash
hostname
whoami
pwd
/home/a5000/anaconda3/bin/conda env list
df -h /home/a5000/6666_raw /mnt/drive/3333_raw
tmux ls || true
```

预期：

- host 为 `A5000`
- user 为 `a5000`
- `dav3` 环境存在
- `/mnt/drive` 剩余空间足够容纳约 `70G` pseudo 数据

### 2. 复制代码到 186

在本机执行：

```bash
rsync -a --info=progress2 \
  /home/caq/6666_raw/dav2_raw_0603/ \
  a5000@10.97.8.186:/home/a5000/6666_raw/dav2_raw_0603/
```

说明：

- 不加 exclude，完整复制项目目录内容。
- 代码体积约 `118M`，预计不需要 tmux。
- 如果实际网络很慢或中断风险变高，再改为 tmux 执行。

复制后在 186 检查：

```bash
cd /home/a5000/6666_raw/dav2_raw_0603
git status --short
test -f finetune_stf/train.py
test -f finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh
```

### 3. 将 186 旧名 `LOD` 改名为 `ROD`

在 186 执行前必须先做保护检查：

```bash
test ! -e /mnt/drive/3333_raw/ROD
test -d /mnt/drive/3333_raw/LOD
```

通过后执行：

```bash
mv /mnt/drive/3333_raw/LOD /mnt/drive/3333_raw/ROD
```

改名后检查：

```bash
test -d /mnt/drive/3333_raw/ROD
test ! -e /mnt/drive/3333_raw/LOD
find /mnt/drive/3333_raw/ROD/00Train-raws/00Train -maxdepth 1 -type f -name '*.raw' | wc -l
find /mnt/drive/3333_raw/ROD/01Valid-raws/01Valid -maxdepth 1 -type f -name '*.raw' | wc -l
find /mnt/drive/3333_raw/ROD/00Train-rggb/00Train -maxdepth 1 -type f -name '*.npy' | wc -l
find /mnt/drive/3333_raw/ROD/01Valid-rggb/01Valid -maxdepth 1 -type f -name '*.npy' | wc -l
```

预期数量：

- train raw：`16089`
- valid raw：`4000`
- train rggb：`16089`
- valid rggb：`4000`

### 4. 用 tmux 迁移缺失 teacherbright pseudo 数据

数据约 `70G`，必须用 tmux。

建议在本机启动 rsync tmux，因为数据源在本机，目标在 186：

```bash
session="0606_rod_teacherbright_rsync_to_186"
log="/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/${session}.log"

tmux new-session -d -s "${session}" \
  "rsync -a --info=progress2 \
    /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/ \
    a5000@10.97.8.186:/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/ \
    2>&1 | tee -a '${log}'"
```

启动后报告：

```bash
tmux attach -t 0606_rod_teacherbright_rsync_to_186
tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0606_rod_teacherbright_rsync_to_186.log
```

完成后在 186 检查：

```bash
test -f /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv
test -f /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/run_config.json
find /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/00Train -maxdepth 1 -type f -name '*.npy' | wc -l
find /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/01Valid -maxdepth 1 -type f -name '*.npy' | wc -l
wc -l /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv
```

预期：

- train pseudo：`12036`
- valid pseudo：`2000`
- manifest 行数：`14037`

### 5. 远端路径适配

186 已有 DAv2-S checkpoint：

```bash
/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth
```

当前脚本默认 checkpoint 路径是本机路径：

```bash
/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
```

不改实验语义参数。执行 smoke 或后续正式脚本时显式覆盖工程路径：

```bash
export ROOT=/home/a5000/6666_raw/dav2_raw_0603
export CONDA_BIN=/home/a5000/anaconda3/bin/conda
export CONDA_ENV=dav3
export PRETRAINED=/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth
export ROD_ROOT=/mnt/drive/3333_raw/ROD
export ROD_MANIFEST=/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv
export HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ckpt_186
export EXP_ROOT=${ROOT}/finetune_stf/exp
export LOG_ROOT=${ROOT}/finetune_stf/logs
```

说明：

- `ROOT`、`CONDA_BIN`、`PRETRAINED`、`HEAVY_ROOT` 是工程路径适配。
- `ROD_ROOT` 改为规范名 `ROD`。
- 不通过路径字符串推断实验语义。

## 小 batch smoke test 计划

smoke 只验证：

- import 可用
- config 参数可解析
- ROD manifest 可读
- student RGB 数据链路可读
- raw RAM 数据链路可读
- 能跑极小训练步和极小验证样本

不启动 formal 实验，不跑完整 epoch。

所有 smoke 输出路径必须包含 `codex_smoke`，例如：

- `/tmp/codex_smoke_186_rod_studentrgb`
- `/tmp/codex_smoke_186_rod_rawram`
- `/tmp/codex_smoke_186_heavy`

成功后只删除这些明确 smoke 临时目录；失败则保留并报告路径。

### 1. student RGB smoke

在 186 的项目目录执行：

```bash
cd /home/a5000/6666_raw/dav2_raw_0603

CUDA_VISIBLE_DEVICES=0 /home/a5000/anaconda3/bin/conda run --live-stream -n dav3 \
  torchrun --nproc_per_node=1 --master_port=29961 finetune_stf/train.py \
    --encoder vits \
    --stage rod_only \
    --input-domain rgb \
    --front-end dav2_rgb \
    --model-input-tensor image \
    --dataset-family rod_raw_student_rgb \
    --dataset-input-mode raw24_student_rgb \
    --raw-storage-format n_a \
    --bridge none \
    --decoder-feature-adapter none \
    --lora none \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0 \
    --rod-root /mnt/drive/3333_raw/ROD \
    --rod-night-manifest /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv \
    --rod-raw-source raw24 \
    --rod-rgb-pipeline student_dark_degreen_v1 \
    --rod-student-white-percentile 99.9 \
    --rod-student-gamma 0.9 \
    --rod-student-channel-gains 1.08 0.95 1.10 \
    --rod-label-space inverse_relative \
    --input-height 512 \
    --input-width 960 \
    --rod-train-crop-mode random \
    --rod-val-crop-mode center \
    --no-eval-stf \
    --eval-rod \
    --best-metric rod \
    --bs 1 \
    --accum-steps 1 \
    --lr 1e-5 \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --epochs 1 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 0 \
    --log-interval 1 \
    --debug-max-train-steps 1 \
    --debug-max-val-samples 2 \
    --no-enable-fixed-viz-dump \
    --no-enable-train-source-viz-dump \
    --pretrained-from /mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth \
    --heavy-save-root /tmp/codex_smoke_186_heavy_studentrgb \
    --save-path /tmp/codex_smoke_186_rod_studentrgb
```

### 2. raw RAM smoke

在 186 的项目目录执行：

```bash
cd /home/a5000/6666_raw/dav2_raw_0603

PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES=0 /home/a5000/anaconda3/bin/conda run --live-stream -n dav3 \
  torchrun --nproc_per_node=1 --master_port=29962 finetune_stf/train.py \
    --encoder vits \
    --stage rod_only \
    --dataset-family rod_raw \
    --dataset-input-mode raw_ram \
    --input-domain raw4 \
    --front-end raw_to_base_rgb_ram3 \
    --model-input-tensor raw \
    --raw-storage-format n_a \
    --bridge none \
    --decoder-feature-adapter none \
    --lora none \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0 \
    --raw-front-end-lr 5e-5 \
    --raw-ram-rgb-tail identity \
    --rod-root /mnt/drive/3333_raw/ROD \
    --rod-night-manifest /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv \
    --rod-raw-source raw24 \
    --rod-label-space inverse_relative \
    --input-height 512 \
    --input-width 960 \
    --rod-train-crop-mode random \
    --rod-val-crop-mode center \
    --no-eval-stf \
    --eval-rod \
    --best-metric rod \
    --bs 1 \
    --accum-steps 1 \
    --lr 1e-5 \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --epochs 1 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 0 \
    --log-interval 1 \
    --debug-max-train-steps 1 \
    --debug-max-val-samples 2 \
    --no-enable-fixed-viz-dump \
    --no-enable-train-source-viz-dump \
    --pretrained-from /mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth \
    --heavy-save-root /tmp/codex_smoke_186_heavy_rawram \
    --save-path /tmp/codex_smoke_186_rod_rawram
```

### 3. smoke 清理规则

如果 smoke 成功：

```bash
rm -rf /tmp/codex_smoke_186_rod_studentrgb
rm -rf /tmp/codex_smoke_186_heavy_studentrgb
rm -rf /tmp/codex_smoke_186_rod_rawram
rm -rf /tmp/codex_smoke_186_heavy_rawram
```

如果 smoke 失败：

- 不删除任何 smoke 输出。
- 报告失败命令、日志路径、保留目录。

## 后续正式实验提醒

本轮不启动 formal 实验。

如果后续要启动正式实验：

- 使用远端本地时间生成 `MMDD_HHMM` 前缀。
- 所有长时间训练必须放入 tmux。
- 必须显式覆盖 186 工程路径：
  - `ROOT=/home/a5000/6666_raw/dav2_raw_0603`
  - `CONDA_BIN=/home/a5000/anaconda3/bin/conda`
  - `PRETRAINED=/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth`
  - `ROD_ROOT=/mnt/drive/3333_raw/ROD`
  - `ROD_MANIFEST=/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv`
  - `HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ckpt_186`
- 不依赖路径名或默认值隐式决定实验语义。

## 后续正式启动参考

以下命令在 186 上执行。formal 脚本不加 `--run-internal` 时会自动创建 tmux session；不要手动复用已有 session。启动后脚本会打印 `tmux session`、`queue log`、`attach` 和 `monitor` 命令。

### 1. 进入项目并设置 186 路径

```bash
cd /home/a5000/6666_raw/dav2_raw_0603

export ROOT=/home/a5000/6666_raw/dav2_raw_0603
export EXP_ROOT=${ROOT}/finetune_stf/exp
export LOG_ROOT=${ROOT}/finetune_stf/logs
export HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ckpt_186
export CONDA_BIN=/home/a5000/anaconda3/bin/conda
export CONDA_ENV=dav3
export PRETRAINED=/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth
export ROD_ROOT=/mnt/drive/3333_raw/ROD
export ROD_MANIFEST=/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv
export GPU=0
```

启动前建议检查：

```bash
hostname
whoami
pwd
${CONDA_BIN} env list | grep -E '(^| )dav3( |$)'
nvidia-smi
test -f "${PRETRAINED}"
test -f "${ROD_MANIFEST}"
test -d "${ROD_ROOT}"
tmux ls || true
```

### 2. 可选 audit

下面两个队列脚本支持 `--audit`，会验证 resolved config，不启动训练：

```bash
bash finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh --audit

PHASE1_BNCLEAN_REVIEWED=1 \
bash finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh --audit
```

### 3. 启动 formal 队列

三个入口分别覆盖当前 ROD-night 复现实验范围。每次启动会自动使用远端本地时间生成 `MMDD_HHMM` 前缀。

student RGB decoder baseline，epochs=10：

```bash
bash finetune_stf/scripts/formal/0604_run_rod_night_student_rgb_decoder_e10_queue.sh
```

student RGB encoder budget 队列，顺序运行 LoRA tap + decoder 和 full backbone LLRD + decoder，epochs=10：

```bash
bash finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh
```

raw RAM v1 队列，顺序运行 raw RAM decoder、raw RAM LoRA tap + decoder、RGB-ref low-lr backbone reference，epochs=5：

```bash
PHASE1_BNCLEAN_REVIEWED=1 \
bash finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh
```

如果并行启动多个队列，必须显式分配不同 GPU 和 master port。例如：

```bash
GPU=1 B_MASTER_PORT=29625 C_MASTER_PORT=29626 \
bash finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh
```

```bash
GPU=1 R1_MASTER_PORT=29631 R2_MASTER_PORT=29632 RGB_MASTER_PORT=29633 \
PHASE1_BNCLEAN_REVIEWED=1 \
bash finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh
```

### 4. 查看运行状态

启动脚本会打印实际 session 和 log。通用查看方式：

```bash
tmux ls
tmux attach -t <SESSION>
tail -f <QUEUE_LOG>
```

单个实验的详细日志在：

```bash
ls -lh ${LOG_ROOT}/*rod_night*.tmux.log
tail -f ${LOG_ROOT}/<RUN_NAME>.tmux.log
```

输出位置：

```bash
ls -lh ${EXP_ROOT}
ls -lh ${HEAVY_ROOT}
```

### 5. 启动约束

- 这些是 formal 实验，启动前确认不是 smoke/debug/tmp/codex_smoke 路径。
- 不要依赖默认本机路径；必须使用上面的 186 显式路径覆盖。
- 不要手动传 `--run-internal`，它只给脚本创建的 tmux 内部进程使用。
- 若脚本报已有 `SAVE` 或 `HEAVY`，不要覆盖旧实验；换新的启动时间或先确认旧目录含义。
- 长时间训练必须保留在 tmux 中运行。
