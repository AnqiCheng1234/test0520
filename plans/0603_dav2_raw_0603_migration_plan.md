# 0603 dav2_raw_0603 代码迁移计划

## 目标

在 64 机器上，从当前项目：

`/home/caq/6666_raw/dav2_raw_0522`

新复制一份代码项目到：

`/home/caq/6666_raw/dav2_raw_0603`

新项目先不启动 residual 专用实验，重点继续围绕原主线做 DAv2 backbone、decoder、RAW/RAM、LoRA、feature adapter、bridge 等微调与评估。

本计划只定义迁移方案和验证标准；正式执行迁移前不应删除 0522 中的任何代码、数据、checkpoint、正式实验输出或用户文件。

## 已确认需求

1. 复制当前 0522 工作树现状，包括当前未提交修改和新增文件。
2. 先完整复制代码，暂不做激进删减。
3. 保留 RAM 主线中的 `rgb_interface_mode=residual_tanh`。
4. 保留 `finetune_stf/scripts/formal/`，但 0603 新 formal 脚本优先服务 backbone/decoder/RAM/LoRA/adapter/bridge 主线。
5. 迁移时排除 `plans/`、实验输出、日志、checkpoint、第三方目录和缓存。
6. 本次只写中文迁移计划文件到 0522 的 `plans/` 目录中。

## 当前 0522 状态摘要

- 当前机器：64，host/user 均为 `caq`。
- 当前项目目录：`/home/caq/6666_raw/dav2_raw_0522`。
- 默认 conda 环境：`dav3`，该环境存在；在当前 SSH 非交互命令环境中应优先使用显式路径 `/home/caq/anaconda3/bin/conda`，不要假设裸 `conda` 一定在 `PATH` 中。
- 目标目录 `/home/caq/6666_raw/dav2_raw_0603` 当前不存在。
- 0522 当前工作树有未提交修改和新增文件，迁移应复制当前工作树现状，不应只复制 Git 已跟踪文件。
- 训练主入口仍是 `finetune_stf/train.py`。
- 参数集中解析在 `finetune_stf/config/resolved.py`。
- `finetune_stf/train.py` 已支持 `--dav2-train-mode none|decoder|full|last:N|first:N|range:a-b`。
- optimizer 已按 `raw_front_end`、`bridge`、`decoder_feature_adapter`、`lora`、`dav2_decoder` 等组做学习率分组。

## residual 边界定义

0603 中需要区分两类 residual：

1. 保留的 RAM 主线 residual 接口

这些属于当前 RAW/RAM 主线能力，不应因“不做 residual 专用实验”而删除：

- `rgb_interface_mode=residual_tanh`
- `RGBInterfaceHead` 中的 residual RGB interface
- 普通 `raw_ram` / `raw_ram_rgb` / RAM + LoRA / RAM + adapter / RAM + bridge 路径

2. 暂不作为 0603 主线的 residual 专用实验

这些保留代码副本，但默认不作为 0603 新实验入口：

- `foundation/engine/models/dav2_residual_control.py`
- `foundation/engine/models/raw_residual_dav2.py`
- `foundation/engine/models/dav2_incremental_residual.py`
- `foundation/tools/train_vkitti2_raw_residual.py`
- `foundation/tools/train_vkitti2_residual_control.py`
- `foundation/tools/train_vkitti2_incremental_residual.py`
- `foundation/tools/train_led_hb_residual_control.py`
- `foundation/tools/train_led_hb_incremental_residual.py`
- residual panel、summary、diagnostic、feature ablation 相关工具
- `finetune_stf/scripts/formal/0524` 及之后明显以 residual/control/incremental 为主题的 formal 脚本

0603 初期原则：这些文件可以完整复制以保留审计和回滚能力，但新实验脚本、README、计划和 launch 命令不应默认引用它们。

## 迁移前置：先上传 0522 到 GitHub

因为 0522 当前有未提交改动，正式迁移前建议先将 0522 的当前代码状态备份到 GitHub。

建议步骤：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
git status --short
git diff --stat
git ls-files --others --exclude-standard | sed -n '1,120p'
```

当前 0522 已确认存在大量未跟踪 `plans/` 产物和未跟踪 `third_party/` 文件，因此不要在未收敛 `.gitignore` 前直接执行 `git add -A`。

备份前应先补充或确认忽略规则，至少覆盖不应上传的运行产物和第三方目录：

```bash
git status --short
git ls-files --others --exclude-standard | sed -n '1,200p'
```

如果需要修改 `.gitignore`，建议优先加入：

```gitignore
plans/**/stf_val/
plans/**/logs/
plans/**/*.npy
plans/**/*.npz
plans/**/*.png
plans/**/*.jpg
plans/**/*.jsonl
third_party/
```

然后先暂存已跟踪文件的修改，再只选择性暂存确认属于代码和必要配置的新增文件。这样可以覆盖当前 tracked 代码改动，同时避免把未跟踪数据、checkpoint、实验输出、日志、图像面板和第三方包拉进提交：

```bash
git add -u
git add .gitignore
git add anqi_eval/depth_metric_common.py anqi_eval/eval_vkitti_metric_depth.py
git add finetune_stf/tools/calibrate_unprocessing_to_stf_decoded.py finetune_stf/tools/realraw_unprocessing_calibration.json
git add foundation/tools/eval_stf_led_hb_formal.py foundation/tools/visualize_led_stf_raw_distribution.py
git add plans/0603_dav2_raw_0603_migration_plan.md
git diff --cached --stat
git status --short
```

如果确认 staged 内容就是 0522 备份版本：

```bash
git commit -m "0603 backup before dav2_raw_0603 migration"
git remote -v
git push
```

如果当前没有 GitHub remote，应先在 GitHub 创建仓库，然后添加 remote：

```bash
git remote add origin <GITHUB_REPO_URL>
git push -u origin HEAD
```

注意：

- 不要把数据集、checkpoint、正式实验输出、日志大文件误提交。
- 若 `.gitignore` 不足，应先补充忽略规则，再提交；不要用未经检查的 `git add -A`。
- 提交前必须检查 `git diff --cached --stat` 和 `git status --short`，确认 staged 内容没有 `plans/` 大量产物、`third_party/`、checkpoint 或实验输出。
- 如果 GitHub 仓库尚未确定，应停止并让用户确认 repo 地址。

## 建议复制策略

使用 `rsync` 复制当前工作树文件，而不是依赖 Git clone。这样可以包含当前未提交修改和新增文件。

建议命令：

```bash
cd /home/caq/6666_raw
rsync -a --info=progress2 \
  --exclude '.git' \
  --exclude '.git/' \
  --exclude 'plans/' \
  --exclude 'finetune_stf/exp/' \
  --exclude 'finetune_stf/exp_0520/' \
  --exclude 'finetune_stf/logs/' \
  --exclude 'checkpoints/' \
  --exclude 'third_party/' \
  --exclude 'runs/' \
  --exclude 'wandb/' \
  --exclude 'logs/' \
  --exclude 'codex_debug/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /home/caq/6666_raw/dav2_raw_0522/ \
  /home/caq/6666_raw/dav2_raw_0603/
```

执行正式复制前，先做 dry-run 检查排除规则：

```bash
cd /home/caq/6666_raw
rsync -an --info=progress2 \
  --exclude '.git' \
  --exclude '.git/' \
  --exclude 'plans/' \
  --exclude 'finetune_stf/exp/' \
  --exclude 'finetune_stf/exp_0520/' \
  --exclude 'finetune_stf/logs/' \
  --exclude 'checkpoints/' \
  --exclude 'third_party/' \
  --exclude 'runs/' \
  --exclude 'wandb/' \
  --exclude 'logs/' \
  --exclude 'codex_debug/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /home/caq/6666_raw/dav2_raw_0522/ \
  /home/caq/6666_raw/dav2_raw_0603/
```

排除原因：

- `plans/`：0522 的计划和报告不复制到新项目，避免旧计划污染 0603。
- `finetune_stf/exp/`、`finetune_stf/exp_0520/`：正式实验输出不复制。
- `finetune_stf/logs/`、`logs/`、`runs/`、`wandb/`：日志和运行记录不复制。
- `checkpoints/`：预训练或实验 checkpoint 不复制，后续显式引用外部路径。
- `third_party/`：第三方目录不复制，除非后续确认 0603 必须依赖。
- `.git`、`.git/`：0522 当前是 linked worktree，`.git` 是指向 `/home/caq/6666_raw/dav2_raw_0520/.git/worktrees/...` 的文件；必须排除，避免 0603 继承 0522 的 Git 元数据。
- `__pycache__/`、`.pytest_cache/`、`codex_debug/`：缓存和调试产物不复制。

## 复制后目录检查

复制后先检查目标目录和规模：

```bash
cd /home/caq/6666_raw/dav2_raw_0603
pwd
find . -maxdepth 2 -type d | sort | head -200
du -sh .
test ! -e .git
```

确认排除项不存在：

```bash
find . -maxdepth 2 \( \
  -name 'plans' -o \
  -name 'exp' -o \
  -name 'exp_0520' -o \
  -name 'logs' -o \
  -name 'runs' -o \
  -name 'wandb' -o \
  -name 'checkpoints' -o \
  -name 'third_party' -o \
  -name '__pycache__' -o \
  -name '.pytest_cache' -o \
  -name '.git' \
\) | sort
```

如果发现正式实验输出、checkpoint 或旧 plans 被复制，应停止并只删除明确属于本次复制错误产生的目标目录内容，不要删除 0522 源项目中的任何文件。

## 0603 主线保留能力

0603 初期应优先维护以下能力：

- RGB DAv2 decoder 微调：`front_end=dav2_rgb`，`--dav2-train-mode decoder`。
- RGB DAv2 backbone 微调：`--dav2-train-mode full|last:N|first:N|range:a-b`。
- RAW naive front-end：`front_end=raw_to_rgb_head`。
- 4 通道 RAM：`front_end=raw_ram4`。
- 3 通道 RamCore3：`front_end=raw_to_base_rgb_ram3`。
- LoRA：`lora=dav2_lora`。
- RAW feature bridge：`bridge=raw_feature_bridge`。
- decoder feature adapter：`decoder_feature_adapter=raw_feature_adapter`。
- STF RGB / STF RAW train and eval。
- KITTI、NYU、ETH3D、RobotCar、RobotCar night 等现有 eval path，但启用 eval 时必须显式设置实验语义参数。

## 0603 参数规范

新 formal launch 脚本必须显式写出实验语义参数，不依赖隐藏默认值、路径名推断或字符串匹配。

必须显式设置的典型参数包括：

- `--input-domain`
- `--front-end`
- `--dataset-family`
- `--dataset-input-mode`
- `--model-input-tensor`
- `--raw-storage-format`
- `--dav2-train-mode`
- `--bridge`
- `--bridge-source`，当启用 bridge 或 decoder feature adapter 时
- `--bridge-feature-source-channels`，当启用 bridge 时
- `--bridge-feature-keys`，当启用 bridge 时
- `--bridge-layers`，当启用 bridge 时
- `--decoder-feature-adapter`
- `--adapter-feature-source-channels`，当启用 decoder feature adapter 时
- `--feature-adapter-keys`，当启用 decoder feature adapter 时
- `--lora`
- `--lora-block-mode`，当启用 LoRA 时
- `--lora-rank`，当启用 LoRA 时
- `--lora-alpha`，当启用 LoRA 时
- `--lora-tap-layers`，当启用 LoRA 且 `--lora-block-mode tap` 时
- `--rgb-interface-mode`，当启用 RAM / RamCore3 RGB interface 时
- `--rgb-residual-scale`，当 `--rgb-interface-mode residual_tanh|residual_linear` 时
- `--raw-ram-rgb-tail`，当启用 `raw_to_base_rgb_ram3` / RAM RGB tail 时
- `--raw-front-end-lr`，当启用 RAW front-end 时
- `--bridge-lr`，当启用 bridge 或 decoder feature adapter 时
- `--lora-lr`，当启用 LoRA 时
- `--kitti-eval-protocol`，当启用 KITTI eval 时
- `--loss-type` 以及 `ssi_grad` 所需的 `--loss-lambda-grad`、`--loss-grad-scales`

不适用参数应显式使用 `none`、`n_a` 或 `not_applicable`，并由 `finetune_stf/config/resolved.py` 的 resolved config 阶段统一校验。

0603 新 formal 脚本不应让路径名、checkpoint 名、旧脚本名或 `input-type` alias 隐式决定上述实验语义字段。

## 0603 formal 脚本整理建议

复制后建议新增或整理一个 0603 专用 formal 区域，例如：

```text
finetune_stf/scripts/formal/0603/
```

0603 formal 脚本只覆盖当前主线：

- RGB decoder baseline
- RGB full/partial backbone 微调
- RAW naive front-end
- raw_ram4
- raw_to_base_rgb_ram3
- RAM + LoRA
- RAM + feature adapter
- RAM + bridge
- RAM + bridge + feature adapter

0524 之后 residual/control/incremental 相关脚本保留在原位置或移入 legacy 标记区，但 0603 初期不从这些脚本派生新实验。

复制后必须审计保留脚本中的 0522 硬编码路径，尤其是默认 `ROOT=/home/caq/6666_raw/dav2_raw_0522` 的脚本。建议检查：

```bash
cd /home/caq/6666_raw/dav2_raw_0603
grep -RIn '/home/caq/6666_raw/dav2_raw_0522\|dav2_raw_0522' \
  finetune_stf/scripts finetune_stf/tools anqi_eval foundation scripts tools README.md \
  --exclude-dir='__pycache__'
```

处理原则：

- 0603 新 formal/smoke 脚本应使用脚本相对路径或默认 `ROOT=/home/caq/6666_raw/dav2_raw_0603`。
- 保留的 legacy 脚本如果仍引用 0522，应加 legacy 标记，且不得作为 0603 初期 launch 入口。
- 任何输出路径、log 路径、`save-path`、`c2-run-dir`、数据 split 路径如果仍指向 0522，都必须逐项确认后才能用于 0603。

## smoke 验证

迁移后先使用 `dav3` 进行轻量验证。

编译检查：

```bash
cd /home/caq/6666_raw/dav2_raw_0603
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
"${CONDA_BIN}" run -n dav3 python -m compileall finetune_stf foundation anqi_eval scripts tools
```

训练入口 help：

```bash
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
"${CONDA_BIN}" run -n dav3 python finetune_stf/train.py --help \
  > /tmp/dav2_raw_0603_train_help_codex_smoke.txt
```

resolved config smoke 建议生成临时脚本或直接用已有 smoke 脚本验证：

```bash
CONDA_BIN=/home/caq/anaconda3/bin/conda CONDA_ENV=dav3 \
  bash finetune_stf/scripts/smoke/0522_2054_smoke_resolved_input.sh
```

如果 smoke 成功，可以删除明确临时产物：

```bash
rm -f /tmp/dav2_raw_0603_train_help_codex_smoke.txt
```

如果 smoke 失败，不删除失败日志或临时目录，保留并报告路径。

## 长任务和正式实验约束

迁移本身通常不是数小时任务，可以前台执行。

正式训练、长评估、大规模 pseudo generation、dataset conversion、checkpoint conversion 等长任务必须放入 tmux。

正式实验命名必须以远程服务器本地时间的 `MMDD_HHMM` 开头，例如：

```text
0603_1420_<experiment_name>
```

启动 tmux 后必须记录：

- tmux session name
- log path
- attach command
- tail command

示例：

```bash
tmux attach -t <SESSION>
tail -f <LOG_PATH>
```

## 迁移后建议 Git 初始化

如果 0603 作为新项目独立管理，复制后可以在 0603 初始化新 Git：

```bash
cd /home/caq/6666_raw/dav2_raw_0603
test ! -e .git
git init
git status --short
git add -A
git diff --cached --stat
git commit -m "0603 initialize dav2_raw_0603 from 0522 code"
```

如果需要上传到 GitHub，再由用户确认新仓库地址后添加 remote 并 push。

## 风险点

1. 0522 当前有未提交修改和大量未跟踪文件，必须先确认 GitHub 备份或至少记录当前 `git status --short`；备份时不要未经检查直接 `git add -A`。
2. residual 字样不能一刀切删除；RAM 主线中的 `residual_tanh` 需要保留。
3. 0522 是 linked worktree，`.git` 是文件而不是目录；rsync 必须排除 `.git` 文件，否则 0603 会继承旧 Git 元数据。
4. `foundation` 目录同时包含主线 raw naive / unprocessing / VKITTI 工具和 residual 专用代码，复制后应靠脚本入口和 README/计划约束使用范围，而不是先做大规模删除。
5. `third_party/` 被排除后，`anqi_eval/eval_vkitti_metric_depth.py` 等需要 official metric package 的脚本必须显式传入外部 `--metric-package-root`，或后续单独确认是否恢复 vendored package。
6. `checkpoints/` 被排除后，所有 launch 脚本必须显式引用外部 checkpoint 路径。
7. `plans/` 被排除后，0603 需要新建自己的计划目录，而不是复用 0522 的历史计划。
8. 多个保留的 0524-0531 formal/smoke 脚本默认 `ROOT=/home/caq/6666_raw/dav2_raw_0522`，复制后若不审计，可能把 0603 实验输出写回旧项目。
9. 当前 SSH 非交互环境中裸 `conda` 可能不可用；迁移验证和 launch 脚本应使用 `/home/caq/anaconda3/bin/conda` 或显式设置 `CONDA_BIN`。

## 完成标准

迁移执行完成后，应满足：

- `/home/caq/6666_raw/dav2_raw_0603` 存在。
- 代码已从 0522 当前工作树复制。
- 0603 中不存在 `.git` 文件或目录；如需独立 Git，必须在确认后重新 `git init`。
- `plans/`、`exp/`、`logs/`、`checkpoints/`、`third_party/`、cache 等排除项未进入 0603。
- `/home/caq/anaconda3/bin/conda run -n dav3 python -m compileall ...` 通过。
- `/home/caq/anaconda3/bin/conda run -n dav3 python finetune_stf/train.py --help` 通过。
- 0603 新 formal/smoke 入口不再默认指向 `/home/caq/6666_raw/dav2_raw_0522`。
- smoke 成功后的明确临时产物已删除。
- smoke 失败时保留失败日志并报告路径。
- 0603 新实验计划明确不以 residual-control / incremental-residual 为初期主线。
