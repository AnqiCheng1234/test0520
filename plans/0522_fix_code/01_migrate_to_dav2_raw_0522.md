# 01 迁移到 dav2_raw_0522

目标：从当前审计项目 `/home/caq/6666_raw/dav2_raw_0520` 迁移出一份新的工作项目 `/home/caq/6666_raw/dav2_raw_0522`。新项目用于后续重构，当前项目保持为旧实验和审计参考。

## 输入

- 源项目：`/home/caq/6666_raw/dav2_raw_0520`
- 目标项目：`/home/caq/6666_raw/dav2_raw_0522`
- 参考文档：`plans/0522_fix_code/parameter_coupling_audit.md`

## 执行步骤

1. 确认源项目状态：

```bash
cd /home/caq/6666_raw/dav2_raw_0520
pwd
git status --short || true
rg --files plans/0522_fix_code
```

2. 检查目标路径是否已经存在。若已存在，不直接覆盖；先人工确认里面是否有用户文件：

```bash
test -e /home/caq/6666_raw/dav2_raw_0522 && find /home/caq/6666_raw/dav2_raw_0522 -maxdepth 2 -type f | head
```

3. 创建目标项目。优先只迁移代码、脚本和轻量配置，不迁移 `plans/`、正式实验输出、checkpoint、cache、可视化大文件。

建议排除项：

- `finetune_stf/exp/`
- `plans/`
- `runs/`
- `wandb/`
- `__pycache__/`
- `.pytest_cache/`
- 大型 checkpoint / npz / png / jpg 结果目录，除非它们是代码依赖

示例命令：

```bash
mkdir -p /home/caq/6666_raw/dav2_raw_0522
rsync -a --info=progress2 \
  --exclude 'finetune_stf/exp/' \
  --exclude 'plans/' \
  --exclude 'runs/' \
  --exclude 'wandb/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /home/caq/6666_raw/dav2_raw_0520/ \
  /home/caq/6666_raw/dav2_raw_0522/
```

如果实际复制预计很久，把 `rsync` 放进 tmux，并写日志：

```bash
tmux new -s migrate_0522 -d 'rsync -a --info=progress2 --exclude "finetune_stf/exp/" --exclude "plans/" --exclude "runs/" --exclude "wandb/" --exclude "__pycache__/" --exclude ".pytest_cache/" /home/caq/6666_raw/dav2_raw_0520/ /home/caq/6666_raw/dav2_raw_0522/ > /home/caq/6666_raw/dav2_raw_0522_migrate.log 2>&1'
tmux attach -t migrate_0522
tail -f /home/caq/6666_raw/dav2_raw_0522_migrate.log
```

4. 在目标项目生成迁移记录，例如 `/home/caq/6666_raw/dav2_raw_0522/migration_manifest_0522.md`，记录：

- 源路径和目标路径
- 执行时间
- rsync 排除项
- 是否保留 git metadata
- 未迁移的大目录
- `plans/` 已明确不迁移；后续计划文档仍以源项目中的 `plans/0522_fix_code/` 为准
- 后续重构只在目标项目执行

5. 在目标项目做基础可用性检查：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
conda run -n dav3 python -m compileall finetune_stf foundation
conda run -n dav3 python finetune_stf/train.py --help >/tmp/dav2_raw_0522_train_help_codex_smoke.txt
```

成功后删除明确临时 help 输出：

```bash
rm -f /tmp/dav2_raw_0522_train_help_codex_smoke.txt
```

## 验收标准

- `/home/caq/6666_raw/dav2_raw_0522` 存在并包含核心代码目录。
- `conda run -n dav3 python -m compileall finetune_stf foundation` 通过。
- `finetune_stf/train.py --help` 可运行。
- 迁移记录已写入目标项目。
- 没有删除源项目里的数据、checkpoint、正式实验输出或用户文件。

## 风险点

- 目标路径若已存在，不能直接覆盖。
- 不应迁移大量正式实验输出到新项目，否则后续重构和搜索会被历史结果干扰。
- 如果后续需要复现实验结果，应该从当前项目或正式 checkpoint 根目录读取，而不是把结果目录复制进新项目。

## 完成记录

执行时间：2026-05-22 19:11 CST

- 已创建 `/home/caq/6666_raw/dav2_raw_0522`。
- 迁移排除了 `finetune_stf/exp/`、`plans/`、`runs/`、`wandb/`、`logs/`、`codex_debug/`、`__pycache__/`、`.pytest_cache/`。
- 新项目中的复制 `.git` 已移除，避免因计划文档未迁移而产生误导性的 dirty status。
- 迁移记录已写入 `/home/caq/6666_raw/dav2_raw_0522/migration_manifest_0522.md`。
- `conda run -n dav3 python -m compileall finetune_stf foundation` 通过。
- `conda run -n dav3 python finetune_stf/train.py --help` 通过，临时输出 `/tmp/dav2_raw_0522_train_help_codex_smoke.txt` 已删除。
