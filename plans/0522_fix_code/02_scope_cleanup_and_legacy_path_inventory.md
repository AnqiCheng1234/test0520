# 02 清理新项目范围与旧路径清单

目标：在新项目中明确“后续要维护的训练路径”和“仅旧实验兼容的路径”。审计文档已经说明后续不再需要 mix 数据训练，因此新项目优先移除或禁用 raw mix / VKITTI mix / HyperSim mix / LOD mix 相关入口，降低后续配置矩阵复杂度。

## 主要触点

- `finetune_stf/train.py`
- `foundation/engine/datasets/hypersim_processed_raw.py`
- `finetune_stf/scripts/`
- `finetune_stf/dataset/lod_raw.py`
- `finetune_stf/dataset/vkitti2.py`

## 执行步骤

1. 在新项目中盘点所有 mix 相关符号：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
rg -n "raw_mix|vkitti_lod|train_sources|train_source_ratios|HypersimProcessedRaw|VKITTI2Raw|CachedVKITTI2Raw|lod_per_vkitti|lod_fraction" .
```

2. 把路径分成三类：

| 类别 | 处理方式 |
| --- | --- |
| 后续仍使用 | 保留并纳入 resolved config |
| 只服务旧 mix 训练 | 从新训练入口删除或硬报错 |
| 只服务旧脚本记录 | 移到 `scripts/legacy/` 或在文件头标注 deprecated |

3. 修改 `train.py` 的 stage choices。新项目建议先只保留当前后续确实要维护的 stage，例如：

- `stf_only`
- `eval_only`

如果仍需要单独 LOD 或 VKITTI 评估，保留 eval dataset，但不要保留 mix 训练调度。

4. 删除或禁用以下 parser 参数的活跃路径：

- `--train-sources`
- `--train-source-ratios`
- `--train-steps-per-epoch` 中只服务 raw mix 的语义
- `--lod-per-vkitti`
- `--lod-fraction`
- `--vkitti-cache-root`
- `--vkitti-randomize-unprocessing`
- `--vkitti-unprocessing-preset`
- `--vkitti-unprocessing-mix-weights`
- HyperSim raw train 专用参数

如果暂时不删除参数，必须在新项目中加启动校验：非支持 stage 下传入有效值直接报错。

5. 清理 `build_datasets()` 和 dataloader 构建逻辑：

- 删除 raw mix source set
- 删除 `lod_day_train` / `lod_night_train` mix 分支
- 删除 VKITTI / HyperSim train dataset 构造
- 保留 STF train/val 与明确需要的 eval dataset

6. 清理脚本：

- 正式保留脚本移到 `finetune_stf/scripts/formal/`
- 旧实验脚本移到 `finetune_stf/scripts/legacy/`
- 对 `train_raw_ram_e3_bridge.sh` 这类尺寸不兼容脚本加 deprecated 注释，或不迁入新项目正式脚本目录

## 验收标准

- `rg -n "raw_mix|train_sources|train_source_ratios|HypersimProcessedRaw|VKITTI2Raw"` 在新训练入口中不再出现活跃分支。
- `train.py --help` 不再把 mix 训练作为新项目推荐路径。
- 旧脚本不会被误认为当前正式实验脚本。
- compile 通过。

## 风险点

- 不要删除 eval 仍需要的 dataset 类，除非已经确认后续不做对应 eval。
- 如果某个旧 checkpoint 加载逻辑依赖旧类名，先保留兼容 adapter，并在文档里标注只读兼容用途。
- 删除路径前先用 `rg` 确认没有新项目正式入口引用。

## 完成记录

执行时间：2026-05-22 19:11 CST

- 已在 `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/train.py` 将 stage choices 收敛为 `stf_only` / `eval_only`。
- `--eval-only` 保留为兼容 alias，并解析为 `stage=eval_only`。
- 已从新训练入口删除 raw mix、VKITTI train、HyperSim train、LOD train、LOD/VKITTI mix schedule 和相关 CLI 参数的活跃路径。
- 旧 shell 启动脚本已移动到 `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/scripts/legacy/`。
- 当前保留的 STF-only 队列脚本已移动到 `/home/caq/6666_raw/dav2_raw_0522/finetune_stf/scripts/formal/`。
- 清理清单已写入 `/home/caq/6666_raw/dav2_raw_0522/scope_cleanup_legacy_inventory_0522.md`。
- `rg -n "raw_mix|train_sources|train_source_ratios|HypersimProcessedRaw|VKITTI2Raw" finetune_stf/train.py` 无匹配。
- `conda run -n dav3 python -m compileall finetune_stf foundation` 通过。
- `conda run -n dav3 python finetune_stf/train.py --help` 通过，临时 smoke 输出已删除。
- `--stage raw_mix` 已验证会在参数解析阶段报错；临时 invalid-stage smoke 日志已删除。
