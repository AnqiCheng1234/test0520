# 09 正式脚本与实验矩阵

目标：把新项目的正式实验脚本改成显式配置驱动。脚本本身要说明本实验只改变哪些维度，避免“看似改 A，实际联动 B/C/D”。

## 脚本目录建议

```text
finetune_stf/scripts/
  formal/
  smoke/
  legacy/
```

- `formal/`：后续正式实验脚本
- `smoke/`：小样本验证脚本，输出路径必须含 `smoke` / `debug` / `tmp` / `codex_smoke`
- `legacy/`：从当前项目迁移来的旧脚本，只作参考

## 正式脚本显式参数规则

每个 formal 脚本必须显式写出顶层实验语义：

```text
--input-domain
--front-end
--dataset-family
--dataset-input-mode
--model-input-tensor
--input-height
--input-width
--raw-storage-format
--bridge
--decoder-feature-adapter
--lora
--dav2-train-mode
--loss-type
```

功能开启时，必须额外显式写出对应子参数：

```text
bridge=raw_feature_bridge:
--bridge-source
--bridge-feature-source-channels
--bridge-feature-keys
--bridge-layers

decoder-feature-adapter=raw_feature_adapter:
--adapter-feature-source-channels
--feature-adapter-keys

lora=dav2_lora:
--lora-rank
--lora-alpha
--lora-lr
--lora-tap-layers

eval-kitti enabled:
--kitti-eval-protocol
```

功能关闭时，顶层开关显式为 `none` / `n_a`，对应子参数不传；resolved config 中记录为 `not_applicable`。不要在 formal 脚本中写 `--bridge-layers none` 或 `--lora-tap-layers none`，除非 CLI parser 已明确支持这种语法。

## 推荐实验矩阵

第一批只覆盖必要单变量对照：

| 编号 | 目的 | 关键配置 |
| --- | --- | --- |
| A | RGB baseline | `input_domain=rgb`, `front_end=dav2_rgb` |
| B | raw4 -> 3ch RAM front-end | `front_end=raw_to_base_rgb_ram3`, no bridge/adaptor |
| C | raw4 -> 4ch RAM front-end | `front_end=raw_ram4`, no bridge/adaptor |
| D | 3ch RAM + bridge | B + `bridge=raw_feature_bridge`, `bridge_feature_source_channels=x3` |
| E | 3ch RAM + decoder feature adaptor | B + `decoder_feature_adapter=raw_feature_adapter`, `adapter_feature_source_channels=x3` |
| F | 3ch RAM + bridge + adaptor | B + `bridge_feature_source_channels=x3`, `adapter_feature_source_channels=x3` |
| G | RGB + LoRA | A + `lora=dav2_lora` |

后续如果需要 4ch adaptor 对照，再加：

- `raw_ram4 + decoder_feature_adapter(x4)`
- `raw_ram4 + bridge(x4)`
- `raw_ram4 + bridge + adaptor(x4)`

## 执行步骤

1. 迁移旧脚本：

- 保留历史脚本到 `legacy/`
- 文件头添加注释：旧脚本可能依赖 `input_type` 隐式联动，不作为新项目正式实验入口

2. 编写 smoke 脚本：

- 每个核心路径一个 smoke
- `--save-path` / `--heavy-save-root` 包含 `codex_smoke`
- 使用 debug max samples 或极小 epoch
- 成功后删除临时输出

3. 编写 formal 脚本：

- 文件名和实验名都以 launch time `MMDD_HHMM` 开头
- 注释写明“唯一改变维度”
- 设置 `conda run -n dav3` 或在脚本开头明确激活 `dav3`
- 只在功能开启时写对应子参数；功能关闭时依赖适用性校验把子参数标为 `not_applicable`
- raw_to_base_rgb_ram3 路径如果仍保留 guard，脚本中显式 `PHASE1_BNCLEAN_REVIEWED=1`，并在实验名中体现 `bnclean`

4. 给每个 formal 脚本写日志路径：

```bash
LOG=/path/to/logs/${EXP_NAME}.log
```

长时间正式训练必须用 tmux：

```bash
tmux new -s ${EXP_NAME} -d "bash finetune_stf/scripts/formal/${EXP_NAME}.sh > ${LOG} 2>&1"
tmux attach -t ${EXP_NAME}
tail -f ${LOG}
```

## 验收标准

- `formal/` 里没有依赖旧 `--input-type` 隐式语义的新脚本。
- 每个 formal 脚本能从文件内容看出实验改变的唯一维度。
- smoke 脚本输出路径都带清晰临时标记。
- 旧脚本不会被误运行成新正式实验。

## 风险点

- 正式脚本不能为了省参数而依赖 resolver 默认值。
- bridge、decoder feature adaptor、LoRA 开启时，其 keys、layers、tap layers 必须显式，尤其是 encoder 变化时。

## 执行记录

2026-05-22 已在 `/home/caq/6666_raw/dav2_raw_0522` 执行：

- 更新 `finetune_stf/scripts/formal/README.md`，明确 maintained formal 脚本必须使用 orthogonal resolved-config flags，不再使用 `--input-type`。
- 更新两个 maintained formal queue 脚本，移除旧 `--input-type`、`--stf-repeat`、`--stf-raw-decode-mode`，显式写出 input/domain/front-end/dataset/input tensor/raw storage/bridge/adaptor/LoRA/loss/train mode。
- 新增 `finetune_stf/scripts/formal/EXPERIMENT_MATRIX.md`，记录 A-G 首批单变量对照矩阵。
- 新增 `finetune_stf/scripts/smoke/0522_2054_smoke_resolved_input.sh`，作为 resolved config 和输入选择的基础 smoke。
