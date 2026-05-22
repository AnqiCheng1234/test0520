# 参数联动修订总览

日期：2026-05-22

来源：`plans/0522_fix_code/parameter_coupling_audit.md`

目标：把后续迁移与修订拆成可按序执行的步骤。当前项目 `/home/caq/6666_raw/dav2_raw_0520` 作为审计来源和兼容参考；实际代码重整优先在新项目 `/home/caq/6666_raw/dav2_raw_0522` 中完成。

## 执行顺序

| 步骤 | 文档 | 主要目标 | 完成标志 |
| --- | --- | --- | --- |
| 01 | `01_migrate_to_dav2_raw_0522.md` | 迁移一份干净项目到 `/home/caq/6666_raw/dav2_raw_0522` | 新项目可 import，基础 help/compile 通过，迁移清单留档 |
| 02 | `02_scope_cleanup_and_legacy_path_inventory.md` | 清点并移除新项目不再需要维护的 mix 训练路径 | raw_mix / VKITTI mix / HyperSim mix / LOD mix 不再进入新训练入口 |
| 03 | `03_resolved_config_schema_and_alias.md` | 建立正交 resolved config，替代 `input_type` 一变量多义 | 新配置能表达 input/domain/front-end/bridge/adaptor/LoRA 等维度 |
| 04 | `04_applicability_validation_and_config_dump.md` | 增加适用性校验和完整 config dump | 活跃参数必须显式；非活跃参数显示 not applicable 或直接报错 |
| 05 | `05_stf_raw_storage_and_dataset_plan.md` | 将 STF raw 解码、通道顺序、norm 统一为 `raw_storage_format` | 不再依赖 raw root 路径字符串推断 canonical/legacy |
| 06 | `06_model_frontend_decomposition_plan.md` | 拆开 RGB / raw4 / raw_to_rgb / raw_ram4 / raw_to_base_rgb_ram3 前端 | `front_end` 决定模型前端，`input_domain` 只决定 wrapper 输入张量域 |
| 07 | `07_bridge_feature_adapter_lora_plan.md` | 解耦 bridge、decoder feature adaptor、LoRA、各自 feature source | 可以构造“只加 decoder feature adaptor”等干净对照路径 |
| 08 | `08_train_eval_viz_input_and_eval_protocol_plan.md` | 统一 train/eval/viz 的输入选择，并明确 KITTI eval protocol | `model_input_tensor` 单点控制输入；KITTI 指标明确模型来源 |
| 09 | `09_formal_scripts_and_experiment_matrix_plan.md` | 重写正式脚本和实验矩阵 | 正式脚本显式写出关键变量，并标注实际改变的实验维度 |
| 10 | `10_smoke_tests_and_acceptance_plan.md` | 建立 smoke test 和验收流程 | 每一步有最小验证，成功 smoke 临时产物被清理 |

## 总体原则

1. 先迁移，后修改。不要在 `/home/caq/6666_raw/dav2_raw_0520` 上直接做破坏性重构。
2. 新项目正式实验不依赖隐式默认值。会影响实验语义的参数必须由脚本显式传入。
3. 必要联动改成显式适用性校验。顶层功能开关关闭时应显式为 `none` / `n_a`；关闭功能的子参数不在 CLI 中传入，若传入有效值则报错，并在 resolved config 中记录为 `not_applicable`。
4. `input_type` 只保留为兼容 alias。新代码内部使用 resolved config，不再用字符串包含关系推断行为。
5. 数据、模型、eval、viz 共享同一个 resolved config，避免训练读 `image`、可视化读 `raw` 这类 latent bug。
6. 移除新项目不再需要的 mix 训练分支，当前项目保留旧实验兼容。
7. 所有 smoke 输出路径必须包含 `smoke`、`debug`、`tmp` 或 `codex_smoke`。成功后只删除明确临时产物；失败时保留并报告路径。
8. 正式实验名必须以远端本地时间 `MMDD_HHMM` 开头；长时间训练必须放进 tmux。

## 推荐执行方式

每次只执行一个步骤文档。完成一个步骤后，至少更新：

- 新项目中的代码或脚本
- 对应步骤文档中的完成记录
- `resolved_config` 或 smoke 日志中的关键验证结果

执行时优先使用 `dav3` conda 环境：

```bash
conda run -n dav3 python -m compileall finetune_stf foundation
```

涉及正式训练、数据转换、checkpoint 转换或预计持续数小时的命令时，按远程开发习惯使用 tmux，并记录 session、日志、attach 和 tail 命令。
