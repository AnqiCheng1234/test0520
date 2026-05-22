# 04 适用性校验与完整 config dump

目标：让实验配置只记录真正参与当前路径的变量。活跃路径参数必须显式；非活跃路径参数不能静默以默认值进入实验记录。

## 主要触点

- `finetune_stf/train.py`
- `finetune_stf/config/resolved.py`
- `save_args()` 或新的 config 保存函数
- 日志初始化处

## 适用性规则

### 模型功能

CLI 规则：顶层功能开关必须显式写出具体值或 `none` / `n_a`；子参数只在对应功能开启时传入。功能关闭时，子参数不要在 CLI 中传入，resolver 在 `resolved_config.json` 中把它们记录为 `not_applicable`。不要要求 `--bridge-layers none`、`--lora-tap-layers none` 这类 list 参数写法，除非 parser 明确实现了该语法。

| 功能 | 开启时必须显式 | 关闭时要求 |
| --- | --- | --- |
| bridge | `bridge_feature_source_channels`, `bridge_feature_keys`, `bridge_layers`, `bridge_source` | `bridge=none`，不传 bridge 子参数 |
| decoder feature adaptor | `adapter_feature_source_channels`, `feature_adapter_keys` | `decoder_feature_adapter=none`，不传 adaptor 子参数 |
| LoRA | `lora_rank`, `lora_alpha`, `lora_lr`, `lora_tap_layers` | `lora=none`，不传 LoRA 子参数 |
| raw4 输入 | `raw_storage_format`, `dataset_family`, `model_input_tensor` | `raw_storage_format=none`，不传 raw 解码/norm/channel 旧组合参数 |

### loss

- 如果 `loss_type=ssi` 且未启用 grad/edge loss，则 grad/edge 子参数不在 CLI 中传入，并在 resolved config 中记录为 `not_applicable`。
- 如果 `loss_type=ssi_grad`，必须显式传入 `loss_lambda_grad` 和 `loss_grad_scales`。
- 如果未来新增 edge loss，启用时必须显式传入 type、权重、作用层级或 mask 规则。

### eval

- `eval_kitti=false` 时，不传 `kitti_eval_protocol`；resolved config 中记录为 `none` / `not_applicable`。
- `eval_kitti=true` 时，必须显式传入 `kitti_eval_protocol`。
- raw live KITTI 未实现前，`live_raw_model` 应报错而不是静默 fallback。

## dump 内容

启动时写两个文件：

- `config.json`：原始 CLI args，保留兼容
- `resolved_config.json`：只包含展开后的真实实验语义

`resolved_config.json` 至少包含：

```text
input_domain
front_end
dataset_family
dataset_input_mode
model_class
model_input_tensor
raw_storage_format
raw_storage_channel_order
raw_decompand
raw_post_decode_norm
raw_channel_count
ram_core_type
imagenet_norm_enabled
bridge_enabled
decoder_feature_adapter_enabled
bridge_feature_source_channels
adapter_feature_source_channels
bridge_feature_keys
bridge_feature_keys_source
bridge_layers
bridge_layers_source
lora_enabled
lora_tap_layers
lora_tap_layers_source
optimizer_param_groups
kitti_model_source
eval_input_domain
not_applicable
```

## 执行步骤

1. 在 `ResolvedConfig` 中保留 `source` 字段，记录每个关键变量来自：

- `explicit`
- `alias_from_input_type`
- `default_from_encoder`
- `not_applicable`

2. 在 parse 阶段区分显式传入与默认值。可以用 argparse 默认值设为 `None`，再在 resolver 中填充。

3. 添加 `validate_applicability(resolved, args)`：

- 功能开启，子参数缺失 -> 报错
- 功能关闭，子参数有效 -> 报错
- 功能关闭，resolver 将对应子参数写入 `not_applicable`
- 只有顶层功能开关使用 `none` / `n_a` 表达关闭；list / numeric 子参数默认通过“不传”表达不适用

4. 在 optimizer 构造后统计 param group：

```text
group_name / lr / trainable_param_count / trainable_tensor_count
```

写入日志和 `resolved_config.json`，避免 `bridge_lr` 这类参数在无 bridge 路径中变成 dead config。

5. 在日志开头打印一段紧凑摘要：

```text
[RESOLVED] input_domain=raw4 front_end=raw_to_base_rgb_ram3 model_input_tensor=raw
[RESOLVED] bridge=none decoder_feature_adapter=raw_feature_adapter adapter_feature_source_channels=x3
[RESOLVED] raw_storage_format=legacy_bggR_decomp16 channel_order=[B,G,G,R]->[R,Gr,Gb,B] norm=passthrough
```

## 验收标准

- 每次运行都会生成 `resolved_config.json`。
- `bridge_feature_keys_source`、`lora_tap_layers_source` 等来源可见。
- 无关参数不能以有效默认值污染实验记录。
- 关闭功能却传入子参数会启动报错。
- compile 和 `train.py --help` 通过。

## 风险点

- argparse 默认值从有效值改成 `None` 可能影响旧脚本。迁移期可以先保留旧 alias 路径，但正式脚本必须显式。
- 如果校验太早执行，可能还没有 optimizer param group 信息；param group dump 可以在模型和 optimizer 构建后补写。

## 完成记录

2026-05-22 已在新项目 `/home/caq/6666_raw/dav2_raw_0522` 执行本步骤：

- `finetune_stf/config/resolved.py` 增加来源追踪、`not_applicable` 输出、`validate_applicability(resolved, args)`，并支持顶层 `none` / `n_a` 关闭语法。
- `finetune_stf/train.py` 在 parse 阶段记录显式 CLI 参数，新增 `--lora-tap-layers`，启动日志打印 `[RESOLVED]` 摘要。
- `finetune_stf/scripts/formal/0522_run_stf_ram*_feature_adapter_bridge_from_0521_1542_queue.sh` 去掉 `loss_type=ssi` 下无效的 `--loss-mask-downsample`。
- optimizer 构造后统计 `group_name / lr / trainable_param_count / trainable_tensor_count`，回写 `resolved_config.json` 和 `config.json`。
- 已验证：`conda run -n dav3 python finetune_stf/train.py --help`、`conda run -n dav3 python -m compileall finetune_stf foundation`、resolver 正/反例快速校验。
