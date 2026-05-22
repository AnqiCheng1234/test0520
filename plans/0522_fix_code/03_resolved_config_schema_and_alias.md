# 03 建立 resolved config schema 与 input_type alias

目标：把当前 `input_type` 中混在一起的职责拆成正交字段。新项目内部不再用 `input_type` 字符串推断数据集、模型前端、bridge、feature adaptor、LoRA 或 eval 行为。

## 建议新增模块

优先新增：

- `finetune_stf/config/__init__.py`
- `finetune_stf/config/resolved.py`

`resolved.py` 负责：

- 定义正交配置 dataclass
- 从 CLI args 构建 resolved config
- 兼容旧 `--input-type` alias
- 执行字段间一致性校验
- 导出可写入 `resolved_config.json` 的 dict

## 核心字段

建议第一版至少包含：

```text
input_domain = rgb | raw4
front_end = dav2_rgb | raw_to_rgb_head | raw_ram4 | raw_to_base_rgb_ram3
dataset_family = stf_rgb | stf_raw
dataset_input_mode = rgb | raw_naive | raw_ram
model_input_tensor = image | raw
bridge = none | raw_feature_bridge
decoder_feature_adapter = none | raw_feature_adapter
lora = none | dav2_lora
bridge_feature_source_channels = none | x3 | x4
adapter_feature_source_channels = none | x3 | x4
feature_adapter_keys = none | x_cat,ffm_mid,x3 | x_cat,ffm_mid,x4
bridge_feature_keys = none | x_cat,ffm_mid,x3 | x_cat,ffm_mid,x4
bridge_layers = none | csv-int-list
lora_tap_layers = none | csv-int-list
raw_storage_format = none | legacy_bggR_decomp16 | raw_future
kitti_eval_protocol = none | rgb_pretrained_ref | rgb_checkpoint_decoder | live_raw_model
```

## alias 展开规则

为了迁移平滑，旧 `input_type` 可以先保留为 alias，但只在 parse 阶段展开。展开后所有内部逻辑只读 resolved config。

示例：

| 旧 `input_type` | 展开后含义 |
| --- | --- |
| `rgb` | `input_domain=rgb`, `front_end=dav2_rgb`, `dataset_family=stf_rgb`, `dataset_input_mode=rgb`, `model_input_tensor=image` |
| `rgb_lora` | `rgb` + `lora=dav2_lora` |
| `raw` | `input_domain=rgb`, `front_end=dav2_rgb`, `dataset_family=stf_raw`, `dataset_input_mode=raw_naive`, `model_input_tensor=image` |
| `raw_packed` | `input_domain=raw4`, `front_end=raw_to_rgb_head`, `dataset_family=stf_raw`, `dataset_input_mode=raw_ram`, `model_input_tensor=raw` |
| `raw_ram` | `input_domain=raw4`, `front_end=raw_ram4`, `dataset_family=stf_raw`, `dataset_input_mode=raw_ram`, `model_input_tensor=raw` |
| `raw_ram_rgb` | `input_domain=raw4`, `front_end=raw_to_base_rgb_ram3`, `dataset_family=stf_raw`, `dataset_input_mode=raw_ram`, `model_input_tensor=raw` |
| `raw_ram_bridge` | `raw_ram4` + `bridge=raw_feature_bridge`, `bridge_feature_source_channels=x4` |
| `raw_ram_rgb_bridge` | `raw_to_base_rgb_ram3` + `bridge=raw_feature_bridge`, `bridge_feature_source_channels=x3` |
| `raw_ram_feature_adapter` | `raw_ram4` + `decoder_feature_adapter=raw_feature_adapter`, `adapter_feature_source_channels=x4` |
| `raw_ram_bridge_feature_adapter` | `raw_ram4` + bridge + decoder feature adaptor, `bridge_feature_source_channels=x4`, `adapter_feature_source_channels=x4` |
| `raw_ram_rgb_bridge_feature_adapter` | `raw_to_base_rgb_ram3` + bridge + decoder feature adaptor, `bridge_feature_source_channels=x3`, `adapter_feature_source_channels=x3` |

## 执行步骤

1. 新增 resolved config dataclass，例如：

```python
@dataclass(frozen=True)
class ResolvedConfig:
    input_domain: str
    front_end: str
    dataset_family: str
    dataset_input_mode: str
    model_input_tensor: str
    bridge: str
    decoder_feature_adapter: str
    lora: str
    bridge_feature_source_channels: str
    adapter_feature_source_channels: str
    raw_storage_format: str
    kitti_eval_protocol: str
```

2. 在 `parse_args()` 结束后调用：

```python
resolved = resolve_config_from_args(args)
validate_resolved_config(resolved, args)
args.resolved_config = resolved
```

3. 逐步替换内部判断：

- `args.input_type in RAW_MODEL_INPUT_TYPES` -> `resolved.input_domain == "raw4"` 或 `resolved.model_input_tensor == "raw"`
- `args.input_type in RGB_INPUT_TYPES` -> `resolved.dataset_family == "stf_rgb"` 或 `resolved.front_end == "dav2_rgb"`
- `resolve_stf_raw_input_mode(args.input_type)` -> `resolved.dataset_input_mode`
- bridge/adaptor/LoRA 判断 -> 对应 resolved 字段

4. 若用户同时传旧 alias 和新正交字段：

- 完全一致时允许，但日志标记 alias 已展开
- 有冲突时启动报错

5. 新正式脚本禁止依赖旧 alias。旧 alias 只为迁移期脚本和 checkpoint 兼容服务。

## 验收标准

- `train.py` 中不再需要新增 `input_type` 字符串组合才能表达新实验路径。
- 能构造“固定 front_end，只打开 decoder_feature_adapter”的配置。
- 旧 `--input-type raw_ram_rgb_bridge_feature_adapter` 能展开成明确 resolved config。
- `dataset_input_mode` 对 RGB、raw naive 兼容路径、raw4/RAM 路径都有明确值，并能被 `build_datasets()` 直接使用。
- resolved config 可序列化为 JSON。

## 风险点

- 一次性替换所有 `input_type` 判断风险较高。建议先实现 resolver 和 dump，再分批替换 build_datasets、build_model、input selection。
- alias 展开期间不能改变旧脚本行为；行为变化必须通过 resolved dump 明确展示。

## 完成记录

执行时间：2026-05-22

完成内容：

- 在新项目 `/home/caq/6666_raw/dav2_raw_0522` 新增 `finetune_stf/config/__init__.py` 与 `finetune_stf/config/resolved.py`。
- 新增 `ResolvedConfig` dataclass，覆盖 input/domain/front-end/dataset/model-input/bridge/decoder feature adapter/LoRA/feature source/raw storage/KITTI protocol 等正交字段。
- 将旧 `--input-type` 改为兼容 alias；未传 alias 时仍默认展开为 `rgb`，传入正交字段时可反推当前训练代码支持的 legacy factory alias。
- `train.py` 已在 `parse_args()` 中解析并校验 resolved config，冲突的 alias 与正交字段会启动时报错。
- `build_datasets()`、`build_model()`、`prepare_model_input()`、optimizer 分组、RAW native size guard、bridge-init 判断和核心 setup 日志已改为读取 resolved config。
- `save_args()` 现在同时写出 `config.json` 中的 `resolved_config` 字段和独立 `resolved_config.json`。

验证结果：

- `conda run -n dav3 python -m compileall finetune_stf foundation` 通过。
- 解析 smoke 覆盖 `rgb`、`rgb_lora`、`raw`、`raw_packed`、`raw_ram`、`raw_ram_rgb`、`raw_ram_bridge`、`raw_ram_rgb_bridge`、`raw_ram_feature_adapter`、`raw_ram_bridge_feature_adapter`、`raw_ram_rgb_bridge_feature_adapter`。
- 旧 alias `--input-type raw_ram_rgb_bridge_feature_adapter` 可展开为 `front_end=raw_to_base_rgb_ram3`、`bridge=raw_feature_bridge`、`decoder_feature_adapter=raw_feature_adapter`、`model_input_tensor=raw`。
- 正交字段可构造 adapter-only 路径：`--front-end raw_ram4 --decoder-feature-adapter raw_feature_adapter --adapter-feature-source-channels x4`，解析为 `input_type_alias=raw_ram_feature_adapter`。
- 冲突 smoke `--input-type rgb --front-end raw_ram4` 按预期启动报错。
- feature source 与 keys 不一致的 smoke `--input-type raw_ram_feature_adapter --bridge-feature-keys x3` 按预期启动报错。
- `save_args()` smoke 成功写出 `config.json` 与 `resolved_config.json`，临时目录 `/tmp/codex_smoke_resolved_save` 已清理。
