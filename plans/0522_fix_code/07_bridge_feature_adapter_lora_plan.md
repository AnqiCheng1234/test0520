# 07 bridge / decoder feature adaptor / LoRA 解耦

目标：把 bridge、decoder feature adaptor、LoRA 从 `input_type` 字符串中拆出来，支持单变量对照实验，尤其是“固定前端，只加 decoder feature adaptor”。

## 主要触点

- `finetune_stf/train.py`
- `finetune_stf/models/lora_bridge.py`
- `finetune_stf/models/raw_feature_adapter.py`
- optimizer param group 构建逻辑

## 需要支持的干净对照路径

| 实验目的 | 配置表达 |
| --- | --- |
| 只测 RGB baseline | `input_domain=rgb`, `front_end=dav2_rgb`, bridge/adaptor/LoRA 全关 |
| 只加 3ch RAM front-end | `input_domain=raw4`, `front_end=raw_to_base_rgb_ram3` |
| 只加 4ch RAM front-end | `input_domain=raw4`, `front_end=raw_ram4` |
| 只加 bridge | 固定 `front_end`，`bridge=raw_feature_bridge` |
| 只加 decoder feature adaptor | 固定 `front_end`，`decoder_feature_adapter=raw_feature_adapter`, `bridge=none` |
| bridge + feature adaptor | bridge 和 adaptor 两个开关同时显式开启 |
| 只加 LoRA | 固定其他路径，`lora=dav2_lora` |

## 执行步骤

1. 在 resolved config 中新增独立开关：

```text
bridge = none | raw_feature_bridge
decoder_feature_adapter = none | raw_feature_adapter
lora = none | dav2_lora
bridge_feature_source_channels = none | x3 | x4
adapter_feature_source_channels = none | x3 | x4
```

2. bridge feature keys 必须显式：

- `bridge_feature_source_channels=x3` 时只允许 `x3` 相关 bridge key
- `bridge_feature_source_channels=x4` 时只允许 `x4` 相关 bridge key
- 不再由 `input_type` 默认推导
- resolved dump 中记录 `bridge_feature_keys_source=explicit`

3. decoder feature adaptor keys 独立于 bridge keys：

```text
bridge_feature_keys
bridge_feature_source_channels
feature_adapter_keys
adapter_feature_source_channels
```

不要继续复用 `bridge_feature_keys` 或 bridge 的 feature source 作为 adaptor keys/source 的唯一入口。adaptor 启用时，`adapter_feature_source_channels` 只约束 `feature_adapter_keys`；bridge 启用时，`bridge_feature_source_channels` 只约束 `bridge_feature_keys`。

4. LoRA tap layers 从 `bridge_layers` 拆出：

```text
bridge_layers
lora_tap_layers
```

当 `lora_block_mode=tap` 时，`lora_tap_layers` 必须显式传入；不能再用 `DEFAULT_BRIDGE_LAYERS_BY_ENCODER` 静默推导。

5. 修改模型 builder：

- front-end builder 只负责 raw/RGB 前端
- bridge wrapper 按 `bridge` 包装
- decoder adaptor wrapper 按 `decoder_feature_adapter` 包装
- LoRA patch 按 `lora` 独立执行

6. 修改 optimizer param groups：

- `base`
- `raw_front_end`
- `bridge`
- `decoder_feature_adapter`
- `lora`
- `dav2_decoder`

每组记录 lr 和 trainable 参数量；如果某组开启但参数量为 0，启动报错。

## 验收标准

- 能表达并构造“只加 decoder feature adaptor，不带 bridge”的 3ch 和 4ch 版本。
- bridge keys、feature adaptor keys、LoRA tap layers 都有独立来源记录。
- bridge 和 decoder feature adaptor 各自的 feature source channels 独立记录；同时开启时不共享同一个隐式开关。
- 改 `encoder` 不会隐式改变 LoRA tap layers。
- `bridge_lr` 不会在无 bridge/adaptor 参数时静默无效。

## 风险点

- 当前 `raw_feature_adapter.py` 中 3ch 和 4ch feature channel 表不同，必须先校验 key/channel map，再构建 projector。
- bridge 和 adaptor 都可能使用 `x_cat` / `ffm_mid`，变量名不能继续混用，否则实验记录仍不清楚。

## 完成记录

执行时间：2026-05-22

代码位置：`/home/caq/6666_raw/dav2_raw_0522`

完成内容：

- `finetune_stf/config/resolved.py`
  - 增加 `raw_ram_lora`、`raw_ram_rgb_feature_adapter`、adapter-only LoRA、3ch bridge+adapter+LoRA 等兼容 alias。
  - bridge keys 与 feature adapter keys 独立解析；不再在 bridge+adapter 同开时把 adapter keys 回填为 bridge keys。
  - `lora_block_mode=tap` 时只接受显式 `--lora-tap-layers`，不再从 `bridge_layers` 或 encoder 默认层继承。
  - resolved dump 保留 `bridge_feature_keys_source`、`feature_adapter_keys_source`、`lora_tap_layers_source`。
- `finetune_stf/train.py`
  - model builder 改为由 resolved config 推导非 LoRA wrapper，再按 `lora` 独立 patch DAv2。
  - `args.bridge_feature_keys`、`args.feature_adapter_keys`、`args.bridge_layers`、`args.lora_tap_layers` 不再互相回填。
  - optimizer 分组改为 `base`、`raw_front_end`、`bridge`、`decoder_feature_adapter`、`lora`、`dav2_decoder`，并对已启用但 0 参数的组报错。
- `finetune_stf/models/raw_feature_adapter.py`
  - 增加 3ch `RawRamRgbFeatureAdapterDepthModel`，支持“只加 decoder feature adapter，不带 bridge”的 x3 路径。
  - bridge+adapter wrapper 支持独立 `bridge_feature_keys` 和 `feature_keys`。
- `finetune_stf/models/lora_bridge.py`
  - LoRA wrapper 接受独立 `lora_tap_layers`；缺失 tap layers 时不再回退到 bridge layers。
- `finetune_stf/models/raw_ram.py`
  - 增加 `raw_ram_lora` alias 的 raw_ram4 front-end 兼容入口。

验证：

- `conda run -n dav3 python -m compileall finetune_stf foundation` 通过。
- 参数解析 smoke 通过：
  - `adapter4`: `front_end=raw_ram4`, `decoder_feature_adapter=raw_feature_adapter`, `bridge=none`
  - `adapter3`: `front_end=raw_to_base_rgb_ram3`, `decoder_feature_adapter=raw_feature_adapter`, `bridge=none`
  - `bridge_adapter_independent_keys`: `bridge_feature_keys=["x_cat"]`, `feature_adapter_keys=["ffm_mid", "x4"]`
  - `lora_missing_tap`: 预期失败，报 `lora=dav2_lora with lora_block_mode=tap requires lora_tap_layers`
  - `lora_explicit_tap`: 显式 `lora_tap_layers=[4, 11]` 通过
- CPU 构造 smoke 通过：
  - adapter-only x4 构造为 `RawRamFeatureAdapterDepthModel`
  - adapter-only x3 构造为 `RawRamRgbFeatureAdapterDepthModel`
  - bridge+adapter 独立 keys 构造为 `RawRamBridgeFeatureAdapterDepthModel`
  - LoRA-only raw_ram4 构造为 `RawRamDepthModel`
  - optimizer summary 中对应启用组均有非零 trainable 参数量。
