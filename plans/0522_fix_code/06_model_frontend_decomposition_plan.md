# 06 模型前端拆分

目标：把“模型接收什么输入域”和“raw 如何进入 DAv2”拆开。`input_domain` 只表达 wrapper 输入张量域；`front_end` 表达具体前端实现。

## 主要触点

- `finetune_stf/train.py`
- `finetune_stf/models/raw_ram.py`
- `finetune_stf/models/spatial_adapter.py`
- `foundation/engine/models` 中 raw naive wrapper

## 前端定义

| `front_end` | 输入域 | 含义 |
| --- | --- | --- |
| `dav2_rgb` | `rgb` | 普通 RGB DAv2 路径 |
| `raw_to_rgb_head` | `raw4` | 4ch raw 经过 4-to-3 head 后进入 DAv2 |
| `raw_ram4` | `raw4` | 4ch `RawRamCore` + RGB interface 后进入 DAv2 |
| `raw_to_base_rgb_ram3` | `raw4` | raw4 先转 base RGB，再进入 `RamCore3` |

`raw_ram_rgb*` 命名容易和普通 RGB baseline 混淆，新项目内部使用 `raw_to_base_rgb_ram3`。

## 执行步骤

1. 在 `build_model()` 中改为以 `resolved.front_end` 分派：

```python
if resolved.front_end == "dav2_rgb":
    ...
elif resolved.front_end == "raw_to_rgb_head":
    ...
elif resolved.front_end == "raw_ram4":
    ...
elif resolved.front_end == "raw_to_base_rgb_ram3":
    ...
```

2. 把 `build_raw_ram_depth_model(..., input_type=...)` 改成显式参数：

```python
build_raw_ram_depth_model(
    dav2_model,
    front_end="raw_ram4",
    rgb_interface_mode=...,
    raw_ram_rgb_tail=...,
)
```

3. 在模型类中保留明确属性，供 resolved dump 使用：

- `front_end`
- `ram_core_type`
- `imagenet_norm_enabled`
- `uses_base_rgb`
- `uses_clamp`
- `raw_ram_rgb_tail`

4. 对 `RawRamRgbDepthModel` 做命名层面的迁移：

- 新类名建议：`RawToBaseRgbRam3DepthModel`
- 旧类名可作为 alias 保留一段时间
- 日志中不再只写 `raw_ram_rgb`，而写 `front_end=raw_to_base_rgb_ram3`

5. 明确 BN-clean/no-ImageNet-norm：

- `front_end=raw_to_base_rgb_ram3` 时，在 resolved config 中写 `imagenet_norm_enabled=false`
- 如果仍需要 guard，错误信息应引用 front_end 而不是旧 `input_type`

## 验收标准

- 新增前端组合不需要新增 `input_type` 字符串。
- `raw_to_base_rgb_ram3` 与普通 `dav2_rgb` 在日志和 config 中清楚区分。
- 旧 alias 展开后和新前端行为一致。
- compile 通过，模型构建 smoke 能覆盖四种前端。

## 风险点

- `raw_to_base_rgb_ram3` 当前走 no-clamp/no-ImageNet-norm，不能被重构时误加回 ImageNet norm。
- `raw_to_rgb_head` 和 `raw_ram4` 都可能输出 RGB-like tensor，但实验语义不同，必须在 resolved config 中保持可见。
