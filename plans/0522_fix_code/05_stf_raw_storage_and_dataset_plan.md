# 05 STF raw storage 与 dataset 配置

目标：把 STF raw 的存储格式、通道顺序、decompand、norm 从脚本组合变量中收束成单一公开变量 `raw_storage_format`，避免换 raw root 时隐式改变解码语义。

## 主要触点

- `finetune_stf/dataset/stf_raw.py`
- `finetune_stf/dataset/raw_utils.py`
- `finetune_stf/train.py`
- `finetune_stf/config/resolved.py`

## 新格式定义

第一版只实现当前本地可用格式：

```text
raw_storage_format=legacy_bggR_decomp16
storage_channel_order=[B,G,G,R]
model_channel_order=[R,Gr,Gb,B]
channel_reorder=[3,1,2,0]
decompand=stf_lut_to_0_1
post_decode_norm=passthrough
raw_channel_count=4
```

预留但不实现：

```text
raw_storage_format=raw_future
```

如果用户选择 `raw_future`，启动阶段直接报错，不能 fallback。

## 执行步骤

1. 新增 raw storage spec，例如 `finetune_stf/dataset/raw_storage.py`：

```python
@dataclass(frozen=True)
class RawStorageSpec:
    name: str
    storage_channel_order: tuple[str, ...]
    model_channel_order: tuple[str, ...]
    channel_reorder: tuple[int, ...]
    decompand: str
    post_decode_norm: str
```

2. 在 `raw_utils.py` 中提供单一入口：

```python
decode_stf_raw_by_storage_format(npz_array, spec) -> np.ndarray
```

输出必须是 model channel order `[R, Gr, Gb, B]` 且数值为 `[0,1]`。

3. 修改 `STF_RAW`：

- 参数改为 `raw_storage_format`
- `stf_raw_decode_mode`、`norm_mode`、`channel_mode` 不作为正式公开组合变量
- `input_mode=raw_ram` 时输出 `sample["image"] = raw4` 和 `sample["raw"] = raw4`
- `input_mode=raw_naive` 若仍保留，应明确是 raw4 转 3ch 的兼容路径

4. 删除路径字符串推断：

- 不再根据 `raw_npz_root` 是否含 `canonical` 决定 decode mode
- 不再根据 root 名称拒绝/允许 canonical
- 如果 root 和 format 不匹配，应通过显式 manifest/spec 校验，而不是字符串猜测

5. 保留 raw-like STF native size 校验，但错误信息改成 resolved config 语义：

```text
dataset_family=stf_raw requires input_size=(512,960)
```

6. resolved config dump 中写入：

- `raw_storage_format`
- `raw_storage_channel_order`
- `raw_model_channel_order`
- `raw_decompand`
- `raw_post_decode_norm`
- `raw_channel_count`

## 验收标准

- STF raw 数据集构造只需要 `raw_storage_format` 表达 raw 解释方式。
- `legacy_bggR_decomp16` 明确执行 `[B,G,G,R] -> [R,Gr,Gb,B]`。
- 不再通过 `raw_npz_root` 字符串推断 canonical/legacy。
- `raw_storage_format=raw_future` 直接报错。
- raw4 模型路径的 sample 中 `raw` 与 `image` 来源清楚，且由 `model_input_tensor` 决定取哪个。

## 风险点

- 历史 `stf_raw_decode_mode=legacy_online_decomp16` / `canonical_decomp16` 可能有旧脚本依赖。新项目中不要静默兼容；如需复现实验，在旧项目跑。
- raw 归一化改动会直接影响指标，必须通过 small smoke 检查 decoded min/max、shape 和通道顺序。
