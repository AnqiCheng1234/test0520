# 08 train / eval / viz 输入选择与 eval protocol

目标：训练、评估、可视化都使用同一个 `model_input_tensor` 选择规则，不再各自通过 `input_type != "rgb"` 或 raw-like 字符串集合推断输入。同时明确 KITTI 指标来自哪个模型 protocol。

## 主要触点

- `finetune_stf/train.py`
- `finetune_stf/util/viz_dump.py`
- `finetune_stf/dataset/stf_raw.py`
- `finetune_stf/dataset/kitti_eval.py`
- eval loop 与 fixed viz dump 调用点

## 执行步骤

1. 新增共享输入选择函数，例如：

```python
def select_model_input(sample, resolved):
    if resolved.model_input_tensor == "raw":
        return sample["raw"]
    if resolved.model_input_tensor == "image":
        return sample["image"]
    raise ValueError(...)
```

2. 修改 `prepare_model_input()`：

- 不再判断 `input_type in RAW_MODEL_INPUT_TYPES`
- 只读 `resolved.model_input_tensor`
- 如果需要的 key 不存在，错误信息要打印 dataset family 和 sample keys

3. 修改 `viz_dump.py` 的 `_select_model_input()`：

- 接收 `resolved` 或至少接收 `model_input_tensor`
- 禁止使用 `str(input_type) != "rgb" and "raw" in sample`

4. 修改 fixed viz / train source viz 调用：

- 传入 resolved config
- viz manifest 中写 `model_input_tensor`
- 如果额外构造 RGB baseline，也写明 `baseline_model_source`

5. 修改 `build_datasets()`：

- STF dataset class 由 `resolved.dataset_family` 决定
- STF raw `input_mode` 由 `resolved.dataset_input_mode` 决定
- eval dataset 的 raw/RGB 输入域写入 resolved config

6. KITTI eval protocol：

- 当前保留 `rgb_pretrained_ref` 和 `rgb_checkpoint_decoder`
- 日志和指标 JSON 中写 `kitti_model_source`
- 在 raw live eval 没实现前，`live_raw_model` 作为枚举预留但启动报错
- 未来实现时新增明确路径，不要让 raw 实验默认落入 RGB reference protocol

## 验收标准

- train/eval/viz 都从同一个字段决定读 `sample["image"]` 还是 `sample["raw"]`。
- `rgb_lora` 或未来非纯 `"rgb"` 命名路径不会在 viz 中误读 raw。
- KITTI eval 指标文件能看出是 `rgb_pretrained_ref`、`rgb_checkpoint_decoder` 还是未来的 `live_raw_model`。
- fixed viz 的 manifest 能记录输入张量来源。

## 风险点

- 部分 eval dataset 可能只提供 `image`，不提供 `raw`。错误应在启动或第一个 batch 明确暴露。
- `train_viz_rgb_baseline` 会额外跑 RGB baseline，必须在输出中标注，避免和 live model 混淆。

## 执行记录

2026-05-22 已在 `/home/caq/6666_raw/dav2_raw_0522` 执行：

- 新增 `finetune_stf/util/model_input.py`，提供共享 `select_model_input()`，缺少 `image` / `raw` 时错误会带 dataset/source 和 sample keys。
- `finetune_stf/train.py` 的训练和 eval 输入选择改为读取 `resolved_config.model_input_tensor`，KITTI/NYU RGB protocol 显式使用 `image`。
- `finetune_stf/util/viz_dump.py` 的 fixed viz 和 train-source viz 改为同一输入选择函数，并在 manifest / metrics 中记录 `model_input_tensor`、baseline/model source。
- KITTI protocol 记录补充 `kitti_model_source`、`eval_input_domain`；`live_raw_model` 加入 CLI 枚举但继续在启动解析阶段报错。
