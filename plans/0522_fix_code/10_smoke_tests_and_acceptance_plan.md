# 10 smoke test 与验收

目标：每个重构步骤都有最小验证，避免迁移后才发现 train/eval/viz 或配置记录已经分叉。所有 smoke 输出必须是清晰临时路径，成功后只删除临时产物，失败则保留并报告。

## 通用命令

在新项目中执行：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
conda run -n dav3 python -m compileall finetune_stf foundation
conda run -n dav3 python finetune_stf/train.py --help >/tmp/dav2_raw_0522_train_help_codex_smoke.txt
```

成功后：

```bash
rm -f /tmp/dav2_raw_0522_train_help_codex_smoke.txt
```

## 分步 smoke

### 配置 resolver

检查 alias 展开：

```bash
conda run -n dav3 python - <<'PY'
from finetune_stf.config.resolved import resolve_legacy_input_type
for name in ["rgb", "rgb_lora", "raw_ram", "raw_ram_rgb_bridge_feature_adapter"]:
    print(name, resolve_legacy_input_type(name))
PY
```

验收：

- 每个旧 alias 都能展开。
- 展开结果包含 `input_domain`、`front_end`、`model_input_tensor`。

### raw storage

用极小样本或单个 NPZ 检查：

- shape 是 `[4,H,W]` 或 `[H,W,4]` 转换后的预期格式
- channel order 已从 `[B,G,G,R]` 转为 `[R,Gr,Gb,B]`
- min/max 在 `[0,1]`
- `post_decode_norm=passthrough`

输出文件如需落盘，使用：

```text
/tmp/dav2_raw_0522_raw_storage_codex_smoke/
```

成功后删除该目录。

### 模型构建

最小覆盖：

- `front_end=dav2_rgb`
- `front_end=raw_to_rgb_head`
- `front_end=raw_ram4`
- `front_end=raw_to_base_rgb_ram3`
- `front_end=raw_to_base_rgb_ram3 + decoder_feature_adapter`
- `front_end=raw_to_base_rgb_ram3 + bridge`
- `lora=dav2_lora`

验收：

- forward 能跑一个随机 tensor。
- 输出 shape 合理。
- resolved config 中 param group 计数非 0。

### train/eval/viz 输入选择

构造假 sample：

```python
sample = {
    "image": torch.zeros(1, 3, 16, 16),
    "raw": torch.ones(1, 4, 16, 16),
}
```

验收：

- `model_input_tensor=image` 返回全 0。
- `model_input_tensor=raw` 返回全 1。
- `viz_dump.py` 和 train 使用同一个选择函数。

### 训练入口 dry run

使用明确临时路径：

```text
/tmp/dav2_raw_0522_train_codex_smoke/
```

若需要真实训练 smoke，限制样本和步数，并确保：

- save path 含 `codex_smoke`
- heavy save path 含 `codex_smoke`
- 成功后删除临时输出
- 失败后保留输出并报告路径

## 正式实验验收

正式实验启动前必须检查：

- 实验名以 `MMDD_HHMM` 开头。
- 长时间任务使用 tmux。
- 日志路径明确。
- `resolved_config.json` 已生成。
- `raw_storage_format`、`model_input_tensor`、`front_end`、feature keys、LoRA tap layers 来源清楚。
- eval 指标中包含 `kitti_model_source`。

## 失败处理

失败 smoke 不删除产物。最终报告必须包含：

- 失败命令
- 临时输出目录
- log 路径
- 首个关键报错
- 是否影响后续步骤

## 验收标准

- 所有 smoke 都能在 `dav3` 环境下运行。
- 成功 smoke 的临时产物已清理。
- 失败 smoke 产物保留且路径明确。
- 新项目不再依赖隐式 `input_type` 联动来完成核心路径。

## 执行记录

2026-05-22 已在 `/home/caq/6666_raw/dav2_raw_0522` 执行：

- `bash -n` 通过：
  - `finetune_stf/scripts/formal/0522_run_stf_ram_feature_adapter_bridge_from_0521_1542_queue.sh`
  - `finetune_stf/scripts/formal/0522_run_stf_ram_rgb_feature_adapter_bridge_from_0521_1542_queue.sh`
  - `finetune_stf/scripts/smoke/0522_2054_smoke_resolved_input.sh`
- `bash finetune_stf/scripts/smoke/0522_2054_smoke_resolved_input.sh` 在 `dav3` 下通过，覆盖 compileall、train help、alias 展开、`select_model_input()` 和 `live_raw_model` 启动报错。
- 成功后已删除 `/tmp/dav2_raw_0522_train_help_codex_smoke.txt`、`/tmp/dav2_raw_0522_model_input_codex_smoke.txt`、`/tmp/dav2_raw_0522_live_raw_model_codex_smoke.txt`。
- 额外用 `train.parse_args()` 验证两个 formal 配置能解析为预期的 `raw_ram4/x4` 与 `raw_to_base_rgb_ram3/x3` resolved config；未启动训练。
