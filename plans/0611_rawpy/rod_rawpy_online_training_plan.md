# 0611 ROD RawPy 在线 RGB 训练执行计划

## 目标与范围

目标：把 ROD night 现有在线 `student_dark_degreen_v1` RGB 渲染路径替换为在线 RawPy 默认 RGB 路径，并只复刻两个已有 student-RGB 训练设定。

锁定范围：

- 在线生成，不预缓存全量 PNG。
- RawPy 调用保持默认：`raw.postprocess()`，不传自定义 kwargs。
- RawPy 输入构造沿用 0611 debug preview：`raw24 -> unpack -> uint16 RGGB DNG -> rawpy.imread(DNG).postprocess()`。
- 网络仍走 RGB 图像路径：`input_domain=rgb`、`front_end=dav2_rgb`、`model_input_tensor=image`。
- 只复刻两组：
  - `0604_0752_rod_night_studentrgb_dav2s_decoder_e10`
  - `0605_0139_rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10`
- 不包含 `0605` encoder-budget 脚本里的 full-backbone C 组。

## 基线复刻定义

### A. RawPy RGB decoder-only

对应原始实验：

```text
0604_0752_rod_night_studentrgb_dav2s_decoder_e10
```

除输入 RGB renderer 从 `student_dark_degreen_v1` 换成 RawPy default 外，其余训练语义保持一致：

```text
encoder = vits
pretrained = depth_anything_v2_vits.pth
stage = rod_only
front_end = dav2_rgb
model_input_tensor = image
lora = none
dav2_train_mode = decoder
backbone_layer_decay = 1.0
epochs = 10
bs = 8
lr = 1e-5
loss = ssi + target normalization
amp = bf16
seed = 42
train crop = random 512x960
val crop = center 512x960
eval_rod = true
```

建议正式 run suffix：

```text
rod_night_rawpyrgb_dav2s_decoder_e10
```

### B. RawPy RGB LoRA tap + decoder

对应原始实验：

```text
0605_0139_rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10
```

除输入 RGB renderer 从 `student_dark_degreen_v1` 换成 RawPy default 外，其余训练语义保持一致：

```text
encoder = vits
pretrained = depth_anything_v2_vits.pth
stage = rod_only
front_end = dav2_rgb
model_input_tensor = image
lora = dav2_lora
lora_block_mode = tap
lora_tap_layers = 2 5 8 11
lora_rank = 8
lora_alpha = 16
lora_lr = 5e-5
dav2_train_mode = decoder
backbone_layer_decay = 1.0
epochs = 10
bs = 8
lr = 1e-5
loss = ssi + target normalization
amp = bf16
seed = 42
train crop = random 512x960
val crop = center 512x960
eval_rod = true
```

建议正式 run suffix：

```text
rod_night_rawpyrgb_dav2s_lora_tap_r8a16_decoder_e10
```

正式实验名仍按规范在 launch 时加当前时间戳：

```text
MMDD_HHMM_rod_night_rawpyrgb_...
```

## 显式实验语义参数

为避免与 student/teacher RGB 混淆，建议新增并在正式 launch 脚本中显式设置：

```text
dataset_family = rod_raw_rawpy_rgb
dataset_input_mode = raw24_rawpy_rgb
rod_rgb_pipeline = rawpy_default_dng_v1
rod_rawpy_dng_profile = rggb_uint16_srgb_d65_black0_white65535_v1
rod_rawpy_postprocess_profile = default_kwargs_empty_v1
```

其中：

- `rod_raw_rawpy_rgb`：表示 ROD raw24 在线转 RawPy RGB 的 dataset family。
- `raw24_rawpy_rgb`：表示输入源仍是 raw24，但模型看到的是 RawPy 生成的 RGB。
- `rawpy_default_dng_v1`：表示使用 debug preview 同款 RawPy pipeline。
- `rggb_uint16_srgb_d65_black0_white65535_v1`：表示 DNG 构造 profile 固定为 RGGB、black 0、white 65535、D65、sRGB matrix。
- `default_kwargs_empty_v1`：表示 `raw.postprocess()` kwargs 为空。

RawPy 数据集不应继续叫 `student_rgb`，也不应复用 `raw24_student_rgb`，否则后续读 config 时会误判输入域。

## 代码改动计划

### 1. 下沉 RawPy renderer

新增模块建议：

```text
finetune_stf/dataset/rod_rawpy_render.py
```

职责：

- 复用 `unpack_raw24` 的输出 shape 和归一化约定。
- 提供 DNG extratags 常量，与 `build_rod_rawpy_debug_preview.py` 当前实现一致。
- 提供：

```python
render_rawpy_default_rgb_from_raw24(raw_path: Path, *, temp_root: Path | None = None) -> np.ndarray
render_rawpy_default_rgb_from_raw_norm(raw_norm: np.ndarray, *, temp_root: Path | None = None) -> np.ndarray
```

输出约定：

```text
dtype = uint8
shape = (928, 1440, 3)
color = RawPy/LibRaw default sRGB output
```

实现细节：

- DNG 写入 full-res `(1856, 2880)` uint16 mosaic。
- RawPy `postprocess()` 得到 full-res `(1856, 2880, 3)` uint8 RGB。
- 使用和 debug preview 一致的 `cv2.INTER_AREA` resize 到 `(928, 1440, 3)`。
- 临时 DNG 使用唯一临时文件名，后缀 `.dng`，适配 DataLoader 多 worker。
- 成功或失败都要避免常规训练中堆积临时 DNG；如果需要保留失败样本，应通过 debug 参数显式开启。

注意：`rawpy` 和 `tifffile` 建议在 renderer 函数内 lazy import，避免非 RawPy 路径 import dataset 时被环境依赖影响。

### 2. 让 debug preview 复用同一 renderer

修改：

```text
finetune_stf/scripts/build_rod_rawpy_debug_preview.py
```

计划：

- 把 DNG constants、`dng_extratags()`、`write_dng()`、`rawpy_postprocess_default()`、`resize_to_native()` 切到 `rod_rawpy_render.py`。
- 保持当前输出不变。
- 用已有 preview 结果做 parity，确认重构不改变 RawPy RGB。

这样训练路径和 debug preview 不会有两份 RawPy 实现。

### 3. 新增 RawPy RGB dataset family

可选实现路径：

- 最小改动：在 `finetune_stf/dataset/rod_raw_student_rgb.py` 中新增 `RODRawRawPyRGB`，复用 `RODRawRenderedRGB` 的 manifest/crop/transform 逻辑，只替换 image render。
- 更清晰改动：新增 `finetune_stf/dataset/rod_rawpy_rgb.py`，从现有模块复用 `_load_rod_manifest_rows`、`ROD_NATIVE_HW`、crop 和 transform。

建议类名：

```python
class RODRawRawPyRGB(Dataset):
    ...
```

`build_sample()` 逻辑：

```text
raw_path -> render_rawpy_default_rgb_from_raw24(raw_path) -> float [0,1]
target_path -> np.load pseudo depth
shape check image == target == (928,1440)
same crop box applied to image and target
NormalizeImage + PrepareForNet
metadata sample["rgb_pipeline"] = "rawpy_default_dng_v1"
metadata sample["dataset_input_mode"] = "raw24_rawpy_rgb"
```

保留当前 ROD manifest、split、label、crop 逻辑，不改 pseudo label。

### 4. config/resolved.py 扩展

修改：

```text
finetune_stf/config/resolved.py
```

新增 choices：

```text
DATASET_FAMILY_CHOICES += "rod_raw_rawpy_rgb"
DATASET_INPUT_MODE_CHOICES += "raw24_rawpy_rgb"
```

新增映射，建议把 RGB image 输入类统一管理：

```python
ROD_RENDERED_RGB_INPUT_MODE_BY_FAMILY = {
    "rod_raw_student_rgb": "raw24_student_rgb",
    "rod_raw_teacher_rgb": "raw24_teacher_rgb",
    "rod_raw_rawpy_rgb": "raw24_rawpy_rgb",
}
```

或新建更准确命名：

```python
ROD_RGB_IMAGE_INPUT_MODE_BY_FAMILY = {...}
```

必须同步更新：

- `_infer_from_front_end()`：`front_end=dav2_rgb` 且 family 为 RawPy 时推断 `dataset_input_mode=raw24_rawpy_rgb`。
- `_legacy_alias_from_config()`：RawPy RGB 仍应映射到 `rgb` / `rgb_lora`，因为模型路径与普通 RGB 一致。
- `validate_resolved_config()`：RawPy family 要求：

```text
dataset_input_mode = raw24_rawpy_rgb
input_domain = rgb
front_end = dav2_rgb
model_input_tensor = image
bridge = none
decoder_feature_adapter = none
lora in {none, dav2_lora}
raw_storage_format = none / n_a
```

此外，`validate_resolved_config(resolved, args)` 当前会接收 `args`，因此 RawPy 相关 pipeline/profile 负例也要在 resolved 层做兜底校验，而不能只依赖 `train.py::parse_args()` 的 parser error。至少覆盖：

```text
dataset_family=rod_raw_rawpy_rgb 时，args.rod_rgb_pipeline 必须是 rawpy_default_dng_v1
dataset_family=rod_raw_rawpy_rgb 时，args.rod_rawpy_dng_profile 必须是 rggb_uint16_srgb_d65_black0_white65535_v1
dataset_family=rod_raw_rawpy_rgb 时，args.rod_rawpy_postprocess_profile 必须是 default_kwargs_empty_v1
dataset_family=rod_raw_student_rgb 时，args.rod_rgb_pipeline 不得是 rawpy_default_dng_v1
dataset_family=rod_raw_teacher_rgb 时，args.rod_rgb_pipeline 不得是 rawpy_default_dng_v1
```

这样即使后续有人绕过 CLI 直接构造 args/config，也不会把 RawPy family 和 student/teacher renderer 静默混用。

### 5. train.py 参数与集中校验

修改：

```text
finetune_stf/train.py
```

新增/扩展参数：

```text
--rod-rgb-pipeline choices += rawpy_default_dng_v1
--rod-rawpy-dng-profile choices n_a rggb_uint16_srgb_d65_black0_white65535_v1
--rod-rawpy-postprocess-profile choices n_a default_kwargs_empty_v1
--rod-rawpy-temp-root default /tmp/rod_rawpy_dng_tmp
```

语义建议：

- `rod_rgb_pipeline`、`rod_rawpy_dng_profile`、`rod_rawpy_postprocess_profile` 是实验语义参数，正式脚本必须显式传。
- `rod_rawpy_temp_root` 是工程参数，可有默认值，但 smoke/formal 脚本也可以显式传到清晰路径。

新增 helper：

```python
def uses_rod_rawpy_rgb_dataset(args):
    return resolved_config(args).dataset_family == "rod_raw_rawpy_rgb"
```

集中校验：

```text
dataset_family=rod_raw_rawpy_rgb 时：
  rod_rgb_pipeline 必须是 rawpy_default_dng_v1
  rod_rawpy_dng_profile 必须是 rggb_uint16_srgb_d65_black0_white65535_v1
  rod_rawpy_postprocess_profile 必须是 default_kwargs_empty_v1
  不校验 student/teacher white_percentile/gamma/channel_gains
  如果显式传入 student/teacher 渲染参数，建议报错，避免以为它们影响 RawPy

dataset_family=rod_raw_student_rgb 时：
  继续要求 student_dark_degreen_v1 + student 参数

dataset_family=rod_raw_teacher_rgb 时：
  继续要求 teacher_bright_degreen_v1 + teacher 参数
```

训练接入必须同时处理两类集合语义：

1. RawPy family 在“RGB image 输入路径”语义上属于 rendered/image RGB family，因此 train.py 中用于 gate 的集合必须包含它，例如 `uses_rod_dataset()`、`uses_rod_rendered_rgb_dataset()`、`supports_rgb_eval_inputs()` 等仍要识别 RawPy RGB。
2. 类选择和 kwargs 注入不能继续使用“非 teacher 即 student”的隐式分支；RawPy 必须有显式 dataset class 和独立 kwargs。

当前 train.py 有一个本地常量：

```python
ROD_RENDERED_RGB_DATASET_FAMILIES = {"rod_raw_student_rgb", "rod_raw_teacher_rgb"}
```

它与 `finetune_stf/config/resolved.py` 中的同名概念不是同一个对象。执行时必须分别更新两处语义，不能只改 resolved.py。

类选择必须改成显式映射，禁止三元式：

```python
ROD_RGB_DATASET_CLS_BY_FAMILY = {
    "rod_raw_student_rgb": RODRawStudentRGB,
    "rod_raw_teacher_rgb": RODRawTeacherRGB,
    "rod_raw_rawpy_rgb": RODRawRawPyRGB,
}

rod_dataset_cls = ROD_RGB_DATASET_CLS_BY_FAMILY[cfg.dataset_family]
```

不要保留下面这种逻辑：

```python
RODRawTeacherRGB if cfg.dataset_family == "rod_raw_teacher_rgb" else RODRawStudentRGB
```

否则新增 RawPy family 后会被静默当成 student RGB 渲染，实验会跑错且不报错。

kwargs 注入也必须按 family 拆分。基础公共参数只包含：

```python
rod_common = {
    "rod_root": args.rod_root,
    "manifest_path": args.rod_night_manifest,
    "size": size,
    "raw_source": args.rod_raw_source,
    "label_space": args.rod_label_space,
}
```

student/teacher family 才能追加：

```python
rgb_pipeline
student_white_percentile
student_gamma
student_channel_gains
teacher_white_percentile
teacher_gamma
teacher_channel_gains
```

RawPy family 只能追加：

```python
rgb_pipeline = "rawpy_default_dng_v1"
rawpy_dng_profile = args.rod_rawpy_dng_profile
rawpy_postprocess_profile = args.rod_rawpy_postprocess_profile
rawpy_temp_root = args.rod_rawpy_temp_root
```

RawPy dataset 不应接收 student/teacher 渲染参数；不能通过 `**kwargs` 静默吞掉这些参数。`RODRawRawPyRGB.__init__` 应使用显式参数签名，不提供泛用 `**kwargs`，让错传 student/teacher 参数在构造阶段直接失败。

- `ROD_RENDERED_RGB_DATASET_FAMILIES` 或更名后的 `ROD_RGB_IMAGE_DATASET_FAMILIES` 必须包含 RawPy family，用于 gate。
- `build_datasets()` 中 RawPy family 必须显式选择 `RODRawRawPyRGB`。
- dataset common args 传入 RawPy profiles 和 temp root。
- 日志新增 `[ROD_RAWPY_RGB]`，记录 rawpy version、LibRaw version、DNG profile、postprocess profile、temp root。

### 6. formal launch 脚本

新增一份队列脚本即可，顺序跑两组：

```text
finetune_stf/scripts/formal/0611_run_rod_night_rawpy_rgb_e10_queue.sh
```

默认行为：

- 使用 conda env `dav3`。
- 使用 tmux。
- 不复用已有 tmux session。
- 每个正式 run 名以 launch 时 `MMDD_HHMM` 开头。
- 输出 attach 和 tail 命令。
- 明确写出全部实验语义参数。

公共参数应包括：

```bash
--encoder vits
--stage rod_only
--input-domain rgb
--front-end dav2_rgb
--model-input-tensor image
--dataset-family rod_raw_rawpy_rgb
--dataset-input-mode raw24_rawpy_rgb
--raw-storage-format n_a
--bridge none
--decoder-feature-adapter none
--rod-root "${ROD_ROOT}"
--rod-night-manifest "${ROD_MANIFEST}"
--rod-raw-source raw24
--rod-rgb-pipeline rawpy_default_dng_v1
--rod-rawpy-dng-profile rggb_uint16_srgb_d65_black0_white65535_v1
--rod-rawpy-postprocess-profile default_kwargs_empty_v1
--rod-label-space inverse_relative
--input-height 512
--input-width 960
--rod-train-crop-mode random
--rod-val-crop-mode center
--no-eval-stf
--eval-rod
--best-metric rod
--save-best-checkpoint
--bs 8
--accum-steps 1
--lr 1e-5
--loss-type ssi
--loss-target-normalization
--loss-norm-min-scale 1e-3
--epochs 10
--amp
--amp-dtype bf16
--seed 42
--num-workers 4
--log-interval 500
--no-enable-fixed-viz-dump
--no-enable-train-source-viz-dump
```

A 组追加：

```bash
--lora none
--dav2-train-mode decoder
--backbone-layer-decay 1.0
```

B 组追加：

```bash
--lora dav2_lora
--lora-block-mode tap
--lora-tap-layers 2 5 8 11
--lora-rank 8
--lora-alpha 16
--lora-lr 5e-5
--dav2-train-mode decoder
--backbone-layer-decay 1.0
```

建议端口：

```text
A_MASTER_PORT=29611
B_MASTER_PORT=29612
```

## 验证计划

### 1. renderer parity

目的：证明训练 renderer 与 0611 debug preview 的 RawPy RGB 完全一致。

使用已有 preview oracle：

```text
finetune_stf/analysis/rod_rawpy_debug/0611_1253_rod_rawpy_preview_n10_dav2l_spectral_r/rawpy_default_928x1440/
```

测试逻辑：

- 读取该 preview manifest 中若干样本。
- 先断言 `rawpy.__version__` 和 `rawpy.libraw_version` 与 preview 记录一致；当前 oracle 对应 `rawpy 0.27.0`、`LibRaw (0, 22, 1)`。
- 用新 renderer 从原始 `raw_path` 在线生成 RawPy RGB。
- 与已保存 `rawpy_928x1440_rgb_path` 对比。
- 期望 byte-exact；若不完全 byte-exact，至少必须定位到 resize/interpolation 或 tifffile tag 顺序差异，不能直接放过。

输出路径必须使用 clearly temporary marker，例如：

```text
/tmp/codex_smoke_rod_rawpy_parity
```

成功后删除该 smoke 输出；失败保留并报告。

### 2. config audit

用 `parse_args()` 验证两组正式配置：

- RawPy decoder-only resolves to `input_type_alias=rgb`。
- RawPy LoRA resolves to `input_type_alias=rgb_lora`。
- `dataset_family=rod_raw_rawpy_rgb`。
- `dataset_input_mode=raw24_rawpy_rgb`。
- `front_end=dav2_rgb`。
- `model_input_tensor=image`。
- LoRA group 和 decoder group 与 `0605_0139` 匹配。

同时加负例：

- RawPy family + `rod_rgb_pipeline=student_dark_degreen_v1` 应报错。
- RawPy family + active student/teacher render 参数显式传入时应报错或被明确拒绝。
- Student family + `rawpy_default_dng_v1` 应报错。

### 3. dataset smoke

运行小样本 dataloader smoke：

```text
dataset_family=rod_raw_rawpy_rgb
split=00Train and 01Valid
num_workers=0 and num_workers=4
```

检查：

- `image` shape 为训练输入 expected CHW。
- `depth` / `valid_mask` shape 正确。
- `rgb_pipeline=rawpy_default_dng_v1`。
- 临时 DNG 不堆积。
- 多 worker 下无文件名冲突。

输出路径：

```text
/tmp/codex_smoke_rod_rawpy_dataset
```

成功删除，失败保留。

### 4. train smoke

分别跑两组极短训练：

```text
debug_max_train_steps = 2
debug_max_val_samples = 4
epochs = 1
save_path = /tmp/codex_smoke_rod_rawpy_decoder
save_path = /tmp/codex_smoke_rod_rawpy_lora
```

检查：

- 能完成 forward/backward/eval。
- `resolved_config.json` 语义正确。
- `config.json` 中记录 RawPy profiles。
- LoRA 组 trainable params 与基线量级一致。
- 日志中能看到 `[ROD_RAWPY_RGB]`。

成功后只删除上述 smoke 路径；失败保留。

## 性能与风险

### 在线 RawPy 开销

RawPy 在线路径每个样本都会：

```text
unpack raw24
write temporary DNG
rawpy.imread + postprocess
resize to native
```

这会明显慢于当前 numpy student render。正式训练先保持 `num_workers=4` 复刻基线；如果 dataloader 成为瓶颈，再单独记录为工程问题，不在第一版实验中改语义。

### 临时文件管理

必须避免使用固定 DNG 路径。建议用：

```python
tempfile.NamedTemporaryFile(prefix="rod_rawpy_", suffix=".dng", dir=temp_root, delete=False)
```

写完关闭后给 rawpy 读取，最后必须用 `try/finally` 清理：

```python
dng_path = Path(tmp.name)
try:
    ...
finally:
    dng_path.unlink(missing_ok=True)
```

不能只在成功路径删除；否则 forward / DataLoader worker 抛异常时，`/tmp` 会持续堆积 DNG。

多 worker 下 temp 文件名必须唯一。`temp_root` 可默认在 `/tmp` 下，正式脚本可显式设为：

```text
/tmp/rod_rawpy_dng_tmp
```

后续可评估内存路径以减少磁盘 I/O：

```text
tifffile.imwrite(BytesIO, ...) -> rawpy.imread(BytesIO)
```

这属于后续工程优化，不进入第一版复刻。只有在确认内存路径与磁盘路径 byte-exact 后，才能替换正式训练 renderer。

### 版本可复现

训练日志和 `config.json` 应记录：

```text
rawpy version
LibRaw version
tifffile version
DNG profile
postprocess profile
```

当前已确认环境：

```text
rawpy 0.27.0
LibRaw 0.22.1
```

### RawPy 默认参数边界

RawPy 默认结果不是 student/teacher RGB recipe：

```text
student_dark_degreen_v1: percentile/gamma/gains hand-rendered RGB
teacher_bright_degreen_v1: percentile/gamma/gains hand-rendered RGB
rawpy_default_dng_v1: LibRaw default ISP from generated DNG
```

后续报告中必须把 RawPy 作为第三种 RGB 输入域，不要归到 student RGB。

## 预期文件清单

新增：

```text
finetune_stf/dataset/rod_rawpy_render.py
finetune_stf/scripts/verify_rod_rawpy_rgb_pipeline.py
finetune_stf/scripts/formal/0611_run_rod_night_rawpy_rgb_e10_queue.sh
```

可能新增：

```text
finetune_stf/dataset/rod_rawpy_rgb.py
```

修改：

```text
finetune_stf/scripts/build_rod_rawpy_debug_preview.py
finetune_stf/config/resolved.py
finetune_stf/train.py
finetune_stf/dataset/rod_raw_student_rgb.py
```

如果采用独立 dataset 文件，则 `rod_raw_student_rgb.py` 可少改或不改。

`finetune_stf/config/__init__.py` 只有在新增需要 re-export 的常量或名字时才需要修改；如果只是扩展 `resolved.py` 中已有 `DATASET_FAMILY_CHOICES` / `DATASET_INPUT_MODE_CHOICES` 的内容，则不必改。

## 执行顺序

1. 抽出 RawPy renderer，并用已有 preview 做 parity。
2. 接入 RawPy dataset family 和 config/resolved 校验。
3. 写 dataset smoke 和 config audit。
4. 写 combined formal queue 脚本，只包含 decoder-only 和 LoRA 两组。
5. 跑 smoke；成功后清理 smoke artifacts。
6. 审核 `resolved_config.json` / `config.json` / 日志字段。
7. 确认后再启动正式 tmux 训练。
