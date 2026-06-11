# ROD 数据集 RawPy RGB Baseline 执行计划

## 1. 目标

当前 ROD 数据集只提供 RAW，没有官方 paired RGB。你已经通过自定义 ISP-like 方法从 ROD RAW 生成了两套 RGB：

- `studentRGB`：偏暗，用作 RGB baseline 输入。
- `teacherRGB`：更亮，DAv2-L depth 预测效果较好，用于生成 pseudo depth label。

现在需要额外生成一套 **RawPy default ISP 风格的 RGB baseline**，用于和 RAW 输入方法做对比。该 baseline 的目标不是追求视觉质量或最好 depth 结果，而是提供一条更接近已有 RAW perception 论文中 “RawPy / default ISP / sRGB baseline” 的可复现实验线。

核心原则：

> 不为性能调参，不优化视觉效果，只生成一条固定、可解释、可复现的 RawPy RGB baseline。

执行口径更新：

> 先做小样本 debug 预览，而不是直接全量生成或训练。小样本预览需要同时输出 `student_dark_degreen_v1`、`teacher_bright_degreen_v1`、`DNG + rawpy.postprocess()` 的并排 panel 和 manifest；确认 scaling、CFA、方向、亮度统计都正常后，再决定是否全量落盘与训练。

---

## 2. 总体路线

建议按三级方案执行：

```text
Level 1: 原始 .raw 能被 rawpy.imread() 直接读取
         → 直接 rawpy.postprocess()

Level 2: 原始 .raw 不能被 rawpy.imread() 直接读取
         → unpack24 → Bayer mosaic → 16-bit DNG → rawpy.postprocess()

Level 3: DNG 封装失败或 RawPy 仍无法稳定读取
         → 使用 RawPy-like software ISP，但实验命名中不能叫 RawPy default
```

对 ROD 来说，**Level 1 大概率失败**，因为 ROD 的 `.raw` 更像 headerless sensor dump，不一定是 LibRaw/RawPy 可直接识别的标准 RAW 格式。因此实际重点应放在 **Level 2：DNG 封装 + RawPy default postprocess**。

---

## 3. 实验命名建议

不建议直接叫：

```text
ROD_rgb_default_isp
```

更建议使用明确命名：

```text
ROD_rgb_rawpy_default
```

如果中间经过 DNG 封装，则更准确叫：

```text
ROD_rgb_dng_rawpy_default
```

这样可以避免把三种不同 RGB 线混在一起：

| 名称 | 含义 |
|---|---|
| `rgb_student_ours_dark` | 你当前的暗 RGB，`wp=99.9, gamma=0.9` |
| `rgb_teacher_ours_bright` | 你当前的亮 RGB，`wp=99.5, gamma=0.454545` |
| `rgb_dng_rawpy_default` | unpack 后封装 DNG，再调用 `rawpy.postprocess()` 默认参数 |
| `raw_input` | RAW / RAW4 / RAM 输入线 |

---

## 4. Step 1：测试原始 `.raw` 是否可被 RawPy 直接读取

先抽 10 张 ROD RAW 做快速测试：

```python
import rawpy
from pathlib import Path

raw_paths = list(Path("/path/to/ROD/raw").glob("*.raw"))[:10]

for p in raw_paths:
    try:
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess()
        print("OK:", p, rgb.shape, rgb.dtype)
    except Exception as e:
        print("FAIL:", p, type(e).__name__, str(e))
```

### 预期

大概率会输出 `FAIL`。如果失败，不需要花太多时间修 `rawpy.imread(.raw)`，直接进入 DNG 封装路线。

### 如果成功

如果 `.raw` 能直接读取，则直接使用：

```python
with rawpy.imread(str(raw_path)) as raw:
    rgb = raw.postprocess()
```

注意不要传额外参数，例如：

```python
raw.postprocess(use_auto_wb=True)
raw.postprocess(use_camera_wb=True)
raw.postprocess(no_auto_bright=True)
raw.postprocess(gamma=(1, 1))
```

这些都会改变 RawPy 默认 ISP 设置。

---

## 5. Step 2：用已有 unpack24 还原 Bayer mosaic

你当前已有处理逻辑：

```text
raw_norm = unpack_raw24(.raw)
RAW4 = [R, Gr, Gb, B]
baseRGB = [R, (Gr+Gb)/2, B]
```

注意：当前工程里的 `finetune_stf.dataset.rod_raw_rgb.unpack_raw24()` 已经返回归一化到 `[0, 1]` 的 `float32` Bayer plane，不能再次除以 `2^24 - 1`。如果未来改成直接读 byte/uint24，再由调用方除以 `2^24 - 1`，必须在 manifest 里写清楚。

但 RawPy/DNG 需要的是 **单通道 Bayer mosaic**，不是 `RAW4` 或 `baseRGB`。

如果 CFA pattern 是 RGGB，则还原方式为：

```python
mosaic[0::2, 0::2] = R
mosaic[0::2, 1::2] = Gr
mosaic[1::2, 0::2] = Gb
mosaic[1::2, 1::2] = B
```

需要确认以下几点：

1. Bayer pattern 是否确实是 RGGB。
2. 图像方向是否正确。
3. 是否需要 crop 边界。
4. RAW4 的四个通道是否和原始 mosaic 空间位置严格对应。

---

## 6. Step 3：24-bit RAW 线性归一化并量化到 16-bit

ROD RAW 是 24-bit HDR RAW。为了 DNG / RawPy 兼容，建议先线性压到 16-bit：

```python
mosaic24_norm = unpack_raw24(raw_path)  # already normalized to [0, 1] in this repo
mosaic16 = np.round(np.clip(mosaic24_norm, 0, 1) * 65535).astype(np.uint16)
```

注意：这一步会损失一部分 24-bit 动态范围，但这条线只用于 RGB baseline，不用于最大化 RAW 方法性能，因此可以接受。

实验记录中建议明确写：

```text
为了兼容 DNG / RawPy，我们将 ROD 的 24-bit Bayer RAW 线性归一化到 [0, 1] 后量化为 16-bit，再封装为 DNG。
```

---

## 7. Step 4：封装 DNG

DNG 封装时，metadata 应固定，不要根据图像质量手动调参。

建议固定字段如下：

| 字段 | 建议设置 |
|---|---|
| CFA pattern | RGGB，除非确认不是 |
| Bit depth | 16-bit |
| Black level | 0 |
| White level | 65535 |
| Image size | 原始 Bayer mosaic 尺寸 |
| Color matrix | 使用固定 generic matrix，不针对结果调优 |
| White balance | 不手动调优；如果必须填，则使用固定值 |
| Orientation | normal |

关键原则：

> DNG metadata 只用于让 RawPy/LibRaw 能读取数据，不用于优化 RGB baseline 的视觉效果或任务性能。

---

## 8. Step 5：调用 RawPy 默认 postprocess

DNG 生成后，调用：

```python
import rawpy
import imageio.v3 as iio

with rawpy.imread(str(dng_path)) as raw:
    rgb = raw.postprocess()

iio.imwrite(str(out_png_path), rgb)
```

这里必须保持：

```python
raw.postprocess()
```

不要传额外参数。否则就不是 RawPy default baseline。

RawPy 默认 postprocess 通常包括：

| 环节 | 默认行为 |
|---|---|
| Demosaic | AHD |
| Output color | sRGB |
| Output bit depth | 8-bit |
| Auto bright | 开启 |
| Gamma | 默认 Rec.709 风格 gamma |
| Denoise | 默认关闭 |

注意：RawPy 默认参数是固定 postprocess 参数和 DNG/RAW metadata 共同作用的结果。因此你的 DNG metadata 必须记录清楚。

---

## 9. Step 6：输出尺寸对齐

你的当前 `baseRGB = [R, (Gr+Gb)/2, B]` 本质上是每个 2×2 Bayer block 生成一个 RGB pixel，因此通常是半分辨率。

RawPy demosaic 输出通常是 full-resolution RGB。

为了保证训练对比公平，建议：

```text
RawPy full-resolution RGB
→ resize 到当前 studentRGB / teacherRGB 相同尺寸
→ 保存为 rgb_dng_rawpy_default
```

必须检查：

```text
rawpy_rgb.shape[:2] == studentRGB.shape[:2] == teacherRGB.shape[:2] == pseudo_depth.shape[:2]
```

如果尺寸不一致，会引入额外变量，影响实验解释。

---

## 10. Step 7：小样本质量检查

先不要全量生成。第一轮只抽 10–20 张样本做 debug preview；如果亮度、方向、CFA 和尺寸都正常，再扩大到 50–100 张做稳定性检查。

建议第一轮输出：

```text
student_dark_degreen_v1 RGB
teacher_bright_degreen_v1 RGB
DNG + rawpy.postprocess() full-resolution RGB
DNG + rawpy.postprocess() resized-to-928x1440 RGB
per-sample panel
contact sheet
manifest_rawpy_preview.jsonl
run_config.json
```

### 10.1 检查图像是否全黑或全白

统计：

```python
import numpy as np

print("mean:", rgb.mean())
print("std:", rgb.std())
print("percentiles:", np.percentile(rgb, [0, 1, 50, 99, 99.9, 100]))
```

如果大量图像满足以下情况，则说明 scaling 或 DNG metadata 有问题：

```text
mean < 2      → 可能整体过黑
mean > 250    → 可能整体过曝
std 很小      → 可能几乎无有效图像内容
```

### 10.2 检查 CFA pattern 是否错误

可视化几张图，重点看：

1. 是否严重偏紫、偏绿。
2. 是否有棋盘格。
3. 边缘是否明显错位。
4. 图像是否上下或左右翻转。
5. 车辆、道路、天空是否有基本可识别结构。

如果出现明显通道错乱，优先检查 CFA pattern：

```text
RGGB / BGGR / GRBG / GBRG
```

### 10.3 检查和现有数据是否对齐

确认文件命名和 split 完全一致：

```text
raw file:              xxx.raw
studentRGB:            xxx.png
teacherRGB:            xxx.png
rawpy_default_rgb:     xxx.png
pseudo_depth_teacher:  xxx.npy / xxx.png
```

---

## 11. Step 8：全量生成并保存 manifest

全量生成时，建议额外保存一个：

```text
manifest_rawpy_default.jsonl
```

每一行记录一张图的信息：

```json
{
  "raw_path": "...",
  "dng_path": "...",
  "rgb_path": "...",
  "raw_shape": [1856, 2880],
  "output_shape": [928, 1440, 3],
  "raw_quantization": "24bit_norm_to_uint16",
  "cfa": "RGGB",
  "black_level": 0,
  "white_level": 65535,
  "rawpy_version": "...",
  "libraw_version": "...",
  "postprocess_kwargs": {},
  "rgb_mean": 0.0,
  "rgb_std": 0.0,
  "rgb_p99": 0.0
}
```

这个 manifest 很重要。后续写论文、报告或答辩时，可以直接说明 baseline 生成过程没有针对任务调优。

---

## 12. 推荐数据目录结构

建议组织成：

```text
ROD_processed/
  raw/
    xxx.raw

  rgb_student_ours_dark/
    xxx.png

  rgb_teacher_ours_bright/
    xxx.png

  rgb_dng_rawpy_default/
    xxx.png

  pseudo_depth_teacher/
    xxx.npy

  manifests/
    manifest_rawpy_default.jsonl
```

训练时保持 pseudo label 固定：

```text
pseudo label = teacherRGB depth
student input = rgb_student_ours_dark / rgb_dng_rawpy_default / raw_input
```

也就是说，RawPy RGB baseline 只替换输入，不重新生成 teacher label。这样变量最干净。

---

## 13. 训练实验矩阵

建议至少跑以下三组：

| 实验名 | 输入 | pseudo label | 目的 |
|---|---|---|---|
| `RGB_ours_dark` | 当前 studentRGB，`wp=99.9, gamma=0.9` | teacherRGB depth | 当前 RGB baseline |
| `RGB_rawpy_default` | DNG + RawPy default RGB | teacherRGB depth | 新增 RawPy RGB baseline |
| `RAW_model` | RAW / RAW4 / RAM 输入 | teacherRGB depth | 你的 RAW 方法 |

所有训练变量必须保持一致：

```text
same train/val split
same pseudo label
same DAv2 backbone
same image size
same optimizer
same learning rate
same schedule
same epochs
same seed
same augmentation
```

只替换输入图像来源。

---

## 14. 结果解释方式

如果结果为：

```text
RAW_model > RGB_ours_dark
RAW_model > RGB_rawpy_default
```

可以支持你的 claim：

> 在 ROD 这种只提供 RAW 的场景下，即使使用 RawPy default 风格的 RGB baseline，RAW 输入方法仍然更有效。

如果 `RGB_rawpy_default` 结果比 `RGB_ours_dark` 更差，也可以解释为：

> 默认 ISP 风格的 RGB 并不一定适合下游 depth perception，尤其在 ROD 这种 24-bit HDR RAW 场景中，简单转 sRGB 可能会损失对任务有用的 RAW 信息。

如果 `RGB_rawpy_default` 比 `RGB_ours_dark` 更好，也不矛盾：

> 说明 DAv2 backbone 对更接近自然 sRGB 分布的输入更友好，但只要 RAW 输入仍然更好，claim 仍然成立。

---

## 15. 报告中建议写法

中文表述可写成：

```text
由于 ROD 数据集只提供 24-bit RAW sensor data，不提供对应 RGB 图像，我们额外构建了一条 RawPy default 风格的 RGB baseline。具体而言，我们首先使用与 RAW 分支一致的 unpack 逻辑读取 ROD RAW，将 24-bit Bayer RAW 线性归一化到 [0, 1] 并量化为 16-bit，然后以固定 CFA pattern、black level 和 white level metadata 封装为 DNG。随后调用 rawpy.postprocess() 默认参数生成 RGB 图像，并将输出 resize 到与现有 studentRGB / teacherRGB 相同的分辨率。该流程不针对视觉质量或下游性能进行额外调参，仅用于构造可复现的默认 ISP 风格 RGB baseline。
```

如果需要更严谨，可以补一句：

```text
由于 ROD 原始 .raw 文件不是标准相机 RAW/DNG 格式，RawPy 无法直接读取，因此该 baseline 更准确地记为 DNG-wrapped RawPy default RGB baseline。
```

---

## 16. 风险点与处理建议

| 风险 | 表现 | 处理 |
|---|---|---|
| `.raw` 不能被 RawPy 直接读取 | `rawpy.imread()` 报错 | 走 DNG 封装路线 |
| CFA pattern 错误 | 图像严重偏色、棋盘格 | 尝试 RGGB / BGGR / GRBG / GBRG，固定正确 pattern |
| white level 不合适 | 图像全黑或全白 | 检查 24-bit normalize 和 16-bit quantization |
| DNG metadata 不完整 | RawPy 读取失败或输出异常 | 补齐必要 DNG tags |
| 输出尺寸不一致 | depth label 对不上 | preprocessing 阶段统一 resize |
| baseline 被质疑调参 | 结果解释不干净 | 全部参数固定，保存 manifest |

---

## 17. 最终执行清单

按顺序完成：

```text
[ ] 1. 抽 10 张 ROD .raw 测试 rawpy.imread() 是否可直接读取
[ ] 2. 如果失败，使用已有 unpack_raw24 得到已归一化 `[0,1]` 的单通道 Bayer mosaic
[ ] 3. 直接用 full-resolution Bayer mosaic 封装 DNG；只在需要 RAW4 对齐检查时才 pack/unpack RAW4
[ ] 4. 确认 CFA pattern 和图像方向
[ ] 5. 将 `[0,1]` Bayer mosaic 量化为 uint16，不重复除以 `2^24 - 1`
[ ] 6. 封装为 DNG，固定 metadata
[ ] 7. 调用 rawpy.postprocess()，不传额外参数
[ ] 8. resize 到 studentRGB / teacherRGB 相同尺寸
[ ] 9. 先生成 10–20 张 debug preview panel 和 manifest
[ ] 10. 预览通过后，抽 50–100 张做稳定性视觉和统计检查
[ ] 11. 全量生成 rgb_dng_rawpy_default
[ ] 12. 保存 manifest_rawpy_default.jsonl
[ ] 13. 使用相同 pseudo label 跑 RGB_rawpy_default baseline
[ ] 14. 与 RGB_ours_dark 和 RAW_model 对比
```

---

## 18. 最重要的结论

这条 RawPy baseline 的实验意义不是让 RGB 结果更好，而是让对比更完整：

```text
RAW_model
vs
你当前的暗 pseudo-RGB baseline
vs
RawPy default 风格的 RGB baseline
```

这样最终 claim 会更稳：

> RAW 输入方法不仅优于手工生成的暗 RGB baseline，也优于通过默认 ISP 风格处理得到的 RGB baseline，说明 RAW 中保留的原始信息对下游 depth perception 更有价值。
