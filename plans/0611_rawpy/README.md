# 0611 ROD RawPy 设置核对摘要

## 结论

ROD rawpy 预览中，`rawpy.postprocess()` 使用的是 rawpy/LibRaw 默认参数；脚本没有传入自定义 postprocess 参数。

需要区分两件事：

- rawpy 后处理参数：未自定义，调用形式是 `raw.postprocess()`。
- rawpy 输入文件：不是原始 ROD `.raw` 直接输入，而是脚本先把 `.raw` unpack 成 RGGB mosaic，再写成临时 DNG，最后让 rawpy 读取这个 DNG。

## 相关生成物

当前本机可见的两个 contact sheet：

- `finetune_stf/analysis/rod_rawpy_debug/0611_0015_rod_rawpy_preview_n10/rod_rawpy_preview_contact_sheet.png`
- `finetune_stf/analysis/rod_rawpy_debug/0611_1253_rod_rawpy_preview_n10_dav2l_spectral_r/rod_rawpy_preview_contact_sheet.png`

两个目录的 `run_config.json` 均记录：

```json
"rawpy_postprocess_kwargs": {}
```

## 代码入口

脚本：

```text
finetune_stf/scripts/build_rod_rawpy_debug_preview.py
```

核心调用：

```python
def rawpy_postprocess_default(dng_path: Path) -> np.ndarray:
    with rawpy.imread(str(dng_path)) as raw:
        return raw.postprocess()
```

正式预览流程：

```text
ROD .raw
  -> unpack_raw24()
  -> float [0, 1] RGGB mosaic
  -> uint16 DNG
  -> rawpy.imread(DNG)
  -> raw.postprocess()
  -> uint8 sRGB RGB PNG
```

manifest 中也记录过直接用 rawpy 读 ROD `.raw` 会失败：

```text
Unsupported file format or not RAW file
```

因此 rawpy 实际处理的是脚本生成的 DNG，不是项目自定义 raw24 容器。

## 本次环境记录

`dav3` 环境中确认：

```text
rawpy 0.27.0
LibRaw 0.22.1
```

`rawpy.Params()` 默认关键项：

```text
use_camera_wb = False
use_auto_wb = False
user_wb = None
output_color = sRGB
output_bps = 8
no_auto_bright = False
bright = 1.0
highlight_mode = Clip
gamma = LibRaw default, observed as gamm=(0.450045..., 4.5)
demosaic_algorithm = default / LibRaw auto choice
```

## 我们写入 DNG 的关键元数据

生成 DNG 时脚本写入的主要元数据：

```text
CFA = RGGB
CFA pattern = [0, 1, 1, 2]
black_level = 0
white_level = 65535
as_shot_neutral = [1, 1, 1]
calibration_illuminant_1 = D65
color_matrix_1 = standard_xyz_to_srgb_d65
```

这部分属于“输入 DNG 构造”，不是 rawpy `postprocess()` 参数。

## 与项目 RGB recipe 的区别

rawpy 默认结果不是下面这些项目内手工 RGB pipeline：

- `student_dark_degreen_v1`
- `teacher_bright_degreen_v1`

student/teacher pipeline 会显式使用项目设定的参数，例如：

```text
channel_gains = [1.08, 0.95, 1.10]
student gamma = 0.9
teacher gamma = 0.454545
white_percentile = 99.9 / 99.5
```

rawpy 预览列没有使用这些 student/teacher 参数；它走的是 LibRaw 默认 ISP，包括 demosaic、色彩空间转换、默认亮度/高光处理、默认 gamma/tone curve，并输出 8-bit sRGB。

## 后续核对命令

```bash
grep -n "raw.postprocess" finetune_stf/scripts/build_rod_rawpy_debug_preview.py
grep -n "rawpy_postprocess_kwargs" \
  finetune_stf/analysis/rod_rawpy_debug/*/run_config.json

conda run -n dav3 python -c "import rawpy; print(rawpy.__version__, rawpy.libraw_version); print(rawpy.Params())"
```

