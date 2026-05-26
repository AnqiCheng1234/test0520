# 新旧 unprocessing 方法差异分析

本文比较两套不同的 RGB -> RAW-like unprocessing 实现：

1. 新方法：`plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py`
   - 来自这次 RAW-Adapter-style 文档和脚本。
   - 目标是离线生成 `.npz/.json/preview`，用于先看 KITTI train/val unprocessing 效果。

2. 本机旧方法：`foundation/engine/transforms/unprocessing.py`
   - 当前工程已有的 PyTorch 在线 unprocessing transform。
   - 被 VKITTI/Hypersim/KITTI 诊断或旧训练路径调用，带有多个 preset 和真实 RAW 校准倾向。

结论先写清楚：这两套方法不是同一个方法，也不应在实验记录里混写。它们虽然都把 RGB 变成 4-channel packed Bayer-like tensor，但参数来源、CFA 语义、曝光/噪声模型、输出接口和用途都不同。

## 1. 总体定位差异

| 维度 | 新 RAW-Adapter-style 脚本 | 本机旧 UnprocessingTransform |
|---|---|---|
| 文件 | `plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py` | `foundation/engine/transforms/unprocessing.py` |
| 主要用途 | 离线生成 RAW-like 文件，检查 normal/dark/over 效果 | 在线 transform，用于数据集、训练、缓存或诊断 |
| 实现框架 | NumPy + PIL | PyTorch `nn.Module` |
| 输入 | RGB 图片目录，或外部 raw-RGB `.npy/.npz` | `torch.Tensor` RGB，shape `(3,H,W)` 或 `(N,3,H,W)` |
| 输出 | `.npz` + `.json` + optional preview PNG | tensor + metadata dict |
| 主输出 shape | `[4,H/2,W/2]` | `(4,H/2,W/2)` 或 `(N,4,H/2,W/2)` |
| 固定/随机 | 每张图用 path hash 固定随机参数；离线可复现 | 可 `randomize=True/False`；可传 `torch.Generator` |
| 主要目标域 | RAW-Adapter-style synthetic RAW-like | Brooks-style / sensor-linear / STF/ETH3D/RobotCar 校准 preset |

## 2. pipeline 顺序差异

### 新 RAW-Adapter-style 脚本

核心顺序：

```text
RGB file
-> load as sRGB float [0,1]
-> sRGB exact inverse gamma
-> optional inverse global tone
-> optional simple RGB-to-camera CCM
-> inverse white balance
-> optional normal/dark/over exposure synthesis
-> black/white level mapping
-> fixed RGGB packing
-> save npz/json/preview
```

对应代码位置：

```text
load_rgb_float
srgb_to_linear
inverse_global_tone
apply_ccm
inverse_white_balance
apply_light_synthesis
pack_rggb
```

### 本机旧方法

核心顺序：

```text
RGB tensor
-> clamp [0,1]
-> approximate gamma: pow(2.2)
-> inverse smoothstep tone
-> sampled/mean RGB->camera CCM from Brooks-style XYZ->camera library
-> safe inverse white balance with highlight protection
-> exposure_gain multiplication
-> configurable CFA packing
-> black level addition
-> optional Poisson-Gaussian noise
-> clamp [0,1]
-> return tensor + metadata
```

对应代码位置：

```text
UnprocessingTransform.forward
srgb_to_linear
inverse_smoothstep
_sample_rgb2cam
safe_invert_white_balance
pack_bayer
add_poisson_gaussian_noise
```

关键差异：新脚本更接近 RAW-Adapter 文档里描述的“RGB/InvISP raw-RGB -> inverse WB -> mosaic -> dark/over”离线生成流程。本机旧方法是工程内部的在线 Brooks-style unprocessing，并围绕真实 RAW 数据集统计做了 preset 化。

## 3. Gamma 和 tone mapping 差异

新脚本使用标准 sRGB piecewise inverse gamma：

```text
x <= 0.04045: x / 12.92
x > 0.04045: ((x + 0.055) / 1.055) ** 2.4
```

本机旧方法使用简化 gamma：

```text
x ** 2.2
```

并且旧方法默认一定执行 inverse smoothstep：

```text
0.5 - sin(asin(1 - 2*x) / 3)
```

新脚本的 inverse tone 是可选参数 `--inverse-tone`，默认不启用。也就是说，在默认配置下，新脚本的 tone 处理更保守；旧方法默认会做 tone 反变换。

## 4. CCM / camera color matrix 差异

新脚本只有两个简单选项：

```text
--ccm identity
--ccm generic_d65
```

默认建议使用 `identity`，避免在前期 KITTI 可视化检查时引入不可解释的颜色变化。

本机旧方法使用 Brooks-style 的 `RGB_TO_XYZ` 和 `XYZ_TO_CAM_LIBRARY`。当 `randomize=True` 且 `randomize_ccm=True` 时，会随机混合多个 XYZ->camera 矩阵；当 `randomize=False` 时，会使用这些矩阵的均值。部分 preset 也允许 `xyz_to_cam_override` 或关闭随机 CCM。

因此，旧方法的 camera color transform 更强，也更贴近“传感器域随机化/校准”的工程需求；新脚本更适合先做可解释的离线检查。

## 5. White balance gain 差异

新脚本默认使用 RAW-Adapter 文档口径：

```text
red_gain  ~ Uniform(1.9, 2.4)
blue_gain ~ Uniform(1.5, 1.9)
inverse WB: R /= red_gain, B /= blue_gain
```

本机旧方法的 gain 来自 preset，不同 preset 差异很大。例如：

```text
stf_legacy:
  red_gain  = 0.9358 ~ 1.3502
  blue_gain = 0.8538 ~ 1.0275
  CFA       = GBRG

eth3d_sensor_linear:
  red_gain  = 1.4736 ~ 2.2989
  blue_gain = 1.2264 ~ 2.6062
  CFA       = RGGB

robotcar_subset100_sensor_linear:
  red_gain  = 0.9762 ~ 1.0617
  blue_gain = 0.8611 ~ 0.9190
  CFA       = GBRG
```

注意：两边参数名都叫 `red_gain/blue_gain`，但取值范围来自完全不同的来源。新脚本的范围来自 RAW-Adapter inverse WB 描述；旧方法的范围来自本机历史实验或真实 RAW 校准 preset。不能只看参数名就认为可比较。

## 6. CFA pattern 和 packed channel 差异

新脚本固定 RGGB：

```text
row 0: R G R G ...
row 1: G B G B ...
packed = [R, G1, G2, B]
R  = raw_rgb[0::2, 0::2, 0]
G1 = raw_rgb[0::2, 1::2, 1]
G2 = raw_rgb[1::2, 0::2, 1]
B  = raw_rgb[1::2, 1::2, 2]
```

本机旧方法支持多种 CFA：

```text
RGGB, BGGR, GRBG, GBRG
```

但输出通道统一成 canonical order：

```text
[R, Gr, Gb, B]
```

也就是说，旧方法的第 0/1/2/3 通道总是语义上的 R/Gr/Gb/B，但这些通道在 full-resolution mosaic 里的空间 offset 取决于 `cfa_pattern`。例如 `stf_legacy` preset 是 `GBRG`，不是 RGGB。

因此：

```text
新脚本: 固定 RGGB + [R,G1,G2,B]
旧方法: 可变 CFA + canonical [R,Gr,Gb,B]
```

这会影响 preview、mosaic 还原、与真实 RAW 数据的对齐，以及后续如果要接模型时的输入语义。

## 7. 曝光与 dark/over 合成差异

新脚本显式有三个 variant：

```text
normal: light_scale = 1.0
dark:   light_scale ~ Uniform(0.05, 0.4)
over:   light_scale ~ Uniform(1.5, 2.5)
```

并用：

```text
y = clip(l*x + n)
n ~ N(0, read_noise^2 + shot_noise*l*x)
```

默认 `--noise-mean-mode zero`。也保留了 `rawadapter_text` 选项，用于严格复现论文公式文字中可能导致期望翻倍的写法，但默认不建议使用。

本机旧方法没有 `normal/dark/over` 这种离线 variant 概念。它使用 preset 里的：

```text
exposure_gain_range
shot_log_gain_range
read_noise_std_range
black_level_range
```

在线随机时，每个样本只生成一个当前采样到的 RAW-like tensor。暗光、过曝不是明确的三个输出子目录，而是由 exposure/noise preset 的随机范围隐式覆盖。

## 8. 噪声模型差异

新脚本噪声：

```text
Gaussian approximation
variance = read_noise^2 + shot_noise * signal
```

参数默认：

```text
shot_noise = 0.001
read_noise = 0.0005
```

本机旧方法噪声：

```text
Poisson shot noise + Gaussian read noise
photon_counts = image / shot_noise_scale
shot = poisson(photon_counts) * shot_noise_scale
read = randn * read_noise_std
```

其中 `shot_noise_scale = exp(shot_log_gain)`，具体范围由 preset 决定。

所以新脚本是用于可控离线效果检查的近似噪声；旧方法是更像传感器采样的 Poisson-Gaussian 在线噪声。

## 9. 黑白电平差异

新脚本：

```text
black_level default = 0.0
white_level default = 1.0
output = black + x * (white - black)
```

本机旧方法：

```text
black_level 从 preset 范围采样或取中点
没有 white_level 参数
加 black level 后 clamp 到 [0,1]
```

旧方法的黑电平是 preset 语义的一部分；新脚本的 black/white level 是离线保存时的简单线性映射。

## 10. Preview / reprocess 差异

新脚本可以保存 debug preview PNG。preview 使用非常简单的 nearest-fill demosaic，仅用于肉眼检查，不是正式 ISP。

本机旧方法提供 `reprocess()`：

```text
packed Bayer + metadata
-> subtract black level
-> undo exposure
-> demosaic
-> apply WB
-> camera->RGB CCM
-> smoothstep
-> gamma
```

旧方法的 reprocess 更完整，能用 metadata 近似回到 RGB；新脚本的 preview 只是检查用图。

## 11. 保存格式和数据流差异

新脚本保存：

```text
output_dir/
  normal/**/*.npz
  dark/**/*.npz
  over/**/*.npz
  preview/{normal,dark,over}/**/*.png
  manifest.json
  each sample json
```

`.npz` key：

```text
raw_packed
raw_mosaic  # only if --save-mosaic
```

旧方法不保存文件，直接在 dataset 或训练代码中返回：

```text
raw tensor
metadata dict
```

如果需要离线缓存，旧工程另有 `cache_vkitti2_pseudoraw.py` 等工具围绕旧 transform 做缓存。

## 12. 对 KITTI 当前检查的影响

这次只检查你给的新 RAW-Adapter-style 方法，因此 KITTI train/val 的执行应使用：

```text
plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py
```

不要使用：

```text
foundation.engine.transforms.build_unprocessing_transform_from_preset(...)
```

尤其不要把旧方法的 preset 名称写入这次新方法的实验记录，例如：

```text
stf_legacy
sensor_linear_dual
robotcar_day_night_sensor_linear_dual
```

这些属于本机旧方法，不属于 RAW-Adapter-style 脚本。

这次 KITTI 样本检查建议明确记录：

```text
method = raw_adapter_style_analytic
backend = analytic
cfa = RGGB
packed_channel_order = [R,G1,G2,B]
red_gain_range = [1.9,2.4]
blue_gain_range = [1.5,1.9]
variants = normal,dark,over
dark_light_scale = [0.05,0.4]
over_light_scale = [1.5,2.5]
noise_mean_mode = zero
ccm = identity
storage = float16
```

## 13. 最容易混淆的点

1. 两边都叫 unprocessing，但不是同一套 pipeline。
2. 两边都输出 4-channel packed Bayer，但 CFA 和 green channel 语义不一定相同。
3. 两边都有 `red_gain/blue_gain`，但范围和来源不同。
4. 新脚本显式生成 `normal/dark/over` 三份；旧方法在线随机只生成一份。
5. 新脚本是离线 NumPy 保存文件；旧方法是在线 PyTorch transform。
6. 新脚本默认 `RGGB`；旧方法常用 preset 里很多是 `GBRG`。
7. 新脚本 preview 不是正式 ISP；旧方法有 metadata-aware reprocess。

## 14. 后续如果要合并或接入时必须先确认的实验语义参数

如果未来要把这次新方法接入现有训练/评估框架，必须把下面参数显式化，不能沿用旧方法隐藏默认值：

```text
unprocessing_method = raw_adapter_style_analytic | old_brooks_preset
backend = analytic | external_npy
input_domain = rgb_srgb_to_synthetic_raw_like
cfa_pattern = RGGB
packed_channel_order = [R,G1,G2,B] 或 [R,Gr,Gb,B]
red_gain_range
blue_gain_range
ccm_mode
inverse_tone
exposure_variant
dark_light_scale_range
over_light_scale_range
noise_model
noise_mean_mode
storage_dtype
```

当前阶段只看 unprocessing 效果，因此暂时不需要改训练代码。
