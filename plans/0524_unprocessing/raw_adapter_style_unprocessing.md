# RAW-Adapter-style unprocessing: 从 RGB 生成 packed Bayer RAW-like 数据

## 0. 目的

本文档说明如何参考 RAW-Adapter 对 ADE20K-RAW 的处理方式，把已有 RGB/sRGB 图像转换为可用于训练的 RAW-like 表示。这里的 **unprocessing** 指从已有 RGB 图像反推或近似生成线性 RAW-like 数据的过程。

本文只覆盖 RGB 到 RAW-like 的数据生成，不讨论 depth 网络、损失函数、下游模型结构或训练策略。

目标输出格式固定为：

```text
4-channel packed Bayer, RGGB, channel order = [R, G1, G2, B]
shape = [4, H/2, W/2]
value range = usually [0, 1]
main file = .npz or .npy
metadata = .json
debug preview = .png, optional
```

## 1. 来源依据和复现口径

RAW-Adapter 官方 mmsegmentation README 对 ADE20K-RAW 的描述是：先采用 InverseISP 将 RGB 转换为 raw-RGB data，然后应用 inverse white balance 和 mosaic。论文实验部分也写明，ADE20K 的 sRGB 图像通过 InvISP 投影到 RAW format，再使用 PASCAL RAW 的 dark/over-exp 合成方式生成低光和过曝版本。

官方仓库里没有找到完整的 ADE20K sRGB 到 RAW 的批处理脚本。可以直接复用的部分主要是：

1. 数据生成的文字描述：`RGB -> InvISP raw-RGB -> inverse WB -> mosaic`。
2. `inverse_wb.py` 中的 white balance gain 采样范围，默认 red gain 大约为 `[1.9, 2.4]`，blue gain 大约为 `[1.5, 1.9]`。
3. InvISP 官方仓库和论文中提供的 RGB 到 RAW 反向恢复思想。
4. RAW-Adapter 论文中低光和过曝 RAW 合成的曝光范围。

因此本文采用两个复现层级：

```text
A. 严格贴近 RAW-Adapter 描述的路线
RGB/sRGB -> InvISP/InverseISP -> raw-RGB -> inverse WB -> RGGB packed Bayer -> optional dark/over exposure

B. 在没有官方 ADE20K 脚本时的工程可运行路线
RGB/sRGB -> inverse sRGB gamma -> optional inverse tone -> optional RGB-to-camera CCM -> inverse WB -> RGGB packed Bayer -> optional dark/over exposure
```

A 是更接近论文描述的路线，但需要你先跑通 InvISP 代码。B 是本文附带脚本默认使用的可运行近似版，适合前期观察、debug、ablation 和在线训练中的数据增强原型。

## 2. 推荐目录结构

离线小样本检查时，不建议一开始把全量 NYUv2、KITTI、Hypersim、Virtual KITTI 都预处理保存。建议先每个数据集抽 50 到 500 张，检查数值范围、preview、低光/过曝效果和网络输入维度。

```text
project_root/
  data_rgb/
    nyuv2/
      images/...
    kitti/
      images/...
    hypersim/
      images/...
    vkitti/
      images/...
  raw_like_debug/
    normal/
      .../*.npz
      .../*.json
    dark/
      .../*.npz
      .../*.json
    over/
      .../*.npz
      .../*.json
    preview/
      normal/.../*.png
      dark/.../*.png
      over/.../*.png
    manifest.json
  scripts/
    unprocess_rgb_to_packed_raw.py
```

每个 `.npz` 默认至少包含：

```text
raw_packed: np.ndarray, shape [4, H/2, W/2], channel order [R, G1, G2, B]
```

如果启用 `--save-mosaic`，还会保存：

```text
raw_mosaic: np.ndarray, shape [H, W], RGGB single-channel mosaic
```

`json` 里记录输入路径、gain、曝光系数、噪声参数、随机种子和 packed shape。在线训练时也建议把这些参数写进 experiment log，至少保证复现实验时能固定随机性。

## 3. 核心 pipeline

### 3.1 输入假设

常规 depth 数据集中的 RGB 图像通常可以按 8-bit sRGB 处理，例如 NYUv2、KITTI、Hypersim、Virtual KITTI。虚拟数据集有时来自线性渲染或经过 tone mapping 的 RGB。如果不能确认，工程上先按 sRGB 处理，并在文档里记录这个假设。

输入图像统一记为：

```text
I_rgb: H x W x 3, uint8 or float, sRGB domain
```

脚本会转换为：

```text
I_rgb_float in [0, 1]
```

### 3.2 RGB 到 raw-RGB

严格的 RAW-Adapter-style 处理应该优先使用 InvISP：

```text
I_rgb -> InvISP reverse pass -> I_raw_rgb
```

InvISP 的核心思想是把传统 ISP 设计成可逆流程。正向从 Bayer RAW 渲染到 sRGB，反向从压缩 RGB 恢复 RAW。实际使用时要注意：InvISP 官方 README 明确提示，一个训练好的模型只适用于特定 camera setting。因此，如果你没有和目标数据集相匹配的相机 RAW/RGB 对，InvISP 输出更适合作为 RAW-like 合成数据，而不是物理真实 RAW。

本文脚本提供两个 backend：

```text
backend = analytic
  用可运行的解析近似替代 InvISP。

backend = external_npy
  读取外部 InvISP 或其他方法已经输出的 raw-RGB .npy/.npz。
```

解析近似 backend 的默认处理是：

```text
sRGB -> linear RGB -> optional inverse tone -> optional RGB-to-camera CCM -> raw-RGB-like
```

建议前期使用 `--ccm identity`，避免引入不可解释的颜色变换。需要更像 camera color space 时，再尝试 `--ccm generic_d65` 或使用真实相机标定 CCM。

### 3.3 Inverse white balance

RAW-Adapter 的 README 写明在 InvISP raw-RGB 后使用 inverse white balance。官方 `inverse_wb.py` 中 red gain 和 blue gain 的范围大致为：

```text
red_gain  ~ Uniform(1.9, 2.4)
blue_gain ~ Uniform(1.5, 1.9)
green_gain = 1.0
```

inverse WB 的实现是：

```text
R_raw = R / red_gain
G_raw = G
B_raw = B / blue_gain
```

注意这里的 red/blue gain 是正向 white balance 中会乘上的 gain。unprocessing 要做反向处理，所以除以 gain。

### 3.4 RGGB mosaic 和 packed Bayer

本文最终保存 4-channel packed Bayer，不保存 3-channel raw-RGB 作为主输出。

RGGB pattern 定义如下：

```text
row 0: R G R G ...
row 1: G B G B ...
row 2: R G R G ...
row 3: G B G B ...
```

从 raw-RGB 到 packed Bayer 的映射为：

```python
R  = raw_rgb[0::2, 0::2, 0]
G1 = raw_rgb[0::2, 1::2, 1]
G2 = raw_rgb[1::2, 0::2, 1]
B  = raw_rgb[1::2, 1::2, 2]
raw_packed = stack([R, G1, G2, B], axis=0)
```

如果 H 或 W 是奇数，最简单做法是裁掉最后一行或最后一列，让图像尺寸变成偶数。在线训练时，crop 的起点也最好保持偶数，否则 Bayer grid 会错位。

## 4. 低光和过曝合成

RAW-Adapter 论文对 PASCAL RAW dark/over-exp 合成给出的设置是：

```text
light scale l ~ Uniform(0.05, 0.4)   for dark
light scale l ~ Uniform(1.5, 2.5)    for over-exposure
```

论文公式写成：

```text
x_n ~ N(mu = l*x, sigma^2 = delta_r^2 + delta_s*l*x)
y = l*x + x_n
```

其中 `x` 是原始 normal-light RAW，`y` 是退化后的 RAW，`delta_r` 表示 read noise，`delta_s` 和 signal-dependent shot noise 有关。

工程实现时需要小心：如果完全照公式文本使用 `mu = l*x` 且 `y = l*x + x_n`，输出的期望会变成 `2*l*x`。这可能是论文排版或符号表达上的歧义。附带脚本默认使用更常见的物理噪声形式：

```text
y = clip(l*x + n)
n ~ N(0, read_noise^2 + shot_noise*l*x)
```

如果需要严格按照论文打印公式做 text-level reproduction，可以在脚本里使用：

```text
--noise-mean-mode rawadapter_text
```

建议默认使用：

```text
--noise-mean-mode zero
```

低光和过曝应在 RAW-like 线性域处理，而不是在 sRGB gamma 域处理。

## 5. 脚本使用

脚本文件：`unprocess_rgb_to_packed_raw.py`

安装依赖：

```bash
pip install numpy pillow tqdm
```

### 5.1 用解析近似 backend 生成小样本

```bash
python unprocess_rgb_to_packed_raw.py --input-dir /path/to/rgb_images --output-dir /path/to/raw_like_debug --backend analytic --variants normal,dark,over --max-images 100 --save-preview --save-mosaic --storage float16
```

输出包括：

```text
/path/to/raw_like_debug/normal/**/*.npz
/path/to/raw_like_debug/dark/**/*.npz
/path/to/raw_like_debug/over/**/*.npz
/path/to/raw_like_debug/preview/**/*.png
/path/to/raw_like_debug/manifest.json
```

### 5.2 检查一个输出文件

```bash
python -c "import numpy as np; d=np.load('/path/to/raw_like_debug/normal/sample.npz'); x=d['raw_packed']; print(x.shape, x.dtype, float(x.min()), float(x.max()))"
```

期望看到：

```text
(4, H/2, W/2), float16 or float32 or uint16, min/max usually within [0, 1]
```

### 5.3 接入外部 InvISP 输出

先用 InvISP 或你自己的 learned unprocessing 模型把 RGB 反推成 raw-RGB，并保存为 `.npy` 或 `.npz`。推荐形状：

```text
H x W x 3, float32, range [0, 1]
```

然后用本文脚本继续做 inverse WB、低光/过曝和 packed Bayer：

```bash
python unprocess_rgb_to_packed_raw.py --input-dir /path/to/rgb_images --external-raw-dir /path/to/invisp_raw_rgb --output-dir /path/to/raw_like_debug --backend external_npy --variants normal,dark,over --max-images 100 --save-preview --save-mosaic --storage float16
```

`external_npy` 会根据输入 RGB 的相对路径寻找同名 `.npy` 或 `.npz`。例如：

```text
input:  /path/to/rgb_images/scene_001/frame_0001.png
search: /path/to/invisp_raw_rgb/scene_001/frame_0001.npy
search: /path/to/invisp_raw_rgb/scene_001/frame_0001.npz
```

`.npz` 中建议使用 key：

```text
raw_rgb
```

如果你的 InvISP 输出已经是单通道 Bayer 或 4-channel packed Bayer，就不要重复做 mosaic。此时应把脚本中 `load_external_raw_rgb` 和 `pack_rggb` 部分改成直接读取并标准化 packed Bayer。

## 6. 在线训练时的建议

你的计划是训练时在线生成 RAW-like 输入，这个方向是合理的。推荐流程如下：

```text
读取 RGB + depth
-> 对 RGB 和 depth 做一致的几何增强，比如 resize/crop/flip
-> 对增强后的 RGB 做 unprocessing
-> 得到 raw_packed [4, H/2, W/2]
-> 对 raw_packed 做 RAW-domain exposure/noise augmentation
-> 输入网络
```

关键原则：

1. 几何变换优先在 RGB/depth 阶段做。mosaic/pack 之后再做任意 resize 或奇数 crop 会破坏 CFA grid。
2. 如果必须在 packed Bayer 上 crop，建议在 packed 坐标中 crop，或者保证原图坐标中的 crop offset 是偶数。
3. 不建议在 RAW-like 输入上做 RGB color jitter。可以改成 RAW-domain gain、exposure、read noise、shot noise、black level、white level 等增强。
4. 随机参数需要可复现。建议使用 `global_seed + sample_index` 或 `global_seed + image_path_hash` 生成每个样本的随机数。
5. depth、semantic label、camera intrinsics 不应该被 photometric unprocessing 改变。只对 RGB 分支做 unprocessing。
6. 如果使用 KITTI 一类相机内参敏感的数据，resize/crop 之后要同步更新 intrinsics。这个和 RAW-like 生成无关，但会影响 depth 任务。

一个 PyTorch Dataset 中的伪代码：

```python
class DepthDatasetWithOnlineUnprocess(torch.utils.data.Dataset):
    def __getitem__(self, index):
        rgb, depth, meta = self.load_rgb_depth(index)
        rgb, depth, meta = self.geometric_transform(rgb, depth, meta)
        rng = make_deterministic_rng(self.base_seed, meta["path"])
        raw_packed, raw_meta = self.unprocessor(rgb, rng)
        return {
            "image": raw_packed,
            "depth": depth,
            "meta": {**meta, "unprocess": raw_meta},
        }
```

实际实现时建议把本文脚本中的 numpy 函数迁移成 torch tensor 版本，这样在线生成可以放到 CPU worker 或 GPU transform 中。若数据加载成为瓶颈，先将一小部分样本离线缓存，确认效果后再做全量在线版本。

## 7. 常规 depth 数据集适配建议

| 数据集 | 场景 | RGB 处理建议 | 备注 |
|---|---|---|---|
| NYUv2 | 室内 RGB-D | 按 8-bit sRGB 处理 | 室内低动态范围，适合先 debug inverse WB 和 packed Bayer。 |
| KITTI | 自动驾驶 RGB + LiDAR depth | 按相机 sRGB 处理 | 注意 resize/crop 后同步 intrinsics。 |
| Hypersim | 合成室内 | 先确认导出的 RGB 是否已经 tone mapped | 合成数据可能有更干净的信号和不同噪声分布。 |
| Virtual KITTI / VKITTI2 | 合成自动驾驶 | 按 tone-mapped RGB 处理 | 适合做 domain gap 和天气/光照 ablation。 |
| SUN RGB-D / DIML / DIODE | 常规 RGB-D depth | 按各自 RGB 格式处理 | 可作为额外验证集或跨数据集测试。 |
| TartanAir | 合成 SLAM / navigation | 可筛选低光、天气或高动态条件 | 有 RGB、depth、segmentation、flow、pose、LiDAR 等多模态。 |

## 8. 暗光或夜间 RGB-D / depth 相关数据集调研

下面列的是除了 nuScenes-Night 和 RobotCar-Night 之外，比较值得优先看的候选。它们不是全部都完全等价于“低光 RGB-D depth estimation benchmark”，但都和低光、夜间、RGB-D 或 depth 任务有直接关系。

| 数据集 | 类型 | 数据内容 | 适合用途 | 注意点 |
|---|---|---|---|---|
| LLRGBD | 低光室内，真实 + 合成 | low-light / normal-light image pairs，另有 real-depth 下载 | 室内低光 RGB-D、低光 scene understanding | 规模相对小，任务偏语义理解和反射恢复。 |
| LED Nighttime Synthetic Drive Dataset | 合成夜间自动驾驶 | 约 49,990 张夜间合成图，带 depth、normal、2D/3D detection、semantic/instance labels | 夜间单目 depth、夜间语义和检测 | 合成数据，和真实车载相机有 domain gap。 |
| MS2 | 真实户外多光谱 stereo | stereo RGB、NIR、thermal、LiDAR、projected depth、GNSS/IMU，含 morning/day/night | 夜间或恶劣条件 depth、多模态鲁棒性 | 不是纯 RGB-D，数据工程复杂度较高。 |
| CARLA-Night-DC | 合成夜间 depth completion | 7532 RGB-D pairs，KITTI-like sparse LiDAR pattern | 夜间 depth completion | 合成数据，偏 sparse depth completion。 |
| OpenLORIS-Object / Scene | 真实机器人 RGB-D | RGB-D、IMU、不同 illumination、occlusion、viewpoint 等变化 | 低光鲁棒性、SLAM、机器人视觉 | 更偏 lifelong robotics / SLAM，不是标准 monocular depth benchmark。 |
| QueensCAMP | 真实室内 RGB-D VSLAM | RGB-D，动态物体、motion blur、varying illumination | 鲁棒 RGB-D SLAM 和低光/变化光照测试 | 主要是 SLAM 评测，不一定直接适配 dense depth training。 |
| TartanAir | 合成多模态 | stereo RGB、depth、segmentation、flow、pose、LiDAR，含低光/天气/动态条件 | 大规模合成低光和恶劣条件预训练 | 合成数据，需控制和真实数据的 domain gap。 |

建议优先顺序：

```text
indoor RGB-D low-light: LLRGBD -> OpenLORIS -> QueensCAMP
outdoor/night driving depth: LED Nighttime Synthetic Drive -> MS2 -> CARLA-Night-DC -> TartanAir selected low-light subsets
```

## 9. 常见坑

### 9.1 RGB 不包含真实 RAW 的全部信息

8-bit sRGB 已经经历了 demosaic、white balance、color correction、tone mapping、gamma、compression 等步骤。unprocessing 只能生成 RAW-like 数据，不能恢复真实传感器的全部动态范围和噪声统计。InvISP 会比简单解析近似更接近真实 RAW，但仍受相机域和训练数据限制。

### 9.2 InvISP 是 camera-specific

InvISP 官方仓库提示，一个训练模型只适用于特定 camera setting。对 NYUv2、KITTI、Hypersim、VKITTI 这种混合来源数据，除非你有对应相机的 paired RAW/RGB，否则不要把输出称作真实 RAW。更稳妥的写法是：

```text
synthetic RAW-like packed Bayer
```

### 9.3 不要在 sRGB 域做低光后再当 RAW

RAW-Adapter 的低光和过曝合成基于 RAW 线性域。推荐顺序是：

```text
RGB -> unprocessing -> RAW-like -> low-light / over-exposure synthesis
```

而不是：

```text
RGB -> darken in sRGB -> unprocessing
```

### 9.4 Preview 不是训练输入

脚本保存的 preview PNG 只是为了肉眼检查。它使用很简单的 debug demosaic，不代表真实 ISP 输出，也不应该作为训练输入。

### 9.5 Packed Bayer 的尺寸变化

`H x W x 3` RGB 会变成 `[4, H/2, W/2]` packed Bayer。如果你的网络 backbone 期望和原 RGB 相同的 spatial size，需要在模型侧处理这个分辨率变化，或者设计适配层把 packed Bayer 映射到所需尺度。

## 10. 推荐最小实验

先用每个数据集抽样 100 张：

```bash
python unprocess_rgb_to_packed_raw.py --input-dir /path/to/nyuv2_rgb --output-dir /path/to/debug_nyuv2_rawlike --backend analytic --variants normal,dark,over --max-images 100 --save-preview --save-mosaic --storage float16
```

检查内容：

```text
1. raw_packed shape 是否为 [4, H/2, W/2]
2. 数值是否主要落在 [0, 1]
3. normal preview 是否没有明显通道错位
4. dark preview 是否明显更暗但仍保留结构
5. over preview 是否出现合理饱和，而不是全白
6. metadata 中 gain、light_scale、seed 是否完整记录
```

确认小样本后，再迁移到在线训练 transform。在线训练阶段不需要保存全部 `.npz`，但建议周期性保存少量 debug batch：

```text
train_debug/epoch_000/sample_000_raw_packed.npz
train_debug/epoch_000/sample_000_preview.png
train_debug/epoch_000/sample_000_meta.json
```

## 11. 参考来源

1. RAW-Adapter 官方 mmsegmentation README，ADE20K-RAW 描述：InverseISP -> raw-RGB -> inverse white balance -> mosaic。https://github.com/cuiziteng/ECCV_RAW_Adapter/tree/main/mmsegmentation_github
2. RAW-Adapter 论文 PDF，实验部分说明 ADE20K RAW 使用 InvISP，并给出 dark/over-exp 合成公式和 `l` 的范围。https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00484.pdf
3. RAW-Adapter `inverse_wb.py`，red/blue gain 采样范围和 inverse WB 实现。https://github.com/cuiziteng/ECCV_RAW_Adapter/blob/main/mmsegmentation_github/inverse_wb.py
4. InvISP 官方代码仓库。https://github.com/yzxing87/Invertible-ISP
5. InvISP project page。https://yzxing87.github.io/InvISP/index.html
6. NYU Depth V2 官方页面。https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
7. KITTI depth completion / depth prediction benchmark。https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
8. Hypersim 官方仓库。https://github.com/apple/ml-hypersim
9. Virtual KITTI 2 announcement。https://europe.naverlabs.com/blog/announcing-virtual-kitti-2/
10. LLRGBD / LISU GitHub。https://github.com/noahzn/LISU
11. LED Nighttime Synthetic Drive project page。https://simondemoreau.github.io/LED/
12. MS2 Multi-Spectral Stereo Dataset GitHub。https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset
13. Learnable Differencing Center for Nighttime Depth Perception，CARLA-Night-DC 和 RobotCar-Night-DC 描述。https://link.springer.com/article/10.1007/s44267-024-00048-9
14. OpenLORIS-Object dataset page。https://lifelong-robotic-vision.github.io/dataset/object.html
15. QueensCAMP RGB-D dataset for robust Visual SLAM。https://arxiv.org/html/2410.12520v1
16. TartanAir dataset page。https://theairlab.org/tartanair-dataset/
