# 0608 RAW 输入链路 · 网络设计 Brief（专家讨论用）

> 用途：和其他专家讨论 **RAW 输入这条线的网络设计修正**。核心议题——当前 DAv2
> backbone 的输入**完全是 RAM 模块的输出**，希望引入"结合"（让 backbone 不只看 RAM 的
> 重建图）。本文只陈述现状与代码里**已实现但未启用**的结合接口，不预设修正方案、不放性能指标。
>
> 重点章节：[§4 RAW 链路怎么走](#4-raw-输入这条线具体怎么走重点) 和
> [§5 核心讨论点：backbone 输入 = RAM 纯替代](#5-核心讨论点backbone-输入--ram-纯替代)。

---

## 1. 数据准备（精炼）

两套数据，同一种"teacher 出伪标签、student 学伪标签"的蒸馏范式。teacher 都是冻结的
**DAv2-L**，student 都是 **DAv2-S**；标签空间统一为 `inverse_relative`（逆相对深度），
验证只在伪标签空间做 proxy 指标，不解释为 metric depth。

### 1.1 ROD-night（早期链路）

- 来源：ROD 夜间 RAW24。
- **teacher**：对夜间图提亮后过冻结 DAv2-L，离线生成 inverse-relative 伪标签
  （`ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/`，native `1440×928`）。
- **student-RGB**（[rod_raw_student_rgb.py](finetune_stf/dataset/rod_raw_student_rgb.py)）：
  RAW24 → `unpack_raw24` → 可配置 ISP 渲染管线在线生成 student dark RGB。
- **student-RAW**（[rod_raw.py](finetune_stf/dataset/rod_raw.py)）：RGGB cache（缺失时 raw24
  回退）→ `pack_raw24_to_rggb` → **4 通道 packed Bayer `[R, Gr, Gb, B]`**。

### 1.2 LOD-true（当前主线，0608 接入）

真实 LOD 数据集（四目录 `RGB_normal / RGB_Dark / RAW_normal / RAW_Dark`，详见
[0608_true_lod_integration_execution_plan.md](plans/0608_true_lod_integration_execution_plan.md)）。

```text
teacher : LOD RGB_normal -> 冻结 DAv2-L(input_size=924) -> 离线 inverse-relative 伪标签 (1200x800)
student-RGB : LOD RGB_Dark -> DAv2-S                （对照组）
student-RAW : LOD RAW_Dark -> RAW-RGB16 3ch 前端 -> DAv2-S  （RAW 主线）
```

- RAW 物理格式与 ROD **不同**：LOD RAW 是 **3 通道 16-bit PNG**（非 Bayer），
  归一化 `uint16 / 65535`，dataset 内把 OpenCV 的 BGR 重排为模型 RGB 序。
- split：`block8excl10`（8 个均匀验证块 + 邻域剔除缓冲），消除了 pair-random 的邻近泄漏。

### 1.3 两条 RAW 的物理格式对比

| 维度 | ROD-night RAW | LOD-true RAW |
|---|---|---|
| 物理格式 | 4ch packed Bayer `[R,Gr,Gb,B]` | 3ch uint16 PNG（类 RGB） |
| 归一化 | ISP 渲染 / `/255` 域 | `uint16 / 65535` |
| `input_domain` | `raw4` | `raw3` |
| RAM 核 | `RawRamCore`（4ch） | `RamCore3`（3ch） |
| dataset family | `rod_raw` / `rod_raw_student_rgb` | `lod_true_raw_dark_rgb16` |

---

## 2. 训练方式（精炼）

- **范式**：离线伪标签蒸馏。训练拟合 dense inverse-relative 伪标签；验证在伪标签空间做
  proxy 指标；LOD 正式实验 best checkpoint 按 `d1` 最大化保存。
- **stage**（[train.py:368](finetune_stf/train.py#L368)）：`stf_only / rod_only / lod_only /
  eval_only`，与 dataset family 强校验绑定。
- **可训练范围消融阶梯**（实现均已就位）：
  1. **decoder-only**：backbone 冻结（[train.py:1227](finetune_stf/train.py#L1227)
     `dav2_module.requires_grad_(False)`），只训 RAM 前端 + DPT decoder。
  2. **+LoRA**：backbone 注入低秩可训练（见 §5）。
  3. **+backbone layer-decay**：逐层学习率衰减
     （[`_build_layer_decay_param_groups`](finetune_stf/train.py#L2914)，
     `lr = base_lr * layer_decay^(max_layer - layer)`）。
- **优化器分组**（[`_build_named_param_groups`](finetune_stf/train.py#L2998)）：按
  `raw_front_end_lr / bridge_lr / adapter_lr / base(decoder/backbone)_lr` 分组，分别给学习率，
  并在 resolved config 与日志中显式记录每组可训练参数量。
- **当前 LOD-true RAW 正式配置**：decoder-only，RAM 前端 lr 与 decoder lr 分离，backbone 全冻结。

---

## 3. 统一网络范式

不管 RGB 还是 RAW、不管 ROD 还是 LOD，最终都收敛到同一条主干：

```text
输入 -> [前端 front_end] -> 3ch 张量 -> center pad(512x960 -> 518x966)
     -> DAv2 backbone (DINOv2 ViT) -> DPT decoder -> center crop -> depth
```

- **空间适配**（[spatial_adapter.py](finetune_stf/models/spatial_adapter.py)）：传感器原生
  `512×960` 中心 pad 到 patch 对齐的 `518×966` 进 backbone，输出 depth 再中心 crop 回 `512×960`。
- **backbone**：DINOv2 ViT（LOD/ROD 用 **DAv2-S**），DPT decoder 在 refinenet 路径逐级上采样。
- 差异只在 **front_end**：RGB 走 `dav2_rgb`（恒等透传 + ImageNet norm）；RAW 走 RAM 系列前端。

`front_end` 选项（[resolved.py:20](finetune_stf/config/resolved.py#L20)）：
`dav2_rgb · raw_to_rgb_head · raw_ram4 · raw_to_base_rgb_ram3 · raw_rgb16_ram3`。

---

## 4. RAW 输入这条线具体怎么走（重点）

以 **当前 LOD-true RAW 主线**（front_end = `raw_rgb16_ram3`，wrapper =
[`RawRgb16Ram3DepthModel`](finetune_stf/models/raw_ram.py#L619)）为例，完整数据流：

```text
RAW_Dark (3ch uint16 PNG)
  │  dataset: /65535 归一化，BGR -> RGB 重排
  ▼
┌─ RamCore3 ───────────────────────────────────────────────┐   (raw_ram.py:316)
│  RPEncoder: 下采样到 256² → 全局 ISP 参数向量 z (128维)    │
│  4 个并行 ISP 分支(各自吃 [原图, z]):                       │
│     WB(逐通道增益) · CCM(3×3 混色) · Gamma · Brightness     │
│  concat → 12ch → FFM3(12→16→64→16→3) → BatchNorm(3)        │
└────────────────────────────────────────────────────────────┘
  │  tail = identity   (可选 tanh2p5 软压重尾，当前用 identity)
  ▼
center pad 512×960 → 518×966
  ▼
DAv2 backbone (DINOv2 ViT, 冻结)
  ▼
DPT decoder (可训练)
  ▼
center crop → depth (inverse-relative)
```

要点：

1. **RAM 是一个可学习的"轻量 ISP"**：用一个下采样图预测全局 ISP 参数（WB/CCM/Gamma/
   Brightness），四个分支并行处理原分辨率输入，再用 FFM 融合成一张 3 通道图。它的产物是一张
   **与输入同分辨率的"伪 RGB 重建图"**。
2. **ROD RAW 的差异**：物理输入是 4ch Bayer，所以走 `raw_ram4`
   （[`RawRamCore`](finetune_stf/models/raw_ram.py#L198) 4ch + `RGBInterfaceHead` 把 4ch→3ch），
   或 `raw_to_base_rgb_ram3`（先 `[R,(Gr+Gb)/2,B]` 降成 3ch 再过 RamCore3）。
   LOD RAW 已经是类 RGB 的 3ch，所以 `raw_rgb16_ram3` 直接喂 RamCore3，**不做 Bayer 投影**。
3. **训练时只有 RAM 前端 + DPT decoder 在更新**，backbone 冻结。

---

## 5. 核心讨论点：backbone 输入 = RAM 纯替代

### 5.1 现状（确认）

当前 RAW 主线里，**RamCore3 输出的那张 3 通道伪 RGB 图，是唯一进入 backbone 的张量**。
backbone 看不到任何原始 RAW 信息——它完全"透过 RAM 的眼睛"看世界。这正是需要讨论修正的点：
是否、以及如何让 backbone（或 decoder）**额外结合**原始 RAW / 其它特征，而不是被 RAM 完全中介。

### 5.2 代码里已实现、但 LOD 正式实验未启用的"结合"接口

框架里其实已经预置了三类结合机制（都已接通、gate 从 0 起步以保护预训练权重），切换只需改
`--input-type`（dataset 与 front_end 不变），无需改模型代码：

| 结合方式 | 机制 | backbone 输入是否还是 RAM 图 | 实现位置 |
|---|---|---|---|
| **Bridge 注入** | RAM 中间特征（`x3`/`x_cat`/`ffm_mid`）投影成 token，按 `tanh(gate)·tokens` 注入 ViT 的若干 tap block | 是，但 **ViT 内部各层额外融合** RAM 特征 | [lora_bridge.py](finetune_stf/models/lora_bridge.py#L204) `RawFeatureBridgeAdapter` |
| **Decoder feature adapter** | RAM 中间特征投影到 a1/a2/a3 三尺度，在 DPT refinenet 路径用 `DepthMergeBlock` 融合 | 是，但 **decoder 侧结合** RAM 特征 | [raw_feature_adapter.py](finetune_stf/models/raw_feature_adapter.py#L131) |
| **Bridge + adapter** | 上两者叠加 | ViT 内部 + decoder 都结合 | 同上 |
| **LoRA** | backbone 注入低秩可训练（qkv/proj），不再全冻结 | 输入不变，但 backbone 本身被微调 | [lora_bridge.py:125](finetune_stf/models/lora_bridge.py#L125) `apply_lora_to_vit` |

> 注意边界：以上结合的**输入张量仍是 RAM 重建图**，只是把 RAM 的**中间特征**旁路注入到
> backbone 内部或 decoder。它们都还**没有**把"原始 RAW / 低光 RGB"等 RAM 之外的信息直接喂给
> backbone 输入端——若专家想要的是**输入级**结合（如 RAM 图与原始 RAW base 的 concat/残差、
> RAW+RGB 双流、normal-guided 引导等），那属于**新增设计**，不在现有开关内。

### 5.3 LOD RAW 的"结合"开关速查（resolved.py 已预置别名）

[resolved.py:407](finetune_stf/config/resolved.py#L407) 起，LOD RAW 的各 `input_type` 已定义好：

| `--input-type` | bridge | decoder adapter | LoRA |
|---|:--:|:--:|:--:|
| `lod_true_raw_dark_rgb16` （**当前正式用**） | — | — | — |
| `lod_true_raw_dark_rgb16_bridge` | ✓(`x3`) | — | — |
| `lod_true_raw_dark_rgb16_feature_adapter` | — | ✓(`x3`) | — |
| `lod_true_raw_dark_rgb16_bridge_feature_adapter` | ✓ | ✓ | — |
| `..._lora` 各变体 | — | — | ✓ |

切到任一"结合"档，`dataset_family` / `front_end` / `model_input_tensor` / `raw_storage_format`
均不变，仅打开对应 `bridge` / `decoder_feature_adapter` / `lora` 分支与其特征源通道。

---

## 6. 实验语义参数速查

`resolved.py` 把会改变实验语义的参数集中校验（避免靠路径名/字符串推断）。RAW 主线取值：

| 参数 | LOD RAW 当前值 | 含义 |
|---|---|---|
| `input_domain` | `raw3` | 输入域（`rgb` / `raw4` / `raw3`） |
| `front_end` | `raw_rgb16_ram3` | 前端类型 |
| `model_input_tensor` | `raw` | 喂模型的张量是 raw 还是 image |
| `raw_storage_format` | `raw_rgb16_png_3ch` | RAW 物理存储格式 |
| `bridge` | `none` | backbone 内部注入开关 |
| `decoder_feature_adapter` | `none` | decoder 侧融合开关 |
| `lora` | `none` | backbone 低秩微调开关 |

---

## 7. 关键文件索引

| 模块 | 文件 |
|---|---|
| RAM 核 / 各 RAW wrapper | [models/raw_ram.py](finetune_stf/models/raw_ram.py) |
| Bridge 注入 + LoRA | [models/lora_bridge.py](finetune_stf/models/lora_bridge.py) |
| Decoder feature adapter | [models/raw_feature_adapter.py](finetune_stf/models/raw_feature_adapter.py) |
| 空间 pad/crop 适配 | [models/spatial_adapter.py](finetune_stf/models/spatial_adapter.py) |
| backbone bridge 注入接入点 | [depth_anything_v2/dinov2.py:271](depth_anything_v2/dinov2.py#L271) |
| 实验语义参数中心 | [config/resolved.py](finetune_stf/config/resolved.py) |
| 训练主脚本（构建/分组/冻结） | [finetune_stf/train.py](finetune_stf/train.py) |
| LOD-true 数据 | [dataset/lod_true.py](finetune_stf/dataset/lod_true.py) |
| ROD 数据 | [dataset/rod_raw.py](finetune_stf/dataset/rod_raw.py) · [dataset/rod_raw_student_rgb.py](finetune_stf/dataset/rod_raw_student_rgb.py) |
| LOD 接入全过程 | [plans/0608_true_lod_integration_execution_plan.md](plans/0608_true_lod_integration_execution_plan.md) |
