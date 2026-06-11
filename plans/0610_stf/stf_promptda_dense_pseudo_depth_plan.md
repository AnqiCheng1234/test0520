# STF Dense Pseudo Depth Label 生成方案：DAv2-L + PromptDA

## 1. 目标

针对 Seeing Through Fog（STF）数据集，生成可用于后续训练的 dense pseudo depth label。

STF 提供 sparse LiDAR / depth GT，但其空间覆盖非常稀疏，不能直接作为 dense depth supervision。当前目标不是重新设计训练 loss，也不是比较多个 depth completion 模型，而是先实现一条稳定的离线 label generation pipeline：

> 使用 STF 官方 RGB 图像作为视觉输入，使用 DAv2-L relative inverse depth 作为 dense structure prior，使用 STF sparse LiDAR depth 作为 metric anchor，并通过 PromptDA 生成最终 dense metric depth label。

最终生成结果应具备：

1. 尽量保留 DAv2-L 的 dense、object boundary、semantic structure 优势；
2. 通过 STF sparse LiDAR GT 提供 metric correctness；
3. 输出 dense metric depth，单位为 meters；
4. 尽量减少额外模型和规则引入，避免第一版过度复杂。

---

## 2. 当前版本边界

### 2.1 本版本只做什么

本版本只负责：

1. 读取 STF RGB；
2. 读取 STF sparse LiDAR depth；
3. 使用 DAv2-L 预测 dense relative inverse depth；
4. 在 inverse-depth space 使用 sparse LiDAR 做 metric alignment；
5. 构造 PromptDA 的 prompt depth；
6. 使用 PromptDA 生成 dense metric depth label；
7. 保存 dense label 和必要的 debug / meta 文件。

### 2.2 本版本暂时不做什么

暂时不做以下内容：

1. 不比较多个 depth completion 模型；
2. 不加入 Marigold-DC、CompletionFormer、NLSPN、GuideNet 等其他模型；
3. 不做 Phase 1 / Phase 2 大规模验证流程；
4. 不负责后续监督训练 loss 设计；
5. 不生成复杂 confidence map；
6. 不把 dense pseudo label 直接称为 dense GT。

推荐命名：

- dense pseudo depth label
- LiDAR-anchored dense pseudo depth
- sparse-GT-calibrated dense depth
- PromptDA-generated dense pseudo supervision

不推荐命名：

- dense ground truth
- dense GT
- true dense label

---

## 3. 输入数据选择

### 3.1 RGB 输入

建议固定使用：

```text
cam_stereo_left_lut
```

理由：

1. 这是 STF 官方提供的 left-view RGB / LUT 图像；
2. 适合作为 DAv2-L 和 PromptDA 的 RGB 输入；
3. label generation 阶段不应使用你自己的 RAW ISP 结果，否则会让 label generation 和后续要研究的 ISP 变量耦合；
4. RAW 只作为后续模型输入，不参与 pseudo label 生成。

### 3.2 Sparse depth 输入

第一版建议使用：

```text
lidar_hdl64_last_stereo_left
```

备用保存或后续对照：

```text
lidar_hdl64_strongest_stereo_left
```

第一版不建议引入复杂的 last / strongest fusion 规则。  
如果需要保持方案最简单，可以只使用 `last_stereo_left`。

---

## 4. 核心 pipeline

整体流程如下：

```text
for each STF sample:

1. 读取 RGB:
   cam_stereo_left_lut

2. 读取 sparse depth:
   lidar_hdl64_last_stereo_left

3. 跑 DAv2-L:
   r = relative inverse depth

4. 在 sparse valid pixels 上拟合:
   1 / Z_lidar = a * r + b

5. 得到 dense metric prior:
   Z_prior = 1 / (a * r + b)

6. 构造 PromptDA prompt:
   Z_prompt = downsample(Z_prior, 192×256)
   对有 LiDAR 点的 low-res cells:
       用该 cell 内 LiDAR median depth 覆盖

7. 跑 PromptDA:
   Z_dense = PromptDA(RGB, Z_prompt)

8. 保存:
   dense_depth_promptda.npy
   dav2_aligned_prior.npy
   sparse_lidar_mask.png
   alignment_meta.json
   debug_vis.png
```

---

## 5. DAv2-L relative inverse depth alignment

### 5.1 为什么必须在 inverse-depth space 拟合

DAv2-L relative depth 输出更接近 relative inverse depth，而不是 metric depth。

因此不应拟合：

```text
Z = a * r + b
```

而应该拟合：

```text
1 / Z = a * r + b
```

其中：

- `r`：DAv2-L 输出的 relative inverse depth；
- `Z`：STF sparse LiDAR metric depth，单位 meters；
- `a, b`：每张图单独拟合得到的 scale / shift 参数。

最终 metric prior 为：

```text
Z_prior = 1 / (a * r + b)
```

### 5.2 拟合点选择

只在 sparse LiDAR valid pixels 上拟合：

```text
valid = sparse_depth > 0
valid = valid & (sparse_depth >= Z_min)
valid = valid & (sparse_depth <= Z_max)
```

建议初始深度范围：

```text
Z_min = 1.0 m
Z_max = 80.0 m
```

如果后续发现 STF 中远距离结构很重要，可以额外尝试：

```text
Z_max = 100.0 m
```

第一版建议先用 `80m`，因为远距离点在雾天 / 夜间 / 低能见度场景中更容易不稳定。

### 5.3 Robust fitting 建议

不要直接用普通 least squares 裸拟合。建议使用以下之一：

优先级：

1. RANSAC linear regression；
2. Huber regression；
3. median-ratio 初始化 + residual trimming；
4. 普通 least squares 只作为 fallback。

拟合目标：

```text
y = 1 / Z_lidar
x = r
y = a * x + b
```

约束：

```text
a > 0
a * r + b > eps
```

建议：

```text
eps = 1e-6
```

如果拟合失败，例如：

- valid sparse points 太少；
- 拟合得到 `a <= 0`；
- 大面积 `a*r+b <= 0`；
- residual 过大；

则该样本标记为 `alignment_failed`，不要直接生成可靠 pseudo label。可以保存 debug 信息，后续人工检查或跳过。

---

## 6. PromptDA prompt depth 构造

### 6.1 不建议直接使用纯 sparse LiDAR prompt

不建议第一版直接使用：

```text
RGB + sparse LiDAR 点图 -> PromptDA
```

原因：

1. STF sparse LiDAR 投影到图像后非常稀疏；
2. PromptDA 更适合输入低分辨率但相对连续的 metric depth prompt；
3. 如果无效区域直接填 0，模型可能把 0 误解为真实近距离深度；
4. 官方接口没有显式 confidence / validity mask 输入，因此无效区域编码需要特别慎重。

### 6.2 推荐 prompt 构造方式

推荐使用：

```text
RGB + DAv2 inverse-aligned dense metric prior with sparse LiDAR overwrite
```

具体做法：

1. 先根据 DAv2-L 和 sparse LiDAR alignment 得到原图分辨率的 `Z_prior`；
2. 将 `Z_prior` 下采样到 PromptDA 所需尺寸；
3. 如果某个 low-res cell 内存在 sparse LiDAR valid points，则用该 cell 内 LiDAR median depth 覆盖原来的 `Z_prior`；
4. 如果某个 low-res cell 内没有 LiDAR 点，则保留 DAv2-aligned `Z_prior`。

推荐 prompt 尺寸：

```text
192 × 256
```

即：

```text
Z_prompt.shape = (192, 256)
```

### 6.3 下采样细节

对于 `Z_prior`：

```text
Z_prompt = resize_or_area_downsample(Z_prior, target_size=(192, 256))
```

对于 sparse LiDAR：

```text
for each low-res cell:
    collect all valid LiDAR points projected into this cell
    if number_of_points > 0:
        Z_prompt[cell] = median(lidar_depth_values)
```

使用 median 而不是 mean 的原因：

1. 更鲁棒；
2. 可以降低少数投影错误点或离群点的影响；
3. 对 dynamic object / reflective surface 的异常值更稳。

---

## 7. 是否生成 confidence map

第一版不建议生成 continuous confidence map。

原因：

1. PromptDA 输出通常是 depth，而不是 depth + uncertainty；
2. 如果模型没有显式 confidence head，手工构造 confidence map 需要很多主观规则；
3. confidence map 会增加后续训练解释成本；
4. 当前目标只是生成第一版 dense label，应先降低变量数量。

第一版建议保存以下辅助文件，而不是 confidence map：

```text
dense_depth_promptda.npy        # 最终 dense pseudo label, meters
dav2_aligned_prior.npy          # PromptDA 输入前的 aligned dense metric prior
sparse_lidar_mask.png           # sparse LiDAR valid mask
alignment_meta.json             # 每张图的 alignment 参数和统计信息
debug_vis.png                   # RGB / sparse / prior / final 的可视化
```

如果后续训练确实需要 confidence，可以再基于以下信息构造 conservative confidence：

1. sparse LiDAR valid mask；
2. alignment residual；
3. depth range；
4. day/night/weather metadata；
5. DAv2 prior 与 sparse LiDAR 的局部一致性；
6. PromptDA 输出与 sparse LiDAR 的误差。

但这不属于第一版。

---

## 8. 输出文件设计

建议每个样本保存：

```text
outputs/
  dense_depth_promptda/
    sample_xxx.npy

  dav2_aligned_prior/
    sample_xxx.npy

  sparse_lidar_mask/
    sample_xxx.png

  alignment_meta/
    sample_xxx.json

  debug_vis/
    sample_xxx.png
```

### 8.1 dense_depth_promptda

格式：

```text
.npy
float32
unit: meters
shape: H × W
```

用途：

- 后续训练中的 dense pseudo supervision；
- 不建议直接作为真正 GT 评价指标。

### 8.2 dav2_aligned_prior

格式：

```text
.npy
float32
unit: meters
shape: H × W
```

用途：

- debug；
- 对比 PromptDA 前后的变化；
- 判断 PromptDA 是否过度修改 DAv2 prior；
- 后续可能用于 ablation。

### 8.3 sparse_lidar_mask

格式：

```text
.png
uint8
0: invalid
255: valid
```

用途：

- 标记真实 sparse LiDAR GT 的位置；
- 后续训练或评价时可用于区分 real sparse GT 和 pseudo dense label。

### 8.4 alignment_meta

格式：

```json
{
  "sample_id": "xxx",
  "rgb_path": "...",
  "sparse_depth_path": "...",
  "dav2_model": "DAv2-L relative depth",
  "promptda_model": "PromptDA-Large",
  "alignment_space": "inverse_depth",
  "a": 0.0,
  "b": 0.0,
  "z_min": 1.0,
  "z_max": 80.0,
  "num_valid_sparse_points": 0,
  "num_points_used_for_fit": 0,
  "fit_method": "RANSAC or Huber",
  "fit_failed": false,
  "median_abs_inverse_residual": 0.0,
  "mean_abs_inverse_residual": 0.0,
  "notes": ""
}
```

### 8.5 debug_vis

建议可视化内容：

1. RGB；
2. sparse LiDAR depth overlay；
3. DAv2 aligned prior；
4. PromptDA final dense depth；
5. absolute difference between prior and final；
6. sparse valid mask。

---

## 9. 推荐第一版配置表

| 项目 | 第一版配置 |
|---|---|
| Dataset | Seeing Through Fog |
| RGB 输入 | `cam_stereo_left_lut` |
| Sparse depth | `lidar_hdl64_last_stereo_left` |
| Dense prior | DAv2-L relative inverse depth |
| Alignment space | inverse depth |
| Alignment formula | `1/Z = a*r + b` |
| Robust fitting | RANSAC / Huber / residual trimming |
| Depth range | `[1m, 80m]` |
| PromptDA 输入 | RGB + low-res metric prompt depth |
| Prompt size | `192×256` |
| Prompt 构造 | DAv2-aligned dense prior 下采样，有 LiDAR cell 用 LiDAR median 覆盖 |
| Final output | PromptDA dense metric depth |
| Output unit | meters |
| Output format | `.npy` float32 |
| Confidence map | 第一版不做 |
| Debug output | sparse mask + alignment meta + visualization |

---

## 10. 需要最后敲定的关键点

在正式写代码前，建议只需要确认以下几件事：

### 10.1 Sparse depth source

第一版默认：

```text
lidar_hdl64_last_stereo_left
```

是否需要同时保存 strongest 版本作为备用？

建议：

```text
第一版只跑 last；
如果后续发现 fog / reflective object 里 last 不稳定，再检查 strongest。
```

### 10.2 Depth max range

默认：

```text
Z_max = 80m
```

备选：

```text
Z_max = 100m
```

建议：

```text
第一版用 80m；
后续根据 STF sparse depth 分布和可视化再调整。
```

### 10.3 Alignment failure policy

建议：

```text
如果有效 sparse 点太少或 robust fitting 失败，则跳过该样本，并在 alignment_meta.json 中标记失败。
```

不要强行生成 label，否则会污染后续训练。

### 10.4 PromptDA prompt 输入

建议最终固定为：

```text
DAv2 inverse-aligned dense prior + sparse LiDAR low-res overwrite
```

不建议使用：

```text
纯 sparse LiDAR prompt
```

---

## 11. 一句话总结

第一版最合理的方案是：

> 用 STF 官方 RGB 跑 DAv2-L 得到 dense relative inverse depth，在 sparse LiDAR 有效点上做 inverse-depth metric alignment，得到 dense metric prior；再将该 prior 下采样为 PromptDA 的 low-res metric prompt，并用 sparse LiDAR 在对应 cell 内进行 median overwrite，最终由 PromptDA 输出 dense metric pseudo label。

这个方案的优点是：

1. 保留 DAv2-L 的 dense structure 和 object boundary；
2. 利用 STF sparse LiDAR 提供 metric anchor；
3. 避免直接把过稀疏的 LiDAR 点图喂给 PromptDA；
4. 不引入额外 depth completion 模型；
5. 第一版没有 confidence map，变量更少，解释更清楚；
6. 后续如果需要，可以自然扩展到 confidence-weighted supervision。
