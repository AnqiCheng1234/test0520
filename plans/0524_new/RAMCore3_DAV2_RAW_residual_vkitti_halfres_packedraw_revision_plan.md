# RAMCore3 + Frozen DAV2 RAW Residual：VKITTI Half-size Packed RAW 修订计划

## 0. 修订结论

原 `M` 系列计划里的 `512x960` 对 VKITTI2 不合适：

```text
VKITTI2 原图: 375 x 1242
旧设置:       512 x 960
```

旧 dataset 路径会先把原图放大到 `1024 x 3391`，再裁成 `1024 x 1920`，最后压回 `512 x 960`。这会改变宽高比，并裁掉大量水平视野。本修订废弃该尺寸路径。

新的本轮决定：

```text
原始 VKITTI2 RGB/depth:      375 x 1242
fullres even policy:         crop bottom 1 row
even fullres RGB/depth:      374 x 1242
4ch packed raw grid:         187 x 621
RGB baseline / DAV2 input:   187 x 621
depth target / valid mask:   187 x 621
```

也就是说，模型看到的 `raw` 是：

```text
sample["raw"]: [4, 187, 621]
channel order: [R, Gr, Gb, B]
```

这里的 `187 x 621` 是 packed Bayer grid，不是单通道 Bayer mosaic 尺寸。对应的 full-resolution synthetic Bayer mosaic 是 `374 x 1242`。

底部裁剪规则固定为：

```text
if original height is odd:
  remove the last row only
if original width is odd:
  raise error for this VKITTI run unless explicitly configured
```

本轮 VKITTI2 宽度 `1242` 已经是偶数，所以只裁底部 1 行。

---

## 1. 参考依据

本修订遵循三类已有实践：

```text
1. RAW Bayer learning 常规做法：
   Bayer raw pack 成 4 通道，空间分辨率每个维度减半。

2. KITTI / driving depth 常规做法：
   为了得到稳定尺寸，常裁剪到偶数或 stride-friendly 尺寸，而不是强行保留奇数高度。

3. DAV2 / DPT 类模型：
   输入只需要 pad 到 patch size 的倍数；不需要把数据本身 reshape 到错误宽高比。
```

可参考：

```text
Learning to See in the Dark:
https://vladlen.info/papers/learning-to-see-in-the-dark.pdf

BTS official dataloader, KITTI crop to 352x1216:
https://raw.githubusercontent.com/cleinc/bts/master/pytorch/bts_dataloader.py

Monodepth2 official fixed input sizes / resizing:
https://github.com/nianticlabs/monodepth2
https://raw.githubusercontent.com/nianticlabs/monodepth2/master/datasets/mono_dataset.py

Depth Anything V2 keep aspect ratio + ensure multiple of 14:
https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/depth_anything_v2/dpt.py
https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/depth_anything_v2/util/transform.py
```

---

## 2. 参数语义修订

必须显式区分：

```text
source_fullres_hw:        原始 sRGB/depth 尺寸，本轮 VKITTI2 是 375x1242
fullres_even_policy:      crop_bottom_to_even
raw_storage_format:       synthetic_packed_bayer_4ch_halfres
model_input_tensor:       raw
model_input_hw:           packed/model grid 尺寸，本轮是 187x621
rgb_input_space:          halfres_2x2_area_from_even_fullres
depth_target_space:       halfres_2x2_valid_mean_from_even_fullres
```

正式脚本中不得只写 `--input-height 187 --input-width 621` 而隐藏这些语义。训练入口和检查工具都必须先扩展 arg surface，否则本计划里的命令会直接因为 `unrecognized arguments` 或 choices 不匹配失败。

### 2.1 训练入口参数面

修改：

```text
foundation/tools/train_vkitti2_raw_residual.py
```

`parse_args()` 必须显式加入：

```bash
--input-domain raw4
--model-input-tensor raw
--raw-storage-format synthetic_packed_bayer_4ch_halfres
--fullres-even-policy crop_bottom_to_even
--rgb-input-space halfres_2x2_area
--depth-target-space halfres_2x2_valid_mean
--input-height 187
--input-width 621
```

其中：

```text
--input-height / --input-width
```

在这个训练入口中明确表示模型输入 grid，也就是 packed raw / half-size RGB / half-size depth 的共同空间尺寸。

具体修改：

```text
1. --raw-storage-format choices 从
   synthetic_packed_bayer_4ch
   扩为：
   synthetic_packed_bayer_4ch
   synthetic_packed_bayer_4ch_halfres

2. 新增 required 参数：
   --fullres-even-policy choices=[not_applicable,crop_bottom_to_even]
   --rgb-input-space choices=[not_applicable,halfres_2x2_area]
   --depth-target-space choices=[not_applicable,halfres_2x2_valid_mean]

3. validate_args() 做集中语义耦合校验：
   raw_storage_format=synthetic_packed_bayer_4ch_halfres 时：
     fullres_even_policy 必须是 crop_bottom_to_even
     rgb_input_space 必须是 halfres_2x2_area
     depth_target_space 必须是 halfres_2x2_valid_mean

   raw_storage_format=synthetic_packed_bayer_4ch 时：
     fullres_even_policy 必须是 not_applicable
     rgb_input_space 必须是 not_applicable
     depth_target_space 必须是 not_applicable

4. build_loaders() 必须把 raw_storage_format / fullres_even_policy /
   rgb_input_space / depth_target_space 透传给 VKITTI2Raw(...)。

5. config.json 和 train.log 必须记录这四个 experiment-semantic 字段。
   现有 train.log/config.json 使用 vars(args)，新增参数后仍要确认字段实际出现；
   如果后续还记录 resolved dataset geometry，也必须写入 config.json 或单独
   dataset_geometry.json，不能只打印到 stdout。
```

旧路径如果还需要保留，正式旧脚本也要显式传：

```bash
--raw-storage-format synthetic_packed_bayer_4ch
--fullres-even-policy not_applicable
--rgb-input-space not_applicable
--depth-target-space not_applicable
```

### 2.2 Sign check 参数面

修改：

```text
foundation/tools/check_vkitti_dav2_sign.py
```

该工具当前只有 `--input-height / --input-width`，必须同步新增：

```bash
--raw-storage-format synthetic_packed_bayer_4ch_halfres
--fullres-even-policy crop_bottom_to_even
--rgb-input-space halfres_2x2_area
--depth-target-space halfres_2x2_valid_mean
```

并复用与训练入口一致的语义耦合校验，然后把这些参数透传给 `VKITTI2Raw(...)`。输出 summary JSON 也必须记录这四个字段，避免 sign 结果和数据语义脱钩。

---

## 3. Dataset 修改目标

修改：

```text
foundation/engine/datasets/vkitti2_raw.py
```

新增模式：

```python
raw_storage_format="synthetic_packed_bayer_4ch_halfres"
fullres_even_policy="crop_bottom_to_even"
rgb_input_space="halfres_2x2_area"
depth_target_space="halfres_2x2_valid_mean"
```

这四个字段是 dataset 构造参数，不允许从 `size`、split path、cache path、run name 字符串隐式推断。

### 3.1 新模式下禁止旧 resize/crop 路径

新模式下不要调用旧逻辑：

```python
_resize_short_edge(..., short_edge=self.fullres_size[0])
_random_crop(...)
_center_crop(...)
cv2.resize(..., self.size)
```

旧 `synthetic_packed_bayer_4ch` 可以继续使用上述旧路径；`synthetic_packed_bayer_4ch_halfres` 必须走独立分支。

新模式流程固定为：

```text
1. 读 VKITTI RGB/depth 原图。
2. 检查原图 H,W。
3. 对 fullres image/depth/valid 做 bottom crop 到偶数高：
   375 x 1242 -> 374 x 1242
4. train 可以 hflip；val 不 hflip。
5. unprocessing 在 374 x 1242 fullres RGB 上执行。
6. pack Bayer 后得到 raw4: 4 x 187 x 621。
7. RGB input 从 bottom-cropped fullres RGB 做显式 2x2 area downsample，得到 3 x 187 x 621。
8. depth target 从 bottom-cropped fullres depth + valid 做显式 2x2 valid mean，得到 187 x 621。
9. valid_mask 从 2x2 block valid 像素数得到，建议 any-valid 即 valid。
```

`self.fullres_size = (self.size[0] * 2, self.size[1] * 2)` 不能作为新模式的 fullres 来源。新模式应以实际读到的图像 shape 为准：

```text
even_fullres = bottom_crop_to_even(original_image)
assert even_fullres_h == 2 * input_height
assert even_fullres_w == 2 * input_width
```

不满足时直接抛错，不做 resize/crop 兜底。这样以后换 scene 或复用到 KITTI 时不会静默错位。

CFA + hflip 顺序固定为：

```text
bottom-crop 到 even fullres
-> train hflip
-> assert image_even.shape[0] % 2 == 0 and image_even.shape[1] % 2 == 0
-> unprocess / pack Bayer
```

hflip 必须发生在 pack 前。pack 时 `unprocessing.py` 会按当前 sample 的 CFA pattern 取 offsets，hflip 后再 pack 才能保证 packed 通道和 `isp_params["cfa_pattern"]` 一致。

### 3.2 2x2 downsample 定义

RGB：

```text
rgb_half[i, j] = mean(rgb_even[2i:2i+2, 2j:2j+2, :])
```

Depth：

```text
block_depth = depth_even[2i:2i+2, 2j:2j+2]
block_valid = valid_even[2i:2i+2, 2j:2j+2]

if any(block_valid):
  depth_half[i, j] = mean(block_depth[block_valid])
  valid_half[i, j] = True
else:
  depth_half[i, j] = 0
  valid_half[i, j] = False
```

当前 `_imagenet_normalize_rgb_tensor(..., target_hw)` 内部用 `cv2.INTER_AREA`。当输入严格是 `2H x 2W`、输出严格是 `H x W` 时，OpenCV area resize 在数学上等价于 2x2 block mean。

为避免实现歧义，halfres 分支二选一即可：

```text
方案 A：新增显式 _downsample_rgb_2x2_area_from_even_fullres()，用 reshape/block mean 实现；
方案 B：复用 _imagenet_normalize_rgb_tensor()，但调用前必须 assert 输入 shape 正好是 2x target_hw，并在代码注释里说明 INTER_AREA 的 2x 等价性。
```

Depth/valid 不能复用旧 `cv2.INTER_NEAREST` 路径。`INTER_NEAREST` 仅保留给旧 `synthetic_packed_bayer_4ch`；新 `synthetic_packed_bayer_4ch_halfres` 必须使用上面的 2x2 valid-mean 分支。

### 3.3 Geometry metadata

`include_geometry=True` 时必须记录：

```json
{
  "original_hw": [375, 1242],
  "even_fullres_hw": [374, 1242],
  "fullres_even_policy": "crop_bottom_to_even",
  "cropped_bottom_rows": 1,
  "crop_box": [0, 0, 374, 1242],
  "crop_box_format": "h_start_w_start_h_end_w_end",
  "crop_box_semantics": "source_fullres_to_even_fullres",
  "packed_hw": [187, 621],
  "rgb_input_space": "halfres_2x2_area",
  "depth_target_space": "halfres_2x2_valid_mean",
  "hflip_applied": false
}
```

不要再记录旧模式里的 `resized_hw=1024x3391` 或 `crop_box=1024x1920`。保留 `crop_box` 是为了兼容 `viz_dump.py` / cached dataset 等消费者，但它必须表达 source fullres 到 even fullres 的 bottom-crop。

注意当前 `_crop_hwew()` 使用 half-open `[h_start, w_start, h_end, w_end]`。因此 bottom-crop 最后一行的等价 crop box 是 `[0, 0, 374, 1242]`，不是 `[1, 0, 375, 1242]`；后者会裁掉顶部第一行。

### 3.4 RGB baseline 可视化路径

修改：

```text
VKITTI2Raw.build_rgb_baseline_input(...)
CachedVKITTI2Raw.build_rgb_baseline_input(...)
```

新模式下不能再走：

```python
_resize_rgb_short_edge(image, short_edge=self.fullres_size[0])
_crop_hwew(image, geometry["crop_box"])
```

`375 x 1242` 用 short edge `374` resize 会得到约 `374 x 1238`，与 `374 x 1242` 不一致；如果 geometry 没有旧 crop_box 还会直接崩。

halfres baseline 分支固定为：

```text
1. 读 source RGB 原图。
2. 按 geometry/fullres_even_policy 做 bottom-crop 到 even fullres。
3. 如果 geometry["hflip_applied"] 为 true，则在 fullres 阶段做同样 hflip。
4. 从 even fullres 做 2x2 area downsample 到 target_hw。
5. 返回 ImageNet normalized tensor 和 rgb_preview。
```

这条路径是 `viz_dump.py` train-viz RGB baseline 的依赖项，必须在 formal run 出图前 smoke。

### 3.5 include_geometry 开关和消费者审计

训练 DataLoader 主路径不需要每个 batch 都带 geometry，但所有 fixed-source viz/eval/cache 路径只要会调用 `build_rgb_baseline_input()`，就必须能拿到 geometry。

实现要求：

```text
1. VKITTI2Raw 保留 build_sample(..., include_geometry=True) 显式参数。
2. 新增 dataset 构造参数 include_geometry=False，使 __getitem__ 在需要时也能返回 geometry；
   formal training 默认 False，避免无用 batch metadata。
3. viz_dump.py 现有 fixed-source 路径会探测 build_sample 的 include_geometry 参数并传 True；
   修改后必须 smoke 该路径。
4. cache_vkitti2_pseudoraw.py 生成 cache 时必须 include_geometry=True，
   cached_vkitti2_raw.py 必须能读取并按新 geometry 还原 RGB baseline。
```

---

## 4. 模型路径确认

`RawResidualDAV2` 的输入保持 dict：

```python
out = model({
    "image": image_rgb_norm,  # [B,3,187,621]
    "raw": raw4,              # [B,4,187,621]
    "valid_mask": valid_mask, # [B,187,621]
})
```

DAV2 path：

```text
image [B,3,187,621]
-> CenterPadCropAdapter dynamic pad
-> pad to patch multiple, expected 196 x 630
-> frozen DAV2
-> center crop back to 187 x 621
```

RAW path：

```text
raw4 [B,4,187,621]
-> packed_bayer_to_base_rgb(raw4)
-> RamCore3.forward_with_features()
-> residual head
```

`CenterPadCropAdapter` 已支持 dynamic shape；不要再把 sensor size 固定理解为 `512x960`。

---

## 5. Smoke 测试

所有 smoke 产物路径必须包含 `codex_smoke`，成功后只删除这些临时产物。

### 5.1 Dataset smoke

```bash
conda run --live-stream -n dav3 python - <<'PY'
from foundation.engine.datasets import VKITTI2Raw

ds = VKITTI2Raw(
    filelist_path="finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt",
    mode="val",
    size=(187, 621),
    raw_storage_format="synthetic_packed_bayer_4ch_halfres",
    fullres_even_policy="crop_bottom_to_even",
    rgb_input_space="halfres_2x2_area",
    depth_target_space="halfres_2x2_valid_mean",
    randomize_unprocessing=False,
    include_rgb_input=True,
    include_rgb_preview=True,
)
s = ds.build_sample(0, include_geometry=True)
print("raw", tuple(s["raw"].shape), s["raw"].dtype, float(s["raw"].min()), float(s["raw"].max()))
print("image", tuple(s["image"].shape), s["image"].dtype, float(s["image"].mean()))
print("depth", tuple(s["depth"].shape), s["depth"].dtype)
print("valid", tuple(s["valid_mask"].shape), int(s["valid_mask"].sum()))
print("geometry", s["geometry_params"])
PY
```

预期：

```text
raw:   (4,187,621), float32
image: (3,187,621), float32, ImageNet normalized
depth: (187,621), float32
valid: (187,621), bool
geometry original_hw=[375,1242]
geometry even_fullres_hw=[374,1242]
geometry cropped_bottom_rows=1
```

### 5.2 Forward smoke

```bash
SMOKE_DIR=plans/0524_new/codex_smoke_vkitti_halfres_packedraw_forward

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_raw_residual.py \
  --input-domain raw4 \
  --model-input-tensor raw \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end raw_to_base_rgb_ram3 \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source ffm_mid \
  --residual-alpha 0.5 \
  --vkitti-unprocessing-preset sensor_linear_dual \
  --randomize-unprocessing \
  --hflip-prob 0.5 \
  --d0-sign 1 \
  --epochs 1 \
  --bs 8 \
  --accum-steps 1 \
  --num-workers 0 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --log-interval 1 \
  --max-train-steps 2 \
  --max-val-samples 4 \
  --save-path "${SMOKE_DIR}/exp" \
  --heavy-save-path "${SMOKE_DIR}/heavy"
```

预期：

```text
[BATCH] image=(8,3,187,621) raw=(8,4,187,621) depth=(8,187,621) valid=(8,187,621)
训练 2 step 成功
峰值显存显著低于旧 512x960 版本
```

显存预期：

```text
pixel count: 512x960=491520 -> 187x621=116127，约 4.2x 缩减
DAV2 token: 37x69=2553 -> 14x45=630，约 4.0x 缩减
attention QK/V 量级约 16x 缩减
```

`bs=8 / vits / bf16` 在 A5000 上不应 OOM。若 forward smoke 或 formal M2 仍在首个 batch OOM，优先视为代码路径仍走旧尺寸、重复保留大 tensor、或 pad/crop bug，不要立刻用 `bs=4 acc=2` 掩盖。

---

## 6. D0 sign / quality baseline

正式 M2 前重新跑 sign check，因为输入尺寸和 RGB baseline space 已变。该工具必须先按 §2.2 扩展参数面：

```bash
conda run --live-stream -n dav3 python foundation/tools/check_vkitti_dav2_sign.py \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --max-samples 64 \
  --output plans/0524_new/codex_smoke_vkitti_halfres_d0_sign_summary.json
```

sign check summary 是 smoke 产物；确认 sign 写入 formal log 后可以删除。

仅确认 sign 不够。`vits + 187x621` 的 DAV2 输入 token 明显减少，必须同步记录 D0 质量基线，避免 halfres D0 已严重退化而残差训练无效。

扩展 `check_vkitti_dav2_sign.py` 或新增同级检查工具，输出至少：

```text
halfres_187x621_D0_abs_rel_mean_over_64
halfres_187x621_D0_d1_mean_over_64
halfres_187x621_D0_silog_mean_over_64
fullres_original_375x1242_D0_abs_rel_mean_over_64
fullres_original_375x1242_D0_d1_mean_over_64
fullres_original_375x1242_D0_silog_mean_over_64
```

质量指标应使用与训练 eval 一致的 D0 对齐/metric 计算协议，例如复用 `affine_align_disp` + `compute_metrics`，并在 summary JSON 里记录协议名。

如果 `187x621` halfres D0 相比原始 `375x1242` 无 crop DAV2 输出严重退化，不继续直接跑 10 epoch residual。先调整策略，例如：

```text
--input-height 374
--input-width 1242
DAV2/RGB baseline 使用 even fullres
packed raw 仍是 187x621
RAM core 输出或 adapter 在融合前上采样到 374x1242
```

---

## 7. 正式 M2 修订

旧失败 run：

```text
0524_2100_vkitti_m2_ffm_mid_residual_vits_bs8_e10
```

该 run 同时存在两个问题：

```text
1. 使用错误尺寸 512x960。
2. 在第一个 batch OOM。
```

不要把它作为正式结果。除非用户明确要求，否则不要删除旧失败产物。

### 7.1 正式 run name

正式实验仍按 `MMDD_HHMM` 前缀命名，建议 suffix：

```text
vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10
```

### 7.2 正式参数

```bash
RUN_SUFFIX="vkitti_m2_ffm_mid_residual_vits_halfraw187x621_bs8_e10"

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_raw_residual.py \
  --input-domain raw4 \
  --model-input-tensor raw \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end raw_to_base_rgb_ram3 \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source ffm_mid \
  --residual-alpha 0.5 \
  --vkitti-unprocessing-preset sensor_linear_dual \
  --randomize-unprocessing \
  --hflip-prob 0.5 \
  --d0-sign <SIGN_FROM_HALFRES_CHECK> \
  --epochs 10 \
  --bs 8 \
  --accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --num-workers 4 \
  --log-interval 100 \
  --save-interval 1 \
  --eval-interval 1 \
  --save-best-checkpoint \
  --save-path finetune_stf/exp/<MMDD_HHMM>_${RUN_SUFFIX} \
  --heavy-save-path /mnt/drive/3333_raw/0000_exp_ckpt/<MMDD_HHMM>_${RUN_SUFFIX}
```

原则上不要先降 batch。按 §5.2 的显存估算，`bs=8 / vits / bf16 / 187x621` 在 A5000 上 OOM 应视为代码 bug，先检查是否仍走旧 `512x960` / fullres resize、是否保存了异常大的中间 tensor、以及 DAV2 adapter 是否 pad 到错误尺寸。

只有定位确认不是代码路径问题后，才允许临时改为：

```text
--bs 4
--accum-steps 2
```

并且 run suffix 必须明确体现：

```text
halfraw187x621_bs4acc2_e10
```

---

## 8. M1 / M3 修订

M2 诊断通过后再跑 M1/M3。

除下面两项外，其余参数必须与 M2 完全一致：

```text
M1:
  --residual-feature-source x3
  suffix vkitti_m1_x3_residual_vits_halfraw187x621_bs8_e10

M3:
  --residual-feature-source x3_ffm_mid
  suffix vkitti_m3_x3_ffm_mid_residual_vits_halfraw187x621_bs8_e10
```

如果 M2 使用 `bs4acc2`，M1/M3 也必须使用同一 effective batch 设置，并在 suffix 中同步标注。

---

## 9. 汇总要求

修改：

```text
foundation/tools/summarize_vkitti_mseries.py
```

该脚本当前只抽取 feature/source 和指标列，必须新增字段抽取并写入 markdown/json/csv 输出：

```text
raw_storage_format
fullres_even_policy
rgb_input_space
depth_target_space
input_height
input_width
source_original_hw
even_fullres_hw
packed_hw
```

字段来源：

```text
raw_storage_format / fullres_even_policy / rgb_input_space / depth_target_space:
  run_dir/config.json

input_height / input_width:
  run_dir/config.json

source_original_hw / even_fullres_hw / packed_hw:
  优先读 run_dir/config.json 中的 resolved dataset geometry；
  source_original_hw 对应 geometry_params.original_hw；
  如果旧 run 没有这些字段，写 n/a，不做路径名推断。
```

旧 `512x960` run 不得混入 half-size packed raw 的正式 M 系列汇总。

### 9.1 Cache 路径要求

修改或标注：

```text
foundation/tools/cache_vkitti2_pseudoraw.py
foundation/engine/datasets/cached_vkitti2_raw.py
```

旧 `512x960` cache 对新 halfres 模式无效，不能复用。若后续要跑 cached 路径：

```text
1. 另起 cache 目录，目录名或 config 必须明确包含 halfres / raw_storage_format。
2. cache config.json 必须记录 raw_storage_format / fullres_even_policy /
   rgb_input_space / depth_target_space / source_original_hw / even_fullres_hw / packed_hw。
3. cache manifest/payload 必须保存 geometry_params。
4. CachedVKITTI2Raw 加载时校验请求 size 和 cache size；
   如果 raw_storage_format 不匹配或缺失，直接报错，不自动兼容。
5. CachedVKITTI2Raw.build_rgb_baseline_input() 必须按 §3.4 的 halfres baseline 分支处理。
```

---

## 10. 完成检查表

```text
[ ] dataset 支持 synthetic_packed_bayer_4ch_halfres
[ ] train_vkitti2_raw_residual.py parse_args/validate_args 已加新参数 + 扩 raw_storage_format choices
[ ] train_vkitti2_raw_residual.py build_loaders 已透传新参数到 VKITTI2Raw
[ ] check_vkitti_dav2_sign.py parse_args/validate_args 已加新参数并透传到 VKITTI2Raw
[ ] bottom crop 1 row 到 374x1242
[ ] new halfres 模式不再使用 self.fullres_size 作为 fullres 来源
[ ] hflip 后 / pack 前断言 even fullres H,W 为偶数
[ ] raw shape = (4,187,621)
[ ] image shape = (3,187,621)
[ ] depth / valid shape = (187,621)
[ ] depth / valid halfres 使用 2x2 valid mean，不复用 INTER_NEAREST
[ ] geometry metadata 记录 crop_bottom_to_even，并保留 source->even 的 crop_box=[0,0,374,1242]
[ ] build_rgb_baseline_input 已加 halfres 分支，viz_dump 可正常出图
[ ] include_geometry 路径已覆盖 viz_dump / cache fixed-source consumer
[ ] train/eval args 显式记录 raw_storage_format / fullres_even_policy / rgb_input_space / depth_target_space
[ ] summarize_vkitti_mseries.py 已读取并输出新语义字段
[ ] cached_vkitti2_raw / cache_vkitti2_pseudoraw 不复用旧 512x960 cache，halfres cache 有独立 config 标注
[ ] D0 sign check 用 half-size RGB/DAV2 input 重跑
[ ] D0 quality baseline 已记录 abs_rel / d1 / silog，并与原始 375x1242 DAV2 输出对照
[ ] M2 formal 用 halfraw187x621 新 run name 重跑
[ ] M2 诊断通过后再跑 M1/M3
```
