# RAMCore3 + Frozen DAV2 RAW Residual：VKITTI C1/C2 控制实验执行计划

## 0. 本轮范围

本文件只覆盖主计划中的 C1 / C2：

```text
C1: frozen DAV2 + RGB residual branch
C2: frozen DAV2 + D0-only residual branch
```

C3 是 parameter-matched plain U-Net RAW residual，需要单独决定 U-Net 参数匹配方式，本文件不展开。

本轮目标不是重新定义 M 系列，而是在 M2 已确认有效后，用同一数据、同一 halfres 几何、同一训练长度和同一 evaluation protocol 判断：

```text
M2/M3 的提升是否只是来自 RGB residual refinement
M2/M3 的提升是否只是来自 D0 post-processing
```

启动条件：

```text
M2 final_abs_rel 优于同 run 中的 D0_halfres，或至少 region metrics 有明确收益，即可启动 C1/C2。
M1/M3 不要求先完成；后续如果 M1/M3 完成，summary 再把它们并入对照表。
如果 M2 不优于 D0_halfres，先不要启动 C 系列，优先 debug residual formulation。
```

---

## 1. 必须继承的 halfres 设置

必须继承 `RAMCore3_DAV2_RAW_residual_vkitti_halfres_packedraw_revision_plan.md` 的分辨率决策。不要再使用旧的 `512x960` 路径。

```text
VKITTI2 original RGB/depth: 375 x 1242
even fullres policy:       crop bottom 1 row
even fullres RGB/depth:    374 x 1242
model/grid input:          187 x 621
rgb_input_space:           halfres_2x2_area
depth_target_space:        halfres_2x2_valid_mean
```

C1/C2 不使用 RAW cue，但必须使用同一 halfres RGB 和 halfres depth target：

```text
image:      ImageNet-normalized halfres RGB, shape [3,187,621]
D0:         frozen DAV2(image), center-padded/cropped back to [187,621]
depth:      2x2 valid-mean halfres GT, shape [187,621]
valid_mask: halfres valid mask, shape [187,621]
```

已有 halfres D0 sign / quality baseline 只能作为参考：

```text
plans/0524_new/0524_2140_vkitti_halfres_d0_sign_quality_baseline.json
recommended_d0_sign: 1
halfres_187x621_D0_abs_rel_mean_over_64: 0.17075571896599684
```

正式 C 系列 queue 必须默认在新 control dataset 上重新跑一次 sign check，再启动 formal。原因是 C 系列会新建 dataset 路径，即使几何语义相同，也必须确认新 RGB pipeline 输出的 D0 与 inverse-depth 方向一致。

如果用户显式传入 `D0_SIGN=1` 或 `D0_SIGN=-1`，queue 可以跳过自动推断；默认必须是：

```text
RUN_SIGN_CHECK=1
D0_SIGN=""
```

如果 halfres 参数、pretrained checkpoint、split 或 encoder 有任何变化，也必须重新跑 sign check。

---

## 2. 实验语义定义

### 2.1 C1: RGB residual branch

C1 只允许 residual head 使用 RGB/D0 信息，不允许使用 RAW、RamCore3、x3、ffm_mid 或 unprocessing 参数。

这里的 `image_rgb_norm` 是 ImageNet 标准化后的 RGB，和 frozen DAV2 实际接收的输入完全一致。C1 因此是一个偏严格的 RGB residual refinement baseline，而不是 `[0,1]` 普通 RGB baseline。

```text
DAV2 path:
  image_rgb_norm -> frozen DAV2 -> D0

residual head input:
  concat(D0_norm, image_rgb_norm)

channels:
  D0_norm:        1
  image_rgb_norm: 3
  total:          4

trainable:
  ResidualGateHead only

frozen / unused:
  DAV2 frozen + eval + no_grad
  RamCore3 unused
  RAW unused
```

实验含义：

```text
如果 C1 接近或超过 M2/M3，说明主要收益可能来自普通 RGB residual refinement。
如果 M2/M3 在 high-error / dark / saturated / boundary region 明显超过 C1，才支持 RAW-like cue 有额外贡献。
```

### 2.2 C2: D0-only residual branch

C2 只允许 residual head 使用 D0_norm，不允许使用 RGB feature、RAW、RamCore3、x3、ffm_mid 或 unprocessing 参数。

```text
DAV2 path:
  image_rgb_norm -> frozen DAV2 -> D0

residual head input:
  D0_norm

channels:
  D0_norm: 1

trainable:
  ResidualGateHead only

frozen / unused:
  DAV2 frozen + eval + no_grad
  RGB only used to compute D0 and region diagnostics
  RamCore3 unused
  RAW unused
```

实验含义：

```text
如果 C2 接近或超过 M2/M3，说明 residual branch 可能只是学到 D0 post-processing。
如果 C1 > C2 但 M2/M3 > C1，说明 RGB refinement 有用，但 RAW-like cue 仍有额外贡献。
```

---

## 3. 需要补齐的代码接口

当前代码不能直接复用：

```text
foundation/tools/train_vkitti2_raw_residual.py:
  --input-domain choices=["raw4"]
  --model-input-tensor choices=["raw"]
  --front-end choices=["raw_to_base_rgb_ram3"]
  --vkitti-unprocessing-preset required
  --randomize-unprocessing / --no-randomize-unprocessing required
  --residual-feature-source choices=["ffm_mid","x3","x3_ffm_mid"]

foundation/engine/datasets/vkitti2_raw.py:
  RAW_STORAGE_FORMAT_CHOICES does not include not_applicable
  validate_vkitti_raw_semantics is RAW-specific

foundation/engine/models/raw_residual_dav2.py:
  RESIDUAL_FEATURE_SOURCES = ("ffm_mid","x3","x3_ffm_mid")
```

正式 C1/C2 前必须新建控制分支接口。不要为了复用旧入口而把 C1/C2 伪装成 `raw4 + raw_to_base_rgb_ram3`，因为这会让 `input_domain/front_end/raw_storage_format` 的语义变脏。

### 3.1 必须新增训练入口

必须新增：

```text
foundation/tools/train_vkitti2_residual_control.py
```

不要 sed 改造 `foundation/tools/train_vkitti2_raw_residual.py`。新入口可以复制 M 系列中 loss/eval/checkpoint/logging 的安全实现，但 CLI 和 validate stage 必须独立表达 C1/C2 的实验语义。

必须支持：

```text
--experiment-id C1|C2
--input-domain rgb
--model-input-tensor image
--dataset-geometry-mode vkitti2_even_fullres_halfres_2x2
--front-end dav2_rgb_frozen
--fullres-even-policy crop_bottom_to_even
--rgb-input-space halfres_2x2_area
--depth-target-space halfres_2x2_valid_mean
--raw-storage-format not_applicable
--residual-feature-source rgb|d0
--d0-sign -1|1
```

新入口内部必须独立定义：

```text
CONTROL_RAW_STORAGE_CHOICES = ("not_applicable",)
CONTROL_FRONT_END_CHOICES = ("dav2_rgb_frozen",)
CONTROL_FEATURE_SOURCES = ("rgb", "d0")
```

禁止为了 C 系列修改：

```text
foundation/engine/datasets/vkitti2_raw.py::RAW_STORAGE_FORMAT_CHOICES
foundation/engine/datasets/vkitti2_raw.py::validate_vkitti_raw_semantics
foundation/engine/models/raw_residual_dav2.py::RESIDUAL_FEATURE_SOURCES
```

集中校验规则：

```text
C1:
  experiment_id == C1
  input_domain == rgb
  model_input_tensor == image
  raw_storage_format == not_applicable
  front_end == dav2_rgb_frozen
  residual_feature_source == rgb

C2:
  experiment_id == C2
  input_domain == rgb
  model_input_tensor == image
  raw_storage_format == not_applicable
  front_end == dav2_rgb_frozen
  residual_feature_source == d0

Both:
  dataset_geometry_mode == vkitti2_even_fullres_halfres_2x2
  fullres_even_policy == crop_bottom_to_even
  rgb_input_space == halfres_2x2_area
  depth_target_space == halfres_2x2_valid_mean
  input_height == 187
  input_width == 621
```

如果 parser 仍保留 RAW/unprocessing 形态的参数，则 C1/C2 必须要求它们是 `not_applicable`，不能接受 `sensor_linear_dual` 这种 active RAW value。

### 3.2 Dataset 要求

必须新增 control dataset 文件：

```text
foundation/engine/datasets/vkitti2_halfres_rgb_depth.py
class VKITTI2HalfresRGBDepth
```

它必须和 M 系列 halfres 路径使用同一几何处理：

```text
1. 读取原始 VKITTI2 RGB/depth。
2. bottom-crop 到 374x1242。
3. train 时执行 hflip_prob=0.5，val 时 hflip_prob=0.0。
4. RGB 用 2x2 area downsample 到 187x621。
5. depth/valid 用 2x2 valid mean downsample 到 187x621。
6. image 输出 ImageNet normalized tensor。
7. rgb_preview 输出 [0,1] tensor，用于 dark/saturated region diagnostics。
8. 不执行 unprocessing，不生成 raw，不构造 isp_params。
9. 不得 import foundation.engine.transforms.unprocessing* 符号，避免引入空 unprocessing pipeline。
```

`describe_geometry()` 必须记录：

```text
source_original_hw: [375,1242]
even_fullres_hw: [374,1242]
cropped_bottom_rows: 1
crop_box: [0,0,374,1242]
packed_hw: not_applicable
input_hw: [187,621]
rgb_input_space: halfres_2x2_area
depth_target_space: halfres_2x2_valid_mean
```

### 3.3 Model 要求

必须新增 model 文件：

```text
foundation/engine/models/dav2_residual_control.py
class DAV2ResidualControl
CONTROL_FEATURE_SOURCES = ("rgb", "d0")
```

不要重构 `RawResidualDAV2`，因为 M2/M1/M3 formal 路径已经依赖该文件。C 系列 model 行为必须等价于：

```text
C1 head_input = concat(D0_norm.unsqueeze(1), image_rgb_norm)
C2 head_input = D0_norm.unsqueeze(1)
```

输入通道：

```text
rgb: 4
d0:  1
```

训练约束：

```text
DAV2 always frozen, eval mode, no_grad
RamCore3 not constructed or not used
ResidualGateHead trainable
```

日志约束：

```text
trainable_params 必须写入日志和 config。
C1/C2 日志不能要求 x3_mean / ffm_mid_mean。
如果 summary 里保留 x3/ffm_mid 列，C1/C2 写 n/a。
```

C1/C2/M2 trainable parameter count 应处于同一量级，方便后续说明 parameter budget 基本匹配。差异主要来自 residual head 第一层：

```text
C2: Conv2d(1 -> 64)
C1: Conv2d(4 -> 64)
M2: Conv2d(65 -> 64)
```

### 3.4 Loss / eval 要求

Loss 和 M 系列完全一致：

```text
pred = D0_norm + gate * delta
target = inverse-depth target, per-sample robust normalized
L_depth + 0.5 L_grad + 0.1 L_keep + 0.01 L_res + 0.005 L_gate + 0.05 L_gate_sup
```

Eval 和 M 系列完全一致：

```text
per-image affine alignment
overall metrics: AbsRel, SqRel, RMSE, RMSE_log, log10, SILog, d1/d2/d3
region metrics: boundary, D0 high-error, far50, dark, saturated
diagnostics: mean_gate, max_gate, mean_abs_delta, mean_abs_gate_delta
```

---

## 4. Smoke tests

Smoke 输出路径必须包含 `codex_smoke`，成功后只删除这些 smoke artifacts。

### 4.1 Dataset shape smoke

先单独验证新 control dataset，避免训练入口启动后才发现 shape 或 key 语义错误。

新增：

```text
foundation/tools/smoke_vkitti2_residual_control_dataset.py
```

必须断言：

```text
raw key 不存在
isp_params key 不存在
image shape = (3,187,621)
depth shape = (187,621)
valid_mask shape = (187,621)
rgb_preview shape = (3,187,621)
geometry original_hw = [375,1242]
geometry even_fullres_hw = [374,1242]
```

示例命令：

```bash
SMOKE_ROOT=plans/0524_new/codex_smoke_vkitti_cseries_dataset_shape

conda run --live-stream -n dav3 python foundation/tools/smoke_vkitti2_residual_control_dataset.py \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --hflip-prob 0.5 \
  --output "${SMOKE_ROOT}/dataset_shape_summary.json"
```

成功后删除：

```bash
rm -rf plans/0524_new/codex_smoke_vkitti_cseries_dataset_shape
```

### 4.2 C2 smoke

C2 是最小 residual branch，建议先跑它验证 D0-only path、loss、eval 和日志。

```bash
SMOKE_ROOT=plans/0524_new/codex_smoke_vkitti_cseries_c2_d0_halfd0_187x621

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_residual_control.py \
  --experiment-id C2 \
  --input-domain rgb \
  --model-input-tensor image \
  --dataset-geometry-mode vkitti2_even_fullres_halfres_2x2 \
  --raw-storage-format not_applicable \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end dav2_rgb_frozen \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source d0 \
  --residual-alpha 0.5 \
  --d0-sign 1 \
  --hflip-prob 0.5 \
  --epochs 1 \
  --bs 8 \
  --accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --num-workers 0 \
  --log-interval 1 \
  --save-interval 1 \
  --eval-interval 1 \
  --max-train-steps 2 \
  --max-val-samples 4 \
  --save-path "${SMOKE_ROOT}/exp" \
  --heavy-save-path "${SMOKE_ROOT}/heavy"
```

成功后删除：

```bash
rm -rf plans/0524_new/codex_smoke_vkitti_cseries_c2_d0_halfd0_187x621
```

### 4.3 C1 smoke

```bash
SMOKE_ROOT=plans/0524_new/codex_smoke_vkitti_cseries_c1_rgb_halfrgb_187x621

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_residual_control.py \
  --experiment-id C1 \
  --input-domain rgb \
  --model-input-tensor image \
  --dataset-geometry-mode vkitti2_even_fullres_halfres_2x2 \
  --raw-storage-format not_applicable \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end dav2_rgb_frozen \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source rgb \
  --residual-alpha 0.5 \
  --d0-sign 1 \
  --hflip-prob 0.5 \
  --epochs 1 \
  --bs 8 \
  --accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --num-workers 0 \
  --log-interval 1 \
  --save-interval 1 \
  --eval-interval 1 \
  --max-train-steps 2 \
  --max-val-samples 4 \
  --save-path "${SMOKE_ROOT}/exp" \
  --heavy-save-path "${SMOKE_ROOT}/heavy"
```

成功后删除：

```bash
rm -rf plans/0524_new/codex_smoke_vkitti_cseries_c1_rgb_halfrgb_187x621
```

失败时保留 smoke 目录并报告路径。

---

## 5. 正式 queue 脚本

新增：

```text
finetune_stf/scripts/formal/0524_run_vkitti_cseries_residual_controls_queue.sh
```

职责：

```text
1. 顶层生成一次 RUN_TIMESTAMP=$(date +%m%d_%H%M)。
2. 创建新的 tmux session，不复用已有 session。
3. 检查 M2 已确认有效；如果用户显式设置 SKIP_M2_GATE=1，必须在 queue log 中记录。
4. 可选运行 dataset/C2/C1 smoke，默认 RUN_SMOKE=1。
5. 默认在 VKITTI2HalfresRGBDepth 上跑 D0 sign check：RUN_SIGN_CHECK=1 且 D0_SIGN=""。
6. 如果用户显式设置 D0_SIGN=1 或 D0_SIGN=-1，允许跳过自动 sign 推断。
7. 如果 save_path 或 heavy_save_path 已存在，直接报错，不覆盖。
8. 正式跑 C2，再跑 C1。
9. 写 queue log 到 finetune_stf/logs/<session>.queue.log。
10. 每个正式 run 写独立 train log 到 finetune_stf/logs/<run_name>.tmux.log。
```

需要新增 control sign check 工具，或在 queue 内调用等价逻辑：

```text
foundation/tools/check_vkitti_control_dav2_sign.py
```

该工具必须直接使用 `VKITTI2HalfresRGBDepth`，计算 `corr(D0, 1/depth)` 并输出：

```text
recommended_d0_sign
halfres_187x621_D0_abs_rel_mean_over_64
halfres_187x621_D0_d1_mean_over_64
halfres_187x621_D0_silog_mean_over_64
```

不要通过 `VKITTI2Raw + dummy preset` 临时复用 sign check，因为那会重新触发 RAW/unprocessing 路径。

§6/§7 的正式命令是 §3.1 参数面的实际取值，二者必须逐字一致。

tmux 启动后必须报告：

```text
tmux session: <SESSION>
queue log: finetune_stf/logs/<SESSION>.queue.log
attach: tmux attach -t <SESSION>
monitor: tail -f finetune_stf/logs/<SESSION>.queue.log
```

---

## 6. 正式 C2 参数

命名规则：

```text
M 系列已使用 halfraw187x621，本计划不改 M 系列 run name。
C 系列没有 RAW 输入，因此用 halfd0_187x621 / halfrgb_187x621 明确控制输入来源。
```

正式 run name：

```text
<MMDD_HHMM>_vkitti_c2_d0only_residual_vits_halfd0_187x621_bs8_e10
```

正式参数：

```bash
RUN_SUFFIX="vkitti_c2_d0only_residual_vits_halfd0_187x621_bs8_e10"
RUN_NAME="${RUN_TIMESTAMP}_${RUN_SUFFIX}"

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_residual_control.py \
  --experiment-id C2 \
  --input-domain rgb \
  --model-input-tensor image \
  --dataset-geometry-mode vkitti2_even_fullres_halfres_2x2 \
  --raw-storage-format not_applicable \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end dav2_rgb_frozen \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source d0 \
  --residual-alpha 0.5 \
  --d0-sign "${D0_SIGN}" \
  --hflip-prob 0.5 \
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
  --save-path "finetune_stf/exp/${RUN_NAME}" \
  --heavy-save-path "/mnt/drive/3333_raw/0000_exp_ckpt/${RUN_NAME}"
```

---

## 7. 正式 C1 参数

正式 run name：

```text
<MMDD_HHMM>_vkitti_c1_rgb_residual_vits_halfrgb_187x621_bs8_e10
```

正式参数：

```bash
RUN_SUFFIX="vkitti_c1_rgb_residual_vits_halfrgb_187x621_bs8_e10"
RUN_NAME="${RUN_TIMESTAMP}_${RUN_SUFFIX}"

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_residual_control.py \
  --experiment-id C1 \
  --input-domain rgb \
  --model-input-tensor image \
  --dataset-geometry-mode vkitti2_even_fullres_halfres_2x2 \
  --raw-storage-format not_applicable \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end dav2_rgb_frozen \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source rgb \
  --residual-alpha 0.5 \
  --d0-sign "${D0_SIGN}" \
  --hflip-prob 0.5 \
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
  --save-path "finetune_stf/exp/${RUN_NAME}" \
  --heavy-save-path "/mnt/drive/3333_raw/0000_exp_ckpt/${RUN_NAME}"
```

---

## 8. 汇总脚本

新增或泛化：

```text
foundation/tools/summarize_vkitti_residual_series.py
```

输入：

```text
required:
  finetune_stf/exp/<M2_RUN>
  finetune_stf/exp/<C1_RUN>
  finetune_stf/exp/<C2_RUN>

optional:
  finetune_stf/exp/<M1_RUN>
  finetune_stf/exp/<M3_RUN>
```

M1/M3 不存在时 summary 不能失败；缺失行写 `n/a` 或直接省略，并在 summary metadata 中记录 `missing_optional_runs`。M 系列旧 config 中没有 `experiment_id` 时，summary 用 `method` 列复制出 `experiment_id=M1/M2/M3`。

输出：

```text
plans/0524_new/vkitti_residual_m_vs_c_summary_<MMDD_HHMM>.md
plans/0524_new/vkitti_residual_m_vs_c_summary_<MMDD_HHMM>.json
plans/0524_new/vkitti_residual_m_vs_c_summary_<MMDD_HHMM>.csv
```

表格字段至少包含：

```text
method
experiment_id
residual_feature_source
input_domain
model_input_tensor
raw_storage_format
front_end
dataset_geometry_mode
fullres_even_policy
rgb_input_space
depth_target_space
input_height
input_width
source_original_hw
even_fullres_hw
packed_hw
trainable_params
best_epoch
D0_abs_rel
final_abs_rel
delta_abs_rel
D0_d1
final_d1
delta_d1
boundary_abs_rel
high_error_abs_rel
far50_abs_rel
dark_abs_rel
saturated_abs_rel
mean_gate
mean_abs_delta
mean_abs_gate_delta
```

汇总脚本必须校验：

```text
所有 run 的 split 相同
所有 run 的 encoder 相同
所有 run 的 input_height/input_width 都是 187/621
所有 run 的 fullres_even_policy/rgb_input_space/depth_target_space 相同
所有 run 的 min_depth/max_depth/eval protocol 相同
旧 512x960 run 不得混入
```

---

## 9. 结果判读

下面的 `M2/M3` 表示当前已经完成的 M 系列主候选。若 M3 尚未完成，先用 M2 做 provisional 判读；M3 完成后再更新 summary 和结论。

### 9.1 支持 RAW-like cue 的结果

```text
M2/M3 overall > C1
M2/M3 high-error region > C1
M2/M3 dark/saturated region > C1
M2/M3 boundary region >= C1
M2/M3 > C2
```

可以继续主张：

```text
RAW-like representation provides residual cues beyond RGB residual refinement and D0-only post-processing.
```

### 9.2 只能支持 RGB refinement 的结果

```text
C1 ≈ M2/M3
C1 > C2
```

说明 trainable residual head 加 RGB 本身已经解释大部分收益，论文 claim 必须降低。

### 9.3 只能支持 D0 post-processing 的结果

```text
C2 ≈ M2/M3
```

说明 residual branch 可能主要是在学 frozen DAV2 输出的后处理，而不是 RAW cue。

### 9.4 危险结果

```text
C1/C2 train loss 下降但 val 不提升
mean_gate 长期接近 1
mean_abs_delta 接近 residual_alpha
C2 显著超过 C1 和 M2/M3
```

优先检查：

```text
d0_sign
D0_norm / GT norm 是否一致
gate supervision 是否过强
val affine alignment 是否和 M 系列一致
halfres geometry 是否误回到 512x960 或 fullres resize/crop
```

---

## 10. 完成检查表

```text
[ ] C1/C2 control dataset 不生成 RAW，不执行 unprocessing。
[ ] control dataset 不 import foundation.engine.transforms.unprocessing*。
[ ] C1/C2 仍使用 crop_bottom_to_even + halfres_2x2_area + halfres_2x2_valid_mean。
[ ] input_height/input_width 固定为 187/621。
[ ] C1 residual head input 是 concat(D0_norm, image_rgb_norm)，channels=4。
[ ] C2 residual head input 是 D0_norm，channels=1。
[ ] DAV2 frozen + eval + no_grad。
[ ] RamCore3 在 C1/C2 中不构造或不使用。
[ ] CLI 明确记录 experiment_id/input_domain/model_input_tensor/front_end/residual_feature_source。
[ ] raw_storage_format 对 C1/C2 为 not_applicable，不接受 active RAW 参数。
[ ] 新 control dataset 单独跑过 shape smoke，且 raw/isp_params key 不存在。
[ ] 新 entry 写入的 config.json 里 raw_storage_format=not_applicable。
[ ] 新 dataset 上的 D0 sign check 默认执行，并写出 recommended_d0_sign。
[ ] train/eval loss 和 metrics 与 M 系列一致。
[ ] C2 smoke 成功并删除 codex_smoke 输出。
[ ] C1 smoke 成功并删除 codex_smoke 输出。
[ ] 正式 C2 10 epoch tmux 完成。
[ ] 正式 C1 10 epoch tmux 完成。
[ ] summary 脚本输出 M2/C1/C2 必需对照表，M1/M3 可选，并拒绝混入 512x960 run。
[ ] summary 脚本能正确显示 raw_storage_format=not_applicable。
```
