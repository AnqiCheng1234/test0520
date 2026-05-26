# RAMCore3 + Frozen DAV2 RAW Residual：VKITTI M 系列执行计划

## 0. 本轮决定

本轮只做 VKITTI2 上的 M 系列主方法验证，不跑原计划 4.1 / 4.2：

```text
不跑 B0: sRGB -> frozen DAV2
不跑 B1: raw4 -> base_rgb -> RamCore3 -> x3 -> frozen DAV2
先跑 M2 / M1 / M3
后续再跑 C1 / C2 / C3
```

训练设置固定为：

```text
dataset: VKITTI2
target: 真实 GT depth
batch_size: 8
epochs: 10
DAV2: frozen + eval mode + no_grad
RamCore3: trainable
ResidualGateHead: trainable
```

默认工作目录：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
conda activate dav3
```

正式实验必须用 tmux，实验名必须以服务器本地时间 `MMDD_HHMM` 开头。

---

## 1. 总体顺序

按下面顺序执行，不要跳步：

```text
Step A. 生成 VKITTI train/val split
Step B. 扩展 VKITTI2Raw dataset，使一个 sample 同时返回 image(rgb_norm) 和 raw
Step C. 新增 RawResidualDAV2 + ResidualGateHead
Step D. 新增 VKITTI residual 专用 train/eval 入口、D0 sign check 脚本、M 系列汇总脚本
Step E. 跑 smoke：dataset、forward、2-step train、短 val
Step F. 正式跑 M2
Step G. 看 M2 诊断；如果没有明显实现问题，再跑 M1 和 M3
Step H. 正式跑 M1 和 M3
Step I. 汇总 M 系列结果，决定后续 C 系列
```

---

## 2. Step A：生成 VKITTI train/val split

当前已有 split：

```text
finetune_stf/dataset/splits/vkitti2/train.txt
entries: 19559
```

当前没有 val split。本轮从这个 train.txt 中人为划出 10% 作为 validation。

### 2.1 新增 split 脚本

新增：

```text
foundation/tools/split_vkitti2_train_val.py
```

功能要求：

```text
input:  finetune_stf/dataset/splits/vkitti2/train.txt
output:
  finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt
  finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt
  finetune_stf/dataset/splits/vkitti2/split_mseries_seed42_summary.json

val_fraction: 0.10
seed: 42
stratify key: scene + condition + camera
```

路径解析规则：

```text
/mnt/drive/1111_new_works/VKITTI2/rgb/Scene20/sunset/frames/rgb/Camera_0/rgb_00560.jpg
scene     = Scene20
condition = sunset
camera    = Camera_0
```

每个 stratum 内 shuffle 后取 10% 到 val，剩余到 train。小 stratum 规则必须固定，避免不同实现产生不同 split：

```text
if n >= 2: val_count = max(1, round(n * 0.10))
if n == 1: val_count = 0, 样本保留在 train
```

输出 summary 至少记录：

```text
total / train / val
missing_rgb / missing_depth
overlap_count
per_scene_condition_camera counts, including train_count and val_count
seed
val_fraction
```

### 2.2 生成命令

```bash
conda run --live-stream -n dav3 python foundation/tools/split_vkitti2_train_val.py \
  --input finetune_stf/dataset/splits/vkitti2/train.txt \
  --train-output finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --val-output finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --summary-output finetune_stf/dataset/splits/vkitti2/split_mseries_seed42_summary.json \
  --val-fraction 0.10 \
  --seed 42 \
  --strict
```

预期：

```text
train ≈ 17600
val   ≈ 1950
missing_rgb = 0
missing_depth = 0
overlap_count = 0
```

这些 split 是正式配置，不是 smoke 产物，不要删除。

---

## 3. Step B：扩展 VKITTI2Raw dataset

现状：

```text
foundation/engine/datasets/vkitti2_raw.py
```

已经能：

```text
RGB -> crop/flip -> online unprocessing -> raw4
GT depth -> 同 crop/resize
```

但目前训练 sample 主要返回：

```python
sample["raw"]
sample["depth"]
sample["valid_mask"]
```

主方法还需要同一个 crop/flip 下的 RGB normalized input 给 frozen DAV2。

### 3.1 修改目标

给 `VKITTI2Raw` 增加参数：

```python
include_rgb_input: bool = False
include_rgb_preview: bool = False
```

这两个参数应进入 dataset 构造参数，并传入 `build_sample()`；`include_geometry` 只保留为调试几何信息开关，不要复用它控制 RGB preview。

当 `include_rgb_input=True` 或 `include_rgb_preview=True` 时，在 `build_sample()` 中按需额外返回：

```python
sample["image"]       # include_rgb_input=True: [3,H,W], ImageNet normalized, same crop/flip as raw/depth
sample["rgb_preview"] # include_rgb_preview=True: [3,H,W], 0..1, optional for visualization
```

注意：

```text
1. image 必须从同一次 crop/flip 后的 fullres image 数组生成，不能重新读图后独立 crop。
   直接复用 foundation/engine/datasets/vkitti2_raw.py 的 _imagenet_normalize_rgb_tensor(image, self.size)。
2. image 的 H,W 必须等于 raw 的 sensor H,W，即默认 512x960。
3. train: random crop + random hflip + random unprocessing。
4. val: center crop + no hflip + deterministic unprocessing。
5. include_rgb_preview 必须独立于 include_geometry；需要 preview 时由 include_rgb_preview=True 返回 sample["rgb_preview"]，不能强行带上 geometry_params。
```

### 3.2 验证点

新增 dataset smoke：

```bash
conda run --live-stream -n dav3 python - <<'PY'
from foundation.engine.datasets import VKITTI2Raw

ds = VKITTI2Raw(
    filelist_path="finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt",
    mode="val",
    size=(512, 960),
    randomize_unprocessing=False,
    include_rgb_input=True,
    include_rgb_preview=True,
)
s = ds[0]
print("raw", tuple(s["raw"].shape), s["raw"].dtype, float(s["raw"].min()), float(s["raw"].max()))
print("image", tuple(s["image"].shape), s["image"].dtype, float(s["image"].mean()))
print("depth", tuple(s["depth"].shape), s["depth"].dtype)
print("valid", tuple(s["valid_mask"].shape), int(s["valid_mask"].sum()))
print("sample", s["sample_name"])
PY
```

预期：

```text
raw:   (4,512,960), float32
image: (3,512,960), float32, ImageNet normalized
depth: (512,960), float32
valid: (512,960), bool
```

---

## 4. Step C：新增 RawResidualDAV2 模型

新增文件：

```text
foundation/engine/models/raw_residual_dav2.py
```

并在：

```text
foundation/engine/models/__init__.py
```

导出 builder。

### 4.1 模型输入输出

模型 forward 输入建议为 dict，避免以后参数顺序混乱：

```python
out = model({
    "image": image_rgb_norm,  # [B,3,H,W]
    "raw": raw4,              # [B,4,H,W]
    "valid_mask": valid_mask, # optional, [B,H,W]
})
```

返回：

```python
{
    "pred": D_final,          # [B,H,W], normalized inverse-depth space
    "D0": D0,                 # [B,H,W], frozen DAV2 raw output
    "D0_norm": D0_norm,       # [B,H,W]
    "delta": delta,           # [B,H,W]
    "gate": gate,             # [B,H,W]
    "x3": x3,                 # [B,3,H,W]
    "ffm_mid": ffm_mid,       # [B,64,H,W]
    "ram_out": ram_out,
}
```

训练脚本里用 `out["pred"]` 算 loss。

### 4.2 DAV2 path

```text
image(rgb_norm) -> center pad -> DAV2 -> center crop -> D0
```

要求：

```python
self.dav2.eval()
for p in self.dav2.parameters():
    p.requires_grad = False

with torch.no_grad():
    D0 = self.rgb_dav2(image)
D0 = D0.detach()
```

不要把 raw/RamCore3 输出送进 DAV2。

center pad/crop 复用已有空间适配逻辑，不重复实现：

```text
优先复用 finetune_stf/models/spatial_adapter.py 的 CenterPadCropAdapter，
参考 foundation/engine/models/dav2_raw_naive.py 中的 "center pad -> DAv2 -> center crop" wrapper。
对默认 sensor size 512x960，pad target 是 518x966（H/W 均为 14 的倍数），DAV2 输出后 center crop 回 512x960。
```

### 4.3 RAW path

```text
raw4 -> base_rgb = packed_bayer_to_base_rgb(raw4) -> RamCore3.forward_with_features()
```

直接复用现有：

```text
finetune_stf/models/raw_ram.py
RamCore3.forward_with_features()
packed_bayer_to_base_rgb()
```

必须调用 `packed_bayer_to_base_rgb(raw4)`，不要在新模型里手写 `torch.cat([R, (Gr+Gb)/2, B])`，避免 packed Bayer 通道顺序以后变更时失配。

不要破坏 `RamCore3.forward()` 的旧行为。新模型只调用 `forward_with_features()`。

### 4.4 M 系列 feature source

新增显式参数：

```text
residual_feature_source = ffm_mid | x3 | x3_ffm_mid
```

对应：

```text
M2: ffm_mid
M1: x3
M3: x3_ffm_mid
```

输入到 residual head：

```text
M2: concat(D0_norm, ffm_mid)      channels = 65
M1: concat(D0_norm, x3)           channels = 4
M3: concat(D0_norm, x3, ffm_mid)  channels = 68
```

`D0_norm` 在模型输出和 loss 里保持 `[B,H,W]`。送入 residual head 前必须显式做：

```python
D0_norm_ch = D0_norm.unsqueeze(1)  # [B,1,H,W]
```

随后用 `D0_norm_ch` 与 `x3` / `ffm_mid` concat。

### 4.5 D0 和 GT 的训练空间

本轮 VKITTI 是真实 metric depth，但模型按 relative inverse-depth refinement 训练。

训练脚本中构造：

```python
Y = 1.0 / clamp(depth_gt, min=1e-6)
Y_norm = robust_norm(Y, valid_mask)
D0_norm = robust_norm(sign * D0, valid_mask)
```

`sign` 用 val 子集 sanity check 决定：

```text
先统计 corr(D0, 1/depth_gt)
如果平均 corr < 0，则 sign = -1
否则 sign = +1
```

不要每张图单独翻转。sign 是整个 run 的显式参数：

```text
--d0-sign 1
或
--d0-sign -1
```

### 4.6 ResidualGateHead

结构：

```text
Conv 3x3 in_ch -> 64
GroupNorm(8)
GELU
ResBlock 64

Down 64 -> 128
ResBlock 128
Down 128 -> 256
ResBlock 256

Up 256 -> 128 + skip
ResBlock 128
Up 128 -> 64 + skip
ResBlock 64

delta_head: Conv 3x3 64->32 + GELU + Conv 1x1 32->1
gate_head:  Conv 3x3 64->32 + GELU + Conv 1x1 32->1
```

不要用 BatchNorm。使用：

```python
GroupNorm(num_groups=8)
```

输出：

```python
delta = alpha * tanh(delta_raw)
gate = sigmoid(gate_logit)
D_final = D0_norm + gate * delta
```

默认：

```text
alpha = 0.5
```

Identity init 必须做：

```text
delta_head last conv weight = 0
delta_head last conv bias   = 0
gate_head last conv weight  = 0
gate_head last conv bias    = -4
```

---

## 5. Step D：新增 VKITTI residual 训练入口

新增：

```text
foundation/tools/train_vkitti2_raw_residual.py
```

建议从：

```text
foundation/tools/train_phase1_vkitti2_naive.py
```

复制并改造，因为它已经使用 `VKITTI2Raw` 和 VKITTI GT depth。

### 5.1 必需参数

新增/保留参数：

```text
--input-domain raw4
--model-input-tensor raw
--raw-storage-format synthetic_packed_bayer_4ch
--front-end raw_to_base_rgb_ram3
--encoder vits
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt
--vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt
--input-height 512
--input-width 960
--min-depth 1.0
--max-depth 80.0
--residual-feature-source ffm_mid|x3|x3_ffm_mid
--residual-alpha 0.5
--d0-sign 1|-1
--vkitti-unprocessing-preset sensor_linear_dual
--randomize-unprocessing / --no-randomize-unprocessing
--hflip-prob 0.5
--epochs 10
--bs 8
--accum-steps 1
--lr 1e-4
--weight-decay 1e-4
--num-workers 4
--log-interval 100
--save-interval 1
--eval-interval 1
--save-best-checkpoint
--max-train-steps <int, optional smoke/debug only>
--max-val-samples <int, optional smoke/debug only>
--amp
--amp-dtype bf16
--seed 42
--save-path <EXP_ROOT>/<RUN_NAME>
--heavy-save-path <HEAVY_ROOT>/<RUN_NAME>
```

实验语义参数必须在正式脚本里显式写出，不靠默认值：

```text
input_domain
model_input_tensor
raw_storage_format
front_end
residual_feature_source
residual_alpha
d0_sign
vkitti_unprocessing_preset
randomize_unprocessing
hflip_prob
input_height/input_width
min_depth/max_depth
encoder
pretrained_from
```

`raw_storage_format=synthetic_packed_bayer_4ch` 是本入口对 VKITTI online unprocessing 输出的显式语义标记；不要从数据路径或 preset 名推断。配置解析阶段必须集中校验：

```text
input_domain == raw4
model_input_tensor == raw
raw_storage_format == synthetic_packed_bayer_4ch
front_end == raw_to_base_rgb_ram3
front_end 与 residual_feature_source/x3/ffm_mid 的适用关系
```

本轮 `bs=8` 直接使用 `lr=1e-4`，显式不做 learning-rate linear scaling。

同一实现阶段还必须新增两个辅助脚本，不要等正式实验结束后再临时补：

```text
foundation/tools/check_vkitti_dav2_sign.py
foundation/tools/summarize_vkitti_mseries.py
```

### 5.2 Loss

训练 loss 在这个新入口内实现，不直接复用旧 `AlignedInverseSigLoss`。

总 loss：

```text
L = L_depth
  + 0.5   * L_grad
  + 0.1   * L_keep
  + 0.01  * L_res
  + 0.005 * L_gate
  + 0.05  * L_gate_sup
```

定义：

```text
L_depth = mean_valid(abs(D_final - Y_norm))

L_grad = mean_valid(abs(grad_x(D_final) - grad_x(Y_norm)))
       + mean_valid(abs(grad_y(D_final) - grad_y(Y_norm)))

E0 = abs(D0_norm - Y_norm)
M_error = clamp((E0 - q80) / (q95 - q80 + eps), 0, 1)

L_keep = mean_valid((1 - M_error) * abs(gate * delta))
L_res  = mean_valid(abs(gate * delta))
L_gate = mean_valid(gate)
L_gate_sup = BCE(gate, M_error)
```

所有 quantile 都按每张图自己的 valid pixels 计算。

### 5.3 训练日志

每个 log interval 记录：

```text
loss_total
L_depth
L_grad
L_keep
L_res
L_gate
L_gate_sup
mean(gate)
max(gate)
mean(abs(delta))
mean(abs(gate * delta))
mean(abs(D_final - D0_norm))
x3 mean/std/min/max
ffm_mid mean/std
lr
max_mem_mb
```

`max_mem_mb` 定义为：

```python
torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
```

每个 epoch 开始时调用 `torch.cuda.reset_peak_memory_stats(device)`，因此日志含义是 epoch-local peak，不是从进程启动以来累计的峰值。

每个 epoch 记录：

```text
train_loss_summary.json
val_metrics.json
current checkpoint
best checkpoint by val abs_rel
```

### 5.4 Evaluation

每个 epoch 在 val split 上评估：

```text
Final vs GT
D0 vs GT
```

使用同一个 affine alignment 协议：

```text
pred_disp -> affine_align_disp(gt_depth, pred_disp, valid_mask)
aligned_depth -> compute_metrics()
```

可复用：

```text
anqi_eval/eval_rel_depth_strict.py
affine_align_disp
compute_metrics
```

必须输出：

```text
overall:
  abs_rel, sq_rel, rmse, rmse_log, log10, silog, silog_x100, d1, d2, d3

region:
  boundary_abs_rel
  dav2_high_error_abs_rel
  far50_abs_rel
  dark_abs_rel
  saturated_abs_rel

delta:
  final_abs_rel - D0_abs_rel
  final_d1 - D0_d1
```

Region mask 定义：

```text
boundary: grad(depth_gt) top 10% valid pixels
DAV2 high-error: abs(D0_norm - Y_norm) top 20% valid pixels
far50: depth_gt > 50m
dark: RGB luma < 0.15
saturated: max(R,G,B) > 0.95
```

注意：dark/saturated 需要 RGB preview 或反归一化的 `sample["image"]`。

---

## 6. Step E：smoke tests

所有 smoke 输出必须放在含 `codex_smoke` 的路径。成功后只删除这些明确临时产物；失败时保留。

### 6.1 Dataset smoke

```bash
SMOKE_DIR=plans/0524_new/codex_smoke_vkitti_mseries_dataset
mkdir -p "$SMOKE_DIR"

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_raw_residual.py \
  --input-domain raw4 \
  --model-input-tensor raw \
  --raw-storage-format synthetic_packed_bayer_4ch \
  --front-end raw_to_base_rgb_ram3 \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 512 \
  --input-width 960 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source ffm_mid \
  --residual-alpha 0.5 \
  --d0-sign 1 \
  --vkitti-unprocessing-preset sensor_linear_dual \
  --randomize-unprocessing \
  --hflip-prob 0.5 \
  --epochs 1 \
  --bs 2 \
  --accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --log-interval 1 \
  --max-train-steps 2 \
  --max-val-samples 4 \
  --save-path "$SMOKE_DIR/exp" \
  --heavy-save-path "$SMOKE_DIR/heavy" \
  --amp \
  --amp-dtype bf16
```

成功后清理：

```bash
rm -rf "$SMOKE_DIR"
```

失败时不要删，记录：

```text
plans/0524_new/codex_smoke_vkitti_mseries_dataset
```

### 6.2 D0 sign check

在正式 M2 前，跑一个只遍历 val 前 64 张的 sign check：

```bash
conda run --live-stream -n dav3 python foundation/tools/check_vkitti_dav2_sign.py \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 512 \
  --input-width 960 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --max-samples 64 \
  --output plans/0524_new/codex_smoke_vkitti_d0_sign_summary.json
```

如果输出平均相关性为正：

```text
formal args: --d0-sign 1
```

如果为负：

```text
formal args: --d0-sign -1
```

这个 summary 是 smoke 产物；确认结果写进正式实验日志后可以删除。

---

## 7. Step F：正式 M2

先只跑 M2：

```text
M2 = frozen DAV2 + ffm_mid residual
```

建议新增正式 queue 脚本：

```text
finetune_stf/scripts/formal/0524_run_vkitti_mseries_residual_queue.sh
```

脚本职责：

```text
1. 在脚本顶层生成一次当前时间戳 MMDD_HHMM，并给 M2/M1/M3 复用
2. 创建 tmux session，不复用旧 session
3. 先跑 smoke，可用 RUN_SMOKE=0 跳过
4. 跑正式 M2
5. 把日志写到 finetune_stf/logs/<session>.queue.log
6. 输出 attach / tail 命令
```

M2 正式参数：

```bash
TIMESTAMP=$(date +%m%d_%H%M)
RUN_SUFFIX="vkitti_m2_ffm_mid_residual_vits_bs8_e10"
RUN_NAME="${TIMESTAMP}_${RUN_SUFFIX}"

conda run --live-stream -n dav3 python foundation/tools/train_vkitti2_raw_residual.py \
  --input-domain raw4 \
  --model-input-tensor raw \
  --raw-storage-format synthetic_packed_bayer_4ch \
  --front-end raw_to_base_rgb_ram3 \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_mseries_seed42.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_mseries_seed42.txt \
  --input-height 512 \
  --input-width 960 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source ffm_mid \
  --residual-alpha 0.5 \
  --d0-sign <FROM_SIGN_CHECK> \
  --vkitti-unprocessing-preset sensor_linear_dual \
  --randomize-unprocessing \
  --hflip-prob 0.5 \
  --epochs 10 \
  --bs 8 \
  --accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --log-interval 100 \
  --save-interval 1 \
  --eval-interval 1 \
  --save-best-checkpoint \
  --save-path "finetune_stf/exp/${RUN_NAME}" \
  --heavy-save-path "/mnt/drive/3333_raw/0000_exp_ckpt/${RUN_NAME}" \
  --amp \
  --amp-dtype bf16 \
  --seed 42
```

正式 tmux 启动后必须报告：

```text
tmux session: <SESSION>
queue log: finetune_stf/logs/<SESSION>.queue.log
attach: tmux attach -t <SESSION>
monitor: tail -f finetune_stf/logs/<SESSION>.queue.log
```

---

## 8. Step G：M2 结束后检查

M2 跑完后先检查，不要立刻解释论文结论。

必须看：

```text
1. final overall abs_rel 是否优于 D0
2. final high-error region abs_rel 是否优于 D0
3. boundary_abs_rel 是否优于 D0
4. low-error region 是否明显变差
5. mean(gate) 是否长期接近 1
6. mean(abs(delta)) 是否长期打满 alpha=0.5
7. x3 min/max 是否爆炸
8. val 曲线是否只在 train loss 降但 val 不动
```

判断：

```text
如果 M2 明显坏于 D0:
  先查 normalization / d0_sign / loss 权重 / gate 初始化。
  暂停 M1/M3。

如果 M2 接近 D0 但 high-error 或 boundary 有提升:
  继续跑 M1/M3。

如果 M2 overall 和 region 都提升:
  继续跑 M1/M3，并把 M2 作为当前主候选。
```

---

## 9. Step H：正式跑 M1 和 M3

M1：

```text
residual_feature_source = x3
run suffix = vkitti_m1_x3_residual_vits_bs8_e10
```

M3：

```text
residual_feature_source = x3_ffm_mid
run suffix = vkitti_m3_x3_ffm_mid_residual_vits_bs8_e10
```

除 `--residual-feature-source` 和 `RUN_SUFFIX` 外，其余参数必须和 M2 完全一致。

queue 脚本顶层只取一次 `TIMESTAMP=$(date +%m%d_%H%M)`。M1/M3 必须复用 M2 同一个 `TIMESTAMP`，分别设置：

```bash
RUN_SUFFIX="vkitti_m1_x3_residual_vits_bs8_e10"
RUN_NAME="${TIMESTAMP}_${RUN_SUFFIX}"

RUN_SUFFIX="vkitti_m3_x3_ffm_mid_residual_vits_bs8_e10"
RUN_NAME="${TIMESTAMP}_${RUN_SUFFIX}"
```

运行顺序：

```text
1. M1
2. M3
```

M1/M3 不需要重新划分 split，不需要更换 unprocessing preset，不需要更换 seed。

---

## 10. Step I：M 系列汇总表

新增汇总脚本：

```text
foundation/tools/summarize_vkitti_mseries.py
```

输入：

```text
finetune_stf/exp/<M2_RUN>
finetune_stf/exp/<M1_RUN>
finetune_stf/exp/<M3_RUN>
```

输出：

```text
plans/0524_new/vkitti_mseries_summary_<MMDD_HHMM>.md
plans/0524_new/vkitti_mseries_summary_<MMDD_HHMM>.json
```

表格至少包含：

```text
method
feature_source
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

选择规则：

```text
如果 M3 明显优于 M2，用 M3 作为 main。
如果 M3 和 M2 接近，用 M2 作为 main。
如果 M1 优于 M2/M3，说明 x3 cue 比 ffm_mid 更有效，需要回看 ffm_mid 是否过强/过宽或 normalization 是否不合适。
如果 M 系列都不优于 D0，先不要跑 C 系列，优先 debug residual formulation。
```

---

## 11. 后续 C 系列，只在 M 系列后执行

本轮先不实现 C 系列，但预留接口：

```text
C1: frozen DAV2 + RGB residual branch
C2: frozen DAV2 + D0-only residual branch
C3: frozen DAV2 + parameter-matched plain U-Net RAW residual
```

C 系列必须使用同一份：

```text
train_mseries_seed42.txt
val_mseries_seed42.txt
bs=8
epochs=10
input size=512x960
encoder=vits
unprocessing preset=sensor_linear_dual
```

不要在 C 系列里改变数据、backbone、训练长度或 eval protocol。

---

## 12. 最小完成定义

本执行计划完成的最低标准：

```text
[ ] train/val split 已生成并通过 no-overlap / no-missing 检查
[ ] VKITTI2Raw 返回 raw + image，并通过 shape smoke
[ ] RawResidualDAV2 支持 ffm_mid / x3 / x3_ffm_mid
[ ] DAV2 frozen + eval + no_grad 已在日志中确认
[ ] D0 sign check 已完成并写入 formal args
[ ] 2-step smoke train 成功，成功 smoke 产物已删除
[ ] M2 正式 10 epoch 完成
[ ] M2 诊断通过后，M1/M3 正式 10 epoch 完成
[ ] M 系列 summary md/json 已输出
[ ] 已明确下一步是 debug M 系列还是进入 C 系列
```

---

## 13. 当前不做的事情

本轮不做：

```text
B0 / B1 formal training
fine-tune DAV2
LoRA
bridge injection
decoder feature adapter
STF 训练
RobotCar / ETH3D / KITTI 跨域评估
Raw4 / Raw5 green-channel ablation
```

这些都等 M 系列在 VKITTI GT 上跑通并看完诊断后再决定。
