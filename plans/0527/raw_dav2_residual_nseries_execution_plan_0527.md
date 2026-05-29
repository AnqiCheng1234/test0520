# RAW-like Residual Correction N 系列具体执行计划

日期：2026-05-27  
输入计划：`plans/0527/raw_dav2_residual_reanalysis_plan_0527.md`  
目标：把当前复盘中的“C2 frozen + RAW/x3 incremental correction”落实为可执行的代码修改、诊断工具、smoke、formal 实验和结果汇总流程。

---

## 0. 执行原则

后续操作默认在项目根目录执行：

```bash
cd /home/caq/6666_raw/dav2_raw_0522
conda run --live-stream -n dav3 <command>
```

正式训练、全量 eval、全量诊断、全量 panel 生成都按长任务处理，必须通过 tmux 启动。smoke 输出路径必须包含 `codex_smoke` / `smoke` / `debug` / `tmp` 之一；smoke 成功后只删除明确的 smoke 产物，失败则保留并报告路径。

所有 formal experiment name 必须由启动时服务器本地时间 `MMDD_HHMM` 开头。所有改变实验语义的参数必须在 formal launch script 中显式出现，不能依赖默认值、路径名推断或一个参数隐式决定多个无关行为。若某参数对当前 method 不适用，也必须显式写成 `not_applicable` / `none`，并由 `validate_args` 中央校验。至少包括：

```text
method_id
encoder
pretrained_from
vkitti_train_list
vkitti_val_list
input_height
input_width
min_depth
max_depth
input_domain
model_input_tensor
dataset_geometry_mode
raw_storage_format
fullres_even_policy
rgb_input_space
depth_target_space
front_end
c2_checkpoint
c2_run_dir
incremental_feature_source
delta_condition
gate_condition
raw_feature_encoder_trainable
residual_alpha
d0_sign
lambda_lp
lowpass_kernel
q_good
lambda_final
lambda_boundary
lambda_grad
lambda_keep_good_d1
lambda_gate_sparse
lambda_lowfreq_loss
lambda_invalid_keep
eval_protocol
kitti_eval_protocol
kitti_val_split
kitti_base
unprocessing_method
vkitti_unprocessing_preset
randomize_unprocessing
raw_adapter_backend
raw_adapter_cfa_pattern
raw_adapter_packed_channel_order
raw_adapter_rgb_transfer
raw_adapter_inverse_tone
raw_adapter_ccm
raw_adapter_red_gain_range
raw_adapter_blue_gain_range
raw_adapter_fixed_red_gain
raw_adapter_fixed_blue_gain
raw_adapter_fixed_light_scale
raw_adapter_dark_light_scale_range
raw_adapter_over_light_scale_range
raw_adapter_shot_noise
raw_adapter_read_noise
raw_adapter_noise_mean_mode
raw_adapter_black_level
raw_adapter_white_level
raw_adapter_random_seed_policy
raw_adapter_variant_policy
raw_adapter_variant_weights
```

本轮只新增计划和后续执行项，不在本文件中直接修改训练代码。

---

## 1. 当前代码地图

已有关键入口：

```text
foundation/engine/models/raw_residual_dav2.py
  当前 M 系列：D0_norm 可选 concat raw/RAM feature，然后单个 ResidualGateHead 输出 delta/gate。

foundation/engine/models/dav2_residual_control.py
  当前 C1/C2：C2 是 D0-only residual calibrator。

foundation/tools/residual_training_common.py
  C/M 系列共享 loss、region metric、checkpoint/json helper。

foundation/tools/train_vkitti2_raw_residual.py
  M 系列训练/eval/KITTI sanity 入口。

foundation/tools/train_vkitti2_residual_control.py
  C1/C2 control 训练/eval/KITTI sanity 入口。

foundation/tools/make_vkitti_raw_residual_qual_panels.py
foundation/tools/eval_raw_residual_kitti.py
foundation/tools/residual_control_kitti_eval.py
  现有 qualitative / KITTI 评测工具。

finetune_stf/scripts/formal/0524_run_vkitti_cseries_residual_controls_queue.sh
finetune_stf/scripts/formal/0526_run_vkitti_rawadapter_feature_d0_ablation_queue.sh
  可复用的 tmux queue / smoke / formal script 模式。
```

当前核心缺口：

```text
1. 没有 eval-time raw/x3 true/zero/mean/shuffle 诊断。
2. 没有以 C2 为 baseline 的 improvement map。
3. residual/gate energy 只统计 valid 区域均值，没有 valid/invalid/region/frequency 分布。
4. 没有 C2 frozen + x3 incremental correction 模型。
5. 现有 loss 是相对 D0 的单头 residual loss，不适合 D1=C2 后的 incremental branch。
```

---

## 2. 总体执行顺序

按下面顺序执行，不要先重训 N 系列：

```text
Step A. 给现有 M1/M2/C2 checkpoint 补无重训诊断。
Step B. 实现 N 系列模型：C2 frozen base + incremental branch。
Step C. 实现 incremental loss、eval 指标、summary 和 panel。
Step D. 写 smoke 脚本并跑通 2-step train / tiny val / tiny KITTI。
Step E. 写 formal queue script。
Step F. 先跑 N2 x3 的 lambda_lp sweep；再跑 q_good sweep；必要时单独跑 lambda_lowfreq_loss sweep。
Step G. 跑 N3 RGB、N4 ffm_mid、N5 D1-only、N7 stop-gradient ablation。
Step H. 对 N2 checkpoint 做 true/zero/mean/shuffle x3 eval。
Step I. 汇总 VKITTI formal、KITTI sanity、region、energy、frequency、panel。
```

---

## 3. Step A：无重训诊断

### A1. 新增 eval-time feature ablation 工具

新增：

```text
foundation/tools/eval_raw_residual_feature_ablation.py
```

用途：对现有 M1/M2 checkpoint 做：

```text
true feature
zero feature
mean feature
shuffled feature from another validation sample
```

最低支持对象：

```text
M1 RA0 x3 + D0:
  run_dir=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20
  checkpoint=/mnt/drive/3333_raw/0000_exp_ckpt/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_14.pth

M2 RA0 ffm_mid + D0:
  run_dir=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20
  checkpoint=/mnt/drive/3333_raw/0000_exp_ckpt/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_09.pth
```

实现方式：

```text
1. 复用 train_vkitti2_raw_residual.py 的 config/dataset/model 构建逻辑。
2. 在 eval loop 中加入 --feature-source {x3,ffm_mid}、--feature-ablation-mode {true,zero,mean,shuffle}、--shuffle-policy {next,stable_hash_far}、--shuffle-seed <int>。
3. true：保持原输出。
4. zero：将被测试 feature 替换为 zeros_like(feature)。
5. mean：先扫一遍 val set，计算对应 feature 的 per-channel global mean，再替换为同形状常数图。
6. shuffle：从同一个 validation split 内的 donor sample 重新提取“被测试 feature tensor”，然后只把该 tensor splice 到当前 sample 的 head_input 前。
7. 对 x3/ffm_mid 都支持，不要只支持 x3。
```

关键实现约束：

```text
ablation 操作对象必须是 `_head_input(...)` 之前的中间 feature tensor，即 x3 或 ffm_mid。
不要替换当前 sample 的 batch["raw"]，也不要把 x3 和 ffm_mid 同时替换，除非后续另设 --feature-source all_raw_frontend。
因此 "x3 shuffle" 的含义是当前 sample 的 D0/D0_norm、raw 输入和非目标 feature 保持不变，仅 head input 中的 x3 来自 donor sample。
"ffm_mid shuffle" 同理，仅替换 ffm_mid。
shuffle 不改变评测 split：target samples 仍然是原始 val split 的同一批样本，只改变 ablation donor 映射。
manifest 必须记录 shuffle_policy、shuffle_seed、target_index、donor_index、target_sample_name、donor_sample_name 和 donor_mapping_sha256。
```

shuffle donor policy：

```text
stable_hash_far:
  formal 主结论默认使用。
  对每个 target idx，从同一 val split 的候选 donor 中选择 condition 不同且 abs(frame_i - frame_j) >= 50 的样本。
  候选按 stable hash 排序：sha256(f"{shuffle_seed}:{target_idx}:{candidate_idx}:{candidate_sample_name}")。
  取 hash 最小的候选；若无候选，退化为 condition 不同；仍无候选再退化为 target_idx != candidate_idx。
  映射必须 deterministic，不能使用运行时随机数状态。

next:
  只用于复现旧/弱负控协议，donor_idx = (target_idx + 1) % N。
  如果和 stable_hash_far 结论不同，summary 中优先采用 stable_hash_far 作为 RAW/RAM feature 是否有效的判断。
```

建议先在模型里加一个内部 helper，而不是在 evaluator 里复制 head 逻辑：

```python
RawResidualDAV2.forward(
    batch,
    feature_override: dict[str, torch.Tensor] | None = None,  # optional keys: "x3", "ffm_mid"
    feature_ablation_mode: str = "true",
)
```

如果不想改现有 forward 签名，可以新增私有方法：

```python
RawResidualDAV2.forward_with_feature_override(...)
```

输出文件：

```text
plans/0527/diagnostics/<timestamp>_raw_feature_ablation/
  m1_x3_d0/
    true_metrics.json
    zero_metrics.json
    mean_metrics.json
    shuffle_metrics.json
    summary.csv
    summary.md
  m2_ffm_mid_d0/
    ...
```

必须记录：

```text
overall abs_rel/d1
boundary/high-error/far50/dark/saturated abs_rel
mean_gate
mean_abs_delta
mean_abs_gate_delta
feature_ablation_mode
feature_source
checkpoint
run_dir
D0 consistency
```

判断：

```text
shuffle_minus_true_abs_rel = shuffle_abs_rel - true_abs_rel
abs(shuffle_minus_true_abs_rel) < 0.0005：当前模型没有稳定使用 RAW/RAM feature。
shuffle_minus_true_abs_rel >= 0.001，且 boundary/fog/saturated 至少一个 target region 同向提升 >= 0.003：RAW/RAM 有条件性贡献。
```

### A2. 新增 residual/gate energy 和 frequency 诊断

新增：

```text
foundation/tools/analyze_residual_energy_frequency.py
```

支持 C2、M1、M2、后续 N2/N3/N4/N5/N7。建议通过 `--run-kind {control,raw,nseries}` 区分构建模型。

每张图统计 masks：

```text
valid
invalid
boundary = GT depth gradient top-10% within valid GT pixels
high_error = D0/D1 base normalized error top-20% within valid GT pixels
far50 = depth > 50m
dark = RGB luma < 0.15
saturated = max RGB > 0.95
```

每个 mask 记录：

```text
gate_mass_ratio = sum(M * gate) / (sum(gate) + eps)
residual_energy_ratio = sum(M * abs(gate * delta)) / (sum(abs(gate * delta)) + eps)
mean_gate
mean_abs_gate_delta
pixel_ratio
```

若分母为 0 或 mask 为空，对应 ratio/mean 写 `null`，并在 per-sample 记录 `empty_mask=true`；summary 聚合时只平均有限值，不要把空 mask 当作 0。

frequency 统计：

```python
residual = gate * delta
low = avgpool(residual, kernel_size=31, stride=1, padding=15)
high = residual - low
denom = mean(abs(residual)) + eps
low_ratio = mean(abs(low)) / denom
high_ratio = mean(abs(high)) / denom
```

输出：

```text
plans/0527/diagnostics/<timestamp>_residual_energy_frequency/
  per_sample.jsonl
  summary.json
  summary.csv
  summary.md
```

### A3. 新增 improvement-over-C2 panel

新增：

```text
foundation/tools/make_residual_vs_c2_panels.py
```

功能：

```text
1. 加载 C2 checkpoint。
2. 加载一个 method checkpoint：M1/M2/N2/N3/N4/N5/N7。
3. 对同一 VKITTI val sample 计算 C2 aligned error 和 method aligned error。
4. 输出 method - C2 的可视化。
```

panel 至少包含：

```text
RGB
GT depth
C2 depth
method depth
C2 absrel error
method absrel error
improvement over C2: green = method better, red = method worse
method gate*delta
method gate
```

输出路径：

```text
plans/0527/panels/<timestamp>_vs_c2_<method>/
  manifest.json
  *.jpg
```

注意：后续 qualitative 证据优先看 `method vs C2`，不要继续只看 `method vs D0`。

### A4. 无重训诊断启动脚本

新增：

```text
finetune_stf/scripts/formal/0527_run_existing_residual_diagnostics_queue.sh
```

要求：

```text
1. 外层启动 tmux，session 名形如 <MMDD_HHMM>_existing_residual_diagnostics。
2. queue log 写到 finetune_stf/logs/<session>.queue.log。
3. 不覆盖已有输出目录。
4. 默认 RUN_FEATURE_ABLATION=1, RUN_ENERGY_FREQ=1, RUN_VS_C2_PANELS=1。
5. 命令全部用 conda run --live-stream -n dav3。
```

启动后必须报告：

```text
tmux attach -t <session>
tail -f <queue_log>
```

---

## 4. Step B：N 系列模型代码

### B1. 新增模型文件

新增：

```text
foundation/engine/models/dav2_incremental_residual.py
```

并在：

```text
foundation/engine/models/__init__.py
```

导出：

```python
C2FrozenIncrementalResidualDAV2
build_c2_frozen_incremental_residual_model
```

模型语义：

```text
C2_out = frozen C2({"image": image, "valid_mask": valid_mask})
D0 = C2_out["D0"]                  # DAV2 raw output from the C2 wrapper
D0_norm = C2_out["D0_norm"]        # C2 wrapper's normalized DAV2 base
D1_norm = C2_out["pred"]           # frozen C2 calibrator output in normalized target space
feature = x3 / ffm_mid / rgb / D1
delta_raw = DeltaHead(feature path)
gate_raw = GateHead(feature + D1 path)
D_final = D1_norm + gate_raw * delta_effective
```

实现约束：

```text
1. 当前 C2 路径是在线调用 frozen DAV2，不读取本地 D0 cache/file。
2. 本轮唯一允许的实现路径：N wrapper 只调用一次 frozen C2.forward，并从 C2_out 读取 D0/D0_norm/D1_norm。
3. N wrapper 内不要额外单独创建或调用 DAV2；不采用“双 DAV2 forward 但假设 bit-identical”的实现。
4. 本轮也不重构 C2.forward_from_d0_norm；若后续需要提速，单独立项并重新 smoke/对照。
5. frozen C2 必须 eval()，所有参数 requires_grad=False，N wrapper 的 train(mode) 也必须保持 c2_model.eval()。
6. `D1` / `D1_norm` 在本计划中专指 frozen C2 base，不是 depth metric 里的 `d1` threshold 指标。
```

推荐 forward 输入：

```python
out = model({
    "image": image_rgb_norm,   # [B,3,H,W]
    "raw": raw4,               # required for x3/ffm_mid, optional otherwise
    "valid_mask": valid_mask,  # [B,H,W]
})
```

必须返回：

```python
{
    "pred": final_norm,          # final normalized prediction
    "D0": d0_raw,                # DAV2 raw output, for D0 consistency
    "D0_norm": d0_norm,          # DAV2 normalized base
    "D1_norm": d1_norm,          # frozen C2 base
    "base_norm": d1_norm,        # loss/eval 通用 base
    "C2_delta": c2_out.get("delta"),
    "C2_gate": c2_out.get("gate"),
    "delta": delta_raw,
    "delta_effective": delta_effective,
    "gate": gate_raw,
    "gate_delta": gate_raw * delta_effective,
    "x3": x3 or None,
    "ffm_mid": ffm_mid or None,
}
```

不要把 `D1_norm` 命名成 `D0_norm` 来复用旧 loss，这会污染后续解释。旧 evaluator 可以保留 D0 指标，但 N 系列必须额外报告 C2/D1 base 指标。

### B2. 模型参数

新增并中央验证：

```text
--method-id {N2,N3,N4,N5,N7}
--c2-checkpoint <path>
--incremental-feature-source {x3,ffm_mid,rgb,d1}
--delta-condition {feature_only,feature_d1_stopgrad,d1_only}
--gate-condition {feature_d1,d1_only}
--raw-feature-encoder-trainable {true,false,not_applicable}
--lambda-lp <float>
--lowpass-kernel 31
```

`raw_feature_encoder_trainable` 是实验语义参数，必须显式记录：

```text
true:
  RamCore3 / RAW feature encoder 与 incremental head 一起训练。
  N2/N4/N7 formal 默认使用 true，以匹配当前 M 系列“RAW/RAM feature path 可学习”的设定。

false:
  冻结 RAW feature encoder，仅训练 delta/gate head。只能作为后续隔离 ablation，不作为本轮默认。

not_applicable:
  N3 RGB 和 N5 D1-only 必须使用 not_applicable；传 true/false 应报错。
```

参数关系：

```text
N2:
  incremental_feature_source=x3
  delta_condition=feature_only
  gate_condition=feature_d1
  raw_feature_encoder_trainable=true

N3:
  incremental_feature_source=rgb
  delta_condition=feature_only
  gate_condition=feature_d1
  raw_feature_encoder_trainable=not_applicable

N4:
  incremental_feature_source=ffm_mid
  delta_condition=feature_only
  gate_condition=feature_d1
  raw_feature_encoder_trainable=true

N5:
  incremental_feature_source=d1
  delta_condition=d1_only
  gate_condition=d1_only
  raw_feature_encoder_trainable=not_applicable

N7:
  incremental_feature_source=x3
  delta_condition=feature_d1_stopgrad
  gate_condition=feature_d1
  raw_feature_encoder_trainable=true
```

`feature_d1_stopgrad` 的实现：

```python
d1_feature_sg = d1_feature.detach()
delta_input = torch.cat([feature, d1_feature_sg], dim=1)
```

N2 默认不允许 D1 进入 delta head：

```text
delta_head input: x3 only
gate_head input: x3 + D1 feature
```

### B3. Head 结构建议

先不要引入复杂新模块，保持和现有 `ResidualGateHead` 接近，便于比较参数量。

新增轻量 encoder：

```text
FeatureEncoder:
  Conv3x3 -> GN -> GELU -> ResidualBlock

D1Encoder:
  Conv3x3(1->32) -> GN -> GELU -> ResidualBlock

DeltaHead:
  feature_encoder -> small U-Net or residual stack -> tanh scaled by residual_alpha

GateHead:
  concat(feature_encoder, d1_encoder) -> small U-Net or residual stack -> sigmoid
```

固定 channel contract：

```text
raw feature tensor channels:
  x3: 3
  ffm_mid: 64
  rgb: 3
  d1: 1

all encoders output 32 channels:
  FeatureEncoder(3 or 64 -> 32)
  D1Encoder(1 -> 32)

method channel mapping:
  N2:
    delta_input = enc(x3)                         # 32 ch
    gate_input  = concat(enc(x3), enc(d1))         # 64 ch

  N3:
    delta_input = enc(rgb)                        # 32 ch
    gate_input  = concat(enc(rgb), enc(d1))        # 64 ch

  N4:
    delta_input = enc(ffm_mid)                    # 32 ch
    gate_input  = concat(enc(ffm_mid), enc(d1))    # 64 ch

  N5:
    delta_input = enc(d1)                         # 32 ch
    gate_input  = enc(d1)                         # 32 ch

  N7:
    delta_input = concat(enc(x3), enc(d1).detach()) # 64 ch
    gate_input  = concat(enc(x3), enc(d1))          # 64 ch
```

模型构建时必须根据 `method_id/incremental_feature_source/delta_condition/gate_condition` 派生并 assert `delta_in_ch` 和 `gate_in_ch`，不要让 head lazy infer 或依赖运行时 tensor shape。若以后改 encoder 输出通道，必须同步更新这张表和 config 记录。

如果为了减少代码量，可以抽象现有 `ResidualBlock/DownBlock/UpBlock/ResidualGateHead` 到共享文件；但不要做无关重构。第一版允许在新文件中局部复用同名 block，后续稳定后再合并。

### B4. C2 checkpoint 加载

实现时建议直接组合已有 C2 model：

```python
c2_model = build_dav2_residual_control_model(
    DepthAnythingV2(...),
    residual_feature_source="d0",
    ...
)
c2_model.load_state_dict(c2_state, strict=True)
c2_model.eval()
for p in c2_model.parameters():
    p.requires_grad = False
```

N wrapper forward 中只调用：

```python
with torch.no_grad():
    c2_out = self.c2_model({"image": image, "valid_mask": valid_mask})
d0_raw = c2_out["D0"].detach()
d0_norm = c2_out["D0_norm"].detach()
d1_norm = c2_out["pred"].detach()
```

不要在 N wrapper 里创建第二个 DAV2 base model 来重算 D0；否则 final、D1、D0 可能来自不同 wrapper 路径，诊断会变得不可解释。本轮明确不选择“C2.forward 内部跑一次 DAV2 + N wrapper 再跑一次 DAV2”的方案。

从 `--c2-checkpoint` 读取 checkpoint 时必须验证：

```text
正式脚本启动训练前先 dump C2 metadata：
  conda run --live-stream -n dav3 python -c "import torch; ck=torch.load(C2_CHECKPOINT,map_location='cpu'); print(ck.get('args',{}))"

C2 identity 校验：
  如果 checkpoint/config args 含 experiment_id，则 lower(experiment_id) 必须等于 c2。
  residual_feature_source 必须等于 d0。
  不使用宽松 OR 接受矛盾字段；例如 experiment_id=C1 但 residual_feature_source=d0 必须报错。

checkpoint args encoder == 当前训练设置
checkpoint args residual_alpha == 当前训练设置
checkpoint args front_end == dav2_rgb_frozen
checkpoint args raw_storage_format == not_applicable
checkpoint args input_height/input_width == 当前训练设置
checkpoint args min_depth/max_depth == 当前训练设置
checkpoint args d0_sign == 当前训练设置
checkpoint args fullres_even_policy/rgb_input_space/depth_target_space == 当前训练设置
checkpoint args vkitti_train_list/vkitti_val_list == 当前训练设置
```

如果 checkpoint 没有 `args`，必须使用显式传入的 `--c2-run-dir/config.json` 补充校验；如果 `experiment_id` 缺失但其它字段完整，则要求 `residual_feature_source=d0`、`front_end=dav2_rgb_frozen`、`input_domain=rgb`、`model_input_tensor=image`、`raw_storage_format=not_applicable` 全部成立才允许继续。如果 checkpoint 和 run dir 都缺少可校验元数据，formal queue 直接报错停止，不要静默继续，也不要在队列里交互式询问。

---

## 5. Step C：incremental loss 和 eval

### C1. 新增 loss helper

在：

```text
foundation/tools/residual_training_common.py
```

新增：

```python
compute_incremental_residual_loss(...)
build_gt_boundary_mask(...)
lowpass_avgpool(...)
```

不要改坏 `compute_residual_loss`，C/M 系列历史入口应保持可跑。

新 loss：

```text
L = lambda_final * L_final
  + lambda_boundary * L_boundary
  + lambda_grad * L_grad
  + lambda_keep_good_d1 * L_keep_good_D1
  + lambda_gate_sparse * L_gate_sparse
  + lambda_lowfreq_loss * L_lowfreq
  + lambda_invalid_keep * L_invalid_keep
```

固定实现公式：

```python
inv_gt = build_training_target(depth, valid_mask, target_space="metric_depth")
y_norm, _ = robust_normalize_target_per_sample(inv_gt, valid_mask, min_valid_pixels=min_valid_pixels)

base = out["base_norm"]          # same tensor as D1_norm
pred = out["pred"]
gate = out["gate"]
delta_raw = out["delta"]
delta_effective = out["delta_effective"]
gate_delta = gate * delta_effective

sample_ok = valid_mask.flatten(1).sum(dim=1) >= min_valid_pixels
effective_valid = valid_mask & sample_ok[:, None, None]

L_final = masked_mean(abs(pred - y_norm), effective_valid)
L_boundary = masked_mean(abs(pred - y_norm), effective_valid & gt_boundary_top10)
L_grad = gradient_l1(pred, y_norm, effective_valid)
L_keep_good_D1 = masked_mean(abs(gate_delta), effective_valid & good_d1_mask)
L_gate_sparse = masked_mean(gate, effective_valid)
L_lowfreq = masked_mean(abs(lowpass_avgpool(gate_delta, kernel_size=lowpass_kernel)), effective_valid)
L_invalid_keep = masked_mean(abs(gate_delta), sample_ok[:, None, None] & (~valid_mask))
```

`masked_mean` 必须对空 mask 返回连接到计算图的 0，不得返回 NaN。`L_lowfreq` 只惩罚最终实际施加到 D1 上的 `gate_delta` 的低频部分；`lambda_lp` 只用于生成 `delta_effective`，两者不能复用同一个参数含义。

建议 formal 基础 loss 权重显式写：

```text
lambda_final=1.0
lambda_boundary=2.0
lambda_grad=0.5
lambda_keep_good_d1=0.2
lambda_gate_sparse=0.05
lambda_invalid_keep=0.1
```

`lambda_lowfreq_loss` 不设全局默认值；必须由实验阶段显式指定：

```text
F1 lambda_lp sweep: lambda_lowfreq_loss=0.0
F2 q_good sweep: lambda_lowfreq_loss=0.0
F2b lowfreq_loss sweep: lambda_lowfreq_loss in 0.0 / 0.05 / 0.1
controls/ablation: 使用 SELECTED_LAMBDA_LOWFREQ_LOSS，若未跑 F2b 则显式为 0.0
```

原因：`lambda_lp` 是 architectural high-pass subtract，`lambda_lowfreq_loss` 是 loss-side low-frequency regularizer；两者都会抑制低频 correction。第一批只扫 `lambda_lp` 时必须令 `lambda_lowfreq_loss=0.0`，否则失败原因无法区分。

`L_boundary` 固定定义：

```text
GT depth gradient magnitude top-10% within each image's valid GT pixels
```

`L_keep_good_D1` 固定 per-image quantile：

```python
E1 = abs(D1_norm - y_norm)
threshold_i = quantile(E1_i[valid_i], q_good)
M_good_i = valid_i & (E1_i < threshold_i)
L_keep_good_D1 = mean(M_good_i * abs(gate * delta_effective))
```

`q_good` 必须显式传入，formal sweep 用：

```text
0.3 / 0.5 / 0.7
```

`lambda_lp` 的含义固定为 delta high-pass subtract strength：

```python
low = avgpool(delta_raw, kernel_size=31)
delta_effective = delta_raw - lambda_lp * low
```

不要把 `lambda_lp` 和 `lambda_lowfreq_loss` 混成同一个参数。

`L_invalid_keep`：

```python
invalid = sample_ok & (~valid_mask)
L_invalid_keep = mean(invalid * abs(gate * delta_effective))
```

如果某张图 invalid pixels 为空，返回 0，不要产生 NaN。

### C2. 新增训练入口

新增：

```text
foundation/tools/train_vkitti2_incremental_residual.py
```

要求：

```text
1. 支持 N2/N4 raw4 输入，复用 VKITTI2Raw。
2. 支持 N3/N5 RGB 输入，复用 VKITTI2HalfresRGBDepth。
3. 支持 KITTI sanity eval；N2/N4 用 halfres_raw_canonical_even_pad_crop_affine_disp，N3/N5 用 halfres_rgb_canonical_even_pad_crop_affine_disp。
4. 每次 eval 同时报告 final、D1/C2 base、D0。
5. config.json 保存全部实验语义参数和 C2 checkpoint path。
6. run_summary.json 保存 best_abs_rel、best_kitti_abs_rel、best_boundary_abs_rel、best_target_region_score。
```

`best_target_region_score` 固定定义为越低越好：

```text
target_region_score =
  mean_finite([
    region.final.boundary_abs_rel - region.D1.boundary_abs_rel,
    region.final.far50_abs_rel - region.D1.far50_abs_rel,
    region.final.dark_abs_rel - region.D1.dark_abs_rel,
    region.final.saturated_abs_rel - region.D1.saturated_abs_rel,
    region.final.fog_low_contrast_abs_rel - region.D1.fog_low_contrast_abs_rel  # if available
  ])
```

`best_abs_rel` 按 `overall.final.abs_rel` 选择；`best_boundary_abs_rel` 按 `region.final.boundary_abs_rel` 选择；`best_target_region_score` 按上面的 delta score 选择。每个 best 都要同时记录 epoch、checkpoint path、final/D1/KITTI delta，不能只记录一个数。

新增 `validate_args` 必须检查：

```text
raw feature source x3/ffm_mid:
  input_domain=raw4
  model_input_tensor=raw
  dataset_geometry_mode=vkitti2_even_fullres_halfres_2x2
  raw_storage_format=synthetic_packed_bayer_4ch_halfres
  front_end=c2_frozen_raw_ram_incremental
  raw_feature_encoder_trainable=true 或 false，但 formal 默认必须显式 true

rgb/d1 feature source:
  input_domain=rgb
  model_input_tensor=image
  dataset_geometry_mode=vkitti2_even_fullres_halfres_2x2
  raw_storage_format=not_applicable
  front_end=c2_frozen_rgb_incremental 或 c2_frozen_d1_incremental
  raw_feature_encoder_trainable=not_applicable

use_bridge/use_lora 等不相关参数不要隐式出现。
```

### C3. N 系列 eval summary schema

N 系列 evaluator 必须对三路 disparity field 分别独立 alignment 和 metric，不得复用 final 的 alignment 参数：

```python
final_disp = out["pred"][0].float().cpu().numpy()
d1_disp = out["D1_norm"][0].float().cpu().numpy()
d0_disp = (float(args.d0_sign) * out["D0"][0].float()).cpu().numpy()

aligned_final, align_final = affine_align_disp(depth_np, final_disp, valid_np)
aligned_d1, align_d1 = affine_align_disp(depth_np, d1_disp, valid_np)
aligned_d0, align_d0 = affine_align_disp(depth_np, d0_disp, valid_np)

metrics_final = compute_metrics(depth_np, aligned_final, valid_np, ...)
metrics_d1 = compute_metrics(depth_np, aligned_d1, valid_np, ...)
metrics_d0 = compute_metrics(depth_np, aligned_d0, valid_np, ...)
```

`overall.final`、`overall.D1`、`overall.D0` 和 region metrics 都必须使用各自独立 aligned depth/disparity 结果。`delta_final_minus_D1` 和 `delta_D1_minus_D0` 只在 metric 聚合后相减，不要用共享 align 参数制造差值。

VKITTI `val_metrics.json` 每个 epoch 推荐 schema：

```json
{
  "epoch": 0,
  "samples": 1000,
  "overall": {
    "final": {},
    "D1": {},
    "D0": {},
    "delta_final_minus_D1": {},
    "delta_D1_minus_D0": {}
  },
  "region": {
    "final": {},
    "D1": {},
    "D0": {},
    "delta_final_minus_D1": {},
    "delta_D1_minus_D0": {}
  },
  "diagnostics": {
    "mean_gate": 0.0,
    "mean_abs_delta": 0.0,
    "mean_abs_delta_effective": 0.0,
    "mean_abs_gate_delta": 0.0,
    "low_ratio": 0.0,
    "high_ratio": 0.0
  }
}
```

注意：成功标准比较的是 `final` vs `D1`，不是只看 `final` vs `D0`。

### C4. KITTI sanity

N 系列 KITTI 输出必须包含：

```text
overall.final
overall.D1
overall.D0
delta_final_minus_D1
```

报告时固定写：

```text
N2 - C2 on KITTI sanity = final_abs_rel - D1_abs_rel
```

如果 N2 在 VKITTI target regions 提升但 `KITTI final - D1 > kitti_regress_eps`，需要解释为可能继承或放大 VKITTI-specific calibration。

---

## 6. Step D：smoke

新增 smoke 脚本：

```text
finetune_stf/scripts/smoke/0527_smoke_vkitti_incremental_nseries.sh
```

smoke 路径：

```text
plans/0527/codex_smoke_nseries_<MMDD_HHMM>/
```

smoke 覆盖：

```text
1. py_compile:
   foundation/engine/models/dav2_incremental_residual.py
   foundation/tools/train_vkitti2_incremental_residual.py
   foundation/tools/eval_raw_residual_feature_ablation.py
   foundation/tools/analyze_residual_energy_frequency.py
   foundation/tools/make_residual_vs_c2_panels.py

2. N2 tiny train:
   epochs=1
   max_train_steps=2
   max_val_samples=4
   max_kitti_val_samples=4
   incremental_feature_source=x3
   lambda_lp=0.5
   lambda_lowfreq_loss=0.0
   q_good=0.5

3. N3 tiny train:
   同上，但 input_domain=rgb, incremental_feature_source=rgb。

4. N5 tiny train:
   同上，但 incremental_feature_source=d1。

5. feature ablation tiny eval:
   对 M1 x3 checkpoint 跑 max_val_samples=4 的 true/zero/shuffle。

6. energy/frequency tiny eval:
   对 C2 和 M1 各跑 max_val_samples=4。

7. vs-C2 panel tiny:
   sample_indices=0,72，输出到 smoke panel dir。
```

成功后删除：

```text
plans/0527/codex_smoke_nseries_<MMDD_HHMM>/
```

删除前必须做路径 guard：

```bash
case "${SMOKE_ROOT}" in
  *codex_smoke*|*smoke*|*debug*|*tmp*) ;;
  *) echo "[ERROR] Refusing to delete non-smoke path: ${SMOKE_ROOT}" >&2; exit 2 ;;
esac
[[ "${SMOKE_ROOT}" == "${ROOT}/plans/0527/"* ]] || {
  echo "[ERROR] Refusing to delete outside plans/0527: ${SMOKE_ROOT}" >&2
  exit 2
}
rm -rf "${SMOKE_ROOT}"
```

只允许删除这个明确 smoke root；不得删除 dataset、pretrained checkpoint、formal exp/heavy 输出或任何不含 smoke/debug/tmp/codex_smoke 标记的路径。

失败时保留，并报告：

```text
smoke root
失败命令
log path
```

---

## 7. Step E：formal queue script

新增：

```text
finetune_stf/scripts/formal/0527_run_vkitti_nseries_incremental_queue.sh
```

外层行为沿用现有 queue script：

```text
1. 自动创建 tmux session：<MMDD_HHMM>_vkitti_nseries_incremental。
2. queue log：finetune_stf/logs/<session>.queue.log。
3. 启动前检查 tmux session 不存在，拒绝复用。
4. 所有 run 启动时单独取 run_timestamp，保证 formal run name 是真实启动时间。
5. 每个 run 检查 save/heavy 路径不存在，拒绝覆盖。
6. 默认先 RUN_SMOKE=1。
```

formal run name 必须冗余编码关键 sweep 参数，方便人工检查和 summary fallback；但 `config.json` 仍然是解析实验语义的权威来源，run name 不能替代 config 校验。

```text
run_name =
  <MMDD_HHMM>_vkitti_<method_id_lower>_<incremental_feature_source>_
  lp<lambda_lp_tag>_q<q_good_tag>_lfl<lambda_lowfreq_loss_tag>_
  rft<raw_feature_encoder_trainable_tag>_vits_half187x621_<split_tag>_bs<bs>_e<epochs>

tag formatting:
  0.0 -> 0p0
  0.05 -> 0p05
  0.5 -> 0p5
  not_applicable -> na

examples:
  0527_1530_vkitti_n2_x3_lp0p5_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20
  0527_1712_vkitti_n5_d1_lp0p5_q0p5_lfl0p0_rftna_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20
```

默认路径：

```text
C2_CHECKPOINT=/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_11.pth
C2_RUN_DIR=/home/caq/6666_raw/dav2_raw_0522/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20
PRETRAINED=/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
VKITTI_TRAIN_LIST=finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt
VKITTI_VAL_LIST=finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt
KITTI_BASE=/mnt/drive/kitti
KITTI_VAL_SPLIT=/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt
EPOCHS=20
```

每个 formal run 的命令必须显式带上：

```text
--method-id
--encoder
--pretrained-from
--vkitti-train-list
--vkitti-val-list
--dataset-geometry-mode
--input-domain
--model-input-tensor
--raw-storage-format
--fullres-even-policy crop_bottom_to_even
--rgb-input-space halfres_2x2_area
--depth-target-space halfres_2x2_valid_mean
--front-end
--c2-checkpoint
--c2-run-dir
--incremental-feature-source
--delta-condition
--gate-condition
--raw-feature-encoder-trainable {true,false,not_applicable}
--input-height 187
--input-width 621
--min-depth 1.0
--max-depth 80.0
--residual-alpha 0.5
--d0-sign 1
--unprocessing-method <explicit>
--vkitti-unprocessing-preset <explicit>
--randomize-unprocessing 或 --no-randomize-unprocessing
--raw-adapter-backend <explicit>
--raw-adapter-cfa-pattern <explicit>
--raw-adapter-packed-channel-order <explicit>
--raw-adapter-rgb-transfer <explicit>
--raw-adapter-inverse-tone <explicit>
--raw-adapter-ccm <explicit>
--raw-adapter-red-gain-range <explicit or not_applicable>
--raw-adapter-blue-gain-range <explicit or not_applicable>
--raw-adapter-fixed-red-gain <explicit or not_applicable>
--raw-adapter-fixed-blue-gain <explicit or not_applicable>
--raw-adapter-fixed-light-scale <explicit or not_applicable>
--raw-adapter-dark-light-scale-range <explicit or not_applicable>
--raw-adapter-over-light-scale-range <explicit or not_applicable>
--raw-adapter-shot-noise <explicit or not_applicable>
--raw-adapter-read-noise <explicit or not_applicable>
--raw-adapter-noise-mean-mode <explicit>
--raw-adapter-black-level <explicit or not_applicable>
--raw-adapter-white-level <explicit or not_applicable>
--raw-adapter-random-seed-policy <explicit>
--raw-adapter-variant-policy <explicit>
--raw-adapter-variant-weights <explicit>
--lambda-lp
--lowpass-kernel 31
--q-good
--lambda-final 1.0
--lambda-boundary 2.0
--lambda-grad 0.5
--lambda-keep-good-d1 0.2
--lambda-gate-sparse 0.05
--lambda-lowfreq-loss <stage-explicit>
--lambda-invalid-keep 0.1
--eval-protocol per_image_affine_disp_depth_anything_v2
--eval-kitti
--kitti-base
--kitti-val-split
--kitti-eval-protocol <explicit>
--kitti-expected-val-samples 652
```

脚本环境开关：

```text
RUN_N2_LP_SWEEP=1
RUN_N2_Q_SWEEP=0
RUN_N2_LOWFREQ_SWEEP=0
RUN_N3_RGB=0
RUN_N4_FFM=0
RUN_N5_D1=0
RUN_N7_STOPGRAD=0

N2_LAMBDA_LP_LIST="0.0 0.3 0.5 0.8"
N2_Q_GOOD_LIST="0.3 0.5 0.7"
N2_LOWFREQ_LOSS_LIST="0.0 0.05 0.1"
SELECTED_LAMBDA_LP=0.5
SELECTED_Q_GOOD=0.5
SELECTED_LAMBDA_LOWFREQ_LOSS=0.0
```

不要一次默认全跑所有矩阵。先跑 N2 lambda sweep，看结果后再开启后续开关。

---

## 8. Step F：formal 实验矩阵

### F1. 第一批：N2 lambda_lp sweep

固定：

```text
method_id=N2
incremental_feature_source=x3
delta_condition=feature_only
gate_condition=feature_d1
q_good=0.5
lambda_lowfreq_loss=0.0
lambda_lp in 0.0 / 0.3 / 0.5 / 0.8
```

目的：

```text
在不叠加 loss-side low-frequency regularizer 的情况下，判断 architectural high-pass strength 是否压制 fog/far/large-object 低频 correction。
```

必须报告：

```text
overall
boundary
high-error
far50
dark
saturated
fog/low-contrast subset if available
low_ratio/high_ratio
KITTI final - D1
```

### F2. 第二批：N2 q_good sweep

先从 F1 选择一个 `SELECTED_LAMBDA_LP`，再跑：

```text
q_good in 0.3 / 0.5 / 0.7
lambda_lowfreq_loss=0.0
```

如果 F1 全部失败，不跑 q_good sweep，先回到诊断。

### F2b. 第三批：N2 lambda_lowfreq_loss sweep

先从 F1/F2 选择 `SELECTED_LAMBDA_LP` 和 `SELECTED_Q_GOOD`，再单独跑：

```text
lambda_lowfreq_loss in 0.0 / 0.05 / 0.1
```

目的：

```text
只评估 loss-side low-frequency regularizer 是否有额外收益或是否压制 far/fog/large-object correction。
```

如果未跑本阶段，`SELECTED_LAMBDA_LOWFREQ_LOSS` 必须显式保持 0.0，后续 controls 不得隐式使用 0.1。

### F3. 第四批：controls 和 ablation

使用预声明的 `SELECTED_LAMBDA_LP`、`SELECTED_Q_GOOD` 和 `SELECTED_LAMBDA_LOWFREQ_LOSS`：

```text
N3: C2 frozen + RGB incremental correction
N4: C2 frozen + ffm_mid incremental correction
N5: C2 frozen + D1-only extra head
N7: C2 frozen + x3 incremental, feature_d1_stopgrad delta
```

N3/N5 使用 RGB/KITTI RGB eval protocol；N4/N7 使用 RAW/KITTI RAW eval protocol。不要把 KITTI protocol 靠字符串推断，脚本中显式设置。

### F4. 第五批：N2 eval-time x3 ablation

对最佳 N2 checkpoint 跑。选择规则固定为：

```text
1. 首选 best_target_region_score 对应 checkpoint，但要求 VKITTI overall.final - D1 <= +0.002 且 KITTI final - D1 <= +0.005。
2. 如果没有 checkpoint 满足上面两个 no-regression 条件，则选择 overall.final.abs_rel 最低的 checkpoint，并在 summary 中标记 selected_by=overall_fallback。
3. 选择结果必须写入 feature ablation manifest：selected_run、selected_epoch、selected_checkpoint、selected_by、overall_delta、target_region_score、kitti_delta。
```

然后跑：

```text
true x3
zero x3
mean x3
shuffled x3 with --shuffle-policy stable_hash_far
```

这是成功标准必需项，不需要单独训练。若需要和旧 ablation protocol 对照，可额外跑 `--shuffle-policy next`，但正式结论和 success/failure 分类以 `stable_hash_far` 为准。

---

## 9. Step G：汇总工具

新增：

```text
foundation/tools/summarize_vkitti_nseries.py
```

输入：

```text
--c2-run <C2 run dir>
--n2-runs <one or more>
--n3-run optional
--n4-run optional
--n5-run optional
--n7-run optional
--feature-ablation-dir optional
--energy-frequency-dir optional
--output-dir plans/0527/result
```

输出：

```text
plans/0527/result/<timestamp>_vkitti_nseries_summary.md
plans/0527/result/<timestamp>_vkitti_nseries_summary.csv
plans/0527/result/<timestamp>_vkitti_nseries_summary.json
```

summary 表至少包含：

```text
method_id
run_name
best_epoch
lambda_lp
q_good
incremental_feature_source
delta_condition
gate_condition
raw_feature_encoder_trainable
trainable_params
c2_checkpoint
VKITTI final abs_rel
VKITTI D1 abs_rel
final - D1 abs_rel
boundary final
boundary final - D1
far50 final
far50 final - D1
dark final
saturated final
mean_gate
mean_abs_gate_delta
low_ratio
high_ratio
KITTI final abs_rel
KITTI D1 abs_rel
KITTI final - D1
feature ablation shuffled - true
x3_shuffle_gain
x3_zero_gain
shuffle_policy
shuffle_seed
donor_mapping_sha256
```

同时生成一段结论模板：

```text
strong_success / medium_success / failed
```

不要自动夸大结果；只按阈值填分类。

---

## 10. 验收标准

### 10.1 代码验收

必须全部通过：

```text
1. py_compile 成功。
2. N2/N3/N5 tiny train 能写出 config.json、val_metrics.json、run_summary.json。
3. tiny KITTI sanity 能写出 kitti_val_metrics.json。
4. N 系列 val_metrics 同时含 final/D1/D0。
5. C/M 系列旧训练入口不受影响，至少 import/py_compile 成功。
6. failed smoke 产物保留；successful smoke 产物只删除 codex_smoke 路径。
```

### 10.2 实验验收

所有比较默认使用 `abs_rel`，越低越好。先定义：

```text
overall_delta = VKITTI overall.final.abs_rel - VKITTI overall.D1.abs_rel
boundary_delta = VKITTI region.final.boundary_abs_rel - VKITTI region.D1.boundary_abs_rel
target_region_delta = best_target_region_score
kitti_delta = KITTI overall.final.abs_rel - KITTI overall.D1.abs_rel
x3_shuffle_gain = shuffled_x3_abs_rel - true_x3_abs_rel
x3_zero_gain = zero_x3_abs_rel - true_x3_abs_rel
```

默认阈值：

```text
overall_improve_eps = 0.002
region_improve_eps = 0.003
feature_gain_eps = 0.001
kitti_regress_eps = 0.005
tie_eps = 0.002
```

强成功：

```text
overall_delta <= -overall_improve_eps
boundary_delta <= -region_improve_eps
fog/low-contrast delta <= -region_improve_eps if available
N2 overall.abs_rel <= N3 overall.abs_rel - feature_gain_eps
x3_shuffle_gain >= feature_gain_eps
kitti_delta <= kitti_regress_eps
```

中等成功：

```text
overall_delta <= tie_eps
boundary_delta <= -region_improve_eps 或 target_region_delta <= -region_improve_eps
x3_shuffle_gain >= feature_gain_eps
但 N2 不一定超过 N3 RGB control
```

失败：

```text
overall_delta > tie_eps
且 boundary_delta > -region_improve_eps
且 target_region_delta > -region_improve_eps
或 x3_shuffle_gain < 0.5 * feature_gain_eps
```

失败时不要继续堆模型复杂度。优先检查：

```text
1. x3 是否在 feature ablation 中真实有效。
2. lambda_lp 是否压死 fog/far 低频 correction。
3. invalid_keep/gate_sparse 是否过强。
4. synthetic RAW 生成是否仍然不够接近真实 RAW。
```

---

## 11. 推荐实际执行命令顺序

### 11.1 先跑现有 checkpoint 诊断

```bash
bash finetune_stf/scripts/formal/0527_run_existing_residual_diagnostics_queue.sh
```

记录脚本输出：

```text
tmux attach -t <session>
tail -f <queue_log>
```

### 11.2 实现 N 系列后跑 smoke

```bash
bash finetune_stf/scripts/smoke/0527_smoke_vkitti_incremental_nseries.sh
```

### 11.3 跑 N2 lambda sweep

```bash
RUN_N2_LP_SWEEP=1 \
RUN_N2_Q_SWEEP=0 \
RUN_N3_RGB=0 \
RUN_N4_FFM=0 \
RUN_N5_D1=0 \
RUN_N7_STOPGRAD=0 \
bash finetune_stf/scripts/formal/0527_run_vkitti_nseries_incremental_queue.sh
```

### 11.4 根据 N2 lambda 结果跑 q_good sweep

```bash
RUN_N2_LP_SWEEP=0 \
RUN_N2_Q_SWEEP=1 \
RUN_N2_LOWFREQ_SWEEP=0 \
SELECTED_LAMBDA_LP=<chosen> \
bash finetune_stf/scripts/formal/0527_run_vkitti_nseries_incremental_queue.sh
```

### 11.5 根据 q_good 结果可选跑 lowfreq_loss sweep

```bash
RUN_N2_LP_SWEEP=0 \
RUN_N2_Q_SWEEP=0 \
RUN_N2_LOWFREQ_SWEEP=1 \
SELECTED_LAMBDA_LP=<chosen> \
SELECTED_Q_GOOD=<chosen> \
bash finetune_stf/scripts/formal/0527_run_vkitti_nseries_incremental_queue.sh
```

如跳过本步，后续显式使用 `SELECTED_LAMBDA_LOWFREQ_LOSS=0.0`。

### 11.6 跑 controls 和 ablation

```bash
RUN_N2_LP_SWEEP=0 \
RUN_N2_Q_SWEEP=0 \
RUN_N2_LOWFREQ_SWEEP=0 \
RUN_N3_RGB=1 \
RUN_N4_FFM=1 \
RUN_N5_D1=1 \
RUN_N7_STOPGRAD=1 \
SELECTED_LAMBDA_LP=<chosen> \
SELECTED_Q_GOOD=<chosen> \
SELECTED_LAMBDA_LOWFREQ_LOSS=<chosen-or-0.0> \
bash finetune_stf/scripts/formal/0527_run_vkitti_nseries_incremental_queue.sh
```

### 11.7 汇总

```bash
conda run --live-stream -n dav3 python foundation/tools/summarize_vkitti_nseries.py \
  --c2-run finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20 \
  --output-dir plans/0527/result
```

实际执行时把 `--n2-runs/--n3-run/--n4-run/--n5-run/--n7-run` 补成完成的 run dir。

---

## 12. 结果解读口径

后续写结论时固定比较顺序：

```text
1. 先看 N2 final vs C2/D1。
2. 再看 N2 vs N3 RGB control。
3. 再看 x3_shuffle_gain 是否达到 feature_gain_eps。
4. 最后看 N2 vs old M1 x3+D0 single-head。
```

不要把 `N2 > D0` 当作 RAW 有效的证据，因为 C2 已经证明 D0-only calibration 很强。

如果 N2 只在 boundary/fog/saturated 有收益，结论写成：

```text
RAW-like x3 provides region-specific local-detail refinement on top of a strong frozen RGB depth prior.
```

如果 N2 不能超过 C2/D1 且 `x3_shuffle_gain < 0.5 * feature_gain_eps`，结论写成：

```text
Under the current inverse-ISP synthetic RAW setting, residual gains are still explained mainly by D0-conditioned calibration rather than robust marginal RAW-like cues.
```
