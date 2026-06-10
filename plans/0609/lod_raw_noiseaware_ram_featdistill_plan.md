# LOD RAW_Dark 下一阶段执行计划：Noise-aware RAM + mid/deep feature distillation

版本：v1.1（审核修订：teacher 风险与备选、gate 初始化、短训规格、阈值统一、loader 前提）  
日期：2026-06-09  
目标：在不引入过多变量的前提下，验证 **RAW_dark 的可学习去噪/恢复能力** 与 **RAW_normal expert feature matching** 是否能恢复 RAW_dark → RAW_normal 的性能 gap。

---

## 执行进度（Codex 实时回写）

更新时间：2026-06-09 22:31

- [x] 13.1.1 paired LOD RAW dataset 代码实现：新增 `lod_true_raw_dark_normal_pair_rgb16` / `raw_rgb16_dark_normal_pair`，batch 返回 `raw/raw_dark/raw_normal`，并同步注册 `resolved.py`、`train.py`、`raw_ram.py`。
- [x] 13.2.1 paired dataset smoke：通过，检查 4 个样本的 `raw/raw_dark/raw_normal/image/depth/valid_mask` shape、路径、`sample_id`、`crop_box`；成功后已删除 `/tmp/codex_smoke_lod_pair_dataset_*`。
- [x] 13.1.2/13.1.5 resolved config 代码实现：新增 `student_init_from`、`teacher_ckpt`、`feat_distill*`、`raw_ram_local_*` 语义字段与 disabled/active 校验；L0/L3 parser smoke 已通过。
- [x] 13.1.3/13.1.4 模型代码实现：`RamCore3` 支持 gated `noiseaware_v1` local residual branch；`RawRgb16Ram3DepthModel.forward(..., return_features=True)` 返回 DAv2 depth-head layers token 和 RAM debug。
- [x] 13.2.2 model/init smoke：L3 student/teacher 单 batch 通过；student compatible load missing=9（新增 local branch），teacher missing=0；`loss_depth`/`loss_feat` finite；local branch 参数确认在 `raw_front_end` optimizer group；teacher grad 全为 None。
- [x] 13.2.3 backward smoke：`torchrun --nproc_per_node=1`、L3、`debug-max-train-steps=2` 通过；完成 optimizer step、TensorBoard 写入、checkpoint 保存；成功后已删除 `/tmp/codex_smoke_lod_featdistill_train_*` 与 heavy 输出目录。
- [x] 13.3 short-train formal script：新增 `finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_short_e10_queue.sh`，覆盖 L0/L1/L2/L3/L0_seed123；`--audit` 已通过。
- [x] 13.3 short-train 正式队列已启动：tmux session=`0609_2045_lod_raw_noiseaware_featdistill_short_e10_queue`，queue log=`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0609_2045_lod_raw_noiseaware_featdistill_short_e10_queue.queue.log`；attach=`tmux attach -t 0609_2045_lod_raw_noiseaware_featdistill_short_e10_queue`；monitor=`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0609_2045_lod_raw_noiseaware_featdistill_short_e10_queue.queue.log`。
- [x] 13.3 short-train 监控节点（2026-06-09 20:56）：L0 完成，`status=0`；pretrain `lod_val d1=0.8451`，最终 best_lod_d1=`0.8456`（epoch 9），best/current/last checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2045_L0_lod_raw_pair_lora_decoder_e10_featnone_localnone/`。
- [x] 13.3 short-train 监控节点（2026-06-09 20:56）：L1 已启动，run=`0609_2045_L1_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_localnone`，配置为 `feat_distill=dav2_middeep_cosine`、layers=`5,8,11`、lambda=`0.05`、local residual=`none`；student/teacher compatible load 均 missing=0/unexpected=0，pretrain `lod_val d1=0.8451`、`lod_train_proxy d1=0.8667`，已进入 epoch 0/10。
- [x] 13.3 short-train 监控节点（2026-06-09 21:00）：L1 已完成 epoch 0-2；epoch 1 best_lod_d1 暂为 `0.8454`（`lod_train_proxy d1=0.8628`），低于 L0 最终 best `0.8456`，但训练/teacher distill 路径运行正常；当前进入 epoch 3/10。
- [x] 13.3 short-train 监控节点（2026-06-09 21:06）：L1 已完成 epoch 0-7；epoch 7 best_lod_d1 刷新到 `0.8466`（`lod_train_proxy d1=0.8721`），已高于 L0 最终 best `0.8456`；best/current checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2045_L1_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_localnone/`，当前进入 epoch 8/10。
- [x] 13.3 short-train 监控节点（2026-06-09 21:08）：L1 完成，`status=0`；最终 best_lod_d1=`0.8484`（epoch 9，`lod_train_proxy d1=0.8718`），高于 L0 最终 best `0.8456`，best/current/last checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2045_L1_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_localnone/`。
- [x] 13.3 short-train 监控节点（2026-06-09 21:10）：L2 已启动，run=`0609_2045_L2_lod_raw_pair_lora_decoder_e10_featnone_noiseawarev1_gate003_s01`，配置为 `feat_distill=none`、local residual=`noiseaware_v1`、hidden_ch=`16`、scale=`0.1`、gate_init=`0.03`、gate_mode=`channel`；新增 local branch missing_keys=9 且 compatible load 正常；pretrain `lod_val d1=0.8449`；epoch 1 best_lod_d1 暂为 `0.8441`，当前进入 epoch 2/10。
- [x] 13.3 short-train 监控节点（2026-06-09 21:18）：L2 完成，`status=0`；最终 best_lod_d1=`0.8460`（epoch 7，`lod_train_proxy d1=0.8715`），略高于 L0 最终 best `0.8456`，低于 L1 `0.8484`；best/current/last checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2045_L2_lod_raw_pair_lora_decoder_e10_featnone_noiseawarev1_gate003_s01/`。
- [x] 13.3 short-train 监控节点（2026-06-09 21:19）：L3 已启动，run=`0609_2045_L3_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_noiseawarev1_gate003_s01`，配置为 `feat_distill=dav2_middeep_cosine`、layers=`5,8,11`、lambda=`0.05`、local residual=`noiseaware_v1`、hidden_ch=`16`、scale=`0.1`、gate_init=`0.03`、gate_mode=`channel`；student 新增 local branch missing_keys=9 且 compatible load 正常，teacher missing=0/unexpected=0；pretrain `lod_val d1=0.8449`、`lod_train_proxy d1=0.8667`，已进入 epoch 0/10。
- [x] 13.3 short-train 监控节点（2026-06-09 21:31）：L3 完成，`status=0`；最终 best_lod_d1=`0.8485`（epoch 9，`lod_train_proxy d1=0.8726`），略高于 L1 `0.8484`、高于 L2 `0.8460` 和 L0 `0.8456`，但与 L1 差距仅 `+0.0001`；best/current/last checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2045_L3_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_noiseawarev1_gate003_s01/`。
- [x] 13.3 short-train 监控节点（2026-06-09 21:31）：L0_seed123 已启动，run=`0609_2045_L0_seed123_lod_raw_pair_lora_decoder_e10_featnone_localnone`，配置为 L0 seed=`123`、`feat_distill=none`、local residual=`none`；student compatible load missing=0/unexpected=0，已进入 pretrain eval。
- [x] 13.3 short-train 监控节点（2026-06-09 21:38）：L0_seed123 已完成 epoch 0-6；epoch 5 best_lod_d1 暂为 `0.8485`（`lod_train_proxy d1=0.8575`），已达到 L3 最终 best `0.8485`，说明 short-train 中 L1/L3 相对 L0(seed42) 的 `+0.0028/+0.0029` 很可能处在 seed 波动范围内；当前进入 epoch 7/10。
- [x] 13.3 short-train 队列结果汇总（2026-06-09 21:41）：队列正常结束，`[QUEUE_DONE]`；最终 best D1：L0=`0.8456`、L1=`0.8484`、L2=`0.8460`、L3=`0.8485`、L0_seed123=`0.8502`。L0_seed123 明显高于 L1/L3，说明本轮 e10 单 seed ablation 不能证明 feature distill 或 local residual 有稳定增益。
- [x] 13.4 short-train feature probe（2026-06-09 21:43）：已对 L0/L1/L2/L3/L0_seed123 的 best checkpoint 跑 112-sample dual-model probe；每个 run 的 `feature_probe/feature_probe.csv` 与 `feature_probe_summary.json` 已生成。汇总显示 L1/L3 的 L5/L8/L11 cosine distance 低于 L0/L0_seed123，但 D1 不超过 L0_seed123。
- [x] 13.3/13.4 short-train 汇总文件：已生成 `plans/0609/lod_raw_noiseaware_featdistill_short_0609_2045_summary/matrix.csv`、`plans/0609/lod_raw_noiseaware_featdistill_short_0609_2045_summary/recovery.csv`、`plans/0609/lod_raw_noiseaware_featdistill_short_0609_2045_summary/ablation_summary.md`。
- [x] 13.5 决策门槛（2026-06-09 21:43）：不进入 full e40。原因：`L3=0.8485` 未满足 `L3 > L0 + 0.010`（seed42 L0 阈值为 `0.8556`），且 `L0_seed123=0.8502` 高于 L3；L3 相对 L0(seed42) 的 AbsRel 从 `3.5956` 到 `3.7276`，约 `+3.7%`，超过 `<2%` 恶化门槛。`L3 >= max(L1,L2)-0.002` 虽满足，但不足以触发 full training。
- [x] 13.5 L3 去噪效果复盘（2026-06-09 21:50）：已补充 L3 vs L1 RAM 前端诊断和 L3 train-viz 面板；`gate_tanh=[0.03134,0.03290,0.03090]`、`0.1*gate≈0.0031`，L3 local residual 平均贡献 `0.000842`，仅为 `x3_ram` 平均幅度的 `0.124%`。可视化显示 `L3 RAM pre-local` 与 `L3 RAM out` 肉眼几乎一致；因此 L3 的 `+0.000045` D1 相对 L1 不能解释为强显式去噪，主要结论仍是不进入 full e40。
- [x] 13.6 full-training formal script：新增 `finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_full_e40_queue.sh`，默认 L3、支持 `FULL_GROUPS=L1,L2,L3`；默认 L3 `--audit` 已通过。
- [x] 13.4 dual-model feature probe：新增 `tools/lod_raw_featdistill_feature_probe.py`，支持 Student RAW_dark checkpoint vs Teacher RAW_normal checkpoint 的 layer5/8/11 token cosine distance；1-sample smoke 已通过并删除成功 smoke 输出。
- [x] 13.4 matrix/summary 工具：新增 `tools/lod_raw_featdistill_matrix_summary.py`，输出 `matrix.csv`、`recovery.csv`、`ablation_summary.md`；单 run smoke 已通过并删除成功 smoke 输出。
- [x] 13.7 no-LoRA short diagnostic 脚本（2026-06-09 21:52）：新增 `finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_w0_w3_short_e10_queue.sh`，覆盖 W0/W3/W0_seed123；student=`0608_1739...W0.../best_model.pth`，teacher=`0609_0010...W0.../best_model.pth`；`--audit` 已通过，确认 `lora=none` 下 `lora_rank/lora_alpha/lora_lr/lora_tap_layers=not_applicable`，W3 teacher recipe=`decoder_w0`。
- [x] 13.7 no-LoRA short diagnostic 队列已启动（2026-06-09 21:53）：tmux session=`0609_2153_lod_raw_noiseaware_featdistill_w0_w3_short_e10_queue`，queue log=`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0609_2153_lod_raw_noiseaware_featdistill_w0_w3_short_e10_queue.queue.log`；attach=`tmux attach -t 0609_2153_lod_raw_noiseaware_featdistill_w0_w3_short_e10_queue`；monitor=`tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0609_2153_lod_raw_noiseaware_featdistill_w0_w3_short_e10_queue.queue.log`。
- [x] 13.7 no-LoRA W0 监控节点（2026-06-09 22:03）：W0 完成，`status=0`；pretrain `lod_val d1≈0.8298`，最终 best_lod_d1=`0.8312`（epoch 7，`lod_train_proxy d1=0.8348`），best/current/last checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2153_W0_lod_raw_pair_decoder_e10_featnone_localnone/`。
- [x] 13.7 no-LoRA W3 监控节点（2026-06-09 22:03）：W3 已启动，配置为 `lora=none`、`feat_distill=dav2_middeep_cosine`、layers=`5,8,11`、lambda=`0.05`、local residual=`noiseaware_v1`、teacher recipe=`decoder_w0`；student 新增 local branch missing_keys=9 且 compatible load 正常，teacher missing=0/unexpected=0；pretrain `lod_val d1=0.8298`、`lod_train_proxy d1=0.8317`，已进入 epoch 0/10。
- [x] 13.7 no-LoRA W3 监控节点（2026-06-09 22:15）：W3 完成，`status=0`；最终 best_lod_d1=`0.8310`（epoch 9，`lod_train_proxy d1=0.8326`），低于 W0 `0.8312`，明显未达到 W0+0.005 诊断阈值；best/current/last checkpoint 已保存到 `/mnt/drive/3333_raw/0000_exp_ckpt/0609_2153_W3_lod_raw_pair_decoder_e10_featmiddeep_lam005_noiseawarev1_gate003_s01/`。
- [x] 13.7 no-LoRA W0_seed123 监控节点（2026-06-09 22:15）：W0_seed123 已启动，配置为 W0 seed=`123`、`lora=none`、`feat_distill=none`、local residual=`none`；student compatible load missing=0/unexpected=0；pretrain `lod_val d1=0.8299`、`lod_train_proxy d1=0.8353`，已进入 epoch 0/10。
- [x] 13.7 no-LoRA short diagnostic 队列结果（2026-06-09 22:25）：队列正常结束，`[QUEUE_DONE]`；最终 best D1：W0=`0.8312`、W3=`0.8310`、W0_seed123=`0.8308`。W3 没有超过 W0，且 seed123 W0 也与 W0 同量级，当前没有证据支持 no-LoRA 下补跑 W1/W2 或进入 no-LoRA full e40。
- [x] 13.7 no-LoRA feature probe / 汇总（2026-06-09 22:31）：已对 W0/W3/W0_seed123 的 best checkpoint 跑 112-sample dual-model probe；汇总目录为 `plans/0609/lod_raw_noiseaware_featdistill_w0_w3_short_0609_2153_summary/`，包含 `matrix.csv`、`recovery.csv`、`ablation_summary.md`。recovery 使用 no-LoRA strict 端点 `M_DD=0.829694`、`M_NN=0.887073`；W3 的 token cosine distance 略低于 W0（L5 `0.116468` vs `0.117962`，L8 `0.096780` vs `0.098118`，L11 `0.121787` vs `0.123409`），但 D1=`0.8310` 仍低于 W0=`0.8312`，因此 feature matching 没有转化为有效验证集增益。
- [x] 13.7 no-LoRA 决策（2026-06-09 22:31）：停止 no-LoRA 分支扩展；不补 W1/W2，不启动 no-LoRA full e40。后续主结论保持：LoRA short 也未过 full gate，no-LoRA short 进一步排除了“只是不需要 LoRA 才有效”的解释。

- [x] 14 第二轮诊断（round-2）设计与执行（2026-06-09 22:55）：用户确认第一轮“去噪失败”。新增两条独立轴的 e10 诊断队列脚本 `finetune_stf/scripts/formal/0609_run_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue.sh`（A=放大 local 预算 scale0.1→0.3/0.5、gate0.03→0.1；B=same-param teacher `C_dark(RAW_normal)`；组合 AB；baseline 第3 seed），新增 `RUN_GROUPS` 组过滤；6 组 `--audit` 全过、`same_param_clean` 路径代码已支持、无需改代码。按用户决定先单跑最强组合 `AB_s05_sp` 作“上限探针”，不达标即停、不跑其余组。tmux session=`0609_2255_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue`。详见 §14。

---

## 0. 当前结论与本计划立场

### 0.1 已确认事实

Canonical LoRA recipe 的 2×2 cross-eval：

| matrix | checkpoint | input | D1 | 解释 |
|---|---|---|---:|---|
| M_DD | C_dark | RAW_dark | 0.845279 | 当前 RAW_dark baseline |
| M_DN | C_dark | RAW_normal | 0.880533 | dark ckpt 可以吃 normal input，显著提升 |
| M_NN | C_normal | RAW_normal | 0.898946 | RAW_normal expert / oracle |
| M_ND | C_normal | RAW_dark | 0.816972 | normal ckpt 直接吃 dark input 不鲁棒 |

补充结论：

- `R_DN = 0.656898`，说明输入质量是重要因素，但 `C_dark(RAW_normal)` 只能解释约 66% 的 gap，不能单独作为完整 oracle。
- BN recalibration 对 `M_DN` 没有实质提升，不支持“主要是 BN running stats 错位”。
- RAW_dark attribution 已显示：噪声 / 低 SNR 是主导退化源；单纯 exposure / dynamic range 不足不是主因。
- 手工 median / bilateral / tone / oracle tone+denoise 没有稳定提升 D1，因此下一步不应走 test-time hand preprocessing。
- feature probe 显示：手工 preprocessing 只轻微改善 RAM output 和 early token，不能有效拉近 DAv2 layer5/8/11。

### 0.2 本计划认可的方向

采用：

```text
Teacher: frozen C_normal, input = RAW_normal
Student: init from C_dark, input = RAW_dark
Training: 继续使用原始 depth loss，再只新增 mid/deep feature loss
Architecture: 不外接 denoiser，先在 RAM 内部加一个轻量 noise-aware local residual branch
Noise augmentation: 暂不加入
Loss: 第一版只用 L_depth + λ_feat * L_feat_middeep
LoRA: 同时做 with-LoRA 与 no-LoRA 对照，但以 canonical LoRA recipe 为主线
```

核心原则：

> 先验证“训练式 task-aware denoise + RAW_normal expert feature matching”是否有效；不要在第一版同时加入 noise aug、edge loss、RAM loss、multi-teacher、reliability loss 等多个变量。

---

## 1. Teacher / Student 定义

### 1.1 主线 teacher

使用 RAW_normal expert：

```text
Teacher T_N:
  checkpoint = C_normal
  input      = RAW_normal
  mode       = eval
  grad       = off
```

Teacher 只用于提供 feature target，第一版不额外使用 teacher depth 作为新 depth label。

理由：

1. `C_normal(RAW_normal)` 是当前最干净、最强的 RAW-normal-domain expert。
2. `C_dark(RAW_normal)` 虽然有效，但 D1=0.880533，距离 `C_normal(RAW_normal)=0.898946` 仍有明显差距。
3. `C_normal(RAW_dark)` 明显低于 `C_dark(RAW_dark)`，所以不能直接用 C_normal 初始化 student，也不能期待 normal ckpt 本身对 RAW_dark 噪声鲁棒。

> 风险与边界（重要）：
> teacher 选 `C_normal(RAW_normal)` 而非同参数的 `C_dark(RAW_normal)`，意味着 feature target 落在与 student **不同的参数空间**（LoRA 线尤甚，backbone LoRA 不同）。带来两点风险：
> 1. feature loss 可能把 student 的 LoRA/decoder 拉向 normal-expert 解，而该解对 RAW_dark 噪声并不鲁棒（参考 `M_ND = 0.816972`）。典型表现是 L1（只 feature loss、不清理输入）出现 M_ND 式退化（见 §8）。
> 2. Q1「feature matching 是否有效」会与「参数迁移」交织，归因不纯。
>
> 本计划第一版仍**先按 `C_normal` teacher 跑**，目的是冲过 `M_DN = 0.880533` 去够 `M_NN = 0.898946`。但若出现 §8 中 `L1 < L0` 退化、或 feature distance 降而 D1 不升，**优先改用同参数 teacher `C_dark(RAW_normal)`** 作对照——它的 token 完全可由 student 自身参数靠清理输入到达，能干净分离「输入清理 vs 参数迁移」两条轴。该对照在 §9 中已由纯 v2 提升为「随时可触发的备选 teacher」。

### 1.2 Student

```text
Student S:
  checkpoint init = C_dark
  input           = RAW_dark
  trainable       = RAM + decoder；LoRA 视实验组决定是否开启
```

对于 canonical LoRA recipe：

```text
Student init = canonical C_dark LoRA checkpoint
Teacher      = canonical C_normal LoRA checkpoint
Trainable    = RAM + LoRA + decoder
```

对于 no-LoRA 对照：

```text
Student init = C_dark_W0 decoder-only checkpoint
Teacher      = C_normal_W0 decoder-only checkpoint
Trainable    = RAM + decoder
```

不要混用 LoRA teacher 与 no-LoRA student，除非后续专门研究 cross-recipe distillation。第一版应保持 teacher/student recipe 对齐。

---

## 2. 模型改动：Noise-aware RAM local residual branch

### 2.1 不采用外接 denoiser

不做：

```text
RAW_dark -> external image denoiser -> RamCore3 -> DAv2
```

原因：

- 手工 denoise / tone preprocessing 已经没有收益。
- 外接 denoiser 会把“输入改变”和“RAM 学习能力改变”混在一起。
- 普通图像去噪可能抹掉 depth boundary。

采用：

```text
RAW_dark -> RamCore3 + internal local residual branch -> DAv2
```

### 2.2 第一版建议的最小结构

为了减少变量，不改 RamCore3 的四个 ISP branch 和 FFM 主体。只在 RamCore3 输出 `x3` 后，增加一个 **gated local residual branch**：

```text
raw input x_raw
        │
        ├── RamCore3 原路径 -> x3_ram
        │
        └── LocalDenoiseBranch([x_raw, x3_ram]) -> delta_x3

x3_out = x3_ram + s * tanh(g) * tanh(delta_x3)
```

建议默认：

```text
s = 0.1                          # x3 已过 RamCore3 末端 BatchNorm2d(3)，近似单位方差；
                                 # 故 s=0.1 ≈ 对单位方差信号最多 ±10% 扰动，量级合理
g : 标量或逐通道(3) 皆可          # 第一版建议逐通道，便于在日志里观察各通道行为
初始化 g 使 tanh(g) ≈ 0.02~0.05   # 小非零，不要用 g=0
```

不要用 `g = 0` 初始化。因为

```text
x3_out = x3_ram + s · tanh(g) · tanh(delta_x3)
当 g=0 时 ∂x3_out/∂(branch 权重) = s · tanh(g) · … = 0
```

即开局只有 g 收到梯度、delta-branch 的卷积权重收不到梯度，必须等 g 离开 0 才开始学，造成被 g 限速的耦合慢启动。改用 `tanh(g) ≈ 0.02~0.05` 的小非零初始化：既近似恒等、几乎不破坏 C_dark checkpoint，又能让 branch 从第一步起就拿到梯度。

### 2.3 LocalDenoiseBranch 建议

第一版保持轻量：

```text
input channels : 6  # concat(raw_dark, x3_ram)
hidden channels: 16 or 32
structure      : Conv3x3 -> GN/ReLU -> depthwise/local Conv3x3 -> GN/ReLU -> Conv1x1 -> 3ch residual
normalization  : GroupNorm or no norm；优先避免新 BN
output         : delta_x3
```

推荐不用 BatchNorm 的原因：cross-eval 已经显示 BN recalibration 不是主因；新增 branch 如果用 BN，可能引入新的 domain-stat 变量。GroupNorm 或 no norm 更干净。

### 2.4 第一版不加入显式 denoise target

不要加入：

```text
x3_out ≈ teacher x3
RAW_restored ≈ RAW_normal
image L1 / SSIM / PSNR loss
```

第一版只让 local branch 通过 depth loss 和 mid/deep feature loss 学到 task-aware correction。

---

## 3. Feature matching 设计

### 3.1 对齐对象

对齐 DAv2 中后层 token，而不是 RAM output。

第一版使用：

```text
layers = [5, 8, 11]
```

原因：之前 feature probe 显示，hand preprocessing 轻微改善 RAM x3 / layer2，但 layer5/8/11 没有接近 RAW_normal，因此真正需要约束的是中后层表示。

### 3.2 Loss 形式

建议第一版使用 normalized token cosine loss：

```text
F_s_l = normalize(Student layer-l tokens, dim=-1)
F_t_l = normalize(Teacher layer-l tokens, dim=-1)

L_feat_l = mean(1 - cosine(F_s_l, stopgrad(F_t_l)))
L_feat   = mean_l L_feat_l, l in {5,8,11}
```

说明：

- `get_intermediate_layers(..., norm=True)` 返回的已是 **LayerNorm 后、且剔除 CLS/register 的 patch token**（dinov2 切片 `out[:, 1+num_register_tokens:]`）。feature loss 只用这些 patch token，teacher/student 逐 patch 对齐，无需自己再排除 CLS。
- cosine 会归一掉 magnitude；因 DPT 头本身吃 LayerNorm 后的 token，重方向是合理的。若后续发现域差主要体现在范数上，可在归一化特征上改/加 smooth-L1 作为备选（不在第一版默认）。

第一版不加 token mask、不加 edge weighting、不加 reliability weighting。保持变量少。

### 3.3 总 loss

第一版只使用：

```text
L_total = L_depth + λ_feat * L_feat_middeep
```

其中：

- `L_depth`：沿用当前 depth pseudo-label loss，不改变伪标签来源。
- `L_feat_middeep`：新增项，teacher 为 `C_normal(RAW_normal)`。

### 3.4 λ_feat 建议

因为 teacher 是 `C_normal` 而不是 same-parameter teacher，feature loss 需要保守。

建议第一轮 sweep：

```text
λ_feat ∈ {0.02, 0.05, 0.10}
```

执行顺序：

1. 先跑 `0.05`。
2. 如果 D1 下降但 feature distance 明显下降，说明约束过强或目标层不合适，再降到 `0.02`。
3. 如果 D1 小幅提升且 feature distance 下降，可以试 `0.10`。

### 3.5 暂不加入的 loss

第一版暂不加入：

```text
L_ram
L_edge
L_reliability
L_image_reconstruction
L_teacher_depth
L_noise_aug_consistency
```

如果 local residual branch 出现明显过强修改，再加入一个很小的 residual penalty：

```text
L_res = mean(delta_x3^2)
```

但这属于 v1b，不属于第一版默认设置。

---

## 4. 实验矩阵

### 4.1 主实验：canonical LoRA recipe

主线使用当前 canonical LoRA pair：

```text
C_dark   = 0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
C_normal = 0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly
```

训练集：RAW_dark input + RAW_normal paired teacher input。  
验证集：RAW_dark input。  
Teacher 只在训练时使用，不在推理时使用。

| ID | Local denoise branch | Feature loss | LoRA trainable | 目的 |
|---|---:|---:|---:|---|
| L0 | 否 | 否 | 是 | current recipe 继续训练 sanity / 复现 baseline |
| L1 | 否 | 是 | 是 | 单独验证 feature matching 是否有效 |
| L2 | 是 | 否 | 是 | 单独验证 learnable local denoise branch 是否有效 |
| L3 | 是 | 是 | 是 | 主实验：denoise + feature matching |

优先级：

```text
先跑 L0/L1/L2/L3 的短训版本。
若 L3 > max(L1,L2) 且 > L0，则进入 full training。
```

### 4.2 no-LoRA 对照实验

使用 W0 decoder-only pair：

```text
C_dark_W0   = 0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly
C_normal_W0 = 0609_0010_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly
```

| ID | Local denoise branch | Feature loss | LoRA trainable | 目的 |
|---|---:|---:|---:|---|
| W0 | 否 | 否 | 否 | no-LoRA baseline |
| W1 | 否 | 是 | 否 | no-LoRA 下 feature matching 是否有效 |
| W2 | 是 | 否 | 否 | no-LoRA 下 local branch 是否有效 |
| W3 | 是 | 是 | 否 | no-LoRA 主实验 |

如果计算资源有限，先跑：

```text
L0/L1/L2/L3 + W0/W3
```

如果 L3 有明显收益，再补 W1/W2 做完整归因。

---

## 5. 训练流程

### 5.1 数据 batch 要求

每个 training sample 需要同时返回：

```text
raw_dark       # student input
raw_normal     # teacher input
pseudo_depth   # 原有 RGB_normal DAv2-L pseudo label
valid_mask     # 原有 depth valid mask，如已有
sample_id      # 用于 debug / feature dump
```

注意：

- `raw_dark` 和 `raw_normal` 必须来自同一样本对。
- spatial crop / pad 必须保持一致；teacher 和 student 的 token 才能逐 patch 对齐。
- 不要对 teacher input 使用额外随机增强，除非 student input 做完全一致的空间增强。

### 5.2 每步 forward

伪代码：

```python
# 1. Student forward
pred_s, feats_s = student.forward_with_features(raw_dark, return_layers=[5,8,11])

# 2. Teacher forward, no grad
with torch.no_grad():
    pred_t, feats_t = teacher.forward_with_features(raw_normal, return_layers=[5,8,11])

# 3. Original depth loss
loss_depth = depth_loss(pred_s, pseudo_depth, valid_mask)

# 4. Mid/deep feature loss
loss_feat = 0
for l in [5, 8, 11]:
    fs = normalize(feats_s[l], dim=-1)
    ft = normalize(feats_t[l], dim=-1)
    loss_feat += mean(1 - cosine(fs, ft))
loss_feat /= 3

# 5. Total
loss = loss_depth + lambda_feat * loss_feat
```

### 5.3 Teacher 模式

必须确保：

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
```

Teacher forward 不更新 BN running stats，也不参与 AMP scaler backward。

Teacher（ViT-S）与 student 同时常驻显存、每步多一次前向。ViT-S 体量小可接受，但需在显存预算里计入这份额外开销。

### 5.4 Student 可训练范围

Canonical LoRA 组：

```text
trainable = RAM + LoRA + decoder
```

No-LoRA 组：

```text
trainable = RAM + decoder
```

新增 LocalDenoiseBranch 属于 RAM/front_end 参数组，学习率跟 `raw_front_end_lr`。

---

## 6. 日志与诊断指标

每个实验至少记录：

### 6.1 原有指标

```text
D1
AbsRel
RMSE
loss_depth
```

### 6.2 新增训练日志

```text
loss_feat_total
loss_feat_l5
loss_feat_l8
loss_feat_l11
lambda_feat
```

如果启用 local residual branch，再记录：

```text
gate_value = tanh(g)
delta_x3_l1
delta_x3_l2
x3_ram_range / x3_out_range
```

### 6.3 eval-time feature probe

每个 checkpoint 至少在 val 上跑一次 feature probe：

```text
Student RAW_dark vs Teacher RAW_normal
layers: 5 / 8 / 11
metric: token cosine distance
```

用于判断：

- feature distance 是否真的下降；
- D1 提升是否和 layer5/8/11 接近一致；
- 是否出现“feature 近了但 depth 没涨”的失配。

---

## 7. 成功标准

Canonical LoRA 主线以 M_DD 与 M_NN 之间的 gap 评估：

```text
M_DD = 0.845279
M_NN = 0.898946
G    = 0.053667
```

定义：

```text
recover_ratio = (D1_new - 0.845279) / 0.053667
```

阶段性标准：

| 结果 | 判断 |
|---|---|
| D1_new <= 0.845 | 方案无效或训练配置有问题 |
| 0.855 左右 | 刚超过 RGB_dark 对照线，有继续价值 |
| 0.860–0.865 | 明确有效，恢复约 28%–37% gap |
| 0.870–0.875 | 强信号，恢复约 46%–55% gap |
| 接近 0.880 | 接近 `C_dark(RAW_normal)` input-swap 水平，说明 RAW_dark restoration 很有效 |
| >0.880 | 超出 same-parameter clean input teacher，说明 student 可能学到更好的 denoise/feature adaptation |

第一轮 short-train 成功标准：

```text
L3 > L0 + 0.010
且 L3 >= max(L1, L2) - 0.002      # 与 §10 Step5 统一；留 0.002 容差吸收短训波动
且 AbsRel/RMSE 相对 L0 恶化 < 2%（相对）
```

如果 L1 或 L2 单独明显优于 L3，不直接否定方向；先调 `λ_feat` 或 local branch gate/residual scale。

---

## 8. 结果解释规则

| 观察 | 解释 | 下一步 |
|---|---|---|
| L1 < L0（feature loss 单独反而掉点） | feature loss 把 LoRA/backbone 拉向 normal-expert 解，对 dark 输入不鲁棒（M_ND 式退化） | 降 λ_feat，或冻结 LoRA 只动 decoder；或改用同参数 teacher `C_dark(RAW_normal)`（见 §1.1） |
| L1 有提升，L2 无提升 | 主要是 feature domain mismatch | 继续 feature loss，考虑层权重/λ sweep |
| L2 有提升，L1 无提升 | 主要是可学习 local restoration | 保留 local branch，feature loss 降权或暂缓 |
| L3 > L1 且 L3 > L2 | denoise 与 feature matching 互补 | 进入 full training |
| L3 < L1 | local branch 干扰 feature alignment | 降低 residual scale 或 gate 学习率 |
| L3 < L2 | feature loss 过强或 teacher mismatch | 降低 λ_feat，先试 0.02 |
| feature distance 降但 D1 不升 | 对齐目标不够 depth-aware | 后续考虑 DPT feature 或 teacher-depth consistency，但不在 v1 加 |
| D1 升但 feature distance 不降 | local branch/LoRA 改善了任务但未对齐 teacher | feature loss 可能不是必要主因，保留 denoise-only 方向 |
| 高噪声样本涨、低噪声样本跌 | denoise 过强 | 后续加入 reliability gate 或 residual penalty |
| no-LoRA 无效、LoRA 有效 | backbone adaptation 必要 | 后续主线保留 LoRA |
| no-LoRA 有效、LoRA 也有效 | RAM-side correction 是主要因素 | 后续可简化 LoRA 或做参数效率研究 |

---

## 9. 暂不做的内容

第一版不做：

```text
1. noise augmentation
2. RGB_normal feature teacher
3. C_dark(RAW_normal) same-parameter teacher loss（默认不做；但已在 §1.1 列为「随时可触发的备选 teacher」：若 C_normal teacher 出现 §8 中 L1<L0 等退化即切换）
4. dual-teacher loss
5. RAM output L1 / pixel reconstruction loss
6. edge loss
7. reliability-weighted loss
8. external denoiser
9. full SID/LED-style restoration network
10. bridge / decoder feature adapter 同时开启
```

这些都可以作为 v2，但不应进入当前第一版，否则变量过多，无法解释收益来源。

---

## 10. 建议执行顺序

### Step 1：代码最小改动

0. 前提核验：先确认 LOD train loader 能按**同一样本对**返回 raw_dark / raw_normal，且二者空间 crop/pad 完全一致（teacher/student token 才能逐 patch 对齐）。这是整份计划的前提，不成立则后续全部无效。
1. 给 LOD RAW_dark training batch 增加 paired `raw_normal` 返回。
2. 增加 frozen teacher loading：`teacher_ckpt = C_normal`。
3. 增加 DAv2 layer5/8/11 feature hook / return。
4. 增加 `L_feat_middeep`。
5. 增加 `NoiseAwareRam3` 或 `RamCore3LocalResidual` 开关。

### Step 2：只跑 canonical LoRA 短训

```text
L0: baseline continue
L1: + feature loss only
L2: + local branch only
L3: + local branch + feature loss
```

短训规格（固定，确保 +0.010 阈值有意义）：

```text
short-train epochs = 10          # full = 40，与现有 e40 recipe 对齐
L0 额外跑 1 个不同 seed           # 估计 run 间 D1 噪声地板；若噪声 ≈ 0.010，需放宽阈值或加长短训
其余组单 seed
```

初始：

```text
λ_feat = 0.05
residual_scale (s) = 0.1
gate_init: tanh(g) ≈ 0.02~0.05   # 小非零，见 §2.2
```

### Step 3：根据短训结果调参

- 如果 L3 下降，先试 `λ_feat = 0.02`。
- 如果 L2 下降，先减 residual scale，或确认 gate 是否打开过快（gate 学习率过大）。
- 如果 L1/L3 feature distance 降但 D1 不涨，暂不加新 loss，先改 layer 权重：降低 layer11，提高 layer5/8。

### Step 4：补 no-LoRA 对照

至少补：

```text
W0, W3
```

如果 W3 有收益，再补 W1/W2。

### Step 5：full training

只有当 short-train 满足以下条件时进入 full training：

```text
主实验 L3 > L0 + 0.010
且 L3 >= max(L1, L2) - 0.002
```

允许 `L3` 与 `max(L1,L2)` 接近，因为短训波动可能较大。

---

## 11. 最终交付物

每轮实验结束后输出：

```text
1. matrix.csv：D1 / AbsRel / RMSE / best epoch / ckpt
2. recovery.csv：相对 M_DD/M_NN 的 gap recovery
3. feature_probe.csv：layer5/8/11 token distance
4. ablation_summary.md：L0-L3 / W0-W3 横向表格
5. failure_gallery：高噪声、高 edge loss、高 feature mismatch 各 8–12 例
```

建议 ablation_summary 第一张表固定为：

| group | local branch | feature loss | lora | D1 | ΔD1 vs baseline | recover ratio | L5 dist | L8 dist | L11 dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

---

## 12. 本计划的判断边界

如果第一轮结果不理想，不应立刻否定“denoise + feature matching”总方向。应先判断失败属于哪一类：

1. local branch 位置不对；
2. feature loss 权重过大；
3. teacher feature 层选择不对；
4. C_normal teacher 与 C_dark student 的 checkpoint gap 过强；
5. LoRA/decoder 已经吸收了部分 feature gap，导致新增约束边际小；
6. RAW_dark 中存在不可恢复区域，需要 reliability gate 或 synthetic noise augmentation。

但这些都属于 v2。v1 的目标不是一次达到 RAW_normal oracle，而是回答三个问题：

```text
Q1: 只加 mid/deep feature matching 是否有用？
Q2: 只加 learnable local RAM denoise branch 是否有用？
Q3: 二者结合是否优于任一单项？
```

---

## 13. 具体执行操作（落地版）

本节把上面的实验计划转成代码改动、smoke test、短训、full training 与结果汇总的实际操作。  
原则：科学假设保持在 §1-§12；工程操作集中放在本节，避免把实验逻辑和实现细节混在一起。

### 13.1 代码改动顺序

按下面顺序实现，不跳步：

```text
1. paired LOD RAW dataset
2. student init / teacher init 加载路径
3. model forward feature API
4. LocalDenoiseBranch / NoiseAwareRam3 开关
5. train loop feature distillation loss
6. resolved config / CLI semantic 参数校验
7. feature probe / summary 脚本
8. formal launch scripts
```

#### 13.1.1 Paired LOD RAW dataset

修改文件：

```text
finetune_stf/dataset/lod_true.py
finetune_stf/config/resolved.py
finetune_stf/models/raw_ram.py
finetune_stf/train.py
finetune_stf/util/model_input.py（如需要）
```

新增 dataset 行为：

```text
dataset_family = lod_true_raw_dark_normal_pair_rgb16
dataset_input_mode = raw_rgb16_dark_normal_pair
model_input_tensor = raw
```

必须同步注册的枚举 / 映射点：

```text
finetune_stf/config/resolved.py
  DATASET_FAMILY_CHOICES += lod_true_raw_dark_normal_pair_rgb16
  DATASET_INPUT_MODE_CHOICES += raw_rgb16_dark_normal_pair
  LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY[lod_true_raw_dark_normal_pair_rgb16] = raw_rgb16_dark_normal_pair
  LOD_TRUE_RAW_RGB16_DATASET_FAMILIES 自动包含新 family
  INPUT_TYPE_ALIASES / _legacy_alias_from_config 增加 paired family 对应 alias（如继续使用 legacy alias 机制）

finetune_stf/models/raw_ram.py
  RAW_RGB16_RAM3_INPUT_TYPES += lod_true_raw_dark_normal_pair_rgb16

finetune_stf/train.py
  uses_lod_dataset / uses_lod_raw_rgb16_dataset / build_datasets / log_setup 等按 LOD_TRUE_RAW_RGB16_DATASET_FAMILIES 触发的路径必须覆盖新 family
  LOD_TRUE_RAW_RGB16_INPUT_MODE_BY_FAMILY 新增映射后，build_datasets 应能选择 paired dataset class，而不是仍走单路 LODTrueRawDarkRGB16
```

如果这些注册点缺任意一个，`argparse choices`、resolved config 校验、`build_model` 或 LOD dataloader 分支会在 `--audit` 阶段直接失败；因此它们属于 dataset 改动的一部分，不是后续清理项。

每个 sample 必须返回：

```text
raw              = raw_dark tensor，保持现有 student 默认输入语义
raw_dark         = raw_dark tensor，显式 student input
raw_normal       = raw_normal tensor，teacher input
image            = raw_dark tensor，用于兼容现有日志/可视化路径
depth            = pseudo_depth
valid_mask       = valid mask
sample_id        = pair_id
geometry_params  = crop / hflip 信息（开启 include_geometry 时）
raw_dark_path
raw_normal_path
```

实现要求：

- `raw_dark` 与 `raw_normal` 必须先同时读入，再应用同一个 crop box。
- 如果启用几何增强，必须对 `raw_dark`、`raw_normal`、`depth`、`valid_mask` 使用同一组 geometric 参数。
- 第一版不对 teacher input 做 photometric/raw noise augmentation；如果 student input 启用 photometric raw augmentation，则 paired teacher feature alignment 不再纯净，v1 默认应禁用或显式报错。
- dataloader smoke 必须检查：

```text
raw_dark.shape == raw_normal.shape == raw.shape
sample_id 一致
crop_box 一致
raw_dark_path 与 raw_normal_path 来自同一 manifest row
```

#### 13.1.2 Student init 与 teacher init

不要用当前 `--resume-from` 作为 short/full 训练的 student 初始化入口。原因：

```text
resume-from 语义 = 继续旧 run
- strict load
- start_epoch = checkpoint_epoch + 1
- 可能加载 optimizer
- 新增 local branch 会 missing keys
```

新增参数：

```text
--student-init-from <path>
--student-init-strict compatible
--teacher-ckpt <path>
--teacher-recipe lora_tap_r8a16 | decoder_w0
--teacher-input-mode raw_rgb16_normal
```

与现有 `--pretrained-from` 的关系必须先定清楚。当前 `--pretrained-from` 是必填参数；formal scripts 仍要显式传它，且推荐保留为 base DAv2 权重入口：

```text
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--student-init-from /mnt/drive/.../C_dark/best_model.pth
```

加载顺序：

```text
1. build student model（含 LoRA / local branch 开关）
2. load_initial_weights(model, --pretrained-from) 灌入 base DAv2 权重，保持现有必填语义
3. load_student_init(model, --student-init-from, strict=compatible) 覆盖 RAM / LoRA / decoder / DAv2 已训练权重
4. start_epoch = 0，不加载 optimizer / scheduler / best_metrics
```

如果未来决定复用 `--pretrained-from` 承担 student checkpoint 初始化，也必须先把它的语义改名或增加 explicit mode；不要让同一个参数有时表示 base DAv2、有时表示 full student checkpoint。v1 推荐保留两个入口，避免破坏现有脚本。

推荐语义：

```text
--student-init-from:
  只加载模型权重，重置 epoch=0，不加载 optimizer/scheduler。
  允许新增 local branch / distill-only buffer 的 missing keys。
  对已有 RAM / DAv2 / LoRA / decoder 权重必须严格匹配。

--teacher-ckpt:
  单独 build teacher model，teacher recipe 必须和 teacher checkpoint 匹配。
  teacher 也复用 build_model + load_initial_weights(--pretrained-from) + compatible checkpoint 覆盖路径；
  对 LoRA teacher，必须先构建 LoRA 包裹模型再加载 teacher checkpoint。
  加载后 teacher.eval()。
  所有 teacher 参数 requires_grad_(False)。
```

兼容加载必须打印并保存：

```text
student_init_missing_keys
student_init_unexpected_keys
teacher_missing_keys
teacher_unexpected_keys
loaded_lora_keys_count
loaded_ram_keys_count
loaded_decoder_keys_count
```

若出现非预期 missing/unexpected，smoke 直接失败。

#### 13.1.3 Model forward feature API

修改文件：

```text
finetune_stf/models/raw_ram.py
finetune_stf/train.py
```

给 `RawRgb16Ram3DepthModel` 增加一个内部 helper，而不是把训练逻辑散落在 train loop：

```python
def forward_with_dav2_features(self, x_raw, *, return_layers):
    # returns:
    #   pred_depth
    #   layer_features: dict[int, patch_tokens]
    #   ram_debug: dict[str, tensor or scalar]
```

实现要求：

- `return_layers=[5,8,11]` 是 feature loss 层；depth prediction 仍需要 DAv2 原生 depth head 的 4 个 intermediate layers（ViT-S/B 为 `[2,5,8,11]`）。
- helper 内部应请求 `depth_layers = self.dav2.intermediate_layer_idx[self.dav2.encoder]`，并验证 feature loss layers 是 depth_layers 的子集；v1 不支持任意非 depth-head 层。
- 使用 `self.dav2.pretrained.get_intermediate_layers(..., norm=True, return_class_token=True)`，保持与 `DepthAnythingV2.forward()` 喂给 `DPTHead` 的格式一致。
- feature loss 只使用 patch tokens，不使用 CLS。
- depth prediction 必须继续走同一批 intermediate features 或等价路径，避免 student 前向为了 feature loss 重复跑 backbone 两次。
- 推荐实现路径：在 helper 内手动完成 `ram_core -> optional local residual -> optional tail -> pad -> dav2.pretrained.get_intermediate_layers(depth_layers, return_class_token=True)`，拿到 DAv2 intermediate features 后，直接调用 `self.dav2.depth_head(features, patch_h, patch_w)` 生成 depth，再 crop。不要改 `depth_anything_v2/dpt.py`，也不要先跑 `self.dav2(x_norm)` 再额外跑一次 `get_intermediate_layers`。
- feature loss 从同一份 `features` 中按 layer id 取 `patch_tokens`，例如 `{5: features[1][0], 8: features[2][0], 11: features[3][0]}`。
- teacher forward 使用相同 helper，但 `torch.no_grad()` 包裹。

#### 13.1.4 LocalDenoiseBranch / NoiseAwareRam3

修改文件：

```text
finetune_stf/models/raw_ram.py
finetune_stf/train.py
finetune_stf/config/resolved.py
```

新增参数：

```text
--raw-ram-local-residual none | noiseaware_v1
--raw-ram-local-hidden-ch 16
--raw-ram-local-residual-scale 0.1
--raw-ram-local-gate-init 0.03
--raw-ram-local-gate-mode channel
```

解析原则：

- categorical 参数必须把禁用 sentinel 纳入合法 choices：`none` / `n_a` 不能在 argparse 阶段失败。
- numeric 参数可以用 `0` 表示禁用值，但 resolver 必须在 disabled branch 里把它记录成 `not_applicable` 或明确的 disabled value。
- list 参数不要直接用 `nargs="+", type=int` 解析禁用态；应先按字符串解析，再把 `none` / `n_a` 转成空 tuple，并在 resolved config 中写成 `not_applicable`。

实现约束：

- local branch 必须挂在 `ram_core.` 下，或同步更新 optimizer param-group 前缀，确保它进入 `raw_front_end` lr。
- `--raw-ram-local-residual none` 时，不允许传 active 的 hidden/scale/gate 参数；或这些参数必须解析成 `not_applicable`。
- `gate_init` 表示 `tanh(g)` 的目标初值，不是 g 本身。初始化时使用 `atanh(gate_init)`。
- log 中同时记录：

```text
raw_ram_local_residual
raw_ram_local_hidden_ch
raw_ram_local_residual_scale
raw_ram_local_gate_init
raw_ram_local_gate_value/tanh
delta_x3_l1
delta_x3_l2
x3_ram_p1/p50/p99
x3_out_p1/p50/p99
```

#### 13.1.5 Feature distillation loss

修改文件：

```text
finetune_stf/train.py
finetune_stf/util/loss.py（如需抽 helper）
```

新增参数：

```text
--feat-distill none | dav2_middeep_cosine
--feat-distill-layers 5 8 11
--feat-distill-lambda 0.05
--feat-distill-teacher normal_expert | same_param_clean
```

解析原则同 §13.1.4：

- `--feat-distill none` 必须是合法 choice。
- `--feat-distill-teacher n_a` 必须是合法 disabled sentinel，并解析成 `not_applicable`。
- `--teacher-recipe n_a`、`--teacher-input-mode n_a` 也必须是合法 disabled sentinel。
- `--feat-distill-layers none` 必须能解析成空 tuple / `not_applicable`；不要实现成会尝试 `int("none")` 的 argparse 类型。

参数校验：

```text
feat-distill=none:
  teacher-ckpt 必须是 n_a/none
  feat-distill-lambda 必须是 0 或 not_applicable
  feat-distill-layers 必须是 none/not_applicable

feat-distill=dav2_middeep_cosine:
  teacher-ckpt 必须存在
  teacher-input-mode 必须是 raw_rgb16_normal
  paired dataset 必须启用 raw_normal
  feat-distill-lambda > 0
  feat-distill-layers 非空
```

训练 loop 中 loss 组成：

```text
loss_depth = existing depth criterion
loss_feat  = mean_l mean(1 - cosine(normalize(Fs_l), normalize(Ft_l)))
loss_total = loss_depth + lambda_feat * loss_feat
```

日志必须区分：

```text
train/loss_depth
train/loss_feat_total
train/loss_feat_l5
train/loss_feat_l8
train/loss_feat_l11
train/lambda_feat
train/loss_total
```

`loss_info["used_samples"] == 0` 时，不做 feature backward，并保持现有 skip 逻辑。

#### 13.1.6 Resolved config 与 config.json

修改文件：

```text
finetune_stf/config/resolved.py
finetune_stf/train.py
```

以下都属于 experiment-semantic 参数，必须进入 `resolved_config.json`：

```text
student_init_from
student_init_strict
teacher_ckpt
teacher_recipe
teacher_input_mode
feat_distill
feat_distill_layers
feat_distill_lambda
feat_distill_teacher
raw_ram_local_residual
raw_ram_local_hidden_ch
raw_ram_local_residual_scale
raw_ram_local_gate_init
raw_ram_local_gate_mode
```

formal launch scripts 必须显式传这些参数；即使禁用也要传 `none` / `n_a` / `0`，不要依赖 hidden defaults。

### 13.2 Smoke tests

所有 smoke 输出路径必须包含 `codex_smoke`，成功后删除；失败则保留并报告路径。

#### 13.2.1 Dataset paired smoke

命令目标：

```text
只构建 paired dataset，取 4 个 sample，不训练。
```

检查项：

```text
raw_dark/raw_normal/raw/image/depth/valid_mask shape
same crop_box
same sample_id
raw_dark_path/raw_normal_path 均存在
teacher input 没有 photometric augmentation
```

输出：

```text
/tmp/codex_smoke_lod_pair_dataset_<timestamp>/
```

成功后删除该目录。

#### 13.2.2 Model / init smoke

命令目标：

```text
build student L0/L3 model
load student-init-from C_dark
build teacher
load teacher-ckpt C_normal
forward one batch
```

检查项：

```text
student pred shape == depth shape
teacher feature layers 5/8/11 shape 与 student 一致
loss_depth finite
loss_feat finite
loss_total finite
teacher has no grad
local branch 参数在 raw_front_end optimizer group
LoRA 参数在 lora optimizer group（LoRA 组）
```

#### 13.2.3 Backward smoke

命令目标：

```text
debug-max-train-steps=2
epochs=1
bs=1 或 2
save_path=/tmp/codex_smoke_lod_featdistill_train_<timestamp>
```

必须通过：

```text
optimizer step
TensorBoard scalar write
checkpoint save 如开启
eval 1-2 samples 如开启
```

成功后删除 clearly temporary smoke artifacts。

### 13.3 Short-train launch scripts

新增 formal script：

```text
finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_short_e10_queue.sh
```

脚本要求：

- 默认 conda env：`dav3`。
- 必须用 tmux 启动。
- session name 和 run name 以 launch time 的 `MMDD_HHMM` 开头。
- 每个 run 写独立 `.tmux.log`。
- 不复用已有 tmux session。
- 不覆盖已有 `EXP_ROOT` / `HEAVY_ROOT` artifacts。
- 支持 `--audit`，只解析参数并打印 semantic payload，不训练。

短训 run matrix：

```text
L0: local none, feat none
L1: local none, feat dav2_middeep_cosine, lambda=0.05
L2: local noiseaware_v1, feat none
L3: local noiseaware_v1, feat dav2_middeep_cosine, lambda=0.05
L0_seed123: local none, feat none, seed=123
```

所有 run 必须显式传：

```text
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--dataset-family lod_true_raw_dark_normal_pair_rgb16
--dataset-input-mode raw_rgb16_dark_normal_pair
--input-domain raw3
--front-end raw_rgb16_ram3
--model-input-tensor raw
--raw-storage-format raw_rgb16_png_3ch
--lod-raw-norm-mode uint16_div_65535
--raw-ram-rgb-tail identity
--student-init-from /mnt/drive/.../0608_2026.../best_model.pth
--student-init-strict compatible
--teacher-ckpt /mnt/drive/.../0609_0044.../best_model.pth 或 n_a
--teacher-recipe lora_tap_r8a16 或 n_a
--teacher-input-mode raw_rgb16_normal 或 n_a
--feat-distill ...
--raw-ram-local-residual ...
--lora dav2_lora
--lora-block-mode tap
--lora-tap-layers 2 5 8 11
--lora-rank 8
--lora-alpha 16
--lora-lr 5e-5
--dav2-train-mode decoder
--raw-front-end-lr 5e-5
--lr 1e-5
--epochs 10
```

L0/L2 禁用 teacher 时也要显式传：

```text
--teacher-ckpt n_a
--teacher-recipe n_a
--teacher-input-mode n_a
--feat-distill none
--feat-distill-lambda 0
--feat-distill-layers none
--feat-distill-teacher n_a
```

L1/L3 启用 feature distill 时显式传：

```text
--feat-distill dav2_middeep_cosine
--feat-distill-layers 5 8 11
--feat-distill-lambda 0.05
--feat-distill-teacher normal_expert
```

L0/L1 禁用 local branch 时显式传：

```text
--raw-ram-local-residual none
--raw-ram-local-hidden-ch 0
--raw-ram-local-residual-scale 0
--raw-ram-local-gate-init 0
--raw-ram-local-gate-mode n_a
```

L2/L3 启用 local branch 时显式传：

```text
--raw-ram-local-residual noiseaware_v1
--raw-ram-local-hidden-ch 16
--raw-ram-local-residual-scale 0.1
--raw-ram-local-gate-init 0.03
--raw-ram-local-gate-mode channel
```

### 13.4 Short-train 后处理

新增或扩展脚本：

```text
tools/lod_raw_featdistill_matrix_summary.py
tools/lod_raw_ram_feature_probe.py
```

每个 short run 完成后输出：

```text
matrix.csv
recovery.csv
feature_probe.csv
ablation_summary.md
```

feature probe 必须支持双模型：

```text
student_checkpoint = short run best_model.pth
student_input      = RAW_dark
teacher_checkpoint = C_normal 或 same-param teacher
teacher_input      = RAW_normal
layers             = 5,8,11
metric             = token cosine distance
```

如果工具暂时只支持单模型 input-swap probe，则不能声称完成 §6.3；必须在 summary 中标记为 `single_model_probe_only`。

### 13.5 决策门槛与下一步

短训完成后先看：

```text
L0_seed42 vs L0_seed123 的 D1 差
L3 - L0
L3 - max(L1,L2)
AbsRel/RMSE 是否恶化
feature distance 是否下降
gate 是否异常打开
delta_x3 是否过大
```

进入 full training 的条件仍用 §7 / §10：

```text
L3 > L0 + 0.010
且 L3 >= max(L1, L2) - 0.002
且 AbsRel/RMSE 相对 L0 恶化 < 2%
```

若失败，按以下顺序处理：

```text
1. L1/L3 掉点且 feature distance 降：lambda 0.05 -> 0.02
2. L2/L3 掉点且 gate/delta 过大：scale 0.1 -> 0.05 或 gate_init 0.03 -> 0.02
3. feature 近但 D1 不涨：降低 layer11 权重，提高 layer5/8
4. 仍失败：切换 teacher 到 same_param_clean = C_dark(RAW_normal)
```

### 13.5.1 L3 去噪效果复盘（2026-06-09 21:50）

本节记录用户要求“看 L3 去噪到底效果如何”后的追加诊断。结论：**L3 最终 D1 略高，但 noise-aware local residual 本身不是强显式去噪器；本轮收益主要不能归因于 local branch 的像素级清理。**

短训 best checkpoint 精确值：

| group | 配置摘要 | best epoch | best_lod_d1 |
|---|---|---:|---:|
| L0 | no feature distill, no local residual | 9 | 0.845572 |
| L1 | feature distill, no local residual | 9 | 0.848410 |
| L2 | no feature distill, `noiseaware_v1` | 7 | 0.845970 |
| L3 | feature distill + `noiseaware_v1` | 9 | 0.848454 |

关键比较：

```text
L3 - L1 = +0.000045
L2 - L0 = +0.000398
L3 - L0 = +0.002882
L0_seed123 best_lod_d1 = 0.8502
```

因此：

1. `noiseaware_v1` 单独带来的 L2-L0 增益很小。
2. L3 只比 L1 高 `+0.000045`，远小于 seed 波动；不能证明 feature distill + local residual 有稳定叠加收益。
3. L0_seed123 高于 L3，进一步说明 e10 单 seed ablation 不能作为 full e40 的充分依据。

L3 vs L1 RAM 前端诊断输出：

```text
summary:
  finetune_stf/analysis/lod_denoise_effect/0609_2045_L3_vs_L1_ram_frontend_val112/summary.json
per-sample stats:
  finetune_stf/analysis/lod_denoise_effect/0609_2045_L3_vs_L1_ram_frontend_val112/per_sample_ram_stats.csv
montage:
  finetune_stf/analysis/lod_denoise_effect/0609_2045_L3_vs_L1_ram_frontend_val112/selected_ram_panels_montage.jpg
L3 train-viz panels:
  finetune_stf/analysis/lod_denoise_effect/0609_2045_L3_train_viz_n4_nobaseline/train_viz/epoch_09/lod
```

RAM local residual 的实测幅度：

```text
gate_tanh = [0.031340, 0.032899, 0.030897]
0.1 * gate_tanh = [0.003134, 0.003290, 0.003090]

mean |RAW_dark - RAW_normal|      = 0.077674
mean |L3 local contribution|      = 0.000842
p99  |L3 local contribution|      = 0.002483
mean local / mean |x3_ram|        = 0.001244  # 约 0.124%
mean |L3 RAM out - L1 RAM out|    = 0.097128
mean |L3 x3_ram|                  = 0.796950
mean |L3 x3_out|                  = 0.796843
```

图像解读：

```text
selected_ram_panels_montage.jpg 每行一个 val 样本，列含义：
1. RAW dark
2. RAW normal
3. |dark-normal|
4. L1 RAM out
5. L3 RAM pre-local
6. L3 RAM out
7. |L3 local contrib|
8. |L3-L1 RAM|
```

注意：所有 RAW/RAM 图都做了 per-image robust stretch，`|L3 local contrib|` 热图也做了幅度归一化；因此亮色表示结构位置，不表示绝对改变量很大。应结合上面的 `mean/p99/local_ratio` 数值判断。

实际观察：

1. `L3 RAM pre-local` 与 `L3 RAM out` 肉眼几乎一致。
2. `|L3 local contrib|` 有边缘/纹理结构，但绝对幅度仅约 `0.0008`，约为 `x3_ram` 的 `0.124%`。
3. `|L3-L1 RAM|` 明显大得多，说明 L3 与 L1 的 RAM 输出差异主要来自整体训练后的权重差异，而不是 L3 local residual 这一项。

复盘判断：

```text
L3 可以记录为本组 seed42 best D1 最高；
但不能表述为 noiseaware_v1 明显完成了 RAW 去噪。
更准确的表述是：
  noiseaware_v1 在当前 gate/scale 下只产生很弱的 task-aware local correction；
  e10 结果未证明它带来稳定、可复现、可肉眼确认的去噪收益；
  不应据此启动 full e40。
```

后续若继续研究 local branch，应优先考虑：

```text
1. 先复现实验到多 seed，确认 L2/L3 是否真的超过无 local 对照。
2. 若仍希望放大 local branch，可单独调 `local_residual_scale` 或 `gate_init`，但必须保持 L1/L2/L3 对照和 seed 对照。
3. 若 feature distance 降而 D1 不稳定，优先尝试 same-param teacher：C_dark(RAW_normal)，而不是继续加强 C_normal feature target。
```

### 13.6 Full training

新增 formal script：

```text
finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_full_e40_queue.sh
```

full training 只启动通过短训门槛的配置。  
run name 必须以 full launch time 的 `MMDD_HHMM` 开头，且 suffix 写清：

```text
lod_true_raw_dark_pair_noiseawarev1_featmiddeep_lora_tap_r8a16_decoder_e40_lam005_gate003_s01
```

如果 full 采用调参后的 `lambda=0.02` 或 `scale=0.05`，run name 与 resolved config 都必须反映该值。

### 13.7 no-LoRA 对照

no-LoRA 不与 LoRA teacher/student 混用。  
W0/W3 的执行脚本可以复用 short/full 框架，但必须替换：

```text
student-init-from = 0608_1739...W0.../best_model.pth
teacher-ckpt      = 0609_0010...W0.../best_model.pth
--lora none
--teacher-recipe decoder_w0
--dav2-train-mode decoder
```

并显式禁用 LoRA：

```text
--lora none
```

resolved config 中必须记录：

```text
lora = none
lora_rank = not_applicable
lora_alpha = not_applicable
lora_lr = not_applicable
lora_tap_layers = not_applicable
```

no-LoRA run 不要传 active 的 `--lora-rank/--lora-alpha/--lora-lr/--lora-tap-layers`。如果后续统一 parser 支持 `n_a` 字面值，也必须先在 `--audit` 中确认这些字段被解析为 `not_applicable`，不能被当作有效 LoRA 配置。

### 13.8 最小代码完成定义

代码完成不以“脚本写好”为准，必须同时满足：

```text
1. paired dataset smoke 通过
2. student/teacher init smoke 通过
3. L0/L1/L2/L3 backward smoke 通过
4. formal short script --audit 通过
5. 至少一个 L3 codex_smoke 训练 2 step 通过
6. config.json / resolved_config.json 含全部新增语义参数
7. optimizer_param_groups 显示 local branch 属于 raw_front_end
8. teacher 参数 requires_grad=false，且 backward 后 teacher grad 全为 None
```

---

## 14. 第二轮诊断：放大 local 分支 + same-param teacher（round-2）

日期：2026-06-09 22:55  
触发：用户确认第一轮（§13.5.1）“去噪失败”，要求换方向重跑。本节自包含，记录第二轮的失败复盘、假设、矩阵、工程落地、判据与执行状态。

### 14.0 第一轮失败结论（量化复盘）

第一轮三机制（`noiseaware_v1` local residual + mid/deep feature distill + `C_normal` teacher）的增益全部落在 seed 噪声地板以下：

| 组 | 配置 | best D1 | AbsRel↓ |
|---|---|---:|---:|
| L0 (seed42) | baseline | 0.845572 | 3.5956 |
| L0_seed123 | 只换 seed | 0.8502 | 3.6286 |
| L1 | +feat distill (C_normal) | 0.848410 | 3.6448 |
| L2 | +local (s0.1) | 0.845970 | 3.6692 |
| L3 | feat+local | 0.848454 | 3.7276 |

- **致命点**：`L0_seed123=0.8502 > L3=0.8485`。seed-only 抖动（+0.0046）比 L3 相对 L0 的“增益”（+0.0029）还大 → 增益不可信。
- full gate `L3 > L0+0.010 = 0.8556` 未达（只 recover 约 6% gap）。
- **所有干预都让 AbsRel 变差**（L0 最低 3.5956，L3 最差 3.7276，+3.7%，超过 <2% 闸）。
- **local 分支空转**：`x3_out = x3_ram + 0.1·tanh(g)·tanh(delta)`，gate 训练后≈[0.0313,0.0329,0.0309]（几乎卡在 init 0.03）；`mean|local|=0.00084`、占 `x3_ram` 幅度 0.124%；而 input 级 `mean|dark−normal|=0.0777` 是它的 ~92×。`|L3−L1 RAM|=0.097` ≫ `|local|=0.0008`，说明 L3/L1 差异来自整体训练后的权重差，不是 local 残差这一项。
- no-LoRA W0/W3 同样：W3=0.8310 < W0=0.8312，排除“是 LoRA 挡路”。

机制根因：
1. `scale=0.1` + 小 gate 把 local 预算锁死在比目标 gap 小约两个数量级，优化器宁可走 LoRA/decoder 这条更省的下降方向；
2. `C_normal` teacher 把 student 往 dark-fragile 的 normal-expert 解拉（M_ND=0.817），中后层 token 表面对齐、metric depth（AbsRel）反而退；
3. 真正合理的 oracle 是 `M_DN = C_dark(RAW_normal) = 0.8805`（同参换 clean input 即 +0.035），gap 大头在输入域；x3 后 0.1-scale 残差位置太靠后、幅度太小，物理上够不到。

### 14.1 第二轮两条独立轴（保持单变量）

不把两个改动塞进一个 run；A/B 各自有干净对照：

- **A 轴**（放大 local 预算）：local-only、teacher none、对照原 L2。scale 0.1→{0.3, 0.5}，gate_init 0.03→0.1（预算上限从 0.003 提到约 0.05，约 16×）。验证“是不是预算被锁死”——若放大后 gate 仍不长大、L2 仍不动，则可干净否决 internal residual 路线。
- **B 轴**（换 teacher）：feat-only、local none、对照原 L1。teacher 从 `C_normal` 换成 **`C_dark(RAW_normal)`** same-param（`feat_distill_teacher=same_param_clean`）。其 token 可由 student 自身参数靠清理输入到达，干净分离“输入清理 vs 参数迁移”，并止住 AbsRel 退化。对应 §1.1 备选与 §13.5 fallback 第4步。
- **组合 AB**：放大 local + same-param teacher，作“上限探针”。
- **baseline 第3 seed**（seed7）：把 run 间 D1 噪声地板 σ 钉死（现仅 2 seed 就差 0.0046，吃掉了上一轮全部“增益”）。

矩阵（e10，复用已有 L0/L0_seed123/L1/L2 作对照，不重跑）：

| run | 轴 | local scale/gate | teacher | seed | 对照 |
|---|---|---|---|---|---|
| C0_seed7 | baseline | none | none | 7 | 量 σ |
| A_s03 | A | 0.3 / 0.1 | none | 42 | vs L2(0.1/0.03) |
| A_s05 | A | 0.5 / 0.1 | none | 42 | vs L2 |
| B_sp | B | none | C_dark(RAW_normal) | 42 | vs L1(C_normal) |
| B_sp_s123 | B | none | C_dark(RAW_normal) | 123 | vs L0_seed123 |
| AB_s05_sp | 组合 | 0.5 / 0.1 | C_dark(RAW_normal) | 42 | vs A_s05, B_sp |

### 14.2 工程落地（无需改模型/配置代码）

- 新脚本：`finetune_stf/scripts/formal/0609_run_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue.sh`（照搬现有 short 队列框架：tmux 启动、`--audit`、run_train、refuse-overwrite）。
- `same_param_clean` 已是 `FEAT_DISTILL_TEACHER_CHOICES` 合法值（`finetune_stf/config/resolved.py:52`）；且 `feat_distill_teacher` 只是写进 `resolved_config.json` 的语义标签，**真正决定 teacher 的是 `teacher_ckpt`**（指向 C_dark）+ `teacher_recipe=lora_tap_r8a16` + `teacher_input_mode=raw_rgb16_normal`。scale/gate_init 都是现成 CLI 参数，放大纯改脚本值。
- 新增 `RUN_GROUPS` 组过滤（`RUN_GROUPS=AB_s05_sp bash …` 只跑指定组）。**踩坑记录**：变量不能命名为 `GROUPS`——它是 bash 内置只读数组（当前用户组 id），`${GROUPS}` 返回 GID `1000`，会导致全部 SKIP。已改名 `RUN_GROUPS`。
- 6 组 `--audit` 全过；B/AB 组确认 `teacher_ckpt` 指向 C_dark、`feat_distill_teacher=same_param_clean`、`teacher_input_mode=raw_rgb16_normal`；A/AB 组 `scale=0.3/0.5`、`gate_init=0.1`。

### 14.3 判据（AB 作上限探针）

| 看什么 | 失败阈值 | 含义 |
|---|---|---|
| AB best D1 vs `L0_seed123=0.8502` | ≤0.8502 | 最强组合都没超 seed 上沿 → 去噪无效 |
| gate 训练后是否从 0.1 长大 | 卡在≈0.1 | 放大预算仍不用 → 结构性问题 |
| local contrib ratio | 仍≈0.1%级 | 没起到去噪作用（上轮 0.124%） |
| AbsRel vs `L0=3.5956` | 明显恶化 | same-param teacher 本应比 C_normal 更不伤 AbsRel |

决策：只要 AB 这个最强组合不达标 → 直接停，不跑 A/B 单轴与其余 seed；若 AB 明显起效，再回补单轴归因（A_s03/A_s05/B_sp/B_sp_s123/C0_seed7）。

### 14.4 执行状态（实时回写）

- 误启动的 6-run 全队列已 `tmux kill-session`；留下废弃半成品目录 `finetune_stf/exp/0609_2251_C0seed7_lod_raw_pair_lora_decoder_e10_featnone_localnone`（仅 epoch0、无 best checkpoint）+ heavy 下一个 tfevents，待清理（删除不影响后续）。
- 按用户决定改为先单跑 `AB_s05_sp`：
  - tmux session=`0609_2255_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue`
  - queue log=`finetune_stf/logs/0609_2255_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue.queue.log`
  - attach=`tmux attach -t 0609_2255_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue`
- 启动正常：teacher missing=0（C_dark same-param）、student missing=9（新增 local branch）、pretrain `lod_val d1=0.8447`。
- 进展（22:59）：epoch0 best=0.8423 → epoch1 best=0.8467 → epoch2/10 进行中。
- 结束（23:07）：`AB_s05_sp` status=0，最终 best 在 epoch9：`lod_val d1=0.8469`、`AbsRel=3.5605`、`rmse=24.3927`、train_proxy `d1=0.8721`、gap=0.0253。
- 判据：`0.8469 < L0_seed123=0.8502`（差 -0.0033），最强组合 AB 仍未超过 seed 上沿；按 §14.3，第二轮去噪上限探针失败，停止回补 A/B 单轴与额外 seed，除非后续显式改判据。
- best checkpoint gate/local 诊断（`01Valid` 112 样本，center crop，只跑 RAM 前端）：
  - `tanh(local_residual_gate)=[0.0983, 0.1004, 0.1000]`，基本贴着 init=0.1，没有继续长大。
  - `mean|local|=0.01283`、`mean|x3_ram|=0.79344`，`local/x3_ram=1.617%`；比第一轮 0.124% 放大约 13×，但仍未转化为 D1 增益。
  - 量级参照：`mean|RAW_dark-RAW_normal|=0.07767`，`local/RAW_gap≈16.5%`。
  - AbsRel 未恶化（best epoch `3.5605`，略优于 L0 `3.5956`），说明 same-param teacher 比 C_normal 更不伤 AbsRel；但主判据 D1 仍失败。

### 14.5 第二轮结论与后续方向（2026-06-09 23:08）

- **决策：停。** AB（放大 local + same-param teacher）作为最强组合，best D1=0.8469 未超 seed 上沿 L0_seed123=0.8502（且 < L1=0.8484、< L3=0.8485），按 §14.3 判据失败；不回补 A/B 单轴与额外 seed。
- **机制结论**：把 local 预算放大约 16×（scale 0.1→0.5、gate_init 0.03→0.1）后，gate 仍停在 init 0.1、local 实际贡献虽从 0.124% 升到 1.617% 却仍无 D1 增益——说明在 x3（RamCore3 末端、过 BN、近单位方差）之后加 gated local residual，对 depth loss 不存在有效下降方向；这是**结构性问题，不是调参或预算问题**。same-param teacher 唯一确认的收益是不伤 AbsRel。
- **若将来仍要推进去噪 / 恢复方向**：需要 v2 结构性重设计，而非继续 sweep λ/scale/gate。候选：把 restoration 放到更早、更大容量的位置（输入级或 RamCore3 之前），或改变监督信号（如直接的 RAW restoration target、teacher-depth consistency）。这些超出第一版“不外接 denoiser、只加 mid/deep feature loss”的克制边界，应另立 v2 计划。
