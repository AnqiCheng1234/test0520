# 0604 ROD Night Train+Eval 代码修改计划 (rev2)

> rev2 (2026-06-04)：根据代码/数据实测核对 + 当日决策重写。关键改动：
> - **第一版明确为方案 A 跨渲染蒸馏**：student 输入 = 在线 `student_dark_degreen_v1`，label = **新生成** 的 DAv2-L(`teacher_bright_degreen_v1`)，**不复用** 现有 sdr_rgb 版 label。
> - 新增 **Phase 0 离线 label 生成** 这一独立步骤；正式训练只读缓存 label。
> - raw→student/teacher RGB 管线已在 `make_0603_...py` 中**重建为可执行代码并通过 byte-exact parity**（20 样本 max_abs=0）；本计划把它收敛为仓库内单一真值模块。
> - 大量复用现有 `dataset/lod_raw.py` 的 `LODRGB/LODRaw`，而非从零新写。
> - DAv2-L label `input_size=924`（≈native，带 smoke 闸 / 回退 700）。

## 1. 目标和范围

只为 `/home/caq/6666_raw/dav2_raw_0603` 生成代码修改计划，不直接改训练代码。目标：接入 ROD-night 的 train+eval。

- 只做 ROD night，不接 day。
- **label = 离线新生成的 DAv2-L pseudo inverse-relative depth**，来源 RGB = `teacher_bright_degreen_v1`（见 §2.2），存到全新目录，不动现有 sdr_rgb 版 label。
- label 是 relative inverse depth，网络预测同类相对 inverse；loss/eval 都不按 metric depth 处理。
- **训练/验证 student 输入 = 训练时从 RAW 在线生成的 `student_dark_degreen_v1` RGB**，不读任何 `*-sdr_rgb`。
- raw→RGB 参数严格使用 `make_0603_...py` 中已 parity 验证的可执行管线（§2.2）。
- 第一版基于官方 DAv2-S checkpoint，只训练 decoder。
- 训练参数复刻 `dav2_raw_0520` 的 `0521_0112` 系列，正式 epoch 改为 10。
- 新接口/文件/日志/配置统一用 `rod`；历史 `lod` 只用于理解旧记录，不在新代码扩散。

### 1.1 实验语义声明（必须显式记录）

第一版是**跨渲染蒸馏 (方案 A)**：

```
teacher 域 (亮): teacher_bright_degreen_v1  --DAv2-L-->  pseudo inverse-rel label (离线缓存)
student 域 (暗): student_dark_degreen_v1    --DAv2-S(decoder finetune)-->  预测，拟合上面 label
```

- student 与 teacher 渲染**仅差 tone**：同一 `R,mean(G),B` 投影、同一 degreen gains `(1.08,0.95,1.10)`；只有 `white_percentile` 与 `gamma` 不同（student `wp=99.9/γ=0.9`，teacher `wp=99.5/γ=0.4545`）。结构/几何完全一致，蒸馏难度差异来自亮度/gamma。
- 该跨渲染事实必须写进 run 名/日志/分析，不能当成同域蒸馏解释。

## 2. 已核对事实（2026-06-04 实测）

### 2.1 ROD 数据

根目录 `/mnt/drive/3333_raw/ROD`（`ROD==LOD`，旧 `LOD` 目录已删，旧 manifest 全死链；见 `README_name_note.md`）。

| split | raw24 | rggb npy | 现有 pseudo(sdr_rgb 版) |
| --- | ---: | ---: | ---: |
| 00Train | 12036 | 12036 | 12036 |
| 01Valid | 2000 | 2000 | 2000 |

- raw24 `.raw`：单文件 `1856*2880*3` bytes；unpack 后 `/(2^24-1)` 归一化，shape (1856,2880)。
- Bayer = **真 RGGB**：`(0,0)=R,(0,1)=Gr,(1,0)=Gb,(1,1)=B`。packed RGGB npy = (928,1440,4)，通道 `0=R,1=Gr,2=Gb,3=B`（**勿套 STF 的 (3,1,2,0)**）。
- **夜间 raw 极暗**：linear `mean≈1e-4`，`p99≈3e-4`（全幅的 ~0.03%），少量亮点 `max≈0.9`。→ 直接影响 DAv2-L `input_size` 选择（§3.4）。

现有 `pseudo_depth_dav2_night_rel_1440x928/` 的 label = **DAv2-L 跑在 `sdr_rgb` jpg 上**生成（manifest `rgb_path` 列与实测确认，重建 nMAE→0.0000），与本计划要用的 student 渲染**不同域**——所以**不复用**，改新生成（§3.3 / §4.1）。

### 2.2 raw→student/teacher RGB 管线（已 byte-exact 验证）

真值脚本：`/home/caq/rod_tools/make_0603_6col_raw_rggb_student_teacher_rgb_channel_hists.py`（已含可执行管线 + `--verify-rendered-inputs`）。

2026-06-04 跑 `--verify-rendered-inputs` 对 20 样本：`raw_visualized / student / teacher` 三者 `max_abs=0, nonzero=0, mean_abs=0` → **在线渲染与原离线 PNG 逐像素完全一致**。

管线（`rawvis_degreen_rgb`）：

```
unpack_raw24 -> (1856,2880) float[0,1] (linear)
-> 投影 R, mean(Gr,Gb), B -> (928,1440,3)  # 2x2 binning, 非 demosaic 插值
-> * gains(1.08,0.95,1.10), clip[0,1]       # degreen, R↑ G↓ B↑
-> stretch_white_point(wp)                   # 全局(整图+三通道联合)百分位, 单标量, 非逐通道
-> gamma_adjust(gamma)                        # clip ** gamma
-> clip(*255).astype(uint8)                   # RGB 顺序 (非 BGR)
```

- `student_dark_degreen_v1`：`wp=99.9, gamma=0.9`
- `teacher_bright_degreen_v1`：`wp=99.5, gamma=0.454545 (1/2.2)`
- 输出 (928,1440,3) uint8，与 label 同尺寸。底层 `unpack_raw24/stretch_white_point/gamma_adjust` 来自 `raw24_to_sdr_rgb.py`。
- **白点全局且依赖整图内容** → 必须在 native 全图渲染、再 crop；不可先 crop 再渲染。

### 2.3 checkpoint

- DAv2-S（student 训练初始化）：`/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth`
- DAv2-L（teacher label 生成）：`/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth`

### 2.4 现有代码状态（0603/finetune_stf，实测）

- `dataset/lod_raw.py` 已有 **`LODRGB`** 与 **`LODRaw`**：已实现 manifest 解析、resize→native、train random / val center crop、ImageNet 归一化、`target_space="inverse_relative"`、`valid_mask=isfinite&>0`。→ **大量可复用**；ROD dataset 只需把"读 jpg"换成"raw→在线 student RGB"。
- 但 `LODRGB/LODRaw` **未接入 train.py 训练循环**：train.py 仅 `from ...lod_raw import DEFAULT_LOD_*`（路径常量），未 import/实例化这两个类。
- `--stage` 仅 `{stf_only, eval_only}`；`DATASET_FAMILY_CHOICES=("stf_rgb","stf_raw")`；`BEST_METRIC_CHOICES` 无 rod；dataloader train key 硬编码 `stf_train`；`build_checkpoint_payload` 里 `best_*` 是逐 metric 硬编码。
- `util/loss.py::build_training_target(target_space="inverse_relative")` **已支持**（不取倒数，直接用 inverse label）。
- `train.py::evaluate()` 现把 `sample["depth"]` 当 metric depth、用 `affine_align_disp` 对齐后算指标 → **对 inverse-rel label 是错的**，需新增按 `target_space` 分支（§4.6）。`resolve_batch_target_space()`(train.py:1498) 已存在，loss 已按它分支。

### 2.5 0521_0112 参考训练参数

参考实验 `0521_0112_stf_..._identity_decoder_e5`：

| 参数 | 0521_0112 | ROD-night v1 |
| --- | --- | --- |
| encoder / ckpt | vits / 官方 DAv2-S | 同 |
| train scope | decoder | decoder |
| epochs | 5 | **10** |
| bs / accum / eff | 8 / 1 / 8 | 8 / 1 / 8 |
| lr | 1e-5 | 1e-5 |
| loss | ssi + target-norm + min-scale 1e-3 | 同 |
| amp / dtype | on / bf16 | 同 |
| seed / workers / log | 42 / 4 / 500 | 同 |
| input H×W | 512×960 | 显式沿用 512×960 (native 928×1440 上 crop) |
| dataset / input | STF pseudo / STF raw-RAM | ROD-night teacher_bright label / 在线 student_dark RGB |

## 3. 总体设计

### 3.1 命名

- 数据集类：`RODRawStudentRGB`；family：`rod_raw_student_rgb`；input mode：`raw24_student_rgb`。
- stage：`rod_only`；best_metric：`rod`；eval tag：`rod_night_val`。
- manifest：`rod_night_dav2_rel_manifest.csv`。
- 新 label 目录：`/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/`。
- 渲染单一真值模块：`finetune_stf/dataset/rod_raw_rgb.py`。

### 3.2 两阶段流程

```
Phase 0 (离线, 一次性, tmux):  raw -> teacher_bright RGB -> DAv2-L -> inverse-rel .npy + manifest + run_config
Phase 1 (正式训练):           raw -> 在线 student_dark RGB -> DAv2-S(decoder) -> 拟合 Phase0 缓存 label
```

### 3.3 输入链路与显式配置

```
input_domain=rgb
front_end=dav2_rgb
model_input_tensor=image
dataset_family=rod_raw_student_rgb
dataset_input_mode=raw24_student_rgb
raw_storage_format=n_a
bridge=none
decoder_feature_adapter=none
use_lora=false
rod_label_space=inverse_relative
rod_rgb_pipeline=student_dark_degreen_v1   # 训练 student 输入
```

不使用：`*-sdr_rgb`、现有 sdr_rgb 版 label、raw-RAM front-end、bridge、LoRA、feature adapter。

### 3.4 DAv2-L label 生成的 input_size 决定

**决定 `input_size=924`（≈native 928，=14×66），带 smoke 视觉闸 + 回退 700。**

- teacher 图只有 928 真实行（1856 的 2×2 binning）；924 用尽全部真实分辨率，>928 只是插值放大喂假细节，对 label 有害 → 924 是"最大真实细节"上限。
- label 存 native 928×1440；若在 700 上算再上采样会丢边界细节。
- `input_size` 只改 DAv2-L **计算/细节分辨率**，不改存储尺寸（输出 interpolate 到 928×1440）。
- 风险：§2.1 夜间 raw 极暗，高分辨率可能放大噪声为虚假深度。→ **闸**：Phase 0 脚本先对 20 样本出 `{700,924}` DAv2-L spectral 对比，确认 924 更锐且不放大夜噪；若更糟回退 700。**选定值写进 label 目录 `run_config.json`**。
- 成本：924 vs 700 token~1.7×、attention~3×，一次性离线 tmux ~2–3h，可接受。

### 3.5 label 与 loss

ROD dataset 每个 sample 显式 `sample["target_space"]="inverse_relative"`；loss 走已支持的 `build_training_target(..., target_space="inverse_relative")`，不取倒数。

### 3.6 eval 口径

现有 metric-depth eval 不适用 inverse-rel label。新增 inverse-rel 分支（§4.6），**按 `sample["target_space"]` 分支**（不靠数据集名字符串）：

```
pred_inverse -> interpolate/crop 到 target -> valid=finite&>0
-> 最小二乘 fit s,t 使 s*pred+t ≈ pseudo_inverse
-> aligned=clamp_positive(s*pred+t)
-> 在 inverse space 算 AbsRel/SqRel/RMSE/RMSE_log/SILog/d1/d2/d3
```

指标是对 DAv2-L pseudo inverse label 的 **proxy 一致性**，非真实 metric depth；日志/key 名含 `inverse`/`pseudo_inverse` 以免混淆。

## 4. 代码修改计划

### 4.0 新增 raw→RGB 单一真值模块（前置）

`finetune_stf/dataset/rod_raw_rgb.py`：把 §2.2 已验证管线收敛进仓库（避免 import 仓库外的 rod_tools viz 脚本）。

- 函数：`unpack_raw24 / raw_rggb_project_to_rgb / stretch_white_point / gamma_adjust / rawvis_degreen_rgb / student_dark_degreen_v1 / teacher_bright_degreen_v1`，及常量 `DEGREEN_GAINS / *_WHITE_PERCENTILE / *_GAMMA / RAW_H=1856 / RAW_W=2880 / RAW_MAX=2**24-1`。
- 配套 parity 测试：复用 `make_0603` 的 `--verify-rendered-inputs` 思路，对那 20 样本断言本模块输出 vs 离线 PNG `max_abs==0`，保证拷进仓库后仍 byte-exact。
- 所有参数显式常量，不藏默认；正式脚本显式传。

### 4.1 Phase 0：离线 teacher label 生成脚本

`finetune_stf/scripts/build_rod_night_teacher_labels.py`（长任务，tmux）：

- 遍历 `00Train/01Valid` 的 `night-*.raw`（各 12036 / 2000）。
- 每张：`teacher_bright_degreen_v1(raw)`（§4.0 模块）→ DAv2-L(vitl, ckpt §2.3, `input_size`§3.4) → 输出 interpolate 到 (928,1440) → 存 inverse-rel `.npy` 到新目录对应 split。
- 复刻原 pseudo 生成的推理设置（见 §6.1）：infer 细节（`input_size`、native target、输出为模型原始 inverse 输出，不额外归一化）。
- 写 `run_config.json`（encoder/ckpt/input_size/teacher 管线参数/通道顺序/target_hw/sample_count）+ `rod_night_dav2_rel_manifest.csv`。
- 内置 `--input-size-sweep` 子模式：对 20 样本出 `{700,924}` spectral 对比图到临时路径，供 §3.4 闸决策。

manifest 列：`sample_id,split,scene,raw24_path,rggb_path,pseudo_depth_path,label_space,height,width`

校验：所有路径含 `/mnt/drive/3333_raw/ROD/` 且无 `LOD`；train=12036/val=2000；raw size=`1856*2880*3`；label shape=(928,1440)；`label_space=inverse_relative`。

### 4.2 ROD dataset（改造现有 `LODRGB`，不从零写）

> `finetune_stf/dataset/lod_raw.py::LODRGB/LODRaw` 已覆盖本轮所需绝大部分逻辑：manifest 解析、对齐 native (928,1440)、train random / val center crop、ImageNet 归一化、`target_space="inverse_relative"`、`valid_mask=isfinite&>0`。**唯一替换输入来源**。可复用：`_load_manifest_rows/_load_rgb_manifest_rows`、`_sample_crop_box`、`_apply_crop`、`NormalizeImage`+`PrepareForNet`。

`finetune_stf/dataset/rod_raw_student_rgb.py::RODRawStudentRGB`：

- 输入不再 `cv2.imread(jpg)`，而是 `rod_raw_rgb.student_dark_degreen_v1(unpack_raw24(raw_path))`（在线，native 全图后再 crop）。student 投影输出本身就是 (928,1440,3)，与 native label 同尺寸，**省掉 sdr_rgb→native 的 resize**，crop 几何天然对齐。
- label = `np.load(pseudo_depth_path)`（Phase 0 缓存），与 image 同 crop box。

入参（全部显式）：`rod_root / manifest_path / split / raw_source=raw24 / rgb_pipeline=student_dark_degreen_v1 / student RGB 参数 / label_space=inverse_relative / input_height=512 / input_width=960 / train_crop_mode=random / val_crop_mode=center`。

sample 输出：`image[3,512,960] / depth[512,960] / valid_mask / target_space="inverse_relative" / dataset="rod" / split / sample_id / raw_path / depth_path`。

尺寸：native (928,1440) 上 train random / val center crop 到 512×960（复刻 0521_0112，避免 resize 扭曲几何）。full-frame eval 留作后续显式参数。valid mask 只做 `isfinite & >0`，不隐式重标定。

### 4.3 resolved config + 校验

`finetune_stf/config/resolved.py`：

```
DATASET_FAMILY_CHOICES += "rod_raw_student_rgb"
dataset_input_mode      += "raw24_student_rgb"
```

新增显式参数：`rod_root / rod_night_manifest / rod_raw_source / rod_rgb_pipeline / rod_label_space / rod_train_crop_mode / rod_val_crop_mode / rod_student_white_percentile / rod_student_gamma / rod_student_channel_gains / eval_rod`。

集中校验（`dataset_family=rod_raw_student_rgb` 时）：`input_domain=rgb`、`front_end=dav2_rgb`、`model_input_tensor=image`、`dataset_input_mode=raw24_student_rgb`、`raw_storage_format∈{n_a}`、`bridge=none`、`decoder_feature_adapter=none`、`use_lora=false`、`rod_label_space=inverse_relative`、`rod_raw_source=raw24` 时 raw 路径必须存在。禁止 path 字符串推断 family/storage/label/pipeline/split。传入 active bridge/raw-front-end/LoRA → 直接报错。

### 4.4 train.py 接线

`finetune_stf/train.py`：

- `--stage` choices += `rod_only`；`BEST_METRIC_CHOICES` += `rod`（同步 `build_checkpoint_payload` 的 `best_*`，新增 `best_rod`）。
- parser 加 §4.3 ROD 参数；import `RODRawStudentRGB`。
- `build_datasets` 支持 `dataset_family=rod_raw_student_rgb`；train dataloader key 从硬编码 `stf_train` 改为通用/新增 `rod_train`。
- stage=`rod_only` 的 train/eval 走 ROD train/val loader + ROD inverse-rel eval。
- 日志打印：ROD train/val count、raw source、student RGB pipeline 全参数、label space、crop policy、ckpt、trainable/frozen 参数量、label 目录与 run_config。
- 不复用 `stf` 作为 ROD 的日志/metric 名。

### 4.5 eval inverse-rel 分支

`finetune_stf/train.py` + `finetune_stf/util/metric.py`：新增

```
affine_align_to_inverse_target(pred_inverse, target_inverse, valid_mask)
compute_inverse_relative_metrics(aligned_pred_inverse, target_inverse, valid_mask)
```

`evaluate()` 按 `resolve_batch_target_space(sample)` 分支：`inverse_relative` 走新 helper（§3.6），`metric_depth` 保留旧 `affine_align_disp` 路径给 STF/其他。

### 4.6 parity / 链路一致性脚本

`finetune_stf/scripts/verify_rod_student_rgb_pipeline.py`：抽样 ROD night → 用训练 dataset 的在线函数生成 student RGB → 与 `make_0603` 离线 student PNG / `rod_raw_rgb` 模块 byte-exact 比对 + channel hist + MAE/RMSE/max-diff。输出到临时路径（`/tmp/codex_smoke_rod_student_rgb_verify_...`）。成功删，失败留并报告。

### 4.7 formal launch 脚本

`finetune_stf/scripts/formal/0604_run_rod_night_student_rgb_decoder_e10_queue.sh`：conda `dav3`；tmux；清晰 session/log；不复用已有 session；run 名 `$(date +%m%d_%H%M)_rod_night_studentrgb_dav2s_decoder_e10`；所有实验语义参数显式写出。

核心参数：

```bash
--encoder vits
--pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
--stage rod_only
--input-domain rgb --front-end dav2_rgb --model-input-tensor image
--dataset-family rod_raw_student_rgb --dataset-input-mode raw24_student_rgb
--raw-storage-format n_a --bridge none --decoder-feature-adapter none
--dav2-train-mode decoder --backbone-layer-decay 1.0
--rod-root /mnt/drive/3333_raw/ROD
--rod-night-manifest /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv
--rod-raw-source raw24 --rod-rgb-pipeline student_dark_degreen_v1
--rod-student-white-percentile 99.9 --rod-student-gamma 0.9 --rod-student-channel-gains 1.08 0.95 1.10
--rod-label-space inverse_relative
--input-height 512 --input-width 960 --rod-train-crop-mode random --rod-val-crop-mode center
--eval-rod --best-metric rod --save-best-checkpoint
--bs 8 --accum-steps 1 --lr 1e-5
--loss-type ssi --loss-target-normalization --loss-norm-min-scale 1e-3
--epochs 10 --amp --amp-dtype bf16 --seed 42 --num-workers 4 --log-interval 500
```

### 4.8 smoke 计划

1. **render parity**：`rod_raw_rgb` 模块 vs 20 离线 PNG，`max_abs==0`。
2. **input_size 闸**：20 样本 `{700,924}` DAv2-L spectral 对比，定 label input_size。
3. **label smoke**：Phase 0 脚本跑少量样本到 `/tmp/codex_smoke_rod_labels_...`，校验 shape/label_space/manifest 路径。
4. **dataset smoke**：读 2 train+2 val，校验不访问 sdr_rgb、`image=(3,512,960)`、`depth=(512,960)`、`target_space=inverse_relative`、crop box 一致。
5. **train/eval smoke**：`/tmp/codex_smoke_rod_train_eval_...`，1–2 train step + 极小 eval，查 loss/metric finite、ckpt/log 正常。
成功只删明确含 `codex_smoke` 的临时产物；失败保留并报告路径。

## 5. 需要避免的实现错误

- 不把新 ROD 接口命名为 `lod`。
- 训练/eval 不读 `sdr_rgb`。
- 不复用 sdr_rgb 版 label（域不匹配）；不依赖旧 manifest 中 `LOD` 死链路径。
- 不把 ROD inverse pseudo label 当 metric depth；不在 eval 对 inverse label 先取倒数再算 metric 指标。
- 不靠 path 字符串推断 family/raw storage/label space/pipeline/split。
- formal 脚本不依赖隐藏默认决定实验语义参数。
- 不把 bridge/LoRA/raw-RAM active 参数和 v1 direct-RGB decoder 混用。
- raw→RGB 渲染参数只来自 §4.0 单一真值模块，禁止散落复制导致漂移；先 crop 后渲染会让全局白点漂移，禁止。

## 6. 风险点 / 实现时确认

1. **找回原始 DAv2-L pseudo 生成脚本** 以照搬 infer 细节（§4.1）；若找不到，按现有 `pseudo_depth_dav2_night_rel_1440x928/run_config.json`（encoder=vitl, input_size=700, target_hw=(928,1440)）精确重建，并对旧 sdr_rgb label 做一次回归（同 sdr_rgb 输入应复现旧 .npy）以验证 infer 等价后，再切到 teacher_bright + input_size=924。
2. **input_size 闸**(§3.4)：924 在夜间噪声上若劣于 700 则回退；决定落 run_config。
3. ROD eval 的 AbsRel/delta 是 pseudo inverse proxy，非真实 metric depth。
4. Phase 0 全量 ~2–3h，必须 tmux + 清晰 log；不可中途复用他人 session。
5. 跨渲染(student_dark vs teacher_bright)仅 tone 差异，分析时如此表述，勿夸大为颜色/几何域差。
6. 旁注：sdr/raw_vis/teacher 三种渲染的 DAv2-L 输出相关性 0.996–1.0（即复用旧 label 在数值上也接近可行），但本版仍选同族 teacher 重生成——更干净、消融更清楚、避免 sdr ISP（grayworld AWB+bilateral）引入的额外域差。

## 7. 验收标准

- 新 label 目录 `pseudo_depth_dav2l_night_teacherbright_rel_1440x928/` 生成完成，所有 manifest 路径指向 `/mnt/drive/3333_raw/ROD`，train/val=12036/2000，label shape (928,1440)，`label_space=inverse_relative`，且 `run_config.json` 记录 teacher 管线参数 + 选定 input_size。
- `rod_raw_rgb` 模块对 20 样本 student/teacher 渲染 byte-exact (`max_abs=0`)。
- 训练/eval 都从 raw24 在线生成 student RGB，均不读 `*-sdr_rgb`。
- dataset sample 显式 `target_space=inverse_relative`；image=(3,512,960)、depth=(512,960)、crop box 一致。
- decoder-only trainable 参数与 0521_0112 同口径；loss=`ssi+target-norm+min-scale 1e-3`；epochs=10。
- eval 在 inverse label space 做 scale/shift alignment 与指标计算，按 `target_space` 分支、保留 metric-depth 旧分支。
- formal run 名以 `MMDD_HHMM` 开头，所有实验语义参数显式。
- smoke 临时产物路径含 `codex_smoke`/`debug`/`tmp`，成功后只删这些。

---

## 8. 后续 train-scope 消融执行计划 (rev3 追加, 2026-06-05)

> 目标：在 0604 decoder-only baseline (`0604_0752_rod_night_studentrgb_dav2s_decoder_e10`) 之上，增加两组 **encoder 适配预算 (encoder-adaptation budget)** 对照，**单变量、其余全锁死 = baseline**。配方对齐先例项目 `/home/caq/6666_raw/dav2_raw_0520`（`finetune_stf/scripts/0521_run_stf_lora_full_dav2s_queue.sh`，同 vits 上已跑过 decoder / LoRA / full+layer-decay 三件套）。
>
> 实测确认 (2026-06-05)：0603 仓库已具备全部 CLI flag（`--lora/--lora-block-mode/--lora-tap-layers/--lora-rank/--lora-alpha/--lora-lr`、`--dav2-train-mode full`、`--backbone-layer-decay`）、LLRD 参数组构建器 `_build_layer_decay_param_groups`（`train.py:2552`，`backbone_layer_decay<1.0` 时自动触发）、以及 `lora=dav2_lora` 时对 rank/alpha/lr/tap-layers 的显式强制（`resolved.py:814-829`）。**只差一处 v1 闸需放开（见 8.1）。**

### 8.0 三组定义

| 组 | dav2_train_mode | lora | backbone_layer_decay | 相对 baseline 的差异 |
| --- | --- | --- | --- | --- |
| A decoder-only（已完成） | decoder | none | 1.0 | — |
| **B LoRA + decoder** | decoder | dav2_lora（tap, r8/α16, lr 5e-5） | 1.0 | 增 6 行 `--lora-*` |
| **C backbone 小 lr + decoder** | **full** | none | **0.9** | 改 2 行 train-mode / layer-decay |

锁死项（三组完全一致，= baseline）：teacher_bright label / `student_dark_degreen_v1`（wp 99.9、γ 0.9、gains 1.08 0.95 1.10）/ 512×960（random·center crop）/ `ssi + target-norm + min-scale 1e-3` / epochs 10 / bs 8 / **decoder lr 1e-5** / seed 42 / amp bf16 / `--eval-rod` inverse-rel 分支 / `--best-metric rod`。

语义说明（写进日志/分析）：
- **C 组 LLRD**：decoder/head 在 base lr 1e-5；backbone 第 L 层 lr = `1e-5 × 0.9^(max_layer−L)`，越靠输入端越小。即"decoder 正常 + backbone 温和适配"，非全网单一小 lr。
- **B 组 tap 层** = vits 的 DPT reassemble 点 `[2,5,8,11]`（`depth_anything_v2/dpt.py:176`）。0603 强制 `block-mode=tap` 时显式传 `--lora-tap-layers`，故脚本显式写 `2 5 8 11`。
- **lr 不照搬**：B 组 LoRA 参数用 `--lora-lr 5e-5`（= decoder lr 的 5×，沿用 0520 先例），decoder 仍 1e-5。

### 8.1 唯一代码改动：放开 rod family 的 LoRA 闸（仅 B 组前置；C 组不需要）

> C 组（backbone full + layer decay）**零代码改动**即可跑——§4.3 的 v1 闸只拦 `bridge/adapter/lora`，不拦 `dav2_train_mode`。只有 B 组（引入 `lora=dav2_lora`）需要这处放开。

文件 `finetune_stf/config/resolved.py`（约 line 878-879），把：

```python
        if cfg.bridge != NONE or cfg.decoder_feature_adapter != NONE or cfg.lora != NONE:
            raise ValueError("dataset_family=rod_raw_student_rgb v1 requires bridge/decoder_feature_adapter/lora all none")
```

改为：

```python
        if cfg.bridge != NONE or cfg.decoder_feature_adapter != NONE:
            raise ValueError("dataset_family=rod_raw_student_rgb requires bridge/decoder_feature_adapter none")
        if cfg.lora not in (NONE, "dav2_lora"):
            raise ValueError("dataset_family=rod_raw_student_rgb only supports lora in {none, dav2_lora}")
```

无需再加显式校验：`resolved.py:814-829` 已在 `lora=dav2_lora` 时强制 `lora_rank/alpha/lr` 显式、`block-mode=tap` 时强制 `lora_tap_layers` 显式 → 只要 8.3/8.2 脚本把这些显式传入即满足 CLAUDE.md §6。改完跑一次任意现有 rod 命令确认无回归报错。

补充审核发现：B 组还需要把 `rod_raw_student_rgb + dav2_rgb + lora=dav2_lora` 的 orthogonal config 映射到现有 legacy alias `rgb_lora`。否则解析阶段会报 "no legacy model factory alias"。对应修改在 `finetune_stf/config/resolved.py::_legacy_alias_from_config` 的 ROD 分支：`return "rgb_lora" if lora else "rgb"`。

### 8.2 单份顺序启动脚本（基于 baseline 脚本 + 多实验 queue pattern 派生）

正式入口改为一份脚本：`finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh`。

脚本要求：
- 外层只启动 **一个 tmux session**，`SESSION_PREFIX` 默认 `rod_night_studentrgb_encoder_budget_e10`。
- tmux 内部按顺序执行 **B 组 → C 组**；上一组失败时 `set -e` 停止队列，保留失败产物。
- B/C 各自在真正开始训练前执行 `date +%m%d_%H%M` 生成 run name，满足 formal experiment 命名使用 launch time 的要求。
- 打印命令和实跑命令共享同一组 bash array，避免 baseline 单实验脚本里“打印块/实跑块两处 CLI 需要同步改”的风险。
- 默认 conda env 仍为 `dav3`；默认 GPU 为 `${CUDA_VISIBLE_DEVICES:-0}`；B/C 端口分别为 `B_MASTER_PORT=29605`、`C_MASTER_PORT=29606`。
- 提供 `--audit` 模式，只解析并打印 resolved config，不启动训练。

**B 组**（脚本内 `run_formal_exp "B LoRA tap r8/a16 + decoder"`）：
- suffix：`rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10`
- 额外显式参数：
```
    --lora dav2_lora
    --lora-block-mode tap
    --lora-tap-layers 2 5 8 11
    --lora-rank 8
    --lora-alpha 16
    --lora-lr 5e-5
    --dav2-train-mode decoder
    --backbone-layer-decay 1.0
```

**C 组**（脚本内 `run_formal_exp "C full backbone LLRD 0.9 + decoder"`）：
- suffix：`rod_night_studentrgb_dav2s_backbone_ld09_decoder_e10`
- 额外显式参数：
```
    --lora none
    --dav2-train-mode full
    --backbone-layer-decay 0.9
```

### 8.3 smoke（每组各 1 次，先 B 后 C；失败保留产物）

不经 queue 脚本，直接命令行跑极小步数，输出到 `codex_smoke` 路径。B 组示例（**C 组**：删掉 6 行 `--lora-*`、`--dav2-train-mode decoder`→`--dav2-train-mode full`、`--backbone-layer-decay 1.0`→`0.9`）：

```bash
SMK=/mnt/drive/3333_raw/0000_exp_ckpt/codex_smoke_$(date +%m%d_%H%M)_rod_lora_decoder
CUDA_VISIBLE_DEVICES=0 conda run --live-stream -n dav3 \
  torchrun --nproc_per_node=1 --master_port=29607 finetune_stf/train.py \
    --encoder vits --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
    --stage rod_only --input-domain rgb --front-end dav2_rgb --model-input-tensor image \
    --dataset-family rod_raw_student_rgb --dataset-input-mode raw24_student_rgb --raw-storage-format n_a \
    --bridge none --decoder-feature-adapter none \
    --lora dav2_lora --lora-block-mode tap --lora-tap-layers 2 5 8 11 --lora-rank 8 --lora-alpha 16 --lora-lr 5e-5 \
    --dav2-train-mode decoder --backbone-layer-decay 1.0 \
    --rod-root /mnt/drive/3333_raw/ROD \
    --rod-night-manifest /mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv \
    --rod-raw-source raw24 --rod-rgb-pipeline student_dark_degreen_v1 \
    --rod-student-white-percentile 99.9 --rod-student-gamma 0.9 --rod-student-channel-gains 1.08 0.95 1.10 \
    --rod-label-space inverse_relative --input-height 512 --input-width 960 \
    --rod-train-crop-mode random --rod-val-crop-mode center \
    --no-eval-stf --eval-rod --best-metric rod --save-best-checkpoint \
    --bs 8 --accum-steps 1 --lr 1e-5 --loss-type ssi --loss-target-normalization --loss-norm-min-scale 1e-3 \
    --epochs 1 --debug-max-train-steps 2 --debug-max-val-samples 8 \
    --amp --amp-dtype bf16 --seed 42 --num-workers 4 --log-interval 1 \
    --no-enable-fixed-viz-dump --no-enable-train-source-viz-dump \
    --heavy-save-root "${SMK}" --save-path "${SMK}"
```

检查 `${SMK}/resolved_config.json` + train.log：
- B：`lora=dav2_lora`、`lora_tap_layers=[2,5,8,11]`、`lora_rank=8/alpha=16/lr=5e-5`、`dav2_train_mode=decoder`；`optimizer_param_groups` 出现 decoder + lora 两组，lora 组 lr=5e-5。
- C：`lora=none`、`dav2_train_mode=full`、`backbone_layer_decay=0.9`；`optimizer_param_groups` 出现多组（backbone 各层 lr 递减 + decoder 1e-5），`trainable_params` 远大于 baseline 的 2,728,513。
- B 组 `input_type_alias` 会随 LoRA 变为 `rgb_lora`，这是 legacy model factory alias，不是额外输入语义；C 组仍为 `rgb`。
- loss / eval metric 全 finite。
- 成功 → 删 `${SMK}`（含 codex_smoke 标记）；失败 → 保留并报告路径。

### 8.4 正式启动 + 监控

smoke 过后，只跑这一份顺序队列脚本：
```bash
bash finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh
```
脚本会启动一个 tmux session，并打印 `tmux attach -t <session>` 与 `tail -f <queue_log>`。tmux 内部单卡顺跑 B→C（每组 ~5.5h，参考 baseline 10 epoch ≈ 5h40m）。**不复用、不 kill 已有 session。**

### 8.5 验收 + 比较口径

- `diff` 三组的 `resolved_config.json`：**只应在 `lora* / input_type_alias / dav2_train_mode / backbone_layer_decay / optimizer_param_groups` 上不同**，其余语义字段必须与 baseline 完全一致。B 的 `input_type_alias=rgb_lora` 仅是 LoRA 模型工厂 alias，输入仍为 `rgb/dav2_rgb/image`。
- B/C 的 `[DATASET] rod_train=12036 rod_night_val=2000`、`target_space=inverse_relative`、`image=(8,3,512,960)` 与 baseline 一致。
- **比较只看 `d1` / `silog`，不看 `abs_rel`/`rmse`**（ROD inverse-rel eval 的 abs_rel 被近零 inverse 分母放大，是 proxy，不可跨组比绝对值；baseline d1≈0.80）。注意当前 `--best-metric rod` 仍按代码里的 `summary["abs_rel"]` 选 best checkpoint；分析时应从每轮 eval log/metrics 中读 d1/silog，不把 best checkpoint 的 abs_rel 选择当作结论依据。三组天花板都是 DAv2-L 亮图 teacher，衡量"暗域 student 逼近 teacher 的能力"，非真实 metric 深度。
- 三组**统一 epochs=10** 以对齐 baseline（纯公平对比；baseline 虽 epoch 2 即 plateau，不为此改 budget）。
- 结论维度：相对 decoder-only baseline，LoRA(B) 与 backbone-LLRD(C) 是否在 d1/silog 上带来一致提升 → 回答"冻结 encoder 是否为此跨渲染暗域蒸馏的瓶颈"。

## 9. rev3 增补：ROD packed Bayer -> RamCore3 raw 输入第一版（2026-06-05）

> 本节是 0605 新增计划，覆盖下一轮实现与实验口径；前文 rev2/student RGB 计划保留为历史记录和已完成 RGB baseline 参照。下一轮第一版不再把 ROD RAW 渲染成 student RGB 后喂 DAv2 backbone，而是复刻 `dav2_raw_0520` 的 raw-RAM 输入路径。

### 9.1 新目标和边界

目标：在当前 ROD night train/eval 框架上新增 `rod_raw` 数据集族，让模型输入为 packed Bayer raw tensor，并走：

```text
ROD raw24 / rggb cache -> packed Bayer raw4 [R,Gr,Gb,B]
-> [R,(Gr+Gb)/2,B]
-> RamCore3
-> DAv2-S backbone + decoder
```

第一版明确不做 bridge、decoder feature adapter、backbone-layer-to-decoder-layer 连接。只实现：

| 组别 | 输入 | front-end | 训练部分 | epoch |
| --- | --- | --- | --- | ---: |
| R1 | ROD raw4 | `raw_to_base_rgb_ram3` + `identity` tail | RamCore3 + DAv2 decoder | 5 |
| R2 | ROD raw4 | `raw_to_base_rgb_ram3` + `identity` tail | RamCore3 + DAv2 LoRA + DAv2 decoder | 5 |
| RGB-ref | ROD student RGB | `dav2_rgb` | DAv2 backbone + decoder，低学习率 | 5 |

锁死项：
- label 仍使用现有 DAv2-L `teacher_bright_degreen_v1` inverse-relative pseudo label manifest：`/mnt/drive/3333_raw/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv`。
- ROD split、crop、loss、eval、batch、seed 先沿用当前 student RGB ROD 实验：512x960 train random crop / val center crop，`ssi + target-normalization + min-scale 1e-3`，bs 8，amp bf16，seed 42。
- 正式实验统一 `epochs=5`。

### 9.2 Tail 选择

本轮 raw-RAM 路径显式使用：

```bash
--raw-ram-rgb-tail identity
```

原因：`identity` 对应最直接的语义：

```text
raw4 -> [R,(Gr+Gb)/2,B] -> RamCore3 -> DAv2
```

即 RamCore3 BN 输出不再经过额外 soft squash，直接作为 DAv2 输入。`tanh2p5` 是稳定化对照项，会把 RamCore3 输出软限制到约 `[-2.5,2.5]`，本轮不默认使用；如 R1/R2 训练不稳，再单独加 `tanh2p5` ablation。

注意：当前 `front_end=raw_to_base_rgb_ram3` 有 `PHASE1_BNCLEAN_REVIEWED` 环境闸。实现/脚本必须在确认本节语义后显式设置：

```bash
PHASE1_BNCLEAN_REVIEWED=1
```

### 9.3 命名和显式配置

新增语义命名：

```text
dataset_family=rod_raw
dataset_input_mode=raw_ram
rod_raw_source=raw24
model_input_tensor=raw
input_domain=raw4
front_end=raw_to_base_rgb_ram3
raw_storage_format=n_a
```

说明：
- `rod_raw` 表示 ROD 数据集族，不再使用 `rod_raw_student_rgb`，避免把 raw-RAM 实验误解释成 student RGB 输入。
- `raw_ram` 表示 dataset 返回 `sample["raw"]`，给模型的是 4ch packed Bayer tensor。
- `rod_raw_source=raw24` 表示 ROD 原始采集文件格式是 24-bit raw；第一版数据加载优先使用 manifest 中的 `rggb_path` 缓存 `(928,1440,4)`，并记录为 `packed_bayer_source=rggb_cache_prefer_raw24_fallback`。如果缓存缺失，再从 `raw24_path` 解包并 pack 成 `[R,Gr,Gb,B]`。
- `raw_storage_format=n_a`：ROD packed Bayer 已是 `[R,Gr,Gb,B]`、float `[0,1]`，不能套 STF 的 `legacy_bggR_decomp16` 或任何路径名推断。

### 9.4 数据集代码计划

新增文件建议：`finetune_stf/dataset/rod_raw.py`。

新增 `RODRaw` dataset：
- 复用当前 `rod_raw_student_rgb.py` 的 manifest 解析、split、target、crop、metadata 逻辑。
- 每个 sample 读取 `rggb_path` 缓存，期望 shape `(928,1440,4)`、通道 `[R,Gr,Gb,B]`、float `[0,1]`。
- fallback：如果 `rggb_path` 缺失且允许 fallback，则用 `unpack_raw24(raw24_path)` 后 pack：

```text
R  = raw[0::2, 0::2]
Gr = raw[0::2, 1::2]
Gb = raw[1::2, 0::2]
B  = raw[1::2, 1::2]
raw4 = stack([R,Gr,Gb,B], axis=-1)
```

- 在 native `(928,1440)` 上与 pseudo label 使用同一个 crop box，再输出：

```python
sample["raw"] = torch.Tensor(4, 512, 960)
sample["image"] = optional preview only, not model input
sample["depth"] = torch.Tensor(512, 960)
sample["valid_mask"] = target finite & > 0
sample["target_space"] = "inverse_relative"
sample["dataset"] = "rod"
```

第一版可以不生成 normalized student RGB `image`，但为了现有可视化/日志兼容，建议 `image` 用 `packed_bayer_to_base_rgb` 或 ROD student preview 生成，只作为 preview，不参与模型输入。训练入口必须以 `model_input_tensor=raw` 选择输入。

### 9.5 Config/resolved 修改计划

`finetune_stf/config/resolved.py`：
- `DATASET_FAMILY_CHOICES += ("rod_raw",)`。
- 复用已有 `DATASET_INPUT_MODE_CHOICES` 中的 `raw_ram`。
- `validate_resolved_config` 新增 `rod_raw` 分支：

```text
dataset_family=rod_raw requires:
  input_domain=raw4
  front_end in {raw_to_base_rgb_ram3}        # 第一版只放开 RamCore3 路径
  dataset_input_mode=raw_ram
  model_input_tensor=raw
  raw_storage_format=none/n_a
  bridge=none
  decoder_feature_adapter=none
  lora in {none,dav2_lora}
```

- **必改**：修改现有 raw front-end 通用校验 `validate_resolved_config`（`resolved.py:884-888`），现硬编码 `if cfg.dataset_family != "stf_raw" or cfg.dataset_input_mode != "raw_ram": raise ...`。改为允许 `front_end in {raw_to_rgb_head, raw_ram4, raw_to_base_rgb_ram3}` 时 `dataset_family in {stf_raw, rod_raw}` 且 `dataset_input_mode == raw_ram`。这是 R1/R2 能通过解析的**前置硬条件**。
- **无需改动（已满足，勿误加编辑）**：`_legacy_alias_from_config`（`resolved.py:689-691`）的 `raw_to_base_rgb_ram3` 分支**只看 `front_end`、不看 `dataset_family`**，因此 `rod_raw + raw_to_base_rgb_ram3`（无 bridge/adapter）会原样返回 `raw_ram_rgb` / `raw_ram_rgb_lora`。这与 §8.1 中 `rod_raw_student_rgb` 必须新增 dataset_family 分支的情况不同——本节**不要**再去改这个函数。
- **无需改动**：`raw_storage_format=n_a` 经 `_raw_storage_format_from_args`（`resolved.py:537`）归一化为 `none`，故能通过 `resolved.py:933` 的 `!= stf_raw` 校验，§9.3 传 `n_a` 不会报错。
- `raw_front_end_lr` 必须在 raw-RAM 正式脚本中显式传入，不能依赖 resolver 默认（`resolved.py:795-797` 在 orthogonal raw front-end 下已强制显式，缺失会报错）。

### 9.6 train.py 接线计划

`finetune_stf/train.py`：
- import `RODRaw`。
- `uses_rod_dataset` 或等价判断扩展为 `dataset_family in {"rod_raw_student_rgb", "rod_raw"}`。
- **必改 / 拆分 parser 校验块（关键，否则 R1/R2 无法启动）**：现 `train.py:642-664` 整块由 `if uses_rod_dataset(args):` 包裹，其中 `647-648` 强制 `--rod-rgb-pipeline == student_dark_degreen_v1`、`651-656` 强制 `--rod-student-white-percentile/gamma/channel-gains == 99.9/0.9/(1.08,0.95,1.10)`。一旦 `uses_rod_dataset` 含 `rod_raw`，这些**纯 student-RGB 渲染参数会被强加给 raw 输入的 R1/R2**，与本节"preview-only、不让 RGB pipeline 决定模型输入"矛盾，也违反 CLAUDE.md §6（耦合不相关语义参数）。拆分为：
  - **公共项（两族共用）**：`rod_raw_source == raw24`、`rod_label_space == inverse_relative`、`rod_root`/`rod_night_manifest` 存在性校验（现 645-646、649-650、657-664）。
  - **仅 `dataset_family == "rod_raw_student_rgb"` 时**才校验 `rod_rgb_pipeline / rod_student_white_percentile / rod_student_gamma / rod_student_channel_gains`（现 647-648、651-656）下沉进该分支。
  - 同步更新 `640-641` / `643-644` 的报错文案（现写死 "requires --dataset-family rod_raw_student_rgb"），改为接受 `rod_raw_student_rgb` 与 `rod_raw`。
- `build_datasets` 中新增 `cfg.dataset_family == "rod_raw"` 分支，构建 `RODRaw`，**key 必须沿用现有 rod 分支：train=`rod_train`、val=`val`**（对齐 `train.py:1438` 的 train_key 选择与 `train.py:2833` 的 `uses_rod_dataset` eval 命名），共享 `--eval-rod` / `--best-metric rod`。
- ROD parser 参数继续复用：

```bash
--rod-root
--rod-night-manifest
--rod-raw-source raw24
--rod-label-space inverse_relative
--rod-train-crop-mode random
--rod-val-crop-mode center
```

- 对 `rod_raw` 不要求 `--rod-rgb-pipeline student_dark_degreen_v1`，或者将其标记为 preview-only；不要让 RGB pipeline 决定模型输入。
- 日志必须打印：

```text
[RESOLVED] input_domain=raw4 front_end=raw_to_base_rgb_ram3 model_input_tensor=raw
[DATASET] dataset_family=rod_raw dataset_input_mode=raw_ram packed_bayer_source=...
[MODEL] raw4 -> [R,(Gr+Gb)/2,B] -> RamCore3 -> identity -> DAv2
```

模型侧预计无需新增 wrapper：当前 `build_model()` 已支持 `front_end=raw_to_base_rgb_ram3` 且无 bridge/adapter 时构建 `RawToBaseRgbRam3DepthModel`。R1/R2 只需确保 `dataset_family=rod_raw` 能通过 config 校验并提供 `sample["raw"]`。

### 9.7 三组正式实验参数

建议新增顺序队列脚本：

```text
finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh
```

脚本要求：
- 外层启动一个 tmux session，不复用已有 session。
- tmux 内部顺序跑 R1 -> R2 -> RGB-ref。
- 每个正式 run 在真正 launch 前用 `date +%m%d_%H%M` 生成 run name。
- `CONDA_ENV=dav3`，日志和 heavy root 沿用当前正式脚本习惯。
- 三组都显式 `--epochs 5`。
- **下面各组只列相对公共基座的增量参数**。公共基座（三组都必须显式传）：`--rod-root /mnt/drive/3333_raw/ROD`、`--rod-night-manifest <teacherbright manifest>`、`--rod-raw-source raw24`、`--rod-label-space inverse_relative`、`--rod-train-crop-mode random --rod-val-crop-mode center`、`--input-height 512 --input-width 960`、`--lr 1e-5`（decoder base lr，experiment-semantic，按 CLAUDE.md §6 显式写，**不吃默认**；RGB-ref 例外，见其块）、`--loss-type ssi --loss-target-normalization --loss-norm-min-scale 1e-3`、`--bs 8 --accum-steps 1 --amp --amp-dtype bf16 --seed 42 --num-workers 4`、`--eval-rod --best-metric rod --save-best-checkpoint`、`--raw-storage-format n_a`。
- **RGB-ref 额外**：因其 `dataset_family=rod_raw_student_rgb`，仍会过 §9.6 拆分后的 student 校验，故公共基座之外**还须显式带** `--rod-rgb-pipeline student_dark_degreen_v1 --rod-student-white-percentile 99.9 --rod-student-gamma 0.9 --rod-student-channel-gains 1.08 0.95 1.10`；R1/R2（`rod_raw`）则**不传**这些 student 参数。

R1：raw RAM + decoder

```bash
--stage rod_only
--dataset-family rod_raw
--dataset-input-mode raw_ram
--input-domain raw4
--front-end raw_to_base_rgb_ram3
--model-input-tensor raw
--raw-storage-format n_a
--rod-raw-source raw24
--bridge none
--decoder-feature-adapter none
--lora none
--dav2-train-mode decoder
--backbone-layer-decay 1.0
--raw-front-end-lr 5e-5
--raw-ram-rgb-tail identity
--epochs 5
```

Run suffix：

```text
rod_night_rawram3_identity_dav2s_ram_decoder_e5
```

R2：raw RAM + LoRA + decoder

```bash
--stage rod_only
--dataset-family rod_raw
--dataset-input-mode raw_ram
--input-domain raw4
--front-end raw_to_base_rgb_ram3
--model-input-tensor raw
--raw-storage-format n_a
--rod-raw-source raw24
--bridge none
--decoder-feature-adapter none
--lora dav2_lora
--lora-block-mode tap
--lora-tap-layers 2 5 8 11
--lora-rank 8
--lora-alpha 16
--lora-lr 5e-5
--dav2-train-mode decoder
--backbone-layer-decay 1.0
--raw-front-end-lr 5e-5
--raw-ram-rgb-tail identity
--epochs 5
```

Run suffix：

```text
rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e5
```

RGB-ref：student RGB + backbone low-lr reference

```bash
--stage rod_only
--dataset-family rod_raw_student_rgb
--dataset-input-mode raw24_student_rgb
--input-domain rgb
--front-end dav2_rgb
--model-input-tensor image
--raw-storage-format n_a
--bridge none
--decoder-feature-adapter none
--lora none
--dav2-train-mode full
--backbone-layer-decay 0.9
--lr 1e-6
--epochs 5
```

Run suffix：

```text
rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5
```

说明：当前 optimizer 没有独立的 backbone LR 参数。`--lr 1e-6` 会同时降低 DAv2 decoder 和 backbone 最后一层 LR，backbone 早期层再乘 `0.9^k`。如果后续目标是“decoder 仍 1e-5、只把 backbone 调小”，需要新增独立 `--backbone-lr` 或 optimizer group 逻辑；第一版参考 RGB 先不改 optimizer。

注意：本 RGB-ref 是 `epochs=5 + full backbone + lr 1e-6` 的**全新 run**，并非 §8 已完成的 decoder-only/e10 baseline，不要混为一谈。它与 R1/R2 的 lr 预算口径**不同**（R1/R2 = RamCore3@`5e-5` + decoder@`1e-5`；RGB-ref = 全网 `1e-6`×layer-decay），分析时按此口径表述，不要当成同一 lr 下的纯对照——它主要回答"student RGB 输入下温和适配 backbone 能到什么水平"，作为 raw-RAM 路径的旁参照。

### 9.8 Smoke 和验收

每组先跑 foreground smoke，输出路径必须含 `codex_smoke`，例如：

```bash
--epochs 1
--debug-max-train-steps 2
--debug-max-val-samples 8
--save-path /tmp/codex_smoke_rod_rawram3_identity_decoder
--heavy-save-root /tmp/codex_smoke_rod_rawram3_identity_decoder_heavy
```

成功后只删除这些明确含 `codex_smoke` 的 smoke 产物；失败保留并报告路径。

R1/R2 smoke 检查：
- `resolved_config.json` 为 `dataset_family=rod_raw`、`dataset_input_mode=raw_ram`、`front_end=raw_to_base_rgb_ram3`、`model_input_tensor=raw`、`raw_ram_rgb_tail=identity`。
- batch input tensor shape 为 `(B,4,512,960)`。
- R1 optimizer groups 至少包含 `raw_front_end` 和 `dav2_decoder`。
- R2 optimizer groups 至少包含 `raw_front_end`、`lora`、`dav2_decoder`。
- `lora_tap_layers=[2,5,8,11]`，rank/alpha/lr 显式记录。

RGB-ref smoke 检查：
- `dataset_family=rod_raw_student_rgb`、`front_end=dav2_rgb`、`model_input_tensor=image`。
- `dav2_train_mode=full`、`backbone_layer_decay=0.9`、`lr=1e-6`。
- optimizer groups 出现 backbone layer groups + `dav2_decoder`。

三组共同验收：
- `[DATASET] rod_train=12036`、`rod val=2000`。
- `target_space=inverse_relative`，loss/eval finite。
- ROD eval 仍按 inverse-relative pseudo label 口径比较，主要看 `d1` / `silog`，不要把 `abs_rel` 当最终结论。

### 9.9 后续不在第一版范围内

- raw bridge / decoder feature adapter。
- RamCore3 feature 接 DAv2 backbone layer 或 decoder layer。
- `tanh2p5` tail ablation。
- 独立 backbone LR optimizer group。
- full-frame ROD eval 或新的 best checkpoint metric 选择逻辑。
