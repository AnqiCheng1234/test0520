# 参数联动关系审计

日期：2026-05-22

范围：当前项目训练入口、模型封装、数据集、评估与常用启动脚本。本文档只记录审计结论，不包含代码修订。

## 1. 总体结论

当前项目里确实存在多处“改一个实验变量，实际联带改变另一组行为”的问题。最核心的问题是 `input_type` 承担了过多职责：

- 决定 RGB / raw / raw RAM / bridge / LoRA / feature adaptor 的模型路径
- 决定使用 `STF` 还是 `STF_RAW`
- 决定模型输入取 `sample["image"]` 还是 `sample["raw"]`
- 决定 raw-like STF 是否强制 native 尺寸 `512x960`
- 决定 bridge / feature adaptor 默认使用 `x3` 还是 `x4`
- 决定 eval dataset 的 raw/RGB 路径
- 决定是否触发 BN-clean guard

因此，当前很多实验变量并不是正交变量。比如“加入 feature adaptor”在部分路径里会同时引入 4ch raw front-end、`RawRamCore`、`x4` feature key、raw dataset input mode 等隐含变化。

## 2. 高风险联动点

### 2.1 `input_type` 是最大耦合源

位置：

- `finetune_stf/train.py:149`
- `finetune_stf/train.py:894`
- `finetune_stf/train.py:1218`

问题：

`input_type` 同时表达输入域、模型前端、bridge、LoRA、feature adaptor。比如：

- `rgb`
- `rgb_lora`
- `raw`
- `raw_packed`
- `raw_ram`
- `raw_ram_rgb`
- `raw_ram_bridge`
- `raw_ram_rgb_bridge`
- `raw_ram_feature_adapter`
- `raw_ram_bridge_feature_adapter`
- `raw_ram_rgb_bridge_feature_adapter`

这些名字看起来像一个变量，但实际包含多个维度。后续修订时应尽量拆成正交配置：

- `input_domain`: `rgb` / `raw4`
- `front_end`: `dav2_rgb` / `raw_to_rgb_head` / `raw_ram4` / `raw_to_base_rgb_ram3`
- `bridge_enabled`: `true` / `false`
- `feature_adapter_enabled`: `true` / `false`
- `lora_enabled`: `true` / `false`
- `bridge_feature_source_channels`: `x3` / `x4`
- `adapter_feature_source_channels`: `x3` / `x4`

其中 `input_domain` 只表示整个模型 wrapper 接收的输入张量域，而不是 DAv2 内部输入。`raw4` 指 4ch packed Bayer 输入 `[R, Gr, Gb, B]`。当前 `raw_packed` / `packed_raw` 更适合作为一种 front-end 路径理解：`raw4 -> 4to3 1x1/raw_to_rgb_head -> DAv2`，不应作为独立 `input_domain`。

### 2.2 Feature adaptor 与 4ch/3ch 路径没有解耦

位置：

- `finetune_stf/models/raw_feature_adapter.py:26`
- `finetune_stf/models/raw_feature_adapter.py:136`
- `finetune_stf/models/raw_feature_adapter.py:155`
- `finetune_stf/models/raw_feature_adapter.py:364`
- `finetune_stf/train.py:761`
- `finetune_stf/train.py:950`

当前路径：

| `input_type` | 前端 | bridge | decoder feature adaptor | 默认 feature key |
| --- | --- | --- | --- | --- |
| `raw_ram_feature_adapter` | 4ch `RawRamCore` | 否 | 是 | `x_cat,ffm_mid,x4` |
| `raw_ram_bridge_feature_adapter` | 4ch `RawRamCore` | 是 | 是 | `x_cat,ffm_mid,x4` |
| `raw_ram_bridge_feature_adapter_lora` | 4ch `RawRamCore` | 是 | 是 | `x_cat,ffm_mid,x4` |
| `raw_ram_rgb_bridge_feature_adapter` | 3ch `RamCore3` | 是 | 是 | `x_cat,ffm_mid,x3` |

问题：

如果实验目标只是“加入 feature adaptor”，使用 `raw_ram_bridge_feature_adapter` 会同时改变到 4ch raw RAM 路径。当前没有一个干净的变量表示“只在现有 3ch/RGB 路径上加入 decoder feature adaptor”。

建议：

- 把 feature adaptor 变成独立开关，而不是编码进 `input_type`
- 单独暴露 `feature_adapter_source_channels=x3|x4`
- 单独暴露 `feature_adapter_keys`
- 明确区分 `decoder_feature_adapter` 和 `bridge_adapter`

### 2.3 raw-like STF 会强制使用 `STF_RAW` 和 raw RAM input mode

位置：

- `finetune_stf/train.py:869`
- `finetune_stf/train.py:1222`
- `finetune_stf/dataset/stf_raw.py:260`
- `finetune_stf/dataset/stf_raw.py:322`

问题：

只要 `input_type` 落入 raw-like 集合，STF dataset 就从 `STF` 切换到 `STF_RAW`。并且 `resolve_stf_raw_input_mode()` 对 raw RAM / bridge / feature adaptor 路径返回 `raw_ram`，最终 dataset 会输出：

- `sample["image"] = raw4`
- `sample["raw"] = raw4`

这意味着模型路径变化会联带改变数据加载路径和输入张量语义。

建议：

- 显式增加 `dataset_family=stf_rgb|stf_raw`
- 显式增加 `model_input_tensor=image|raw`
- 启动时打印 resolved dataset config

### 2.4 raw-like STF 强制输入尺寸为 `512x960`

位置：

- `finetune_stf/train.py:626`
- `finetune_stf/dataset/stf_raw.py:43`

问题：

当 `stage=stf_only` 且 `input_type` 属于 raw-like，训练尺寸必须等于 `STF_RAW_NATIVE_HW=(512,960)`。因此切换 raw-like model 不只是切模型，也可能联带改变或限制输入分辨率。

脚本风险：

`finetune_stf/scripts/train_raw_ram_e3_bridge.sh` 中 raw-like `stf_only` 使用 `518x966`，按当前检查逻辑会失败，可能是过时脚本。

### 2.5 `stf_raw_decode_mode` 和 `norm_mode` 强耦合

位置：

- `finetune_stf/train.py:604`
- `finetune_stf/dataset/stf_raw.py:170`
- `finetune_stf/dataset/raw_utils.py:72`

当前规则：

- `legacy_companded` 可以使用非 `passthrough` 的 `norm_mode`
- `legacy_online_decomp16` 必须使用 `norm_mode=passthrough`
- `canonical_decomp16` 必须使用 `norm_mode=passthrough`
- `raw_npz_root` 名字里如果带 `canonical`，会要求 `canonical_decomp16`
- legacy root 会拒绝 `canonical_decomp16`

问题：

raw 解码、root 命名、归一化模式互相约束。一个实验脚本里如果只想换 raw root，很可能被迫换 decode/norm。

建议：

- STF raw 只保留一个公开格式变量：`raw_storage_format`
- 当前本地 STF raw root 使用 `raw_storage_format=legacy_bggR_decomp16`
- `legacy_bggR_decomp16` 绑定完整输入解释，不再让脚本分别选择 channel/decompand/norm：
  - storage channel order: `[B, G, G, R]`
  - model channel order: `[R, Gr, Gb, B]`
  - required transform: channel reorder `[3,1,2,0]`
  - required decompand: STF LUT decompand to `[0,1]`
  - required post-decode norm: `passthrough`
- 预留一个未来格式占位，例如 `raw_future`，但当前不实现；如果用户选择该格式，应在启动阶段直接报错，避免静默走错 raw 解释路径
- 不依赖路径字符串推断 canonical/legacy
- 启动时输出 `raw_storage_format` 以及它展开后的 channel order / decompand / post-decode norm

### 2.6 bridge 默认 key 由 `input_type` 隐式推导

位置：

- `finetune_stf/train.py:761`
- `finetune_stf/models/lora_bridge.py:20`
- `finetune_stf/models/raw_feature_adapter.py:39`

当前默认：

- 4ch raw RAM bridge / feature adaptor：`x_cat,ffm_mid,x4`
- 3ch RAM RGB bridge / feature adaptor：`x_cat,ffm_mid,x3`

问题：

如果脚本不显式传 `--bridge-feature-keys`，切换 `input_type` 会自动改变 feature source。这样实验结果不再是单变量对照。

建议：

- 对所有 bridge / adaptor 实验强制显式传 `bridge_feature_keys`
- 在 config dump 里标记 key 来源：`explicit` 或 `default_from_input_type`

### 2.7 `raw_ram_rgb*` 不是普通 RGB 路径

位置：

- `finetune_stf/models/raw_ram.py:545`
- `finetune_stf/train.py:728`

问题：

`raw_ram_rgb` / `raw_ram_rgb_bridge` / `raw_ram_rgb_bridge_feature_adapter` 名字里有 `rgb`，但它不是普通 RGB DAv2 路径。它的实际流程是：

1. raw4 通过 `packed_bayer_to_base_rgb()` 转 base RGB
2. base RGB 进入 `RamCore3`
3. 可选 `raw_ram_rgb_tail`
4. 直接送 DAv2，且走 no-clamp / no-ImageNet-norm 路径

训练前还要求环境变量 `PHASE1_BNCLEAN_REVIEWED=1`。

建议：

- 命名上避免把它和 `rgb` baseline 混淆
- 可以改名为 `raw_to_base_rgb_ram3`
- 把 BN-clean/no-ImageNet norm 作为显式 resolved config 输出

### 2.8 KITTI eval 默认不是当前 raw 模型的 live eval

位置：

- `finetune_stf/train.py:1304`
- `finetune_stf/train.py:3092`

问题：

当前 KITTI eval dataset 在 `build_datasets()` 中固定用 `input_type="rgb"` 构造。后续 eval 默认协议还可能使用单独的 frozen RGB pretrained reference，而不是当前 raw model 的直接输出。

当前已有 `kitti_eval_protocol` 只有两个：

- `rgb_pretrained_ref`
- `rgb_checkpoint_decoder`

这意味着 raw 实验里的 KITTI 指标默认不是“当前 raw 模型在 KITTI 上的表现”，而是某种 RGB reference / decoder-sync protocol。`live_model` / raw live eval 目前不是现存 protocol，只能作为后续新增路径。

建议：

- 把 KITTI eval protocol 显式打印到日志
- 当前指标表先标注 `kitti_model_source=rgb_pretrained_ref|rgb_checkpoint_decoder`
- 如果要评估 raw 泛化，应单独实现明确的 raw-like KITTI eval 路径，例如新增 `kitti_eval_protocol=live_raw_model`，并在实现后再把它加入指标表枚举

### 2.9 HyperSim raw 使用 `vkitti_*` 变量

位置：

- `finetune_stf/train.py:1432`
- `foundation/engine/datasets/hypersim_processed_raw.py`

问题：

HyperSim raw dataset 复用了这些变量：

- `vkitti_randomize_unprocessing`
- `vkitti_unprocessing_preset`
- `vkitti_unprocessing_mix_weights`
- `vkitti_hflip_prob`

变量名显示是 VKITTI，但实际也影响 HyperSim。后续做 `raw_mix` 时容易误判。

建议：

- 改成通用变量：`synthetic_raw_randomize_unprocessing`
- 或拆成 `vkitti_*` 与 `hypersim_*`

### 2.10 train / viz 的输入选择逻辑不完全一致

位置：

- `finetune_stf/train.py:2506`
- `finetune_stf/util/viz_dump.py:234`

问题：

训练用 `RAW_MODEL_INPUT_TYPES` 判断是否优先取 `sample["raw"]`；可视化中 `_select_model_input()` 使用 `input_type != "rgb"` 且样本中存在 `raw` 就取 raw。

当前 `rgb_lora` 不属于已发生问题：它在 `train.py` 中属于 `RGB_INPUT_TYPES`，不属于 `RAW_MODEL_INPUT_TYPES`，因此 STF 训练/验证 dataset 走 `STF` RGB 路径，样本里通常没有 `raw` 键。`viz_dump.py` 里的 `"raw" in sample` 短路检查会让 `rgb_lora` 实际仍读取 `sample["image"]`。

真正风险是 latent bug：如果后续新增某个 `input_type`，命名上不是纯 `"rgb"`，但语义上仍使用 RGB dataset 或 RGB model input，只要 sample 中同时带有 `raw` 键，可视化就可能改读 `sample["raw"]`，而训练仍读 `sample["image"]`。这会造成 train / viz 输入不一致，而且容易被误判为“当前不影响运行所以不用修”。

建议：

- 可视化复用训练入口的 `RAW_MODEL_INPUT_TYPES`
- 不要用字符串 `input_type != "rgb"` 推断 raw 输入
- 在 resolved config 中明确记录 `model_input_tensor=image|raw`，并让 train / eval / viz 共用同一套输入选择函数

### 2.11 encoder 与 LoRA tap layers 隐式耦合

位置：

- `finetune_stf/train.py:907`
- `finetune_stf/train.py:935`

问题：

当前 RGB LoRA 和 `raw_ram_rgb_lora` 路径调用：

```python
apply_lora_to_vit(..., tap_layers=args.bridge_layers or DEFAULT_BRIDGE_LAYERS_BY_ENCODER[args.encoder], ...)
```

因此如果脚本没有显式传 `--bridge-layers`，切换 `encoder` 会同时改变 LoRA 注入位置：

- `vits` / `vitb`: `[2,5,8,11]`
- `vitl`: `[4,11,17,23]`
- `vitg`: `[9,19,29,39]`

这和 §2.6 里 `bridge_feature_keys` 默认由 `input_type` 推导是同类问题：看似只改 backbone size，实际也改了 LoRA tap layers。并且当前变量名仍叫 `bridge_layers`，但它也被 LoRA 路径复用，容易把 bridge 注入位置和 LoRA 注入位置混在一起。

建议：

- LoRA 实验必须显式传入 LoRA tap layers，不依赖 `DEFAULT_BRIDGE_LAYERS_BY_ENCODER`
- 拆分命名，例如 `bridge_layers` 与 `lora_tap_layers`
- 在 resolved config 中记录 `lora_tap_layers` 以及来源：`explicit` / `default_from_encoder`

## 3. 脚本层风险点

### 3.1 Feature adaptor 脚本不是单变量对照

相关脚本：

- `finetune_stf/scripts/0521_run_stf_feature_adapter_pair_186.sh`
- `finetune_stf/scripts/0522_run_stf_ram_feature_adapter_bridge_from_0521_1542_queue.sh`
- `finetune_stf/scripts/0522_run_stf_ram_rgb_feature_adapter_bridge_from_0521_1542_queue.sh`

风险：

`0521_run_stf_feature_adapter_pair_186.sh` 和 `0522_run_stf_ram_feature_adapter_bridge_from_0521_1542_queue.sh` 使用 `raw_ram_bridge_feature_adapter`，实际包含：

- 4ch raw input
- `RawRamCore`
- `x4` feature
- bridge
- decoder feature adaptor

这不是“只加 feature adaptor”。

`0522_run_stf_ram_rgb_feature_adapter_bridge_from_0521_1542_queue.sh` 使用 `raw_ram_rgb_bridge_feature_adapter`，更接近 3ch RAM 路径，但仍然包含 bridge，也不是纯 decoder feature adaptor。

因此，当前 STF 正式脚本里仍缺一个“只用 decoder feature adaptor、不带 bridge”的干净对照脚本。虽然旧 VKITTI/LOD 脚本中存在 `raw_ram_feature_adapter` 路径，但它不是当前 STF feature-adapter 对照实验的直接替代。这正是 §5.3 中“只加 decoder feature adaptor”路径缺失的具体实例。

### 3.2 旧 raw RAM bridge 脚本可能已不兼容当前尺寸检查

相关脚本：

- `finetune_stf/scripts/train_raw_ram_e3_bridge.sh`

风险：

脚本中 raw-like `stf_only` 使用 `518x966`，但当前 `train.py` 要求 raw-like STF 使用 `512x960`。

### 3.3 `PHASE1_BNCLEAN_REVIEWED` guard 需要在脚本里明确

涉及：

- `raw_ram_rgb`
- `raw_ram_rgb_lora`
- `raw_ram_rgb_bridge`
- `raw_ram_rgb_bridge_lora`
- `raw_ram_rgb_bridge_feature_adapter`

风险：

如果脚本里没有设置 `PHASE1_BNCLEAN_REVIEWED=1`，训练会直接失败。即便设置了，也应在实验名或日志中体现该路径用了 BN-clean/no-ImageNet-norm。

## 4. 变量清单

以下清单面向 `finetune_stf/train.py` 的命令行入口，按职责分类。后续修订时建议把这些变量从“一个大 `input_type` 派生一切”改成显式 resolved config。

### 4.1 实验身份与运行控制

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `encoder` | DAv2 encoder 规模 | 决定默认 bridge layers |
| `stage` | 训练 stage | 决定 dataset composition 和 loader schedule |
| `input_type` | 模型/输入总开关 | 最大耦合源 |
| `save_path` | 输出目录 | 正式实验命名需注意时间戳 |
| `heavy_save_root` | 大文件保存目录 | 影响 checkpoint/viz 等重文件位置 |
| `seed` | 随机种子 | 影响 loader、采样、viz |
| `port` | 分布式端口 | 多实验并发时需避免冲突 |

### 4.2 数据源与 split

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `stf_root` | STF RGB/root | 与 `stage`、`input_type` 联动 |
| `raw_npz_root` | STF raw npz root | 与 decode mode、norm mode 联动 |
| `vkitti_train_list` | VKITTI train list | 受 raw/RGB dataset 切换影响 |
| `hypersim_processed_base` | HyperSim processed base | raw_mix 时使用 |
| `hypersim_train_root` | HyperSim train root | raw_mix 时使用 |
| `hypersim_train_list` | HyperSim train list | raw_mix 时使用 |
| `hypersim_train_meta` | HyperSim meta | raw_mix 时使用 |
| `lod_root` | LOD root | RGB/raw dataset 由 `input_type` 决定 |
| `lod_day_manifest` | LOD day manifest | raw_mix/vkitti_lod 使用 |
| `lod_night_manifest` | LOD night manifest | raw_mix/vkitti_lod 使用 |
| `lod_day_max_samples` | LOD day 限制 | 改变数据混合比例 |
| `lod_night_max_samples` | LOD night 限制 | 改变数据混合比例 |
| `stf_pseudo_manifest` | STF pseudo target manifest | 与 target mode 联动 |

### 4.3 输入尺寸、raw 解码与归一化

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `input_height` | 输入高 | raw-like STF 被强制为 512 |
| `input_width` | 输入宽 | raw-like STF 被强制为 960 |
| `raw_storage_format` | 建议新增的 STF raw 存储格式变量 | 应替代脚本层手动组合 decode/norm/channel |
| `stf_raw_decode_mode` | STF raw 解码模式 | 强约束 `norm_mode` 和 root 类型 |
| `norm_mode` | raw 归一化模式 | 与 decode mode、dataset domain 强联动 |
| `channel_mode` | raw 转 3ch 的通道策略 | 影响 raw naive / eval raw RGB |
| `use_imagenet_norm` | 是否 ImageNet norm | raw_ram 路径通常绕过 |
| `lod_crop_mode` | LOD crop 策略 | train/eval crop 行为变化 |

### 4.4 raw domain 配置

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `lod_raw_domain_config` | LOD raw domain transform | 直接改变 raw 输入分布 |
| `eth3d_raw_domain_config` | ETH3D raw eval domain transform | eval 指标可变 |
| `robotcar_raw_domain_config` | RobotCar day raw domain transform | eval 指标可变 |
| `robotcar_night_raw_domain_config` | RobotCar night raw domain transform | eval 指标可变 |
| `eth3d_eval_norm_mode` | ETH3D eval norm | 与 train norm 不一定一致 |
| `robotcar_eval_norm_mode` | RobotCar day eval norm | 与 train norm 不一定一致 |
| `robotcar_night_eval_norm_mode` | RobotCar night eval norm | 与 train norm 不一定一致 |

### 4.5 模型前端与 RGB interface

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `dav2_train_mode` | DAv2 训练范围 | 与 LoRA/bridge 参数组联动 |
| `rgb_interface_mode` | 4ch raw 到 3ch interface | 只影响部分 raw RAM 路径 |
| `rgb_residual_scale` | residual interface scale | 与 interface mode 绑定 |
| `raw_ram_rgb_tail` | 3ch RAM tail | 只影响 `raw_ram_rgb*` |
| `backbone_layer_decay` | backbone layer-wise LR decay | 只在 full train mode 生效 |

### 4.6 Bridge / Feature adaptor / LoRA

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `bridge_source` | bridge feature 来源 | 当前多数路径默认 raw RAM |
| `bridge_feature_keys` | bridge 使用的 feature keys | 默认由 `input_type` 推导 |
| `feature_adapter_keys` | decoder feature adaptor 使用的 feature keys | 当前部分路径与 bridge keys/source 混用 |
| `bridge_layers` | bridge 注入 blocks，也被部分 LoRA 路径当 tap layers 使用 | 默认由 `encoder` 推导，改 encoder 会联带改变注入位置 |
| `bridge_lr` | bridge/adaptor 参数 LR | 不是纯 bridge-only；`raw_ram_feature_adapter` 这类无 bridge 路径也会用它训练 `feature_projector` / `merge*` 等 adapter 参数，需输出 param group 计数避免 dead 配置 |
| `lora_rank` | LoRA rank | 只在 LoRA input_type 生效 |
| `lora_alpha` | LoRA alpha | 只在 LoRA input_type 生效 |
| `lora_lr` | LoRA LR | 与参数名匹配和 input_type 相关 |
| `lora_block_mode` | LoRA 插入 block 范围 | 与 bridge/tap 实验联动 |

### 4.7 训练采样与数据混合

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `epochs` | epoch 数 | 与 stage schedule 联动 |
| `bs` | batch size | DDP 每卡 batch |
| `accum_steps` | 梯度累积 | 改变有效 batch |
| `lr` | 主学习率 | 与参数分组联动 |
| `num_workers` | dataloader workers | 性能变量 |
| `stf_repeat` | STF repeat | 改变 stage 数据权重 |
| `lod_per_vkitti` | vkitti_lod 中 LOD 数量 | 与 `lod_fraction` 联动 |
| `lod_fraction` | LOD fraction | 会重写 vkitti_lod schedule |
| `train_sources` | raw_mix sources | 与 ratios 必须匹配 |
| `train_source_ratios` | raw_mix ratios | 直接改变混合数据分布 |
| `train_steps_per_epoch` | raw_mix epoch 长度 | 改变每轮采样次数 |

### 4.8 loss 与 target

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `loss_type` | loss 类型 | 决定其他 loss 参数是否生效 |
| `loss_lambda_grad` | gradient loss 权重 | 只在 `ssi_grad` 生效 |
| `loss_grad_scales` | gradient scales | 只在 `ssi_grad` 生效 |
| `loss_mask_downsample` | mask downsample | 影响 loss mask |
| `loss_target_normalization` | target normalization | `aligned_sig` 不使用 |
| `loss_norm_min_scale` | normalization min scale | 只在部分 loss 生效 |
| `stf_train_target_mode` | STF train target | 与 pseudo/GT target 语义联动 |

### 4.9 VKITTI / HyperSim unprocessing

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `vkitti_randomize_unprocessing` | 是否随机 unprocessing | 同时影响 VKITTI raw 和 HyperSim raw |
| `vkitti_unprocessing_preset` | unprocessing preset | 名字是 VKITTI，但可影响 HyperSim |
| `vkitti_unprocessing_mix_weights` | preset mix weights | 改变合成 raw sensor 分布 |
| `vkitti_hflip_prob` | VKITTI/HyperSim raw hflip | 影响 raw dataset 增广 |
| `vkitti_cache_root` | VKITTI cache | 只允许 raw-like input |

### 4.10 eval、checkpoint 与可视化

| 变量 | 作用 | 联动风险 |
| --- | --- | --- |
| `eval_only` | 只评估 | 与 checkpoint 加载联动 |
| `eval_kitti` | 是否 eval KITTI | 默认可能不是 live raw 模型 |
| `eval_eth3d` | 是否 eval ETH3D | RGB/raw 路径由 input_type 决定 |
| `eval_robotcar` | 是否 eval RobotCar day | RGB/raw 路径由 input_type 决定 |
| `eval_robotcar_night` | 是否 eval RobotCar night | RGB/raw 路径由 input_type 决定 |
| `kitti_eval_protocol` | KITTI eval 协议 | 直接决定模型来源 |
| `best_metric` | best checkpoint 指标 | 可能被 eval protocol 影响 |
| `save_best_checkpoint` | 是否保存 best ckpt | 与 best metric 联动 |
| `eval_interval` | eval 间隔 | 影响 checkpoint 选择 |
| `debug_max_*` | debug sample 限制 | 影响 eval/训练有效样本 |
| `enable_fixed_viz_dump` | fixed viz dump | 与 raw/RGB 输入选择联动 |
| `fixed_viz_splits` | fixed viz split | 可混合多个 eval source |
| `fixed_viz_lod_n` | fixed LOD viz 数量 | 影响可视化对比 |
| `enable_train_source_viz_dump` | train source viz | 可能自动创建 RGB baseline |
| `train_viz_sources` | train viz sources | 与 raw_mix source 名称相关 |
| `train_viz_rgb_baseline` | 是否加 RGB baseline | 改变可视化输出语义 |
| `train_viz_rgb_checkpoint` | RGB baseline ckpt | baseline 来源变量 |

## 5. 建议的修订路线

### 5.1 第一步：增加 resolved config dump

在真正大改前，先在启动时输出完整 resolved config，包括：

- `input_domain`
- `dataset_class`
- `dataset_input_mode`
- `model_class`
- `model_input_tensor`
- `raw_storage_format`
- `raw_storage_channel_order`
- `raw_decompand`
- `raw_post_decode_norm`
- `raw_channel_count`
- `ram_core_type`
- `imagenet_norm_enabled`
- `bridge_enabled`
- `decoder_feature_adapter_enabled`
- `bridge_feature_keys`
- `bridge_layers`
- `feature_key_source`: explicit/default
- `lora_tap_layers`
- `lora_tap_layers_source`: explicit/default_from_encoder
- `optimizer_param_groups`: group name / lr / trainable parameter count
- `kitti_model_source`
- `eval_input_domain`

这样即使暂时不重构，也能先避免“不知情地改了别的变量”。

### 5.2 第二步：把 `input_type` 拆成正交变量

建议新结构：

```text
input_domain = rgb | raw4
front_end = dav2_rgb | raw_to_rgb_head | raw_ram4 | raw_to_base_rgb_ram3
bridge = none | raw_feature_bridge
decoder_feature_adapter = none | raw_feature_adapter
lora = none | dav2_lora
bridge_feature_source_channels = none | x3 | x4
adapter_feature_source_channels = none | x3 | x4
```

这里 `raw_packed` 应展开为 `input_domain=raw4` + `front_end=raw_to_rgb_head`，避免把“输入数据域”和“raw 到 DAv2 的前端实现”混成同一个变量。

旧的 `input_type` 可以保留为兼容 alias，但必须在日志里打印 alias 展开后的真实配置。

### 5.3 第三步：补齐干净对照路径

为了支持单变量实验，至少需要这些组合：

| 实验目的 | 应有路径 |
| --- | --- |
| 只测 3ch baseline | RGB / base RGB path，无 raw RAM |
| 只加 3ch RAM front-end | raw4 -> base RGB -> RamCore3 |
| 只加 4ch RAM front-end | raw4 -> RawRamCore -> RGB interface |
| 只加 bridge | 固定 front-end，只开 bridge |
| 只加 decoder feature adaptor | 固定 front-end，只开 decoder feature adaptor |
| bridge + feature adaptor | 显式组合，而不是由 input_type 暗含 |
| 只加 LoRA | 固定其他路径，只开 LoRA |

### 5.4 第四步：脚本层强制显式关键变量

建议所有正式实验脚本必须显式写出：

- `--input-type`
- `--input-height`
- `--input-width`
- `--stf-raw-decode-mode`
- `--norm-mode`
- `--bridge-feature-keys`
- `--bridge-layers`
- `--dav2-train-mode`
- `--kitti-eval-protocol`

并在脚本注释中写明该实验实际改变的维度。

## 6. 优先修订清单

建议按下面顺序修：

1. 修 `viz_dump.py` 输入选择逻辑，和训练里的 `RAW_MODEL_INPUT_TYPES` 对齐。
2. 给 `train.py` 增加 resolved config dump。
3. 禁止 raw 域解释类配置依赖危险默认值，尤其是 `stf_raw_decode_mode=legacy_companded` 这类会改变通道/decompand/norm 语义的默认值；正式实验必须由脚本显式传入 `raw_storage_format` 或等价 resolved 配置。
4. 把 feature adaptor 从 `input_type` 中拆成独立开关。
5. 明确区分 3ch feature adaptor 和 4ch feature adaptor。
6. 把 `bridge_feature_keys` 默认来源打印出来，正式实验强制显式传入。
7. 把 LoRA tap layers 从 `bridge_layers` 中拆出或至少显式打印来源，避免改 `encoder` 时隐式改变 LoRA 注入位置。
8. 把 HyperSim 使用的 `vkitti_*` unprocessing 变量重命名或拆分。
9. 清理或标注过时脚本，尤其是 raw-like STF 使用 `518x966` 的脚本。
10. 在 KITTI eval 输出中标注当前已有的 `rgb_pretrained_ref` / `rgb_checkpoint_decoder`；如果后续实现 raw live eval，再新增类似 `live_raw_model` 的明确 protocol/source。

## 7. 新项目迁移补充原则

### 7.1 迁移目标

后续计划迁移一份新的项目/计划，在保持当前任务不动的前提下，用更干净的代码重新组织训练入口、配置解析、数据集和模型 wrapper。当前项目可以继续作为现有实验记录和兼容参考，新项目优先服务后续可控对照实验。

### 7.2 正式实验不依赖隐式默认值

新项目中，正式实验参数原则上不设置隐式默认值。所有会影响实验语义的变量都应由启动脚本显式传入，目标是降低“没有意识到某个默认值改变了实验变量”的风险。

需要区分两类参数：

- 活跃路径参数：当前实验实际使用到的参数，必须在 `.sh` 文件中显式传入。
- 非活跃路径参数：当前实验不会使用到的参数，不能静默进入实验记录；应由启动校验要求其不传，或显式设置为 `none` / `n_a` 这类无效占位，并在 resolved config 中显示为 not applicable。

这种策略是否完全能避免连带变量还需要实现时验证，但它至少可以把隐式默认值暴露为显式实验配置或显式不适用项。

### 7.3 移除不再需要的 mix 训练路径

后续训练不再需要 mix 数据训练。新项目中可以直接移除 raw mix / VKITTI mix / HyperSim mix / LOD mix 等仅服务混合训练的代码路径、配置项和脚本逻辑。保留当前项目作为旧实验兼容，不在迁移项目里继续维护这些分支。

### 7.4 必要联动应改成显式适用性校验

“避免连带变量”不是禁止所有条件约束。必要的联动应该保留，但必须变成显式校验，而不是靠默认值或字符串推断。

例如 `lod_crop_mode` 只在 LOD train/eval 启用时有意义：

- 如果启用了 LOD train/eval，则必须显式传入 `lod_crop_mode`。
- 如果没有启用任何 LOD train/eval，则 `lod_crop_mode` 必须不传，或必须为 `none` / `n_a`。
- 如果无 LOD 路径但传入了有效 crop mode，应在启动阶段报错，避免污染实验记录。

同理，LoRA、bridge、feature adapter、eval protocol、dataset-specific crop / norm / decode 等参数都应有适用性校验：只有对应功能开启时才允许传入其子参数；功能关闭时传入子参数应报错或必须显式为 `none` / `n_a`。

### 7.5 loss 相关参数也需要适用性校验

loss 配置应避免把未使用的 loss 分支写进实验记录。例如当主 loss 是 `ssi` 且不使用 grad/edge loss 时，grad 相关参数不应以默认值形式进入 resolved config。

建议规则：

- 如果 `loss=ssi` 且未启用 grad/edge loss，则 grad/edge 相关参数必须不传，或必须显式为 `none` / `0` / `n_a`，并在 resolved config 中显示为 not applicable。
- 如果启用了 grad/edge loss，则必须显式传入 grad/edge loss type、权重、作用层级或 mask 规则。
- 如果 loss 组合不需要某个子参数但脚本传入了有效值，应在启动阶段报错。

目标是让实验记录只包含真正参与训练目标的 loss 项，避免未启用分支的默认参数看起来像实验变量。
