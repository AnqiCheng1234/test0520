# 0525 RAW-Adapter-style / InvISP 方法接入当前在线训练执行计划

## 0. 结论和边界

结论：`plans/0524_unprocessing` 中的 RAW-Adapter-style / InvISP 路线不能直接热切换到已经启动或已经完成的当前训练中。它可以作为一条新的在线 unprocessing 方法接入下一轮训练，但必须新增显式方法分支，不能复用当前 `sensor_linear_dual` / Brooks-style preset 的实验语义。

当前 raw 训练入口已经在线生成 pseudo-RAW：`VKITTI2Raw.__getitem__` 在读取 RGB/depth、完成几何处理后调用 `foundation.engine.transforms.UnprocessingTransform` 返回 `raw`。要接入本计划方法，目标不是“把离线脚本直接塞进训练”，而是把 RAW-Adapter-style 的核心步骤迁移为 PyTorch transform，并让训练、VKITTI val、KITTI val 使用同一套语义。

本计划分两层：

1. 第一阶段只接入 `analytic` 在线版本：可训练、可 smoke、可正式对照。
2. 第二阶段再接入外部 InvISP 输出：以离线 raw-RGB cache 为主，不建议把 InvISP 模型直接放进 dataloader 在线推理。

## 1. 当前代码事实

离线计划方法：

- 文件：`plans/0524_unprocessing/unprocess_rgb_to_packed_raw.py`
- backend：
  - `analytic`: sRGB inverse gamma -> optional inverse tone -> optional CCM -> inverse WB -> exposure/noise -> RGGB pack
  - `external_npy`: 读取外部 InvISP / learned unprocessing 已经输出的 raw-RGB `.npy/.npz`
- 输出：`raw_packed [4,H/2,W/2]`，固定 RGGB，channel order `[R,G1,G2,B]`
- 它是 NumPy + PIL 离线脚本，保存 `.npz/.json/preview`。

当前训练方法：

- Dataset：`foundation/engine/datasets/vkitti2_raw.py`
- Transform：`foundation/engine/transforms/unprocessing.py`
- 训练入口：`foundation/tools/train_vkitti2_raw_residual.py`
- 当前 raw 实验参数：
  - `input_domain=raw4`
  - `model_input_tensor=raw`
  - `front_end=raw_to_base_rgb_ram3`
  - `raw_storage_format=synthetic_packed_bayer_4ch_halfres`
  - `vkitti_unprocessing_preset=sensor_linear_dual`
  - `randomize_unprocessing=True`
- KITTI eval 当前从训练 config 读取 `vkitti_unprocessing_preset`，并用同一个旧 transform 的 `randomize=False` 版本生成 eval raw。

关键差异：

- RAW-Adapter-style 方法固定 RGGB；旧方法支持 RGGB/BGGR/GRBG/GBRG 并输出 canonical `[R,Gr,Gb,B]`。
- RAW-Adapter-style 方法有 `normal/dark/over` 显式 variant；旧方法用 preset 的 exposure/noise range 隐式采样。
- tone curve 不同，不能混写。旧 `UnprocessingTransform.forward()` 当前无条件执行 Brooks-style `inverse_smoothstep`：
  `0.5 - sin(asin(1 - 2*x) / 3)`。RAW-Adapter-style 离线脚本的 tone 选项是 `inverse_global_tone(strength=0.15)`：
  `x / max(1 - 0.15 * (1 - x), 1e-6)`。本地 2x2 ablation 中 `tone_only` / `ccm_tone` 传了 `--inverse-tone`，metadata 为 `inverse_tone=true`；`baseline` / `ccm_only` 没传，metadata 为 `inverse_tone=false`。执行文档把 `ccm_tone` 标为最接近 RAW-Adapter 完整解析近似路线，因此在线计划主线应使用 `raw_adapter_inverse_tone=global_0p15`，`none` 只作为 no-tone ablation。`--raw-adapter-inverse-tone none` 不等价于旧方法“关 tone”，因为旧方法当前没有关 tone 分支。
- RAW-Adapter-style `external_npy` 不是在线 InvISP，而是外部 raw-RGB 产物读取接口。
- 两者都能产出 4ch packed tensor，但实验语义不同，必须显式区分。

## 2. 目标实验语义

新增统一语义参数，所有会改变实验含义的参数必须在训练脚本和正式 launch 脚本里显式传入。

一致性约束：

- 训练 VKITTI、验证 VKITTI、验证 KITTI 必须从同一个 resolved unprocessing config 构造 transform。
- `config.json` 是 KITTI offline eval 的唯一参数来源；KITTI eval 不允许重新指定、推断或覆盖 raw-adapter / InvISP 相关参数。
- `raw_adapter_style` 的 resolved config 必须在训练入口只构造一次，然后原样传给 VKITTI train、VKITTI val 和 KITTI val builder；不得在不同 split 上各自走本地默认值。
- 如果训练使用 `unprocessing_method=raw_adapter_style`，则 VKITTI train、VKITTI val、KITTI val 都必须记录并使用同一组 method-level 参数：backend、CFA、packed channel order、RGB transfer、inverse tone、CCM、WB gain range、variant policy、light scale range、noise model、black/white level。
- `raw_adapter_inverse_tone` 必须作为 resolved config 的一部分传递。训练、VKITTI val、KITTI val 对同一次实验必须使用同一个值：`none` 就全部不做 tone 反变换，`global_0p15` 就全部使用离线脚本同一公式。任何一个 split 都不得回退到旧 `inverse_smoothstep`。
- 第一版 raw-adapter 主线默认按本地 `tone_only` / `ccm_tone` 口径使用 `raw_adapter_inverse_tone=global_0p15`。`none` 保留给明确命名的 no-tone ablation，不作为主线默认。
- 当前 `raw_adapter_style` / InvISP 路线默认必须使用 `randomize_unprocessing=false`，正式脚本和 smoke 命令都要显式传 `--no-randomize-unprocessing`。
- `raw_adapter_style` 默认要求 VKITTI train、VKITTI val、KITTI val 使用完全相同的固定 gain/noise/light/variant 参数。第一版不要用 train 随机采样、eval canonical 的模式。
- `train_dataset.describe_unprocessing()`、`val_dataset.describe_unprocessing()`、`KittiHalfresRawDataset.describe_unprocessing()` 必须包含完全相同的 `raw_adapter_config_hash`。hash 不一致时直接 raise，不允许只打印 warning。
- `raw_adapter_config_hash` 只覆盖 unprocessing 实验语义字段：method、backend、CFA、channel order、RGB transfer、tone、CCM、fixed gain、variant、light scale、noise policy、black/white level、randomize policy、external cache 语义参数。它不得包含 split 名、filelist 路径、batch size、worker 数、日志路径、checkpoint 路径或实验名。
- hash 相同仍要在 summary 里记录关键 fixed 字段的明文值，至少包括 `raw_adapter_inverse_tone`、`raw_adapter_ccm`、`fixed_red_gain`、`fixed_blue_gain`、`variant_policy`、`fixed_light_scale`、`noise_model`、`noise_realization_applied`、`shot_noise`、`read_noise`、`noise_mean_mode`。这样可以人工确认训练和验证确实同参。
- 只有当实验明确要研究 RAW-Adapter 随机增强时，才允许显式打开 `--randomize-unprocessing`，并且实验名、config、eval summary 必须写明 train/eval 不再是同一组具体 unprocessing 参数。

建议新增参数：

```text
--unprocessing-method old_brooks_preset | raw_adapter_style

# old_brooks_preset 分支保留现状
--vkitti-unprocessing-preset sensor_linear_dual
--vkitti-unprocessing-mix-weights ...
--randomize-unprocessing / --no-randomize-unprocessing

# raw_adapter_style 分支必须显式给出
--raw-adapter-backend analytic | external_raw_rgb_cache
--raw-adapter-cfa-pattern RGGB
--raw-adapter-packed-channel-order R_Gr_Gb_B
--raw-adapter-rgb-transfer srgb_piecewise
--raw-adapter-inverse-tone none | global_0p15
--raw-adapter-ccm identity | generic_d65
--raw-adapter-red-gain-range 1.9 2.4
--raw-adapter-blue-gain-range 1.5 1.9
--raw-adapter-fixed-red-gain 2.15
--raw-adapter-fixed-blue-gain 1.70
--raw-adapter-variant-policy normal | dark | over | mix
--raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0
--raw-adapter-fixed-light-scale 1.0
--raw-adapter-dark-light-scale-range 0.05 0.4
--raw-adapter-over-light-scale-range 1.5 2.5
--raw-adapter-shot-noise 0.001
--raw-adapter-read-noise 0.0005
--raw-adapter-noise-mean-mode zero | rawadapter_text
--raw-adapter-black-level 0.0
--raw-adapter-white-level 1.0
--raw-adapter-random-seed-policy dataloader_generator
```

中央校验规则：

- 采用方案 A：把 `--vkitti-unprocessing-preset` 从当前 `required=True` 改成 `default="not_applicable"`，并在 `validate_args()` 中按 `unprocessing_method` 做互斥校验。不要采用方案 B，不要在 preset registry 里注册 `"not_applicable"` 占位 entry；preset registry 只应包含能真实构造旧 `UnprocessingTransform` 的 preset，避免未来误调用 `build_unprocessing_transform_from_preset("not_applicable")`。
- `unprocessing_method=old_brooks_preset` 时，raw-adapter 参数必须是 `not_applicable` 或不传；传 active 值应报错。
- `unprocessing_method=old_brooks_preset` 时，`vkitti_unprocessing_preset` 必须是 active preset，例如 `sensor_linear_dual`；传 `not_applicable` 应报错。
- `unprocessing_method=raw_adapter_style` 时，`vkitti_unprocessing_preset` 必须是 `not_applicable`，并且 `vkitti_unprocessing_mix_weights` 必须是 `None` / `not_applicable`。不能默默沿用 `sensor_linear_dual`。
- `vkitti_unprocessing_mix_weights` 和 `raw_adapter_variant_weights` 是两个不同维度的 mix，必须互斥使用：前者只属于 `old_brooks_preset` 的 dual preset/sub-preset 混合，后者只属于 `raw_adapter_style` 的 exposure variant 混合。formal launch 脚本里不得同时设置两者，避免把 preset mix 误传成 normal/dark/over mix。
- `unprocessing_method=raw_adapter_style` 时，`randomize_unprocessing` 默认必须解析为 `False`。正式 launch 若未显式传 `--no-randomize-unprocessing`，应在脚本层报错，避免旧 raw 实验的 `--randomize-unprocessing` 习惯被带入。
- `raw_adapter_backend=analytic` 时，不能传 external cache 路径。
- `raw_adapter_backend=external_raw_rgb_cache` 时，必须显式传 cache root，并校验 cache 的 RGB source path、shape、split 与当前 filelist 一致。
- `raw_adapter_random_seed_policy` 第一版仅允许 `dataloader_generator`。`path_hash` 尚未实现；如果 CLI/parser 暂时保留该 choice，`validate_args()` 必须显式报错，提示 `path_hash` is not implemented yet。不要让 launch 脚本作者误以为 path-hash sampling 已可用。
- `raw_adapter_style` 且 `randomize_unprocessing=False` 时，必须使用显式 fixed 参数：`fixed_red_gain`、`fixed_blue_gain`、`variant_policy`、`fixed_light_scale`、固定 noise policy。train 和 val 必须一致。
- `randomize_unprocessing=False` 时，fixed/range 共存的规则必须写进中央校验：
  - transform 只使用 `fixed_red_gain`、`fixed_blue_gain`、`fixed_light_scale` 和固定 `variant_policy`；range 字段只写入 config/metadata 作为记录和边界校验依据，不参与采样。
  - `fixed_red_gain` 必须落在 `raw_adapter_red_gain_range` 内，否则报错。
  - `fixed_blue_gain` 必须落在 `raw_adapter_blue_gain_range` 内，否则报错。
  - `variant_policy=normal` 时，`fixed_light_scale` 语义上必须等于 `1.0`，不能使用 dark/over range。
  - `variant_policy=dark` 时，`fixed_light_scale` 必须落在 `raw_adapter_dark_light_scale_range` 内。
  - `variant_policy=over` 时，`fixed_light_scale` 必须落在 `raw_adapter_over_light_scale_range` 内。
  - `variant_policy=mix` 不允许和 `randomize_unprocessing=False` 同时启用；固定第一版只能是 `normal`、`dark` 或 `over` 之一。
  - `raw_adapter_variant_weights` 在固定模式下只允许作为记录字段；如果传入，必须是与固定 `variant_policy` 一致的 one-hot 权重，否则报错。
- noise 采样开关与 `randomize_unprocessing` 耦合：
  - `randomize_unprocessing=False` 时，强制 `noise_model="none"`，`shot_noise/read_noise` 仍写入 config 作为实验记录但不采样 noise realization，metadata 必须写 `noise_realization_applied=false`。
  - `randomize_unprocessing=True` 时，`noise_model="raw_adapter_gaussian_signal_dependent"`，按 `shot_noise/read_noise` 采样。
  - 如果未来要支持 `randomize_unprocessing=False + fixed deterministic noise`，必须新增显式参数，不得复用当前 `shot_noise/read_noise` 字段隐式启用。
- `raw_adapter_style` 必须从完整 raw-adapter 语义参数生成 `raw_adapter_config_hash`，写入 transform metadata、Dataset `describe_unprocessing()` 和训练 `config.json`。hash 规则建议沿用旧 preset 思路：对规范化 JSON payload 做 SHA256，取前 16 个字符。payload 至少包含 backend、CFA、channel order、RGB transfer、tone、CCM、gain、variant、light scale、noise model、noise realization policy、black/white level、randomize policy、external cache 语义参数；不应包含临时路径、日志路径或实验名。

## 3. Phase A: 新增 PyTorch RAW-Adapter-style transform

新增文件建议：

```text
foundation/engine/transforms/raw_adapter_style_unprocessing.py
```

新增类：

```text
RawAdapterStyleUnprocessingTransform(torch.nn.Module)
```

输入输出：

```text
input:  image tensor [3,H,W] or [N,3,H,W], float, RGB, [0,1]
output: raw tensor [4,H/2,W/2] or [N,4,H/2,W/2], float32
metadata: dict
```

必须实现的 analytic pipeline：

```text
RGB [0,1]
-> exact sRGB piecewise inverse gamma
-> raw-adapter tone branch:
   none:        identity, no tone inverse
   global_0p15: inverse_global_tone(strength=0.15), same formula as offline script
-> CCM identity/generic_d65
-> inverse WB: R /= red_gain, B /= blue_gain
-> variant exposure:
   normal: light_scale=1.0
   dark:   Uniform(0.05,0.4)
   over:   Uniform(1.5,2.5)
-> noise branch:
   none: no noise realization
   raw_adapter_gaussian_signal_dependent:
     variance = read_noise^2 + shot_noise * signal
-> black/white level mapping
-> crop bottom/right to even H/W
-> fixed RGGB pack to canonical [R,Gr,Gb,B]
```

禁止事项：

- `RawAdapterStyleUnprocessingTransform` 不得调用旧 `UnprocessingTransform.inverse_smoothstep()`。
- 不要把 `global_0p15` 实现成 Brooks inverse smoothstep；它必须复刻离线脚本的 `x / max(1 - 0.15 * (1 - x), 1e-6)`。
- 不要把 `raw_adapter_inverse_tone=none` 解释成旧方法 tone 关闭。它只属于 raw-adapter-style 方法，旧方法仍按旧 transform 语义无条件使用 inverse smoothstep。

注意 channel order：脚本写作 `[R,G1,G2,B]`。训练模型的 `packed_bayer_to_base_rgb()` 只要求第 1/2 通道是两个 green plane。因此在线 transform metadata 中写：

```text
cfa_pattern=RGGB
packed_channel_order=[R,Gr,Gb,B]
raw_adapter_original_order=[R,G1,G2,B]
```

这里的 `[R,G1,G2,B]` 与 `[R,Gr,Gb,B]` 数值等价只在 `cfa_pattern=RGGB` 时成立：RGGB 下 `G1=raw_rgb[0::2,1::2,1]` 对应 `Gr`，`G2=raw_rgb[1::2,0::2,1]` 对应 `Gb`。如果第二阶段允许 RGGB 以外的 CFA，不能沿用这个等价注释，必须重新对照 `PATTERN_TO_OFFSETS` 定义每个 packed channel 的空间 offset 和 green plane 语义。

随机性：

- 第一版 `raw_adapter_style` 默认不随机采样，不依赖 `torch.Generator` 来决定 gain/light/noise。
- transform 仍可支持 `torch.Generator`，但只在显式 `--randomize-unprocessing` 的后续随机增强实验中启用。
- 第一版仅实现 `random_seed_policy=dataloader_generator`。传入 `path_hash` 必须在 `validate_args()` 报错，提示尚未实现。
- 如果后续使用 `path_hash`，Dataset 需要根据 `seed + image_path + epoch/local_variant` 生成 per-sample generator；这会改变当前训练随机语义，必须作为显式实验参数。

`randomize_unprocessing=False` 行为定义：

```text
cfa_pattern    = raw_adapter_cfa_pattern (Phase A: RGGB)
rgb2cam        = identity 或 generic_d65 矩阵，由 raw_adapter_ccm 决定，不在 library 中随机加权
red_gain       = raw_adapter_fixed_red_gain
blue_gain      = raw_adapter_fixed_blue_gain
inverse_tone   = raw_adapter_inverse_tone 指定的 none / global_0p15
variant        = raw_adapter_variant_policy，必须是 normal/dark/over，不允许 mix
light_scale    = raw_adapter_fixed_light_scale
noise_model    = none
noise          = none，不采样
black/white    = raw_adapter_black_level / raw_adapter_white_level
metadata.randomize = false
metadata.noise_realization_applied = false
```

metadata schema 必须兼容旧下游。旧 Dataset 会向 `isp_params` 注入 `isp_profile_name`、`isp_profile_group`、`selected_sub_preset_name`、`preset_version`、`preset_hash`、`preset_mix_weights`、`selected_sub_preset_hash` 等字段；KITTI eval、可视化和 checkpoint dump 可能直接读取这些 key。`raw_adapter_style` 不能只返回一套全新的 key，而应该同时提供兼容字段和 raw-adapter 专属字段。

`raw_adapter_style` metadata 必须包含：

```text
unprocessing_method

# old-schema compatibility fields
isp_profile_name                 # "raw_adapter_style"
isp_profile_group                # "raw_adapter_style"
selected_sub_preset_name          # "raw_adapter_style"
preset_version                   # e.g. "2026-05-25.raw_adapter_style_v1"
preset_hash                      # compatibility alias of raw_adapter_config_hash
preset_mix_weights               # [{"name": "raw_adapter_style", "weight": 1.0}]
selected_sub_preset_hash          # compatibility alias of raw_adapter_config_hash
packed_channel_order             # [R,Gr,Gb,B]
cfa_pattern                      # RGGB
noise_model                      # raw_adapter_gaussian_signal_dependent | none
noise_realization_applied        # false in randomize_unprocessing=False Phase A
randomize

# raw-adapter-specific fields
raw_adapter_backend
raw_adapter_config_hash
raw_adapter_original_order        # [R,G1,G2,B], valid-equivalent to [R,Gr,Gb,B] only under RGGB
rgb_transfer
inverse_tone
ccm
red_gain
blue_gain
variant
variant_policy
variant_weights
light_scale
shot_noise
read_noise
noise_mean_mode
black_level
white_level
random_seed_policy
external_raw_rgb_root             # "not_applicable" for analytic backend
external_raw_rgb_key              # "not_applicable" for analytic backend
external_cache_space              # "not_applicable" for analytic backend
```

注意：`preset_hash` / `selected_sub_preset_hash` 在 `raw_adapter_style` 下只是 `raw_adapter_config_hash` 的兼容别名，仅用于不破坏旧 metadata schema。下游若按 `preset_hash` 分组，或用它推断 sensor 来源 / preset 名称，必须先按 `isp_profile_group` / `unprocessing_method` 分流，再在各自方法内部做分组。

`noise_model` 取值规则：

- no random noise realization: `none`
- RAW-Adapter-style signal-dependent Gaussian: `raw_adapter_gaussian_signal_dependent`
- legacy old transform: `poisson_gaussian`

旧 `old_brooks_preset` 分支也应在 `describe_unprocessing()` / config 层补 raw-adapter 占位字段，值统一为 `"not_applicable"` 或空结构，便于 downstream 统一读字段：

```text
unprocessing_method=old_brooks_preset
raw_adapter_backend=not_applicable
raw_adapter_config_hash=not_applicable
raw_adapter_original_order=not_applicable
raw_adapter_variant_weights={}
raw_adapter_external_raw_rgb_root=not_applicable
```

不要把这些占位字段传进旧 `UnprocessingTransform` 构造函数；它们只用于 config/metadata/schema 稳定。

## 4. Phase B: 接入 VKITTI2Raw Dataset

修改文件：

```text
foundation/engine/datasets/vkitti2_raw.py
```

改动点：

1. `VKITTI2Raw.__init__` 新增 `unprocessing_method` 和 raw-adapter 参数。
2. 保留旧 `old_brooks_preset` 行为，确保现有 M-series 不受影响。
3. `unprocessing_method=old_brooks_preset` 时，才允许调用 `get_unprocessing_preset()`、`resolve_unprocessing_mix_weights()`、`build_unprocessing_transform_from_preset()`。
4. `unprocessing_method=raw_adapter_style` 时，不得调用 `get_unprocessing_preset("not_applicable")`。应直接构造 `RawAdapterStyleUnprocessingTransform`，并把旧 preset 相关字段写成 `not_applicable`。
5. 当 `raw_storage_format=synthetic_packed_bayer_4ch_halfres` 时，仍按当前流程先做 fullres even crop / hflip，再调用 unprocessing。
6. 当 `raw_storage_format=synthetic_packed_bayer_4ch` 时，仍先 resize/crop/flip，再调用 unprocessing。
7. `describe_unprocessing()` 根据方法返回不同 schema，不能把 raw-adapter 方法伪装成 preset。

目标调用位置保持不变：

```text
image_tensor = RGB after geometry
raw_tensor, isp_params = unprocessing_transform(image_tensor, generator=torch_generator)
```

半分辨率几何原则保持：

- 几何增强必须在 RGB/depth 阶段完成。
- pack 之前 fullres H/W 必须为偶数。
- hflip 后直接固定 RGGB pack 会改变真实 CFA 起点语义。当前旧方法也没有按 flip 调整 CFA。第一版为了和当前工程约束一致，可以保持固定 RGGB，但必须在 metadata 里记录 `hflip_applied`，并在计划/实验说明中承认这是 synthetic RAW-like，不是物理真实 CFA 对齐。

## 5. Phase C: 接入训练入口和配置保存

修改文件：

```text
foundation/tools/train_vkitti2_raw_residual.py
```

改动点：

1. argparse 新增 `--unprocessing-method` 和 raw-adapter 参数。
2. 把当前 `--vkitti-unprocessing-preset required=True` 改成 `default="not_applicable"`。这是为了让 raw-adapter 分支可以显式声明旧 preset 不适用；不能通过 registry 占位解决。
3. `validate_args()` 中做中央语义校验：
   - `old_brooks_preset`: 要求 active `vkitti_unprocessing_preset`，拒绝 active raw-adapter 参数。
   - `raw_adapter_style`: 要求 `vkitti_unprocessing_preset == "not_applicable"`，拒绝 active `vkitti_unprocessing_mix_weights`，要求 raw-adapter 必填参数齐全。
4. `build_loaders()` 把方法和参数传给 train/val Dataset。它不能无条件假设 `vkitti_unprocessing_preset` 可用于旧 preset 构造。
5. `config.json` 保存完整 raw-adapter 参数和 `raw_adapter_config_hash`。
   `config.json` 必须至少包含以下顶层 key，作为 eval dispatch 的唯一来源，避免未来重构 argparse 字段名时无意改变评估分支：

```text
unprocessing_method
vkitti_unprocessing_preset                    # old_brooks_preset 分支用
vkitti_unprocessing_mix_weights               # old_brooks_preset 分支用
randomize_unprocessing
noise_model
noise_realization_applied
raw_adapter_*                                  # raw_adapter_style 分支用
raw_adapter_config_hash                       # raw_adapter_style 分支用
```

6. `build_loaders()` 必须先生成一个 `resolved_unprocessing_config`，再用同一个对象或同一份规范化 dict 构造 VKITTI train 和 VKITTI val Dataset。
7. Dataset 构造完成后立即比较：
   - `train_dataset.describe_unprocessing()["raw_adapter_config_hash"]`
   - `val_dataset.describe_unprocessing()["raw_adapter_config_hash"]`
   - 关键 fixed 字段：tone、CCM、gain、variant、light scale、noise、black/white level
   任一不一致都 raise，不能等到 eval 阶段才发现。
8. logger 输出 `train_dataset.describe_unprocessing()` 和 `val_dataset.describe_unprocessing()`，确保日志里能明确看到方法、backend、variant policy、gain/noise 参数和 `raw_adapter_config_hash`。
9. `config.json` 中保存的 `raw_adapter_config_hash` 必须等于 train/val Dataset 的 hash；保存前做一次断言。

兼容策略：

- 默认不改旧脚本行为。旧 formal script 如果不传 `--unprocessing-method`，可以临时默认为 `old_brooks_preset`，但正式新实验脚本必须显式传。
- 更严格方案是把 `--unprocessing-method` 设为 required；如果会影响现有脚本，先在新脚本显式传，后续再收紧。
- 不要为了兼容 raw-adapter 分支而在 `foundation/engine/transforms/unprocessing.py` 的 preset registry 里注册 `"not_applicable"`。那会污染旧方法 registry，并让错误路径更晚暴露。
- 把 `--vkitti-unprocessing-preset` 改成 `default="not_applicable"` 后，必须确认所有仍走 raw old-preset 入口的现存 formal scripts 都显式传 `--vkitti-unprocessing-preset`，否则旧脚本会被 `default="not_applicable" + validate_args()` 拒绝。当前 `finetune_stf/scripts/formal/0524_run_vkitti_mseries_residual_queue.sh` 已显式传 `sensor_linear_dual`；`finetune_stf/scripts/formal/0524_run_vkitti_cseries_residual_controls_queue.sh` 是 RGB control 入口，不走 raw unprocessing preset，不属于这条校验对象。新增 raw formal script 也必须显式传。

## 6. Phase D: 接入 KITTI eval 一致性

修改文件：

```text
foundation/tools/eval_raw_residual_kitti.py
```

改动点：

1. 必须先重写 `validate_run_config(config)`。当前 `eval_raw_residual_kitti.py` 里 `expected` 字典硬编码了：

```text
"vkitti_unprocessing_preset": "sensor_linear_dual"
```

这会直接拒绝 `unprocessing_method=raw_adapter_style` 且 `vkitti_unprocessing_preset=not_applicable` 的新实验。改法不是删除所有校验，而是拆成 method-无关校验和 method-相关校验。

2. `validate_run_config` 的 method-无关硬校验继续保留：

```text
input_domain = raw4
front_end = raw_to_base_rgb_ram3
model_input_tensor = raw
raw_storage_format = synthetic_packed_bayer_4ch_halfres
fullres_even_policy = crop_bottom_to_even
rgb_input_space = halfres_2x2_area
depth_target_space = halfres_2x2_valid_mean
input_height = 187
input_width = 621
min_depth = 1.0
max_depth = 80.0
```

3. `validate_run_config` 的 method-相关部分按 `unprocessing_method` 分支：

```text
if unprocessing_method == old_brooks_preset:
  require vkitti_unprocessing_preset == sensor_linear_dual
  validate optional vkitti_unprocessing_mix_weights
  reject active raw-adapter fields

if unprocessing_method == raw_adapter_style:
  require vkitti_unprocessing_preset in [not_applicable, None]
  require raw-adapter required fields are present
  require raw_adapter_backend == analytic for Phase A
  require randomize_unprocessing == false for Phase A fixed experiments
  require raw_adapter_random_seed_policy == dataloader_generator
  require raw_adapter_cfa_pattern == RGGB
  require raw_adapter_packed_channel_order == R_Gr_Gb_B
  require raw_adapter_rgb_transfer == srgb_piecewise
  require raw_adapter_inverse_tone in [none, global_0p15]
  require raw_adapter_noise_mean_mode in [zero, rawadapter_text]
  require fixed red/blue gains are inside their explicit ranges
  require fixed variant is normal/dark/over, not mix
  require variant_weights, if present, is one-hot and matches fixed variant
  require fixed_light_scale is valid for the fixed variant
  require noise_model == none and noise_realization_applied == false
  reject active old preset / mix-weight semantics
```

如果为了兼容旧 config 需要缺省，规则只能是：

```text
missing unprocessing_method -> old_brooks_preset
```

不能把 `raw_adapter_style` 的缺失字段默默补默认值。

4. `KittiHalfresRawDataset` 不再只接收 `unprocessing_preset`，而是接收完整 unprocessing config。
5. 从训练 `config.json` 读取：
   - `unprocessing_method`
   - raw-adapter 参数
   - old preset 参数
6. 当训练用 `raw_adapter_style` 时，KITTI val 使用同一 raw-adapter transform、同一 `raw_adapter_inverse_tone` 和同一组 fixed 参数，不再切换到单独的 eval canonical 默认值。
7. `KittiHalfresRawDataset` 构造完成后必须调用 `describe_unprocessing()`，并与训练 `config.json` 中的 `raw_adapter_config_hash` 和关键 fixed 字段逐项比较；不一致直接 raise。
8. eval summary 写入 `unprocessing_method`、`validate_run_config` 采用的 method 分支、完整 fixed unprocessing policy、`raw_adapter_config_hash`，以及“来自训练 config”的标记。
9. 增加一个 shared builder，例如：

```text
build_unprocessing_from_resolved_config(config, split="train"|"vkitti_val"|"kitti_val")
```

训练 Dataset 和 KITTI eval Dataset 都必须调用这个 builder。`raw_adapter_style` 的第一版 builder 不允许根据 `split` 改变具体 unprocessing 参数；train/vkitti_val/kitti_val 都应得到同一组 fixed raw-adapter 参数，包括同一个 tone 分支。旧 `old_brooks_preset` 分支可以保留当前 train randomized、val deterministic 的行为。

shared builder 的返回值应包含规范化后的 unprocessing summary 和 hash。训练入口保存 config 前、VKITTI val 构造后、KITTI eval 构造后都用同一个比较函数校验 summary；这样训练和验证的一致性由代码保证，不依赖人工检查日志。

必须避免：

- `validate_run_config` 仍在顶层硬校验 `vkitti_unprocessing_preset == sensor_linear_dual`。
- 训练用 raw-adapter，KITTI eval 仍用 `sensor_linear_dual randomize=False`。
- 训练用旧 preset，KITTI eval 误用 raw-adapter。
- VKITTI train 使用 launch 参数，KITTI eval 使用 eval 脚本本地默认值，导致两边看似同名但参数不同。
- `raw_adapter_style` train 随机采样，但 VKITTI/KITTI val 使用 canonical 固定值，导致三者不是同一组具体 InvISP/raw-adapter 参数。
- `raw_adapter_style` train 使用 `inverse_tone=none`，但 VKITTI/KITTI val 默认启用 `global_0p15` 或旧 `inverse_smoothstep`。

## 7. Phase E: smoke 测试

所有 smoke 输出必须写入包含 `codex_smoke` 的路径。成功后删除；失败保留并报告。

### 7.1 Transform parity smoke

新增测试脚本建议：

```text
foundation/tools/smoke_raw_adapter_style_unprocessing.py
```

目标：

- 构造固定 RGB tensor。
- 关闭随机噪声或固定 generator。
- 分别覆盖 `raw_adapter_inverse_tone=none` 和 `raw_adapter_inverse_tone=global_0p15`。
- 对比 NumPy 脚本核心函数与 PyTorch transform 的输出 shape、range、关键数值；`global_0p15` 必须和离线脚本公式一致，不能等价成旧 inverse smoothstep。

命令：

```bash
conda run --live-stream -n dav3 python foundation/tools/smoke_raw_adapter_style_unprocessing.py \
  --output plans/0524_unprocessing/codex_smoke_0525_rawadapter_transform/parity.json
```

成功后删除：

```bash
rm -rf plans/0524_unprocessing/codex_smoke_0525_rawadapter_transform
```

### 7.2 Dataset shape smoke

目标：

- 构造 `VKITTI2Raw` train/val 各 2-4 个样本。
- 检查 `raw=[4,187,621]`、`image=[3,187,621]`、`depth=[187,621]`、`valid_mask=[187,621]`。
- 检查 metadata 中 `unprocessing_method=raw_adapter_style`。
- 检查 train/val `describe_unprocessing()` 的 `raw_adapter_config_hash` 完全相同，并逐项比较 tone、CCM、fixed gain、variant、fixed light scale、noise、black/white level。
- 即使命令记录了非零 `shot_noise/read_noise`，在 `--no-randomize-unprocessing` 下也必须检查 `noise_model=none` 且 `noise_realization_applied=false`。

命令：

```bash
conda run --live-stream -n dav3 python foundation/tools/smoke_vkitti2_raw_adapter_dataset.py \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt \
  --input-height 187 \
  --input-width 621 \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --unprocessing-method raw_adapter_style \
  --vkitti-unprocessing-preset not_applicable \
  --no-randomize-unprocessing \
  --raw-adapter-backend analytic \
  --raw-adapter-cfa-pattern RGGB \
  --raw-adapter-packed-channel-order R_Gr_Gb_B \
  --raw-adapter-rgb-transfer srgb_piecewise \
  --raw-adapter-inverse-tone global_0p15 \
  --raw-adapter-ccm identity \
  --raw-adapter-red-gain-range 1.9 2.4 \
  --raw-adapter-blue-gain-range 1.5 1.9 \
  --raw-adapter-fixed-red-gain 2.15 \
  --raw-adapter-fixed-blue-gain 1.70 \
  --raw-adapter-variant-policy normal \
  --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0 \
  --raw-adapter-fixed-light-scale 1.0 \
  --raw-adapter-dark-light-scale-range 0.05 0.4 \
  --raw-adapter-over-light-scale-range 1.5 2.5 \
  --raw-adapter-shot-noise 0.001 \
  --raw-adapter-read-noise 0.0005 \
  --raw-adapter-noise-mean-mode zero \
  --raw-adapter-black-level 0.0 \
  --raw-adapter-white-level 1.0 \
  --raw-adapter-random-seed-policy dataloader_generator \
  --output plans/0524_unprocessing/codex_smoke_0525_rawadapter_dataset/summary.json
```

### 7.3 Training smoke

目标：

- 1 epoch
- `--max-train-steps 2`
- `--max-val-samples 4`
- KITTI eval 最多 4 张
- 验证 forward/backward/eval/config 写出。
- 验证 `config.json`、VKITTI train summary、VKITTI val summary、KITTI eval summary 中的 `raw_adapter_config_hash` 一致；关键 fixed 字段也必须一致。
- 验证 `--no-randomize-unprocessing` 下 `config.json` / Dataset metadata / KITTI eval summary 都记录 `noise_model=none` 和 `noise_realization_applied=false`。

命令模板：

```bash
SMOKE_ROOT=plans/0524_unprocessing/codex_smoke_0525_rawadapter_train

CUDA_VISIBLE_DEVICES=0 conda run --live-stream -n dav3 \
  python foundation/tools/train_vkitti2_raw_residual.py \
  --input-domain raw4 \
  --model-input-tensor raw \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --fullres-even-policy crop_bottom_to_even \
  --rgb-input-space halfres_2x2_area \
  --depth-target-space halfres_2x2_valid_mean \
  --front-end raw_to_base_rgb_ram3 \
  --encoder vits \
  --pretrained-from /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth \
  --vkitti-train-list finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt \
  --vkitti-val-list finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt \
  --eval-kitti \
  --kitti-base /mnt/drive/kitti \
  --kitti-val-split /home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt \
  --kitti-eval-protocol halfres_raw_canonical_even_pad_crop_affine_disp \
  --kitti-expected-val-samples 652 \
  --max-kitti-val-samples 4 \
  --input-height 187 \
  --input-width 621 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --residual-feature-source ffm_mid \
  --residual-alpha 0.5 \
  --d0-sign 1 \
  --unprocessing-method raw_adapter_style \
  --vkitti-unprocessing-preset not_applicable \
  --no-randomize-unprocessing \
  --hflip-prob 0.5 \
  --raw-adapter-backend analytic \
  --raw-adapter-cfa-pattern RGGB \
  --raw-adapter-packed-channel-order R_Gr_Gb_B \
  --raw-adapter-rgb-transfer srgb_piecewise \
  --raw-adapter-inverse-tone global_0p15 \
  --raw-adapter-ccm identity \
  --raw-adapter-red-gain-range 1.9 2.4 \
  --raw-adapter-blue-gain-range 1.5 1.9 \
  --raw-adapter-fixed-red-gain 2.15 \
  --raw-adapter-fixed-blue-gain 1.70 \
  --raw-adapter-variant-policy normal \
  --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0 \
  --raw-adapter-fixed-light-scale 1.0 \
  --raw-adapter-dark-light-scale-range 0.05 0.4 \
  --raw-adapter-over-light-scale-range 1.5 2.5 \
  --raw-adapter-shot-noise 0.001 \
  --raw-adapter-read-noise 0.0005 \
  --raw-adapter-noise-mean-mode zero \
  --raw-adapter-black-level 0.0 \
  --raw-adapter-white-level 1.0 \
  --raw-adapter-random-seed-policy dataloader_generator \
  --epochs 1 \
  --bs 8 \
  --accum-steps 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --log-interval 1 \
  --save-interval 1 \
  --eval-interval 1 \
  --max-train-steps 2 \
  --max-val-samples 4 \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --save-path "${SMOKE_ROOT}/exp" \
  --heavy-save-path "${SMOKE_ROOT}/heavy" \
  2>&1 | tee "${SMOKE_ROOT}/train_smoke.log"
```

成功后删除：

```bash
rm -rf "${SMOKE_ROOT}"
```

失败时保留整个 `${SMOKE_ROOT}`。

## 8. Phase F: 正式 launch 脚本

新增正式脚本建议：

```text
finetune_stf/scripts/formal/0525_run_vkitti_mseries_rawadapter_queue.sh
```

要求：

- 使用 tmux。
- 默认 conda env 为 `dav3`。
- 实验名以服务器本地时间 `MMDD_HHMM` 开头。
- 不覆盖已有 save/heavy 目录。
- smoke 成功后删除 `codex_smoke` 目录。
- formal 参数必须显式写全，不依赖 transform 默认值。
- `raw_adapter_style` / 当前 InvISP 路线默认必须显式传 `--no-randomize-unprocessing`，保证 VKITTI train、VKITTI val、KITTI val 使用同一组固定 unprocessing 参数。

第一组正式实验建议：

```text
M2-RA0:
  method=raw_adapter_style
  backend=analytic
  randomize_unprocessing=false
  ccm=identity
  inverse_tone=global_0p15
  variant_policy=normal
  fixed_light_scale=1.0

M2-RA1-fixed-dark:
  method=raw_adapter_style
  backend=analytic
  randomize_unprocessing=false
  ccm=identity
  inverse_tone=global_0p15
  variant_policy=dark
  fixed_light_scale=<explicit value in 0.05..0.4>

M2-RA2-fixed-over:
  method=raw_adapter_style
  backend=analytic
  randomize_unprocessing=false
  ccm=identity
  inverse_tone=global_0p15
  variant_policy=over
  fixed_light_scale=<explicit value in 1.5..2.5>

M2-RA3-ccm-tone-normal:
  method=raw_adapter_style
  backend=analytic
  randomize_unprocessing=false
  ccm=generic_d65
  inverse_tone=global_0p15
  variant_policy=normal
  fixed_light_scale=1.0

M2-RA4-no-tone-normal:
  method=raw_adapter_style
  backend=analytic
  randomize_unprocessing=false
  ccm=identity
  inverse_tone=none
  variant_policy=normal
  fixed_light_scale=1.0
```

先跑 `M2-RA0`，不要一次性开全矩阵。RA0 对齐本地 `tone_only` 的 tone 口径，只保留 CCM 为 identity，能回答“带离线脚本 inverse_global_tone 的 RAW-Adapter-style analytic normal 固定参数是否比当前 sensor_linear_dual 更适合训练”。RA3 再对齐本地 `ccm_tone` 完整解析近似路线；RA4 是 no-tone 消融，不作为主线默认。RA1/RA2 再用固定 dark/over 参数检查低光/过曝，不使用随机 mix 作为第一轮正式实验。

formal run name 示例：

```text
0525_HHMM_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20
```

tmux 输出必须包含：

```text
Started tmux session: <SESSION>
Queue log: <LOG>
Attach: tmux attach -t <SESSION>
Monitor: tail -f <LOG>
```

## 9. Phase G: 外部 InvISP / learned unprocessing cache

不建议第一版把 InvISP 模型直接放进 dataloader，原因：

- InvISP 是 camera-specific，VKITTI/KITTI/NYU/Hypersim 混合来源不应称为真实 RAW。
- 在线跑 learned InvISP 会显著拖慢 dataloader，且多 worker / GPU 资源调度复杂。
- 当前计划脚本的 `external_npy` 本质是读外部 raw-RGB，不是模型调用。

推荐第二阶段实现：

```text
1. 离线生成 raw-RGB cache，shape 与原始 RGB 对齐。
2. Dataset 读取 RGB/depth，同时按 image_path 找 raw-RGB cache。
3. 对 raw-RGB 应用与 RGB/depth 一致的几何处理。
4. 几何处理后再做 inverse WB / exposure / noise / pack。
```

新增参数：

```text
--raw-adapter-backend external_raw_rgb_cache
--raw-adapter-external-raw-rgb-root <PATH>
--raw-adapter-external-key raw_rgb
--raw-adapter-external-cache-space original_rgb_aligned
```

校验：

- cache 路径必须存在。
- raw-RGB shape 与 RGB 原始 shape 对齐。
- filelist 中每个样本必须找到对应 raw-RGB，否则报错。
- 如果 external 输出已经是 packed Bayer，不走第一阶段。需新增 backend `external_packed_bayer_cache`，并单独处理 CFA/crop/flip 语义。

## 10. 验收标准

代码层：

- 旧 `sensor_linear_dual` 训练脚本不受影响。
- 新 raw-adapter transform 能通过 parity smoke。
- VKITTI dataset smoke shape 正确。
- training smoke 能完成 2 step train + 4 sample val + 4 sample KITTI eval。
- `config.json` 和 `train.log` 明确记录 `unprocessing_method=raw_adapter_style`、所有 raw-adapter 语义参数和 `raw_adapter_config_hash`。
- VKITTI train、VKITTI val、KITTI val 的 `raw_adapter_config_hash` 完全一致；tone、CCM、fixed gain、variant、fixed light scale、noise、black/white level 明文字段也一致。

实验层：

- 正式 RA0 与当前 M2 使用同一 split、同一 geometry、同一 model/front-end/residual head，只改变 unprocessing 方法。
- VKITTI val 和 KITTI eval 都使用训练 config 中同一 unprocessing 方法、同一 tone 曲线和同一组固定 raw-adapter 参数；不允许 eval 脚本使用本地默认值覆盖训练 config。
- 所有 formal run 名以 launch 时间 `MMDD_HHMM` 开头。

## 11. 风险和处理

风险 1：hflip 后固定 RGGB 与物理 CFA 不一致。

处理：第一版标注 synthetic RAW-like，并记录 `hflip_applied`。如果后续要更物理，需要 hflip 后切换 CFA pattern 或在 packed 坐标处理 flip。

风险 2：RA1/RA2 的 dark/over 会改变输入分布，可能影响 DAv2 RGB baseline 与 RAW residual 的相对学习难度。

处理：先跑 RA0 normal-only，再逐步加固定 dark/over。第一轮不使用 random mix，保证 train/val 的 unprocessing 参数完全一致。

风险 3：RAW-Adapter-style 固定 no-noise 第一版与后续 Gaussian noise 随机增强实验不可直接比较；Gaussian noise 与旧 Poisson-Gaussian 也不可直接比较。

处理：第一版固定实验名和 config 明确写 `noise_model=none`、`noise_realization_applied=false`。后续随机增强实验必须在实验名和 config 明确写 `noise_model=raw_adapter_gaussian_signal_dependent`，并单独与旧 Poisson-Gaussian 区分。

风险 4：InvISP cache 与几何增强顺序不一致。

处理：external raw-RGB cache 必须与原 RGB 对齐，并在 Dataset 中应用同一几何处理；不要直接读取已经 crop/pack 的离线 `.npz` 替代当前训练路径，除非另起 cache dataset。

## 12. 推荐执行顺序

1. 实现 `RawAdapterStyleUnprocessingTransform` analytic 分支。
2. 加 transform parity smoke。
3. 接入 `VKITTI2Raw`，加 dataset smoke。
4. 接入 `train_vkitti2_raw_residual.py` 参数和 config 保存。
5. 接入 `eval_raw_residual_kitti.py`，保证 KITTI eval 方法一致。
6. 跑 training smoke。
7. 新建 `0525_run_vkitti_mseries_rawadapter_queue.sh`。
8. 跑正式 RA0。
9. 根据 RA0 结果决定是否跑 RA1/RA2。
10. 若 RA0/RA1 有价值，再做 external InvISP raw-RGB cache 分支。
