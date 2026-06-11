# ROD zero-shot eval with LOD checkpoints 结果记录

日期：2026-06-10

本文记录 `/home/caq/6666_raw/dav2_raw_0603` 中三组 ROD-night zero-shot eval。这里的 zero-shot 指：不在 ROD 上训练，直接加载 LOD checkpoint，在 ROD `01Valid` 上评估。

指标口径：ROD GT 使用 ROD manifest 中的 DAv2-L pseudo inverse-relative label；指标为 `rod_night_val` protocol 下 affine align 后的 inverse-relative proxy，不是 metric depth benchmark。`d1/d2/d3` 越高越好，`abs_rel/rmse` 越低越好。

## 1. Eval protocol

- ROD root：`/home/caq/6666_raw/0000_dataset/ROD`
- ROD manifest：`/home/caq/6666_raw/0000_dataset/ROD/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv`
- eval split：`01Valid`
- eval samples：`2000`
- target：`rod_pseudo_depth_dav2l_teacherbright_inverse_relative`
- crop：center crop to `512x960`

输入视图：

- `rod_raw_student_rgb`：ROD RAW24 -> `student_dark_degreen_v1` RGB，`front_end=dav2_rgb`，`model_input_tensor=image`。
- `rod_raw_rgb3`：ROD RAW24 / packed Bayer -> `[R,(Gr+Gb)/2,B]` raw3，`front_end=raw_rgb16_ram3`，`model_input_tensor=raw`。该视图不走 student RGB ISP。

## 2. Results

| Run | Checkpoint | Eval input | D1 | D2 | D3 | abs_rel | rmse | silog |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `0610_2323_rod_studentrgb_zeroshot_from_0608_2246_lodrgb_lora_eval` | `0608_2246` LOD RGB_Dark LoRA tap r8/a16 best | ROD student RGB | 0.5308 | 0.7746 | 0.8306 | 38.9013 | 47.3037 | 0.9826 |
| `0610_2338_rod_raw_rgb3_zeroshot_from_0608_2026_lod_raw_dark_lora_eval` | `0608_2026` LOD RAW_Dark RGB16 RamCore3 LoRA tap r8/a16 best | ROD raw3 `[R,Gavg,B]` | 0.3442 | 0.5820 | 0.7140 | 75.5859 | 75.3958 | 1.4209 |
| `0610_2340_rod_raw_rgb3_zeroshot_from_0609_0044_lod_raw_normal_lora_eval` | `0609_0044` LOD RAW_normal RGB16 RamCore3 LoRA tap r8/a16 best | ROD raw3 `[R,Gavg,B]` | 0.6422 | 0.7710 | 0.8335 | 32.7645 | 35.2631 | 1.5775 |

完整指标 JSON：

- `0608_2246` -> ROD student RGB：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_2323_rod_studentrgb_zeroshot_from_0608_2246_lodrgb_lora_eval/eval_only_rod_studentrgb_zeroshot_summary.json`
- `0608_2026` -> ROD raw3：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_2338_rod_raw_rgb3_zeroshot_from_0608_2026_lod_raw_dark_lora_eval/eval_only_rod_raw_rgb3_zeroshot_summary.json`
- `0609_0044` -> ROD raw3：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_2340_rod_raw_rgb3_zeroshot_from_0609_0044_lod_raw_normal_lora_eval/eval_only_rod_raw_rgb3_zeroshot_summary.json`

Queue logs：

- `0608_2246` -> ROD student RGB：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_2323_rod_studentrgb_zeroshot_0608_2246.queue.log`
- `0608_2026` / `0609_0044` -> ROD raw3：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_2338_rod_raw_rgb3_zeroshot_0608_2026_0609_0044.queue.log`

## 3. Notes

- `0609_0044` RAW_normal checkpoint 在 ROD raw3 zero-shot 上明显好于 `0608_2026` RAW_Dark checkpoint：D1 `0.6422` vs `0.3442`。
- `0609_0044` ROD raw3 zero-shot 的 D1 高于 `0608_2246` ROD student RGB zero-shot：`0.6422` vs `0.5308`；但二者输入视图和 checkpoint 训练域不同，只能作为 cross-domain probe，不应当当作同协议 formal ranking。
- edge metrics 在三组 summary 中为 `nan`，JSON 中记录为 `null`，主表未列入。
- 本次 raw3 eval 为兼容 LOD `raw_rgb16_ram3` checkpoint，新增了显式 `rod_raw_rgb3/raw24_base_rgb3` 输入语义；它不同于已有 ROD `rod_raw/raw_ram` 的 raw4 `[R,Gr,Gb,B] -> raw_to_base_rgb_ram3` 路径。
