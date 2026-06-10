# LOD Post-RAM Stage B Strict Summary

## Strict Rows

| run_id | seed | kind | epoch | training_best | strict_D1 | AbsRel | RMSE | delta_ratio | resolved_config |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly | n_a | strict_noop_reference | 37 | 0.845068 | 0.845279 | 3.666699 | 24.878314 | 0.000000 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/resolved_config.json` |
| 0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42 | 42 | B0_LORA_s01_baseline | 9 | 0.847611 | 0.847592 | 3.666920 | 24.367533 | 0.000000 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/resolved_config.json` |
| 0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42 | 42 | T_LORA_s01_cleanup | 9 | 0.848464 | 0.848504 | 3.754150 | 24.267365 | 0.091457 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/resolved_config.json` |
| 0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123 | 123 | B0_LORA_s01_baseline | 9 | 0.848650 | 0.848470 | 3.860734 | 24.346629 | 0.000000 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/resolved_config.json` |
| 0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123 | 123 | T_LORA_s01_cleanup | 9 | 0.850784 | 0.850708 | 3.746498 | 24.166887 | 0.102271 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/resolved_config.json` |
| 0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777 | 777 | B0_LORA_s01_baseline | 9 | 0.845163 | 0.845085 | 3.778942 | 24.514426 | 0.000000 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/resolved_config.json` |
| 0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777 | 777 | T_LORA_s01_cleanup | 1 | 0.846308 | 0.846413 | 4.324509 | 25.026583 | 0.039808 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/exp/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/resolved_config.json` |

## Matched Seed Deltas

| seed | training_delta | strict_delta | baseline_strict | cleanup_strict | cleanup_delta_ratio |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.000853 | 0.000912 | 0.847592 | 0.848504 | 0.091457 |
| 123 | 0.002134 | 0.002238 | 0.848470 | 0.850708 | 0.102271 |
| 777 | 0.001145 | 0.001329 | 0.845085 | 0.846413 | 0.039808 |

## Decision Notes

- Stage A strict no-op reference: D1 `0.845279`, AbsRel `3.666699`, RMSE `24.878314` from `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/EXT_NOOP_A0/metrics.json`.
- Best cleanup strict row: `0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123` with strict D1 `0.850708` and training-best `0.850784`.
- Mean matched strict delta: `0.001493` over `3` seeds.
- Mean matched training-best delta: `0.001377` over `3` seeds.
- Go-gate status: D1 has a positive matched-seed signal, but only seed123 cleanup exceeds the matched baseline training-best upper bound, and winner AbsRel is worse than the Stage A strict no-op reference. Do not automatically launch formal ablation from this pass.
- `viz_path` is `n_a`: this strict pass generated metrics/per-sample CSV only; Stage A external sanity generated post-RAM panels.
