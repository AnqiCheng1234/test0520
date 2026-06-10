# LOD RAW Cross-Eval

- output_dir: `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_stageB_strict/0610_0212_lod_postram_stageB_strict/T_seed42`
- eval_split: `01Valid`
- bn_recalib_split: `00Train`
- bn_recalib_definition: reset running stats, cumulative average, RAM/front-end only, no gradient

## Matrix

| variant | matrix | checkpoint | input | D1 | AbsRel | RMSE | post_delta_ratio | samples | BN samples |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| strict | M_DD | C_dark | I_dark | 0.848504 | 3.754150 | 24.267365 | 0.091457 | 112 | 0 |

## Recovery

| variant | G=M_NN-M_DD | R_DN | M_DN-M_DD | M_ND-M_DD |
|---|---:|---:|---:|---:|
