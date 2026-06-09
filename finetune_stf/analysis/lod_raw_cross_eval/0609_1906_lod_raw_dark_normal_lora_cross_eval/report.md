# LOD RAW Cross-Eval

- output_dir: `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1906_lod_raw_dark_normal_lora_cross_eval`
- eval_split: `01Valid`
- bn_recalib_split: `00Train`
- bn_recalib_definition: reset running stats, cumulative average, RAM/front-end only, no gradient

## Matrix

| variant | matrix | checkpoint | input | D1 | AbsRel | RMSE | samples | BN samples |
|---|---|---|---|---:|---:|---:|---:|---:|
| strict | M_DD | C_dark | I_dark | 0.845279 | 3.666699 | 24.878314 | 112 | 0 |
| bn_recalib | M_DD | C_dark | I_dark | 0.842198 | 3.764893 | 25.189090 | 112 | 1958 |
| strict | M_DN | C_dark | I_normal | 0.880533 | 3.052718 | 20.466344 | 112 | 0 |
| bn_recalib | M_DN | C_dark | I_normal | 0.878907 | 3.179445 | 20.650424 | 112 | 1958 |
| strict | M_NN | C_normal | I_normal | 0.898946 | 2.469963 | 18.394070 | 112 | 0 |
| bn_recalib | M_NN | C_normal | I_normal | 0.898185 | 2.572419 | 18.362419 | 112 | 1958 |
| strict | M_ND | C_normal | I_dark | 0.816972 | 3.237003 | 28.824663 | 112 | 0 |
| bn_recalib | M_ND | C_normal | I_dark | 0.820890 | 3.422002 | 28.445251 | 112 | 1958 |

## Recovery

| variant | G=M_NN-M_DD | R_DN | M_DN-M_DD | M_ND-M_DD |
|---|---:|---:|---:|---:|
| bn_recalib | 0.055987 | 0.655682 | 0.036710 | -0.021308 |
| strict | 0.053667 | 0.656898 | 0.035253 | -0.028307 |
