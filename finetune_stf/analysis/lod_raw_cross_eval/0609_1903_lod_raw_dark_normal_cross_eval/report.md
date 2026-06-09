# LOD RAW Cross-Eval

- output_dir: `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_raw_cross_eval/0609_1903_lod_raw_dark_normal_cross_eval`
- eval_split: `01Valid`
- bn_recalib_split: `00Train`
- bn_recalib_definition: reset running stats, cumulative average, RAM/front-end only, no gradient

## Matrix

| variant | matrix | checkpoint | input | D1 | AbsRel | RMSE | samples | BN samples |
|---|---|---|---|---:|---:|---:|---:|---:|
| strict | M_DD | C_dark | I_dark | 0.829694 | 3.904160 | 27.016630 | 112 | 0 |
| bn_recalib | M_DD | C_dark | I_dark | 0.828288 | 4.000188 | 27.199405 | 112 | 1958 |
| strict | M_DN | C_dark | I_normal | 0.875950 | 2.811440 | 20.942291 | 112 | 0 |
| bn_recalib | M_DN | C_dark | I_normal | 0.874958 | 3.191405 | 21.184416 | 112 | 1958 |
| strict | M_NN | C_normal | I_normal | 0.887073 | 2.380199 | 19.731537 | 112 | 0 |
| bn_recalib | M_NN | C_normal | I_normal | 0.887237 | 2.767326 | 19.900387 | 112 | 1958 |
| strict | M_ND | C_normal | I_dark | 0.792805 | 3.684180 | 31.994471 | 112 | 0 |
| bn_recalib | M_ND | C_normal | I_dark | 0.801819 | 3.762086 | 30.455739 | 112 | 1958 |

## Recovery

| variant | G=M_NN-M_DD | R_DN | M_DN-M_DD | M_ND-M_DD |
|---|---:|---:|---:|---:|
| bn_recalib | 0.058949 | 0.791702 | 0.046670 | -0.026468 |
| strict | 0.057380 | 0.806138 | 0.046256 | -0.036889 |
