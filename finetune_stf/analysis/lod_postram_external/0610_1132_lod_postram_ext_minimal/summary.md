# Post-RAM External Eval Summary

- baseline_ext_id: `EXT_NOOP_A0`

| EXT | denoiser | sigma | alpha | affine | D1 | delta D1 | AbsRel | RMSE | clamp | delta_ratio | samples | viz |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| EXT_D1 | drunet | 10.0 | 0.25 | q001q999 | 0.841947 | -0.003332 | 3.789210 | 25.035134 | 0.001919 | 0.034174 | 112 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D1/viz/EXT_D1/manifest.json` |
| EXT_D3 | drunet | 25.0 | 0.25 | q001q999 | 0.846295 | 0.001016 | 3.593061 | 24.742452 | 0.001919 | 0.044219 | 112 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D3/viz/EXT_D3/manifest.json` |
| EXT_D4 | drunet | 25.0 | 0.5 | q001q999 | 0.844203 | -0.001076 | 3.689165 | 24.803842 | 0.001919 | 0.088437 | 112 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_D4/viz/EXT_D4/manifest.json` |
| EXT_ID_Q001 | identity | n_a | 1.0 | q001q999 | 0.845080 | -0.000200 | 3.691696 | 24.874620 | 0.001919 | 0.000936 | 112 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_ID_Q001/viz/EXT_ID_Q001/manifest.json` |
| EXT_NOOP_A0 | none | n_a | 0.0 | n_a | 0.845279 | 0.000000 | 3.666699 | 24.878314 | 0.000000 | 0.000000 | 112 | `/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/lod_postram_external/0610_1132_lod_postram_ext_minimal/EXT_NOOP_A0/viz/EXT_NOOP_A0/manifest.json` |
