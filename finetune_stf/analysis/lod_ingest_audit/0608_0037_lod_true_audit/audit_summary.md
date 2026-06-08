# LOD True Ingest Audit

- run_name: `0608_0037_lod_true_audit`
- manifest: `/home/caq/6666_raw/0000_dataset/LOD/manifests/lod_true_pairs_2118_112_seed42.csv`
- rows: `2230`
- split_counts: `{'00Train': 2118, '01Valid': 112}`

## Gates

- normalization: `pass` - RAW values are uint16 with low clipping; RAW_Dark is brighter than RAW_normal, so this remains an explicit uint16_div_65535 semantic choice rather than inferred exposure linearization.
- alignment: `pass` - Paired RGB NCC is clearly above unrelated dark frames and downsample phase shifts are small.
- channel_order: `pass` - Correlation best mapping is OpenCV BGR read order: cv2_channel_0->B, 1->G, 2->R. Use channel_reorder=(2,1,0) to feed model RGB.

## RAW Intensity

| subset | sampled pairs | mean | p1 | p50 | p99 | max | clip@65535 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 2230 | 9282.56 | 0 | 5386 | 50532 | 65535 | 0.00317509 |
| dark | 2230 | 11746.50 | 0 | 8491 | 54250 | 65535 | 0.00286339 |

- dark_over_normal_mean_ratio: `1.2654`
- normalization_recommendation: `uint16_div_65535`

## Alignment

- paired RGB NCC p10/p50/p90: `0.5017` / `0.8765` / `0.9628`
- unrelated RGB NCC p90: `0.2154`
- downsample phase shift abs p90: `0.3236` pixels at 128x192 audit scale

## Channel Order

- best_cv2_to_rgb_mapping: `{'cv2_channel_0': 'B', 'cv2_channel_1': 'G', 'cv2_channel_2': 'R'}`
- recommended_channel_reorder_to_model_rgb: `[2, 1, 0]`
