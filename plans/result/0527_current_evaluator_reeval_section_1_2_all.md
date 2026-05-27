# Current Evaluator Re-eval for Summary Section 1.2

Comparison rule: `same_4dp` is true only when the current re-evaluation rounded to 4 decimals matches the value printed in `rgb_raw_baseline_fairness_summary.md` section 1.2.

| Experiment | Epoch | Changed metrics | boundary | high-error | far50 | dark | saturated | mean_gate | mean_abs_delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| C1 RGB-residual | 3 | boundary, high-error, far50, dark | 0.5169 != 0.8588 (+0.3419) | 0.3279 != 0.4926 (+0.1647) | 0.9089 != 1.1212 (+0.2123) | 0.1160 != 0.1216 (+0.0056) | 0.1747 = 0.1747 (+0.0000) | 0.3213 = 0.3213 (+0.0000) | 0.4671 = 0.4671 (-0.0000) |
| C2 D0-only residual | 11 | boundary, high-error, far50, dark, saturated | 0.4748 != 0.4285 (-0.0463) | 0.2703 != 0.2771 (+0.0068) | 0.5908 != 0.6736 (+0.0828) | 0.1143 != 0.1196 (+0.0053) | 0.1382 != 0.1383 (+0.0001) | 0.3358 = 0.3358 (+0.0000) | 0.4537 = 0.4537 (+0.0000) |
| M2 FFM-mid residual | 8 | boundary, high-error, far50, dark, saturated | 0.5723 != 0.5240 (-0.0483) | 0.3366 != 0.3282 (-0.0084) | 0.6057 != 0.6499 (+0.0442) | 0.1214 != 0.1304 (+0.0090) | 0.1347 != 0.1348 (+0.0001) | 0.3122 = 0.3122 (-0.0000) | 0.4665 = 0.4665 (+0.0000) |
| M2 RA0 rawadapter | 9 | boundary, high-error, far50, dark, saturated, mean_gate, mean_abs_delta | 0.5444 != 0.5519 (+0.0075) | 0.3434 != 0.3214 (-0.0220) | 0.9874 != 1.0496 (+0.0622) | 0.1260 != 0.1237 (-0.0023) | 0.1504 != 0.1503 (-0.0001) | 0.3174 != 0.3175 (+0.0001) | 0.4706 != 0.4707 (+0.0001) |
| M1 RA0 x3 D0-concat | 14 | boundary, high-error, far50, dark, saturated, mean_gate | 0.4120 != 0.3879 (-0.0241) | 0.3062 != 0.2732 (-0.0330) | 0.6866 != 0.6248 (-0.0618) | 0.1380 != 0.1141 (-0.0239) | 0.1496 != 0.1492 (-0.0004) | 0.3234 != 0.3233 (-0.0001) | 0.4698 = 0.4698 (+0.0000) |
| M2 no-D0 FFM-mid only | 6 | boundary, high-error, far50, dark, saturated | 0.7841 != 0.8570 (+0.0729) | 0.5269 != 0.4809 (-0.0460) | 1.0226 != 1.0255 (+0.0029) | 0.1714 != 0.1691 (-0.0023) | 0.2101 != 0.2102 (+0.0001) | 0.2473 = 0.2473 (-0.0000) | 0.4365 = 0.4365 (-0.0000) |
| M1 no-D0 x3 only | 14 | boundary, high-error, far50, dark, saturated | 0.8605 != 0.6499 (-0.2106) | 0.5868 != 0.4733 (-0.1135) | 1.2515 != 0.8626 (-0.3889) | 0.1739 != 0.1641 (-0.0098) | 0.1999 != 0.1997 (-0.0002) | 0.2335 = 0.2335 (+0.0000) | 0.4445 = 0.4445 (-0.0000) |

## Full Per-row Details

### C1 RGB-residual (0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 03)

- evaluator: `foundation.tools.train_vkitti2_residual_control.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c1_rgb_residual_vits_halfrgb_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_03.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.5169 | 0.8588 | +0.3419 | False |
| high-error | 0.3279 | 0.4926 | +0.1647 | False |
| far50 | 0.9089 | 1.1212 | +0.2123 | False |
| dark | 0.1160 | 0.1216 | +0.0056 | False |
| saturated | 0.1747 | 0.1747 | +0.0000 | True |
| mean_gate | 0.3213 | 0.3213 | +0.0000 | True |
| mean_abs_delta | 0.4671 | 0.4671 | -0.0000 | True |

### C2 D0-only residual (0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 11)

- evaluator: `foundation.tools.train_vkitti2_residual_control.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_11.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.4748 | 0.4285 | -0.0463 | False |
| high-error | 0.2703 | 0.2771 | +0.0068 | False |
| far50 | 0.5908 | 0.6736 | +0.0828 | False |
| dark | 0.1143 | 0.1196 | +0.0053 | False |
| saturated | 0.1382 | 0.1383 | +0.0001 | False |
| mean_gate | 0.3358 | 0.3358 | +0.0000 | True |
| mean_abs_delta | 0.4537 | 0.4537 | +0.0000 | True |

### M2 FFM-mid residual (0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 08)

- evaluator: `foundation.tools.train_vkitti2_raw_residual.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_0204_vkitti_m2_ffm_mid_residual_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_08.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.5723 | 0.5240 | -0.0483 | False |
| high-error | 0.3366 | 0.3282 | -0.0084 | False |
| far50 | 0.6057 | 0.6499 | +0.0442 | False |
| dark | 0.1214 | 0.1304 | +0.0090 | False |
| saturated | 0.1347 | 0.1348 | +0.0001 | False |
| mean_gate | 0.3122 | 0.3122 | -0.0000 | True |
| mean_abs_delta | 0.4665 | 0.4665 | +0.0000 | True |

### M2 RA0 rawadapter (0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 09)

- evaluator: `foundation.tools.train_vkitti2_raw_residual.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_09.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.5444 | 0.5519 | +0.0075 | False |
| high-error | 0.3434 | 0.3214 | -0.0220 | False |
| far50 | 0.9874 | 1.0496 | +0.0622 | False |
| dark | 0.1260 | 0.1237 | -0.0023 | False |
| saturated | 0.1504 | 0.1503 | -0.0001 | False |
| mean_gate | 0.3174 | 0.3175 | +0.0001 | False |
| mean_abs_delta | 0.4706 | 0.4707 | +0.0001 | False |

### M1 RA0 x3 D0-concat (0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 14)

- evaluator: `foundation.tools.train_vkitti2_raw_residual.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_14.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.4120 | 0.3879 | -0.0241 | False |
| high-error | 0.3062 | 0.2732 | -0.0330 | False |
| far50 | 0.6866 | 0.6248 | -0.0618 | False |
| dark | 0.1380 | 0.1141 | -0.0239 | False |
| saturated | 0.1496 | 0.1492 | -0.0004 | False |
| mean_gate | 0.3234 | 0.3233 | -0.0001 | False |
| mean_abs_delta | 0.4698 | 0.4698 | +0.0000 | True |

### M2 no-D0 FFM-mid only (0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 06)

- evaluator: `foundation.tools.train_vkitti2_raw_residual.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0213_vkitti_m2nod0_ra0_ffm_mid_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_06.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.7841 | 0.8570 | +0.0729 | False |
| high-error | 0.5269 | 0.4809 | -0.0460 | False |
| far50 | 1.0226 | 1.0255 | +0.0029 | False |
| dark | 0.1714 | 0.1691 | -0.0023 | False |
| saturated | 0.2101 | 0.2102 | +0.0001 | False |
| mean_gate | 0.2473 | 0.2473 | -0.0000 | True |
| mean_abs_delta | 0.4365 | 0.4365 | -0.0000 | True |

### M1 no-D0 x3 only (0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20, epoch 14)

- evaluator: `foundation.tools.train_vkitti2_raw_residual.evaluate_model`
- checkpoint: `/mnt/drive/3333_raw/0000_exp_ckpt/0526_0344_vkitti_m1nod0_ra0_x3_only_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_14.pth`
- samples: `1000`

| Metric | Old 1.2 | Current | Delta | Same at 4dp |
|---|---:|---:|---:|---|
| boundary | 0.8605 | 0.6499 | -0.2106 | False |
| high-error | 0.5868 | 0.4733 | -0.1135 | False |
| far50 | 1.2515 | 0.8626 | -0.3889 | False |
| dark | 0.1739 | 0.1641 | -0.0098 | False |
| saturated | 0.1999 | 0.1997 | -0.0002 | False |
| mean_gate | 0.2335 | 0.2335 | +0.0000 | True |
| mean_abs_delta | 0.4445 | 0.4445 | -0.0000 | True |

