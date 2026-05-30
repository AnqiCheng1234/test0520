# Feature Ablation Summary

- source_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_09.pth
- mode: shuffle
- scope: both
- key: x3
- seed: 42
- method_id: N7
- c2_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth

| dataset | samples | final abs_rel | D1 abs_rel | D0 abs_rel | final-D1 abs_rel | boundary final | boundary final-D1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VKITTI | 1000 | 0.1212763077369966 | 0.12097560103065506 | 0.15309758370696205 | 0.0003007067063415386 | 0.2623648877519317 | -0.006791939147684223 |
| KITTI | 652 | 0.09637372228767646 | 0.09644374810658646 | 0.11837322563408595 | -7.00258189099967e-05 |  |  |
