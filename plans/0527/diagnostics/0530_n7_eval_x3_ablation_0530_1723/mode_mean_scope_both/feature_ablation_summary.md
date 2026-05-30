# Feature Ablation Summary

- source_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_09.pth
- mode: mean
- scope: both
- key: x3
- seed: 42
- method_id: N7
- c2_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth

| dataset | samples | final abs_rel | D1 abs_rel | D0 abs_rel | final-D1 abs_rel | boundary final | boundary final-D1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VKITTI | 1000 | 0.12205316245870751 | 0.12097560103065506 | 0.15309758370696205 | 0.0010775614280524454 | 0.2611416398187342 | -0.008015187080881725 |
| KITTI | 652 | 0.09622222485165156 | 0.09644374810658646 | 0.11837322563408595 | -0.0002215232549348939 |  |  |
