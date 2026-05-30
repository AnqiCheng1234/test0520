# N6 x3 Feature Ablation Summary

- source_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0529_1523_vkitti_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_03.pth
- mode: shuffle
- key: x3
- seed: 42
- method_id: N2
- c2_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth

| dataset | samples | final abs_rel | D1 abs_rel | D0 abs_rel | final-D1 abs_rel | boundary final | boundary final-D1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VKITTI | 1000 | 0.12049659647073724 | 0.12096944073420056 | 0.15309450501543295 | -0.00047284426346332065 | 0.26171254890765144 | -0.007431190918632757 |
| KITTI | 652 | 0.09681922555418239 | 0.09644540205881978 | 0.11837387988765452 | 0.0003738234953626074 |  |  |
