# N6 x3 Feature Ablation Summary

- source_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0529_1523_vkitti_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_03.pth
- mode: true
- key: x3
- seed: 42
- method_id: N2
- c2_checkpoint: /mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth

| dataset | samples | final abs_rel | D1 abs_rel | D0 abs_rel | final-D1 abs_rel | boundary final | boundary final-D1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VKITTI | 1000 | 0.11860904558390478 | 0.12097898364715157 | 0.153095935985279 | -0.002369938063246785 | 0.24947469360429209 | -0.019698218294812997 |
| KITTI | 652 | 0.09670720289773695 | 0.09644603888416722 | 0.11837516457299688 | 0.00026116401356972296 |  |  |
