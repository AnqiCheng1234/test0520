# N7 controls summary

## P0 N7 eval-time x3 ablation

| mode | scope | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | dark | saturated | KITTI abs_rel | mean_gate | mean_abs_gate_delta | low_ratio | high_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | both | 0.12205316245870751 | 0.8490567526805992 | 0.0010775614280524454 | 0.2611416398187342 | 0.2638143172781727 | 0.2773215067164773 | 0.10462152203326305 | 0.13278410076879146 | 0.09622222485165156 | 0.06343822929635644 | 0.01716998299653642 | 0.6588110572695732 | 1.0436413599401713 |
| shuffle | both | 0.1212763077369966 | 0.8523574394866202 | 0.0003007067063415386 | 0.2623648877519317 | 0.2622923610810758 | 0.269874657823988 | 0.10451852727678052 | 0.13847696452242156 | 0.09637372228767646 | 0.06390454685967416 | 0.01804094842588529 | 0.6810292302668095 | 0.9922378859817982 |
| true | both | 0.11727715445647811 | 0.8604269477100324 | -0.00369844657417695 | 0.24309784451661057 | 0.2533408199173369 | 0.2604315246590059 | 0.10299681289143413 | 0.11831445955335475 | 0.09619027758433964 | 0.06019652613066137 | 0.016516755028627813 | 0.6469549541771412 | 1.0039352102577686 |
| zero | both | 0.12367251282883175 | 0.8434971853550892 | 0.0026969117981766877 | 0.25961176570172234 | 0.2570481447013855 | 0.2950208514818695 | 0.10945085114695935 | 0.13391052311190305 | 0.0959792890594671 | 0.12268796339165419 | 0.033880523376632485 | 0.8143501342236996 | 0.667268808066845 |

D0/D1 invariance check:
- tolerance: 1e-07
- ok: True
- D0 max diffs: {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0, 'silog_x100': 0.0, 'd1': 0.0, 'd2': 0.0, 'd3': 0.0}
- D1 max diffs: {'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0, 'silog_x100': 0.0, 'd1': 0.0, 'd2': 0.0, 'd3': 0.0}

Threshold check:
- overall true-shuffle improvement: 0.003999153280518489 pass=True
- boundary true-shuffle improvement: 0.019267043235321107 pass=True
- saturated true-shuffle improvement: 0.02016250496906681 pass=True

## P1 N7-zero-x3-train vs N7 true

| method | selected ckpt | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | saturated | KITTI abs_rel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N7 | /mnt/drive/3333_raw/0000_exp_ckpt/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_09.pth | 0.11729256744108649 | 0.8603872998098767 | -0.0037013088180168213 | 0.243164209949122 | 0.2533209184689911 | 0.2605253634921313 | 0.11852521828034376 | 0.09622246028771116 |

## P2 N7-RGB vs N7 true

| method | selected ckpt | VK abs_rel | VK d1 | final-D1 | boundary | high-error | far50 | saturated | KITTI abs_rel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N7 | /mnt/drive/3333_raw/0000_exp_ckpt/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_09.pth | 0.11729256744108649 | 0.8603872998098767 | -0.0037013088180168213 | 0.243164209949122 | 0.2533209184689911 | 0.2605253634921313 | 0.11852521828034376 | 0.09622246028771116 |

## Interpretation guardrails

If true ~= shuffle/zero/mean, N7 cannot be used as evidence that image-corresponding x3 matters.
If N7 true ~= N7-zero-x3-train, N7 improvement is mostly D1-conditioned head capacity, not x3.
If N7 true ~= N7RGB, RAW/RAM x3 is effective but not clearly better than matched RGB cue under clean VKITTI.
