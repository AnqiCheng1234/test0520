# 0606_0330 vs 0606_1431 loss and eval summary

Source logs: `train.log`; eval split: `rod_night_val`; train points are logged every 500 optimizer steps.

| exp | best AbsRel | best epoch | final epoch | final AbsRel | final RMSE | final SILog | final d1 | final avg_loss | AbsRel drop e0->best | pretrain AbsRel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0606_0330 decoder | 4.2575 | 3 | 9 | 5.5728 | 12.0940 | 1.0218 | 0.8700 | 0.012158 | 69.1% | 130.3823 |
| 0606_1431 lowlr | 6.4195 | 7 | 9 | 6.4649 | 13.4667 | 1.0279 | 0.8562 | 0.016309 | 35.2% | 130.3823 |

Notes:
- 0606_1431 uses 0.1x lr for the backbone and DAV2 decoder parameter groups compared with 0606_0330, according to `resolved_config.json`.
- Main eval plots omit the pretrain point because its AbsRel is 130.3823 for both experiments and would compress the epoch 0-9 trends.
- Lower is better for AbsRel/RMSE/SILog; higher is better for d1/d2/d3.

Generated files:
- `loss_curves.png`
- `eval_trends.png`
- `absrel_zoom.png`
- `train_loss_points.csv`
- `epoch_loss_eval.csv`
- `pretrain_eval.csv`
