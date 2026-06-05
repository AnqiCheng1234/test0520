# Migration Manifest 0522

Created: 2026-05-22 19:11:37 CST

## Paths

- Source: `/home/caq/6666_raw/dav2_raw_0520`
- Target: `/home/caq/6666_raw/dav2_raw_0522`
- Follow-up plan source: `/home/caq/6666_raw/dav2_raw_0520/plans/0522_fix_code/`

## Copy Policy

Command pattern:

```bash
rsync -a --info=progress2 \
  --exclude 'finetune_stf/exp/' \
  --exclude 'plans/' \
  --exclude 'runs/' \
  --exclude 'wandb/' \
  --exclude 'logs/' \
  --exclude 'codex_debug/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  /home/caq/6666_raw/dav2_raw_0520/ \
  /home/caq/6666_raw/dav2_raw_0522/
```

Excluded from the new project:

- `finetune_stf/exp/`
- `plans/`
- `runs/`
- `wandb/`
- `logs/`
- `codex_debug/`
- Python and pytest cache directories

## Git Metadata

The copied `.git` directory was removed from the new project after migration. This avoids a misleading dirty worktree caused by intentionally excluded tracked `plans/` files. The 0520 project remains the audit and old-experiment reference.

## Size Check

- Target project size after post-review cleanup: `5.9M`
- `finetune_stf/`: `4.6M`
- `anqi_eval/`: `468K`
- `foundation/`: `208K`
- `configs/`: `344K`

## Notes

- No source datasets, pretrained checkpoints, or formal experiment outputs were deleted.
- Follow-up code restructuring is scoped to `/home/caq/6666_raw/dav2_raw_0522`.
- Post-review cleanup removed copied `anqi_eval/results/` visual/eval outputs and all generated `__pycache__/` directories.

## Verification Results

- `conda run -n dav3 python -m compileall finetune_stf foundation` passed.
- `conda run -n dav3 python finetune_stf/train.py --help` passed.
- `/tmp/dav2_raw_0522_train_help_codex_smoke.txt` was deleted after the successful help smoke.
