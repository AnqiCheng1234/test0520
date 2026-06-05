# Scope Cleanup And Legacy Inventory 0522

Created: 2026-05-22 19:11:37 CST

## Maintained Training Scope

The new training entrypoint `finetune_stf/train.py` now supports:

- `--stage stf_only`
- `--stage eval_only`
- `--eval-only` as a compatibility alias that resolves to `stage=eval_only`

STF train/val and existing eval datasets remain available. Raw mix, VKITTI mix, HyperSim mix, and LOD/VKITTI mix training are no longer active paths in the new training entrypoint.

## Removed From Active Train Entrypoint

- `raw_mix`, `mixed`, `vkitti_only`, `lod_only`, and `vkitti_lod` stage choices
- `--train-sources`
- `--train-source-ratios`
- `--train-steps-per-epoch`
- `--lod-per-vkitti`
- `--lod-fraction`
- `--vkitti-cache-root`
- `--vkitti-randomize-unprocessing`
- `--vkitti-unprocessing-preset`
- `--vkitti-unprocessing-mix-weights`
- HyperSim raw train CLI parameters and dataset construction
- VKITTI train dataset construction
- LOD train dataset construction
- Mixed training schedules and mixed source iteration

## Script Layout

- Maintained scripts: `finetune_stf/scripts/formal/`
- Historical 0520-era shell launch scripts: `finetune_stf/scripts/legacy/`
- Python analysis helpers remain at `finetune_stf/scripts/` for now.

## Verification Targets

Run from `/home/caq/6666_raw/dav2_raw_0522`:

```bash
rg -n "raw_mix|train_sources|train_source_ratios|HypersimProcessedRaw|VKITTI2Raw" finetune_stf/train.py
conda run -n dav3 python -m compileall finetune_stf foundation
conda run -n dav3 python finetune_stf/train.py --help >/tmp/dav2_raw_0522_train_help_codex_smoke.txt
```

The temporary help output is a smoke artifact and should be deleted after successful verification.

## Verification Results

Completed on 2026-05-22 19:11 CST:

- `rg -n "raw_mix|train_sources|train_source_ratios|HypersimProcessedRaw|VKITTI2Raw" finetune_stf/train.py` returned no matches.
- `rg -n -- "--stage (raw_mix|vkitti_lod|mixed|vkitti_only|lod_only)|raw_mix|vkitti_lod|--train-sources|--lod-per-vkitti|HypersimProcessedRaw|VKITTI2Raw|CachedVKITTI2Raw" finetune_stf/scripts/formal` returned no matches.
- `conda run -n dav3 python -m compileall finetune_stf foundation` passed.
- `conda run -n dav3 python finetune_stf/train.py --help` passed.
- `/tmp/dav2_raw_0522_train_help_codex_smoke.txt` was deleted after the successful help smoke.
- `bash -n` passed for both formal shell scripts.
- `--stage raw_mix` was verified to fail at argument parsing; temporary invalid-stage smoke log was deleted.
