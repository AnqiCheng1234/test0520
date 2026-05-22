# Formal Scripts

This directory is reserved for maintained scripts that target the 0522 project scope.

Current policy:

- Use `--stage stf_only` for training runs.
- Use `--stage eval_only` or `--eval-only` for evaluation-only runs.
- Do not add raw mix, VKITTI mix, HyperSim mix, or LOD mix training scripts here.
- Formal experiment names must start with the launch timestamp in `MMDD_HHMM` format.
- Maintained formal scripts must use orthogonal resolved-config flags instead of `--input-type`.
- Required top-level semantic flags: `--input-domain`, `--front-end`, `--dataset-family`, `--dataset-input-mode`, `--model-input-tensor`, `--input-height`, `--input-width`, `--raw-storage-format`, `--bridge`, `--decoder-feature-adapter`, `--lora`, `--dav2-train-mode`, and `--loss-type`.
- Only pass feature-specific sub-parameters when the feature is enabled. For example, bridge/adaptor scripts pass feature source channels, keys, layers, and source; `lora=none` scripts do not pass LoRA sub-parameters.
- Smoke scripts live in `finetune_stf/scripts/smoke/` and must write only to paths containing `smoke`, `debug`, `tmp`, or `codex_smoke`.

Current maintained entries:

- `0522_run_stf_ram_feature_adapter_bridge_from_0521_1542_queue.sh`: `front_end=raw_ram4`, `bridge=x4`, `decoder_feature_adapter=x4`.
- `0522_run_stf_ram_rgb_feature_adapter_bridge_from_0521_1542_queue.sh`: `front_end=raw_to_base_rgb_ram3`, `bridge=x3`, `decoder_feature_adapter=x3`, with `PHASE1_BNCLEAN_REVIEWED=1`.
