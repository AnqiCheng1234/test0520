#!/usr/bin/env bash
set -euo pipefail
export RUN_NAME="${RUN_NAME:-0430_b2_vkitti_lod_dn_bridge_feature_adapter_lora_decoder_sigmoid_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_e10}"
export INPUT_TYPE="${INPUT_TYPE:-raw_ram_bridge_feature_adapter_lora}"
export DAV2_TRAIN_MODE="${DAV2_TRAIN_MODE:-decoder}"
export RGB_INTERFACE_MODE="${RGB_INTERFACE_MODE:-sigmoid}"
export LORA_BLOCK_MODE="${LORA_BLOCK_MODE:-tap}"
export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export BS="${BS:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-4}"
exec "$(dirname "$0")/0430_common_train.sh" "$@"
