#!/usr/bin/env bash
set -euo pipefail
export RUN_NAME="${RUN_NAME:-0430_a3_vkitti_lod_dn_raw4_residual_tanh_s1_bridge_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_e10}"
export INPUT_TYPE="${INPUT_TYPE:-raw_ram_bridge}"
export DAV2_TRAIN_MODE="${DAV2_TRAIN_MODE:-decoder}"
export RGB_INTERFACE_MODE="${RGB_INTERFACE_MODE:-residual_tanh}"
export RGB_RESIDUAL_SCALE="${RGB_RESIDUAL_SCALE:-1.0}"
export BS="${BS:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-4}"
exec "$(dirname "$0")/0430_common_train.sh" "$@"
