#!/usr/bin/env bash
set -euo pipefail
export RUN_NAME="${RUN_NAME:-0430_c4_vkitti_lod_dn_bridge_first8_644x1008_bs2acc8_ssi_lod3_eth3d_kitti_rnight_e10}"
export INPUT_TYPE="${INPUT_TYPE:-raw_ram_bridge}"
export DAV2_TRAIN_MODE="${DAV2_TRAIN_MODE:-first:8}"
export RGB_INTERFACE_MODE="${RGB_INTERFACE_MODE:-sigmoid}"
export BS="${BS:-2}"
export ACCUM_STEPS="${ACCUM_STEPS:-8}"
exec "$(dirname "$0")/0430_common_train.sh" "$@"
