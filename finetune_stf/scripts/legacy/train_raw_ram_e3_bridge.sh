#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
BRIDGE_INIT_FROM="${BRIDGE_INIT_FROM:-}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BRIDGE_LR="${BRIDGE_LR:-5e-5}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-ram_core}"
BRIDGE_FEATURE_KEYS=(${BRIDGE_FEATURE_KEYS:-x_cat ffm_mid x4})
BRIDGE_LAYERS=(${BRIDGE_LAYERS:-4 11 17 23})
TS="${TS:-$(date +%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-${TS}_e3_raw_ram_bridge_bs${BS}_acc${ACCUM_STEPS}}"
SAVE_PATH="${SAVE_PATH:-finetune_stf/exp/${RUN_TAG}}"

EXTRA_ARGS=()
if [[ -n "${BRIDGE_INIT_FROM}" ]]; then
  EXTRA_ARGS+=(--bridge-init-from "${BRIDGE_INIT_FROM}")
fi

torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
  --stage stf_only \
  --input-type raw_ram_bridge \
  --encoder "${ENCODER}" \
  --pretrained-from "${PRETRAINED_FROM}" \
  --stf-root "${STF_ROOT}" \
  --raw-npz-root "${RAW_NPZ_ROOT}" \
  --input-height 518 \
  --input-width 966 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --norm-mode companded \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --lr 1e-5 \
  --bridge-lr "${BRIDGE_LR}" \
  --bridge-source "${BRIDGE_SOURCE}" \
  --bridge-feature-keys "${BRIDGE_FEATURE_KEYS[@]}" \
  --bridge-layers "${BRIDGE_LAYERS[@]}" \
  --num-workers "${NUM_WORKERS}" \
  --save-path "${SAVE_PATH}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
