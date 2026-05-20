#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TS="${TS:-$(date +%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-${TS}_e1_raw_naive_bs${BS}_acc${ACCUM_STEPS}}"
SAVE_PATH="${SAVE_PATH:-finetune_stf/exp/${RUN_TAG}}"

torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
  --stage stf_only \
  --input-type raw \
  --encoder "${ENCODER}" \
  --pretrained-from "${PRETRAINED_FROM}" \
  --stf-root "${STF_ROOT}" \
  --raw-npz-root "${RAW_NPZ_ROOT}" \
  --input-height 518 \
  --input-width 966 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --norm-mode companded \
  --channel-mode rgb_avg_g \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --lr 1e-5 \
  --num-workers "${NUM_WORKERS}" \
  --save-path "${SAVE_PATH}" \
  "$@"
