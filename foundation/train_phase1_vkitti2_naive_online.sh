#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-/home/caq/6666_raw/dav2_raw/finetune_stf/dataset/splits/vkitti2/train.txt}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-5}"
TS="${TS:-$(date +%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-${TS}_phase1_vkitti2_naive_rand_on_bs${BS}_acc${ACCUM_STEPS}}"
SAVE_PATH="${SAVE_PATH:-foundation/exp/${RUN_TAG}}"

python foundation/tools/train_phase1_vkitti2_naive.py \
  --encoder "${ENCODER}" \
  --pretrained-from "${PRETRAINED_FROM}" \
  --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
  --input-height 518 \
  --input-width 966 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --lr "${LR}" \
  --num-workers "${NUM_WORKERS}" \
  --save-path "${SAVE_PATH}" \
  --freeze-backbone \
  --randomize-unprocessing \
  "$@"
