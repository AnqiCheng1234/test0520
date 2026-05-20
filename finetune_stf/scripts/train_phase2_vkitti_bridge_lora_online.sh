#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
KITTI_BASE="${KITTI_BASE:-/mnt/drive/kitti}"
KITTI_VAL_SPLIT="${KITTI_VAL_SPLIT:-/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-/home/caq/6666_raw/dav2_raw/finetune_stf/dataset/splits/vkitti2/train.txt}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-5}"
BRIDGE_LR="${BRIDGE_LR:-5e-5}"
LORA_LR="${LORA_LR:-5e-5}"
LORA_BLOCK_MODE="${LORA_BLOCK_MODE:-tap}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-ram_core}"
BRIDGE_FEATURE_KEYS=(${BRIDGE_FEATURE_KEYS:-x_cat ffm_mid x4})
BRIDGE_LAYERS=(${BRIDGE_LAYERS:-4 11 17 23})
TS="${TS:-$(date +%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-${TS}_phase2_vkitti_bridge_lora_online_${LORA_BLOCK_MODE}_r${LORA_RANK}_bs${BS}_acc${ACCUM_STEPS}}"
SAVE_PATH="${SAVE_PATH:-finetune_stf/exp/${RUN_TAG}}"

torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
  --stage vkitti_only \
  --input-type raw_ram_bridge_lora \
  --encoder "${ENCODER}" \
  --pretrained-from "${PRETRAINED_FROM}" \
  --stf-root "${STF_ROOT}" \
  --raw-npz-root "${RAW_NPZ_ROOT}" \
  --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
  --input-height 518 \
  --input-width 966 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --norm-mode companded \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --lr "${LR}" \
  --bridge-lr "${BRIDGE_LR}" \
  --lora-lr "${LORA_LR}" \
  --lora-block-mode "${LORA_BLOCK_MODE}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --bridge-source "${BRIDGE_SOURCE}" \
  --bridge-feature-keys "${BRIDGE_FEATURE_KEYS[@]}" \
  --bridge-layers "${BRIDGE_LAYERS[@]}" \
  --num-workers "${NUM_WORKERS}" \
  --eval-kitti \
  --kitti-base "${KITTI_BASE}" \
  --kitti-val-split "${KITTI_VAL_SPLIT}" \
  --amp \
  --amp-dtype bf16 \
  --save-path "${SAVE_PATH}" \
  "$@"
