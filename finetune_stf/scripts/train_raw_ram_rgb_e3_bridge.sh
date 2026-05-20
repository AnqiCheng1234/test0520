#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CONDA_ENV="${CONDA_ENV:-dav3}"
MASTER_PORT="${MASTER_PORT:-29772}"
GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
BRIDGE_INIT_FROM="${BRIDGE_INIT_FROM:-}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
LOD_ROOT="${LOD_ROOT:-/mnt/drive/3333_raw/LOD}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${REPO_ROOT}/finetune_stf/dataset/splits/vkitti2/train.txt}"
VKITTI_UNPROCESSING_PRESET="${VKITTI_UNPROCESSING_PRESET:-sensor_linear_dual}"
VKITTI_UNPROCESSING_MIX_WEIGHTS="${VKITTI_UNPROCESSING_MIX_WEIGHTS:-0.5,0.5}"
ETH3D_ROOT="${ETH3D_ROOT:-/mnt/drive/3333_raw/eth3d_raw_depth_640960}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-5}"
BRIDGE_LR="${BRIDGE_LR:-5e-5}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-ram_core}"
LOD_PER_VKITTI="${LOD_PER_VKITTI:-1}"
BRIDGE_FEATURE_KEYS=(${BRIDGE_FEATURE_KEYS:-x_cat ffm_mid x3})
BRIDGE_LAYERS=(${BRIDGE_LAYERS:-4 11 17 23})
TS="${TS:-$(date +%m%d_%H%M)}"
RUN_TAG="${RUN_TAG:-${TS}_vkitti_lod_mixed_raw_ram_rgb_bridge_644x1008_bs4acc4_ssi_lod1_eth3d_best_e10}"
SAVE_PATH="${SAVE_PATH:-${EXP_ROOT}/${RUN_TAG}}"

EXTRA_ARGS=()
if [[ -n "${BRIDGE_INIT_FROM}" ]]; then
  EXTRA_ARGS+=(--bridge-init-from "${BRIDGE_INIT_FROM}")
fi

mkdir -p "${SAVE_PATH}"
cd "${REPO_ROOT}"

echo "[RUN] starting: ${RUN_TAG}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${GPUS}" --master_port "${MASTER_PORT}" finetune_stf/train.py \
  --stage vkitti_lod \
  --input-type raw_ram_rgb_bridge \
  --encoder "${ENCODER}" \
  --dav2-train-mode none \
  --loss-type ssi \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${PRETRAINED_FROM}" \
  --stf-root "${STF_ROOT}" \
  --raw-npz-root "${RAW_NPZ_ROOT}" \
  --lod-root "${LOD_ROOT}" \
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \
  --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
  --vkitti-unprocessing-preset "${VKITTI_UNPROCESSING_PRESET}" \
  --vkitti-unprocessing-mix-weights "${VKITTI_UNPROCESSING_MIX_WEIGHTS}" \
  --lod-per-vkitti "${LOD_PER_VKITTI}" \
  --input-height 644 \
  --input-width 1008 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --norm-mode sensor_linear \
  --channel-mode rgb_avg_g \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --bridge-lr "${BRIDGE_LR}" \
  --bridge-source "${BRIDGE_SOURCE}" \
  --bridge-feature-keys "${BRIDGE_FEATURE_KEYS[@]}" \
  --bridge-layers "${BRIDGE_LAYERS[@]}" \
  --eval-eth3d \
  --eth3d-root "${ETH3D_ROOT}" \
  --eth3d-eval-mode fast \
  --eth3d-fast-eval-backend proxy \
  --eval-robotcar \
  --robotcar-root "${ROBOTCAR_ROOT}" \
  --robotcar-eval-mode fast \
  --robotcar-fast-eval-backend sparse \
  --best-metric eth3d \
  --amp \
  --amp-dtype bf16 \
  --log-interval 250 \
  --save-path "${SAVE_PATH}" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee "${SAVE_PATH}/train.log"

echo "[RUN] finished: ${RUN_TAG}"
