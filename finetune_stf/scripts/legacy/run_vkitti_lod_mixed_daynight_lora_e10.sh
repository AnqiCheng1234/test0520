#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_0424"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CONDA_ENV="${CONDA_ENV:-dav3}"
MASTER_PORT="${MASTER_PORT:-29777}"
GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
LOD_ROOT="${LOD_ROOT:-/mnt/drive/3333_raw/LOD}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest.csv}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${REPO_ROOT}/finetune_stf/dataset/splits/vkitti2/train.txt}"
VKITTI_UNPROCESSING_PRESET="${VKITTI_UNPROCESSING_PRESET:-sensor_linear_dual}"
VKITTI_UNPROCESSING_MIX_WEIGHTS="${VKITTI_UNPROCESSING_MIX_WEIGHTS:-0.5,0.5}"
ETH3D_ROOT="${ETH3D_ROOT:-/mnt/drive/3333_raw/eth3d_raw_depth_640960}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100}"
ROBOTCAR_NIGHT_ROOT="${ROBOTCAR_NIGHT_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo}"
ROBOTCAR_NIGHT_MANIFEST_NAME="${ROBOTCAR_NIGHT_MANIFEST_NAME:-robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv}"
ROBOTCAR_NIGHT_MAX_SAMPLES="${ROBOTCAR_NIGHT_MAX_SAMPLES:-}"
KITTI_EVAL_PROTOCOL="${KITTI_EVAL_PROTOCOL:-rgb_checkpoint_decoder}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-5}"
BRIDGE_LR="${BRIDGE_LR:-5e-5}"
LORA_LR="${LORA_LR:-5e-5}"
LORA_BLOCK_MODE="${LORA_BLOCK_MODE:-tap}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LOD_PER_VKITTI="${LOD_PER_VKITTI:-3}"
BRIDGE_FEATURE_KEYS=(${BRIDGE_FEATURE_KEYS:-x_cat ffm_mid x4})
TS="${TS:-$(date +%m%d_%H%M)}"
RUN_NAME="${RUN_NAME:-${TS}_vkitti_lod_dn_lora_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_best_e10}"
RUN_DIR="${RUN_DIR:-${EXP_ROOT}/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${ROBOTCAR_NIGHT_MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--robotcar-night-max-samples "${ROBOTCAR_NIGHT_MAX_SAMPLES}")
fi

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

echo "[RUN] starting: ${RUN_NAME}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${GPUS}" --master_port "${MASTER_PORT}" finetune_stf/train.py \
  --stage vkitti_lod \
  --input-type raw_ram_bridge_lora \
  --encoder "${ENCODER}" \
  --dav2-train-mode decoder \
  --loss-type ssi \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${PRETRAINED_FROM}" \
  --lod-root "${LOD_ROOT}" \
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \
  --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
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
  --lora-lr "${LORA_LR}" \
  --lora-block-mode "${LORA_BLOCK_MODE}" \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --bridge-feature-keys "${BRIDGE_FEATURE_KEYS[@]}" \
  --no-eval-stf \
  --eval-eth3d \
  --eth3d-root "${ETH3D_ROOT}" \
  --eth3d-eval-mode fast \
  --eth3d-fast-eval-backend proxy \
  --eval-robotcar \
  --robotcar-root "${ROBOTCAR_ROOT}" \
  --robotcar-eval-mode fast \
  --robotcar-fast-eval-backend sparse \
  --eval-robotcar-night \
  --robotcar-night-root "${ROBOTCAR_NIGHT_ROOT}" \
  --robotcar-night-manifest-name "${ROBOTCAR_NIGHT_MANIFEST_NAME}" \
  --robotcar-night-fast-eval-backend sparse \
  --eval-kitti \
  --kitti-eval-protocol "${KITTI_EVAL_PROTOCOL}" \
  --best-metric eth3d \
  --amp \
  --amp-dtype bf16 \
  --log-interval 250 \
  --save-path "${RUN_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[RUN] finished: ${RUN_NAME}"
