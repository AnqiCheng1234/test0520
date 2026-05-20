#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_0424"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CONDA_ENV="${CONDA_ENV:-dav3}"
MASTER_PORT="${MASTER_PORT:-29831}"
GPUS="${GPUS:-1}"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-0430_c1_vkitti_lod_dn_bridge_full_lrd09_644x1008_bs2acc8_ssi_lod3_eth3d_kitti_rnight_e10}"
RUN_NAME="${RUN_NAME:-${SOURCE_RUN_NAME}_nyu_eval_only}"
RUN_DIR="${RUN_DIR:-${EXP_ROOT}/${RUN_NAME}}"
RESUME_FROM="${RESUME_FROM:-/mnt/drive/3333_raw/0000_exp_ckpt/${SOURCE_RUN_NAME}/current_model.pth}"

PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
LOD_ROOT="${LOD_ROOT:-/mnt/drive/3333_raw/LOD}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest.csv}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${REPO_ROOT}/finetune_stf/dataset/splits/vkitti2/train.txt}"
NYU_DIR="${NYU_DIR:-/mnt/drive/nyu/nyu_test}"
NYU_MAX_SAMPLES="${NYU_MAX_SAMPLES:-}"
DEBUG_MAX_KITTI_SAMPLES="${DEBUG_MAX_KITTI_SAMPLES:-}"

EXTRA_ARGS=()
if [[ -n "${NYU_MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--nyu-max-samples "${NYU_MAX_SAMPLES}")
fi
if [[ -n "${DEBUG_MAX_KITTI_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--debug-max-kitti-samples "${DEBUG_MAX_KITTI_SAMPLES}")
fi

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

echo "[RUN] eval-only NYUv2 RGB decoder: ${RUN_NAME}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${GPUS}" --master_port "${MASTER_PORT}" finetune_stf/train.py \
  --eval-only \
  --stage vkitti_lod \
  --input-type raw_ram_bridge \
  --encoder vitl \
  --dav2-train-mode full \
  --loss-type ssi \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${PRETRAINED_FROM}" \
  --resume-from "${RESUME_FROM}" \
  --lod-root "${LOD_ROOT}" \
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \
  --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
  --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
  --vkitti-unprocessing-preset sensor_linear_dual \
  --vkitti-unprocessing-mix-weights 0.5,0.5 \
  --lod-per-vkitti 3 \
  --input-height 644 \
  --input-width 1008 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --norm-mode sensor_linear \
  --channel-mode rgb_avg_g \
  --epochs 1 \
  --bs 2 \
  --accum-steps 8 \
  --num-workers "${NUM_WORKERS:-4}" \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --lora-lr 5e-5 \
  --bridge-feature-keys x_cat ffm_mid x4 \
  --no-eval-stf \
  --eval-kitti \
  --kitti-eval-protocol rgb_checkpoint_decoder \
  --eval-nyu \
  --nyu-dir "${NYU_DIR}" \
  --best-metric kitti \
  --amp \
  --amp-dtype bf16 \
  --log-interval 250 \
  --save-path "${RUN_DIR}" \
  --rgb-interface-mode sigmoid \
  --backbone-layer-decay 0.9 \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[RUN] finished eval-only NYUv2 RGB decoder: ${RUN_NAME}"
