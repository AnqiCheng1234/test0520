#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CKPT="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"
LOD_ROOT="/mnt/drive/3333_raw/LOD"
LOD_DAY_MANIFEST="${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv"
# Transitional RobotCar eval root:
#   - lms_front-only
#   - 480x640 rgb/raw eval inputs
#   - full-resolution 960x1280 sparse GT
#   - current local asset contains 65 quality-filtered subset samples
ROBOTCAR_ROOT="/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100"
CONDA_ENV="${CONDA_ENV:-dav3}"
TS="${TS:-$(date +%m%d_%H%M)}"

RUN_NAME="${TS}_lod_day_only_raw_ram_bridge_644x1008_bs4acc4_ssi_eth3d_best"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"

echo "[RUN] starting: ${RUN_NAME}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node=1 --master_port 29771 finetune_stf/train.py \
  --encoder vitl \
  --stage lod_only \
  --input-type raw_ram_bridge \
  --dav2-train-mode none \
  --loss-type ssi \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN_DIR}" \
  --lod-root "${LOD_ROOT}" \
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \
  --input-height 644 \
  --input-width 1008 \
  --bs 4 \
  --accum-steps 4 \
  --epochs 20 \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --bridge-source ram_core \
  --bridge-feature-keys x_cat ffm_mid x4 \
  --bridge-layers 4 11 17 23 \
  --norm-mode sensor_linear \
  --channel-mode rgb_avg_g \
  --eval-eth3d \
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
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[RUN] finished: ${RUN_NAME}"
