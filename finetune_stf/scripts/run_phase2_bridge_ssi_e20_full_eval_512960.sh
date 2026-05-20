#!/usr/bin/env bash
set -euo pipefail

# Phase2 bridge frontend-only follow-up:
#   - drop feature-adapter path; use raw_ram_bridge (best frontend-only structure so far)
#   - simplify loss back to plain SSI (no grad term) per 2026-04-20 plan
#   - extend training to 20 epochs since the ssi_grad e10 run still trended downward
#     at epoch 8 (best=0.1263), suggesting room with more iterations
#   - enable all three downstream evals (STF + ETH3D fast/proxy + RobotCar fast/sparse)
#     and only keep the STF-best ckpt as the canonical model

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CKPT="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"
CONDA_ENV="${CONDA_ENV:-dav3}"
TS="${TS:-$(date +%m%d_%H%M)}"

RUN_NAME="${TS}_phase2_vkitti_only_raw_ram_bridge_frontend_only_512960_bs4acc4_full_eval_v2calib_loss_ssi_e20"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"

echo "[RUN] starting: ${RUN_NAME}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node=1 --master_port 29761 finetune_stf/train.py \
  --encoder vitl \
  --stage vkitti_only \
  --input-type raw_ram_bridge \
  --dav2-train-mode none \
  --loss-type ssi \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN_DIR}" \
  --input-height 512 \
  --input-width 960 \
  --bs 4 \
  --accum-steps 4 \
  --epochs 20 \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --bridge-source ram_core \
  --bridge-feature-keys x_cat ffm_mid x4 \
  --bridge-layers 4 11 17 23 \
  --norm-mode companded \
  --channel-mode rgb_avg_g \
  --eval-kitti \
  --eval-eth3d \
  --eth3d-eval-mode fast \
  --eth3d-fast-eval-backend proxy \
  --eval-robotcar \
  --robotcar-eval-mode fast \
  --robotcar-fast-eval-backend sparse \
  --save-best-stf-only \
  --amp \
  --amp-dtype bf16 \
  --vkitti-randomize-unprocessing \
  --log-interval 500 \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[RUN] finished: ${RUN_NAME}"
