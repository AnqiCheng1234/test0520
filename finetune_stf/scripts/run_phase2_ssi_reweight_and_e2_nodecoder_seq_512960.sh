#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CKPT="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"
CONDA_ENV="${CONDA_ENV:-dav3}"
TS="${TS:-$(date +%m%d_%H%M)}"

# 2026-04-19 scalar check on the reference phase2 run:
#   loss_ssi  ~= 0.009
#   loss_grad ~= 0.118
# Keep grad guidance, but reduce its dominance so SSI has a comparable effect.
RUN1_LAMBDA_GRAD="${RUN1_LAMBDA_GRAD:-0.1}"

RUN1_NAME="${TS}_phase2_vkitti_only_raw_ram_bridge_frontend_only_512960_bs4acc4_dual_eval_v2calib_loss_ssi_grad_lg${RUN1_LAMBDA_GRAD//./p}_eth3d_fast_e10"
RUN2_NAME="${TS}_e2_vkitti_only_raw_ram_feature_adapter_nodecoder_512960_bs4acc4_dual_eval_v2calib_loss_ssi_grad_e10"

RUN1_DIR="${EXP_ROOT}/${RUN1_NAME}"
RUN2_DIR="${EXP_ROOT}/${RUN2_NAME}"

mkdir -p "${RUN1_DIR}" "${RUN2_DIR}"

cd "${REPO_ROOT}"

echo "[SEQ] starting run 1/2: ${RUN1_NAME}"
echo "[SEQ] run 1 uses loss_lambda_grad=${RUN1_LAMBDA_GRAD} so SSI and grad are closer in scale"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node=1 --master_port 29741 finetune_stf/train.py \
  --encoder vitl \
  --stage vkitti_only \
  --input-type raw_ram_bridge \
  --dav2-train-mode none \
  --loss-type ssi_grad \
  --loss-lambda-grad "${RUN1_LAMBDA_GRAD}" \
  --loss-grad-scales 4 \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN1_DIR}" \
  --input-height 512 \
  --input-width 960 \
  --bs 4 \
  --accum-steps 4 \
  --epochs 10 \
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
  --amp \
  --amp-dtype bf16 \
  --vkitti-randomize-unprocessing \
  --log-interval 200 \
  2>&1 | tee "${RUN1_DIR}/train.log"

echo "[SEQ] finished run 1/2: ${RUN1_NAME}"
echo "[SEQ] starting run 2/2: ${RUN2_NAME}"
echo "[SEQ] run 2 uses bs=4 accum=4; smoke-tested on 2026-04-19 with dav2_train_mode=none"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node=1 --master_port 29742 finetune_stf/train.py \
  --encoder vitl \
  --stage vkitti_only \
  --input-type raw_ram_feature_adapter \
  --dav2-train-mode none \
  --loss-type ssi_grad \
  --loss-lambda-grad 2.0 \
  --loss-grad-scales 4 \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN2_DIR}" \
  --input-height 512 \
  --input-width 960 \
  --bs 4 \
  --accum-steps 4 \
  --epochs 10 \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --bridge-feature-keys x_cat ffm_mid x4 \
  --norm-mode companded \
  --channel-mode rgb_avg_g \
  --eval-kitti \
  --amp \
  --amp-dtype bf16 \
  --vkitti-randomize-unprocessing \
  --log-interval 200 \
  2>&1 | tee "${RUN2_DIR}/train.log"

echo "[SEQ] finished run 2/2: ${RUN2_NAME}"
