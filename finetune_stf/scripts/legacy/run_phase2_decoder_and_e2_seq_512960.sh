#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CKPT="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth"
TS="${TS:-$(date +%m%d_%H%M)}"

RUN1_NAME="${TS}_phase2_vkitti_only_raw_ram_bridge_frontend_decoder_512960_bs2acc8_dual_eval_v2calib_loss_ssi_grad_e10"
RUN2_NAME="${TS}_e2_vkitti_only_raw_ram_feature_adapter_decoder_512960_bs2acc8_dual_eval_v2calib_loss_ssi_grad_e10"

RUN1_DIR="${EXP_ROOT}/${RUN1_NAME}"
RUN2_DIR="${EXP_ROOT}/${RUN2_NAME}"

mkdir -p "${RUN1_DIR}" "${RUN2_DIR}"

cd "${REPO_ROOT}"

echo "[SEQ] starting run 1/2: ${RUN1_NAME}"
conda run -n dav3 torchrun --nproc_per_node=1 --master_port 29681 finetune_stf/train.py \
  --encoder vitl \
  --stage vkitti_only \
  --input-type raw_ram_bridge \
  --dav2-train-mode decoder \
  --loss-type ssi_grad \
  --loss-lambda-grad 2.0 \
  --loss-grad-scales 4 \
  --loss-mask-downsample strict \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN1_DIR}" \
  --bs 2 \
  --accum-steps 8 \
  --epochs 10 \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --eval-kitti \
  --amp \
  --amp-dtype bf16 \
  --vkitti-randomize-unprocessing \
  --log-interval 200 \
  2>&1 | tee "${RUN1_DIR}/train.log"

echo "[SEQ] finished run 1/2: ${RUN1_NAME}"
echo "[SEQ] starting run 2/2: ${RUN2_NAME}"
conda run -n dav3 torchrun --nproc_per_node=1 --master_port 29682 finetune_stf/train.py \
  --encoder vitl \
  --stage vkitti_only \
  --input-type raw_ram_feature_adapter \
  --dav2-train-mode decoder \
  --loss-type ssi_grad \
  --loss-lambda-grad 2.0 \
  --loss-grad-scales 4 \
  --loss-mask-downsample strict \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN2_DIR}" \
  --bs 2 \
  --accum-steps 8 \
  --epochs 10 \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --eval-kitti \
  --amp \
  --amp-dtype bf16 \
  --vkitti-randomize-unprocessing \
  --log-interval 200 \
  2>&1 | tee "${RUN2_DIR}/train.log"

echo "[SEQ] finished run 2/2: ${RUN2_NAME}"
