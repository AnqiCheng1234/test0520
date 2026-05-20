#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CONDA_ENV="${CONDA_ENV:-dav3}"
CKPT="${CKPT:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
TS="${TS:-$(date +%m%d_%H%M)}"
MASTER_PORT="${MASTER_PORT:-29691}"
QUAL_NUM_SAMPLES="${QUAL_NUM_SAMPLES:-10}"

RUN_NAME="${RUN_NAME:-${TS}_phase2_vkitti_only_raw_ram_residual_frontend_only_512960_bs4acc4_dual_eval_v2calib_loss_ssi_grad_e10}"
RUN_DIR="${SAVE_PATH:-${EXP_ROOT}/${RUN_NAME}}"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

echo "[RUN] residual-head control save path: ${RUN_DIR}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node=1 --master_port "${MASTER_PORT}" finetune_stf/train.py \
  --encoder vitl \
  --stage vkitti_only \
  --input-type raw_ram_residual \
  --dav2-train-mode none \
  --loss-type ssi_grad \
  --loss-lambda-grad 2.0 \
  --loss-grad-scales 4 \
  --loss-mask-downsample strict \
  --pretrained-from "${CKPT}" \
  --save-path "${RUN_DIR}" \
  --bs 4 \
  --accum-steps 4 \
  --epochs 10 \
  --lr 1e-5 \
  --eval-kitti \
  --amp \
  --amp-dtype bf16 \
  --vkitti-randomize-unprocessing \
  --log-interval 200 \
  2>&1 | tee "${RUN_DIR}/train.log"

if [[ -f "${RUN_DIR}/best_model.pth" ]]; then
  echo "[RUN] dumping Appendix-A triplets"
  conda run -n "${CONDA_ENV}" python anqi_eval/dump_stf_rgb_triplets.py \
    "${RUN_DIR}" \
    --split val \
    --checkpoint best \
    --num-samples "${QUAL_NUM_SAMPLES}"

  echo "[RUN] dumping qualitative predictions"
  conda run -n "${CONDA_ENV}" python anqi_eval/visualize_stf_predictions.py \
    "${RUN_DIR}" \
    --split val \
    --checkpoint best \
    --num-samples "${QUAL_NUM_SAMPLES}"
fi
