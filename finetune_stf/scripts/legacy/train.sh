#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

STAGE="${1:-stf_only}"
GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
SAVE_ROOT="${SAVE_ROOT:-finetune_stf/exp}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
STF_REPEAT="${STF_REPEAT:-7}"

COMMON_ARGS=(
  --encoder "${ENCODER}"
  --pretrained-from "${PRETRAINED_FROM}"
  --stf-root "${STF_ROOT}"
  --min-depth 1.0
  --max-depth 80.0
  --input-height 518
  --input-width 966
  --epochs "${EPOCHS}"
  --lr 1e-5
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
)

case "${STAGE}" in
  stf_only)
    torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
      --stage stf_only \
      --bs "${BS}" \
      --save-path "${SAVE_ROOT}/stf_rel_${ENCODER}" \
      "${COMMON_ARGS[@]}"
    ;;
  mixed)
    torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
      --stage mixed \
      --bs "${BS}" \
      --stf-repeat "${STF_REPEAT}" \
      --save-path "${SAVE_ROOT}/stf_rel_${ENCODER}_mixed" \
      "${COMMON_ARGS[@]}"
    ;;
  smoke)
    torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
      --stage stf_only \
      --bs 2 \
      --epochs 1 \
      --debug-max-train-steps 10 \
      --debug-max-val-samples 16 \
      --save-path "${SAVE_ROOT}/stf_rel_${ENCODER}_smoke" \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Usage: $0 {stf_only|mixed|smoke}" >&2
    exit 1
    ;;
esac
