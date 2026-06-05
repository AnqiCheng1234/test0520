#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTAMP="${TIMESTAMP:-$(date +%m%d_%H%M)}"
GPUS="${GPUS:-1}"
ENCODER="${ENCODER:-vitl}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
VKITTI2_ROOT="${VKITTI2_ROOT:-/mnt/drive/1111_new_works/VKITTI2}"
VKITTI2_TRAIN_LIST="${VKITTI2_TRAIN_LIST:-finetune_stf/dataset/splits/vkitti2/train.txt}"
SAVE_ROOT="${SAVE_ROOT:-finetune_stf/exp}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INPUT_HEIGHT="${INPUT_HEIGHT:-518}"
INPUT_WIDTH="${INPUT_WIDTH:-966}"
MIN_DEPTH="${MIN_DEPTH:-1.0}"
MAX_DEPTH="${MAX_DEPTH:-80.0}"

PHASEA_EPOCHS="${PHASEA_EPOCHS:-20}"
PHASEA_BS="${PHASEA_BS:-10}"
PHASEA_LR="${PHASEA_LR:-1e-5}"

PHASEB_EPOCHS="${PHASEB_EPOCHS:-20}"
PHASEB_BS="${PHASEB_BS:-10}"
PHASEB_LR="${PHASEB_LR:-1e-5}"
PHASEB_STF_REPEAT="${PHASEB_STF_REPEAT:-7}"
PHASEB_INIT="${PHASEB_INIT:-official}"

FALLBACK_EPOCHS="${FALLBACK_EPOCHS:-10}"
FALLBACK_BS="${FALLBACK_BS:-10}"
FALLBACK_LR="${FALLBACK_LR:-5e-6}"

PHASEA_SAVE="${SAVE_ROOT}/${TIMESTAMP}_stf_rel_${ENCODER}_phaseA"
PHASEB_SAVE="${SAVE_ROOT}/${TIMESTAMP}_stf_rel_${ENCODER}_phaseB"
FALLBACK_SAVE="${SAVE_ROOT}/${TIMESTAMP}_stf_rel_${ENCODER}_phaseA_fallback"

common_args=(
  --encoder "${ENCODER}"
  --stf-root "${STF_ROOT}"
  --min-depth "${MIN_DEPTH}"
  --max-depth "${MAX_DEPTH}"
  --input-height "${INPUT_HEIGHT}"
  --input-width "${INPUT_WIDTH}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
)

check_vkitti2_ready() {
  [[ -d "${VKITTI2_ROOT}/rgb" ]] &&
  [[ -d "${VKITTI2_ROOT}/depth" ]] &&
  [[ -f "${VKITTI2_TRAIN_LIST}" ]] &&
  [[ -s "${VKITTI2_TRAIN_LIST}" ]]
}

echo "[pipeline] timestamp=${TIMESTAMP}"
echo "[pipeline] phase A save path: ${PHASEA_SAVE}"
echo "[pipeline] phase B save path: ${PHASEB_SAVE}"
echo "[pipeline] fallback save path: ${FALLBACK_SAVE}"

torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
  --stage stf_only \
  --pretrained-from "${PRETRAINED_FROM}" \
  --epochs "${PHASEA_EPOCHS}" \
  --bs "${PHASEA_BS}" \
  --lr "${PHASEA_LR}" \
  --save-path "${PHASEA_SAVE}" \
  "${common_args[@]}"

PHASEA_MODEL="${PHASEA_SAVE}/best_model.pth"
if [[ ! -f "${PHASEA_MODEL}" ]]; then
  echo "[pipeline] phase A did not produce ${PHASEA_MODEL}" >&2
  exit 1
fi

if check_vkitti2_ready; then
  PHASEB_PRETRAINED="${PRETRAINED_FROM}"
  if [[ "${PHASEB_INIT}" == "phasea" ]]; then
    PHASEB_PRETRAINED="${PHASEA_MODEL}"
  elif [[ "${PHASEB_INIT}" != "official" ]]; then
    echo "[pipeline] unsupported PHASEB_INIT=${PHASEB_INIT} (expected official|phasea)" >&2
    exit 1
  fi
  echo "[pipeline] VKITTI2 is ready, starting phase B mixed training."
  echo "[pipeline] phase B init: ${PHASEB_INIT} -> ${PHASEB_PRETRAINED}"
  torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
    --stage mixed \
    --pretrained-from "${PHASEB_PRETRAINED}" \
    --vkitti-train-list "${VKITTI2_TRAIN_LIST}" \
    --epochs "${PHASEB_EPOCHS}" \
    --bs "${PHASEB_BS}" \
    --lr "${PHASEB_LR}" \
    --stf-repeat "${PHASEB_STF_REPEAT}" \
    --save-path "${PHASEB_SAVE}" \
    "${common_args[@]}"
else
  echo "[pipeline] VKITTI2 is not ready, running STF-only fallback from phase A best model."
  torchrun --nproc_per_node="${GPUS}" finetune_stf/train.py \
    --stage stf_only \
    --pretrained-from "${PHASEA_MODEL}" \
    --epochs "${FALLBACK_EPOCHS}" \
    --bs "${FALLBACK_BS}" \
    --lr "${FALLBACK_LR}" \
    --save-path "${FALLBACK_SAVE}" \
    "${common_args[@]}"
fi
