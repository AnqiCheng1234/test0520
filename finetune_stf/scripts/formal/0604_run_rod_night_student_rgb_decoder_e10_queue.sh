#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
ROD_ROOT="${ROD_ROOT:-/mnt/drive/3333_raw/ROD}"
ROD_MANIFEST="${ROD_MANIFEST:-${ROD_ROOT}/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-rod_night_studentrgb_decoder_e10}"
MASTER_PORT="${MASTER_PORT:-29604}"

if [[ "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  run_timestamp="$(date +%m%d_%H%M)"
  session="${run_timestamp}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' ROD_ROOT='${ROD_ROOT}' ROD_MANIFEST='${ROD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' RUN_TIMESTAMP='${run_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
tmux session: ${session}
queue log: ${queue_log}
attach: tmux attach -t ${session}
monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}" "${HEAVY_ROOT}"
require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }; }

require_file "${PRETRAINED}"
require_file "${ROD_MANIFEST}"
require_dir "${ROD_ROOT}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}"
RUN_NAME="${RUN_TIMESTAMP}_rod_night_studentrgb_dav2s_decoder_e10"
SAVE="${EXP_ROOT}/${RUN_NAME}"
HEAVY="${HEAVY_ROOT}/${RUN_NAME}"
LOG="${LOG_ROOT}/${RUN_NAME}.tmux.log"

if [[ -e "${SAVE}" || -e "${HEAVY}" ]]; then
  echo "[ERROR] refusing to overwrite ${SAVE} or ${HEAVY}" >&2
  exit 2
fi

mkdir -p "${SAVE}"
echo "[FORMAL] run=${RUN_NAME}"
echo "[FORMAL] save=${SAVE}"
echo "[FORMAL] heavy_root=${HEAVY_ROOT}"
echo "[FORMAL] log=${LOG}"
echo "[FORMAL] manifest=${ROD_MANIFEST}"
echo "[FORMAL] teacher=teacher_bright_degreen_v1 label_space=inverse_relative student=student_dark_degreen_v1"
{
  echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} finetune_stf/train.py"
  printf ' %q' \
    --encoder vits \
    --pretrained-from "${PRETRAINED}" \
    --stage rod_only \
    --input-domain rgb \
    --front-end dav2_rgb \
    --model-input-tensor image \
    --dataset-family rod_raw_student_rgb \
    --dataset-input-mode raw24_student_rgb \
    --raw-storage-format n_a \
    --bridge none \
    --decoder-feature-adapter none \
    --lora none \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0 \
    --rod-root "${ROD_ROOT}" \
    --rod-night-manifest "${ROD_MANIFEST}" \
    --rod-raw-source raw24 \
    --rod-rgb-pipeline student_dark_degreen_v1 \
    --rod-student-white-percentile 99.9 \
    --rod-student-gamma 0.9 \
    --rod-student-channel-gains 1.08 0.95 1.10 \
    --rod-label-space inverse_relative \
    --input-height 512 \
    --input-width 960 \
    --rod-train-crop-mode random \
    --rod-val-crop-mode center \
    --no-eval-stf \
    --eval-rod \
    --best-metric rod \
    --save-best-checkpoint \
    --bs 8 \
    --accum-steps 1 \
    --lr 1e-5 \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --epochs 10 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 4 \
    --log-interval 500 \
    --no-enable-fixed-viz-dump \
    --no-enable-train-source-viz-dump \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${SAVE}"
  printf '\n'
} 2>&1 | tee -a "${LOG}"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
  torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" finetune_stf/train.py \
    --encoder vits \
    --pretrained-from "${PRETRAINED}" \
    --stage rod_only \
    --input-domain rgb \
    --front-end dav2_rgb \
    --model-input-tensor image \
    --dataset-family rod_raw_student_rgb \
    --dataset-input-mode raw24_student_rgb \
    --raw-storage-format n_a \
    --bridge none \
    --decoder-feature-adapter none \
    --lora none \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0 \
    --rod-root "${ROD_ROOT}" \
    --rod-night-manifest "${ROD_MANIFEST}" \
    --rod-raw-source raw24 \
    --rod-rgb-pipeline student_dark_degreen_v1 \
    --rod-student-white-percentile 99.9 \
    --rod-student-gamma 0.9 \
    --rod-student-channel-gains 1.08 0.95 1.10 \
    --rod-label-space inverse_relative \
    --input-height 512 \
    --input-width 960 \
    --rod-train-crop-mode random \
    --rod-val-crop-mode center \
    --no-eval-stf \
    --eval-rod \
    --best-metric rod \
    --save-best-checkpoint \
    --bs 8 \
    --accum-steps 1 \
    --lr 1e-5 \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --epochs 10 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 4 \
    --log-interval 500 \
    --no-enable-fixed-viz-dump \
    --no-enable-train-source-viz-dump \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${SAVE}" 2>&1 | tee -a "${LOG}"
status=${PIPESTATUS[0]}
set -e

echo "[END] $(date -Iseconds) status=${status}" | tee -a "${LOG}"
exit "${status}"
