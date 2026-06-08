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
LOD_ROOT="${LOD_ROOT:-/home/caq/6666_raw/0000_dataset/LOD}"
LOD_MANIFEST="${LOD_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_true_raw_rgb16_ram3_decoder_e10}"
MASTER_PORT="${MASTER_PORT:-29619}"

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' RUN_TIMESTAMP='${run_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
require_file "${LOD_MANIFEST}"
require_dir "${LOD_ROOT}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}"
RUN_NAME="${RUN_TIMESTAMP}_lod_true_raw_rgb16_ram3_dav2s_decoder_e10"
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
echo "[FORMAL] lod_root=${LOD_ROOT}"
echo "[FORMAL] manifest=${LOD_MANIFEST}"
echo "[FORMAL] student=LOD_RAW_Dark_RGB16 raw_storage=raw_rgb16_png_3ch norm=uint16_div_65535 teacher=LOD_RGB_normal_DAv2L best_metric=lod_d1"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
  torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" finetune_stf/train.py \
    --encoder vits \
    --pretrained-from "${PRETRAINED}" \
    --stage lod_only \
    --lod-root "${LOD_ROOT}" \
    --lod-manifest "${LOD_MANIFEST}" \
    --dataset-family lod_true_raw_dark_rgb16 \
    --dataset-input-mode raw_rgb16_dark \
    --input-domain raw3 \
    --front-end raw_rgb16_ram3 \
    --model-input-tensor raw \
    --raw-storage-format raw_rgb16_png_3ch \
    --lod-raw-norm-mode uint16_div_65535 \
    --bridge none \
    --decoder-feature-adapter none \
    --lora none \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0 \
    --raw-front-end-lr 5e-5 \
    --raw-ram-rgb-tail identity \
    --input-height 512 \
    --input-width 960 \
    --lod-label-space inverse_relative \
    --lod-train-crop-mode random \
    --lod-val-crop-mode center \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --epochs 10 \
    --bs 8 \
    --accum-steps 1 \
    --lr 1e-5 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 4 \
    --log-interval 500 \
    --no-eval-stf \
    --eval-lod \
    --best-metric lod_d1 \
    --save-best-checkpoint \
    --no-enable-fixed-viz-dump \
    --no-enable-train-source-viz-dump \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${SAVE}" 2>&1 | tee -a "${LOG}"
status=${PIPESTATUS[0]}
set -e

echo "[END] $(date -Iseconds) status=${status}" | tee -a "${LOG}"
exit "${status}"
