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
LOD_MANIFEST="${LOD_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_true_rgb_aug_e40_block8excl10_queue}"
MASTER_PORT="${MASTER_PORT:-29628}"

if [[ "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  queue_timestamp="$(date +%m%d_%H%M)"
  session="${queue_timestamp}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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

run_one() {
  local run_id="$1"
  local aug_preset="$2"
  local run_timestamp
  run_timestamp="$(date +%m%d_%H%M)"
  local run_name="${run_timestamp}_lod_true_rgb_dark_block8excl10_dav2s_decoder_e40_${run_id}_aug-${aug_preset}_poly"
  local save="${EXP_ROOT}/${run_name}"
  local heavy="${HEAVY_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}.tmux.log"

  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] refusing to overwrite ${save} or ${heavy}" >&2
    exit 2
  fi

  mkdir -p "${save}"
  echo "[FORMAL] run=${run_name}"
  echo "[FORMAL] save=${save}"
  echo "[FORMAL] heavy=${heavy}"
  echo "[FORMAL] log=${log}"
  echo "[FORMAL] matrix=${run_id} link=RGB_Dark split=block8_val112_excl10_uniform train=1958 val=112 buffer=160 nearest_val_train_pair_distance_min=11 epochs=40 aug_preset=${aug_preset} lr_schedule=poly warmup_steps=0 train_proxy=112 lod_manifest=${LOD_MANIFEST}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" finetune_stf/train.py \
      --encoder vits \
      --pretrained-from "${PRETRAINED}" \
      --stage lod_only \
      --lod-root "${LOD_ROOT}" \
      --lod-manifest "${LOD_MANIFEST}" \
      --dataset-family lod_true_rgb_dark \
      --dataset-input-mode rgb_dark \
      --input-domain rgb \
      --front-end dav2_rgb \
      --model-input-tensor image \
      --raw-storage-format n_a \
      --bridge none \
      --decoder-feature-adapter none \
      --lora none \
      --dav2-train-mode decoder \
      --backbone-layer-decay 1.0 \
      --input-height 512 \
      --input-width 960 \
      --lod-label-space inverse_relative \
      --lod-train-crop-mode random \
      --lod-val-crop-mode center \
      --loss-type ssi \
      --loss-target-normalization \
      --loss-norm-min-scale 1e-3 \
      --epochs 40 \
      --bs 8 \
      --accum-steps 1 \
      --lr 1e-5 \
      --lr-schedule poly \
      --warmup-steps 0 \
      --aug-preset "${aug_preset}" \
      --eval-lod-train-proxy \
      --lod-train-proxy-count 112 \
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
      --save-path "${save}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) run=${run_name} status=${status}" | tee -a "${log}"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

run_one "R0" "baseline_e10"
run_one "Rg" "geom"
run_one "R1" "medium"

echo "[QUEUE] all RGB aug e40 runs done $(date -Iseconds)"
