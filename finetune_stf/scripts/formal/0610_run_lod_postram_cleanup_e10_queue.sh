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
C_DARK="${C_DARK:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
C_DARK_W0="${C_DARK_W0:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_1739_lod_true_raw_rgb16_block8excl10_ram3_dav2s_decoder_e40_W0_aug-baseline_e10_poly/best_model.pth}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_postram_cleanup_e10_queue}"
MASTER_PORT="${MASTER_PORT:-29661}"
RUN_MATRIX="${RUN_MATRIX:-lora}"
SEEDS="${SEEDS:-42 123 777}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
CLEANUP_SCALE="${CLEANUP_SCALE:-0.1}"
CLEANUP_LR="${CLEANUP_LR:-5e-5}"
DEBUG_MAX_TRAIN_STEPS="${DEBUG_MAX_TRAIN_STEPS:-0}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0610_run_lod_postram_cleanup_e10_queue.sh
  bash finetune_stf/scripts/formal/0610_run_lod_postram_cleanup_e10_queue.sh --audit

Defaults:
  RUN_MATRIX=lora
  SEEDS="42 123 777"
  EPOCHS=10

RUN_MATRIX values:
  lora  -> B0_LORA_s01 + T_LORA_s01
  w0    -> B0_W0_s01 + T_W0_s01
  both  -> lora and w0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--audit" ]]; then
  AUDIT_ONLY=1
  shift
fi

if [[ "${AUDIT_ONLY}" != "1" && "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  queue_timestamp="$(date +%m%d_%H%M)"
  session="${queue_timestamp}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' C_DARK='${C_DARK}' C_DARK_W0='${C_DARK_W0}' MASTER_PORT='${MASTER_PORT}' RUN_MATRIX='${RUN_MATRIX}' SEEDS='${SEEDS}' EPOCHS='${EPOCHS}' BS='${BS}' CLEANUP_SCALE='${CLEANUP_SCALE}' CLEANUP_LR='${CLEANUP_LR}' DEBUG_MAX_TRAIN_STEPS='${DEBUG_MAX_TRAIN_STEPS}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
tmux session: ${session}
queue log: ${queue_log}
attach: tmux attach -t ${session}
monitor: tail -f ${queue_log}
EOF
  exit 0
fi

if [[ "${1:-}" == "--run-internal" ]]; then
  shift
fi
if [[ "$#" -gt 0 ]]; then
  usage >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}" "${HEAVY_ROOT}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }; }
require_file "${PRETRAINED}"
require_file "${LOD_MANIFEST}"
require_file "${C_DARK}"
require_file "${C_DARK_W0}"
require_dir "${LOD_ROOT}"

echo "[START] $(date -Iseconds)"
echo "[SESSION] ${QUEUE_SESSION:-internal}"
echo "[HOST] $(hostname)"
echo "[USER] $(whoami)"
echo "[PWD] $(pwd)"
echo "[GPU] ${GPU}"
echo "[RUN_MATRIX] ${RUN_MATRIX}"
echo "[SEEDS] ${SEEDS}"
echo "[EPOCHS] ${EPOCHS}"
echo "[CLEANUP_SCALE] ${CLEANUP_SCALE}"
echo "[DEBUG_MAX_TRAIN_STEPS] ${DEBUG_MAX_TRAIN_STEPS}"

run_one() {
  local label="$1"
  local recipe="$2"
  local cleanup="$3"
  local seed="$4"
  local init_ckpt="$5"
  local port="$6"
  local run_timestamp
  run_timestamp="$(date +%m%d_%H%M)"
  local run_name="${run_timestamp}_${label}_lod_postram_cleanup_e${EPOCHS}_seed${seed}"
  local save="${EXP_ROOT}/${run_name}"
  local heavy="${HEAVY_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}.tmux.log"
  if [[ "${AUDIT_ONLY}" == "1" ]]; then
    echo "[AUDIT][OK] ${run_name} recipe=${recipe} cleanup=${cleanup} seed=${seed} save=${save} heavy=${heavy} log=${log}"
    return 0
  fi
  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] refusing to overwrite ${save} or ${heavy}" >&2
    exit 2
  fi
  mkdir -p "${save}"
  {
    echo "[RUN] ${run_name}"
    echo "[LABEL] ${label}"
    echo "[RECIPE] ${recipe}"
    echo "[CLEANUP] ${cleanup}"
    echo "[SEED] ${seed}"
    echo "[INIT] ${init_ckpt}"
    echo "[SAVE] ${save}"
    echo "[HEAVY] ${heavy}"
    echo "[LOG] ${log}"
  } | tee -a "${log}"

  local -a lora_args=()
  if [[ "${recipe}" == "lora" ]]; then
    lora_args=(
      --lora dav2_lora
      --lora-block-mode tap
      --lora-tap-layers 2 5 8 11
      --lora-rank 8
      --lora-alpha 16
      --lora-lr 5e-5
    )
  elif [[ "${recipe}" == "w0" ]]; then
    lora_args=(--lora none)
  else
    echo "[ERROR] unsupported recipe=${recipe}" >&2
    exit 2
  fi

  local -a cleanup_args=()
  if [[ "${cleanup}" == "none" ]]; then
    cleanup_args=(
      --post-ram-cleanup none
      --post-ram-cleanup-channels n_a
      --post-ram-cleanup-blocks n_a
      --post-ram-cleanup-scale n_a
      --post-ram-cleanup-norm n_a
      --post-ram-cleanup-zero-init n_a
      --post-ram-cleanup-lr n_a
      --post-ram-external-denoiser none
      --post-ram-denoiser-sigma n_a
      --post-ram-denoiser-alpha n_a
      --post-ram-denoiser-affine n_a
      --post-ram-denoiser-frozen n_a
      --post-ram-operation-position n_a
    )
  elif [[ "${cleanup}" == "cnn" ]]; then
    cleanup_args=(
      --post-ram-cleanup cnn
      --post-ram-cleanup-channels 32
      --post-ram-cleanup-blocks 4
      --post-ram-cleanup-scale "${CLEANUP_SCALE}"
      --post-ram-cleanup-norm groupnorm
      --post-ram-cleanup-zero-init true
      --post-ram-cleanup-lr "${CLEANUP_LR}"
      --post-ram-external-denoiser none
      --post-ram-denoiser-sigma n_a
      --post-ram-denoiser-alpha n_a
      --post-ram-denoiser-affine n_a
      --post-ram-denoiser-frozen n_a
      --post-ram-operation-position pre_tail
    )
  else
    echo "[ERROR] unsupported cleanup=${cleanup}" >&2
    exit 2
  fi

  local -a debug_args=()
  if [[ "${DEBUG_MAX_TRAIN_STEPS}" != "0" ]]; then
    debug_args=(--debug-max-train-steps "${DEBUG_MAX_TRAIN_STEPS}")
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port "${port}" finetune_stf/train.py \
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
      "${lora_args[@]}" \
      --dav2-train-mode decoder \
      --backbone-layer-decay 1.0 \
      --student-init-from "${init_ckpt}" \
      --student-init-strict compatible \
      --raw-front-end-lr 5e-5 \
      --raw-ram-rgb-tail identity \
      --raw-ram-local-residual none \
      --raw-ram-local-hidden-ch 0 \
      --raw-ram-local-residual-scale 0 \
      --raw-ram-local-gate-init 0 \
      --raw-ram-local-gate-mode n_a \
      "${cleanup_args[@]}" \
      --input-height 512 \
      --input-width 960 \
      --lod-label-space inverse_relative \
      --lod-train-crop-mode random \
      --lod-val-crop-mode center \
      --loss-type ssi \
      --loss-target-normalization \
      --loss-norm-min-scale 1e-3 \
      --epochs "${EPOCHS}" \
      --bs "${BS}" \
      --accum-steps 1 \
      --lr 1e-5 \
      --lr-schedule poly \
      --warmup-steps 0 \
      --aug-preset off \
      --aug-hflip-prob 0.5 \
      --eval-lod-train-proxy \
      --lod-train-proxy-count 112 \
      --amp \
      --amp-dtype bf16 \
      --seed "${seed}" \
      --num-workers 4 \
      --log-interval 500 \
      --no-eval-stf \
      --eval-lod \
      --best-metric lod_d1 \
      --save-best-checkpoint \
      --no-enable-fixed-viz-dump \
      --no-enable-train-source-viz-dump \
      --heavy-save-root "${HEAVY_ROOT}" \
      --save-path "${save}" \
      "${debug_args[@]}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) run=${run_name} status=${status}" | tee -a "${log}"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

port="${MASTER_PORT}"
run_lora() {
  for seed in ${SEEDS}; do
    run_one "B0_LORA_s01" "lora" "none" "${seed}" "${C_DARK}" "${port}"
    port=$((port + 1))
    run_one "T_LORA_s01" "lora" "cnn" "${seed}" "${C_DARK}" "${port}"
    port=$((port + 1))
  done
}

run_w0() {
  for seed in ${SEEDS}; do
    run_one "B0_W0_s01" "w0" "none" "${seed}" "${C_DARK_W0}" "${port}"
    port=$((port + 1))
    run_one "T_W0_s01" "w0" "cnn" "${seed}" "${C_DARK_W0}" "${port}"
    port=$((port + 1))
  done
}

case "${RUN_MATRIX}" in
  lora) run_lora ;;
  w0) run_w0 ;;
  both) run_lora; run_w0 ;;
  *) echo "[ERROR] RUN_MATRIX must be lora, w0, or both; got ${RUN_MATRIX}" >&2; exit 2 ;;
esac

echo "[QUEUE] done $(date -Iseconds)"
