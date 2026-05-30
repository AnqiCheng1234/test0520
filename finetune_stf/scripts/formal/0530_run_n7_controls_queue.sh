#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
BS="${BS:-8}"
EPOCHS="${EPOCHS:-10}"
DEFAULT_N7_RUN_DIR="${ROOT}/finetune_stf/exp/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10"
DEFAULT_N7_CKPT="/mnt/drive/3333_raw/0000_exp_ckpt/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_09.pth"
N7_RUN_DIR="${N7_RUN_DIR:-${DEFAULT_N7_RUN_DIR}}"
N7_CKPT="${N7_CKPT:-${DEFAULT_N7_CKPT}}"
SESSION_PREFIX="${SESSION_PREFIX:-n7_controls}"

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
    "cd '${ROOT}' && ROOT='${ROOT}' LOG_ROOT='${LOG_ROOT}' EXP_ROOT='${EXP_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' DEVICE='${DEVICE}' SEED='${SEED}' BS='${BS}' EPOCHS='${EPOCHS}' N7_RUN_DIR='${N7_RUN_DIR}' N7_CKPT='${N7_CKPT}' QUEUE_TIMESTAMP='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
Started tmux session: ${session}
Queue log: ${queue_log}
Attach: tmux attach -t ${session}
Monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "${LOG_ROOT}" "${EXP_ROOT}" "${HEAVY_ROOT}"
run_log_dir="${LOG_ROOT}/${QUEUE_SESSION:-n7_controls_manual}"
mkdir -p "${run_log_dir}"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] Required file not found: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] Required directory not found: $1" >&2; exit 2; }; }
tag_float() { echo "$1" | sed 's/\./p/g'; }

require_dir "${N7_RUN_DIR}"
require_file "${N7_CKPT}"
require_file "${N7_RUN_DIR}/config.json"

run_child() {
  local name="$1"
  local log="$2"
  shift 2
  echo "[QUEUE][START] ${name} $(date -Iseconds)" | tee -a "${log}"
  echo "[QUEUE][CMD] $*" | tee -a "${log}"
  set +e
  "$@" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[QUEUE][END] ${name} status=${status} $(date -Iseconds)" | tee -a "${log}"
  return "${status}"
}

queue_timestamp="${QUEUE_TIMESTAMP:-$(date +%m%d_%H%M)}"
P0_OUT_ROOT="${P0_OUT_ROOT:-${ROOT}/plans/0527/diagnostics/0530_n7_eval_x3_ablation_${queue_timestamp}}"
run_child "P0 N7 eval-time x3 ablation" "${run_log_dir}/p0_n7_eval_x3_ablation.log" \
  env ROOT="${ROOT}" CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" GPU="${GPU}" DEVICE="${DEVICE}" SEED="${SEED}" BS="${BS}" \
  N7_RUN_DIR="${N7_RUN_DIR}" N7_CKPT="${N7_CKPT}" OUT_ROOT="${P0_OUT_ROOT}" RUN_EXTRA_SCOPES="${RUN_EXTRA_SCOPES:-0}" \
  bash finetune_stf/scripts/formal/0530_run_n7_eval_x3_ablation.sh

p1_ts="$(date +%m%d_%H%M)"
P1_RUN_NAME="${P1_RUN_NAME:-${p1_ts}_vkitti_n7_zero_x3_train_lp0p5_q0p3_lfl0p0_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs${BS}_e${EPOCHS}}"
P1_SAVE_PATH="${EXP_ROOT}/${P1_RUN_NAME}"
P1_HEAVY_PATH="${HEAVY_ROOT}/${P1_RUN_NAME}"
run_child "P1 N7-zero-x3-train" "${run_log_dir}/p1_n7_zero_x3_train.log" \
  env ROOT="${ROOT}" EXP_ROOT="${EXP_ROOT}" HEAVY_ROOT="${HEAVY_ROOT}" CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" GPU="${GPU}" DEVICE="${DEVICE}" SEED="${SEED}" BS="${BS}" EPOCHS="${EPOCHS}" \
  N7_RUN_DIR="${N7_RUN_DIR}" RUN_NAME="${P1_RUN_NAME}" SAVE_PATH="${P1_SAVE_PATH}" HEAVY_SAVE_PATH="${P1_HEAVY_PATH}" \
  bash finetune_stf/scripts/formal/0530_run_n7_zero_x3_train.sh

p2_ts="$(date +%m%d_%H%M)"
P2_RUN_NAME="${P2_RUN_NAME:-${p2_ts}_vkitti_n7rgb_lp0p5_q0p3_lfl0p0_rftna_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs${BS}_e${EPOCHS}}"
P2_SAVE_PATH="${EXP_ROOT}/${P2_RUN_NAME}"
P2_HEAVY_PATH="${HEAVY_ROOT}/${P2_RUN_NAME}"
run_child "P2 N7-RGB matched control" "${run_log_dir}/p2_n7_rgb_matched_control.log" \
  env ROOT="${ROOT}" EXP_ROOT="${EXP_ROOT}" HEAVY_ROOT="${HEAVY_ROOT}" CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" GPU="${GPU}" DEVICE="${DEVICE}" SEED="${SEED}" BS="${BS}" EPOCHS="${EPOCHS}" \
  N7_RUN_DIR="${N7_RUN_DIR}" RUN_NAME="${P2_RUN_NAME}" SAVE_PATH="${P2_SAVE_PATH}" HEAVY_SAVE_PATH="${P2_HEAVY_PATH}" \
  bash finetune_stf/scripts/formal/0530_run_n7_rgb_matched_control.sh

SUMMARY_OUT="${SUMMARY_OUT:-${ROOT}/plans/0527/diagnostics/0530_n7_controls_summary_${queue_timestamp}}"
mkdir -p "${SUMMARY_OUT}"
"${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/summarize_n7_controls.py \
  --n7-ablation-root "${P0_OUT_ROOT}" \
  --n7-zero-run-dir "${P1_SAVE_PATH}" \
  --n7-rgb-run-dir "${P2_SAVE_PATH}" \
  --n7-true-run-dir "${N7_RUN_DIR}" \
  --out-dir "${SUMMARY_OUT}" 2>&1 | tee -a "${run_log_dir}/summary.log"

echo "[DONE] N7 controls queue completed at $(date -Iseconds)"
echo "[DONE] P0_OUT_ROOT=${P0_OUT_ROOT}"
echo "[DONE] P1_SAVE_PATH=${P1_SAVE_PATH}"
echo "[DONE] P1_HEAVY_PATH=${P1_HEAVY_PATH}"
echo "[DONE] P2_SAVE_PATH=${P2_SAVE_PATH}"
echo "[DONE] P2_HEAVY_PATH=${P2_HEAVY_PATH}"
echo "[DONE] SUMMARY_OUT=${SUMMARY_OUT}"
