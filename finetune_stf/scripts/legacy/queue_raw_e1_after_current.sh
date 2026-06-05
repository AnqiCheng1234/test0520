#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PROJECT_ROOT="${PROJECT_ROOT:-/home/caq/6666_raw/dav2_raw}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/caq/anaconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-dav3}"

CURRENT_SESSION="${CURRENT_SESSION:-e2_ram}"
CURRENT_SAVE_PATH="${CURRENT_SAVE_PATH:-finetune_stf/exp/e2_raw_ram_bs4_20260412_223717}"
CURRENT_EPOCHS="${CURRENT_EPOCHS:-20}"
POLL_SECONDS="${POLL_SECONDS:-120}"

TARGET_SESSION="${TARGET_SESSION:-e1_accum}"
E1_BS="${E1_BS:-4}"
E1_ACCUM_STEPS="${E1_ACCUM_STEPS:-4}"
E1_SAVE_PATH="${E1_SAVE_PATH:-}"

CURRENT_FINAL_EPOCH="${CURRENT_FINAL_EPOCH:-$((CURRENT_EPOCHS - 1))}"

if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

current_train_running() {
  pgrep -af "finetune_stf/train.py.*--save-path ${CURRENT_SAVE_PATH}" >/dev/null 2>&1
}

show_last_progress() {
  local log_file="${PROJECT_ROOT}/${CURRENT_SAVE_PATH}/train.log"
  if [[ ! -f "${log_file}" ]]; then
    log "waiting for train log: ${CURRENT_SAVE_PATH}/train.log"
    return
  fi

  log "latest progress from ${CURRENT_SAVE_PATH}:"
  grep -E "\\[(TRAIN|EVAL|CHECKPOINT|EPOCH)\\]" "${log_file}" | tail -n 5 || true
}

launch_e1() {
  local e1_save_path="${E1_SAVE_PATH}"
  local run_tag=""
  local launch_cmd=""

  if [[ -z "${e1_save_path}" ]]; then
    run_tag="$(date +%m%d_%H%M)_e1_raw_naive_bs${E1_BS}_acc${E1_ACCUM_STEPS}"
    e1_save_path="finetune_stf/exp/${run_tag}"
  fi

  if tmux has-session -t "=${TARGET_SESSION}" 2>/dev/null; then
    log "target tmux session already exists: ${TARGET_SESSION}"
    return 1
  fi

  launch_cmd="source ${CONDA_ACTIVATE} ${CONDA_ENV} && cd ${PROJECT_ROOT} && BS=${E1_BS} ACCUM_STEPS=${E1_ACCUM_STEPS} SAVE_PATH=${e1_save_path} bash finetune_stf/scripts/train_raw_e1.sh"
  tmux new-session -d -s "${TARGET_SESSION}" "${launch_cmd}"
  log "launched E1 accumulation run in tmux session ${TARGET_SESSION}"
  log "E1 save path: ${e1_save_path}"
}

log "queue watcher started"
log "monitoring current session=${CURRENT_SESSION} save_path=${CURRENT_SAVE_PATH}"
log "E1 will launch in tmux session=${TARGET_SESSION} with bs=${E1_BS} accum_steps=${E1_ACCUM_STEPS} effective_bs=$((E1_BS * E1_ACCUM_STEPS))"

show_last_progress

while current_train_running; do
  sleep "${POLL_SECONDS}"
  show_last_progress
done

log "current training process is no longer running"

current_log="${PROJECT_ROOT}/${CURRENT_SAVE_PATH}/train.log"
if [[ ! -f "${current_log}" ]]; then
  log "missing train log after process exit; refusing to launch E1"
  exit 1
fi

if ! grep -q "\\[EPOCH\\] done epoch=${CURRENT_FINAL_EPOCH} " "${current_log}"; then
  log "did not find final epoch marker epoch=${CURRENT_FINAL_EPOCH}; refusing to launch E1"
  show_last_progress
  exit 1
fi

show_last_progress
launch_e1
