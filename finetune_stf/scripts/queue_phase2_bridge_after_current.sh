#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PROJECT_ROOT="${PROJECT_ROOT:-/home/caq/6666_raw/dav2_raw}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/caq/anaconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-dav3}"

CURRENT_SAVE_PATH="${CURRENT_SAVE_PATH:-finetune_stf/exp/phase2_candidateB_raw_ram_bridge_online_20260415_223850}"
CURRENT_EPOCHS="${CURRENT_EPOCHS:-20}"
POLL_SECONDS="${POLL_SECONDS:-120}"

BRIDGE_SESSION="${BRIDGE_SESSION:-phase2_bridge_blr1e4}"
BRIDGE_BS="${BRIDGE_BS:-4}"
BRIDGE_ACCUM_STEPS="${BRIDGE_ACCUM_STEPS:-4}"
BRIDGE_LR="${BRIDGE_LR:-1e-4}"
BRIDGE_SAVE_PATH="${BRIDGE_SAVE_PATH:-}"

LORA_SESSION="${LORA_SESSION:-phase2_bridge_lora}"
LORA_BS="${LORA_BS:-4}"
LORA_ACCUM_STEPS="${LORA_ACCUM_STEPS:-4}"
LORA_BRIDGE_LR="${LORA_BRIDGE_LR:-5e-5}"
LORA_LR="${LORA_LR:-5e-5}"
LORA_SAVE_PATH="${LORA_SAVE_PATH:-}"

CURRENT_FINAL_EPOCH="${CURRENT_FINAL_EPOCH:-$((CURRENT_EPOCHS - 1))}"

if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}" "${CONDA_ENV}"
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

resolve_abs_save_path() {
  local save_path="$1"
  if [[ "${save_path}" = /* ]]; then
    printf '%s\n' "${save_path}"
  else
    printf '%s/%s\n' "${PROJECT_ROOT}" "${save_path}"
  fi
}

run_is_active() {
  local save_path="$1"
  local abs_save_path
  abs_save_path="$(resolve_abs_save_path "${save_path}")"
  pgrep -af "finetune_stf/train.py" | grep -F -- "--save-path ${save_path}" >/dev/null 2>&1 || \
    pgrep -af "finetune_stf/train.py" | grep -F -- "--save-path ${abs_save_path}" >/dev/null 2>&1
}

show_last_progress() {
  local save_path="$1"
  local log_root
  log_root="$(resolve_abs_save_path "${save_path}")"
  local log_file="${log_root}/train.log"
  if [[ ! -f "${log_file}" ]]; then
    log "waiting for train log: ${save_path}/train.log"
    return
  fi
  log "latest progress from ${save_path}:"
  grep -E "\\[(TRAIN|EVAL|CHECKPOINT|EPOCH)\\]" "${log_file}" | tail -n 5 || true
}

ensure_run_finished() {
  local save_path="$1"
  local final_epoch="$2"
  local log_root
  log_root="$(resolve_abs_save_path "${save_path}")"
  local log_file="${log_root}/train.log"
  if [[ ! -f "${log_file}" ]]; then
    log "missing train log for ${save_path}"
    return 1
  fi
  grep -q "\\[EPOCH\\] done epoch=${final_epoch} " "${log_file}"
}

launch_bridge() {
  local save_path="${BRIDGE_SAVE_PATH}"
  if [[ -z "${save_path}" ]]; then
    save_path="finetune_stf/exp/$(date +%m%d_%H%M)_phase2_vkitti_bridge_online_blr1e4_bs${BRIDGE_BS}_acc${BRIDGE_ACCUM_STEPS}"
  fi
  if tmux has-session -t "=${BRIDGE_SESSION}" 2>/dev/null; then
    log "bridge tmux session already exists: ${BRIDGE_SESSION}"
    return 1
  fi
  local launch_cmd="source ${CONDA_ACTIVATE} ${CONDA_ENV} && cd ${PROJECT_ROOT} && BS=${BRIDGE_BS} ACCUM_STEPS=${BRIDGE_ACCUM_STEPS} BRIDGE_LR=${BRIDGE_LR} SAVE_PATH=${save_path} bash finetune_stf/scripts/train_phase2_vkitti_bridge_online.sh"
  tmux new-session -d -s "${BRIDGE_SESSION}" "${launch_cmd}"
  log "launched bridge-only run in tmux session ${BRIDGE_SESSION}"
  log "bridge-only save path: ${save_path}"
  BRIDGE_SAVE_PATH="${save_path}"
}

launch_lora() {
  local save_path="${LORA_SAVE_PATH}"
  if [[ -z "${save_path}" ]]; then
    save_path="finetune_stf/exp/$(date +%m%d_%H%M)_phase2_vkitti_bridge_lora_online_bs${LORA_BS}_acc${LORA_ACCUM_STEPS}"
  fi
  if tmux has-session -t "=${LORA_SESSION}" 2>/dev/null; then
    log "lora tmux session already exists: ${LORA_SESSION}"
    return 1
  fi
  local launch_cmd="source ${CONDA_ACTIVATE} ${CONDA_ENV} && cd ${PROJECT_ROOT} && BS=${LORA_BS} ACCUM_STEPS=${LORA_ACCUM_STEPS} BRIDGE_LR=${LORA_BRIDGE_LR} LORA_LR=${LORA_LR} SAVE_PATH=${save_path} bash finetune_stf/scripts/train_phase2_vkitti_bridge_lora_online.sh"
  tmux new-session -d -s "${LORA_SESSION}" "${launch_cmd}"
  log "launched bridge+LoRA run in tmux session ${LORA_SESSION}"
  log "bridge+LoRA save path: ${save_path}"
  LORA_SAVE_PATH="${save_path}"
}

log "phase2 queue watcher started"
log "watching current run save_path=${CURRENT_SAVE_PATH}"
show_last_progress "${CURRENT_SAVE_PATH}"

while run_is_active "${CURRENT_SAVE_PATH}"; do
  sleep "${POLL_SECONDS}"
  show_last_progress "${CURRENT_SAVE_PATH}"
done

if ! ensure_run_finished "${CURRENT_SAVE_PATH}" "${CURRENT_FINAL_EPOCH}"; then
  log "current run did not reach final epoch=${CURRENT_FINAL_EPOCH}; refusing to launch follow-up runs"
  show_last_progress "${CURRENT_SAVE_PATH}"
  exit 1
fi

log "current run finished; launching bridge-only follow-up"
launch_bridge

while run_is_active "${BRIDGE_SAVE_PATH}"; do
  sleep "${POLL_SECONDS}"
  show_last_progress "${BRIDGE_SAVE_PATH}"
done

if ! ensure_run_finished "${BRIDGE_SAVE_PATH}" "${CURRENT_FINAL_EPOCH}"; then
  log "bridge-only run did not reach final epoch=${CURRENT_FINAL_EPOCH}; refusing to launch LoRA run"
  show_last_progress "${BRIDGE_SAVE_PATH}"
  exit 1
fi

log "bridge-only run finished; launching bridge+LoRA follow-up"
launch_lora
