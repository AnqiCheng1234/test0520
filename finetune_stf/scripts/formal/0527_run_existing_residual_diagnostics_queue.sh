#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-existing_residual_diagnostics}"

RUN_FEATURE_ABLATION="${RUN_FEATURE_ABLATION:-1}"
RUN_ENERGY_FREQ="${RUN_ENERGY_FREQ:-1}"
RUN_VS_C2_PANELS="${RUN_VS_C2_PANELS:-1}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
PANEL_SAMPLE_INDICES="${PANEL_SAMPLE_INDICES:-0,72}"

C2_RUN_DIR="${C2_RUN_DIR:-${ROOT}/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
C2_CHECKPOINT="${C2_CHECKPOINT:-/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_11.pth}"
M1_RUN_DIR="${M1_RUN_DIR:-${ROOT}/finetune_stf/exp/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
M1_CHECKPOINT="${M1_CHECKPOINT:-/mnt/drive/3333_raw/0000_exp_ckpt/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_14.pth}"
M2_RUN_DIR="${M2_RUN_DIR:-${ROOT}/finetune_stf/exp/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
M2_CHECKPOINT="${M2_CHECKPOINT:-/mnt/drive/3333_raw/0000_exp_ckpt/0525_1425_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_09.pth}"

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
    "cd '${ROOT}' && ROOT='${ROOT}' LOG_ROOT='${LOG_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' RUN_FEATURE_ABLATION='${RUN_FEATURE_ABLATION}' RUN_ENERGY_FREQ='${RUN_ENERGY_FREQ}' RUN_VS_C2_PANELS='${RUN_VS_C2_PANELS}' MAX_VAL_SAMPLES='${MAX_VAL_SAMPLES}' PANEL_SAMPLE_INDICES='${PANEL_SAMPLE_INDICES}' C2_RUN_DIR='${C2_RUN_DIR}' C2_CHECKPOINT='${C2_CHECKPOINT}' M1_RUN_DIR='${M1_RUN_DIR}' M1_CHECKPOINT='${M1_CHECKPOINT}' M2_RUN_DIR='${M2_RUN_DIR}' M2_CHECKPOINT='${M2_CHECKPOINT}' QUEUE_TIMESTAMP='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
Started tmux session: ${session}
Queue log: ${queue_log}
Attach: tmux attach -t ${session}
Monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
timestamp="${QUEUE_TIMESTAMP:-$(date +%m%d_%H%M)}"
mkdir -p "${ROOT}/plans/0527/diagnostics" "${ROOT}/plans/0527/panels" "${LOG_ROOT}"

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] Required file not found: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "[ERROR] Required directory not found: $1" >&2; exit 2; }
}
run_cmd() {
  echo "[CMD] $*" >&2
  CUDA_VISIBLE_DEVICES="${GPU}" "$@"
}
max_arg=()
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  max_arg=(--max-val-samples "${MAX_VAL_SAMPLES}")
fi

require_dir "${C2_RUN_DIR}"
require_file "${C2_CHECKPOINT}"
require_dir "${M1_RUN_DIR}"
require_file "${M1_CHECKPOINT}"
require_dir "${M2_RUN_DIR}"
require_file "${M2_CHECKPOINT}"

if [[ "${RUN_FEATURE_ABLATION}" == "1" ]]; then
  base="${ROOT}/plans/0527/diagnostics/${timestamp}_raw_feature_ablation"
  mkdir "${base}"
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_raw_residual_feature_ablation.py \
    --run-dir "${M1_RUN_DIR}" --checkpoint "${M1_CHECKPOINT}" --feature-source x3 \
    --output-dir "${base}/m1_x3_d0" --feature-ablation-modes true,zero,mean,shuffle \
    --shuffle-policy stable_hash_far --shuffle-seed 42 "${max_arg[@]}"
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_raw_residual_feature_ablation.py \
    --run-dir "${M2_RUN_DIR}" --checkpoint "${M2_CHECKPOINT}" --feature-source ffm_mid \
    --output-dir "${base}/m2_ffm_mid_d0" --feature-ablation-modes true,zero,mean,shuffle \
    --shuffle-policy stable_hash_far --shuffle-seed 42 "${max_arg[@]}"
fi

if [[ "${RUN_ENERGY_FREQ}" == "1" ]]; then
  base="${ROOT}/plans/0527/diagnostics/${timestamp}_residual_energy_frequency"
  mkdir "${base}"
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/analyze_residual_energy_frequency.py \
    --run-kind control --run-dir "${C2_RUN_DIR}" --checkpoint "${C2_CHECKPOINT}" --output-dir "${base}/c2" "${max_arg[@]}"
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/analyze_residual_energy_frequency.py \
    --run-kind raw --run-dir "${M1_RUN_DIR}" --checkpoint "${M1_CHECKPOINT}" --output-dir "${base}/m1_x3_d0" "${max_arg[@]}"
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/analyze_residual_energy_frequency.py \
    --run-kind raw --run-dir "${M2_RUN_DIR}" --checkpoint "${M2_CHECKPOINT}" --output-dir "${base}/m2_ffm_mid_d0" "${max_arg[@]}"
fi

if [[ "${RUN_VS_C2_PANELS}" == "1" ]]; then
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/make_residual_vs_c2_panels.py \
    --c2-run-dir "${C2_RUN_DIR}" --c2-checkpoint "${C2_CHECKPOINT}" \
    --method-run-dir "${M1_RUN_DIR}" --method-checkpoint "${M1_CHECKPOINT}" --method-kind raw \
    --output-dir "${ROOT}/plans/0527/panels/${timestamp}_vs_c2_m1_x3_d0" \
    --sample-indices "${PANEL_SAMPLE_INDICES}" --max-panels 2
  run_cmd "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/make_residual_vs_c2_panels.py \
    --c2-run-dir "${C2_RUN_DIR}" --c2-checkpoint "${C2_CHECKPOINT}" \
    --method-run-dir "${M2_RUN_DIR}" --method-checkpoint "${M2_CHECKPOINT}" --method-kind raw \
    --output-dir "${ROOT}/plans/0527/panels/${timestamp}_vs_c2_m2_ffm_mid_d0" \
    --sample-indices "${PANEL_SAMPLE_INDICES}" --max-panels 2
fi

echo "[DONE] existing residual diagnostics completed at $(date -Iseconds)"
