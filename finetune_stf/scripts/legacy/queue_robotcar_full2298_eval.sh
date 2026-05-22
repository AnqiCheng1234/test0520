#!/usr/bin/env bash
set -euo pipefail

# Queue a stratified RobotCar eval for the lms_front-only + 480x640 + sparse protocol.
# to run after the currently active training finishes.
# Goal: refresh the RobotCar comparison rows in
#   /home/caq/6666_raw/dav2_raw_512960/plans/result/rgb_raw_baseline_fairness_summary.md
# with numbers that align with train-time eval (full 2298 samples, not seed=42 500-subset).
# Checkpoint reused: same Phase2 ssi_grad bridge that was used for the 500-subset comparison.

REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
CONDA_ENV="${CONDA_ENV:-dav3}"
TRAIN_PID="${TRAIN_PID:-}"

EXP_DIR="/home/caq/6666_raw/dav2_raw_512960/finetune_stf/exp/0418_0130_phase2_vkitti_only_raw_ram_bridge_frontend_only_512960_bs4acc4_dual_eval_v2calib_loss_ssi_grad_e10"
OUT_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/mnt/drive/3333_raw/robotcar_lms_front_480640_rgb_raw_compare_${OUT_TS}"
LOG="${OUT_DIR}.log"

mkdir -p "$(dirname "${LOG}")"

if [[ -z "${TRAIN_PID}" ]]; then
  echo "[queue] no TRAIN_PID provided; starting eval immediately" | tee -a "${LOG}"
else
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[queue] TRAIN_PID=${TRAIN_PID} already gone; starting eval immediately" | tee -a "${LOG}"
  else
    echo "[queue] waiting for TRAIN_PID=${TRAIN_PID} to exit before starting eval" | tee -a "${LOG}"
    # tail --pid blocks until the named PID exits; works for non-child PIDs too.
    tail --pid="${TRAIN_PID}" -f /dev/null
    echo "[queue] TRAIN_PID=${TRAIN_PID} exited at $(date -Iseconds); waiting 60s for GPU to settle" | tee -a "${LOG}"
    sleep 60
  fi
fi

echo "[queue] starting eval at $(date -Iseconds)" | tee -a "${LOG}"
echo "[queue] exp_dir=${EXP_DIR}" | tee -a "${LOG}"
echo "[queue] output_dir=${OUT_DIR}" | tee -a "${LOG}"

cd "${REPO_ROOT}"
conda run -n "${CONDA_ENV}" --no-capture-output \
  python -u anqi_eval/eval_robotcar_rgb_raw_compare.py \
  "${EXP_DIR}" \
  --checkpoint best \
  --robotcar-root /mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640 \
  --output-dir "${OUT_DIR}" \
  --fast-eval-backend sparse \
  --sample-count 99999 \
  --sample-seed 42 \
  --raw-source native \
  --max-depth 50 \
  --stratified-eval \
  --no-save-panels \
  2>&1 | tee -a "${LOG}"

echo "[queue] eval finished at $(date -Iseconds)" | tee -a "${LOG}"
echo "[queue] summary_stratified.json at ${OUT_DIR}/summary_stratified.json" | tee -a "${LOG}"
