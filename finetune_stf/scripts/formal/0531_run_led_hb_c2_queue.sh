#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0522}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-led_hb_c2}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LED_TRAIN_LIST="${LED_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_train_all.txt}"
LED_VAL_LIST="${LED_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt}"
D0_SIGN="${D0_SIGN:?set D0_SIGN to recommended_d0_sign from the full-val L0 D0 eval}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
RUN_PROBE="${RUN_PROBE:-1}"

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LED_TRAIN_LIST='${LED_TRAIN_LIST}' LED_VAL_LIST='${LED_VAL_LIST}' D0_SIGN='${D0_SIGN}' EPOCHS='${EPOCHS}' BS='${BS}' ACCUM_STEPS='${ACCUM_STEPS}' RUN_SMOKE='${RUN_SMOKE}' KEEP_SMOKE='${KEEP_SMOKE}' RUN_PROBE='${RUN_PROBE}' RUN_TIMESTAMP='${run_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
require_file "${PRETRAINED}"
require_file "${LED_TRAIN_LIST}"
require_file "${LED_VAL_LIST}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "[SMOKE] running LED-HB stack smoke before C2 formal"
  KEEP_SMOKE="${KEEP_SMOKE}" GPU="${GPU}" CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" PRETRAINED="${PRETRAINED}" \
    bash finetune_stf/scripts/smoke/0531_smoke_led_hb_stack.sh
fi

run_c2_train() {
  local save="$1"
  local heavy="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_led_hb_residual_control.py \
      --experiment-id C2 \
      --dataset-name led_hb \
      --illumination HB \
      --input-domain rgb \
      --model-input-tensor image \
      --dataset-geometry-mode resize_fullres_756x1344_then_halfres_378x672 \
      --raw-storage-format not_applicable \
      --rgb-input-space resize_area_756x1344_then_2x2_area \
      --depth-target-space resize_nearest_756x1344_then_2x2_valid_mean \
      --depth-label distance_to_image_plane \
      --depth-unit meter \
      --front-end dav2_rgb_frozen \
      --encoder vits \
      --pretrained-from "${PRETRAINED}" \
      --led-train-list "${LED_TRAIN_LIST}" \
      --led-val-list "${LED_VAL_LIST}" \
      --train-split hb_train_all \
      --val-split hb_val_stride5_n1000_seed42 \
      --input-height 378 \
      --input-width 672 \
      --min-depth 1.0 \
      --max-depth 200.0 \
      --residual-feature-source d0 \
      --residual-alpha 0.5 \
      --d0-sign "${D0_SIGN}" \
      --hflip-prob 0.5 \
      --epochs "${EPOCHS}" \
      --bs "${BS}" \
      --accum-steps "${ACCUM_STEPS}" \
      --lr 1e-4 \
      --weight-decay 1e-4 \
      --num-workers 4 \
      --log-interval 100 \
      --save-interval 1 \
      --eval-interval 1 \
      --save-best-checkpoint \
      --amp \
      --amp-dtype bf16 \
      --seed 42 \
      "$@" \
      --save-path "${save}" \
      --heavy-save-path "${heavy}"
}

if [[ "${RUN_PROBE}" == "1" ]]; then
  probe_root="${ROOT}/plans/0531_led_night/codex_smoke_led_hb_c2_probe_${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}_bs${BS}_acc${ACCUM_STEPS}"
  echo "[PROBE] bs=${BS} accum=${ACCUM_STEPS} root=${probe_root}"
  set +e
  EPOCHS=1 run_c2_train "${probe_root}/exp" "${probe_root}/heavy" --max-train-steps 2 --max-val-samples 4 --num-workers 0 --log-interval 1
  probe_status=$?
  set -e
  if [[ "${probe_status}" -ne 0 && "${BS}" == "8" && "${ACCUM_STEPS}" == "1" ]]; then
    echo "[PROBE] bs=8 failed; retrying bs=4 accum=2"
    BS=4
    ACCUM_STEPS=2
    probe_root="${ROOT}/plans/0531_led_night/codex_smoke_led_hb_c2_probe_${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}_bs${BS}_acc${ACCUM_STEPS}"
    EPOCHS=1 run_c2_train "${probe_root}/exp" "${probe_root}/heavy" --max-train-steps 2 --max-val-samples 4 --num-workers 0 --log-interval 1
  elif [[ "${probe_status}" -ne 0 ]]; then
    exit "${probe_status}"
  fi
  rm -rf "${probe_root}"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}"
RUN_NAME="${RUN_TIMESTAMP}_led_hb_c2_d0only_residual_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}"
SAVE="${EXP_ROOT}/${RUN_NAME}"
HEAVY="${HEAVY_ROOT}/${RUN_NAME}"
LOG="${LOG_ROOT}/${RUN_NAME}.tmux.log"
if [[ -e "${SAVE}" || -e "${HEAVY}" ]]; then
  echo "[ERROR] refusing to overwrite ${SAVE} or ${HEAVY}" >&2
  exit 2
fi
echo "[FORMAL] run=${RUN_NAME}"
echo "[FORMAL] save=${SAVE}"
echo "[FORMAL] heavy=${HEAVY}"
run_c2_train "${SAVE}" "${HEAVY}" 2>&1 | tee -a "${LOG}"
echo "C2_RUN_DIR=${SAVE}"
echo "C2_CHECKPOINT=${HEAVY}/best_abs_rel.pth"
