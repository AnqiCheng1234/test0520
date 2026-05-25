#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-vkitti_mseries_rawadapter}"
SPLIT_TAG="${SPLIT_TAG:-sceneholdout_Scene20_n1000_seed42}"
EPOCHS="${EPOCHS:-20}"
D0_SIGN="${D0_SIGN:-1}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt}"
VKITTI_VAL_LIST="${VKITTI_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt}"
EVAL_KITTI="${EVAL_KITTI:-1}"
KITTI_BASE="${KITTI_BASE:-/mnt/drive/kitti}"
KITTI_VAL_SPLIT="${KITTI_VAL_SPLIT:-/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt}"
KITTI_EVAL_PROTOCOL="${KITTI_EVAL_PROTOCOL:-halfres_raw_canonical_even_pad_crop_affine_disp}"
KITTI_EXPECTED_VAL_SAMPLES="${KITTI_EXPECTED_VAL_SAMPLES:-652}"
KITTI_NUM_WORKERS="${KITTI_NUM_WORKERS:-2}"

RUN_SMOKE="${RUN_SMOKE:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
RUN_RA0="${RUN_RA0:-1}"
RUN_RA1="${RUN_RA1:-0}"
RUN_RA2="${RUN_RA2:-0}"
RUN_RA3="${RUN_RA3:-0}"
RUN_RA4="${RUN_RA4:-0}"
RA1_DARK_LIGHT_SCALE="${RA1_DARK_LIGHT_SCALE:-0.20}"
RA2_OVER_LIGHT_SCALE="${RA2_OVER_LIGHT_SCALE:-1.80}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0525_run_vkitti_mseries_rawadapter_queue.sh

Starts one new tmux session. By default it runs smoke checks and launches only
RA0. Useful overrides:
  RUN_SMOKE=0 GPU=1 D0_SIGN=1 bash ...
  RUN_RA0=0 RUN_RA3=1 bash ...
  EPOCHS=20 EVAL_KITTI=1 bash ...
EOF
  exit 0
fi

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' VKITTI_TRAIN_LIST='${VKITTI_TRAIN_LIST}' VKITTI_VAL_LIST='${VKITTI_VAL_LIST}' EVAL_KITTI='${EVAL_KITTI}' KITTI_BASE='${KITTI_BASE}' KITTI_VAL_SPLIT='${KITTI_VAL_SPLIT}' KITTI_EVAL_PROTOCOL='${KITTI_EVAL_PROTOCOL}' KITTI_EXPECTED_VAL_SAMPLES='${KITTI_EXPECTED_VAL_SAMPLES}' KITTI_NUM_WORKERS='${KITTI_NUM_WORKERS}' RUN_SMOKE='${RUN_SMOKE}' KEEP_SMOKE='${KEEP_SMOKE}' D0_SIGN='${D0_SIGN}' RUN_RA0='${RUN_RA0}' RUN_RA1='${RUN_RA1}' RUN_RA2='${RUN_RA2}' RUN_RA3='${RUN_RA3}' RUN_RA4='${RUN_RA4}' RA1_DARK_LIGHT_SCALE='${RA1_DARK_LIGHT_SCALE}' RA2_OVER_LIGHT_SCALE='${RA2_OVER_LIGHT_SCALE}' SPLIT_TAG='${SPLIT_TAG}' EPOCHS='${EPOCHS}' RUN_TIMESTAMP='${run_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

  cat <<EOF
Started tmux session: ${session}
Queue log: ${queue_log}
Attach: tmux attach -t ${session}
Monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file not found: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 2
  fi
}

variant_weights() {
  case "$1" in
    normal) echo "normal=1.0,dark=0.0,over=0.0" ;;
    dark) echo "normal=0.0,dark=1.0,over=0.0" ;;
    over) echo "normal=0.0,dark=0.0,over=1.0" ;;
    *) echo "[ERROR] Unknown variant $1" >&2; return 2 ;;
  esac
}

common_args=(
  --input-domain raw4
  --model-input-tensor raw
  --raw-storage-format synthetic_packed_bayer_4ch_halfres
  --fullres-even-policy crop_bottom_to_even
  --rgb-input-space halfres_2x2_area
  --depth-target-space halfres_2x2_valid_mean
  --front-end raw_to_base_rgb_ram3
  --encoder vits
  --pretrained-from "${PRETRAINED}"
  --vkitti-train-list "${VKITTI_TRAIN_LIST}"
  --vkitti-val-list "${VKITTI_VAL_LIST}"
  --input-height 187
  --input-width 621
  --min-depth 1.0
  --max-depth 80.0
  --residual-feature-source ffm_mid
  --residual-alpha 0.5
  --d0-sign "${D0_SIGN}"
  --unprocessing-method raw_adapter_style
  --vkitti-unprocessing-preset not_applicable
  --no-randomize-unprocessing
  --hflip-prob 0.5
  --raw-adapter-backend analytic
  --raw-adapter-cfa-pattern RGGB
  --raw-adapter-packed-channel-order R_Gr_Gb_B
  --raw-adapter-rgb-transfer srgb_piecewise
  --raw-adapter-red-gain-range 1.9 2.4
  --raw-adapter-blue-gain-range 1.5 1.9
  --raw-adapter-fixed-red-gain 2.15
  --raw-adapter-fixed-blue-gain 1.70
  --raw-adapter-dark-light-scale-range 0.05 0.4
  --raw-adapter-over-light-scale-range 1.5 2.5
  --raw-adapter-shot-noise 0.001
  --raw-adapter-read-noise 0.0005
  --raw-adapter-noise-mean-mode zero
  --raw-adapter-black-level 0.0
  --raw-adapter-white-level 1.0
  --raw-adapter-random-seed-policy dataloader_generator
  --accum-steps 1
  --lr 1e-4
  --weight-decay 1e-4
  --amp
  --amp-dtype bf16
  --seed 42
)

if [[ "${EVAL_KITTI}" == "1" ]]; then
  common_args+=(
    --eval-kitti
    --kitti-base "${KITTI_BASE}"
    --kitti-val-split "${KITTI_VAL_SPLIT}"
    --kitti-eval-protocol "${KITTI_EVAL_PROTOCOL}"
    --kitti-expected-val-samples "${KITTI_EXPECTED_VAL_SAMPLES}"
    --kitti-num-workers "${KITTI_NUM_WORKERS}"
  )
fi

log_header() {
  local label="$1"
  local run="$2"
  local log="$3"
  {
    echo "[START] $(date -Iseconds)"
    echo "[LABEL] ${label}"
    echo "[RUN] ${run}"
    echo "[SESSION] ${QUEUE_SESSION:-internal}"
    echo "[HOST] $(hostname)"
    echo "[USER] $(whoami)"
    echo "[PWD] $(pwd)"
    echo "[GPU] ${GPU}"
    echo "[LOG] ${log}"
  } 2>&1 | tee -a "${log}"
}

run_training() {
  local label="$1"
  local run_name="$2"
  local inverse_tone="$3"
  local ccm="$4"
  local variant="$5"
  local fixed_light="$6"

  local save="${EXP_ROOT}/${run_name}"
  local heavy="${HEAVY_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}.tmux.log"

  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] Refusing to overwrite existing artifacts for ${run_name}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi

  mkdir -p "${save}"
  log_header "${label}" "${run_name}" "${log}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_raw_residual.py \
    "${common_args[@]}" \
    --raw-adapter-inverse-tone "${inverse_tone}" \
    --raw-adapter-ccm "${ccm}" \
    --raw-adapter-variant-policy "${variant}" \
    --raw-adapter-variant-weights "$(variant_weights "${variant}")" \
    --raw-adapter-fixed-light-scale "${fixed_light}" \
    --epochs "${EPOCHS}" \
    --bs 8 \
    --num-workers 4 \
    --log-interval 100 \
    --save-interval 1 \
    --eval-interval 1 \
    --save-best-checkpoint \
    --save-path "${save}" \
    --heavy-save-path "${heavy}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

run_smokes() {
  local smoke_root="$1"
  mkdir -p "${smoke_root}"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/smoke_raw_adapter_style_unprocessing.py \
    --output "${smoke_root}/transform/parity.json"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/smoke_vkitti2_raw_adapter_dataset.py \
    --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
    --vkitti-val-list "${VKITTI_VAL_LIST}" \
    --input-height 187 \
    --input-width 621 \
    --raw-storage-format synthetic_packed_bayer_4ch_halfres \
    --fullres-even-policy crop_bottom_to_even \
    --rgb-input-space halfres_2x2_area \
    --depth-target-space halfres_2x2_valid_mean \
    --unprocessing-method raw_adapter_style \
    --vkitti-unprocessing-preset not_applicable \
    --no-randomize-unprocessing \
    --raw-adapter-backend analytic \
    --raw-adapter-cfa-pattern RGGB \
    --raw-adapter-packed-channel-order R_Gr_Gb_B \
    --raw-adapter-rgb-transfer srgb_piecewise \
    --raw-adapter-inverse-tone global_0p15 \
    --raw-adapter-ccm identity \
    --raw-adapter-red-gain-range 1.9 2.4 \
    --raw-adapter-blue-gain-range 1.5 1.9 \
    --raw-adapter-fixed-red-gain 2.15 \
    --raw-adapter-fixed-blue-gain 1.70 \
    --raw-adapter-variant-policy normal \
    --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0 \
    --raw-adapter-fixed-light-scale 1.0 \
    --raw-adapter-dark-light-scale-range 0.05 0.4 \
    --raw-adapter-over-light-scale-range 1.5 2.5 \
    --raw-adapter-shot-noise 0.001 \
    --raw-adapter-read-noise 0.0005 \
    --raw-adapter-noise-mean-mode zero \
    --raw-adapter-black-level 0.0 \
    --raw-adapter-white-level 1.0 \
    --raw-adapter-random-seed-policy dataloader_generator \
    --output "${smoke_root}/dataset/summary.json"

  local smoke_kitti_args=()
  if [[ "${EVAL_KITTI}" == "1" ]]; then
    smoke_kitti_args+=(--max-kitti-val-samples 4)
  fi
  mkdir -p "${smoke_root}/train"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_raw_residual.py \
    "${common_args[@]}" \
    --raw-adapter-inverse-tone global_0p15 \
    --raw-adapter-ccm identity \
    --raw-adapter-variant-policy normal \
    --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0 \
    --raw-adapter-fixed-light-scale 1.0 \
    --epochs 1 \
    --bs 8 \
    --num-workers 0 \
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --max-train-steps 2 \
    --max-val-samples 4 \
    "${smoke_kitti_args[@]}" \
    --save-path "${smoke_root}/train/exp" \
    --heavy-save-path "${smoke_root}/train/heavy" 2>&1 | tee -a "${smoke_root}/train/train_smoke.log"
}

require_file "${PRETRAINED}"
require_file "${VKITTI_TRAIN_LIST}"
require_file "${VKITTI_VAL_LIST}"
if [[ "${EVAL_KITTI}" == "1" ]]; then
  require_file "${KITTI_VAL_SPLIT}"
  require_dir "${KITTI_BASE}"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}"
SMOKE_ROOT="${ROOT}/plans/0524_unprocessing/codex_smoke_0525_rawadapter_queue_${RUN_TIMESTAMP}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[RUN_TIMESTAMP] ${RUN_TIMESTAMP}"
echo "[SPLIT_TAG] ${SPLIT_TAG}"
echo "[D0_SIGN] ${D0_SIGN}"
echo "[EPOCHS] ${EPOCHS}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_smokes "${SMOKE_ROOT}"
fi

if [[ "${KEEP_SMOKE}" != "1" && -d "${SMOKE_ROOT}" ]]; then
  rm -rf "${SMOKE_ROOT}"
  echo "[SMOKE] removed successful smoke artifacts: ${SMOKE_ROOT}"
fi

if [[ "${RUN_RA0}" == "1" ]]; then
  run_training "formal M2-RA0 rawadapter analytic identity normal" \
    "${RUN_TIMESTAMP}_vkitti_m2_ra0_rawadapter_analytic_identity_normal_vits_halfraw187x621_${SPLIT_TAG}_bs8_e${EPOCHS}" \
    "global_0p15" "identity" "normal" "1.0"
fi

if [[ "${RUN_RA1}" == "1" ]]; then
  run_training "formal M2-RA1 rawadapter fixed dark" \
    "${RUN_TIMESTAMP}_vkitti_m2_ra1_rawadapter_analytic_identity_dark_vits_halfraw187x621_${SPLIT_TAG}_bs8_e${EPOCHS}" \
    "global_0p15" "identity" "dark" "${RA1_DARK_LIGHT_SCALE}"
fi

if [[ "${RUN_RA2}" == "1" ]]; then
  run_training "formal M2-RA2 rawadapter fixed over" \
    "${RUN_TIMESTAMP}_vkitti_m2_ra2_rawadapter_analytic_identity_over_vits_halfraw187x621_${SPLIT_TAG}_bs8_e${EPOCHS}" \
    "global_0p15" "identity" "over" "${RA2_OVER_LIGHT_SCALE}"
fi

if [[ "${RUN_RA3}" == "1" ]]; then
  run_training "formal M2-RA3 rawadapter ccm tone normal" \
    "${RUN_TIMESTAMP}_vkitti_m2_ra3_rawadapter_analytic_genericd65_normal_vits_halfraw187x621_${SPLIT_TAG}_bs8_e${EPOCHS}" \
    "global_0p15" "generic_d65" "normal" "1.0"
fi

if [[ "${RUN_RA4}" == "1" ]]; then
  run_training "formal M2-RA4 rawadapter no-tone normal" \
    "${RUN_TIMESTAMP}_vkitti_m2_ra4_rawadapter_analytic_notone_identity_normal_vits_halfraw187x621_${SPLIT_TAG}_bs8_e${EPOCHS}" \
    "none" "identity" "normal" "1.0"
fi

echo "[QUEUE_END] $(date -Iseconds) status=0"
