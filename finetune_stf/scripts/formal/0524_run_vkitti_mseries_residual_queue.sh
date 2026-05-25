#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-vkitti_mseries_residual}"
SPLIT_TAG="${SPLIT_TAG:-sceneholdout_Scene20_n1000_seed42}"
EPOCHS="${EPOCHS:-20}"

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
RUN_SIGN_CHECK="${RUN_SIGN_CHECK:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
D0_SIGN="${D0_SIGN:-}"
RUN_M2="${RUN_M2:-1}"
RUN_M1="${RUN_M1:-0}"
RUN_M3="${RUN_M3:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0524_run_vkitti_mseries_residual_queue.sh

Starts one new tmux session. By default it runs smoke checks, infers D0 sign,
and launches formal M2 only.

Useful overrides:
  GPU=1 RUN_SMOKE=0 D0_SIGN=1 bash ...
  EPOCHS=20 bash ...
  RUN_M2=0 RUN_M1=1 RUN_M3=1 D0_SIGN=1 bash ...
  SPLIT_TAG=sceneholdout_Scene20 VKITTI_TRAIN_LIST=... VKITTI_VAL_LIST=... bash ...
  EVAL_KITTI=0 bash ...
  KITTI_BASE=/mnt/drive/kitti KITTI_VAL_SPLIT=/path/to/kitti/val.txt bash ...
  KITTI_EVAL_PROTOCOL=halfres_raw_canonical_even_pad_crop_affine_disp bash ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' VKITTI_TRAIN_LIST='${VKITTI_TRAIN_LIST}' VKITTI_VAL_LIST='${VKITTI_VAL_LIST}' EVAL_KITTI='${EVAL_KITTI}' KITTI_BASE='${KITTI_BASE}' KITTI_VAL_SPLIT='${KITTI_VAL_SPLIT}' KITTI_EVAL_PROTOCOL='${KITTI_EVAL_PROTOCOL}' KITTI_EXPECTED_VAL_SAMPLES='${KITTI_EXPECTED_VAL_SAMPLES}' KITTI_NUM_WORKERS='${KITTI_NUM_WORKERS}' RUN_SMOKE='${RUN_SMOKE}' RUN_SIGN_CHECK='${RUN_SIGN_CHECK}' KEEP_SMOKE='${KEEP_SMOKE}' D0_SIGN='${D0_SIGN}' RUN_M2='${RUN_M2}' RUN_M1='${RUN_M1}' RUN_M3='${RUN_M3}' SPLIT_TAG='${SPLIT_TAG}' EPOCHS='${EPOCHS}' RUN_TIMESTAMP='${run_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

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
  --residual-alpha 0.5
  --vkitti-unprocessing-preset sensor_linear_dual
  --randomize-unprocessing
  --hflip-prob 0.5
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

run_training() {
  local label="$1"
  local run_name="$2"
  local feature_source="$3"
  shift 3

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
  {
    echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} python foundation/tools/train_vkitti2_raw_residual.py"
    printf ' %q' "${common_args[@]}"
    printf ' --residual-feature-source %q --d0-sign %q' "${feature_source}" "${D0_SIGN}"
    printf ' %q' "$@"
    printf ' --save-path %q --heavy-save-path %q\n' "${save}" "${heavy}"
  } 2>&1 | tee -a "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_raw_residual.py \
    "${common_args[@]}" \
    --residual-feature-source "${feature_source}" \
    --d0-sign "${D0_SIGN}" \
    "$@" \
    --save-path "${save}" \
    --heavy-save-path "${heavy}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

run_smoke_train() {
  local smoke_root="$1"
  local smoke_log="${smoke_root}/train_smoke.log"
  local smoke_kitti_args=()
  if [[ "${EVAL_KITTI}" == "1" ]]; then
    smoke_kitti_args+=(--max-kitti-val-samples 4)
  fi
  mkdir -p "${smoke_root}"
  echo "[SMOKE] train smoke root=${smoke_root}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_raw_residual.py \
    "${common_args[@]}" \
    --residual-feature-source ffm_mid \
    --d0-sign 1 \
    --epochs 1 \
    --bs 8 \
    --num-workers 0 \
    --log-interval 1 \
    --max-train-steps 2 \
    --max-val-samples 4 \
    "${smoke_kitti_args[@]}" \
    --save-path "${smoke_root}/exp" \
    --heavy-save-path "${smoke_root}/heavy" 2>&1 | tee -a "${smoke_log}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[SMOKE][ERROR] train smoke failed; kept ${smoke_root}" >&2
    return "${status}"
  fi
}

run_sign_check() {
  local smoke_root="$1"
  local sign_json="${smoke_root}/vkitti_d0_sign_summary.json"
  mkdir -p "${smoke_root}"
  echo "[SIGN] checking D0 sign -> ${sign_json}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/check_vkitti_dav2_sign.py \
    --encoder vits \
    --pretrained-from "${PRETRAINED}" \
    --vkitti-val-list "${VKITTI_VAL_LIST}" \
    --input-height 187 \
    --input-width 621 \
    --raw-storage-format synthetic_packed_bayer_4ch_halfres \
    --fullres-even-policy crop_bottom_to_even \
    --rgb-input-space halfres_2x2_area \
    --depth-target-space halfres_2x2_valid_mean \
    --min-depth 1.0 \
    --max-depth 80.0 \
    --max-samples 64 \
    --output "${sign_json}"
  D0_SIGN="$("${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import json; print(json.load(open('${sign_json}', 'r', encoding='utf-8'))['recommended_d0_sign'])")"
  export D0_SIGN
  echo "[SIGN] recommended --d0-sign ${D0_SIGN}"
}

require_file "${PRETRAINED}"
require_file "${VKITTI_TRAIN_LIST}"
require_file "${VKITTI_VAL_LIST}"
if [[ "${EVAL_KITTI}" == "1" ]]; then
  require_file "${KITTI_VAL_SPLIT}"
  require_dir "${KITTI_BASE}"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}"
SMOKE_ROOT="${ROOT}/plans/0524_new/codex_smoke_vkitti_mseries_queue_${RUN_TIMESTAMP}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[RUN_TIMESTAMP] ${RUN_TIMESTAMP}"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[TRAIN_LIST] ${VKITTI_TRAIN_LIST}"
echo "[VAL_LIST] ${VKITTI_VAL_LIST}"
echo "[EPOCHS] ${EPOCHS}"
echo "[EVAL_KITTI] ${EVAL_KITTI}"
if [[ "${EVAL_KITTI}" == "1" ]]; then
  echo "[KITTI_BASE] ${KITTI_BASE}"
  echo "[KITTI_VAL_SPLIT] ${KITTI_VAL_SPLIT}"
  echo "[KITTI_EVAL_PROTOCOL] ${KITTI_EVAL_PROTOCOL}"
  echo "[KITTI_EXPECTED_VAL_SAMPLES] ${KITTI_EXPECTED_VAL_SAMPLES}"
fi
echo "[SPLIT_TAG] ${SPLIT_TAG:-none}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_smoke_train "${SMOKE_ROOT}"
fi

if [[ -z "${D0_SIGN}" && "${RUN_SIGN_CHECK}" == "1" ]]; then
  run_sign_check "${SMOKE_ROOT}"
fi

if [[ -z "${D0_SIGN}" ]]; then
  echo "[ERROR] D0_SIGN is empty. Set D0_SIGN=1 or D0_SIGN=-1, or enable RUN_SIGN_CHECK=1." >&2
  exit 2
fi

if [[ "${KEEP_SMOKE}" != "1" && -d "${SMOKE_ROOT}" ]]; then
  rm -rf "${SMOKE_ROOT}"
  echo "[SMOKE] removed successful smoke artifacts: ${SMOKE_ROOT}"
fi

if [[ "${RUN_M2}" == "1" ]]; then
  run_training "formal M2 ffm_mid residual" "${RUN_TIMESTAMP}_vkitti_m2_ffm_mid_residual_vits_halfraw187x621${SPLIT_TAG:+_${SPLIT_TAG}}_bs8_e${EPOCHS}" "ffm_mid" \
    --epochs "${EPOCHS}" \
    --bs 8 \
    --num-workers 4 \
    --log-interval 100 \
    --save-interval 1 \
    --eval-interval 1 \
    --save-best-checkpoint
fi

if [[ "${RUN_M1}" == "1" ]]; then
  run_training "formal M1 x3 residual" "${RUN_TIMESTAMP}_vkitti_m1_x3_residual_vits_halfraw187x621${SPLIT_TAG:+_${SPLIT_TAG}}_bs8_e${EPOCHS}" "x3" \
    --epochs "${EPOCHS}" \
    --bs 8 \
    --num-workers 4 \
    --log-interval 100 \
    --save-interval 1 \
    --eval-interval 1 \
    --save-best-checkpoint
fi

if [[ "${RUN_M3}" == "1" ]]; then
  run_training "formal M3 x3_ffm_mid residual" "${RUN_TIMESTAMP}_vkitti_m3_x3_ffm_mid_residual_vits_halfraw187x621${SPLIT_TAG:+_${SPLIT_TAG}}_bs8_e${EPOCHS}" "x3_ffm_mid" \
    --epochs "${EPOCHS}" \
    --bs 8 \
    --num-workers 4 \
    --log-interval 100 \
    --save-interval 1 \
    --eval-interval 1 \
    --save-best-checkpoint
fi

echo "[QUEUE_END] $(date -Iseconds) status=0 d0_sign=${D0_SIGN}"
