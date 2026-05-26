#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-vkitti_rawadapter_feature_d0_ablation}"
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
RUN_M1_X3_D0="${RUN_M1_X3_D0:-1}"
RUN_FFM_MID_ONLY="${RUN_FFM_MID_ONLY:-1}"
RUN_X3_ONLY="${RUN_X3_ONLY:-1}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0526_run_vkitti_rawadapter_feature_d0_ablation_queue.sh

Starts one tmux queue. Each formal run computes its own MMDD_HHMM timestamp
at actual launch time. By default it runs smoke checks, then launches:
  1. M1 x3 with D0_norm concatenated
  2. ffm_mid only, no D0_norm in residual head input
  3. x3 only, no D0_norm in residual head input

Useful overrides:
  RUN_SMOKE=0 GPU=1 bash ...
  RUN_M1_X3_D0=0 RUN_FFM_MID_ONLY=1 RUN_X3_ONLY=0 bash ...
  EPOCHS=20 EVAL_KITTI=1 bash ...
EOF
  exit 0
fi

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' VKITTI_TRAIN_LIST='${VKITTI_TRAIN_LIST}' VKITTI_VAL_LIST='${VKITTI_VAL_LIST}' EVAL_KITTI='${EVAL_KITTI}' KITTI_BASE='${KITTI_BASE}' KITTI_VAL_SPLIT='${KITTI_VAL_SPLIT}' KITTI_EVAL_PROTOCOL='${KITTI_EVAL_PROTOCOL}' KITTI_EXPECTED_VAL_SAMPLES='${KITTI_EXPECTED_VAL_SAMPLES}' KITTI_NUM_WORKERS='${KITTI_NUM_WORKERS}' RUN_SMOKE='${RUN_SMOKE}' KEEP_SMOKE='${KEEP_SMOKE}' D0_SIGN='${D0_SIGN}' RUN_M1_X3_D0='${RUN_M1_X3_D0}' RUN_FFM_MID_ONLY='${RUN_FFM_MID_ONLY}' RUN_X3_ONLY='${RUN_X3_ONLY}' SPLIT_TAG='${SPLIT_TAG}' EPOCHS='${EPOCHS}' QUEUE_TIMESTAMP='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

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
  --raw-adapter-inverse-tone global_0p15
  --raw-adapter-ccm identity
  --raw-adapter-variant-policy normal
  --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0
  --raw-adapter-fixed-light-scale 1.0
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
  local run_tag="$2"
  local feature_source="$3"
  local d0_mode="$4"

  local run_timestamp
  run_timestamp="$(date +%m%d_%H%M)"
  local run_name="${run_timestamp}_${run_tag}_rawadapter_analytic_identity_normal_vits_halfraw187x621_${SPLIT_TAG}_bs8_e${EPOCHS}"
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
    printf ' --residual-feature-source %q --residual-head-d0-mode %q' "${feature_source}" "${d0_mode}"
    printf ' --epochs %q --bs 8 --num-workers 4 --log-interval 100 --save-interval 1 --eval-interval 1 --save-best-checkpoint' "${EPOCHS}"
    printf ' --save-path %q --heavy-save-path %q\n' "${save}" "${heavy}"
  } 2>&1 | tee -a "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_raw_residual.py \
    "${common_args[@]}" \
    --residual-feature-source "${feature_source}" \
    --residual-head-d0-mode "${d0_mode}" \
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

run_train_smoke() {
  local label="$1"
  local feature_source="$2"
  local d0_mode="$3"
  local smoke_root="$4"
  local smoke_dir="${smoke_root}/${label}"
  local smoke_log="${smoke_dir}/train_smoke.log"
  local smoke_kitti_args=()
  if [[ "${EVAL_KITTI}" == "1" ]]; then
    smoke_kitti_args+=(--max-kitti-val-samples 4)
  fi

  mkdir -p "${smoke_dir}"
  echo "[SMOKE] ${label} feature=${feature_source} d0_mode=${d0_mode} root=${smoke_dir}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_raw_residual.py \
    "${common_args[@]}" \
    --residual-feature-source "${feature_source}" \
    --residual-head-d0-mode "${d0_mode}" \
    --epochs 1 \
    --bs 8 \
    --num-workers 0 \
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --max-train-steps 2 \
    --max-val-samples 4 \
    "${smoke_kitti_args[@]}" \
    --save-path "${smoke_dir}/exp" \
    --heavy-save-path "${smoke_dir}/heavy" 2>&1 | tee -a "${smoke_log}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[SMOKE][ERROR] ${label} failed; kept ${smoke_dir}" >&2
    return "${status}"
  fi
}

run_smokes() {
  local smoke_root="$1"
  mkdir -p "${smoke_root}"
  run_train_smoke "smoke_m1_x3_d0concat" "x3" "concat" "${smoke_root}"
  run_train_smoke "smoke_ffm_mid_only" "ffm_mid" "none" "${smoke_root}"
  run_train_smoke "smoke_x3_only" "x3" "none" "${smoke_root}"
}

require_file "${PRETRAINED}"
require_file "${VKITTI_TRAIN_LIST}"
require_file "${VKITTI_VAL_LIST}"
if [[ "${EVAL_KITTI}" == "1" ]]; then
  require_file "${KITTI_VAL_SPLIT}"
  require_dir "${KITTI_BASE}"
fi

QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP:-$(date +%m%d_%H%M)}"
SMOKE_ROOT="${ROOT}/plans/0524_unprocessing/codex_smoke_0526_feature_d0_ablation_queue_${QUEUE_TIMESTAMP}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[QUEUE_TIMESTAMP] ${QUEUE_TIMESTAMP}"
echo "[SPLIT_TAG] ${SPLIT_TAG}"
echo "[D0_SIGN] ${D0_SIGN}"
echo "[EPOCHS] ${EPOCHS}"
echo "[PER_RUN_TIMESTAMP] enabled"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_smokes "${SMOKE_ROOT}"
fi

if [[ "${KEEP_SMOKE}" != "1" && -d "${SMOKE_ROOT}" ]]; then
  rm -rf "${SMOKE_ROOT}"
  echo "[SMOKE] removed successful smoke artifacts: ${SMOKE_ROOT}"
fi

if [[ "${RUN_M1_X3_D0}" == "1" ]]; then
  run_training "formal M1 x3 residual with D0_norm" \
    "vkitti_m1_ra0_x3_d0concat" \
    "x3" \
    "concat"
fi

if [[ "${RUN_FFM_MID_ONLY}" == "1" ]]; then
  run_training "formal ffm_mid-only residual head, no D0_norm input" \
    "vkitti_m2nod0_ra0_ffm_mid_only" \
    "ffm_mid" \
    "none"
fi

if [[ "${RUN_X3_ONLY}" == "1" ]]; then
  run_training "formal x3-only residual head, no D0_norm input" \
    "vkitti_m1nod0_ra0_x3_only" \
    "x3" \
    "none"
fi

echo "[QUEUE_END] $(date -Iseconds) status=0"
