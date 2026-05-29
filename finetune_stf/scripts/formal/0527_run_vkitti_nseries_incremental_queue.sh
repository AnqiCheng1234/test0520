#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-vkitti_nseries_incremental}"
SPLIT_TAG="${SPLIT_TAG:-sceneholdout_Scene20_n1000_seed42}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"

RUN_SMOKE="${RUN_SMOKE:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
RUN_N2_LP_SWEEP="${RUN_N2_LP_SWEEP:-1}"
RUN_N2_Q_SWEEP="${RUN_N2_Q_SWEEP:-0}"
RUN_N2_LOWFREQ_SWEEP="${RUN_N2_LOWFREQ_SWEEP:-0}"
RUN_N3_RGB="${RUN_N3_RGB:-0}"
RUN_N4_FFM="${RUN_N4_FFM:-0}"
RUN_N5_D1="${RUN_N5_D1:-0}"
RUN_N7_STOPGRAD="${RUN_N7_STOPGRAD:-0}"

N2_LAMBDA_LP_LIST="${N2_LAMBDA_LP_LIST:-0.0 0.3 0.5 0.8}"
N2_Q_GOOD_LIST="${N2_Q_GOOD_LIST:-0.3 0.5 0.7}"
N2_LOWFREQ_LOSS_LIST="${N2_LOWFREQ_LOSS_LIST:-0.0 0.05 0.1}"
SELECTED_LAMBDA_LP="${SELECTED_LAMBDA_LP:-0.5}"
SELECTED_Q_GOOD="${SELECTED_Q_GOOD:-0.5}"
SELECTED_LAMBDA_LOWFREQ_LOSS="${SELECTED_LAMBDA_LOWFREQ_LOSS:-0.0}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
C2_CHECKPOINT="${C2_CHECKPOINT:-/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_11.pth}"
C2_RUN_DIR="${C2_RUN_DIR:-${ROOT}/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt}"
VKITTI_VAL_LIST="${VKITTI_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt}"
KITTI_BASE="${KITTI_BASE:-/mnt/drive/kitti}"
KITTI_VAL_SPLIT="${KITTI_VAL_SPLIT:-/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt}"
KITTI_EXPECTED_VAL_SAMPLES="${KITTI_EXPECTED_VAL_SAMPLES:-652}"
KITTI_NUM_WORKERS="${KITTI_NUM_WORKERS:-2}"

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' C2_CHECKPOINT='${C2_CHECKPOINT}' C2_RUN_DIR='${C2_RUN_DIR}' VKITTI_TRAIN_LIST='${VKITTI_TRAIN_LIST}' VKITTI_VAL_LIST='${VKITTI_VAL_LIST}' KITTI_BASE='${KITTI_BASE}' KITTI_VAL_SPLIT='${KITTI_VAL_SPLIT}' KITTI_EXPECTED_VAL_SAMPLES='${KITTI_EXPECTED_VAL_SAMPLES}' KITTI_NUM_WORKERS='${KITTI_NUM_WORKERS}' RUN_SMOKE='${RUN_SMOKE}' KEEP_SMOKE='${KEEP_SMOKE}' RUN_N2_LP_SWEEP='${RUN_N2_LP_SWEEP}' RUN_N2_Q_SWEEP='${RUN_N2_Q_SWEEP}' RUN_N2_LOWFREQ_SWEEP='${RUN_N2_LOWFREQ_SWEEP}' RUN_N3_RGB='${RUN_N3_RGB}' RUN_N4_FFM='${RUN_N4_FFM}' RUN_N5_D1='${RUN_N5_D1}' RUN_N7_STOPGRAD='${RUN_N7_STOPGRAD}' N2_LAMBDA_LP_LIST='${N2_LAMBDA_LP_LIST}' N2_Q_GOOD_LIST='${N2_Q_GOOD_LIST}' N2_LOWFREQ_LOSS_LIST='${N2_LOWFREQ_LOSS_LIST}' SELECTED_LAMBDA_LP='${SELECTED_LAMBDA_LP}' SELECTED_Q_GOOD='${SELECTED_Q_GOOD}' SELECTED_LAMBDA_LOWFREQ_LOSS='${SELECTED_LAMBDA_LOWFREQ_LOSS}' SPLIT_TAG='${SPLIT_TAG}' EPOCHS='${EPOCHS}' BS='${BS}' QUEUE_TIMESTAMP='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
Started tmux session: ${session}
Queue log: ${queue_log}
Attach: tmux attach -t ${session}
Monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}" "${HEAVY_ROOT}"

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] Required file not found: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "[ERROR] Required directory not found: $1" >&2; exit 2; }
}
tag_float() {
  local value="$1"
  if [[ "${value}" == "not_applicable" ]]; then
    echo "na"
  else
    echo "${value}" | sed 's/\./p/g'
  fi
}

require_file "${PRETRAINED}"
require_file "${C2_CHECKPOINT}"
require_dir "${C2_RUN_DIR}"
require_file "${VKITTI_TRAIN_LIST}"
require_file "${VKITTI_VAL_LIST}"
require_file "${KITTI_VAL_SPLIT}"
require_dir "${KITTI_BASE}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "[SMOKE] starting smoke before formal queue"
  KEEP_SMOKE="${KEEP_SMOKE}" GPU="${GPU}" CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" \
    PRETRAINED="${PRETRAINED}" C2_CHECKPOINT="${C2_CHECKPOINT}" C2_RUN_DIR="${C2_RUN_DIR}" \
    VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST}" VKITTI_VAL_LIST="${VKITTI_VAL_LIST}" \
    KITTI_BASE="${KITTI_BASE}" KITTI_VAL_SPLIT="${KITTI_VAL_SPLIT}" \
    bash finetune_stf/scripts/smoke/0527_smoke_vkitti_incremental_nseries.sh
fi

common_args=(
  --encoder vits
  --pretrained-from "${PRETRAINED}"
  --c2-checkpoint "${C2_CHECKPOINT}"
  --c2-run-dir "${C2_RUN_DIR}"
  --vkitti-train-list "${VKITTI_TRAIN_LIST}"
  --vkitti-val-list "${VKITTI_VAL_LIST}"
  --dataset-geometry-mode vkitti2_even_fullres_halfres_2x2
  --fullres-even-policy crop_bottom_to_even
  --rgb-input-space halfres_2x2_area
  --depth-target-space halfres_2x2_valid_mean
  --input-height 187
  --input-width 621
  --min-depth 1.0
  --max-depth 80.0
  --residual-alpha 0.5
  --d0-sign 1
  --lowpass-kernel 31
  --lambda-final 1.0
  --lambda-boundary 2.0
  --lambda-grad 0.5
  --lambda-keep-good-d1 0.2
  --lambda-gate-sparse 0.05
  --lambda-invalid-keep 0.1
  --eval-protocol per_image_affine_disp_depth_anything_v2
  --eval-kitti
  --kitti-base "${KITTI_BASE}"
  --kitti-val-split "${KITTI_VAL_SPLIT}"
  --kitti-expected-val-samples "${KITTI_EXPECTED_VAL_SAMPLES}"
  --kitti-num-workers "${KITTI_NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --bs "${BS}"
  --accum-steps 1
  --lr 1e-4
  --weight-decay 1e-4
  --num-workers 4
  --hflip-prob 0.5
  --log-interval 100
  --save-interval 1
  --eval-interval 1
  --save-best-checkpoint
  --amp
  --amp-dtype bf16
  --seed 42
)

raw_adapter_args=(
  --unprocessing-method raw_adapter_style
  --vkitti-unprocessing-preset not_applicable
  --no-randomize-unprocessing
  --raw-adapter-backend analytic
  --raw-adapter-cfa-pattern RGGB
  --raw-adapter-packed-channel-order R_Gr_Gb_B
  --raw-adapter-rgb-transfer srgb_piecewise
  --raw-adapter-inverse-tone global_0p15
  --raw-adapter-ccm identity
  --raw-adapter-red-gain-range 1.9 2.4
  --raw-adapter-blue-gain-range 1.5 1.9
  --raw-adapter-fixed-red-gain 2.15
  --raw-adapter-fixed-blue-gain 1.70
  --raw-adapter-fixed-light-scale 1.0
  --raw-adapter-dark-light-scale-range 0.05 0.4
  --raw-adapter-over-light-scale-range 1.5 2.5
  --raw-adapter-shot-noise 0.001
  --raw-adapter-read-noise 0.0005
  --raw-adapter-noise-mean-mode zero
  --raw-adapter-black-level 0.0
  --raw-adapter-white-level 1.0
  --raw-adapter-random-seed-policy dataloader_generator
  --raw-adapter-variant-policy normal
  --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0
)

rgb_na_args=(
  --unprocessing-method not_applicable
  --vkitti-unprocessing-preset not_applicable
  --raw-adapter-backend not_applicable
  --raw-adapter-cfa-pattern not_applicable
  --raw-adapter-packed-channel-order not_applicable
  --raw-adapter-rgb-transfer not_applicable
  --raw-adapter-inverse-tone not_applicable
  --raw-adapter-ccm not_applicable
  --raw-adapter-red-gain-range not_applicable
  --raw-adapter-blue-gain-range not_applicable
  --raw-adapter-fixed-red-gain not_applicable
  --raw-adapter-fixed-blue-gain not_applicable
  --raw-adapter-fixed-light-scale not_applicable
  --raw-adapter-dark-light-scale-range not_applicable
  --raw-adapter-over-light-scale-range not_applicable
  --raw-adapter-shot-noise not_applicable
  --raw-adapter-read-noise not_applicable
  --raw-adapter-noise-mean-mode not_applicable
  --raw-adapter-black-level not_applicable
  --raw-adapter-white-level not_applicable
  --raw-adapter-random-seed-policy not_applicable
  --raw-adapter-variant-policy not_applicable
  --raw-adapter-variant-weights not_applicable
)

method_contract() {
  local method="$1"
  case "${method}" in
    N2) echo "x3 feature_only feature_d1 true raw4 raw c2_frozen_raw_ram_incremental halfres_raw_canonical_even_pad_crop_affine_disp" ;;
    N3) echo "rgb feature_only feature_d1 not_applicable rgb image c2_frozen_rgb_incremental halfres_rgb_canonical_even_pad_crop_affine_disp" ;;
    N4) echo "ffm_mid feature_only feature_d1 true raw4 raw c2_frozen_raw_ram_incremental halfres_raw_canonical_even_pad_crop_affine_disp" ;;
    N5) echo "d1 d1_only d1_only not_applicable rgb image c2_frozen_d1_incremental halfres_rgb_canonical_even_pad_crop_affine_disp" ;;
    N7) echo "x3 feature_d1_stopgrad feature_d1 true raw4 raw c2_frozen_raw_ram_incremental halfres_raw_canonical_even_pad_crop_affine_disp" ;;
    *) echo "[ERROR] Unknown method ${method}" >&2; return 2 ;;
  esac
}

run_training() {
  local method="$1"
  local lambda_lp="$2"
  local q_good="$3"
  local lambda_lowfreq_loss="$4"
  read -r feature delta_condition gate_condition rft input_domain model_input front_end kitti_protocol < <(method_contract "${method}")
  local raw_storage="not_applicable"
  local unproc_args=("${rgb_na_args[@]}")
  if [[ "${input_domain}" == "raw4" ]]; then
    raw_storage="synthetic_packed_bayer_4ch_halfres"
    unproc_args=("${raw_adapter_args[@]}")
  fi
  local run_timestamp
  run_timestamp="$(date +%m%d_%H%M)"
  local run_name="${run_timestamp}_vkitti_${method,,}_${feature}_lp$(tag_float "${lambda_lp}")_q$(tag_float "${q_good}")_lfl$(tag_float "${lambda_lowfreq_loss}")_rft$(tag_float "${rft}")_vits_half187x621_${SPLIT_TAG}_bs${BS}_e${EPOCHS}"
  local save="${EXP_ROOT}/${run_name}"
  local heavy="${HEAVY_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}.tmux.log"
  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] Refusing to overwrite artifacts for ${run_name}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi
  mkdir -p "${save}"
  {
    echo "[START] $(date -Iseconds)"
    echo "[RUN] ${run_name}"
    echo "[METHOD] ${method}"
    echo "[HOST] $(hostname)"
    echo "[USER] $(whoami)"
    echo "[PWD] $(pwd)"
    echo "[GPU] ${GPU}"
    echo "[C2_METADATA]"
  } 2>&1 | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python -c "import torch; ck=torch.load('${C2_CHECKPOINT}', map_location='cpu'); print(ck.get('args', {}))" 2>&1 | tee -a "${log}"
  {
    echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} python foundation/tools/train_vkitti2_incremental_residual.py"
    printf ' %q' "${common_args[@]}"
    printf ' --method-id %q --input-domain %q --model-input-tensor %q --raw-storage-format %q --front-end %q' "${method}" "${input_domain}" "${model_input}" "${raw_storage}" "${front_end}"
    printf ' --incremental-feature-source %q --delta-condition %q --gate-condition %q --raw-feature-encoder-trainable %q' "${feature}" "${delta_condition}" "${gate_condition}" "${rft}"
    printf ' --lambda-lp %q --q-good %q --lambda-lowfreq-loss %q --kitti-eval-protocol %q' "${lambda_lp}" "${q_good}" "${lambda_lowfreq_loss}" "${kitti_protocol}"
    printf ' %q' "${unproc_args[@]}"
    printf ' --save-path %q --heavy-save-path %q\n' "${save}" "${heavy}"
  } 2>&1 | tee -a "${log}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_incremental_residual.py \
    "${common_args[@]}" \
    --method-id "${method}" \
    --input-domain "${input_domain}" \
    --model-input-tensor "${model_input}" \
    --raw-storage-format "${raw_storage}" \
    --front-end "${front_end}" \
    --incremental-feature-source "${feature}" \
    --delta-condition "${delta_condition}" \
    --gate-condition "${gate_condition}" \
    --raw-feature-encoder-trainable "${rft}" \
    --lambda-lp "${lambda_lp}" \
    --q-good "${q_good}" \
    --lambda-lowfreq-loss "${lambda_lowfreq_loss}" \
    --kitti-eval-protocol "${kitti_protocol}" \
    "${unproc_args[@]}" \
    --save-path "${save}" \
    --heavy-save-path "${heavy}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

if [[ "${RUN_N2_LP_SWEEP}" == "1" ]]; then
  for lp in ${N2_LAMBDA_LP_LIST}; do
    run_training N2 "${lp}" 0.5 0.0
  done
fi
if [[ "${RUN_N2_Q_SWEEP}" == "1" ]]; then
  for q in ${N2_Q_GOOD_LIST}; do
    run_training N2 "${SELECTED_LAMBDA_LP}" "${q}" 0.0
  done
fi
if [[ "${RUN_N2_LOWFREQ_SWEEP}" == "1" ]]; then
  for lfl in ${N2_LOWFREQ_LOSS_LIST}; do
    run_training N2 "${SELECTED_LAMBDA_LP}" "${SELECTED_Q_GOOD}" "${lfl}"
  done
fi
if [[ "${RUN_N3_RGB}" == "1" ]]; then
  run_training N3 "${SELECTED_LAMBDA_LP}" "${SELECTED_Q_GOOD}" "${SELECTED_LAMBDA_LOWFREQ_LOSS}"
fi
if [[ "${RUN_N4_FFM}" == "1" ]]; then
  run_training N4 "${SELECTED_LAMBDA_LP}" "${SELECTED_Q_GOOD}" "${SELECTED_LAMBDA_LOWFREQ_LOSS}"
fi
if [[ "${RUN_N5_D1}" == "1" ]]; then
  run_training N5 "${SELECTED_LAMBDA_LP}" "${SELECTED_Q_GOOD}" "${SELECTED_LAMBDA_LOWFREQ_LOSS}"
fi
if [[ "${RUN_N7_STOPGRAD}" == "1" ]]; then
  run_training N7 "${SELECTED_LAMBDA_LP}" "${SELECTED_Q_GOOD}" "${SELECTED_LAMBDA_LOWFREQ_LOSS}"
fi

echo "[DONE] N-series queue completed at $(date -Iseconds)"
