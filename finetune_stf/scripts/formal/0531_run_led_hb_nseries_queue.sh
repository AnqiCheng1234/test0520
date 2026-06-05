#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-led_hb_nseries}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LED_TRAIN_LIST="${LED_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_train_all.txt}"
LED_VAL_LIST="${LED_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/led_hb/hb_val_stride5_n1000_seed42.txt}"
C2_RUN_DIR="${C2_RUN_DIR:?set completed LED-HB C2 run dir}"
C2_CHECKPOINT="${C2_CHECKPOINT:?set completed LED-HB C2 checkpoint}"
D0_SIGN="${D0_SIGN:-}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
RUN_N5_D1="${RUN_N5_D1:-1}"
RUN_N3_RGB="${RUN_N3_RGB:-1}"
RUN_N2_X3="${RUN_N2_X3:-1}"
RUN_N7_STOPGRAD="${RUN_N7_STOPGRAD:-1}"

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LED_TRAIN_LIST='${LED_TRAIN_LIST}' LED_VAL_LIST='${LED_VAL_LIST}' C2_RUN_DIR='${C2_RUN_DIR}' C2_CHECKPOINT='${C2_CHECKPOINT}' D0_SIGN='${D0_SIGN}' EPOCHS='${EPOCHS}' BS='${BS}' ACCUM_STEPS='${ACCUM_STEPS}' RUN_SMOKE='${RUN_SMOKE}' KEEP_SMOKE='${KEEP_SMOKE}' RUN_N5_D1='${RUN_N5_D1}' RUN_N3_RGB='${RUN_N3_RGB}' RUN_N2_X3='${RUN_N2_X3}' RUN_N7_STOPGRAD='${RUN_N7_STOPGRAD}' QUEUE_TIMESTAMP='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing dir: $1" >&2; exit 2; }; }
tag_float() { echo "$1" | sed 's/\./p/g'; }
require_file "${PRETRAINED}"
require_file "${LED_TRAIN_LIST}"
require_file "${LED_VAL_LIST}"
require_file "${C2_CHECKPOINT}"
require_dir "${C2_RUN_DIR}"

if [[ -z "${D0_SIGN}" ]]; then
  D0_SIGN="$("${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import json; print(json.load(open('${C2_RUN_DIR}/config.json'))['d0_sign'])")"
fi
echo "[CONFIG] D0_SIGN=${D0_SIGN}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "[SMOKE] running LED-HB stack smoke before N-series formal"
  KEEP_SMOKE="${KEEP_SMOKE}" GPU="${GPU}" CONDA_BIN="${CONDA_BIN}" CONDA_ENV="${CONDA_ENV}" PRETRAINED="${PRETRAINED}" \
    bash finetune_stf/scripts/smoke/0531_smoke_led_hb_stack.sh
fi

common_args=(
  --dataset-name led_hb
  --illumination HB
  --dataset-geometry-mode resize_fullres_756x1344_then_halfres_378x672
  --rgb-input-space resize_area_756x1344_then_2x2_area
  --depth-target-space resize_nearest_756x1344_then_2x2_valid_mean
  --depth-label distance_to_image_plane
  --depth-unit meter
  --train-split hb_train_all
  --val-split hb_val_stride5_n1000_seed42
  --eval-protocol per_image_affine_disp_depth_anything_v2
  --encoder vits
  --pretrained-from "${PRETRAINED}"
  --c2-checkpoint "${C2_CHECKPOINT}"
  --c2-run-dir "${C2_RUN_DIR}"
  --led-train-list "${LED_TRAIN_LIST}"
  --led-val-list "${LED_VAL_LIST}"
  --input-height 378
  --input-width 672
  --min-depth 1.0
  --max-depth 200.0
  --residual-alpha 0.5
  --d0-sign "${D0_SIGN}"
  --lowpass-kernel 31
  --q-good 0.3
  --lambda-final 1.0
  --lambda-boundary 2.0
  --lambda-grad 0.5
  --lambda-keep-good-d1 0.2
  --lambda-gate-sparse 0.05
  --lambda-invalid-keep 0.1
  --lambda-lowfreq-loss 0.0
  --hflip-prob 0.5
  --epochs "${EPOCHS}"
  --bs "${BS}"
  --accum-steps "${ACCUM_STEPS}"
  --lr 1e-4
  --weight-decay 1e-4
  --num-workers 4
  --log-interval 100
  --save-interval 1
  --eval-interval 1
  --save-best-checkpoint
  --amp
  --amp-dtype bf16
  --seed 42
)
raw_adapter_args=(--unprocessing-method raw_adapter_style --vkitti-unprocessing-preset not_applicable --no-randomize-unprocessing --raw-adapter-backend analytic --raw-adapter-cfa-pattern RGGB --raw-adapter-packed-channel-order R_Gr_Gb_B --raw-adapter-rgb-transfer srgb_piecewise --raw-adapter-inverse-tone global_0p15 --raw-adapter-ccm identity --raw-adapter-red-gain-range 1.9 2.4 --raw-adapter-blue-gain-range 1.5 1.9 --raw-adapter-fixed-red-gain 2.15 --raw-adapter-fixed-blue-gain 1.70 --raw-adapter-fixed-light-scale 1.0 --raw-adapter-dark-light-scale-range 0.05 0.4 --raw-adapter-over-light-scale-range 1.5 2.5 --raw-adapter-shot-noise 0.001 --raw-adapter-read-noise 0.0005 --raw-adapter-noise-mean-mode zero --raw-adapter-black-level 0.0 --raw-adapter-white-level 1.0 --raw-adapter-random-seed-policy dataloader_generator --raw-adapter-variant-policy normal --raw-adapter-variant-weights normal=1.0,dark=0.0,over=0.0)
rgb_na_args=(--unprocessing-method not_applicable --vkitti-unprocessing-preset not_applicable --raw-adapter-backend not_applicable --raw-adapter-cfa-pattern not_applicable --raw-adapter-packed-channel-order not_applicable --raw-adapter-rgb-transfer not_applicable --raw-adapter-inverse-tone not_applicable --raw-adapter-ccm not_applicable --raw-adapter-red-gain-range not_applicable --raw-adapter-blue-gain-range not_applicable --raw-adapter-fixed-red-gain not_applicable --raw-adapter-fixed-blue-gain not_applicable --raw-adapter-fixed-light-scale not_applicable --raw-adapter-dark-light-scale-range not_applicable --raw-adapter-over-light-scale-range not_applicable --raw-adapter-shot-noise not_applicable --raw-adapter-read-noise not_applicable --raw-adapter-noise-mean-mode not_applicable --raw-adapter-black-level not_applicable --raw-adapter-white-level not_applicable --raw-adapter-random-seed-policy not_applicable --raw-adapter-variant-policy not_applicable --raw-adapter-variant-weights not_applicable)

run_method() {
  local method="$1" feature="$2" delta="$3" gate="$4" rft="$5" input_domain="$6" model_input="$7" front_end="$8" raw_storage="$9" lambda_lp="${10}"
  local run_timestamp; run_timestamp="$(date +%m%d_%H%M)"
  local run_name="${run_timestamp}_led_hb_${method,,}_${feature}_lp$(tag_float "${lambda_lp}")_q0p3_lfl0p0_rft${rft}_vits_half378x672_trainall_valstride5n1000_seed42_bs${BS}_acc${ACCUM_STEPS}_e${EPOCHS}"
  local save="${EXP_ROOT}/${run_name}"
  local heavy="${HEAVY_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}.tmux.log"
  [[ ! -e "${save}" && ! -e "${heavy}" ]] || { echo "[ERROR] refusing overwrite ${run_name}" >&2; exit 2; }
  local unproc_args=("${rgb_na_args[@]}")
  [[ "${input_domain}" == "raw4" ]] && unproc_args=("${raw_adapter_args[@]}")
  echo "[FORMAL] ${method} run=${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_led_hb_incremental_residual.py \
    "${common_args[@]}" \
    --method-id "${method}" \
    --input-domain "${input_domain}" \
    --model-input-tensor "${model_input}" \
    --raw-storage-format "${raw_storage}" \
    --front-end "${front_end}" \
    --incremental-feature-source "${feature}" \
    --delta-condition "${delta}" \
    --gate-condition "${gate}" \
    --raw-feature-encoder-trainable "${rft}" \
    --lambda-lp "${lambda_lp}" \
    "${unproc_args[@]}" \
    --save-path "${save}" \
    --heavy-save-path "${heavy}" 2>&1 | tee -a "${log}"
  echo "${method}_RUN_DIR=${save}"
  echo "${method}_CHECKPOINT=${heavy}/best_abs_rel.pth"
}

[[ "${RUN_N5_D1}" == "1" ]] && run_method N5 d1 d1_only d1_only not_applicable rgb image c2_frozen_d1_incremental not_applicable 0.5
[[ "${RUN_N3_RGB}" == "1" ]] && run_method N3 rgb feature_only feature_d1 not_applicable rgb image c2_frozen_rgb_incremental not_applicable 0.5
[[ "${RUN_N2_X3}" == "1" ]] && run_method N2 x3 feature_only feature_d1 true raw4 raw c2_frozen_raw_ram_incremental synthetic_packed_bayer_4ch_halfres 0.8
[[ "${RUN_N7_STOPGRAD}" == "1" ]] && run_method N7 x3 feature_d1_stopgrad feature_d1 true raw4 raw c2_frozen_raw_ram_incremental synthetic_packed_bayer_4ch_halfres 0.5
