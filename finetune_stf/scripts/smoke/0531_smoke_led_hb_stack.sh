#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LED_ROOT="${LED_ROOT:-/mnt/drive/3333_raw/led_night}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"

cd "${ROOT}"
SMOKE_TS="$(date +%m%d_%H%M)"
SMOKE_ROOT="${SMOKE_ROOT:-${ROOT}/plans/0531_led_night/codex_smoke_led_hb_stack_${SMOKE_TS}}"
SPLIT_ROOT="${SMOKE_ROOT}/splits"
EXP_ROOT="${SMOKE_ROOT}/exp"
HEAVY_ROOT="${SMOKE_ROOT}/heavy"
DIAG_ROOT="${SMOKE_ROOT}/diagnostics"
mkdir -p "${SMOKE_ROOT}" "${SPLIT_ROOT}" "${EXP_ROOT}" "${HEAVY_ROOT}" "${DIAG_ROOT}"
trap 'echo "[SMOKE_FAILED] root=${SMOKE_ROOT}" >&2' ERR

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }
}
safe_rm_smoke() {
  local path="$1"
  case "${path}" in
    *codex_smoke*|*smoke*|*debug*|*tmp*) rm -rf "${path}" ;;
    *) echo "[ERROR] refusing to delete non-smoke path: ${path}" >&2; exit 2 ;;
  esac
}

require_file "${PRETRAINED}"
require_dir "${LED_ROOT}/extracted/HB"

NEW_PY=(
  foundation/engine/datasets/led_hb.py
  foundation/tools/build_led_hb_filelists.py
  foundation/tools/smoke_led_hb_dataset.py
  foundation/tools/eval_led_hb_d0.py
  foundation/tools/train_led_hb_residual_control.py
  foundation/tools/train_led_hb_incremental_residual.py
  foundation/tools/eval_led_hb_formal.py
  foundation/tools/make_led_hb_residual_panels.py
  foundation/tools/summarize_led_hb_formal.py
)

echo "[SMOKE] py_compile"
"${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python -m py_compile "${NEW_PY[@]}"

echo "[SMOKE] build filelists"
"${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/build_led_hb_filelists.py \
  --led-root "${LED_ROOT}" \
  --illum HB \
  --train-maps china,herrenberg,ottosuhrallee \
  --val-maps hamburg \
  --val-stride 5 \
  --val-n 1000 \
  --seed 42 \
  --out-dir "${SPLIT_ROOT}"
LED_TRAIN_LIST="${SPLIT_ROOT}/hb_train_all.txt"
LED_VAL_LIST="${SPLIT_ROOT}/hb_val_stride5_n1000_seed42.txt"
require_file "${LED_TRAIN_LIST}"
require_file "${LED_VAL_LIST}"

echo "[SMOKE] dataset"
"${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/smoke_led_hb_dataset.py \
  --led-train-list "${LED_TRAIN_LIST}" \
  --led-val-list "${LED_VAL_LIST}" \
  --input-height 378 \
  --input-width 672 \
  --min-depth 1.0 \
  --max-depth 200.0 \
  --led-geometry-mode resize_fullres_756x1344_then_halfres_378x672 \
  --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --output "${SMOKE_ROOT}/dataset_smoke_summary.json"

echo "[SMOKE] D0 max_val_samples=8"
D0_OUT="${DIAG_ROOT}/d0_smoke"
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_led_hb_d0.py \
  --encoder vits \
  --pretrained-from "${PRETRAINED}" \
  --led-val-list "${LED_VAL_LIST}" \
  --input-height 378 \
  --input-width 672 \
  --min-depth 1.0 \
  --max-depth 200.0 \
  --max-val-samples 8 \
  --output-dir "${D0_OUT}" \
  --device cuda
D0_SIGN="$("${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import json; print(json.load(open('${D0_OUT}/d0_summary.json'))['recommended_d0_sign'])")"
echo "[SMOKE] D0_SIGN=${D0_SIGN}"

common_c2=(
  --experiment-id C2
  --dataset-name led_hb
  --illumination HB
  --input-domain rgb
  --model-input-tensor image
  --dataset-geometry-mode resize_fullres_756x1344_then_halfres_378x672
  --raw-storage-format not_applicable
  --rgb-input-space resize_area_756x1344_then_2x2_area
  --depth-target-space resize_nearest_756x1344_then_2x2_valid_mean
  --depth-label distance_to_image_plane
  --depth-unit meter
  --front-end dav2_rgb_frozen
  --encoder vits
  --pretrained-from "${PRETRAINED}"
  --led-train-list "${LED_TRAIN_LIST}"
  --led-val-list "${LED_VAL_LIST}"
  --train-split hb_train_all
  --val-split hb_val_stride5_n1000_seed42
  --input-height 378
  --input-width 672
  --min-depth 1.0
  --max-depth 200.0
  --residual-feature-source d0
  --residual-alpha 0.5
  --d0-sign "${D0_SIGN}"
  --hflip-prob 0.5
  --epochs 1
  --bs 1
  --accum-steps 1
  --lr 1e-4
  --weight-decay 1e-4
  --num-workers 0
  --log-interval 1
  --save-interval 1
  --eval-interval 1
  --save-best-checkpoint
  --max-train-steps 2
  --max-val-samples 4
  --amp
  --amp-dtype bf16
  --seed 42
)

C2_RUN="${EXP_ROOT}/smoke_led_hb_c2"
C2_HEAVY="${HEAVY_ROOT}/smoke_led_hb_c2"
echo "[SMOKE] C2 tiny train"
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/train_led_hb_residual_control.py \
  "${common_c2[@]}" \
  --save-path "${C2_RUN}" \
  --heavy-save-path "${C2_HEAVY}"
C2_CKPT="${C2_HEAVY}/best_abs_rel.pth"
require_file "${C2_RUN}/config.json"
require_file "${C2_RUN}/val_metrics.json"
require_file "${C2_RUN}/run_summary.json"
require_file "${C2_CKPT}"

common_n=(
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
  --c2-checkpoint "${C2_CKPT}"
  --c2-run-dir "${C2_RUN}"
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
  --epochs 1
  --bs 1
  --accum-steps 1
  --lr 1e-4
  --weight-decay 1e-4
  --num-workers 0
  --log-interval 1
  --save-interval 1
  --eval-interval 1
  --save-best-checkpoint
  --max-train-steps 2
  --max-val-samples 4
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

run_n() {
  local method="$1"; shift
  local feature="$1"; shift
  local delta="$1"; shift
  local gate="$1"; shift
  local rft="$1"; shift
  local input_domain="$1"; shift
  local model_input="$1"; shift
  local front_end="$1"; shift
  local raw_storage="$1"; shift
  local lambda_lp="$1"; shift
  local save="${EXP_ROOT}/smoke_led_hb_${method,,}"
  local heavy="${HEAVY_ROOT}/smoke_led_hb_${method,,}"
  local unproc_args=("${rgb_na_args[@]}")
  if [[ "${input_domain}" == "raw4" ]]; then
    unproc_args=("${raw_adapter_args[@]}")
  fi
  echo "[SMOKE] ${method} tiny train"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/train_led_hb_incremental_residual.py \
    "${common_n[@]}" \
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
    --heavy-save-path "${heavy}"
  require_file "${save}/config.json"
  require_file "${save}/val_metrics.json"
  require_file "${save}/run_summary.json"
  require_file "${heavy}/best_abs_rel.pth"
}

run_n N5 d1 d1_only d1_only not_applicable rgb image c2_frozen_d1_incremental not_applicable 0.5
run_n N3 rgb feature_only feature_d1 not_applicable rgb image c2_frozen_rgb_incremental not_applicable 0.5
run_n N2 x3 feature_only feature_d1 true raw4 raw c2_frozen_raw_ram_incremental synthetic_packed_bayer_4ch_halfres 0.8
run_n N7 x3 feature_d1_stopgrad feature_d1 true raw4 raw c2_frozen_raw_ram_incremental synthetic_packed_bayer_4ch_halfres 0.5

N7_RUN="${EXP_ROOT}/smoke_led_hb_n7"
N7_CKPT="${HEAVY_ROOT}/smoke_led_hb_n7/best_abs_rel.pth"
for mode in true zero mean shuffle; do
  echo "[SMOKE] N7 feature ablation ${mode}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_led_hb_formal.py \
    --run-kind nseries \
    --run-dir "${N7_RUN}" \
    --checkpoint "${N7_CKPT}" \
    --max-depth 200.0 \
    --feature-ablation-mode "${mode}" \
    --feature-ablation-scope both \
    --feature-ablation-key x3 \
    --feature-ablation-donor-offset 1 \
    --max-val-samples 4 \
    --output-dir "${DIAG_ROOT}/n7_x3_${mode}_max200" \
    --device cuda
  require_file "${DIAG_ROOT}/n7_x3_${mode}_max200/metrics.json"
done

echo "[SMOKE] N7 tiny panels"
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/make_led_hb_residual_panels.py \
  --run-kind nseries \
  --run-dir "${N7_RUN}" \
  --checkpoint "${N7_CKPT}" \
  --output-dir "${DIAG_ROOT}/n7_panels" \
  --num-panels 1 \
  --selection-mode fixed \
  --max-depth 200.0 \
  --device cuda
require_file "${DIAG_ROOT}/n7_panels/panel_manifest.json"
if ! find "${DIAG_ROOT}/n7_panels" -name '*.png' -print -quit | grep -q .; then
  echo "[ERROR] no panel PNG generated" >&2
  exit 2
fi

echo "[SMOKE] passed root=${SMOKE_ROOT}"
if [[ "${KEEP_SMOKE}" != "1" ]]; then
  safe_rm_smoke "${SMOKE_ROOT}"
  echo "[SMOKE] removed ${SMOKE_ROOT}"
else
  echo "[SMOKE] kept ${SMOKE_ROOT}"
fi
