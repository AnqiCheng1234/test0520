#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
timestamp="$(date +%m%d_%H%M)"
SMOKE_ROOT="${SMOKE_ROOT:-${ROOT}/plans/0527/codex_smoke_nseries_${timestamp}}"
LOG_PATH="${SMOKE_ROOT}/smoke.log"
KEEP_SMOKE="${KEEP_SMOKE:-0}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
C2_CHECKPOINT="${C2_CHECKPOINT:-/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_11.pth}"
C2_RUN_DIR="${C2_RUN_DIR:-${ROOT}/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt}"
VKITTI_VAL_LIST="${VKITTI_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt}"
KITTI_BASE="${KITTI_BASE:-/mnt/drive/kitti}"
KITTI_VAL_SPLIT="${KITTI_VAL_SPLIT:-/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt}"
M1_RUN_DIR="${M1_RUN_DIR:-${ROOT}/finetune_stf/exp/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
M1_CHECKPOINT="${M1_CHECKPOINT:-/mnt/drive/3333_raw/0000_exp_ckpt/0526_0040_vkitti_m1_ra0_x3_d0concat_rawadapter_analytic_identity_normal_vits_halfraw187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/epoch_14.pth}"

mkdir -p "${SMOKE_ROOT}"
cd "${ROOT}"

fail() {
  local status="$1"
  local cmd="$2"
  echo "[SMOKE][FAIL] status=${status}" | tee -a "${LOG_PATH}" >&2
  echo "[SMOKE][FAIL] root=${SMOKE_ROOT}" | tee -a "${LOG_PATH}" >&2
  echo "[SMOKE][FAIL] log=${LOG_PATH}" | tee -a "${LOG_PATH}" >&2
  echo "[SMOKE][FAIL] command=${cmd}" | tee -a "${LOG_PATH}" >&2
  exit "${status}"
}

run_step() {
  local label="$1"
  shift
  echo "[SMOKE][STEP] ${label}" | tee -a "${LOG_PATH}"
  echo "[CMD] $*" | tee -a "${LOG_PATH}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "$@" 2>&1 | tee -a "${LOG_PATH}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    fail "${status}" "$*"
  fi
}

cleanup_success() {
  if [[ "${KEEP_SMOKE}" == "1" ]]; then
    echo "[SMOKE] success; keeping ${SMOKE_ROOT}" | tee -a "${LOG_PATH}"
    return
  fi
  case "${SMOKE_ROOT}" in
    *codex_smoke*|*smoke*|*debug*|*tmp*) ;;
    *) echo "[ERROR] Refusing to delete non-smoke path: ${SMOKE_ROOT}" >&2; exit 2 ;;
  esac
  [[ "${SMOKE_ROOT}" == "${ROOT}/plans/0527/"* ]] || {
    echo "[ERROR] Refusing to delete outside plans/0527: ${SMOKE_ROOT}" >&2
    exit 2
  }
  rm -rf "${SMOKE_ROOT}"
}

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

common_train_args=(
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
  --lambda-lp 0.5
  --lowpass-kernel 31
  --q-good 0.5
  --lambda-final 1.0
  --lambda-boundary 2.0
  --lambda-grad 0.5
  --lambda-keep-good-d1 0.2
  --lambda-gate-sparse 0.05
  --lambda-lowfreq-loss 0.0
  --lambda-invalid-keep 0.1
  --eval-protocol per_image_affine_disp_depth_anything_v2
  --eval-kitti
  --kitti-base "${KITTI_BASE}"
  --kitti-val-split "${KITTI_VAL_SPLIT}"
  --kitti-expected-val-samples 652
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
  --max-kitti-val-samples 4
  --amp
  --amp-dtype bf16
  --seed 42
)

run_step "py_compile" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python -m py_compile \
  foundation/engine/models/dav2_incremental_residual.py \
  foundation/tools/train_vkitti2_incremental_residual.py \
  foundation/tools/eval_raw_residual_feature_ablation.py \
  foundation/tools/analyze_residual_energy_frequency.py \
  foundation/tools/make_residual_vs_c2_panels.py \
  foundation/tools/train_vkitti2_raw_residual.py \
  foundation/tools/train_vkitti2_residual_control.py

run_step "N2 tiny train" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/train_vkitti2_incremental_residual.py \
  "${common_train_args[@]}" \
  --method-id N2 --input-domain raw4 --model-input-tensor raw --raw-storage-format synthetic_packed_bayer_4ch_halfres \
  --front-end c2_frozen_raw_ram_incremental --incremental-feature-source x3 --delta-condition feature_only \
  --gate-condition feature_d1 --raw-feature-encoder-trainable true --hflip-prob 0.5 \
  --kitti-eval-protocol halfres_raw_canonical_even_pad_crop_affine_disp \
  "${raw_adapter_args[@]}" \
  --save-path "${SMOKE_ROOT}/n2_save" --heavy-save-path "${SMOKE_ROOT}/n2_heavy"

run_step "N3 tiny train" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/train_vkitti2_incremental_residual.py \
  "${common_train_args[@]}" \
  --method-id N3 --input-domain rgb --model-input-tensor image --raw-storage-format not_applicable \
  --front-end c2_frozen_rgb_incremental --incremental-feature-source rgb --delta-condition feature_only \
  --gate-condition feature_d1 --raw-feature-encoder-trainable not_applicable --hflip-prob 0.5 \
  --kitti-eval-protocol halfres_rgb_canonical_even_pad_crop_affine_disp \
  "${rgb_na_args[@]}" \
  --save-path "${SMOKE_ROOT}/n3_save" --heavy-save-path "${SMOKE_ROOT}/n3_heavy"

run_step "N5 tiny train" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/train_vkitti2_incremental_residual.py \
  "${common_train_args[@]}" \
  --method-id N5 --input-domain rgb --model-input-tensor image --raw-storage-format not_applicable \
  --front-end c2_frozen_d1_incremental --incremental-feature-source d1 --delta-condition d1_only \
  --gate-condition d1_only --raw-feature-encoder-trainable not_applicable --hflip-prob 0.5 \
  --kitti-eval-protocol halfres_rgb_canonical_even_pad_crop_affine_disp \
  "${rgb_na_args[@]}" \
  --save-path "${SMOKE_ROOT}/n5_save" --heavy-save-path "${SMOKE_ROOT}/n5_heavy"

run_step "feature ablation tiny" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_raw_residual_feature_ablation.py \
  --run-dir "${M1_RUN_DIR}" --checkpoint "${M1_CHECKPOINT}" --feature-source x3 \
  --feature-ablation-modes true,zero,shuffle --shuffle-policy stable_hash_far --shuffle-seed 42 \
  --max-val-samples 4 --output-dir "${SMOKE_ROOT}/feature_ablation_m1"

run_step "energy C2 tiny" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/analyze_residual_energy_frequency.py \
  --run-kind control --run-dir "${C2_RUN_DIR}" --checkpoint "${C2_CHECKPOINT}" \
  --max-val-samples 4 --output-dir "${SMOKE_ROOT}/energy_c2"

run_step "energy M1 tiny" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/analyze_residual_energy_frequency.py \
  --run-kind raw --run-dir "${M1_RUN_DIR}" --checkpoint "${M1_CHECKPOINT}" \
  --max-val-samples 4 --output-dir "${SMOKE_ROOT}/energy_m1"

run_step "vs-C2 panels tiny" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/make_residual_vs_c2_panels.py \
  --c2-run-dir "${C2_RUN_DIR}" --c2-checkpoint "${C2_CHECKPOINT}" \
  --method-run-dir "${M1_RUN_DIR}" --method-checkpoint "${M1_CHECKPOINT}" --method-kind raw \
  --sample-indices 0,72 --max-panels 2 --output-dir "${SMOKE_ROOT}/panels_m1_vs_c2"

for required in \
  "${SMOKE_ROOT}/n2_save/config.json" "${SMOKE_ROOT}/n2_save/val_metrics.json" "${SMOKE_ROOT}/n2_save/run_summary.json" "${SMOKE_ROOT}/n2_save/kitti_val_metrics.json" \
  "${SMOKE_ROOT}/n3_save/config.json" "${SMOKE_ROOT}/n3_save/val_metrics.json" "${SMOKE_ROOT}/n3_save/run_summary.json" "${SMOKE_ROOT}/n3_save/kitti_val_metrics.json" \
  "${SMOKE_ROOT}/n5_save/config.json" "${SMOKE_ROOT}/n5_save/val_metrics.json" "${SMOKE_ROOT}/n5_save/run_summary.json" "${SMOKE_ROOT}/n5_save/kitti_val_metrics.json"; do
  [[ -f "${required}" ]] || fail 3 "missing required smoke artifact ${required}"
done

run_step "schema check" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - <<PY
import json
from pathlib import Path
root = Path("${SMOKE_ROOT}")
for name in ("n2_save", "n3_save", "n5_save"):
    data = json.loads((root / name / "val_metrics.json").read_text())
    latest = data["latest"]
    assert "final" in latest["overall"], name
    assert "D1" in latest["overall"], name
    assert "D0" in latest["overall"], name
    kitti = json.loads((root / name / "kitti_val_metrics.json").read_text())["latest"]
    assert "delta_final_minus_D1" in kitti["overall"], name
print("schema ok")
PY

echo "[SMOKE] success root=${SMOKE_ROOT}" | tee -a "${LOG_PATH}"
cleanup_success
