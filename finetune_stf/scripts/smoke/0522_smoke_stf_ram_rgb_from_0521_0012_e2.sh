#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-29782}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
DAV2_MANIFEST="${DAV2_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"

RUN_SUFFIX="stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_ram_only_e2_from_0521_0012_setting"

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

is_temp_path() {
  case "$1" in
    *smoke*|*debug*|*tmp*|*codex_smoke*) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup_smoke_artifacts() {
  local path
  for path in "$@"; do
    if [[ -e "${path}" ]]; then
      if is_temp_path "${path}"; then
        echo "[CLEANUP] rm -rf ${path}"
        rm -rf "${path}"
      else
        echo "[CLEANUP][KEEP] ambiguous path: ${path}"
      fi
    fi
  done
}

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}"
require_file "${PRETRAINED}"
require_file "${DAV2_MANIFEST}"
require_dir "${STF_ROOT}"
require_dir "${RAW_NPZ_ROOT}"

ts="$(date +%m%d_%H%M)"
run="codex_smoke_${ts}_${RUN_SUFFIX}"
save="${EXP_ROOT}/${run}"
heavy="${HEAVY_ROOT}/${run}"
log="${LOG_ROOT}/${run}.smoke.log"

if [[ -e "${save}" || -e "${heavy}" || -e "${log}" ]]; then
  echo "[ERROR] Refusing to overwrite existing smoke artifacts for ${run}" >&2
  echo "  save=${save}" >&2
  echo "  heavy=${heavy}" >&2
  echo "  log=${log}" >&2
  exit 2
fi

echo "[SMOKE] run=${run}"
echo "[SMOKE] log=${log}"

set +e
PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
  torchrun --nproc_per_node=1 --master_port="${PORT}" finetune_stf/train.py \
  --encoder vits \
  --stage stf_only \
  --stf-root "${STF_ROOT}" \
  --input-height 512 \
  --input-width 960 \
  --stf-fast-eval-backend sparse \
  --eval-stf \
  --best-metric stf \
  --save-best-checkpoint \
  --bs 2 \
  --accum-steps 1 \
  --lr 1e-5 \
  --raw-front-end-lr 5e-5 \
  --loss-type ssi \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --amp \
  --amp-dtype bf16 \
  --seed 42 \
  --num-workers 0 \
  --log-interval 1 \
  --raw-npz-root "${RAW_NPZ_ROOT}" \
  --input-domain raw4 \
  --front-end raw_to_base_rgb_ram3 \
  --dataset-family stf_raw \
  --dataset-input-mode raw_ram \
  --model-input-tensor raw \
  --raw-storage-format legacy_bggR_decomp16 \
  --bridge none \
  --decoder-feature-adapter none \
  --lora none \
  --norm-mode passthrough \
  --channel-mode rgb_avg_g \
  --raw-ram-rgb-tail identity \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode none \
  --epochs 1 \
  --debug-max-train-steps 2 \
  --debug-max-val-samples 8 \
  --no-enable-fixed-viz-dump \
  --no-enable-train-source-viz-dump \
  --no-train-viz-rgb-baseline \
  --pretrained-from "${PRETRAINED}" \
  --heavy-save-root "${HEAVY_ROOT}" \
  --save-path "${save}" 2>&1 | tee -a "${log}"
status=${PIPESTATUS[0]}
set -e

if [[ "${status}" -eq 0 ]]; then
  grep -q "group=raw_front_end lr=5.00e-05" "${log}"
  grep -q "input_type=raw_ram_rgb" "${log}"
  cleanup_smoke_artifacts "${save}" "${heavy}" "${log}"
  echo "[SMOKE][PASS] ${run}"
else
  echo "[SMOKE][FAIL] ${run} status=${status}"
  echo "[SMOKE][KEEP] ${save}"
  echo "[SMOKE][KEEP] ${heavy}"
  echo "[SMOKE][KEEP] ${log}"
  exit "${status}"
fi
