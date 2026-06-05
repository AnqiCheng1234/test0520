#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/caq/6666_raw/dav2_raw_0520}"
EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/finetune_stf/exp}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPUS="${GPUS:-1}"

ENCODER="${ENCODER:-vits}"
PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LOD_ROOT="${LOD_ROOT:-/mnt/drive/3333_raw/LOD}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest_subset50_split_seed42.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest_subset50_split_seed42.csv}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${REPO_ROOT}/finetune_stf/dataset/splits/vkitti2/train.txt}"
VKITTI_UNPROCESSING_PRESET="${VKITTI_UNPROCESSING_PRESET:-sensor_linear_dual}"
VKITTI_UNPROCESSING_MIX_WEIGHTS="${VKITTI_UNPROCESSING_MIX_WEIGHTS:-0.5,0.5}"
ETH3D_ROOT="${ETH3D_ROOT:-/mnt/drive/3333_raw/eth3d_raw_depth_640960}"
ETH3D_MAX_SAMPLES="${ETH3D_MAX_SAMPLES:-150}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100}"
ROBOTCAR_NIGHT_ROOT="${ROBOTCAR_NIGHT_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo}"
ROBOTCAR_NIGHT_MANIFEST_NAME="${ROBOTCAR_NIGHT_MANIFEST_NAME:-robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv}"
ROBOTCAR_NIGHT_MAX_SAMPLES="${ROBOTCAR_NIGHT_MAX_SAMPLES:-}"
KITTI_EVAL_PROTOCOL="${KITTI_EVAL_PROTOCOL:-rgb_checkpoint_decoder}"
NYU_DIR="${NYU_DIR:-/mnt/drive/nyu/nyu_test}"
NYU_MAX_SAMPLES="${NYU_MAX_SAMPLES:-}"

EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-5}"
BRIDGE_LR="${BRIDGE_LR:-5e-5}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-ram_core}"
LOD_PER_VKITTI="${LOD_PER_VKITTI:-1}"
INPUT_HEIGHT="${INPUT_HEIGHT:-518}"
INPUT_WIDTH="${INPUT_WIDTH:-812}"
LOD_CROP_MODE="${LOD_CROP_MODE:-random}"

TARGET="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_TRAIN_ARGS=("$@")

cd "${REPO_ROOT}"

run_rgb_bridge_1317_e10() {
  # Timestamp each sub-run when it actually starts. In "all" mode, the second
  # experiment may begin hours later and should not inherit the launcher's TS.
  local ts="${RGB_TS:-$(date +%m%d_%H%M)}"
  local master_port="${RGB_MASTER_PORT:-29831}"
  local run_name="${RGB_RUN_NAME:-${ts}_vits_vkitti_lod_dn_rgb_bridge_decoder_518x812_bs8acc1_lod50_lod1vk1_randomcrop_ssi_eth3d150_e10}"
  local run_dir="${RGB_RUN_DIR:-${EXP_ROOT}/${run_name}}"
  local bridge_feature_keys=(${RGB_BRIDGE_FEATURE_KEYS:-x_cat ffm_mid x3})
  local extra_args=()

  if [[ -n "${ROBOTCAR_NIGHT_MAX_SAMPLES}" ]]; then
    extra_args+=(--robotcar-night-max-samples "${ROBOTCAR_NIGHT_MAX_SAMPLES}")
  fi
  if [[ -n "${NYU_MAX_SAMPLES}" ]]; then
    extra_args+=(--nyu-max-samples "${NYU_MAX_SAMPLES}")
  fi

  mkdir -p "${run_dir}"
  echo "[RUN][rgb_bridge_1317_e10] starting: ${run_name}"
  echo "[RUN][rgb_bridge_1317_e10] best_metric=eth3d save_best_checkpoint=on"
  echo "[RUN][rgb_bridge_1317_e10] bs=${BS} accum_steps=${ACCUM_STEPS} effective_bs=$((BS * ACCUM_STEPS)) epochs=${EPOCHS}"
  echo "[RUN][rgb_bridge_1317_e10] input_hw=${INPUT_HEIGHT}x${INPUT_WIDTH} lod_crop_mode=${LOD_CROP_MODE} lod_per_vkitti=${LOD_PER_VKITTI}"

  conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${GPUS}" --master_port "${master_port}" finetune_stf/train.py \
    --stage vkitti_lod \
    --input-type raw_ram_rgb_bridge \
    --encoder "${ENCODER}" \
    --dav2-train-mode decoder \
    --loss-type ssi \
    --loss-mask-downsample strict \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --pretrained-from "${PRETRAINED_FROM}" \
    --lod-root "${LOD_ROOT}" \
    --lod-day-manifest "${LOD_DAY_MANIFEST}" \
    --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
    --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
    --vkitti-unprocessing-preset "${VKITTI_UNPROCESSING_PRESET}" \
    --vkitti-unprocessing-mix-weights "${VKITTI_UNPROCESSING_MIX_WEIGHTS}" \
    --lod-per-vkitti "${LOD_PER_VKITTI}" \
    --input-height "${INPUT_HEIGHT}" \
    --input-width "${INPUT_WIDTH}" \
    --lod-crop-mode "${LOD_CROP_MODE}" \
    --min-depth 1.0 \
    --max-depth 80.0 \
    --norm-mode sensor_linear \
    --channel-mode rgb_avg_g \
    --epochs "${EPOCHS}" \
    --bs "${BS}" \
    --accum-steps "${ACCUM_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --bridge-lr "${BRIDGE_LR}" \
    --bridge-source "${BRIDGE_SOURCE}" \
    --bridge-feature-keys "${bridge_feature_keys[@]}" \
    --rgb-interface-mode residual_tanh \
    --no-eval-stf \
    --eval-eth3d \
    --eth3d-root "${ETH3D_ROOT}" \
    --eth3d-eval-mode fast \
    --eth3d-fast-eval-backend proxy \
    --eth3d-max-samples "${ETH3D_MAX_SAMPLES}" \
    --eval-robotcar \
    --robotcar-root "${ROBOTCAR_ROOT}" \
    --robotcar-eval-mode fast \
    --robotcar-fast-eval-backend sparse \
    --eval-robotcar-night \
    --robotcar-night-root "${ROBOTCAR_NIGHT_ROOT}" \
    --robotcar-night-manifest-name "${ROBOTCAR_NIGHT_MANIFEST_NAME}" \
    --robotcar-night-fast-eval-backend sparse \
    --eval-kitti \
    --kitti-eval-protocol "${KITTI_EVAL_PROTOCOL}" \
    --eval-nyu \
    --nyu-dir "${NYU_DIR}" \
    --best-metric eth3d \
    --save-best-checkpoint \
    --amp \
    --amp-dtype bf16 \
    --log-interval 250 \
    --save-path "${run_dir}" \
    "${extra_args[@]}" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    2>&1 | tee "${run_dir}/train.log"

  echo "[RUN][rgb_bridge_1317_e10] finished: ${run_name}"
}

run_4ch_bridge_sigmoid_2108_e10() {
  # Timestamp each sub-run when it actually starts. In "all" mode, the second
  # experiment may begin hours later and should not inherit the launcher's TS.
  local ts="${RAW4_TS:-$(date +%m%d_%H%M)}"
  local master_port="${RAW4_MASTER_PORT:-29832}"
  local run_name="${RAW4_RUN_NAME:-${ts}_vits_vkitti_lod_dn_4ch_bridge_sigmoid_decoder_518x812_bs8acc1_lod50_lod1vk1_randomcrop_ssi_eth3d150_e10}"
  local run_dir="${RAW4_RUN_DIR:-${EXP_ROOT}/${run_name}}"
  local bridge_feature_keys=(${RAW4_BRIDGE_FEATURE_KEYS:-x_cat ffm_mid x4})
  local extra_args=()

  if [[ -n "${ROBOTCAR_NIGHT_MAX_SAMPLES}" ]]; then
    extra_args+=(--robotcar-night-max-samples "${ROBOTCAR_NIGHT_MAX_SAMPLES}")
  fi
  if [[ -n "${NYU_MAX_SAMPLES}" ]]; then
    extra_args+=(--nyu-max-samples "${NYU_MAX_SAMPLES}")
  fi

  mkdir -p "${run_dir}"
  echo "[RUN][4ch_bridge_sigmoid_2108_e10] starting: ${run_name}"
  echo "[RUN][4ch_bridge_sigmoid_2108_e10] best_metric=eth3d save_best_checkpoint=on"
  echo "[RUN][4ch_bridge_sigmoid_2108_e10] bs=${BS} accum_steps=${ACCUM_STEPS} effective_bs=$((BS * ACCUM_STEPS)) epochs=${EPOCHS}"
  echo "[RUN][4ch_bridge_sigmoid_2108_e10] input_hw=${INPUT_HEIGHT}x${INPUT_WIDTH} lod_crop_mode=${LOD_CROP_MODE} lod_per_vkitti=${LOD_PER_VKITTI}"

  conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${GPUS}" --master_port "${master_port}" finetune_stf/train.py \
    --stage vkitti_lod \
    --input-type raw_ram_bridge \
    --encoder "${ENCODER}" \
    --dav2-train-mode decoder \
    --loss-type ssi \
    --loss-mask-downsample strict \
    --loss-target-normalization \
    --loss-norm-min-scale 1e-3 \
    --pretrained-from "${PRETRAINED_FROM}" \
    --lod-root "${LOD_ROOT}" \
    --lod-day-manifest "${LOD_DAY_MANIFEST}" \
    --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
    --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
    --vkitti-unprocessing-preset "${VKITTI_UNPROCESSING_PRESET}" \
    --vkitti-unprocessing-mix-weights "${VKITTI_UNPROCESSING_MIX_WEIGHTS}" \
    --lod-per-vkitti "${LOD_PER_VKITTI}" \
    --input-height "${INPUT_HEIGHT}" \
    --input-width "${INPUT_WIDTH}" \
    --lod-crop-mode "${LOD_CROP_MODE}" \
    --min-depth 1.0 \
    --max-depth 80.0 \
    --norm-mode sensor_linear \
    --channel-mode rgb_avg_g \
    --epochs "${EPOCHS}" \
    --bs "${BS}" \
    --accum-steps "${ACCUM_STEPS}" \
    --num-workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --bridge-lr "${BRIDGE_LR}" \
    --bridge-source "${BRIDGE_SOURCE}" \
    --bridge-feature-keys "${bridge_feature_keys[@]}" \
    --rgb-interface-mode sigmoid \
    --no-eval-stf \
    --eval-eth3d \
    --eth3d-root "${ETH3D_ROOT}" \
    --eth3d-eval-mode fast \
    --eth3d-fast-eval-backend proxy \
    --eth3d-max-samples "${ETH3D_MAX_SAMPLES}" \
    --eval-robotcar \
    --robotcar-root "${ROBOTCAR_ROOT}" \
    --robotcar-eval-mode fast \
    --robotcar-fast-eval-backend sparse \
    --eval-robotcar-night \
    --robotcar-night-root "${ROBOTCAR_NIGHT_ROOT}" \
    --robotcar-night-manifest-name "${ROBOTCAR_NIGHT_MANIFEST_NAME}" \
    --robotcar-night-fast-eval-backend sparse \
    --eval-kitti \
    --kitti-eval-protocol "${KITTI_EVAL_PROTOCOL}" \
    --eval-nyu \
    --nyu-dir "${NYU_DIR}" \
    --best-metric eth3d \
    --save-best-checkpoint \
    --amp \
    --amp-dtype bf16 \
    --log-interval 250 \
    --save-path "${run_dir}" \
    "${extra_args[@]}" \
    "${EXTRA_TRAIN_ARGS[@]}" \
    2>&1 | tee "${run_dir}/train.log"

  echo "[RUN][4ch_bridge_sigmoid_2108_e10] finished: ${run_name}"
}

case "${TARGET}" in
  rgb|1317|rgb_bridge)
    run_rgb_bridge_1317_e10
    ;;
  raw4|4ch|2108|sigmoid)
    run_4ch_bridge_sigmoid_2108_e10
    ;;
  all|both)
    run_rgb_bridge_1317_e10
    run_4ch_bridge_sigmoid_2108_e10
    ;;
  *)
    echo "Usage: $0 [rgb|4ch|all] [extra train.py args...]" >&2
    exit 2
    ;;
esac
