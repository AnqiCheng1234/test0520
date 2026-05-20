#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/caq/6666_raw/dav2_raw_0424"
EXP_ROOT="${REPO_ROOT}/finetune_stf/exp"
CONDA_ENV="${CONDA_ENV:-dav3}"
MASTER_PORT="${MASTER_PORT:-29831}"
GPUS="${GPUS:-1}"

PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth}"
LOD_ROOT="${LOD_ROOT:-/mnt/drive/3333_raw/LOD}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest.csv}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${REPO_ROOT}/finetune_stf/dataset/splits/vkitti2/train.txt}"
HYPERSIM_PROCESSED_BASE="${HYPERSIM_PROCESSED_BASE:-/mnt/drive/1111_new_works/hypersim_marigold_processed/hypersim}"
ETH3D_ROOT="${ETH3D_ROOT:-/mnt/drive/3333_raw/eth3d_raw_depth_640960}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100}"
ROBOTCAR_NIGHT_ROOT="${ROBOTCAR_NIGHT_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo}"
ROBOTCAR_NIGHT_MANIFEST_NAME="${ROBOTCAR_NIGHT_MANIFEST_NAME:-robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv}"

CASE_NAME="${1:?Usage: $0 <H0_rawmix_ref|H1_hypersim_only|H2_vkitti_hypersim|H3_vkitti_hypersim_lod> [extra train.py args...]}"
shift

case "${CASE_NAME}" in
  H0_rawmix_ref)
    RUN_NAME="0513_h0_rawmix_ref_lod_dn_vkitti_lora_decoder_644x1008_bs4acc4_e10"
    TRAIN_SOURCES="lod_day,lod_night,vkitti"
    TRAIN_SOURCE_RATIOS="3,3,2"
    BEST_METRIC="avg4"
    ;;
  H1_hypersim_only)
    RUN_NAME="0513_h1_hypersim_only_lora_decoder_644x1008_bs4acc4_e10"
    TRAIN_SOURCES="hypersim"
    TRAIN_SOURCE_RATIOS="1"
    BEST_METRIC="eth3d"
    ;;
  H2_vkitti_hypersim)
    RUN_NAME="0513_h2_vkitti_hypersim_lora_decoder_644x1008_bs4acc4_e10"
    TRAIN_SOURCES="vkitti,hypersim"
    TRAIN_SOURCE_RATIOS="1,1"
    BEST_METRIC="avg4"
    ;;
  H3_vkitti_hypersim_lod)
    RUN_NAME="0513_h3_lod_dn_vkitti_hypersim_lora_decoder_644x1008_bs4acc4_e10"
    TRAIN_SOURCES="lod_day,lod_night,vkitti,hypersim"
    TRAIN_SOURCE_RATIOS="3,3,2,2"
    BEST_METRIC="avg4"
    ;;
  *)
    echo "Unknown CASE_NAME: ${CASE_NAME}" >&2
    exit 2
    ;;
esac

RUN_DIR="${RUN_DIR:-${EXP_ROOT}/${RUN_NAME}}"
mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

echo "[RUN] starting ${CASE_NAME}: ${RUN_NAME}"
conda run -n "${CONDA_ENV}" torchrun --nproc_per_node="${GPUS}" --master_port "${MASTER_PORT}" finetune_stf/train.py \
  --stage raw_mix \
  --train-sources "${TRAIN_SOURCES}" \
  --train-source-ratios "${TRAIN_SOURCE_RATIOS}" \
  --train-steps-per-epoch 6696 \
  --input-type raw_ram_bridge_lora \
  --encoder vitl \
  --dav2-train-mode decoder \
  --rgb-interface-mode sigmoid \
  --lora-rank 8 \
  --lora-alpha 16 \
  --loss-type ssi \
  --loss-mask-downsample strict \
  --loss-target-normalization \
  --loss-norm-min-scale 1e-3 \
  --pretrained-from "${PRETRAINED_FROM}" \
  --lod-root "${LOD_ROOT}" \
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \
  --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
  --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
  --vkitti-unprocessing-preset sensor_linear_dual \
  --vkitti-unprocessing-mix-weights 0.5,0.5 \
  --hypersim-processed-base "${HYPERSIM_PROCESSED_BASE}" \
  --hypersim-min-depth 0.1 \
  --hypersim-max-depth 80.0 \
  --input-height 644 \
  --input-width 1008 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --norm-mode sensor_linear \
  --channel-mode rgb_avg_g \
  --epochs 10 \
  --bs 4 \
  --accum-steps 4 \
  --num-workers 4 \
  --lr 1e-5 \
  --bridge-lr 5e-5 \
  --lora-lr 5e-5 \
  --bridge-feature-keys x_cat ffm_mid x4 \
  --no-eval-stf \
  --eval-eth3d \
  --eth3d-root "${ETH3D_ROOT}" \
  --eth3d-eval-mode fast \
  --eth3d-fast-eval-backend proxy \
  --eval-robotcar \
  --robotcar-root "${ROBOTCAR_ROOT}" \
  --robotcar-eval-mode fast \
  --robotcar-fast-eval-backend sparse \
  --eval-robotcar-night \
  --robotcar-night-root "${ROBOTCAR_NIGHT_ROOT}" \
  --robotcar-night-manifest-name "${ROBOTCAR_NIGHT_MANIFEST_NAME}" \
  --robotcar-night-fast-eval-backend sparse \
  --eval-kitti \
  --kitti-eval-protocol rgb_checkpoint_decoder \
  --best-metric "${BEST_METRIC}" \
  --amp \
  --amp-dtype bf16 \
  --log-interval 250 \
  --save-path "${RUN_DIR}" \
  "$@" \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[RUN] finished ${CASE_NAME}: ${RUN_NAME}"
