#!/usr/bin/env bash
set -euo pipefail

cd /home/caq/6666_raw/dav2_raw_0520

CONDA_BIN="${CONDA_BIN:-conda}"
GPU="${GPU:-0}"
MASTER_PORT="${MASTER_PORT:-29670}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
EXP_ROOT="${EXP_ROOT:-/home/caq/6666_raw/dav2_raw_0520/finetune_stf/exp}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest_subset50_split_seed42.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest_subset50_split_seed42.csv}"

run_one() {
  local tag="$1"
  local preset_args="$2"
  local suffix="$3"
  local ts run save

  ts="$(date +%m%d_%H%M)"
  run="${ts}_${suffix}"
  save="${EXP_ROOT}/${run}"

  echo "[RUN] ${run} tag=${tag} start=$(date -Is)"
  echo "[GPU] CUDA_VISIBLE_DEVICES=${GPU}"
  echo "[PORT] ${MASTER_PORT}"
  echo "[PRESET_ARGS] ${preset_args}"

  PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    "${CONDA_BIN}" run --live-stream -n dav3 torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" finetune_stf/train.py \
      --encoder vits \
      --stage raw_mix \
      --train-sources vkitti \
      --train-source-ratios 1.0 \
      --train-steps-per-epoch 877 \
      --input-type raw_ram_rgb_bridge_lora \
      --norm-mode sensor_linear \
      --channel-mode rgb_avg_g \
      --dav2-train-mode decoder \
      --bridge-feature-keys x_cat ffm_mid x3 \
      --bridge-layers 2 5 8 11 \
      --bridge-source ram_core \
      --lora-block-mode tap \
      --lora-rank 8 \
      --lora-alpha 16 \
      --lod-day-manifest "${LOD_DAY_MANIFEST}" \
      --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
      --lod-crop-mode random \
      --input-height 518 \
      --input-width 812 \
      --bs 8 \
      --accum-steps 1 \
      --epochs 10 \
      --lr 1e-5 \
      --bridge-lr 5e-5 \
      --lora-lr 5e-5 \
      --loss-type ssi \
      --loss-target-normalization \
      --loss-lambda-grad 2.0 \
      --amp \
      --amp-dtype bf16 \
      --seed 42 \
      --num-workers 8 \
      --log-interval 250 \
      --no-eval-stf \
      --eval-robotcar-night \
      --robotcar-night-fast-eval-backend sparse \
      --best-metric robotcar_night \
      --save-best-checkpoint \
      --pretrained-from "${PRETRAINED}" \
      --heavy-save-root "${HEAVY_ROOT}" \
      --save-path "${save}" \
      ${preset_args}

  echo "[DONE] ${run} tag=${tag} end=$(date -Is)"
}

run_one robotcarday "--vkitti-unprocessing-preset robotcar_subset100_sensor_linear" \
  "vits_vkitti_robotcarday_raw_ram_rgb_bridge_lora_tap_r8a16_bnclean_518x812_bs8acc1_ssi_rcnightbest_e10"

run_one robotcarnight "--vkitti-unprocessing-preset robotcar_night_sensor_linear" \
  "vits_vkitti_robotcarnight_raw_ram_rgb_bridge_lora_tap_r8a16_bnclean_518x812_bs8acc1_ssi_rcnightbest_e10"

run_one robotcardaynight50 "--vkitti-unprocessing-preset robotcar_day_night_sensor_linear_dual --vkitti-unprocessing-mix-weights robotcar_subset100_sensor_linear=0.5,robotcar_night_sensor_linear=0.5" \
  "vits_vkitti_robotcardaynight50_raw_ram_rgb_bridge_lora_tap_r8a16_bnclean_518x812_bs8acc1_ssi_rcnightbest_e10"
