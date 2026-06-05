#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/caq/6666_raw/dav2_raw_0424"
HEAVY_ROOT="/mnt/drive/3333_raw/0000_exp_ckpt"

RUNS=(
  "0430_a1_vkitti_lod_dn_raw4_tanh01_bridge_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_e10"
  "0430_a2_vkitti_lod_dn_raw4_tanh01_bridge_lora_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_e10"
  "0430_a3_vkitti_lod_dn_raw4_residual_tanh_s1_bridge_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_e10"
  "0430_b1_vkitti_lod_dn_bridge_decoder_sigmoid_no_lora_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_e10"
  "0430_b2_vkitti_lod_dn_bridge_feature_adapter_lora_decoder_sigmoid_644x1008_bs2acc8_ssi_lod3_eth3d_kitti_rnight_e10"
)

DAY_SUBSET="${ROOT}/finetune_stf/exp/0429_0021_vkitti_lod_dn_raw4_residual_tanh_bridge_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_best_e10/robotcar_fast_rgb_raw_compare_last_5panels_only/subset_indices.json"
NIGHT_SUBSET="${ROOT}/finetune_stf/exp/0429_0021_vkitti_lod_dn_raw4_residual_tanh_bridge_decoder_644x1008_bs4acc4_ssi_lod3_eth3d_kitti_rnight_best_e10/robotcar_night_fast_rgb_raw_compare_last_5panels_only/subset_indices.json"

export PYTHONUNBUFFERED=1
cd "${ROOT}"

for run in "${RUNS[@]}"; do
  exp_dir="${ROOT}/finetune_stf/exp/${run}"
  ckpt="${HEAVY_ROOT}/${run}/last_epoch_model.pth"

  echo "===== ${run} ====="
  test -f "${exp_dir}/config.json"
  test -f "${ckpt}"

  echo "[1/4] ETH3D panels"
  python anqi_eval/eval_eth3d_rgb_raw_compare.py \
    "${exp_dir}" \
    --checkpoint "${ckpt}" \
    --eth3d-root /mnt/drive/3333_raw/eth3d_raw_depth_640960 \
    --output-dir "${exp_dir}/eth3d_fast_rgb_raw_compare_last_5panels_only" \
    --fast-eval-backend proxy \
    --max-samples 5 \
    --panels-only

  echo "[2/4] RobotCar day panels"
  python anqi_eval/eval_robotcar_rgb_raw_compare.py \
    "${exp_dir}" \
    --checkpoint "${ckpt}" \
    --robotcar-root /mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100 \
    --output-dir "${exp_dir}/robotcar_fast_rgb_raw_compare_last_5panels_only" \
    --fast-eval-backend sparse \
    --sample-count 5 \
    --subset-indices "${DAY_SUBSET}" \
    --raw-source native \
    --min-depth 0.1 \
    --max-depth 50.0 \
    --panels-only

  echo "[3/4] RobotCar night panels"
  python anqi_eval/eval_robotcar_rgb_raw_compare.py \
    "${exp_dir}" \
    --checkpoint "${ckpt}" \
    --robotcar-root /mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo \
    --manifest-name robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv \
    --output-dir "${exp_dir}/robotcar_night_fast_rgb_raw_compare_last_5panels_only" \
    --fast-eval-backend sparse \
    --sample-count 5 \
    --subset-indices "${NIGHT_SUBSET}" \
    --raw-source native \
    --min-depth 0.1 \
    --max-depth 50.0 \
    --panels-only

  echo "[4/4] KITTI 2x3 panels"
  python anqi_eval/compare_kitti_rgb_zeroshot_vs_ours.py \
    "${exp_dir}" \
    --ours-checkpoint "${ckpt}" \
    --output-dir "${exp_dir}/kitti_rgb_compare_dav2_vs_ours_2x3" \
    --indices 0 162 325 488 651 \
    --min-depth 0.1 \
    --max-depth 80.0

  echo "===== done ${run} ====="
done

echo "All 0430 A/B visual analyses finished."
