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

export PYTHONUNBUFFERED=1
cd "${ROOT}"

for run in "${RUNS[@]}"; do
  exp_dir="${ROOT}/finetune_stf/exp/${run}"
  ckpt="${HEAVY_ROOT}/${run}/last_epoch_model.pth"

  echo "===== KITTI full-frame panels: ${run} ====="
  test -f "${exp_dir}/config.json"
  test -f "${ckpt}"

  python anqi_eval/compare_kitti_rgb_zeroshot_vs_ours.py \
    "${exp_dir}" \
    --ours-checkpoint "${ckpt}" \
    --output-dir "${exp_dir}/kitti_rgb_compare_dav2_vs_ours_2x3" \
    --indices 0 162 325 488 651 \
    --min-depth 0.1 \
    --max-depth 80.0

  echo "===== done ${run} ====="
done

echo "All 0430 A/B KITTI full-frame visual analyses finished."
