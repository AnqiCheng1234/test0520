#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/a5000/6666_raw/dav2_raw_0515_vits}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp_186}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt_186}"
CONDA_BIN="${CONDA_BIN:-/home/a5000/anaconda3/bin/conda}"
PRETRAINED="${PRETRAINED:-/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest_subset50_split_seed42.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest_subset50_split_seed42.csv}"

make_entry() {
  local run="$1"
  local session="$2"
  local gpu="$3"
  local port="$4"
  local lod_fraction_arg="$5"
  local save="${EXP_ROOT}/${run}"
  local log="${save}/tmux_launch.log"
  local entry="${save}/tmux_entry.sh"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    return 2
  fi
  if [[ -e "${save}" ]]; then
    echo "[ERROR] Refusing to overwrite existing run directory: ${save}" >&2
    return 2
  fi

  mkdir -p "${save}"
  cat > "${entry}" <<EOF
#!/usr/bin/env bash
set -u
cd "${ROOT}"
{
  echo "[RUN] ${run}"
  echo "[TMUX] ${session}"
  echo "[HOST] \$(hostname) [USER] \$(whoami) [PWD] \$(pwd)"
  echo "[START] \$(date -Iseconds)"
  echo "[GPU] CUDA_VISIBLE_DEVICES=${gpu}"
  echo "[CMD] CUDA_VISIBLE_DEVICES=${gpu} ${CONDA_BIN} run -n dav3 torchrun --nproc_per_node=1 --master_port=${port} finetune_stf/train.py --encoder vits --stage vkitti_lod --lod-per-vkitti 1 ${lod_fraction_arg} --lod-day-manifest ${LOD_DAY_MANIFEST} --lod-night-manifest ${LOD_NIGHT_MANIFEST} --lod-crop-mode random --input-type rgb --input-height 518 --input-width 812 --bs 8 --accum-steps 1 --epochs 10 --lr 1e-5 --loss-type ssi --loss-target-normalization --amp --amp-dtype bf16 --seed 42 --num-workers 8 --log-interval 250 --no-eval-stf --eval-kitti --kitti-eval-protocol rgb_checkpoint_decoder --eval-nyu --eval-eth3d --eth3d-eval-mode fast --eth3d-fast-eval-backend proxy --eth3d-max-samples 150 --eval-robotcar --robotcar-eval-mode fast --robotcar-fast-eval-backend sparse --eval-robotcar-night --robotcar-night-fast-eval-backend sparse --best-metric robotcar_night --save-best-checkpoint --pretrained-from ${PRETRAINED} --heavy-save-root ${HEAVY_ROOT} --save-path ${save}"
} 2>&1 | tee -a "${log}"

set +e
CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run -n dav3 torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \\
  --encoder vits \\
  --stage vkitti_lod \\
  --lod-per-vkitti 1 \\
  ${lod_fraction_arg} \\
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \\
  --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \\
  --lod-crop-mode random \\
  --input-type rgb \\
  --input-height 518 \\
  --input-width 812 \\
  --bs 8 \\
  --accum-steps 1 \\
  --epochs 10 \\
  --lr 1e-5 \\
  --loss-type ssi \\
  --loss-target-normalization \\
  --amp \\
  --amp-dtype bf16 \\
  --seed 42 \\
  --num-workers 8 \\
  --log-interval 250 \\
  --no-eval-stf \\
  --eval-kitti \\
  --kitti-eval-protocol rgb_checkpoint_decoder \\
  --eval-nyu \\
  --eval-eth3d \\
  --eth3d-eval-mode fast \\
  --eth3d-fast-eval-backend proxy \\
  --eth3d-max-samples 150 \\
  --eval-robotcar \\
  --robotcar-eval-mode fast \\
  --robotcar-fast-eval-backend sparse \\
  --eval-robotcar-night \\
  --robotcar-night-fast-eval-backend sparse \\
  --best-metric robotcar_night \\
  --save-best-checkpoint \\
  --pretrained-from "${PRETRAINED}" \\
  --heavy-save-root "${HEAVY_ROOT}" \\
  --save-path "${save}" 2>&1 | tee -a "${log}"
status=\${PIPESTATUS[0]}
set -e
echo "[END] \$(date -Iseconds) status=\${status}" 2>&1 | tee -a "${log}"
exit "\${status}"
EOF
  chmod +x "${entry}"
}

launch_run() {
  local run="$1"
  local session="$2"
  local gpu="$3"
  local port="$4"
  local lod_fraction_arg="$5"
  local save="${EXP_ROOT}/${run}"
  local log="${save}/tmux_launch.log"
  local entry="${save}/tmux_entry.sh"

  make_entry "${run}" "${session}" "${gpu}" "${port}" "${lod_fraction_arg}"
  tmux new-session -d -s "${session}" "${entry}"

  echo "[LAUNCHED] ${run}"
  echo "session=${session}"
  echo "log=${log}"
  echo "attach=tmux attach -t ${session}"
  echo "tail=tail -f ${log}"
}

mkdir -p "${EXP_ROOT}"
cd "${ROOT}"

ts=$(date +%m%d_%H%M)
run15="${ts}_vits_rgb_vkitti_lod_decoder_518x812_bs8acc1_lod15_vk85_randomcrop_ssi_rcnightbest_eth3d150_e10"
run50="${ts}_vits_rgb_vkitti_lod_decoder_518x812_bs8acc1_lod50_lod1vk1_randomcrop_ssi_rcnightbest_eth3d150_e10"

session15="${ts}_rgb_vkitti_lod15_gpu0"
session50="${ts}_rgb_vkitti_lod50_gpu1"

launch_run "${run15}" "${session15}" 0 29641 "--lod-fraction 0.15"
launch_run "${run50}" "${session50}" 1 29642 ""
