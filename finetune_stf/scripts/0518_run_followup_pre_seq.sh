#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/caq/6666_raw/dav2_raw_0520
EXP_ROOT=${ROOT}/finetune_stf/exp
HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ckpt
LOD_DAY_MANIFEST=/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest_subset50_split_seed42.csv
LOD_NIGHT_MANIFEST=/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest_subset50_split_seed42.csv
PRETRAINED=/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
QUEUE_LOG_DIR=${ROOT}/plans/0518_followup_pre/logs

SESSION_PREFIX=followup_pre_seq
GPU=${CUDA_VISIBLE_DEVICES:-0}

usage() {
  cat <<'EOF'
Usage:
  finetune_stf/scripts/0518_run_followup_pre_seq.sh

Starts one tmux session that runs these experiments sequentially:
  1. robotcar_subset100_sensor_linear + random CCM, raw_ram_rgb_bridge, VKITTI+LOD 1:1
  2. RGB VKITTI+LOD baseline, based on 0517_2314 with LOD added 1:1

Each run name gets its own MMDD_HHMM prefix at the actual start time of that run.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${QUEUE_LOG_DIR}"
  QUEUE_TS=$(date +%m%d_%H%M)
  SESSION=${QUEUE_TS}_${SESSION_PREFIX}
  QUEUE_LOG=${QUEUE_LOG_DIR}/${SESSION}.log

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
  fi

  tmux new -d -s "${SESSION}" "cd '${ROOT}' && QUEUE_SESSION='${SESSION}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${QUEUE_LOG}'"

  cat <<EOF
Started tmux session: ${SESSION}
Queue log: ${QUEUE_LOG}
Attach: tmux attach -t ${SESSION}
Monitor: tail -f ${QUEUE_LOG}
EOF
  exit 0
fi

cd "${ROOT}"

log_header() {
  local run="$1"
  local session="$2"
  local port="$3"
  local log="$4"

  {
    echo "[START] $(date -Iseconds)"
    echo "[HOST] $(hostname)"
    echo "[PWD] $(pwd)"
    echo "[RUN] ${run}"
    echo "[SESSION] ${session}"
    echo "[PORT] ${port}"
    echo "[GPU] ${GPU}"
  } 2>&1 | tee -a "${log}"
}

run_robotcar_randomccm() {
  local ts run save log port
  ts=$(date +%m%d_%H%M)
  run=${ts}_vits_vkitti_lod_dn_robotcar_randomccm_3ch_bridge_decoder_518x812_bs8acc1_lod50_lod1vk1_randomcrop_ssi_rcnightbest_eth3d150_e10
  save=${EXP_ROOT}/${run}
  log=${save}/tmux_launch.log
  port=29631

  if [[ -e "${save}" ]]; then
    echo "[ERROR] Refusing to overwrite existing run directory: ${save}" >&2
    return 2
  fi
  mkdir -p "${save}"
  log_header "${run}" "${QUEUE_SESSION:-${SESSION_PREFIX}}" "${port}" "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" conda run -n dav3 torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    --encoder vits \
    --stage vkitti_lod \
    --lod-per-vkitti 1 \
    --lod-day-manifest "${LOD_DAY_MANIFEST}" \
    --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
    --lod-crop-mode random \
    --input-type raw_ram_rgb_bridge \
    --norm-mode sensor_linear \
    --channel-mode rgb_avg_g \
    --bridge-feature-keys x_cat ffm_mid x3 \
    --bridge-layers 2 5 8 11 \
    --bridge-source ram_core \
    --vkitti-unprocessing-preset robotcar_subset100_sensor_linear \
    --vkitti-randomize-unprocessing \
    --vkitti-hflip-prob 0.5 \
    --input-height 518 \
    --input-width 812 \
    --bs 8 \
    --accum-steps 1 \
    --epochs 10 \
    --lr 1e-5 \
    --bridge-lr 5e-5 \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-lambda-grad 2.0 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 8 \
    --log-interval 250 \
    --no-eval-stf \
    --eval-kitti \
    --kitti-eval-protocol rgb_checkpoint_decoder \
    --eval-nyu \
    --eval-eth3d \
    --eth3d-eval-mode fast \
    --eth3d-fast-eval-backend proxy \
    --eth3d-max-samples 150 \
    --eval-robotcar \
    --robotcar-eval-mode fast \
    --robotcar-fast-eval-backend sparse \
    --eval-robotcar-night \
    --robotcar-night-fast-eval-backend sparse \
    --best-metric robotcar_night \
    --save-best-checkpoint \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}" 2>&1 | tee -a "${log}"

  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

run_rgb_vkitti_lod() {
  local ts run save log port
  ts=$(date +%m%d_%H%M)
  run=${ts}_vits_rgb_vkitti_lod_decoder_518x812_bs8acc1_lod50_lod1vk1_randomcrop_ssi_rcnightbest_eth3d150_e10
  save=${EXP_ROOT}/${run}
  log=${save}/tmux_launch.log
  port=29632

  if [[ -e "${save}" ]]; then
    echo "[ERROR] Refusing to overwrite existing run directory: ${save}" >&2
    return 2
  fi
  mkdir -p "${save}"
  log_header "${run}" "${QUEUE_SESSION:-${SESSION_PREFIX}}" "${port}" "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" conda run -n dav3 torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    --encoder vits \
    --stage vkitti_lod \
    --lod-per-vkitti 1 \
    --lod-day-manifest "${LOD_DAY_MANIFEST}" \
    --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
    --lod-crop-mode random \
    --input-type rgb \
    --input-height 518 \
    --input-width 812 \
    --bs 8 \
    --accum-steps 1 \
    --epochs 10 \
    --lr 1e-5 \
    --loss-type ssi \
    --loss-target-normalization \
    --loss-lambda-grad 2.0 \
    --amp \
    --amp-dtype bf16 \
    --seed 42 \
    --num-workers 8 \
    --log-interval 250 \
    --no-eval-stf \
    --eval-kitti \
    --kitti-eval-protocol rgb_checkpoint_decoder \
    --eval-nyu \
    --eval-eth3d \
    --eth3d-eval-mode fast \
    --eth3d-fast-eval-backend proxy \
    --eth3d-max-samples 150 \
    --eval-robotcar \
    --robotcar-eval-mode fast \
    --robotcar-fast-eval-backend sparse \
    --eval-robotcar-night \
    --robotcar-night-fast-eval-backend sparse \
    --best-metric robotcar_night \
    --save-best-checkpoint \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}" 2>&1 | tee -a "${log}"

  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_HOST] $(hostname)"
echo "[QUEUE_PWD] $(pwd)"
echo "[QUEUE_GPU] ${GPU}"
echo "[QUEUE_POLICY] second run starts only if the first run exits with status 0"

if run_robotcar_randomccm; then
  echo "[QUEUE] first run completed successfully; starting RGB VKITTI+LOD"
else
  status=$?
  echo "[QUEUE_END] $(date -Iseconds) status=${status}"
  exit "${status}"
fi

if run_rgb_vkitti_lod; then
  echo "[QUEUE_END] $(date -Iseconds) status=0"
else
  status=$?
  echo "[QUEUE_END] $(date -Iseconds) status=${status}"
  exit "${status}"
fi
