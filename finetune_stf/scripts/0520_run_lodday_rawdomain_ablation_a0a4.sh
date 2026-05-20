#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/caq/6666_raw/dav2_raw_0520}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-0}"
GPUS="${GPUS:-1}"
TS="${TS:-$(date +%m%d_%H%M)}"

EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/finetune_stf/exp}"
HEAVY_SAVE_ROOT="${HEAVY_SAVE_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CFG_DIR="${CFG_DIR:-${REPO_ROOT}/configs/raw_domain_ablation}"

PRETRAINED_FROM="${PRETRAINED_FROM:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LOD_ROOT="${LOD_ROOT:-/mnt/drive/3333_raw/LOD}"
LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest.csv}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_subset100}"
ROBOTCAR_NIGHT_ROOT="${ROBOTCAR_NIGHT_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo}"
ROBOTCAR_NIGHT_MANIFEST_NAME="${ROBOTCAR_NIGHT_MANIFEST_NAME:-robotcar_raw_depth_v1_val_balanced250_scene_interleaved.csv}"

EPOCHS="${EPOCHS:-2}"
TRAIN_STEPS_PER_EPOCH="${TRAIN_STEPS_PER_EPOCH:-757}"
BS="${BS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-5}"
INPUT_HEIGHT="${INPUT_HEIGHT:-518}"
INPUT_WIDTH="${INPUT_WIDTH:-812}"
BASE_PORT="${BASE_PORT:-29680}"

ABLS=("$@")
if [[ ${#ABLS[@]} -eq 0 ]]; then
  ABLS=(A0 A1 A2 A3 A4)
fi

cd "${REPO_ROOT}"

echo "[RUN] lodday raw-domain ablation ${ABLS[*]}"
echo "[START] $(date --iso-8601=seconds)"
echo "[HOST] $(hostname) [USER] $(whoami) [PWD] $(pwd)"
echo "[GPU] CUDA_VISIBLE_DEVICES=${GPU}"
echo "[TS] ${TS}"
echo "[CFG_DIR] ${CFG_DIR}"
echo "[EPOCHS] ${EPOCHS} [TRAIN_STEPS_PER_EPOCH] ${TRAIN_STEPS_PER_EPOCH} [BS] ${BS} [ACCUM] ${ACCUM_STEPS}"

for idx in "${!ABLS[@]}"; do
  abl="${ABLS[$idx]}"
  run_name="${TS}_lodday_rawdomain_${abl}_raw_ram_rgb_e${EPOCHS}_rcdaybest"
  run_dir="${EXP_ROOT}/${run_name}"
  master_port=$((BASE_PORT + idx))

  mkdir -p "${run_dir}"
  {
    echo "[RUN][${abl}] run_name=${run_name}"
    echo "[RUN][${abl}] run_dir=${run_dir}"
    echo "[RUN][${abl}] heavy_dir=${HEAVY_SAVE_ROOT}/${run_name}"
    echo "[RUN][${abl}] master_port=${master_port}"
    echo "[RUN][${abl}] lod_config=${CFG_DIR}/${abl}_lod_raw_domain.json"
    echo "[RUN][${abl}] robotcar_day_config=${CFG_DIR}/${abl}_robotcar_day_raw_domain.json"
    echo "[RUN][${abl}] robotcar_night_config=${CFG_DIR}/${abl}_robotcar_night_raw_domain.json"
    echo "[RUN][${abl}] start=$(date --iso-8601=seconds)"
  } | tee "${run_dir}/tmux_launch.log"

  PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    /home/caq/anaconda3/bin/conda run -n "${CONDA_ENV}" torchrun \
      --nproc_per_node="${GPUS}" \
      --master_port "${master_port}" \
      finetune_stf/train.py \
        --encoder vits \
        --stage raw_mix \
        --train-sources lod_day \
        --train-source-ratios 1.0 \
        --train-steps-per-epoch "${TRAIN_STEPS_PER_EPOCH}" \
        --input-type raw_ram_rgb \
        --norm-mode sensor_linear \
        --dav2-train-mode full \
        --backbone-layer-decay 0.9 \
        --lod-root "${LOD_ROOT}" \
        --lod-day-manifest "${LOD_DAY_MANIFEST}" \
        --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \
        --lod-crop-mode random \
        --lod-raw-domain-config "${CFG_DIR}/${abl}_lod_raw_domain.json" \
        --input-height "${INPUT_HEIGHT}" \
        --input-width "${INPUT_WIDTH}" \
        --bs "${BS}" \
        --accum-steps "${ACCUM_STEPS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --loss-type ssi \
        --loss-target-normalization \
        --loss-lambda-grad 2.0 \
        --amp \
        --amp-dtype bf16 \
        --seed 42 \
        --num-workers "${NUM_WORKERS}" \
        --log-interval 250 \
        --no-eval-stf \
        --eval-robotcar \
        --robotcar-root "${ROBOTCAR_ROOT}" \
        --robotcar-eval-mode fast \
        --robotcar-fast-eval-backend sparse \
        --robotcar-raw-domain-config "${CFG_DIR}/${abl}_robotcar_day_raw_domain.json" \
        --eval-robotcar-night \
        --robotcar-night-root "${ROBOTCAR_NIGHT_ROOT}" \
        --robotcar-night-manifest-name "${ROBOTCAR_NIGHT_MANIFEST_NAME}" \
        --robotcar-night-fast-eval-backend sparse \
        --robotcar-night-raw-domain-config "${CFG_DIR}/${abl}_robotcar_night_raw_domain.json" \
        --best-metric robotcar_day \
        --save-best-checkpoint \
        --pretrained-from "${PRETRAINED_FROM}" \
        --heavy-save-root "${HEAVY_SAVE_ROOT}" \
        --save-path "${run_dir}" \
        2>&1 | tee "${run_dir}/train.log"

  status=${PIPESTATUS[0]}
  echo "[RUN][${abl}] end=$(date --iso-8601=seconds) status=${status}" | tee -a "${run_dir}/tmux_launch.log"
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR][${abl}] stopping sequence because the run failed"
    exit "${status}"
  fi
done

echo "[END] $(date --iso-8601=seconds) status=0"
