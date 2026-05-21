#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"

PRETRAINED="${PRETRAINED:-/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth}"
STF_ROOT="${STF_ROOT:-/home/a5000/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
DAV2_MANIFEST="${DAV2_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"

GPU_FULL="${GPU_FULL:-0}"
GPU_LORA="${GPU_LORA:-1}"
PORT_FULL="${PORT_FULL:-29741}"
PORT_LORA="${PORT_LORA:-29742}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"

TS="${TS:-$(date +%m%d_%H%M)}"
RUN_FULL="${RUN_FULL:-${TS}_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_full_lrd09_e10_from_0521_1004_setting}"
RUN_LORA="${RUN_LORA:-${TS}_stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_lora_decoder_e10_from_0521_0835_setting}"
SESSION_FULL="${SESSION_FULL:-${RUN_FULL}}"
SESSION_LORA="${SESSION_LORA:-${RUN_LORA}}"

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

check_run_paths() {
  local run="$1"
  local save="${EXP_ROOT}/${run}"
  local heavy="${HEAVY_ROOT}/${run}"
  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] Refusing to overwrite existing artifacts for ${run}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    exit 2
  fi
}

wait_for_gpu_idle() {
  local gpu="$1"
  if [[ "${WAIT_FOR_GPU_IDLE}" != "1" ]]; then
    return 0
  fi

  while true; do
    local apps
    apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${apps//[[:space:]]/}" ]]; then
      echo "[GPU] idle on CUDA_VISIBLE_DEVICES=${gpu}"
      return 0
    fi
    echo "[GPU] busy on CUDA_VISIBLE_DEVICES=${gpu}; waiting ${WAIT_INTERVAL_SEC}s"
    echo "${apps}"
    sleep "${WAIT_INTERVAL_SEC}"
  done
}

common_args=(
  --encoder vits
  --stage stf_only
  --stf-root "${STF_ROOT}"
  --input-height 512
  --input-width 960
  --stf-fast-eval-backend sparse
  --eval-stf
  --best-metric stf
  --save-best-checkpoint
  --bs 8
  --accum-steps 1
  --lr 1e-5
  --loss-type ssi
  --loss-target-normalization
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --stf-repeat 7
  --raw-npz-root "${RAW_NPZ_ROOT}"
  --stf-raw-decode-mode legacy_online_decomp16
  --norm-mode passthrough
  --channel-mode rgb_avg_g
  --raw-ram-rgb-tail identity
  --rgb-interface-mode residual_tanh
  --bridge-feature-keys x_cat ffm_mid x4
  --bridge-layers 2 5 8 11
  --bridge-source ram_core
  --bridge-lr 5e-5
  --stf-train-target-mode dav2_pseudo
  --stf-pseudo-manifest "${DAV2_MANIFEST}"
  --epochs 10
)

lora_args=(
  --lora-block-mode tap
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
)

run_one() {
  local variant="$1"
  local label run gpu port log
  local -a variant_args

  case "${variant}" in
    full)
      label="0521_1004 setting + decoder feature adapter, full DAv2"
      run="${RUN_FULL}"
      gpu="${GPU_FULL}"
      port="${PORT_FULL}"
      variant_args=(
        --input-type raw_ram_bridge_feature_adapter
        --dav2-train-mode full
        --backbone-layer-decay 0.9
      )
      ;;
    lora)
      label="0521_0835 setting + decoder feature adapter, LoRA decoder"
      run="${RUN_LORA}"
      gpu="${GPU_LORA}"
      port="${PORT_LORA}"
      variant_args=(
        "${lora_args[@]}"
        --input-type raw_ram_bridge_feature_adapter_lora
        --dav2-train-mode decoder
        --backbone-layer-decay 1.0
      )
      ;;
    *)
      echo "[ERROR] Unknown variant: ${variant}" >&2
      exit 2
      ;;
  esac

  local save="${EXP_ROOT}/${run}"
  log="${LOG_ROOT}/${run}.tmux.log"

  mkdir -p "${save}" "${LOG_ROOT}"
  wait_for_gpu_idle "${gpu}"

  echo "[START] $(date -Iseconds)"
  echo "[LABEL] ${label}"
  echo "[RUN] ${run}"
  echo "[HOST] $(hostname)"
  echo "[USER] $(whoami)"
  echo "[PWD] $(pwd)"
  echo "[GPU] ${gpu}"
  echo "[PORT] ${port}"
  echo "[LOG] ${log}"
  echo "[PRETRAINED] ${PRETRAINED}"
  echo "[DAV2_MANIFEST] ${DAV2_MANIFEST}"
  echo "[NOTE] No resume_from or bridge_init_from is used; this follows settings only."

  set +e
  PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES="${gpu}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python -m torch.distributed.run --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    "${common_args[@]}" \
    "${variant_args[@]}" \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}"
  local status=$?
  set -e

  echo "[END] $(date -Iseconds) status=${status}"
  exit "${status}"
}

if [[ "${1:-}" == "--run-one" ]]; then
  cd "${ROOT}"
  mkdir -p "${EXP_ROOT}" "${LOG_ROOT}"
  require_file "${PRETRAINED}"
  require_file "${DAV2_MANIFEST}"
  require_dir "${STF_ROOT}"
  require_dir "${RAW_NPZ_ROOT}"
  run_one "${2:-}"
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}"
require_file "${PRETRAINED}"
require_file "${DAV2_MANIFEST}"
require_dir "${STF_ROOT}"
require_dir "${RAW_NPZ_ROOT}"
check_run_paths "${RUN_FULL}"
check_run_paths "${RUN_LORA}"

if tmux has-session -t "${SESSION_FULL}" 2>/dev/null; then
  echo "[ERROR] Refusing to reuse existing tmux session: ${SESSION_FULL}" >&2
  exit 2
fi
if tmux has-session -t "${SESSION_LORA}" 2>/dev/null; then
  echo "[ERROR] Refusing to reuse existing tmux session: ${SESSION_LORA}" >&2
  exit 2
fi

LOG_FULL="${LOG_ROOT}/${RUN_FULL}.tmux.log"
LOG_LORA="${LOG_ROOT}/${RUN_LORA}.tmux.log"

tmux new-session -d -s "${SESSION_FULL}" \
  "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' PRETRAINED='${PRETRAINED}' STF_ROOT='${STF_ROOT}' RAW_NPZ_ROOT='${RAW_NPZ_ROOT}' DAV2_MANIFEST='${DAV2_MANIFEST}' GPU_FULL='${GPU_FULL}' PORT_FULL='${PORT_FULL}' WAIT_FOR_GPU_IDLE='${WAIT_FOR_GPU_IDLE}' WAIT_INTERVAL_SEC='${WAIT_INTERVAL_SEC}' RUN_FULL='${RUN_FULL}' bash '$0' --run-one full 2>&1 | tee -a '${LOG_FULL}'"

tmux new-session -d -s "${SESSION_LORA}" \
  "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' PRETRAINED='${PRETRAINED}' STF_ROOT='${STF_ROOT}' RAW_NPZ_ROOT='${RAW_NPZ_ROOT}' DAV2_MANIFEST='${DAV2_MANIFEST}' GPU_LORA='${GPU_LORA}' PORT_LORA='${PORT_LORA}' WAIT_FOR_GPU_IDLE='${WAIT_FOR_GPU_IDLE}' WAIT_INTERVAL_SEC='${WAIT_INTERVAL_SEC}' RUN_LORA='${RUN_LORA}' bash '$0' --run-one lora 2>&1 | tee -a '${LOG_LORA}'"

cat <<EOF
Started tmux sessions:
  ${SESSION_FULL}
  ${SESSION_LORA}

Logs:
  ${LOG_FULL}
  ${LOG_LORA}

Attach:
  tmux attach -t ${SESSION_FULL}
  tmux attach -t ${SESSION_LORA}

Monitor:
  tail -f ${LOG_FULL}
  tail -f ${LOG_LORA}
EOF
