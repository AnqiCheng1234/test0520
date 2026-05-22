#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
SESSION_PREFIX="${SESSION_PREFIX:-stf_ram_fa_bridge_from_0521_1542}"
RUN_SMOKE="${RUN_SMOKE:-1}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
DAV2_MANIFEST="${DAV2_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"

SMOKE_PORT="${SMOKE_PORT:-29784}"
FORMAL_PORT="${FORMAL_PORT:-29734}"
RUN_SUFFIX="${RUN_SUFFIX:-stf_train_test_pseudovitl_raw_ram_bridge_feature_adapter_ram_e10_from_0521_1542_setting}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/0522_run_stf_ram_feature_adapter_bridge_from_0521_1542_queue.sh

Starts one tmux session that:
  1. optionally runs a codex_smoke validation run,
  2. deletes successful codex_smoke artifacts only,
  3. launches the formal STF experiment based on 0521_1542 with feature adapter added.

Overrides:
  GPU=1 WAIT_FOR_GPU_IDLE=0 RUN_SMOKE=0 bash ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  queue_ts="$(date +%m%d_%H%M)"
  session="${queue_ts}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi

  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' WAIT_FOR_GPU_IDLE='${WAIT_FOR_GPU_IDLE}' WAIT_INTERVAL_SEC='${WAIT_INTERVAL_SEC}' RUN_SMOKE='${RUN_SMOKE}' PRETRAINED='${PRETRAINED}' STF_ROOT='${STF_ROOT}' RAW_NPZ_ROOT='${RAW_NPZ_ROOT}' DAV2_MANIFEST='${DAV2_MANIFEST}' SMOKE_PORT='${SMOKE_PORT}' FORMAL_PORT='${FORMAL_PORT}' RUN_SUFFIX='${RUN_SUFFIX}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

  cat <<EOF
Started tmux session: ${session}
Queue log: ${queue_log}
Attach: tmux attach -t ${session}
Monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}"
export PHASE1_BNCLEAN_REVIEWED=1

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

wait_for_gpu_idle() {
  if [[ "${WAIT_FOR_GPU_IDLE}" != "1" ]]; then
    return 0
  fi

  while true; do
    local apps
    apps="$(CUDA_VISIBLE_DEVICES="${GPU}" nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${apps//[[:space:]]/}" ]]; then
      echo "[GPU] idle on CUDA_VISIBLE_DEVICES=${GPU}"
      return 0
    fi
    echo "[GPU] busy on CUDA_VISIBLE_DEVICES=${GPU}; waiting ${WAIT_INTERVAL_SEC}s"
    echo "${apps}"
    sleep "${WAIT_INTERVAL_SEC}"
  done
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

log_header() {
  local label="$1"
  local run="$2"
  local port="$3"
  local log="$4"
  {
    echo "[START] $(date -Iseconds)"
    echo "[LABEL] ${label}"
    echo "[RUN] ${run}"
    echo "[SESSION] ${QUEUE_SESSION:-internal}"
    echo "[HOST] $(hostname)"
    echo "[USER] $(whoami)"
    echo "[PWD] $(pwd)"
    echo "[GPU] ${GPU}"
    echo "[PORT] ${port}"
    echo "[LOG] ${log}"
  } 2>&1 | tee -a "${log}"
}

run_train() {
  local label="$1"
  local run="$2"
  local port="$3"
  shift 3

  local save="${EXP_ROOT}/${run}"
  local heavy="${HEAVY_ROOT}/${run}"
  local log="${LOG_ROOT}/${run}.tmux.log"

  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] Refusing to overwrite existing artifacts for ${run}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi

  wait_for_gpu_idle
  mkdir -p "${save}"
  log_header "${label}" "${run}" "${port}" "${log}"

  {
    echo -n "[CMD] PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} torchrun --nproc_per_node=1 --master_port=${port} finetune_stf/train.py"
    printf ' %q' "$@"
    printf ' --pretrained-from %q --heavy-save-root %q --save-path %q\n' "${PRETRAINED}" "${HEAVY_ROOT}" "${save}"
  } 2>&1 | tee -a "${log}"

  set +e
  PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    "$@" \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e

  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
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
  --loss-mask-downsample strict
  --loss-target-normalization
  --loss-norm-min-scale 1e-3
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
  --input-type raw_ram_bridge_feature_adapter
  --dav2-train-mode none
  --epochs 10
)

smoke_args=(
  --epochs 1
  --bs 2
  --accum-steps 1
  --num-workers 0
  --log-interval 1
  --debug-max-train-steps 2
  --debug-max-val-samples 8
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
  --no-train-viz-rgb-baseline
)

run_smoke() {
  local smoke_ts run
  smoke_ts="$(date +%m%d_%H%M)"
  run="codex_smoke_${smoke_ts}_${RUN_SUFFIX}"
  if run_train "smoke 0521_1542 + feature adapter, train RAM/FA/bridge" "${run}" "${SMOKE_PORT}" \
    "${common_args[@]}" "${smoke_args[@]}"; then
    cleanup_smoke_artifacts "${EXP_ROOT}/${run}" "${HEAVY_ROOT}/${run}" "${LOG_ROOT}/${run}.tmux.log"
  else
    local status=$?
    echo "[SMOKE][FAIL] ${run} status=${status}"
    echo "[SMOKE][KEEP] ${EXP_ROOT}/${run}"
    echo "[SMOKE][KEEP] ${HEAVY_ROOT}/${run}"
    echo "[SMOKE][KEEP] ${LOG_ROOT}/${run}.tmux.log"
    return "${status}"
  fi
}

run_formal() {
  local ts run
  ts="$(date +%m%d_%H%M)"
  run="${ts}_${RUN_SUFFIX}"
  run_train "formal 0521_1542 + feature adapter, train RAM/FA/bridge only" "${run}" "${FORMAL_PORT}" \
    "${common_args[@]}"
}

require_file "${PRETRAINED}"
require_file "${DAV2_MANIFEST}"
require_dir "${STF_ROOT}"
require_dir "${RAW_NPZ_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[QUEUE_POLICY] RUN_SMOKE=${RUN_SMOKE}; stop on first failure"
echo "[REFERENCE] 0521_1542_stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_ram_e10"
echo "[CHANGE] input_type=raw_ram_bridge_feature_adapter dav2_train_mode=none feature_keys=x_cat,ffm_mid,x4"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[DAV2_MANIFEST] ${DAV2_MANIFEST}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_smoke
else
  echo "[SKIP] RUN_SMOKE=${RUN_SMOKE}; not running smoke test"
fi

run_formal

echo "[QUEUE_END] $(date -Iseconds) status=0"
