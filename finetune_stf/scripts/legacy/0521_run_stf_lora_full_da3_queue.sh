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
SESSION_PREFIX="${SESSION_PREFIX:-stf_0521_lora_full_da3_seq}"
RUN_SMOKES="${RUN_SMOKES:-1}"
START_FORMAL_EXP="${START_FORMAL_EXP:-1}"
END_FORMAL_EXP="${END_FORMAL_EXP:-8}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
DAV2_MANIFEST="${DAV2_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"
DA3_MANIFEST="${DA3_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/0521_run_stf_lora_full_da3_queue.sh

Starts one tmux session that:
  1. waits for GPU idle by default,
  2. runs codex_smoke experiments for exp7 and exp8,
  3. deletes successful codex_smoke artifacts only,
  4. runs formal experiments 1..8 sequentially.

Overrides:
  GPU=0 WAIT_FOR_GPU_IDLE=0 CONDA_BIN=/path/to/conda bash ...
  RUN_SMOKES=0 START_FORMAL_EXP=2 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' WAIT_FOR_GPU_IDLE='${WAIT_FOR_GPU_IDLE}' WAIT_INTERVAL_SEC='${WAIT_INTERVAL_SEC}' RUN_SMOKES='${RUN_SMOKES}' START_FORMAL_EXP='${START_FORMAL_EXP}' END_FORMAL_EXP='${END_FORMAL_EXP}' PRETRAINED='${PRETRAINED}' STF_ROOT='${STF_ROOT}' RAW_NPZ_ROOT='${RAW_NPZ_ROOT}' DAV2_MANIFEST='${DAV2_MANIFEST}' DA3_MANIFEST='${DA3_MANIFEST}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

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
  --loss-target-normalization
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --stf-repeat 7
)

raw_common_args=(
  --raw-npz-root "${RAW_NPZ_ROOT}"
  --stf-raw-decode-mode legacy_online_decomp16
  --norm-mode passthrough
  --channel-mode rgb_avg_g
  --raw-ram-rgb-tail identity
  --rgb-interface-mode residual_tanh
)

bridge_common_args=(
  --norm-mode passthrough
  --channel-mode rgb_avg_g
  --bridge-feature-keys x_cat ffm_mid x3
  --bridge-layers 2 5 8 11
  --bridge-source ram_core
  --bridge-lr 5e-5
)

lora_args=(
  --lora-block-mode tap
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
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
  local label="$1"
  local run="$2"
  local port="$3"
  shift 3

  if run_train "${label}" "${run}" "${port}" "$@"; then
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
  local label="$1"
  local suffix="$2"
  local port="$3"
  shift 3

  local ts run
  ts="$(date +%m%d_%H%M)"
  run="${ts}_${suffix}"
  run_train "${label}" "${run}" "${port}" "$@"
}

run_formal_exp() {
  local exp_idx="$1"
  shift

  if (( exp_idx < START_FORMAL_EXP || exp_idx > END_FORMAL_EXP )); then
    echo "[SKIP] exp${exp_idx} outside formal range ${START_FORMAL_EXP}..${END_FORMAL_EXP}"
    return 0
  fi

  run_formal "$@"
}

require_file "${PRETRAINED}"
require_file "${DAV2_MANIFEST}"
require_file "${DA3_MANIFEST}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[QUEUE_POLICY] RUN_SMOKES=${RUN_SMOKES}; formal exp${START_FORMAL_EXP}..exp${END_FORMAL_EXP}; stop on first failure"
echo "[DAV2_MANIFEST] ${DAV2_MANIFEST}"
echo "[DA3_MANIFEST] ${DA3_MANIFEST}"

if [[ "${RUN_SMOKES}" == "1" ]]; then
  smoke_ts="$(date +%m%d_%H%M)"
  run_smoke "smoke exp7 da3 rgb_lora decoder" "codex_smoke_${smoke_ts}_exp7_da3_rgb_lora_decoder" 29681 \
    "${common_args[@]}" "${smoke_args[@]}" "${lora_args[@]}" \
    --input-type rgb_lora \
    --stf-train-target-mode da3_pseudo_sparse_metric \
    --stf-pseudo-manifest "${DA3_MANIFEST}" \
    --dav2-train-mode decoder

  smoke_ts="$(date +%m%d_%H%M)"
  run_smoke "smoke exp8 da3 raw_ram_rgb_lora decoder" "codex_smoke_${smoke_ts}_exp8_da3_raw_ram_rgb_lora_decoder" 29682 \
    "${common_args[@]}" "${smoke_args[@]}" "${raw_common_args[@]}" "${lora_args[@]}" \
    --input-type raw_ram_rgb_lora \
    --stf-train-target-mode da3_pseudo_sparse_metric \
    --stf-pseudo-manifest "${DA3_MANIFEST}" \
    --dav2-train-mode decoder
else
  echo "[SKIP] RUN_SMOKES=${RUN_SMOKES}; not running smoke tests"
fi

run_formal_exp 1 "exp1 dav2 rgb_lora decoder" "stf_train_test_pseudovitl_rgb_lora_decoder_e5" 29691 \
  "${common_args[@]}" "${lora_args[@]}" \
  --input-type rgb_lora \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode decoder \
  --epochs 5

run_formal_exp 2 "exp2 dav2 rgb full" "stf_train_test_pseudovitl_rgb_full_lrd09_e5" 29692 \
  "${common_args[@]}" \
  --input-type rgb \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode full \
  --backbone-layer-decay 0.9 \
  --epochs 5

run_formal_exp 3 "exp3 dav2 raw_ram_rgb_lora decoder" "stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_lora_decoder_e10" 29693 \
  "${common_args[@]}" "${raw_common_args[@]}" "${lora_args[@]}" \
  --input-type raw_ram_rgb_lora \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode decoder \
  --epochs 10

run_formal_exp 4 "exp4 dav2 raw_ram_rgb full" "stf_train_test_pseudovitl_raw_ram_rgb_bnclean_identity_full_lrd09_e10" 29694 \
  "${common_args[@]}" "${raw_common_args[@]}" \
  --input-type raw_ram_rgb \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode full \
  --backbone-layer-decay 0.9 \
  --bridge-lr 5e-5 \
  --epochs 10

run_formal_exp 5 "exp5 dav2 raw_ram_rgb_bridge_lora decoder" "stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_lora_decoder_e10" 29695 \
  "${common_args[@]}" "${raw_common_args[@]}" "${bridge_common_args[@]}" "${lora_args[@]}" \
  --input-type raw_ram_rgb_bridge_lora \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode decoder \
  --epochs 10

run_formal_exp 6 "exp6 dav2 raw_ram_rgb_bridge full" "stf_train_test_pseudovitl_raw_ram_rgb_bnclean_bridge_full_lrd09_e10" 29696 \
  "${common_args[@]}" "${raw_common_args[@]}" "${bridge_common_args[@]}" \
  --input-type raw_ram_rgb_bridge \
  --stf-train-target-mode dav2_pseudo \
  --stf-pseudo-manifest "${DAV2_MANIFEST}" \
  --dav2-train-mode full \
  --backbone-layer-decay 0.9 \
  --epochs 10

run_formal_exp 7 "exp7 da3 rgb_lora decoder" "stf_train_test_pseudoda3_sparse_metric_rgb_lora_decoder_e5" 29697 \
  "${common_args[@]}" "${lora_args[@]}" \
  --input-type rgb_lora \
  --stf-train-target-mode da3_pseudo_sparse_metric \
  --stf-pseudo-manifest "${DA3_MANIFEST}" \
  --dav2-train-mode decoder \
  --epochs 5

run_formal_exp 8 "exp8 da3 raw_ram_rgb_lora decoder" "stf_train_test_pseudoda3_sparse_metric_raw_ram_rgb_bnclean_identity_lora_decoder_e10" 29698 \
  "${common_args[@]}" "${raw_common_args[@]}" "${lora_args[@]}" \
  --input-type raw_ram_rgb_lora \
  --stf-train-target-mode da3_pseudo_sparse_metric \
  --stf-pseudo-manifest "${DA3_MANIFEST}" \
  --dav2-train-mode decoder \
  --epochs 10

echo "[QUEUE_END] $(date -Iseconds) status=0"
