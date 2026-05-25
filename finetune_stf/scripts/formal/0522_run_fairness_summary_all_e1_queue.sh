#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
SESSION_PREFIX="${SESSION_PREFIX:-stf_fairness_summary_all_e1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

PRETRAINED_VITS="${PRETRAINED_VITS:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
DAV2_MANIFEST="${DAV2_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"
DA3_MANIFEST="${DA3_MANIFEST:-/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_da3mono_large_rgb_lut_6216_0521_0051/stf_rgb_lut_manifest_6216.csv}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0522_run_fairness_summary_all_e1_queue.sh
  bash finetune_stf/scripts/formal/0522_run_fairness_summary_all_e1_queue.sh --audit

Starts one tmux session that sequentially runs the training experiments listed in
plans/result/rgb_raw_baseline_fairness_summary.md, each with epochs=1.

The --audit mode validates the resolved configs without launching training.

Overrides:
  GPU=1 WAIT_FOR_GPU_IDLE=0 bash ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--audit" ]]; then
  AUDIT_ONLY=1
  shift
fi

if [[ "${AUDIT_ONLY}" != "1" && "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  queue_ts="$(date +%m%d_%H%M)"
  session="${queue_ts}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi

  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' WAIT_FOR_GPU_IDLE='${WAIT_FOR_GPU_IDLE}' WAIT_INTERVAL_SEC='${WAIT_INTERVAL_SEC}' PRETRAINED_VITS='${PRETRAINED_VITS}' STF_ROOT='${STF_ROOT}' RAW_NPZ_ROOT='${RAW_NPZ_ROOT}' DAV2_MANIFEST='${DAV2_MANIFEST}' DA3_MANIFEST='${DA3_MANIFEST}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

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

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_audit_${run}"
  PHASE1_BNCLEAN_REVIEWED=1 "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "$@" \
    --pretrained-from "${PRETRAINED_VITS}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${audit_save}" <<'PY'
import json
import sys

from finetune_stf.train import parse_args

args = parse_args()
cfg = args.resolved_config.to_dict()
payload = {
    "run": args.save_path.rsplit("/", 1)[-1],
    "input_type": args.input_type,
    "front_end": cfg["front_end"],
    "bridge": cfg["bridge"],
    "adapter": cfg["decoder_feature_adapter"],
    "lora": cfg["lora"],
    "dav2_train_mode": args.dav2_train_mode,
    "epochs": args.epochs,
    "target": args.stf_train_target_mode,
    "loss": args.loss_type,
}
print("[AUDIT][OK] " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

run_train() {
  local label="$1"
  local run="$2"
  local port="$3"
  shift 3

  if [[ "${AUDIT_ONLY}" == "1" ]]; then
    audit_train_args "${run}" "$@"
    return 0
  fi

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
    printf ' --pretrained-from %q --heavy-save-root %q --save-path %q\n' "${PRETRAINED_VITS}" "${HEAVY_ROOT}" "${save}"
  } 2>&1 | tee -a "${log}"

  set +e
  PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    "$@" \
    --pretrained-from "${PRETRAINED_VITS}" \
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
  --loss-norm-min-scale 1e-3
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --epochs 1
)

rgb_base_args=(
  --input-domain rgb
  --front-end dav2_rgb
  --dataset-family stf_rgb
  --dataset-input-mode rgb
  --model-input-tensor image
  --raw-storage-format n_a
  --bridge none
  --decoder-feature-adapter none
)

raw3_base_args=(
  --raw-npz-root "${RAW_NPZ_ROOT}"
  --input-domain raw4
  --front-end raw_to_base_rgb_ram3
  --dataset-family stf_raw
  --dataset-input-mode raw_ram
  --model-input-tensor raw
  --raw-storage-format legacy_bggR_decomp16
  --raw-front-end-lr 5e-5
  --norm-mode passthrough
  --channel-mode rgb_avg_g
  --raw-ram-rgb-tail identity
)

raw4_base_args=(
  --raw-npz-root "${RAW_NPZ_ROOT}"
  --input-domain raw4
  --front-end raw_ram4
  --dataset-family stf_raw
  --dataset-input-mode raw_ram
  --model-input-tensor raw
  --raw-storage-format legacy_bggR_decomp16
  --raw-front-end-lr 5e-5
  --norm-mode passthrough
  --channel-mode rgb_avg_g
  --rgb-interface-mode residual_tanh
)

lora_args=(
  --lora dav2_lora
  --lora-block-mode tap
  --lora-tap-layers 2 5 8 11
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
)

no_lora_args=(
  --lora none
)

bridge3_args=(
  --bridge raw_feature_bridge
  --bridge-feature-source-channels x3
  --bridge-feature-keys x_cat ffm_mid x3
  --bridge-layers 2 5 8 11
  --bridge-source ram_core
  --bridge-lr 5e-5
)

bridge4_args=(
  --bridge raw_feature_bridge
  --bridge-feature-source-channels x4
  --bridge-feature-keys x_cat ffm_mid x4
  --bridge-layers 2 5 8 11
  --bridge-source ram_core
  --bridge-lr 5e-5
)

adapter3_args=(
  --decoder-feature-adapter raw_feature_adapter
  --adapter-feature-source-channels x3
  --feature-adapter-keys x_cat ffm_mid x3
)

adapter4_args=(
  --decoder-feature-adapter raw_feature_adapter
  --adapter-feature-source-channels x4
  --feature-adapter-keys x_cat ffm_mid x4
)

no_adapter_args=(
  --decoder-feature-adapter none
)

dav2_target_args=(
  --stf-train-target-mode dav2_pseudo
  --stf-pseudo-manifest "${DAV2_MANIFEST}"
)

da3_target_args=(
  --stf-train-target-mode da3_pseudo_sparse_metric
  --stf-pseudo-manifest "${DA3_MANIFEST}"
)

run_formal_exp() {
  local label="$1"
  local suffix="$2"
  local port="$3"
  shift 3
  local ts run
  ts="$(date +%m%d_%H%M)"
  run="${ts}_${suffix}"
  echo "[QUEUE][RUN] ${label} -> ${run}"
  run_train "${label}" "${run}" "${port}" "$@"
}

run_all() {
  run_formal_exp "0521_0012 raw_ram_rgb frozen" "repro_0521_0012_raw_ram_rgb_bnclean_identity_ram_only_e1" 29801 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${no_adapter_args[@]}" "${no_lora_args[@]}" \
    --bridge none --dav2-train-mode none --backbone-layer-decay 1.0

  run_formal_exp "0521_1542 raw_ram_rgb_bridge frozen" "repro_0521_1542_raw_ram_rgb_bridge_ram_e1" 29802 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${bridge3_args[@]}" "${no_adapter_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode none --backbone-layer-decay 1.0

  run_formal_exp "0522_0137 raw_ram4 bridge+feature_adapter frozen" "repro_0522_0137_raw_ram_bridge_feature_adapter_ram_e1" 29803 \
    "${common_args[@]}" "${raw4_base_args[@]}" "${dav2_target_args[@]}" "${bridge4_args[@]}" "${adapter4_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode none --backbone-layer-decay 1.0

  run_formal_exp "0522_1423 raw_ram_rgb bridge+feature_adapter frozen" "repro_0522_1423_raw_ram_rgb_bridge_feature_adapter_ram_e1" 29804 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${bridge3_args[@]}" "${adapter3_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode none --backbone-layer-decay 1.0

  run_formal_exp "0521_0112 raw_ram_rgb decoder" "repro_0521_0112_raw_ram_rgb_bnclean_identity_decoder_e1" 29805 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${no_adapter_args[@]}" "${no_lora_args[@]}" \
    --bridge none --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_0133 rgb decoder" "repro_0521_0133_rgb_decoder_e1" 29806 \
    "${common_args[@]}" "${rgb_base_args[@]}" "${dav2_target_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_0306 rgb lora decoder" "repro_0521_0306_rgb_lora_decoder_e1" 29807 \
    "${common_args[@]}" "${rgb_base_args[@]}" "${dav2_target_args[@]}" "${lora_args[@]}" \
    --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_0402 rgb full lrd09" "repro_0521_0402_rgb_full_lrd09_e1" 29808 \
    "${common_args[@]}" "${rgb_base_args[@]}" "${dav2_target_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode full --backbone-layer-decay 0.9

  run_formal_exp "0521_0522 raw_ram_rgb lora decoder" "repro_0521_0522_raw_ram_rgb_bnclean_identity_lora_decoder_e1" 29809 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${no_adapter_args[@]}" "${lora_args[@]}" \
    --bridge none --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_0656 raw_ram_rgb full lrd09" "repro_0521_0656_raw_ram_rgb_bnclean_identity_full_lrd09_e1" 29810 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${no_adapter_args[@]}" "${no_lora_args[@]}" \
    --bridge none --dav2-train-mode full --backbone-layer-decay 0.9

  run_formal_exp "0521_0835 raw_ram_rgb_bridge lora decoder" "repro_0521_0835_raw_ram_rgb_bridge_lora_decoder_e1" 29811 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${bridge3_args[@]}" "${no_adapter_args[@]}" "${lora_args[@]}" \
    --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_1606 raw_ram4 bridge+feature_adapter lora decoder" "repro_0521_1606_raw_ram_bridge_feature_adapter_lora_decoder_e1" 29812 \
    "${common_args[@]}" "${raw4_base_args[@]}" "${dav2_target_args[@]}" "${bridge4_args[@]}" "${adapter4_args[@]}" "${lora_args[@]}" \
    --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_1004 raw_ram_rgb_bridge full lrd09" "repro_0521_1004_raw_ram_rgb_bridge_full_lrd09_e1" 29813 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${dav2_target_args[@]}" "${bridge3_args[@]}" "${no_adapter_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode full --backbone-layer-decay 0.9

  run_formal_exp "0521_1606 raw_ram4 bridge+feature_adapter full lrd09" "repro_0521_1606_raw_ram_bridge_feature_adapter_full_lrd09_e1" 29814 \
    "${common_args[@]}" "${raw4_base_args[@]}" "${dav2_target_args[@]}" "${bridge4_args[@]}" "${adapter4_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode full --backbone-layer-decay 0.9

  run_formal_exp "0521_2217 raw_ram4 bridge+feature_adapter full no lrd" "repro_0521_2217_raw_ram_bridge_feature_adapter_full_no_lrd_e1" 29815 \
    "${common_args[@]}" "${raw4_base_args[@]}" "${dav2_target_args[@]}" "${bridge4_args[@]}" "${adapter4_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode full --backbone-layer-decay 1.0

  run_formal_exp "0521_1137 da3 rgb lora decoder" "repro_0521_1137_pseudoda3_rgb_lora_decoder_e1" 29816 \
    "${common_args[@]}" "${rgb_base_args[@]}" "${da3_target_args[@]}" "${lora_args[@]}" \
    --dav2-train-mode decoder --backbone-layer-decay 1.0

  run_formal_exp "0521_1308 da3 raw_ram_rgb lora decoder" "repro_0521_1308_pseudoda3_raw_ram_rgb_lora_decoder_e1" 29817 \
    "${common_args[@]}" "${raw3_base_args[@]}" "${da3_target_args[@]}" "${no_adapter_args[@]}" "${lora_args[@]}" \
    --bridge none --dav2-train-mode decoder --backbone-layer-decay 1.0
}

require_file "${PRETRAINED_VITS}"
require_file "${DAV2_MANIFEST}"
require_file "${DA3_MANIFEST}"
require_dir "${STF_ROOT}"
require_dir "${RAW_NPZ_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SOURCE] /home/caq/6666_raw/dav2_raw_0520/plans/result/rgb_raw_baseline_fairness_summary.md"
echo "[SCOPE] 17 training runs, epochs=1; direct zero-shot baselines are eval-only and not queued here"
echo "[PRETRAINED_VITS] ${PRETRAINED_VITS}"
echo "[DAV2_MANIFEST] ${DAV2_MANIFEST}"
echo "[DA3_MANIFEST] ${DA3_MANIFEST}"

run_all

echo "[QUEUE_END] $(date -Iseconds) status=0"
