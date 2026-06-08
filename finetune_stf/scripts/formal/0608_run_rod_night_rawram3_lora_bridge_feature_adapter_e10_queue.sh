#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
ROD_ROOT="${ROD_ROOT:-/home/caq/6666_raw/0000_dataset/ROD}"
ROD_MANIFEST="${ROD_MANIFEST:-${ROD_ROOT}/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-rod_night_rawram3_lora_bridge_fa_e10}"
MASTER_PORT="${MASTER_PORT:-29617}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
EPOCHS="${EPOCHS:-10}"
RUN_SUFFIX="${RUN_SUFFIX:-rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e${EPOCHS}}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0608_run_rod_night_rawram3_lora_bridge_feature_adapter_e10_queue.sh
  bash finetune_stf/scripts/formal/0608_run_rod_night_rawram3_lora_bridge_feature_adapter_e10_queue.sh --audit

Starts one tmux session for:
  ROD raw4 -> base RGB -> RamCore3 identity -> DAv2-S
  + raw feature bridge to backbone
  + decoder feature adapter
  + LoRA tap r8/a16
  + DAv2 decoder training, epochs=10 by default.

Overrides:
  GPU=1 MASTER_PORT=29627 EPOCHS=10 bash ...
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
  queue_timestamp="$(date +%m%d_%H%M)"
  session="${queue_timestamp}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' ROD_ROOT='${ROD_ROOT}' ROD_MANIFEST='${ROD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' EPOCHS='${EPOCHS}' RUN_SUFFIX='${RUN_SUFFIX}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' PHASE1_BNCLEAN_REVIEWED=1 bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
tmux session: ${session}
queue log: ${queue_log}
attach: tmux attach -t ${session}
monitor: tail -f ${queue_log}
EOF
  exit 0
fi

if [[ "${1:-}" == "--run-internal" ]]; then
  shift
fi

if [[ "$#" -gt 0 ]]; then
  usage >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}" "${HEAVY_ROOT}"
export PHASE1_BNCLEAN_REVIEWED=1

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing file: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] missing directory: ${path}" >&2
    exit 2
  fi
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
    echo "[MANIFEST] ${ROD_MANIFEST}"
    echo "[SEMANTICS] reference=0605_1844 rawram3 identity lora_tap r8/a16 decoder; add bridge=x3 and decoder_feature_adapter=x3"
  } 2>&1 | tee -a "${log}"
}

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_audit_${run}"
  PHASE1_BNCLEAN_REVIEWED=1 "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "$@" \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${audit_save}" <<'PY'
import json

from finetune_stf.train import parse_args, metric_direction

args = parse_args()
cfg = args.resolved_config.to_dict()
payload = {
    "run": args.save_path.rsplit("/", 1)[-1],
    "dataset_family": cfg["dataset_family"],
    "dataset_input_mode": cfg["dataset_input_mode"],
    "input_domain": cfg["input_domain"],
    "front_end": cfg["front_end"],
    "model_input_tensor": cfg["model_input_tensor"],
    "raw_storage_format": cfg["raw_storage_format"],
    "raw_ram_rgb_tail": args.raw_ram_rgb_tail,
    "bridge": cfg["bridge"],
    "bridge_feature_source_channels": cfg["bridge_feature_source_channels"],
    "bridge_feature_keys": cfg["bridge_feature_keys"],
    "bridge_layers": cfg["bridge_layers"],
    "decoder_feature_adapter": cfg["decoder_feature_adapter"],
    "adapter_feature_source_channels": cfg["adapter_feature_source_channels"],
    "feature_adapter_keys": cfg["feature_adapter_keys"],
    "bridge_source": cfg["bridge_source"],
    "lora": cfg["lora"],
    "lora_tap_layers": cfg["lora_tap_layers"],
    "lora_rank": cfg["lora_rank"],
    "lora_alpha": cfg["lora_alpha"],
    "lora_lr": cfg["lora_lr"],
    "dav2_train_mode": args.dav2_train_mode,
    "best_metric": args.best_metric,
    "best_metric_direction": metric_direction(args.best_metric),
    "rod_root": args.rod_root,
    "rod_night_manifest": args.rod_night_manifest,
    "epochs": args.epochs,
    "input_type_alias": cfg["input_type_alias"],
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
    echo "[ERROR] refusing to overwrite existing artifacts for ${run}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi

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
  --stage rod_only
  --rod-root "${ROD_ROOT}"
  --rod-night-manifest "${ROD_MANIFEST}"
  --rod-raw-source raw24
  --rod-label-space inverse_relative
  --input-height 512
  --input-width 960
  --rod-train-crop-mode random
  --rod-val-crop-mode center
  --no-eval-stf
  --eval-rod
  --best-metric rod
  --save-best-checkpoint
  --bs 8
  --accum-steps 1
  --loss-type ssi
  --loss-target-normalization
  --loss-norm-min-scale 1e-3
  --epochs "${EPOCHS}"
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
)

model_args=(
  --dataset-family rod_raw
  --dataset-input-mode raw_ram
  --input-domain raw4
  --front-end raw_to_base_rgb_ram3
  --model-input-tensor raw
  --raw-storage-format n_a
  --raw-front-end-lr 5e-5
  --raw-ram-rgb-tail identity
  --bridge raw_feature_bridge
  --decoder-feature-adapter raw_feature_adapter
  --bridge-feature-source-channels x3
  --adapter-feature-source-channels x3
  --bridge-feature-keys x_cat ffm_mid x3
  --feature-adapter-keys x_cat ffm_mid x3
  --bridge-layers 2 5 8 11
  --bridge-source ram_core
  --bridge-lr 5e-5
  --lora dav2_lora
  --lora-block-mode tap
  --lora-tap-layers 2 5 8 11
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
  --dav2-train-mode decoder
  --backbone-layer-decay 1.0
  --lr 1e-5
)

run_formal() {
  local ts run
  ts="$(date +%m%d_%H%M)"
  run="${ts}_${RUN_SUFFIX}"
  echo "[QUEUE][RUN] ROD rawram3 identity + LoRA tap + bridge + feature adapter -> ${run}"
  run_train "ROD rawram3 identity + LoRA tap + bridge + feature adapter" "${run}" "${MASTER_PORT}" \
    "${common_args[@]}" "${model_args[@]}"
}

require_file "${PRETRAINED}"
require_file "${ROD_MANIFEST}"
require_dir "${ROD_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SCOPE] 0605_1844 config plus bridge and decoder feature adapter, epochs=${EPOCHS}"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[ROD_ROOT] ${ROD_ROOT}"
echo "[ROD_MANIFEST] ${ROD_MANIFEST}"
echo "[PORT] ${MASTER_PORT}"
echo "[PHASE1_BNCLEAN_REVIEWED] ${PHASE1_BNCLEAN_REVIEWED:-unset}"

run_formal

echo "[QUEUE_END] $(date -Iseconds) status=0"
