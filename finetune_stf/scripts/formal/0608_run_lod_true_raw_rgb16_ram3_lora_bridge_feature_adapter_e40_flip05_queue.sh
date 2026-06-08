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
LOD_ROOT="${LOD_ROOT:-/home/caq/6666_raw/0000_dataset/LOD}"
LOD_MANIFEST="${LOD_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_true_raw_rgb16_ram3_lora_bridge_fa_e40_flip05_queue}"
MASTER_PORT="${MASTER_PORT:-29631}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
EPOCHS="${EPOCHS:-40}"
RUN_SUFFIX_LORA="${RUN_SUFFIX_LORA:-lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e${EPOCHS}_flip05_poly}"
RUN_SUFFIX_BRIDGE_FA="${RUN_SUFFIX_BRIDGE_FA:-lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_bridge_feature_adapter_decoder_e${EPOCHS}_flip05_poly}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0608_run_lod_true_raw_rgb16_ram3_lora_bridge_feature_adapter_e40_flip05_queue.sh
  bash finetune_stf/scripts/formal/0608_run_lod_true_raw_rgb16_ram3_lora_bridge_feature_adapter_e40_flip05_queue.sh --audit

Starts one tmux session for two LOD RAW_RGB16 block8/excl10 runs:
  1. RamCore3 + DAv2 decoder + LoRA tap r8/a16
  2. RamCore3 + DAv2 decoder + LoRA tap r8/a16 + x3 bridge + x3 decoder feature adapter

Both use epochs=40, poly LR schedule, and train-only hflip=0.5 with no other LOD augmentation.

Overrides:
  GPU=0 MASTER_PORT=29631 EPOCHS=40 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' EPOCHS='${EPOCHS}' RUN_SUFFIX_LORA='${RUN_SUFFIX_LORA}' RUN_SUFFIX_BRIDGE_FA='${RUN_SUFFIX_BRIDGE_FA}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
    echo "[LOD_ROOT] ${LOD_ROOT}"
    echo "[LOD_MANIFEST] ${LOD_MANIFEST}"
    echo "[SEMANTICS] split=block8_val112_excl10_uniform train=1958 val=112 buffer=160 epochs=${EPOCHS} aug=flip0.5_only lr_schedule=poly warmup_steps=0"
  } 2>&1 | tee -a "${log}"
}

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_lod_lora_audit_${run}"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "$@" \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${audit_save}" <<'PY'
import json

from finetune_stf.train import parse_args, metric_direction, resolve_lod_aug_config

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
    "epochs": args.epochs,
    "lr_schedule": args.lr_schedule,
    "warmup_steps": args.warmup_steps,
    "augmentation": resolve_lod_aug_config(args).to_dict(),
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
    echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} torchrun --nproc_per_node=1 --master_port=${port} finetune_stf/train.py"
    printf ' %q' "$@"
    printf ' --pretrained-from %q --heavy-save-root %q --save-path %q\n' "${PRETRAINED}" "${HEAVY_ROOT}" "${save}"
  } 2>&1 | tee -a "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
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
  --stage lod_only
  --lod-root "${LOD_ROOT}"
  --lod-manifest "${LOD_MANIFEST}"
  --lod-label-space inverse_relative
  --input-height 512
  --input-width 960
  --lod-train-crop-mode random
  --lod-val-crop-mode center
  --no-eval-stf
  --eval-lod
  --eval-lod-train-proxy
  --lod-train-proxy-count 112
  --best-metric lod_d1
  --save-best-checkpoint
  --bs 8
  --accum-steps 1
  --loss-type ssi
  --loss-target-normalization
  --loss-norm-min-scale 1e-3
  --epochs "${EPOCHS}"
  --lr-schedule poly
  --warmup-steps 0
  --aug-preset off
  --aug-hflip-prob 0.5
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
)

raw_rgb16_args=(
  --dataset-family lod_true_raw_dark_rgb16
  --dataset-input-mode raw_rgb16_dark
  --input-domain raw3
  --front-end raw_rgb16_ram3
  --model-input-tensor raw
  --raw-storage-format raw_rgb16_png_3ch
  --lod-raw-norm-mode uint16_div_65535
  --raw-front-end-lr 5e-5
  --raw-ram-rgb-tail identity
  --dav2-train-mode decoder
  --backbone-layer-decay 1.0
  --lr 1e-5
)

lora_common_args=(
  --lora dav2_lora
  --lora-block-mode tap
  --lora-tap-layers 2 5 8 11
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
)

lora_only_disabled_args=(
  --bridge none
  --decoder-feature-adapter none
)

bridge_feature_adapter_args=(
  --bridge raw_feature_bridge
  --decoder-feature-adapter raw_feature_adapter
  --bridge-feature-source-channels x3
  --adapter-feature-source-channels x3
  --bridge-feature-keys x_cat ffm_mid x3
  --feature-adapter-keys x_cat ffm_mid x3
  --bridge-layers 2 5 8 11
  --bridge-source ram_core
  --bridge-lr 5e-5
)

run_formal() {
  local ts run_lora run_bridge_fa

  ts="$(date +%m%d_%H%M)"
  run_lora="${ts}_${RUN_SUFFIX_LORA}"
  echo "[QUEUE][RUN] LOD RAW_RGB16 RamCore3 + LoRA tap -> ${run_lora}"
  run_train "LOD RAW_RGB16 RamCore3 + LoRA tap" "${run_lora}" "${MASTER_PORT}" \
    "${common_args[@]}" "${raw_rgb16_args[@]}" "${lora_only_disabled_args[@]}" "${lora_common_args[@]}"

  ts="$(date +%m%d_%H%M)"
  run_bridge_fa="${ts}_${RUN_SUFFIX_BRIDGE_FA}"
  echo "[QUEUE][RUN] LOD RAW_RGB16 RamCore3 + LoRA tap + bridge + feature adapter -> ${run_bridge_fa}"
  run_train "LOD RAW_RGB16 RamCore3 + LoRA tap + bridge + feature adapter" "${run_bridge_fa}" "${MASTER_PORT}" \
    "${common_args[@]}" "${raw_rgb16_args[@]}" "${lora_common_args[@]}" "${bridge_feature_adapter_args[@]}"
}

require_file "${PRETRAINED}"
require_file "${LOD_MANIFEST}"
require_dir "${LOD_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SCOPE] LOD RAW_RGB16 block8/excl10 e${EPOCHS}, flip0.5 only, LoRA-only then LoRA+bridge+feature-adapter"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[LOD_ROOT] ${LOD_ROOT}"
echo "[LOD_MANIFEST] ${LOD_MANIFEST}"
echo "[PORT] ${MASTER_PORT}"

run_formal

echo "[QUEUE_END] $(date -Iseconds) status=0"
