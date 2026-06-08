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
SESSION_PREFIX="${SESSION_PREFIX:-rod_night_raw_ram_v1_e5}"
R1_MASTER_PORT="${R1_MASTER_PORT:-29611}"
R2_MASTER_PORT="${R2_MASTER_PORT:-29612}"
RGB_MASTER_PORT="${RGB_MASTER_PORT:-29613}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh
  bash finetune_stf/scripts/formal/0605_run_rod_night_raw_ram_v1_e5_queue.sh --audit

Starts one tmux session that sequentially runs:
  R1: ROD raw4 -> RamCore3 identity -> DAv2-S decoder, epochs=5
  R2: ROD raw4 -> RamCore3 identity -> DAv2-S LoRA tap r8/a16 + decoder, epochs=5
  RGB-ref: ROD student RGB -> DAv2-S full backbone LLRD 0.9 with lr=1e-6, epochs=5

The --audit mode validates resolved configs without launching training.

Overrides:
  GPU=1 R1_MASTER_PORT=29621 R2_MASTER_PORT=29622 RGB_MASTER_PORT=29623 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' ROD_ROOT='${ROD_ROOT}' ROD_MANIFEST='${ROD_MANIFEST}' R1_MASTER_PORT='${R1_MASTER_PORT}' R2_MASTER_PORT='${R2_MASTER_PORT}' RGB_MASTER_PORT='${RGB_MASTER_PORT}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' PHASE1_BNCLEAN_REVIEWED=1 bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
    echo "[MANIFEST] ${ROD_MANIFEST}"
    echo "[SEMANTICS] label=teacher_bright_degreen_v1 inverse_relative, ROD raw-RAM v1, raw_ram_rgb_tail=identity for R1/R2"
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

from finetune_stf.train import parse_args

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
    "raw_front_end_lr": cfg["raw_front_end_lr"],
    "raw_ram_rgb_tail": args.raw_ram_rgb_tail,
    "lora": cfg["lora"],
    "lora_tap_layers": cfg["lora_tap_layers"],
    "lora_rank": cfg["lora_rank"],
    "lora_alpha": cfg["lora_alpha"],
    "lora_lr": cfg["lora_lr"],
    "dav2_train_mode": args.dav2_train_mode,
    "backbone_layer_decay": args.backbone_layer_decay,
    "lr": args.lr,
    "rod_label_space": args.rod_label_space,
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

rod_common_args=(
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
  --epochs 5
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
)

rod_raw_base_args=(
  --dataset-family rod_raw
  --dataset-input-mode raw_ram
  --input-domain raw4
  --front-end raw_to_base_rgb_ram3
  --model-input-tensor raw
  --raw-storage-format n_a
  --bridge none
  --decoder-feature-adapter none
  --lr 1e-5
  --raw-front-end-lr 5e-5
  --raw-ram-rgb-tail identity
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

rgb_ref_args=(
  --dataset-family rod_raw_student_rgb
  --dataset-input-mode raw24_student_rgb
  --input-domain rgb
  --front-end dav2_rgb
  --model-input-tensor image
  --raw-storage-format n_a
  --bridge none
  --decoder-feature-adapter none
  --lora none
  --dav2-train-mode full
  --backbone-layer-decay 0.9
  --lr 1e-6
  --rod-rgb-pipeline student_dark_degreen_v1
  --rod-student-white-percentile 99.9
  --rod-student-gamma 0.9
  --rod-student-channel-gains 1.08 0.95 1.10
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
  run_formal_exp "R1 raw RAM identity + decoder" "rod_night_rawram3_identity_dav2s_ram_decoder_e5" "${R1_MASTER_PORT}" \
    "${rod_common_args[@]}" "${rod_raw_base_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0

  run_formal_exp "R2 raw RAM identity + LoRA tap r8/a16 + decoder" "rod_night_rawram3_identity_dav2s_ram_lora_tap_r8a16_decoder_e5" "${R2_MASTER_PORT}" \
    "${rod_common_args[@]}" "${rod_raw_base_args[@]}" "${lora_args[@]}" \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0

  run_formal_exp "RGB-ref student RGB low-lr backbone LLRD 0.9" "rod_night_studentrgb_dav2s_backbone_lowlr_ld09_e5" "${RGB_MASTER_PORT}" \
    "${rod_common_args[@]}" "${rgb_ref_args[@]}"
}

require_file "${PRETRAINED}"
require_file "${ROD_MANIFEST}"
require_dir "${ROD_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SCOPE] R1 -> R2 -> RGB-ref ROD night raw-RAM v1, epochs=5"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[ROD_ROOT] ${ROD_ROOT}"
echo "[ROD_MANIFEST] ${ROD_MANIFEST}"
echo "[PORTS] R1=${R1_MASTER_PORT} R2=${R2_MASTER_PORT} RGB=${RGB_MASTER_PORT}"
echo "[PHASE1_BNCLEAN_REVIEWED] ${PHASE1_BNCLEAN_REVIEWED:-unset}"

run_all

echo "[QUEUE_END] $(date -Iseconds) status=0"
