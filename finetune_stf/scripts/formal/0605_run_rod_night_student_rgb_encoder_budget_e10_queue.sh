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
SESSION_PREFIX="${SESSION_PREFIX:-rod_night_studentrgb_encoder_budget_e10}"
B_MASTER_PORT="${B_MASTER_PORT:-29605}"
C_MASTER_PORT="${C_MASTER_PORT:-29606}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh
  bash finetune_stf/scripts/formal/0605_run_rod_night_student_rgb_encoder_budget_e10_queue.sh --audit

Starts one tmux session that sequentially runs:
  B: ROD student RGB LoRA tap r8/a16 + decoder
  C: ROD student RGB full backbone with layer decay 0.9 + decoder

The --audit mode validates the resolved configs without launching training.

Overrides:
  GPU=1 B_MASTER_PORT=29615 C_MASTER_PORT=29616 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' ROD_ROOT='${ROD_ROOT}' ROD_MANIFEST='${ROD_MANIFEST}' B_MASTER_PORT='${B_MASTER_PORT}' C_MASTER_PORT='${C_MASTER_PORT}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
    echo "[SEMANTICS] teacher=teacher_bright_degreen_v1 label_space=inverse_relative student=student_dark_degreen_v1"
  } 2>&1 | tee -a "${log}"
}

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_audit_${run}"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "$@" \
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
    "front_end": cfg["front_end"],
    "model_input_tensor": cfg["model_input_tensor"],
    "lora": cfg["lora"],
    "lora_tap_layers": cfg["lora_tap_layers"],
    "lora_rank": cfg["lora_rank"],
    "lora_alpha": cfg["lora_alpha"],
    "lora_lr": cfg["lora_lr"],
    "dav2_train_mode": args.dav2_train_mode,
    "backbone_layer_decay": args.backbone_layer_decay,
    "rod_label_space": args.rod_label_space,
    "epochs": args.epochs,
    "save_path": args.save_path,
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
  --stage rod_only
  --input-domain rgb
  --front-end dav2_rgb
  --model-input-tensor image
  --dataset-family rod_raw_student_rgb
  --dataset-input-mode raw24_student_rgb
  --raw-storage-format n_a
  --bridge none
  --decoder-feature-adapter none
  --rod-root "${ROD_ROOT}"
  --rod-night-manifest "${ROD_MANIFEST}"
  --rod-raw-source raw24
  --rod-rgb-pipeline student_dark_degreen_v1
  --rod-student-white-percentile 99.9
  --rod-student-gamma 0.9
  --rod-student-channel-gains 1.08 0.95 1.10
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
  --lr 1e-5
  --loss-type ssi
  --loss-target-normalization
  --loss-norm-min-scale 1e-3
  --epochs 10
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
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
  run_formal_exp "B LoRA tap r8/a16 + decoder" "rod_night_studentrgb_dav2s_lora_tap_r8a16_decoder_e10" "${B_MASTER_PORT}" \
    "${common_args[@]}" "${lora_args[@]}" \
    --dav2-train-mode decoder \
    --backbone-layer-decay 1.0

  run_formal_exp "C full backbone LLRD 0.9 + decoder" "rod_night_studentrgb_dav2s_backbone_ld09_decoder_e10" "${C_MASTER_PORT}" \
    "${common_args[@]}" "${no_lora_args[@]}" \
    --dav2-train-mode full \
    --backbone-layer-decay 0.9
}

require_file "${PRETRAINED}"
require_file "${ROD_MANIFEST}"
require_dir "${ROD_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SCOPE] B then C ROD night student RGB encoder adaptation budget, epochs=10"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[ROD_ROOT] ${ROD_ROOT}"
echo "[ROD_MANIFEST] ${ROD_MANIFEST}"
echo "[PORTS] B=${B_MASTER_PORT} C=${C_MASTER_PORT}"

run_all

echo "[QUEUE_END] $(date -Iseconds) status=0"
