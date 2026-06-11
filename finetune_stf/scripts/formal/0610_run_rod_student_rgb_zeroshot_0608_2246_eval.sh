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
CKPT="${CKPT:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_2246_lod_true_rgb_dark_block8excl10_dav2s_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
ROD_ROOT="${ROD_ROOT:-/home/caq/6666_raw/0000_dataset/ROD}"
ROD_MANIFEST="${ROD_MANIFEST:-${ROD_ROOT}/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-rod_studentrgb_zeroshot_0608_2246}"
MASTER_PORT="${MASTER_PORT:-29611}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0610_run_rod_student_rgb_zeroshot_0608_2246_eval.sh
  bash finetune_stf/scripts/formal/0610_run_rod_student_rgb_zeroshot_0608_2246_eval.sh --audit

Starts one tmux session for zero-shot eval:
  checkpoint: 0608_2246 LOD true RGB_dark DAv2-S LoRA tap r8/a16 decoder best_model
  eval input: ROD night student RGB rendered from RAW24
  eval target: ROD night pseudo label from the ROD manifest

Overrides:
  GPU=1 MASTER_PORT=29620 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' CKPT='${CKPT}' ROD_ROOT='${ROD_ROOT}' ROD_MANIFEST='${ROD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
  local run="$1"
  local port="$2"
  local log="$3"
  {
    echo "[START] $(date -Iseconds)"
    echo "[RUN] ${run}"
    echo "[SESSION] ${QUEUE_SESSION:-internal}"
    echo "[HOST] $(hostname)"
    echo "[USER] $(whoami)"
    echo "[PWD] $(pwd)"
    echo "[GPU] ${GPU}"
    echo "[PORT] ${port}"
    echo "[LOG] ${log}"
    echo "[CKPT] ${CKPT}"
    echo "[MANIFEST] ${ROD_MANIFEST}"
    echo "[SEMANTICS] zero_shot=true source_ckpt=0608_2246_lod_true_rgb_dark_lora eval_input=rod_student_dark_degreen_v1 gt=rod_pseudo_depth_dav2l_teacherbright split=01Valid crop=center"
  } 2>&1 | tee -a "${log}"
}

common_args=(
  --encoder vits
  --stage eval_only
  --eval-only
  --input-domain rgb
  --front-end dav2_rgb
  --model-input-tensor image
  --dataset-family rod_raw_student_rgb
  --dataset-input-mode raw24_student_rgb
  --raw-storage-format n_a
  --bridge none
  --decoder-feature-adapter none
  --lora dav2_lora
  --lora-block-mode tap
  --lora-tap-layers 2 5 8 11
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
  --dav2-train-mode decoder
  --backbone-layer-decay 1.0
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
  --bs 8
  --accum-steps 1
  --lr 1e-5
  --loss-type ssi
  --loss-target-normalization
  --loss-norm-min-scale 1e-3
  --epochs 0
  --amp
  --amp-dtype bf16
  --seed 42
  --num-workers 4
  --log-interval 500
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
)

audit_train_args() {
  local audit_save="/tmp/codex_smoke_audit_rod_studentrgb_zeroshot_0608_2246"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "${common_args[@]}" \
    --pretrained-from "${PRETRAINED}" \
    --resume-from "${CKPT}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${audit_save}" <<'PY'
import json

from finetune_stf.train import parse_args

args = parse_args()
cfg = args.resolved_config.to_dict()
payload = {
    "stage": args.stage,
    "eval_only": args.eval_only,
    "dataset_family": cfg["dataset_family"],
    "dataset_input_mode": cfg["dataset_input_mode"],
    "front_end": cfg["front_end"],
    "model_input_tensor": cfg["model_input_tensor"],
    "lora": cfg["lora"],
    "lora_tap_layers": cfg["lora_tap_layers"],
    "lora_rank": cfg["lora_rank"],
    "lora_alpha": cfg["lora_alpha"],
    "dav2_train_mode": args.dav2_train_mode,
    "rod_rgb_pipeline": args.rod_rgb_pipeline,
    "rod_label_space": args.rod_label_space,
    "resume_from": args.resume_from,
    "pretrained_from": args.pretrained_from,
    "save_path": args.save_path,
}
print("[AUDIT][OK] " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

extract_summary() {
  local run="$1"
  local save="$2"
  local log="$3"
  local summary="${save}/eval_only_rod_studentrgb_zeroshot_summary.json"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "${save}/train.log" "${summary}" "${run}" "${CKPT}" "${ROD_MANIFEST}" <<'PY'
import ast
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
run = sys.argv[3]
checkpoint = sys.argv[4]
manifest = sys.argv[5]

target = "[EVAL][eval_only_rod_night_val] summary="
summary = None
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if target in line:
        summary_text = re.sub(r"\bnan\b", "None", line.split(target, 1)[1].strip())
        summary = ast.literal_eval(summary_text)

if summary is None:
    raise SystemExit(f"missing eval summary line in {log_path}")

summary = {
    key: (None if isinstance(value, float) and not math.isfinite(value) else value)
    for key, value in summary.items()
}
payload = {
    "run": run,
    "checkpoint": checkpoint,
    "rod_manifest": manifest,
    "split": "01Valid",
    "eval_input": "rod_raw_student_rgb/student_dark_degreen_v1",
    "target": "rod_pseudo_depth_dav2l_teacherbright_inverse_relative",
    "metrics": summary,
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[SUMMARY] {summary_path}")
print(json.dumps(payload, sort_keys=True), flush=True)
PY
}

run_eval() {
  local run="$1"
  local port="$2"
  local save="${EXP_ROOT}/${run}"
  local heavy="${HEAVY_ROOT}/${run}"
  local log="${LOG_ROOT}/${run}.tmux.log"

  if [[ "${AUDIT_ONLY}" == "1" ]]; then
    audit_train_args
    return 0
  fi

  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] refusing to overwrite existing artifacts for ${run}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi

  mkdir -p "${save}"
  log_header "${run}" "${port}" "${log}"

  {
    echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} torchrun --nproc_per_node=1 --master_port=${port} finetune_stf/train.py"
    printf ' %q' "${common_args[@]}"
    printf ' --pretrained-from %q --resume-from %q --heavy-save-root %q --save-path %q\n' "${PRETRAINED}" "${CKPT}" "${HEAVY_ROOT}" "${save}"
  } 2>&1 | tee -a "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    "${common_args[@]}" \
    --pretrained-from "${PRETRAINED}" \
    --resume-from "${CKPT}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" == "0" ]]; then
    extract_summary "${run}" "${save}" "${log}" 2>&1 | tee -a "${log}"
  fi

  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

require_file "${PRETRAINED}"
require_file "${CKPT}"
require_file "${ROD_MANIFEST}"
require_dir "${ROD_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SCOPE] ROD student RGB zero-shot eval from 0608_2246 LOD RGB_dark checkpoint"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[CKPT] ${CKPT}"
echo "[ROD_ROOT] ${ROD_ROOT}"
echo "[ROD_MANIFEST] ${ROD_MANIFEST}"
echo "[PORT] ${MASTER_PORT}"

run_ts="$(date +%m%d_%H%M)"
run="${run_ts}_rod_studentrgb_zeroshot_from_0608_2246_lodrgb_lora_eval"
run_eval "${run}" "${MASTER_PORT}"

echo "[QUEUE_END] $(date -Iseconds) status=0"
