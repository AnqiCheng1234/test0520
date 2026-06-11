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
CKPT_0608_2026="${CKPT_0608_2026:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
CKPT_0609_0044="${CKPT_0609_0044:-/mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
ROD_ROOT="${ROD_ROOT:-/home/caq/6666_raw/0000_dataset/ROD}"
ROD_MANIFEST="${ROD_MANIFEST:-${ROD_ROOT}/pseudo_depth_dav2l_night_teacherbright_rel_1440x928/rod_night_dav2_rel_manifest.csv}"
SESSION_PREFIX="${SESSION_PREFIX:-rod_raw_rgb3_zeroshot_0608_2026_0609_0044}"
MASTER_PORT="${MASTER_PORT:-29612}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0610_run_rod_raw_rgb3_zeroshot_0608_2026_0609_0044_eval.sh
  bash finetune_stf/scripts/formal/0610_run_rod_raw_rgb3_zeroshot_0608_2026_0609_0044_eval.sh --audit

Starts one tmux session for two zero-shot evals:
  eval input: ROD night RAW24 packed Bayer, converted explicitly to raw3 [R,(Gr+Gb)/2,B]
  eval target: ROD night pseudo label from the ROD manifest
  checkpoints:
    0608_2026 LOD RAW_dark RGB16 RamCore3 LoRA best_model
    0609_0044 LOD RAW_normal RGB16 RamCore3 LoRA best_model

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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' CKPT_0608_2026='${CKPT_0608_2026}' CKPT_0609_0044='${CKPT_0609_0044}' ROD_ROOT='${ROD_ROOT}' ROD_MANIFEST='${ROD_MANIFEST}' MASTER_PORT='${MASTER_PORT}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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

common_args=(
  --encoder vits
  --stage eval_only
  --eval-only
  --input-domain raw3
  --front-end raw_rgb16_ram3
  --model-input-tensor raw
  --dataset-family rod_raw_rgb3
  --dataset-input-mode raw24_base_rgb3
  --raw-storage-format n_a
  --bridge none
  --decoder-feature-adapter none
  --lora dav2_lora
  --lora-block-mode tap
  --lora-tap-layers 2 5 8 11
  --lora-rank 8
  --lora-alpha 16
  --lora-lr 5e-5
  --raw-front-end-lr 5e-5
  --raw-ram-rgb-tail identity
  --dav2-train-mode decoder
  --backbone-layer-decay 1.0
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
  local label="$1"
  local ckpt="$2"
  local audit_save="/tmp/codex_smoke_audit_${label}"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "${common_args[@]}" \
    --pretrained-from "${PRETRAINED}" \
    --resume-from "${ckpt}" \
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
    "rod_label_space": args.rod_label_space,
    "resume_from": args.resume_from,
    "save_path": args.save_path,
}
print("[AUDIT][OK] " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

log_header() {
  local label="$1"
  local run="$2"
  local ckpt="$3"
  local port="$4"
  local log="$5"
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
    echo "[CKPT] ${ckpt}"
    echo "[MANIFEST] ${ROD_MANIFEST}"
    echo "[SEMANTICS] zero_shot=true eval_input=rod_raw_rgb3_from_raw24_base_rgb target=rod_pseudo_depth_dav2l_teacherbright split=01Valid crop=center raw_view=[R,(Gr+Gb)/2,B]"
  } 2>&1 | tee -a "${log}"
}

extract_summary() {
  local label="$1"
  local run="$2"
  local save="$3"
  local ckpt="$4"
  local summary="${save}/eval_only_rod_raw_rgb3_zeroshot_summary.json"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "${save}/train.log" "${summary}" "${label}" "${run}" "${ckpt}" "${ROD_MANIFEST}" <<'PY'
import ast
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
label = sys.argv[3]
run = sys.argv[4]
checkpoint = sys.argv[5]
manifest = sys.argv[6]

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
    "label": label,
    "run": run,
    "checkpoint": checkpoint,
    "rod_manifest": manifest,
    "split": "01Valid",
    "eval_input": "rod_raw_rgb3/raw24_base_rgb3/[R,(Gr+Gb)/2,B]",
    "target": "rod_pseudo_depth_dav2l_teacherbright_inverse_relative",
    "metrics": summary,
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[SUMMARY] {summary_path}")
print(json.dumps(payload, sort_keys=True), flush=True)
PY
}

run_eval() {
  local label="$1"
  local suffix="$2"
  local ckpt="$3"
  local port="$4"

  if [[ "${AUDIT_ONLY}" == "1" ]]; then
    audit_train_args "${label}" "${ckpt}"
    return 0
  fi

  local ts run save heavy log
  ts="$(date +%m%d_%H%M)"
  run="${ts}_${suffix}"
  save="${EXP_ROOT}/${run}"
  heavy="${HEAVY_ROOT}/${run}"
  log="${LOG_ROOT}/${run}.tmux.log"

  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] refusing to overwrite existing artifacts for ${run}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi

  mkdir -p "${save}"
  log_header "${label}" "${run}" "${ckpt}" "${port}" "${log}"

  {
    echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} torchrun --nproc_per_node=1 --master_port=${port} finetune_stf/train.py"
    printf ' %q' "${common_args[@]}"
    printf ' --pretrained-from %q --resume-from %q --heavy-save-root %q --save-path %q\n' "${PRETRAINED}" "${ckpt}" "${HEAVY_ROOT}" "${save}"
  } 2>&1 | tee -a "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    "${common_args[@]}" \
    --pretrained-from "${PRETRAINED}" \
    --resume-from "${ckpt}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" == "0" ]]; then
    extract_summary "${label}" "${run}" "${save}" "${ckpt}" 2>&1 | tee -a "${log}"
  fi

  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

require_file "${PRETRAINED}"
require_file "${CKPT_0608_2026}"
require_file "${CKPT_0609_0044}"
require_file "${ROD_MANIFEST}"
require_dir "${ROD_ROOT}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[MODE] AUDIT_ONLY=${AUDIT_ONLY}"
echo "[SCOPE] ROD raw3 zero-shot eval from 0608_2026 and 0609_0044 LOD RAW checkpoints"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[CKPT_0608_2026] ${CKPT_0608_2026}"
echo "[CKPT_0609_0044] ${CKPT_0609_0044}"
echo "[ROD_ROOT] ${ROD_ROOT}"
echo "[ROD_MANIFEST] ${ROD_MANIFEST}"
echo "[PORT] ${MASTER_PORT}"

run_eval "0608_2026_lod_raw_dark_rgb16_lora_best" \
  "rod_raw_rgb3_zeroshot_from_0608_2026_lod_raw_dark_lora_eval" \
  "${CKPT_0608_2026}" "${MASTER_PORT}"

run_eval "0609_0044_lod_raw_normal_rgb16_lora_best" \
  "rod_raw_rgb3_zeroshot_from_0609_0044_lod_raw_normal_lora_eval" \
  "${CKPT_0609_0044}" "$((MASTER_PORT + 1))"

echo "[QUEUE_END] $(date -Iseconds) status=0"
