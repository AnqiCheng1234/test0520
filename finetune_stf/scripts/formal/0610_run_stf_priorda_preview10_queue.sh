#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
OUT_PARENT="${OUT_PARENT:-/mnt/drive/3333_raw/seeing_through_fog}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
MANIFEST="${MANIFEST:-${OUT_PARENT}/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"
SCRIPT="${SCRIPT:-${ROOT}/finetune_stf/scripts/generate_stf_priorda_dense_pseudo_depth.py}"
SESSION_PREFIX="${SESSION_PREFIX:-priorda_stf_preview10}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0610_run_stf_priorda_preview10_queue.sh

Overrides:
  GPU=1 OUT_PARENT=/mnt/drive/3333_raw/seeing_through_fog bash ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--run-internal" ]]; then
  ts="$(date +%m%d_%H%M)"
  session="${SESSION_PREFIX}_${ts}"
  out_root="${OUT_PARENT}/debug_priorda_preview10_${ts}"
  log_path="${LOG_ROOT}/${ts}_priorda_stf_preview10.log"
  mkdir -p "${LOG_ROOT}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "ROOT='${ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' OUT_ROOT='${out_root}' LOG_PATH='${log_path}' MANIFEST='${MANIFEST}' SCRIPT='${SCRIPT}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal"
  cat <<EOF
tmux session: ${session}
log path: ${log_path}
attach: tmux attach -t ${session}
monitor: tail -f ${log_path}
preview output: ${out_root}
preview contact sheet: ${out_root}/preview10_contact_sheet.png
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "$(dirname "${LOG_PATH}")"
{
  echo "[START] $(date -Iseconds)"
  echo "[HOST] $(hostname)"
  echo "[USER] $(whoami)"
  echo "[PWD] $(pwd)"
  echo "[GPU] ${GPU}"
  echo "[CONDA_ENV] ${CONDA_ENV}"
  echo "[MANIFEST] ${MANIFEST}"
  echo "[OUT_ROOT] ${OUT_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python "${SCRIPT}" \
      --manifest "${MANIFEST}" \
      --output-root "${OUT_ROOT}" \
      --max-samples 10 \
      --preview-contact-sheet \
      --device cuda:0
  echo "[END] $(date -Iseconds)"
} 2>&1 | tee -a "${LOG_PATH}"
