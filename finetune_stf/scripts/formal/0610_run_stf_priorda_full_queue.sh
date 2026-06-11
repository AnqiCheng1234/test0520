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
SESSION_PREFIX="${SESSION_PREFIX:-priorda_stf}"
PREVIEW_APPROVED="${PREVIEW_APPROVED:-}"
FINE_HEIT="${FINE_HEIT:-native}"

usage() {
  cat <<'EOF'
Usage:
  PREVIEW_APPROVED=/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_MMDD_HHMM \
    bash finetune_stf/scripts/formal/0610_run_stf_priorda_full_queue.sh

Overrides:
  GPU=1 OUT_PARENT=/mnt/drive/3333_raw/seeing_through_fog bash ...
  FINE_HEIT=native bash ...   # default hires-fine
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${PREVIEW_APPROVED}" ]]; then
  echo "[ERROR] PREVIEW_APPROVED is required for full 6216 generation." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${PREVIEW_APPROVED}/preview10_summary.json" || ! -f "${PREVIEW_APPROVED}/preview10_contact_sheet.png" ]]; then
  echo "[ERROR] PREVIEW_APPROVED does not look like a completed preview output: ${PREVIEW_APPROVED}" >&2
  exit 2
fi

if [[ "${1:-}" != "--run-internal" ]]; then
  ts="$(date +%m%d_%H%M)"
  session="${SESSION_PREFIX}_${ts}"
  out_root="${OUT_PARENT}/${ts}_pseudo_depth_priorda_hiresfine_dav2l_geo_lidar_last_6216"
  log_path="${LOG_ROOT}/${ts}_priorda_stf_generate.log"
  mkdir -p "${LOG_ROOT}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "ROOT='${ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' OUT_ROOT='${out_root}' LOG_PATH='${log_path}' MANIFEST='${MANIFEST}' SCRIPT='${SCRIPT}' PREVIEW_APPROVED='${PREVIEW_APPROVED}' FINE_HEIT='${FINE_HEIT}' CUDA_VISIBLE_DEVICES='${GPU}' PRIORDA_FINE_HEIT='${FINE_HEIT}' bash '$0' --run-internal"
  cat <<EOF
tmux session: ${session}
log path: ${log_path}
attach: tmux attach -t ${session}
monitor: tail -f ${log_path}
output root: ${out_root}
preview approved: ${PREVIEW_APPROVED}
fine heit: PRIORDA_FINE_HEIT=${FINE_HEIT}
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
  echo "[PREVIEW_APPROVED] ${PREVIEW_APPROVED}"
  echo "[PRIORDA_FINE_HEIT] ${FINE_HEIT}"
  PRIORDA_FINE_HEIT="${FINE_HEIT}" CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python "${SCRIPT}" \
      --manifest "${MANIFEST}" \
      --output-root "${OUT_ROOT}" \
      --preview-approved "${PREVIEW_APPROVED}" \
      --device cuda:0
  echo "[END] $(date -Iseconds)"
} 2>&1 | tee -a "${LOG_PATH}"
