#!/usr/bin/env bash
# One-off comparison: diagnose PriorDA fine-stage blur on STF dense pseudo depth.
#   A) coarse-only  : full-res KNN/global aligned coarse output (skip 518 fine ViT)
#   B) hires-fine   : keep fine stage but PRIORDA_FINE_HEIT=native (H//14*14)
# Same 10 deterministic preview samples as debug_priorda_preview10_*, so panels
# are directly comparable against the original 518-fine preview.
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-0}"
OUT_PARENT="${OUT_PARENT:-/mnt/drive/3333_raw/seeing_through_fog}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
MANIFEST="${MANIFEST:-${OUT_PARENT}/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv}"
SCRIPT="${SCRIPT:-${ROOT}/finetune_stf/scripts/generate_stf_priorda_dense_pseudo_depth.py}"

TS="$(date +%m%d_%H%M)"
SESSION="priorda_blurcmp_${TS}"
LOG_PATH="${LOG_ROOT}/${TS}_priorda_blurcmp.log"
COARSE_OUT="${OUT_PARENT}/debug_priorda_preview10_coarse_${TS}"
HIRES_OUT="${OUT_PARENT}/debug_priorda_preview10_hiresfine_${TS}"

mkdir -p "${LOG_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[ERROR] refusing to reuse existing tmux session: ${SESSION}" >&2
  exit 2
fi

read -r -d '' INNER <<EOF || true
set -euo pipefail
echo "[START] \$(date -Iseconds)"
echo "[HOST] \$(hostname)  [USER] \$(whoami)  [GPU] ${GPU}"

echo "[A coarse-only] -> ${COARSE_OUT}"
CUDA_VISIBLE_DEVICES=${GPU} conda run --live-stream -n ${CONDA_ENV} \
  python "${SCRIPT}" \
    --manifest "${MANIFEST}" \
    --output-root "${COARSE_OUT}" \
    --max-samples 10 \
    --preview-contact-sheet \
    --device cuda:0 \
    --coarse-only

echo "[B hires-fine native] -> ${HIRES_OUT}"
PRIORDA_FINE_HEIT=native CUDA_VISIBLE_DEVICES=${GPU} conda run --live-stream -n ${CONDA_ENV} \
  python "${SCRIPT}" \
    --manifest "${MANIFEST}" \
    --output-root "${HIRES_OUT}" \
    --max-samples 10 \
    --preview-contact-sheet \
    --device cuda:0

echo "[ALL DONE] \$(date -Iseconds)"
EOF

cd "${ROOT}"
tmux new-session -d -s "${SESSION}" "bash -lc '${INNER}' 2>&1 | tee '${LOG_PATH}'"

cat <<EOF
tmux session: ${SESSION}
log path: ${LOG_PATH}
attach:  tmux attach -t ${SESSION}
monitor: tail -f ${LOG_PATH}

A coarse-only  contact sheet: ${COARSE_OUT}/preview10_contact_sheet.png
B hires-fine   contact sheet: ${HIRES_OUT}/preview10_contact_sheet.png
EOF
