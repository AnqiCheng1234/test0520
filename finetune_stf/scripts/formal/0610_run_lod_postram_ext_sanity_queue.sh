#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${ROOT}/finetune_stf/analysis/lod_postram_external}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
C_DARK_RUN="${C_DARK_RUN:-${ROOT}/finetune_stf/exp/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly}"
C_DARK_CKPT="${C_DARK_CKPT:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
LOD_ROOT="${LOD_ROOT:-/home/caq/6666_raw/0000_dataset/LOD}"
LOD_MANIFEST="${LOD_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_postram_ext_sanity_queue}"
RUN_PREFIX="${RUN_PREFIX:-}"
EXT_SET="${EXT_SET:-sanity}"
DRUNET_TS="${DRUNET_TS:-}"
RESTORMER_TS="${RESTORMER_TS:-}"
NAFNET_TS="${NAFNET_TS:-}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0610_run_lod_postram_ext_sanity_queue.sh
  bash finetune_stf/scripts/formal/0610_run_lod_postram_ext_sanity_queue.sh --audit

Default EXT_SET=sanity runs:
  collect_stats, EXT_NOOP_A0, EXT_ID_Q001, visualizations, summary

Optional:
  EXT_SET=minimal DRUNET_TS=/path/drunet.ts bash ...
  RESTORMER_TS and NAFNET_TS are only used when supplied.
EOF
}

AUDIT_ONLY=0
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
    "cd '${ROOT}' && ROOT='${ROOT}' LOG_ROOT='${LOG_ROOT}' ANALYSIS_ROOT='${ANALYSIS_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' C_DARK_RUN='${C_DARK_RUN}' C_DARK_CKPT='${C_DARK_CKPT}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' PRETRAINED='${PRETRAINED}' EXT_SET='${EXT_SET}' DRUNET_TS='${DRUNET_TS}' RESTORMER_TS='${RESTORMER_TS}' NAFNET_TS='${NAFNET_TS}' RUN_PREFIX='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
RUN_PREFIX="${RUN_PREFIX:-$(date +%m%d_%H%M)}"
OUT_DIR="${ANALYSIS_ROOT}/${RUN_PREFIX}_lod_postram_ext_${EXT_SET}"
STATS_JSON="${OUT_DIR}/xram_stats_00Train.json"

require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 2; }; }
require_file "${PRETRAINED}"
require_file "${LOD_MANIFEST}"
require_file "${C_DARK_CKPT}"
require_dir "${LOD_ROOT}"
require_dir "${C_DARK_RUN}"
mkdir -p "${OUT_DIR}" "${LOG_ROOT}"

echo "[START] $(date -Iseconds)"
echo "[SESSION] ${QUEUE_SESSION:-internal}"
echo "[HOST] $(hostname)"
echo "[USER] $(whoami)"
echo "[PWD] $(pwd)"
echo "[OUT_DIR] ${OUT_DIR}"
echo "[EXT_SET] ${EXT_SET}"
echo "[C_DARK_RUN] ${C_DARK_RUN}"
echo "[C_DARK_CKPT] ${C_DARK_CKPT}"

if [[ "${AUDIT_ONLY}" == "1" ]]; then
  echo "[AUDIT][OK] would write ${OUT_DIR}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
  python tools/lod_postram_collect_stats.py \
    --run-dir "${C_DARK_RUN}" \
    --checkpoint "${C_DARK_CKPT}" \
    --output-json "${STATS_JSON}" \
    --lod-root "${LOD_ROOT}" \
    --lod-manifest "${LOD_MANIFEST}" \
    --pretrained-from "${PRETRAINED}" \
    --split 00Train \
    --crop-mode center \
    --batch-size 4 \
    --num-workers 4 \
    --progress-interval 25

run_ext() {
  local ext_id="$1"
  shift
  local ext_dir="${OUT_DIR}/${ext_id}"
  echo "[EXT] start ${ext_id}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python tools/lod_postram_external_eval.py \
      --run-dir "${C_DARK_RUN}" \
      --checkpoint "${C_DARK_CKPT}" \
      --stats-json "${STATS_JSON}" \
      --output-dir "${ext_dir}" \
      --ext-id "${ext_id}" \
      --lod-root "${LOD_ROOT}" \
      --lod-manifest "${LOD_MANIFEST}" \
      --pretrained-from "${PRETRAINED}" \
      --eval-split 01Valid \
      --eval-batch-size 1 \
      --num-workers 4 \
      --artifact-max-samples 16 \
      --progress-interval 25 \
      "$@"
  local affine="n_a"
  local prev=""
  for arg in "$@"; do
    if [[ "${prev}" == "--affine" ]]; then
      affine="${arg}"
    fi
    prev="${arg}"
  done
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python tools/lod_postram_visualize.py \
      --eval-dir "${ext_dir}" \
      --output-dir "${ext_dir}/viz/${ext_id}" \
      --stats-json "${STATS_JSON}" \
      --affine "${affine}" \
      --max-samples 16
  echo "[EXT] done ${ext_id}"
}

run_ext EXT_NOOP_A0 --denoiser none --alpha 0 --affine n_a
run_ext EXT_ID_Q001 --denoiser identity --alpha 1.0 --affine q001q999

EXTRA_DIRS=()
if [[ "${EXT_SET}" == "minimal" || "${EXT_SET}" == "full" ]]; then
  if [[ -n "${DRUNET_TS}" ]]; then
    require_file "${DRUNET_TS}"
    run_ext EXT_D1 --denoiser drunet --sigma 10 --alpha 0.25 --affine q001q999 --denoiser-torchscript "${DRUNET_TS}" --denoiser-concat-sigma-channel
    run_ext EXT_D3 --denoiser drunet --sigma 25 --alpha 0.25 --affine q001q999 --denoiser-torchscript "${DRUNET_TS}" --denoiser-concat-sigma-channel
    run_ext EXT_D4 --denoiser drunet --sigma 25 --alpha 0.5 --affine q001q999 --denoiser-torchscript "${DRUNET_TS}" --denoiser-concat-sigma-channel
  else
    echo "[WARN] EXT_SET=${EXT_SET} requested DRUNet cells but DRUNET_TS is empty; skipping EXT_D*"
  fi
  if [[ -n "${RESTORMER_TS}" ]]; then
    require_file "${RESTORMER_TS}"
    run_ext EXT_R1 --denoiser restormer --alpha 0.25 --affine q001q999 --denoiser-torchscript "${RESTORMER_TS}"
  fi
  if [[ -n "${NAFNET_TS}" ]]; then
    require_file "${NAFNET_TS}"
    run_ext EXT_N1 --denoiser nafnet --alpha 0.25 --affine q001q999 --denoiser-torchscript "${NAFNET_TS}"
  fi
fi

mapfile -t EXT_DIRS < <(find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'EXT_*' | sort)
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
  python tools/lod_postram_ext_summary.py \
    --eval-dirs "${EXT_DIRS[@]}" \
    --output-csv "${OUT_DIR}/summary.csv" \
    --output-md "${OUT_DIR}/summary.md"

echo "[DONE] $(date -Iseconds) output=${OUT_DIR}"
