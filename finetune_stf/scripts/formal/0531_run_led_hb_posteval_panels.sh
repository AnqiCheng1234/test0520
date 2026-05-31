#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0522}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-led_hb_posteval}"

if [[ "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  ts="$(date +%m%d_%H%M)"
  session="${ts}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' LOG_ROOT='${LOG_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' C2_RUN_DIR='${C2_RUN_DIR:-}' C2_CHECKPOINT='${C2_CHECKPOINT:-}' N5_RUN_DIR='${N5_RUN_DIR:-}' N5_CHECKPOINT='${N5_CHECKPOINT:-}' N3_RUN_DIR='${N3_RUN_DIR:-}' N3_CHECKPOINT='${N3_CHECKPOINT:-}' N2_RUN_DIR='${N2_RUN_DIR:-}' N2_CHECKPOINT='${N2_CHECKPOINT:-}' N7_RUN_DIR='${N7_RUN_DIR:-}' N7_CHECKPOINT='${N7_CHECKPOINT:-}' POST_TS='${ts}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
  cat <<EOF
tmux session: ${session}
queue log: ${queue_log}
attach: tmux attach -t ${session}
monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
require_file() { [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[ERROR] missing dir: $1" >&2; exit 2; }; }
POST_TS="${POST_TS:-$(date +%m%d_%H%M)}"
OUT_ROOT="${ROOT}/plans/0531_led_night/diagnostics/${POST_TS}_led_hb_posteval"
mkdir -p "${OUT_ROOT}"

for var in C2_RUN_DIR C2_CHECKPOINT N5_RUN_DIR N5_CHECKPOINT N3_RUN_DIR N3_CHECKPOINT N2_RUN_DIR N2_CHECKPOINT N7_RUN_DIR N7_CHECKPOINT; do
  [[ -n "${!var:-}" ]] || { echo "[ERROR] ${var} is required" >&2; exit 2; }
done
require_dir "${C2_RUN_DIR}"; require_file "${C2_CHECKPOINT}"
require_dir "${N5_RUN_DIR}"; require_file "${N5_CHECKPOINT}"
require_dir "${N3_RUN_DIR}"; require_file "${N3_CHECKPOINT}"
require_dir "${N2_RUN_DIR}"; require_file "${N2_CHECKPOINT}"
require_dir "${N7_RUN_DIR}"; require_file "${N7_CHECKPOINT}"

eval_run() {
  local kind="$1" run_dir="$2" ckpt="$3" max_depth="$4" name="$5"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_led_hb_formal.py \
    --run-kind "${kind}" --run-dir "${run_dir}" --checkpoint "${ckpt}" --max-depth "${max_depth}" \
    --output-dir "${OUT_ROOT}/${name}_max${max_depth/./}" --device cuda
}

for depth in 200.0 80.0; do
  eval_run c2 "${C2_RUN_DIR}" "${C2_CHECKPOINT}" "${depth}" c2
  eval_run nseries "${N5_RUN_DIR}" "${N5_CHECKPOINT}" "${depth}" n5
  eval_run nseries "${N3_RUN_DIR}" "${N3_CHECKPOINT}" "${depth}" n3
  eval_run nseries "${N2_RUN_DIR}" "${N2_CHECKPOINT}" "${depth}" n2
  eval_run nseries "${N7_RUN_DIR}" "${N7_CHECKPOINT}" "${depth}" n7
done

for run in n2 "${N2_RUN_DIR}" "${N2_CHECKPOINT}" n7 "${N7_RUN_DIR}" "${N7_CHECKPOINT}"; do :; done
for method in n2 n7; do
  if [[ "${method}" == "n2" ]]; then run_dir="${N2_RUN_DIR}"; ckpt="${N2_CHECKPOINT}"; else run_dir="${N7_RUN_DIR}"; ckpt="${N7_CHECKPOINT}"; fi
  for mode in true zero mean shuffle; do
    CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_led_hb_formal.py \
      --run-kind nseries --run-dir "${run_dir}" --checkpoint "${ckpt}" --max-depth 200.0 \
      --feature-ablation-mode "${mode}" --feature-ablation-scope both --feature-ablation-key x3 --feature-ablation-donor-offset 1 \
      --output-dir "${OUT_ROOT}/${method}_x3_${mode}_max200" --device cuda
  done
  for mode in true shuffle; do
    CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/eval_led_hb_formal.py \
      --run-kind nseries --run-dir "${run_dir}" --checkpoint "${ckpt}" --max-depth 80.0 \
      --feature-ablation-mode "${mode}" --feature-ablation-scope both --feature-ablation-key x3 --feature-ablation-donor-offset 1 \
      --output-dir "${OUT_ROOT}/${method}_x3_${mode}_max80" --device cuda
  done
done

for method in c2 n5 n3 n2 n7; do
  case "${method}" in
    c2) kind=c2; run_dir="${C2_RUN_DIR}"; ckpt="${C2_CHECKPOINT}" ;;
    n5) kind=nseries; run_dir="${N5_RUN_DIR}"; ckpt="${N5_CHECKPOINT}" ;;
    n3) kind=nseries; run_dir="${N3_RUN_DIR}"; ckpt="${N3_CHECKPOINT}" ;;
    n2) kind=nseries; run_dir="${N2_RUN_DIR}"; ckpt="${N2_CHECKPOINT}" ;;
    n7) kind=nseries; run_dir="${N7_RUN_DIR}"; ckpt="${N7_CHECKPOINT}" ;;
  esac
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/make_led_hb_residual_panels.py \
    --run-kind "${kind}" --run-dir "${run_dir}" --checkpoint "${ckpt}" --output-dir "${OUT_ROOT}/panels_${method}" \
    --num-panels 8 --selection-mode linspace --max-depth 200.0 --device cuda
done

"${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/summarize_led_hb_formal.py \
  --input-root "${OUT_ROOT}" --output-dir "${OUT_ROOT}"
echo "POSTEVAL_DIR=${OUT_ROOT}"
