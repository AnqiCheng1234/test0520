#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXPECTED_ROOT="${EXPECTED_ROOT:-/home/caq/6666_raw/dav2_raw_0522}"
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
PYTHON_BIN="${PYTHON_BIN:-/home/caq/anaconda3/envs/${CONDA_ENV}/bin/python}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

SELECTION_MODE="${SELECTION_MODE:-mixed}"
FIXED_SAMPLE_INDICES="${FIXED_SAMPLE_INDICES:-0,72}"
TOPK_BETTER="${TOPK_BETTER:-4}"
TOPK_WORSE="${TOPK_WORSE:-4}"
MAX_SCAN_SAMPLES="${MAX_SCAN_SAMPLES:-}"

C2_RUN_DIR="${C2_RUN_DIR:-${ROOT}/finetune_stf/exp/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20}"
C2_CKPT="${C2_CKPT:-/mnt/drive/3333_raw/0000_exp_ckpt/0525_0203_vkitti_c2_d0only_residual_vits_halfd0_187x621_sceneholdout_Scene20_n1000_seed42_bs8_e20/best_abs_rel.pth}"
C2_CHECKPOINT="${C2_CHECKPOINT:-${C2_CKPT}}"

N2_LP08_RUN_DIR="${N2_LP08_RUN_DIR:-${ROOT}/finetune_stf/exp/0529_1523_vkitti_n2_x3_lp0p8_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10}"
N2_LP08_CKPT="${N2_LP08_CKPT:-}"
N2_LP08_FALLBACK_RUN_DIR="${N2_LP08_FALLBACK_RUN_DIR:-${ROOT}/finetune_stf/exp/0528_0049_vkitti_n2_x3_lp0p8_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10}"
N2_LP08_FALLBACK_CKPT="${N2_LP08_FALLBACK_CKPT:-}"

N2_LP05_RUN_DIR="${N2_LP05_RUN_DIR:-${ROOT}/finetune_stf/exp/0527_2354_vkitti_n2_x3_lp0p5_q0p5_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10}"
N2_LP05_CKPT="${N2_LP05_CKPT:-}"

N4_RUN_DIR="${N4_RUN_DIR:-${ROOT}/finetune_stf/exp/0530_0213_vkitti_n4_ffm_mid_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10}"
N4_CKPT="${N4_CKPT:-}"
N5_RUN_DIR="${N5_RUN_DIR:-${ROOT}/finetune_stf/exp/0530_0350_vkitti_n5_d1_lp0p5_q0p3_lfl0p0_rftna_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10}"
N5_CKPT="${N5_CKPT:-}"
N7_RUN_DIR="${N7_RUN_DIR:-${ROOT}/finetune_stf/exp/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10}"
N7_CKPT="${N7_CKPT:-}"

timestamp="$(date +%m%d_%H%M)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/plans/0527/diagnostics/0530_improvement_vs_c2_${timestamp}}"

if [[ "${ROOT}" != "${EXPECTED_ROOT}" ]]; then
  echo "[ERROR] Unexpected project root: ${ROOT}; expected ${EXPECTED_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${ROOT}/foundation/tools/make_residual_vs_c2_panels.py" ]]; then
  echo "[ERROR] Project root check failed; missing foundation/tools/make_residual_vs_c2_panels.py under ${ROOT}" >&2
  exit 2
fi
if [[ -e "${OUT_ROOT}" ]]; then
  echo "[ERROR] Refusing to overwrite existing OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}"
RUN_LOG="${OUT_ROOT}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[INFO] host=$(hostname)"
echo "[INFO] user=$(whoami)"
echo "[INFO] root=${ROOT}"
echo "[INFO] out_root=${OUT_ROOT}"
echo "[INFO] conda_env=${CONDA_ENV}"
echo "[INFO] python_bin=${PYTHON_BIN}"
echo "[INFO] selection_mode=${SELECTION_MODE} fixed=${FIXED_SAMPLE_INDICES} topk_better=${TOPK_BETTER} topk_worse=${TOPK_WORSE} max_scan_samples=${MAX_SCAN_SAMPLES:-none}"

missing=()
method_labels=()
method_run_dirs=()
method_ckpts=()
method_kinds=()
method_out_dirs=()

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    missing+=("${label} run dir missing: ${path}")
    return 1
  fi
  return 0
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    missing+=("${label} checkpoint missing: ${path}")
    return 1
  fi
  return 0
}

quote_cmd() {
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
}

try_resolve_checkpoint() {
  local run_dir="$1"
  local manual_ckpt="$2"
  RESOLVED_CKPT=""
  CHECKED_CANDIDATES=()
  CHECKPOINT_ERROR=""
  if [[ ! -d "${run_dir}" ]]; then
    CHECKPOINT_ERROR="run dir missing: ${run_dir}"
    return 1
  fi
  if [[ -n "${manual_ckpt}" ]]; then
    CHECKED_CANDIDATES+=("${manual_ckpt}")
    if [[ -f "${manual_ckpt}" ]]; then
      RESOLVED_CKPT="${manual_ckpt}"
      return 0
    fi
    CHECKPOINT_ERROR="manual checkpoint not found: ${manual_ckpt}"
    return 1
  fi
  mapfile -t CHECKED_CANDIDATES < <("${PYTHON_BIN}" - "${run_dir}" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).expanduser().resolve()
out = []

def add(value):
    if not value:
        return
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = (run_dir / p).resolve()
    s = str(p)
    if s not in out:
        out.append(s)

def load(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

for name in ("best_val_metrics.json", "best_kitti_metrics.json"):
    payload = load(run_dir / name)
    if isinstance(payload, dict):
        add(payload.get("checkpoint_path"))
        add(payload.get("best_checkpoint"))
        for key in ("best_abs_rel", "best_target_region_score", "best_kitti_abs_rel"):
            record = payload.get(key)
            if isinstance(record, dict):
                add(record.get("checkpoint_path"))

payload = load(run_dir / "run_summary.json")
if isinstance(payload, dict):
    for key in ("best_abs_rel", "best_target_region_score", "best_kitti_abs_rel", "best_val_metrics", "best_kitti_metrics"):
        record = payload.get(key)
        if isinstance(record, dict):
            add(record.get("checkpoint_path"))

payload = load(run_dir / "config.json")
if isinstance(payload, dict):
    add(payload.get("checkpoint_path"))

latest = run_dir / "latest_checkpoint"
if latest.is_file():
    try:
        add(latest.read_text(encoding="utf-8").strip())
    except Exception:
        pass

for sub in ("checkpoints", "ckpts"):
    d = run_dir / sub
    if d.is_dir():
        for pattern in ("best_abs_rel.pth", "best*.pth", "epoch_*.pth", "latest.pth", "*.pt"):
            for p in sorted(d.glob(pattern)):
                add(p)

heavy = Path("/mnt/drive/3333_raw/0000_exp_ckpt") / run_dir.name
for pattern in ("best_abs_rel.pth", "epoch_*.pth", "latest.pth", "*.pt"):
    paths = list(heavy.glob(pattern)) if heavy.is_dir() else []
    if pattern == "epoch_*.pth":
        def epoch_num(path):
            match = re.search(r"epoch_(\d+)", path.name)
            return int(match.group(1)) if match else -1
        paths = sorted(paths, key=epoch_num)
    else:
        paths = sorted(paths)
    for p in paths:
        add(p)

for item in out:
    print(item)
PY
)
  local candidate
  for candidate in "${CHECKED_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
      RESOLVED_CKPT="${candidate}"
      return 0
    fi
  done
  CHECKPOINT_ERROR="no existing checkpoint candidate for ${run_dir}; checked: ${CHECKED_CANDIDATES[*]:-none}"
  return 1
}

add_method() {
  local label="$1"
  local run_dir="$2"
  local ckpt="$3"
  local kind="$4"
  method_labels+=("${label}")
  method_run_dirs+=("${run_dir}")
  method_ckpts+=("${ckpt}")
  method_kinds+=("${kind}")
}

add_required_nseries() {
  local label="$1"
  local run_dir="$2"
  local manual_ckpt="$3"
  if ! require_dir "${label}" "${run_dir}"; then
    return 1
  fi
  if ! try_resolve_checkpoint "${run_dir}" "${manual_ckpt}"; then
    missing+=("${label} checkpoint unresolved: ${CHECKPOINT_ERROR}")
    return 1
  fi
  add_method "${label}" "${run_dir}" "${RESOLVED_CKPT}" "nseries"
  return 0
}

add_n2_lp08_with_fallback() {
  if [[ -d "${N2_LP08_RUN_DIR}" ]] && try_resolve_checkpoint "${N2_LP08_RUN_DIR}" "${N2_LP08_CKPT}"; then
    add_method "N2_lp0p8_q0p3_aggressive" "${N2_LP08_RUN_DIR}" "${RESOLVED_CKPT}" "nseries"
    return 0
  fi
  echo "[WARN] N2 lp0.8 q0.3 checkpoint unavailable; trying q0.5 fallback. Reason: ${CHECKPOINT_ERROR:-run dir missing}"
  if [[ -d "${N2_LP08_FALLBACK_RUN_DIR}" ]] && try_resolve_checkpoint "${N2_LP08_FALLBACK_RUN_DIR}" "${N2_LP08_FALLBACK_CKPT}"; then
    add_method "N2_lp0p8_q0p5_aggressive_fallback" "${N2_LP08_FALLBACK_RUN_DIR}" "${RESOLVED_CKPT}" "nseries"
    return 0
  fi
  missing+=("N2 aggressive checkpoint unresolved: primary=${N2_LP08_RUN_DIR} fallback=${N2_LP08_FALLBACK_RUN_DIR} reason=${CHECKPOINT_ERROR:-run dir missing}")
  return 1
}

require_dir "C2" "${C2_RUN_DIR}" || true
require_file "C2" "${C2_CHECKPOINT}" || true
add_n2_lp08_with_fallback || true
add_required_nseries "N2_lp0p5_q0p5_conservative" "${N2_LP05_RUN_DIR}" "${N2_LP05_CKPT}" || true

if [[ -n "${N4_RUN_DIR}" ]]; then
  add_required_nseries "N4_$(basename "${N4_RUN_DIR}")" "${N4_RUN_DIR}" "${N4_CKPT}" || true
fi
if [[ -n "${N5_RUN_DIR}" ]]; then
  add_required_nseries "N5_$(basename "${N5_RUN_DIR}")" "${N5_RUN_DIR}" "${N5_CKPT}" || true
fi
if [[ -n "${N7_RUN_DIR}" ]]; then
  add_required_nseries "N7_$(basename "${N7_RUN_DIR}")" "${N7_RUN_DIR}" "${N7_CKPT}" || true
fi

if (( ${#missing[@]} > 0 )); then
  echo "[ERROR] Missing required paths:"
  printf ' - %s\n' "${missing[@]}"
  exit 2
fi
if (( ${#method_labels[@]} == 0 )); then
  echo "[ERROR] No methods resolved."
  exit 2
fi

echo "[INFO] resolved methods:"
for i in "${!method_labels[@]}"; do
  echo "[INFO] ${method_labels[$i]} run=${method_run_dirs[$i]} checkpoint=${method_ckpts[$i]}"
done

for i in "${!method_labels[@]}"; do
  label="${method_labels[$i]}"
  run_dir="${method_run_dirs[$i]}"
  ckpt="${method_ckpts[$i]}"
  kind="${method_kinds[$i]}"
  method_out="${OUT_ROOT}/${label}"
  method_out_dirs+=("${method_out}")
  method_log="${OUT_ROOT}/${label}.log"
  cmd=(
    "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/make_residual_vs_c2_panels.py
    --c2-run-dir "${C2_RUN_DIR}"
    --c2-checkpoint "${C2_CHECKPOINT}"
    --method-run-dir "${run_dir}"
    --method-checkpoint "${ckpt}"
    --method-kind "${kind}"
    --method-label "${label}"
    --output-dir "${method_out}"
    --selection-mode "${SELECTION_MODE}"
    --sample-indices "${FIXED_SAMPLE_INDICES}"
    --topk-better "${TOPK_BETTER}"
    --topk-worse "${TOPK_WORSE}"
    --device "${DEVICE}"
    --write-summary
  )
  if [[ -n "${MAX_SCAN_SAMPLES}" ]]; then
    cmd+=(--max-scan-samples "${MAX_SCAN_SAMPLES}")
  fi
  {
    quote_cmd "${cmd[@]}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
  } 2>&1 | tee -a "${method_log}"
done

summary_cmd=(
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/summarize_improvement_vs_c2_panels.py
  --output-dir "${OUT_ROOT}"
  --method-dirs
)
summary_cmd+=("${method_out_dirs[@]}")
quote_cmd "${summary_cmd[@]}"
"${summary_cmd[@]}"

echo "[DONE] improvement-over-C2 visualization completed"
echo "[DONE] output=${OUT_ROOT}"
echo "[DONE] run_log=${RUN_LOG}"
