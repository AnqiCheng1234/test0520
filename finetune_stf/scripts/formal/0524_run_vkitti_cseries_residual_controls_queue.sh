#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
SESSION_PREFIX="${SESSION_PREFIX:-vkitti_cseries_residual_controls}"
SPLIT_TAG="${SPLIT_TAG:-sceneholdout_Scene20_n1000_seed42}"

PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
VKITTI_TRAIN_LIST="${VKITTI_TRAIN_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/train_sceneholdout_Scene20.txt}"
VKITTI_VAL_LIST="${VKITTI_VAL_LIST:-${ROOT}/finetune_stf/dataset/splits/vkitti2/val_sceneholdout_Scene20_n1000_seed42.txt}"
EVAL_KITTI="${EVAL_KITTI:-1}"
KITTI_BASE="${KITTI_BASE:-/mnt/drive/kitti}"
KITTI_VAL_SPLIT="${KITTI_VAL_SPLIT:-/home/caq/6666_raw/dav2_raw/metric_depth/dataset/splits/kitti/val.txt}"
KITTI_EVAL_PROTOCOL="${KITTI_EVAL_PROTOCOL:-halfres_rgb_canonical_even_pad_crop_affine_disp}"
KITTI_EXPECTED_VAL_SAMPLES="${KITTI_EXPECTED_VAL_SAMPLES:-652}"
KITTI_NUM_WORKERS="${KITTI_NUM_WORKERS:-2}"
M2_RUN="${M2_RUN:-}"
SKIP_M2_GATE="${SKIP_M2_GATE:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_SIGN_CHECK="${RUN_SIGN_CHECK:-1}"
KEEP_SMOKE="${KEEP_SMOKE:-0}"
D0_SIGN="${D0_SIGN:-}"
RUN_C2="${RUN_C2:-1}"
RUN_C1="${RUN_C1:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0524_run_vkitti_cseries_residual_controls_queue.sh

Starts one new tmux session. By default it runs dataset/C1/C2 smoke checks,
checks M2 is useful, infers D0 sign on the control dataset, then launches
formal C1 followed by C2.

Useful overrides:
  GPU=1 RUN_SMOKE=0 D0_SIGN=1 bash ...
  SKIP_M2_GATE=1 RUN_SIGN_CHECK=0 D0_SIGN=1 bash ...
  M2_RUN=finetune_stf/exp/<matching_m2_sceneholdout_run> bash ...
  EVAL_KITTI=0 bash ...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" != "--run-internal" ]]; then
  mkdir -p "${LOG_ROOT}"
  run_timestamp="$(date +%m%d_%H%M)"
  session="${run_timestamp}_${SESSION_PREFIX}"
  queue_log="${LOG_ROOT}/${session}.queue.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[ERROR] Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi

  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' VKITTI_TRAIN_LIST='${VKITTI_TRAIN_LIST}' VKITTI_VAL_LIST='${VKITTI_VAL_LIST}' EVAL_KITTI='${EVAL_KITTI}' KITTI_BASE='${KITTI_BASE}' KITTI_VAL_SPLIT='${KITTI_VAL_SPLIT}' KITTI_EVAL_PROTOCOL='${KITTI_EVAL_PROTOCOL}' KITTI_EXPECTED_VAL_SAMPLES='${KITTI_EXPECTED_VAL_SAMPLES}' KITTI_NUM_WORKERS='${KITTI_NUM_WORKERS}' M2_RUN='${M2_RUN}' SKIP_M2_GATE='${SKIP_M2_GATE}' RUN_SMOKE='${RUN_SMOKE}' RUN_SIGN_CHECK='${RUN_SIGN_CHECK}' KEEP_SMOKE='${KEEP_SMOKE}' D0_SIGN='${D0_SIGN}' RUN_C2='${RUN_C2}' RUN_C1='${RUN_C1}' SPLIT_TAG='${SPLIT_TAG}' RUN_TIMESTAMP='${run_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"

  cat <<EOF
tmux session: ${session}
queue log: ${queue_log}
attach: tmux attach -t ${session}
monitor: tail -f ${queue_log}
EOF
  exit 0
fi

cd "${ROOT}"
mkdir -p "${EXP_ROOT}" "${LOG_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file not found: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 2
  fi
}

log_header() {
  local label="$1"
  local run="$2"
  local log="$3"
  {
    echo "[START] $(date -Iseconds)"
    echo "[LABEL] ${label}"
    echo "[RUN] ${run}"
    echo "[SESSION] ${QUEUE_SESSION:-internal}"
    echo "[HOST] $(hostname)"
    echo "[USER] $(whoami)"
    echo "[PWD] $(pwd)"
    echo "[GPU] ${GPU}"
    echo "[LOG] ${log}"
  } 2>&1 | tee -a "${log}"
}

common_args=(
  --input-domain rgb
  --model-input-tensor image
  --dataset-geometry-mode vkitti2_even_fullres_halfres_2x2
  --raw-storage-format not_applicable
  --fullres-even-policy crop_bottom_to_even
  --rgb-input-space halfres_2x2_area
  --depth-target-space halfres_2x2_valid_mean
  --front-end dav2_rgb_frozen
  --encoder vits
  --pretrained-from "${PRETRAINED}"
  --vkitti-train-list "${VKITTI_TRAIN_LIST}"
  --vkitti-val-list "${VKITTI_VAL_LIST}"
  --input-height 187
  --input-width 621
  --min-depth 1.0
  --max-depth 80.0
  --residual-alpha 0.5
  --hflip-prob 0.5
  --accum-steps 1
  --lr 1e-4
  --weight-decay 1e-4
  --amp
  --amp-dtype bf16
  --seed 42
)

if [[ "${EVAL_KITTI}" == "1" ]]; then
  common_args+=(
    --eval-kitti
    --kitti-base "${KITTI_BASE}"
    --kitti-val-split "${KITTI_VAL_SPLIT}"
    --kitti-eval-protocol "${KITTI_EVAL_PROTOCOL}"
    --kitti-expected-val-samples "${KITTI_EXPECTED_VAL_SAMPLES}"
    --kitti-num-workers "${KITTI_NUM_WORKERS}"
  )
fi

check_m2_gate() {
  if [[ "${SKIP_M2_GATE}" == "1" ]]; then
    echo "[M2_GATE] skipped by SKIP_M2_GATE=1"
    return 0
  fi
  if [[ -z "${M2_RUN}" ]]; then
    echo "[ERROR] M2_RUN is empty. Set M2_RUN to a completed matching M2 run, or SKIP_M2_GATE=1 to override." >&2
    return 2
  fi
  if [[ ! -d "${M2_RUN}" ]]; then
    echo "[ERROR] M2_RUN not found: ${M2_RUN}" >&2
    echo "[ERROR] Set M2_RUN to a completed M2 run or SKIP_M2_GATE=1 to override." >&2
    return 2
  fi
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c '
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected_train = str(Path(sys.argv[2]).expanduser().resolve())
expected_val = str(Path(sys.argv[3]).expanduser().resolve())
config_path = run / "config.json"
val_path = run / "val_metrics.json"
if not config_path.is_file():
    raise SystemExit(f"[ERROR] missing config.json: {config_path}")
if not val_path.is_file():
    raise SystemExit(f"[ERROR] missing val_metrics.json: {val_path}")
config = json.load(open(config_path, "r", encoding="utf-8"))
actual_train = str(Path(config.get("vkitti_train_list", "")).expanduser().resolve())
actual_val = str(Path(config.get("vkitti_val_list", "")).expanduser().resolve())
if actual_train != expected_train or actual_val != expected_val:
    raise SystemExit(
        "[ERROR] M2 split mismatch. "
        f"M2 train={actual_train} val={actual_val}; "
        f"queue train={expected_train} val={expected_val}. "
        "Set M2_RUN to the matching run or SKIP_M2_GATE=1."
    )
payload = json.load(open(val_path, "r", encoding="utf-8"))
epochs = payload.get("epochs") or [payload.get("latest")]
epochs = [e for e in epochs if isinstance(e, dict)]
if not epochs:
    raise SystemExit("[ERROR] M2 val metrics empty")
best = min(
    epochs,
    key=lambda e: float(e.get("overall", {}).get("final", {}).get("abs_rel", float("inf"))),
)
overall = best.get("overall", {})
final = overall.get("final", {})
d0 = overall.get("D0", {})
final_abs = final.get("abs_rel")
d0_abs = d0.get("abs_rel")
if final_abs is None or d0_abs is None:
    raise SystemExit("[ERROR] M2 missing final/D0 abs_rel")
final_abs = float(final_abs)
d0_abs = float(d0_abs)
region_delta = best.get("region", {}).get("delta", {})
region_benefits = [
    float(v) for v in region_delta.values()
    if v is not None and math.isfinite(float(v)) and float(v) < 0.0
]
if final_abs < d0_abs or region_benefits:
    print(f"[M2_GATE] pass run={run.name} final_abs_rel={final_abs:.6f} D0_abs_rel={d0_abs:.6f} region_benefits={len(region_benefits)}")
else:
    raise SystemExit(
        f"[ERROR] M2 gate failed: final_abs_rel={final_abs:.6f} D0_abs_rel={d0_abs:.6f} and no improving region deltas"
    )
' "${M2_RUN}" "${VKITTI_TRAIN_LIST}" "${VKITTI_VAL_LIST}"
}

run_dataset_smoke() {
  local smoke_root="$1"
  local out="${smoke_root}/dataset_shape_summary.json"
  mkdir -p "${smoke_root}"
  echo "[SMOKE] dataset shape -> ${out}"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/smoke_vkitti2_residual_control_dataset.py \
    --vkitti-train-list "${VKITTI_TRAIN_LIST}" \
    --vkitti-val-list "${VKITTI_VAL_LIST}" \
    --input-height 187 \
    --input-width 621 \
    --fullres-even-policy crop_bottom_to_even \
    --rgb-input-space halfres_2x2_area \
    --depth-target-space halfres_2x2_valid_mean \
    --hflip-prob 0.5 \
    --output "${out}"
}

run_smoke_train() {
  local label="$1"
  local experiment_id="$2"
  local feature_source="$3"
  local smoke_root="$4"
  local smoke_log="${smoke_root}/train_smoke.log"
  local smoke_kitti_args=()
  if [[ "${EVAL_KITTI}" == "1" ]]; then
    smoke_kitti_args+=(--max-kitti-val-samples 4)
  fi
  mkdir -p "${smoke_root}"
  echo "[SMOKE] ${label} root=${smoke_root}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_residual_control.py \
    --experiment-id "${experiment_id}" \
    "${common_args[@]}" \
    --residual-feature-source "${feature_source}" \
    --d0-sign 1 \
    --epochs 1 \
    --bs 8 \
    --num-workers 0 \
    --log-interval 1 \
    --save-interval 1 \
    --eval-interval 1 \
    --max-train-steps 2 \
    --max-val-samples 4 \
    "${smoke_kitti_args[@]}" \
    --save-path "${smoke_root}/exp" \
    --heavy-save-path "${smoke_root}/heavy" 2>&1 | tee -a "${smoke_log}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[SMOKE][ERROR] ${label} failed; kept ${smoke_root}" >&2
    return "${status}"
  fi
}

run_sign_check() {
  local smoke_root="$1"
  local sign_json="${smoke_root}/vkitti_control_d0_sign_summary.json"
  mkdir -p "${smoke_root}"
  echo "[SIGN] checking control D0 sign -> ${sign_json}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/check_vkitti_control_dav2_sign.py \
    --encoder vits \
    --pretrained-from "${PRETRAINED}" \
    --vkitti-val-list "${VKITTI_VAL_LIST}" \
    --input-height 187 \
    --input-width 621 \
    --fullres-even-policy crop_bottom_to_even \
    --rgb-input-space halfres_2x2_area \
    --depth-target-space halfres_2x2_valid_mean \
    --min-depth 1.0 \
    --max-depth 80.0 \
    --max-samples 64 \
    --output "${sign_json}"
  D0_SIGN="$("${CONDA_BIN}" run -n "${CONDA_ENV}" python -c "import json; print(json.load(open('${sign_json}', 'r', encoding='utf-8'))['recommended_d0_sign'])")"
  export D0_SIGN
  echo "[SIGN] recommended --d0-sign ${D0_SIGN}"
}

run_training() {
  local label="$1"
  local run_name="$2"
  local experiment_id="$3"
  local feature_source="$4"
  shift 4

  local save="${EXP_ROOT}/${run_name}"
  local heavy="${HEAVY_ROOT}/${run_name}"
  local log="${LOG_ROOT}/${run_name}.tmux.log"

  if [[ -e "${save}" || -e "${heavy}" ]]; then
    echo "[ERROR] Refusing to overwrite existing artifacts for ${run_name}" >&2
    echo "  save=${save}" >&2
    echo "  heavy=${heavy}" >&2
    return 2
  fi

  mkdir -p "${save}"
  log_header "${label}" "${run_name}" "${log}"
  {
    echo -n "[CMD] CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} python foundation/tools/train_vkitti2_residual_control.py --experiment-id ${experiment_id}"
    printf ' %q' "${common_args[@]}"
    printf ' --residual-feature-source %q --d0-sign %q' "${feature_source}" "${D0_SIGN}"
    printf ' %q' "$@"
    printf ' --save-path %q --heavy-save-path %q\n' "${save}" "${heavy}"
  } 2>&1 | tee -a "${log}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    python foundation/tools/train_vkitti2_residual_control.py \
    --experiment-id "${experiment_id}" \
    "${common_args[@]}" \
    --residual-feature-source "${feature_source}" \
    --d0-sign "${D0_SIGN}" \
    "$@" \
    --save-path "${save}" \
    --heavy-save-path "${heavy}" 2>&1 | tee -a "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[END] $(date -Iseconds) status=${status}" 2>&1 | tee -a "${log}"
  return "${status}"
}

require_file "${PRETRAINED}"
require_file "${VKITTI_TRAIN_LIST}"
require_file "${VKITTI_VAL_LIST}"
if [[ "${EVAL_KITTI}" == "1" ]]; then
  require_file "${KITTI_VAL_SPLIT}"
  require_dir "${KITTI_BASE}"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%m%d_%H%M)}"
SMOKE_ROOT="${ROOT}/plans/0524_new/codex_smoke_vkitti_cseries_queue_${RUN_TIMESTAMP}"

echo "[QUEUE_START] $(date -Iseconds)"
echo "[QUEUE_SESSION] ${QUEUE_SESSION:-internal}"
echo "[RUN_TIMESTAMP] ${RUN_TIMESTAMP}"
echo "[PRETRAINED] ${PRETRAINED}"
echo "[TRAIN_LIST] ${VKITTI_TRAIN_LIST}"
echo "[VAL_LIST] ${VKITTI_VAL_LIST}"
echo "[M2_RUN] ${M2_RUN}"
echo "[EVAL_KITTI] ${EVAL_KITTI}"
if [[ "${EVAL_KITTI}" == "1" ]]; then
  echo "[KITTI_BASE] ${KITTI_BASE}"
  echo "[KITTI_VAL_SPLIT] ${KITTI_VAL_SPLIT}"
  echo "[KITTI_EVAL_PROTOCOL] ${KITTI_EVAL_PROTOCOL}"
  echo "[KITTI_EXPECTED_VAL_SAMPLES] ${KITTI_EXPECTED_VAL_SAMPLES}"
fi
echo "[SPLIT_TAG] ${SPLIT_TAG:-none}"

check_m2_gate

if [[ "${RUN_SMOKE}" == "1" ]]; then
  run_dataset_smoke "${SMOKE_ROOT}/dataset_shape"
  run_smoke_train "C1 RGB control smoke" "C1" "rgb" "${SMOKE_ROOT}/c1_rgb"
  run_smoke_train "C2 D0-only control smoke" "C2" "d0" "${SMOKE_ROOT}/c2_d0"
fi

if [[ "${D0_SIGN}" == "1" || "${D0_SIGN}" == "-1" ]]; then
  echo "[SIGN] using explicit D0_SIGN=${D0_SIGN}; automatic sign inference skipped"
elif [[ -z "${D0_SIGN}" && "${RUN_SIGN_CHECK}" == "1" ]]; then
  run_sign_check "${SMOKE_ROOT}/sign_check"
fi

if [[ -z "${D0_SIGN}" ]]; then
  echo "[ERROR] D0_SIGN is empty. Set D0_SIGN=1 or D0_SIGN=-1, or enable RUN_SIGN_CHECK=1." >&2
  exit 2
fi
if [[ "${D0_SIGN}" != "1" && "${D0_SIGN}" != "-1" ]]; then
  echo "[ERROR] D0_SIGN must be 1 or -1, got ${D0_SIGN}" >&2
  exit 2
fi

if [[ "${KEEP_SMOKE}" != "1" && -d "${SMOKE_ROOT}" ]]; then
  rm -rf "${SMOKE_ROOT}"
  echo "[SMOKE] removed successful smoke artifacts: ${SMOKE_ROOT}"
fi

if [[ "${RUN_C1}" == "1" ]]; then
  run_training "formal C1 RGB residual control" "${RUN_TIMESTAMP}_vkitti_c1_rgb_residual_vits_halfrgb_187x621${SPLIT_TAG:+_${SPLIT_TAG}}_bs8_e20" "C1" "rgb" \
    --epochs 20 \
    --bs 8 \
    --num-workers 4 \
    --log-interval 100 \
    --save-interval 1 \
    --eval-interval 1 \
    --save-best-checkpoint
fi

if [[ "${RUN_C2}" == "1" ]]; then
  run_training "formal C2 D0-only residual control" "${RUN_TIMESTAMP}_vkitti_c2_d0only_residual_vits_halfd0_187x621${SPLIT_TAG:+_${SPLIT_TAG}}_bs8_e20" "C2" "d0" \
    --epochs 20 \
    --bs 8 \
    --num-workers 4 \
    --log-interval 100 \
    --save-interval 1 \
    --eval-interval 1 \
    --save-best-checkpoint
fi

echo "[QUEUE_END] $(date -Iseconds) status=0 d0_sign=${D0_SIGN}"
