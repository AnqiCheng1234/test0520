#!/usr/bin/env bash
set -euo pipefail

# Round-2 e10 diagnostic queue after the first noiseaware+featdistill short-train failed.
# Two independent axes (kept single-variable, NOT mixed into one run):
#   A-axis : enlarge the noiseaware_v1 local residual budget (scale/gate up), feat=none.
#   B-axis : switch feature-distill teacher to same-param C_dark(RAW_normal), local=none.
# Plus one combination run and one extra baseline seed to pin the seed-noise floor.
# Reuses existing L0/L0_seed123/L1/L2 checkpoints as the comparison anchors (not re-run here).

ROOT="${ROOT:-/home/caq/6666_raw/dav2_raw_0603}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/finetune_stf/logs}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
PRETRAINED="${PRETRAINED:-/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth}"
LOD_ROOT="${LOD_ROOT:-/home/caq/6666_raw/0000_dataset/LOD}"
LOD_MANIFEST="${LOD_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv}"
# Student init AND same-param teacher both use the canonical C_dark LoRA checkpoint.
C_DARK="${C_DARK:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_raw_featdistill_round2_amp_sameparam_short_e10_queue}"
MASTER_PORT="${MASTER_PORT:-29651}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
LAMBDA_FEAT="${LAMBDA_FEAT:-0.05}"
# Enlarged local-residual budget (vs first round scale=0.1 gate_init=0.03).
LOCAL_HIDDEN="${LOCAL_HIDDEN:-16}"
LOCAL_GATE_INIT="${LOCAL_GATE_INIT:-0.1}"
LOCAL_SCALE_A1="${LOCAL_SCALE_A1:-0.3}"
LOCAL_SCALE_A2="${LOCAL_SCALE_A2:-0.5}"
RUN_PREFIX="${RUN_PREFIX:-}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0609_run_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue.sh
  bash finetune_stf/scripts/formal/0609_run_lod_raw_featdistill_round2_amp_sameparam_short_e10_queue.sh --audit

Round-2 short-train matrix (6 new runs, single-variable per axis):
  C0_seed7   baseline   feat=none  local=none           seed=7    (3rd baseline seed -> sigma)
  A_s03      A-axis     feat=none  local=noiseaware s0.3 seed=42   (vs L2 s0.1)
  A_s05      A-axis     feat=none  local=noiseaware s0.5 seed=42   (vs L2 s0.1)
  B_sp       B-axis     feat=sameparam(C_dark)  local=none seed=42 (vs L1 normal_expert)
  B_sp_s123  B-axis     feat=sameparam(C_dark)  local=none seed=123(vs L0_seed123)
  AB_s05_sp  combo      feat=sameparam  local=noiseaware s0.5 seed=42

Overrides:
  GPU=0 MASTER_PORT=29651 EPOCHS=10 BS=8 LAMBDA_FEAT=0.05 \
  LOCAL_GATE_INIT=0.1 LOCAL_SCALE_A1=0.3 LOCAL_SCALE_A2=0.5 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' C_DARK='${C_DARK}' MASTER_PORT='${MASTER_PORT}' EPOCHS='${EPOCHS}' BS='${BS}' LAMBDA_FEAT='${LAMBDA_FEAT}' LOCAL_HIDDEN='${LOCAL_HIDDEN}' LOCAL_GATE_INIT='${LOCAL_GATE_INIT}' LOCAL_SCALE_A1='${LOCAL_SCALE_A1}' LOCAL_SCALE_A2='${LOCAL_SCALE_A2}' RUN_GROUPS='${RUN_GROUPS:-}' RUN_PREFIX='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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

RUN_PREFIX="${RUN_PREFIX:-$(date +%m%d_%H%M)}"

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

require_file "${PRETRAINED}"
require_file "${LOD_MANIFEST}"
require_file "${C_DARK}"
require_dir "${LOD_ROOT}"

log_header() {
  local label="$1"
  local run="$2"
  local port="$3"
  local log="$4"
  local teacher_disp="$5"
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
    echo "[LOD_ROOT] ${LOD_ROOT}"
    echo "[LOD_MANIFEST] ${LOD_MANIFEST}"
    echo "[STUDENT_INIT] ${C_DARK}"
    echo "[TEACHER] ${teacher_disp}"
    echo "[SEMANTICS] matrix=${label} epochs=${EPOCHS} bs=${BS} lambda_feat=${LAMBDA_FEAT} gate_init=${LOCAL_GATE_INIT} pair_input=RAW_Dark+RAW_normal aug=hflip0.5_only"
  } 2>&1 | tee -a "${log}"
}

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_lod_round2_audit_${run}"
  "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python - "$@" \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${audit_save}" <<'PY'
import json
from finetune_stf.train import parse_args, metric_direction, resolve_lod_aug_config

args = parse_args()
cfg = args.resolved_config.to_dict()
payload = {
    "run": args.save_path.rsplit("/", 1)[-1],
    "dataset_family": cfg["dataset_family"],
    "dataset_input_mode": cfg["dataset_input_mode"],
    "input_domain": cfg["input_domain"],
    "front_end": cfg["front_end"],
    "model_input_tensor": cfg["model_input_tensor"],
    "raw_storage_format": cfg["raw_storage_format"],
    "student_init_from": cfg["student_init_from"],
    "teacher_ckpt": cfg["teacher_ckpt"],
    "teacher_recipe": cfg["teacher_recipe"],
    "teacher_input_mode": cfg["teacher_input_mode"],
    "feat_distill": cfg["feat_distill"],
    "feat_distill_layers": cfg["feat_distill_layers"],
    "feat_distill_lambda": cfg["feat_distill_lambda"],
    "feat_distill_teacher": cfg["feat_distill_teacher"],
    "raw_ram_local_residual": cfg["raw_ram_local_residual"],
    "raw_ram_local_hidden_ch": cfg["raw_ram_local_hidden_ch"],
    "raw_ram_local_residual_scale": cfg["raw_ram_local_residual_scale"],
    "raw_ram_local_gate_init": cfg["raw_ram_local_gate_init"],
    "raw_ram_local_gate_mode": cfg["raw_ram_local_gate_mode"],
    "lora": cfg["lora"],
    "seed": getattr(args, "seed", None),
    "epochs": args.epochs,
    "bs": args.bs,
}
print("[AUDIT][OK] " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

run_train() {
  local label="$1"
  local run="$2"
  local port="$3"
  local teacher_disp="$4"
  shift 4

  # Optional group filter: RUN_GROUPS="AB_s05_sp" or "A_s05,B_sp" runs only those labels.
  # NOTE: must NOT be named GROUPS (bash builtin readonly group-id array).
  if [[ -n "${RUN_GROUPS:-}" ]]; then
    case ",${RUN_GROUPS}," in
      *",${label},"*) ;;
      *) echo "[SKIP] ${label} (not in RUN_GROUPS=${RUN_GROUPS})"; return 0 ;;
    esac
  fi

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
  log_header "${label}" "${run}" "${port}" "${log}" "${teacher_disp}"

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
  --stage lod_only
  --lod-root "${LOD_ROOT}"
  --lod-manifest "${LOD_MANIFEST}"
  --lod-label-space inverse_relative
  --input-height 512
  --input-width 960
  --lod-train-crop-mode random
  --lod-val-crop-mode center
  --no-eval-stf
  --eval-lod
  --eval-lod-train-proxy
  --lod-train-proxy-count 112
  --best-metric lod_d1
  --save-best-checkpoint
  --bs "${BS}"
  --accum-steps 1
  --loss-type ssi
  --loss-target-normalization
  --loss-norm-min-scale 1e-3
  --epochs "${EPOCHS}"
  --lr-schedule poly
  --warmup-steps 0
  --aug-preset off
  --aug-hflip-prob 0.5
  --amp
  --amp-dtype bf16
  --num-workers 4
  --log-interval 500
  --no-enable-fixed-viz-dump
  --no-enable-train-source-viz-dump
  --dataset-family lod_true_raw_dark_normal_pair_rgb16
  --dataset-input-mode raw_rgb16_dark_normal_pair
  --input-domain raw3
  --front-end raw_rgb16_ram3
  --model-input-tensor raw
  --raw-storage-format raw_rgb16_png_3ch
  --lod-raw-norm-mode uint16_div_65535
  --raw-front-end-lr 5e-5
  --raw-ram-rgb-tail identity
  --student-init-from "${C_DARK}"
  --student-init-strict compatible
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
  --lr 1e-5
)

feat_none_args=(
  --teacher-ckpt n_a
  --teacher-recipe n_a
  --teacher-input-mode n_a
  --feat-distill none
  --feat-distill-layers none
  --feat-distill-lambda 0
  --feat-distill-teacher n_a
)

# B-axis: same-param teacher = C_dark fed RAW_normal (== M_DN endpoint), recipe aligned to student.
feat_sameparam_args=(
  --teacher-ckpt "${C_DARK}"
  --teacher-recipe lora_tap_r8a16
  --teacher-input-mode raw_rgb16_normal
  --feat-distill dav2_middeep_cosine
  --feat-distill-layers 5 8 11
  --feat-distill-lambda "${LAMBDA_FEAT}"
  --feat-distill-teacher same_param_clean
)

local_none_args=(
  --raw-ram-local-residual none
  --raw-ram-local-hidden-ch 0
  --raw-ram-local-residual-scale 0
  --raw-ram-local-gate-init 0
  --raw-ram-local-gate-mode n_a
)

local_s03_args=(
  --raw-ram-local-residual noiseaware_v1
  --raw-ram-local-hidden-ch "${LOCAL_HIDDEN}"
  --raw-ram-local-residual-scale "${LOCAL_SCALE_A1}"
  --raw-ram-local-gate-init "${LOCAL_GATE_INIT}"
  --raw-ram-local-gate-mode channel
)

local_s05_args=(
  --raw-ram-local-residual noiseaware_v1
  --raw-ram-local-hidden-ch "${LOCAL_HIDDEN}"
  --raw-ram-local-residual-scale "${LOCAL_SCALE_A2}"
  --raw-ram-local-gate-init "${LOCAL_GATE_INIT}"
  --raw-ram-local-gate-mode channel
)

# --- baseline 3rd seed: pin the run-to-run D1 noise floor ---
run_train "C0_seed7" "${RUN_PREFIX}_C0seed7_lod_raw_pair_lora_decoder_e${EPOCHS}_featnone_localnone" "$((MASTER_PORT + 0))" "none" \
  "${common_args[@]}" --seed 7 "${feat_none_args[@]}" "${local_none_args[@]}"

# --- A-axis: enlarge local residual budget, feat none ---
run_train "A_s03" "${RUN_PREFIX}_A_s03g01_lod_raw_pair_lora_decoder_e${EPOCHS}_featnone_localnoiseaware_s03_g01" "$((MASTER_PORT + 1))" "none" \
  "${common_args[@]}" --seed 42 "${feat_none_args[@]}" "${local_s03_args[@]}"

run_train "A_s05" "${RUN_PREFIX}_A_s05g01_lod_raw_pair_lora_decoder_e${EPOCHS}_featnone_localnoiseaware_s05_g01" "$((MASTER_PORT + 2))" "none" \
  "${common_args[@]}" --seed 42 "${feat_none_args[@]}" "${local_s05_args[@]}"

# --- B-axis: same-param teacher C_dark(RAW_normal), local none ---
run_train "B_sp" "${RUN_PREFIX}_B_sp_lod_raw_pair_lora_decoder_e${EPOCHS}_featmiddeep_sameparam_localnone" "$((MASTER_PORT + 3))" "${C_DARK}" \
  "${common_args[@]}" --seed 42 "${feat_sameparam_args[@]}" "${local_none_args[@]}"

run_train "B_sp_s123" "${RUN_PREFIX}_B_sp_s123_lod_raw_pair_lora_decoder_e${EPOCHS}_featmiddeep_sameparam_localnone" "$((MASTER_PORT + 4))" "${C_DARK}" \
  "${common_args[@]}" --seed 123 "${feat_sameparam_args[@]}" "${local_none_args[@]}"

# --- combination: enlarged local residual + same-param teacher ---
run_train "AB_s05_sp" "${RUN_PREFIX}_AB_s05sp_lod_raw_pair_lora_decoder_e${EPOCHS}_featmiddeep_sameparam_localnoiseaware_s05_g01" "$((MASTER_PORT + 5))" "${C_DARK}" \
  "${common_args[@]}" --seed 42 "${feat_sameparam_args[@]}" "${local_s05_args[@]}"

echo "[QUEUE_DONE] $(date -Iseconds)"
