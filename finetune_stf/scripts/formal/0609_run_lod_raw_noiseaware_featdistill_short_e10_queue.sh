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
LOD_ROOT="${LOD_ROOT:-/home/caq/6666_raw/0000_dataset/LOD}"
LOD_MANIFEST="${LOD_MANIFEST:-${LOD_ROOT}/pseudo_depth_dav2l_rgb_normal_rel_1200x800/lod_true_rgb_normal_dav2l_rel_manifest_block8_val112_excl10_uniform.csv}"
C_DARK="${C_DARK:-/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
C_NORMAL="${C_NORMAL:-/mnt/drive/3333_raw/0000_exp_ckpt/0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth}"
SESSION_PREFIX="${SESSION_PREFIX:-lod_raw_noiseaware_featdistill_short_e10_queue}"
MASTER_PORT="${MASTER_PORT:-29641}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"
LAMBDA_FEAT="${LAMBDA_FEAT:-0.05}"
RUN_PREFIX="${RUN_PREFIX:-}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_short_e10_queue.sh
  bash finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_short_e10_queue.sh --audit

Runs canonical LoRA short-train matrix:
  L0, L1, L2, L3, L0_seed123

Overrides:
  GPU=0 MASTER_PORT=29641 EPOCHS=10 BS=8 LAMBDA_FEAT=0.05 bash ...
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' C_DARK='${C_DARK}' C_NORMAL='${C_NORMAL}' MASTER_PORT='${MASTER_PORT}' EPOCHS='${EPOCHS}' BS='${BS}' LAMBDA_FEAT='${LAMBDA_FEAT}' RUN_PREFIX='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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
require_file "${C_NORMAL}"
require_dir "${LOD_ROOT}"

log_header() {
  local label="$1"
  local run="$2"
  local port="$3"
  local log="$4"
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
    echo "[TEACHER] ${C_NORMAL}"
    echo "[SEMANTICS] matrix=${label} epochs=${EPOCHS} bs=${BS} lambda_feat=${LAMBDA_FEAT} pair_input=RAW_Dark+RAW_normal aug=hflip0.5_only"
  } 2>&1 | tee -a "${log}"
}

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_lod_noiseaware_audit_${run}"
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
    "raw_ram_rgb_tail": args.raw_ram_rgb_tail,
    "student_init_from": cfg["student_init_from"],
    "student_init_strict": cfg["student_init_strict"],
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
    "lora_tap_layers": cfg["lora_tap_layers"],
    "lora_rank": cfg["lora_rank"],
    "lora_alpha": cfg["lora_alpha"],
    "lora_lr": cfg["lora_lr"],
    "dav2_train_mode": args.dav2_train_mode,
    "best_metric": args.best_metric,
    "best_metric_direction": metric_direction(args.best_metric),
    "epochs": args.epochs,
    "bs": args.bs,
    "lr_schedule": args.lr_schedule,
    "augmentation": resolve_lod_aug_config(args).to_dict(),
    "input_type_alias": cfg["input_type_alias"],
}
print("[AUDIT][OK] " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

run_train() {
  local label="$1"
  local run="$2"
  local port="$3"
  shift 3

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
  log_header "${label}" "${run}" "${port}" "${log}"

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

feat_on_args=(
  --teacher-ckpt "${C_NORMAL}"
  --teacher-recipe lora_tap_r8a16
  --teacher-input-mode raw_rgb16_normal
  --feat-distill dav2_middeep_cosine
  --feat-distill-layers 5 8 11
  --feat-distill-lambda "${LAMBDA_FEAT}"
  --feat-distill-teacher normal_expert
)

local_none_args=(
  --raw-ram-local-residual none
  --raw-ram-local-hidden-ch 0
  --raw-ram-local-residual-scale 0
  --raw-ram-local-gate-init 0
  --raw-ram-local-gate-mode n_a
)

local_on_args=(
  --raw-ram-local-residual noiseaware_v1
  --raw-ram-local-hidden-ch 16
  --raw-ram-local-residual-scale 0.1
  --raw-ram-local-gate-init 0.03
  --raw-ram-local-gate-mode channel
)

run_train "L0" "${RUN_PREFIX}_L0_lod_raw_pair_lora_decoder_e${EPOCHS}_featnone_localnone" "$((MASTER_PORT + 0))" \
  "${common_args[@]}" --seed 42 "${feat_none_args[@]}" "${local_none_args[@]}"

run_train "L1" "${RUN_PREFIX}_L1_lod_raw_pair_lora_decoder_e${EPOCHS}_featmiddeep_lam${LAMBDA_FEAT/./}_localnone" "$((MASTER_PORT + 1))" \
  "${common_args[@]}" --seed 42 "${feat_on_args[@]}" "${local_none_args[@]}"

run_train "L2" "${RUN_PREFIX}_L2_lod_raw_pair_lora_decoder_e${EPOCHS}_featnone_noiseawarev1_gate003_s01" "$((MASTER_PORT + 2))" \
  "${common_args[@]}" --seed 42 "${feat_none_args[@]}" "${local_on_args[@]}"

run_train "L3" "${RUN_PREFIX}_L3_lod_raw_pair_lora_decoder_e${EPOCHS}_featmiddeep_lam${LAMBDA_FEAT/./}_noiseawarev1_gate003_s01" "$((MASTER_PORT + 3))" \
  "${common_args[@]}" --seed 42 "${feat_on_args[@]}" "${local_on_args[@]}"

run_train "L0_seed123" "${RUN_PREFIX}_L0_seed123_lod_raw_pair_lora_decoder_e${EPOCHS}_featnone_localnone" "$((MASTER_PORT + 4))" \
  "${common_args[@]}" --seed 123 "${feat_none_args[@]}" "${local_none_args[@]}"

echo "[QUEUE_DONE] $(date -Iseconds)"
