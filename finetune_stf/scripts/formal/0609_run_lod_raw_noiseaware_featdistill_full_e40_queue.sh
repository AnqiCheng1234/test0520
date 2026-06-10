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
SESSION_PREFIX="${SESSION_PREFIX:-lod_raw_noiseaware_featdistill_full_e40_queue}"
MASTER_PORT="${MASTER_PORT:-29651}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
EPOCHS="${EPOCHS:-40}"
BS="${BS:-8}"
LAMBDA_FEAT="${LAMBDA_FEAT:-0.05}"
FULL_GROUPS="${FULL_GROUPS:-L3}"
RUN_PREFIX="${RUN_PREFIX:-}"

usage() {
  cat <<'EOF'
Usage:
  bash finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_full_e40_queue.sh
  bash finetune_stf/scripts/formal/0609_run_lod_raw_noiseaware_featdistill_full_e40_queue.sh --audit

Default runs only L3 full training. Override with:
  FULL_GROUPS=L1,L2,L3 LAMBDA_FEAT=0.05 bash ...

Only launch groups that passed the short-train gate.
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
    "cd '${ROOT}' && ROOT='${ROOT}' EXP_ROOT='${EXP_ROOT}' LOG_ROOT='${LOG_ROOT}' HEAVY_ROOT='${HEAVY_ROOT}' CONDA_BIN='${CONDA_BIN}' CONDA_ENV='${CONDA_ENV}' GPU='${GPU}' PRETRAINED='${PRETRAINED}' LOD_ROOT='${LOD_ROOT}' LOD_MANIFEST='${LOD_MANIFEST}' C_DARK='${C_DARK}' C_NORMAL='${C_NORMAL}' MASTER_PORT='${MASTER_PORT}' EPOCHS='${EPOCHS}' BS='${BS}' LAMBDA_FEAT='${LAMBDA_FEAT}' FULL_GROUPS='${FULL_GROUPS}' RUN_PREFIX='${queue_timestamp}' QUEUE_SESSION='${session}' CUDA_VISIBLE_DEVICES='${GPU}' bash '$0' --run-internal 2>&1 | tee -a '${queue_log}'"
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

audit_train_args() {
  local run="$1"
  shift
  local audit_save="/tmp/codex_smoke_lod_noiseaware_full_audit_${run}"
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
    "epochs": args.epochs,
    "bs": args.bs,
    "dataset_family": cfg["dataset_family"],
    "dataset_input_mode": cfg["dataset_input_mode"],
    "student_init_from": cfg["student_init_from"],
    "teacher_ckpt": cfg["teacher_ckpt"],
    "feat_distill": cfg["feat_distill"],
    "feat_distill_layers": cfg["feat_distill_layers"],
    "feat_distill_lambda": cfg["feat_distill_lambda"],
    "raw_ram_local_residual": cfg["raw_ram_local_residual"],
    "raw_ram_local_residual_scale": cfg["raw_ram_local_residual_scale"],
    "raw_ram_local_gate_init": cfg["raw_ram_local_gate_init"],
    "lora": cfg["lora"],
    "lora_tap_layers": cfg["lora_tap_layers"],
    "best_metric": args.best_metric,
    "best_metric_direction": metric_direction(args.best_metric),
    "augmentation": resolve_lod_aug_config(args).to_dict(),
    "input_type_alias": cfg["input_type_alias"],
}
print("[AUDIT][OK] " + json.dumps(payload, sort_keys=True), flush=True)
PY
}

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
    echo "[SEMANTICS] full groups=${FULL_GROUPS} epochs=${EPOCHS} bs=${BS} lambda_feat=${LAMBDA_FEAT}"
  } 2>&1 | tee -a "${log}"
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
  CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" \
    torchrun --nproc_per_node=1 --master_port="${port}" finetune_stf/train.py \
    "$@" \
    --pretrained-from "${PRETRAINED}" \
    --heavy-save-root "${HEAVY_ROOT}" \
    --save-path "${save}" 2>&1 | tee -a "${log}"
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

IFS=',' read -r -a requested_groups <<< "${FULL_GROUPS}"
for idx in "${!requested_groups[@]}"; do
  group="$(echo "${requested_groups[$idx]}" | xargs)"
  case "${group}" in
    L1)
      run_train "L1_full" "${RUN_PREFIX}_L1_lod_true_raw_dark_pair_featmiddeep_lora_tap_r8a16_decoder_e${EPOCHS}_lam${LAMBDA_FEAT/./}_localnone" "$((MASTER_PORT + idx))" \
        "${common_args[@]}" --seed 42 "${feat_on_args[@]}" "${local_none_args[@]}"
      ;;
    L2)
      run_train "L2_full" "${RUN_PREFIX}_L2_lod_true_raw_dark_pair_noiseawarev1_lora_tap_r8a16_decoder_e${EPOCHS}_gate003_s01" "$((MASTER_PORT + idx))" \
        "${common_args[@]}" --seed 42 "${feat_none_args[@]}" "${local_on_args[@]}"
      ;;
    L3)
      run_train "L3_full" "${RUN_PREFIX}_L3_lod_true_raw_dark_pair_noiseawarev1_featmiddeep_lora_tap_r8a16_decoder_e${EPOCHS}_lam${LAMBDA_FEAT/./}_gate003_s01" "$((MASTER_PORT + idx))" \
        "${common_args[@]}" --seed 42 "${feat_on_args[@]}" "${local_on_args[@]}"
      ;;
    *)
      echo "[ERROR] Unsupported FULL_GROUPS item: ${group}. Expected L1,L2,L3" >&2
      exit 2
      ;;
  esac
done

echo "[QUEUE_DONE] $(date -Iseconds)"
