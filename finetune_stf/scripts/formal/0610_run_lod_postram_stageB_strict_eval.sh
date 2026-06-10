#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-dav3}"
RUN_STAMP="${RUN_STAMP:-$(date +%m%d_%H%M)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/finetune_stf/analysis/lod_postram_stageB_strict/${RUN_STAMP}_lod_postram_stageB_strict}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
AUDIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit)
      AUDIT=1
      shift
      ;;
    *)
      echo "[ERROR] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  PY=(python)
else
  PY=(conda run -n "${CONDA_ENV}" python)
fi

NORMAL_RUN_ID="0609_0044_lod_true_raw_normal_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly"
NORMAL_RUN_DIR="${ROOT}/finetune_stf/exp/${NORMAL_RUN_ID}"
NORMAL_CKPT="/mnt/drive/3333_raw/0000_exp_ckpt/${NORMAL_RUN_ID}/best_model.pth"
STAGE_A_NOOP="${ROOT}/finetune_stf/analysis/lod_postram_external/0610_0050_lod_postram_ext_sanity/EXT_NOOP_A0/metrics.json"

RUNS=(
  "REF_C_DARK_0608|0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly|/mnt/drive/3333_raw/0000_exp_ckpt/0608_2026_lod_true_raw_rgb16_block8excl10_ram3_dav2s_ram_lora_tap_r8a16_decoder_e40_flip05_poly/best_model.pth"
  "B0_seed42|0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42|/mnt/drive/3333_raw/0000_exp_ckpt/0610_0054_B0_LORA_s01_lod_postram_cleanup_e10_seed42/best_model.pth"
  "T_seed42|0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42|/mnt/drive/3333_raw/0000_exp_ckpt/0610_0103_T_LORA_s01_lod_postram_cleanup_e10_seed42/best_model.pth"
  "B0_seed123|0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123|/mnt/drive/3333_raw/0000_exp_ckpt/0610_0119_B0_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth"
  "T_seed123|0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123|/mnt/drive/3333_raw/0000_exp_ckpt/0610_0128_T_LORA_s01_lod_postram_cleanup_e10_seed123/best_model.pth"
  "B0_seed777|0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777|/mnt/drive/3333_raw/0000_exp_ckpt/0610_0144_B0_LORA_s01_lod_postram_cleanup_e10_seed777/best_model.pth"
  "T_seed777|0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777|/mnt/drive/3333_raw/0000_exp_ckpt/0610_0153_T_LORA_s01_lod_postram_cleanup_e10_seed777/best_model.pth"
)

missing=()
for item in "${RUNS[@]}"; do
  IFS="|" read -r label run_id ckpt <<<"${item}"
  run_dir="${ROOT}/finetune_stf/exp/${run_id}"
  [[ -f "${run_dir}/config.json" ]] || missing+=("${label} config missing: ${run_dir}/config.json")
  [[ -f "${run_dir}/resolved_config.json" ]] || missing+=("${label} resolved_config missing: ${run_dir}/resolved_config.json")
  [[ -f "${ckpt}" ]] || missing+=("${label} checkpoint missing: ${ckpt}")
done
[[ -f "${NORMAL_RUN_DIR}/config.json" ]] || missing+=("normal config missing: ${NORMAL_RUN_DIR}/config.json")
[[ -f "${NORMAL_RUN_DIR}/resolved_config.json" ]] || missing+=("normal resolved_config missing: ${NORMAL_RUN_DIR}/resolved_config.json")
[[ -f "${NORMAL_CKPT}" ]] || missing+=("normal checkpoint missing: ${NORMAL_CKPT}")
[[ -f "${STAGE_A_NOOP}" ]] || missing+=("stage A noop metrics missing: ${STAGE_A_NOOP}")

if (( ${#missing[@]} > 0 )); then
  printf '[ERROR] missing required inputs:\n' >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

if (( AUDIT == 1 )); then
  echo "[AUDIT] OK"
  echo "[AUDIT] output_root=${OUT_ROOT}"
  echo "[AUDIT] device=${DEVICE} num_workers=${NUM_WORKERS} eval_batch_size=${EVAL_BATCH_SIZE}"
  exit 0
fi

mkdir -p "${OUT_ROOT}/logs"
MANIFEST="${OUT_ROOT}/manifest.tsv"
printf 'label\trun_id\tcheckpoint\n' > "${MANIFEST}"

for item in "${RUNS[@]}"; do
  IFS="|" read -r label run_id ckpt <<<"${item}"
  run_dir="${ROOT}/finetune_stf/exp/${run_id}"
  eval_dir="${OUT_ROOT}/${label}"
  log_path="${OUT_ROOT}/logs/${label}.log"
  printf '%s\t%s\t%s\n' "${label}" "${run_id}" "${ckpt}" >> "${MANIFEST}"
  echo "[STRICT] start label=${label} run=${run_id}"
  cmd=(
    "${PY[@]}" "${ROOT}/tools/lod_raw_cross_eval.py"
    --dark-run-dir "${run_dir}"
    --dark-checkpoint "${ckpt}"
    --normal-run-dir "${NORMAL_RUN_DIR}"
    --normal-checkpoint "${NORMAL_CKPT}"
    --output-dir "${eval_dir}"
    --variants strict
    --combos M_DD
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --progress-interval "${PROGRESS_INTERVAL}"
  )
  if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
    cmd+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
  fi
  "${cmd[@]}" 2>&1 | tee "${log_path}"
  echo "[STRICT] done label=${label}"
done

"${PY[@]}" "${ROOT}/tools/lod_postram_stageb_summary.py" \
  --eval-root "${OUT_ROOT}" \
  --stage-a-noop-metrics "${STAGE_A_NOOP}" \
  --output-csv "${OUT_ROOT}/stageB_strict_summary.csv" \
  --output-md "${OUT_ROOT}/summary.md" \
  2>&1 | tee "${OUT_ROOT}/logs/summary.log"

echo "[STRICT] wrote ${OUT_ROOT}/summary.md"
