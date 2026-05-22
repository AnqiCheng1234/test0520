#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/caq/6666_raw/dav2_raw_0520}"
CONDA_ENV="${CONDA_ENV:-dav3}"
WAIT_SECONDS="${WAIT_SECONDS:-10800}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_CACHE_WAIT_SECONDS="${MAX_CACHE_WAIT_SECONDS:-43200}"

CACHE_ROOT="${VKITTI_CACHE_ROOT:-/mnt/drive/1111_new_works/VKITTI2/cache_raw_sensor_linear_dual_644x1008_k1rand_fp32_seed20260516}"
RUN_SCRIPT="${RUN_SCRIPT:-${REPO_ROOT}/finetune_stf/scripts/0516_run_vits_vkitti_lod_rgb_bridge_decoder_ac_speedup_e10.sh}"
RUN_NAME="${RUN_NAME:-0516_vits_vkitti_lod_dn_rgb_bridge_decoder_644x1008_bs8acc1_vkcachek1rand_fp32_ssi_lod3_eth3d_best_speedup_e10}"
MASTER_PORT="${MASTER_PORT:-29788}"

cd "${REPO_ROOT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

cache_expected_rows() {
  conda run -n "${CONDA_ENV}" python -c \
    'import json, sys; from pathlib import Path; path = Path(sys.argv[1]); print(0 if not path.is_file() else int(json.loads(path.read_text(encoding="utf-8")).get("num_samples", 0)))' \
    "${CACHE_ROOT}/config.json"
}

cache_manifest_rows() {
  if [[ ! -f "${CACHE_ROOT}/manifest.jsonl" ]]; then
    echo 0
    return
  fi
  wc -l < "${CACHE_ROOT}/manifest.jsonl"
}

cache_is_complete() {
  [[ -f "${CACHE_ROOT}/config.json" ]] || return 1
  [[ -f "${CACHE_ROOT}/manifest.jsonl" ]] || return 1
  [[ -f "${CACHE_ROOT}/summary.json" ]] || return 1

  local expected rows
  expected="$(cache_expected_rows)"
  rows="$(cache_manifest_rows)"
  [[ "${expected}" =~ ^[0-9]+$ ]] || return 1
  [[ "${rows}" =~ ^[0-9]+$ ]] || return 1
  [[ "${expected}" -gt 0 ]] || return 1
  [[ "${rows}" -eq "${expected}" ]]
}

echo "[$(timestamp)] waiting ${WAIT_SECONDS}s before formal training"
echo "[$(timestamp)] cache_root=${CACHE_ROOT}"
echo "[$(timestamp)] run_name=${RUN_NAME}"
sleep "${WAIT_SECONDS}"

echo "[$(timestamp)] initial wait done; checking cache completeness"
cache_wait_start="$(date +%s)"
while ! cache_is_complete; do
  rows="$(cache_manifest_rows)"
  expected="$(cache_expected_rows || echo 0)"
  echo "[$(timestamp)] cache incomplete rows=${rows}/${expected}; polling again in ${POLL_SECONDS}s"
  now="$(date +%s)"
  if (( now - cache_wait_start >= MAX_CACHE_WAIT_SECONDS )); then
    echo "[$(timestamp)] ERROR: cache did not complete within ${MAX_CACHE_WAIT_SECONDS}s after initial wait" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

echo "[$(timestamp)] cache complete; starting formal run"
exec env \
  CONDA_ENV="${CONDA_ENV}" \
  MASTER_PORT="${MASTER_PORT}" \
  VKITTI_CACHE_ROOT="${CACHE_ROOT}" \
  RUN_NAME="${RUN_NAME}" \
  "${RUN_SCRIPT}"
