#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -gt 0 && -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT="$1"
  shift
fi

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Pass the checkpoint as the first argument or set CHECKPOINT=/path/to/ckpt" >&2
  exit 1
fi

ENCODER="${ENCODER:-vitl}"
STF_ROOT="${STF_ROOT:-/home/caq/6666_raw/seeingthroughfog}"
RAW_NPZ_ROOT="${RAW_NPZ_ROOT:-/mnt/drive/3333_raw/seeing_through_fog/cam_stereo_left_bayer_rect/npz}"
SAVE_DIR="${SAVE_DIR:-anqi_eval/results}"

python anqi_eval/eval_stf_rel_depth.py \
  --encoder "${ENCODER}" \
  --checkpoint "${CHECKPOINT}" \
  --input-type raw_ram \
  --stf-root "${STF_ROOT}" \
  --raw-npz-root "${RAW_NPZ_ROOT}" \
  --split val \
  --input-height 518 \
  --input-width 966 \
  --min-depth 1.0 \
  --max-depth 80.0 \
  --save-dir "${SAVE_DIR}" \
  "$@"
