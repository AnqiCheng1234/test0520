#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/caq/6666_raw/dav2"
TIMESTAMP="${TIMESTAMP:-$(date +%m%d_%H%M)}"
SESSION="${SESSION:-${TIMESTAMP}_stf_pipeline}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "source /home/caq/anaconda3/bin/activate depth_anything && \
   cd ${PROJECT_ROOT} && \
   TIMESTAMP=${TIMESTAMP} finetune_stf/scripts/run_pipeline.sh"

echo "Started tmux session: ${SESSION}"
echo "Per-experiment logs will be written to each exp directory as train.log"
