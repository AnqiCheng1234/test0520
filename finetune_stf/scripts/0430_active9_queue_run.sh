#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

runs=(
  "0430_a1_run.sh"
  "0430_a2_run.sh"
  "0430_a3_run.sh"
  "0430_b1_run.sh"
  "0430_b2_run.sh"
  "0430_c1_run.sh"
  "0430_c2_run.sh"
  "0430_c3_run.sh"
  "0430_c4_run.sh"
)

for run_script in "${runs[@]}"; do
  echo "[QUEUE] start ${run_script} at $(date '+%F %T')"
  "${SCRIPT_DIR}/${run_script}" "$@"
  echo "[QUEUE] finish ${run_script} at $(date '+%F %T')"
  echo ""
done

echo "[QUEUE] all 9 experiments completed at $(date '+%F %T')"
