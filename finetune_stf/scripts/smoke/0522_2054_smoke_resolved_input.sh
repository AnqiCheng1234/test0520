#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"

HELP_LOG="${HELP_LOG:-/tmp/dav2_raw_0522_train_help_codex_smoke.txt}"
INPUT_LOG="${INPUT_LOG:-/tmp/dav2_raw_0522_model_input_codex_smoke.txt}"
LIVE_RAW_LOG="${LIVE_RAW_LOG:-/tmp/dav2_raw_0522_live_raw_model_codex_smoke.txt}"

cd "${ROOT}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m compileall finetune_stf foundation
"${CONDA_BIN}" run -n "${CONDA_ENV}" python finetune_stf/train.py --help > "${HELP_LOG}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY' > "${INPUT_LOG}"
import torch

from finetune_stf.config import resolve_legacy_input_type
from finetune_stf.util.model_input import select_model_input

sample = {
    "image": torch.zeros(1, 3, 16, 16),
    "raw": torch.ones(1, 4, 16, 16),
}

image = select_model_input(sample, "image", dataset_family="codex_smoke")
raw = select_model_input(sample, "raw", dataset_family="codex_smoke")
assert torch.all(image == 0)
assert torch.all(raw == 1)

for name in ("rgb", "rgb_lora", "raw_ram", "raw_ram_rgb_bridge_feature_adapter"):
    resolved = resolve_legacy_input_type(name)
    for key in ("input_domain", "front_end", "dataset_family", "model_input_tensor"):
        assert key in resolved, (name, key, resolved)
    print(name, resolved["input_domain"], resolved["front_end"], resolved["model_input_tensor"])
PY

set +e
"${CONDA_BIN}" run -n "${CONDA_ENV}" python finetune_stf/train.py \
  --pretrained-from /tmp/dav2_raw_0522_missing_pretrained_codex_smoke.pth \
  --save-path /tmp/dav2_raw_0522_live_raw_model_codex_smoke \
  --eval-kitti \
  --kitti-eval-protocol live_raw_model \
  > "${LIVE_RAW_LOG}" 2>&1
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
  echo "[SMOKE][FAIL] live_raw_model unexpectedly passed" >&2
  echo "[SMOKE][KEEP] ${LIVE_RAW_LOG}" >&2
  exit 1
fi
grep -q "live_raw_model is reserved" "${LIVE_RAW_LOG}"

rm -f "${HELP_LOG}" "${INPUT_LOG}" "${LIVE_RAW_LOG}"
