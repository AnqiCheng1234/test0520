#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-conda}"

if [[ -z "${PRETRAINED:-}" ]]; then
  if [[ -f /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth ]]; then
    PRETRAINED="/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth"
  else
    PRETRAINED="/mnt/drive/3333_raw/checkpoints/depth_anything_v2_vits.pth"
  fi
fi

LOD_DAY_MANIFEST="${LOD_DAY_MANIFEST:-/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_day_rel_1440x928/lod_day_dav2_rel_manifest_subset50_split_seed42.csv}"
LOD_NIGHT_MANIFEST="${LOD_NIGHT_MANIFEST:-/mnt/drive/3333_raw/LOD/pseudo_depth_dav2_night_rel_1440x928/lod_night_dav2_rel_manifest_subset50_split_seed42.csv}"
GPU="${GPU:-0}"
PORT="${PORT:-29651}"
INPUT_TYPE="${INPUT_TYPE:-raw_ram_rgb}"

case "${INPUT_TYPE}" in
  raw_ram_rgb|raw_ram_rgb_bridge|raw_ram_rgb_bridge_lora) ;;
  *)
    echo "[ERROR] INPUT_TYPE must be raw_ram_rgb, raw_ram_rgb_bridge, or raw_ram_rgb_bridge_lora; got ${INPUT_TYPE}" >&2
    exit 2
    ;;
esac

if [[ "${PHASE1_BNCLEAN_REVIEWED:-0}" != "1" ]]; then
  cat >&2 <<'MSG'
[STOP] Phase-1 BN-clean guard
This run would use the 2026-05-19 Phase-1 change:
- raw_ram_rgb removes torch.clamp(x3, 0, 1) and ImageNet normalization after RamCore3 BN.
- raw_ram_rgb_bridge/raw_ram_rgb_bridge_lora now use the same no-clamp/no-ImageNet-norm path.

Before launching, re-audit:
- finetune_stf/models/raw_ram.py
- finetune_stf/models/lora_bridge.py
- plans/0519_aaa_important/phase1_lod_night_only_plan.md

If this is still intended, rerun with:
PHASE1_BNCLEAN_REVIEWED=1 bash finetune_stf/scripts/0519_phaseA_lodnight_bnclean_smoke.sh
MSG
  exit 3
fi

mkdir -p "${EXP_ROOT}"
cd "${ROOT}"

ts="$(date +%m%d_%H%M)"
RUN_NAME="${RUN_NAME:-${ts}_phase1_lodnight_${INPUT_TYPE}_bnclean_smoke200}"
SESSION="${SESSION:-${ts}_phase1_lodnight_bnclean_smoke_gpu${GPU}}"
SAVE="${EXP_ROOT}/${RUN_NAME}"
LOG="${SAVE}/tmux_launch.log"
ENTRY="${SAVE}/tmux_entry.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[ERROR] Refusing to reuse existing tmux session: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${SAVE}" ]]; then
  echo "[ERROR] Refusing to overwrite existing run directory: ${SAVE}" >&2
  exit 2
fi

mkdir -p "${SAVE}"
cat > "${ENTRY}" <<EOF
#!/usr/bin/env bash
set -u
cd "${ROOT}"
export PHASE1_BNCLEAN_REVIEWED=1
{
  echo "[RUN] ${RUN_NAME}"
  echo "[TMUX] ${SESSION}"
  echo "[HOST] \$(hostname) [USER] \$(whoami) [PWD] \$(pwd)"
  echo "[START] \$(date -Iseconds)"
  echo "[GPU] CUDA_VISIBLE_DEVICES=${GPU}"
  echo "[NOTICE] Phase-1 BN-clean: ${INPUT_TYPE} uses RamCore3 BN output directly; no clamp and no ImageNet norm after x3."
  echo "[CMD] PHASE1_BNCLEAN_REVIEWED=1 CUDA_VISIBLE_DEVICES=${GPU} ${CONDA_BIN} run -n dav3 torchrun --nproc_per_node=1 --master_port=${PORT} finetune_stf/train.py --encoder vits --stage raw_mix --train-sources lod_night --train-source-ratios 1.0 --train-steps-per-epoch 200 --input-type ${INPUT_TYPE} --norm-mode sensor_linear --dav2-train-mode full --backbone-layer-decay 0.9 --lod-day-manifest ${LOD_DAY_MANIFEST} --lod-night-manifest ${LOD_NIGHT_MANIFEST} --lod-crop-mode random --input-height 518 --input-width 812 --bs 8 --accum-steps 1 --epochs 1 --lr 1e-5 --loss-type ssi --loss-target-normalization --amp --amp-dtype bf16 --seed 42 --num-workers 8 --log-interval 50 --no-eval-stf --eval-robotcar-night --robotcar-night-fast-eval-backend sparse --robotcar-night-max-samples 30 --best-metric robotcar_night --save-best-checkpoint --pretrained-from ${PRETRAINED} --heavy-save-root ${HEAVY_ROOT} --save-path ${SAVE}"
} 2>&1 | tee -a "${LOG}"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_BIN}" run -n dav3 torchrun --nproc_per_node=1 --master_port="${PORT}" finetune_stf/train.py \\
  --encoder vits \\
  --stage raw_mix \\
  --train-sources lod_night \\
  --train-source-ratios 1.0 \\
  --train-steps-per-epoch 200 \\
  --input-type "${INPUT_TYPE}" \\
  --norm-mode sensor_linear \\
  --dav2-train-mode full \\
  --backbone-layer-decay 0.9 \\
  --lod-day-manifest "${LOD_DAY_MANIFEST}" \\
  --lod-night-manifest "${LOD_NIGHT_MANIFEST}" \\
  --lod-crop-mode random \\
  --input-height 518 \\
  --input-width 812 \\
  --bs 8 \\
  --accum-steps 1 \\
  --epochs 1 \\
  --lr 1e-5 \\
  --loss-type ssi \\
  --loss-target-normalization \\
  --amp \\
  --amp-dtype bf16 \\
  --seed 42 \\
  --num-workers 8 \\
  --log-interval 50 \\
  --no-eval-stf \\
  --eval-robotcar-night \\
  --robotcar-night-fast-eval-backend sparse \\
  --robotcar-night-max-samples 30 \\
  --best-metric robotcar_night \\
  --save-best-checkpoint \\
  --pretrained-from "${PRETRAINED}" \\
  --heavy-save-root "${HEAVY_ROOT}" \\
  --save-path "${SAVE}" 2>&1 | tee -a "${LOG}"
status=\${PIPESTATUS[0]}
set -e
echo "[END] \$(date -Iseconds) status=\${status}" 2>&1 | tee -a "${LOG}"
exit "\${status}"
EOF
chmod +x "${ENTRY}"

tmux new-session -d -s "${SESSION}" "${ENTRY}"

echo "[LAUNCHED] ${RUN_NAME}"
echo "session=${SESSION}"
echo "log=${LOG}"
echo "attach=tmux attach -t ${SESSION}"
echo "tail=tail -f ${LOG}"
