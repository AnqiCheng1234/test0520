#!/usr/bin/env bash
# Short-train batch launcher for LOD RAW degradation diagnosis (plan §6.3).
# Runs cold-start (from pretrained, NOT resume L0) short trains for the
# noise-vs-exposure dissociation set, ≥2 seeds, writes best_model by lod_d1.
#
# Usage:
#   SET=anchors bash tools/lod_raw_short_train_batch.sh        # A1 dark_identity + A2 normal_identity
#   SET=degrade bash tools/lod_raw_short_train_batch.sh        # E7 exposure_only + E8 noise_only
# Env overrides: EPOCHS (default 5), MAXSTEPS (default 80), SEEDS (default "42 123")
set -euo pipefail

SET="${SET:-anchors}"
EPOCHS="${EPOCHS:-5}"
MAXSTEPS="${MAXSTEPS:-80}"
SEEDS="${SEEDS:-42 123}"

REPO=/home/caq/6666_raw/dav2_raw_0603
MAN_DIR=/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/ablation_inputs/manifests
LIGHT_ROOT=$REPO/finetune_stf/exp/lod_raw_diag/short_train
HEAVY_ROOT=/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/short_train
LOG_DIR=/mnt/drive/3333_raw/0000_exp_ckpt/lod_raw_diag/logs
PRETRAINED=/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
LOD_ROOT=/home/caq/6666_raw/0000_dataset/LOD
mkdir -p "$LIGHT_ROOT" "$HEAVY_ROOT" "$LOG_DIR"

# job := "SHORT_NAME|DATASET_FAMILY|DATASET_INPUT_MODE|MANIFEST_BASENAME"
if [ "$SET" = "anchors" ]; then
  JOBS=(
    "raw_dark_identity|lod_true_raw_dark_rgb16|raw_rgb16_dark|raw_dark_identity.csv"
    "raw_normal_identity|lod_true_raw_normal_rgb16|raw_rgb16_normal|raw_normal_identity.csv"
  )
elif [ "$SET" = "degrade" ]; then
  JOBS=(
    "raw_normal_to_dark_exposure|lod_true_raw_normal_rgb16|raw_rgb16_normal|raw_normal_to_dark_exposure_trainfit.csv"
    "raw_normal_to_dark_noise_only|lod_true_raw_normal_rgb16|raw_rgb16_normal|raw_normal_to_dark_noise_only_trainfit.csv"
  )
else
  echo "unknown SET=$SET (use anchors|degrade)"; exit 2
fi

PORT=29670
SUMMARY=$HEAVY_ROOT/short_train_summary_${SET}_e${EPOCHS}_s${MAXSTEPS}.csv
echo "short_name,seed,best_lod_d1,returncode,run_dir,log" > "$SUMMARY"

for job in "${JOBS[@]}"; do
  IFS='|' read -r SHORT FAMILY INPUT_MODE MAN <<< "$job"
  for SEED in $SEEDS; do
    RUN_NAME=${SHORT}_e${EPOCHS}_s${MAXSTEPS}_seed${SEED}
    LOG=$LOG_DIR/0609_short_train_${RUN_NAME}.log
    echo "[QUEUE] launching $RUN_NAME (family=$FAMILY mode=$INPUT_MODE port=$PORT)"
    set +e
    conda run --live-stream -n dav3 torchrun --nproc_per_node=1 --master_port ${PORT} ${REPO}/finetune_stf/train.py \
      --stage lod_only \
      --encoder vits \
      --dataset-family "$FAMILY" \
      --dataset-input-mode "$INPUT_MODE" \
      --input-domain raw3 --front-end raw_rgb16_ram3 --model-input-tensor raw \
      --bridge none --decoder-feature-adapter none \
      --pretrained-from "$PRETRAINED" \
      --lod-root "$LOD_ROOT" \
      --lod-manifest "$MAN_DIR/$MAN" \
      --lod-label-space inverse_relative --lod-train-crop-mode random --lod-val-crop-mode center \
      --raw-storage-format raw_rgb16_png_3ch --lod-raw-norm-mode uint16_div_65535 --raw-ram-rgb-tail identity \
      --lora dav2_lora --lora-block-mode tap --lora-tap-layers 2 5 8 11 \
      --lora-rank 8 --lora-alpha 16 --lora-lr 5e-5 --raw-front-end-lr 5e-5 \
      --dav2-train-mode decoder --backbone-layer-decay 1.0 \
      --lr 1e-5 --lr-schedule poly --warmup-steps 0 --loss-type ssi \
      --loss-target-normalization --loss-norm-min-scale 1e-3 \
      --bs 8 --accum-steps 1 --input-height 512 --input-width 960 \
      --aug-preset off --aug-hflip-prob 0.5 \
      --eval-lod --no-eval-stf --eval-lod-train-proxy --lod-train-proxy-count 112 \
      --amp --amp-dtype bf16 --num-workers 4 --log-interval 500 \
      --no-enable-fixed-viz-dump --no-enable-train-source-viz-dump \
      --save-best-checkpoint --best-metric lod_d1 \
      --epochs ${EPOCHS} --debug-max-train-steps ${MAXSTEPS} \
      --save-path ${LIGHT_ROOT}/${RUN_NAME} \
      --heavy-save-root ${HEAVY_ROOT} \
      --seed ${SEED} > "$LOG" 2>&1
    RC=$?
    set -e
    BEST=$(grep -oE "best_lod_d1 improved to [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$" || echo "NA")
    echo "[QUEUE] done $RUN_NAME rc=$RC best_lod_d1=$BEST"
    echo "${SHORT},${SEED},${BEST},${RC},${HEAVY_ROOT}/${RUN_NAME},${LOG}" >> "$SUMMARY"
    PORT=$((PORT+1))
  done
done

echo "[QUEUE] SET=$SET complete. summary: $SUMMARY"
column -s, -t "$SUMMARY" || cat "$SUMMARY"
