#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/finetune_stf/exp}"
HEAVY_ROOT="${HEAVY_ROOT:-/mnt/drive/3333_raw/0000_exp_ckpt}"
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
BS="${BS:-8}"
EPOCHS="${EPOCHS:-10}"
timestamp="$(date +%m%d_%H%M)"
DEFAULT_N7_RUN_DIR="${ROOT}/finetune_stf/exp/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10"
N7_RUN_DIR="${N7_RUN_DIR:-${DEFAULT_N7_RUN_DIR}}"
RUN_NAME="${RUN_NAME:-${timestamp}_vkitti_n7rgb_lp0p5_q0p3_lfl0p0_rftna_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs${BS}_e${EPOCHS}}"
SAVE_PATH="${SAVE_PATH:-${OUT_ROOT:-${EXP_ROOT}/${RUN_NAME}}}"
if [[ -n "${HEAVY_SAVE_PATH:-}" ]]; then
  :
elif [[ -n "${OUT_ROOT:-}" ]]; then
  HEAVY_SAVE_PATH="${SAVE_PATH}/heavy"
else
  HEAVY_SAVE_PATH="${HEAVY_ROOT}/${RUN_NAME}"
fi

fail() { echo "[ERROR] $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || fail "Required file not found: $1"; }
require_dir() { [[ -d "$1" ]] || fail "Required directory not found: $1"; }

require_dir "${N7_RUN_DIR}"
require_file "${N7_RUN_DIR}/config.json"
[[ ! -e "${SAVE_PATH}" ]] || fail "SAVE_PATH already exists, refusing to overwrite: ${SAVE_PATH}"
[[ ! -e "${HEAVY_SAVE_PATH}" ]] || fail "HEAVY_SAVE_PATH already exists, refusing to overwrite: ${HEAVY_SAVE_PATH}"

COMMANDS_FILE="${SAVE_PATH}.command.sh"
mkdir -p "$(dirname "${COMMANDS_FILE}")"

python3 - "${N7_RUN_DIR}/config.json" "${COMMANDS_FILE}" "${SAVE_PATH}" "${HEAVY_SAVE_PATH}" "${BS}" "${EPOCHS}" "${SEED}" "${CONDA_BIN}" "${CONDA_ENV}" "${GPU}" "${DEVICE}" "${MAX_TRAIN_STEPS:-}" "${MAX_VAL_SAMPLES:-}" "${MAX_KITTI_VAL_SAMPLES:-}" <<'PY'
import json, shlex, sys
from pathlib import Path

config_path, commands_path, save_path, heavy_save_path, bs, epochs, seed, conda_bin, conda_env, gpu, device, max_train_steps, max_val_samples, max_kitti_val_samples = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text())
arg_keys = [
    "method_id", "input_domain", "model_input_tensor", "dataset_geometry_mode", "raw_storage_format",
    "fullres_even_policy", "rgb_input_space", "depth_target_space", "front_end", "encoder",
    "pretrained_from", "c2_checkpoint", "c2_run_dir", "vkitti_train_list", "vkitti_val_list",
    "eval_protocol", "kitti_val_split", "kitti_base", "kitti_eval_protocol", "kitti_expected_val_samples",
    "kitti_num_workers", "max_kitti_val_samples", "input_height", "input_width", "min_depth", "max_depth",
    "incremental_feature_source", "delta_condition", "gate_condition", "raw_feature_encoder_trainable",
    "residual_alpha", "d0_sign", "lambda_lp", "lowpass_kernel", "q_good", "lambda_final",
    "lambda_boundary", "lambda_grad", "lambda_keep_good_d1", "lambda_gate_sparse", "lambda_lowfreq_loss",
    "lambda_invalid_keep", "unprocessing_method", "vkitti_unprocessing_preset",
    "vkitti_unprocessing_mix_weights", "raw_adapter_backend", "raw_adapter_cfa_pattern",
    "raw_adapter_packed_channel_order", "raw_adapter_rgb_transfer", "raw_adapter_inverse_tone",
    "raw_adapter_ccm", "raw_adapter_red_gain_range", "raw_adapter_blue_gain_range",
    "raw_adapter_fixed_red_gain", "raw_adapter_fixed_blue_gain", "raw_adapter_fixed_light_scale",
    "raw_adapter_dark_light_scale_range", "raw_adapter_over_light_scale_range", "raw_adapter_shot_noise",
    "raw_adapter_read_noise", "raw_adapter_noise_mean_mode", "raw_adapter_black_level",
    "raw_adapter_white_level", "raw_adapter_random_seed_policy", "raw_adapter_external_raw_rgb_root",
    "raw_adapter_external_key", "raw_adapter_external_cache_space", "raw_adapter_variant_policy",
    "raw_adapter_variant_weights", "hflip_prob", "epochs", "bs", "accum_steps", "lr", "weight_decay",
    "num_workers", "log_interval", "save_interval", "eval_interval", "max_train_steps", "max_val_samples",
    "amp_dtype", "seed", "device",
]
bool_keys = {"eval_kitti": "eval-kitti", "save_best_checkpoint": "save-best-checkpoint", "amp": "amp"}
na = "not_applicable"
overrides = {
    "method_id": "N7RGB",
    "input_domain": "rgb",
    "model_input_tensor": "image",
    "front_end": "c2_frozen_rgb_incremental",
    "raw_storage_format": na,
    "unprocessing_method": na,
    "vkitti_unprocessing_preset": na,
    "vkitti_unprocessing_mix_weights": None,
    "incremental_feature_source": "rgb",
    "delta_condition": "feature_d1_stopgrad",
    "gate_condition": "feature_d1",
    "raw_feature_encoder_trainable": na,
    "kitti_eval_protocol": "halfres_rgb_canonical_even_pad_crop_affine_disp",
    "lambda_lp": 0.5,
    "q_good": 0.3,
    "lambda_lowfreq_loss": 0.0,
    "epochs": int(epochs),
    "bs": int(bs),
    "lr": "1e-4",
    "weight_decay": "1e-4",
    "seed": int(seed),
    "device": device,
}
for key in [
    "raw_adapter_backend", "raw_adapter_cfa_pattern", "raw_adapter_packed_channel_order",
    "raw_adapter_rgb_transfer", "raw_adapter_inverse_tone", "raw_adapter_ccm",
    "raw_adapter_red_gain_range", "raw_adapter_blue_gain_range", "raw_adapter_fixed_red_gain",
    "raw_adapter_fixed_blue_gain", "raw_adapter_fixed_light_scale", "raw_adapter_dark_light_scale_range",
    "raw_adapter_over_light_scale_range", "raw_adapter_shot_noise", "raw_adapter_read_noise",
    "raw_adapter_noise_mean_mode", "raw_adapter_black_level", "raw_adapter_white_level",
    "raw_adapter_random_seed_policy", "raw_adapter_external_raw_rgb_root", "raw_adapter_external_key",
    "raw_adapter_external_cache_space", "raw_adapter_variant_policy", "raw_adapter_variant_weights",
]:
    overrides[key] = na
if max_train_steps:
    overrides["max_train_steps"] = int(max_train_steps)
if max_val_samples:
    overrides["max_val_samples"] = int(max_val_samples)
if max_kitti_val_samples:
    overrides["max_kitti_val_samples"] = int(max_kitti_val_samples)

def value_for(key):
    if key in overrides:
        return overrides[key]
    if key == "eval_protocol" and isinstance(cfg.get(key), dict):
        return cfg[key].get("vkitti_val")
    return cfg.get(key)

def add_arg(cmd, key, value):
    if value is None:
        return
    flag = "--" + key.replace("_", "-")
    if isinstance(value, list):
        if value:
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
    elif isinstance(value, dict):
        cmd.extend([flag, json.dumps(value, sort_keys=True)])
    else:
        cmd.extend([flag, str(value)])

cmd = [f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))}", shlex.quote(conda_bin), "run", "--live-stream", "-n", shlex.quote(conda_env), "python", "foundation/tools/train_vkitti2_incremental_residual.py"]
for bool_key, flag in bool_keys.items():
    if bool(cfg.get(bool_key)):
        cmd.append("--" + flag)
    elif bool_key == "amp":
        cmd.append("--no-amp")
for key in arg_keys:
    add_arg(cmd, key, value_for(key))
cmd.extend([
    "--train-feature-ablation-mode", "true",
    "--eval-feature-ablation-mode", "true",
    "--feature-ablation-scope", "both",
    "--feature-ablation-key", "x3",
    "--feature-ablation-seed", str(seed),
    "--experiment-label", "N7-RGB",
    "--save-path", save_path,
    "--heavy-save-path", heavy_save_path,
])
Path(commands_path).write_text(" ".join(part if part.startswith("CUDA_VISIBLE_DEVICES=") else shlex.quote(part) for part in cmd) + "\n", encoding="utf-8")
PY

mkdir -p "${SAVE_PATH}"
LOG_PATH="${SAVE_PATH}/n7_rgb_matched_control.log"
{
  printf '[INFO] host=%s user=%s pwd=%s\n' "$(hostname)" "$(whoami)" "$(pwd)"
  echo "[INFO] N7_RUN_DIR=${N7_RUN_DIR}"
  echo "[INFO] SAVE_PATH=${SAVE_PATH}"
  echo "[INFO] HEAVY_SAVE_PATH=${HEAVY_SAVE_PATH}"
  echo "[INFO] N7RGB differs from N3 because delta uses concat(F_rgb, stopgrad(F_d1)); N3 delta uses F_rgb only."
  echo "[CMD] $(cat "${COMMANDS_FILE}")"
} 2>&1 | tee -a "${LOG_PATH}"

bash -lc "cd '${ROOT}' && $(cat "${COMMANDS_FILE}")" 2>&1 | tee -a "${LOG_PATH}"
echo "[DONE] N7 RGB matched control outputs: ${SAVE_PATH}" | tee -a "${LOG_PATH}"
