#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
BS="${BS:-8}"
RUN_EXTRA_SCOPES="${RUN_EXTRA_SCOPES:-0}"
timestamp="$(date +%m%d_%H%M)"
DEFAULT_N7_RUN_DIR="${ROOT}/finetune_stf/exp/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10"
DEFAULT_N7_CKPT="/mnt/drive/3333_raw/0000_exp_ckpt/0530_0216_vkitti_n7_x3_lp0p5_q0p3_lfl0p0_rfttrue_vits_half187x621_sceneholdout_Scene20_n1000_seed42_bs8_e10/epoch_09.pth"
N7_RUN_DIR="${N7_RUN_DIR:-${DEFAULT_N7_RUN_DIR}}"
N7_CKPT="${N7_CKPT:-${DEFAULT_N7_CKPT}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/plans/0527/diagnostics/0530_n7_eval_x3_ablation_${timestamp}}"

fail() { echo "[ERROR] $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || fail "Required file not found: $1"; }
require_dir() { [[ -d "$1" ]] || fail "Required directory not found: $1"; }

require_dir "${N7_RUN_DIR}"
require_file "${N7_CKPT}"
require_file "${N7_RUN_DIR}/config.json"
[[ ! -e "${OUT_ROOT}" ]] || fail "OUT_ROOT already exists, refusing to overwrite: ${OUT_ROOT}"

read -r cfg_c2_ckpt cfg_c2_run_dir < <(python3 - "${N7_RUN_DIR}/config.json" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("c2_checkpoint", ""), cfg.get("c2_run_dir", ""))
PY
)
C2_CKPT="${C2_CKPT:-${C2_CHECKPOINT:-${cfg_c2_ckpt}}}"
C2_RUN_DIR="${C2_RUN_DIR:-${cfg_c2_run_dir}}"
require_file "${C2_CKPT}"
require_dir "${C2_RUN_DIR}"

mkdir -p "${OUT_ROOT}"
LOG_PATH="${OUT_ROOT}/n7_eval_x3_ablation.log"
COMMANDS_FILE="${OUT_ROOT}/n7_eval_x3_ablation_commands.sh"

python3 - "${N7_RUN_DIR}/config.json" "${COMMANDS_FILE}" "${OUT_ROOT}" "${N7_CKPT}" "${C2_CKPT}" "${C2_RUN_DIR}" "${BS}" "${SEED}" "${CONDA_BIN}" "${CONDA_ENV}" "${GPU}" "${DEVICE}" "${RUN_EXTRA_SCOPES}" "${MAX_VAL_SAMPLES:-}" "${MAX_KITTI_VAL_SAMPLES:-}" <<'PY'
import json, shlex, sys
from pathlib import Path

config_path, commands_path, out_root, n7_ckpt, c2_ckpt, c2_run_dir, bs, seed, conda_bin, conda_env, gpu, device, run_extra_scopes, max_val_samples, max_kitti_val_samples = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text())
if str(cfg.get("method_id", "")).upper() != "N7":
    raise SystemExit(f"N7 config method_id must be N7, got {cfg.get('method_id')!r}")
required = {
    "incremental_feature_source": "x3",
    "delta_condition": "feature_d1_stopgrad",
    "gate_condition": "feature_d1",
    "raw_feature_encoder_trainable": "true",
}
for key, expected in required.items():
    if str(cfg.get(key)) != expected:
        raise SystemExit(f"N7 config {key} must be {expected!r}, got {cfg.get(key)!r}")

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

def value_for(key):
    if key == "eval_protocol" and isinstance(cfg.get(key), dict):
        return cfg[key].get("vkitti_val")
    if key == "c2_checkpoint":
        return c2_ckpt
    if key == "c2_run_dir":
        return c2_run_dir
    if key == "bs":
        return int(bs)
    if key == "seed":
        return int(seed)
    if key == "device":
        return device
    if key == "max_val_samples":
        return int(max_val_samples) if max_val_samples else cfg.get(key)
    if key == "max_kitti_val_samples":
        return int(max_kitti_val_samples) if max_kitti_val_samples else cfg.get(key)
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

pairs = [("true", "both"), ("shuffle", "both"), ("zero", "both"), ("mean", "both")]
if str(run_extra_scopes) == "1":
    pairs.extend([("shuffle", "delta"), ("shuffle", "gate")])
lines = []
for mode, scope in pairs:
    mode_out = str(Path(out_root) / f"mode_{mode}_scope_{scope}")
    cmd = [f"CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))}", shlex.quote(conda_bin), "run", "--live-stream", "-n", shlex.quote(conda_env), "python", "foundation/tools/train_vkitti2_incremental_residual.py"]
    for bool_key, flag in bool_keys.items():
        if bool(cfg.get(bool_key)):
            cmd.append("--" + flag)
        elif bool_key == "amp":
            cmd.append("--no-amp")
    randomize = cfg.get("randomize_unprocessing")
    if randomize is True:
        cmd.append("--randomize-unprocessing")
    elif randomize is False:
        cmd.append("--no-randomize-unprocessing")
    for key in arg_keys:
        add_arg(cmd, key, value_for(key))
    cmd.extend([
        "--eval-only", "--resume-from", str(n7_ckpt),
        "--train-feature-ablation-mode", "true",
        "--eval-feature-ablation-mode", mode,
        "--feature-ablation-scope", scope,
        "--feature-ablation-key", "x3",
        "--feature-ablation-seed", str(seed),
        "--feature-ablation-donor-offset", "1",
        "--experiment-label", "N7-eval-x3-ablation",
        "--n6-output-dir", mode_out,
        "--save-path", str(Path(out_root) / "driver"),
        "--heavy-save-path", str(Path(out_root) / "heavy_unused"),
    ])
    lines.append(" ".join(part if part.startswith("CUDA_VISIBLE_DEVICES=") else shlex.quote(part) for part in cmd))
Path(commands_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

{
  printf '[INFO] host=%s user=%s pwd=%s\n' "$(hostname)" "$(whoami)" "$(pwd)"
  echo "[INFO] N7_RUN_DIR=${N7_RUN_DIR}"
  echo "[INFO] N7_CKPT=${N7_CKPT}"
  echo "[INFO] C2_RUN_DIR=${C2_RUN_DIR}"
  echo "[INFO] C2_CKPT=${C2_CKPT}"
  echo "[INFO] OUT_ROOT=${OUT_ROOT}"
} 2>&1 | tee -a "${LOG_PATH}"

while IFS= read -r cmd; do
  echo "[CMD] ${cmd}" | tee -a "${LOG_PATH}"
  bash -lc "cd '${ROOT}' && ${cmd}" 2>&1 | tee -a "${LOG_PATH}"
done < "${COMMANDS_FILE}"

"${CONDA_BIN}" run --live-stream -n "${CONDA_ENV}" python foundation/tools/summarize_n7_controls.py \
  --n7-ablation-root "${OUT_ROOT}" \
  --n7-true-run-dir "${N7_RUN_DIR}" \
  --out-dir "${OUT_ROOT}" \
  --p0-prefix n7_eval_x3_ablation 2>&1 | tee -a "${LOG_PATH}"

echo "[DONE] N7 eval x3 ablation outputs: ${OUT_ROOT}" | tee -a "${LOG_PATH}"
