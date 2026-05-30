#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-/home/caq/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dav3}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
BS="${BS:-8}"
SEED="${SEED:-42}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
MAX_KITTI_VAL_SAMPLES="${MAX_KITTI_VAL_SAMPLES:-}"
timestamp="$(date +%m%d_%H%M)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/plans/0527/diagnostics/n6_x3_ablation_${timestamp}}"
MODES="${MODES:-true shuffle}"
N2_CKPT="${N2_CKPT:-}"
N2_RUN_DIR="${N2_RUN_DIR:-}"

fail() { echo "[ERROR] $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || fail "Required file not found: $1"; }
require_dir() { [[ -d "$1" ]] || fail "Required directory not found: $1"; }

[[ -n "${N2_CKPT}" ]] || fail "Set N2_CKPT to the trained N2 checkpoint."
require_file "${N2_CKPT}"
if [[ -z "${N2_RUN_DIR}" ]]; then ckpt_dir="$(cd "$(dirname "${N2_CKPT}")" && pwd)"; run_name="$(basename "${ckpt_dir}")"; if [[ -f "${ckpt_dir}/config.json" ]]; then N2_RUN_DIR="${ckpt_dir}"; elif [[ -f "${ROOT}/finetune_stf/exp/${run_name}/config.json" ]]; then N2_RUN_DIR="${ROOT}/finetune_stf/exp/${run_name}"; else fail "Could not infer N2_RUN_DIR from N2_CKPT=${N2_CKPT}; set N2_RUN_DIR explicitly."; fi; fi
require_dir "${N2_RUN_DIR}"
require_file "${N2_RUN_DIR}/config.json"

read -r cfg_c2_checkpoint cfg_c2_run_dir < <(python3 - "${N2_RUN_DIR}/config.json" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("c2_checkpoint", ""), cfg.get("c2_run_dir", ""))
PY
)
C2_CHECKPOINT="${C2_CHECKPOINT:-${cfg_c2_checkpoint}}"
C2_RUN_DIR="${C2_RUN_DIR:-${cfg_c2_run_dir}}"
require_file "${C2_CHECKPOINT}"
require_dir "${C2_RUN_DIR}"
[[ ! -e "${OUT_ROOT}" ]] || fail "OUT_ROOT already exists, refusing to overwrite: ${OUT_ROOT}"
[[ "${BS}" =~ ^[0-9]+$ ]] || fail "BS must be an integer, got ${BS}"
(( BS > 1 )) || fail "BS must be > 1 for shuffle donor batches."

mkdir -p "${OUT_ROOT}"
LOG_PATH="${OUT_ROOT}/n6_eval.log"
COMMANDS_FILE="${OUT_ROOT}/n6_eval_commands.sh"

python3 - "${N2_RUN_DIR}/config.json" "${COMMANDS_FILE}" "${OUT_ROOT}" "${N2_CKPT}" "${C2_CHECKPOINT}" "${C2_RUN_DIR}" "${BS}" "${SEED}" "${CONDA_BIN}" "${CONDA_ENV}" "${GPU}" "${MODES}" "${MAX_VAL_SAMPLES}" "${MAX_KITTI_VAL_SAMPLES}" <<'PY'
import json, shlex, sys
from pathlib import Path
config_path, commands_path, out_root, n2_ckpt, c2_checkpoint, c2_run_dir, bs, seed, conda_bin, conda_env, gpu, modes, max_val_samples, max_kitti_val_samples = sys.argv[1:]
cfg = json.loads(Path(config_path).read_text())
if str(cfg.get("method_id", "")).upper() != "N2":
    raise SystemExit(f"N2 config method_id must be N2, got {cfg.get('method_id')!r}")
if cfg.get("incremental_feature_source") != "x3":
    raise SystemExit(f"N2 config incremental_feature_source must be x3, got {cfg.get('incremental_feature_source')!r}")
arg_keys = ["method_id", "input_domain", "model_input_tensor", "dataset_geometry_mode", "raw_storage_format", "fullres_even_policy", "rgb_input_space", "depth_target_space", "front_end", "encoder", "pretrained_from", "c2_checkpoint", "c2_run_dir", "vkitti_train_list", "vkitti_val_list", "eval_protocol", "kitti_val_split", "kitti_base", "kitti_eval_protocol", "kitti_expected_val_samples", "kitti_num_workers", "max_kitti_val_samples", "input_height", "input_width", "min_depth", "max_depth", "incremental_feature_source", "delta_condition", "gate_condition", "raw_feature_encoder_trainable", "residual_alpha", "d0_sign", "lambda_lp", "lowpass_kernel", "q_good", "lambda_final", "lambda_boundary", "lambda_grad", "lambda_keep_good_d1", "lambda_gate_sparse", "lambda_lowfreq_loss", "lambda_invalid_keep", "unprocessing_method", "vkitti_unprocessing_preset", "vkitti_unprocessing_mix_weights", "raw_adapter_backend", "raw_adapter_cfa_pattern", "raw_adapter_packed_channel_order", "raw_adapter_rgb_transfer", "raw_adapter_inverse_tone", "raw_adapter_ccm", "raw_adapter_red_gain_range", "raw_adapter_blue_gain_range", "raw_adapter_fixed_red_gain", "raw_adapter_fixed_blue_gain", "raw_adapter_fixed_light_scale", "raw_adapter_dark_light_scale_range", "raw_adapter_over_light_scale_range", "raw_adapter_shot_noise", "raw_adapter_read_noise", "raw_adapter_noise_mean_mode", "raw_adapter_black_level", "raw_adapter_white_level", "raw_adapter_random_seed_policy", "raw_adapter_external_raw_rgb_root", "raw_adapter_external_key", "raw_adapter_external_cache_space", "raw_adapter_variant_policy", "raw_adapter_variant_weights", "hflip_prob", "epochs", "bs", "accum_steps", "lr", "weight_decay", "num_workers", "log_interval", "save_interval", "eval_interval", "max_train_steps", "max_val_samples", "amp_dtype", "seed", "device"]
bool_keys = {"eval_kitti": "eval-kitti", "save_best_checkpoint": "save-best-checkpoint", "amp": "amp"}
def value_for(key):
    if key == "eval_protocol" and isinstance(cfg.get(key), dict):
        return cfg[key].get("vkitti_val")
    if key == "c2_checkpoint":
        return c2_checkpoint
    if key == "c2_run_dir":
        return c2_run_dir
    if key == "bs":
        return int(bs)
    if key == "seed":
        return int(seed)
    if key == "max_val_samples" and max_val_samples:
        return int(max_val_samples)
    if key == "max_kitti_val_samples" and max_kitti_val_samples:
        return int(max_kitti_val_samples)
    return cfg.get(key)
def add_arg(cmd, key, value):
    if value is None:
        return
    flag = "--" + key.replace("_", "-")
    if isinstance(value, list):
        if not value:
            return
        cmd.append(flag)
        cmd.extend(str(v) for v in value)
    elif isinstance(value, dict):
        cmd.extend([flag, json.dumps(value, sort_keys=True)])
    else:
        cmd.extend([flag, str(value)])
lines = []
for mode in modes.split():
    mode_out = str(Path(out_root) / mode)
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
    cmd.extend(["--eval-only", "--resume-from", str(n2_ckpt), "--n6-feature-ablation-mode", mode, "--n6-feature-ablation-seed", str(seed), "--n6-feature-ablation-key", "x3", "--n6-output-dir", mode_out, "--save-path", str(Path(out_root) / "driver"), "--heavy-save-path", str(Path(out_root) / "heavy_unused")])
    lines.append(" ".join(shlex.quote(part) if not part.startswith("CUDA_VISIBLE_DEVICES=") else part for part in cmd))
Path(commands_path).write_text("\n".join(lines) + "\n")
PY

printf '[INFO] host=%s user=%s pwd=%s\n' "$(hostname)" "$(whoami)" "$(pwd)" | tee -a "${LOG_PATH}"
echo "[INFO] N2_CKPT=${N2_CKPT}" | tee -a "${LOG_PATH}"
echo "[INFO] N2_RUN_DIR=${N2_RUN_DIR}" | tee -a "${LOG_PATH}"
echo "[INFO] C2_CHECKPOINT=${C2_CHECKPOINT}" | tee -a "${LOG_PATH}"
echo "[INFO] C2_RUN_DIR=${C2_RUN_DIR}" | tee -a "${LOG_PATH}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}" | tee -a "${LOG_PATH}"
echo "[SMOKE_CMD] ${CONDA_BIN} run --live-stream -n ${CONDA_ENV} python -m py_compile foundation/engine/models/dav2_incremental_residual.py foundation/tools/train_vkitti2_incremental_residual.py" | tee -a "${LOG_PATH}"

while IFS= read -r cmd; do echo "[CMD] ${cmd}" | tee -a "${LOG_PATH}"; bash -lc "cd '${ROOT}' && ${cmd}" 2>&1 | tee -a "${LOG_PATH}"; done < "${COMMANDS_FILE}"

python3 - "${OUT_ROOT}" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
def load(mode):
    path = root / mode / "n6_summary.json"
    return json.loads(path.read_text()) if path.is_file() else None
def get(payload, path):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur
def diff(a, b, path):
    av, bv = get(a, path), get(b, path)
    if av is None or bv is None:
        return None
    av, bv = float(av), float(bv)
    return bv - av if math.isfinite(av) and math.isfinite(bv) else None
true = load("true")
shuffle = load("shuffle")
if true and shuffle:
    payload = {"experiment": "N6_compare", "true": true, "shuffle": shuffle, "shuffle_minus_true": {"vkitti_final_abs_rel": diff(true, shuffle, ["vkitti", "overall", "final", "abs_rel"]), "vkitti_boundary_abs_rel": diff(true, shuffle, ["vkitti", "region", "final", "boundary_abs_rel"]), "vkitti_far50_abs_rel": diff(true, shuffle, ["vkitti", "region", "final", "far50_abs_rel"]), "vkitti_dark_abs_rel": diff(true, shuffle, ["vkitti", "region", "final", "dark_abs_rel"]), "vkitti_saturated_abs_rel": diff(true, shuffle, ["vkitti", "region", "final", "saturated_abs_rel"]), "kitti_final_abs_rel": diff(true, shuffle, ["kitti", "overall", "final", "abs_rel"])}}
    (root / "n6_compare.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# N6 Compare", "", "| metric | shuffle - true |", "|---|---:|"]
    for key, value in payload["shuffle_minus_true"].items():
        lines.append(f"| {key} | {value} |")
    (root / "n6_compare.md").write_text("\n".join(lines) + "\n")
PY

echo "[DONE] N6 eval outputs: ${OUT_ROOT}" | tee -a "${LOG_PATH}"
