#!/usr/bin/env bash
# Watcher: wait until TARGET_PID exits, then launch the rgb-bridge training.
# Primary signal: PID disappearance. Sanity signal: GPU memory < GPU_MEM_THRESHOLD_MB.
# Both must hold for CONSECUTIVE_OK consecutive checks before trigger fires.
set -uo pipefail

TARGET_PID="${TARGET_PID:-3735748}"
CHECK_INTERVAL_S="${CHECK_INTERVAL_S:-30}"
CONSECUTIVE_OK="${CONSECUTIVE_OK:-3}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-500}"
GPU_INDEX="${GPU_INDEX:-0}"
REPO_ROOT="/home/caq/6666_raw/dav2_raw_512960"
LAUNCH_SCRIPT="${REPO_ROOT}/finetune_stf/scripts/train_raw_ram_rgb_e3_bridge.sh"
WATCHER_LOG_DIR="${REPO_ROOT}/finetune_stf/exp/_watcher_logs"
mkdir -p "${WATCHER_LOG_DIR}"
WATCHER_LOG="${WATCHER_LOG_DIR}/watch_pid_${TARGET_PID}_to_rgb_bridge_$(date +%Y%m%d_%H%M%S).log"

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${WATCHER_LOG}"
}

gpu_mem_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null | tr -d ' '
}

pid_alive() {
    kill -0 "${TARGET_PID}" 2>/dev/null
}

log "watcher starting: TARGET_PID=${TARGET_PID}  interval=${CHECK_INTERVAL_S}s  consecutive_ok=${CONSECUTIVE_OK}  gpu_mem_threshold=${GPU_MEM_THRESHOLD_MB}MB"
log "launch target: ${LAUNCH_SCRIPT}"
if [[ ! -x "${LAUNCH_SCRIPT}" ]]; then
    log "ERROR: launch script not executable: ${LAUNCH_SCRIPT}"
    log "attempting to fix with chmod +x"
    chmod +x "${LAUNCH_SCRIPT}" || { log "chmod failed; aborting"; exit 1; }
fi

if ! pid_alive; then
    log "WARNING: TARGET_PID=${TARGET_PID} is not alive at startup. Sleeping 10s to avoid accidental immediate trigger."
    sleep 10
fi

ok_streak=0
while true; do
    mem_mb=$(gpu_mem_mb)
    if pid_alive; then
        ok_streak=0
        log "PID ${TARGET_PID} alive  gpu_mem=${mem_mb}MB  streak=0/${CONSECUTIVE_OK}"
    else
        if [[ "${mem_mb}" =~ ^[0-9]+$ ]] && (( mem_mb < GPU_MEM_THRESHOLD_MB )); then
            ok_streak=$(( ok_streak + 1 ))
            log "PID ${TARGET_PID} gone  gpu_mem=${mem_mb}MB (<${GPU_MEM_THRESHOLD_MB})  streak=${ok_streak}/${CONSECUTIVE_OK}"
        else
            ok_streak=0
            log "PID ${TARGET_PID} gone BUT gpu_mem=${mem_mb}MB (>=${GPU_MEM_THRESHOLD_MB})  streak reset"
        fi
    fi

    if (( ok_streak >= CONSECUTIVE_OK )); then
        log "TRIGGER: PID ${TARGET_PID} gone for ${ok_streak} consecutive checks, gpu clear. Launching rgb-bridge training."
        break
    fi
    sleep "${CHECK_INTERVAL_S}"
done

log "--- handing off to ${LAUNCH_SCRIPT} ---"
cd "${REPO_ROOT}"
exec bash "${LAUNCH_SCRIPT}"
