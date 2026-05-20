#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_SESSION="${DOWNLOAD_SESSION:-robotcar_night_dl_20260424}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-/mnt/drive/3333_raw/robotcar}"
RUNS_FILE="${RUNS_FILE:-/mnt/drive/3333_raw/robotcar/robotcar_runs_night.txt}"
PILOT_ROOT="${PILOT_ROOT:-/mnt/drive/3333_raw/robotcar/batch_rgb_raw_gt_depth_lms_front_night_pilot}"
STAGING_ROOT="${STAGING_ROOT:-/mnt/drive/3333_raw/robotcar/batch_rgb_raw_gt_depth_stride10_lms_front_night_vo_2runs}"
FINAL_ROOT="${FINAL_ROOT:-/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo}"
REPO_ROOT="${REPO_ROOT:-/home/caq/6666_raw/dav2_raw_0424}"
DOWNLOAD_TOOLS_ROOT="${DOWNLOAD_TOOLS_ROOT:-/home/caq/6666_raw/robotcar_download}"
LOG_DIR="${LOG_DIR:-/mnt/drive/3333_raw/robotcar/logs}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/drive/3333_raw/robotcar/tools/venv/bin/python}"
MIN_WINDOW_TRAVEL_M="${MIN_WINDOW_TRAVEL_M:-5.0}"
PILOT_STRIDE="${PILOT_STRIDE:-100}"
FULL_STRIDE="${FULL_STRIDE:-10}"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/night_post_download_pipeline_${STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[info] log_file=$LOG_FILE"
echo "[info] python_bin=$PYTHON_BIN"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] python binary not executable: $PYTHON_BIN"
  exit 1
fi
echo "[info] waiting download tmux session: $DOWNLOAD_SESSION"
while tmux has-session -t "$DOWNLOAD_SESSION" 2>/dev/null; do
  sleep 30
done

echo "[info] download session ended, start prepare_runs"
bash "$DOWNLOAD_TOOLS_ROOT/robotcar_prepare_runs.sh"

echo "[check] download-layer required files"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

root = Path('/mnt/drive/3333_raw/robotcar/downloads')
runs = ['2014-11-14-16-34-33', '2014-12-16-18-44-24']
required = [
    'stereo/centre',
    'stereo.timestamps',
    'lms_front',
    'lms_front.timestamps',
    'vo/vo.csv',
    'gps/ins.csv',
]
missing = []
for run in runs:
    base = root / run
    for rel in required:
        path = base / rel
        if rel.endswith('centre') or rel.endswith('lms_front'):
            ok = path.is_dir() and any(path.iterdir())
        else:
            ok = path.exists()
        if not ok:
            missing.append(str(path))
if missing:
    print('[error] missing required download files:')
    for p in missing:
        print(' -', p)
    raise SystemExit(1)
print('[ok] download-layer checks passed')
PY

echo "[run] pilot build (stride=$PILOT_STRIDE, min_window_travel_m=$MIN_WINDOW_TRAVEL_M)"
"$PYTHON_BIN" "$DOWNLOAD_TOOLS_ROOT/robotcar_batch_build_samples.py" \
  --root "$ROBOTCAR_ROOT" \
  --runs-file "$RUNS_FILE" \
  --output-root "$PILOT_ROOT" \
  --poses-source vo \
  --laser-sensors lms_front \
  --window-sec 10.0 \
  --stride "$PILOT_STRIDE" \
  --limit-per-run 24 \
  --min-window-travel-m "$MIN_WINDOW_TRAVEL_M"

echo "[check] pilot quality threshold (>=8 quality_ok per run)"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

pilot_root = Path('/mnt/drive/3333_raw/robotcar/batch_rgb_raw_gt_depth_lms_front_night_pilot')
runs = ['2014-11-14-16-34-33', '2014-12-16-18-44-24']
counts = {r: {'total': 0, 'quality_ok': 0} for r in runs}
for sample_dir in sorted(pilot_root.iterdir()):
    if not sample_dir.is_dir():
        continue
    meta = sample_dir / 'meta.json'
    if not meta.is_file():
        continue
    with meta.open('r', encoding='utf-8') as f:
        m = json.load(f)
    run = m.get('run')
    if run not in counts:
        continue
    counts[run]['total'] += 1
    if bool(m.get('quality_ok', False)):
        counts[run]['quality_ok'] += 1
print('[pilot-counts]', json.dumps(counts, indent=2, ensure_ascii=False))
failed = [r for r, c in counts.items() if c['quality_ok'] < 8]
if failed:
    print('[error] pilot threshold failed runs:', ', '.join(failed))
    raise SystemExit(2)
print('[ok] pilot threshold passed')
PY

echo "[run] full night staging build (stride=$FULL_STRIDE, min_window_travel_m=$MIN_WINDOW_TRAVEL_M)"
"$PYTHON_BIN" "$DOWNLOAD_TOOLS_ROOT/robotcar_batch_build_samples.py" \
  --root "$ROBOTCAR_ROOT" \
  --runs-file "$RUNS_FILE" \
  --output-root "$STAGING_ROOT" \
  --poses-source vo \
  --laser-sensors lms_front \
  --window-sec 10.0 \
  --stride "$FULL_STRIDE" \
  --skip-existing \
  --min-window-travel-m "$MIN_WINDOW_TRAVEL_M"

echo "[run] prepare final 480x640 root"
"$PYTHON_BIN" "$REPO_ROOT/finetune_stf/tools/prepare_robotcar_raw_depth.py" \
  --source-root "$STAGING_ROOT" \
  --output-root "$FINAL_ROOT" \
  --target-height 480 \
  --target-width 640

echo "[run] write night_runs.json"
cat > "$FINAL_ROOT/night_runs.json" <<'JSON'
{
  "condition": "night",
  "runs": [
    "2014-11-14-16-34-33",
    "2014-12-16-18-44-24"
  ],
  "poses_source": "vo",
  "laser_sensors": [
    "lms_front"
  ],
  "target_hw": "480x640"
}
JSON

echo "[check] final-root acceptance and protocol spot checks"
"$PYTHON_BIN" - <<'PY'
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

final_root = Path('/mnt/drive/3333_raw/robotcar_raw_depth_lms_front_480640_night_2runs_vo')
summary_path = final_root / 'manifests/robotcar_summary.json'
manifest_path = final_root / 'manifests/robotcar_raw_depth_v1_val.csv'
night_runs_path = final_root / 'night_runs.json'

for p in [summary_path, manifest_path, night_runs_path]:
    if not p.exists():
        raise SystemExit(f'[error] missing required file: {p}')

summary = json.loads(summary_path.read_text(encoding='utf-8'))
scene_stats = summary.get('scene_stats', {})
if not scene_stats:
    raise SystemExit('[error] summary has empty scene_stats')
if summary.get('failed_samples', 1) != 0:
    print('[warn] failed_samples != 0:', summary.get('failed_samples'))

rows = list(csv.DictReader(manifest_path.open('r', encoding='utf-8')))
if not rows:
    raise SystemExit('[error] manifest is empty')

missing_paths = []
scene_counts = {}
for r in rows:
    scene = r['scene']
    scene_counts[scene] = scene_counts.get(scene, 0) + 1
    for key in ['rgb_eval_path', 'raw_eval_path', 'depth_src_path', 'depth_proxy_path', 'meta_src_path']:
        p = Path(r[key])
        if not p.exists():
            missing_paths.append(str(p))
if missing_paths:
    raise SystemExit('[error] missing manifest paths, first=' + missing_paths[0])

for scene, count in sorted(scene_counts.items()):
    if count < 20:
        raise SystemExit(f'[error] scene {scene} has <20 kept samples: {count}')

sample_rows = random.sample(rows, k=min(5, len(rows)))
for r in sample_rows:
    rgb = Image.open(r['rgb_eval_path']).convert('RGB')
    if rgb.size != (640, 480):
        raise SystemExit(f"[error] bad rgb_eval size: {r['rgb_eval_path']} -> {rgb.size}")
    raw_eval = np.load(r['raw_eval_path'])['bayer_rect']
    if raw_eval.shape != (480, 640, 4):
        raise SystemExit(f"[error] bad raw_eval shape: {r['raw_eval_path']} -> {raw_eval.shape}")
    depth_src = np.load(r['depth_src_path'])
    if depth_src.shape != (960, 1280):
        raise SystemExit(f"[error] bad depth_src shape: {r['depth_src_path']} -> {depth_src.shape}")
    proxy = np.load(r['depth_proxy_path'])
    if 'depth' not in proxy or 'valid_mask' not in proxy:
        raise SystemExit(f"[error] proxy missing keys: {r['depth_proxy_path']}")

night = json.loads(night_runs_path.read_text(encoding='utf-8'))
night_set = set(night.get('runs', []))
manifest_set = set(scene_counts.keys())
if night_set != manifest_set:
    raise SystemExit(f"[error] night_runs mismatch: night={sorted(night_set)} manifest={sorted(manifest_set)}")

print('[ok] final checks passed')
print('[summary] scene_counts=', json.dumps(scene_counts, ensure_ascii=False))
PY

echo "[done] pipeline finished"
