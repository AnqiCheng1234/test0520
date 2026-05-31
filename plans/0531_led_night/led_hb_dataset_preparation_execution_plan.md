# LED-HB dataset download and preparation execution plan

Date: 2026-05-31

Scope: only download, extract, organize, and verify the LED Nighttime Synthetic Drive Dataset HB data needed for the first stage. This plan intentionally does not include project loader integration, model training, eval, panels, or ablation.

---

## 0. Remote setup

Connect to server 64:

```bash
ssh caq64
hostname
whoami
cd /home/caq/6666_raw/dav2_raw_0522
pwd
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3
conda env list
```

Use this dataset root:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
```

Long-running steps, including download and unzip, must run in tmux.

---

## 1. Target directory format

Create a layout that separates original downloads, extracted files, neutral manifests, and logs:

```text
/mnt/drive/3333_raw/led_night/
  downloads/
    google_drive/
      HB/
        train/
        val/
  extracted/
    HB/
      train/
      val/
  manifests/
  logs/
  scripts/
  tmp/
  codex_smoke/
```

Create it:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
mkdir -p \
  "${LED_ROOT}/downloads/google_drive/HB/train" \
  "${LED_ROOT}/downloads/google_drive/HB/val" \
  "${LED_ROOT}/extracted/HB/train" \
  "${LED_ROOT}/extracted/HB/val" \
  "${LED_ROOT}/manifests" \
  "${LED_ROOT}/logs" \
  "${LED_ROOT}/scripts" \
  "${LED_ROOT}/tmp" \
  "${LED_ROOT}/codex_smoke"
df -h /mnt/drive/3333_raw
```

Do not delete downloaded zip files after extraction. Keeping zips makes later verification and re-extraction possible.

Recommended free space before starting:

```text
minimum:     200 GB
recommended: 300 GB
safe:        500 GB
```

---

## 2. Official source

Official project page:

```text
https://simondemoreau.github.io/LED/
```

Official Google Drive root folder:

```text
https://drive.google.com/drive/folders/1VK_KZURZQOZ0ejXI3IoiwyGobzmQVc9R?usp=sharing
```

Dataset terms and license:

```text
https://drive.google.com/file/d/1lWFFWHZpRq1U2C0Lx74UIpp2Ie0GfFO8/view?usp=sharing
https://github.com/SimondeMoreau/LED/blob/main/LICENSE_DATASET
```

Before downloading, record that the dataset terms have been checked and that the intended use is research/non-commercial:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
{
  echo "[DATE] $(date -Iseconds)"
  echo "[HOST] $(hostname)"
  echo "[USER] $(whoami)"
  echo "[PROJECT] /home/caq/6666_raw/dav2_raw_0522"
  echo "[PROJECT_PAGE] https://simondemoreau.github.io/LED/"
  echo "[DATASET_TERMS] https://drive.google.com/file/d/1lWFFWHZpRq1U2C0Lx74UIpp2Ie0GfFO8/view?usp=sharing"
  echo "[LICENSE_DATASET] https://github.com/SimondeMoreau/LED/blob/main/LICENSE_DATASET"
  echo "[INTENDED_USE] research/non-commercial experiment preparation"
} > "${LED_ROOT}/manifests/dataset_terms_record.txt"
```

Observed Google Drive folder IDs:

| Folder | ID |
|---|---|
| HB | `1AxxJPfQLR_m9y-UdQb83q4sdp0bZifbl` |
| HB/train | `1iK6N40d6z7HvD7bDSlY3URY29T3sc3a2` |
| HB/val | `1dEVhnt-6NKQhVqYLNfiZaT4aY4bGOZt0` |

For first-stage HB, download only:

| Split | Content | File | Google Drive ID | Expected bytes |
|---|---|---|---|---:|
| train | RGB | `NSDD_HB_train_RGB.zip` | `1l_Nz75v5z6dDae8lADP41lKB5tvk3DQ1` | 8606802418 |
| train | Depth | `NSDD_HB_train_Depth.zip` | `1O_oKjY6rENw_zLVk5H_YlygPWwGJBEXf` | 10739356645 |
| train | Metadata | `NSDD_HB_train_Metadata.zip` | `1JKkRo2rAnUD8OWwWZ5sFAa5Z5oBevEha` | 20641363800 |
| val | RGB | `NSDD_HB_val_RGB.zip` | `1Frea7eSA4F5_GWVdoQOg1kdWaTCQhRek` | 3047883186 |
| val | Depth | `NSDD_HB_val_Depth.zip` | `1oN8gLSzUmvMtXLeeLwnK6LzzoDlsO7yU` | 7792087314 |
| val | Metadata | `NSDD_HB_val_Metadata.zip` | `18hjR_19RmDt-ftVNoI7lfv1L7qHoOC-s` | 2730018211 |

Do not download for this stage:

```text
Pattern/*
HB/test/*
NSDD_HB_*_Normals.zip
NSDD_HB_*_Semantic_Segmentation.zip
NSDD_HB_*_Instance_Segmentation.zip
NSDD_HB_*_Object_Detection.zip
```

---

## 3. Download script

Run this preflight before creating or launching the download script:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3

command -v tmux
command -v unzip
command -v zipinfo
python -V

if ! python -m gdown --version >/dev/null 2>&1; then
  python -m pip install gdown
fi
python -m gdown --version
```

Do not install `gdown` inside the long-running download job. Dependency installation should be a separate, visible preflight step so the download log only reflects download status.

Create the script:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
cat > "${LED_ROOT}/scripts/download_led_hb_required.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

source /home/caq/anaconda3/etc/profile.d/conda.sh
conda activate dav3

LED_ROOT=/mnt/drive/3333_raw/led_night
mkdir -p "${LED_ROOT}/downloads/google_drive/HB/train" "${LED_ROOT}/downloads/google_drive/HB/val"

python -m gdown --version

download_one() {
  local file_id="$1"
  local out="$2"
  local expected_bytes="$3"
  local tmp="${out}.tmp"

  if [[ -f "${out}" ]]; then
    local actual
    actual="$(stat -c '%s' "${out}")"
    if [[ "${actual}" == "${expected_bytes}" ]]; then
      echo "[SKIP] ${out} already exists with expected size ${actual}"
      return 0
    fi
    echo "[ERROR] Existing file has wrong size: ${out} actual=${actual} expected=${expected_bytes}" >&2
    echo "[ACTION] Move it aside manually if you want to re-download." >&2
    exit 2
  fi
  if [[ -e "${tmp}" ]]; then
    echo "[ERROR] Temporary download already exists: ${tmp}" >&2
    echo "[ACTION] Inspect or move it aside before retrying; it may be a partial download." >&2
    exit 2
  fi

  echo "[DOWNLOAD] ${out} via ${tmp}"
  python -m gdown "${file_id}" -O "${tmp}"

  local actual
  actual="$(stat -c '%s' "${tmp}")"
  if [[ "${actual}" != "${expected_bytes}" ]]; then
    echo "[ERROR] Size mismatch: ${tmp} actual=${actual} expected=${expected_bytes}" >&2
    echo "[ACTION] Keeping ${tmp} for debugging; do not proceed until it is fixed." >&2
    exit 2
  fi
  mv "${tmp}" "${out}"
  echo "[OK] ${out} ${actual} bytes"
}

download_one 1l_Nz75v5z6dDae8lADP41lKB5tvk3DQ1 "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_RGB.zip" 8606802418
download_one 1O_oKjY6rENw_zLVk5H_YlygPWwGJBEXf "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_Depth.zip" 10739356645
download_one 1JKkRo2rAnUD8OWwWZ5sFAa5Z5oBevEha "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_Metadata.zip" 20641363800

download_one 1Frea7eSA4F5_GWVdoQOg1kdWaTCQhRek "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_RGB.zip" 3047883186
download_one 1oN8gLSzUmvMtXLeeLwnK6LzzoDlsO7yU "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_Depth.zip" 7792087314
download_one 18hjR_19RmDt-ftVNoI7lfv1L7qHoOC-s "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_Metadata.zip" 2730018211

du -h "${LED_ROOT}/downloads/google_drive/HB/train/"*.zip "${LED_ROOT}/downloads/google_drive/HB/val/"*.zip
BASH

chmod +x "${LED_ROOT}/scripts/download_led_hb_required.sh"
```

Launch the download in tmux:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
SESSION="$(date +%m%d_%H%M)_led_hb_download"
LOG="${LED_ROOT}/logs/${SESSION}.log"
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'set -o pipefail; bash \"${LED_ROOT}/scripts/download_led_hb_required.sh\" 2>&1 | tee -a \"${LOG}\"; status=\${PIPESTATUS[0]}; echo \"[EXIT] \${status}\" | tee -a \"${LOG}\"; exit \${status}'"
echo "tmux session: ${SESSION}"
echo "log path: ${LOG}"
echo "attach: tmux attach -t ${SESSION}"
echo "monitor: tail -f ${LOG}"
```

If Google Drive quota blocks `gdown`, use the official browser folder to manually download the six files in Section 2, place them at the exact target paths, then rerun the script. The script should skip files with matching byte sizes.

---

## 4. Download verification

Run after the download tmux job exits successfully:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
cat > "${LED_ROOT}/manifests/required_zip_expected_sizes.tsv" <<'TSV'
split	content	file	expected_bytes
train	RGB	NSDD_HB_train_RGB.zip	8606802418
train	Depth	NSDD_HB_train_Depth.zip	10739356645
train	Metadata	NSDD_HB_train_Metadata.zip	20641363800
val	RGB	NSDD_HB_val_RGB.zip	3047883186
val	Depth	NSDD_HB_val_Depth.zip	7792087314
val	Metadata	NSDD_HB_val_Metadata.zip	2730018211
TSV

python - <<'PY'
from pathlib import Path

root = Path("/mnt/drive/3333_raw/led_night")
rows = [
    ("train", "NSDD_HB_train_RGB.zip", 8606802418),
    ("train", "NSDD_HB_train_Depth.zip", 10739356645),
    ("train", "NSDD_HB_train_Metadata.zip", 20641363800),
    ("val", "NSDD_HB_val_RGB.zip", 3047883186),
    ("val", "NSDD_HB_val_Depth.zip", 7792087314),
    ("val", "NSDD_HB_val_Metadata.zip", 2730018211),
]
for split, name, expected in rows:
    path = root / "downloads" / "google_drive" / "HB" / split / name
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(f"size mismatch: {path} actual={actual} expected={expected}")
    print(f"OK {split} {name} {actual}")
PY
```

Do not proceed to extraction until all six files pass the size check.

---

## 5. Zip integrity test

`unzip -t` can take time, so run it in tmux:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
cat > "${LED_ROOT}/scripts/test_led_hb_zips.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
LED_ROOT=/mnt/drive/3333_raw/led_night
for zip_path in \
  "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_RGB.zip" \
  "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_Depth.zip" \
  "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_Metadata.zip" \
  "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_RGB.zip" \
  "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_Depth.zip" \
  "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_Metadata.zip"
do
  echo "[TEST] ${zip_path}"
  unzip -t "${zip_path}" >/dev/null
  echo "[OK] ${zip_path}"
done
BASH
chmod +x "${LED_ROOT}/scripts/test_led_hb_zips.sh"

SESSION="$(date +%m%d_%H%M)_led_hb_ziptest"
LOG="${LED_ROOT}/logs/${SESSION}.log"
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'set -o pipefail; bash \"${LED_ROOT}/scripts/test_led_hb_zips.sh\" 2>&1 | tee -a \"${LOG}\"; status=\${PIPESTATUS[0]}; echo \"[EXIT] \${status}\" | tee -a \"${LOG}\"; exit \${status}'"
echo "tmux session: ${SESSION}"
echo "log path: ${LOG}"
echo "attach: tmux attach -t ${SESSION}"
echo "monitor: tail -f ${LOG}"
```

---

## 6. Zip internal layout audit

Run this after the zip integrity test passes and before extraction. This step records the top-level paths inside each zip, so the extraction script can preserve the intended final layout instead of blindly creating nested directories.

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
cat > "${LED_ROOT}/scripts/audit_led_hb_zip_layout.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

LED_ROOT = Path("/mnt/drive/3333_raw/led_night")
MANIFESTS = LED_ROOT / "manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)

ZIPS = [
    ("train", LED_ROOT / "downloads/google_drive/HB/train/NSDD_HB_train_RGB.zip"),
    ("train", LED_ROOT / "downloads/google_drive/HB/train/NSDD_HB_train_Depth.zip"),
    ("train", LED_ROOT / "downloads/google_drive/HB/train/NSDD_HB_train_Metadata.zip"),
    ("val", LED_ROOT / "downloads/google_drive/HB/val/NSDD_HB_val_RGB.zip"),
    ("val", LED_ROOT / "downloads/google_drive/HB/val/NSDD_HB_val_Depth.zip"),
    ("val", LED_ROOT / "downloads/google_drive/HB/val/NSDD_HB_val_Metadata.zip"),
]

def prefix_mode(name: str, split: str) -> str:
    parts = Path(name).parts
    if len(parts) >= 3 and parts[0] == "HB" and parts[1] == split:
        return "HB_split_prefix"
    if len(parts) >= 2 and parts[0] == split:
        return "split_prefix"
    if len(parts) >= 2 and parts[0] in {"china", "herrenberg", "ottosuhrallee", "hamburg"}:
        return "map_prefix"
    return "unknown"

summary = {}
lines = []
for split, zip_path in ZIPS:
    if not zip_path.is_file():
        raise SystemExit(f"missing zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    modes = Counter(prefix_mode(n, split) for n in names)
    first = names[:80]
    if not modes:
        raise SystemExit(f"empty zip: {zip_path}")
    if len(modes) != 1 or "unknown" in modes:
        raise SystemExit(
            f"unsupported mixed/unknown zip prefix: {zip_path} modes={dict(modes)}; inspect before extraction"
        )
    summary[zip_path.name] = {
        "split": split,
        "file_count": len(names),
        "prefix_mode": next(iter(modes)),
        "first_80_entries": first,
    }
    lines.append(f"===== {zip_path} =====")
    lines.append(f"split={split} prefix_mode={next(iter(modes))} file_count={len(names)}")
    lines.extend(first)
    lines.append("")

(MANIFESTS / "zip_internal_layout_audit.txt").write_text("\n".join(lines), encoding="utf-8")
(MANIFESTS / "zip_internal_layout_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
chmod +x "${LED_ROOT}/scripts/audit_led_hb_zip_layout.py"

python "${LED_ROOT}/scripts/audit_led_hb_zip_layout.py"
```

Do not proceed if this script reports `unknown` or mixed prefixes. In that case, inspect:

```bash
less /mnt/drive/3333_raw/led_night/manifests/zip_internal_layout_audit.txt
```

Supported prefix modes are:

```text
HB_split_prefix: zip contains HB/train/... or HB/val/...; extract under ${LED_ROOT}/extracted
split_prefix:    zip contains train/... or val/...; extract under ${LED_ROOT}/extracted/HB
map_prefix:      zip contains china/... / hamburg/...; extract under ${LED_ROOT}/extracted/HB/{train,val}
```

---

## 7. Extract script

Create the script:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
cat > "${LED_ROOT}/scripts/extract_led_hb_required.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

LED_ROOT=/mnt/drive/3333_raw/led_night
MANIFESTS="${LED_ROOT}/manifests"

detect_prefix_mode() {
  local zip_path="$1"
  local split="$2"
  python - "${zip_path}" "${split}" <<'PY'
from pathlib import Path
import sys
import zipfile

zip_path = Path(sys.argv[1])
split = sys.argv[2]

def prefix_mode(name: str) -> str:
    parts = Path(name).parts
    if len(parts) >= 3 and parts[0] == "HB" and parts[1] == split:
        return "HB_split_prefix"
    if len(parts) >= 2 and parts[0] == split:
        return "split_prefix"
    if len(parts) >= 2 and parts[0] in {"china", "herrenberg", "ottosuhrallee", "hamburg"}:
        return "map_prefix"
    return "unknown"

with zipfile.ZipFile(zip_path) as zf:
    modes = {prefix_mode(n) for n in zf.namelist() if not n.endswith("/")}

if len(modes) != 1 or "unknown" in modes:
    raise SystemExit(f"unsupported mixed/unknown zip prefix: {zip_path} modes={sorted(modes)}")
print(next(iter(modes)))
PY
}

extract_one() {
  local zip_path="$1"
  local split="$2"
  [[ -f "${zip_path}" ]] || { echo "[ERROR] missing zip: ${zip_path}" >&2; exit 2; }

  local base
  base="$(basename "${zip_path}" .zip)"
  local done_marker="${MANIFESTS}/${base}.extracted.done"
  if [[ -f "${done_marker}" ]]; then
    echo "[SKIP] ${zip_path} already has marker ${done_marker}"
    return 0
  fi

  local prefix_mode
  prefix_mode="$(detect_prefix_mode "${zip_path}" "${split}")"

  local out_dir
  case "${prefix_mode}" in
    HB_split_prefix)
      out_dir="${LED_ROOT}/extracted"
      ;;
    split_prefix)
      out_dir="${LED_ROOT}/extracted/HB"
      ;;
    map_prefix)
      out_dir="${LED_ROOT}/extracted/HB/${split}"
      ;;
    *)
      echo "[ERROR] unsupported prefix mode ${prefix_mode} for ${zip_path}" >&2
      exit 2
      ;;
  esac

  local tmp_dir="${LED_ROOT}/tmp/extract_${base}_tmp_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "${tmp_dir}" "${out_dir}" "${MANIFESTS}"

  echo "[EXTRACT] ${zip_path} prefix_mode=${prefix_mode} tmp=${tmp_dir} -> ${out_dir}"
  unzip -q "${zip_path}" -d "${tmp_dir}"

  local collisions="${MANIFESTS}/${base}.extract_collisions.txt"
  : > "${collisions}"
  while IFS= read -r -d '' file_path; do
    rel="${file_path#${tmp_dir}/}"
    if [[ -e "${out_dir}/${rel}" ]]; then
      echo "${out_dir}/${rel}" >> "${collisions}"
    fi
  done < <(find "${tmp_dir}" -type f -print0)

  if [[ -s "${collisions}" ]]; then
    echo "[ERROR] Existing files would be overwritten while extracting ${zip_path}" >&2
    echo "[ACTION] Keeping tmp extraction for debugging: ${tmp_dir}" >&2
    echo "[ACTION] Collision list: ${collisions}" >&2
    exit 2
  fi
  rm -f "${collisions}"

  cp -a "${tmp_dir}/." "${out_dir}/"
  {
    echo "zip=${zip_path}"
    echo "split=${split}"
    echo "prefix_mode=${prefix_mode}"
    echo "out_dir=${out_dir}"
    echo "completed_at=$(date -Iseconds)"
  } > "${done_marker}"

  case "${tmp_dir}" in
    "${LED_ROOT}/tmp/extract_"*"_tmp_"*)
      rm -rf "${tmp_dir}"
      ;;
    *)
      echo "[ERROR] Refusing to remove unexpected tmp dir: ${tmp_dir}" >&2
      exit 2
      ;;
  esac

  echo "[OK] ${zip_path}"
}

extract_one "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_RGB.zip" train
extract_one "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_Depth.zip" train
extract_one "${LED_ROOT}/downloads/google_drive/HB/train/NSDD_HB_train_Metadata.zip" train

extract_one "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_RGB.zip" val
extract_one "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_Depth.zip" val
extract_one "${LED_ROOT}/downloads/google_drive/HB/val/NSDD_HB_val_Metadata.zip" val

python - <<'PY'
from pathlib import Path

root = Path("/mnt/drive/3333_raw/led_night")
files = sorted(p.as_posix() for p in (root / "extracted" / "HB").rglob("*") if p.is_file())
(root / "manifests" / "extracted_first_200_files.txt").write_text(
    "\n".join(files[:200]) + ("\n" if files[:200] else ""),
    encoding="utf-8",
)
PY
du -sh "${LED_ROOT}/extracted/HB" | tee "${LED_ROOT}/manifests/extracted_hb_du.txt"
BASH

chmod +x "${LED_ROOT}/scripts/extract_led_hb_required.sh"
```

Launch extraction in tmux:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
SESSION="$(date +%m%d_%H%M)_led_hb_extract"
LOG="${LED_ROOT}/logs/${SESSION}.log"
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'set -o pipefail; bash \"${LED_ROOT}/scripts/extract_led_hb_required.sh\" 2>&1 | tee -a \"${LOG}\"; status=\${PIPESTATUS[0]}; echo \"[EXIT] \${status}\" | tee -a \"${LOG}\"; exit \${status}'"
echo "tmux session: ${SESSION}"
echo "log path: ${LOG}"
echo "attach: tmux attach -t ${SESSION}"
echo "monitor: tail -f ${LOG}"
```

---

## 8. Build a neutral extracted-file inventory

This inventory is not a project loader filelist. It only records what was extracted.
It must distinguish `distance_to_image_plane` from `distance_to_camera`; the first-stage depth GT is `distance_to_image_plane`.

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
cat > "${LED_ROOT}/scripts/build_led_hb_extracted_inventory.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

LED_ROOT = Path("/mnt/drive/3333_raw/led_night")
EXTRACTED = LED_ROOT / "extracted" / "HB"
MANIFESTS = LED_ROOT / "manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)

EXTS_IMAGE = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXTS_DEPTH = {".exr", ".png", ".npy", ".npz", ".tif", ".tiff"}
EXTS_META = {".json", ".txt", ".yaml", ".yml", ".csv", ".xml"}

def classify(path: Path) -> str:
    parts = [p.lower() for p in path.relative_to(EXTRACTED).parts]
    suffix = path.suffix.lower()
    if "ldr_color" in parts and suffix in EXTS_IMAGE:
        return "rgb_ldr_color"
    if "distance_to_image_plane" in parts and suffix in EXTS_DEPTH:
        return "depth_image_plane"
    if "distance_to_camera" in parts and suffix in EXTS_DEPTH:
        return "depth_camera"
    if "camera_params" in parts and suffix in EXTS_META:
        return "metadata_camera_params"
    if "transforms" in parts and suffix in EXTS_META:
        return "metadata_transforms"
    if suffix in EXTS_META:
        return "metadata_other"
    return "other"

def map_and_frame_key(path: Path) -> tuple[str, str] | None:
    parts = path.relative_to(EXTRACTED).parts
    if len(parts) < 4:
        return None
    split, map_name, _kind = parts[0], parts[1], parts[2]
    if split not in {"train", "val"}:
        return None
    if map_name not in {"china", "herrenberg", "ottosuhrallee", "hamburg"}:
        return None
    return map_name, path.stem

summary = {}
for split in ("train", "val"):
    split_root = EXTRACTED / split
    records = []
    counts = {
        "rgb_ldr_color": 0,
        "depth_image_plane": 0,
        "depth_camera": 0,
        "metadata_camera_params": 0,
        "metadata_transforms": 0,
        "metadata_other": 0,
        "other": 0,
    }
    frame_keys = {
        "rgb_ldr_color": set(),
        "depth_image_plane": set(),
        "metadata_camera_params": set(),
        "metadata_transforms": set(),
    }
    by_map = {}
    by_suffix = {}
    for path in sorted(split_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(LED_ROOT).as_posix()
        kind = classify(path)
        counts[kind] += 1
        key = map_and_frame_key(path)
        if key is not None:
            by_map.setdefault(key[0], {}).setdefault(kind, 0)
            by_map[key[0]][kind] = by_map[key[0]].get(kind, 0) + 1
            if kind in frame_keys:
                frame_keys[kind].add(key)
        by_suffix[path.suffix.lower() or "<none>"] = by_suffix.get(path.suffix.lower() or "<none>", 0) + 1
        records.append({"kind": kind, "path": rel, "bytes": path.stat().st_size})

    out_jsonl = MANIFESTS / f"hb_{split}_extracted_inventory.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    rgb_depth_pairs = frame_keys["rgb_ldr_color"] & frame_keys["depth_image_plane"]
    full_pairs = (
        rgb_depth_pairs
        & frame_keys["metadata_camera_params"]
        & frame_keys["metadata_transforms"]
    )
    summary[split] = {
        "root": str(split_root),
        "total_files": len(records),
        "counts": counts,
        "by_map": by_map,
        "rgb_depth_image_plane_pair_count": len(rgb_depth_pairs),
        "rgb_depth_camera_params_transforms_pair_count": len(full_pairs),
        "suffix_counts": by_suffix,
        "inventory": str(out_jsonl),
    }

summary_path = MANIFESTS / "led_hb_extracted_inventory_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
chmod +x "${LED_ROOT}/scripts/build_led_hb_extracted_inventory.py"

python "${LED_ROOT}/scripts/build_led_hb_extracted_inventory.py"
```

Inspect:

```bash
cat /mnt/drive/3333_raw/led_night/manifests/led_hb_extracted_inventory_summary.json
head -50 /mnt/drive/3333_raw/led_night/manifests/extracted_first_200_files.txt
```

Expected approximate target counts:

```text
HB/train rgb_ldr_color:                    about 14,997
HB/train depth_image_plane:                about 14,997
HB/train rgb_depth_image_plane_pair_count: about 14,997
HB/val rgb_ldr_color:                      about 4,999
HB/val depth_image_plane:                  about 4,999
HB/val rgb_depth_image_plane_pair_count:   about 4,999
```

`depth_camera` is recorded for audit only and must not be counted as the first-stage GT. The raw inventory may count more metadata files than frames, depending on how `Metadata.zip` is structured. That is acceptable at this stage. The important check is that `rgb_ldr_color` and `depth_image_plane` counts are near the expected values and can be paired by stable map/frame ids.

---

## 9. Manual structure audit

Record the actual extracted structure:

```bash
export LED_ROOT=/mnt/drive/3333_raw/led_night
{
  echo "[DATE] $(date -Iseconds)"
  echo "[HOST] $(hostname)"
  echo "[ROOT] ${LED_ROOT}"
  echo
  echo "[TOP DIRS]"
  find "${LED_ROOT}/extracted/HB" -maxdepth 4 -type d | sort
  echo
  echo "[SAMPLE FILES]"
  find "${LED_ROOT}/extracted/HB" -maxdepth 8 -type f | sort | head -300
} > "${LED_ROOT}/manifests/extracted_structure_audit.txt"
```

Open:

```bash
less /mnt/drive/3333_raw/led_night/manifests/extracted_structure_audit.txt
```

Confirm these by visual inspection of file names and directories:

```text
train maps include china, herrenberg, ottosuhrallee
val map includes hamburg
ldr_color files exist for train and val
distance_to_image_plane files exist for train and val
camera_params files exist for train and val, or metadata documents the equivalent camera fields
transforms files exist for train and val, or metadata documents the equivalent transform fields
```

If `distance_to_camera` is present, keep it in the audit but do not use it as the first-stage depth GT. If map names are not explicit in paths, inspect metadata files to determine how map/frame ids are encoded.

---

## 10. Data-preparation completion criteria

Data preparation is complete when all are true:

- The six required zip files exist under `/mnt/drive/3333_raw/led_night/downloads/google_drive/HB/{train,val}`.
- Each zip file matches the expected byte size in Section 2.
- `unzip -t` passes for all six zips.
- `zip_internal_layout_summary.json` exists and reports one supported prefix mode for each zip.
- All six zips are extracted under `/mnt/drive/3333_raw/led_night/extracted/HB/{train,val}`.
- Six extraction marker files named `NSDD_HB_{train,val}_{RGB,Depth,Metadata}.extracted.done` exist under `/mnt/drive/3333_raw/led_night/manifests`.
- `led_hb_extracted_inventory_summary.json` exists.
- `rgb_ldr_color`, `depth_image_plane`, and `rgb_depth_image_plane_pair_count` are close to:
  - HB/train: about 14,997 frames
  - HB/val: about 4,999 frames
- `distance_to_camera`, if present, is documented separately and is not counted as first-stage GT.
- The structure audit identifies train maps and val map, or documents how map/frame ids are represented if the zip layout differs.
- No Pattern, HB/test, normals, semantic segmentation, instance segmentation, or object detection zip was downloaded for this stage.

Stop and report the exact path/log if any check fails.

---

## 11. Handoff to later project-integration plan

Do not implement these in this data-only plan:

```text
LED dataset class
project-specific train/val filelists
online raw4 generation smoke
D0/D1/x3 model forward checks
training queue
eval queue
panel generation
ablation
```

The next plan should start from these prepared artifacts:

```text
/mnt/drive/3333_raw/led_night/downloads/google_drive/HB/
/mnt/drive/3333_raw/led_night/extracted/HB/
/mnt/drive/3333_raw/led_night/manifests/dataset_terms_record.txt
/mnt/drive/3333_raw/led_night/manifests/zip_internal_layout_summary.json
/mnt/drive/3333_raw/led_night/manifests/led_hb_extracted_inventory_summary.json
/mnt/drive/3333_raw/led_night/manifests/hb_train_extracted_inventory.jsonl
/mnt/drive/3333_raw/led_night/manifests/hb_val_extracted_inventory.jsonl
/mnt/drive/3333_raw/led_night/manifests/extracted_structure_audit.txt
```
