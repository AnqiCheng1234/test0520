#!/usr/bin/env bash
set -euo pipefail

VKITTI2_ROOT="${VKITTI2_ROOT:-/mnt/drive/1111_new_works/VKITTI2}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${VKITTI2_ROOT}_downloads}"
INCLUDE_TEXTGT="${INCLUDE_TEXTGT:-0}"

RGB_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar"
DEPTH_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar"
TEXTGT_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz"

mkdir -p "${VKITTI2_ROOT}" "${DOWNLOAD_DIR}"

download_file() {
  local url="$1"
  local out="$2"
  echo "[download] ${url}"
  wget -c -O "${out}" "${url}"
}

extract_archive() {
  local archive="$1"
  local target="$2"
  mkdir -p "${target}"

  local scene_entry
  scene_entry="$(tar -tf "${archive}" | grep -m1 'Scene[0-9][0-9]/' || true)"
  if [[ -z "${scene_entry}" ]]; then
    echo "[warn] no SceneXX entry found in ${archive}, extracting without strip-components"
    tar -xf "${archive}" -C "${target}"
    return
  fi

  IFS='/' read -r -a parts <<< "${scene_entry}"
  local strip_components=0
  while [[ "${strip_components}" -lt "${#parts[@]}" ]]; do
    if [[ "${parts[${strip_components}]}" =~ ^Scene[0-9][0-9]$ ]]; then
      break
    fi
    strip_components=$((strip_components + 1))
  done

  echo "[extract] ${archive} -> ${target} (strip-components=${strip_components})"
  tar --skip-old-files -xf "${archive}" -C "${target}" --strip-components="${strip_components}"
}

RGB_ARCHIVE="${DOWNLOAD_DIR}/vkitti_2.0.3_rgb.tar"
DEPTH_ARCHIVE="${DOWNLOAD_DIR}/vkitti_2.0.3_depth.tar"
TEXTGT_ARCHIVE="${DOWNLOAD_DIR}/vkitti_2.0.3_textgt.tar.gz"

download_file "${RGB_URL}" "${RGB_ARCHIVE}"
download_file "${DEPTH_URL}" "${DEPTH_ARCHIVE}"

extract_archive "${RGB_ARCHIVE}" "${VKITTI2_ROOT}/rgb"
extract_archive "${DEPTH_ARCHIVE}" "${VKITTI2_ROOT}/depth"

if [[ "${INCLUDE_TEXTGT}" == "1" ]]; then
  download_file "${TEXTGT_URL}" "${TEXTGT_ARCHIVE}"
  extract_archive "${TEXTGT_ARCHIVE}" "${VKITTI2_ROOT}/textgt"
fi

python "$(dirname "$0")/generate_vkitti2_split.py" \
  --vkitti2-root "${VKITTI2_ROOT}" \
  --output "$(dirname "$0")/../finetune_stf/dataset/splits/vkitti2/train.txt"

echo "[done] VKITTI2 download and split generation finished"

