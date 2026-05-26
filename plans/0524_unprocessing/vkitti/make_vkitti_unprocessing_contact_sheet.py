#!/usr/bin/env python3
"""VKITTI2 entrypoint for manifest-compatible RAW-like contact sheets."""

from __future__ import annotations

import sys
from pathlib import Path


KITTI_HELPER_DIR = Path(__file__).resolve().parents[1] / "kitti"
sys.path.insert(0, str(KITTI_HELPER_DIR))

from make_kitti_unprocessing_contact_sheet import main  # noqa: E402


if __name__ == "__main__":
    main()
