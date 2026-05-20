#!/usr/bin/env python3
"""Compatibility wrapper for the renamed multi-split analyzer."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_multi_split_analysis.py")), run_name="__main__")
