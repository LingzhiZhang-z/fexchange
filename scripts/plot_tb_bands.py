#!/usr/bin/env python3
"""Compatibility wrapper for scripts/tb/plot_tb_bands.py."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tb" / "plot_tb_bands.py"), run_name="__main__")
