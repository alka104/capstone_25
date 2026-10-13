"""
config.py -- single place for every path and constant the site uses.
Override paths with environment variables if your folders differ.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- Paths ---------------------------------------------------------------
MODEL_REPO = Path(os.getenv("UHI_MODEL_REPO", r"C:\uhi-model"))
MODEL_PYTHON = MODEL_REPO / ".venv" / "Scripts" / "python.exe"
CHECKPOINT = MODEL_REPO / "checkpoints" / "baseline_full" / "best.pt"
RUN_NAME = "baseline_full"

# Frozen preprocessing outputs (READ-ONLY; README section C says the dataset lives here)
FROZEN_ROOT = Path(os.getenv("UHI_FROZEN_ROOT", r"C:\Users\PC-012\Desktop\uhi"))

SITE_ROOT = Path(__file__).resolve().parent
PRECOMPUTED = SITE_ROOT / "data" / "precomputed"

# ---- Month selector -------------------------------------------------------
# Handoff section 9: catalogue runs 2019-01..2024-12. Hindcasts 2020-01..2024-12 are
# supported; 2025-01 is the one supported forward target (inputs = Jan-Dec 2024).
FIRST_TARGET, LAST_TARGET = (2020, 1), (2025, 1)


def target_months() -> list[str]:
    out, (y, m) = [], FIRST_TARGET
    while (y, m) <= LAST_TARGET:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out  # 60 labels, '2020-01' ... '2024-12'


DEFAULT_MONTH = "2024-12"

# Data-quality notes from README section J. level: "high" | "elevated" | "note"
MONTH_WARNINGS: dict[str, tuple[str, str]] = {
    "2024-07": ("high", "July 2024 has a confirmed LST anomaly (~28.5 C city mean vs ~43 C in June) and heavy "
                        "temporal interpolation. This is the model's worst month; treat results with caution."),
    "2024-06": ("elevated", "June 2024 contains heavy temporal interpolation in the source data."),
    "2024-08": ("elevated", "August 2024 is adjacent to the July anomaly and has heavy interpolation; "
                            "its input window also contains July 2024."),
}
MONTH_WARNINGS["2025-01"] = ("elevated", "Forward target, NOT a current forecast: built from Jan-Dec 2024 inputs "
                                         "(which include the anomalous July 2024 frame). Newer months are unsupported.")
for _m in ("2024-09", "2024-10", "2024-11", "2024-12"):
    MONTH_WARNINGS[_m] = ("note", "This forecast's 12-month input window includes the anomalous July 2024 frame.")


def month_warning(month: str) -> tuple[str, str] | None:
    return MONTH_WARNINGS.get(month)


def fullgrid_dir(month: str) -> Path:
    """Folder written by `python -m evaluation.full_grid` (README section P)."""
    return MODEL_REPO / "evaluation" / f"fullgrid_{month}_{RUN_NAME}"


# ---- Real output file names (from the folder listing) ---------------------
FULLGRID_FILES = {
    "lst": "lst_forecast_celsius.tif",
    "uhi": "uhi_forecast_celsius.tif",
    "valid": "lst_valid_uint8.tif",
    "summary": "inference_summary.json",
}
SCENARIO_FILES = {
    "baseline": "baseline_lst_celsius.tif",
    "scenario": "scenario_lst_celsius.tif",
    "delta": "delta_lst_celsius.tif",
    "report": "scenario_report.json",
}
EVAL_ROOT = MODEL_REPO / "evaluation"