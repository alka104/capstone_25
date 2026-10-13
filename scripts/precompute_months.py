"""
precompute_months.py -- run full-grid inference once per target month so the website's
"Analyze" step is a file read, not a model run.  Resumable: finished months are skipped.

Usage (from C:\\uhi_site, site venv active):
    python scripts\\precompute_months.py                 # all 60 months
    python scripts\\precompute_months.py 2024-12 2024-11 # only these
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main(months: list[str]) -> int:
    valid = set(config.target_months())
    bad = [m for m in months if m not in valid]
    if bad:
        print(f"ERROR: not valid target months: {bad}")
        return 2

    failures = []
    for i, month in enumerate(months, 1):
        out = config.fullgrid_dir(month)
        if (out / "inference_summary.json").exists():
            print(f"[{i}/{len(months)}] {month}: already done, skipping")
            continue
        print(f"[{i}/{len(months)}] {month}: running ...", flush=True)
        t0 = time.time()
        cmd = [str(config.MODEL_PYTHON), "-m", "evaluation.full_grid",
               "--checkpoint", str(config.CHECKPOINT),
               "--target-month", month, "--batch-size", "8"]
        proc = subprocess.run(cmd, cwd=config.MODEL_REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    FAILED ({month}):\n{proc.stderr[-1500:]}")
            failures.append(month)
            continue
        print(f"    done in {time.time() - t0:.0f}s -> {out}")

    print(f"\nFinished. {len(months) - len(failures)} ok, {len(failures)} failed {failures or ''}")
    return 1 if failures else 0


if __name__ == "__main__":
    args = sys.argv[1:] or config.target_months()
    raise SystemExit(main(args))