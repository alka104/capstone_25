"""
uhi_backend.py -- grid maths + scenario runner for the Streamlit site.

Uses ONLY the documented scenario CLI (`python -m scenarios.run_scenario`, see its --help)
inside the model repo's own venv. The model repo is never modified.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Frozen grid (README section C / handoff section 4)
CRS_GRID = "EPSG:32643"
WIDTH, HEIGHT, RES_M = 1608, 1493, 30.0
X0, Y0 = 757950.0, 1458780.0

_to_grid = Transformer.from_crs("EPSG:4326", CRS_GRID, always_xy=True)
_to_wgs84 = Transformer.from_crs(CRS_GRID, "EPSG:4326", always_xy=True)

STRATEGIES = {
    "vegetation_restoration": "Vegetation restoration",
    "reflective_surface": "Reflective surfaces",
    "bare_soil_restoration": "Bare-soil restoration",
}


def latlon_to_rowcol(lat: float, lon: float) -> tuple[int, int]:
    x, y = _to_grid.transform(lon, lat)
    return int((Y0 - y) // RES_M), int((x - X0) // RES_M)


def in_study_grid(lat: float, lon: float) -> bool:
    r, c = latlon_to_rowcol(lat, lon)
    return 0 <= r < HEIGHT and 0 <= c < WIDTH


def grid_bounds_wgs84() -> tuple[float, float, float, float]:
    """(south, west, north, east)."""
    pts = [(X0, Y0), (X0 + WIDTH * RES_M, Y0), (X0, Y0 - HEIGHT * RES_M),
           (X0 + WIDTH * RES_M, Y0 - HEIGHT * RES_M)]
    lons, lats = zip(*[_to_wgs84.transform(x, y) for x, y in pts])
    return min(lats), min(lons), max(lats), max(lons)


def square_polygon_geojson(lat: float, lon: float, half_size_m: float) -> dict:
    x, y = _to_grid.transform(lon, lat)
    sq = Point(x, y).buffer(half_size_m, cap_style=3)
    ring = [list(_to_wgs84.transform(px, py)) for px, py in sq.exterior.coords]
    return {"type": "Polygon", "coordinates": [ring]}


def polygon_vertices_in_grid(geojson: dict) -> bool:
    ring = geojson["coordinates"][0]
    return all(in_study_grid(lat, lon) for lon, lat in ring)


@dataclass
class ScenarioResult:
    run_id: str
    out_dir: Path
    report: dict
    stdout: str

    def path(self, key: str) -> Path:
        return self.out_dir / config.SCENARIO_FILES[key]


def _find_output_dir(run_id: str, started: float) -> Path:
    direct = [config.EVAL_ROOT / f"scenario_{run_id}", config.EVAL_ROOT / run_id]
    for d in direct:
        if (d / config.SCENARIO_FILES["report"]).exists():
            return d
    fresh = [r.parent for r in config.EVAL_ROOT.glob("*/scenario_report.json") if r.stat().st_mtime >= started - 2]
    if fresh:
        return max(fresh, key=lambda d: (d / config.SCENARIO_FILES["report"]).stat().st_mtime)
    raise FileNotFoundError("Scenario finished but no fresh scenario_report.json was found under "
                            f"{config.EVAL_ROOT}. Check the CLI output.")


def run_scenario(strategy: str, polygon_geojson: dict, target_month: str, coverage_pct: float,
                 target_vf: float | None = None, target_albedo: float | None = None,
                 max_area_m2: float | None = None, soil_weight: float | None = None,
                 min_impervious_fraction: float | None = None, min_soil_fraction: float | None = None,
                 max_veg_fraction: float | None = None, evaluation_domain: str = "polygon",
                 timeout_s: int = 1800) -> ScenarioResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if target_month not in config.target_months():
        raise ValueError(f"Unsupported target month: {target_month}")
    if not 0 < coverage_pct <= 100:
        raise ValueError("coverage_pct must be in (0, 100]")
    if not polygon_vertices_in_grid(polygon_geojson):
        raise ValueError("Polygon extends outside the supported Bengaluru grid.")

    run_id = f"web_{uuid.uuid4().hex[:8]}"
    poly_file = Path(tempfile.gettempdir()) / f"{run_id}_polygon.geojson"
    poly_file.write_text(json.dumps(polygon_geojson), encoding="utf-8")

    cmd = [str(config.MODEL_PYTHON), "-m", "scenarios.run_scenario",
           "--strategy", strategy, "--polygon-json", str(poly_file),
           "--coverage-pct", str(coverage_pct), "--target-month", target_month,
           "--evaluation-domain", evaluation_domain, "--run-id", run_id,
           "--checkpoint", str(config.CHECKPOINT)]
    opt = {"--target-vf": target_vf, "--target-albedo": target_albedo, "--max-area-m2": max_area_m2,
           "--soil-weight": soil_weight, "--min-impervious-fraction": min_impervious_fraction,
           "--min-soil-fraction": min_soil_fraction, "--max-veg-fraction": max_veg_fraction}
    for flag, val in opt.items():
        if val is not None:
            cmd += [flag, str(val)]

    started = time.time()
    proc = subprocess.run(cmd, cwd=config.MODEL_REPO, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError("Scenario run failed.\nCommand: " + " ".join(cmd) + "\n\n" + proc.stderr[-3000:])
    out_dir = _find_output_dir(run_id, started)
    report = json.loads((out_dir / config.SCENARIO_FILES["report"]).read_text(encoding="utf-8"))
    return ScenarioResult(run_id, out_dir, report, proc.stdout)