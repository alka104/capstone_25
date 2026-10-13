"""
raster_io.py -- read model output GeoTIFFs and prepare them for the map.

Nothing here predicts anything. It reads rasters the model repo wrote (using each file's OWN
CRS/transform, so we never assume the grid) and colourises them. NaN / nodata stay transparent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from matplotlib import colormaps
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


def read_array(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan)


def value_range(path: Path, symmetric: bool = False) -> tuple[float, float]:
    a = read_array(path)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(a), 98)) or 1e-6
        return -m, m
    return float(np.percentile(a, 2)), float(np.percentile(a, 98))


def overlay(path: Path, vmin: float, vmax: float, cmap: str = "RdYlGn_r"):
    """-> (rgba uint8 array, [[south, west], [north, east]]) in EPSG:4326, cropped to valid pixels."""
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float32").filled(np.nan)
        tf, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(data, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=tf,
                  dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    valid = np.isfinite(dst)
    if not valid.any():
        raise ValueError(f"{path.name} has no valid pixels")
    rows, cols = np.where(valid)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    crop = dst[r0:r1 + 1, c0:c1 + 1]
    norm = np.clip((crop - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = (colormaps[cmap](norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(crop), 215, 0)
    west, north = tf.c + c0 * tf.a, tf.f + r0 * tf.e
    east, south = tf.c + (c1 + 1) * tf.a, tf.f + (r1 + 1) * tf.e
    return rgba, [[south, west], [north, east]]


def sample_point(path: Path, lat: float, lon: float, radius_m: float = 150.0) -> dict:
    """Value at the pixel and mean over a small neighbourhood (NaN-aware)."""
    with rasterio.open(path) as src:
        x, y = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
        row, col = src.index(x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return {"inside": False, "pixel": np.nan, "mean": np.nan}
        res = abs(src.res[0])
        k = max(1, int(round(radius_m / res)))
        r0, c0 = max(row - k, 0), max(col - k, 0)
        win = rasterio.windows.Window(c0, r0, min(2 * k + 1, src.width - c0), min(2 * k + 1, src.height - r0))
        a = src.read(1, window=win, masked=True).astype("float32").filled(np.nan)
        centre = src.read(1, window=rasterio.windows.Window(col, row, 1, 1), masked=True).astype("float32").filled(np.nan)
    return {"inside": True, "pixel": float(centre[0, 0]),
            "mean": float(np.nanmean(a)) if np.isfinite(a).any() else np.nan}


def polygon_stats(path: Path, polygon_geojson: dict) -> dict:
    """NaN-aware stats of a raster inside a WGS84 polygon. Invalid pixels are excluded, never zero-filled."""
    with rasterio.open(path) as src:
        proj = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        geom = shp_transform(proj, shape(polygon_geojson))
        inside = geometry_mask([geom], out_shape=(src.height, src.width), transform=src.transform, invert=True)
        a = src.read(1, masked=True).astype("float32").filled(np.nan)
        px_area = abs(src.res[0] * src.res[1])
    vals = a[inside & np.isfinite(a)]
    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "min": np.nan, "max": np.nan, "area_m2": 0.0}
    return {"n": int(vals.size), "mean": float(vals.mean()), "min": float(vals.min()),
            "max": float(vals.max()), "area_m2": float(vals.size * px_area)}


def domain_mean(path: Path) -> float:
    a = read_array(path)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def polygon_values(path: Path, polygon_geojson: dict) -> np.ndarray:
    """1-D array of the valid (finite) pixel values inside a WGS84 polygon. Invalid pixels are excluded."""
    with rasterio.open(path) as src:
        proj = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        geom = shp_transform(proj, shape(polygon_geojson))
        inside = geometry_mask([geom], out_shape=(src.height, src.width), transform=src.transform, invert=True)
        a = src.read(1, masked=True).astype("float32").filled(np.nan)
    return a[inside & np.isfinite(a)]