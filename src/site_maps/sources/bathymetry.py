"""Seafloor and coastal topography from NOAA NCEI's Continuously Updated DEM (CUDEM) on the NOAA AWS bucket.

CUDEM tiles are 0.25° GeoTIFFs: 1/9 arc-second (~3 m) where lidar and dense soundings exist and 1/3 arc-second (~10 m) farther offshore, in
NAD83 (EPSG:4269) with NAVD88 heights in metres (negative below the datum), as stated in Amante et al. (2023) and in each file's TIFF tags.
This module reads the tiles that cover a map frame onto the frame grid by averaging, composites them (finest first), and derives contours,
hillshade, depth-zone areas, a hypsometric curve and transects, all from that one grid.
"""
from __future__ import annotations

import json
import re

import numpy as np
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union

from ..core.cache import cache_path, s3_listing
from ..core.frame import SiteFrame
from ..core.provenance import Source
from .rasters import Grid, read_to_grid

BASE = "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
PRODUCTS = {"19": "dem/NCEI_ninth_Topobathy_2014_8483/", "13": "dem/NCEI_third_Topobathy_2014_8580/"}
SOURCE = Source("CUDEM", "NOAA National Centers for Environmental Information (NCEI)", "Continuously Updated DEM (CUDEM): 1/9 and 1/3 arc-second bathymetric-topographic tiles",
                "tiles generated 2021 (version in each tile name)", "https://doi.org/10.3390/rs15061702",
                "Public domain (NOAA); Amante et al. 2023, Remote Sensing 15(6):1702",
                "NAVD88 heights; land RMSE 0.76 m vs ICESat-2 (Amante et al. 2023); offshore accuracy not stated per tile")
_TILE = re.compile(r"_n(\d+)[xX](\d+)_[wW](\d+)[xX](\d+)_")


def index() -> dict:
    """{'19': [(key, north_edge, west_edge), ...], '13': [...]} parsed from the bucket listing (tile names give the upper-left corner)."""
    p = cache_path("cudem/index.json")
    if not p.exists():
        out = {}
        for tag, pre in PRODUCTS.items():
            rows = []
            for k in s3_listing(BASE, pre, None):
                m = _TILE.search(k) if k.endswith(".tif") else None
                if m:
                    rows.append((k, float(f"{m.group(1)}.{m.group(2)}"), -float(f"{m.group(3)}.{m.group(4)}")))
            out[tag] = rows
        p.write_text(json.dumps(out))
    return json.loads(p.read_text())


def tile_box(north: float, west: float):
    return box(west, north - 0.25, west + 0.25, north)


def coverage_polygon(near: tuple[float, float, float, float]):
    """Union of the tile footprints that touch `near` (W, S, E, N), from both resolutions."""
    nb = box(*near)
    return unary_union([tile_box(la, lo) for rows in index().values() for _, la, lo in rows if tile_box(la, lo).intersects(nb)])


def max_width_inside(center: tuple[float, float], aspect: float, limit_m: float = 90000.0, step: float = 1000.0, margin_frac: float = 0.02) -> float:
    """Largest frame width (m) whose padded window stays inside the tile coverage, so the extent is limited by the data, not chosen by eye."""
    cov = coverage_polygon((center[0] - 1.5, center[1] - 1.0, center[0] + 1.5, center[1] + 1.0))
    w = limit_m
    while w > 5000:
        fr = SiteFrame.around(center[0], center[1], w, aspect)
        if cov.contains(box(*fr.padded_bounds_ll(margin_frac))):
            return w
        w -= step
    raise RuntimeError("no frame fits inside the CUDEM coverage")


def topobathy(frame: SiteFrame, res: float = 60.0):
    """Elevation (m NAVD88) averaged onto the frame grid; 1/9 arc-second tiles first, 1/3 arc-second tiles fill the rest. Returns (masked array, Grid, tiles used)."""
    grid = Grid.for_frame(frame, res)
    key = cache_path(f"derived/cudem_{frame.epsg}_{frame.center[0]:.4f}_{frame.center[1]:.4f}_{int(frame.width_m)}_{res:g}.npz")
    if key.exists():
        z = np.load(key, allow_pickle=True)
        return np.ma.masked_invalid(z["z"]), grid, [str(u) for u in z["used"]]
    w, s, e, n = frame.padded_bounds_ll(0.05)
    nb = box(w, s, e, n)
    out, used = None, []
    idx = index()
    for tag in ("19", "13"):
        for k, la, lo in idx[tag]:
            if not tile_box(la, lo).intersects(nb):
                continue
            a = read_to_grid(f"/vsicurl/{BASE}/{k}", grid, frame.crs, resampling=Resampling.average, nodata=-999999.0)
            if a.count() == 0:
                continue
            used.append(k.split("/")[-1])
            out = a if out is None else np.ma.where(np.ma.getmaskarray(out), a, out)
    z = np.ma.masked_where(np.ma.filled(out, np.nan) < -9000, out)
    np.savez_compressed(key, z=np.ma.filled(z.astype(float), np.nan), used=np.array(used))
    return z, grid, used


def utm_scale_factor(x: np.ndarray, k0: float = 0.9996, R: float = 6371008.8) -> np.ndarray:
    """Transverse-Mercator point scale factor from the distance to the central meridian (x is UTM easting in metres)."""
    return k0 * (1 + ((x - 500000.0) ** 2) / (2 * R ** 2 * k0 ** 2))


def cell_area_km2(grid: Grid) -> np.ndarray:
    """Ground area of each cell in km² (grid area divided by the squared scale factor, which removes the UTM distortion)."""
    h, w = grid.shape
    x0 = grid.extent[0]
    xs = x0 + (np.arange(w) + 0.5) * grid.res
    return np.tile((grid.res ** 2) / (utm_scale_factor(xs) ** 2) / 1e6, (h, 1))


def depth_zone_areas(z, grid: Grid, edges=(0, 10, 20, 40, 60, 80, 100, 150)) -> list[tuple[str, float, float]]:
    """(label, area km², fraction of sea area) for depth bands below 0 m NAVD88; `edges` are positive depths."""
    a = cell_area_km2(grid)
    zz = np.ma.filled(z, np.nan)
    sea = np.isfinite(zz) & (zz < 0)
    tot = a[sea].sum()
    rows = []
    for lo, hi in zip(edges, list(edges[1:]) + [1e9]):
        m = sea & (-zz >= lo) & (-zz < hi)
        rows.append((f"{lo}–{hi} m" if hi < 1e8 else f"> {lo} m", float(a[m].sum()), float(a[m].sum() / tot) if tot else 0.0))
    return rows


def hypsometric_curve(z, grid: Grid, n: int = 200):
    """Sea area (km²) deeper than each depth (m): returns (depths, areas)."""
    a = cell_area_km2(grid)
    zz = np.ma.filled(z, np.nan)
    sea = np.isfinite(zz) & (zz < 0)
    d, w = -zz[sea], a[sea]
    order = np.argsort(d)
    d, w = d[order], w[order]
    cum = w.sum() - np.cumsum(w) + w                 # area of cells at least as deep as each sorted depth (the cell itself included)
    gd = np.linspace(0, d.max(), n)
    return gd, np.interp(gd, d, cum)


def deepest_cell(z) -> tuple[int, int]:
    zz = np.ma.filled(z, np.nan)
    return tuple(int(i) for i in np.unravel_index(np.nanargmin(zz), zz.shape))


def nearest_land_cell(z, grid: Grid, to: tuple[int, int]) -> tuple[int, int]:
    """Land cell (elevation >= 0 m) closest to the cell `to`."""
    zz = np.ma.filled(z, np.nan)
    land = np.isfinite(zz) & (zz >= 0)
    dist, (ir, ic) = ndimage.distance_transform_edt(~land, return_indices=True)
    return int(ir[to]), int(ic[to])


def transect(z, grid: Grid, a: tuple[int, int], b: tuple[int, int], step_m: float | None = None):
    """Elevation sampled along the straight line a -> b (cells). Returns distance (km), elevation (m), and the sample points as projected x, y."""
    zz = np.ma.filled(z, np.nan)
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    r = np.linspace(a[0], b[0], n)
    c = np.linspace(a[1], b[1], n)
    vals = ndimage.map_coordinates(np.where(np.isfinite(zz), zz, np.nan), [r, c], order=1, mode="nearest")
    d = np.hypot(r - a[0], c - a[1]) * grid.res / 1000.0
    x0, y0, x1, y1 = grid.extent
    return d, vals, (x0 + (c + 0.5) * grid.res, y1 - (r + 0.5) * grid.res)


def rc_to_xy(grid: Grid, rc):
    x0, y0, x1, y1 = grid.extent
    return x0 + (rc[1] + 0.5) * grid.res, y1 - (rc[0] + 0.5) * grid.res
