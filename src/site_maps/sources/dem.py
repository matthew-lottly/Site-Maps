"""Elevation: USGS 3DEP 1 arc-second (US) with Copernicus GLO-30 as the global fallback."""
from __future__ import annotations

import math

import numpy as np

from ..core.cache import download
from ..core.frame import SiteFrame
from ..core.provenance import Source
from .rasters import Grid, read_to_grid

TNM = Source("3DEP", "U.S. Geological Survey", "3D Elevation Program (3DEP) 1 arc-second DEM", "current staged product at retrieval", "https://www.usgs.gov/3d-elevation-program", "Public domain (U.S. Government work)",
             "Vertical datum NAVD88, horizontal NAD83; bare-earth DEM, ~30 m posting")
COP = Source("GLO30", "European Space Agency / Copernicus", "Copernicus DEM GLO-30", "2021 release", "https://registry.opendata.aws/copernicus-dem/", "Copernicus DEM licence (free, attribution required)",
             "Digital surface model, 30 m posting, EGM2008 heights")


def _tiles(bounds):
    w, s, e, n = bounds
    return [(la, lo) for la in range(math.floor(s), math.floor(n) + 1) for lo in range(math.floor(w), math.floor(e) + 1)]


def tnm_tile_url(lat: int, lon: int) -> str:
    """3DEP 1 arc-second tiles are named by their north-west corner (e.g. n30w096 covers 29-30 N, 96-95 W)."""
    n, w = lat + 1, -lon
    t = f"n{n:02d}w{w:03d}"
    return f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/{t}/USGS_1_{t}.tif"


def cop_tile_url(lat: int, lon: int) -> str:
    ns = f"N{abs(lat):02d}_00" if lat >= 0 else f"S{abs(lat):02d}_00"
    ew = f"E{abs(lon):03d}_00" if lon >= 0 else f"W{abs(lon):03d}_00"
    t = f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{t}/{t}.tif"


def elevation(frame: SiteFrame, res: float, source: str = "3dep", pad_frac: float = 0.03):
    """Returns (masked elevation array in metres on the frame grid, Grid, Source)."""
    grid = Grid.for_frame(frame, res, pad_frac)
    bounds = frame.padded_bounds_ll(0.1)
    stack = None
    for la, lo in _tiles(bounds):
        url = tnm_tile_url(la, lo) if source == "3dep" else cop_tile_url(la, lo)
        local = download(url, f"dem/{source}_{la}_{lo}.tif")
        a = read_to_grid(str(local), grid, frame.crs)
        stack = a if stack is None else np.ma.where(np.ma.getmaskarray(stack), a, stack)
    return stack, grid, (TNM if source == "3dep" else COP)
