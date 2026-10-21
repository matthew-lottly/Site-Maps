"""Meta / CIESIN High Resolution Settlement Layer (HRSL), general population, 1 arc-second (~30 m) cloud-optimised GeoTIFF tiles on AWS."""
from __future__ import annotations

import re

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

from ..core.cache import cache_path
from ..core.frame import SiteFrame
from ..core.provenance import Source
from .rasters import remote_env

BASE = "https://dataforgood-fb-data.s3.amazonaws.com/hrsl-cogs/hrsl_general"
SOURCE = Source("HRSL", "Meta Data for Good / CIESIN (Columbia University)", "High Resolution Settlement Layer, general population (hrsl_general, latest.vrt mosaic)",
                "tile file version as named on the sheet", "https://dataforgood.facebook.com/dfg/tools/high-resolution-population-density-maps", "CC BY 4.0",
                "Modelled population per ~30 m cell from census counts and building detection; not a census count")


def tile_for(lon: float, lat: float) -> tuple[str, str]:
    """Find the source tile in the published VRT mosaic whose destination rectangle contains the point. Returns (relative path, vrt tile version note)."""
    vrt = cache_path("hrsl/hrsl_general-latest.vrt")
    if not vrt.exists():
        vrt.write_text(requests.get(f"{BASE}/hrsl_general-latest.vrt", timeout=120).text)
    t = vrt.read_text()
    gt = [float(x) for x in re.search(r"<GeoTransform>([^<]*)</GeoTransform>", t).group(1).split(",")]
    px, py = (lon - gt[0]) / gt[1], (lat - gt[3]) / gt[5]
    for m in re.finditer(r"<SourceFilename[^>]*>([^<]*)</SourceFilename>.*?<DstRect xOff=\"(\d+)\" yOff=\"(\d+)\" xSize=\"(\d+)\" ySize=\"(\d+)\"", t, re.S):
        f, xo, yo, xs, ys = m.group(1), *map(int, m.groups()[1:])
        if xo <= px < xo + xs and yo <= py < yo + ys:
            return f, f
    raise LookupError("no HRSL tile covers this point")


def cells(frame: SiteFrame, bounds_ll: tuple[float, float, float, float]):
    """Population cells inside `bounds_ll` (W,S,E,N): returns (x, y, people) arrays, x/y = cell-centre in the frame CRS, plus the tile name."""
    lon, lat = frame.center
    rel, _ = tile_for(lon, lat)
    w, s, e, n = bounds_ll
    with remote_env(), rasterio.open(f"/vsicurl/{BASE}/{rel}") as src:
        win = from_bounds(w, s, e, n, src.transform)
        a = src.read(1, window=win, masked=True, boundless=False)
        tr = src.window_transform(win.round_offsets().round_lengths())
    rows, cols = np.indices(a.shape)
    lons, lats = tr * (cols + 0.5, rows + 0.5)
    x, y = frame.to_proj.transform(np.asarray(lons).ravel(), np.asarray(lats).ravel())
    v = np.ma.filled(a, 0).astype(float).ravel()
    m = v > 0
    return np.asarray(x)[m], np.asarray(y)[m], v[m], rel
