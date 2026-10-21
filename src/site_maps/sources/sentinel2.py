"""Sentinel-2 L2A cloud-optimised GeoTIFFs on AWS (Element 84 'sentinel-cogs'): scene search by S3 listing, cloud-masked reads, NDVI."""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass

import mgrs
import numpy as np
import requests

from ..core.cache import cache_path, s3_listing
from ..core.frame import SiteFrame
from ..core.provenance import Source
from .rasters import Grid, read_to_grid

BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
SOURCE = Source("S2", "ESA Copernicus (COGs by Element 84)", "Sentinel-2 L2A COGs", "scene dates as labelled on the sheet",
                "https://registry.opendata.aws/sentinel-2-l2a-cogs/", "free and open; contains modified Copernicus Sentinel data", "Contains modified Copernicus Sentinel data.")
GOOD_SCL = (4, 5)              # vegetation and bare soil; water, cloud, shadow, snow and unclassified pixels are masked out of NDVI


@dataclass
class Scene:
    prefix: str
    date: dt.date
    cloud: float
    epsg: int

    def url(self, asset: str) -> str:
        return f"/vsicurl/{BASE}/{self.prefix}/{asset}.tif"


def tile_for(lon: float, lat: float) -> tuple[str, str, str]:
    m = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)           # e.g. 15RTN
    mm = re.match(r"(\d+)([A-Z])([A-Z]{2})", m)
    return mm.group(1), mm.group(2), mm.group(3)


def scenes(lon: float, lat: float, start: dt.date, end: dt.date) -> list[Scene]:
    z, band, sq = tile_for(lon, lat)
    out = []
    ym = dt.date(start.year, start.month, 1)
    while ym <= end:
        for pfx in s3_listing(BASE, f"sentinel-s2-l2a-cogs/{z}/{band}/{sq}/{ym.year}/{ym.month}/"):
            if not pfx.endswith("/"):
                continue
            m = re.search(r"_(\d{8})_\d_L2A/$", pfx)
            if not m:
                continue
            d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            if start <= d <= end:
                jp = cache_path(f"s2/{pfx.strip('/').split('/')[-1]}.json")
                if not jp.exists():
                    jp.write_text(requests.get(f"{BASE}/{pfx}{pfx.strip('/').split('/')[-1]}.json", timeout=60).text)
                j = json.loads(jp.read_text())
                out.append(Scene(pfx.strip("/"), d, float(j["properties"].get("eo:cloud_cover", 100)), int(j["properties"]["proj:epsg"])))
        ym = dt.date(ym.year + (ym.month == 12), ym.month % 12 + 1, 1)
    return sorted(out, key=lambda s: s.date)


def _boa_offset(red, nir) -> float:
    """The STAC item lists offset -0.1 (processing baseline >= 04.00 adds 1000 to the DNs), but the DNs in these COGs are sometimes already without it.
    Decide from the data: negative reflectance is impossible, so if the darkest 0.5 % of red/NIR pixels are well below 1000 DN the offset is not present."""
    v = np.concatenate([np.ma.compressed(red).astype(float), np.ma.compressed(nir).astype(float)])
    v = v[v > 0]
    return -0.1 if len(v) and np.percentile(v, 0.5) >= 800 else 0.0


def read_masked(scene: Scene, frame: SiteFrame, res: float = 10.0, pad_frac: float = 0.0):
    """Returns dict(red, nir, ndvi, good) on the frame grid; reflectance applies the scale/offset published in the STAC item (0.0001, -0.1)."""
    grid = Grid.for_frame(frame, res, pad_frac)
    from rasterio.enums import Resampling

    red = read_to_grid(scene.url("B04"), grid, frame.crs, resampling=Resampling.bilinear)
    nir = read_to_grid(scene.url("B08"), grid, frame.crs, resampling=Resampling.bilinear)
    scl = read_to_grid(scene.url("SCL"), grid, frame.crs, resampling=Resampling.nearest)
    off = _boa_offset(red, nir)
    r = np.clip(np.ma.filled(red.astype(float), np.nan) * 1e-4 + off, 0, 1)
    n = np.clip(np.ma.filled(nir.astype(float), np.nan) * 1e-4 + off, 0, 1)
    good = np.isin(np.ma.filled(scl, 0), GOOD_SCL) & np.isfinite(r) & np.isfinite(n) & ((n + r) > 0.05)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (n - r) / (n + r)
    ndvi = np.ma.masked_where(~good, np.clip(np.where(good, ndvi, 0.0), -1, 1))
    return {"red": r, "nir": n, "ndvi": ndvi, "good": good, "grid": grid}


def true_color(scene: Scene, frame: SiteFrame, res: float = 10.0, pad_frac: float = 0.0) -> tuple[np.ndarray, Grid]:
    grid = Grid.for_frame(frame, res, pad_frac)
    from rasterio.enums import Resampling

    a = read_to_grid(scene.url("TCI"), grid, frame.crs, band=[1, 2, 3], resampling=Resampling.bilinear)
    return np.moveaxis(np.ma.filled(a, 0).astype(np.uint8), 0, -1), grid


def window_quality(scene: Scene, frame: SiteFrame, res: float = 60.0) -> tuple[float, float]:
    """(coverage, cloud fraction) inside the map window from the scene classification layer. Coverage < 1 means the window crosses the swath edge;
    scene-level cloud cover describes the whole 110 km tile, not this window."""
    grid = Grid.for_frame(frame, res)
    from rasterio.enums import Resampling

    scl = np.ma.filled(read_to_grid(scene.url("SCL"), grid, frame.crs, resampling=Resampling.nearest), 0)
    covered = scl != 0
    cov = float(covered.mean())
    cloud = float(np.isin(scl, (1, 3, 8, 9, 10)).sum() / max(covered.sum(), 1))
    return cov, cloud
