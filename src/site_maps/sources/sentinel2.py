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
                    resp = requests.get(f"{BASE}/{pfx}{pfx.strip('/').split('/')[-1]}.json", timeout=60)
                    if resp.status_code != 200:               # a listed scene without its item JSON is skipped
                        continue
                    jp.write_text(resp.text)
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


def tiles_for_frame(frame: SiteFrame) -> list[tuple[str, str, str]]:
    w, s, e, n = frame.bounds_ll
    pts = [(w, s), (w, n), (e, s), (e, n), frame.center, ((w + e) / 2, s), ((w + e) / 2, n), (w, (s + n) / 2), (e, (s + n) / 2)]
    return sorted({tile_for(lo, la) for lo, la in pts})


def best_scene_per_tile(frame: SiteFrame, start: dt.date, end: dt.date, max_scene_cloud: float = 40.0) -> list[tuple[Scene, float, float]]:
    """For each MGRS tile touching the frame: the scene with the least cloud/shadow inside the window, among scenes that cover the window as fully as the
    best-covering scene does. Returns [(scene, window coverage, window cloud fraction)]."""
    out = []
    for z, band, sq in tiles_for_frame(frame):
        lon, lat = _tile_probe(frame, (z, band, sq))
        cands = []
        for sc in scenes(lon, lat, start, end):
            if sc.cloud > max_scene_cloud:
                continue
            cov, cl = window_quality(sc, frame, res=120.0)
            cands.append((sc, cov, cl))
        if not cands:
            continue
        cmax = max(c[1] for c in cands)
        good = [c for c in cands if c[1] >= cmax - 0.02]
        out.append(min(good, key=lambda c: (round(c[2], 3), -c[0].date.toordinal())))
    return out


def _tile_probe(frame: SiteFrame, tile: tuple[str, str, str]) -> tuple[float, float]:
    """A lon/lat inside the frame that falls in `tile` (used to list that tile's scenes)."""
    w, s, e, n = frame.bounds_ll
    for fx in np.linspace(0.05, 0.95, 10):
        for fy in np.linspace(0.05, 0.95, 10):
            lo, la = w + fx * (e - w), s + fy * (n - s)
            if tile_for(lo, la) == tile:
                return lo, la
    return frame.center


def true_color_mosaic(frame: SiteFrame, picks: list[Scene], res: float = 40.0):
    """Composite of the picked scenes' true-colour product on the frame grid (first scene wins where it has data)."""
    grid = Grid.for_frame(frame, res)
    out = None
    for sc in picks:
        rgb, _ = true_color(sc, frame, res)
        valid = rgb.max(axis=2) > 0
        out = rgb if out is None else np.where(valid[..., None], rgb, out) if False else np.where((out.max(axis=2) > 0)[..., None], out, rgb)
    return out, grid
