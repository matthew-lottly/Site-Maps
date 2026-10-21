"""Raster helpers: read any (remote) raster onto the map grid, hillshade, contours."""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

from ..core.frame import SiteFrame


@contextmanager
def remote_env():
    proxy = os.environ.get("HTTPS_PROXY", "").replace("http://", "").replace("https://", "")
    opts = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.json", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                CURL_CA_BUNDLE="/root/.ccr/ca-bundle.crt", GDAL_HTTP_UNSAFESSL="NO")
    if proxy:
        opts["GDAL_HTTP_PROXY"] = proxy
    with rasterio.Env(**opts):
        yield


@dataclass
class Grid:
    """Raster grid in the frame CRS: array(s) + extent (x0, y0, x1, y1)."""

    extent: tuple[float, float, float, float]
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.extent
        return int(round((y1 - y0) / self.res)), int(round((x1 - x0) / self.res))

    @property
    def transform(self):
        x0, y0, x1, y1 = self.extent
        return from_bounds(x0, y0, x1, y1, self.shape[1], self.shape[0])

    @classmethod
    def for_frame(cls, frame: SiteFrame, res: float, pad_frac: float = 0.0) -> "Grid":
        x0, y0, x1, y1 = frame.extent
        px, py = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        return cls((x0 - px, y0 - py, x1 + px, y1 + py), res)


def read_to_grid(path: str, grid: Grid, crs, band: int | list[int] = 1, resampling=Resampling.bilinear, nodata=None) -> np.ndarray:
    """Read `path` (local file or /vsicurl/ URL) resampled onto `grid`. Returns (rows, cols) or (bands, rows, cols)."""
    h, w = grid.shape
    with remote_env(), rasterio.open(path) as src:
        with WarpedVRT(src, crs=crs, transform=grid.transform, width=w, height=h, resampling=resampling, nodata=nodata if nodata is not None else src.nodata) as vrt:
            arr = vrt.read(band, masked=True)
    return arr


def hillshade(z: np.ndarray, res: float, azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """Standard Horn hillshade, 0-1. `z` in metres on a square grid of `res` metres."""
    z = np.ma.filled(z.astype(float), np.nan)
    z = np.where(np.isnan(z), np.nanmean(z), z)
    dzdx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * res)
    dzdy = ((z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:])) / (8 * res)
    dzdx, dzdy = dzdx * z_factor, dzdy * z_factor
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.full(z.shape, np.nan)
    out[1:-1, 1:-1] = np.clip(hs, 0, 1)
    return np.where(np.isnan(out), np.nanmean(out), out)


def slope_degrees(z: np.ndarray, res: float) -> np.ndarray:
    z = np.ma.filled(z.astype(float), np.nan)
    gy, gx = np.gradient(np.where(np.isnan(z), np.nanmean(z), z), res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))
