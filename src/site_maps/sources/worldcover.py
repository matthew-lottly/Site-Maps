"""ESA WorldCover 10 m land cover (2021, v200) read from the public AWS bucket."""
from __future__ import annotations

import math

import numpy as np

from ..core.frame import SiteFrame
from ..core.provenance import Source
from .rasters import Grid, read_to_grid
from rasterio.enums import Resampling

SOURCE = Source("WC", "ESA WorldCover consortium", "ESA WorldCover 10 m 2021 v200", "v200, reference year 2021", "https://esa-worldcover.org/", "CC BY 4.0",
                "Class names and colours as in the WorldCover v200 Product User Manual")
CLASSES = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#FFBB22"), 30: ("Grassland", "#FFFF4C"), 40: ("Cropland", "#F096FF"), 50: ("Built-up", "#FA0000"),
           60: ("Bare / sparse vegetation", "#B4B4B4"), 70: ("Snow and ice", "#F0F0F0"), 80: ("Permanent water bodies", "#0064C8"), 90: ("Herbaceous wetland", "#0096A0"),
           95: ("Mangroves", "#00CF75"), 100: ("Moss and lichen", "#FAE6A0")}


def tile_urls(bounds) -> list[str]:
    w, s, e, n = bounds
    urls = []
    for la in range(math.floor(s / 3) * 3, math.floor(n / 3) * 3 + 1, 3):
        for lo in range(math.floor(w / 3) * 3, math.floor(e / 3) * 3 + 1, 3):
            ns = f"N{la:02d}" if la >= 0 else f"S{-la:02d}"
            ew = f"E{lo:03d}" if lo >= 0 else f"W{-lo:03d}"
            urls.append(f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")
    return urls


def land_cover(frame: SiteFrame, res: float = 10.0, pad_frac: float = 0.0):
    grid = Grid.for_frame(frame, res, pad_frac)
    out = None
    for u in tile_urls(frame.padded_bounds_ll(0.1)):
        a = read_to_grid(u, grid, frame.crs, resampling=Resampling.nearest)
        out = a if out is None else np.ma.where(np.ma.getmaskarray(out), a, out)
    return out, grid
