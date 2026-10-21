"""ESA WorldCover 10 m land cover (2021, v200) read from the public AWS bucket."""
from __future__ import annotations

import math

import numpy as np

from ..core.frame import SiteFrame
from ..core.provenance import Source
from .rasters import Grid, read_to_grid
from rasterio.enums import Resampling

SOURCE = Source("WC", "ESA WorldCover", "WorldCover 10 m 2021", "v200, reference year 2021", "https://esa-worldcover.org/", "CC BY 4.0",
                "Class names and colours read from the GeoTIFF legend and colour table")
_CLASSES = None


def classes() -> dict[int, tuple[str, str]]:
    """Class code -> (name, colour), read from the product itself: the GeoTIFF's embedded colour table and its `legend` tag."""
    global _CLASSES
    if _CLASSES is None:
        import rasterio

        from .rasters import remote_env

        with remote_env(), rasterio.open(tile_urls((-95.6, 29.4, -95.5, 29.5))[0]) as src:
            cm, legend = src.colormap(1), src.tags()["legend"]
        out = {}
        for line in legend.strip().splitlines():
            code, name = line.split(None, 1)
            r, g, b, _ = cm[int(code)]
            out[int(code)] = (name.strip().replace("Bare/sparse", "Bare / sparse"), "#%02X%02X%02X" % (r, g, b))
        _CLASSES = out
    return _CLASSES


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
