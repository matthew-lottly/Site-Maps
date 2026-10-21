"""Shared bits for the five site-map types."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from matplotlib.colors import to_rgba
from shapely.geometry import box
from shapely.ops import unary_union

from ..core.frame import SiteFrame
from ..core.layout import Sheet, halo
from ..core.provenance import ProvenanceLog
from ..sources import overture as ov


def raster_extent(grid) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = grid.extent
    return (x0, x1, y0, y1)


def mask_outside(arr: np.ndarray, geom, grid, crs=None) -> np.ndarray:
    """Boolean raster, True inside `geom` (projected like the grid)."""
    from rasterio.features import rasterize

    return rasterize([(geom, 1)], out_shape=grid.shape, transform=grid.transform, fill=0, dtype="uint8").astype(bool)


def clip_to_frame(g: gpd.GeoDataFrame, frame: SiteFrame) -> gpd.GeoDataFrame:
    x0, y0, x1, y1 = frame.extent
    win = box(x0, y0, x1, y1)
    g = g[g.intersects(win)].copy()
    g["geometry"] = g.geometry.intersection(win)
    return g[~g.geometry.is_empty]


def overture_layer(kind: str, frame: SiteFrame, columns: list[str], prov: ProvenanceLog, layer: str, detail: str = "", extra_filter=None, tag: str = "") -> gpd.GeoDataFrame:
    g = ov.read(kind, frame.padded_bounds_ll(0.12), columns, extra_filter=extra_filter, crs=frame.crs, tag=tag)
    prov.add(layer, ov.SOURCE, len(g), detail or f"Overture '{kind}' type, bounding-box filter, reprojected to {frame.epsg}")
    return g


def label_lines(ax, gdf, text_col: str, frame: SiteFrame, min_len_frac: float = 0.12, fontsize: float = 6.4, color="#1F2933", exclude=None, max_labels: int = 14, placer=None):
    """Name roads / rivers along the longest visible piece of each name, rotated to follow it."""
    x0, y0, x1, y1 = frame.extent
    W = x1 - x0
    view = box(x0, y0, x1, y1).buffer(-W * 0.02)
    placed = []
    n = 0
    for name, grp in gdf[gdf[text_col].notna()].groupby(text_col):
        best = None
        for geom in grp.geometry:
            if geom is None or geom.is_empty:
                continue
            part = geom.intersection(view)
            if part.is_empty:
                continue
            parts = list(part.geoms) if part.geom_type.startswith("Multi") else [part]
            for p in parts:
                if p.geom_type == "LineString" and (best is None or p.length > best.length):
                    best = p
        if best is None or best.length < W * min_len_frac:
            continue
        mid = best.interpolate(0.5, normalized=True)
        if exclude is not None and exclude(mid.x, mid.y):
            continue
        if any(np.hypot(mid.x - q[0], mid.y - q[1]) < W * 0.14 for q in placed):
            continue
        a, c = best.interpolate(0.46, normalized=True), best.interpolate(0.54, normalized=True)
        ang = np.degrees(np.arctan2(c.y - a.y, c.x - a.x))
        ang = ang + 180 if ang > 90 or ang < -90 else ang
        if placer is not None:
            if placer.place_centered(mid.x, mid.y, name, rotation=ang, fontsize=fontsize, color=color) is None:
                continue
        else:
            ax.text(mid.x, mid.y, name, rotation=ang, rotation_mode="anchor", ha="center", va="center", fontsize=fontsize, color=color, zorder=50, path_effects=halo(2.4))
        placed.append((mid.x, mid.y))
        n += 1
        if n >= max_labels:
            break
