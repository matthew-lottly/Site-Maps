"""Campground selection logic: confidence filter, name/distance de-duplication, valley clustering, facility distances. Pure functions on projected GeoDataFrames."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import geopandas as gpd

MIN_CONFIDENCE = 0.80           # Overture place confidence below this is dropped (entries at exactly 0.77 carry no operating status and are all conflated defaults)
DEDUPE_M = 1500.0               # same normalised name within this distance = one campground
CLUSTER_M = 3000.0              # campgrounds closer than this (single linkage) belong to the same cluster

_STOP = re.compile(r"\b(campground|camp ground|yosemite valley|yosemite|national park|np|camp site|campsite)\b", re.I)


def norm_name(name: str) -> str:
    n = _STOP.sub(" ", name.lower())
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return re.sub(r"\s+", " ", n).strip() or name.lower()


def filter_and_dedupe(places: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Keep category 'campground', confidence >= MIN_CONFIDENCE and not marked closed. Among entries with the same normalised name
    within DEDUPE_M keep the one with the highest confidence (operating_status 'open' wins ties). Returns (kept, log of dropped rows)."""
    dropped: list[dict] = []
    c = places[places["category"] == "campground"].copy()
    keep_mask = []
    for _, r in c.iterrows():
        reason = None
        if r["confidence"] < MIN_CONFIDENCE:
            reason = f"confidence {r['confidence']:.2f} < {MIN_CONFIDENCE}"
        elif str(r.get("operating_status")).lower() in ("closed", "permanently_closed", "temporarily_closed"):
            reason = f"operating_status {r['operating_status']}"
        keep_mask.append(reason is None)
        if reason:
            dropped.append({"name": r["name"], "id": r["id"], "reason": reason})
    c = c[np.array(keep_mask, dtype=bool)].copy()
    c["_key"] = c["name"].map(norm_name)
    c["_rank"] = c["confidence"] + (c["operating_status"].astype(str).str.lower() == "open") * 0.001
    c = c.sort_values("_rank", ascending=False)
    kept_idx: list = []
    for i, r in c.iterrows():
        dup = [j for j in kept_idx if c.loc[j, "_key"] == r["_key"] and c.loc[j, "geometry"].distance(r.geometry) <= DEDUPE_M]
        if dup:
            dropped.append({"name": r["name"], "id": r["id"], "reason": f"duplicate of '{c.loc[dup[0], 'name']}' (same name key, {c.loc[dup[0], 'geometry'].distance(r.geometry):.0f} m)"})
        else:
            kept_idx.append(i)
    return c.loc[kept_idx].drop(columns=["_key", "_rank"]), dropped


def cluster_labels(points: gpd.GeoDataFrame, link_m: float = CLUSTER_M) -> np.ndarray:
    """Single-linkage cluster label per point (points closer than `link_m` share a label)."""
    xy = np.c_[points.geometry.x, points.geometry.y]
    n = len(xy)
    lab = -np.ones(n, dtype=int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = k
        while stack:
            a = stack.pop()
            d = np.hypot(*(xy - xy[a]).T)
            for b in np.where((d <= link_m) & (lab < 0))[0]:
                lab[b] = k
                stack.append(b)
        k += 1
    return lab


def cluster_near(points: gpd.GeoDataFrame, anchor, link_m: float = CLUSTER_M) -> np.ndarray:
    """Positional indices of the cluster that contains the point nearest to `anchor` (a shapely Point in the same CRS)."""
    lab = cluster_labels(points, link_m)
    nearest = int(np.argmin(points.geometry.distance(anchor).to_numpy()))
    return np.where(lab == lab[nearest])[0]


def largest_cluster(points: gpd.GeoDataFrame, link_m: float = CLUSTER_M) -> np.ndarray:
    """Indices (positional) of the largest single-linkage cluster of points; ties broken by summed confidence."""
    lab = cluster_labels(points, link_m)
    conf = points["confidence"].to_numpy()
    best = max(set(lab), key=lambda c: ((lab == c).sum(), conf[lab == c].sum()))
    return np.where(lab == best)[0]


def nearest_distance(origin, targets: gpd.GeoSeries) -> tuple[float, object]:
    """Distance (m) from a point to the nearest target geometry and that target's index."""
    if len(targets) == 0:
        return float("nan"), None
    d = targets.distance(origin)
    return float(d.min()), d.idxmin()


def nice_interval(zmin: float, zmax: float, target_lines: int = 14) -> float:
    """Contour interval from the data range: the 'nice' step (1,2,5 x 10^k) giving about `target_lines` lines."""
    span = max(zmax - zmin, 1.0)
    raw = span / target_lines
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return float(m * mag)
    return float(10 * mag)


def drop_outside_cluster(kept: gpd.GeoDataFrame, cluster_idx: np.ndarray, centre) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """Campgrounds outside the chosen cluster are dropped; the log gives the distance to the cluster centre so displaced points are visible."""
    inside = kept.iloc[cluster_idx]
    out = kept.drop(index=inside.index)
    return inside, [{"name": r["name"], "id": r["id"], "reason": f"outside the valley cluster ({r.geometry.distance(centre) / 1000:.1f} km from its centre)"} for _, r in out.iterrows()]


def match_polygons(points: gpd.GeoDataFrame, polys: gpd.GeoDataFrame, max_m: float = 300.0) -> dict:
    """For each point, the nearest polygon (index into `polys`) within `max_m`, else None. Returns {point index: (poly index, distance m)}."""
    out = {}
    for i, r in points.iterrows():
        if len(polys) == 0:
            out[i] = None
            continue
        d = polys.geometry.distance(r.geometry)
        out[i] = (d.idxmin(), float(d.min())) if d.min() <= max_m else None
    return out


def slope_stats(slope: np.ndarray, grid_extent, res: float, geom, radius_m: float = 100.0) -> dict:
    """Mean / 90th-percentile / max slope (degrees) of the cells within `radius_m` of a geometry."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    x0, y0, x1, y1 = grid_extent
    h, w = slope.shape
    m = rasterize([(geom.buffer(radius_m), 1)], out_shape=(h, w), transform=from_bounds(x0, y0, x1, y1, w, h), fill=0, dtype="uint8").astype(bool)
    v = slope[m & np.isfinite(slope)]
    if len(v) == 0:
        return {"mean": float("nan"), "p90": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "p90": float(np.percentile(v, 90)), "max": float(v.max()), "n": int(len(v))}


def sample(arr: np.ndarray, grid_extent, x: float, y: float) -> float:
    """Bilinear sample of a north-up raster at (x, y); NaN outside."""
    x0, y0, x1, y1 = grid_extent
    h, w = arr.shape
    fx, fy = (x - x0) / (x1 - x0) * w - 0.5, (y1 - y) / (y1 - y0) * h - 0.5
    if not (0 <= fx <= w - 1 and 0 <= fy <= h - 1):
        return float("nan")
    ix, iy = int(np.floor(fx)), int(np.floor(fy))
    ix, iy = min(ix, w - 2), min(iy, h - 2)
    tx, ty = fx - ix, fy - iy
    a = arr[iy:iy + 2, ix:ix + 2]
    return float(a[0, 0] * (1 - tx) * (1 - ty) + a[0, 1] * tx * (1 - ty) + a[1, 0] * (1 - tx) * ty + a[1, 1] * tx * ty)
