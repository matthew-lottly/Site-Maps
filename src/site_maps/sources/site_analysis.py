"""Pure geometry / table helpers for the real-estate site analysis (no network)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

M_PER_MILE = 1609.344
RINGS_MI = (0.25, 0.5, 1.0)
ARTERIALS = ("motorway", "trunk", "primary", "secondary", "tertiary")


def max_speed_mph(js) -> float | None:
    """Largest posted maximum speed in an Overture `speed_limits` JSON string, converted to mph (None if absent)."""
    if js is None or (isinstance(js, float) and js != js):
        return None
    best = None
    for rule in json.loads(js):
        ms = (rule or {}).get("max_speed") or {}
        if ms.get("value") is None:
            continue
        v = float(ms["value"]) * (0.621371 if ms.get("unit") == "km/h" else 1.0)
        best = v if best is None else max(best, v)
    return best


def ring_geoms(center: Point, radii_mi=RINGS_MI):
    return {r: center.buffer(r * M_PER_MILE, 128) for r in radii_mi}


def population_in_ring(x: np.ndarray, y: np.ndarray, people: np.ndarray, center: tuple[float, float], radius_m: float) -> float:
    """Sum of cell values whose centres lie within `radius_m` of `center` (cells straddling the ring are counted by centre)."""
    return float(people[np.hypot(x - center[0], y - center[1]) <= radius_m].sum())


def footprint_stats(bld: gpd.GeoDataFrame, ring) -> dict:
    inside = bld.geometry.intersection(ring)
    inside = inside[~inside.is_empty]
    area = float(inside.area.sum())
    return {"buildings": int(len(inside)), "footprint_m2": area, "coverage": area / ring.area}


def nearest_by_class(point: Point, seg: gpd.GeoDataFrame, classes=ARTERIALS) -> pd.DataFrame:
    rows = []
    for c in classes:
        s = seg[seg["class"] == c]
        if s.empty:
            rows.append({"class": c, "distance_m": np.nan, "name": None, "max_mph": None})
            continue
        d = s.geometry.distance(point)
        i = d.idxmin()
        rows.append({"class": c, "distance_m": float(d.loc[i]), "name": s.loc[i].get("name"), "max_mph": max_speed_mph(s.loc[i].get("speed_limits"))})
    return pd.DataFrame(rows)


def places_by_ring(places: gpd.GeoDataFrame, center: Point, radii_mi=RINGS_MI, group_col="category_group") -> pd.DataFrame:
    """Counts per category group in each distance band (cumulative: everything within r)."""
    d = places.geometry.distance(center) / M_PER_MILE
    out = {}
    for r in radii_mi:
        out[f"≤ {r:g} mi"] = places[d <= r].groupby(group_col).size()
    return pd.DataFrame(out).fillna(0).astype(int)
