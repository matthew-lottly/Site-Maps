"""Power-sector data: WRI Global Power Plant Database records and parsing of OSM power tags carried by Overture infrastructure features."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..core.cache import download
from ..core.provenance import Source

GPPD_URL = "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv"
GPPD = Source("WRI", "World Resources Institute (Global Power Plant Database)", "Global Power Plant Database", "v1.3.0 (WRI states it has not been maintained since early 2022)",
              "https://github.com/wri/global-power-plant-database", "CC BY 4.0", "Plant attributes are as of the database vintage, not today")

VOLTAGE_CLASSES = [("≥ 345 kV", 345.0, 1e9), ("230–344 kV", 230.0, 345.0), ("115–229 kV", 115.0, 230.0), ("< 115 kV", 0.0, 115.0)]
UNKNOWN = "voltage not tagged"
NE_SOURCE = Source("NE", "Natural Earth", "Admin 1 – States, Provinces 1:50m", "GitHub master at retrieval", "https://www.naturalearthdata.com/", "Public domain")


def load_gppd() -> pd.DataFrame:
    return pd.read_csv(download(GPPD_URL, "wri/global_power_plant_database.csv"), low_memory=False)


def tag_dict(source_tags) -> dict:
    """Overture stores OSM tags as JSON text of [key, value] pairs (see sources.overture._flatten)."""
    if source_tags is None or (isinstance(source_tags, float) and source_tags != source_tags):
        return {}
    try:
        return {k: v for k, v in json.loads(source_tags)}
    except (ValueError, TypeError):
        return {}


def max_voltage_kv(tags: dict) -> float | None:
    """OSM `voltage` is volts, possibly several values separated by ';' (e.g. '138000;34500'). Returns the highest in kV, or None."""
    v = tags.get("voltage")
    if not v:
        return None
    vals = []
    for part in str(v).split(";"):
        try:
            vals.append(float(part.strip()) / 1000.0)
        except ValueError:
            continue
    return max(vals) if vals else None


def voltage_class(kv: float | None) -> str:
    if kv is None or not np.isfinite(kv):
        return UNKNOWN
    for name, lo, hi in VOLTAGE_CLASSES:
        if lo <= kv < hi:
            return name
    return UNKNOWN


def length_by_class_km(lines, center, radius_m: float) -> dict[str, float]:
    """Kilometres of line (projected metres) inside a circle around `center`, per voltage class. `lines` needs geometry and a 'vclass' column."""
    from shapely.geometry import Point

    disc = Point(center).buffer(radius_m, 128)
    out: dict[str, float] = {}
    for cls, g in lines.groupby("vclass"):
        out[cls] = float(g.geometry.intersection(disc).length.sum() / 1000.0)
    return out


def counts_by_band(points, center, edges_m) -> list[int]:
    """Count of point features per distance band [e0,e1), [e1,e2)... from `center`."""
    d = np.hypot(points.geometry.x.to_numpy() - center[0], points.geometry.y.to_numpy() - center[1])
    return [int(((d >= a) & (d < b)).sum()) for a, b in zip(edges_m[:-1], edges_m[1:])]
