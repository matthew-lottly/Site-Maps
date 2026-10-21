"""Where is a point? State / country names come from Natural Earth admin-1 polygons, never from typed text."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from .cache import download

_ADM = None


def admin1():
    global _ADM
    if _ADM is None:
        _ADM = gpd.read_file(download("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson", "naturalearth/admin1_50m.geojson"))
    return _ADM


def context(lon: float, lat: float) -> dict:
    """{'region': state/province name, 'country': country name} of the admin-1 polygon containing the point (nearest if it falls offshore)."""
    a = admin1()
    p = Point(lon, lat)
    hit = a[a.contains(p)]
    if hit.empty:
        hit = a.iloc[[a.distance(p).idxmin()]]
    r = hit.iloc[0]
    return {"region": r.get("name"), "country": r.get("admin"), "iso": r.get("iso_a2"), "postal": r.get("postal")}
