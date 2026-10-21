"""Overture Maps (release pinned) read straight from S3 with a bounding-box filter."""
from __future__ import annotations

import os

import geopandas as gpd
import pyarrow.dataset as ds
import shapely
from pyarrow import fs

from ..core.cache import cache_path
from ..core.frame import SiteFrame
from ..core.provenance import Source

RELEASE = "2026-09-23.1"
S3_PATH = f"overturemaps-us-west-2/release/{RELEASE}"
SOURCE = Source("OV", "Overture Maps Foundation", f"Overture Maps {RELEASE}", RELEASE, "https://docs.overturemaps.org/",
                "ODbL / CDLA-Permissive-2.0 per theme; see source tags (OpenStreetMap data © OpenStreetMap contributors)")

THEMES = {"places": "theme=places/type=place", "segments": "theme=transportation/type=segment", "buildings": "theme=buildings/type=building",
          "land_use": "theme=base/type=land_use", "land": "theme=base/type=land", "water": "theme=base/type=water", "infrastructure": "theme=base/type=infrastructure",
          "land_cover": "theme=base/type=land_cover", "bathymetry": "theme=base/type=bathymetry", "connectors": "theme=transportation/type=connector",
          "divisions": "theme=divisions/type=division_area"}


def _fs() -> fs.S3FileSystem:
    return fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))


def read(kind: str, bounds_ll: tuple[float, float, float, float], columns: list[str], extra_filter=None, crs=None, tag: str = "") -> gpd.GeoDataFrame:
    """Features of one Overture type whose bounding box lies inside `bounds_ll` (W,S,E,N). Cached as parquet per request."""
    W, S, E, N = bounds_ll
    key = f"overture/{RELEASE}/{kind}_{tag}_{W:.4f}_{S:.4f}_{E:.4f}_{N:.4f}.parquet"
    p = cache_path(key)
    if p.exists():
        g = gpd.read_parquet(p)
    else:
        flt = (ds.field("bbox", "xmin") < E) & (ds.field("bbox", "xmax") > W) & (ds.field("bbox", "ymin") < N) & (ds.field("bbox", "ymax") > S)
        if extra_filter is not None:
            flt = flt & extra_filter
        d = ds.dataset(f"{S3_PATH}/{THEMES[kind]}", filesystem=_fs(), format="parquet")
        cols = list(dict.fromkeys(columns + ["geometry"]))
        t = d.to_table(filter=flt, columns=cols).to_pandas()
        geom = shapely.from_wkb(t["geometry"].to_numpy())
        t = _flatten(t.drop(columns=["geometry"]))
        g = gpd.GeoDataFrame(t, geometry=geom, crs="EPSG:4326")
        g.to_parquet(p)
    return g.to_crs(crs) if crs is not None else g


def _flatten(t):
    """Parquet struct columns -> plain columns: names -> name, taxonomy -> category fields, brand -> brand_name, anything else nested -> JSON text."""
    import json

    def js(v):
        if v is None or (isinstance(v, float) and v != v):
            return None
        return json.dumps(v.tolist() if hasattr(v, "tolist") else v, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    if "names" in t:
        t["name"] = t["names"].map(primary_name)
        t = t.drop(columns=["names"])
    if "taxonomy" in t:
        t["category"] = t["taxonomy"].map(lambda x: x["primary"] if x is not None else None)
        t["category_group"] = t["taxonomy"].map(lambda x: x["hierarchy"][0] if x is not None and len(x["hierarchy"]) else None)
        t = t.drop(columns=["taxonomy"])
    if "brand" in t:
        t["brand_name"] = t["brand"].map(lambda b: primary_name(b["names"]) if b is not None and b.get("names") is not None else None)
        t = t.drop(columns=["brand"])
    if "bbox" in t:
        t = t.drop(columns=["bbox"])
    for c in list(t.columns):
        if t[c].map(lambda v: isinstance(v, (dict, list)) or hasattr(v, "tolist")).any():
            t[c] = t[c].map(js)
    return t


def primary_name(names) -> str | None:
    return names.get("primary") if isinstance(names, dict) else None
