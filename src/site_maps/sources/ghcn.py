"""NOAA GHCN-Daily (AWS Open Data): nearest long-record station and monthly climate normals computed from the daily values."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..core.cache import download
from ..core.provenance import Source

BASE = "https://noaa-ghcn-pds.s3.amazonaws.com"
SOURCE = Source("GHCN", "NOAA National Centers for Environmental Information", "Global Historical Climatology Network – Daily (GHCN-Daily)", "AWS Open Data snapshot at retrieval",
                "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily", "Public domain (U.S. Government work)",
                "Normals computed here from daily values; quality-flagged values removed")


def stations() -> pd.DataFrame:
    p = download(f"{BASE}/ghcnd-stations.txt", "ghcn/ghcnd-stations.txt")
    rows = []
    for ln in open(p, encoding="utf-8", errors="ignore"):
        rows.append((ln[0:11], float(ln[12:20]), float(ln[21:30]), float(ln[31:37]), ln[41:71].strip()))
    return pd.DataFrame(rows, columns=["id", "lat", "lon", "elev_m", "name"])


def haversine_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def daily(station_id: str) -> pd.DataFrame:
    p = download(f"{BASE}/csv/by_station/{station_id}.csv", f"ghcn/{station_id}.csv")
    d = pd.read_csv(p, dtype={"Q_FLAG": str}, usecols=["DATE", "ELEMENT", "DATA_VALUE", "Q_FLAG"], low_memory=False)
    d = d[d["Q_FLAG"].isna() & d["ELEMENT"].isin(["TMAX", "TMIN", "PRCP"])]
    d["DATE"] = pd.to_datetime(d["DATE"].astype(str), format="%Y%m%d")
    return d.pivot_table(index="DATE", columns="ELEMENT", values="DATA_VALUE", aggfunc="first")


def best_station(lon: float, lat: float, max_km: float = 90, years=(1991, 2020), min_years: int = 20, limit: int = 30):
    """Nearest station with enough complete TMAX, TMIN and PRCP years in the normals period. Returns (row, daily frame, distance km)."""
    st = stations()
    st["km"] = haversine_km(lon, lat, st["lon"].to_numpy(), st["lat"].to_numpy())
    st = st[st["id"].str.startswith(("USC", "USW"))]            # NWS cooperative / airport stations have the long records
    for _, r in st[st["km"] <= max_km].sort_values("km").head(limit).iterrows():
        try:
            d = daily(r["id"])
        except Exception:
            continue
        d = d[(d.index.year >= years[0]) & (d.index.year <= years[1])]
        if not {"TMAX", "TMIN", "PRCP"} <= set(d.columns):
            continue
        if d.empty:
            continue
        full = [y for y, g in d.groupby(d.index.year) if min(g[c].notna().sum() for c in ("TMAX", "TMIN", "PRCP")) >= 300]
        if len(full) >= min_years:
            return r, d[d.index.year.isin(full)], float(r["km"])
    return None, None, None


def monthly_normals(d: pd.DataFrame) -> pd.DataFrame:
    """Mean daily max/min temperature (°C) and mean monthly precipitation total (mm) over the years present. GHCN stores tenths of °C and tenths of mm."""
    t = d.copy()
    t["TMAX"] /= 10.0
    t["TMIN"] /= 10.0
    t["PRCP"] /= 10.0
    m = t.groupby([t.index.year, t.index.month]).agg(tmax=("TMAX", "mean"), tmin=("TMIN", "mean"), prcp=("PRCP", "sum"))
    return m.groupby(level=1).mean().rename_axis("month")
