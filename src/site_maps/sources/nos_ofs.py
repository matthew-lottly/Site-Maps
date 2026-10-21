"""NOAA NOS Operational Forecast Systems (ROMS-based) from the AWS bucket: the latest surface nowcast fields (temperature, currents, free surface).

Model output is not observation: it is the operational nowcast of a numerical model, labelled with its cycle and valid time on the sheet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..core.cache import download, s3_listing
from ..core.provenance import Source

BASE = "https://noaa-nos-ofs-pds.s3.amazonaws.com"


def source(ofs: str) -> Source:
    return Source("OFS", "NOAA National Ocean Service, Coast Survey Development Laboratory", f"{ofs.upper()} operational nowcast, 2-D surface fields (ROMS)", "cycle shown on the sheet",
                  "https://registry.opendata.aws/noaa-nos-cora/", "Public domain (U.S. Government work)", "Model output, not observation")


@dataclass
class Surface:
    lon: np.ndarray
    lat: np.ndarray
    temp: np.ndarray            # degC
    east: np.ndarray            # m/s (rotated to geographic east)
    north: np.ndarray           # m/s
    zeta: np.ndarray            # m (free surface)
    mask: np.ndarray
    valid_time: str
    cycle: str
    file: str
    title: str


def latest_nowcast_key(ofs: str, max_days_back: int = 10) -> str:
    """Key of the last nowcast 2-D file of the most recent day that has any."""
    import datetime as dt

    d = dt.date.today()
    for _ in range(max_days_back):
        ks = [k for k in s3_listing(BASE, f"{ofs}/netcdf/{d:%Y/%m/%d}/", None) if ".2ds.n" in k and k.endswith(".nc")]
        if ks:
            def stamp(k):
                m = re.search(r"\.t(\d\d)z\.(\d{8})\.2ds\.n(\d+)\.nc", k)
                return (m.group(2), int(m.group(1)), int(m.group(3)))
            return sorted(ks, key=stamp)[-1]
        d -= dt.timedelta(days=1)
    raise RuntimeError(f"no {ofs} nowcast found in the last {max_days_back} days")


def read_surface(ofs: str, bounds: tuple[float, float, float, float]) -> Surface:
    """Surface fields inside `bounds` (W, S, E, N). Staggered u/v are averaged to the rho points and rotated with the grid angle."""
    key = latest_nowcast_key(ofs)
    ds = xr.open_dataset(download(f"{BASE}/{key}", f"ofs/{key.split('/')[-1]}"))
    lon, lat = ds["lon_rho"].values, ds["lat_rho"].values
    W, S, E, N = bounds
    sel = (lon >= W) & (lon <= E) & (lat >= S) & (lat <= N)
    rows, cols = np.where(sel)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    sl = (slice(r0, r1), slice(c0, c1))
    u = ds["u_sur"].values[0]
    v = ds["v_sur"].values[0]
    u_rho = np.full(lon.shape, np.nan)
    v_rho = np.full(lon.shape, np.nan)
    u_rho[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])
    v_rho[1:-1, :] = 0.5 * (v[:-1, :] + v[1:, :])
    ang = ds["angle"].values
    east = u_rho * np.cos(ang) - v_rho * np.sin(ang)
    north = u_rho * np.sin(ang) + v_rho * np.cos(ang)
    m = re.search(r"\.t(\d\d)z\.(\d{8})\.", key)
    return Surface(lon[sl], lat[sl], ds["temp_sur"].values[0][sl], east[sl], north[sl], ds["zeta"].values[0][sl], ds["mask_rho"].values[sl],
                   str(ds["ocean_time"].values[0])[:16].replace("T", " ") + " UTC", f"{m.group(2)} t{m.group(1)}z", key.split("/")[-1], str(ds.attrs.get("title", "")))
