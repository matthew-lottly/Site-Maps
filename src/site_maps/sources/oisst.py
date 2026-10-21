"""NOAA Optimum Interpolation SST (OISST v2.1, AVHRR-only, 0.25°) daily files from the NOAA CDR bucket on AWS."""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import xarray as xr

from ..core.cache import cache_path
from ..core.provenance import Source

BASE = "https://noaa-cdr-sea-surface-temp-optimum-interpolation-pds.s3.amazonaws.com/data/v2.1/avhrr"
SOURCE = Source("OISST", "NOAA National Centers for Environmental Information", "Daily Optimum Interpolation SST (OISST) v2.1, AVHRR-only, 0.25° grid", "daily files; dates on the sheet",
                "https://www.ncei.noaa.gov/products/optimum-interpolation-sst", "Public domain (U.S. Government work)",
                "Satellite + in-situ blended analysis; a recent file may be the preliminary version where no final file exists yet")


def _fetch(date: dt.date) -> tuple[str, str] | None:
    for suffix in ("", "_preliminary"):
        name = f"oisst-avhrr-v02r01.{date:%Y%m%d}{suffix}.nc"
        p = cache_path(f"oisst/{name}")
        if p.exists() and p.stat().st_size > 0:
            return str(p), "preliminary" if suffix else "final"
        r = requests.get(f"{BASE}/{date:%Y%m}/{name}", timeout=120)
        if r.status_code == 200:
            p.write_bytes(r.content)
            return str(p), "preliminary" if suffix else "final"
    return None


def box_mean(date: dt.date, bounds: tuple[float, float, float, float]):
    """Mean SST (°C) over the valid ocean cells inside `bounds` (W, S, E, N), the number of cells, and whether the file was final or preliminary."""
    got = _fetch(date)
    if got is None:
        return None
    ds = xr.open_dataset(got[0])
    W, S, E, N = bounds
    sub = ds["sst"].isel(time=0, zlev=0).sel(lat=slice(S, N), lon=slice(W % 360, E % 360))
    v = sub.values
    ok = np.isfinite(v)
    return (float(v[ok].mean()) if ok.any() else None), int(ok.sum()), got[1]


def series(dates: list[dt.date], bounds, workers: int = 8):
    # downloads run in threads; the netCDF/HDF5 library is not thread-safe, so files are opened one at a time
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_fetch, dates))
    return [box_mean(d, bounds) for d in dates]
