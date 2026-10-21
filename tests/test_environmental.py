import numpy as np
import pandas as pd

from site_maps.sources import ghcn
from site_maps.sources.sentinel2 import _boa_offset, tile_for
from site_maps.sources.worldcover import CLASSES, tile_urls


def test_worldcover_tile_naming():
    urls = tile_urls((-95.66, 29.33, -95.55, 29.41))
    assert len(urls) == 1 and "N27W096" in urls[0]
    assert len(tile_urls((-96.5, 29.5, -95.5, 29.6))) == 2          # 96.5 W lies in the W099 tile, 95.5 W in W096
    assert set(CLASSES) >= {10, 80, 90}


def test_sentinel_tile_and_offset_detection():
    assert tile_for(-95.6, 29.38) == ("15", "R", "TN")
    dark = np.ma.masked_array(np.full((50, 50), 300.0))
    bright = np.ma.masked_array(np.full((50, 50), 1500.0))
    assert _boa_offset(dark, bright) == 0.0                  # darkest pixels well below 1000 DN: no offset in the data
    assert _boa_offset(bright, bright) == -0.1               # nothing below 1000 DN: the -0.1 offset is present


def test_ghcn_units_and_normals():
    idx = pd.date_range("2000-01-01", "2001-12-31")
    d = pd.DataFrame({"TMAX": 300.0, "TMIN": 100.0, "PRCP": 10.0}, index=idx)      # tenths of °C and tenths of mm
    n = ghcn.monthly_normals(d)
    assert n.loc[1, "tmax"] == 30.0 and n.loc[1, "tmin"] == 10.0
    assert round(n.loc[1, "prcp"], 1) == 31.0                  # 31 days x 1.0 mm
    assert ghcn.haversine_km(0, 0, 0, 1) - 111.2 < 0.5
