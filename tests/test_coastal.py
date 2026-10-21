import datetime as dt

import numpy as np
import pytest

from site_maps.maps import coastal
from site_maps.sources import bathymetry as bt, nos_ofs, obis
from site_maps.sources.rasters import Grid


def _grid(n=40, res=100.0):
    return Grid((500000.0, 4700000.0, 500000.0 + n * res, 4700000.0 + n * res), res)


def test_cudem_tile_name_is_upper_left_corner(monkeypatch):
    m = bt._TILE.search("ncei13_n42x50_w070x75_2021v1.tif")
    assert (float(f"{m.group(1)}.{m.group(2)}"), -float(f"{m.group(3)}.{m.group(4)}")) == (42.5, -70.75)
    assert bt._TILE.search("ncei13_n29X50_w084X75_2019v1.tif")                  # upper-case X variants exist in the bucket
    b = bt.tile_box(42.5, -70.75)
    assert b.bounds == (-70.75, 42.25, -70.5, 42.5)


def test_scale_factor_and_cell_area():
    assert bt.utm_scale_factor(np.array([500000.0]))[0] == pytest.approx(0.9996)
    assert bt.utm_scale_factor(np.array([700000.0]))[0] > 0.9996                  # grows away from the central meridian
    g = _grid(10, 100.0)
    a = bt.cell_area_km2(g)
    assert a.shape == (10, 10) and a[0, 0] == pytest.approx(0.01 / 0.9996 ** 2, rel=1e-3)


def test_depth_zones_and_hypsometry_on_a_ramp():
    g = _grid(40, 100.0)
    z = np.ma.masked_invalid(np.tile(np.linspace(20, -60, 40), (40, 1)))        # land on the left, 60 m deep on the right
    rows = bt.depth_zone_areas(z, g, edges=(0, 20, 40, 60))
    assert [r[0] for r in rows] == ["0–20 m", "20–40 m", "40–60 m", "> 60 m"]
    assert sum(r[2] for r in rows) == pytest.approx(1.0)
    d, area = bt.hypsometric_curve(z, g)
    assert area[0] == pytest.approx(sum(r[1] for r in rows)) and np.all(np.diff(area) <= 1e-9)


def test_deepest_land_and_transect():
    g = _grid(30, 50.0)
    z = np.tile(np.linspace(10, -100, 30), (30, 1))
    z[5, 25] = -250.0
    deep = bt.deepest_cell(z)
    assert deep == (5, 25)
    land = bt.nearest_land_cell(z, g, deep)
    assert z[land] >= 0
    d, v, (x, y) = bt.transect(z, g, land, deep)
    assert d[0] == 0 and d[-1] == pytest.approx(np.hypot(deep[0] - land[0], deep[1] - land[1]) * 50 / 1000)
    assert v[-1] == pytest.approx(-250.0, abs=1.0)


def test_row_group_selection_uses_footer_ranges():
    idx = {"a.parquet": [[[0, 42.0, 43.0, -71.0, -70.0], [1, 10.0, 11.0, 5.0, 6.0]]], "b.parquet": [[[0, 0.0, 1.0, 0.0, 1.0]]], "bad.parquet": [None, "err"]}
    assert obis.select_row_groups(idx, (-70.5, 42.2, -70.4, 42.3)) == [("a.parquet", [0])]


def test_month_dates_and_sample_grid():
    ds = coastal.month_dates(12, 15, today=dt.date(2026, 10, 21))
    assert ds[0] == dt.date(2025, 10, 15) and ds[-1] == dt.date(2026, 9, 15) and len(ds) == 12
    g = _grid(4, 10.0)
    arr = np.arange(16.0).reshape(4, 4)
    out = coastal.sample_grid(arr, g, [500005.0, 500035.0, 600000.0], [4700035.0, 4700005.0, 4700005.0])
    assert out[0] == 0.0 and out[1] == 15.0 and np.isnan(out[2])


def test_ofs_stamp_ordering_prefers_latest_cycle():
    keys = ["g/gomofs.t00z.20261020.2ds.n006.nc", "g/gomofs.t18z.20261020.2ds.n006.nc", "g/gomofs.t18z.20261020.2ds.n003.nc"]
    import re
    def stamp(k):
        m = re.search(r"\.t(\d\d)z\.(\d{8})\.2ds\.n(\d+)\.nc", k)
        return (m.group(2), int(m.group(1)), int(m.group(3)))
    assert sorted(keys, key=stamp)[-1].endswith("t18z.20261020.2ds.n006.nc")
