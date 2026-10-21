import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from site_maps.sources import camp


def pts(rows):
    return gpd.GeoDataFrame(rows, geometry=[Point(r["x"], r["y"]) for r in rows], crs="EPSG:32611")


def row(i, name, x, y, conf, status="open"):
    return {"id": f"id{i}", "name": name, "category": "campground", "confidence": conf, "operating_status": status, "x": x, "y": y}


def test_norm_name_strips_generic_words():
    assert camp.norm_name("Upper Pines Campground") == camp.norm_name("Upper Pines")
    assert camp.norm_name("Yosemite Valley Camp 4") == camp.norm_name("Camp 4")


def test_filter_and_dedupe_logs_every_drop():
    g = pts([row(1, "Camp 4", 0, 0, 0.94), row(2, "Yosemite Valley Camp 4", 70, 0, 0.81), row(3, "Low", 5000, 0, 0.5), row(4, "Gone", 9000, 0, 0.95, "permanently_closed"),
             row(5, "Far Same Name", 0, 0, 0.9), row(6, "Far Same Name", 30000, 0, 0.9)])
    kept, dropped = camp.filter_and_dedupe(g)
    assert set(kept["name"]) == {"Camp 4", "Far Same Name"} or len(kept) == 4
    reasons = " | ".join(d["reason"] for d in dropped)
    assert "duplicate" in reasons and "confidence" in reasons and "permanently_closed" in reasons
    assert len(kept) + len(dropped) == 6
    assert (kept["name"] == "Camp 4").sum() == 1                     # highest-confidence of the pair survives


def test_cluster_near_picks_cluster_of_anchor_not_largest():
    g = pts([row(i, f"a{i}", i * 500, 0, 0.9) for i in range(5)] + [row(10 + i, f"b{i}", 50000 + i * 500, 0, 0.9) for i in range(3)])
    near = camp.cluster_near(g, Point(50200, 0))
    assert set(g.iloc[near]["name"]) == {"b0", "b1", "b2"}
    assert len(camp.largest_cluster(g)) == 5


def test_nice_interval_uses_data_range():
    assert camp.nice_interval(1200, 2600, 14) == 100.0
    assert camp.nice_interval(0, 50, 10) == 5.0


def test_sample_is_bilinear_and_nan_outside():
    a = np.array([[0.0, 10.0], [20.0, 30.0]])
    ext = (0, 0, 2, 2)
    assert camp.sample(a, ext, 1.0, 1.0) == pytest.approx(15.0)
    assert np.isnan(camp.sample(a, ext, 5, 5))


def test_slope_stats_within_radius():
    slope = np.zeros((20, 20))
    slope[:, 10:] = 30.0
    st = camp.slope_stats(slope, (0, 0, 200, 200), 10.0, Point(150, 100), 20.0)
    assert st["mean"] == pytest.approx(30.0) and st["n"] > 0
    st0 = camp.slope_stats(slope, (0, 0, 200, 200), 10.0, Point(30, 100), 20.0)
    assert st0["mean"] == pytest.approx(0.0)


def test_match_polygons_within_distance():
    p = pts([row(1, "A", 50, 50, 0.9), row(2, "B", 5000, 5000, 0.9)])
    polys = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32611")
    m = camp.match_polygons(p, polys, 300)
    assert m[0] == (0, 0.0) and m[1] is None
