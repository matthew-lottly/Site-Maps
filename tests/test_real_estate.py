import json

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, box

from site_maps.sources.site_analysis import M_PER_MILE, footprint_stats, max_speed_mph, nearest_by_class, places_by_ring, population_in_ring, ring_geoms


def test_max_speed_units_and_missing():
    assert max_speed_mph(json.dumps([{"max_speed": {"value": 45, "unit": "mph"}}, {"max_speed": {"value": 30, "unit": "mph"}}])) == 45
    assert round(max_speed_mph(json.dumps([{"max_speed": {"value": 100, "unit": "km/h"}}])), 1) == 62.1
    assert max_speed_mph(None) is None
    assert max_speed_mph(json.dumps([{"max_speed": None}])) is None


def test_population_in_ring_uses_cell_centres():
    x = np.array([0, 100, 400, 2000.0]); y = np.zeros(4); p = np.array([5, 7, 11, 13.0])
    assert population_in_ring(x, y, p, (0, 0), 500) == 23
    assert population_in_ring(x, y, p, (0, 0), 50) == 5


def test_footprint_coverage_is_exact_for_a_square():
    ring = Point(0, 0).buffer(M_PER_MILE)
    sq = gpd.GeoDataFrame(geometry=[box(-100, -100, 100, 100), box(5000, 5000, 5100, 5100)], crs=32615)
    s = footprint_stats(sq, ring)
    assert s["buildings"] == 1 and abs(s["footprint_m2"] - 40000) < 1e-6
    assert abs(s["coverage"] - 40000 / ring.area) < 1e-12


def test_nearest_by_class_and_ring_counts():
    seg = gpd.GeoDataFrame({"class": ["primary", "primary", "secondary"], "name": ["A", "B", "C"], "speed_limits": [None] * 3,
                            "geometry": [LineString([(300, -50), (300, 50)]), LineString([(900, -50), (900, 50)]), LineString([(100, -50), (100, 50)])]}, crs=32615)
    t = nearest_by_class(Point(0, 0), seg, ("primary", "secondary", "trunk")).set_index("class")
    assert t.loc["primary", "name"] == "A" and t.loc["primary", "distance_m"] == 300 and t.loc["secondary", "distance_m"] == 100 and np.isnan(t.loc["trunk", "distance_m"])
    pl = gpd.GeoDataFrame({"category_group": ["shopping", "shopping", "health_care"], "geometry": [Point(100, 0), Point(1000, 0), Point(300, 0)]}, crs=32615)
    c = places_by_ring(pl, Point(0, 0), (0.25, 1.0))
    assert c.loc["shopping", "≤ 0.25 mi"] == 1 and c.loc["shopping", "≤ 1 mi"] == 2 and c.loc["health_care", "≤ 0.25 mi"] == 1
    assert set(ring_geoms(Point(0, 0))) == {0.25, 0.5, 1.0}
