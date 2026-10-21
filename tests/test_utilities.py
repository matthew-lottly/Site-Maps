import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from site_maps.sources import power


def test_voltage_parsing():
    assert power.max_voltage_kv({"voltage": "138000;34500"}) == 138.0
    assert power.max_voltage_kv({"voltage": "345000"}) == 345.0
    assert power.max_voltage_kv({"voltage": "high"}) is None
    assert power.max_voltage_kv({}) is None


def test_voltage_classes():
    assert power.voltage_class(345) == "≥ 345 kV"
    assert power.voltage_class(138) == "115–229 kV"
    assert power.voltage_class(34.5) == "< 115 kV"
    assert power.voltage_class(None) == power.UNKNOWN


def test_tag_dict_handles_json_and_missing():
    assert power.tag_dict('[["power","line"],["voltage","138000"]]') == {"power": "line", "voltage": "138000"}
    assert power.tag_dict(None) == {}
    assert power.tag_dict("not json") == {}


def test_length_by_class_clips_to_radius():
    g = gpd.GeoDataFrame({"vclass": ["≥ 345 kV"]}, geometry=[LineString([(-10000, 0), (10000, 0)])], crs="EPSG:32615")
    out = power.length_by_class_km(g, (0, 0), 5000)
    assert out["≥ 345 kV"] == pytest.approx(10.0, rel=0.01)


def test_counts_by_band():
    pts = gpd.GeoDataFrame(geometry=[Point(500, 0), Point(1500, 0), Point(1600, 0), Point(9000, 0)], crs="EPSG:32615")
    assert power.counts_by_band(pts, (0, 0), [0, 1000, 2000, 10000]) == [1, 2, 1]
