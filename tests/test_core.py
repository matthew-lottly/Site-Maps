import math

import pytest

from site_maps.core.frame import SiteFrame, utm_crs
from site_maps.core.layout import Legend, Sheet
from site_maps.core.provenance import ProvenanceLog, Source

SRC = Source("T", "Test publisher", "Test dataset", "v1", "https://example.org", "CC0")


def test_utm_zone_selection():
    assert utm_crs(-95.6, 29.4).to_epsg() == 32615
    assert utm_crs(-119.6, 37.7).to_epsg() == 32611
    assert utm_crs(151.2, -33.9).to_epsg() == 32756


def test_frame_extent_and_convergence():
    f = SiteFrame.around(-95.6, 29.4, 10000, 1.25)
    x0, y0, x1, y1 = f.extent
    assert math.isclose(x1 - x0, 10000) and math.isclose((y1 - y0) * 1.25, 10000)
    assert -3 < f.convergence_deg() < 3
    w, s, e, n = f.bounds_ll
    assert w < -95.6 < e and s < 29.4 < n


def test_provenance_dedupes_sources_and_renders_tables():
    log = ProvenanceLog()
    log.add("A", SRC, 3, "clip")
    log.add("B", SRC, 4)
    assert [s.key for s in log.sources()] == ["T"]
    md = log.to_markdown()
    assert "Test dataset" in md and "| A |" in md


def test_sheet_exports_png_pdf_and_passes_layout_audit(tmp_path):
    f = SiteFrame.around(-95.6, 29.4, 10000, Sheet.map_aspect())
    log = ProvenanceLog()
    log.add("demo", SRC, None, "none")
    sh = Sheet("Demo", "subtitle", "site line", f, log)
    sh.graticule(); sh.scale_bar(); sh.north_arrow()
    lg = Legend("Legend").header("H").patch("x", "#ccc").line("y", "#000")
    sh.legend(lg)
    sh.sources_box()
    out = sh.export("demo", tmp_path, dpi=60)
    assert out["png"].exists() and out["pdf"].exists()
    assert (tmp_path / "demo.provenance.json").exists()
    assert out["issues"] == []
