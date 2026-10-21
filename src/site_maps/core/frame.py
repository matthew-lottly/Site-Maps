"""Map frame: a site extent in a local UTM projection, with the numbers a research sheet needs (CRS, scale, true-north angle)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from pyproj import CRS, Transformer


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


@dataclass(frozen=True)
class SiteFrame:
    """`center` is lon/lat (WGS84); `width_m` is the map width in projected metres; `aspect` = width / height of the map window."""

    center: tuple[float, float]
    width_m: float
    aspect: float
    crs: CRS

    @classmethod
    def around(cls, lon: float, lat: float, width_m: float, aspect: float, crs: CRS | None = None) -> "SiteFrame":
        return cls((lon, lat), width_m, aspect, crs or utm_crs(lon, lat))

    @property
    def to_proj(self) -> Transformer:
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    @property
    def to_ll(self) -> Transformer:
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    @property
    def cx_cy(self) -> tuple[float, float]:
        return self.to_proj.transform(*self.center)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        cx, cy = self.cx_cy
        w, h = self.width_m, self.width_m / self.aspect
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    @property
    def bounds_ll(self) -> tuple[float, float, float, float]:
        """West, south, east, north of the window (sampled around the edge so a rotated grid is still covered)."""
        x0, y0, x1, y1 = self.extent
        pts = [(x, y) for x in (x0, (x0 + x1) / 2, x1) for y in (y0, (y0 + y1) / 2, y1)]
        lons, lats = zip(*[self.to_ll.transform(x, y) for x, y in pts])
        return min(lons), min(lats), max(lons), max(lats)

    def padded_bounds_ll(self, frac: float = 0.06) -> tuple[float, float, float, float]:
        w, s, e, n = self.bounds_ll
        dx, dy = (e - w) * frac, (n - s) * frac
        return w - dx, s - dy, e + dx, n + dy

    def convergence_deg(self) -> float:
        """Angle between grid north and true north at the centre (positive = grid north is east of true north)."""
        from pyproj import Proj

        return float(Proj(self.crs).get_factors(*self.center).meridian_convergence)

    def scale_denominator(self, map_width_in: float) -> float:
        return self.width_m / (map_width_in * 0.0254)

    @property
    def epsg(self) -> str:
        a = self.crs.to_authority()
        return f"{a[0]}:{a[1]}" if a else self.crs.name
