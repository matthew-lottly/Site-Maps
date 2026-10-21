"""Real-estate site-analysis map: a named office tower with its surroundings measured in distance rings (Seattle, Washington)."""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgba
import matplotlib
from shapely.geometry import Point, box

from ..core.cache import OUTPUTS
from ..core.frame import SiteFrame
from ..core import place
from ..core.labels import LabelPlacer
from ..core.layout import Legend, Sheet, halo
from ..core.provenance import ProvenanceLog
from ..sources import dem as demsrc, hrsl, overture as ov, sentinel2 as s2
from ..sources.site_analysis import ARTERIALS, M_PER_MILE, RINGS_MI, footprint_stats, max_speed_mph, nearest_by_class, places_by_ring, population_in_ring, ring_geoms
from .base import clip_to_frame, label_lines, mask_outside, overture_layer, raster_extent

SITE_ID = "bffb77cb-5b55-4536-94a2-e52504282cea"            # Overture building feature
SITE_NAME = "Rainier Square"
SEARCH_BOX = (-122.345, 47.598, -122.320, 47.616)                  # used only to fetch the feature by id
GROUPS = {"services_and_business": "Services & business", "shopping": "Shopping", "health_care": "Health care", "food_and_drink": "Food & drink", "lifestyle_services": "Lifestyle services",
          "travel_and_transportation": "Travel & transport", "community_and_government": "Community & gov.", "sports_and_recreation": "Sports & recreation", "education": "Education",
          "lodging": "Lodging", "arts_and_entertainment": "Arts & entertainment", "cultural_and_historic": "Cultural & historic"}
MAP_GROUPS = {"shopping": "#B45309", "food_and_drink": "#BE123C", "lodging": "#6D28D9", "health_care": "#0F766E"}
LAND_USE = {"retail": ("Retail", "#FDBA74"), "commercial": ("Commercial", "#FCA5A5"), "residential": ("Residential", "#FDE68A"), "industrial": ("Industrial", "#C4B5FD"),
            "park": ("Park", "#86EFAC"), "recreation": ("Recreation / golf", "#BBF7D0"), "education": ("Education", "#93C5FD")}
ROAD_STYLE = {"motorway": (2.0, "#7F1D1D"), "trunk": (1.8, "#9A3412"), "primary": (1.5, "#B45309"), "secondary": (1.2, "#6B7280"), "tertiary": (0.8, "#6B7280"),
              "residential": (0.6, "#9CA3AF"), "unclassified": (0.6, "#9CA3AF"), "service": (0.35, "#CBD5E1")}


class RealEstateSiteMap:
    key = "real_estate_rainier_square_seattle"

    def __init__(self, outdir=OUTPUTS):
        self.outdir = outdir
        self.prov = ProvenanceLog()
        self.log: dict = {}

    def build(self) -> Sheet:
        prov = self.prov
        b = ov.read("buildings", SEARCH_BOX, ["id", "names", "subtype", "class", "height", "num_floors"], tag="re_site")
        s = b[b["id"] == SITE_ID]
        assert len(s) == 1 and s.iloc[0]["name"] == SITE_NAME, "site building not found in the pinned Overture release"
        c = s.geometry.iloc[0].centroid
        frame = SiteFrame.around(c.x, c.y, 5300, Sheet.map_aspect())
        site = s.to_crs(frame.crs).iloc[0]
        sg = site.geometry
        cen = sg.centroid
        cxy = (cen.x, cen.y)
        rings = ring_geoms(cen)
        self.log["site"] = {"name": SITE_NAME, "id": SITE_ID, "height_m": float(site["height"]), "floors": None if pd.isna(site["num_floors"]) else int(site["num_floors"]),
                            "footprint_m2": round(sg.area, 1)}
        prov.add(f"Subject building: {SITE_NAME}", ov.SOURCE, 1, f"Overture building {SITE_ID}; height and floor count as published in the feature")
        div = ov.read("divisions", (c.x - 0.001, c.y - 0.001, c.x + 0.001, c.y + 0.001), ["id", "names", "subtype", "country", "region"], tag="re_div", crs=frame.crs)
        div = div[div.contains(cen)]
        pick = lambda sub: (div[div["subtype"] == sub].assign(a=lambda d_: d_.geometry.area).sort_values("a").iloc[0]["name"] if (div["subtype"] == sub).any() else None)
        self.log["divisions"] = {k: pick(k) for k in ("neighborhood", "locality", "county", "region")}
        ctx = place.context(c.x, c.y)
        self.log["admin1"] = ctx
        where = ", ".join(v for v in (self.log["divisions"]["neighborhood"], self.log["divisions"]["county"], ctx["region"]) if v)
        prov.add("Place names (neighbourhood, county, region)", ov.SOURCE, int(len(div)), "names of the Overture division polygons that contain the subject building centroid")
        sheet = Sheet(f"{SITE_NAME}", f"Real-estate site analysis · {where}", f"{c.y:.4f}° N  {abs(c.x):.4f}° W  ·  rings 0.25 / 0.5 / 1 mile", frame, prov)
        ax = sheet.ax
        x0, y0, x1, y1 = frame.extent

        # imagery base (clearest recent scene), washed back so the vector layers read
        scs = [q for q in s2.scenes(c.x, c.y, dt.date(2026, 3, 1), dt.date(2026, 10, 15)) if q.cloud < 40]
        scored = []
        for q in scs:
            cov, cl = s2.window_quality(q, frame)
            if cov > 0.99:
                scored.append((round(cl, 2), -q.date.toordinal(), q))
        scored.sort(key=lambda t: (t[0], t[1]))
        cl, _, base = scored[0]
        rgb, grid = s2.true_color(base, frame, 10.0)
        ax.imshow(rgb, extent=raster_extent(grid), origin="upper", zorder=1, interpolation="lanczos")
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", alpha=0.45, ec="none", zorder=1.5))
        prov.add(f"Imagery: Sentinel-2 true colour, {base.date}", s2.SOURCE, None, f"scene {base.prefix.split('/')[-1]}, 10 m TCI, window cloud/shadow {cl:.1%}; washed 45 % toward white under the vector layers")
        self.log["imagery_scene"] = base.prefix.split("/")[-1]

        # land use
        lu = overture_layer("land_use", frame, ["id", "names", "subtype", "class"], prov, "Land use (OpenStreetMap-derived)", tag="re")
        lu = clip_to_frame(lu, frame)
        lu = lu[lu["class"].isin(LAND_USE) | lu["subtype"].isin(["recreation", "park", "education"])]
        lu["lu"] = np.where(lu["class"].isin(LAND_USE), lu["class"], np.where(lu["subtype"] == "recreation", "recreation", lu["subtype"]))
        for k, (nm, col) in LAND_USE.items():
            sub = lu[lu["lu"] == k]
            if len(sub):
                sub.plot(ax=ax, fc=col, ec="none", alpha=0.30, zorder=2)

        # buildings coloured by height
        bld = clip_to_frame(overture_layer("buildings", frame, ["id", "names", "subtype", "class", "height", "num_floors"], prov, "Building footprints", tag="re"), frame)
        norm = Normalize(0, 150)
        cmap = matplotlib.colormaps["YlOrBr"]
        h = bld["height"]
        known = bld[h.notna()]
        known.plot(ax=ax, color=[cmap(norm(min(v, 150))) for v in known["height"]], ec="#374151", lw=0.0, zorder=5)
        bld[h.isna()].plot(ax=ax, fc="#E5E7EB", ec="#374151", lw=0.0, zorder=5)
        gpd.GeoSeries([sg], crs=frame.crs).plot(ax=ax, fc="#DC2626", ec="white", lw=1.6, zorder=9)
        gpd.GeoSeries([sg], crs=frame.crs).boundary.plot(ax=ax, color="#7F1D1D", lw=0.8, zorder=9.1)

        # streets
        seg = overture_layer("segments", frame, ["id", "names", "class", "subclass", "speed_limits"], prov, "Streets (by road class, speed limits where tagged)", tag="re")
        seg_c = clip_to_frame(seg, frame)
        for cls in ["tertiary", "secondary", "primary", "trunk", "motorway"]:
            sub = seg_c[seg_c["class"] == cls]
            if len(sub):
                lw, col = ROAD_STYLE[cls]
                sub.plot(ax=ax, color="white", lw=lw + 1.2, zorder=3 + 0.01 * len(cls), capstyle="round")
                sub.plot(ax=ax, color=col, lw=lw, zorder=3.5 + 0.01 * len(cls), capstyle="round")

        # places
        pl = overture_layer("places", frame, ["id", "names", "taxonomy", "confidence", "operating_status"], prov, "Places (businesses and facilities)", tag="re")
        pl = pl[(pl["operating_status"] == "open") & (pl["confidence"] >= 0.7)]
        prov.records[-1].detail += "; kept open places with confidence ≥ 0.7"
        prov.records[-1].features = len(pl)
        # rings
        for r, g in rings.items():
            gpd.GeoSeries([g], crs=frame.crs).boundary.plot(ax=ax, color="#111827", lw=1.0, ls="--", zorder=14)
        sheet.graticule(); sheet.scale_bar(); sheet.north_arrow()
        lp = LabelPlacer(ax)
        lp.reserve_rect_axes(0.0, 0.0, 0.34, 0.17)
        lp.reserve_rect_axes(0.89, 0.84, 1.0, 1.0)
        ang_pt = np.radians(35)
        for r in rings:
            lp.place_centered(cen.x + r * M_PER_MILE * np.cos(ang_pt), cen.y + r * M_PER_MILE * np.sin(ang_pt), f"{r:g} mi", fontsize=7, fontweight="bold", color="#111827", halo_w=2.6)
        lp.reserve_point(cen.x, cen.y, 14)
        lp.place(cen.x, cen.y, f"{SITE_NAME} · {self.log['site']['height_m']:g} m", fontsize=7.2, fontweight="bold", color="#7F1D1D", halo_w=2.8,
                 offsets=[(14, -22, "left", "top"), (14, 22, "left", "bottom"), (-14, -22, "right", "top"), (-14, 22, "right", "bottom")])
        self._street_labels(lp, seg_c, frame)

        # ---- ring statistics
        r1 = rings[1.0]
        stats = {}
        pcx, pcy, ppl, tile = hrsl.cells(frame, frame.padded_bounds_ll(0.05))
        prov.add("Population density", hrsl.SOURCE, int(len(ppl)), f"tile {tile}; cells with centre inside each ring summed; {len(ppl):,} populated cells in the window")
        g_all = bld.copy()
        for r, g in rings.items():
            fs = footprint_stats(g_all, g)
            hh = g_all[g_all.intersects(g)]["height"].dropna()
            stats[r] = {"pop": population_in_ring(pcx, pcy, ppl, cxy, r * M_PER_MILE), "area_km2": g.area / 1e6, "buildings": fs["buildings"], "coverage": fs["coverage"],
                        "median_h": float(hh.median()) if len(hh) else float("nan"), "n_h": int(len(hh))}
            stats[r]["density"] = stats[r]["pop"] / stats[r]["area_km2"]
        counts = places_by_ring(pl, cen)
        for r in RINGS_MI:
            stats[r]["places"] = int(counts[f"≤ {r:g} mi"].sum())
        self.log["rings"] = {f"{r:g} mi": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for r, d in stats.items()}
        # terrain inside the 1 mile ring
        z, zgrid, zsrc = demsrc.elevation(frame, 30.0)
        prov.add("Terrain: 3DEP elevation", zsrc, None, "30 m grid resampled to the map grid; minimum and maximum inside each ring")
        zf = np.ma.filled(z, np.nan)
        for r, g in rings.items():
            zv = zf[mask_outside(None, g, zgrid)]
            zv = zv[np.isfinite(zv)]
            stats[r]["zmin"], stats[r]["zmax"] = float(zv.min()), float(zv.max())
        self.log["elevation_1mi_m"] = {"min": round(stats[1.0]["zmin"], 1), "max": round(stats[1.0]["zmax"], 1)}
        near = nearest_by_class(cen, seg, ARTERIALS)
        near["dist_to_footprint_m"] = [float(seg[seg["class"] == c].geometry.distance(sg).min()) if (seg["class"] == c).any() else np.nan for c in near["class"]]
        self.log["access"] = {r["class"]: {"name": r["name"], "m": None if pd.isna(r["dist_to_footprint_m"]) else round(r["dist_to_footprint_m"]), "max_mph": r["max_mph"]} for _, r in near.iterrows()}
        sp = seg_c.assign(mph=seg_c["speed_limits"].map(max_speed_mph))
        self.log["speed_tagged_segments_in_window"] = int(sp["mph"].notna().sum())
        self.log["segments_in_window"] = int(len(seg_c))

        # ---- right column
        sheet.locator(width_deg=7.0, height_in=1.0)
        lg = Legend("Legend", cols=4)
        lg.header("Buildings (colour = Overture height)")
        lg.ramp("Height 0–150 m (taller capped)", "YlOrBr", 0, 150, "m")
        lg.patch("No height", "#E5E7EB")
        lg.patch("Subject", "#DC2626")
        lg.header("Streets (by class)")
        for c_, lbl in (("motorway", "Motorway"), ("primary", "Primary"), ("secondary", "Secondary"), ("tertiary", "Tertiary")):
            lg.line(lbl, ROAD_STYLE[c_][1], ROAD_STYLE[c_][0])
        lg.header("Land use (OSM-derived, 30 % opacity)")
        present = [k for k in LAND_USE if (lu["lu"] == k).any()]
        for k in present:
            lg.patch(LAND_USE[k][0], LAND_USE[k][1])
        sheet.legend(lg)
        self._tables(sheet, stats, counts, near, sg)
        hk = bld["height"].notna().mean()
        self.log["height_recorded_share"] = round(float(hk), 3)
        sheet.notes += [f"Heights are recorded for {hk:.0%} of footprints in the window; the rest are grey.", "Population is a modelled estimate (HRSL), counted by cell centre within each ring.", "Land use, streets, places and buildings are OpenStreetMap-derived (Overture) and may lag the ground."]
        self.log['remaining_in_before_sources'] = round(sheet.remaining_in(), 2)
        sheet.sources_box("Data sources")
        out = sheet.export(self.key, self.outdir)
        self.log["layout_issues"] = out["issues"]
        return sheet

    @staticmethod
    def _street_labels(lp, seg, frame, max_labels=16):
        """Name each arterial once along its longest visible piece, rotated to follow it; collisions are skipped, not forced."""
        x0, y0, x1, y1 = frame.extent
        W = x1 - x0
        view = box(x0, y0, x1, y1).buffer(-W * 0.02)
        n = 0
        for cls in ("motorway", "trunk", "primary", "secondary"):
            sub = seg[(seg["class"] == cls) & seg["name"].notna()]
            for name, grp in sub.groupby("name"):
                best = None
                for geom in grp.geometry:
                    part = geom.intersection(view)
                    parts = list(part.geoms) if part.geom_type.startswith("Multi") else [part]
                    for q in parts:
                        if q.geom_type == "LineString" and (best is None or q.length > best.length):
                            best = q
                if best is None or best.length < W * 0.10:
                    continue
                m = best.interpolate(0.5, normalized=True)
                a_, c_ = best.interpolate(0.46, normalized=True), best.interpolate(0.54, normalized=True)
                ang = np.degrees(np.arctan2(c_.y - a_.y, c_.x - a_.x))
                ang = ang + 180 if ang > 90 or ang < -90 else ang
                if lp.place_centered(m.x, m.y, name, rotation=ang, fontsize=6.2, color="#1F2933"):
                    n += 1
                if n >= max_labels:
                    return

    def _tables(self, sheet: Sheet, stats, counts, near, sg):
        t = sheet.t
        p = sheet.panel(1.5, "Surroundings by distance ring (cumulative)")
        H = 1.62
        rows = [("Population (HRSL model)", lambda s: f"{s['pop']:,.0f}"), ("People per km²", lambda s: f"{s['density']:,.0f}"), ("Buildings", lambda s: f"{s['buildings']:,}"),
                ("Footprint coverage", lambda s: f"{s['coverage']:.1%}"), ("Median height (m)", lambda s: f"{s['median_h']:.0f}" if np.isfinite(s["median_h"]) else "n/a"),
                ("Open places", lambda s: f"{s['places']:,}"), ("Elevation range, 3DEP (m)", lambda s: f"{int(round(s['zmin']))} to {int(round(s['zmax']))}")]
        xs = [0.50, 0.68, 0.86]
        for x, r in zip(xs, RINGS_MI):
            p.text(x, 1 - 0.46 / H, f"≤ {r:g} mi", fontsize=6.6, fontweight="bold", color=t.ink, ha="center", va="center", transform=p.transAxes)
        for i, (lab, f) in enumerate(rows):
            y = 1 - (0.70 + i * 0.125) / H
            p.text(0.03, y, lab, fontsize=6.4, color=t.muted, va="center", transform=p.transAxes)
            for x, r in zip(xs, RINGS_MI):
                p.text(x, y, f(stats[r]), fontsize=6.8, color=t.ink, ha="center", va="center", fontweight="bold", transform=p.transAxes)
            p.plot([0.03, 0.97], [y - 0.07 / H] * 2, color=t.rule, lw=0.4, transform=p.transAxes)
        # places by group chart
        p = sheet.panel(1.05, "Open places by category and distance band")
        ax = sheet.chart_axes(p, (0.34, 0.22, 0.60, 0.50))
        cnt = counts.copy()
        cnt["b1"], cnt["b2"], cnt["b3"] = cnt["≤ 0.25 mi"], cnt["≤ 0.5 mi"] - cnt["≤ 0.25 mi"], cnt["≤ 1 mi"] - cnt["≤ 0.5 mi"]
        cnt = cnt.sort_values("≤ 1 mi").tail(5)
        lab = [GROUPS.get(i, i) for i in cnt.index]
        left = np.zeros(len(cnt))
        for k, col, nm in (("b1", "#1E3A8A", "0–0.25 mi"), ("b2", "#3B82F6", "0.25–0.5 mi"), ("b3", "#BFDBFE", "0.5–1 mi")):
            ax.barh(lab, cnt[k], left=left, color=col, height=0.7, label=nm, edgecolor="white", linewidth=0.3)
            left += cnt[k].to_numpy()
        for i, v in enumerate(cnt["≤ 1 mi"]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=5.8, color=t.ink)
        ax.set_xlim(0, cnt["≤ 1 mi"].max() * 1.15)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.legend(fontsize=5.4, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0))
        # access table
        p = sheet.panel(1.0, "Nearest streets by class (to the building footprint)")
        H = 1.16
        hdr = [(0.03, "Class", "left"), (0.24, "Street (as named in data)", "left"), (0.74, "Distance", "right"), (0.97, "Posted max", "right")]
        for x, h_, ha in hdr:
            p.text(x, 1 - 0.46 / H, h_, fontsize=5.8, fontweight="bold", color=t.muted, ha=ha, va="center", transform=p.transAxes)
        for i, (_, r) in enumerate(near.iterrows()):
            y = 1 - (0.64 + i * 0.105) / H
            d = r["dist_to_footprint_m"]
            p.text(0.03, y, r["class"].title(), fontsize=6.2, color=t.ink, va="center", transform=p.transAxes)
            p.text(0.22, y, str(r["name"])[:32] if isinstance(r["name"], str) else "(unnamed in data)", fontsize=6.2, color=t.ink, va="center", transform=p.transAxes)
            p.text(0.76, y, "none in window" if pd.isna(d) else f"{d:,.0f} m", fontsize=6.2, color=t.ink, ha="right", va="center", transform=p.transAxes)
            mph = r["max_mph"]
            p.text(0.97, y, "not tagged" if mph is None or pd.isna(mph) else f"{mph:.0f} mph", fontsize=6.2, color=t.muted, ha="right", va="center", transform=p.transAxes)
