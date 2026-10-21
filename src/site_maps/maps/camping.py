"""Camping site map: campgrounds of one valley on imagery + relief, with trails, water, facilities, field observations, terrain metrics and climate.

Everything printed is computed from, or copied verbatim from, a registered dataset. The site is resolved from Overture campground places (see sources/camp.py).
"""
from __future__ import annotations

import datetime as dt
import textwrap

import geopandas as gpd
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union

from ..core.cache import OUTPUTS, cache_path, download
from ..core.frame import SiteFrame
from ..core import place
from ..core.labels import LabelPlacer
from ..core.layout import Legend, Sheet, halo
from ..core.provenance import ProvenanceLog, Source
from ..sources import camp, dem as demsrc, ghcn, inat, overture as ov, sentinel2 as s2, worldcover as wc
from ..sources.inat import GROUP_COLORS
from ..sources.rasters import Grid, hillshade, read_to_grid, slope_degrees
from .base import clip_to_frame, label_lines, overture_layer, raster_extent

SEARCH_BOX = (-119.9, 37.5, -119.2, 38.1)           # used only to fetch candidate campground places; the site frame is computed from the data
SITE_PLACE = "Yosemite Valley"                  # text used to find the Overture places that anchor the site
INAT_FILE = "yosemite_valley"                       # cached iNaturalist extract covering the same area
TRAIL_CLASSES = ["path", "footway", "track", "bridleway", "cycleway", "pedestrian", "steps"]
ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service"]
FAC = {"toilets": ("Toilets", "#7C3AED", "s"), "drinking_water": ("Drinking water", "#0284C7", "o"), "parking": ("Parking", "#475569", "P"), "bus_stop": ("Bus stop", "#B45309", "D")}


WIDE = [(10, 9, "left", "bottom"), (10, -9, "left", "top"), (-10, 9, "right", "bottom"), (-10, -9, "right", "top"), (0, 15, "center", "bottom"), (0, -15, "center", "top"),
        (22, 0, "left", "center"), (-22, 0, "right", "center"), (24, 22, "left", "bottom"), (-24, 22, "right", "bottom"), (24, -22, "left", "top"), (-24, -22, "right", "top"),
        (0, 30, "center", "bottom"), (0, -30, "center", "top")]


def sheet_clear(frame, x, y) -> bool:
    x0, y0, x1, y1 = frame.extent
    W, H = x1 - x0, y1 - y0
    return (x > x1 - 0.14 * W and y > y1 - 0.20 * H) or (x < x0 + 0.27 * W and y < y0 + 0.19 * H)


class CampingSiteMap:
    key = "camping_yosemite_valley"

    def __init__(self, outdir=OUTPUTS):
        self.outdir = outdir
        self.prov = ProvenanceLog()
        self.log: dict = {}

    # ---------------- site resolution ----------------
    def resolve(self):
        raw = ov.read("places", SEARCH_BOX, ["id", "names", "taxonomy", "confidence", "operating_status"], tag="yos")
        crs = SiteFrame.around(-119.6, 37.74, 1000, 1.2).crs
        pts = raw.to_crs(crs)
        kept, dropped = camp.filter_and_dedupe(pts)
        named = pts[pts["name"].fillna("").str.contains(SITE_PLACE, case=False) & (pts["confidence"] >= 0.7)]
        assert len(named) >= 5, f"too few Overture places mention {SITE_PLACE!r}"
        anchor_pt = Point(float(np.median(named.geometry.x)), float(np.median(named.geometry.y)))     # median is robust to the few mis-located entries
        idx = camp.cluster_near(kept, anchor_pt)
        cluster = kept.iloc[idx]
        centre = Point(unary_union(cluster.geometry).centroid)
        cluster, out = camp.drop_outside_cluster(kept, idx, centre)
        dropped += out
        assert len(cluster) >= 3, "expected several campgrounds in the valley cluster"
        x0, y0, x1, y1 = cluster.total_bounds
        span = max(x1 - x0, y1 - y0)
        width = float(np.ceil(max(7000.0, 2.3 * span) / 500) * 500)
        lon, lat = SiteFrame.around(-119.6, 37.74, 1000, 1.2).to_ll.transform((x0 + x1) / 2, (y0 + y1) / 2)
        frame = SiteFrame.around(lon, lat, width, Sheet.map_aspect())
        self.site_name = SITE_PLACE
        self.log["site_name_source"] = f"anchor = median position of {len(named)} Overture places (confidence ≥ 0.7) whose name contains '{SITE_PLACE}'; the campground cluster nearest to it is mapped"
        self.log["candidates_total"] = int((raw["category"] == "campground").sum())
        self.log["campgrounds_kept"] = [r["name"] for _, r in cluster.iterrows()]
        self.log["campgrounds_dropped"] = [f"{d['name']}: {d['reason']}" for d in dropped]
        self.log["frame_width_m"] = width
        self.log["cluster_span_m"] = round(float(span))
        return frame, cluster.to_crs(frame.crs)

    # ---------------- imagery ----------------
    def imagery(self, frame):
        lon, lat = frame.center
        best = None
        grid = Grid.for_frame(frame, 60.0)
        for s in s2.scenes(lon, lat, dt.date(2025, 6, 1), dt.date(2026, 10, 15)):
            if s.cloud > 40:
                continue
            cov, cl = s2.window_quality(s, frame)
            if cov < 0.99 or cl > 0.02:
                continue
            scl = np.ma.filled(read_to_grid(s.url("SCL"), grid, frame.crs, resampling=Resampling.nearest), 0)
            snow = float((scl == 11).mean())
            key = (round(snow, 2), -s.date.toordinal())
            if best is None or key < best[0]:
                best = (key, s, cl, snow)
        assert best is not None, "no usable Sentinel-2 scene found"
        _, scene, cl, snow = best
        rgb, igrid = s2.true_color(scene, frame, 10.0)
        self.log["imagery_scene"] = scene.prefix.split("/")[-1]
        self.log["imagery_window_cloud"] = round(cl, 4)
        self.log["imagery_window_snow"] = round(snow, 4)
        return scene, rgb, igrid

    # ---------------- build ----------------
    def build(self):
        prov = self.prov
        frame, camps = self.resolve()
        ctx = place.context(*frame.center)
        state, country = ctx["region"], ctx["country"]
        prov.add("Campgrounds (named places)", ov.SOURCE, len(camps),
                 f"Overture places, category 'campground'; kept confidence ≥ {camp.MIN_CONFIDENCE} and not closed; same-name entries within {camp.DEDUPE_M:.0f} m merged (highest confidence kept); "
                 f"largest single-linkage cluster (link {camp.CLUSTER_M / 1000:.0f} km) taken as the site; {len(self.log['campgrounds_dropped'])} candidates dropped")
        prov.add("State / country name in subtitle", Source("NE", "Natural Earth", "Admin 1 – States, Provinces 1:50m", "GitHub master at retrieval", "https://www.naturalearthdata.com/", "Public domain"), None, "admin-1 polygon containing the map centre")
        lon0, lat0 = frame.center
        sheet = Sheet(f"{self.site_name}: campgrounds", f"Camping site map · {state}, {country} · imagery with relief, contours, trails, water, facilities and field observations",
                      f"{lat0:.4f}° N  {abs(lon0):.4f}° W  ·  {len(camps)} campgrounds", frame, prov)
        ax = sheet.ax

        # --- terrain (same 10 m grid as the imagery)
        z, dgrid, dsrc = demsrc.elevation(frame, 10.0, pad_frac=0.0)
        zf = np.ma.filled(z, np.nan).astype(float)
        hs = hillshade(z, dgrid.res, 315, 40, 1.6)
        slope = slope_degrees(z, dgrid.res)
        prov.add("Terrain: hillshade, contours, slope", dsrc, None, "1 arc-second DEM resampled bilinearly to 10 m; Horn hillshade (az 315°, alt 40°, vertical exaggeration 1.6); slope from the same grid")
        zmin, zmax = float(np.nanmin(zf)), float(np.nanmax(zf))
        interval = camp.nice_interval(zmin, zmax, 16)
        self.log["elevation_range_m"] = [round(zmin), round(zmax)]
        self.log["contour_interval_m"] = interval

        scene, rgb, igrid = self.imagery(frame)
        assert rgb.shape[:2] == hs.shape, (rgb.shape, hs.shape)
        img = np.clip((rgb.astype(float) / 255.0) ** 0.8 * (0.62 + 0.55 * hs)[..., None], 0, 1)
        ax.imshow(img, extent=raster_extent(igrid), origin="upper", zorder=1, interpolation="lanczos")
        prov.add(f"Imagery: Sentinel-2 true colour, {scene.date}", s2.SOURCE, None,
                 f"scene {scene.prefix.split('/')[-1]}; TCI at 10 m multiplied by the hillshade; window cloud {self.log['imagery_window_cloud']:.1%}, snow {self.log['imagery_window_snow']:.1%}")
        x0, y0, x1, y1 = frame.extent
        X = np.linspace(x0 + dgrid.res / 2, x1 - dgrid.res / 2, zf.shape[1])
        Y = np.linspace(y1 - dgrid.res / 2, y0 + dgrid.res / 2, zf.shape[0])
        levels = np.arange(np.ceil(zmin / interval) * interval, zmax, interval)
        idx = [l for l in levels if np.isclose(l % (interval * 5), 0)]
        ax.contour(X, Y, zf, levels=[l for l in levels if l not in idx], colors="#FFFFFF", linewidths=0.25, alpha=0.55, zorder=2)
        cs = ax.contour(X, Y, zf, levels=idx, colors="#FFFFFF", linewidths=0.7, alpha=0.9, zorder=2.1)
        ax.clabel(cs, fmt="%d m", fontsize=5.4, colors="#FFFFFF", inline=True, inline_spacing=6)

        # --- water
        water = clip_to_frame(overture_layer("water", frame, ["id", "names", "subtype", "class"], prov, "Surface water", tag="camp"), frame)
        wl = water[water.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        wp = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(wp):
            wp.plot(ax=ax, fc="#38BDF8", ec="#0369A1", lw=0.6, alpha=0.6, zorder=3)
        rivers = wl[wl["subtype"] == "river"]
        streams = wl[wl["subtype"] == "stream"]
        if len(streams):
            streams.plot(ax=ax, color="#38BDF8", lw=0.7, zorder=3)
        if len(rivers):
            rivers.plot(ax=ax, color="#0EA5E9", lw=1.8, zorder=3.1)
        
        # --- roads and trails
        seg = clip_to_frame(overture_layer("segments", frame, ["id", "names", "class", "subclass"], prov, "Roads and trails", tag="camp"), frame)
        roads = seg[seg["class"].isin(ROAD_CLASSES)]
        trails = seg[seg["class"].isin(TRAIL_CLASSES)]
        major = roads[~roads["class"].isin(["service", "living_street"])]
        minor = roads[roads["class"].isin(["service", "living_street"])]
        if len(minor):
            minor.plot(ax=ax, color="#F8FAFC", lw=0.5, zorder=4)
        if len(major):
            major.plot(ax=ax, color="#1F2933", lw=2.6, zorder=4.1)
            major.plot(ax=ax, color="#F8FAFC", lw=1.6, zorder=4.2)
        if len(trails):
            trails.plot(ax=ax, color="#FDE68A", lw=0.55, ls="--", zorder=4.3)
        
        # --- camp footprints
        lu = clip_to_frame(overture_layer("land_use", frame, ["id", "names", "subtype", "class"], prov, "Camp site polygons", tag="camp"), frame)
        cpoly = lu[lu["class"] == "camp_site"]
        match = camp.match_polygons(camps, cpoly)
        if len(cpoly):
            cpoly.plot(ax=ax, fc="#F97316", ec="#9A3412", lw=0.9, alpha=0.32, zorder=5)
        self.log["camp_site_polygons_in_view"] = int(len(cpoly))
        self.log["campground_polygon_match_m"] = {camps.loc[i, "name"]: (round(v[1]) if v else None) for i, v in match.items()}

        # --- facilities
        infra = clip_to_frame(overture_layer("infrastructure", frame, ["id", "names", "subtype", "class"], prov, "Facilities (infrastructure points)", tag="camp"), frame)
        fac = {k: infra[infra["class"] == k] for k in FAC}
        for k, (lab, col, mk) in FAC.items():
            g = fac[k]
            if len(g):
                ax.scatter(g.geometry.representative_point().x, g.geometry.representative_point().y, s=14, marker=mk, c=col, edgecolors="white", linewidths=0.5, zorder=8)
        self.log["facility_counts_in_view"] = {FAC[k][0]: int(len(g)) for k, g in fac.items()}

        # --- named places: trailheads and visitor centres
        places = clip_to_frame(overture_layer("places", frame, ["id", "names", "taxonomy", "confidence", "operating_status"], prov, "Named places", tag="camp2"), frame)
        thead = places[(places["category"] == "hiking_trail") & places["name"].fillna("").str.contains("Trailhead|Trail Head", case=False) & (places["confidence"] >= 0.8)]
        visitor = places[places["name"].fillna("").str.contains("Visitor Center", case=False) & (places["confidence"] >= 0.7)]
        for g, mk, col in ((thead, "^", "#16A34A"), (visitor, "*", "#DC2626")):
            for _, r in g.iterrows():
                ax.plot(r.geometry.x, r.geometry.y, marker=mk, ms=8, mfc=col, mec="white", mew=0.7, zorder=9)
        ex0, ey0, ex1, ey1 = frame.extent
        inside = lambda g: g[(g.geometry.x > ex0 + 0.04 * (ex1 - ex0)) & (g.geometry.x < ex1 - 0.04 * (ex1 - ex0)) & (g.geometry.y > ey0 + 0.04 * (ey1 - ey0)) & (g.geometry.y < ey1 - 0.04 * (ey1 - ey0))]
        thead, visitor = inside(thead), inside(visitor)
        self.log["trailheads_in_view"] = [r["name"] for _, r in thead.iterrows()]
        self.log["visitor_centres_in_view"] = [r["name"] for _, r in visitor.iterrows()]

        # --- field observations
        obs = self.observations(frame)
        if len(obs):
            for g, gd in obs.groupby("group"):
                ax.scatter(gd["x"], gd["y"], s=2.2, c=GROUP_COLORS.get(g, "#64748B"), alpha=0.55, linewidths=0, zorder=7)
            prov.add("Field observations (research grade)", inat.SOURCE, len(obs), "research-grade, positional accuracy ≤ 100 m, inside the map window")
        self.log["observations_in_window"] = int(len(obs))

        # --- campground markers, numbered west to east
        order = camps.assign(_x=camps.geometry.x).sort_values("_x")
        nums = {i: n + 1 for n, i in enumerate(order.index)}
        for i, r in camps.iterrows():
            ax.plot(r.geometry.x, r.geometry.y, marker="^", ms=12, mfc="#F97316", mec="white", mew=1.2, zorder=20)
            ax.annotate(str(nums[i]), (r.geometry.x, r.geometry.y), xytext=(0, -1.5), textcoords="offset points", fontsize=5.6, fontweight="bold", color="white", ha="center", va="center", zorder=21)

        sheet.graticule()
        sheet.scale_bar()
        sheet.north_arrow()

        # labels, placed with collision avoidance in priority order: campgrounds, trailheads / visitor centres, river names, road names
        lp = LabelPlacer(ax)
        lp.reserve_rect_axes(0.0, 0.0, 0.31, 0.2)
        lp.reserve_rect_axes(0.88, 0.84, 1.0, 1.0)
        for _, r in camps.iterrows():
            lp.reserve_point(r.geometry.x, r.geometry.y, 7)
        for g in (thead, visitor):
            for _, r in g.iterrows():
                lp.reserve_point(r.geometry.x, r.geometry.y, 5)
        for _, r in camps.iterrows():
            lp.place(r.geometry.x, r.geometry.y, r["name"], fontsize=7, fontweight="bold", color="#111827", halo_w=2.8, offsets=WIDE)
        for g in (visitor, thead):
            for _, r in g.iterrows():
                lp.place(r.geometry.x, r.geometry.y, r["name"], fontsize=5.8, color="#111827")
        label_lines(ax, rivers, "name", frame, 0.08, 6.4, color="#075985", max_labels=3, placer=lp)
        label_lines(ax, major, "name", frame, 0.10, 6.0, max_labels=8, placer=lp)
        sheet.notes += [f"Campground positions are Overture place points and can be offset from the real site; {len(self.log['campgrounds_dropped'])} candidate(s) were dropped by the documented filter.",
                        "Elevation is a 1 arc-second DEM, so slope and height above the river are indicative at campground scale, not survey values.",
                        "Observation points show where people record species, not abundance."]

        metrics = self.metrics(camps, nums, zf, dgrid, slope, rivers, fac, thead, cpoly, match, obs)
        self.right_column(sheet, frame, camps, nums, metrics, obs, zf, dgrid, rivers, fac, thead, visitor, cpoly, interval)
        sheet.sources_box("Data sources")
        out = sheet.export(self.key, self.outdir)
        self.log["layout_issues"] = out["issues"]
        return sheet

    # ---------------- helpers ----------------
    def observations(self, frame):
        f = cache_path(f"inat/{INAT_FILE}.csv")
        if not f.exists():
            return pd.DataFrame()
        o = pd.read_csv(f, dtype={"observation_uuid": str})
        o = o[(o["quality_grade"] == "research") & (o["positional_accuracy"].fillna(1e9) <= 100)]
        w, s, e, n = frame.padded_bounds_ll(0.0)
        o = o[o["longitude"].between(w, e) & o["latitude"].between(s, n)].copy()
        o = inat.add_groups(o)
        xs, ys = frame.to_proj.transform(o["longitude"].to_numpy(), o["latitude"].to_numpy())
        o["x"], o["y"] = xs, ys
        x0, y0, x1, y1 = frame.extent
        return o[(o["x"] >= x0) & (o["x"] <= x1) & (o["y"] >= y0) & (o["y"] <= y1)]

    def main_river(self, rivers):
        """Longest merged river line in view (name copied from the data)."""
        if not len(rivers):
            return None, None
        named = rivers[rivers["name"].notna()]
        name = named.assign(L=named.geometry.length).groupby("name")["L"].sum().idxmax() if len(named) else None
        sel = rivers[rivers["name"] == name] if name else rivers
        merged = linemerge(list(sel.geometry))
        parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
        line = max(parts, key=lambda g: g.length)
        xs = np.array(line.coords)[:, 0]
        if xs[0] > xs[-1]:
            line = type(line)(list(line.coords)[::-1])
        return name, line

    def metrics(self, camps, nums, zf, dgrid, slope, rivers, fac, thead, cpoly, match, obs):
        """Per-campground numbers computed from the data on the sheet."""
        rname, rline = self.main_river(rivers)
        ext = dgrid.extent
        out = []
        for i, r in camps.iterrows():
            geom = cpoly.loc[match[i][0], "geometry"] if match.get(i) else r.geometry
            e_here = camp.sample(zf, ext, r.geometry.x, r.geometry.y)
            st = camp.slope_stats(slope, ext, dgrid.res, geom, 100.0)
            row = {"n": nums[i], "name": r["name"], "elev_m": e_here, "slope_mean": st["mean"], "slope_p90": st["p90"]}
            if rline is not None:
                npnt = rline.interpolate(rline.project(r.geometry))
                row["river_m"] = r.geometry.distance(rline)
                row["above_river_m"] = e_here - camp.sample(zf, ext, npnt.x, npnt.y)
            for k in ("toilets", "drinking_water"):
                row[k] = camp.nearest_distance(r.geometry, fac[k].geometry)[0]
            row["trailhead"] = camp.nearest_distance(r.geometry, thead.geometry)[0]
            if len(obs):
                row["obs_500m"] = int((np.hypot(obs["x"] - r.geometry.x, obs["y"] - r.geometry.y) <= 500).sum())
            out.append(row)
        self.log["river_used"] = rname
        self.log["campground_metrics"] = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()} for row in out]
        self.prov.add("Per-campground metrics (elevation, slope, distances)", demsrc.TNM, None,
                      "DEM sampled bilinearly at the Overture point; slope = mean / 90th percentile of 10 m-grid slope within 100 m of the camp polygon (or point if none within 300 m); distances are "
                      "straight-line metres to the nearest Overture feature; height above river = DEM at the campground minus DEM at the nearest point of the longest named river line in view")
        return pd.DataFrame(out)

    # ---------------- right column ----------------
    def right_column(self, sheet, frame, camps, nums, m, obs, zf, dgrid, rivers, fac, thead, visitor, cpoly, interval):
        t = sheet.t
        WCC = wc.classes()
        sheet.locator(width_deg=9.0, height_in=0.9)
        lg = Legend("Legend", cols=3)
        lg.header("Map symbols")
        lg.marker("Campground", "#F97316", "^", 7)
        lg.patch("Camp site polygon", "#F97316")
        lg.line("River / stream", "#0EA5E9", 1.6)
        lg.line("Road", "#1F2933", 1.8)
        lg.line("Trail / path / track", "#FDE68A", 1.0, "--")
        lg.line(f"Contour, {interval:g} m", "#CBD5E1", 0.8)
        lg.header("Facilities in view (count)")
        for k, (lab, col, mk) in FAC.items():
            if len(fac[k]):
                lg.marker(f"{lab} ({len(fac[k])})", col, mk, 5)
        if len(thead):
            lg.marker(f"Trailhead ({len(thead)})", "#16A34A", "^", 6)
        if len(visitor):
            lg.marker(f"Visitor centre ({len(visitor)})", "#DC2626", "*", 7)
        if len(obs):
            lg.header("Research-grade observations (count)")
            cnt = obs.groupby("group").size().sort_values(ascending=False)
            for g, n in cnt.head(5).items():
                lg.marker(f"{g} ({n:,})", GROUP_COLORS.get(g, "#64748B"), "o", 4.5)
            if len(cnt) > 5:
                lg.marker(f"{len(cnt) - 5} other groups ({int(cnt.iloc[5:].sum()):,})", "#64748B", "o", 4.5)
        sheet.legend(lg)

        # ---- table of terrain and facility distances
        rows = len(m)
        h = 0.26 + 0.12 + 0.42 + rows * 0.2 + 0.08
        ax = sheet.panel(h, "Terrain and access at each campground")
        cols = [("#", 0.03, "l"), ("Campground", 0.075, "l"), ("Elev.", 0.56, "r"), ("Above\nriver", 0.655, "r"), ("Slope\nmean", 0.745, "r"), ("Toilet", 0.83, "r"), ("Water", 0.905, "r"), ("Trailhead", 0.985, "r")]
        units = {"Elev.": "m", "Above\nriver": "m", "Slope\nmean": "°", "Toilet": "m", "Water": "m", "Trailhead": "m"}
        top = 1 - (0.26 + 0.12) / h
        for label, x, al in cols:
            txt = label + (f"\n({units[label]})" if label in units and "\n" not in label else (f"\n({units[label]})" if label in units else ""))
            ax.text(x, top, txt, fontsize=5.4, fontweight="bold", color=t.muted, ha="right" if al == "r" else "left", va="top", transform=ax.transAxes, linespacing=1.15)
        ax.plot([0.02, 0.99], [top - 0.36 / h] * 2, color=t.rule, lw=0.6, transform=ax.transAxes)
        for k, r in m.sort_values("n").reset_index(drop=True).iterrows():
            y = top - (0.42 + 0.12 + k * 0.2) / h
            vals = [str(int(r["n"])), r["name"][:26], f"{r['elev_m']:.0f}", f"{r.get('above_river_m', np.nan):+.0f}", f"{r['slope_mean']:.1f}", f"{r['toilets']:.0f}", f"{r['drinking_water']:.0f}", f"{r['trailhead']:.0f}"]
            for (label, x, al), v in zip(cols, vals):
                ax.text(x, y, v, fontsize=6.2, color=t.ink, ha="right" if al == "r" else "left", va="center", transform=ax.transAxes)

        # ---- elevation profile along the main river
        rname, rline = self.main_river(rivers)
        panel = sheet.panel(0.98, f"Elevation profile along {rname}" if rname else "Elevation profile")
        if rline is not None:
            L = rline.length
            step = 15.0
            d = np.arange(0, L, step)
            pts = [rline.interpolate(s) for s in d]
            e = np.array([camp.sample(zf, dgrid.extent, p.x, p.y) for p in pts])
            a = sheet.chart_axes(panel, (0.12, 0.30, 0.84, 0.42))
            a.fill_between(d / 1000, e, e.min() - 5, color="#BAE6FD", alpha=0.8, lw=0)
            a.plot(d / 1000, e, color="#0369A1", lw=1.2)
            for _, r in m.iterrows():
                cg = camps.loc[camps["name"] == r["name"]].iloc[0].geometry
                if cg.distance(rline) <= 1500:
                    s_ = rline.project(cg)
                    a.plot([s_ / 1000], [r["elev_m"]], marker="^", ms=6, mfc="#F97316", mec="white", mew=0.6, ls="none", zorder=5)
                    a.text(s_ / 1000, r["elev_m"] + (np.nanmax(e) - np.nanmin(e)) * 0.14, str(int(r["n"])), fontsize=5.8, ha="center", va="bottom", color="#9A3412", fontweight="bold")
            a.set_xlabel("distance along river line, west to east (km)", fontsize=5.6, color=t.muted, labelpad=1)
            a.set_ylabel("m", fontsize=5.6, color=t.muted, labelpad=1)
            a.set_ylim(np.nanmin(e) - 5, np.nanmax(e) + (np.nanmax(e) - np.nanmin(e)) * 0.45)
            self.log["river_profile_m"] = {"length": round(L), "min": round(float(np.nanmin(e))), "max": round(float(np.nanmax(e)))}

        # ---- climate + land cover
        panel = sheet.panel(1.3, "Climate normals and land cover")
        a1 = sheet.chart_axes(panel, (0.10, 0.31, 0.30, 0.38))
        st, dd, km = ghcn.best_station(*frame.center, max_km=60, min_years=20)
        if st is not None:
            n = ghcn.monthly_normals(dd)
            a2 = a1.twinx()
            a1.bar(n.index, n["prcp"], color="#93C5FD", width=0.7)
            a2.plot(n.index, n["tmax"], color="#DC2626", lw=1.2)
            a2.plot(n.index, n["tmin"], color="#2563EB", lw=1.2)
            a1.set_xticks(range(1, 13))
            a1.set_xticklabels(list("JFMAMJJASOND"), fontsize=5.6)
            a1.set_ylabel("precip (mm)", fontsize=5.6, color=t.muted)
            a2.set_ylabel("°C", fontsize=5.6, color=t.muted)
            a2.tick_params(labelsize=5.6, length=2, colors=t.muted)
            a2.spines["top"].set_visible(False)
            a1.set_title(f"{st['name']}: {km:.0f} km, {st['elev_m']:.0f} m", fontsize=5.8, color=t.ink, loc="left", pad=3)
            mean_e = float(m["elev_m"].mean())
            years = sorted(set(dd.index.year))
            diff = float(m["elev_m"].mean()) - float(st["elev_m"])
            panel.text(0.03, 0.045, f"Campgrounds average {abs(diff):.0f} m {'below' if diff < 0 else 'above'} the station (DEM)", fontsize=5.4, color=t.muted, va="center", transform=panel.transAxes)
            self.log["climate_station"] = {"id": st["id"], "name": st["name"], "km": round(km, 1), "elev_m": float(st["elev_m"]), "years": len(years), "campground_mean_elev_m": round(mean_e),
                                           "elevation_difference_m": round(mean_e - float(st["elev_m"]))}
            sheet.prov.add(f"Climate normals: GHCN {st['id']}", ghcn.SOURCE, None,
                           f"{st['name']}, {km:.0f} km from the map centre, station elevation {st['elev_m']:.0f} m vs campground mean {mean_e:.0f} m (DEM); {len(years)} complete years in 1991–2020; "
                           "mean daily max/min temperature and mean monthly precipitation computed from daily values")
        lc, lgrid = wc.land_cover(frame, 20.0)
        arr = np.ma.filled(lc, 0).astype(int)
        valid = arr > 0
        a3 = sheet.chart_axes(panel, (0.66, 0.31, 0.29, 0.38))
        share = {c: (arr == c).sum() / max(valid.sum(), 1) * 100 for c in WCC if (arr == c).any()}
        top = sorted([kv for kv in share.items() if kv[1] >= 0.5], key=lambda kv: kv[1])[-4:]
        a3.barh([textwrap.fill(WCC[c][0], 15) for c, _ in top], [v for _, v in top], color=[WCC[c][1] for c, _ in top], height=0.7, edgecolor="#6B7280", linewidth=0.3)
        for i, (c, v) in enumerate(top):
            a3.text(v, i, f" {v:.0f}%", va="center", fontsize=5.6, color=t.ink)
        a3.set_xlim(0, max(v for _, v in top) * 1.3)
        a3.tick_params(axis="y", labelsize=5.6)
        a3.set_title("Land cover, % of window", fontsize=5.8, color=t.ink, loc="right", pad=3)
        self.log["land_cover_pct"] = {WCC[c][0]: round(v, 1) for c, v in share.items()}
        sheet.prov.add("Land cover in the map window", wc.SOURCE, None, "10 m classes counted on a 20 m nearest-neighbour grid over the whole window")
