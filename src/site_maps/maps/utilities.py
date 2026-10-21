"""Utilities site map: a power generating station and the grid, pipelines, access and water around it."""
from __future__ import annotations

import datetime as dt
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from shapely.geometry import Point, box

from ..core.cache import OUTPUTS
from ..core.frame import SiteFrame
from ..core.layout import Legend, Sheet, halo
from ..core.provenance import ProvenanceLog
from ..sources import dem as demsrc, overture as ov, power, sentinel2 as s2, worldcover as wc
from ..sources.rasters import Grid
from .base import clip_to_frame, label_lines, overture_layer, raster_extent

GPPD_ID = "USA0003470"                      # WRI Global Power Plant Database record
GPPD_NAME = "W A Parish"
MAP_WIDTH_M = 15000.0
STAT_RADII_M = (5000.0, 20000.0)
BAND_EDGES_M = [0, 1000, 2500, 5000, 10000, 20000]
LINE_STYLE = {"≥ 345 kV": ("#B91C1C", 2.3), "230–344 kV": ("#EA580C", 1.9), "115–229 kV": ("#EAB308", 1.5), "< 115 kV": ("#14B8A6", 1.0), power.UNKNOWN: ("#CBD5E1", 0.9)}
ORDER = ["≥ 345 kV", "230–344 kV", "115–229 kV", "< 115 kV", power.UNKNOWN]


def _geom(g, kinds):
    return g[g.geom_type.isin(kinds)]


class UtilitiesSiteMap:
    key = "utilities_wa_parish"

    def __init__(self, outdir=OUTPUTS):
        self.outdir = outdir
        self.prov = ProvenanceLog()
        self.log: dict = {}

    # ---------------- data ----------------
    def facility(self):
        gp = power.load_gppd()
        r = gp[gp["gppd_idnr"] == GPPD_ID]
        assert len(r) == 1 and r.iloc[0]["name"] == GPPD_NAME, "WRI record not found"
        return r.iloc[0]

    def _wide(self, kind, frame, columns, tag, radius_m=22000.0, extra_filter=None):
        wide = SiteFrame.around(*frame.center, radius_m * 2, 1.0, frame.crs)
        g = ov.read(kind, wide.padded_bounds_ll(0.02), columns, extra_filter=extra_filter, crs=frame.crs, tag=tag)
        return g

    def build(self):
        prov = self.prov
        rec = self.facility()
        lon, lat = float(rec["longitude"]), float(rec["latitude"])
        frame = SiteFrame.around(lon, lat, MAP_WIDTH_M, Sheet.map_aspect())
        cx, cy = frame.cx_cy
        plant_xy = (cx, cy)
        prov.add("Facility record (name, capacity, fuels, owner, location)", power.GPPD, 1, f"record {GPPD_ID}; geolocation source in record: {rec['geolocation_source']}")

        # ---- infrastructure within 22 km (statistics) and the map window (drawing)
        infra = self._wide("infrastructure", frame, ["id", "names", "subtype", "class", "source_tags"], "ut_wide")
        prov.add("Power, pipeline and utility features", ov.SOURCE, len(infra), "Overture base/infrastructure (OpenStreetMap-derived) within 22 km of the plant")
        infra["tags"] = infra["source_tags"].map(power.tag_dict)
        pw = infra[infra["subtype"] == "power"].copy()
        pw["kv"] = pw["tags"].map(power.max_voltage_kv)
        pw["vclass"] = pw["kv"].map(power.voltage_class)

        # resolve the plant polygon in Overture by its EIA plant code (the numeric tail of the WRI id)
        eia = str(int(GPPD_ID[3:]))
        plants = pw[(pw["class"] == "plant") & (pw["tags"].map(lambda t: t.get("ref:US:EIA")) == eia)]
        assert len(plants) == 1, f"expected one Overture plant with ref:US:EIA={eia}, got {len(plants)}"
        plant = plants.iloc[0]
        offset_m = float(Point(plant_xy).distance(plant.geometry.centroid))
        pt = plant["tags"]
        self.log["overture_plant"] = {"name": plant["name"], "eia_code": eia, "source_tag": pt.get("source"), "output_tag": pt.get("plant:output:electricity"), "offset_from_wri_point_m": round(offset_m)}

        lines = _geom(pw[pw["class"].isin(["power_line", "minor_line"])], ["LineString", "MultiLineString"])
        cables = _geom(pw[pw["class"] == "cable"], ["LineString", "MultiLineString"])
        towers = _geom(pw[pw["class"] == "power_tower"], ["Point"])
        poles = _geom(pw[pw["class"] == "power_pole"], ["Point"])
        subs = pw[pw["class"] == "substation"].copy()
        subs["pt"] = subs.geometry.map(lambda g: g.centroid)
        pipes = _geom(infra[(infra["subtype"] == "utility") & (infra["class"] == "pipeline")], ["LineString", "MultiLineString"])

        # ---- statistics (computed from the data)
        stats = {}
        for R in STAT_RADII_M:
            stats[f"line_km_{int(R / 1000)}km"] = {k: round(v, 1) for k, v in power.length_by_class_km(lines, plant_xy, R).items()}
        self.log["line_km"] = stats
        self.log["tower_bands"] = dict(zip([f"{a / 1000:g}-{b / 1000:g} km" for a, b in zip(BAND_EDGES_M[:-1], BAND_EDGES_M[1:])], power.counts_by_band(towers, plant_xy, BAND_EDGES_M)))
        subs_xy = gpd.GeoSeries(subs["pt"], crs=frame.crs)
        sd = subs_xy.distance(Point(plant_xy))
        subs = subs.assign(dist_m=sd.to_numpy())
        self.log["substation_bands"] = dict(zip(self.log["tower_bands"].keys(), power.counts_by_band(gpd.GeoDataFrame(geometry=subs_xy), plant_xy, BAND_EDGES_M)))
        trans = subs[subs["tags"].map(lambda t: t.get("substation")) == "transmission"]
        self.log["nearest_substation"] = {"name": subs.sort_values("dist_m").iloc[0]["name"], "km": round(subs["dist_m"].min() / 1000, 2)}
        self.log["nearest_transmission_substation"] = {"name": trans.sort_values("dist_m").iloc[0]["name"], "km": round(trans["dist_m"].min() / 1000, 2)} if len(trans) else None
        self.log["counts_22km"] = {"lines": int(len(lines)), "towers": int(len(towers)), "poles": int(len(poles)), "substations": int(len(subs)), "pipelines": int(len(pipes))}

        # ---- sheet
        sheet = Sheet(f"{rec['name']} Generating Station, Texas", "Utility site map · generating plant, transmission and distribution lines, substations, pipelines, access and water",
                      f"{lat:.4f}° N  {abs(lon):.4f}° W  ·  {rec['capacity_mw']:,.0f} MW ({rec['primary_fuel']})", frame, prov)
        ax = sheet.ax

        # imagery base
        scs = [s for s in s2.scenes(lon, lat, dt.date(2026, 3, 1), dt.date(2026, 10, 15)) if s.cloud < 50]
        scored = []
        for s in scs:
            cov, cl = s2.window_quality(s, frame)
            if cov > 0.99:
                scored.append((round(cl, 2), -s.date.toordinal(), s))
        scored.sort(key=lambda t: (t[0], t[1]))
        cl, _, base = scored[0]
        rgb, grid = s2.true_color(base, frame, 10.0)
        ax.imshow(rgb, extent=raster_extent(grid), origin="upper", zorder=1, interpolation="lanczos")
        prov.add(f"Imagery: Sentinel-2 true colour, {base.date}", s2.SOURCE, None, f"scene {base.prefix.split('/')[-1]}; TCI at 10 m; window cloud/shadow {cl:.1%}")
        self.log["imagery_scene"] = base.prefix.split("/")[-1]

        # water
        water = overture_layer("water", frame, ["id", "names", "subtype", "class"], prov, "Surface water", tag="ut")
        wpoly = clip_to_frame(_geom(water, ["Polygon", "MultiPolygon"]), frame)
        wline = clip_to_frame(_geom(water, ["LineString", "MultiLineString"]), frame)
        if len(wpoly):
            wpoly.plot(ax=ax, fc="#38BDF8", ec="#0369A1", lw=0.5, alpha=0.45, zorder=3)
        if len(wline):
            wline.plot(ax=ax, color="#38BDF8", lw=0.9, zorder=3)

        # roads and rail
        seg = overture_layer("segments", frame, ["id", "names", "subtype", "class"], prov, "Roads and railways", tag="ut")
        seg = clip_to_frame(seg, frame)
        rail = seg[seg["subtype"] == "rail"]
        roads = seg[(seg["subtype"] == "road") & seg["class"].isin(["motorway", "trunk", "primary", "secondary", "tertiary"])]
        if len(roads):
            roads.plot(ax=ax, color="white", lw=2.6, zorder=4)
            roads.plot(ax=ax, color="#475569", lw=1.1, zorder=4.1)
        if len(rail):
            rail.plot(ax=ax, color="white", lw=2.4, zorder=4.2)
            rail.plot(ax=ax, color="#111827", lw=1.2, ls="--", zorder=4.3)
        label_lines(ax, roads, "name", frame, 0.10, 6.2, exclude=lambda x, y: _in_furniture(frame, x, y), max_labels=8)

        # land-use industrial footprint nearest the plant
        lu = overture_layer("land_use", frame, ["id", "names", "subtype", "class"], prov, "Industrial land use", tag="ut")
        lu_ind = clip_to_frame(lu[lu["class"] == "industrial"], frame)

        # power layers
        pipes_f = clip_to_frame(pipes, frame)
        if len(pipes_f):
            pipes_f.plot(ax=ax, color="#7C3AED", lw=1.0, ls=(0, (1, 1.5)), zorder=5)
        lines_f = clip_to_frame(lines, frame)
        for cls in reversed(ORDER):
            g = lines_f[lines_f["vclass"] == cls]
            if len(g):
                col, lw = LINE_STYLE[cls]
                g.plot(ax=ax, color="#0F172A", lw=lw + 1.2, alpha=0.55, zorder=6)
                g.plot(ax=ax, color=col, lw=lw, zorder=6.1, ls="-" if cls != power.UNKNOWN else "--")
        cables_f = clip_to_frame(cables, frame)
        if len(cables_f):
            cables_f.plot(ax=ax, color="#C026D3", lw=1.0, ls=":", zorder=6.2)
        tw = towers[towers.within(box(*frame.extent))]
        ax.scatter(tw.geometry.x, tw.geometry.y, s=2.2, c="#111827", zorder=7, linewidths=0)
        prov.add("Transmission towers", ov.SOURCE, len(tw), "power_tower points inside the map window")

        # substations and plant footprint
        sx = subs[subs["pt"].map(lambda p: box(*frame.extent).contains(p))]
        for _, r in sx.iterrows():
            kv = power.max_voltage_kv(r["tags"])
            if r.geometry.geom_type in ("Polygon", "MultiPolygon"):
                gpd.GeoSeries([r.geometry], crs=frame.crs).plot(ax=ax, fc="#FDBA74", ec="#9A3412", lw=0.8, alpha=0.9, zorder=8)
            ax.plot(r["pt"].x, r["pt"].y, marker="s", ms=5.5, mfc="#F97316", mec="white", mew=0.8, zorder=9)
            if r["name"]:
                ax.annotate(f"{r['name']}" + (f" ({kv:g} kV)" if kv else ""), (r["pt"].x, r["pt"].y), xytext=(6, -9), textcoords="offset points", fontsize=5.9, color="#1F2933", zorder=10, path_effects=halo(2.2))
        gpd.GeoSeries([plant.geometry], crs=frame.crs).plot(ax=ax, fc="#EF4444", ec="white", lw=1.6, alpha=0.5, zorder=8.5)
        gpd.GeoSeries([plant.geometry], crs=frame.crs).boundary.plot(ax=ax, color="#7F1D1D", lw=1.3, zorder=8.6)
        ax.plot(cx, cy, marker="*", ms=13, mfc="#FACC15", mec="#111827", mew=1.0, zorder=12)
        ax.annotate(f"{rec['name']} (WRI {GPPD_ID})", (cx, cy), xytext=(10, 12), textcoords="offset points", fontsize=7.4, fontweight="bold", zorder=13, path_effects=halo(2.6))
        prov.add("Plant footprint", ov.SOURCE, 1, f"Overture infrastructure feature (power=plant, ref:US:EIA={eia}), drawn as published; WRI point is {offset_m:,.0f} m from its centroid")
        prov.add("Substations", ov.SOURCE, len(sx), "power=substation features whose centre is inside the map window; names and voltages as tagged")

        # nearest water (computed from the polygons in the window)
        water_all = self._wide("water", frame, ["id", "names", "subtype", "class"], "ut_wide", 12000.0)
        wp_all = _geom(water_all, ["Polygon", "MultiPolygon"])
        wp_all = wp_all.assign(d=wp_all.geometry.distance(Point(plant_xy)), a=wp_all.geometry.area / 1e6)
        nearest = wp_all.sort_values("d").iloc[0]
        big = wp_all[wp_all["d"] < 5000].sort_values("a", ascending=False)
        self.log["nearest_water_polygon"] = {"name": nearest["name"], "km": round(nearest["d"] / 1000, 2), "class": nearest["class"]}
        self.log["largest_water_within_5km"] = {"name": big.iloc[0]["name"], "km2": round(big.iloc[0]["a"], 2)} if len(big) else None

        sheet.graticule()
        sheet.scale_bar()
        sheet.north_arrow()
        self._detail_inset(sheet, frame, plant, rec, lon, lat)

        # ---- land cover within 5 and 20 km of the plant (30 m grid for statistics)
        lcstats = self._landcover_stats(plant_xy, frame)

        # ---- right column
        sheet.locator(width_deg=7.0, height_in=1.1)
        lg = Legend("Legend")
        lg.header("Power system (OpenStreetMap-derived)")
        for cls in ORDER:
            if cls in set(lines["vclass"]):
                lg.line(f"Line {cls}", LINE_STYLE[cls][0], LINE_STYLE[cls][1] + 0.4, "-" if cls != power.UNKNOWN else "--")
        if len(cables_f):
            lg.line("Cable", "#C026D3", 1.0, ":")
        lg.marker("Transmission tower", "#111827", "o", 2.5)
        lg.marker("Substation", "#F97316", "s", 5)
        lg.patch("Plant footprint", "#EF4444")
        lg.marker("WRI plant location", "#FACC15", "*", 9, "#111827")
        lg.line("Pipeline", "#7C3AED", 1.0, (0, (1, 1.5)))
        lg.header("Base")
        lg.line("Road (primary–tertiary)", "#475569", 1.1)
        lg.line("Railway", "#111827", 1.2, "--")
        lg.patch("Surface water", "#38BDF8")
        sheet.legend(lg)
        self._facts(sheet, rec, plant, lcstats)
        self._charts(sheet, stats, lcstats)
        sheet.note_box("Limits", [
            "Power, pipeline and substation features are OpenStreetMap contributions carried by Overture; they are not an engineering or utility record, voltages are only as tagged, and absence of a feature does not mean absence on the ground.",
            f"Plant attributes (capacity, fuel, owner, location) are from WRI Global Power Plant Database v1.3.0, a snapshot WRI says is unmaintained since early 2022; Overture's plant tag cites {pt.get('source')}.",
            "Land cover is the ESA WorldCover 2021 10 m product (statistics at 30 m).",
        ], width_chars=92, fontsize=5.5)
        sheet.sources_box("Data sources")
        sheet.export(self.key, self.outdir)
        return sheet

    # ---------------- pieces ----------------
    def _landcover_stats(self, center, frame):
        out = {}
        wide = SiteFrame.around(*frame.center, STAT_RADII_M[1] * 2.05, 1.0, frame.crs)
        lc, grid = wc.land_cover(wide, 30.0)
        arr = np.ma.filled(lc, 0).astype(int)
        x0, y0, x1, y1 = grid.extent
        ys = y1 - (np.arange(arr.shape[0]) + 0.5) * grid.res
        xs = x0 + (np.arange(arr.shape[1]) + 0.5) * grid.res
        XX, YY = np.meshgrid(xs, ys)
        dist = np.hypot(XX - center[0], YY - center[1])
        for R in STAT_RADII_M:
            m = dist <= R
            tot = m.sum()
            out[int(R / 1000)] = {wc.CLASSES[c][0]: float((arr[m] == c).sum() / tot) for c in wc.CLASSES if (arr[m] == c).any()}
        self.prov.add("Land cover within 5 and 20 km", wc.SOURCE, None, "10 m classes resampled (nearest) to a 30 m grid; shares of pixels inside each radius")
        self.log["land_cover_share_5km"] = {k: round(v, 3) for k, v in out[5].items()}
        self.log["land_cover_share_20km"] = {k: round(v, 3) for k, v in out[20].items()}
        return out

    def _detail_inset(self, sheet, frame, plant, rec, lon, lat):
        """Plant footprint detail (1.6 km window) with buildings and generator/transformer points, placed over the lower right of the main map."""
        c = plant.geometry.centroid
        w = 2400.0
        mb = sheet.ax.get_position()
        iw = 0.19
        ih = iw * 17.0 / 11.0
        left, bottom = mb.x1 - iw - 0.008, mb.y0 + 0.008
        ia = sheet.fig.add_axes((left, bottom, iw, ih))
        ia.set_xlim(c.x - w / 2, c.x + w / 2)
        ia.set_ylim(c.y - w / 2, c.y + w / 2)
        ia.set_aspect("equal")
        ia.set_xticks([]); ia.set_yticks([])
        sub = SiteFrame(self_center(frame, c), w, 1.0, frame.crs)
        bl = ov.read("buildings", sub.padded_bounds_ll(0.05), ["id", "class", "subtype", "height"], crs=frame.crs, tag="ut_detail")
        win = box(c.x - w / 2, c.y - w / 2, c.x + w / 2, c.y + w / 2)
        bl = bl[bl.intersects(win)]
        ia.set_facecolor("#F3F4F6")
        gpd.GeoSeries([plant.geometry], crs=frame.crs).plot(ax=ia, fc="#FCA5A5", ec="#7F1D1D", lw=0.9, alpha=0.7)
        if len(bl):
            bl.plot(ax=ia, fc="#9CA3AF", ec="#374151", lw=0.3, alpha=0.9)
        pw = ov.read("infrastructure", sub.padded_bounds_ll(0.05), ["id", "names", "subtype", "class"], crs=frame.crs, tag="ut_detail")
        gen = pw[(pw["subtype"] == "power") & pw["class"].isin(["generator", "transformer", "switch"])]
        gen = gen[gen.geometry.centroid.within(win)]
        cen = gen.geometry.centroid
        ia.scatter(cen.x, cen.y, s=3.5, c="#1D4ED8", zorder=5, linewidths=0)
        ia.set_xlim(c.x - w / 2, c.x + w / 2); ia.set_ylim(c.y - w / 2, c.y + w / 2)
        for sp in ia.spines.values():
            sp.set_edgecolor("#111827"); sp.set_linewidth(1.2)
        ia.text(0.03, 0.97, "PLANT DETAIL 2.4 km", transform=ia.transAxes, fontsize=5.8, fontweight="bold", va="top", color="#111827", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ia.text(0.03, 0.03, f"{len(bl):,} buildings · {len(gen):,} generator/transformer/switch points", transform=ia.transAxes, fontsize=5.0, va="bottom", color="#111827", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        sheet.prov.add("Plant detail: buildings", ov.SOURCE, len(bl), "Overture building footprints intersecting the 2.4 km detail window")
        sheet.prov.add("Plant detail: generators, transformers, switches", ov.SOURCE, len(gen), "power=generator / transformer / switch features in the detail window, drawn as points")
        self.log["detail_buildings"] = int(len(bl))
        self.log["detail_generator_points"] = int(len(gen))

    def _facts(self, sheet, rec, plant, lcstats):
        ov_t = plant["tags"]
        gen = rec.get("generation_gwh_2019")
        rows = [("Name (WRI)", f"{rec['name']}"), ("Capacity (WRI)", f"{rec['capacity_mw']:,.1f} MW"), ("Primary fuel (WRI)", str(rec["primary_fuel"])),
                ("Owner (WRI)", str(rec["owner"])), ("Commissioning year (WRI)", f"{rec['commissioning_year']:.0f}" if pd.notna(rec["commissioning_year"]) else "n/a"),
                ("Operator (Overture tag)", ov_t.get("operator", "n/a")), ("Fuel tag (Overture)", ov_t.get("plant:source", "n/a")), ("Footprint area (Overture)", f"{plant.geometry.area / 1e6:.2f} km²"),
                ("Nearest substation", f"{self.log['nearest_substation']['name'] or 'unnamed'}, {self.log['nearest_substation']['km']} km"),
                ("Nearest surface water", f"{self.log['nearest_water_polygon']['name'] or 'unnamed ' + str(self.log['nearest_water_polygon']['class'])}, {self.log['nearest_water_polygon']['km']} km")]
        h = 0.26 + 0.08 + len(rows) * 0.145
        ax = sheet.panel(h, "Facility facts")
        y = 1 - (0.26 + 0.07) / h
        for k, v in rows:
            ax.text(0.03, y, k, fontsize=5.8, color=sheet.t.muted, va="top", transform=ax.transAxes)
            ax.text(0.97, y, v, fontsize=6.0, color=sheet.t.ink, va="top", ha="right", transform=ax.transAxes)
            y -= 0.145 / h

    def _charts(self, sheet, stats, lcstats):
        t = sheet.t
        panel = sheet.panel(1.65, "Grid around the plant")
        a1 = sheet.chart_axes(panel, (0.11, 0.18, 0.40, 0.48))
        classes = [c for c in ORDER if any(c in stats[k] for k in stats)]
        x = np.arange(len(classes))
        for i, (k, col) in enumerate((("line_km_5km", "#94A3B8"), ("line_km_20km", "#334155"))):
            a1.bar(x + (i - 0.5) * 0.38, [stats[k].get(c, 0) for c in classes], 0.36, color=col, label=f"within {k.split('_')[-1].replace('km', ' km')}")
        a1.set_xticks(x); a1.set_xticklabels([c.replace("voltage not tagged", "untagged") for c in classes], fontsize=5.2, rotation=20)
        a1.set_title("Line length by voltage class (km)", fontsize=6.4, loc="left", color=t.ink, pad=3)
        a1.legend(fontsize=5.4, frameon=False, loc="upper right")
        a2 = sheet.chart_axes(panel, (0.62, 0.18, 0.35, 0.48))
        bands = list(self.log["tower_bands"].keys())
        xb = np.arange(len(bands))
        a2.bar(xb - 0.19, list(self.log["tower_bands"].values()), 0.36, color="#111827", label="towers")
        a2.bar(xb + 0.19, list(self.log["substation_bands"].values()), 0.36, color="#F97316", label="substations")
        a2.set_xticks(xb); a2.set_xticklabels([b.replace(" km", "") for b in bands], fontsize=5.4)
        a2.set_xlabel("distance from plant (km)", fontsize=5.5, color=t.muted, labelpad=1)
        a2.set_title("Towers and substations by distance band", fontsize=6.4, loc="left", color=t.ink, pad=3)
        a2.legend(fontsize=5.4, frameon=False, loc="upper left")
        panel = sheet.panel(1.45, "Land cover around the plant (share of area)")
        a3 = sheet.chart_axes(panel, (0.30, 0.16, 0.64, 0.62))
        names = [n for n in sorted({n for r in lcstats.values() for n in r}, key=lambda n: -lcstats[20].get(n, 0))][:7][::-1]
        yy = np.arange(len(names))
        a3.barh(yy + 0.2, [lcstats[5].get(n, 0) * 100 for n in names], 0.38, color="#94A3B8", label="within 5 km")
        a3.barh(yy - 0.2, [lcstats[20].get(n, 0) * 100 for n in names], 0.38, color="#334155", label="within 20 km")
        a3.set_yticks(yy); a3.set_yticklabels(names, fontsize=5.6)
        a3.set_xlabel("% of area", fontsize=5.5, color=t.muted, labelpad=1)
        a3.legend(fontsize=5.4, frameon=False, loc="lower right")


def self_center(frame, c):
    lon, lat = frame.to_ll.transform(c.x, c.y)
    return (lon, lat)



def _in_furniture(frame, x, y) -> bool:
    """True under the north arrow (top right) or scale bar (bottom left); labels there are skipped."""
    x0, y0, x1, y1 = frame.extent
    W, H = x1 - x0, y1 - y0
    return (x > x1 - 0.14 * W and y > y1 - 0.20 * H) or (x < x0 + 0.27 * W and y < y0 + 0.19 * H)
