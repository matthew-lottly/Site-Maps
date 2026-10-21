"""Environmental research site map: protected-area context with imagery, land cover, hydrography, field observations, vegetation condition and climate."""
from __future__ import annotations

import datetime as dt
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_rgba
from shapely.geometry import box

from ..core.cache import OUTPUTS, cache_path
from ..core.frame import SiteFrame
from ..core.layout import Legend, Sheet, halo
from ..core.provenance import ProvenanceLog
from ..sources import dem as demsrc, ghcn, inat, overture as ov, sentinel2 as s2, worldcover as wc
from ..sources.rasters import Grid, hillshade
from .base import clip_to_frame, label_lines, mask_outside, overture_layer, raster_extent

SITE_ID = "adb37d48-f604-3b63-b9b0-099010f527b9"            # Overture land_use feature
SITE_NAME = "Brazos Bend State Park"
SITE_BOX = (-96.0, 29.0, -95.2, 29.7)                       # search window used only to fetch the feature by id
INAT_BOX = "brazos_bend"
GROUPS = {"Aves": "Birds", "Reptilia": "Reptiles", "Amphibia": "Amphibians", "Mammalia": "Mammals", "Actinopterygii": "Fishes", "Insecta": "Insects", "Arachnida": "Arachnids",
          "Plantae": "Plants", "Fungi": "Fungi", "Mollusca": "Molluscs"}
GROUP_COLORS = {"Birds": "#1D4ED8", "Reptiles": "#B45309", "Amphibians": "#0E7490", "Mammals": "#7C2D12", "Fishes": "#0369A1", "Insects": "#C026D3", "Arachnids": "#9D174D",
                "Plants": "#15803D", "Fungi": "#6D28D9", "Molluscs": "#475569", "Other": "#64748B"}


class EnvironmentalSiteMap:
    key = "environmental_brazos_bend"

    def __init__(self, outdir=OUTPUTS):
        self.outdir = outdir
        self.prov = ProvenanceLog()
        self.log: dict = {}

    # ---------------- data ----------------
    def site(self):
        g = ov.read("land_use", SITE_BOX, ["id", "names", "subtype", "class"], tag="env_site")
        s = g[g["id"] == SITE_ID]
        assert len(s) == 1 and s.iloc[0]["name"] == SITE_NAME, "site feature not found in the pinned Overture release"
        return s

    def observations(self, frame: SiteFrame, park_geom) -> pd.DataFrame:
        f = cache_path(f"inat/{INAT_BOX}.csv")
        if not f.exists():
            return pd.DataFrame()
        o = pd.read_csv(f, dtype={"observation_uuid": str})
        o = o[(o["quality_grade"] == "research") & (o["positional_accuracy"].fillna(1e9) <= 100)]
        w, s, e, n = frame.padded_bounds_ll(0.0)
        o = o[(o["longitude"].between(w, e)) & (o["latitude"].between(s, n))].copy()
        t = inat.taxa()
        ids = {name: int(t[(t["name"] == name) & (t["active"]) & (t["rank"].isin(["class", "kingdom", "phylum"]))]["taxon_id"].iloc[0]) for name in GROUPS if ((t["name"] == name) & t["active"]).any()}
        anc = t.set_index("taxon_id")["ancestry"].to_dict()
        id2group = {v: GROUPS[k] for k, v in ids.items()}

        def group(tid):
            chain = [int(x) for x in str(anc.get(tid, "")).split("/") if x.isdigit()] + [tid]
            for c in reversed(chain):
                if c in id2group:
                    return id2group[c]
            return "Other"

        o["group"] = o["taxon_id"].map(group)
        xs, ys = frame.to_proj.transform(o["longitude"].to_numpy(), o["latitude"].to_numpy())
        o["x"], o["y"] = xs, ys
        o["species_name"] = o["taxon_id"].map(t.set_index("taxon_id")["name"].to_dict())
        return o

    def ndvi_series(self, frame: SiteFrame, park_geom, start=dt.date(2025, 10, 1), end=dt.date(2026, 10, 15)):
        """Park-mean NDVI for the clearest scene of each month (window coverage > 99 %, cloud/shadow < 15 % of the window)."""
        cache = cache_path(f"derived/ndvi_series_{SITE_ID}_{start}_{end}.json")
        if cache.exists():
            return json.loads(cache.read_text())
        rows = []
        cand = [s for s in s2.scenes(*frame.center, start, end) if s.cloud < 50]
        by_month: dict = {}
        for s in cand:
            cov, cl = s2.window_quality(s, frame)
            if cov > 0.99 and cl < 0.15:
                k = (s.date.year, s.date.month)
                if k not in by_month or cl < by_month[k][1]:
                    by_month[k] = (s, cl)
        for k, (s, cl) in sorted(by_month.items()):
            b = s2.read_masked(s, frame)
            inside = mask_outside(None, park_geom, b["grid"]) & b["good"]
            v = np.ma.filled(b["ndvi"], np.nan)[inside]
            v = v[np.isfinite(v)]
            if len(v) > 500:
                rows.append({"date": s.date.isoformat(), "scene": s.prefix.split("/")[-1], "cloud_window": cl, "mean": float(np.mean(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": int(len(v))})
        cache.write_text(json.dumps(rows))
        return rows

    # ---------------- map ----------------
    def build(self):
        prov = self.prov
        site = self.site()
        park = site.to_crs(SiteFrame.around(*site.geometry.iloc[0].centroid.coords[0], 1000, 1.2).crs)
        c = site.geometry.iloc[0].centroid
        frame = SiteFrame.around(c.x, c.y, 12600, Sheet.map_aspect())
        park = site.to_crs(frame.crs)
        pg = park.geometry.iloc[0]
        prov.add("Site boundary: " + SITE_NAME, ov.SOURCE, 1, f"Overture land_use feature {SITE_ID} (subtype protected / class natural_monument)")
        sheet = Sheet("Brazos Bend State Park, Texas", "Environmental site map · imagery, land cover, hydrography, field observations, vegetation condition and climate",
                      f"{c.y:.4f}° N  {abs(c.x):.4f}° W  ·  park area {pg.area / 1e6:.1f} km²", frame, prov)
        ax = sheet.ax
        sheet.notes += ["Land cover is a 10 m global product and misclassifies small features.", "The park polygon is OSM-derived, not a surveyed boundary.",
                        "Observation density follows where people walk, so it reflects observer effort as well as biodiversity.", "NDVI months without a clear scene are absent, not interpolated."]

        # imagery base: clearest recent scene
        scs = [s for s in s2.scenes(c.x, c.y, dt.date(2026, 3, 1), dt.date(2026, 10, 15)) if s.cloud < 50]
        scored = []
        for s in scs:
            cov, cl = s2.window_quality(s, frame)
            if cov > 0.99:
                scored.append((cl, -s.date.toordinal(), s))
        scored.sort(key=lambda t: (round(t[0], 2), t[1]))
        cloud_frac, _, base = scored[0]
        rgb, grid = s2.true_color(base, frame, 10.0)
        ax.imshow(rgb, extent=raster_extent(grid), origin="upper", zorder=1, interpolation="lanczos")
        prov.add(f"Imagery: Sentinel-2 true colour, {base.date}", s2.SOURCE, None, f"scene {base.prefix.split('/')[-1]}; TCI product at 10 m; window cloud/shadow {cloud_frac:.1%}")
        self.log["imagery_scene"] = base.prefix.split("/")[-1]

        # land cover inside the park
        lc, lgrid = wc.land_cover(frame, 10.0)
        inside = mask_outside(None, pg, lgrid)
        lc_arr = np.ma.filled(lc, 0).astype(int)
        rgba = np.zeros(lc_arr.shape + (4,))
        for code, (nm, col) in wc.CLASSES.items():
            m = (lc_arr == code) & inside
            rgba[m] = to_rgba(col, 0.40)
        ax.imshow(rgba, extent=raster_extent(lgrid), origin="upper", zorder=2, interpolation="nearest")
        prov.add("Land cover inside the park boundary", wc.SOURCE, None, "10 m classes drawn at 40 % opacity over imagery, only inside the Overture park polygon")
        area = {}
        px = (lgrid.res ** 2) / 1e6
        for code, (nm, col) in wc.CLASSES.items():
            n = int(((lc_arr == code) & inside).sum())
            if n:
                area[code] = n * px
        self.log["land_cover_km2"] = {wc.CLASSES[k][0]: round(v, 3) for k, v in area.items()}

        # hydrography
        water = overture_layer("water", frame, ["id", "names", "subtype", "class", "is_intermittent"], prov, "Hydrography (surface water)", tag="env")
        wpoly = clip_to_frame(water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])], frame)
        wline = clip_to_frame(water[water.geometry.geom_type.isin(["LineString", "MultiLineString"])], frame)
        if len(wpoly):
            wpoly.plot(ax=ax, fc="#38BDF8", ec="#0369A1", lw=0.6, alpha=0.55, zorder=3)
        if len(wline):
            wline.plot(ax=ax, color="#0369A1", lw=0.9, zorder=3)

        named = wpoly[wpoly["name"].notna()] if len(wpoly) else wpoly
        for _, r_ in named.iterrows():
            if r_.geometry.area > 4e4:
                p_ = r_.geometry.representative_point()
                ax.text(p_.x, p_.y, r_["name"], fontsize=5.8, style="italic", color="#0C4A6E", ha="center", va="center", zorder=8, path_effects=halo(2.2, "#E0F2FE"))

        # roads and trails
        seg = overture_layer("segments", frame, ["id", "names", "class", "subclass", "road_surface"], prov, "Roads and trails", extra_filter=None, tag="env")
        seg = clip_to_frame(seg, frame)
        roads = seg[seg["class"].isin(["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service"])]
        trails = seg[seg["class"].isin(["path", "footway", "track", "cycleway", "bridleway", "pedestrian"])]
        if len(roads):
            roads.plot(ax=ax, color="white", lw=2.2, zorder=4)
            roads.plot(ax=ax, color="#52606D", lw=1.0, zorder=4.1)
        if len(trails):
            trails.plot(ax=ax, color="#7C2D12", lw=0.8, ls="--", zorder=4.2)
        label_lines(ax, roads, "name", frame, 0.10, 6.2, exclude=lambda x, y: sheet_in_furniture(frame, x, y), max_labels=9)

        # park outline
        gpd.GeoSeries([pg], crs=frame.crs).boundary.plot(ax=ax, color="white", lw=3.6, zorder=6)
        gpd.GeoSeries([pg], crs=frame.crs).boundary.plot(ax=ax, color="#B42318", lw=1.7, zorder=6.1)
        # park-level places
        places = overture_layer("places", frame, ["id", "names", "taxonomy", "confidence", "operating_status"], prov, "Points of interest (named)", tag="env")
        pl = places[places["category"].isin(["campground", "park", "nature_reserve", "visitor_center", "tourist_information_center", "hiking_trail", "picnic_area", "boat_ramp", "fishing_area", "lake"]) & (places["confidence"] >= 0.6)]
        pl = clip_to_frame(pl, frame)
        pl = pl[pl.within(pg.buffer(400))]
        for _, r in pl.iterrows():
            ax.plot(r.geometry.x, r.geometry.y, marker="s", ms=5.2, mfc="#F59E0B", mec="white", mew=0.8, zorder=20)
            ax.annotate(r["name"], (r.geometry.x, r.geometry.y), xytext=(6, 4), textcoords="offset points", fontsize=6.2, fontweight="bold", color="#1F2933", zorder=21, path_effects=halo(2.4))

        # field observations
        obs = self.observations(frame, pg)
        if len(obs):
            for g, gdf in obs.groupby("group"):
                ax.scatter(gdf["x"], gdf["y"], s=4.5, c=GROUP_COLORS.get(g, "#64748B"), edgecolors="white", linewidths=0.15, alpha=0.85, zorder=15)
            prov.add("Field observations (research grade)", inat.SOURCE, len(obs), "research-grade, positional accuracy ≤ 100 m, inside the map window; taxon group from iNaturalist taxonomy")
        self.log["observations_in_window"] = int(len(obs))

        sheet.graticule()
        sheet.scale_bar()
        sheet.north_arrow()

        # --- right column
        sheet.locator(width_deg=9.0, height_in=1.05)
        lg = Legend("Legend")
        lg.header("Boundary and base")
        lg.line("Park boundary (Overture land_use)", "#B42318", 1.7)
        lg.patch("Surface water", "#38BDF8")
        lg.line("Road", "#52606D", 1.2)
        lg.line("Trail / track", "#7C2D12", 1.0, "--")
        lg.marker("Named point of interest", "#F59E0B", "s", 5)
        lg.header("Land cover inside park (WorldCover 10 m, 40 % opacity)")
        for code, v in sorted(area.items(), key=lambda kv: -kv[1]):
            lg.patch(wc.CLASSES[code][0], wc.CLASSES[code][1])
        if len(obs):
            lg.header("Research-grade observations")
            for g in [g for g in GROUP_COLORS if g in set(obs["group"])]:
                lg.marker(g, GROUP_COLORS[g], "o", 4.5)
        sheet.legend(lg)

        self._charts(sheet, frame, pg, area, obs, base)
        sheet.sources_box("Data sources")
        sheet.export(self.key, self.outdir)
        return sheet

    def _charts(self, sheet: Sheet, frame, pg, area, obs, base):
        t = sheet.t
        # row 1: land cover bars | observations by group
        panel = sheet.panel(1.5, "Composition and observations")
        a1 = sheet.chart_axes(panel, (0.20, 0.17, 0.27, 0.50))
        items = sorted(area.items(), key=lambda kv: kv[1])
        a1.barh([wc.CLASSES[k][0] for k, _ in items], [v for _, v in items], color=[wc.CLASSES[k][1] for k, _ in items], height=0.7, edgecolor="#6B7280", linewidth=0.3)
        tot = sum(area.values())
        for i, (k, v) in enumerate(items):
            a1.text(v, i, f" {v:.2f} km² ({v / tot:.0%})", va="center", fontsize=5.8, color=t.ink)
        a1.set_xlim(0, max(area.values()) * 1.9)
        a1.set_title("Land cover in park (km²)", fontsize=6.6, color=t.ink, loc="left", pad=3)
        a1.tick_params(axis="y", labelsize=5.8)
        if len(obs):
            a2 = sheet.chart_axes(panel, (0.66, 0.17, 0.28, 0.50))
            cnt = obs[obs.apply(lambda r: pg.buffer(0).contains(__import__("shapely.geometry", fromlist=["Point"]).Point(r["x"], r["y"])), axis=1)]
            c = cnt.groupby("group").size().sort_values().tail(8)
            a2.barh(c.index, c.values, color=[GROUP_COLORS.get(g, "#64748B") for g in c.index], height=0.7)
            for i, v in enumerate(c.values):
                a2.text(v, i, f" {v:,}", va="center", fontsize=5.8, color=t.ink)
            a2.set_xlim(0, c.max() * 1.25)
            a2.set_title(f"Observations inside park (n={len(cnt):,})", fontsize=6.6, color=t.ink, loc="left", pad=3)
            a2.tick_params(axis="y", labelsize=5.8)
            self.log["observations_in_park"] = int(len(cnt))
            self.log["species_in_park"] = int(cnt["species_name"].nunique())
        # row 2: NDVI series | climograph
        panel = sheet.panel(1.4, "Vegetation condition and climate")
        a3 = sheet.chart_axes(panel, (0.12, 0.20, 0.34, 0.50))
        rows = self.ndvi_series(frame, pg)
        if rows:
            d = [dt.date.fromisoformat(r["date"]) for r in rows]
            import matplotlib.dates as mdates
            d = mdates.date2num(pd.to_datetime(d))
            a3.fill_between(d, [r["p10"] for r in rows], [r["p90"] for r in rows], color="#86EFAC", alpha=0.6, lw=0)
            a3.plot(d, [r["mean"] for r in rows], marker="o", ms=3, color="#166534", lw=1.3)
            a3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            a3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            a3.set_ylim(0.2, 1.0)
            a3.set_title("NDVI inside park, clearest scene per month", fontsize=6.6, color=t.ink, loc="left", pad=3)
            a3.tick_params(axis="x", labelsize=5.8)
            prov_scenes = ", ".join(r["scene"] for r in rows)
            sheet.prov.add("Vegetation condition (NDVI series)", s2.SOURCE, len(rows), f"NDVI=(NIR−Red)/(NIR+Red) from B08/B04 surface reflectance (DN×0.0001; the −0.1 offset is applied only where a scene's DNs carry it), vegetation/bare-soil pixels per the scene classification layer; mean and 10–90 % band over park pixels; scenes {prov_scenes}")
            self.log["ndvi_scenes"] = [r["scene"] for r in rows]
        a4 = sheet.chart_axes(panel, (0.62, 0.20, 0.30, 0.50))
        st, d, km = ghcn.best_station(*frame.center)
        if st is not None:
            n = ghcn.monthly_normals(d)
            ax2 = a4.twinx()
            a4.bar(n.index, n["prcp"], color="#93C5FD", width=0.7)
            ax2.plot(n.index, n["tmax"], color="#DC2626", lw=1.3)
            ax2.plot(n.index, n["tmin"], color="#2563EB", lw=1.3)
            a4.set_xticks(range(1, 13)); a4.set_xticklabels(list("JFMAMJJASOND"), fontsize=5.8)
            a4.set_ylabel("precip (mm)", fontsize=5.8, color=t.muted)
            ax2.set_ylabel("°C", fontsize=5.8, color=t.muted)
            ax2.tick_params(labelsize=5.8, length=2, colors=t.muted)
            for sp in ("top",):
                ax2.spines[sp].set_visible(False)
            a4.set_title(f"{st['name']} ({km:.0f} km)", fontsize=6.4, color=t.ink, loc="left", pad=3)
            yrs = sorted(set(d.index.year))
            sheet.prov.add(f"Climate normals: GHCN {st['id']}", ghcn.SOURCE, None, f"{st['name']}, {km:.0f} km from the park centre; {len(yrs)} complete years in 1991–2020; mean daily max/min and mean monthly precipitation computed from daily values")
            self.log["climate_station"] = {"id": st["id"], "name": st["name"], "km": km, "years": len(yrs)}


def sheet_in_furniture(frame, x, y) -> bool:
    x0, y0, x1, y1 = frame.extent
    W, H = x1 - x0, y1 - y0
    return (x > x1 - 0.14 * W and y > y1 - 0.20 * H) or (x < x0 + 0.27 * W and y < y0 + 0.19 * H)
