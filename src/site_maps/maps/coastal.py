"""Coastal / sea research site map: seafloor relief, shoreline imagery, model surface currents, sea-surface temperature and marine species records."""
from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.spatial import cKDTree

from ..core.cache import OUTPUTS, cache_path
from ..core.frame import SiteFrame
from ..core.labels import LabelPlacer
from ..core.layout import Legend, Sheet, halo
from ..core.place import context as place_context
from ..core.provenance import ProvenanceLog
from ..sources import bathymetry as bt, nos_ofs, obis, oisst, overture as ov, sentinel2 as s2
from ..sources.rasters import hillshade
from .base import clip_to_frame, overture_layer, raster_extent

SITE_ID = "2f309763-10f2-3b98-b2fc-d8444ea81852"        # Overture water feature: physical / bay point named "Massachusetts Bay"
SITE_NAME = "Massachusetts Bay"
SITE_BOX = (-71.2, 41.6, -69.5, 43.2)                    # search window used only to fetch the feature by id
OFS = "gomofs"
OBIS_BOX_PAD = 0.05
COARSE_EDGES = (0, 20, 60, 100)
DEPTH_EDGES = (0, 10, 20, 40, 60, 80, 100, 150)
CLASS_GROUPS = {"Mammalia": ("Marine mammals", "#B91C1C"), "Aves": ("Birds", "#7C3AED"), "Actinopterygii": ("Ray-finned fishes", "#0891B2"), "Elasmobranchii": ("Sharks and rays", "#4B5563"),
                "Malacostraca": ("Crustaceans", "#EA580C"), "Gastropoda": ("Snails", "#CA8A04"), "Bivalvia": ("Bivalves", "#A16207")}
OTHER = ("Other", "#6B7280")
BATHY = LinearSegmentedColormap.from_list("bathy", ["#E3F2F7", "#B8DDE9", "#7DBAD6", "#4A93BE", "#2A6AA0", "#194A82", "#0E2A5C"])


def month_dates(n_back: int = 12, day: int = 15, today: dt.date | None = None) -> list[dt.date]:
    """The `day`-th of each of the last `n_back` complete months, oldest first (days chosen so a final OISST file normally exists)."""
    t = (today or dt.date.today()).replace(day=1)
    out = []
    y, m = t.year, t.month
    for _ in range(n_back):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(dt.date(y, m, day))
    return out[::-1]


def zone_label(lo: float, hi: float | None) -> str:
    return f"{lo:g}–{hi:g}" if hi is not None else f"> {lo:g}"


def sample_grid(arr, grid, x, y):
    """Value of a grid at projected points (nearest cell); NaN outside."""
    zz = np.ma.filled(arr, np.nan)
    x0, y0, x1, y1 = grid.extent
    c = np.floor((np.asarray(x) - x0) / grid.res).astype(int)
    r = np.floor((y1 - np.asarray(y)) / grid.res).astype(int)
    ok = (r >= 0) & (r < zz.shape[0]) & (c >= 0) & (c < zz.shape[1])
    out = np.full(len(c), np.nan)
    out[ok] = zz[r[ok], c[ok]]
    return out


class CoastalSiteMap:
    key = "coastal_massachusetts_bay"

    def __init__(self, outdir=OUTPUTS):
        self.outdir = outdir
        self.prov = ProvenanceLog()
        self.log: dict = {}

    # ---------------- data ----------------
    def site(self):
        w = ov.read("water", SITE_BOX, ["id", "names", "subtype", "class"], tag="coastal_site")
        s = w[w["id"] == SITE_ID]
        assert len(s) == 1 and s.iloc[0]["name"] == SITE_NAME, "site feature not found in the pinned Overture release"
        return s

    def occurrences(self, frame: SiteFrame, z, grid) -> pd.DataFrame:
        w, s, e, n = frame.padded_bounds_ll(0.0)
        df = obis.occurrences((w - OBIS_BOX_PAD, s - OBIS_BOX_PAD, e + OBIS_BOX_PAD, n + OBIS_BOX_PAD), "mass_bay")
        df = df[(df["absence"] != True) & (df["dropped"] != True) & df["scientificName"].notna()]
        df = df[(df["decimalLatitude"].between(s, n)) & (df["decimalLongitude"].between(w, e))].copy()
        x, y = frame.to_proj.transform(df["decimalLongitude"].to_numpy(), df["decimalLatitude"].to_numpy())
        df["x"], df["y"] = x, y
        df["floor_m"] = sample_grid(z, grid, x, y)
        df["group"] = df["class"].map(lambda c: CLASS_GROUPS.get(c, OTHER)[0])
        return df

    def sst_series(self, bounds):
        dates = month_dates()
        latest_ok = dt.date.today() - dt.timedelta(days=20)
        years = range(2016, dates[-1].year + 1)
        want = {d: [dt.date(y, d.month, d.day) for y in years if y != d.year and dt.date(y, d.month, d.day) < latest_ok] for d in dates}
        flat = sorted({d for k, v in want.items() for d in [k] + v if d < latest_ok})
        cache = cache_path(f"derived/oisst_{bounds[0]:.3f}_{bounds[1]:.3f}_{flat[0]}_{flat[-1]}.json")
        import json
        if cache.exists():
            vals = {dt.date.fromisoformat(k): v for k, v in json.loads(cache.read_text()).items()}
        else:
            res = oisst.series(flat, bounds)
            vals = {d: ({"sst": r[0], "n": r[1], "kind": r[2]} if r and r[0] is not None else None) for d, r in zip(flat, res)}
            cache.write_text(json.dumps({d.isoformat(): v for d, v in vals.items()}))
        return dates, want, vals

    # ---------------- map ----------------
    def build(self):
        prov = self.prov
        site = self.site()
        pt = site.geometry.iloc[0]
        asp = Sheet.map_aspect()
        width = bt.max_width_inside((pt.x, pt.y), asp)
        frame = SiteFrame.around(pt.x, pt.y, width, asp)
        prov.add("Site anchor: " + SITE_NAME, ov.SOURCE, 1, f"Overture water feature {SITE_ID} (physical / bay); the frame is centred on it and sized to the largest window inside the CUDEM tile coverage ({width / 1000:.0f} km)")
        z, grid, used = bt.topobathy(frame, 60.0)
        zz = np.ma.filled(z, np.nan)
        sea = np.isfinite(zz) & (zz < 0)
        self.log["frame_width_km"] = width / 1000
        self.log["epsg"] = frame.epsg
        self.log["cudem_tiles"] = used
        where = place_context(pt.x, pt.y)
        sheet = Sheet(f"{SITE_NAME}, {where['region']}", "Coastal and sea research site map · seafloor relief, shoreline imagery, model surface currents, sea-surface temperature and marine species records",
                      f"{pt.y:.4f}° N  {abs(pt.x):.4f}° W  ·  frame {width / 1000:.0f} km wide", frame, prov)
        ax = sheet.ax
        sheet.notes += ["Depths are gridded NAVD88 heights, not a nautical chart and not for navigation.", "Model currents and temperature are a forecast-system nowcast, not measurements.",
                        "Species records are presence records; density reflects observer effort as well as abundance.", "The 0 m NAVD88 line is not the shoreline."]

        # ---- seafloor (hillshaded bathymetric ramp) and land (Sentinel-2)
        depth = np.where(sea, -zz, np.nan)
        norm = PowerNorm(0.6, vmin=0, vmax=np.nanpercentile(depth, 99.5))
        rgb = np.ones(zz.shape + (3,))
        hs = hillshade(z, grid.res, azimuth=315, altitude=40, z_factor=6.0)
        sea_rgb = BATHY(norm(np.nan_to_num(depth)))[..., :3]
        rgb[sea] = (sea_rgb * (0.62 + 0.38 * hs[..., None]))[sea]
        picks = s2.best_scene_per_tile(frame, dt.date(2026, 6, 1), dt.date(2026, 10, 20))
        mosaic, mgrid = s2.true_color_mosaic(frame, [p[0] for p in picks], 60.0)
        land = np.isfinite(zz) & (zz >= 0)
        has = mosaic.max(axis=2) > 0
        shade = (0.78 + 0.22 * hs)[..., None]
        rgb[land & has] = (mosaic / 255.0 * shade)[land & has]
        rgb[land & ~has] = np.array([0.92, 0.9, 0.86]) * shade[land & ~has]
        rgb[~np.isfinite(zz)] = 1.0
        ax.imshow(np.clip(rgb, 0, 1), extent=raster_extent(grid), origin="upper", zorder=1, interpolation="bilinear")
        prov.add("Seafloor relief and topography", bt.SOURCE, None, f"{len(used)} tiles ({sum('ncei19' in u for u in used)} at 1/9 and {sum('ncei13' in u for u in used)} at 1/3 arc-second) averaged onto a 60 m grid; hillshade az 315°, 6x vertical exaggeration; depth ramp 0–{np.nanpercentile(depth, 99.5):.0f} m")
        scene_txt = "; ".join(f"{p[0].prefix.split('/')[-1]} (window cloud {p[2]:.1%})" for p in picks)
        prov.add("Land imagery (shown only where CUDEM elevation is at or above 0 m)", s2.SOURCE, len(picks), f"true-colour product, {scene_txt}")
        self.log["imagery_scenes"] = [p[0].prefix.split("/")[-1] for p in picks]

        X = grid.extent[0] + (np.arange(grid.shape[1]) + 0.5) * grid.res
        Y = grid.extent[3] - (np.arange(grid.shape[0]) + 0.5) * grid.res
        cs = ax.contour(X, Y, np.where(sea, depth, np.nan), levels=[d for d in DEPTH_EDGES[1:]], colors="white", linewidths=0.45, alpha=0.75, zorder=3)
        ax.contour(X, Y, np.nan_to_num(zz, nan=-1), levels=[0.0], colors="#111827", linewidths=0.7, zorder=4)
        lab = ax.clabel(cs, levels=[20, 40, 60], fmt="%d m", fontsize=5.4, inline=True, inline_spacing=6)
        for t in lab:
            t.set_color("#F8FAFC")
            t.set_zorder(5)

        # ---- model surface currents (GoMOFS nowcast)
        w, s, e, n = frame.padded_bounds_ll(0.02)
        sf = nos_ofs.read_surface(OFS, (w, s, e, n))
        prov.add(f"{OFS.upper()} surface nowcast", nos_ofs.source(OFS), int(np.isfinite(sf.temp).sum()), f"file {sf.file}; cycle {sf.cycle}; valid {sf.valid_time}; u/v averaged to cell centres and rotated by the grid angle; arrows on a 5 km grid taken from the nearest model cell")
        fx, fy = frame.to_proj.transform(sf.lon.ravel(), sf.lat.ravel())
        ok = np.isfinite(sf.east.ravel()) & np.isfinite(sf.north.ravel()) & (sf.mask.ravel() > 0)
        pts = np.c_[fx, fy]
        tree = cKDTree(pts[ok])
        x0, y0, x1, y1 = frame.extent
        gx, gy = np.meshgrid(np.arange(x0 + 2500, x1, 5000.0), np.arange(y0 + 2500, y1, 5000.0))
        dist, idx = tree.query(np.c_[gx.ravel(), gy.ravel()])
        east, north = sf.east.ravel()[ok][idx], sf.north.ravel()[ok][idx]
        gxf, gyf = gx.ravel(), gy.ravel()
        inside = (dist < 2500) & (sample_grid(z, grid, gxf, gyf).astype(float) < -2)
        scale = 14000.0                                # data metres of arrow per m/s
        ax.quiver(gxf[inside], gyf[inside], east[inside] * scale, north[inside] * scale, angles="xy", scale_units="xy", scale=1, color="#0B1220", width=0.0018, headwidth=4, headlength=5, zorder=8, alpha=0.9)
        speed = np.hypot(east[inside], north[inside])
        self.log["ofs_cycle"] = sf.cycle
        self.log["ofs_valid"] = sf.valid_time
        self.log["current_speed_ms"] = {"min": float(speed.min()), "median": float(np.median(speed)), "max": float(speed.max())}
        seatemp = sf.temp[(sf.mask > 0) & np.isfinite(sf.temp)]
        self.log["ofs_sst_c"] = {"min": float(seatemp.min()), "mean": float(seatemp.mean()), "max": float(seatemp.max())}

        # ---- transect: deepest cell to the nearest land cell (both read from the grid)
        deep = bt.deepest_cell(z)
        land_cell = bt.nearest_land_cell(z, grid, deep)
        dkm, prof, (tx, ty) = bt.transect(z, grid, land_cell, deep)
        ax.plot(tx, ty, color="#FBBF24", lw=1.5, ls=(0, (4, 2)), zorder=9)
        ax_, ay_ = bt.rc_to_xy(grid, land_cell)
        bx_, by_ = bt.rc_to_xy(grid, deep)
        for (px_, py_, ch) in ((ax_, ay_, "A"), (bx_, by_, "B")):
            ax.plot(px_, py_, marker="o", ms=11, mfc="#FBBF24", mec="#111827", mew=0.9, zorder=11)
            ax.text(px_, py_, ch, ha="center", va="center", fontsize=6.5, fontweight="bold", color="#111827", zorder=12)
        self.log["deepest_m"] = float(np.nanmin(zz))
        self.log["transect_km"] = float(dkm[-1])

        # ---- marine occurrences
        occ = self.occurrences(frame, z, grid)
        if len(occ):
            for gname, gcol in [CLASS_GROUPS[c] for c in CLASS_GROUPS] + [OTHER]:
                sub = occ[occ["group"] == gname]
                if len(sub):
                    ax.scatter(sub["x"], sub["y"], s=3.2, c=gcol, linewidths=0, alpha=0.65, zorder=7)
            prov.add("Marine species occurrences", obis.SOURCE, len(occ), "records inside the frame with absence and OBIS-dropped records removed; class from the record; seafloor depth sampled from the CUDEM grid at each point")
        self.log["occurrences"] = int(len(occ))
        self.log["species"] = int(occ["scientificName"].nunique()) if len(occ) else 0
        self.log["datasets"] = int(occ["dataset_id"].nunique()) if len(occ) else 0

        # ---- names from Overture (capes, bays, islands, peninsulas)
        names = []
        wat = overture_layer("water", frame, ["id", "names", "subtype", "class"], prov, "Place names: water features", tag="coastal")
        lnd = overture_layer("land", frame, ["id", "names", "subtype", "class"], prov, "Place names: land features", tag="coastal")
        for df_, classes in ((wat, ("cape", "bay")), (lnd, ("island", "peninsula"))):
            sel = clip_to_frame(df_[df_["name"].notna() & df_["class"].isin(classes)], frame)
            for _, r in sel.iterrows():
                p = r.geometry.representative_point()
                names.append((classes.index(r["class"]) if r["class"] in classes else 9, r["name"], p.x, p.y, r["class"]))
        sheet.graticule()
        sheet.scale_bar()
        sheet.north_arrow()
        lp = LabelPlacer(ax)
        lp.reserve_rect_axes(0.0, 0.0, 0.31, 0.2)
        lp.reserve_rect_axes(0.89, 0.85, 1.0, 1.0)
        lp.reserve_texts([t for t in ax.texts if t.get_text().strip()])
        lp.reserve_point(ax_, ay_, 8)
        lp.reserve_point(bx_, by_, 8)
        shown = 0
        for prio, nm, px_, py_, cls in sorted(set(names)):
            if shown >= 40:
                break
            lp.reserve_point(px_, py_, 3)
            t_ = lp.place(px_, py_, nm, fontsize=6.3, fontweight="bold" if cls in ("cape", "bay", "peninsula") else "normal", style="italic" if cls == "island" else "normal", color="#0B1220")
            if t_ is not None:
                ax.plot(px_, py_, marker="o", ms=2.6, mfc="white", mec="#111827", mew=0.6, zorder=13)
                shown += 1
        self.log["place_names_in_frame"] = int(len({n_[1] for n_ in names}))
        self.log["place_names_labelled"] = shown

        # ---- right column
        sheet.locator(width_deg=8.0, height_in=0.7)
        lg = Legend("Legend")
        lg.header("Seafloor depth (m below NAVD88)")
        lg.ramp("CUDEM, hillshaded", BATHY, 0, float(norm.vmax), "m")
        lg.line(f"Depth contours ({', '.join(str(d) for d in DEPTH_EDGES[1:])} m)", "#7FB3D5", 0.9)
        lg.line("0 m NAVD88", "#111827", 1.0)
        lg.header(f"Overlays · {OFS.upper()} currents, nowcast {sf.valid_time}")
        lg.marker(f"arrow: 0.25 m/s drawn {0.25 * scale / 1000:.1f} km long", "#0B1220", ">", 5)
        lg.marker("A nearest land cell · B deepest cell", "#FBBF24", "o", 6, "#111827")
        if len(occ):
            lg.marker(f"OBIS records, n={len(occ):,} (colours: chart)", "#64748B", "o", 4.5)
        sheet.legend(lg)
        self._charts(sheet, frame, z, grid, dkm, prof, occ, sf)
        sheet.sources_box("Data sources")
        sheet.export(self.key, self.outdir)
        return sheet

    @staticmethod
    def _trim_ticks(axes):
        """Keep only ticks that fall inside the axis limits so no stray tick text sits outside its panel."""
        for a in axes:
            for get, setv, lim in ((a.get_xticks, a.set_xticks, a.get_xlim()), (a.get_yticks, a.set_yticks, a.get_ylim())):
                lo, hi = sorted(lim)
                tk = [v for v in get() if lo - 1e-9 <= v <= hi + 1e-9]
                if get() is not None and len(tk) != len(get()) and len(tk):
                    setv(tk)

    def _charts(self, sheet, frame, z, grid, dkm, prof, occ, sf):
        t = sheet.t
        # --- transect profile
        panel = sheet.panel(0.95, "Seafloor profile A–B")
        a = sheet.chart_axes(panel, (0.13, 0.26, 0.82, 0.32))
        a.fill_between(dkm, prof, min(np.nanmin(prof), -1) - 5, color="#7DBAD6", lw=0)
        a.plot(dkm, prof, color="#0E2A5C", lw=1.0)
        a.axhline(0, color="#111827", lw=0.5)
        a.set_xlabel("distance from A (km)", fontsize=5.8, color=t.muted, labelpad=1)
        a.set_ylabel("elevation (m)", fontsize=5.8, color=t.muted, labelpad=1)
        a.set_xlim(0, dkm[-1])
        a.yaxis.set_major_locator(MaxNLocator(2))
        slope = np.degrees(np.arctan(np.abs(np.gradient(prof, dkm * 1000.0))))
        a.set_title(f"A to B {dkm[-1]:.1f} km · deepest {np.nanmin(prof):.0f} m · max slope {np.nanmax(slope):.1f}° (60 m cells)", fontsize=6.2, color=t.ink, loc="left", pad=3)
        self.log["profile_max_slope_deg"] = float(np.nanmax(slope))
        # --- depth zones and hypsometry
        panel = sheet.panel(1.0, "Depth zones and hypsometry")
        rows = bt.depth_zone_areas(z, grid, COARSE_EDGES)
        a1 = sheet.chart_axes(panel, (0.20, 0.20, 0.28, 0.45))
        labels = [r[0].replace(" m", "") for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        a1.barh(labels, vals, color=[BATHY(PowerNorm(0.6, 0, 150)(d + 1)) for d in list(COARSE_EDGES)][::-1][: len(rows)], height=0.72, edgecolor="#475569", linewidth=0.3)
        for i, v in enumerate(vals):
            a1.text(v, i, f" {v:,.0f}", va="center", fontsize=4.6, color=t.ink)
        a1.set_xlim(0, max(vals) * 1.3)
        a1.set_title("Sea area by depth (km², depth in m)", fontsize=6.2, color=t.ink, loc="left", pad=3)
        a1.tick_params(axis="y", labelsize=4.6)
        d, ar = bt.hypsometric_curve(z, grid)
        a2 = sheet.chart_axes(panel, (0.66, 0.20, 0.28, 0.45))
        a2.plot(ar, d, color="#0E2A5C", lw=1.2)
        a2.invert_yaxis()
        a2.xaxis.set_major_locator(MaxNLocator(3))
        a2.yaxis.set_major_locator(MaxNLocator(3))
        a2.set_xlabel("area deeper (km²)", fontsize=5.6, color=t.muted, labelpad=1)
        a2.set_ylabel("depth (m)", fontsize=5.6, color=t.muted, labelpad=1)
        a2.set_title("Hypsometric curve", fontsize=6.2, color=t.ink, loc="left", pad=3)
        self.log["depth_zone_km2"] = {r[0]: round(r[1], 1) for r in rows}
        self.log["sea_area_km2"] = round(sum(r[1] for r in rows), 1)
        # --- occurrences by group and by seafloor depth zone
        panel = sheet.panel(1.0, "Species records")
        if len(occ):
            a3 = sheet.chart_axes(panel, (0.27, 0.12, 0.22, 0.52))
            c = occ.groupby("group").size().sort_values().tail(5)
            cols = {v[0]: v[1] for v in CLASS_GROUPS.values()}
            a3.barh(c.index, c.values, color=[cols.get(g, OTHER[1]) for g in c.index], height=0.7)
            for i, v in enumerate(c.values):
                a3.text(v, i, f" {v:,}", va="center", fontsize=4.6, color=t.ink)
            a3.set_xscale('log'); a3.set_xlim(1, c.max() * 12); a3.set_xticks([])
            a3.set_title("By class group (log scale)", fontsize=6.2, color=t.ink, loc="left", pad=3)
            a3.tick_params(axis="y", labelsize=4.6)
            sea_occ = occ[occ["floor_m"] < 0]
            bins = list(COARSE_EDGES) + [1e9]
            cut = pd.cut(-sea_occ["floor_m"], bins=bins, right=False)
            cnt = cut.value_counts().sort_index()
            a4 = sheet.chart_axes(panel, (0.70, 0.12, 0.25, 0.52))
            labs = [f"{int(i.left)}–{int(i.right)}" if i.right < 1e8 else f">{int(i.left)}" for i in cnt.index]
            a4.barh(labs[::-1], cnt.values[::-1], color="#4A93BE", height=0.72)
            for i, v in enumerate(cnt.values[::-1]):
                a4.text(v, i, f" {v:,}", va="center", fontsize=4.6, color=t.ink)
            a4.set_xscale('log'); a4.set_xlim(1, cnt.max() * 12); a4.set_xticks([])
            a4.set_title(f"By seafloor depth, m (n={len(sea_occ):,})", fontsize=6.0, color=t.ink, loc="left", pad=3)
            a4.tick_params(axis="y", labelsize=4.6)
            self.log["records_at_sea"] = int(len(sea_occ))
            self.log["records_by_class_group"] = {k: int(v) for k, v in occ.groupby("group").size().items()}
        # --- SST: OISST seasonal comparison | OFS surface temperature distribution
        panel = sheet.panel(1.0, "Sea-surface temperature")
        w, s, e, n = frame.padded_bounds_ll(0.0)
        dates, want, vals = self.sst_series((w, s, e, n))
        a5 = sheet.chart_axes(panel, (0.12, 0.20, 0.38, 0.42))
        cur = [vals.get(d) for d in dates]
        x = np.arange(len(dates))
        clim = []
        for d in dates:
            ys = [vals[k]["sst"] for k in want[d] if vals.get(k)]
            clim.append((np.mean(ys), np.min(ys), np.max(ys)) if ys else (np.nan,) * 3)
        clim = np.array(clim)
        a5.fill_between(x, clim[:, 1], clim[:, 2], color="#BFDBFE", lw=0)
        a5.plot(x, clim[:, 0], color="#1D4ED8", lw=1.0)
        a5.plot(x, [c["sst"] if c else np.nan for c in cur], color="#DC2626", lw=1.2, marker="o", ms=2.5)
        a5.yaxis.set_major_locator(MaxNLocator(3))
        a5.set_xticks(x[::2])
        a5.set_xticklabels([f"{dates[i]:%b}" for i in range(0, len(dates), 2)], fontsize=5.6)
        a5.set_ylabel("°C", fontsize=5.8, color=t.muted, labelpad=1)
        a5.set_title(f"OISST monthly (red) vs 2016– range", fontsize=5.8, color=t.ink, loc="left", pad=3)
        a6 = sheet.chart_axes(panel, (0.64, 0.20, 0.30, 0.42))
        tmp = sf.temp[(sf.mask > 0) & np.isfinite(sf.temp)]
        a6.hist(tmp, bins=18, color="#F59E0B", edgecolor="white", linewidth=0.3)
        a6.set_xlabel("°C", fontsize=5.8, color=t.muted, labelpad=1)
        a6.set_ylabel("model cells", fontsize=5.6, color=t.muted, labelpad=1)
        a6.set_title(f"{OFS.upper()} surface temp", fontsize=5.8, color=t.ink, loc="left", pad=3)
        self.log["oisst_current_year"] = [None if c is None else round(c["sst"], 2) for c in cur]
        self.log["oisst_dates"] = [d.isoformat() for d in dates]
        self._trim_ticks([ax_ for ax_ in sheet.fig.axes if ax_.get_position().x0 > 0.68 and ax_.get_xlabel() + ax_.get_ylabel() + ax_.get_title("left") != ""])
        prov = sheet.prov
        prov.add("Sea-surface temperature series (OISST)", oisst.SOURCE, len([c for c in cur if c]), "mean of valid 0.25° ocean cells inside the frame; 15th of each month; comparison range from the same calendar day in 2016 onwards; "
                 + ("preliminary files used for " + ", ".join(f"{d:%b %Y}" for d, c in zip(dates, cur) if c and c["kind"] == "preliminary") if any(c and c["kind"] == "preliminary" for c in cur) else "final files only"))
