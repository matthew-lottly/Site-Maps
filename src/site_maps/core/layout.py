"""Research-sheet layout: title block, graticule-labelled map frame, scale bar, true-north arrow, locator, legend, sources, notes.

One `Sheet` per map. Domain code draws on `sheet.ax` in projected metres and registers every layer in `sheet.prov`.
"""
from __future__ import annotations

import datetime as dt
import math
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from shapely.geometry import box

from .. import __version__
from .cache import download
from .frame import SiteFrame
from .provenance import ProvenanceLog, Source

W_IN, H_IN = 17.0, 11.0


@dataclass
class Theme:
    paper: str = "#FFFFFF"
    ink: str = "#1F2933"
    muted: str = "#52606D"
    faint: str = "#9AA5B1"
    rule: str = "#CBD2D9"
    panel: str = "#F5F7FA"
    accent: str = "#B42318"
    water: str = "#CFE3EE"
    land: str = "#F3F1EA"
    fonts: tuple = ("Inter", "Helvetica Neue", "Arial", "DejaVu Sans")

    def font(self) -> str:
        have = {f.name for f in font_manager.fontManager.ttflist}
        return next((f for f in self.fonts if f in have), "DejaVu Sans")


def halo(w: float = 2.5, color: str = "white"):
    return [pe.withStroke(linewidth=w, foreground=color)]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


@dataclass
class Legend:
    """Collects legend rows; items flow into two columns under each header, so `height_in` is known before the panel is placed."""

    title: str = "Legend"
    rows: list = field(default_factory=list)
    cols: int = 2

    def header(self, text):
        self.rows.append(("header", text))
        return self

    def patch(self, label, color, hatch=None, edge=None):
        self.rows.append(("patch", label, color, hatch, edge))
        return self

    def line(self, label, color, lw=1.2, ls="-"):
        self.rows.append(("line", label, color, lw, ls))
        return self

    def marker(self, label, color, marker="o", size=6, edge="white"):
        self.rows.append(("marker", label, color, marker, size, edge))
        return self

    def ramp(self, label, cmap, vmin, vmax, unit=""):
        self.rows.append(("ramp", label, cmap, vmin, vmax, unit))
        return self

    def sections(self):
        out, cur = [], [None, []]
        for r in self.rows:
            if r[0] == "header":
                if cur[0] is not None or cur[1]:
                    out.append(cur)
                cur = [r[1], []]
            else:
                cur[1].append(r)
        out.append(cur)
        return out

    @property
    def height_in(self) -> float:
        h = 0.26 + 0.08
        for head, items in self.sections():
            if head:
                h += 0.2
            plain = [i for i in items if i[0] != "ramp"]
            h += -(-len(plain) // self.cols) * 0.18 + sum(0.55 for i in items if i[0] == "ramp") + 0.05
        return h


class Sheet:
    MAP = (0.03, 0.098, 0.636, 0.786)           # figure fractions: left, bottom, width, height
    COL_X, COL_W = 0.683, 0.295

    def __init__(self, title: str, subtitle: str, site_line: str, frame: SiteFrame, prov: ProvenanceLog | None = None, theme: Theme | None = None):
        self.t = theme or Theme()
        matplotlib.rcParams["font.family"] = self.t.font()
        matplotlib.rcParams["text.parse_math"] = False
        matplotlib.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        self.title, self.subtitle, self.site_line, self.frame = title, subtitle, site_line, frame
        self.prov = prov or ProvenanceLog()
        self.fig = plt.figure(figsize=(W_IN, H_IN), facecolor=self.t.paper)
        self.notes: list[str] = []
        self._y = 0.905                           # top of the next right-column panel (figure fraction)
        self._title_block()
        self.ax = self._map_axes()
        self.panel_axes: list[plt.Axes] = []

    @classmethod
    def map_aspect(cls) -> float:
        return (cls.MAP[2] * W_IN) / (cls.MAP[3] * H_IN)

    # ---------- chrome ----------
    def _title_block(self):
        f, t = self.fig, self.t
        a = f.add_axes((0, 0.915, 1, 0.085))
        a.set_axis_off()
        a.add_patch(Rectangle((0, 0), 1, 1, fc=t.ink, transform=a.transAxes))
        a.text(0.022, 0.62, self.title, color="white", fontsize=21, fontweight="bold", va="center", transform=a.transAxes)
        a.text(0.022, 0.24, self.subtitle, color="#CBD2D9", fontsize=10.5, va="center", transform=a.transAxes)
        a.text(0.978, 0.62, self.site_line, color="white", fontsize=9.5, va="center", ha="right", transform=a.transAxes)
        a.text(0.978, 0.24, f"{self.frame.crs.name} ({self.frame.epsg})", color="#CBD2D9", fontsize=8.5, va="center", ha="right", transform=a.transAxes)

    def _map_axes(self):
        ax = self.fig.add_axes(self.MAP)
        x0, y0, x1, y1 = self.frame.extent
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(self.t.land)
        for s in ax.spines.values():
            s.set_edgecolor(self.t.ink)
            s.set_linewidth(1.1)
        return ax

    # ---------- map furniture ----------
    @staticmethod
    def _nice_step(span_deg: float) -> float:
        for s in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10):
            if span_deg / s <= 6.5:
                return s
        return 10

    def graticule(self, labels: bool = True, lines: bool = True):
        """Lat/lon graticule with labels on all four borders (decimal degrees, WGS 84)."""
        ax, fr = self.ax, self.frame
        x0, y0, x1, y1 = fr.extent
        w, s, e, n = fr.bounds_ll
        step = self._nice_step(max(e - w, n - s))
        lons = np.arange(math.ceil(w / step) * step, e, step)
        lats = np.arange(math.ceil(s / step) * step, n, step)
        dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        t = self.t
        for lo in lons:
            la = np.linspace(s - 0.2 * (n - s), n + 0.2 * (n - s), 200)
            X, Y = fr.to_proj.transform(np.full_like(la, lo), la)
            if lines:
                ax.plot(X, Y, color="#7B8794", lw=0.4, ls=(0, (1, 3)), zorder=40)
            if labels:
                for yy, side in ((y0, 0), (y1, 1)):
                    xi = np.interp(yy, Y, X)
                    fx = (xi - x0) / (x1 - x0)
                    if 0.02 < fx < 0.98:
                        ax.text(fx, -0.012 if side == 0 else 1.012, f"{abs(lo):.{dec}f}°{'W' if lo < 0 else 'E'}", transform=ax.transAxes, ha="center", va="top" if side == 0 else "bottom",
                                fontsize=7, color=t.muted, clip_on=False)
        for la in lats:
            lo = np.linspace(w - 0.2 * (e - w), e + 0.2 * (e - w), 200)
            X, Y = fr.to_proj.transform(lo, np.full_like(lo, la))
            if lines:
                ax.plot(X, Y, color="#7B8794", lw=0.4, ls=(0, (1, 3)), zorder=40)
            if labels:
                order = np.argsort(X)
                for xx, side in ((x0, 0), (x1, 1)):
                    yi = np.interp(xx, X[order], Y[order])
                    fy = (yi - y0) / (y1 - y0)
                    if 0.02 < fy < 0.98:
                        ax.text(-0.008 if side == 0 else 1.008, fy, f"{abs(la):.{dec}f}°{'S' if la < 0 else 'N'}", transform=ax.transAxes, ha="right" if side == 0 else "left", va="center",
                                fontsize=7, color=t.muted, rotation=90 if side == 0 else -90, rotation_mode="anchor", clip_on=False)

    def scale_bar(self, corner: str = "bl"):
        ax, fr = self.ax, self.frame
        x0, y0, x1, y1 = fr.extent
        W, H = x1 - x0, y1 - y0
        target = W * 0.22
        nice = [1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000, 100000]
        L = max(n for n in nice if n <= target)
        n = 4 if L not in (2.5, 25, 250, 2500, 25000) else 5
        seg = L / n
        unit, div = ("km", 1000.0) if L >= 1000 else ("m", 1.0)
        bx, by = x0 + W * 0.03, y0 + H * 0.045
        h = H * 0.009
        ax.add_patch(Rectangle((bx - W * 0.012, by - H * 0.028), L + W * 0.075, H * 0.085, fc="white", alpha=0.88, ec="none", zorder=60))
        for i in range(n):
            ax.add_patch(Rectangle((bx + i * seg, by), seg, h, fc=self.t.ink if i % 2 == 0 else "white", ec=self.t.ink, lw=0.6, zorder=61))
        for i in range(n + 1):
            ax.text(bx + i * seg, by + h * 1.8, f"{i * seg / div:g}", fontsize=6.5, ha="center", va="bottom", color=self.t.ink, zorder=62)
        ax.text(bx + L + W * 0.008, by + h / 2, unit, fontsize=6.5, va="center", color=self.t.ink, zorder=62)
        ax.text(bx, by - H * 0.006, f"1:{fr.scale_denominator(self.MAP[2] * W_IN):,.0f} at {W_IN:.0f}×{H_IN:.0f} in", fontsize=5.8, va="top", color=self.t.muted, zorder=62)

    def north_arrow(self):
        """Arrow points to TRUE north; the sheet's grid north is rotated by the meridian convergence at the map centre."""
        ax, fr = self.ax, self.frame
        x0, y0, x1, y1 = fr.extent
        W, H = x1 - x0, y1 - y0
        bw, bh = W * 0.07, H * 0.115
        bx, by = x1 - bw - W * 0.012, y1 - bh - H * 0.015
        ax.add_patch(Rectangle((bx, by), bw, bh, fc="white", alpha=0.9, ec=self.t.rule, lw=0.5, zorder=60))
        cx, cy, r = bx + bw / 2, by + bh * 0.50, bh * 0.24
        ang = math.radians(fr.convergence_deg())
        rot = lambda dx, dy: (cx + dx * math.cos(ang) - dy * math.sin(ang), cy + dx * math.sin(ang) + dy * math.cos(ang))
        ax.add_patch(MplPolygon([rot(0, r * 1.2), rot(-r * 0.5, -r * 0.8), rot(0, -r * 0.3)], closed=True, fc=self.t.ink, ec=self.t.ink, lw=0.6, zorder=61))
        ax.add_patch(MplPolygon([rot(0, r * 1.2), rot(r * 0.5, -r * 0.8), rot(0, -r * 0.3)], closed=True, fc="white", ec=self.t.ink, lw=0.8, zorder=61))
        ax.text(*rot(0, r * 1.38), "N", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=self.t.ink, zorder=62)
        ax.text(bx + bw / 2, by + bh * 0.07, f"grid N {abs(fr.convergence_deg()):.1f}° {'E' if fr.convergence_deg() > 0 else 'W'} of true N", fontsize=4.6, ha="center", va="bottom", color=self.t.muted, zorder=62)

    def finish_map(self):
        """Drop axis titles/ticks that plotting helpers add; the graticule carries the coordinates."""
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        x0, y0, x1, y1 = self.frame.extent
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def remaining_in(self) -> float:
        return (self._y - 0.082) * H_IN

    # ---------- right column ----------
    def panel(self, height_in: float, title: str | None = None, fill: bool = True) -> plt.Axes:
        h = height_in / H_IN
        ax = self.fig.add_axes((self.COL_X, self._y - h, self.COL_W, h))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        if fill:
            ax.add_patch(Rectangle((0, 0), 1, 1, fc=self.t.panel, ec=self.t.rule, lw=0.8, transform=ax.transAxes))
        if title:
            hh = 0.26 / height_in
            ax.add_patch(Rectangle((0, 1 - hh), 1, hh, fc=self.t.ink, ec="none", transform=ax.transAxes))
            ax.text(0.03, 1 - hh / 2, title.upper(), color="white", fontsize=7.2, fontweight="bold", va="center", transform=ax.transAxes)
        self._y -= h + 0.012
        self.panel_axes.append(ax)
        return ax

    def chart_axes(self, panel: plt.Axes, rect=(0.12, 0.16, 0.82, 0.62)) -> plt.Axes:
        pb = panel.get_position()
        ax = self.fig.add_axes((pb.x0 + rect[0] * pb.width, pb.y0 + rect[1] * pb.height, rect[2] * pb.width, rect[3] * pb.height))
        ax.tick_params(labelsize=6.5, length=2, colors=self.t.muted)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(self.t.faint)
        ax.set_facecolor("none")
        return ax

    def locator(self, width_deg: float = 9.0, height_in: float = 1.5):
        """State/province locator from Natural Earth admin-1 (1:50m) with the map window outlined."""
        src = Source("NE", "Natural Earth", "Admin 1 – States, Provinces 1:50m", "GitHub master at retrieval", "https://www.naturalearthdata.com/", "Public domain")
        path = download("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson", "naturalearth/admin1_50m.geojson")
        adm = gpd.read_file(path)
        cx, cy = self.frame.center
        pan = self.panel(height_in, "Locator")
        pb = pan.get_position()
        ax = self.fig.add_axes((pb.x0 + 0.006, pb.y0 + 0.008, pb.width - 0.012, pb.height - 0.26 / H_IN - 0.012))
        ratio = ((pb.width - 0.012) * W_IN) / ((pb.height - 0.26 / H_IN - 0.012) * H_IN)
        hh = width_deg / ratio / math.cos(math.radians(cy))
        ax.set_xlim(cx - width_deg / 2, cx + width_deg / 2)
        ax.set_ylim(cy - hh / 2, cy + hh / 2)
        ax.set_aspect(1 / math.cos(math.radians(cy)))
        ax.set_facecolor(self.t.water)
        adm.plot(ax=ax, fc=self.t.land, ec=self.t.faint, lw=0.5)
        for _, r in adm.iterrows():
            p = r.geometry.representative_point()
            if abs(p.x - cx) < width_deg * 0.42 and abs(p.y - cy) < hh * 0.42 and r.get("name"):
                ax.text(p.x, p.y, str(r["name"]).upper(), fontsize=5.0, color=self.t.faint, ha="center", va="center", zorder=5)
        w, s, e, n = self.frame.bounds_ll
        ax.add_patch(Rectangle((w, s), max(e - w, 0.05), max(n - s, 0.05), fc="none", ec=self.t.accent, lw=1.5, zorder=10))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xlim(cx - width_deg / 2, cx + width_deg / 2); ax.set_ylim(cy - hh / 2, cy + hh / 2)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.t.rule)
        self.prov.add("Locator: state boundaries", src, len(adm), "1:50m admin-1 polygons, unmodified")
        return ax

    def legend(self, lg: Legend):
        H = lg.height_in
        ax = self.panel(H, lg.title)
        t = self.t
        y = H - 0.26 - 0.08                                   # inches from the panel bottom
        colw = 1.0 / lg.cols
        for head, items in lg.sections():
            if head:
                ax.text(0.03, (y - 0.09) / H, head, fontsize=6.7, fontweight="bold", color=t.ink, va="center", transform=ax.transAxes)
                y -= 0.2
            plain = [i for i in items if i[0] != "ramp"]
            for k, r in enumerate(plain):
                col, row = k // -(-len(plain) // lg.cols) if lg.cols > 1 else 0, k % -(-len(plain) // lg.cols) if lg.cols > 1 else k
                x0 = 0.03 + col * colw
                yy = (y - 0.09 - row * 0.18) / H
                kind = r[0]
                if kind == "patch":
                    ax.add_patch(Rectangle((x0, yy - 0.055 / H), 0.07, 0.11 / H, fc=r[2], ec=r[4] or "#6B7280", lw=0.4, hatch=r[3], transform=ax.transAxes))
                elif kind == "line":
                    ax.plot([x0, x0 + 0.07], [yy, yy], color=r[2], lw=r[3], ls=r[4], transform=ax.transAxes, solid_capstyle="butt")
                else:
                    ax.plot([x0 + 0.035], [yy], marker=r[3], ms=r[4], color=r[2], mec=r[5], mew=0.6, ls="none", transform=ax.transAxes)
                ax.text(x0 + 0.095, yy, r[1], fontsize=6.3, color=t.ink, va="center", transform=ax.transAxes)
            if plain:
                y -= -(-len(plain) // lg.cols) * 0.18
            for r in [i for i in items if i[0] == "ramp"]:
                ib = ax.get_position()
                ia = self.fig.add_axes((ib.x0 + 0.03 * ib.width, ib.y0 + (y - 0.42) / H * ib.height, 0.55 * ib.width, 0.11 / H_IN))
                ia.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=r[2], extent=(r[3], r[4], 0, 1))
                ia.set_yticks([]); ia.tick_params(labelsize=5.6, length=2, colors=t.muted)
                for sp in ia.spines.values():
                    sp.set_linewidth(0.4)
                ax.text(0.03, (y - 0.08) / H, r[1], fontsize=6.3, color=t.ink, va="center", transform=ax.transAxes)
                ax.text(0.62, (y - 0.33) / H, r[5], fontsize=6.0, color=t.muted, va="center", transform=ax.transAxes)
                y -= 0.55
            y -= 0.05
        return ax

    def note_box(self, title: str, lines: list[str], height_in: float | None = None, width_chars: int = 74, fontsize: float = 6.0):
        wrapped = [textwrap.fill(s, width_chars, subsequent_indent="   ") for s in lines]
        n = sum(w.count("\n") + 1 for w in wrapped)
        h = height_in or (0.26 + 0.1 + n * 0.108 + 0.04 * len(wrapped))
        ax = self.panel(h, title)
        y = 1 - (0.26 + 0.09) / h
        for w in wrapped:
            ax.text(0.03, y, w, fontsize=fontsize, color=self.t.ink, va="top", transform=ax.transAxes, linespacing=1.25)
            y -= ((w.count("\n") + 1) * 0.108 + 0.04) / h
        return ax

    def sources_box(self, title: str = "Data sources", note: str = ""):
        lines = []
        for s_ in self.prov.sources():
            lines.append(f"[{s_.key}] {s_.publisher}: {s_.dataset} ({s_.version}); {s_.license}.")
        lines += [note] if note else []
        avail = self.remaining_in()
        for fs, chars in ((5.6, 107), (5.2, 115), (4.9, 123), (4.6, 131)):
            wrapped = [textwrap.fill(x, chars, subsequent_indent="   ") for x in lines]
            n = sum(w.count("\n") + 1 for w in wrapped)
            need = 0.26 + 0.1 + n * fs * 0.0172 + 0.03 * len(wrapped)
            if need <= avail:
                break
        ax = self.panel(min(need, avail), title)
        y = 1 - (0.26 + 0.08) / min(need, avail)
        for w in wrapped:
            ax.text(0.03, y, w, fontsize=fs, color=self.t.ink, va="top", transform=ax.transAxes, linespacing=1.22)
            y -= ((w.count("\n") + 1) * fs * 0.0172 + 0.03) / min(need, avail)
        return ax

    def footer(self):
        f, t = self.fig, self.t
        a = f.add_axes((0, 0, 1, 0.075))
        a.set_axis_off()
        a.plot([0.022, 0.978], [0.99, 0.99], color=t.rule, lw=0.6, transform=a.transAxes)
        a.text(0.022, 0.86, f"Prepared {dt.date.today().isoformat()} · site-maps {__version__} ({git_rev()}) · data retrieved {self.prov.retrieved}", fontsize=6.5, color=t.muted, va="top", transform=a.transAxes)
        a.text(0.022, 0.64, "Every layer on this sheet is drawn from the public datasets listed in Data sources; nothing is simulated or hand-digitised. For orientation and research planning, not for survey, navigation or legal boundaries.",
               fontsize=6.0, color=t.muted, va="top", transform=a.transAxes)
        if self.notes:
            a.text(0.022, 0.43, textwrap.fill("Limits: " + "  ·  ".join(self.notes), 215), fontsize=5.6, color=t.muted, va="top", transform=a.transAxes, linespacing=1.2)
        a.text(0.978, 0.42, f"{self.frame.crs.name} ({self.frame.epsg}) · graticule WGS 84", fontsize=6.0, color=t.muted, va="top", ha="right", transform=a.transAxes)

    # ---------- output ----------
    def export(self, stem: str, outdir: Path, dpi: int = 200) -> dict[str, Path]:
        outdir.mkdir(parents=True, exist_ok=True)
        self.finish_map()
        self.footer()
        out = {k: outdir / f"{stem}.{k}" for k in ("png", "pdf")}
        self.fig.savefig(out["png"], dpi=dpi, facecolor=self.t.paper)
        self.fig.savefig(out["pdf"], facecolor=self.t.paper)
        (outdir / f"{stem}.provenance.json").write_text(self.prov.to_json())
        (outdir / f"{stem}.sources.md").write_text(self.prov.to_markdown())
        out["issues"] = self.audit()
        plt.close(self.fig)
        return out

    def audit(self) -> list[str]:
        from .qa import audit
        return audit(self.fig)
