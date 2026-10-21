"""Collision-aware map labels: try several offsets around a point and keep the first that does not overlap earlier labels, markers or the frame."""
from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.text import Text

#          dx,  dy (points)   ha       va
OFFSETS = [(6, 4, "left", "bottom"), (6, -4, "left", "top"), (-6, 4, "right", "bottom"), (-6, -4, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
           (10, 0, "left", "center"), (-10, 0, "right", "center")]


class LabelPlacer:
    def __init__(self, ax, pad_px: float = 2.0):
        self.ax, self.fig, self.pad = ax, ax.figure, pad_px
        self.boxes: list[tuple[float, float, float, float]] = []

    def _renderer(self):
        self.fig.canvas.draw()
        return self.fig.canvas.get_renderer()

    def reserve_texts(self, texts=None):
        r = self._renderer()
        for t in texts if texts is not None else [t for t in self.ax.texts if t.get_text().strip()]:
            b = t.get_window_extent(r)
            self.boxes.append((b.x0, b.y0, b.x1, b.y1))

    def reserve_point(self, x: float, y: float, radius_pts: float = 4.0):
        """Keep labels off a marker drawn at data coords (x, y)."""
        px, py = self.ax.transData.transform((x, y))
        r = radius_pts * self.fig.dpi / 72.0
        self.boxes.append((px - r, py - r, px + r, py + r))

    def reserve_rect_axes(self, x0, y0, x1, y1):
        """Reserve a rectangle given in axes fractions (e.g. the legend-like furniture inside the map)."""
        (a, b), (c, d) = self.ax.transAxes.transform((x0, y0)), self.ax.transAxes.transform((x1, y1))
        self.boxes.append((a, b, c, d))

    def _free(self, bb) -> bool:
        a = self.ax.get_window_extent(self._renderer())
        if bb.x0 < a.x0 + 2 or bb.x1 > a.x1 - 2 or bb.y0 < a.y0 + 2 or bb.y1 > a.y1 - 2:
            return False
        for x0, y0, x1, y1 in self.boxes:
            if bb.x0 < x1 + self.pad and bb.x1 > x0 - self.pad and bb.y0 < y1 + self.pad and bb.y1 > y0 - self.pad:
                return False
        return True

    def place(self, x: float, y: float, text: str, halo_w: float = 2.4, halo_color: str = "white", offsets=OFFSETS, **kw) -> Text | None:
        kw.setdefault("fontsize", 6.5)
        kw.setdefault("zorder", 50)
        r = self._renderer()
        for dx, dy, ha, va in offsets:
            t = self.ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, path_effects=[pe.withStroke(linewidth=halo_w, foreground=halo_color)], **kw)
            bb = t.get_window_extent(r)
            if self._free(bb):
                self.boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
                return t
            t.remove()
        return None

    def place_centered(self, x: float, y: float, text: str, halo_w: float = 2.2, halo_color: str = "white", rotation: float = 0.0, **kw) -> Text | None:
        """Label centred on (x, y) (areas and lines); skipped if it would collide."""
        kw.setdefault("fontsize", 6.5)
        kw.setdefault("zorder", 50)
        r = self._renderer()
        t = self.ax.text(x, y, text, ha="center", va="center", rotation=rotation, rotation_mode="anchor", path_effects=[pe.withStroke(linewidth=halo_w, foreground=halo_color)], **kw)
        bb = t.get_window_extent(r)
        if self._free(bb):
            self.boxes.append((bb.x0, bb.y0, bb.x1, bb.y1))
            return t
        t.remove()
        return None
