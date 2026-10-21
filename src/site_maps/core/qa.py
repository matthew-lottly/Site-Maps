"""Automatic layout audit: finds text that overlaps other text or spills off the page / out of its panel before a map is exported."""
from __future__ import annotations

import itertools

from matplotlib.text import Annotation, Text


def audit(fig, overlap: float = 0.12, tol_px: float = 2.0) -> list[str]:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    skip = set()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            if not a.axison:
                skip.update(id(x) for x in axis.get_ticklabels(which="both"))
            skip.add(id(axis.get_offset_text()))
    items = []
    for t in fig.findobj(Text):
        if id(t) in skip:
            continue
        s = t.get_text().strip()
        if not s or not t.get_visible() or t.get_alpha() == 0:
            continue
        ax = t.axes
        if ax is not None and (t is ax.title or t is ax._left_title or t is ax._right_title or t is ax.xaxis.label or t is ax.yaxis.label):
            continue
        if ax is not None and any(t in axis.get_ticklabels() for axis in (ax.xaxis, ax.yaxis)):
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((t, bb, s.replace("\n", " ")[:38]))
    issues = []
    for (t1, b1, s1), (t2, b2, s2) in itertools.combinations(items, 2):
        x0, x1 = max(b1.x0, b2.x0), min(b1.x1, b2.x1)
        y0, y1 = max(b1.y0, b2.y0), min(b1.y1, b2.y1)
        if x1 <= x0 or y1 <= y0:
            continue
        inter = (x1 - x0) * (y1 - y0)
        if inter / min(b1.width * b1.height, b2.width * b2.height) > overlap:
            issues.append(f"overlap: '{s1}' x '{s2}'")
    for t, bb, s in items:
        if bb.x0 < -tol_px or bb.y0 < -tol_px or bb.x1 > W + tol_px or bb.y1 > H + tol_px:
            issues.append(f"off page: '{s}'")
            continue
        ax = t.axes
        if ax is not None and not ax.axison:                       # panel axes: text must stay inside the panel
            ab = ax.get_window_extent(r)
            if bb.x1 > ab.x1 + tol_px or bb.x0 < ab.x0 - tol_px or bb.y1 > ab.y1 + tol_px or bb.y0 < ab.y0 - tol_px:
                if ab.width > 20 and ab.height > 20:
                    issues.append(f"outside its panel: '{s}'")
    return issues
