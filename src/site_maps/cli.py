"""Command line: python -m site_maps <map> [--outdir PATH]"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .core.cache import OUTPUTS

MAPS = {"environmental": ("site_maps.maps.environmental", "EnvironmentalSiteMap"),
        "real-estate": ("site_maps.maps.real_estate", "RealEstateSiteMap"),
        "utilities": ("site_maps.maps.utilities", "UtilitiesSiteMap"),
        "camping": ("site_maps.maps.camping", "CampingSiteMap"),
        "coastal": ("site_maps.maps.coastal", "CoastalSiteMap")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="site_maps", description="Research-grade site maps from cited public data")
    p.add_argument("map", choices=[*MAPS, "all"])
    p.add_argument("--outdir", default=str(OUTPUTS))
    a = p.parse_args(argv)
    for key in (MAPS if a.map == "all" else [a.map]):
        try:
            mod, cls = MAPS[key]
            m = getattr(importlib.import_module(mod), cls)(outdir=Path(a.outdir))
        except ModuleNotFoundError:
            print(f"{key}: not built yet")
            continue
        sheet = m.build()
        print(f"✔ {key}: {a.outdir}/{m.key}.png")
        for i in getattr(sheet, "last_issues", []) or []:
            print("   layout:", i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
