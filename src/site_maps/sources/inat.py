"""iNaturalist Open Data (AWS): stream the 13 GB observations export once and keep only the rows inside the requested boxes.

Only observations whose licence permits open redistribution are in this export. Keep the columns the maps need: location,
positional accuracy (metres), taxon, quality grade (research = community-confirmed ID) and date.
"""
from __future__ import annotations

import gzip
import sys

import pandas as pd
import requests

from ..core.cache import cache_path, download
from ..core.provenance import Source

BASE = "https://inaturalist-open-data.s3.amazonaws.com"
SOURCE = Source("iNat", "iNaturalist", "iNaturalist Open Data (observations, taxa)", "export dated 2026-09-27",
                "https://registry.opendata.aws/inaturalist-open-data/", "per-record CC0 / CC-BY / CC-BY-NC")


def extract(boxes: dict[str, tuple[float, float, float, float]], chunk_rows: int = 2_000_000, log=print) -> dict[str, str]:
    """`boxes` maps a name to (W, S, E, N). Writes data/cache/inat/<name>.csv for each box that does not exist yet."""
    todo = {k: v for k, v in boxes.items() if not cache_path(f"inat/{k}.csv").exists()}
    if not todo:
        return {k: str(cache_path(f"inat/{k}.csv")) for k in boxes}
    parts = {k: [] for k in todo}
    seen = 0
    with requests.get(f"{BASE}/observations.csv.gz", stream=True, timeout=600) as r:
        r.raise_for_status()
        gz = gzip.GzipFile(fileobj=r.raw)
        for ch in pd.read_csv(gz, sep="\t", chunksize=chunk_rows, usecols=["observation_uuid", "latitude", "longitude", "positional_accuracy", "taxon_id", "quality_grade", "observed_on"],
                              dtype={"observation_uuid": str, "quality_grade": str, "observed_on": str}, low_memory=False):
            seen += len(ch)
            for k, (W, S, E, N) in todo.items():
                m = (ch["longitude"] >= W) & (ch["longitude"] <= E) & (ch["latitude"] >= S) & (ch["latitude"] <= N)
                if m.any():
                    parts[k].append(ch[m])
            log(f"scanned {seen:,} rows; kept " + ", ".join(f"{k}={sum(len(p) for p in v):,}" for k, v in parts.items()))
    for k, v in parts.items():
        pd.concat(v).to_csv(cache_path(f"inat/{k}.csv"), index=False)
    return {k: str(cache_path(f"inat/{k}.csv")) for k in boxes}


GROUPS = {"Aves": "Birds", "Reptilia": "Reptiles", "Amphibia": "Amphibians", "Mammalia": "Mammals", "Actinopterygii": "Fishes", "Insecta": "Insects", "Arachnida": "Arachnids",
          "Plantae": "Plants", "Fungi": "Fungi", "Mollusca": "Molluscs"}
GROUP_COLORS = {"Birds": "#1D4ED8", "Reptiles": "#B45309", "Amphibians": "#0E7490", "Mammals": "#7C2D12", "Fishes": "#0369A1", "Insects": "#C026D3", "Arachnids": "#9D174D",
                "Plants": "#15803D", "Fungi": "#6D28D9", "Molluscs": "#475569", "Other": "#64748B"}


def add_groups(o: pd.DataFrame) -> pd.DataFrame:
    """Add `group` (broad taxon group from the iNaturalist taxonomy) and `species_name` to an observation table."""
    t = taxa()
    ids = {name: int(t[(t["name"] == name) & (t["active"]) & (t["rank"].isin(["class", "kingdom", "phylum"]))]["taxon_id"].iloc[0]) for name in GROUPS if ((t["name"] == name) & t["active"]).any()}
    anc = t.set_index("taxon_id")["ancestry"].to_dict()
    id2group = {v: GROUPS[k] for k, v in ids.items()}

    def group(tid):
        chain = [int(x) for x in str(anc.get(tid, "")).split("/") if x.isdigit()] + [tid]
        for c in reversed(chain):
            if c in id2group:
                return id2group[c]
        return "Other"

    o = o.copy()
    o["group"] = o["taxon_id"].map(group)
    o["species_name"] = o["taxon_id"].map(t.set_index("taxon_id")["name"].to_dict())
    return o


def taxa() -> pd.DataFrame:
    p = download(f"{BASE}/taxa.csv.gz", "inat/taxa.csv.gz")
    gz = open(p, "rb").read(2) == b"\x1f\x8b"                  # the HTTP layer may already have undone the gzip
    return pd.read_csv(p, sep="\t", dtype={"ancestry": str}, compression="gzip" if gz else None)


if __name__ == "__main__":
    BOXES = {"brazos_bend": (-95.72, 29.28, -95.48, 29.47), "yosemite_valley": (-119.80, 37.65, -119.40, 37.85), "monterey_bay": (-122.40, 36.30, -121.50, 37.10)}
    extract(BOXES, log=lambda s: print(s, flush=True))
