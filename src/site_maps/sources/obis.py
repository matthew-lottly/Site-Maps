"""OBIS Open Data (AWS, GeoParquet): marine species occurrences inside a box.

The export is 7,100+ files (one per dataset) with no spatial index, so this module reads each file's parquet footer once (threaded),
caches the latitude/longitude row-group ranges, and then reads only the row groups that can intersect the requested box.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from pyarrow import fs

from ..core.cache import cache_path, s3_listing
from ..core.provenance import Source

BUCKET = "obis-open-data"
BASE = f"https://{BUCKET}.s3.amazonaws.com"
SOURCE = Source("OBIS", "Ocean Biodiversity Information System (UNESCO-IOC)", "OBIS Occurrence Data (GeoParquet export)", "export of 25 March 2025 (data to 2025-12-31 per STAC item)",
                "https://doi.org/10.25607/obis.occurrence.b89117cd", "CC BY-NC 4.0 (per STAC item); records keep their data-provider licences",
                "Records flagged dropped or absence are excluded")
LAT, LON = "interpreted.decimalLatitude", "interpreted.decimalLongitude"
FIELDS = ["interpreted.scientificName", "interpreted.class", "interpreted.phylum", "interpreted.minimumDepthInMeters", "interpreted.maximumDepthInMeters",
          "interpreted.eventDate", "interpreted.individualCount", "interpreted.basisOfRecord", "dataset_id", "dropped", "absence"]


def _fs():
    return fs.S3FileSystem(anonymous=True, region="us-east-1", proxy_options=os.environ.get("HTTPS_PROXY"))


def _ranges(key: str, filesystem) -> list:
    try:
        pf = pq.ParquetFile(f"{BUCKET}/{key}", filesystem=filesystem)
        md = pf.metadata
        names = [md.schema.column(i).path for i in range(md.num_columns)]
        ia, io = names.index(LAT), names.index(LON)
        out = []
        for rg in range(md.num_row_groups):
            a, o = md.row_group(rg).column(ia).statistics, md.row_group(rg).column(io).statistics
            if a is None or o is None or not a.has_min_max or not o.has_min_max:
                out.append([rg, -90, 90, -180, 180])
            else:
                out.append([rg, a.min, a.max, o.min, o.max])
        return [key, out]
    except Exception as e:                      # a file we cannot read is reported, not silently dropped
        return [key, None, str(e)[:120]]


def footer_index(workers: int = 24, log=print) -> dict:
    p = cache_path("obis/footer_index.json")
    if p.exists():
        return json.loads(p.read_text())
    keys = [k for k in s3_listing(BASE, "occurrence/", None) if k.endswith(".parquet")]
    filesystem = _fs()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda k: _ranges(k, filesystem), keys))
    idx = {r[0]: r[1:] for r in res}
    p.write_text(json.dumps(idx))
    log(f"indexed {len(idx)} files; unreadable: {sum(1 for v in idx.values() if v[0] is None)}")
    return idx


def select_row_groups(idx: dict, box: tuple[float, float, float, float]) -> list[tuple[str, list[int]]]:
    """Files and row groups whose latitude/longitude ranges can intersect `box` (W, S, E, N). Index rows are [row_group, lat_min, lat_max, lon_min, lon_max]."""
    W, S, E, N = box
    hits = [(k, [r[0] for r in v[0] if r[2] >= S and r[1] <= N and r[4] >= W and r[3] <= E]) for k, v in idx.items() if v[0] is not None]
    return [(k, rgs) for k, rgs in hits if rgs]


def occurrences(box: tuple[float, float, float, float], tag: str, log=print) -> pd.DataFrame:
    """Occurrences with a point inside `box` (W, S, E, N). Cached as parquet."""
    out = cache_path(f"obis/{tag}.parquet")
    if out.exists():
        return pd.read_parquet(out)
    W, S, E, N = box
    idx = footer_index(log=log)
    filesystem = _fs()
    hits = select_row_groups(idx, box)
    log(f"{len(hits)} dataset files overlap the box")
    def read(item):
        k, rgs = item
        pf = pq.ParquetFile(f"{BUCKET}/{k}", filesystem=filesystem)
        frames = []
        for rg in rgs:
            ll = pf.read_row_group(rg, columns=[LAT, LON]).flatten().to_pandas()          # stage 1: coordinates only
            m = (ll.iloc[:, 0].between(S, N) & ll.iloc[:, 1].between(W, E)).to_numpy()
            if not m.any():
                continue
            t = pf.read_row_group(rg, columns=[LAT, LON] + FIELDS).flatten().to_pandas()   # stage 2: the other fields, only for row groups with hits
            t.columns = [c.replace("interpreted.", "") for c in t.columns]
            t = t[m]
            t["file"] = k
            frames.append(t)
        return frames

    with ThreadPoolExecutor(16) as ex:
        frames = [f for fs_ in ex.map(read, hits) for f in fs_]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_parquet(out)
    return df
