"""Download cache under data/cache (not committed)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def cache_path(name: str) -> Path:
    p = CACHE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def download(url: str, name: str | None = None, timeout: int = 300) -> Path:
    dest = cache_path(name or hashlib.sha1(url.encode()).hexdigest()[:12] + "_" + url.rsplit("/", 1)[-1])
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def s3_listing(base: str, prefix: str, delimiter: str | None = "/") -> list[str]:
    """Anonymous S3 ListObjectsV2 over HTTPS; returns prefixes and keys."""
    import re

    out, tok = [], None
    while True:
        params = {"list-type": 2, "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if tok:
            params["continuation-token"] = tok
        txt = requests.get(base + "/", params=params, timeout=60).text
        out += [a or b for a, b in re.findall(r"<Prefix>([^<]*)</Prefix>|<Key>([^<]*)</Key>", txt)]
        m = re.search(r"<NextContinuationToken>([^<]*)", txt)
        if not m:
            return [o for o in out if o != prefix]
        tok = m.group(1)
