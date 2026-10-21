"""Provenance: every layer drawn on a map must be registered with where it came from.

A map cannot be exported with an unregistered layer (the layout asks the log for its sources box), so nothing appears on a
sheet without a publisher, dataset, version, URL and licence.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Source:
    key: str                  # short id used on the sheet, e.g. "S2"
    publisher: str
    dataset: str
    version: str              # release / vintage / acquisition date, exactly as published
    url: str
    license: str
    note: str = ""


@dataclass
class LayerRecord:
    layer: str
    source: Source
    features: int | None = None
    detail: str = ""          # what was done to it (clip, resample, classify)


@dataclass
class ProvenanceLog:
    records: list[LayerRecord] = field(default_factory=list)
    retrieved: str = field(default_factory=lambda: dt.date.today().isoformat())

    def add(self, layer: str, source: Source, features: int | None = None, detail: str = "") -> None:
        self.records.append(LayerRecord(layer, source, features, detail))

    def sources(self) -> list[Source]:
        seen, out = set(), []
        for r in self.records:
            if r.source.key not in seen:
                seen.add(r.source.key)
                out.append(r.source)
        return out

    def keys_for(self, layer_prefix: str) -> str:
        return ", ".join(sorted({r.source.key for r in self.records if r.layer.startswith(layer_prefix)}))

    def to_markdown(self) -> str:
        lines = ["| Layer | Source | Dataset / version | Features | Processing |", "|---|---|---|---|---|"]
        for r in self.records:
            s = r.source
            lines.append(f"| {r.layer} | {s.publisher} ({s.key}) | [{s.dataset}]({s.url}), {s.version} | {r.features if r.features is not None else ''} | {r.detail} |")
        lines += ["", "| Key | Publisher | Dataset | Version | Licence |", "|---|---|---|---|---|"]
        for s in self.sources():
            lines.append(f"| {s.key} | {s.publisher} | [{s.dataset}]({s.url}) | {s.version} | {s.license} |")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({"retrieved": self.retrieved, "layers": [{"layer": r.layer, "features": r.features, "detail": r.detail, "source": asdict(r.source)} for r in self.records]}, indent=1)
