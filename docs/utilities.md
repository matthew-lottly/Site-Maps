# Utilities site map: Comanche (CO) power plant, Colorado

`python -m site_maps utilities` → `outputs/utilities_comanche_co.{png,pdf}` plus provenance and text inventory.

## Site resolution (nothing typed in)
The facility is WRI Global Power Plant Database record `USA0000470` (name asserted to be `Comanche (CO)`). The Overture plant footprint is found by matching the
OSM tag `ref:US:EIA` to the numeric tail of the WRI id (exactly one match is asserted). The WRI point sits 249 m from the footprint centroid (computed, logged).
State name comes from Natural Earth admin-1 geometry (`core/place.py`). The plant was chosen after comparing power-feature counts around the largest US plants in
other states; Comanche has 29 mapped substations and ~10k power features within 10 km.

## Data
| Layer | Source | Notes |
|---|---|---|
| Plant attributes | WRI GPPD v1.3.0 | WRI states the project is unmaintained since early 2022; Overture's plant tag cites EIA-860 2021 |
| Lines, towers, poles, substations, cables, pipelines, generators/transformers | Overture base/infrastructure (OSM) | voltage = highest value in the OSM `voltage` tag; untagged lines are a separate class |
| Roads, railways, buildings, land use, water | Overture | release 2026-09-23.1 |
| Imagery | Sentinel-2 L2A COG | clearest recent scene by window cloud fraction (id on the sheet) |
| Land cover | ESA WorldCover 2021 | shares on a 30 m grid within 5 and 20 km |
| Terrain | USGS 3DEP 1 arc-second | elevation range within 5 km only; the area is not hillshaded |

## Computed on the sheet
Line length by voltage class within 5 and 20 km (clipped in UTM metres), tower and substation counts by distance band, nearest substation and nearest surface water to the
WRI point, land-cover shares, elevation min/mean/max within 5 km, building and generator-point counts in the 2.4 km detail window.

## Limits
OSM-derived power data is not an engineering record: voltages are only as tagged (a sizeable share of line length is untagged), connectivity is not modelled, and
missing features are simply missing. Line classes `power_line` and `minor_line` are both drawn. Towers are points; poles are counted (log) but not drawn. Pipelines
are the few OSM `pipeline` ways in Overture and carry no product information on the sheet. The WRI point is a database location, not a survey point.
