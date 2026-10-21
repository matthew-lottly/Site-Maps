# Environmental research site map: Brazos Bend State Park, Texas

`python -m site_maps environmental` → `outputs/environmental_brazos_bend.{png,pdf}` plus `.provenance.json` and `.sources.md`.

## How the site is defined
The park is the Overture Maps `land_use` feature with id `adb37d48-f604-3b63-b9b0-099010f527b9` (name asserted at run time; subtype
`protected`, class `natural_monument`). Its polygon, not a hand-drawn outline, defines every in-park statistic. The map window is 12.6 km wide,
centred on the polygon centroid, in UTM zone 15N (EPSG:32615).

## Layers and what they are
| Layer | Dataset | Notes |
|---|---|---|
| Base image | Sentinel-2 L2A true colour (TCI), clearest scene in the window from March 2026 on (scene id printed in the log) | 10 m; window cloud/shadow measured from the scene classification layer, not the whole-tile figure |
| Land cover (in park, 40 % opacity) | ESA WorldCover 10 m 2021 v200 | official class names and colours; areas computed from pixel counts (100 m² per pixel in UTM) |
| Hydrography, roads, trails, places | Overture Maps release 2026-09-23.1 (OSM-derived) | names exactly as published; roads/trails by segment class |
| Field observations | iNaturalist Open Data, export dated 2026-09-27 | research grade only, positional accuracy ≤ 100 m; taxon group from the iNaturalist taxonomy |
| NDVI | Sentinel-2 B08/B04 | clearest scene per month (window coverage > 99 %, cloud/shadow < 15 %), vegetation and bare-soil pixels only (scene classification), mean and 10–90 % band over park pixels |
| Climate | NOAA GHCN-Daily | nearest NWS cooperative/airport station with ≥ 20 complete years in 1991–2020; normals computed from daily values |

## Numbers produced by the build (`EnvironmentalSiteMap.log`)
Park area 21.7 km². WorldCover area inside the park: tree cover 18.77 km², grassland 1.51, herbaceous wetland 0.99, permanent water 0.19,
cropland 0.15, built-up 0.06. 23,253 research-grade observations inside the park, spanning 1,948 distinct taxa (identifications at any rank).
NDVI inside the park runs from about 0.49 (February) to about 0.80 (May and July 2026). Climate: THOMPSONS 3 WSW (11.5 km, 20 complete years).

## Limits (also printed on the sheet's notes where space allows)
- Land cover is a 10 m global product; it will misclassify small features. The Overture polygon is OSM-derived, not the surveyed park boundary.
- Observation density follows where people walk (trails, observation decks), so it maps observer effort as much as biodiversity.
- Sentinel-2 gaps: a humid-subtropical summer leaves few cloud-free scenes; NDVI months without a clear scene are absent rather than interpolated.
- The scene-classification mask is not perfect; thin cloud or shadow can remain.
- Climate normals come from a station 11.5 km away; elevation and local microclimate can differ.
