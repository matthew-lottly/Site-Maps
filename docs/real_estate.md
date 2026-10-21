# Real-estate site map: Rainier Square, Seattle (King County, Washington)

`python -m site_maps real-estate` → `outputs/real_estate_rainier_square_seattle.{png,pdf}`, plus `.provenance.json`, `.sources.md`, `.text.txt`.

## Site resolution
The subject is the Overture Maps building feature `bffb77cb-5b55-4536-94a2-e52504282cea` (release 2026-09-23.1). The code fetches it from a search
box and asserts that its `name` is "Rainier Square" and that exactly one feature matches. Neighbourhood/county names come from the Overture
division polygons containing the building centroid; the state name comes from the Natural Earth admin-1 polygon (`core/place.py`). Seattle was
chosen after checking that Overture buildings, places and streets are dense there and that Sentinel-2 tile 10TET covers the 4.3 km window
completely (Denver, first tried, straddles two MGRS tiles; Chicago's window falls outside the tile on the dates tested).

## Layers and what they contribute
| Layer | Dataset | Use |
|---|---|---|
| Imagery | Sentinel-2 L2A COG, clearest scene by window cloud/shadow fraction | base, washed 45 % toward white |
| Buildings | Overture buildings (OSM-derived) | footprints coloured by `height`; grey where none recorded; ring statistics |
| Streets | Overture transportation segments | classes, names, posted maximum speed (`speed_limits`) where tagged |
| Land use | Overture base `land_use` | tinted polygons, 30 % opacity |
| Places | Overture places (open, confidence ≥ 0.7) | counts by category group per distance band |
| Population | Meta / CIESIN HRSL general, tile read from the published `latest.vrt` mosaic | sum of ~30 m cell values whose centre lies in each ring |
| Terrain | USGS 3DEP 1 arc-second | minimum and maximum elevation in each ring |

## Computed numbers (all in `RealEstateSiteMap.log`)
Per ring (0.25 / 0.5 / 1 mile, cumulative, measured from the building centroid): population, people per km², building count, footprint
coverage (exact polygon intersection area / ring area), median recorded height, open places, elevation range. Nearest street of each class is
measured to the building footprint, with the posted maximum speed where the segment carries one.

## Limits
- Overture content for buildings, land use, streets and places derives from OpenStreetMap and other contributed sources; completeness and
  currency vary. Heights are recorded for only part of the footprints (stated on the sheet).
- HRSL is a modelled population surface; it is not a census count, and cells straddling a ring are counted by centre.
- Place counts include every category group (many are offices and services); they are not a measure of retail demand.
- No parcel boundaries, zoning, ownership or market data are used: none is available from reachable open sources.
