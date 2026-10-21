# Coastal / sea research site map — Massachusetts Bay

Run: `PYTHONPATH=src python3 -m site_maps coastal` → `outputs/coastal_massachusetts_bay.{png,pdf,provenance.json,sources.md,text.txt}`.

## Data (all public; every layer registered in provenance)
- **Seafloor/land relief** — NOAA NCEI CUDEM 1/9 and 1/3 arc-second topobathy (NAVD88), finest tiles first, averaged to a 60 m UTM 19N grid. Extent is the largest window inside CUDEM tile coverage around the Overture "Massachusetts Bay" point (77 km).
- **Imagery** — Sentinel-2 L2A COGs (best low-cloud scene per MGRS tile), shown over land only.
- **Currents / SST** — NOAA NOS GoMOFS latest nowcast, 2-D surface fields; u/v averaged to cell centres, rotated by the grid angle, arrows on a 5 km grid.
- **SST series** — NOAA OISST v2.1 daily files; mean of valid 0.25° cells in the frame, 15th of each month, vs. same calendar day 2016 onward.
- **Species** — OBIS GeoParquet export, records inside the frame (absences removed); seafloor depth sampled from the CUDEM grid.
- **Names / context** — Overture water and land names; state from `place.context`.

## Method
Depth zones use UTM scale-factor-corrected cell areas; hypsometric curve from the same grid. Transect A–B runs from the nearest land cell to the grid's deepest cell, both read from the grid. All numbers on the sheet are computed; see `CoastalSiteMap.log`.

## Limits
Gridded NAVD88 heights, not a nautical chart; 0 m NAVD88 is not the shoreline. Currents/temperature are model nowcasts. OBIS records are presence-only; density reflects observer effort and a few dominant datasets. No SST map layer; no iNaturalist records. Limited place-name labelling (capes, bays, peninsulas, islands; max 40, dropped on collision). Network-heavy first run (caches under `data/cache`).
