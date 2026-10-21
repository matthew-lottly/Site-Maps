# Camping site map: Yosemite Valley campgrounds

`python -m site_maps camping` → `outputs/camping_yosemite_valley.{png,pdf}`, plus `.provenance.json`, `.sources.md`, `.text.txt` (every string on the sheet).

## How the site is chosen (all from data)
1. Candidates: Overture places, category `campground`, in a search box around Yosemite.
2. Filter: confidence ≥ 0.80 and not closed (Overture entries at exactly 0.77 carry no operating status; this threshold removes them). Same normalised name within 1.5 km merged, highest confidence kept. Every drop is listed in `log["campgrounds_dropped"]` with its reason.
3. Site: the 24 Overture places whose name contains "Yosemite Valley" (confidence ≥ 0.7) give a median anchor; the single-linkage cluster (3 km links) of campgrounds nearest that anchor is mapped. A 0.98-confidence "Upper Pines" place lies about 22 km from the valley and is dropped as outside the cluster, which is why a plain "largest cluster" rule was rejected (it picks Wawona).
4. Frame: cluster span × 2.3, rounded to 0.5 km, in the local UTM zone (EPSG:32611).

## Data
Overture Maps 2026-09-23.1 (places, transportation segments, water, land_use camp-site polygons, infrastructure points: toilets, drinking water, parking, bus stops) · Sentinel-2 L2A COG (clearest, snow-free scene by scene-classification layer in the window) blended with a hillshade · USGS 3DEP 1 arc-second DEM (hillshade, contours at a data-derived interval, slope, profile) · iNaturalist Open Data research-grade observations (positional accuracy ≤ 100 m) · WorldCover 2021 · NOAA GHCN-Daily normals from the nearest station with ≥ 20 complete years in 1991–2020 · Natural Earth admin-1 for the state/country name.

## Numbers on the sheet and how they are computed
- Elevation: DEM sampled bilinearly at the Overture point. Height above river: that value minus the DEM at the nearest point of the longest river line named in the data. Slope: mean of the 10 m-grid slope within 100 m of the matched camp-site polygon.
- Distances: straight-line metres from the campground point to the nearest Overture toilets / drinking-water point / place named "…Trailhead".
- Profile: DEM sampled every 15 m along the river line in view; campgrounds are projected onto it.
- Climate: monthly means of daily TMAX/TMIN and mean monthly precipitation from the station's daily records; the station-to-campground elevation difference is shown.

## Limits
Overture place points can be offset from the real site; the DEM is 30 m native so slope and height above the river are indicative; infrastructure points are OSM-derived and incomplete; observations show recording effort, not abundance; no flood, fire or wildlife-hazard information is shown because none is in these datasets.
