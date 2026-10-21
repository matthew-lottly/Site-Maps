# Data hosts reachable without credentials (checked from the build environment)

Reachable and used: Overture Maps (S3, global: places, buildings, transportation, divisions, base: land, land_use, land_cover, water,
infrastructure, bathymetry) · USGS 3DEP elevation (prd-tnm S3) · USGS 3DEP lidar (usgs-lidar-public) · Copernicus DEM GLO-30 · ESA WorldCover ·
Sentinel-2 L2A COGs (sentinel-cogs S3) · NOAA GHCN-Daily · NOAA NOS Operational Forecast Systems (noaa-nos-ofs-pds) · NOAA CDR sea-surface
temperature (OISST) · NOAA National Water Model (noaa-nwm-pds) · NOAA coastal lidar (noaa-nos-coastal-lidar-pds) · NOAA DCDB bathymetry
(noaa-dcdb-bathymetry-pds) · NOAA GOES · NASA POWER · iNaturalist Open Data · OBIS Open Data · GBIF open data (parquet, not spatially indexed)
· Meta Data for Good (population density, canopy height) · WRI Global Power Plant Database (raw GitHub) · Natural Earth (raw GitHub, S3) ·
Mapzen/AWS terrain tiles · PyPI.

Blocked: Census (api/TIGER), FEMA, USGS/NOAA web APIs and services (waterservices, tidesandcurrents, ndbc, ERDDAP), GBIF/OBIS/iNaturalist
REST APIs, ArcGIS/Esri services, OSM tiles/Overpass/Nominatim, Zillow, FHFA, HUD, recreation.gov, NPS, EPA ECHO.

Python access patterns that work through the session proxy:

- Overture / pyarrow: `fs.S3FileSystem(anonymous=True, region=..., proxy_options=os.environ["HTTPS_PROXY"])` (see `sources/overture.py`)
- Remote rasters: `rasterio` with `/vsicurl/https://...` inside `sources.rasters.remote_env()`; use `read_to_grid` to resample onto the map grid
- S3 listings over HTTPS: `core.cache.s3_listing(base_url, prefix)`
- Large archives: stream and filter (see `sources/inat.py`), keep only the small extract in `data/cache`
- HTTP objects served with `Content-Encoding: gzip` are already decompressed by `requests`
