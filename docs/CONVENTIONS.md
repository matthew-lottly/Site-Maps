# Conventions for site maps in this repo

## The rule: no fake data, no guessing

1. **Every drawn layer is registered** with `sheet.prov.add(layer, Source, features, detail)`. The sources box is generated from that log, so
   an unregistered layer has no citation and must not be drawn.
2. **Sites are resolved from data, not typed in.** Pick the site by a dataset identifier (an Overture feature id, a WRI plant id, a GHCN
   station id) and assert that the record's name matches. A `SEARCH_BOX` used only to fetch the record is fine; a hand-typed coordinate that
   stands in for a real feature is not.
3. **Every number on the sheet is computed from the data on the sheet** (areas, counts, normals, distances). No figures from memory, no
   hand-drawn polygons, no illustrative values. If a quantity can't be computed from an available dataset, leave it off and say so in the notes.
4. **State limits honestly** in the notes box: resolution, vintage, what the dataset can and cannot show (e.g. "OSM-derived power lines are
   not an engineering record", "bathymetry is gridded, not a nautical chart").
5. **Mark vintage**: imagery date, release id, normals period, export date. Pin versions (Overture release, scene ids).
6. Names and labels are printed exactly as the dataset has them.

## Layout API (`site_maps.core.layout.Sheet`)

17 × 11 in landscape. `Sheet(title, subtitle, site_line, frame, prov)` then draw on `sheet.ax` in projected metres (`frame.crs`).

- `sheet.graticule()`, `scale_bar()`, `north_arrow()` (true-north arrow, grid convergence stated)
- right column, top to bottom: `locator()`, `legend(Legend)`, `panel(height_in, title)` + `chart_axes(panel, rect)`, `note_box(title, lines)`, `sources_box()`
- `sheet.export(stem, outdir)` writes PNG (200 dpi), PDF, `<stem>.provenance.json`, `<stem>.sources.md`, and returns `issues` from the automatic text-overlap audit. **The audit must be empty.**
- Panel heights are inches; the column has about 8.9 in in total. `sheet.remaining_in()` tells you what is left.

## Map classes

`maps/<type>.py` holds one class with `key`, `__init__(self, outdir=OUTPUTS)`, `build()` returning the `Sheet`, and a `log` dict of the computed
numbers (used by tests and the README). Data fetching lives in `sources/<name>.py` modules that return arrays/GeoDataFrames plus a `Source`.

## Cartographic standards used on every sheet

Title block with site name and coordinates; projected CRS stated (local UTM); graticule labelled on all four borders (WGS 84); scale bar
and printed ratio; true-north arrow with grid convergence; locator; legend; data-sources box with publisher / dataset / version / licence;
footer with date and software version. Colour-vision-safe palettes (viridis/cividis for continuous data); official class colours where a
product defines them (e.g. ESA WorldCover).

## Tests

Pure-logic tests only (no network): geometry, unit conversions, classification, parsing of dataset records. Network-dependent builds are
verified by running `python -m site_maps <map>` and checking `issues == []` and the printed `log`.

## Audits that run on every export

- **Layout audit** (`core/qa.py`): no text may overlap other text, leave the page or leave its panel. `export()['issues']` must be `[]`.
- **Text inventory** (`<stem>.text.txt`): every string printed on the sheet. Each must come from a dataset field (names exactly as published) or a
  computed value; reviewers can check the file against `<stem>.provenance.json`.
- **Collision-aware labels** (`core/labels.py` `LabelPlacer`): place point, area and line labels in priority order; labels that cannot be placed are
  dropped instead of overprinting. Reserve furniture (scale bar, north arrow) before placing.
- **No typed place facts**: state / province / country names come from `core/place.py` (Natural Earth admin-1 polygons), class names and colours
  from the product files themselves (`sources/worldcover.classes()` reads the GeoTIFF legend and colour table).
