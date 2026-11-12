# Small fires and protected-area effectiveness in eight Colombian National Natural Parks — code and data

Code and derived data for:

Fattore, C., Coronado-Hernández, J.R. *Small fires change the verdict on protected-area effectiveness: matched Sentinel-2, VIIRS and MODIS evidence from eight Colombian National Natural Parks.* Submitted to the Journal of Environmental Management.

## Contents

| Path | Description |
|---|---|
| `config.py` | Paths, parks, years and Google Earth Engine project ID (set `GEE_PROJECT` to your own Cloud project). |
| `01_build_zones_and_sample.py` | Park polygons (WDPA), 10 km external rings minus other protected areas and clipped to land (EPSG:9377), random sample points and covariates extracted from Earth Engine. |
| `02_matching_park_effect.py` | Covariate matching (nearest neighbour with replacement, exact on land-cover group, caliper 0.5 SD) and ATT with 10 km block bootstrap; `OUTCOME = "s2"` or `"modis"`. |
| `03_s2_burn_metrics_points.py` | Annual Sentinel-2 L1C NBR metrics at the points (Cloud Score+ cs ≥ 0.6, MNDWI water mask): dNBR = median − minimum, number of low and clear images, date of the minimum, 2017–2025. |
| `04_s2_fire_confirmation.py` | Fire confirmation: dNBR ≥ 0.15 in ≥ 2 clear images AND (VIIRS S-NPP 375 m hotspot ≤ 500 m between −60 and +5 days of the minimum OR MODIS MCD64A1 burned); < 5 clear images = missing. |
| `05_gam_drivers.py` | Logistic GAM (pyGAM), smoothing chosen by spatial cross-validation on 10 km blocks; Fig. 6. |
| `06_lightgbm_shap.py` | LightGBM with SHAP values (robustness check); Fig. S1. |
| `07_figures.py`, `08_flowchart.py`, `09_extra_figures.py` | Figures 1–5 and S2–S3. |
| `10_tidy_figs.py` | Moves superseded figures to `figs/old`. |
| `11_graphical_abstract.py` | Graphical abstract. |
| `GEE_park_vs_ring_fire_S2.js` | Simplified single-park version of the workflow for the Earth Engine Code Editor. |
| `data/pnn_zones.gpkg` | Park and ring polygons (EPSG:9377). |
| `data/pnn_zone_area.csv` | Area of each park and ring. |
| `data/pnn_matching_sample.csv` | Sample points with covariates and MODIS MCD64A1 burned flags (`b2016`–`b2025`). |
| `data/pnn_s2_burn_metrics.csv` | Sentinel-2 metrics per point (`pid`) and year: `dnbr`, `doy` (date of the NBR minimum), `nlow`, `nobs`. |
| `data/pnn_sample_s2fire.csv` | Final point table: covariates, MODIS flags and Sentinel-2 confirmed fire flags (`f2017`–`f2025`; empty = missing). |
| `results/` | Tables behind the results (ATT per park, covariate balance, GAM and LightGBM cross-validation, SHAP importance, threshold sensitivity). |

## Data not redistributed

- VIIRS S-NPP 375 m active fires (archive, Colombia, 2016–2025): download from NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/download/) and place the CSV in `data/firms/`.
- Official national and departmental boundaries (Fig. 1): DANE / OCHA ROLAC COD-AB, https://data.humdata.org/dataset/cod-ab-col, in `col-administrative-divisions-shapefiles/`.
- All other inputs (Sentinel-2, Cloud Score+, MODIS MCD64A1, SRTM, CHIRPS, ESA WorldCover, accessibility, WDPA) are read from Google Earth Engine.

## Environment

Python 3.11 with earthengine-api, geopandas 1.2, pandas, numpy, scikit-learn 1.9, pygam 0.12, lightgbm 4.7, shap 0.54, matplotlib. Run the scripts in numerical order (Spyder or command line). Steps 01 and 03 need an Earth Engine account; steps 02 and 04–09 run on the files in `data/`.

## Licence

Code: MIT. Derived data: CC BY 4.0.
