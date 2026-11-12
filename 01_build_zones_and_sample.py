"""
PNN Colombia fire study - Step 1 (run in Spyder).

New workflow: geometry is done LOCALLY (geopandas), Earth Engine is only asked
for raster values at points, in small batches. Nothing runs as a GEE task.

  1. Download from GEE only the polygons needed: the 8 parks, the protected areas
     near them, the Colombia land boundary.
  2. Locally: dissolve parks, 10 km ring minus protected areas, clipped to land.
  3. Locally: random points in each zone (oversampled).
  4. GEE: covariates + yearly MODIS burned flags at the points, 1000 points per call.
  5. Drop water / null points, keep N_POINTS per zone, estimate land area.

Outputs (in config.DATA_DIR):
  pnn_zones.gpkg            parks and rings (reusable for maps and figures)
  pnn_zone_area.csv         total and land area of each zone (ha)
  pnn_matching_sample.csv   points with covariates and burned flags b2016..b2025

Requirements (Anaconda Prompt):
  conda install -c conda-forge geopandas earthengine-api
"""
import json
import time

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

import config as cfg

CRS_M = "EPSG:9377"          # MAGNA-SIRGAS / Origen-Nacional (metres)
OVERSAMPLE = 1.6             # extra points to replace water / null pixels
BATCH = 1000                 # points per Earth Engine request
YEARS = list(range(cfg.YEAR_START, cfg.YEAR_END + 1))

# park name -> accent-free search string in WDPA NAME
SEARCH = {
    "El Tuparro": "Tuparro",
    "Sumapaz": "Sumapaz",
    "El Cocuy": "Cocuy",
    "Sierra de la Macarena": "Macarena",
    "Tinigua": "Tinigua",
    "Chiribiquete": "Chiribiquete",
    "Sierra Nevada de Santa Marta": "Sierra Nevada",
    "Isla de Salamanca": "Salamanca",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ee_to_gdf(fc, props):
    """Small FeatureCollection -> GeoDataFrame (EPSG:4326)."""
    feats = fc.select(props).getInfo()["features"]
    rows = [dict(f["properties"], geometry=shape(f["geometry"])) for f in feats]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def union(g):
    """Dissolve a GeoDataFrame/GeoSeries (works with geopandas < 1.0 too)."""
    return g.union_all() if hasattr(g, "union_all") else g.unary_union


def random_points(geom, n, rng):
    """Uniform random points inside a (projected) polygon, by rejection."""
    minx, miny, maxx, maxy = geom.bounds
    pts = []
    while sum(len(p) for p in pts) < n:
        k = max(4 * n, 10000)
        x = rng.uniform(minx, maxx, k)
        y = rng.uniform(miny, maxy, k)
        inside = gpd.GeoSeries(gpd.points_from_xy(x, y)).within(geom).to_numpy()
        pts.append(np.column_stack([x[inside], y[inside]]))
    return np.vstack(pts)[:n]


def covariate_image():
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().rename("worldcover")
    dem = ee.Image("USGS/SRTMGL1_003")
    mcd = ee.ImageCollection("MODIS/061/MCD64A1").select("BurnDate")
    burned = [mcd.filterDate(f"{y}-01-01", f"{y + 1}-01-01")
              .map(lambda im: im.gt(0)).max().unmask(0).rename(f"b{y}") for y in YEARS]
    precip = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(f"{cfg.YEAR_START}-01-01", f"{cfg.YEAR_END + 1}-01-01")
              .sum().divide(len(YEARS)).rename("precip_mm_yr"))
    access = (ee.Image("Oxford/MAP/accessibility_to_cities_2015_v1_0")
              .select("accessibility").rename("travel_time_min"))
    return ee.Image.cat([dem.rename("elevation"), ee.Terrain.slope(dem).rename("slope"),
                         access, precip, wc] + burned)


def extract(img, df):
    """Raster values at points, BATCH points per request, with retries."""
    out = []
    for start in range(0, len(df), BATCH):
        chunk = df.iloc[start:start + BATCH]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lon, lat]), {"pid": int(pid)})
            for pid, lon, lat in zip(chunk.pid, chunk.lon, chunk.lat)])
        for attempt in range(4):
            try:
                res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                        scale=cfg.SAMPLE_SCALE, tileScale=4).getInfo()
                break
            except ee.EEException as e:
                print(f"  retry {attempt + 1} ({e})")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Batch starting at {start} failed 4 times")
        out.extend(f["properties"] for f in res["features"])
        print(f"  {min(start + BATCH, len(df))}/{len(df)} points")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ee.Initialize(project=cfg.GEE_PROJECT)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.SEED)

    # 1. polygons from GEE --------------------------------------------------
    wdpa = (ee.FeatureCollection("WCMC/WDPA/current/polygons")
            .filter(ee.Filter.eq("ISO3", "COL")))
    parks_fc = wdpa.filter(ee.Filter.stringContains("DESIG", "Parque"))

    zones = []
    print("Downloading park polygons...")
    for park, key in SEARCH.items():
        g = ee_to_gdf(parks_fc.filter(ee.Filter.stringContains("NAME", key)),
                      ["NAME", "DESIG"])
        if g.empty:
            raise ValueError(f"No WDPA park matches '{key}'")
        print(f"  {park:30s} <- {sorted(set(g.NAME))}")
        zones.append({"park": park, "ecosystem": cfg.PARKS[park],
                      "geometry": union(g.to_crs(CRS_M))})
    parks = gpd.GeoDataFrame(zones, geometry="geometry", crs=CRS_M)

    print("Downloading land boundary...")
    land = union(ee_to_gdf(ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                     .filter(ee.Filter.eq("country_na", "Colombia")),
                     ["country_na"]).to_crs(CRS_M))

    # 2. rings, locally ----------------------------------------------------
    rows = []
    for _, r in parks.iterrows():
        outer = r.geometry.buffer(cfg.BUFFER_KM * 1000)
        # protected areas touching this ring only (small download)
        bb = gpd.GeoSeries([outer], crs=CRS_M).to_crs(4326).total_bounds
        near = ee_to_gdf(wdpa.filterBounds(ee.Geometry.Rectangle(list(bb)))
                         .map(lambda f: f.simplify(50)), ["NAME"])
        pa_union = union(near.to_crs(CRS_M).buffer(0))
        print(f"  {r.park:30s} {len(near)} nearby protected areas")
        ring = outer.difference(pa_union).difference(r.geometry).intersection(land)
        rows.append({"park": r.park, "ecosystem": r.ecosystem, "zone": "park",
                     "geometry": r.geometry.intersection(land)})
        rows.append({"park": r.park, "ecosystem": r.ecosystem, "zone": "buffer",
                     "geometry": ring})
    zones = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_M)
    zones["zone_ha"] = zones.area / 1e4
    zones.to_file(cfg.DATA_DIR / "pnn_zones.gpkg", driver="GPKG")
    print(zones[["park", "zone", "zone_ha"]].round(0).to_string(index=False))

    # 3. random points, locally ---------------------------------------------
    n_draw = int(cfg.N_POINTS_PER_ZONE * OVERSAMPLE)
    pts = []
    for i, z in zones.iterrows():
        xy = random_points(z.geometry, n_draw, rng)
        pts.append(pd.DataFrame({"park": z.park, "ecosystem": z.ecosystem, "zone": z.zone,
                                 "treated": int(z.zone == "park"), "draw": np.arange(n_draw),
                                 "x": xy[:, 0], "y": xy[:, 1]}))
    pts = pd.concat(pts, ignore_index=True)
    ll = gpd.GeoSeries(gpd.points_from_xy(pts.x, pts.y), crs=CRS_M).to_crs(4326)
    pts["lon"], pts["lat"] = ll.x, ll.y
    pts["pid"] = np.arange(len(pts))

    # 4. covariates from GEE, in batches ----------------------------------
    print(f"Extracting covariates for {len(pts)} points...")
    vals = extract(covariate_image(), pts)
    df = pts.merge(vals, on="pid", how="left")

    # 5. keep land pixels with complete covariates, N per zone --------------
    cov = ["elevation", "slope", "travel_time_min", "precip_mm_yr", "worldcover"]
    df["water"] = df.worldcover.eq(80)
    land_frac = (1 - df.groupby(["park", "zone"]).water.mean()).rename("land_frac")
    ok = df[~df.water].dropna(subset=cov)
    sample = (ok.sort_values("draw").groupby(["park", "zone"]).head(cfg.N_POINTS_PER_ZONE))
    bcols = [f"b{y}" for y in YEARS]
    sample[bcols] = sample[bcols].fillna(0).astype(int)
    sample["n_burned_years"] = sample[bcols].sum(axis=1)
    sample = sample.drop(columns=["draw", "x", "y", "water"])
    sample.to_csv(cfg.DATA_DIR / "pnn_matching_sample.csv", index=False)

    area = zones.drop(columns="geometry").merge(land_frac.reset_index(), on=["park", "zone"])
    area["land_ha"] = area.zone_ha * area.land_frac
    area.to_csv(cfg.DATA_DIR / "pnn_zone_area.csv", index=False)

    print("\nPoints kept per zone:")
    print(sample.groupby(["park", "zone"]).size().unstack())
    print("\nDone. Files written to", cfg.DATA_DIR)


if __name__ == "__main__":
    main()
