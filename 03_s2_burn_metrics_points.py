"""
PNN Colombia fire study - Step 3 (Spyder): Sentinel-2 burn metrics at the sample points.

For every point of data/pnn_matching_sample.csv and every year 2017-2025 it
extracts from Sentinel-2 L1C (harmonized, Cloud Score+ masked):

  dnbr_YYYY   largest NBR drop of the year = median NBR - minimum NBR
  nlow_YYYY   number of clear observations with NBR < median - 0.1
              (a real burn scar persists over several acquisitions;
               cloud/shadow residuals usually affect a single date)
  nobs_YYYY   number of clear observations
  doy_YYYY    day of year of the minimum NBR (fire timing)

Observations that look like water (MNDWI > 0) are excluded, so seasonal
flooding in wetlands (Isla de Salamanca) is not mistaken for fire.
Thresholds are NOT applied here: they are calibrated afterwards in Python
(step 4) against MODIS and FIRMS.

Earth Engine is only asked for values at points, in small spatially compact
batches, several in parallel. Every batch is saved as it finishes, so if the
script stops you can relaunch it and it continues where it left off.

Output: data/pnn_s2_burn_metrics.csv
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config as cfg

S2_YEARS = list(range(2017, cfg.YEAR_END + 1))
BATCH = 250            # points per request
WORKERS = 6            # parallel requests
CS_THRESHOLD = 0.60    # Cloud Score+ clear threshold
LOW_DROP = 0.10        # NBR drop used for nlow
PARTS = cfg.DATA_DIR / "s2_parts"
TEST_BATCHES = 6       # first run: only 6 batches to check speed; then set to None


def s2_year_image(year, region):
    """Bands dnbr, nlow, nobs, doy for one year over the batch region."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    start, end = f"{year}-01-01", f"{year + 1}-01-01"

    def prep(im):
        clear = im.select("cs").gte(CS_THRESHOLD)
        nbr = im.normalizedDifference(["B8", "B12"]).rename("nbr")
        mndwi = im.normalizedDifference(["B3", "B11"])
        doy = ee.Image.constant(ee.Date(im.get("system:time_start")).getRelative("day", "year")
                                .add(1)).toInt16().rename("doy")
        return (nbr.addBands(nbr.multiply(-1).rename("negnbr")).addBands(doy)
                .updateMask(clear.And(mndwi.lte(0))))

    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterDate(start, end).filterBounds(region)
           .linkCollection(cs, ["cs"]).map(prep))

    nbr = col.select("nbr")
    med = nbr.median()
    dnbr = med.subtract(nbr.min()).rename(f"dnbr_{year}")
    nlow = (nbr.map(lambda im: im.lt(med.subtract(LOW_DROP))).sum()
            .rename(f"nlow_{year}"))
    nobs = nbr.count().rename(f"nobs_{year}")
    doy = col.qualityMosaic("negnbr").select("doy").rename(f"doy_{year}")
    return ee.Image.cat([dnbr, nlow, nobs, doy])


def run_batch(i, chunk):
    out = PARTS / f"batch_{i:04d}.csv"
    if out.exists():
        return i, "skipped"
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"pid": int(pid)})
        for pid, lon, lat in zip(chunk.pid, chunk.lon, chunk.lat)])
    region = fc.geometry().bounds()
    img = ee.Image.cat([s2_year_image(y, region) for y in S2_YEARS])
    for attempt in range(5):
        try:
            res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(),
                                    scale=20, tileScale=4).getInfo()
            pd.DataFrame([f["properties"] for f in res["features"]]).to_csv(out, index=False)
            return i, "ok"
        except Exception as e:          # noqa: BLE001 - EE raises several types
            msg = str(e)[:120]
            time.sleep(10 * (attempt + 1))
    return i, f"FAILED: {msg}"


def main():
    ee.Initialize(project=cfg.GEE_PROJECT)
    PARTS.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(cfg.DATA_DIR / "pnn_matching_sample.csv",
                      usecols=["pid", "park", "zone", "lon", "lat"])
    # spatially compact batches: k-means clusters within each park/zone,
    # clusters larger than BATCH are split
    chunks = []
    for _, g in pts.groupby(["park", "zone"]):
        k = int(np.ceil(len(g) / BATCH)) * 2
        lab = KMeans(n_clusters=k, n_init=4, random_state=cfg.SEED).fit_predict(g[["lon", "lat"]])
        for c in range(k):
            gc = g[lab == c]
            chunks += [gc.iloc[s:s + BATCH] for s in range(0, len(gc), BATCH)]
    if TEST_BATCHES:
        chunks = chunks[:TEST_BATCHES]
    n = len(chunks)
    print(f"{len(pts)} points, {n} batches of {BATCH}, {WORKERS} in parallel")

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_batch, i, c) for i, c in enumerate(chunks)]
        for f in as_completed(futures):
            i, status = f.result()
            done += 1
            if status.startswith("FAILED"):
                failed.append(i)
            el = (time.time() - t0) / 60
            print(f"  batch {i:4d} {status:8.8s} | {done}/{n} done, "
                  f"{el:5.1f} min elapsed, ~{el / done * (n - done):5.1f} min left")

    if TEST_BATCHES:
        print("\nTest finished. Set TEST_BATCHES = None and relaunch for all points.")
        return
    if failed:
        print(f"\n{len(failed)} batches failed: relaunch the script to retry them.")
        return

    parts = pd.concat([pd.read_csv(p) for p in sorted(PARTS.glob("batch_*.csv"))])
    parts.to_csv(cfg.DATA_DIR / "pnn_s2_burn_metrics.csv", index=False)
    print(f"\nDone: {len(parts)} points written to data/pnn_s2_burn_metrics.csv")


if __name__ == "__main__":
    main()
