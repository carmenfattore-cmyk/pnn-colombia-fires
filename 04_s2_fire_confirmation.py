"""
PNN Colombia fire study - Step 4 (Spyder): turn Sentinel-2 NBR drops into fire / no fire.

Problem: an NBR drop alone is not fire-specific. Harvest, ploughing, dry-season
leaf loss (Caribbean dry forest) and terrain shadow also lower NBR.
Solution: a Sentinel-2 drop counts as FIRE only when an independent fire
product confirms it:
  - a VIIRS active fire (375 m, NASA FIRMS) close to the point shortly before
    the date of minimum NBR  -> captures small fires, or
  - the MODIS MCD64A1 burned-area flag for that year -> captures large, fast
    savanna fires, whose front is often missed by the VIIRS overpasses.

  fire(point, year) = dnbr >= DNBR_T  and  nlow >= NLOW_K
                      and  ( VIIRS hotspot within HOT_DIST_M
                             between doy - DAYS_BEFORE and doy + DAYS_AFTER
                             or  MODIS burned )
  point-years with < MIN_OBS clear images are set to missing.

Inputs : data/pnn_matching_sample.csv, data/pnn_s2_burn_metrics.csv,
         data/firms/*.csv   (FIRMS VIIRS S-NPP yearly country files, Colombia)
Outputs: data/pnn_sample_s2fire.csv   (sample + f2017..f2025 fire flags)
         results/s2_vs_modis_by_zone.csv, results/s2_sensitivity.csv
"""
import glob

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer

import config as cfg

S2_YEARS = list(range(2017, cfg.YEAR_END + 1))
DNBR_T = 0.15          # main threshold (sensitivity: 0.25, 0.35)
NLOW_K = 2             # drop visible in >= 2 clear images
MIN_OBS = 5            # minimum clear images per year
HOT_DIST_M = 500       # VIIRS pixel 375 m + geolocation error
DAYS_BEFORE = 60       # hotspot may precede the NBR minimum by up to 60 days
DAYS_AFTER = 5

TO_M = Transformer.from_crs("EPSG:4326", "EPSG:9377", always_xy=True)


def load_viirs():
    files = sorted(glob.glob(str(cfg.DATA_DIR / "firms" / "*.csv")))
    if not files:
        raise FileNotFoundError("No FIRMS CSV in data/firms - download them first.")
    v = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "type" in v:                      # 0 = presumed vegetation fire (archive files);
        v = v[v["type"].isna() | (v["type"] == 0)]   # NRT files have no 'type' column
    v = v.drop_duplicates(subset=[c for c in ["latitude", "longitude", "acq_date", "acq_time"]
                                  if c in v])
    v = v[v.confidence.astype(str).str.lower() != "l"]          # drop low confidence
    v["date"] = pd.to_datetime(v.acq_date)
    v = v[v.date.dt.year.between(S2_YEARS[0] - 1, S2_YEARS[-1])]
    v["x"], v["y"] = TO_M.transform(v.longitude.to_numpy(), v.latitude.to_numpy())
    print(f"VIIRS hotspots used: {len(v):,} ({v.date.min():%Y-%m-%d} to {v.date.max():%Y-%m-%d})")
    return v


def hotspot_confirmed(pts, viirs, year):
    """Boolean array: a hotspot near each point in the window before its NBR minimum."""
    doy = pts[f"doy_{year}"].to_numpy()
    tmin = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(np.nan_to_num(doy) - 1 - DAYS_BEFORE, "D")
    tmax = pd.Timestamp(f"{year}-01-01") + pd.to_timedelta(np.nan_to_num(doy) - 1 + DAYS_AFTER, "D")
    v = viirs[(viirs.date >= tmin.min()) & (viirs.date <= tmax.max())]
    ok = np.zeros(len(pts), dtype=bool)
    if v.empty:
        return ok
    tree = cKDTree(v[["x", "y"]].to_numpy())
    vdates = v.date.to_numpy()
    for i, nb in enumerate(tree.query_ball_point(pts[["x", "y"]].to_numpy(), r=HOT_DIST_M)):
        if nb and not np.isnan(doy[i]):
            d = vdates[nb]
            ok[i] = ((d >= tmin[i].to_datetime64()) & (d <= tmax[i].to_datetime64())).any()
    return ok


def classify(m, viirs, dnbr_t):
    flags, hot = {}, {}
    for y in S2_YEARS:
        drop = (m[f"dnbr_{y}"] >= dnbr_t) & (m[f"nlow_{y}"] >= NLOW_K)
        if y not in hot:
            hot[y] = hotspot_confirmed(m, viirs, y)
        f = (drop & (hot[y] | (m[f"b{y}"] == 1))).astype(float)
        f[m[f"nobs_{y}"].fillna(0) < MIN_OBS] = np.nan
        flags[f"f{y}"] = f
        flags[f"drop{y}"] = drop.astype(float).where(m[f"nobs_{y}"].fillna(0) >= MIN_OBS)
    return pd.DataFrame(flags, index=m.index)


def main():
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    s = pd.read_csv(cfg.DATA_DIR / "pnn_matching_sample.csv")
    s2 = pd.read_csv(cfg.DATA_DIR / "pnn_s2_burn_metrics.csv")
    m = s.merge(s2, on="pid")
    m["x"], m["y"] = TO_M.transform(m.lon.to_numpy(), m.lat.to_numpy())
    viirs = load_viirs()

    # sensitivity of the threshold
    sens = []
    for t in [0.15, 0.25, 0.35]:
        fl = classify(m, viirs, t)
        g = pd.concat([m[["park", "zone"]], fl], axis=1)
        g["fire_rate"] = g[[f"f{y}" for y in S2_YEARS]].mean(axis=1) * 100
        g["drop_rate"] = g[[f"drop{y}" for y in S2_YEARS]].mean(axis=1) * 100
        r = g.groupby(["park", "zone"])[["drop_rate", "fire_rate"]].mean().reset_index()
        r["dnbr_t"] = t
        sens.append(r)
        if t == DNBR_T:
            main_flags = fl
    sens = pd.concat(sens)
    sens.round(3).to_csv(cfg.RESULTS_DIR / "s2_sensitivity.csv", index=False)

    # comparison with MODIS on the common period 2017-2025
    out = pd.concat([m, main_flags[[f"f{y}" for y in S2_YEARS]]], axis=1)
    out["modis_rate"] = out[[f"b{y}" for y in S2_YEARS]].mean(axis=1) * 100
    out["s2fire_rate"] = out[[f"f{y}" for y in S2_YEARS]].mean(axis=1) * 100
    comp = (out.groupby(["park", "zone"])[["modis_rate", "s2fire_rate"]].mean()
            .unstack().round(2))
    comp.to_csv(cfg.RESULTS_DIR / "s2_vs_modis_by_zone.csv")
    print("\nAnnual burn rate (% of points per year), 2017-2025:")
    print(comp)
    print("\nThreshold sensitivity (fire_rate, %/yr):")
    print(sens.pivot_table(index=["park", "zone"], columns="dnbr_t", values="fire_rate").round(2))

    keep = [c for c in out.columns if not c.startswith(("dnbr_", "nlow_", "nobs_"))
            and c not in ("x", "y")]
    out[keep].to_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv", index=False)
    print("\nWritten data/pnn_sample_s2fire.csv")


if __name__ == "__main__":
    main()
