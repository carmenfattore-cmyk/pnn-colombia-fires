"""
PNN Colombia fire study - Step 2 (Spyder): does each park reduce fire?

Covariate matching, park by park:
  treated  = points inside the park
  controls = points in the 10 km external ring
Each park point is matched to the most similar ring point (with replacement)
on standardised elevation, slope, travel time to cities and rainfall, with
exact matching on land-cover group. Pairs farther than CALIPER (in SD units)
are discarded.

Outcome: annual burn probability (share of 2016-2025 years burned, MCD64A1).
Effect (ATT) = park - matched ring, in percentage points per year, with a
spatial block bootstrap (BLOCK_KM grid cells) for the 95% CI.

OUTCOME = "s2": MODIS MCD64A1, 2016-2025 (data/pnn_matching_sample.csv)
OUTCOME = "s2"   : Sentinel-2 fires confirmed by VIIRS/MODIS, 2017-2025
                   (data/pnn_sample_s2fire.csv, from step 4)
Outputs: results/park_effect_att_<OUTCOME>.csv, results/covariate_balance_<OUTCOME>.csv,
         figs/fig_park_effect_<OUTCOME>.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import config as cfg

import os
OUTCOME = os.environ.get("OUTCOME", "s2")        # "modis" or "s2"
FOREST_ONLY = os.environ.get("FOREST_ONLY", "0") == "1"   # forest and mangrove points only
SUFFIX = OUTCOME + ("_forest" if FOREST_ONLY else "")
COVS = ["elevation", "slope", "travel_time_min", "precip_mm_yr"]
CALIPER = 0.5          # max matching distance, in pooled-SD units (Euclidean)
BLOCK_KM = 10          # spatial block size for the bootstrap
N_BOOT = 1000
YEARS = list(range(cfg.YEAR_START, cfg.YEAR_END + 1))

# ESA WorldCover -> broad groups for exact matching
LC_GROUP = {10: "forest", 20: "shrub_grass", 30: "shrub_grass", 90: "wetland",
            95: "wetland", 40: "agri_built", 50: "agri_built", 60: "bare",
            70: "bare", 100: "shrub_grass"}


def load():
    if OUTCOME == "modis":
        df = pd.read_csv(cfg.DATA_DIR / "pnn_matching_sample.csv")
        cols = [f"b{y}" for y in YEARS]
    else:
        df = pd.read_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv")
        cols = [f"f{y}" for y in range(2017, cfg.YEAR_END + 1)]
    df = df[df.worldcover != 80].dropna(subset=COVS + ["worldcover"])
    df["lc"] = df.worldcover.astype(int).map(LC_GROUP).fillna("other")
    if FOREST_ONLY:
        df = df[df.worldcover.isin([10, 95])]
    df["burn_rate"] = df[cols].mean(axis=1) * 100  # % of valid years burned
    df = df.dropna(subset=["burn_rate"])
    # spatial blocks (approx. km grid in degrees)
    deg = BLOCK_KM / 111.0
    df["block"] = (np.floor(df.lon / deg).astype(int).astype(str) + "_" +
                   np.floor(df.lat / deg).astype(int).astype(str))
    return df


def smd(a, b):
    s = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def match_park(d):
    """Return matched pairs (index of treated, index of control) for one park."""
    t, c = d[d.treated == 1], d[d.treated == 0]
    sd = d[COVS].std().replace(0, 1)
    pairs = []
    for lc in t.lc.unique():
        tt, cc = t[t.lc == lc], c[c.lc == lc]
        if len(cc) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1).fit((cc[COVS] / sd).to_numpy())
        dist, idx = nn.kneighbors((tt[COVS] / sd).to_numpy())
        keep = dist[:, 0] <= CALIPER
        pairs.append(pd.DataFrame({"t": tt.index[keep], "c": cc.index[idx[keep, 0]]}))
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["t", "c"])


def block_bootstrap(diff, blocks, rng):
    ub = blocks.unique()
    groups = {b: diff[blocks == b].to_numpy() for b in ub}
    est = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(ub, size=len(ub), replace=True)
        est[i] = np.concatenate([groups[b] for b in pick]).mean()
    return np.percentile(est, [2.5, 97.5])


def main():
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    cfg.FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.SEED)
    df = load()

    res, bal = [], []
    for park, d in df.groupby("park"):
        pairs = match_park(d)
        t, c = d[d.treated == 1], d[d.treated == 0]
        mt, mc = d.loc[pairs.t], d.loc[pairs.c]

        # covariate balance before / after
        for v in COVS:
            bal.append({"park": park, "covariate": v,
                        "smd_before": smd(t[v], c[v]), "smd_after": smd(mt[v], mc[v])})

        diff = pd.Series(mt.burn_rate.to_numpy() - mc.burn_rate.to_numpy())
        ci = (block_bootstrap(diff, mt.block.reset_index(drop=True), rng)
              if len(diff) > 1 else [np.nan, np.nan])
        res.append({
            "park": park, "ecosystem": d.ecosystem.iloc[0],
            "n_park": len(t), "n_matched": len(pairs),
            "pct_matched": 100 * len(pairs) / len(t),
            "burn_park_raw": t.burn_rate.mean(), "burn_ring_raw": c.burn_rate.mean(),
            "naive_diff": t.burn_rate.mean() - c.burn_rate.mean(),
            "burn_park_matched": mt.burn_rate.mean(), "burn_ring_matched": mc.burn_rate.mean(),
            "att": diff.mean(), "ci_low": ci[0], "ci_high": ci[1],
        })
        print(f"{park:30s} matched {len(pairs):5d}/{len(t)}  "
              f"naive {res[-1]['naive_diff']:+6.2f}  ATT {diff.mean():+6.2f} "
              f"[{ci[0]:+.2f}, {ci[1]:+.2f}] pp/yr")

    res = pd.DataFrame(res).sort_values("att")
    bal = pd.DataFrame(bal)
    res.round(3).to_csv(cfg.RESULTS_DIR / f"park_effect_att_{SUFFIX}.csv", index=False)
    bal.round(3).to_csv(cfg.RESULTS_DIR / f"covariate_balance_{SUFFIX}.csv", index=False)

    print("\nMax |SMD| after matching (target < 0.1):")
    print(bal.groupby("park").smd_after.apply(lambda s: s.abs().max()).round(3))

    # figure: naive vs matched effect
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(res))
    ax.axvline(0, color="grey", lw=0.8)
    ax.scatter(res.naive_diff, y, marker="o", facecolors="none", edgecolors="grey",
               label="Naive (park - ring)")
    ax.errorbar(res.att, y, xerr=[res.att - res.ci_low, res.ci_high - res.att],
                fmt="o", color="#b2182b", capsize=3, label="Matched ATT (95% CI)")
    ax.set_yticks(y, res.park)
    ax.set_xlabel("Difference in annual burn probability (percentage points)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / f"fig_park_effect_{SUFFIX}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
