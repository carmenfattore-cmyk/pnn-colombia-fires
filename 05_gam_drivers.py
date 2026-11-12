"""
PNN Colombia fire study - Step 5 (Spyder): drivers of fire occurrence with a GAM.

Unit of analysis: point-year (Sentinel-2 fire confirmed by VIIRS/MODIS, 2017-2025).
Logistic GAM:

  logit P(fire) = s(elevation) + s(slope) + s(log travel time) + s(rainfall)
                  + f(land cover) + f(year) + f(park x inside/outside park)

- the smooth terms show HOW the fire probability changes along each gradient;
- f(year) captures inter-annual variability (El Nino years, 2018);
- f(park x zone) gives the inside/outside contrast for each park, net of the gradients.

Validation: 5-fold SPATIAL cross-validation (10 km blocks), AUC.
Note: pointwise CIs from pygam assume independent observations; they are
indicative only (points are spatially clustered). The park effect is tested
formally by the matching (step 2).

Inputs : data/pnn_sample_s2fire.csv
Outputs: results/gam_summary.txt, results/gam_cv_auc.csv,
         results/gam_factor_effects.csv, figs/Fig6_gam_partial_effects.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every PC
from pygam import LogisticGAM, s, f
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config as cfg

YEARS = list(range(2017, cfg.YEAR_END + 1))
BLOCK_KM = 10
N_SPLINES = 12
LC_GROUP = {10: "forest", 20: "shrub/grass", 30: "shrub/grass", 90: "wetland",
            95: "wetland", 40: "agri/built", 50: "agri/built", 60: "bare/sparse",
            70: "bare/sparse", 100: "shrub/grass"}
SMOOTHS = [("elevation", "Elevation (m)"), ("slope", "Slope (deg)"),
           ("log_travel", "Travel time to cities (log min)"),
           ("precip_mm_yr", "Mean annual rainfall (mm)")]


def load_long():
    d = pd.read_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv")
    d = d[d.worldcover != 80].dropna(subset=["elevation", "slope", "travel_time_min",
                                              "precip_mm_yr", "worldcover"])
    d["lc"] = d.worldcover.astype(int).map(LC_GROUP).fillna("bare/sparse")
    d["log_travel"] = np.log1p(d.travel_time_min)
    d["eco_zone"] = d.park + " | " + d.zone          # park x inside/outside
    deg = BLOCK_KM / 111.0
    d["block"] = (np.floor(d.lon / deg).astype(int).astype(str) + "_" +
                  np.floor(d.lat / deg).astype(int).astype(str))
    long = d.melt(id_vars=["pid", "park", "lc", "eco_zone", "block"] + [c for c, _ in SMOOTHS],
                  value_vars=[f"f{y}" for y in YEARS], var_name="year", value_name="fire")
    long = long.dropna(subset=["fire"])
    long["year"] = long.year.str[1:].astype(int)
    return long


def design(long, cats):
    X = long[[c for c, _ in SMOOTHS]].copy()
    for c in ["lc", "year", "eco_zone"]:
        X[c] = pd.Categorical(long[c], categories=cats[c]).codes
    return X.to_numpy(float)


def make_gam(lam=None):
    terms = (s(0, n_splines=N_SPLINES) + s(1, n_splines=N_SPLINES) +
             s(2, n_splines=N_SPLINES) + s(3, n_splines=N_SPLINES) + f(4) + f(5) + f(6))
    return LogisticGAM(terms, lam=lam if lam is not None else 0.6)


def main():
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    cfg.FIG_DIR.mkdir(exist_ok=True)
    long = load_long()
    cats = {c: sorted(long[c].unique()) for c in ["lc", "year", "eco_zone"]}
    X, y = design(long, cats), long.fire.to_numpy()
    print(f"{len(long):,} point-years, fire prevalence {y.mean() * 100:.2f}%")

    # --- smoothing parameter chosen by SPATIAL CV on a 25% subsample ----------
    # (GCV/UBRE pick the least smoothing with 370k correlated rows -> wiggly curves)
    rng = np.random.default_rng(cfg.SEED)
    sub = np.where(rng.random(len(y)) < 0.25)[0]
    scores = {}
    for lam_c in [1, 10, 100, 1000, 10000]:
        aucs = []
        for tr, te in GroupKFold(n_splits=3).split(X[sub], y[sub], long.block.iloc[sub]):
            g = make_gam(lam_c).fit(X[sub][tr], y[sub][tr])
            aucs.append(roc_auc_score(y[sub][te], g.predict_proba(X[sub][te])))
        scores[lam_c] = np.mean(aucs)
        print(f"  lambda {lam_c:>6}: spatial-CV AUC {scores[lam_c]:.4f}")
    best = max(scores.values())
    # largest lambda (smoothest curves) within 0.002 AUC of the best
    lam = max(l for l, a in scores.items() if a >= best - 0.002)
    print("Selected lambda:", lam)

    # --- spatial cross-validation --------------------------------------------
    cv = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, long.block)):
        g = make_gam(lam).fit(X[tr], y[tr])
        cv.append({"fold": k + 1, "auc": roc_auc_score(y[te], g.predict_proba(X[te])),
                   "n_test": len(te)})
        print(f"  fold {k + 1}: AUC = {cv[-1]['auc']:.3f}")
    cv = pd.DataFrame(cv)
    cv.round(4).to_csv(cfg.RESULTS_DIR / "gam_cv_auc.csv", index=False)
    print(f"Spatial CV AUC: {cv.auc.mean():.3f} +/- {cv.auc.std():.3f}")

    # --- final model ----------------------------------------------------------
    gam = make_gam(lam).fit(X, y)
    with open(cfg.RESULTS_DIR / "gam_summary.txt", "w") as fh:
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gam.summary()
        fh.write(buf.getvalue())
        fh.write(f"\nSpatial CV AUC: {cv.auc.mean():.3f} +/- {cv.auc.std():.3f}\n")
    print(f"Deviance explained: {gam.statistics_['pseudo_r2']['explained_deviance']:.3f}")

    # factor effects (log-odds, relative to the mean partial effect of each term)
    rows = []
    for ti, name in [(4, "lc"), (5, "year"), (6, "eco_zone")]:
        XX = gam.generate_X_grid(term=ti)
        pdep, ci = gam.partial_dependence(term=ti, X=XX, width=0.95)
        for code, val, lo, hi in zip(XX[:, ti].astype(int), pdep, ci[:, 0], ci[:, 1]):
            if 0 <= code < len(cats[name]):
                rows.append({"term": name, "level": cats[name][code],
                             "log_odds": val, "ci_low": lo, "ci_high": hi,
                             "odds_ratio": np.exp(val)})
    fac = pd.DataFrame(rows).drop_duplicates(["term", "level"])
    fac.round(3).to_csv(cfg.RESULTS_DIR / "gam_factor_effects.csv", index=False)
    print(fac[fac.term != "lc"].round(2).to_string(index=False))

    # --- figure: 4 smooths + year + ecosystem x zone ---------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for i, (col, label) in enumerate(SMOOTHS):
        ax = axes.flat[i]
        XX = gam.generate_X_grid(term=i)
        pdep, ci = gam.partial_dependence(term=i, X=XX, width=0.95)
        # centre on the mean partial effect over the data (curves show relative change)
        centre = gam.partial_dependence(term=i, X=X[::50]).mean()
        pdep, ci = pdep - centre, ci - centre
        q = np.quantile(long[col], [0.01, 0.99])
        vis = (XX[:, i] >= q[0]) & (XX[:, i] <= q[1])
        ax.set_ylim(ci[vis].min() - 0.3, ci[vis].max() + 0.3)
        ax.plot(XX[:, i], pdep, color="#b2182b")
        ax.fill_between(XX[:, i], ci[:, 0], ci[:, 1], color="#b2182b", alpha=0.2, lw=0)
        ax.set_xlim(q)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel("Partial effect (log-odds)")
    ax = axes.flat[4]
    yr = fac[fac.term == "year"]
    ax.errorbar(yr.level.astype(int), yr.log_odds,
                yerr=[yr.log_odds - yr.ci_low, yr.ci_high - yr.log_odds],
                fmt="o-", color="#2166ac", capsize=2)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Partial effect (log-odds)")
    ax = axes.flat[5]
    ez = fac[fac.term == "eco_zone"].sort_values("level")
    yy = np.arange(len(ez))
    colors = ["#1b7837" if "park" in l else "#999999" for l in ez.level]
    for yi, (_, r), c in zip(yy, ez.iterrows(), colors):
        ax.errorbar(r.log_odds, yi, xerr=[[r.log_odds - r.ci_low], [r.ci_high - r.log_odds]],
                    fmt="o", color=c, capsize=2)
    ax.set_yticks(yy, ez.level, fontsize=7)
    ax.set_xlabel("Partial effect (log-odds)")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "Fig6_gam_partial_effects.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
