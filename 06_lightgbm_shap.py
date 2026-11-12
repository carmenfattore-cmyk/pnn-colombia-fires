"""
PNN Colombia fire study - Step 6 (Spyder): robustness check with LightGBM + SHAP.

Same data, outcome and spatial cross-validation as the GAM (step 5), but a
gradient-boosting model that can capture interactions without assuming
additivity. If both models rank the same drivers and show the same shapes,
the GAM results are robust.

Inputs : data/pnn_sample_s2fire.csv
Outputs: results/lgbm_cv_auc.csv, results/shap_importance.csv,
         figs/fig_shap_summary.png, figs/fig_shap_dependence.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every PC
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config as cfg

YEARS = list(range(2017, cfg.YEAR_END + 1))
BLOCK_KM = 10
N_SHAP = 20000          # point-years used to compute SHAP values
LC_GROUP = {10: "forest", 20: "shrub/grass", 30: "shrub/grass", 90: "wetland",
            95: "wetland", 40: "agri/built", 50: "agri/built", 60: "bare/sparse",
            70: "bare/sparse", 100: "shrub/grass"}
NUM = ["elevation", "slope", "log_travel", "precip_mm_yr"]
CAT = ["lc", "year", "park_zone"]
LABELS = {"elevation": "Elevation", "slope": "Slope", "log_travel": "Travel time (log)",
          "precip_mm_yr": "Mean annual rainfall", "lc": "Land cover", "year": "Year",
          "park_zone": "Park x inside/outside"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              n_estimators=600, verbose=-1, random_state=cfg.SEED)


def load_long():
    d = pd.read_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv")
    d = d[d.worldcover != 80].dropna(subset=["elevation", "slope", "travel_time_min",
                                              "precip_mm_yr", "worldcover"])
    d["lc"] = d.worldcover.astype(int).map(LC_GROUP).fillna("bare/sparse")
    d["log_travel"] = np.log1p(d.travel_time_min)
    d["park_zone"] = d.park + " | " + d.zone
    deg = BLOCK_KM / 111.0
    d["block"] = (np.floor(d.lon / deg).astype(int).astype(str) + "_" +
                  np.floor(d.lat / deg).astype(int).astype(str))
    long = d.melt(id_vars=["pid", "block", "lc", "park_zone"] + NUM,
                  value_vars=[f"f{y}" for y in YEARS], var_name="year", value_name="fire")
    long = long.dropna(subset=["fire"])
    long["year"] = long.year.str[1:]
    for c in CAT:
        long[c] = long[c].astype("category")
    return long


def main():
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    cfg.FIG_DIR.mkdir(exist_ok=True)
    long = load_long()
    X, y = long[NUM + CAT], long.fire.astype(int).to_numpy()
    print(f"{len(long):,} point-years, fire prevalence {y.mean() * 100:.2f}%")

    # --- spatial CV (same 10 km blocks as the GAM) ---------------------------
    cv = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, long.block)):
        m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr])
        cv.append({"fold": k + 1, "auc": roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1])})
        print(f"  fold {k + 1}: AUC = {cv[-1]['auc']:.3f}")
    cv = pd.DataFrame(cv)
    cv.round(4).to_csv(cfg.RESULTS_DIR / "lgbm_cv_auc.csv", index=False)
    print(f"Spatial CV AUC (LightGBM): {cv.auc.mean():.3f} +/- {cv.auc.std():.3f}")

    # --- final model and SHAP --------------------------------------------------
    model = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    idx = np.random.default_rng(cfg.SEED).choice(len(X), size=min(N_SHAP, len(X)), replace=False)
    Xs = X.iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv          # log-odds contributions

    imp = (pd.DataFrame({"feature": NUM + CAT, "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False))
    imp["share_pct"] = 100 * imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp.round(4).to_csv(cfg.RESULTS_DIR / "shap_importance.csv", index=False)
    print("\nSHAP importance (mean |SHAP|, log-odds):")
    print(imp.round(3).to_string(index=False))

    # importance bar chart
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([LABELS[f] for f in imp.feature][::-1], imp.mean_abs_shap[::-1], color="#b2182b")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig_shap_summary.png", dpi=300)

    # dependence plots for the 4 continuous drivers
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, f in zip(axes, NUM):
        j = (NUM + CAT).index(f)
        x = Xs[f].to_numpy()
        ax.scatter(x, sv[:, j], s=2, alpha=0.15, color="#2166ac", rasterized=True)
        # running median to show the shape
        bins = np.quantile(x, np.linspace(0.01, 0.99, 30))
        mid = 0.5 * (bins[1:] + bins[:-1])
        med = [np.median(sv[(x >= a) & (x < b), j]) if ((x >= a) & (x < b)).any() else np.nan
               for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mid, med, color="#b2182b", lw=2)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(LABELS[f])
        ax.set_ylabel("SHAP value (log-odds)")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "fig_shap_dependence.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
