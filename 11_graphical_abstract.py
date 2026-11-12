"""PNN Colombia fire study - graphical abstract (JEM): MODIS omission and park effect, S2 vs MODIS."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every PC

import config as cfg

ORDER = ["Isla de Salamanca", "Sierra Nevada de Santa Marta", "El Cocuy", "Sumapaz",
         "El Tuparro", "Sierra de la Macarena", "Tinigua", "Chiribiquete"]
SHORT = {"Sierra Nevada de Santa Marta": "S. Nevada de S. Marta", "Sierra de la Macarena": "S. de la Macarena"}
YEARS = range(2017, 2026)


def main():
    d = pd.read_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv")
    miss = {}
    for p, g in d.groupby("park"):
        f = pd.concat([g[f"b{y}"][g[f"f{y}"] == 1] for y in YEARS])
        miss[p] = 100 * (f == 0).mean()
    s2 = pd.read_csv(cfg.RESULTS_DIR / "park_effect_att_s2.csv").set_index("park")
    mo = pd.read_csv(cfg.RESULTS_DIR / "park_effect_att_modis.csv").set_index("park")

    fig = plt.figure(figsize=(10, 4.0))
    fig.text(0.5, 0.95, "Small fires change the verdict on protected-area effectiveness",
             ha="center", fontsize=13, weight="bold", color="#1d3b3a")
    y = np.arange(len(ORDER))[::-1]
    lab = [SHORT.get(p, p) for p in ORDER]

    ax = fig.add_axes([0.17, 0.12, 0.3, 0.7])
    v = [miss[p] for p in ORDER]
    ax.barh(y, v, color="#c0392b", height=0.6)
    for yi, vi in zip(y, v):
        ax.text(vi + 1.5, yi, f"{vi:.0f}%", va="center", fontsize=8.5)
    ax.set_yticks(y, lab, fontsize=8.5)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.6, len(ORDER) + 0.7)
    ax.set_xlabel("Sentinel-2 + VIIRS fires missed by MODIS (%)", fontsize=9)
    ax.set_title("1  MODIS misses 46% of fires", fontsize=10.5, loc="left", weight="bold", color="#c0392b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = fig.add_axes([0.6, 0.12, 0.36, 0.7])
    ax.axvline(0, color="#888888", lw=0.8)
    for off, t, c, mk, lb in [(0.15, s2, "#1d7874", "o", "Sentinel-2 + VIIRS/MODIS"),
                              (-0.15, mo, "#8e979f", "s", "MODIS only")]:
        for k, p in enumerate(ORDER):
            r = t.loc[p]
            sig = r.ci_low > 0 or r.ci_high < 0
            ax.plot([r.ci_low, r.ci_high], [y[k] + off] * 2, color=c, lw=1.4)
            ax.plot(r.att, y[k] + off, mk, color=c, mfc=c if sig else "white", ms=6,
                    label=lb if k == 0 else None)
    ax.set_yticks(y, [""] * len(y))
    ax.set_ylim(-0.6, len(ORDER) + 0.7)
    ax.set_xlabel("Park effect on annual burn probability (pp yr$^{-1}$)", fontsize=9)
    ax.set_title("2  Matched park effect depends on the product", fontsize=10.5, loc="left",
                 weight="bold", color="#1d7874")
    ax.text(0.02, 0.02, "filled = significant (95% CI)", transform=ax.transAxes, fontsize=7.5,
            color="#555555")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2, handletextpad=0.3, columnspacing=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(cfg.FIG_DIR / "Graphical_abstract.png", dpi=300)
    print({k: round(v, 1) for k, v in miss.items()})


if __name__ == "__main__":
    main()
