"""
PNN Colombia fire study - Step 7 (Spyder): manuscript figures.

Fig. 1  Study area: parks, 10 km external rings, ecosystems.
Fig. 2  Workflow (separate script 08_flowchart.py).
Fig. 3  Annual burned share inside/outside each park: Sentinel-2 (confirmed) vs MODIS,
        with the share of Sentinel-2 fires missed by MODIS.
Fig. 5  Park effect after matching (ATT), Sentinel-2 vs MODIS outcome.
(Figs. 4, S2 and S3 = 09_extra_figures.py; Fig. 6 = figs/fig_gam_partial_effects.png from step 5;
 Supplementary = figs/fig_shap_*.png from step 6.)

Inputs : data/pnn_zones.gpkg, official boundaries in col-administrative-divisions-shapefiles/,
         data/ne/naturalearth_lowres.shp (inset only), data/pnn_sample_s2fire.csv,
         results/park_effect_att_s2.csv, results/park_effect_att_modis.csv
Outputs: figs/Fig1_study_area.png, figs/Fig3_burned_s2_vs_modis.png, figs/Fig5_park_effect.png
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every PC
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config as cfg

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
YEARS = list(range(2017, cfg.YEAR_END + 1))
ECO_COL = {"Orinoquia savanna": "#e6ab02", "Andean paramo": "#7570b3",
           "Amazon deforestation arc": "#1b9e77",
           "Caribbean mountain": "#d95f02", "Caribbean coast (mangrove/wetland)": "#1f78b4"}
SHORT = {"Sierra Nevada de Santa Marta": "Sierra Nevada de S. Marta",
         "Sierra de la Macarena": "Sierra de la Macarena"}
# order: by ecosystem, north to south
ORDER = ["Isla de Salamanca", "Sierra Nevada de Santa Marta", "El Cocuy", "Sumapaz",
         "El Tuparro", "Sierra de la Macarena", "Tinigua", "Chiribiquete"]
C_IN, C_OUT = "#1b7837", "#999999"


ADMIN_DIR = cfg.BASE_DIR / "col-administrative-divisions-shapefiles"


def _scalebar(ax, lon0, lat0, km=200):
    """Simple scale bar in km at latitude lat0 (geographic axes)."""
    dlon = km / (111.32 * np.cos(np.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color="k", lw=2, solid_capstyle="butt")
    ax.plot([lon0, lon0 + dlon / 2], [lat0, lat0], color="white", lw=1, solid_capstyle="butt")
    ax.text(lon0, lat0 - 0.25, "0", ha="center", va="top", fontsize=7)
    ax.text(lon0 + dlon, lat0 - 0.25, f"{km} km", ha="center", va="top", fontsize=7)


def fig1():
    zones = gpd.read_file(cfg.DATA_DIR / "pnn_zones.gpkg").to_crs(4326)
    zones["geometry"] = zones.geometry.simplify(0.002)
    adm0 = gpd.read_file(ADMIN_DIR / "col_admbnda_adm0_mgn_itos_20200416.shp").to_crs(4326)
    adm1 = gpd.read_file(ADMIN_DIR / "col_admbnda_adm1_mgn_20200416.shp").to_crs(4326)
    adm0["geometry"] = adm0.geometry.simplify(0.005)
    adm1["geometry"] = adm1.geometry.simplify(0.01)

    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.set_facecolor("#e3eef6")                       # sea and neighbouring countries
    adm1.plot(ax=ax, color="#f7f5f0", edgecolor="#c8c3b8", lw=0.3)
    adm0.boundary.plot(ax=ax, color="#4d4d4d", lw=0.7)
    zones[zones.zone == "buffer"].plot(ax=ax, color="#9e9e9e", alpha=0.45, edgecolor="none")
    zones[zones.zone == "buffer"].boundary.plot(ax=ax, color="#5a5a5a", lw=0.35)
    parks = zones[zones.zone == "park"]
    parks.plot(ax=ax, color=[ECO_COL[e] for e in parks.ecosystem], edgecolor="k", lw=0.4)

    # park names with leader lines: label position (lon, lat)
    lab = {"Isla de Salamanca": (-77.7, 11.9), "Sierra Nevada de Santa Marta": (-72.7, 12.0),
           "El Cocuy": (-71.1, 7.5), "Sumapaz": (-77.5, 4.7), "El Tuparro": (-70.4, 6.7),
           "Sierra de la Macarena": (-78.4, 3.3), "Tinigua": (-77.5, 1.8),
           "Chiribiquete": (-71.2, 1.9)}
    for p, (tx, ty) in lab.items():
        g = parks[parks.park == p].geometry.iloc[0]
        x, y = g.representative_point().coords[0]
        ax.annotate(SHORT.get(p, p), (x, y), xytext=(tx, ty), ha="left", va="center",
                    fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.5, color="k",
                                                  shrinkA=1, shrinkB=0))
    for txt, x, y in [("Caribbean Sea", -75.4, 12.7), ("Pacific\nOcean", -79.0, 5.8),
                      ("VENEZUELA", -69.8, 9.3), ("BRAZIL", -68.2, -1.8),
                      ("ECUADOR", -76.2, -1.7), ("PANAMA", -78.9, 8.4)]:
        ax.text(x, y, txt, fontsize=6.5, color="#6b7f8e", style="italic", ha="center")

    ax.set_xlim(-80, -66.5)
    ax.set_ylim(-4.5, 13)
    ax.set_xticks(range(-80, -65, 4))
    ax.set_xticklabels([f"{abs(v)}\u00b0W" for v in range(-80, -65, 4)])
    ax.set_yticks(range(-4, 14, 4))
    ax.set_yticklabels([f"{abs(v)}\u00b0{'S' if v < 0 else 'N'}" if v else "0\u00b0"
                        for v in range(-4, 14, 4)])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    ax.annotate("N", xy=(-67.4, 12.4), xytext=(-67.4, 11.3), ha="center", fontsize=9,
                weight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=1))
    _scalebar(ax, -75.0, -3.9)
    handles = [Patch(facecolor=c, edgecolor="k", lw=0.4, label=e) for e, c in ECO_COL.items()]
    handles.append(Patch(facecolor="#c4c4c4", edgecolor="#5a5a5a", lw=0.35,
                         label="10 km external ring"))
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=True, framealpha=0.95)

    # inset: Colombia in South America
    ins = ax.inset_axes([0.015, 0.015, 0.19, 0.25])
    world = gpd.read_file(cfg.DATA_DIR / "ne" / "naturalearth_lowres.shp")
    world[world.continent == "South America"].plot(ax=ins, color="#d9d9d9", edgecolor="white", lw=0.3)
    adm0.plot(ax=ins, color="#4d4d4d")
    ins.set_xlim(-82, -34)
    ins.set_ylim(-56, 13)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_xlabel("")
    ins.set_ylabel("")
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "Fig1_study_area.png", dpi=300)


def fig2():
    d = pd.read_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv")
    fcols, bcols = [f"f{y}" for y in YEARS], [f"b{y}" for y in YEARS]
    d["s2"] = d[fcols].mean(axis=1) * 100
    d["modis"] = d[bcols].mean(axis=1) * 100
    r = d.groupby(["park", "zone"])[["s2", "modis"]].mean().unstack()
    F, B = d[fcols].to_numpy(), d[bcols].to_numpy()
    missed = {}
    for p in ORDER:
        m = (d.park == p).to_numpy()
        fire = (F[m] == 1)
        missed[p] = 100 * (fire & (B[m] == 0)).sum() / max(fire.sum(), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(ORDER))[::-1]
    h = 0.36
    ax.barh(y + h / 2, r.loc[ORDER, ("s2", "park")], h, color=C_IN, label="Inside park (Sentinel-2)")
    ax.barh(y - h / 2, r.loc[ORDER, ("s2", "buffer")], h, color=C_OUT, label="External ring (Sentinel-2)")
    ax.scatter(r.loc[ORDER, ("modis", "park")], y + h / 2, marker="|", s=120, color="k", zorder=3)
    ax.scatter(r.loc[ORDER, ("modis", "buffer")], y - h / 2, marker="|", s=120, color="k", zorder=3)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks(y, [SHORT.get(p, p) for p in ORDER])
    ax.set_xlabel("Points burned per year, 2017-2025 (%)")
    for yi, p in zip(y, ORDER):
        ax.text(29, yi, f"{missed[p]:.0f}%", va="center", ha="right", fontsize=8, color="#b2182b")
    ax.set_ylim(-0.7, y[0] + 1.1)
    ax.text(29, y[0] + 0.55, "S2 fires missed\nby MODIS", ha="right", va="bottom", fontsize=7,
            color="#b2182b")
    handles = [Patch(color=C_IN, label="Inside park (Sentinel-2)"),
               Patch(color=C_OUT, label="External ring (Sentinel-2)"),
               Line2D([], [], marker="|", ls="none", color="k", ms=10, label="MODIS MCD64A1")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3,
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "Fig3_burned_s2_vs_modis.png", dpi=300)
    return r, missed


def fig3():
    s2 = pd.read_csv(cfg.RESULTS_DIR / "park_effect_att_s2.csv").set_index("park").loc[ORDER]
    mo = pd.read_csv(cfg.RESULTS_DIR / "park_effect_att_modis.csv").set_index("park").loc[ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(ORDER))[::-1]
    ax.axvline(0, color="grey", lw=0.8)
    ax.errorbar(mo.att, y - 0.15, xerr=[mo.att - mo.ci_low, mo.ci_high - mo.att], fmt="o",
                mfc="white", color="#636363", capsize=2, label="MODIS MCD64A1 (2016-2025)")
    ax.errorbar(s2.att, y + 0.15, xerr=[s2.att - s2.ci_low, s2.ci_high - s2.att], fmt="o",
                color="#b2182b", capsize=2, label="Sentinel-2 confirmed (2017-2025)")
    ax.set_yticks(y, [SHORT.get(p, p) for p in ORDER])
    ax.set_xlabel("Park effect on annual burn probability (percentage points)\n"
                  "matched park points - matched ring points")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "Fig5_park_effect.png", dpi=300)


def main():
    cfg.FIG_DIR.mkdir(exist_ok=True)
    fig1()
    r, missed = fig2()
    fig3()
    print(r.round(2))
    print("Sentinel-2 fire point-years missed by MODIS (%):",
          {k: round(v, 1) for k, v in missed.items()})
    plt.show()


if __name__ == "__main__":
    main()
