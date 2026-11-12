"""
PNN Colombia fire study - Step 9 (Spyder): additional figures.

Fig. 4  Isla de Salamanca: small fires confirmed by Sentinel-2 + VIIRS and missed by MODIS
        (maps for 2020 and 2024 + annual counts 2017-2025).
Fig. S2 Annual share of points burned per ecosystem, park vs 10 km ring (Sentinel-2 confirmed),
        with MODIS for comparison.
Fig. S3 Covariate balance before and after matching (standardised mean difference).

Inputs : data/pnn_zones.gpkg, data/pnn_sample_s2fire.csv, data/firms (VIIRS archive),
         official boundaries in col-administrative-divisions-shapefiles/,
         results/covariate_balance_s2.csv
Outputs: figs/Fig4_salamanca_small_fires.png, figs/FigS2_annual_series.png,
         figs/FigS3_covariate_balance.png
"""
from pathlib import Path

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
ECO_LABEL = {"Orinoquia savanna": "(a) Orinoquia savanna\nEl Tuparro",
             "Andean paramo": "(b) Andean paramo\nSumapaz, El Cocuy",
             "Amazon deforestation arc": "(c) Amazon deforestation arc\nMacarena, Tinigua, Chiribiquete",
             "Caribbean mountain": "(d) Caribbean mountain\nSierra Nevada de Santa Marta",
             "Caribbean coast (mangrove/wetland)": "(e) Caribbean coast\nIsla de Salamanca"}
ORDER = ["Isla de Salamanca", "Sierra Nevada de Santa Marta", "El Cocuy", "Sumapaz",
         "El Tuparro", "Sierra de la Macarena", "Tinigua", "Chiribiquete"]
ADMIN_DIR = cfg.BASE_DIR / "col-administrative-divisions-shapefiles"
RED, BLUE, VIIRS_C = "#c0392b", "#1f4e79", "#f39c12"


def load_viirs():
    files = list((cfg.DATA_DIR / "firms").glob("*.csv")) + list(cfg.BASE_DIR.glob("DL_FIRE_*/*.csv"))
    f = pd.concat([pd.read_csv(p, usecols=["latitude", "longitude", "acq_date"]) for p in files])
    f = f.drop_duplicates()
    f["year"] = pd.to_datetime(f.acq_date).dt.year
    return f


# ---------------------------------------------------------------------------
def fig_salamanca(d, zones):
    park = "Isla de Salamanca"
    z = zones[zones.park == park].to_crs(4326)
    x0, y0, x1, y1 = z.total_bounds
    pad = 0.03
    adm0 = gpd.read_file(ADMIN_DIR / "col_admbnda_adm0_mgn_itos_20200416.shp").to_crs(4326)
    adm0 = adm0.clip((x0 - 0.2, y0 - 0.2, x1 + 0.2, y1 + 0.2))
    v = load_viirs()
    v = v[v.longitude.between(x0, x1) & v.latitude.between(y0, y1)]
    vg = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.longitude, v.latitude), crs=4326)
    vg = vg[vg.within(z.union_all())]
    s = d[d.park == park]

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.3, wspace=0.05)
    for k, yr in enumerate([2020, 2024]):
        ax = fig.add_subplot(gs[0, k])
        ax.set_facecolor("#e3eef6")
        adm0.plot(ax=ax, color="#f7f5f0", edgecolor="#9a958c", lw=0.4, aspect=None)
        z[z.zone == "buffer"].plot(ax=ax, color="#9e9e9e", alpha=0.35, edgecolor="none", aspect=None)
        z[z.zone == "park"].explode().boundary.plot(ax=ax, color="#1b7837", lw=1.0, aspect=None)
        ok = s[f"f{yr}"].notna()
        nb = s[ok & (s[f"f{yr}"] == 0)]
        s2only = s[(s[f"f{yr}"] == 1) & (s[f"b{yr}"] == 0)]
        both = s[(s[f"f{yr}"] == 1) & (s[f"b{yr}"] == 1)]
        ax.scatter(nb.lon, nb.lat, s=1.2, color="#b5b5b5", lw=0, zorder=2)
        vy = vg[vg.year == yr]
        ax.scatter(vy.longitude, vy.latitude, s=5, marker="x", color=VIIRS_C, lw=0.6, zorder=3)
        ax.scatter(s2only.lon, s2only.lat, s=9, color=RED, lw=0, zorder=4)
        ax.scatter(both.lon, both.lat, s=11, facecolor="none", edgecolor=BLUE, lw=0.8, zorder=5)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#888888")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"({'ab'[k]}) {yr}", fontsize=8.5, loc="left")
        ax.text(0.98, 0.04, f"Sentinel-2 fire points: {len(s2only) + len(both)}\n"
                f"detected by MODIS: {len(both)}\nVIIRS hotspots: {len(vy)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(fc="white", ec="#bbbbbb", lw=0.5, alpha=0.9))
        if k == 0:
            ax.text(-74.62, 11.07, "Caribbean Sea", fontsize=7, style="italic", color="#4a6f8a")
            dl = 10 / (111.32 * np.cos(np.radians(10.9)))
            ax.plot([x0, x0 + dl], [y0 + 0.005, y0 + 0.005], color="k", lw=2, solid_capstyle="butt")
            ax.text(x0 + dl / 2, y0 + 0.015, "10 km", ha="center", va="bottom", fontsize=7)
    handles = [Line2D([], [], marker="o", ls="", color=RED, ms=4,
                      label="Sentinel-2 confirmed fire, missed by MODIS"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, ms=4.5,
                      label="Sentinel-2 confirmed fire, detected by MODIS"),
               Line2D([], [], marker="x", ls="", color=VIIRS_C, ms=4, label="VIIRS 375 m hotspot"),
               Line2D([], [], marker="o", ls="", color="#b5b5b5", ms=2.5, label="No fire"),
               Line2D([], [], color="#1b7837", lw=1.0, label="Park boundary"),
               Patch(facecolor="#d0d0d0", label="10 km external ring")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.0))

    ax = fig.add_subplot(gs[1, :])
    n_s2 = [(s[f"f{y}"] == 1).sum() for y in YEARS]
    n_both = [((s[f"f{y}"] == 1) & (s[f"b{y}"] == 1)).sum() for y in YEARS]
    n_v = [(vg.year == y).sum() for y in YEARS]
    xx = np.arange(len(YEARS))
    ax.bar(xx, n_s2, 0.62, color=RED, label="Sentinel-2 confirmed fire points")
    ax.bar(xx, n_both, 0.62, color=BLUE, label="of which detected by MODIS")
    ax.set_xticks(xx, YEARS)
    ax.set_ylabel("Fire point-years\n(park and ring)")
    ax2 = ax.twinx()
    ax2.plot(xx, n_v, color=VIIRS_C, marker="x", lw=1.2, label="VIIRS hotspots")
    ax2.set_ylabel("VIIRS hotspots", color="#b9770e")
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, max(n_v) * 1.15)
    ax.set_ylim(0, max(n_s2) * 1.15)
    ax.set_title("(c) Isla de Salamanca and ring, 2017–2025", fontsize=8.5, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.06)
    fig.savefig(cfg.FIG_DIR / "Fig4_salamanca_small_fires.png", dpi=300)
    return pd.DataFrame({"year": YEARS, "s2": n_s2, "s2_and_modis": n_both, "viirs": n_v})


# ---------------------------------------------------------------------------
def annual_rates(d):
    rows = []
    for (eco, zone), g in d.groupby(["ecosystem", "zone"]):
        for y in YEARS:
            ok = g[f"f{y}"].notna()
            rows.append(dict(ecosystem=eco, zone=zone, year=y,
                             s2=100 * g.loc[ok, f"f{y}"].mean(),
                             modis=100 * g[f"b{y}"].mean()))
    return pd.DataFrame(rows)


def fig_series(d):
    r = annual_rates(d)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.4), sharex=True)
    ecos = list(ECO_COL)
    for k, ax in enumerate(axes.flat):
        if k == 5:
            ax.axis("off")
            break
        eco = ecos[k]
        c = ECO_COL[eco]
        g = r[r.ecosystem == eco]
        p, b = g[g.zone == "park"], g[g.zone == "buffer"]
        for yy in (2018, 2020, 2022):
            ax.axvspan(yy - 0.4, yy + 0.4, color="#f2f2f2", zorder=0)
        ax.plot(p.year, p.s2, color=c, lw=1.8, marker="o", ms=3.2, label="Park, Sentinel-2")
        ax.plot(b.year, b.s2, color=c, lw=1.3, ls="--", marker="o", ms=3.2, mfc="white",
                label="Ring, Sentinel-2")
        ax.plot(p.year, p.modis, color="#555555", lw=0.8, label="Park, MODIS")
        ax.plot(b.year, b.modis, color="#555555", lw=0.8, ls="--", label="Ring, MODIS")
        ax.set_title(ECO_LABEL[eco], fontsize=7.6, loc="left", linespacing=1.25)
        ax.set_ylim(bottom=0)
        ax.set_xticks([2017, 2019, 2021, 2023, 2025])
        ax.tick_params(labelsize=7.5)
        if k % 3 == 0:
            ax.set_ylabel("Points burned (% per year)")
    handles = [Line2D([], [], color="k", lw=1.8, marker="o", ms=3.2, label="Park, Sentinel-2 confirmed"),
               Line2D([], [], color="k", lw=1.3, ls="--", marker="o", ms=3.2, mfc="white",
                      label="10 km ring, Sentinel-2 confirmed"),
               Line2D([], [], color="#555555", lw=0.8, label="Park, MODIS MCD64A1"),
               Line2D([], [], color="#555555", lw=0.8, ls="--", label="10 km ring, MODIS MCD64A1"),
               Patch(facecolor="#f2f2f2", label="High fire-probability years\nin the GAM (2018, 2020, 2022)")]
    axes.flat[5].legend(handles=handles, loc="center", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "FigS2_annual_series.png", dpi=300)
    return r


# ---------------------------------------------------------------------------
def fig_balance():
    b = pd.read_csv(cfg.RESULTS_DIR / "covariate_balance_s2.csv")
    cov_lab = {"elevation": "Elevation", "slope": "Slope", "travel_time_min": "Travel time",
               "precip_mm_yr": "Rainfall"}
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 3.3), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, (cv, lab) in zip(axes, cov_lab.items()):
        g = b[b.covariate == cv].set_index("park").loc[ORDER]
        ax.axvspan(-0.1, 0.1, color="#e8f3ee", zorder=0)
        ax.axvline(0, color="#999999", lw=0.6)
        for yi, (bef, aft) in zip(y, zip(g.smd_before.abs(), g.smd_after.abs())):
            ax.plot([bef, aft], [yi, yi], color="#cccccc", lw=0.8, zorder=1)
        ax.scatter(g.smd_before.abs(), y, s=22, facecolor="white", edgecolor="#555555", lw=0.9,
                   zorder=2, label="Before matching")
        ax.scatter(g.smd_after.abs(), y, s=22, color="#1d7874", zorder=3, label="After matching")
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlim(-0.05, max(1.9, b.smd_before.abs().max() + 0.1))
        ax.tick_params(labelsize=7.5)
    axes[0].set_yticks(y, ["S. Nevada de S. Marta" if p.startswith("Sierra Nevada") else p
                           for p in ORDER])
    fig.supxlabel("Absolute standardised mean difference, park vs ring points", fontsize=8.5)
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(cfg.FIG_DIR / "FigS3_covariate_balance.png", dpi=300)


def main():
    cfg.FIG_DIR.mkdir(exist_ok=True)
    d = pd.read_csv(cfg.DATA_DIR / "pnn_sample_s2fire.csv")
    zones = gpd.read_file(cfg.DATA_DIR / "pnn_zones.gpkg")
    sal = fig_salamanca(d, zones)
    r = fig_series(d)
    fig_balance()
    print(sal)
    print(r.pivot_table(index="year", columns=["ecosystem", "zone"], values="s2").round(1).T)
    plt.show()


if __name__ == "__main__":
    main()
