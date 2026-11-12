"""
PNN Colombia fire study - Fig. 2: workflow diagram (Spyder, matplotlib only).
Four highlighted blocks (input data, processing, analysis, outputs) with stage labels on the
right. Box widths adapt to the text, so labels never cross the borders.
Output: figs/Fig2_workflow.png and figs/Fig2_workflow.pdf
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # same font on every PC
from matplotlib.patches import FancyBboxPatch

import config as cfg

TEAL, TEAL_BG = "#1d7874", "#eaf4f3"
GREY, GREY_BG, GREY_BOX = "#8e979f", "#f4f5f7", "#e3e6e9"
OCHRE, OCHRE_BG = "#c08a2b", "#fbf3e4"
PLUM, PLUM_BG = "#6a4c93", "#f1edf7"
INK, LINE = "#222831", "#7d868f"
FONT = 8.2
STEP = 0.32          # line spacing (data units)
PADX, PADY = 0.35, 0.16


def frame(ax, x0, y0, x1, y1, ec, fc):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=fc, ec=ec, lw=1.6, zorder=0))


def box(ax, xc, yc, title, lines, w=None, fc="white", ec=TEAL, tc=INK, rounded=True, lw=1.1):
    """Draw the text first, measure it, then draw a box that fits it (at least width w)."""
    n = len(lines) + 1
    y0 = yc + STEP * (n - 1) / 2
    texts = [ax.text(xc, y0, title, ha="center", va="center", fontsize=FONT, color=tc,
                     weight="bold", zorder=3)]
    for k, ln in enumerate(lines, 1):
        texts.append(ax.text(xc, y0 - k * STEP, ln, ha="center", va="center", fontsize=FONT,
                             color=tc, zorder=3))
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    tw = 0
    for t in texts:
        bb = t.get_window_extent(renderer=r)
        (xa, _), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        tw = max(tw, xb - xa)
    bw = max(w or 0, tw + 2 * PADX)
    bh = STEP * n + 2 * PADY
    style = "round,pad=0,rounding_size=0.18" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((xc - bw / 2, yc - bh / 2), bw, bh, boxstyle=style,
                                fc=fc, ec=ec, lw=lw, zorder=2))
    return dict(l=xc - bw / 2, r=xc + bw / 2, t=yc + bh / 2, b=yc - bh / 2, x=xc, y=yc)


def path(ax, pts, color=LINE, lw=1.0, ls="-", arrow=True):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    if arrow:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=10,
                                    shrinkA=0, shrinkB=0, linestyle="-"))


def main():
    cfg.FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.3, 8.6))
    ax.set_xlim(0, 20.2)
    ax.set_ylim(1.85, 17.1)
    ax.axis("off")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    fig.canvas.draw()

    # ---- block frames and stage labels -------------------------------------
    frame(ax, 0.15, 14.75, 17.75, 16.95, GREY, GREY_BG)
    frame(ax, 3.7, 7.25, 11.9, 14.4, TEAL, TEAL_BG)
    frame(ax, 1.6, 4.4, 17.75, 6.3, OCHRE, OCHRE_BG)
    frame(ax, 1.6, 2.0, 17.75, 3.9, PLUM, PLUM_BG)
    for txt, y, c in [("INPUT\nDATA", 15.85, "#5f6870"), ("PROCESSING", 11.0, TEAL),
                      ("ANALYSIS", 5.35, "#a8741c"), ("OUTPUTS", 2.95, PLUM)]:
        ax.text(17.95, y, txt, ha="left", va="center", fontsize=10.5, weight="bold", color=c,
                linespacing=1.1)

    # ---- input data ------------------------------------------------------------
    kw = dict(fc=GREY_BOX, ec=GREY_BOX, rounded=False, lw=0, w=3.1)
    yi = 15.85
    wdpa = box(ax, 1.85, yi, "Protected areas", ["WDPA boundaries", "8 National Natural Parks"], **kw)
    s2 = box(ax, 5.35, yi, "Sentinel-2 L1C", ["2017–2025", "Cloud Score+ mask"], **kw)
    viirs = box(ax, 8.8, yi, "VIIRS S-NPP", ["375 m active fires", "(NASA FIRMS)"], **kw)
    modis = box(ax, 12.25, yi, "MODIS MCD64A1", ["500 m burned area", "2016–2025"], **kw)
    cov = box(ax, 15.75, yi, "Covariates", ["SRTM, travel time,", "CHIRPS, WorldCover"], **kw)

    # ---- processing ------------------------------------------------------------
    px = 7.8
    design = box(ax, px, 13.5, "Sampling design",
                 ["10 km external rings (land only)", "47,499 random points"], w=5.6)
    nbr = box(ax, px, 11.8, "Annual NBR metrics per point",
              ["dNBR = median − minimum", "clear images, date of minimum"], w=5.6)
    conf = box(ax, px, 9.95, "Fire confirmation",
               ["dNBR ≥ 0.15 in ≥ 2 clear images", "AND (VIIRS fire ≤ 500 m, −60/+5 days",
                "OR MODIS burned)"], fc=TEAL, tc="white", lw=1.4, w=5.6)
    rec = box(ax, px, 8.1, "Fire records per point-year",
              ["Sentinel-2 confirmed (16,140 fires)", "MODIS-only record"], w=5.6)
    for b, nxt in [(design, nbr), (nbr, conf), (conf, rec)]:
        path(ax, [(px, b["b"]), (px, nxt["t"])], color=TEAL, lw=1.6)

    # ---- analysis --------------------------------------------------------------
    ya, w3 = 5.35, 4.6
    xs = (4.45, 9.65, 14.85)
    comp = box(ax, xs[0], ya, "Product comparison",
               ["burn rates inside vs ring", "fires missed by MODIS"], ec=OCHRE, w=w3)
    match = box(ax, xs[1], ya, "Covariate matching",
                ["ATT per park", "10 km block bootstrap"], ec=OCHRE, w=w3)
    gam = box(ax, xs[2], ya, "GAM and LightGBM–SHAP",
              ["fire probability gradients", "5-fold spatial CV"], ec=OCHRE, w=w3)

    # ---- outputs ---------------------------------------------------------------
    yo = 2.95
    o1 = box(ax, xs[0], yo, "MODIS omission",
             ["46% of fires missed (97% Salamanca)", "Figs. 3–4, S2"], ec=PLUM, w=w3)
    o2 = box(ax, xs[1], yo, "Park effect per park",
             ["ATT (pp yr⁻¹), S-2 vs MODIS verdict", "Table 2, Fig. 5"], ec=PLUM, w=w3)
    o3 = box(ax, xs[2], yo, "Drivers of fire",
             ["partial effects, SHAP importance", "Fig. 6, Fig. S1"], ec=PLUM, w=w3)
    for a, o in [(comp, o1), (match, o2), (gam, o3)]:
        path(ax, [(a["x"], a["b"]), (a["x"], o["t"])], color=PLUM, lw=1.6)

    # records -> analyses: one teal trunk and a horizontal bus, drops at box centres
    ybus = 6.8
    ax.plot([px, px], [rec["b"], ybus], color=TEAL, lw=1.6, zorder=1)
    ax.plot([comp["x"], gam["x"]], [ybus, ybus], color=TEAL, lw=1.6, zorder=1)
    for b in (comp, match, gam):
        path(ax, [(b["x"], ybus), (b["x"], b["t"])], color=TEAL, lw=1.6)

    # ---- input arrows (grey) ---------------------------------------------------
    path(ax, [(wdpa["x"], wdpa["b"]), (wdpa["x"], design["y"]), (design["l"], design["y"])])
    path(ax, [(4.3, s2["b"]), (4.3, nbr["y"]), (nbr["l"], nbr["y"])])
    path(ax, [(viirs["x"] + 0.9, viirs["b"]), (viirs["x"] + 0.9, 14.57), (11.3, 14.57),
              (11.3, conf["y"] + 0.35), (conf["r"], conf["y"] + 0.35)])
    path(ax, [(modis["x"], modis["b"]), (modis["x"], conf["y"] - 0.35), (conf["r"], conf["y"] - 0.35)])

    # covariates -> matching and GAM (dashed)
    xc, ydash = 16.4, 6.55
    ax.plot([xc, xc], [cov["b"], ydash], color=LINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.plot([match["x"] + 1.2, xc], [ydash, ydash], color=LINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for x, tgt in ((match["x"] + 1.2, match), (xc, gam)):
        ax.annotate("", xy=(x, tgt["t"]), xytext=(x, ydash), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.0, mutation_scale=10,
                                    shrinkA=0, shrinkB=0))

    fig.savefig(cfg.FIG_DIR / "Fig2_workflow.png", dpi=300)
    fig.savefig(cfg.FIG_DIR / "Fig2_workflow.pdf")
    plt.show()


if __name__ == "__main__":
    main()
