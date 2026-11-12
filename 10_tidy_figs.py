"""
PNN Colombia fire study - Step 10 (Spyder): tidy the figs folder.
Moves superseded figures and Spyder plot-pane saves into figs/old, so that only
the manuscript figures remain (Fig1-Fig6, fig_shap_* (Fig. S1), FigS2, FigS3). Nothing is deleted.
"""
import shutil

import config as cfg

KEEP = {"Fig1_study_area.png", "Fig2_workflow.png", "Fig2_workflow.pdf",
        "Fig3_burned_s2_vs_modis.png", "Fig4_salamanca_small_fires.png",
        "FigS2_annual_series.png", "Fig5_park_effect.png", "Fig6_gam_partial_effects.png",
        "fig_shap_summary.png", "fig_shap_dependence.png", "FigS3_covariate_balance.png"}


def main():
    old = cfg.FIG_DIR / "old"
    old.mkdir(exist_ok=True)
    for f in sorted(cfg.FIG_DIR.iterdir()):
        if f.is_file() and f.name not in KEEP:
            shutil.move(str(f), str(old / f.name))
            print("moved to old:", f.name)
    print("kept:", sorted(p.name for p in cfg.FIG_DIR.iterdir() if p.is_file()))


if __name__ == "__main__":
    main()
