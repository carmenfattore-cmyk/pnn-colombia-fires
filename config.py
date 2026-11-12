"""
Configuration for the Colombian National Natural Parks fire study.
All paths and study settings live here (same flat layout as the Basilicata project).
"""
from pathlib import Path

# --- Earth Engine -----------------------------------------------------------
GEE_PROJECT = "your-gee-project-id"        # <-- set your Cloud project ID
DRIVE_FOLDER = "PNN_Colombia_fires"        # Google Drive folder for exports

# --- Local paths ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"               # CSVs downloaded from Drive go here
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figs"

# --- Study period -----------------------------------------------------------
YEAR_START = 2016                          # peace agreement year; El Nino 2015-16
YEAR_END = 2025                            # last complete year

# --- Parks, grouped by ecosystem -------------------------------------------
# Park names (WDPA search strings are in 01_build_zones_and_sample.py).
PARKS = {
    "El Tuparro":                    "Orinoquia savanna",
    "Sumapaz":                       "Andean paramo",
    "El Cocuy":                      "Andean paramo",
    "Sierra de la Macarena":         "Amazon deforestation arc",
    "Tinigua":                       "Amazon deforestation arc",
    "Chiribiquete":                  "Amazon deforestation arc",
    "Sierra Nevada de Santa Marta":  "Caribbean mountain",
    "Isla de Salamanca":             "Caribbean coast (mangrove/wetland)",
}

# --- Zones ------------------------------------------------------------------
BUFFER_KM = 10                             # outer ring for inside/outside comparison

# --- Sampling for matching --------------------------------------------------
N_POINTS_PER_ZONE = 3000                   # random points per park and per buffer
SAMPLE_SCALE = 100                         # m
SEED = 42

# --- Active fires -----------------------------------------------------------
FIRMS_MIN_CONFIDENCE = 50                  # MODIS FIRMS confidence threshold (%)
