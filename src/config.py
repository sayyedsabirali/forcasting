import os
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
# DATA_PATH = r"F:\task\ev_charging_forecast\data\processed\ev_charging_analysis_ready.csv"
DATA_PATH = r"F:\task\ev_charging_forecast\data\raw\karocharge_delhi_30day_realistic_synthetic_mixed_idle.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_SAVE_PATH = OUTPUT_DIR / "models"
PLOTS_PATH = OUTPUT_DIR / "plots"
OUTPUT_PATH = OUTPUT_DIR / "data"
for path in [MODEL_SAVE_PATH, PLOTS_PATH, OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)
TARGET_COL = "num_sessions"
FORECAST_HORIZON = 24 * 7
RANDOM_STATE = 42