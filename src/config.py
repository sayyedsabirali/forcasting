import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_PATH = r"F:\task\ev_charging_forecast\data\processed\ev_charging_analysis_ready.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_SAVE_PATH = OUTPUT_DIR / "models"
PLOTS_PATH = OUTPUT_DIR / "plots"
OUTPUT_PATH = OUTPUT_DIR / "data"

for path in [MODEL_SAVE_PATH, PLOTS_PATH, OUTPUT_PATH]:
    path.mkdir(parents=True, exist_ok=True)
TARGET_COL = "num_sessions"
FORECAST_HORIZON = 24 * 7
RANDOM_STATE = 42

# SIMPLE REGRESSION PARAMS
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'objective': 'reg:squarederror',
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'verbosity': 0
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_STATE,
    'max_features': 'sqrt',
    'bootstrap': True,
    'verbose': 0
}

LGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'random_state': RANDOM_STATE,
    'objective': 'regression',
    'verbose': -1
}