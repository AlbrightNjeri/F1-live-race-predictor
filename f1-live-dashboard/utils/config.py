"""
utils/config.py — Central configuration for F1 Live Predictor Dashboard
"""

# ── Polling ────────────────────────────────────────────────────────────────
UPDATE_INTERVAL_SECONDS = 45          # 30–60 s range; tweak here
FASTF1_CACHE_DIR = "cache/"

# ── Model paths ────────────────────────────────────────────────────────────
MODEL_PATH = "models/xgb_model.pkl"

# ── Prediction blend ───────────────────────────────────────────────────────
MONTE_CARLO_WEIGHT = 0.60
ML_WEIGHT = 0.40
MONTE_CARLO_SIMULATIONS = 5_000

# ── Base probabilities ─────────────────────────────────────────────────────
BASE_DNF_RATE = 0.05           # ~5 % per race
BASE_SAFETY_CAR_PROB = 0.15
BASE_RED_FLAG_PROB = 0.03
TYRE_AGE_PIT_THRESHOLD = 25    # laps — insight trigger

# ── Circuit overtaking index (1=low, 10=high) ──────────────────────────────
CIRCUIT_OVERTAKING_INDEX: dict[str, int] = {
    "monaco": 1,
    "monte_carlo": 1,
    "singapore": 2,
    "hungaroring": 3,
    "hungary": 3,
    "suzuka": 4,
    "japan": 4,
    "zandvoort": 4,
    "netherlands": 4,
    "barcelona": 5,
    "spain": 5,
    "silverstone": 6,
    "britain": 6,
    "mexico": 6,
    "mexico_city": 6,
    "interlagos": 7,
    "brazil": 7,
    "sao_paulo": 7,
    "austin": 7,
    "cota": 7,
    "bahrain": 7,
    "melbourne": 7,
    "australia": 7,
    "jeddah": 7,
    "saudi_arabia": 7,
    "baku": 8,
    "azerbaijan": 8,
    "shanghai": 8,
    "china": 8,
    "spa": 9,
    "belgium": 9,
    "monza": 9,
    "italy": 9,
    "las_vegas": 9,
    "default": 5,
}

# ── Training data ──────────────────────────────────────────────────────────
TRAINING_SEASONS = [2022, 2023, 2024]

# ── Streamlit ──────────────────────────────────────────────────────────────
AUTOREFRESH_INTERVAL_MS = UPDATE_INTERVAL_SECONDS * 1_000
DASHBOARD_TITLE = "F1 Live Predictor"

# ── Tyre compound colours (hex) ────────────────────────────────────────────
TYRE_COLORS: dict[str, str] = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#EBEBEB",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
    "UNKNOWN": "#888888",
}

# ── Top-N drivers shown in charts ─────────────────────────────────────────
CHART_TOP_N = 5
