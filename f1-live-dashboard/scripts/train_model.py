"""
scripts/train_model.py
──────────────────────
Offline training pipeline.

Uses ONLY OFFICIAL STATE: FIA final race classifications from 2022–2024.
NEVER mixes with live data.

Run once before a race weekend:
    python scripts/train_model.py
"""

from __future__ import annotations

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

from utils.config import FASTF1_CACHE_DIR, TRAINING_SEASONS, MODEL_PATH, CIRCUIT_OVERTAKING_INDEX
from utils.logger import get_logger

log = get_logger("train_model")

fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


# ── Feature columns (must match feature_builder.py) ───────────────────────
FEATURE_COLS = [
    "quali_position",
    "gap_to_leader_s",
    "tyre_age_laps",
    "consistency_score",
    "circuit_overtaking_index",
    "laps_remaining",
    "safety_car_active",
    "current_position",
]
TARGET_COL = "final_position"


def load_race_data(year: int, gp_name: str) -> pd.DataFrame | None:
    """Load official results for one race — OFFICIAL STATE only."""
    try:
        session = fastf1.get_session(year, gp_name, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        log.warning("Could not load %d %s: %s", year, gp_name, exc)
        return None

    laps = session.laps
    results = session.results

    if laps is None or laps.empty or results is None or results.empty:
        log.warning("Empty data for %d %s", year, gp_name)
        return None

    circuit_key = (session.event["Location"] if hasattr(session, "event") else gp_name).lower()
    circuit_key = circuit_key.replace(" ", "_").replace("-", "_")
    overtaking_idx = CIRCUIT_OVERTAKING_INDEX.get(circuit_key, CIRCUIT_OVERTAKING_INDEX["default"])

    rows = []
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 70

    for _, result_row in results.iterrows():
        abbr = result_row.get("Abbreviation", "")
        drv_laps = laps[laps["Driver"] == abbr]

        if drv_laps.empty:
            continue

        # Final position from official results
        final_pos = result_row.get("ClassifiedPosition")
        try:
            final_pos = int(final_pos)
        except (TypeError, ValueError):
            final_pos = 20  # DNF / DSQ → last

        # Mid-race features (use lap ~50% as snapshot)
        mid_lap = total_laps // 2
        mid_laps = drv_laps[drv_laps["LapNumber"] <= mid_lap]
        if mid_laps.empty:
            mid_laps = drv_laps

        last_mid = mid_laps.iloc[-1]

        pos_at_mid = _safe_int(last_mid.get("Position"), 20)
        gap_at_mid = abs(_safe_float(last_mid.get("GapToLeader"), 0.0))
        tyre_age = _safe_int(last_mid.get("TyreLife"), 10)

        lap_times_s = (
            drv_laps["LapTime"].dropna().apply(_td_to_s).dropna()
        )
        consistency = float(lap_times_s.std()) if len(lap_times_s) > 2 else 1.5

        rows.append(
            {
                "driver": abbr,
                "year": year,
                "gp": gp_name,
                "quali_position": _safe_int(result_row.get("GridPosition"), pos_at_mid),
                "gap_to_leader_s": gap_at_mid,
                "tyre_age_laps": tyre_age,
                "consistency_score": consistency,
                "circuit_overtaking_index": overtaking_idx,
                "laps_remaining": total_laps - mid_lap,
                "safety_car_active": 0,   # not used in training snapshot
                "current_position": pos_at_mid,
                TARGET_COL: final_pos,
            }
        )

    return pd.DataFrame(rows) if rows else None


def collect_training_data(seasons: list[int]) -> pd.DataFrame:
    """Iterate over all races in the given seasons and collect features."""
    all_frames: list[pd.DataFrame] = []

    for year in seasons:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        gps = schedule["EventName"].tolist()
        log.info("Season %d — %d races", year, len(gps))

        for gp in gps:
            log.info("  Loading %d %s …", year, gp)
            df = load_race_data(year, gp)
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No training data could be loaded. Check FastF1 cache / network.")

    return pd.concat(all_frames, ignore_index=True)


def train(df: pd.DataFrame) -> xgb.XGBRegressor:
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    train_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
    val_rmse = mean_squared_error(y_val, model.predict(X_val), squared=False)
    log.info("Train RMSE: %.3f | Val RMSE: %.3f", train_rmse, val_rmse)
    print(f"\n✅  Train RMSE: {train_rmse:.3f}  |  Val RMSE: {val_rmse:.3f}\n")

    return model


def main() -> None:
    print("=" * 60)
    print("  F1 Live Predictor — Offline Model Training")
    print(f"  Seasons: {TRAINING_SEASONS}")
    print("=" * 60)

    print("\n📡 Collecting official race results…")
    df = collect_training_data(TRAINING_SEASONS)
    print(f"   → {len(df)} training samples across {df['gp'].nunique()} GPs\n")

    print("🧠 Training XGBoost model…")
    model = train(df)

    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")


# ── Utilities ──────────────────────────────────────────────────────────────

def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _td_to_s(td):
    try:
        return td.total_seconds()
    except Exception:
        return None


if __name__ == "__main__":
    main()
