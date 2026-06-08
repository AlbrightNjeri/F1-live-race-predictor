"""
data/feature_builder.py
───────────────────────
Converts a RaceState snapshot into a feature DataFrame ready for the ML model.

PIPELINE GUARANTEE: reads only LIVE STATE — never touches official results.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from data.fastf1_poller import RaceState, DriverState
from utils.config import CIRCUIT_OVERTAKING_INDEX
from utils.logger import get_logger

log = get_logger("feature_builder")

# ── Feature columns (must match training) ─────────────────────────────────
FEATURE_COLUMNS = [
    "quali_position",           # grid / current position (proxy)
    "gap_to_leader_s",
    "tyre_age_laps",
    "consistency_score",
    "circuit_overtaking_index",
    "laps_remaining",
    "safety_car_active",
    "current_position",
]


def build_features(state: RaceState) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per driver and FEATURE_COLUMNS as columns.
    Index = driver abbreviation.
    """
    records = []
    laps_remaining = max(0, state.total_laps - state.current_lap)

    for driver in state.drivers:
        records.append(
            {
                "driver": driver.abbreviation,
                "quali_position": driver.position,       # best proxy live
                "gap_to_leader_s": driver.gap_to_leader_s,
                "tyre_age_laps": driver.tyre_age_laps,
                "consistency_score": driver.lap_time_std,
                "circuit_overtaking_index": state.circuit_overtaking_index,
                "laps_remaining": laps_remaining,
                "safety_car_active": int(state.safety_car_active),
                "current_position": driver.position,
            }
        )

    df = pd.DataFrame(records).set_index("driver")

    # Fill any NaNs with sensible defaults
    df["gap_to_leader_s"] = df["gap_to_leader_s"].fillna(0.0)
    df["tyre_age_laps"] = df["tyre_age_laps"].fillna(10)
    df["consistency_score"] = df["consistency_score"].fillna(1.5)

    return df[FEATURE_COLUMNS]
