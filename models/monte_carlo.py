"""
models/monte_carlo.py
─────────────────────
5,000-simulation Monte Carlo race engine.
Runs every prediction cycle regardless of whether the ML model exists.

Uses only LIVE STATE inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.fastf1_poller import RaceState
from utils.config import (
    MONTE_CARLO_SIMULATIONS,
    BASE_DNF_RATE,
    BASE_SAFETY_CAR_PROB,
    BASE_RED_FLAG_PROB,
)
from utils.logger import get_logger

log = get_logger("monte_carlo")


@dataclass
class MCResult:
    driver: str
    win_pct: float
    podium_pct: float
    top5_pct: float
    expected_finish: float


def run_monte_carlo(state: RaceState) -> dict[str, MCResult]:
    """
    Run MONTE_CARLO_SIMULATIONS race simulations from the current RaceState.
    Returns a dict keyed by driver abbreviation.
    """
    n = MONTE_CARLO_SIMULATIONS
    drivers = state.drivers
    if not drivers:
        return {}

    n_drivers = len(drivers)
    laps_remaining = max(1, state.total_laps - state.current_lap)

    # ── Per-driver base scores ─────────────────────────────────────────
    # Lower current position = better; gap penalises further
    base_scores = np.array(
        [_base_score(d, laps_remaining) for d in drivers], dtype=np.float64
    )

    # ── Simulation matrix: shape (n_sims, n_drivers) ──────────────────
    rng = np.random.default_rng()

    # Lap time noise — driver-specific std dev scaled by laps remaining
    pace_noise = rng.normal(
        loc=0.0,
        scale=np.array([d.lap_time_std for d in drivers]) * laps_remaining,
        size=(n, n_drivers),
    )

    # DNF draw — per driver per sim
    dnf_prob = _dnf_probability(drivers, laps_remaining)
    dnf_draw = rng.random(size=(n, n_drivers)) < dnf_prob  # bool mask

    # Safety car effect: compresses field by reducing gaps
    sc_factor = np.ones((n, n_drivers))
    sc_events = rng.random(n) < BASE_SAFETY_CAR_PROB
    sc_factor[sc_events, :] *= 0.5   # halve gaps under SC

    # Red flag: complete randomisation of running order
    red_flag_events = rng.random(n) < BASE_RED_FLAG_PROB

    # ── Final simulated scores ─────────────────────────────────────────
    sim_scores = (
        base_scores
        + pace_noise * sc_factor
    )

    # Mark DNF drivers with a very high (bad) score
    sim_scores[dnf_draw] += 1e6

    # Red flag: shuffle order for affected sims
    for i in np.where(red_flag_events)[0]:
        shuffle_idx = rng.permutation(n_drivers)
        sim_scores[i] = sim_scores[i][shuffle_idx]

    # ── Convert scores to finishing positions ──────────────────────────
    # argsort twice gives rank (0-based)
    finishing_ranks = np.argsort(np.argsort(sim_scores, axis=1), axis=1) + 1  # 1-based

    # ── Aggregate stats ────────────────────────────────────────────────
    results: dict[str, MCResult] = {}
    for idx, driver in enumerate(drivers):
        ranks = finishing_ranks[:, idx]
        results[driver.abbreviation] = MCResult(
            driver=driver.abbreviation,
            win_pct=round(float((ranks == 1).mean() * 100), 2),
            podium_pct=round(float((ranks <= 3).mean() * 100), 2),
            top5_pct=round(float((ranks <= 5).mean() * 100), 2),
            expected_finish=round(float(ranks.mean()), 2),
        )

    log.debug("Monte Carlo complete — %d sims, %d drivers", n, n_drivers)
    return results


# ── Helpers ────────────────────────────────────────────────────────────────

def _base_score(driver, laps_remaining: int) -> float:
    """
    Lower is better.
    Combines current position, gap to leader, and tyre age.
    """
    pos_factor = driver.position * 10.0
    gap_factor = driver.gap_to_leader_s * 0.5
    tyre_factor = max(0.0, driver.tyre_age_laps - 20) * 0.3
    return pos_factor + gap_factor + tyre_factor


def _dnf_probability(drivers, laps_remaining: int) -> np.ndarray:
    """
    Per-driver DNF probability scaled by tyre age and position.
    Shape: (n_drivers,)
    """
    probs = []
    for d in drivers:
        # Higher tyre age → higher risk
        tyre_risk = min(0.04, d.tyre_age_laps * 0.0008)
        # Running near the back → slightly higher risk
        pos_risk = min(0.02, (d.position - 1) * 0.001)
        total = BASE_DNF_RATE + tyre_risk + pos_risk
        probs.append(min(total, 0.25))   # cap at 25 %
    return np.array(probs)
