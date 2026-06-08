"""
data/fastf1_poller.py
────────────────────
Polls a FastF1 session every UPDATE_INTERVAL_SECONDS.
Produces a RaceState snapshot consumed by the dashboard.

Pipeline guarantee: LIVE STATE only — never touches official results.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import fastf1
import pandas as pd
import numpy as np

from utils.config import (
    UPDATE_INTERVAL_SECONDS,
    FASTF1_CACHE_DIR,
    CIRCUIT_OVERTAKING_INDEX,
    TYRE_AGE_PIT_THRESHOLD,
)
from utils.logger import get_logger

log = get_logger("poller")

# ── Enable disk cache ──────────────────────────────────────────────────────
fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class DriverState:
    abbreviation: str
    full_name: str
    team: str
    position: int
    gap_to_leader_s: float          # seconds, 0.0 for leader
    tyre_compound: str              # SOFT / MEDIUM / HARD / …
    tyre_age_laps: int
    lap_time_last: Optional[float]  # seconds
    lap_time_std: float             # consistency proxy
    laps_completed: int
    is_retired: bool = False
    # history for charts
    position_history: list[int] = field(default_factory=list)


@dataclass
class RaceState:
    session_name: str
    circuit: str
    total_laps: int
    current_lap: int
    safety_car_active: bool
    red_flag_active: bool
    drivers: list[DriverState] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    stale: bool = False             # True if last poll failed
    circuit_overtaking_index: int = 5


# ── Poller ─────────────────────────────────────────────────────────────────

class F1Poller:
    """Runs on a background thread; callers read .latest_state."""

    def __init__(self, year: int, gp: str, session_type: str = "R"):
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.latest_state: Optional[RaceState] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_laps: Optional[pd.DataFrame] = None

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Spin up background polling thread."""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Poller started — %d %s %s", self.year, self.gp, self.session_type)

    def stop(self) -> None:
        self._stop_event.set()

    # ── Internal polling loop ───────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self._fetch_and_build_state()
                if state:
                    self.latest_state = state
                    log.info(
                        "State updated — lap %d/%d | SC=%s",
                        state.current_lap,
                        state.total_laps,
                        state.safety_car_active,
                    )
            except Exception as exc:  # noqa: BLE001
                log.error("Poll cycle error: %s", exc, exc_info=True)
                if self.latest_state:
                    self.latest_state.stale = True
            time.sleep(UPDATE_INTERVAL_SECONDS)

    def _fetch_and_build_state(self) -> Optional[RaceState]:
        session = fastf1.get_session(self.year, self.gp, self.session_type)
        try:
            session.load(laps=True, telemetry=False, weather=True, messages=True)
        except fastf1.core.DataNotLoadedError as exc:
            log.warning("DataNotLoadedError: %s — will retry", exc)
            return None

        laps: pd.DataFrame = session.laps
        if laps is None or laps.empty:
            log.warning("No lap data available yet")
            return None

        # ── Session metadata ────────────────────────────────────────────
        total_laps = self._get_total_laps(session)
        current_lap = int(laps["LapNumber"].max()) if not laps.empty else 0
        sc_active, red_flag = self._parse_track_status(session)
        circuit_key = _normalise_circuit(session.event["Location"] if hasattr(session, "event") else "default")
        overtaking_idx = CIRCUIT_OVERTAKING_INDEX.get(circuit_key, CIRCUIT_OVERTAKING_INDEX["default"])

        # ── Per-driver state ────────────────────────────────────────────
        drivers = self._build_driver_states(laps, current_lap, total_laps)

        state = RaceState(
            session_name=f"{self.year} {self.gp} {self.session_type}",
            circuit=session.event["Location"] if hasattr(session, "event") else self.gp,
            total_laps=total_laps,
            current_lap=current_lap,
            safety_car_active=sc_active,
            red_flag_active=red_flag,
            drivers=drivers,
            last_updated=datetime.utcnow(),
            stale=False,
            circuit_overtaking_index=overtaking_idx,
        )

        # Merge position history from previous state
        if self.latest_state:
            prev_map = {d.abbreviation: d for d in self.latest_state.drivers}
            for d in state.drivers:
                if d.abbreviation in prev_map:
                    d.position_history = prev_map[d.abbreviation].position_history.copy()
                d.position_history.append(d.position)
                if len(d.position_history) > 100:
                    d.position_history = d.position_history[-100:]

        return state

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_driver_states(
        self, laps: pd.DataFrame, current_lap: int, total_laps: int
    ) -> list[DriverState]:
        """Build one DriverState per driver from the latest lap data."""
        drivers: list[DriverState] = []

        # Group by driver abbreviation
        for abbr, drv_laps in laps.groupby("Driver"):
            if drv_laps.empty:
                continue

            latest_lap = drv_laps[drv_laps["LapNumber"] == drv_laps["LapNumber"].max()].iloc[-1]

            # Position — FastF1 stores as float, can be NaN
            pos = _safe_int(latest_lap.get("Position"), default=99)

            # Gap to leader in seconds
            gap = _safe_float(latest_lap.get("GapToLeader"), default=0.0)
            if gap < 0:
                gap = abs(gap)

            # Tyre
            compound = str(latest_lap.get("Compound", "UNKNOWN")).upper()
            tyre_age = _safe_int(latest_lap.get("TyreLife"), default=0)

            # Lap times
            lap_time_s = _timedelta_to_s(latest_lap.get("LapTime"))
            lap_times_s = drv_laps["LapTime"].dropna().apply(_timedelta_to_s).dropna()
            consistency = float(lap_times_s.std()) if len(lap_times_s) > 2 else 2.0

            laps_done = _safe_int(latest_lap.get("LapNumber"), default=0)
            is_retired = bool(latest_lap.get("IsPersonalBest") is None and pos == 99)

            # Driver full name + team
            full_name = str(latest_lap.get("Driver", abbr))
            team = str(latest_lap.get("Team", "Unknown"))

            drivers.append(
                DriverState(
                    abbreviation=str(abbr),
                    full_name=full_name,
                    team=team,
                    position=pos,
                    gap_to_leader_s=gap,
                    tyre_compound=compound,
                    tyre_age_laps=tyre_age,
                    lap_time_last=lap_time_s,
                    lap_time_std=consistency,
                    laps_completed=laps_done,
                    position_history=[pos],
                )
            )

        # Sort by position
        drivers.sort(key=lambda d: d.position)
        return drivers

    def _get_total_laps(self, session) -> int:
        try:
            return int(session.total_laps)
        except Exception:
            return 70  # sensible default

    def _parse_track_status(self, session) -> tuple[bool, bool]:
        """Parse track status messages for SC / red flag."""
        sc_active = False
        red_flag = False
        try:
            msgs = session.track_status
            if msgs is not None and not msgs.empty:
                latest = msgs.iloc[-1]
                status = str(latest.get("Status", ""))
                sc_active = status in ("4", "5", "6")   # SC, VSC, SC ending
                red_flag = status == "5"
        except Exception:
            pass
        return sc_active, red_flag


# ── Convenience factory ────────────────────────────────────────────────────

def create_demo_state(gp_name: str = "Monaco", year: int = 2024) -> RaceState:
    """
    Returns a synthetic RaceState for UI development / offline testing.
    Uses realistic-looking but entirely fake data.
    """
    import random
    rng = random.Random(42)
    driver_pool = [
        ("VER", "Max Verstappen", "Red Bull Racing"),
        ("LEC", "Charles Leclerc", "Ferrari"),
        ("NOR", "Lando Norris", "McLaren"),
        ("PIA", "Oscar Piastri", "McLaren"),
        ("SAI", "Carlos Sainz", "Ferrari"),
        ("HAM", "Lewis Hamilton", "Mercedes"),
        ("RUS", "George Russell", "Mercedes"),
        ("ALO", "Fernando Alonso", "Aston Martin"),
        ("STR", "Lance Stroll", "Aston Martin"),
        ("GAS", "Pierre Gasly", "Alpine"),
        ("OCO", "Esteban Ocon", "Alpine"),
        ("ALB", "Alex Albon", "Williams"),
        ("SAR", "Logan Sargeant", "Williams"),
        ("BOT", "Valtteri Bottas", "Alfa Romeo"),
        ("ZHO", "Zhou Guanyu", "Alfa Romeo"),
        ("TSU", "Yuki Tsunoda", "AlphaTauri"),
        ("RIC", "Daniel Ricciardo", "AlphaTauri"),
        ("MAG", "Kevin Magnussen", "Haas"),
        ("HUL", "Nico Hulkenberg", "Haas"),
        ("DEV", "Nyck de Vries", "AlphaTauri"),
    ]

    compounds = ["SOFT", "MEDIUM", "HARD"]
    current_lap = 38
    total_laps = 70

    drivers = []
    for pos, (abbr, name, team) in enumerate(driver_pool, start=1):
        gap = 0.0 if pos == 1 else rng.uniform(0.5, 45.0) * (pos ** 0.7)
        tyre_age = rng.randint(5, 30)
        compound = rng.choice(compounds)
        history = [max(1, pos + rng.randint(-3, 3)) for _ in range(10)]
        history.append(pos)
        drivers.append(
            DriverState(
                abbreviation=abbr,
                full_name=name,
                team=team,
                position=pos,
                gap_to_leader_s=round(gap, 3),
                tyre_compound=compound,
                tyre_age_laps=tyre_age,
                lap_time_last=round(rng.uniform(78.0, 95.0), 3),
                lap_time_std=round(rng.uniform(0.3, 1.8), 3),
                laps_completed=current_lap,
                is_retired=False,
                position_history=history,
            )
        )

    return RaceState(
        session_name=f"{year} {gp_name} R (DEMO)",
        circuit=gp_name,
        total_laps=total_laps,
        current_lap=current_lap,
        safety_car_active=False,
        red_flag_active=False,
        drivers=drivers,
        last_updated=datetime.utcnow(),
        stale=False,
        circuit_overtaking_index=5,
    )


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


def _timedelta_to_s(td) -> Optional[float]:
    """Convert a pandas Timedelta or None to seconds float."""
    try:
        return td.total_seconds()
    except Exception:
        return None


def _normalise_circuit(location: str) -> str:
    return location.lower().replace(" ", "_").replace("-", "_")
