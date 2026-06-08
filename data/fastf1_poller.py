"""
data/fastf1_poller.py
─────────────────────
Polls a FastF1 session every UPDATE_INTERVAL_SECONDS.
Produces a RaceState snapshot consumed by the dashboard.

LIVE STATE only — never touches official results.
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
)
from utils.logger import get_logger

log = get_logger("poller")

# Auto-create cache dir (required for Streamlit Cloud deployment)
import os as _os
_os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class DriverState:
    abbreviation: str
    full_name: str
    team: str
    position: int
    gap_to_leader_s: float
    tyre_compound: str
    tyre_age_laps: int
    lap_time_last: Optional[float]
    lap_time_std: float
    laps_completed: int
    is_retired: bool = False
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
    stale: bool = False
    circuit_overtaking_index: int = 5


# ── Poller ─────────────────────────────────────────────────────────────────

class F1Poller:
    def __init__(self, year: int, gp: str, session_type: str = "R"):
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.latest_state: Optional[RaceState] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        # Do an immediate first fetch in a startup thread so data appears fast
        def _startup():
            try:
                state = self._fetch_and_build_state()
                if state:
                    self.latest_state = state
                    log.info("Initial fetch complete — lap %d/%d", state.current_lap, state.total_laps)
            except Exception as exc:
                log.error("Initial fetch error: %s", exc, exc_info=True)
            # Then hand off to the regular polling loop
            self._poll_loop()

        self._thread = threading.Thread(target=_startup, daemon=True)
        self._thread.start()
        log.info("Poller started — %d %s %s", self.year, self.gp, self.session_type)

    def stop(self) -> None:
        self._stop_event.set()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self._fetch_and_build_state()
                if state:
                    self.latest_state = state
                    log.info("State updated — lap %d/%d", state.current_lap, state.total_laps)
            except Exception as exc:
                log.error("Poll cycle error: %s", exc, exc_info=True)
                if self.latest_state:
                    self.latest_state.stale = True
            time.sleep(UPDATE_INTERVAL_SECONDS)

    def _fetch_and_build_state(self) -> Optional[RaceState]:
        try:
            session = fastf1.get_session(self.year, self.gp, self.session_type)
            session.load(laps=True, telemetry=False, weather=False, messages=True)
        except Exception as exc:
            log.warning("Session load failed: %s", exc)
            return None

        laps: pd.DataFrame = session.laps
        if laps is None or laps.empty:
            log.warning("No lap data available")
            return None

        # ── Metadata ───────────────────────────────────────────────────
        total_laps  = self._get_total_laps(session, laps)
        current_lap = int(laps["LapNumber"].max())
        sc_active, red_flag = self._parse_track_status(session)

        try:
            location = session.event["Location"]
        except Exception:
            location = self.gp
        circuit_key  = _normalise(location)
        overtaking_idx = CIRCUIT_OVERTAKING_INDEX.get(circuit_key, CIRCUIT_OVERTAKING_INDEX["default"])

        # ── Build driver name/team lookup from session.results ─────────
        name_map: dict[str, str] = {}
        team_map: dict[str, str] = {}
        try:
            results = session.results
            if results is not None and not results.empty:
                for _, row in results.iterrows():
                    abbr = str(row.get("Abbreviation", ""))
                    name_map[abbr] = str(row.get("FullName", abbr))
                    team_map[abbr] = str(row.get("TeamName", "Unknown"))
        except Exception:
            pass

        drivers = self._build_driver_states(laps, name_map, team_map)

        # Use actual event name from FastF1 if available
        try:
            event_name = session.event["EventName"]
            session_label = f"{self.year} {event_name}"
        except Exception:
            session_label = f"{self.year} {self.gp}"

        state = RaceState(
            session_name=session_label,
            circuit=location,
            total_laps=total_laps,
            current_lap=current_lap,
            safety_car_active=sc_active,
            red_flag_active=red_flag,
            drivers=drivers,
            last_updated=datetime.utcnow(),
            stale=False,
            circuit_overtaking_index=overtaking_idx,
        )

        # Carry forward position history
        if self.latest_state:
            prev = {d.abbreviation: d for d in self.latest_state.drivers}
            for d in state.drivers:
                if d.abbreviation in prev:
                    d.position_history = prev[d.abbreviation].position_history.copy()
                d.position_history.append(d.position)
                if len(d.position_history) > 100:
                    d.position_history = d.position_history[-100:]

        return state

    def _build_driver_states(
        self,
        laps: pd.DataFrame,
        name_map: dict[str, str],
        team_map: dict[str, str],
    ) -> list[DriverState]:
        drivers: list[DriverState] = []

        for abbr, drv_laps in laps.groupby("Driver"):
            if drv_laps.empty:
                continue

            abbr = str(abbr)
            last_lap_num = drv_laps["LapNumber"].max()
            latest = drv_laps[drv_laps["LapNumber"] == last_lap_num].iloc[-1]

            pos      = _safe_int(latest.get("Position"), 99)
            gap      = abs(_safe_float(latest.get("GapToLeader"), 0.0))
            compound = str(latest.get("Compound", "UNKNOWN")).upper()
            if compound in ("NAN", "NONE", ""):
                compound = "UNKNOWN"
            tyre_age = _safe_int(latest.get("TyreLife"), 0)

            lap_time_s  = _td_to_s(latest.get("LapTime"))
            lap_times_s = drv_laps["LapTime"].dropna().apply(_td_to_s).dropna()
            consistency = float(lap_times_s.std()) if len(lap_times_s) > 2 else 2.0

            laps_done   = _safe_int(latest.get("LapNumber"), 0)
            is_retired  = pos == 99 and laps_done < int(laps["LapNumber"].max()) - 5

            drivers.append(DriverState(
                abbreviation   = abbr,
                full_name      = name_map.get(abbr, abbr),
                team           = team_map.get(abbr, "Unknown"),
                position       = pos,
                gap_to_leader_s= gap,
                tyre_compound  = compound,
                tyre_age_laps  = tyre_age,
                lap_time_last  = lap_time_s,
                lap_time_std   = consistency,
                laps_completed = laps_done,
                is_retired     = is_retired,
                position_history=[pos],
            ))

        drivers.sort(key=lambda d: d.position)
        return drivers

    def _get_total_laps(self, session, laps: pd.DataFrame) -> int:
        # Try session attribute first
        for attr in ("total_laps", "laps_in_event", "_total_laps"):
            try:
                val = int(getattr(session, attr))
                if val > 0:
                    return val
            except Exception:
                pass
        # Fall back to lap count data
        try:
            lc = session.lap_count
            if lc is not None and not lc.empty:
                return int(lc["TotalLaps"].iloc[-1])
        except Exception:
            pass
        # Use max lap observed + small buffer
        try:
            return int(laps["LapNumber"].max()) + 2
        except Exception:
            return 70

    def _parse_track_status(self, session) -> tuple[bool, bool]:
        sc_active  = False
        red_flag   = False
        try:
            msgs = session.track_status
            if msgs is not None and not msgs.empty:
                status = str(msgs.iloc[-1].get("Status", ""))
                sc_active = status in ("4", "5", "6")
                red_flag  = status == "5"
        except Exception:
            pass
        return sc_active, red_flag


# ── Demo state factory ─────────────────────────────────────────────────────

def create_demo_state(gp_name: str = "Monaco", year: int = 2026) -> RaceState:
    import random
    rng = random.Random(42)

    # 2026 grid
    driver_pool = [
        ("VER", "Max Verstappen",      "Red Bull Racing"),
        ("NOR", "Lando Norris",        "McLaren"),
        ("LEC", "Charles Leclerc",     "Ferrari"),
        ("PIA", "Oscar Piastri",       "McLaren"),
        ("SAI", "Carlos Sainz",        "Williams"),
        ("HAM", "Lewis Hamilton",      "Ferrari"),
        ("RUS", "George Russell",      "Mercedes"),
        ("ANT", "Kimi Antonelli",      "Mercedes"),
        ("ALO", "Fernando Alonso",     "Aston Martin"),
        ("STR", "Lance Stroll",        "Aston Martin"),
        ("GAS", "Pierre Gasly",        "Alpine"),
        ("COL", "Franco Colapinto",    "Alpine"),
        ("ALB", "Alex Albon",          "Williams"),
        ("TSU", "Yuki Tsunoda",        "Red Bull Racing"),
        ("HUL", "Nico Hulkenberg",     "Sauber"),
        ("BOR", "Gabriel Bortoleto",   "Sauber"),
        ("LAW", "Liam Lawson",         "Racing Bulls"),
        ("HAD", "Isack Hadjar",        "Racing Bulls"),
        ("BEA", "Oliver Bearman",      "Haas"),
        ("OCO", "Esteban Ocon",        "Haas"),
    ]

    compounds    = ["SOFT", "MEDIUM", "HARD"]
    current_lap  = 38
    total_laps   = 78

    drivers = []
    for pos, (abbr, name, team) in enumerate(driver_pool, start=1):
        gap      = 0.0 if pos == 1 else round(rng.uniform(0.4, 40.0) * (pos ** 0.65), 3)
        tyre_age = rng.randint(5, 28)
        compound = rng.choice(compounds)
        history  = [max(1, pos + rng.randint(-3, 3)) for _ in range(10)]
        history.append(pos)
        drivers.append(DriverState(
            abbreviation    = abbr,
            full_name       = name,
            team            = team,
            position        = pos,
            gap_to_leader_s = gap,
            tyre_compound   = compound,
            tyre_age_laps   = tyre_age,
            lap_time_last   = round(rng.uniform(72.0, 88.0), 3),
            lap_time_std    = round(rng.uniform(0.2, 1.5), 3),
            laps_completed  = current_lap,
            is_retired      = False,
            position_history= history,
        ))

    return RaceState(
        session_name         = f"{year} {gp_name} Race (DEMO)",
        circuit              = gp_name,
        total_laps           = total_laps,
        current_lap          = current_lap,
        safety_car_active    = False,
        red_flag_active      = False,
        drivers              = drivers,
        last_updated         = datetime.utcnow(),
        stale                = False,
        circuit_overtaking_index = 1,
    )


# ── Utilities ──────────────────────────────────────────────────────────────

def _safe_int(val: object, default: int = 0) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def _safe_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def _td_to_s(td: object) -> Optional[float]:
    try:
        return td.total_seconds()  # type: ignore[union-attr]
    except Exception:
        return None

def _normalise(location: str) -> str:
    return location.lower().replace(" ", "_").replace("-", "_")