"""
scripts/diagnose.py
───────────────────
Run this to diagnose FastF1 connectivity issues.

    python scripts/diagnose.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=" * 60)
print("  F1 Live Predictor — FastF1 Diagnostic")
print("=" * 60)

# ── 1. Check FastF1 version ────────────────────────────────────────────────
print("\n[1] Checking FastF1 version...")
try:
    import fastf1
    print(f"    FastF1 version: {fastf1.__version__}")
except Exception as e:
    print(f"    ERROR importing FastF1: {e}")
    sys.exit(1)

# ── 2. Enable cache ────────────────────────────────────────────────────────
print("\n[2] Enabling cache...")
try:
    import os
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache/")
    print("    Cache enabled at cache/")
except Exception as e:
    print(f"    ERROR enabling cache: {e}")

# ── 3. Try loading event schedule ─────────────────────────────────────────
print("\n[3] Loading 2024 event schedule...")
try:
    schedule = fastf1.get_event_schedule(2024, include_testing=False)
    print(f"    OK — found {len(schedule)} events")
    print(f"    First event: {schedule.iloc[0]['EventName']}")
    print(f"    Last event:  {schedule.iloc[-1]['EventName']}")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 4. Try loading a known past session ───────────────────────────────────
print("\n[4] Loading 2023 Monaco Race session (metadata only)...")
try:
    session = fastf1.get_session(2023, "Monaco", "R")
    print(f"    Session object created: {session}")
except Exception as e:
    print(f"    ERROR creating session: {e}")
    sys.exit(1)

print("\n[5] Loading session data (laps only, no telemetry)...")
try:
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = session.laps
    print(f"    OK — loaded {len(laps)} lap records")
    print(f"    Drivers found: {laps['Driver'].unique().tolist()}")
    print(f"    Max lap: {laps['LapNumber'].max()}")
except Exception as e:
    print(f"    ERROR loading session data: {e}")
    import traceback
    traceback.print_exc()

# ── 5. Try 2026 schedule ──────────────────────────────────────────────────
print("\n[6] Loading 2026 event schedule...")
try:
    schedule_26 = fastf1.get_event_schedule(2026, include_testing=False)
    print(f"    OK — found {len(schedule_26)} events")
    recent = schedule_26[schedule_26['EventDate'] <= __import__('datetime').datetime.now()]
    if not recent.empty:
        print(f"    Most recent completed event: {recent.iloc[-1]['EventName']} on {recent.iloc[-1]['EventDate'].date()}")
    else:
        print("    No past events found in 2026 schedule")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 6. Try 2026 Monaco ────────────────────────────────────────────────────
print("\n[7] Attempting to load 2026 Monaco Race...")
try:
    session_26 = fastf1.get_session(2026, "Monaco", "R")
    session_26.load(laps=True, telemetry=False, weather=False, messages=False)
    laps_26 = session_26.laps
    print(f"    OK — loaded {len(laps_26)} lap records")
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("  Diagnostic complete — share the output above.")
print("=" * 60)
