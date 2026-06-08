"""
dashboard/app.py — F1 Live Predictor Dashboard
"""

from __future__ import annotations
import os, sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from data.fastf1_poller import F1Poller, RaceState, create_demo_state
from data.feature_builder import build_features
from models.monte_carlo import run_monte_carlo
from models.predictor import RacePredictor, blend_predictions
from models.monte_carlo import MCResult
from dashboard.components import (
    render_race_table, render_probability_table,
    render_win_probability_chart, render_position_tracker,
    render_probability_evolution, render_insights,
)
from utils.config import (
    AUTOREFRESH_INTERVAL_MS, DASHBOARD_TITLE,
    TYRE_AGE_PIT_THRESHOLD, UPDATE_INTERVAL_SECONDS,
)
from utils.logger import get_logger

log = get_logger("app")


def _generate_insights(state: RaceState, sorted_preds: list[MCResult]) -> list[str]:
    insights: list[str] = []
    if state.safety_car_active:
        n = sum(1 for d in state.drivers if d.gap_to_leader_s <= 5.0)
        insights.append(f"Safety car compresses the field — {n} drivers within 5 seconds of the lead.")
    for d in state.drivers[:10]:
        if d.tyre_age_laps > TYRE_AGE_PIT_THRESHOLD:
            insights.append(f"{d.abbreviation} on aging tyres ({d.tyre_age_laps} laps) — pit window likely approaching.")
            if len(insights) >= 3:
                return insights
    for d in state.drivers:
        hist = d.position_history
        if len(hist) >= 6:
            gain = hist[-6] - hist[-1]
            if gain >= 2:
                insights.append(f"{d.abbreviation} on the move — up {gain} places in the last 5 updates.")
            elif gain <= -2:
                insights.append(f"{d.abbreviation} dropping back — lost {abs(gain)} positions in the last 5 updates.")
        if len(insights) >= 3:
            return insights
    for d in state.drivers:
        if d.gap_to_leader_s > 30.0 and d.position <= 8:
            insights.append(f"{d.abbreviation} is {d.gap_to_leader_s:.1f}s off the lead — possible strategy divergence.")
        if len(insights) >= 3:
            return insights
    laps_left = state.total_laps - state.current_lap
    if laps_left <= 10 and sorted_preds:
        leader = sorted_preds[0]
        insights.append(f"{laps_left} laps remaining — {leader.driver} leads with {leader.win_pct:.0f}% win probability.")
    return insights[:3]


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Live Predictor",
    page_icon=":racing_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,700&family=Barlow+Condensed:wght@400;500;600;700;800;900&display=swap');

/* Root font */
*, html, body, [class*="css"], p, div, span, td, th {
    font-family: 'Barlow', sans-serif !important;
}

/* Dark base */
.stApp { background: #111116 !important; }
.stApp > header { background: #111116 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D0D10 !important;
    border-right: 1px solid #222228 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Barlow', sans-serif !important;
    color: #CCCCCC !important;
}
[data-testid="stSidebar"] h2 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
    color: #E10600 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #888 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #E10600 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 0.5rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #FF1A00 !important;
}

/* F1 Header — flat, no gradient */
.f1-header {
    background: #E10600;
    padding: 1.2rem 1.8rem;
    border-radius: 0px;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    border-left: 6px solid #111116;
}
.f1-header-left {
    border-right: 1px solid rgba(255,255,255,0.25);
    padding-right: 1.5rem;
}
.f1-header h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    color: white !important;
    margin: 0 !important;
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    line-height: 1 !important;
}
.f1-header p {
    color: rgba(255,255,255,0.75) !important;
    margin: 0.25rem 0 0 0 !important;
    font-size: 0.68rem !important;
    letter-spacing: 3px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}
.f1-lap-badge {
    background: #111116;
    color: #E10600;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 900;
    font-size: 1.8rem;
    letter-spacing: 2px;
    padding: 0.2rem 1rem;
    border-radius: 2px;
    line-height: 1.2;
}
.f1-lap-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600;
}

/* Section labels */
.section-label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #666 !important;
    margin: 1.2rem 0 0.5rem 0 !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #222228 !important;
    display: block !important;
}

/* Override h3 */
h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #666 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #18181E !important;
    border: 1px solid #222228 !important;
    border-radius: 3px !important;
    padding: 0.7rem 1rem !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.62rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #555 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    color: #FFFFFF !important;
    letter-spacing: 1px !important;
}

/* Status badges */
.badge {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.badge-sc  { background: #FFC906; color: #000000; }
.badge-rf  { background: #E10600; color: #FFFFFF; }
.badge-vsc { background: #FF8C00; color: #000000; }

/* Stale warning */
.stale-bar {
    background: #1A1200;
    border-left: 3px solid #FFC906;
    padding: 0.5rem 1rem;
    border-radius: 2px;
    margin-bottom: 1rem;
    color: #FFC906;
    font-size: 0.78rem;
    letter-spacing: 1px;
    font-weight: 600;
    font-family: 'Barlow Condensed', sans-serif !important;
    text-transform: uppercase;
}

/* Model status */
.model-status {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 2px;
}
.model-mc { background: #1A1200; color: #FFC906; border: 1px solid #332400; }
.model-ml { background: #001A15; color: #00D2BE; border: 1px solid #003328; }

/* Insight cards */
.insight-card {
    background: #18181E;
    border: 1px solid #222228;
    border-left: 3px solid #E10600;
    border-radius: 2px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #CCCCCC;
    font-family: 'Barlow', sans-serif !important;
}

/* Divider */
hr { border-color: #222228 !important; margin: 1rem 0 !important; }

/* Footer */
.f1-footer {
    text-align: center;
    color: #333;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1C1C22;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600;
}

/* Streamlit dataframe tweaks */
[data-testid="stDataFrame"] {
    border: 1px solid #222228 !important;
    border-radius: 2px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────
refresh_count: int = st_autorefresh(interval=AUTOREFRESH_INTERVAL_MS, limit=None, key="f1_ar")

# ── Session state init ─────────────────────────────────────────────────────
if "predictor" not in st.session_state:
    st.session_state.predictor = RacePredictor()
if "win_history" not in st.session_state:
    st.session_state["win_history"]: dict[str, list[float]] = {}
if "poller" not in st.session_state:
    st.session_state.poller = None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Session Config")
    mode = st.radio("Data source", ["Live FastF1", "Demo Mode"], index=1)
    demo_mode = (mode == "Demo Mode")

    demo_gp   = "Monaco"
    demo_year = 2024

    if not demo_mode:
        year         = st.number_input("Season", min_value=2018, max_value=2025, value=2024)
        gp           = st.text_input("Grand Prix", value="Monaco")
        session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3"], index=0)
        if st.button("Start Live Polling"):
            if st.session_state.poller:
                st.session_state.poller.stop()
            poller = F1Poller(year=int(year), gp=gp, session_type=session_type)
            poller.start()
            st.session_state.poller = poller
            st.success(f"Polling {year} {gp} {session_type}")
        if st.button("Stop Polling"):
            if st.session_state.poller:
                st.session_state.poller.stop()
                st.session_state.poller = None
    else:
        demo_gp   = st.text_input("GP name", value="Monaco")
        demo_year = int(st.number_input("Year", min_value=2018, max_value=2025, value=2024))

    st.divider()
    has_model   = st.session_state.predictor.has_model
    badge_cls   = "model-ml" if has_model else "model-mc"
    badge_label = "ML + Monte Carlo active" if has_model else "Monte Carlo only"
    st.markdown(f'<span class="model-status {badge_cls}">{badge_label}</span>', unsafe_allow_html=True)
    st.markdown(f"<br><small style='color:#444;letter-spacing:1px;font-size:0.65rem;'>REFRESH EVERY {UPDATE_INTERVAL_SECONDS}s</small>", unsafe_allow_html=True)
    st.caption("Run scripts/train_model.py to enable XGBoost.")


# ── Get race state ─────────────────────────────────────────────────────────
def get_state() -> RaceState | None:
    if demo_mode:
        import random
        rng  = random.Random(refresh_count % 100)
        base = create_demo_state(gp_name=demo_gp, year=demo_year)
        base.current_lap = min(base.total_laps, 30 + refresh_count * 2)
        for d in base.drivers:
            d.laps_completed = base.current_lap
            d.position_history.append(max(1, d.position + rng.randint(-1, 1)))
        return base
    poller = st.session_state.poller
    return poller.latest_state if poller else None


state = get_state()

# ── Header ─────────────────────────────────────────────────────────────────
lap_display = f"{state.current_lap}/{state.total_laps}" if state else "–/–"
circuit_display = state.circuit if state else "–"

st.markdown(f"""
<div class="f1-header">
    <div class="f1-header-left">
        <h1>F1 Live Predictor</h1>
        <p>Race Prediction Engine &nbsp;·&nbsp; Monte Carlo + XGBoost &nbsp;·&nbsp; FastF1</p>
    </div>
    <div style="margin-left:auto;text-align:right;">
        <div class="f1-lap-label">Lap</div>
        <div class="f1-lap-badge">{lap_display}</div>
    </div>
    <div style="text-align:right;padding-left:1.2rem;border-left:1px solid rgba(255,255,255,0.2);">
        <div class="f1-lap-label">Circuit</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:1rem;letter-spacing:2px;color:#fff;text-transform:uppercase;">{circuit_display}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if state is None:
    st.warning("No race data available. Switch to Demo Mode in the sidebar or start live polling.")
    st.stop()

if state.stale:
    st.markdown('<div class="stale-bar">Warning — data is stale. Last poll failed. Displaying last known state.</div>', unsafe_allow_html=True)

# ── Status badges ──────────────────────────────────────────────────────────
if state.red_flag_active:
    st.markdown('<span class="badge badge-rf">Red Flag</span>', unsafe_allow_html=True)
elif state.safety_car_active:
    st.markdown('<span class="badge badge-sc">Safety Car Deployed</span>', unsafe_allow_html=True)

# ── Metrics row ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Session",      state.session_name.split("(")[0].strip()[:22])
with c2: st.metric("Circuit",      state.circuit)
with c3: st.metric("Laps",         f"{state.current_lap} / {state.total_laps}")
with c4: st.metric("Track Status", "Safety Car" if state.safety_car_active else "Green Flag")
with c5: st.metric("Updated",      state.last_updated.strftime("%H:%M:%S"))

st.divider()

# ── Predictions ────────────────────────────────────────────────────────────
mc_results: dict[str, MCResult] = run_monte_carlo(state)
try:
    features     = build_features(state)
    ml_positions = st.session_state.predictor.predict_positions(features)
except Exception as exc:
    log.error("Feature/ML error: %s", exc)
    ml_positions = None

final_predictions: dict[str, MCResult] = blend_predictions(mc_results, ml_positions, len(state.drivers))
sorted_preds: list[MCResult] = sorted(final_predictions.values(), key=lambda r: r.win_pct, reverse=True)

# Update win history for top 3
win_history: dict[str, list[float]] = st.session_state["win_history"]
for result in sorted_preds[:3]:
    if result.driver not in win_history:
        win_history[result.driver] = []
    win_history[result.driver].append(float(result.win_pct))
    if len(win_history[result.driver]) > 60:
        win_history[result.driver] = win_history[result.driver][-60:]

prob_rows: list[dict[str, Any]] = [
    {"abbreviation": r.driver, "win_pct": float(r.win_pct),
     "podium_pct": float(r.podium_pct), "top5_pct": float(r.top5_pct),
     "expected_finish": float(r.expected_finish)}
    for r in sorted_preds
]
driver_dicts: list[dict[str, Any]] = [
    {"position": int(d.position), "abbreviation": str(d.abbreviation),
     "full_name": str(d.full_name), "team": str(d.team),
     "gap_to_leader_s": float(d.gap_to_leader_s),
     "tyre_compound": str(d.tyre_compound), "tyre_age_laps": int(d.tyre_age_laps)}
    for d in state.drivers if not d.is_retired
]

# ── Main layout ────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9])

with left:
    st.markdown('<span class="section-label">Live Race Order</span>', unsafe_allow_html=True)
    render_race_table(driver_dicts)
    st.markdown('<span class="section-label">Win / Podium Probabilities</span>', unsafe_allow_html=True)
    render_probability_table(prob_rows)

with right:
    render_win_probability_chart(prob_rows)
    history_data: dict[str, list[int]] = {
        d.abbreviation: d.position_history
        for d in state.drivers if d.position_history
    }
    render_position_tracker(history_data)
    render_probability_evolution(win_history)

# ── Insights ───────────────────────────────────────────────────────────────
st.divider()
st.markdown('<span class="section-label">Race Insights</span>', unsafe_allow_html=True)
insights = _generate_insights(state, sorted_preds)
if insights:
    cols = st.columns(len(insights))
    icons = ["", "", ""]
    for col, insight, icon in zip(cols, insights, icons):
        with col:
            st.markdown(f'<div class="insight-card">{icon} {insight}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
mode_label = "Demo" if demo_mode else "Live"
st.markdown(f"""
<div class="f1-footer">
    F1 Live Predictor &nbsp;·&nbsp; FastF1 + XGBoost + Monte Carlo &nbsp;·&nbsp;
    {state.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')} &nbsp;·&nbsp;
    {mode_label} &nbsp;·&nbsp; {'Stale' if state.stale else 'OK'}
</div>
""", unsafe_allow_html=True)
