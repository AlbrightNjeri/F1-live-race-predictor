"""
dashboard/app.py — F1 Live Predictor Dashboard (Streamlit entry point)
"""

from __future__ import annotations
import os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from data.fastf1_poller import F1Poller, RaceState, create_demo_state
from data.feature_builder import build_features
from models.monte_carlo import run_monte_carlo
from models.predictor import RacePredictor, blend_predictions
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

# ── Insight generator ──────────────────────────────────────────────────────
def _generate_insights(state: RaceState, sorted_preds: list) -> list[str]:
    insights: list[str] = []
    if state.safety_car_active:
        n = sum(1 for d in state.drivers if d.gap_to_leader_s <= 5.0)
        insights.append(f"🚗 Safety car compresses the field — {n} drivers within 5 seconds of the lead.")
    for d in state.drivers[:10]:
        if d.tyre_age_laps > TYRE_AGE_PIT_THRESHOLD:
            insights.append(f"⚡ {d.abbreviation} on aging tyres ({d.tyre_age_laps} laps) — pit window likely approaching.")
            if len(insights) >= 3:
                return insights
    for d in state.drivers:
        hist = d.position_history
        if len(hist) >= 6:
            gain = hist[-6] - hist[-1]
            if gain >= 2:
                insights.append(f"📈 {d.abbreviation} on the move — up {gain} places in the last 5 updates.")
            elif gain <= -2:
                insights.append(f"📉 {d.abbreviation} dropping back — lost {abs(gain)} positions in the last 5 updates.")
        if len(insights) >= 3:
            return insights
    for d in state.drivers:
        if d.gap_to_leader_s > 30.0 and d.position <= 8:
            insights.append(f"🔄 {d.abbreviation} is {d.gap_to_leader_s:.1f}s off the lead — possible strategy divergence.")
        if len(insights) >= 3:
            return insights
    laps_left = state.total_laps - state.current_lap
    if laps_left <= 10 and sorted_preds:
        leader = sorted_preds[0]
        insights.append(f"🏁 {laps_left} laps remaining — {leader.driver} leads with {leader.win_pct:.0f}% win probability.")
    return insights[:3]


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Live Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"], .stMarkdown, .stDataFrame, p, div {
    font-family: 'Poppins', sans-serif !important;
}

/* Dark base */
.stApp { background: #0a0a0f !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* Header banner */
.f1-header {
    background: linear-gradient(135deg, #E10600 0%, #8B0000 40%, #0a0a0f 100%);
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(225,6,0,0.3);
    position: relative;
    overflow: hidden;
}
.f1-header::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%);
    border-radius: 50%;
}
.f1-header h1 {
    font-family: 'Montserrat', sans-serif !important;
    color: white !important;
    margin: 0 !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase;
}
.f1-header p {
    color: rgba(255,255,255,0.6) !important;
    margin: 0.2rem 0 0 0 !important;
    font-size: 0.72rem !important;
    letter-spacing: 2.5px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
}

/* Section headers */
h3 { 
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 0.6rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #666 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    color: #FFF !important;
}

/* Stale warning */
.stale-warning {
    background: rgba(255,80,0,0.12);
    border-left: 3px solid #FF5000;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    color: #FF9966;
    font-size: 0.82rem;
}

/* SC / RF badges */
.sc-badge {
    display: inline-block;
    background: #FFC906;
    color: #000;
    border-radius: 5px;
    padding: 3px 10px;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.rf-badge {
    display: inline-block;
    background: #E10600;
    color: #fff;
    border-radius: 5px;
    padding: 3px 10px;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800;
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Panel cards */
.panel-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* Sidebar text */
[data-testid="stSidebar"] * { color: #CCC !important; }
[data-testid="stSidebar"] .stButton button {
    background: #E10600 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #444;
    font-size: 0.72rem;
    letter-spacing: 1px;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* Model status badge */
.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.model-mc { background: rgba(255,165,0,0.15); color: #FFA500; border: 1px solid rgba(255,165,0,0.3); }
.model-ml { background: rgba(0,210,190,0.15); color: #00D2BE; border: 1px solid rgba(0,210,190,0.3); }
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh ───────────────────────────────────────────────────────────
refresh_count = st_autorefresh(interval=AUTOREFRESH_INTERVAL_MS, limit=None, key="f1_autorefresh")

# ── Session state init ─────────────────────────────────────────────────────
if "predictor" not in st.session_state:
    st.session_state.predictor = RacePredictor()
if "win_history" not in st.session_state:
    st.session_state.win_history: dict[str, list[float]] = {}
if "poller" not in st.session_state:
    st.session_state.poller = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Session Config")
    mode = st.radio("Data source", ["🏎️ Live FastF1", "🎮 Demo Mode"], index=1)
    demo_mode = mode == "🎮 Demo Mode"

    if not demo_mode:
        year = st.number_input("Season", min_value=2018, max_value=2025, value=2024)
        gp = st.text_input("Grand Prix", value="Monaco")
        session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3"], index=0)
        if st.button("▶ Start Live Polling"):
            if st.session_state.poller:
                st.session_state.poller.stop()
            poller = F1Poller(year=int(year), gp=gp, session_type=session_type)
            poller.start()
            st.session_state.poller = poller
            st.session_state.demo_mode = False
            st.success(f"Polling {year} {gp} {session_type}")
        if st.button("⏹ Stop"):
            if st.session_state.poller:
                st.session_state.poller.stop()
                st.session_state.poller = None
    else:
        st.session_state.demo_mode = True
        demo_gp = st.text_input("Demo GP name", value="Monaco")
        demo_year = st.number_input("Demo year", min_value=2018, max_value=2025, value=2024)

    st.divider()
    has_model = st.session_state.predictor.has_model
    badge_class = "model-ml" if has_model else "model-mc"
    badge_label = "✅ ML + Monte Carlo" if has_model else "⚠️ Monte Carlo only"
    st.markdown(f'<span class="model-badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)
    st.markdown(f"<br><small>Refresh every {UPDATE_INTERVAL_SECONDS}s</small>", unsafe_allow_html=True)
    st.caption("Run `scripts/train_model.py` to enable XGBoost layer.")


# ── Get race state ─────────────────────────────────────────────────────────
def get_state() -> RaceState | None:
    if st.session_state.demo_mode:
        import random
        rng = random.Random(refresh_count % 100)
        gp_name = "Monaco"
        yr = 2024
        try:
            gp_name = demo_gp
            yr = int(demo_year)
        except Exception:
            pass
        base = create_demo_state(gp_name=gp_name, year=yr)
        base.current_lap = min(base.total_laps, 30 + refresh_count * 2)
        for d in base.drivers:
            d.laps_completed = base.current_lap
            d.position_history.append(max(1, d.position + rng.randint(-1, 1)))
        return base
    else:
        poller = st.session_state.poller
        return poller.latest_state if poller else None


state = get_state()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="f1-header">
    <h1>🏎️ {DASHBOARD_TITLE}</h1>
    <p>Live Race Prediction Engine &nbsp;·&nbsp; Monte Carlo + XGBoost &nbsp;·&nbsp; FastF1</p>
</div>
""", unsafe_allow_html=True)

if state is None:
    st.warning("No race data available. Start live polling from the sidebar or switch to Demo Mode.")
    st.stop()

if state.stale:
    st.markdown('<div class="stale-warning">⚠️ Data is stale — last poll failed. Displaying last known state.</div>', unsafe_allow_html=True)

# ── Top metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Session", state.session_name.split("(")[0].strip()[:20])
with c2: st.metric("Circuit", state.circuit)
with c3: st.metric("Lap", f"{state.current_lap} / {state.total_laps}")
with c4: st.metric("Track Status", "🟡 Safety Car" if state.safety_car_active else "🟢 Clear")
with c5: st.metric("Updated", state.last_updated.strftime("%H:%M:%S"))

if state.red_flag_active:
    st.markdown('<span class="rf-badge">🚩 Red Flag</span>', unsafe_allow_html=True)
elif state.safety_car_active:
    st.markdown('<span class="sc-badge">🚗 Safety Car Deployed</span>', unsafe_allow_html=True)

st.divider()

# ── Predictions ────────────────────────────────────────────────────────────
mc_results = run_monte_carlo(state)
try:
    features = build_features(state)
    ml_positions = st.session_state.predictor.predict_positions(features)
except Exception as exc:
    log.error("Feature/ML error: %s", exc)
    ml_positions = None

final_predictions = blend_predictions(mc_results, ml_positions, len(state.drivers))
sorted_preds = sorted(final_predictions.values(), key=lambda r: r.win_pct, reverse=True)

# Update win history
for result in sorted_preds[:3]:
    if result.driver not in st.session_state.win_history:
        st.session_state.win_history[result.driver] = []
    st.session_state.win_history[result.driver].append(result.win_pct)
    if len(st.session_state.win_history[result.driver]) > 60:
        st.session_state.win_history[result.driver] = st.session_state.win_history[result.driver][-60:]

prob_rows = [
    {"abbreviation": r.driver, "win_pct": r.win_pct,
     "podium_pct": r.podium_pct, "top5_pct": r.top5_pct,
     "expected_finish": r.expected_finish}
    for r in sorted_preds
]

driver_dicts = [
    {"position": d.position, "abbreviation": d.abbreviation,
     "full_name": d.full_name, "team": d.team,
     "gap_to_leader_s": d.gap_to_leader_s,
     "tyre_compound": d.tyre_compound, "tyre_age_laps": d.tyre_age_laps}
    for d in state.drivers if not d.is_retired
]

# ── Main layout ────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("### 🏁 Live Race Order")
    render_race_table(driver_dicts)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Win / Podium Probabilities")
    render_probability_table(prob_rows)

with right:
    render_win_probability_chart(prob_rows)
    history_data = {d.abbreviation: d.position_history for d in state.drivers if d.position_history}
    render_position_tracker(history_data)
    render_probability_evolution(st.session_state.win_history)

# ── Insights ───────────────────────────────────────────────────────────────
st.divider()
insights = _generate_insights(state, sorted_preds)
render_insights(insights)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    F1 LIVE PREDICTOR &nbsp;·&nbsp; FastF1 + XGBoost + Monte Carlo &nbsp;·&nbsp;
    {state.last_updated.strftime('%Y-%m-%d %H:%M:%S UTC')} &nbsp;·&nbsp;
    {'⚠️ STALE' if state.stale else '✅ LIVE'}
</div>
""", unsafe_allow_html=True)
