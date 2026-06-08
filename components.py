"""
dashboard/components.py
───────────────────────
Reusable Streamlit display components for the F1 Live Predictor Dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config import TYRE_COLORS, CHART_TOP_N

# ── Team colour palette ────────────────────────────────────────────────────
DRIVER_COLORS = ["#E10600", "#FF8000", "#00D2BE", "#0090FF", "#39B54A",
                 "#FF1E00", "#B6BABD", "#FFF500", "#006F62", "#2B4562"]


# ── Live Race Table ────────────────────────────────────────────────────────

def render_race_table(drivers: list[dict]) -> None:
    if not drivers:
        st.info("Waiting for race data…")
        return

    rows = []
    for d in drivers:
        compound = d.get("tyre_compound", "UNKNOWN").upper()
        tyre_emoji = _tyre_emoji(compound)
        gap = d.get("gap_to_leader_s", 0.0)
        gap_str = "— LEADER —" if gap == 0.0 else f"+{gap:.3f}s"

        rows.append({
            "Pos": int(d.get("position", 99)),
            "Driver": d.get("abbreviation", "–"),
            "Team": d.get("team", "–"),
            "Gap": gap_str,
            "Tyre": f"{tyre_emoji} {compound}",
            "Tyre Age": f"{d.get('tyre_age_laps', 0)} laps",
        })

    df = pd.DataFrame(rows)

    def _row_style(row):
        pos = row["Pos"]
        try:
            pos = int(pos)
        except (TypeError, ValueError):
            return [""] * len(row)
        if pos == 1:
            return ["background-color: rgba(255,215,0,0.15)"] * len(row)
        if pos <= 3:
            return ["background-color: rgba(192,192,192,0.10)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )


# ── Probability Table ──────────────────────────────────────────────────────

def render_probability_table(prob_rows: list[dict]) -> None:
    if not prob_rows:
        st.info("No prediction data yet…")
        return

    rows = []
    for d in prob_rows:
        rows.append({
            "Driver": d.get("abbreviation", "–"),
            "Win %": f"{d.get('win_pct', 0.0):.1f}%",
            "Podium %": f"{d.get('podium_pct', 0.0):.1f}%",
            "Top 5 %": f"{d.get('top5_pct', 0.0):.1f}%",
            "Exp. Finish": d.get("expected_finish", "–"),
        })

    df = pd.DataFrame(rows)

    def _row_style(row):
        try:
            idx = int(row.name)
        except (TypeError, ValueError):
            return [""] * len(row)
        if idx == 0:
            return ["background-color: rgba(255,215,0,0.20)"] * len(row)
        if idx <= 2:
            return ["background-color: rgba(192,192,192,0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )


# ── Win Probability Bar Chart ──────────────────────────────────────────────

def render_win_probability_chart(prob_rows: list[dict]) -> None:
    if not prob_rows:
        return

    top = prob_rows[:CHART_TOP_N]
    drivers = [r["abbreviation"] for r in top]
    win_pcts = [r["win_pct"] for r in top]

    fig = go.Figure(go.Bar(
        x=win_pcts,
        y=drivers,
        orientation="h",
        marker=dict(
            color=DRIVER_COLORS[:len(drivers)],
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{v:.1f}%" for v in win_pcts],
        textposition="outside",
        textfont=dict(family="Montserrat, sans-serif", size=12, color="#FFFFFF"),
    ))
    fig.update_layout(
        title=dict(text="WIN PROBABILITY", font=dict(family="Montserrat, sans-serif", size=13, color="#999"), x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed", tickfont=dict(family="Montserrat, sans-serif", size=13, color="#FFF")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        height=260,
        margin=dict(l=10, r=60, t=35, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Position Tracker ───────────────────────────────────────────────────────

def render_position_tracker(history_data: dict[str, list[int]]) -> None:
    if not history_data:
        return

    sorted_drivers = sorted(
        history_data.items(), key=lambda kv: kv[1][-1] if kv[1] else 99
    )[:CHART_TOP_N]

    fig = go.Figure()
    for i, (abbr, positions) in enumerate(sorted_drivers):
        x = list(range(1, len(positions) + 1))
        fig.add_trace(go.Scatter(
            x=x, y=positions,
            mode="lines+markers",
            name=abbr,
            line=dict(color=DRIVER_COLORS[i % len(DRIVER_COLORS)], width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=dict(text="POSITION TRACKER — TOP 5", font=dict(family="Montserrat, sans-serif", size=13, color="#999"), x=0),
        xaxis=dict(title="Update", tickfont=dict(family="Montserrat", size=11), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(autorange="reversed", title="Position", dtick=1,
                   tickfont=dict(family="Montserrat", size=11), gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(family="Montserrat", size=11)),
        margin=dict(l=10, r=20, t=45, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Win % Evolution Chart ──────────────────────────────────────────────────

def render_probability_evolution(win_history: dict[str, list[float]]) -> None:
    if not win_history:
        return

    fig = go.Figure()
    for i, (abbr, history) in enumerate(list(win_history.items())[:3]):
        x = list(range(1, len(history) + 1))
        fig.add_trace(go.Scatter(
            x=x, y=history,
            mode="lines+markers",
            name=abbr,
            line=dict(color=DRIVER_COLORS[i % len(DRIVER_COLORS)], width=2),
            fill="tozeroy",
            fillcolor=f"rgba({_hex_to_rgb(DRIVER_COLORS[i])},0.08)",
        ))

    fig.update_layout(
        title=dict(text="WIN % EVOLUTION — TOP 3", font=dict(family="Montserrat, sans-serif", size=13, color="#999"), x=0),
        xaxis=dict(title="Update", tickfont=dict(family="Montserrat", size=11), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Win %", tickfont=dict(family="Montserrat", size=11), gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(family="Montserrat", size=11)),
        margin=dict(l=10, r=20, t=45, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Race Insight Panel ─────────────────────────────────────────────────────

def render_insights(insights: list[str]) -> None:
    if not insights:
        return
    st.markdown("### 🧠 Race Insights")
    cols = st.columns(len(insights))
    for col, insight in zip(cols, insights[:3]):
        with col:
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-left: 3px solid #E10600;
                border-radius: 8px;
                padding: 0.85rem 1rem;
                font-family: 'Montserrat', sans-serif;
                font-size: 0.82rem;
                line-height: 1.5;
                color: #DDD;
            ">{insight}</div>
            """, unsafe_allow_html=True)


# ── Utilities ──────────────────────────────────────────────────────────────

def _tyre_emoji(compound: str) -> str:
    return {"SOFT": "🔴", "MEDIUM": "🟡", "HARD": "⚪",
            "INTERMEDIATE": "🟢", "WET": "🔵"}.get(compound, "⬛")


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
