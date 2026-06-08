"""
dashboard/components.py — Reusable chart/table components
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils.config import CHART_TOP_N

DRIVER_COLORS = ["#E10600","#FF8000","#00D2BE","#0090FF","#39B54A",
                 "#FFFFFF","#B6BABD","#FFF500","#006F62","#2B4562"]

_CHART_LAYOUT = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(family="Barlow Condensed, sans-serif", color="#AAAAAA", size=12),
    margin        = dict(l=10, r=20, t=40, b=20),
)


# ── Race Table ─────────────────────────────────────────────────────────────
def render_race_table(drivers: list[dict[str, Any]]) -> None:
    if not drivers:
        st.info("Waiting for race data…")
        return
    rows = []
    for d in drivers:
        compound: str = str(d.get("tyre_compound", "UNKNOWN")).upper()
        gap: float    = float(d.get("gap_to_leader_s", 0.0))
        rows.append({
            "Pos":      int(d.get("position", 99)),
            "Driver":   str(d.get("abbreviation", "–")),
            "Team":     str(d.get("team", "–")),
            "Gap":      "LEADER" if gap == 0.0 else f"+{gap:.3f}s",
            "Tyre":     f"{_tyre_dot(compound)} {compound}",
            "Tyre Age": f"{int(d.get('tyre_age_laps', 0))} laps",
        })
    df = pd.DataFrame(rows)

    def _style(row: pd.Series) -> list[str]:
        pos = row["Pos"]
        try:
            p = int(pos)
        except (TypeError, ValueError):
            return [""] * len(row)
        if p == 1:
            return ["color:#E10600;font-weight:700"] + [""] * (len(row)-1)
        if p <= 3:
            return ["color:#FFC906;font-weight:600"] + [""] * (len(row)-1)
        return [""] * len(row)

    st.dataframe(df.style.apply(_style, axis=1), width='stretch', hide_index=True)


# ── Probability Table ──────────────────────────────────────────────────────
def render_probability_table(prob_rows: list[dict[str, Any]]) -> None:
    if not prob_rows:
        st.info("No prediction data yet…")
        return
    rows = []
    for d in prob_rows:
        rows.append({
            "Driver":      str(d.get("abbreviation", "–")),
            "Win %":       f"{float(d.get('win_pct', 0.0)):.1f}%",
            "Podium %":    f"{float(d.get('podium_pct', 0.0)):.1f}%",
            "Top 5 %":     f"{float(d.get('top5_pct', 0.0)):.1f}%",
            "Exp. Finish": float(d.get("expected_finish", 0.0)),
        })
    df = pd.DataFrame(rows)

    def _style(row: pd.Series) -> list[str]:
        idx: int
        try:
            idx = int(row.name)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return [""] * len(row)
        if idx == 0:
            return ["color:#E10600;font-weight:700"] + [""] * (len(row)-1)
        if idx <= 2:
            return ["color:#FFC906;font-weight:600"] + [""] * (len(row)-1)
        return [""] * len(row)

    st.dataframe(df.style.apply(_style, axis=1), width='stretch', hide_index=True)


# ── Win Probability Bar Chart ──────────────────────────────────────────────
def render_win_probability_chart(prob_rows: list[dict[str, Any]]) -> None:
    if not prob_rows:
        return
    top      = prob_rows[:CHART_TOP_N]
    drivers  = [str(r["abbreviation"]) for r in top]
    win_pcts = [float(r["win_pct"])    for r in top]

    fig = go.Figure(go.Bar(
        x=win_pcts, y=drivers, orientation="h",
        marker=dict(color=DRIVER_COLORS[:len(drivers)]),
        text=[f"{v:.1f}%" for v in win_pcts],
        textposition="outside",
        textfont=dict(family="Barlow Condensed", size=13, color="#FFF"),
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="WIN PROBABILITY", font=dict(size=11, color="#555", family="Barlow Condensed"), x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13, color="#FFF", family="Barlow Condensed")),
        height=250,
    )
    st.plotly_chart(fig, width='stretch')


# ── Position Tracker ───────────────────────────────────────────────────────
def render_position_tracker(history_data: dict[str, list[int]]) -> None:
    if not history_data:
        return
    sorted_drivers = sorted(
        history_data.items(), key=lambda kv: kv[1][-1] if kv[1] else 99
    )[:CHART_TOP_N]
    fig = go.Figure()
    for i, (abbr, positions) in enumerate(sorted_drivers):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(positions)+1)), y=positions,
            mode="lines+markers", name=abbr,
            line=dict(color=DRIVER_COLORS[i % len(DRIVER_COLORS)], width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="POSITION TRACKER", font=dict(size=11, color="#555", family="Barlow Condensed"), x=0),
        xaxis=dict(title="Update", gridcolor="#1C1C22", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", title="Pos", dtick=1, gridcolor="#1C1C22", tickfont=dict(size=11)),
        height=270,
        legend=dict(orientation="h", y=1.08, font=dict(size=11, family="Barlow Condensed")),
    )
    st.plotly_chart(fig, width='stretch')


# ── Win % Evolution ────────────────────────────────────────────────────────
def render_probability_evolution(win_history: dict[str, list[float]]) -> None:
    if not win_history:
        return
    fig = go.Figure()
    for i, (abbr, history) in enumerate(list(win_history.items())[:3]):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history)+1)), y=history,
            mode="lines+markers", name=abbr,
            line=dict(color=DRIVER_COLORS[i % len(DRIVER_COLORS)], width=2),
        ))
    fig.update_layout(
        **_CHART_LAYOUT,
        title=dict(text="WIN % EVOLUTION", font=dict(size=11, color="#555", family="Barlow Condensed"), x=0),
        xaxis=dict(title="Update", gridcolor="#1C1C22", tickfont=dict(size=11)),
        yaxis=dict(title="Win %", gridcolor="#1C1C22", tickfont=dict(size=11)),
        height=250,
        legend=dict(orientation="h", y=1.08, font=dict(size=11, family="Barlow Condensed")),
    )
    st.plotly_chart(fig, width='stretch')


# ── Insights (kept for backward compat — rendering moved to app.py) ────────
def render_insights(insights: list[str]) -> None:
    for insight in insights[:3]:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)


# ── Utilities ──────────────────────────────────────────────────────────────
def _tyre_dot(compound: str) -> str:
    return {"SOFT":"●","MEDIUM":"●","HARD":"●","INTERMEDIATE":"●","WET":"●"}.get(compound, "●")