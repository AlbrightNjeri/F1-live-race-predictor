# F1 Live Race Predictor Dashboard

A real-time Formula 1 race prediction dashboard that updates win/podium probabilities dynamically during a Grand Prix using live FastF1 data and a Monte Carlo + XGBoost prediction engine.

![F1 Live Predictor](https://img.shields.io/badge/F1-Live%20Predictor-E10600?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMiAxNWwtNS01IDEuNDEtMS40MUwxMCAxNC4xN2w3LjU5LTcuNTlMMTkgOGwtOSA5eiIvPjwvc3ZnPg==)

## Features

- **Live Race Polling** — polls FastF1 session data every 45 seconds
- **Monte Carlo Engine** — 5,000 simulations per update with DNF/safety car/red flag randomness
- **XGBoost ML Layer** — blends with Monte Carlo once trained (60/40 weighting)
- **4 Dashboard Panels** — live race order, win/podium probabilities, position tracker, probability evolution
- **Race Insights** — rule-based alerts for tyre age, position movers, strategy divergence
- **Demo Mode** — fully functional with synthetic 2026 grid data, no FastF1 connection required

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create cache directory
mkdir cache

# Run in Demo Mode (no data connection needed)
python -m streamlit run dashboard/app.py
```

Then open `http://localhost:8501` in your browser. Select **Demo Mode** in the sidebar.

## Live Mode

Switch to **Live FastF1** in the sidebar, enter the season year, Grand Prix name, and session type, then click **Start Live Polling**.

Works with any completed or in-progress F1 session from 2018 onwards.

Example — 2026 Monaco Race:
- Season: `2026`
- Grand Prix: `Monaco`
- Session: `R`

## Train the ML Model (optional)

Trains XGBoost on 2022–2024 official race results. Run once before a race weekend:

```bash
python scripts/train_model.py
```

This takes 10–20 minutes and saves the model to `models/xgb_model.pkl`. Without it, the dashboard runs Monte Carlo-only mode.

## Project Structure

```
f1-live-dashboard/
├── data/
│   ├── fastf1_poller.py      # polling loop, RaceState model
│   └── feature_builder.py    # LIVE STATE → ML features
├── models/
│   ├── monte_carlo.py        # 5,000-sim Monte Carlo engine
│   └── predictor.py          # XGBoost loader + MC/ML blender
├── dashboard/
│   ├── app.py                # Streamlit entry point
│   └── components.py         # charts and tables
├── scripts/
│   ├── train_model.py        # offline training (OFFICIAL STATE only)
│   └── diagnose.py           # FastF1 connectivity diagnostic
├── utils/
│   ├── config.py             # constants and circuit index
│   └── logger.py             # structured logging
└── requirements.txt
```

## Data Pipeline Design

Three strictly separated state layers:

| Layer | Source | Used for |
|---|---|---|
| LIVE STATE | `session.laps` polling | Dashboard predictions |
| PROVISIONAL STATE | Crossing-the-line order | Display only |
| OFFICIAL STATE | FIA final classifications | Model training only |

Live and official pipelines never share data.

## Diagnostics

If live data isn't loading:

```bash
python scripts/diagnose.py
```

## Tech Stack

- **FastF1** — F1 session data API
- **XGBoost** — finishing position regression
- **Streamlit** — dashboard UI
- **Plotly** — interactive charts
- **NumPy** — Monte Carlo simulations

## Requirements

- Python 3.10+
- FastF1 3.3+
- Internet connection for live data

## License

MIT
