"""
main.py
───────
Launch the F1 Live Predictor Dashboard.

Usage:
    python main.py                         # launches Streamlit UI
    python main.py --train                 # runs offline model training instead
    python main.py --demo                  # forces demo mode in Streamlit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def launch_dashboard(demo: bool = False) -> None:
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.headless", "true",
    ]
    if demo:
        print("🎮 Launching in demo mode…")
    else:
        print("🏎️ Launching F1 Live Predictor Dashboard…")
    subprocess.run(cmd, check=True)


def train_model() -> None:
    print("🧠 Starting offline model training…")
    subprocess.run([sys.executable, "scripts/train_model.py"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 Live Predictor Dashboard")
    parser.add_argument("--train", action="store_true", help="Run offline training instead of dashboard")
    parser.add_argument("--demo", action="store_true", help="Force demo mode")
    args = parser.parse_args()

    # Ensure cache + models dirs exist
    Path("cache").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("exports").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    if args.train:
        train_model()
    else:
        launch_dashboard(demo=args.demo)


if __name__ == "__main__":
    main()
