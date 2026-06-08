"""
models/predictor.py
───────────────────
Loads xgb_model.pkl and produces per-driver expected finishing position.
Returns None gracefully if the model file does not exist.

PIPELINE GUARANTEE: uses only LIVE STATE features — never official results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.config import MODEL_PATH, MONTE_CARLO_WEIGHT, ML_WEIGHT
from utils.logger import get_logger

log = get_logger("predictor")


class RacePredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self._model_path = Path(model_path)
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self._model_path.exists():
            log.warning(
                "No trained model found at %s. "
                "Running Monte Carlo-only mode. "
                "Run scripts/train_model.py to train.",
                self._model_path,
            )
            return
        try:
            import joblib
            self._model = joblib.load(self._model_path)
            log.info("XGBoost model loaded from %s", self._model_path)
        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            self._model = None

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def predict_positions(self, features: pd.DataFrame) -> Optional[dict[str, float]]:
        """
        Returns dict[driver_abbr -> expected_finish_position] or None.
        """
        if not self.has_model:
            return None
        try:
            preds = self._model.predict(features.values)
            return {abbr: float(pred) for abbr, pred in zip(features.index, preds)}
        except Exception as exc:
            log.error("Model inference error: %s", exc)
            return None


def blend_predictions(
    mc_results: dict,
    ml_positions: Optional[dict[str, float]],
    n_drivers: int,
) -> dict:
    """
    Blend Monte Carlo win % with ML expected position.

    Strategy:
    - If no ML model: return MC results unchanged.
    - If ML available: convert ML expected positions to win probabilities
      and blend with MC at MONTE_CARLO_WEIGHT / ML_WEIGHT ratio.
    """
    if ml_positions is None:
        return mc_results

    # Convert expected position to win probability (inverse rank, normalised)
    inv_scores = {
        abbr: 1.0 / max(pos, 1.0) for abbr, pos in ml_positions.items()
    }
    total = sum(inv_scores.values()) or 1.0
    ml_win_pcts = {abbr: (v / total) * 100 for abbr, v in inv_scores.items()}

    # Blend
    blended = {}
    for abbr, mc in mc_results.items():
        ml_win = ml_win_pcts.get(abbr, mc.win_pct)
        blended_win = MONTE_CARLO_WEIGHT * mc.win_pct + ML_WEIGHT * ml_win

        # Scale podium / top5 proportionally to preserve ordering
        scale = blended_win / max(mc.win_pct, 0.01)
        blended_podium = min(100.0, mc.podium_pct * scale)
        blended_top5 = min(100.0, mc.top5_pct * scale)

        # Clone and update
        from models.monte_carlo import MCResult
        blended[abbr] = MCResult(
            driver=abbr,
            win_pct=round(blended_win, 2),
            podium_pct=round(blended_podium, 2),
            top5_pct=round(blended_top5, 2),
            expected_finish=round(ml_positions.get(abbr, mc.expected_finish), 2),
        )

    return blended
