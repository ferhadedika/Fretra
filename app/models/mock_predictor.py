"""
app/models/mock_predictor.py
==============================
A mock implementation of BaseForecastModel that runs with zero trained
artifacts. It generates plausible-looking SCFI forecasts (random walk off
the last known value) and synthetic SHAP-shaped output, using the SAME
feature names your real pipeline uses (brent_crude_lag1, cny_usd_lag1,
world_pmi, etc.) so the frontend and Feature Dictionary work unchanged
once real artifacts are swapped in.

HOW TO REPLACE THIS WITH A REAL MODEL:
  1. Create a new file, e.g. app/models/xgboost_model.py
  2. Subclass BaseForecastModel (see base.py for the contract)
  3. In load(): joblib.load(artifacts_dir / "xgboost.pkl") instead of
     the random.seed() call below
  4. In predict(): call self._model.predict(features) instead of the
     random-walk loop
  5. In explain(): use shap.TreeExplainer(self._model) instead of the
     synthetic dict
  6. In app/services/model_registry.py, swap MockForecastModel for your
     new class in the MODEL_CLASSES list — nothing else changes, because
     every router/service only talks to the BaseForecastModel interface.
"""

import random
from pathlib import Path
from typing import Optional

from app.models.anchor import ForecastAnchor
from app.models.base import BaseForecastModel

# Same feature set referenced in the FreightAI dashboard's feature dictionary.
# Keeping these names consistent means swapping in the real SHAP explainer
# later requires zero frontend changes.
_FEATURE_NAMES = [
    "scfi_lag_1",
    "scfi_lag_2",
    "brent_crude_lag1",
    "cny_usd_lag1",
    "world_pmi",
    "world_trade_volume",
    "seasonal_dummy_q4",
]

# Per-model accuracy figures and a small synthetic variance per model so the
# UI's "compare models" view shows believable, distinct numbers.
_MOCK_METRICS = {
    "xgboost":       {"mae": 45.2, "rmse": 60.1, "mape_pct": 3.2, "r2": 0.81},
    "random_forest": {"mae": 50.1, "rmse": 65.4, "mape_pct": 3.6, "r2": 0.77},
    "sarimax":       {"mae": 55.0, "rmse": 70.2, "mape_pct": 4.0, "r2": 0.70},
    "lstm":          {"mae": 48.0, "rmse": 63.0, "mape_pct": 3.4, "r2": 0.79},
    "ensemble":       {"mae": 42.0, "rmse": 58.0, "mape_pct": 3.0, "r2": 0.84},
}


class MockForecastModel(BaseForecastModel):
    """
    Drop-in stand-in for a real model. Constructed once per model_id by
    the ModelRegistry (see services/model_registry.py).
    """

    def __init__(self, model_id: str, label: str, last_known_scfi: float = 1850.0):
        super().__init__()
        self.model_id = model_id
        self.label = label
        self._last_known_scfi = last_known_scfi
        self._rng = random.Random(hash(model_id) % (2**32))

    def load(self, artifacts_dir: Path) -> None:
        # A real model would do something like:
        #   self._model = joblib.load(artifacts_dir / f"{self.model_id}.pkl")
        # There's nothing to load for the mock — just mark ready and
        # populate metrics so /models reports something meaningful.
        self.metrics = _MOCK_METRICS.get(self.model_id, {})
        self._loaded = True

    def predict(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> list[float]:
        # anchor is accepted-but-ignored: the mock has no real feature
        # cascade to re-anchor, it just needs the same call signature as
        # every real model class so a partially-mocked deployment (any
        # model still falling back here) doesn't break the shared interface.
        values = []
        current = self._last_known_scfi
        for _ in range(horizon_weeks):
            current = current + self._rng.gauss(0, 18)
            current = max(current, 100.0)  # keep it physically sane
            values.append(round(current, 2))
        return values

    def explain(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> tuple[dict, dict]:
        rng = self._rng
        global_importance = {
            feat: round(abs(rng.gauss(5, 3)), 3) for feat in _FEATURE_NAMES
        }
        local_importance = {
            feat: round(rng.gauss(0, 4), 3) for feat in _FEATURE_NAMES
        }
        return global_importance, local_importance
