"""
app/services/model_registry.py
=================================
Loads every model ONCE at application startup (see app/main.py's lifespan
handler) and gives the rest of the app a single lookup point. This avoids
re-loading a multi-MB model artifact from disk on every request, which
would defeat the point of "low-latency async endpoints."

HOW THE REAL/MOCK SWITCH WORKS:
  For each model, this tries to load the real artifact-backed class
  first (XGBoostModel, RandomForestModel, SarimaxModel, LstmModel). If
  the required files aren't in MODEL_ARTIFACTS_DIR yet, loading raises
  (missing file / import error), and the registry falls back to
  MockForecastModel for THAT model only - so you can drop in artifacts
  one model at a time and the rest keep working as mocks meanwhile.
  Check GET /api/health or the startup logs to see which models are real
  vs. mock at any point.
"""

import logging
from pathlib import Path

from app.config import MODEL_ARTIFACTS_DIR
from app.models.base import BaseForecastModel
from app.models.mock_predictor import MockForecastModel
from app.models.xgboost_model import XGBoostModel
from app.models.random_forest_model import RandomForestModel
from app.models.sarimax_model import SarimaxModel
from app.models.lstm_model import LstmModel
from app.models.ensemble_model import EnsembleModel

logger = logging.getLogger("freightai.model_registry")

_MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "sarimax": "SARIMAX",
    "lstm": "LSTM",
}

_REAL_CLASSES = {
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "sarimax": SarimaxModel,
    "lstm": LstmModel,
}


def _load_one(model_id: str, label: str, artifacts_dir: Path) -> tuple[BaseForecastModel, bool]:
    """Try the real class; fall back to mock for this model only. Returns (model, is_real)."""
    real_cls = _REAL_CLASSES[model_id]
    try:
        model = real_cls()
        model.load(artifacts_dir)
        logger.info("Loaded REAL model: %s", model_id)
        return model, True
    except Exception as exc:
        logger.warning(
            "Could not load real artifacts for '%s' (%s). Falling back to mock. "
            "Drop the real files into %s to activate it.",
            model_id, exc, artifacts_dir,
        )
        mock = MockForecastModel(model_id=model_id, label=label)
        mock.load(artifacts_dir)
        return mock, False


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, BaseForecastModel] = {}
        self._real_status: dict[str, bool] = {}

    def load_all(self) -> None:
        """Instantiate + load() every model once. Call at app startup."""
        for model_id, label in _MODEL_LABELS.items():
            model, is_real = _load_one(model_id, label, MODEL_ARTIFACTS_DIR)
            self._models[model_id] = model
            self._real_status[model_id] = is_real

        # Ensemble is special: it composes xgboost + random_forest rather
        # than loading its own artifact. It goes real only if BOTH its
        # components are real.
        ensemble = EnsembleModel()
        ensemble.load(MODEL_ARTIFACTS_DIR)
        xgb_real = self._real_status.get("xgboost", False)
        rf_real = self._real_status.get("random_forest", False)

        if xgb_real and rf_real:
            ensemble.set_components(self._models["xgboost"], self._models["random_forest"])
            self._models["ensemble"] = ensemble
            self._real_status["ensemble"] = True
            logger.info("Loaded REAL model: ensemble (xgboost + random_forest)")
        else:
            mock = MockForecastModel(model_id="ensemble", label="Ensemble")
            mock.load(MODEL_ARTIFACTS_DIR)
            self._models["ensemble"] = mock
            self._real_status["ensemble"] = False
            logger.warning(
                "Ensemble falling back to mock - requires BOTH xgboost and "
                "random_forest to be real (xgboost_real=%s, random_forest_real=%s)",
                xgb_real, rf_real,
            )

    def get(self, model_id: str) -> BaseForecastModel:
        if model_id not in self._models:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._models[model_id]

    def list_all(self) -> list[BaseForecastModel]:
        return list(self._models.values())

    def real_status(self) -> dict[str, bool]:
        """Which models are running real artifacts vs. mock fallback."""
        return dict(self._real_status)

    @property
    def is_ready(self) -> bool:
        return bool(self._models) and all(m.is_loaded for m in self._models.values())


# Module-level singleton — imported by main.py (to call load_all() at
# startup) and by routers (via dependency injection, see routers/deps.py).
registry = ModelRegistry()
