"""
app/models/ensemble_model.py
===============================
Real Ensemble model wrapper. Doesn't load its own artifact file - it
composes the four already-loaded component models (XGBoost, RandomForest,
SARIMAX, LSTM) using the SLSQP-optimised blend weights saved in
ensemble_config.json, exactly matching scfi_model_v3.py's:

    ens_preds = w[0]*xgb + w[1]*rf + w[2]*sarimax + w[3]*lstm

REQUIRES:
  ensemble_config.json   - {"weights": {"xgboost": .., "rf": .., "sarimax": .., "lstm": ..}}
  All four component model instances, already loaded, passed in via
  set_components() after the registry loads them (see
  app/services/model_registry.py for the wiring).

Each component's weight has a floor (10% at training time - see
MIN_ENSEMBLE_WEIGHT in scfi_model_v3.py) so this is always a genuine
4-way blend, never a single dominant model, at the cost of a slightly
higher CV MAE than the unconstrained optimum would give.

SHAP for the ensemble is the same weighted blend applied to each
component's SHAP values - mathematically consistent since the ensemble's
prediction IS the weighted sum of the components' predictions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from app.models.anchor import ForecastAnchor
from app.models.base import BaseForecastModel
from app.models.metrics_loader import load_metrics_for

logger = logging.getLogger("freightai.ensemble_model")

COMPONENT_IDS = ("xgboost", "rf", "sarimax", "lstm")


class EnsembleModel(BaseForecastModel):
    model_id = "ensemble"
    label = "Ensemble"

    def __init__(self):
        super().__init__()
        self._weights = {"xgboost": 0.25, "rf": 0.25, "sarimax": 0.25, "lstm": 0.25}
        self._components: dict[str, BaseForecastModel] = {}

    def set_components(
        self,
        xgb_model: BaseForecastModel,
        rf_model: BaseForecastModel,
        sarimax_model: BaseForecastModel,
        lstm_model: BaseForecastModel,
    ) -> None:
        """Called by ModelRegistry after all four component models are loaded."""
        self._components = {
            "xgboost": xgb_model,
            "rf": rf_model,
            "sarimax": sarimax_model,
            "lstm": lstm_model,
        }

    def load(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        cfg_path = artifacts_dir / "ensemble_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            weights = cfg.get("weights", {})
            if weights:
                self._weights = {k: float(weights.get(k, 0.0)) for k in COMPONENT_IDS}
            else:
                logger.warning(
                    "ensemble_config.json has no 'weights' key - keeping equal 25%% "
                    "default weights for %s", COMPONENT_IDS,
                )

        metrics_path = artifacts_dir.parent / "SCFI_model_metrics.json"
        self.metrics = load_metrics_for(metrics_path, "ensemble")

        # Marked loaded here; set_components() must still be called before
        # predict()/explain() work - ModelRegistry guarantees the order.
        self._loaded = True

    def _require_components(self) -> None:
        if len(self._components) != len(COMPONENT_IDS):
            raise RuntimeError(
                "EnsembleModel.set_components() must be called (with all four "
                "components) before predict()/explain()."
            )

    def predict(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> list[float]:
        self._require_components()
        per_model = {
            cid: model.predict(horizon_weeks, anchor=anchor)
            for cid, model in self._components.items()
        }
        return [
            round(sum(self._weights[cid] * per_model[cid][i] for cid in COMPONENT_IDS), 2)
            for i in range(horizon_weeks)
        ]

    def explain(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> tuple[dict, dict]:
        self._require_components()
        per_model_global = {}
        per_model_local = {}
        for cid, model in self._components.items():
            g, l = model.explain(horizon_weeks, anchor=anchor)
            per_model_global[cid] = g
            per_model_local[cid] = l

        all_features = set()
        for g in per_model_global.values():
            all_features |= set(g)

        global_importance = {
            f: sum(self._weights[cid] * per_model_global[cid].get(f, 0.0) for cid in COMPONENT_IDS)
            for f in all_features
        }
        local_importance = {
            f: sum(self._weights[cid] * per_model_local[cid].get(f, 0.0) for cid in COMPONENT_IDS)
            for f in all_features
        }
        return global_importance, local_importance
