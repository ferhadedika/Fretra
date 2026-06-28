"""
app/models/ensemble_model.py
===============================
Real Ensemble model wrapper. Unlike the other four, this one doesn't
load its own artifact file - it composes an already-loaded XGBoostModel
and RandomForestModel instance using the scipy-optimised blend weight
saved in ensemble_config.json (w_star in the training script), exactly
matching:

    ens_preds = w_star * xgb_arr + (1.0 - w_star) * rf_arr

REQUIRES:
  ensemble_config.json   - {"xgb_weight": ..., "rf_weight": ...}
  Already-loaded XGBoostModel and RandomForestModel instances, passed in
  via set_components() after the registry loads all models (see
  app/services/model_registry.py for the wiring).

SHAP for the ensemble is the weighted blend of its two components' SHAP
values - this isn't in the training script (which only computes SHAP for
the four base models), but it's the mathematically consistent extension:
since the ensemble's prediction IS w*xgb + (1-w)*rf, its attribution is
w*xgb_shap + (1-w)*rf_shap for any feature shared by both.
"""

import json
from pathlib import Path

from app.models.base import BaseForecastModel
from app.models.metrics_loader import load_metrics_for


class EnsembleModel(BaseForecastModel):
    model_id = "ensemble"
    label = "Ensemble"

    def __init__(self):
        super().__init__()
        self._xgb_weight = 0.5
        self._rf_weight = 0.5
        self._xgb_model = None
        self._rf_model = None

    def set_components(self, xgb_model: BaseForecastModel, rf_model: BaseForecastModel) -> None:
        """Called by ModelRegistry after both component models are loaded."""
        self._xgb_model = xgb_model
        self._rf_model = rf_model

    def load(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        cfg_path = artifacts_dir / "ensemble_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self._xgb_weight = cfg.get("xgb_weight", 0.5)
            self._rf_weight = cfg.get("rf_weight", 1 - self._xgb_weight)

        metrics_path = artifacts_dir.parent / "SCFI_model_metrics.json"
        self.metrics = load_metrics_for(metrics_path, "ensemble")

        # Marked loaded here; set_components() must still be called before
        # predict()/explain() work - ModelRegistry guarantees the order.
        self._loaded = True

    def predict(self, horizon_weeks: int) -> list[float]:
        if self._xgb_model is None or self._rf_model is None:
            raise RuntimeError(
                "EnsembleModel.set_components() must be called before predict()."
            )
        xgb_vals = self._xgb_model.predict(horizon_weeks)
        rf_vals = self._rf_model.predict(horizon_weeks)
        return [
            round(self._xgb_weight * x + self._rf_weight * r, 2)
            for x, r in zip(xgb_vals, rf_vals)
        ]

    def explain(self, horizon_weeks: int) -> tuple[dict, dict]:
        if self._xgb_model is None or self._rf_model is None:
            raise RuntimeError(
                "EnsembleModel.set_components() must be called before explain()."
            )
        xgb_global, xgb_local = self._xgb_model.explain(horizon_weeks)
        rf_global, rf_local = self._rf_model.explain(horizon_weeks)

        all_features = set(xgb_global) | set(rf_global)
        global_importance = {
            f: self._xgb_weight * xgb_global.get(f, 0.0) + self._rf_weight * rf_global.get(f, 0.0)
            for f in all_features
        }
        local_importance = {
            f: self._xgb_weight * xgb_local.get(f, 0.0) + self._rf_weight * rf_local.get(f, 0.0)
            for f in all_features
        }
        return global_importance, local_importance
