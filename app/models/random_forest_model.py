"""
app/models/random_forest_model.py
====================================
Real Random Forest / ExtraTrees model wrapper. Loads random_forest.pkl
saved by the training script's artifact-saving block. Your training
script picks whichever of RandomForestRegressor or ExtraTreesRegressor
scored best (best_rf_type, saved in ensemble_config.json's "rf_variant")
- this class doesn't need to know which one it got, since both share the
same sklearn .predict()/.feature_importances_ interface and both work
fine with shap.TreeExplainer.

REQUIRES these files in artifacts_dir:
  random_forest.pkl     - joblib.dump(rf_final, ...)
  last_row.pkl           - joblib.dump(last_row, ...)
  feature_config.json    - ml_features, last_scfi
  ensemble_config.json    - rf_variant (used only for the model's label)
  X_train_full.pkl       - SHAP background data
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Optional

from app.models.anchor import ForecastAnchor
from app.models.base import BaseForecastModel
from app.models.feature_cascade import row_to_dict, forecast_tree_delta
from app.models.metrics_loader import load_metrics_for


class RandomForestModel(BaseForecastModel):
    model_id = "random_forest"
    label = "Random Forest"  # overwritten in load() with the actual variant (RF/ET)

    def __init__(self):
        super().__init__()
        self._model = None
        self._last_row: dict = {}
        self._ml_features: list[str] = []
        self._last_scfi: float = 0.0
        self._X_background = None

    def load(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        self._model = joblib.load(artifacts_dir / "random_forest.pkl")

        last_row_series = joblib.load(artifacts_dir / "last_row.pkl")
        self._last_row = row_to_dict(last_row_series)

        with open(artifacts_dir / "feature_config.json") as f:
            cfg = json.load(f)
        self._ml_features = cfg["ml_features"]
        self._last_scfi = cfg["last_scfi"]

        ensemble_cfg_path = artifacts_dir / "ensemble_config.json"
        if ensemble_cfg_path.exists():
            with open(ensemble_cfg_path) as f:
                ens_cfg = json.load(f)
            variant = ens_cfg.get("rf_variant", "RF")
            self.label = "Extra Trees" if variant == "ET" else "Random Forest"

        bg_path = artifacts_dir / "X_train_full.pkl"
        self._X_background = joblib.load(bg_path) if bg_path.exists() else None

        metrics_path = artifacts_dir.parent / "SCFI_model_metrics.json"
        self.metrics = load_metrics_for(metrics_path, "random_forest")

        self._loaded = True

    def predict(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> list[float]:
        row = anchor.last_row if anchor else self._last_row
        last_scfi = anchor.last_scfi if anchor else self._last_scfi
        fc = forecast_tree_delta(
            self._model, row, self._ml_features,
            last_scfi, horizon_weeks,
        )
        return [pt["forecast"] for pt in fc]

    def explain(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> tuple[dict, dict]:
        import shap

        row = anchor.last_row if anchor else self._last_row
        x_last = np.array([[row[f] for f in self._ml_features]])
        explainer = shap.TreeExplainer(self._model)

        bg = self._X_background if self._X_background is not None else x_last
        sv_global = explainer.shap_values(bg)
        global_importance = {
            feat: float(v) for feat, v in zip(self._ml_features, np.abs(sv_global).mean(0))
        }

        sv_local = explainer.shap_values(x_last)
        local_importance = {
            feat: float(v) for feat, v in zip(self._ml_features, sv_local[0])
        }

        return global_importance, local_importance
