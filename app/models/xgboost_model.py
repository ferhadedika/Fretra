"""
app/models/xgboost_model.py
==============================
Real XGBoost model wrapper. Loads xgboost.pkl saved by the training
script's artifact-saving block, and uses SHAP TreeExplainer for
explanations (matching shap_tree() in the training script exactly).

REQUIRES these files in artifacts_dir:
  xgboost.pkl           - joblib.dump(xgb_final, ...)
  last_row.pkl           - joblib.dump(last_row, ...)
  feature_config.json    - ml_features, last_scfi
  X_train_full.pkl       - joblib.dump(X, ...) - SHAP background data
                            (the full training feature matrix; see the
                            updated artifact-saving patch)
"""

import json
import joblib
import numpy as np
from pathlib import Path

from app.models.base import BaseForecastModel
from app.models.feature_cascade import row_to_dict, forecast_tree_delta
from app.models.metrics_loader import load_metrics_for


class XGBoostModel(BaseForecastModel):
    model_id = "xgboost"
    label = "XGBoost"

    def __init__(self):
        super().__init__()
        self._model = None
        self._last_row: dict = {}
        self._ml_features: list[str] = []
        self._last_scfi: float = 0.0
        self._X_background = None

    def load(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        self._model = joblib.load(artifacts_dir / "xgboost.pkl")

        last_row_series = joblib.load(artifacts_dir / "last_row.pkl")
        self._last_row = row_to_dict(last_row_series)

        with open(artifacts_dir / "feature_config.json") as f:
            cfg = json.load(f)
        self._ml_features = cfg["ml_features"]
        self._last_scfi = cfg["last_scfi"]

        bg_path = artifacts_dir / "X_train_full.pkl"
        self._X_background = joblib.load(bg_path) if bg_path.exists() else None

        # Metrics aren't re-derivable from the .pkl alone - read them from
        # SCFI_model_metrics.json if present (same JSON your dashboard reads),
        # else leave blank rather than guessing.
        metrics_path = artifacts_dir.parent / "SCFI_model_metrics.json"
        self.metrics = load_metrics_for(metrics_path, "xgboost")

        self._loaded = True

    def predict(self, horizon_weeks: int) -> list[float]:
        fc = forecast_tree_delta(
            self._model, self._last_row, self._ml_features,
            self._last_scfi, horizon_weeks,
        )
        return [pt["forecast"] for pt in fc]

    def explain(self, horizon_weeks: int) -> tuple[dict, dict]:
        import shap

        x_last = np.array([[self._last_row[f] for f in self._ml_features]])

        explainer = shap.TreeExplainer(self._model)

        if self._X_background is not None:
            sv_global = explainer.shap_values(self._X_background)
            global_importance = {
                feat: float(v)
                for feat, v in zip(self._ml_features, np.abs(sv_global).mean(0))
            }
        else:
            # No background data saved - fall back to explaining just the
            # last row twice (still correct, just a smaller/noisier sample
            # for "global" importance than the full training set would give)
            sv_global = explainer.shap_values(x_last)
            global_importance = {
                feat: float(v) for feat, v in zip(self._ml_features, np.abs(sv_global).mean(0))
            }

        sv_local = explainer.shap_values(x_last)
        local_importance = {
            feat: float(v) for feat, v in zip(self._ml_features, sv_local[0])
        }

        return global_importance, local_importance
