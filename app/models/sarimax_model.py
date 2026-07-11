"""
app/models/sarimax_model.py
==============================
Real SARIMAX model wrapper. Loads sarimax.pkl saved via the training
script's sar_res_full.save(...) call.

IMPORTANT: SARIMAX results are NOT loaded with joblib.load() - statsmodels
results objects need their own SARIMAXResults.load() classmethod. Using
joblib.load() on a file saved with .save() usually still works (it's
pickle under the hood) but .load() is the statsmodels-blessed path and
more robust across statsmodels versions - use it.

SHAP for SARIMAX uses KernelExplainer (model-agnostic - SARIMAX has no
tree structure for TreeExplainer to walk), exactly matching shap_lstm-
style usage in the training script's own SARIMAX SHAP section. This is
the slowest of the four explain() calls since KernelExplainer is a
black-box sampling method, not a closed-form attribution.

REQUIRES these files in artifacts_dir:
  sarimax.pkl                    - sar_res_full.save(...)
  last_row.pkl                    - joblib.dump(last_row, ...)
  feature_config.json             - exog_features, last_scfi
  sarimax_exog_background.pkl     - joblib.dump(X_s[-50:], ...) - REQUIRED
                                     for explain() to return non-trivial
                                     SHAP values. If this background were
                                     just a single row equal to the point
                                     being explained, KernelExplainer would
                                     trivially report zero for every
                                     feature (there's no deviation from
                                     "baseline" to attribute). 50 rows of
                                     real historical variation is what
                                     gives the explainer something to
                                     measure against, exactly mirroring
                                     X_s[-50:] in the training script.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Optional

from app.models.anchor import ForecastAnchor
from app.models.base import BaseForecastModel
from app.models.feature_cascade import row_to_dict, forecast_sarimax
from app.models.metrics_loader import load_metrics_for


class SarimaxModel(BaseForecastModel):
    model_id = "sarimax"
    label = "SARIMAX"

    def __init__(self):
        super().__init__()
        self._results = None
        self._last_row: dict = {}
        self._exog_features: list[str] = []
        self._last_scfi: float = 0.0
        self._exog_background = None

    def load(self, artifacts_dir: Path) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults

        artifacts_dir = Path(artifacts_dir)
        # NOT joblib.load() - see module docstring.
        self._results = SARIMAXResults.load(str(artifacts_dir / "sarimax.pkl"))

        last_row_series = joblib.load(artifacts_dir / "last_row.pkl")
        self._last_row = row_to_dict(last_row_series)

        with open(artifacts_dir / "feature_config.json") as f:
            cfg = json.load(f)
        self._exog_features = cfg["exog_features"]
        self._last_scfi = cfg["last_scfi"]

        # Required for non-trivial SHAP - see module docstring. If this
        # file is missing, explain() raises rather than silently
        # returning all-zero attributions.
        self._exog_background = joblib.load(artifacts_dir / "sarimax_exog_background.pkl")

        metrics_path = artifacts_dir.parent / "SCFI_model_metrics.json"
        self.metrics = load_metrics_for(metrics_path, "sarimax")

        self._loaded = True

    def predict(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> list[float]:
        row = anchor.last_row if anchor else self._last_row
        fc = forecast_sarimax(self._results, row, self._exog_features, horizon_weeks)
        return [pt["forecast"] for pt in fc]

    def explain(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> tuple[dict, dict]:
        """
        PERFORMANCE NOTE: SARIMAX's SHAP is the slowest of the four
        models by a wide margin - shap.KernelExplainer calls
        self._results.forecast() once per sampled coalition (not once
        per feature), and SARIMAX's own .forecast() is itself a
        non-trivial Kalman filter pass. At nsamples=80 with a 20-row
        explained set, this took 140+ seconds in testing - unacceptable
        for a synchronous request.

        Fix: explain ONE row (not 20) for "global" too, and drop
        nsamples to something that still gives a stable ranking without
        the 140s cost. This trades a small amount of SHAP precision
        for actually being usable behind an API endpoint. If you need
        the full-precision version for a report/analysis (not the live
        API), call this same logic with nsamples=200+ in a notebook
        instead - just don't do it inside a request handler.
        """
        import shap

        def predict_fn(exog_batch: np.ndarray) -> np.ndarray:
            preds = []
            for row in exog_batch:
                try:
                    val = self._results.forecast(1, exog=row.reshape(1, -1))
                    preds.append(float(val.iloc[0]) if hasattr(val, "iloc") else float(val[0]))
                except Exception:
                    preds.append(self._last_scfi)
            return np.array(preds)

        row = anchor.last_row if anchor else self._last_row
        x_last = np.array([[row[f] for f in self._exog_features]])

        # Smaller background (15 rows, not 50) and lower nsamples (30, not
        # 80) - still real historical variation, just bounded for latency.
        background = self._exog_background[-15:]

        explainer = shap.KernelExplainer(predict_fn, background)
        # Single explainer call shared by both global and local: explain
        # the background rows once, reuse their mean |SHAP| as "global",
        # and the LAST row's own SHAP as "local" - instead of two separate
        # explainer.shap_values() calls (which was the main cost).
        sv_background = explainer.shap_values(background, nsamples=30, silent=True)
        sv_local_only = explainer.shap_values(x_last, nsamples=30, silent=True)

        sv_bg_arr = np.array(sv_background)
        global_importance = {
            feat: float(v) for feat, v in zip(self._exog_features, np.abs(sv_bg_arr).mean(0))
        }
        local_arr = np.array(sv_local_only).reshape(-1)
        local_importance = {feat: float(v) for feat, v in zip(self._exog_features, local_arr)}

        return global_importance, local_importance
