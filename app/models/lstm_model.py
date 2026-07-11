"""
app/models/lstm_model.py
===========================
Real LSTM model wrapper. Loads lstm.keras + both scalers saved by the
training script's artifact-saving block.

REQUIRES these files in artifacts_dir:
  lstm.keras                  - lstm_final.save(...)
  lstm_feature_scaler.pkl       - joblib.dump(feat_sc, ...)
  lstm_target_scaler.pkl        - joblib.dump(tgt_sc, ...)
  lstm_X_seq_background.pkl     - joblib.dump(X_seq[-30:], ...) - the last
                                   30 scaled sequences, used both as the
                                   forecast starting point AND as SHAP
                                   KernelExplainer background (matches
                                   shap_lstm()'s X_bg argument in the
                                   training script)
  feature_config.json           - ml_features, last_scfi

SHAP for LSTM uses KernelExplainer the same way shap_lstm() does in the
training script: it explains the model as a black box over the LAST
timestep's features only (flattening the sequence dimension), since
KernelExplainer can't reason about a full 3D sequence directly.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Optional

from app.models.anchor import ForecastAnchor
from app.models.base import BaseForecastModel
from app.models.feature_cascade import forecast_lstm_delta
from app.models.metrics_loader import load_metrics_for


class LstmModel(BaseForecastModel):
    model_id = "lstm"
    label = "LSTM"

    def __init__(self):
        super().__init__()
        self._model = None
        self._feat_scaler = None
        self._target_scaler = None
        self._x_seq_background = None  # shape (N, SEQ_LEN, n_features), scaled
        self._ml_features: list[str] = []
        self._last_scfi: float = 0.0

    def load(self, artifacts_dir: Path) -> None:
        from tensorflow import keras

        artifacts_dir = Path(artifacts_dir)
        self._model = keras.models.load_model(artifacts_dir / "lstm.keras")
        self._feat_scaler = joblib.load(artifacts_dir / "lstm_feature_scaler.pkl")
        self._target_scaler = joblib.load(artifacts_dir / "lstm_target_scaler.pkl")
        self._x_seq_background = joblib.load(artifacts_dir / "lstm_X_seq_background.pkl")

        with open(artifacts_dir / "feature_config.json") as f:
            cfg = json.load(f)
        self._ml_features = cfg["ml_features"]
        self._last_scfi = cfg["last_scfi"]

        metrics_path = artifacts_dir.parent / "SCFI_model_metrics.json"
        self.metrics = load_metrics_for(metrics_path, "lstm")

        self._loaded = True

    def predict(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> list[float]:
        # self._x_seq_background has shape (N, SEQ_LEN, n_features) - N
        # separate overlapping sequences. The correct forecast starting
        # point is the LAST single sequence (the most recent SEQ_LEN
        # timesteps), shape (SEQ_LEN, n_features) - NOT the last 12
        # sequences, which would stay 3D and break the LSTM's expected
        # input shape.
        if anchor is not None and anchor.x_seq_tail is not None:
            x_seq_tail = anchor.x_seq_tail
            last_scfi = anchor.last_scfi
        else:
            x_seq_tail = self._x_seq_background[-1]
            last_scfi = self._last_scfi
        fc = forecast_lstm_delta(
            self._model, self._feat_scaler, self._target_scaler, x_seq_tail,
            self._ml_features, last_scfi, horizon_weeks,
        )
        return [pt["forecast"] for pt in fc]

    def transform_sequence(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Scale a live (seq_len, n_features) raw-units matrix with this
        model's own frozen feature scaler, so callers building a fresh
        anchor never need to reach into self._feat_scaler directly."""
        return self._feat_scaler.transform(feature_matrix)

    def explain(self, horizon_weeks: int, anchor: Optional[ForecastAnchor] = None) -> tuple[dict, dict]:
        import shap

        seq_len = self._x_seq_background.shape[1]

        def flat_pred(flat_X: np.ndarray) -> np.ndarray:
            tiled = np.tile(flat_X[:, None, :], (1, seq_len, 1))
            return self._model.predict(tiled, verbose=0).ravel()

        x_bg = self._x_seq_background[:, -1, :]
        if anchor is not None and anchor.x_seq_tail is not None:
            # Same "explain the last few timesteps" shape as the frozen
            # path, but the LAST row (whose SHAP becomes local_importance
            # below) is the live anchor's most recent timestep.
            x_ex = np.vstack([self._x_seq_background[-4:, -1, :], anchor.x_seq_tail[-1:, :]])
        else:
            x_ex = self._x_seq_background[-5:, -1, :]  # explain last few "last timesteps"

        explainer = shap.KernelExplainer(flat_pred, x_bg)
        sv = explainer.shap_values(x_ex, nsamples=100, silent=True)

        sv_arr = np.array(sv)
        global_importance = {
            feat: float(v) for feat, v in zip(self._ml_features, np.abs(sv_arr).mean(0))
        }
        local_importance = {
            feat: float(v) for feat, v in zip(self._ml_features, sv_arr[-1])
        }

        return global_importance, local_importance
