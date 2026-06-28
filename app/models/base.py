"""
app/models/base.py
====================
The contract every model wrapper implements. This is the seam between
"mock predictor that runs today" and "your real trained model" — swap the
implementation of load()/predict()/explain() in a subclass and nothing
else in the app changes.

WHY THIS SHAPE:
  - load() is separate from __init__ so the heavy artifact-loading (reading
    a .pkl/.h5 file into memory) happens once at startup, not per-request.
  - predict() always returns plain Python floats in ORIGINAL SCFI units
    (not normalised [0,1]) — the de-normalisation is each model's own
    responsibility since different artifacts may store different scales.
  - explain() returns (global_importance, local_importance) as plain
    dicts of {feature_name: value} — SHAP-library-specific objects never
    leak past this boundary, so routers/services don't need to know
    whether a model uses TreeExplainer, KernelExplainer, etc.
"""

from abc import ABC, abstractmethod


class BaseForecastModel(ABC):
    """Abstract interface for a single forecasting model."""

    model_id: str = "base"
    label: str = "Base Model"

    def __init__(self):
        self._loaded = False
        self.metrics: dict = {}  # MAE / RMSE / MAPE(%) / R² once load() runs

    @abstractmethod
    def load(self, artifacts_dir) -> None:
        """
        Load the trained artifact(s) from disk (or initialise a mock).
        Must set self._loaded = True and populate self.metrics on success.
        Called once at app startup via ModelRegistry.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, horizon_weeks: int) -> list[float]:
        """
        Return `horizon_weeks` forecasted SCFI values (original units,
        e.g. ~800-5000), ordered from week+1 to week+horizon_weeks.
        """
        raise NotImplementedError

    @abstractmethod
    def explain(self, horizon_weeks: int) -> tuple[dict[str, float], dict[str, float]]:
        """
        Return (global_importance, local_importance) as
        {feature_name: shap_value} dicts. Global = mean |SHAP| across
        training data; local = SHAP values behind the most recent
        prediction specifically.
        """
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return self._loaded
