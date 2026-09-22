"""
app/schemas/rolling_forecast.py
==================================
Pydantic models for the rolling-forecast pipeline's endpoints
(app/routers/rolling_forecast_router.py). Kept separate from
schemas/forecast.py since these describe a different persisted shape
(Supabase forecast_history/forecast_accuracy/forecast_shap rows) than the
existing on-demand ForecastResponse/ShapResponse.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.forecast import ModelId, ShapFeature

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "sarimax": "SARIMAX",
    "lstm": "LSTM",
    "ensemble": "Ensemble",
}


class RollingForecastPoint(BaseModel):
    target_week: date
    week_ahead: int
    predicted_scfi: float
    actual_scfi: Optional[float] = None
    forecast_error: Optional[float] = None
    # Approximate, NOT a statistically rigorous prediction interval - see
    # orchestrator.py's _confidence_interval() docstring. Widens with
    # week_ahead; None if the model had no CV RMSE to derive it from.
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_level: Optional[float] = None
    # False if this point came from MockForecastModel's random-walk
    # fallback rather than the real trained model (its artifact failed to
    # load - see model_registry.py). None if the is_real column doesn't
    # exist yet in Supabase (migration not run - docs/sql/add_is_real_column.sql).
    is_real: Optional[bool] = None


class RollingModelForecast(BaseModel):
    model_name: ModelId
    label: str
    points: list[RollingForecastPoint]
    local_importance: list[ShapFeature] = []


class RollingForecastResponse(BaseModel):
    run_id: str
    forecast_created_at: datetime
    latest_actual_week: date
    horizon_weeks: int
    models: list[RollingModelForecast]


class RollingForecastRunRecord(BaseModel):
    run_id: str
    forecast_created_at: datetime
    latest_actual_week: date
    model_count: int


class AccuracyRecord(BaseModel):
    model_name: str
    forecast_horizon: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    bias: Optional[float] = None
    n_samples: int
    computed_at: datetime


class RunForecastRequest(BaseModel):
    force: bool = False
