"""
app/schemas/forecast.py
=========================
Pydantic models for every request/response shape the API exposes.
Keeping these in one place means the frontend's TypeScript types and the
backend's validation can stay in lockstep — change a field here, and
FastAPI's auto-generated OpenAPI schema (visible at /docs) updates too.
"""

from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field

from app.config import DEFAULT_HORIZON_WEEKS

ModelId = Literal["xgboost", "random_forest", "sarimax", "lstm", "ensemble"]


class HistoricalPoint(BaseModel):
    date: date
    scfi: float


class ForecastPoint(BaseModel):
    week: int = Field(..., description="Weeks ahead of the cutoff date, 1-indexed")
    date: date
    forecast: float
    delta: float = Field(..., description="Change vs. the previous week's value")


class ModelMeta(BaseModel):
    id: ModelId
    label: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape_pct: Optional[float] = None
    r2: Optional[float] = None


class ForecastRequest(BaseModel):
    model_id: ModelId = "ensemble"
    horizon_weeks: int = Field(DEFAULT_HORIZON_WEEKS, ge=1, le=12)
    save_to_history: bool = Field(
        True, description="If true, persist this forecast run to Firestore"
    )


class ForecastResponse(BaseModel):
    model: ModelMeta
    current_scfi: float
    cutoff_date: date
    forecast: list[ForecastPoint]


class ShapFeature(BaseModel):
    feature: str
    label: str
    value: float


class ShapResponse(BaseModel):
    model_id: ModelId
    global_importance: list[ShapFeature]
    local_importance: list[ShapFeature]


class ForecastHistoryRecord(BaseModel):
    id: Optional[str] = None
    model_id: ModelId
    horizon_weeks: int
    current_scfi: float
    cutoff_date: date
    forecast: list[ForecastPoint]
    created_at: Optional[str] = None
