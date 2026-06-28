"""
app/routers/forecast_router.py
=================================
Core forecasting endpoints: generate a new forecast (optionally persisting
it to Firestore) and read back historical SCFI / past forecast runs.

All handlers are async def — FastAPI runs them on the event loop without
blocking a worker thread, which is what "asynchronous endpoints for
low-latency responses" means in practice. The model.predict() call itself
is currently synchronous CPU-bound work (true for XGBoost/RF too — they
don't have async APIs), so for real heavy models you'd wrap that call in
`await run_in_threadpool(model.predict, horizon_weeks)` (from
starlette.concurrency) to avoid blocking the event loop. Noted inline
below where that change would go.
"""

from fastapi import APIRouter, HTTPException, Query

from app.schemas.forecast import (
    ForecastRequest, ForecastResponse, HistoricalPoint, ForecastHistoryRecord,
)
from app.services import forecast_service
from app.services.firestore_service import firestore_service

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("/historical", response_model=list[HistoricalPoint])
async def get_historical():
    """Historical SCFI series (from Firestore if configured, else a
    deterministic synthetic series so the chart always renders)."""
    return await forecast_service.get_historical_series()


@router.post("", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """
    Generate a forecast for the requested model + horizon. If
    save_to_history is true (default), the result is persisted via
    FirestoreService (or kept in memory if Firestore isn't configured
    yet) and shows up in GET /api/forecast/history.

    NOTE for real models: if predict()/explain() become slow (real
    XGBoost/SARIMAX/LSTM inference), swap:
        raw_values = model.predict(horizon_weeks)
    for:
        from starlette.concurrency import run_in_threadpool
        raw_values = await run_in_threadpool(model.predict, horizon_weeks)
    inside forecast_service.run_forecast() to keep the event loop free.
    """
    try:
        return await forecast_service.run_forecast(
            model_id=request.model_id,
            horizon_weeks=request.horizon_weeks,
            save_to_history=request.save_to_history,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/history", response_model=list[ForecastHistoryRecord])
async def get_forecast_history(limit: int = Query(50, ge=1, le=200)):
    """Past forecast runs a user has generated (most recent first)."""
    records = await firestore_service.get_forecast_history(limit=limit)
    return records
