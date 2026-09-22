"""
app/routers/rolling_forecast_router.py
=========================================
Read/trigger endpoints for the rolling-forecast pipeline
(app/forecasting/orchestrator.py). Deliberately under a distinct prefix
from the existing /api/forecast/* (Firestore-backed, on-demand) endpoints
- different persistence, different semantics, no path collision.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies.auth import get_current_user
from app.forecasting.orchestrator import run_rolling_forecast
from app.schemas.rolling_forecast import (
    AccuracyRecord,
    MODEL_LABELS,
    RollingForecastResponse,
    RollingForecastRunRecord,
    RollingModelForecast,
    RunForecastRequest,
)
from app.schemas.forecast import ModelId
from app.services.supabase_service import supabase_service

router = APIRouter(prefix="/api/rolling-forecast", tags=["rolling-forecast"])


@router.get("/latest", response_model=RollingForecastResponse)
async def get_latest(horizon_weeks: int = Query(8, ge=1, le=12)):
    """Latest rolling-forecast run, all models + ensemble, filtered to
    the requested horizon (a prefix of the stored 12-week run)."""
    run = await supabase_service.get_latest_run_summary()
    if run is None:
        raise HTTPException(status_code=404, detail="No rolling forecast run yet.")

    rows = await supabase_service.get_run_rows(run["run_id"])
    shap_by_model = await supabase_service.get_shap_for_run(run["run_id"])

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        if row["week_ahead"] > horizon_weeks:
            continue
        by_model.setdefault(row["model_name"], []).append(row)

    models = [
        RollingModelForecast(
            model_name=model_name,
            label=MODEL_LABELS.get(model_name, model_name),
            points=sorted(model_rows, key=lambda r: r["week_ahead"]),
            local_importance=sorted(
                (
                    {"feature": f, "label": f, "value": v}
                    for f, v in shap_by_model.get(model_name, {}).items()
                ),
                key=lambda f: abs(f["value"]),
                reverse=True,
            ),
        )
        for model_name, model_rows in by_model.items()
    ]

    return RollingForecastResponse(
        run_id=run["run_id"],
        forecast_created_at=run["forecast_created_at"],
        latest_actual_week=run["latest_actual_week"],
        horizon_weeks=horizon_weeks,
        models=models,
    )


@router.get("/history", response_model=list[RollingForecastRunRecord])
async def get_history(limit: int = Query(20, ge=1, le=100)):
    """Past rolling-forecast runs, most recent first."""
    return await supabase_service.list_runs(limit=limit)


@router.get("/accuracy", response_model=list[AccuracyRecord])
async def get_accuracy(
    model_name: ModelId | None = None,
    horizon_weeks: int | None = Query(None, ge=1, le=12),
):
    """Latest accuracy snapshots (MAE/RMSE/MAPE/bias), optionally filtered
    by model and/or horizon."""
    return await supabase_service.get_accuracy(model_name=model_name, horizon_weeks=horizon_weeks)


@router.post("/run")
async def force_run(request: RunForecastRequest, user=Depends(get_current_user)):
    """Force (re)generation of the rolling forecast right now, bypassing
    the scheduler's own new-data check unless `force` is false."""
    created_by = f"api:{getattr(user, 'email', 'unknown')}"
    return await run_rolling_forecast(created_by=created_by, force=request.force)
