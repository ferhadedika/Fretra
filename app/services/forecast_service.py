"""
app/services/forecast_service.py
===================================
Business logic that sits between the routers and (a) the model registry,
(b) Firestore. Routers stay thin; this is where forecast requests turn
into model calls + persisted records.
"""

import random
from datetime import date, timedelta

from app.config import SCFI_MIN, SCFI_MAX
from app.services.model_registry import registry
from app.services.firestore_service import firestore_service
from app.services.supabase_service import supabase_service
from app.schemas.forecast import (
    ForecastResponse, ForecastPoint, ModelMeta, HistoricalPoint, ShapResponse, ShapFeature,
)

# Human-readable labels for the mock feature set — mirrors the dashboard's
# Feature Dictionary so the same names resolve to the same labels here.
_FEATURE_LABELS = {
    "scfi_lag_1": "SCFI (1 Week Ago)",
    "scfi_lag_2": "SCFI (2 Weeks Ago)",
    "brent_crude_lag1": "Brent Crude Price (Lagged)",
    "cny_usd_lag1": "CNY/USD Exchange Rate (Lagged)",
    "world_pmi": "Global Manufacturing PMI",
    "world_trade_volume": "World Trade Volume",
    "seasonal_dummy_q4": "Q4 Seasonal Effect",
}


def _synthetic_historical_series(weeks: int = 120) -> list[dict]:
    """
    Used only when Firestore has no historical_scfi data yet (e.g. before
    you've loaded your real dataset in). Deterministic seed so repeated
    calls produce a stable-looking chart instead of jumping around.
    """
    rng = random.Random(42)
    today = date.today()
    start = today - timedelta(weeks=weeks)
    value = 1500.0
    points = []
    for i in range(weeks):
        value = max(value + rng.gauss(0, 22), SCFI_MIN)
        value = min(value, SCFI_MAX)
        points.append({"date": start + timedelta(weeks=i), "scfi": round(value, 2)})
    return points


async def get_historical_series() -> list[HistoricalPoint]:
    rows = await supabase_service.get_historical_scfi()
    if not rows:
        rows = _synthetic_historical_series()
    return [HistoricalPoint(**row) for row in rows]


async def run_forecast(model_id: str, horizon_weeks: int, save_to_history: bool) -> ForecastResponse:
    model = registry.get(model_id)

    historical = await get_historical_series()
    current_scfi = historical[-1].scfi
    cutoff = historical[-1].date

    raw_values = model.predict(horizon_weeks)

    points = []
    prev = current_scfi
    for week, value in enumerate(raw_values, start=1):
        points.append(ForecastPoint(
            week=week,
            date=cutoff + timedelta(weeks=week),
            forecast=value,
            delta=round(value - prev, 2),
        ))
        prev = value

    response = ForecastResponse(
        model=ModelMeta(
            id=model.model_id,
            label=model.label,
            mae=model.metrics.get("mae"),
            rmse=model.metrics.get("rmse"),
            mape_pct=model.metrics.get("mape_pct"),
            r2=model.metrics.get("r2"),
        ),
        current_scfi=current_scfi,
        cutoff_date=cutoff,
        forecast=points,
    )

    if save_to_history:
        # Persisted shape matches ForecastHistoryRecord (flat model_id +
        # horizon_weeks), NOT the nested ForecastResponse shape returned to
        # the caller - these are deliberately different schemas for two
        # different jobs (API response vs. history-list row).
        await firestore_service.save_forecast_record({
            "model_id": response.model.id,
            "horizon_weeks": horizon_weeks,
            "current_scfi": response.current_scfi,
            "cutoff_date": response.cutoff_date,
            "forecast": [p.model_dump() for p in response.forecast],
        })

    return response


def _build_shap_response_sync(model_id: str, horizon_weeks: int) -> ShapResponse:
    """
    The actual SHAP computation. Synchronous on purpose - model.explain()
    does real CPU-bound work (sub-second for tree models, ~10-15s for
    SARIMAX's KernelExplainer) and awaits nothing, so wrapping it in
    async def would be cosmetic. shap_router.py runs THIS function in a
    threadpool directly via run_in_threadpool, which is the actual fix
    for not blocking the event loop - no nested asyncio.run() needed.
    """
    model = registry.get(model_id)
    global_imp, local_imp = model.explain(horizon_weeks)

    def _to_features(d: dict) -> list[ShapFeature]:
        return sorted(
            (
                ShapFeature(feature=k, label=_FEATURE_LABELS.get(k, k), value=v)
                for k, v in d.items()
            ),
            key=lambda f: abs(f.value),
            reverse=True,
        )

    return ShapResponse(
        model_id=model.model_id,
        global_importance=_to_features(global_imp),
        local_importance=_to_features(local_imp),
    )


async def get_all_models_meta() -> list[ModelMeta]:
    return [
        ModelMeta(
            id=m.model_id,
            label=m.label,
            mae=m.metrics.get("mae"),
            rmse=m.metrics.get("rmse"),
            mape_pct=m.metrics.get("mape_pct"),
            r2=m.metrics.get("r2"),
        )
        for m in registry.list_all()
    ]
