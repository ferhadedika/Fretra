"""
app/forecasting/orchestrator.py
==================================
The rolling-forecast pipeline core: pulls the full live history from
Supabase, engineers features exactly as training did, builds a fresh
ForecastAnchor from the newest row, and generates a 12-week-ahead
forecast (+ SHAP local explanation) for every model using ONLY the
already-loaded saved artifacts (no retraining) - then persists an
immutable prediction log.

Generates ONE run at the max horizon (12 weeks), not three separate
4/8/12 runs: every model's cascade is deterministic and prefix-stable, so
the week+4 value under a 12-week cascade is identical to the week+4 value
under a 4-week cascade. Callers ask for a horizon by filtering
`week_ahead <= N` on the stored rows (see rolling_forecast_router.py).
"""

import json
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta

import pandas as pd
from starlette.concurrency import run_in_threadpool

from app.config import MODEL_ARTIFACTS_DIR, PREDICTION_VERSION
from app.forecasting import feature_engineering
from app.forecasting.accuracy import backfill_and_recompute
from app.models.anchor import ForecastAnchor
from app.services.model_registry import registry
from app.services.supabase_service import supabase_service

logger = logging.getLogger("freightai.rolling_forecast")

RUN_HORIZON_WEEKS = 12
MODEL_IDS = ["xgboost", "random_forest", "sarimax", "lstm", "ensemble"]

# Distributed lock (see supabase_service.try_acquire_lock) - stops two
# backend instances (or a scheduler tick overlapping a manual /run call)
# from executing the pipeline concurrently. TTL is generous relative to
# the ~17s a real run takes, so a crashed holder doesn't block for long.
PIPELINE_LOCK_NAME = "rolling_forecast"
PIPELINE_LOCK_TTL_SECONDS = 300
_HOLDER_ID = f"{socket.gethostname()}:{os.getpid()}"


async def needs_new_run() -> bool:
    """True if Supabase has a newer actual week than the last recorded
    rolling-forecast run (or no run has ever been recorded)."""
    latest_actual = await supabase_service.get_latest_scfi_date()
    if latest_actual is None:
        return False

    last_run = await supabase_service.get_latest_run_summary()
    if last_run is None or not last_run.get("latest_actual_week"):
        return True

    return latest_actual.isoformat() > last_run["latest_actual_week"]


def _json_safe_floats(d: dict) -> dict:
    """SHAP/prediction dicts often contain numpy scalar types - normalise
    to plain Python floats so they serialise cleanly as JSON/JSONB."""
    return {k: float(v) for k, v in d.items()}


# Confidence interval: NOT statistically rigorous (no quantile regression,
# no retraining - per "don't retrain the models"). Uses each model's own
# training-time cross-validated RMSE as a proxy for one-step residual
# spread, widened by sqrt(week_ahead) - a standard, if approximate, way to
# reflect that errors compound across a *recursive* multi-step cascade
# (each week's forecast feeds the next), so uncertainty should grow with
# horizon even though no per-step error covariance was actually modeled.
# Clearly an approximation, not a proper prediction interval - labelled as
# such everywhere it's surfaced (schema field name, frontend UI).
CONFIDENCE_LEVEL = 0.95
CONFIDENCE_Z = 1.96


def _confidence_interval(predicted_value: float, rmse: "float | None", week_ahead: int) -> tuple[float, float]:
    if not rmse:
        # No CV RMSE available for this model (e.g. still running as a
        # mock) - fall back to a 0-width "interval" rather than fabricating
        # a number from nothing.
        return predicted_value, predicted_value
    half_width = CONFIDENCE_Z * rmse * (week_ahead ** 0.5)
    return predicted_value - half_width, predicted_value + half_width


def _predict_and_explain_all(
    anchor: ForecastAnchor, run_id: str, created_at: str, latest_actual_week, created_by: str,
) -> tuple[list[dict], list[dict]]:
    """
    CPU-bound: runs all 5 models' predict()+explain() calls against the
    live anchor. Called via run_in_threadpool (see run_rolling_forecast)
    so it never blocks the event loop - SARIMAX/LSTM SHAP alone can take
    several seconds each, same reasoning as app/routers/shap_router.py.

    Every row is tagged with is_real from registry.real_status() at write
    time - if a model's artifact failed to load and it's silently running
    as MockForecastModel's random-walk fallback, that's recorded on the
    row itself rather than only being visible in in-memory registry state
    (previously exposed solely via GET /api/health, with no link to what
    got persisted). accuracy.py filters to is_real rows only when
    computing MAE/RMSE/MAPE/bias, so a mock-degraded run can never
    silently corrupt a model's real accuracy stats. Requires the
    is_real column (docs/sql/add_is_real_column.sql) - degrades to
    everything reading as real-only accuracy if the migration hasn't
    been run yet, since supabase_service already fails closed (logs and
    returns []) on a missing column rather than crashing.
    """
    history_rows: list[dict] = []
    shap_rows: list[dict] = []
    real_status = registry.real_status()

    for model_id in MODEL_IDS:
        model = registry.get(model_id)
        points = model.predict(RUN_HORIZON_WEEKS, anchor=anchor)
        _, local_importance = model.explain(RUN_HORIZON_WEEKS, anchor=anchor)
        rmse = model.metrics.get("rmse")
        is_real = real_status.get(model_id, False)

        for week_ahead, value in enumerate(points, start=1):
            target_week = latest_actual_week + timedelta(weeks=week_ahead)
            lower, upper = _confidence_interval(float(value), rmse, week_ahead)
            history_rows.append({
                "run_id": run_id,
                "forecast_created_at": created_at,
                "latest_actual_week": latest_actual_week.isoformat(),
                "forecast_horizon": RUN_HORIZON_WEEKS,
                "week_ahead": week_ahead,
                "target_week": target_week.isoformat(),
                "predicted_scfi": float(value),
                "model_name": model_id,
                "prediction_version": PREDICTION_VERSION,
                "created_by": created_by,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "confidence_level": CONFIDENCE_LEVEL,
                "is_real": is_real,
            })

        shap_rows.append({
            "run_id": run_id,
            "model_name": model_id,
            "local_importance": _json_safe_floats(local_importance),
            "is_real": is_real,
        })

    return history_rows, shap_rows


async def run_rolling_forecast(created_by: str, force: bool = False) -> dict:
    """
    Full pipeline: fetch -> engineer -> (maybe skip) -> anchor -> predict/
    explain all models -> persist -> backfill accuracy. Safe to call
    repeatedly - a no-op run when nothing's changed just returns
    {"skipped": True, ...} instead of writing duplicate rows.

    Acquires a distributed lock first so two backend instances (or an
    overlapping scheduler tick + manual /run call) can't run this
    concurrently - see supabase_service.try_acquire_lock().
    """
    if not await supabase_service.try_acquire_lock(PIPELINE_LOCK_NAME, _HOLDER_ID, PIPELINE_LOCK_TTL_SECONDS):
        logger.info("run_rolling_forecast: lock held by another instance/call - skipping.")
        return {"skipped": True, "reason": "lock_held_elsewhere"}

    try:
        return await _run_rolling_forecast_locked(created_by, force)
    finally:
        await supabase_service.release_lock(PIPELINE_LOCK_NAME, _HOLDER_ID)


async def _run_rolling_forecast_locked(created_by: str, force: bool) -> dict:
    """The actual pipeline body, only ever called while holding the lock."""
    raw_rows = await supabase_service.get_full_historical()
    if not raw_rows:
        logger.warning("run_rolling_forecast: no historical data available from Supabase - skipping.")
        return {"skipped": True, "reason": "no_historical_data"}

    engineered = feature_engineering.engineer_features(pd.DataFrame(raw_rows))
    if engineered.empty:
        logger.warning("run_rolling_forecast: feature engineering produced an empty frame - skipping.")
        return {"skipped": True, "reason": "empty_engineered_frame"}

    latest_actual_week = engineered["date"].iloc[-1].date()

    if not force and not await needs_new_run():
        return {
            "skipped": True,
            "reason": "no_new_data",
            "latest_actual_week": latest_actual_week.isoformat(),
        }

    with open(MODEL_ARTIFACTS_DIR / "feature_config.json") as f:
        cfg = json.load(f)

    anchor_inputs = feature_engineering.build_anchor_inputs(engineered, cfg["ml_features"], cfg["seq_len"])

    x_seq_tail = None
    if registry.real_status().get("lstm"):
        x_seq_tail = registry.get("lstm").transform_sequence(anchor_inputs["raw_seq_tail"])

    anchor = ForecastAnchor(
        last_row=anchor_inputs["last_row"],
        last_scfi=anchor_inputs["last_scfi"],
        last_date=anchor_inputs["last_date"],
        x_seq_tail=x_seq_tail,
    )

    run_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    history_rows, shap_rows = await run_in_threadpool(
        _predict_and_explain_all, anchor, run_id, created_at, latest_actual_week, created_by,
    )

    await supabase_service.insert_forecast_history_rows(history_rows)
    await supabase_service.insert_shap_rows(shap_rows)
    await backfill_and_recompute(engineered)

    logger.info(
        "Rolling forecast run %s complete - latest_actual_week=%s, %d rows written.",
        run_id, latest_actual_week, len(history_rows),
    )

    return {
        "skipped": False,
        "run_id": run_id,
        "latest_actual_week": latest_actual_week.isoformat(),
        "forecast_created_at": created_at,
        "models": len(MODEL_IDS),
        "rows_written": len(history_rows),
    }
