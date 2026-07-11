"""
app/forecasting/accuracy.py
==============================
Once the actual SCFI for a previously-predicted target_week becomes
available (i.e. the scraper has now pulled that week's real value into
Supabase), backfill it onto the matching forecast_history row and
recompute aggregate accuracy snapshots.

Design: the ORIGINAL prediction fields (predicted_scfi, model_name,
target_week, ...) are never touched - only actual_scfi/forecast_error/
abs_pct_error/actual_backfilled_at get appended once known, so
forecast_history stays an immutable log of "what was predicted, when."
Aggregate MAE/RMSE/MAPE/bias live in a separate forecast_accuracy table
as periodic snapshots, rather than being recomputed on every read.
"""

import logging
import math
from datetime import date as date_cls, datetime

import pandas as pd

from app.services.supabase_service import supabase_service

logger = logging.getLogger("freightai.accuracy")

HORIZON_BUCKETS = (4, 8, 12)


async def backfill_and_recompute(engineered_df: pd.DataFrame) -> None:
    """`engineered_df` is the freshly engineered historical DataFrame from
    the current rolling-forecast run - its `date`/`SCFI` columns are the
    source of truth for "what actually happened" up through the latest
    actual week."""
    as_of = engineered_df["date"].iloc[-1].date()
    pending = await supabase_service.get_unbackfilled_predictions(as_of)

    if pending:
        actual_by_week = {
            ts.date(): float(scfi)
            for ts, scfi in zip(engineered_df["date"], engineered_df["SCFI"])
        }
        now_iso = datetime.utcnow().isoformat()
        updates = []
        for row in pending:
            target_week = date_cls.fromisoformat(row["target_week"])
            actual = actual_by_week.get(target_week)
            if actual is None:
                continue  # that week's actual isn't in this history slice yet
            predicted = float(row["predicted_scfi"])
            error = actual - predicted
            abs_pct_error = abs(error) / actual * 100 if actual else None
            updates.append({
                "id": row["id"],
                "actual_scfi": actual,
                "forecast_error": round(error, 4),
                "abs_pct_error": round(abs_pct_error, 4) if abs_pct_error is not None else None,
                "actual_backfilled_at": now_iso,
            })
        if updates:
            n = await supabase_service.update_actuals(updates)
            logger.info("Backfilled actuals onto %d forecast_history rows", n)

    await recompute_accuracy_snapshot()


async def recompute_accuracy_snapshot() -> None:
    """For each model x horizon bucket (4/8/12 weeks ahead), compute
    MAE/RMSE/MAPE/bias over every backfilled prediction and store one
    new snapshot row per (model, horizon)."""
    backfilled = await supabase_service.get_backfilled_predictions()
    if not backfilled:
        return

    df = pd.DataFrame(backfilled)
    df["error"] = df["actual_scfi"] - df["predicted_scfi"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["abs_error"] / df["actual_scfi"].abs().replace(0, float("nan")) * 100

    now_iso = datetime.utcnow().isoformat()
    snapshots = []
    for model_name, model_df in df.groupby("model_name"):
        for horizon in HORIZON_BUCKETS:
            bucket = model_df[model_df["week_ahead"] <= horizon]
            if bucket.empty:
                continue
            mae = float(bucket["abs_error"].mean())
            rmse = float(math.sqrt((bucket["error"] ** 2).mean()))
            mape = float(bucket["pct_error"].mean()) if bucket["pct_error"].notna().any() else None
            bias = float(bucket["error"].mean())
            snapshots.append({
                "model_name": model_name,
                "forecast_horizon": horizon,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4) if mape is not None else None,
                "bias": round(bias, 4),
                "n_samples": int(len(bucket)),
                "computed_at": now_iso,
            })

    if snapshots:
        await supabase_service.upsert_accuracy_snapshot(snapshots)
