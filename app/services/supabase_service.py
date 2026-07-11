"""
app/services/supabase_service.py
===================================
Reads the real historical SCFI series from Supabase's `scfi_weekly` table
(populated by backend/supabase_upload.py). This is a separate store from
Firestore (which only holds forecast run history), so it gets its own
thin client instead of being bolted onto firestore_service.py.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from app.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_ENABLED

logger = logging.getLogger("freightai.supabase")

TABLE_NAME = "scfi_weekly"

# ── Rolling-forecast tables (see backend's SQL DDL) ────────────────────────
FORECAST_HISTORY_TABLE = "forecast_history"
FORECAST_ACCURACY_TABLE = "forecast_accuracy"
FORECAST_SHAP_TABLE = "forecast_shap"
FORECAST_RUNS_VIEW = "forecast_runs"
PIPELINE_LOCKS_TABLE = "pipeline_locks"

RAW_HISTORY_COLUMNS = (
    "date, scfi, brent_crude, cny_usd, pmi_manufacturing, pmi_services, "
    "trade_volume_index, trade_volume_yoy"
)


class SupabaseService:
    """
    Real-mode: backed by the Supabase `scfi_weekly` table.
    Fallback-mode: returns an empty list when no credentials are
    configured, matching firestore_service's fallback contract so
    forecast_service's synthetic-series fallback still works unchanged.
    """

    def __init__(self):
        self._enabled = SUPABASE_ENABLED
        self._client = None

        if self._enabled:
            try:
                from supabase import create_client

                self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("Supabase connected")
            except Exception as exc:  # noqa: BLE001 - log and degrade gracefully
                logger.warning(
                    "Supabase credentials configured but connection failed (%s). "
                    "Falling back to synthetic historical series.", exc,
                )
                self._enabled = False

    @property
    def mode(self) -> str:
        return "supabase" if self._enabled else "disabled (no Supabase credentials configured)"

    async def get_historical_scfi(self) -> list[dict]:
        """Returns [{date, scfi}, ...] ordered by date ascending."""
        if not self._enabled:
            return []

        res = (
            self._client.table(TABLE_NAME)
            .select("date, scfi")
            .order("date")
            .execute()
        )
        return res.data or []

    async def get_full_historical(self) -> list[dict]:
        """
        Full raw columns (date, scfi, brent_crude, cny_usd,
        pmi_manufacturing, pmi_services, trade_volume_index,
        trade_volume_yoy) ordered by date ascending - what the rolling
        forecast pipeline's feature engineering needs. Distinct from
        get_historical_scfi() (date+scfi only, used by the existing
        dashboard chart) so that contract never changes.
        """
        if not self._enabled:
            return []
        try:
            res = (
                self._client.table(TABLE_NAME)
                .select(RAW_HISTORY_COLUMNS)
                .order("date")
                .execute()
            )
            return res.data or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_full_historical failed (%s)", exc)
            return []

    async def get_latest_scfi_date(self) -> Optional[date]:
        if not self._enabled:
            return None
        try:
            res = (
                self._client.table(TABLE_NAME)
                .select("date")
                .order("date", desc=True)
                .limit(1)
                .execute()
            )
            if not res.data:
                return None
            return date.fromisoformat(res.data[0]["date"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_latest_scfi_date failed (%s)", exc)
            return None

    # ── Rolling forecast history ────────────────────────────────────────────

    async def insert_forecast_history_rows(self, rows: list[dict]) -> None:
        if not self._enabled or not rows:
            return
        try:
            self._client.table(FORECAST_HISTORY_TABLE).insert(rows).execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "insert_forecast_history_rows failed (%s). Run the "
                "forecast_history SQL DDL in the Supabase SQL editor if "
                "the table doesn't exist yet.", exc,
            )

    async def get_latest_run_summary(self) -> Optional[dict]:
        if not self._enabled:
            return None
        try:
            res = (
                self._client.table(FORECAST_RUNS_VIEW)
                .select("*")
                .order("forecast_created_at", desc=True)
                .limit(1)
                .execute()
            )
            return res.data[0] if res.data else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_latest_run_summary failed (%s)", exc)
            return None

    async def list_runs(self, limit: int = 20) -> list[dict]:
        if not self._enabled:
            return []
        try:
            res = (
                self._client.table(FORECAST_RUNS_VIEW)
                .select("*")
                .order("forecast_created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return res.data or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_runs failed (%s)", exc)
            return []

    async def get_run_rows(self, run_id: str) -> list[dict]:
        if not self._enabled:
            return []
        try:
            res = (
                self._client.table(FORECAST_HISTORY_TABLE)
                .select("*")
                .eq("run_id", run_id)
                .order("model_name")
                .order("week_ahead")
                .execute()
            )
            return res.data or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_run_rows failed (%s)", exc)
            return []

    async def get_unbackfilled_predictions(self, as_of: date) -> list[dict]:
        """Predictions whose target week has already happened but whose
        actual value hasn't been backfilled yet."""
        if not self._enabled:
            return []
        try:
            res = (
                self._client.table(FORECAST_HISTORY_TABLE)
                .select("*")
                .lte("target_week", as_of.isoformat())
                .is_("actual_scfi", "null")
                .execute()
            )
            return res.data or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_unbackfilled_predictions failed (%s)", exc)
            return []

    async def update_actuals(self, updates: list[dict]) -> int:
        """Each update dict: {id, actual_scfi, forecast_error,
        abs_pct_error, actual_backfilled_at}. Appends onto existing rows -
        never touches predicted_scfi/model_name/target_week, keeping the
        prediction itself immutable."""
        if not self._enabled or not updates:
            return 0
        applied = 0
        for u in updates:
            row_id = u["id"]
            payload = {k: v for k, v in u.items() if k != "id"}
            try:
                self._client.table(FORECAST_HISTORY_TABLE).update(payload).eq("id", row_id).execute()
                applied += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("update_actuals failed for row %s (%s)", row_id, exc)
        return applied

    async def get_backfilled_predictions(self) -> list[dict]:
        """All forecast_history rows that already have an actual value
        recorded - the raw material for accuracy snapshot computation."""
        if not self._enabled:
            return []
        try:
            res = (
                self._client.table(FORECAST_HISTORY_TABLE)
                .select("model_name, week_ahead, predicted_scfi, actual_scfi")
                .not_.is_("actual_scfi", "null")
                .execute()
            )
            return res.data or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_backfilled_predictions failed (%s)", exc)
            return []

    async def upsert_accuracy_snapshot(self, rows: list[dict]) -> None:
        if not self._enabled or not rows:
            return
        try:
            self._client.table(FORECAST_ACCURACY_TABLE).insert(rows).execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("upsert_accuracy_snapshot failed (%s)", exc)

    async def get_accuracy(
        self, model_name: Optional[str] = None, horizon_weeks: Optional[int] = None
    ) -> list[dict]:
        if not self._enabled:
            return []
        try:
            query = self._client.table(FORECAST_ACCURACY_TABLE).select("*")
            if model_name:
                query = query.eq("model_name", model_name)
            if horizon_weeks:
                query = query.eq("forecast_horizon", horizon_weeks)
            res = query.order("computed_at", desc=True).execute()
            return res.data or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_accuracy failed (%s)", exc)
            return []

    # ── SHAP per forecast run ────────────────────────────────────────────────

    async def insert_shap_rows(self, rows: list[dict]) -> None:
        if not self._enabled or not rows:
            return
        try:
            self._client.table(FORECAST_SHAP_TABLE).insert(rows).execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("insert_shap_rows failed (%s)", exc)

    async def get_shap_for_run(self, run_id: str) -> dict[str, dict]:
        if not self._enabled:
            return {}
        try:
            res = (
                self._client.table(FORECAST_SHAP_TABLE)
                .select("model_name, local_importance")
                .eq("run_id", run_id)
                .execute()
            )
            return {row["model_name"]: row["local_importance"] for row in (res.data or [])}
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_shap_for_run failed (%s)", exc)
            return {}

    # ── Distributed lock (so only one backend instance runs the pipeline) ───

    async def try_acquire_lock(self, lock_name: str, holder_id: str, ttl_seconds: int) -> bool:
        """
        Two-phase INSERT-else-conditional-UPDATE, both single atomic
        Postgres statements, so this is race-safe across processes without
        needing a custom SQL function:
          1. INSERT succeeds only if no row exists yet for this lock_name
             (primary key) - exactly one concurrent caller can win this.
          2. If a row already exists, UPDATE ... WHERE expires_at < now()
             only succeeds if the existing lock has actually expired;
             Postgres's row-level locking on UPDATE means two concurrent
             callers racing this same statement can't both "see" the lock
             as expired and both win - the second to run re-evaluates the
             WHERE clause against the first's already-committed row and
             correctly finds 0 rows to update.
        If Supabase isn't configured, fails open (returns True) rather than
        blocking the pipeline in local/dev setups with no lock table.
        """
        if not self._enabled:
            return True
        now = datetime.utcnow()
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        try:
            self._client.table(PIPELINE_LOCKS_TABLE).insert({
                "lock_name": lock_name, "locked_by": holder_id,
                "locked_at": now.isoformat(), "expires_at": expires_at,
            }).execute()
            return True
        except Exception:  # noqa: BLE001 - row already exists, fall through
            pass

        try:
            res = (
                self._client.table(PIPELINE_LOCKS_TABLE)
                .update({"locked_by": holder_id, "locked_at": now.isoformat(), "expires_at": expires_at})
                .eq("lock_name", lock_name)
                .lt("expires_at", now.isoformat())
                .execute()
            )
            return bool(res.data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("try_acquire_lock failed (%s) - failing closed (lock not acquired)", exc)
            return False

    async def release_lock(self, lock_name: str, holder_id: str) -> None:
        """Best-effort early release so the next scheduled tick doesn't have
        to wait out the full TTL. Only releases if we're still the holder
        (in case we already lost the lock to a TTL-based takeover)."""
        if not self._enabled:
            return
        try:
            self._client.table(PIPELINE_LOCKS_TABLE).delete().eq("lock_name", lock_name).eq(
                "locked_by", holder_id
            ).execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("release_lock failed (%s)", exc)


# Module-level singleton, mirrors services/model_registry.py's pattern.
supabase_service = SupabaseService()
