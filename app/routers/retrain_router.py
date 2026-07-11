"""
app/routers/retrain_router.py
================================
Manual trigger + status polling for Stage 2 retraining
(app/forecasting/retrain.py). The actual training run can take 30min-2hr,
so POST /run fires it as a background asyncio task and returns
immediately rather than holding the HTTP connection open - status is
polled separately via GET /status.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends

from app.dependencies.auth import get_current_user
from app.forecasting.retrain import read_status, run_retraining_and_maybe_swap

logger = logging.getLogger("freightai.retrain_router")

router = APIRouter(prefix="/api/models/retrain", tags=["retrain"])


@router.post("")
async def trigger_retrain(user=Depends(get_current_user)):
    """Starts a retraining run in the background (does not wait for it to
    finish - poll GET /api/models/retrain/status for progress/result)."""
    triggered_by = f"api:{getattr(user, 'email', 'unknown')}"

    async def _run():
        try:
            await run_retraining_and_maybe_swap(triggered_by=triggered_by)
        except Exception:  # noqa: BLE001 - a failed background task must not vanish silently
            logger.exception("Background retraining task raised unexpectedly")

    asyncio.create_task(_run())
    return {"status": "started", "triggered_by": triggered_by}


@router.get("/status")
async def get_retrain_status():
    return read_status()
