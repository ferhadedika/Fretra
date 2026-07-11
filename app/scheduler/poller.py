"""
app/scheduler/poller.py
==========================
Internal trigger for the rolling-forecast pipeline: instead of the
external scraper (backend/scraper_pipeline.py, run via cron/Task
Scheduler) having to know or care about this app, a periodic in-process
job just checks Supabase for a newer actual week than the last recorded
run and regenerates automatically. Fully decoupled from however/whenever
the scraper runs.

Uses AsyncIOScheduler (not BackgroundScheduler): the job body is a
coroutine awaiting Supabase calls, so it needs to run ON FastAPI/
uvicorn's existing event loop rather than a second thread that would
have to spin up its own asyncio.run() (which fights an already-running
loop). In-memory job store is fine for a single-process deployment - if
the process restarts mid-interval, the next poll's own needs_new_run()
comparison just catches up; nothing is lost.
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import ROLLING_FORECAST_POLL_MINUTES
from app.forecasting.orchestrator import run_rolling_forecast

logger = logging.getLogger("freightai.scheduler")

_scheduler = AsyncIOScheduler()


async def _poll_job() -> None:
    try:
        result = await run_rolling_forecast(created_by="scheduler")
        if result.get("skipped"):
            logger.info("Rolling forecast poll: skipped (%s)", result.get("reason"))
        else:
            logger.info("Rolling forecast poll: new run %s", result.get("run_id"))
    except Exception:  # noqa: BLE001 - a poll failure must never kill the scheduler
        logger.exception("Rolling forecast poll job failed")


def start() -> None:
    # IntervalTrigger defaults to firing first at (now + interval), not
    # immediately - so the very first poll happens one interval after
    # startup, same as every subsequent one.
    _scheduler.add_job(
        _poll_job,
        "interval",
        minutes=ROLLING_FORECAST_POLL_MINUTES,
        id="rolling_forecast_poll",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Rolling forecast poller started (every %d min)", ROLLING_FORECAST_POLL_MINUTES)


def shutdown() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Rolling forecast poller stopped")
