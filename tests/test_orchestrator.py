"""
needs_new_run() is what makes the 30-minute poller (and repeated manual
POST /run calls) cheap and idempotent when nothing's changed - these
tests cover its three decision branches directly, mocking
supabase_service so no real network call happens.
"""
from datetime import date
from unittest.mock import AsyncMock, patch

from app.forecasting.orchestrator import needs_new_run


async def test_no_new_run_when_no_historical_data_at_all():
    with patch("app.forecasting.orchestrator.supabase_service") as mock_svc:
        mock_svc.get_latest_scfi_date = AsyncMock(return_value=None)
        assert await needs_new_run() is False


async def test_needs_run_when_no_prior_run_exists():
    with patch("app.forecasting.orchestrator.supabase_service") as mock_svc:
        mock_svc.get_latest_scfi_date = AsyncMock(return_value=date(2026, 7, 10))
        mock_svc.get_latest_run_summary = AsyncMock(return_value=None)
        assert await needs_new_run() is True


async def test_needs_run_when_newer_data_exists():
    with patch("app.forecasting.orchestrator.supabase_service") as mock_svc:
        mock_svc.get_latest_scfi_date = AsyncMock(return_value=date(2026, 7, 10))
        mock_svc.get_latest_run_summary = AsyncMock(return_value={"latest_actual_week": "2026-07-03"})
        assert await needs_new_run() is True


async def test_no_new_run_when_data_unchanged():
    with patch("app.forecasting.orchestrator.supabase_service") as mock_svc:
        mock_svc.get_latest_scfi_date = AsyncMock(return_value=date(2026, 7, 10))
        mock_svc.get_latest_run_summary = AsyncMock(return_value={"latest_actual_week": "2026-07-10"})
        assert await needs_new_run() is False
