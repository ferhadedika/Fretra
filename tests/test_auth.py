"""
The auth dependency is the entire security boundary for every data-bearing
route (see app/main.py's dependencies=_auth_dep on each router). These
tests cover the two failure modes that matter most: no token at all, and
a token Supabase rejects as invalid/expired.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.dependencies.auth import get_current_user


async def test_rejects_missing_bearer_token():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=None)
    assert exc_info.value.status_code == 401


async def test_rejects_invalid_token():
    fake_credentials = MagicMock()
    fake_credentials.credentials = "not-a-real-token"

    with patch("app.dependencies.auth._auth_client") as mock_client:
        mock_client.auth.get_user.side_effect = Exception("invalid JWT")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=fake_credentials)
        assert exc_info.value.status_code == 401


async def test_rejects_when_supabase_returns_no_user():
    fake_credentials = MagicMock()
    fake_credentials.credentials = "some-token"

    fake_result = MagicMock()
    fake_result.user = None

    with patch("app.dependencies.auth._auth_client") as mock_client:
        mock_client.auth.get_user.return_value = fake_result
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials=fake_credentials)
        assert exc_info.value.status_code == 401


async def test_accepts_valid_token():
    fake_credentials = MagicMock()
    fake_credentials.credentials = "a-real-token"

    fake_user = MagicMock()
    fake_result = MagicMock()
    fake_result.user = fake_user

    with patch("app.dependencies.auth._auth_client") as mock_client:
        mock_client.auth.get_user.return_value = fake_result
        user = await get_current_user(credentials=fake_credentials)
        assert user is fake_user
