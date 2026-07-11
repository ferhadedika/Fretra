"""
app/dependencies/auth.py
=========================
Verifies the Supabase-issued JWT sent by the frontend so the API itself
refuses unauthenticated access — the frontend's client-side route guard on
`/app` doesn't stop someone calling these endpoints directly.
"""
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import SUPABASE_ENABLED, SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger("freightai.auth")

_bearer = HTTPBearer(auto_error=False)

_auth_client = None
if SUPABASE_ENABLED:
    from supabase import create_client

    _auth_client = create_client(SUPABASE_URL, SUPABASE_KEY)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    """Raises 401 unless a valid Supabase session token is supplied."""
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token.")

    if _auth_client is None:
        # No Supabase project configured on this deployment — fail closed
        # instead of silently letting every request through.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Authentication is not configured on this server.",
        )

    try:
        result = _auth_client.auth.get_user(credentials.credentials)
    except Exception as exc:  # noqa: BLE001 - any verification failure -> 401
        logger.warning("Token verification failed: %s", exc)
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or expired session."
        ) from exc

    if result is None or result.user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session.")

    return result.user
