"""
app/services/firestore_service.py
====================================
Wraps all Firestore reads/writes behind one class so the rest of the app
never imports the `firebase_admin` SDK directly.

KEY DESIGN CHOICE: if no Firebase credentials are configured (the normal
state before you've set up a project), this falls back to a simple
in-memory list instead of raising errors. The API works identically
either way; forecast history just won't survive a server restart until
real credentials are added. Flip it on by setting:

    FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccountKey.json
    FIREBASE_PROJECT_ID=your-project-id

in the backend's environment (see backend/.env.example).
"""

import logging
from datetime import datetime, date

from app.config import FIRESTORE_ENABLED, FIREBASE_CREDENTIALS_PATH, FIREBASE_PROJECT_ID

logger = logging.getLogger("freightai.firestore")

HISTORY_COLLECTION = "forecast_history"


def _json_safe(record: dict) -> dict:
    """Convert date objects to ISO strings so they're Firestore/JSON safe."""
    out = {}
    for k, v in record.items():
        if isinstance(v, (date, datetime)):
            out[k] = v.isoformat()
        elif isinstance(v, list):
            out[k] = [_json_safe(item) if isinstance(item, dict) else item for item in v]
        else:
            out[k] = v
    return out


class FirestoreService:
    """
    Real-mode: backed by Google Cloud Firestore via firebase_admin.
    Fallback-mode: backed by an in-memory list, used automatically when
    no credentials are configured. Same public interface either way.
    """

    def __init__(self):
        self._enabled = FIRESTORE_ENABLED
        self._db = None
        self._memory_store: list[dict] = []  # used only in fallback mode
        self._next_id = 1

        if self._enabled:
            try:
                import firebase_admin
                from firebase_admin import credentials, firestore

                if not firebase_admin._apps:
                    cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
                    firebase_admin.initialize_app(
                        cred,
                        {"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None,
                    )
                self._db = firestore.client()
                logger.info("Firestore connected (project=%s)", FIREBASE_PROJECT_ID)
            except Exception as exc:  # noqa: BLE001 - log and degrade gracefully
                logger.warning(
                    "Firestore credentials configured but connection failed (%s). "
                    "Falling back to in-memory storage.", exc,
                )
                self._enabled = False

    @property
    def mode(self) -> str:
        return "firestore" if self._enabled else "in-memory (no Firebase configured)"

    # ── Forecast history ───────────────────────────────────────────────────

    async def save_forecast_record(self, record: dict) -> str:
        safe_record = _json_safe(record)
        safe_record["created_at"] = datetime.utcnow().isoformat()

        if self._enabled:
            doc_ref = self._db.collection(HISTORY_COLLECTION).document()
            doc_ref.set(safe_record)
            return doc_ref.id

        record_id = str(self._next_id)
        self._next_id += 1
        safe_record["id"] = record_id
        self._memory_store.append(safe_record)
        return record_id

    async def get_forecast_history(self, limit: int = 50) -> list[dict]:
        if self._enabled:
            docs = (
                self._db.collection(HISTORY_COLLECTION)
                .order_by("created_at", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            return [{**doc.to_dict(), "id": doc.id} for doc in docs]

        return list(reversed(self._memory_store))[:limit]


# Module-level singleton, mirrors services/model_registry.py's pattern.
firestore_service = FirestoreService()
