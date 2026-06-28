"""
app/config.py
==============
Central settings, all overridable via environment variables so nothing
secret is hardcoded. Works with zero config out of the box (Firestore
calls become no-ops if no credentials are configured) so the API runs
standalone before Firebase is wired up.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/

# ── CORS ──────────────────────────────────────────────────────────────────
# Comma-separated list of allowed frontend origins. Defaults cover local
# dev (Vite/CRA) + Firebase Hosting's default domain pattern.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:5174"
).split(",")

# ── Model artifacts ──────────────────────────────────────────────────────
MODEL_ARTIFACTS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", BASE_DIR / "model_artifacts"))

# ── Firestore / Firebase ─────────────────────────────────────────────────
# Path to a service account JSON key. If unset or the file doesn't exist,
# FirestoreService falls back to in-memory storage so the API still works
# end-to-end without a Firebase project configured yet.
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")

FIRESTORE_ENABLED = bool(FIREBASE_CREDENTIALS_PATH and Path(FIREBASE_CREDENTIALS_PATH).exists())

# ── Forecast defaults ─────────────────────────────────────────────────────
DEFAULT_HORIZON_WEEKS = int(os.getenv("DEFAULT_HORIZON_WEEKS", "8"))
SCFI_MIN = float(os.getenv("SCFI_MIN", "800.0"))
SCFI_MAX = float(os.getenv("SCFI_MAX", "5000.0"))
