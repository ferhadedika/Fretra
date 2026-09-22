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
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(BASE_DIR / ".env")

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

# ── Supabase ──────────────────────────────────────────────────────────────
# Real historical SCFI series lives here (see backend/supabase_upload.py),
# in the `scfi_weekly` table. Falls back to a synthetic series if unset.
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

# Optional. If set, the BACKEND's own Supabase client (app/services/
# supabase_service.py) uses this instead of SUPABASE_KEY. A service-role
# key bypasses Row Level Security entirely, which is what lets
# docs/sql/tighten_rls_to_service_role.sql remove anon/authenticated
# write access on the rolling-pipeline tables without breaking the
# backend/scraper's own writes - see that file's header comment for the
# two-step order this must be applied in. Leave unset and nothing about
# current behavior changes: the backend keeps using SUPABASE_KEY exactly
# as it does today, and the existing permissive RLS policy keeps working.
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ── Forecast defaults ─────────────────────────────────────────────────────
DEFAULT_HORIZON_WEEKS = int(os.getenv("DEFAULT_HORIZON_WEEKS", "8"))
SCFI_MIN = float(os.getenv("SCFI_MIN", "800.0"))
SCFI_MAX = float(os.getenv("SCFI_MAX", "5000.0"))

# ── Debug endpoints ───────────────────────────────────────────────────────
# GET /api/metrics-debug echoes server filesystem paths and a directory
# listing - useful while wiring up METRICS_PATH on a new deployment, but
# it's information disclosure (any signed-in user could hit it, not just
# an admin - there's no admin-role concept in this app) if left on by
# default in production. Off unless explicitly opted into.
DEBUG_ENDPOINTS_ENABLED = os.getenv("DEBUG_ENDPOINTS_ENABLED", "false").lower() == "true"

# ── Rolling forecast pipeline ─────────────────────────────────────────────
# How often the internal poller checks Supabase for a newer actual week
# than the last recorded rolling-forecast run (see app/scheduler/poller.py).
ROLLING_FORECAST_POLL_MINUTES = int(os.getenv("ROLLING_FORECAST_POLL_MINUTES", "30"))
# Tag stored on every forecast_history row - bump this if the artifacts in
# model_artifacts/ are ever swapped out, to distinguish which model
# generation produced a given historical prediction.
PREDICTION_VERSION = os.getenv("PREDICTION_VERSION", "v1")
