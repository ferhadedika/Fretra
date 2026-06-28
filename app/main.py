"""
app/main.py
============
FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000

Run in production (e.g. on Render/Railway):
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""
import os
import json

# ── Write Firebase service-account JSON from env var to disk, if provided ──
# Render (and similar platforms) can't easily mount a JSON file, so we pass
# the credentials as a single env var and materialize it as a file here,
# before app.config reads FIREBASE_CREDENTIALS_PATH.
_fb_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
if _fb_json and not os.getenv("FIREBASE_CREDENTIALS_PATH"):
    _cred_path = "/tmp/firebase-credentials.json"
    with open(_cred_path, "w") as f:
        f.write(_fb_json)
    os.environ["FIREBASE_CREDENTIALS_PATH"] = _cred_path
    
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.services.model_registry import registry
from app.services.firestore_service import firestore_service
from app.routers import models_router, forecast_router, shap_router
from app.routers import metrics_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freightai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load every model ONCE, not per-request ──────────────────
    logger.info("Loading models...")
    registry.load_all()
    logger.info("Models loaded: %s", [m.model_id for m in registry.list_all()])
    logger.info("Real vs. mock status: %s", registry.real_status())
    logger.info("Firestore mode: %s", firestore_service.mode)
    yield
    # ── Shutdown: nothing to clean up for the mock models; a real LSTM/
    # GPU-backed model might release resources here. ────────────────────
    logger.info("Shutting down.")


app = FastAPI(
    title="FreightAI SCFI Forecasting API",
    description="Serves XGBoost, Random Forest, SARIMAX, LSTM, and Ensemble "
                 "SCFI forecasts with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router.router)
app.include_router(forecast_router.router)
app.include_router(shap_router.router)


@app.get("/api/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "models_ready": registry.is_ready,
        "models_real_status": registry.real_status(),
        "firestore_mode": firestore_service.mode,
    }
app.include_router(metrics_router.router)