"""
app/routers/models_router.py
===============================
Endpoints for discovering which models are available and their accuracy
metrics. No heavy computation here — registry.list_all() just reads
already-loaded model objects, so this stays fast without needing to be
anything special.
"""

from fastapi import APIRouter

from app.schemas.forecast import ModelMeta
from app.services import forecast_service

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelMeta])
async def list_models():
    """All available models and their cross-validated accuracy metrics."""
    return await forecast_service.get_all_models_meta()
