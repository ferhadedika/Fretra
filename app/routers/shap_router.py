"""
app/routers/shap_router.py
=============================
SHAP explainability endpoint, separated from forecast_router.py because
it's a conceptually distinct concern (explaining a prediction vs. making
one) even though both currently read from the same model object.

PERFORMANCE NOTE: SARIMAX's explain() uses shap.KernelExplainer, which is
genuinely slow (~10-15s measured) since it's a black-box sampling method
calling SARIMAX's own Kalman-filter forecast() repeatedly. That's too
long to run directly inside an async endpoint - it would block the event
loop and stall every other concurrent request for its full duration.
get_shap_explanation() below runs in a threadpool for exactly this
reason; XGBoost/RandomForest's TreeExplainer is fast enough that this
costs nothing for them, but it's what keeps SARIMAX from blocking the
whole server.
"""

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.schemas.forecast import ShapResponse, ModelId
from app.services import forecast_service

router = APIRouter(prefix="/api/shap", tags=["shap"])


@router.get("/{model_id}", response_model=ShapResponse)
async def get_shap(model_id: ModelId, horizon_weeks: int = Query(8, ge=1, le=12)):
    """
    Global feature importance (mean |SHAP| across training data) and
    local importance (SHAP values behind the most recent prediction) for
    the given model.

    Runs in a threadpool, not directly on the event loop - see module
    docstring. This means concurrent requests to OTHER endpoints stay
    responsive even while a slow SARIMAX explanation is computing, at
    the cost of this specific request still taking as long as the
    underlying SHAP computation does (10-15s for SARIMAX, sub-second
    for the tree models).
    """
    try:
        return await run_in_threadpool(
            forecast_service._build_shap_response_sync, model_id, horizon_weeks
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
