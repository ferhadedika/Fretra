"""
backend/app/routers/metrics_router.py
======================================
Serves the full SCFI_model_metrics.json merged with baseline CV metrics
to the frontend.
Add to main.py:  app.include_router(metrics_router.router)
"""
import json
import pathlib
from fastapi import APIRouter, HTTPException

from app.config import DEBUG_ENDPOINTS_ENABLED

router = APIRouter(prefix="/api", tags=["metrics"])

# File locations (relative to this file):
# This file:       freightai-app/backend/app/routers/metrics_router.py
# parent           = freightai-app/backend/app/routers/
# parent.parent    = freightai-app/backend/app/
# parent x3        = freightai-app/backend/
# parent x4        = freightai-app/

# SCFI_model_metrics.json (primary)  -> freightai-app/backend/
# SCFI_model_metrics.json (fallback) -> freightai-app/
METRICS_PATH  = pathlib.Path(__file__).parent.parent.parent / "SCFI_model_metrics.json"
FALLBACK_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "SCFI_model_metrics.json"

# SCFI_model_metrics_baselines.json -> freightai-app/backend/
BASELINES_PATH          = pathlib.Path(__file__).parent.parent.parent / "SCFI_model_metrics_baselines.json"
BASELINES_FALLBACK_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "SCFI_model_metrics_baselines.json"


def _load_json(*paths):
    for path in paths:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


@router.get("/metrics")
def get_metrics():
    data = _load_json(METRICS_PATH, FALLBACK_PATH)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="SCFI_model_metrics.json not found. Run the training script first.",
        )

    # Merge baseline CV metrics if available
    baselines_data = _load_json(BASELINES_PATH, BASELINES_FALLBACK_PATH)
    if baselines_data:
        data["baselines"]          = baselines_data.get("baselines", [])
        data["baseline_cv_config"] = baselines_data.get("baseline_cv_config", {})

    return data

@router.get("/metrics-debug")
def metrics_debug():
    # Filesystem paths + a directory listing - information disclosure to
    # any signed-in user (this app has no admin-role concept to gate it
    # more tightly), so it's opt-in only. Set DEBUG_ENDPOINTS_ENABLED=true
    # locally while wiring up METRICS_PATH on a new deployment.
    if not DEBUG_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "metrics_path": str(METRICS_PATH),
        "metrics_path_exists": METRICS_PATH.exists(),
        "fallback_path": str(FALLBACK_PATH),
        "fallback_exists": FALLBACK_PATH.exists(),
        "cwd": str(pathlib.Path.cwd()),
        "file_dir_contents": [str(p) for p in pathlib.Path(__file__).parent.parent.parent.iterdir()],
    }