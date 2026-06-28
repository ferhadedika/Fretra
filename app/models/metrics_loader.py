"""
app/models/metrics_loader.py
===============================
One shared helper, used by every real model class, that reads
MAE/RMSE/MAPE(%)/R² for a given model_id out of SCFI_model_metrics.json
(the same JSON file your Streamlit dashboard already reads) - since
those metrics are computed during cross-validation and aren't stored
inside the model artifact (.pkl/.keras) itself.
"""

import json
from pathlib import Path


def load_metrics_for(metrics_json_path: Path, model_id: str) -> dict:
    if not metrics_json_path.exists():
        return {}
    try:
        with open(metrics_json_path) as f:
            data = json.load(f)
        for m in data.get("models", []):
            if m.get("id") == model_id:
                return {
                    "mae": m.get("MAE"),
                    "rmse": m.get("RMSE"),
                    "mape_pct": m.get("MAPE(%)"),
                    "r2": m.get("R²"),
                }
    except Exception:
        pass
    return {}
