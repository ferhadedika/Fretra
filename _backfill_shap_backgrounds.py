"""
One-off backfill for the SHAP global-importance background files that
scfi_model_v3.py's save block didn't produce before this fix (see the
new joblib.dump(X, ...) / joblib.dump(X_s[-50:], ...) calls added
alongside lstm_X_seq_background.pkl in that script).

Recomputes X (68 ml_features) and X_s (19 exog_features, last 50 rows)
from the SAME live Supabase data + feature_engineering.py pipeline the
Jul 10 retrain already used, so the backgrounds match the currently
deployed model without re-running the expensive training.

Run from backend/ with backend/venv (has pandas/numpy/joblib/supabase-py,
no need for the heavy repo-root .venv):
    venv/Scripts/python.exe _backfill_shap_backgrounds.py
"""
import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from app.forecasting.feature_engineering import engineer_features
from app.services.supabase_service import supabase_service

BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BACKEND_DIR / "model_artifacts"
BACKUPS_ROOT = BACKEND_DIR / "model_artifacts_backups"


async def main() -> None:
    print("Fetching full raw history from Supabase...")
    raw_rows = await supabase_service.get_full_historical()
    if not raw_rows:
        raise RuntimeError("get_full_historical() returned no rows - check Supabase connectivity/.env")
    raw_df = pd.DataFrame(raw_rows)
    print(f"  {len(raw_df)} raw rows, {raw_df['date'].min()} -> {raw_df['date'].max()}")

    df = engineer_features(raw_df)
    print(f"  {len(df)} engineered rows after dropna()")

    with open(ARTIFACTS_DIR / "feature_config.json") as f:
        cfg = json.load(f)
    ml_features = cfg["ml_features"]
    exog_features = cfg["exog_features"]

    missing_ml = [c for c in ml_features if c not in df.columns]
    missing_exog = [c for c in exog_features if c not in df.columns]
    if missing_ml or missing_exog:
        raise RuntimeError(f"engineered df missing columns - ml:{missing_ml} exog:{missing_exog}")

    X = df[ml_features].values
    X_s = df[exog_features].values
    print(f"  X_train_full shape: {X.shape}")
    print(f"  sarimax_exog_background shape (last 50 rows): {X_s[-50:].shape}")

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUPS_ROOT / f"{stamp}-shap-bg-backfill"
    backup_dir.mkdir(parents=True, exist_ok=True)
    old_exog = ARTIFACTS_DIR / "sarimax_exog_background.pkl"
    if old_exog.exists():
        shutil.copy2(old_exog, backup_dir / "sarimax_exog_background.pkl")
        print(f"  Backed up old sarimax_exog_background.pkl -> {backup_dir}")
    old_xfull = ARTIFACTS_DIR / "X_train_full.pkl"
    if old_xfull.exists():
        shutil.copy2(old_xfull, backup_dir / "X_train_full.pkl")

    joblib.dump(X, ARTIFACTS_DIR / "X_train_full.pkl")
    print(f"  Saved -> {ARTIFACTS_DIR / 'X_train_full.pkl'}")
    joblib.dump(X_s[-50:], ARTIFACTS_DIR / "sarimax_exog_background.pkl")
    print(f"  Saved -> {ARTIFACTS_DIR / 'sarimax_exog_background.pkl'}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
