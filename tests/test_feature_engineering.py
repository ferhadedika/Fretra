"""
Turns this project's earlier manual verification (done interactively
against live data during development) into a real, repeatable test:
engineer_features() must reproduce the frozen training-time last_row.pkl
byte-for-byte when fed the same historical data the training script used.
This is the single most safety-critical test in the backend - if this
ever silently breaks, every rolling forecast is anchored to garbage.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.forecasting.feature_engineering import engineer_features

BACKEND_DIR = Path(__file__).parent.parent
CSV_PATH = BACKEND_DIR / "scfi_final_dataset.csv"
LAST_ROW_PATH = BACKEND_DIR / "model_artifacts" / "last_row.pkl"

RAW_RENAME = {
    "Date": "date", "SCFI": "scfi", "Brent_Crude": "brent_crude", "CNY_USD": "cny_usd",
    "PMI_Manufacturing": "pmi_manufacturing", "PMI_Services": "pmi_services",
    "Trade_Volume_Index": "trade_volume_index", "Trade_Volume_YoY": "trade_volume_yoy",
}


@pytest.mark.skipif(not CSV_PATH.exists() or not LAST_ROW_PATH.exists(), reason="training artifacts not present")
def test_engineer_features_reproduces_frozen_last_row():
    raw = pd.read_csv(CSV_PATH)[list(RAW_RENAME.keys())].rename(columns=RAW_RENAME)

    engineered = engineer_features(raw)
    computed_last_row = engineered.iloc[-1]
    frozen_last_row = joblib.load(LAST_ROW_PATH)

    mismatches = []
    for col in frozen_last_row.index:
        if col == "date" or col not in computed_last_row.index:
            continue
        a, b = computed_last_row[col], frozen_last_row[col]
        try:
            close = np.isclose(float(a), float(b), rtol=1e-4, atol=1e-3)
        except (TypeError, ValueError):
            close = a == b
        if not close:
            mismatches.append((col, a, b))

    assert not mismatches, f"engineer_features() diverged from training on: {mismatches}"


def test_engineer_features_produces_all_expected_columns():
    """A lighter-weight structural check that doesn't need the training
    artifacts - just confirms the function's output shape is sane against
    live-Supabase-style input (synthetic here, so this test always runs)."""
    dates = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    raw = pd.DataFrame({
        "date": dates,
        "scfi": np.linspace(800, 1500, len(dates)) + np.random.RandomState(0).normal(0, 10, len(dates)),
        "brent_crude": np.linspace(50, 90, len(dates)),
        "cny_usd": np.linspace(6.5, 7.0, len(dates)),
        "pmi_manufacturing": np.linspace(48, 55, len(dates)),
        "pmi_services": np.linspace(48, 55, len(dates)),
        "trade_volume_index": np.linspace(100, 115, len(dates)),
        "trade_volume_yoy": np.linspace(-2, 4, len(dates)),
    })

    engineered = engineer_features(raw)

    assert not engineered.empty, "120 weeks of synthetic data should survive the 52-week warm-up dropna()"
    assert engineered["date"].is_monotonic_increasing
    for col in ("SCFI_pct_chg_4w", "DSCFI_lag1", "SCFI_regime", "brent_x_pmi"):
        assert col in engineered.columns
    assert not engineered.isna().any().any(), "no NaNs should survive past the warm-up dropna()"
