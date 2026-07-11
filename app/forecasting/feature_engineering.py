"""
app/forecasting/feature_engineering.py
=========================================
The ONE place that turns raw weekly SCFI/macro rows into the exact
engineered feature matrix the trained models expect - ported verbatim
from the training script's own feature-engineering section (verified
byte-for-byte against the frozen model_artifacts/last_row.pkl, reproduced
from scfi_final_dataset.csv column-by-column before this module existed).

Deliberately has NO `app.*` imports - only pandas/numpy/stdlib - so this
same function can be dropped straight into the external training script
too. That's the whole point of "shared by training and inference": a
model retrained later and this app's rolling-forecast pipeline must
derive DSCFI_lag1, SCFI_EMA4, SCFI_regime, etc. from raw history the
exact same way, or every week+2..+N forecast silently drifts from what
retraining would have produced.

Input contract: a DataFrame with columns
    date, scfi, brent_crude, cny_usd, pmi_manufacturing, pmi_services,
    trade_volume_index, trade_volume_yoy
(this is exactly Supabase's `scfi_weekly` table shape - see
backend/scraper_pipeline.py's merge_and_upsert()), sorted or unsorted,
at any cadence (older rows may have irregular day-of-week gaps).

Output: the fully engineered DataFrame (67 ml_features + 19 exog_features
columns, dropna()'d exactly as training does - this drops the first ~52
warm-up weeks that lack a full rolling-52-week window, same as training).
"""

import numpy as np
import pandas as pd

MONTH_NAMES = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Supabase raw column -> internal name used by ml_features/exog_features.
# pmi_services has no engineered feature downstream (not in ML_FEATURES/
# EXOG_FEATURES) but is kept for parity with the training dataframe shape.
RAW_TO_INTERNAL = {
    "date": "date",
    "scfi": "SCFI",
    "brent_crude": "Brent_Oil",
    "cny_usd": "CNY_USD",
    "pmi_manufacturing": "World_PMI",
    "pmi_services": "PMI_Services",
    "trade_volume_index": "Trade_Volume_Index",
    "trade_volume_yoy": "Trade_Volume_YoY",
}


def to_weekly_friday(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Reindex onto a strict W-FRI weekly spine, forward-filling gaps -
    mirrors scraper_pipeline.py's to_weekly_friday() helper, which is how
    each source series was normalised before ever landing in Supabase.
    Needed here too because raw rows read back from Supabase aren't
    guaranteed evenly spaced (older backfilled rows can land on
    non-Friday dates), and every rolling/lag feature below assumes one
    row = one week.
    """
    out = df.set_index(date_col).sort_index()
    out = out.resample("W-FRI").last().ffill()
    return out.reset_index()


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full training-parity feature engineering pipeline. `raw_df` must have
    the Supabase `scfi_weekly` raw columns (see RAW_TO_INTERNAL); `date`
    must be parseable by pd.to_datetime.
    """
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = to_weekly_friday(df, "date")

    df = df.rename(columns=RAW_TO_INTERNAL)

    # ── Calendar ────────────────────────────────────────────────────────
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    quarter = df["date"].dt.quarter
    df["Q2"] = (quarter == 2).astype(int)
    df["Q3"] = (quarter == 3).astype(int)
    df["Q4"] = (quarter == 4).astype(int)
    for i, mn in enumerate(MONTH_NAMES, 2):
        df[f"Month_{mn}"] = (df["month"] == i).astype(int)

    # ── Base lags (plain col.shift(k) - verified identical to training's
    #    own pre-computed lag columns) ──────────────────────────────────
    for k in (1, 2, 3, 4):
        df[f"SCFI_lag{k}"] = df["SCFI"].shift(k)
    for k in (1, 2, 4, 8):
        df[f"Brent_Oil_lag{k}"] = df["Brent_Oil"].shift(k)
    for k in (1, 2, 4):
        df[f"CNY_USD_lag{k}"] = df["CNY_USD"].shift(k)
    for k in (1, 4, 8):
        df[f"World_PMI_lag{k}"] = df["World_PMI"].shift(k)
        df[f"Trade_Volume_Index_lag{k}"] = df["Trade_Volume_Index"].shift(k)

    # ── First-differenced target + its lags ────────────────────────────
    df["DELTA_SCFI"] = df["SCFI"] - df["SCFI_lag1"]
    for k in (1, 2, 3, 4, 8, 12):
        df[f"DSCFI_lag{k}"] = df["DELTA_SCFI"].shift(k)

    # ── Base momentum ───────────────────────────────────────────────────
    df["SCFI_pct_chg_4w"] = df["SCFI"].pct_change(4)
    df["SCFI_rolling_mean4"] = df["SCFI"].rolling(4).mean()
    df["SCFI_rolling_std4"] = df["SCFI"].rolling(4).std()

    # ── Regime (52-week window) ─────────────────────────────────────────
    REGIME_WIN = 52
    roll52 = df["SCFI"].rolling(REGIME_WIN)
    df["SCFI_z52"] = (df["SCFI"] - roll52.mean()) / (roll52.std() + 1e-9)
    df["SCFI_dist_hi52"] = (df["SCFI"] - roll52.max()) / (roll52.max() + 1e-9)
    df["SCFI_dist_lo52"] = (df["SCFI"] - roll52.min()) / (roll52.min() + 1e-9)
    df["SCFI_regime"] = pd.cut(
        df["SCFI_z52"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]
    ).astype(float)

    # ── Extended momentum ────────────────────────────────────────────────
    df["SCFI_rolling_mean8"] = df["SCFI"].rolling(8).mean()
    df["SCFI_rolling_mean13"] = df["SCFI"].rolling(13).mean()
    df["SCFI_rolling_std8"] = df["SCFI"].rolling(8).std()
    df["SCFI_EMA4"] = df["SCFI"].ewm(span=4, adjust=False).mean()
    df["SCFI_EMA12"] = df["SCFI"].ewm(span=12, adjust=False).mean()
    df["SCFI_EMA4_vs_EMA12"] = df["SCFI_EMA4"] - df["SCFI_EMA12"]

    # ── Exogenous rate-of-change ─────────────────────────────────────────
    for col in ["Brent_Oil", "World_PMI", "Trade_Volume_Index"]:
        df[f"{col}_delta1"] = df[col].diff(1)
        df[f"{col}_delta4"] = df[col].diff(4)
        df[f"{col}_pct4"] = df[col].pct_change(4)

    brent_roll13 = df["Brent_Oil"].rolling(13)
    df["Brent_z13"] = (df["Brent_Oil"] - brent_roll13.mean()) / (brent_roll13.std() + 1e-9)

    # ── Interactions ─────────────────────────────────────────────────────
    df["brent_x_pmi"] = df["Brent_Oil"] * df["World_PMI"]
    df["brent_x_cny"] = df["Brent_Oil"] * df["CNY_USD"]
    df["trade_x_pmi"] = df["Trade_Volume_Index"] * df["World_PMI"]
    df["DSCFI_x_regime"] = df["DSCFI_lag1"] * df["SCFI_regime"]

    return df.dropna().reset_index(drop=True)


def build_anchor_inputs(engineered_df: pd.DataFrame, ml_features: list[str], seq_len: int) -> dict:
    """
    Returns the pieces a rolling-forecast run needs to build a
    ForecastAnchor: {last_row, last_scfi, last_date, raw_seq_tail}.
    `raw_seq_tail` is left UNSCALED (shape (seq_len, len(ml_features)),
    original units) - the caller applies the model's own frozen feature
    scaler (LstmModel.transform_sequence()), keeping this module
    scaler/model-agnostic.
    """
    last_row = engineered_df.iloc[-1].to_dict()
    last_scfi = float(engineered_df["SCFI"].iloc[-1])
    last_date = engineered_df["date"].iloc[-1].date()
    raw_seq_tail = engineered_df[ml_features].values[-seq_len:]
    return {
        "last_row": last_row,
        "last_scfi": last_scfi,
        "last_date": last_date,
        "raw_seq_tail": raw_seq_tail,
    }
