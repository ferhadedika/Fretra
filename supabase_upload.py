"""
supabase_upload.py
==================
Call this after your pipeline runs to upsert the latest CSV rows
into Supabase so the React frontend always reads a live SCFI value.

Requirements:
    pip install supabase

Setup:
    1. Create a free project at https://supabase.com
    2. Run the SQL below in Supabase SQL Editor to create the table:

        create table if not exists scfi_weekly (
            date        date primary key,
            scfi        float8,
            brent_crude float8,
            cny_usd     float8,
            pmi_manufacturing float8,
            pmi_services      float8,
            trade_volume_index float8,
            trade_volume_yoy   float8,
            updated_at  timestamptz default now()
        );

    3. Set these env vars (or create a .env file):
        SUPABASE_URL=https://xxxx.supabase.co
        SUPABASE_KEY=your-anon-or-service-role-key

Usage:
    python supabase_upload.py                          # uploads default CSV
    python supabase_upload.py --csv path/to/file.csv  # custom path
"""

import os
import argparse
import pathlib
import pandas as pd
from supabase import create_client

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

DEFAULT_CSV  = pathlib.Path(__file__).parent / "cleaned_dataset_raw.csv"
TABLE_NAME   = "scfi_weekly"

# Columns to upload (must match the SQL table above)
CORE_COLS = [
    "Date", "SCFI", "Brent_Crude", "CNY_USD",
    "PMI_Manufacturing", "PMI_Services",
    "Trade_Volume_Index", "Trade_Volume_YoY",
]

SCFI_MIN = 800.0
SCFI_MAX = 5000.0


def denormalize(series: pd.Series) -> pd.Series:
    """Undo [0,1] MinMax scaling if values look scaled."""
    if series.max() <= 1.0:
        return series * (SCFI_MAX - SCFI_MIN) + SCFI_MIN
    return series


def upload(csv_path: pathlib.Path):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set as environment variables."
        )

    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Keep only core columns that exist
    cols = [c for c in CORE_COLS if c in df.columns]
    df = df[cols].copy()

    # Normalise date column
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Denormalize SCFI if needed
    if "SCFI" in df.columns:
        df["SCFI"] = denormalize(df["SCFI"])

    # Rename to snake_case to match DB columns
    df.rename(columns={
        "SCFI":                "scfi",
        "Brent_Crude":         "brent_crude",
        "CNY_USD":             "cny_usd",
        "PMI_Manufacturing":   "pmi_manufacturing",
        "PMI_Services":        "pmi_services",
        "Trade_Volume_Index":  "trade_volume_index",
        "Trade_Volume_YoY":    "trade_volume_yoy",
    }, inplace=True)

    # Drop rows with null date or SCFI
    df = df.dropna(subset=["date", "scfi"])

    records = df.to_dict(orient="records")
    print(f"Upserting {len(records)} rows into '{TABLE_NAME}'…")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Upsert in chunks of 500
    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        client.table(TABLE_NAME).upsert(chunk, on_conflict="date").execute()
        print(f"  → uploaded rows {i+1}–{min(i+chunk_size, len(records))}")

    print("✅ Supabase upload complete.")
    print(f"   Latest SCFI date : {df['date'].max()}")
    print(f"   Latest SCFI value: {df.loc[df['date'] == df['date'].max(), 'scfi'].values[0]:,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    upload(args.csv)
