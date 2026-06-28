"""
compute_baselines.py
=====================
Computes naive baseline forecasting metrics from scfi_final_dataset.csv
and appends them to SCFI_model_metrics.json under a "baselines" key.

Baselines implemented:
  1. Naive          — forecast = last known value (random walk)
  2. Seasonal Naive — forecast = value from 52 weeks ago
  3. MA4            — forecast = 4-week simple moving average
  4. EWM            — forecast = exponentially weighted mean (alpha=0.3)

Metrics computed per baseline (same as your models):
  MAE, RMSE, MAPE(%), R²

Uses the same walk-forward validation approach as your models:
  - Expanding window, 5 folds
  - Each fold trains on past, tests on next chunk
  - No future data leakage

Run:
    python compute_baselines.py
    python compute_baselines.py --csv path/to/file.csv --json path/to/metrics.json
"""

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).parent
DEFAULT_CSV  = SCRIPT_DIR / "scfi_final_dataset.csv"
DEFAULT_JSON = SCRIPT_DIR / "SCFI_model_metrics_baselines.json"

SCFI_MIN = 800.0
SCFI_MAX = 5000.0
N_SPLITS = 5


def denormalize(series: pd.Series) -> pd.Series:
    if series.max() <= 1.0:
        return series * (SCFI_MAX - SCFI_MIN) + SCFI_MIN
    return series


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": None, "RMSE": None, "MAPE(%)": None, "R²": None}

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    r2   = r2_score(y_true, y_pred)

    return {
        "MAE":     round(mae,  2),
        "RMSE":    round(rmse, 2),
        "MAPE(%)": round(mape, 4),
        "R²":      round(r2,   6),
    }


def walk_forward_baseline(scfi: pd.Series, baseline_fn, n_splits=N_SPLITS):
    """
    Run walk-forward CV for a baseline that only needs the history up to
    the test window. baseline_fn(train_series) -> scalar forecast for next step.
    Returns list of fold metric dicts.
    """
    tss        = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(tss.split(scfi), 1):
        train = scfi.iloc[train_idx]
        test  = scfi.iloc[test_idx]

        preds = []
        history = train.copy()
        for i in range(len(test)):
            pred = baseline_fn(history)
            preds.append(pred)
            history = pd.concat([history, pd.Series([test.iloc[i]])], ignore_index=True)

        m = compute_metrics(test.values, preds)
        m["fold"] = fold
        fold_metrics.append(m)
        all_true.extend(test.values)
        all_pred.extend(preds)

    overall = compute_metrics(all_true, all_pred)
    return overall, fold_metrics


def naive_fn(history):
    return history.iloc[-1]

def ma4_fn(history):
    return history.iloc[-4:].mean()

def ewm_fn(history):
    return history.ewm(alpha=0.3, adjust=False).mean().iloc[-1]

def seasonal_naive_fn(history):
    if len(history) >= 52:
        return history.iloc[-52]
    return history.iloc[-1]  # fallback if not enough history


def run(csv_path: pathlib.Path, json_path: pathlib.Path):
    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    scfi = denormalize(df["SCFI"].copy())
    print(f"  {len(scfi)} rows, SCFI range: {scfi.min():.1f} – {scfi.max():.1f}")

    # ── Compute baselines ─────────────────────────────────────────────────────
    baselines_config = [
        {
            "id":    "naive",
            "label": "Naïve (Random Walk)",
            "fn":    naive_fn,
            "description": "Forecast = last observed SCFI value. The simplest possible benchmark.",
        },
        {
            "id":    "seasonal_naive",
            "label": "Seasonal Naïve",
            "fn":    seasonal_naive_fn,
            "description": "Forecast = SCFI value from 52 weeks ago. Captures annual seasonality.",
        },
        {
            "id":    "ma4",
            "label": "Moving Average (4-week)",
            "fn":    ma4_fn,
            "description": "Forecast = simple average of last 4 weeks. Smooths short-term noise.",
        },
        {
            "id":    "ewm",
            "label": "Exponential Smoothing (α=0.3)",
            "fn":    ewm_fn,
            "description": "Forecast = exponentially weighted average, weighting recent values more.",
        },
    ]

    results = []
    for b in baselines_config:
        print(f"  Computing baseline: {b['label']}…")
        overall, fold_mets = walk_forward_baseline(scfi, b["fn"])
        results.append({
            "id":          b["id"],
            "label":       b["label"],
            "description": b["description"],
            "MAE":         overall["MAE"],
            "RMSE":        overall["RMSE"],
            "MAPE(%)":     overall["MAPE(%)"],
            "R²":          overall["R²"],
            "fold_metrics": fold_mets,
        })
        print(f"    MAE={overall['MAE']:.1f}  RMSE={overall['RMSE']:.1f}  "
              f"MAPE={overall['MAPE(%)']:.2f}%  R²={overall['R²']:.4f}")

    # ── Load existing metrics JSON ─────────────────────────────────────────────
    if json_path.exists():
        with open(json_path, "r") as f:
            metrics = json.load(f)
        print(f"\nLoaded existing metrics JSON: {json_path.name}")
    else:
        metrics = {}
        print(f"\nNo existing JSON found — creating new one at {json_path.name}")

    # ── Compute improvement vs naive ─────────────────────────────────────────
    naive_mae = next(b["MAE"] for b in results if b["id"] == "naive")

    # Add improvement stats to each ML model in the JSON
    for m in metrics.get("models", []):
        if m.get("MAE") and naive_mae:
            m["vs_naive_pct"] = round((naive_mae - m["MAE"]) / naive_mae * 100, 2)

    # ── Write baselines into JSON ─────────────────────────────────────────────
    metrics["baselines"] = results
    metrics["baseline_cv_config"] = {
        "n_splits":  N_SPLITS,
        "method":    "TimeSeriesSplit (expanding walk-forward)",
        "computed":  pd.Timestamp.now().strftime("%Y-%m-%d"),
    }

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Baselines written to {json_path.name}")
    print(f"   Naïve MAE (benchmark): {naive_mae:.1f}")

    # Print improvement table
    print("\n── Improvement over Naïve baseline ──────────────────────")
    for m in metrics.get("models", []):
        imp = m.get("vs_naive_pct")
        if imp is not None:
            print(f"  {m['label']:25s}  MAE {m['MAE']:.1f}  ({imp:+.1f}% vs Naïve)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=pathlib.Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    run(args.csv, args.json)
