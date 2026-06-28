"""
app/models/feature_cascade.py
================================
Ports the row-mutation / feature-cascading logic from the training
script's forecast_tree_delta() / forecast_sarimax() / forecast_lstm_delta()
functions, so the API can generate multi-week-ahead forecasts WITHOUT
re-running the training pipeline.

WHY THIS EXISTS AS ITS OWN MODULE:
  Your training script's forecast functions mutate a pandas row across
  weeks - cascading ΔSCFI lags, recomputing EMAs/regime z-scores, advancing
  the calendar - and that exact cascade has to be reproduced bit-for-bit
  at serve time, or each model's week+2...+8 forecasts will silently drift
  from what the training script would have produced. This is ported as
  directly as possible from scfi_model_v3.py, operating on a plain dict
  instead of a DataFrame row (faster to mutate per-request, no pandas
  index overhead).
"""

MONTH_NAMES = ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def row_to_dict(last_row) -> dict:
    """Convert the saved pandas Series (last_row.pkl) into a plain dict."""
    return last_row.to_dict()


def advance_calendar(row: dict) -> None:
    """Mutates row in place: advances month/year/week/quarter/month-dummies by one step."""
    nm = int((row["month"] % 12) + 1)
    row["year"] = int(row["year"]) + (1 if nm == 1 else 0)
    row["month"] = nm
    row["week_of_year"] = int((row["week_of_year"] % 52) + 1)
    row["Q2"] = int(nm in [4, 5, 6])
    row["Q3"] = int(nm in [7, 8, 9])
    row["Q4"] = int(nm in [10, 11, 12])
    for i, mn in enumerate(MONTH_NAMES, 2):
        row[f"Month_{mn}"] = int(nm == i)


def cascade_tree_step(row: dict, delta: float, level: float, prev: float) -> None:
    """
    Mutates row in place after a tree-model (XGBoost/RF) step, exactly
    mirroring forecast_tree_delta()'s cascade in the training script.
    """
    row["DSCFI_lag12"] = row["DSCFI_lag8"]
    row["DSCFI_lag8"] = row["DSCFI_lag4"]
    row["DSCFI_lag4"] = row["DSCFI_lag3"]
    row["DSCFI_lag3"] = row["DSCFI_lag2"]
    row["DSCFI_lag2"] = row["DSCFI_lag1"]
    row["DSCFI_lag1"] = delta

    row["SCFI_pct_chg_4w"] = delta / (prev + 1e-9)
    row["SCFI_rolling_mean4"] = (row["SCFI_rolling_mean4"] * 3 + level) / 4
    row["SCFI_rolling_mean8"] = (row["SCFI_rolling_mean8"] * 7 + level) / 8
    row["SCFI_EMA4"] = 0.4 * level + 0.6 * row["SCFI_EMA4"]
    row["SCFI_EMA12"] = 0.154 * level + 0.846 * row["SCFI_EMA12"]
    row["SCFI_EMA4_vs_EMA12"] = row["SCFI_EMA4"] - row["SCFI_EMA12"]
    row["DSCFI_x_regime"] = row["DSCFI_lag1"] * row["SCFI_regime"]
    row["brent_x_pmi"] = row["Brent_Oil"] * row["World_PMI"]
    row["brent_x_cny"] = row["Brent_Oil"] * row["CNY_USD"]
    row["trade_x_pmi"] = row["Trade_Volume_Index"] * row["World_PMI"]

    advance_calendar(row)


def forecast_tree_delta(model, last_row: dict, ml_features: list[str],
                         last_scfi: float, horizon_weeks: int) -> list[dict]:
    """
    Direct port of forecast_tree_delta() from the training script.
    Works for both XGBoost and Random Forest/ExtraTrees - they share the
    same sklearn-style .predict(X) -> array interface.
    """
    import numpy as np

    row = dict(last_row)
    prev = last_scfi
    out = []
    for w in range(1, horizon_weeks + 1):
        x = np.array([[row[f] for f in ml_features]])
        delta = float(model.predict(x)[0])
        level = prev + delta
        out.append({"week": w, "forecast": round(level, 2), "delta": round(delta, 2)})
        cascade_tree_step(row, delta, level, prev)
        prev = level
    return out


def forecast_sarimax(sar_results, last_row: dict, exog_features: list[str],
                      horizon_weeks: int):
    """
    Direct port of forecast_sarimax() from the training script. SARIMAX's
    exog forecast only needs calendar dummies advanced (its own AR/MA
    terms handle the level dynamics) - exogenous LEVEL variables
    (Brent_Oil, CNY_USD, etc.) are held at their last known value, exactly
    matching the training script's behaviour.
    """
    import numpy as np

    rows = []
    last_exog = {f: last_row[f] for f in exog_features}
    cur_m = int(last_row["month"])
    for _ in range(horizon_weeks):
        cur_m = (cur_m % 12) + 1
        row = dict(last_exog)
        row["Q2"] = int(cur_m in [4, 5, 6])
        row["Q3"] = int(cur_m in [7, 8, 9])
        row["Q4"] = int(cur_m in [10, 11, 12])
        for i, mn in enumerate(MONTH_NAMES, 2):
            row[f"Month_{mn}"] = int(cur_m == i)
        rows.append([row[f] for f in exog_features])

    fc = sar_results.forecast(steps=horizon_weeks, exog=np.array(rows))
    fc_values = list(fc)
    return [{"week": w + 1, "forecast": round(float(v), 2)} for w, v in enumerate(fc_values)]


def forecast_lstm_delta(lstm_model, feat_scaler, target_scaler, x_seq_tail,
                         ml_features: list[str], last_scfi: float, horizon_weeks: int):
    """
    Direct port of forecast_lstm_delta() from the training script.
    x_seq_tail: the last SEQ_LEN rows of SCALED features, shape (SEQ_LEN, n_features) -
    this is X_sc[-SEQ_LEN:] in the training script, reconstructable from
    last_row + feat_scaler if you don't save it directly (see model class
    docstring for the fallback path).
    """
    import numpy as np

    l1 = ml_features.index("DSCFI_lag1")
    l2 = ml_features.index("DSCFI_lag2")
    l3 = ml_features.index("DSCFI_lag3")
    l4 = ml_features.index("DSCFI_lag4")
    l8 = ml_features.index("DSCFI_lag8")
    l12 = ml_features.index("DSCFI_lag12")

    seq = x_seq_tail.copy()
    out = []
    prev = last_scfi
    for _ in range(horizon_weeks):
        ps_sc = float(lstm_model.predict(seq[None], verbose=0).ravel()[0])
        delta = float(target_scaler.inverse_transform([[ps_sc]])[0][0])
        level = prev + delta
        out.append({"week": len(out) + 1, "forecast": round(level, 2), "delta": round(delta, 2)})

        ns = seq[-1].copy()
        ns[l12] = ns[l8]
        ns[l8] = ns[l4]
        ns[l4] = ns[l3]
        ns[l3] = ns[l2]
        ns[l2] = ns[l1]
        ns[l1] = ps_sc
        seq = np.vstack([seq[1:], ns[None]])
        prev = level
    return out
