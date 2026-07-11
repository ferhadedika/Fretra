"""
dm_test.py
============
Diebold-Mariano test for equal predictive accuracy, run against
cv_predictions.json (the per-observation CV predictions captured by
scfi_model_v3.py - see that script's "NEW" comments for provenance).

Explicitly inner-joins each pair of models on date before pairing errors,
rather than trusting positional alignment (even though this run's data
happens to already be positionally aligned - confirmed separately).

Implements the classic DM statistic with a Newey-West/Bartlett-weighted
long-run variance estimator (h-1 lags, h=1 here since every prediction in
cv_predictions.json is a genuine one-step-ahead forecast - both trees and
SARIMAX reconstruct/predict from the TRUE lag-1 level, not a prior
forecast - so h=1 is correct, not an approximation), plus the Harvey,
Leybourne & Newbold (1997) small-sample correction, since n=395 is not
large enough to rely on the plain asymptotic normal approximation.
"""
import json
import numpy as np
from scipy import stats

with open("cv_predictions.json") as f:
    CV = json.load(f)


def series(model_id):
    d = CV[model_id]
    return dict(zip(d["date"], zip(d["actual"], d["predicted"])))


def joined_errors(model_a, model_b):
    """Inner-join model_a and model_b on date; return aligned error arrays
    (actual - predicted) for each, plus the shared dates."""
    sa, sb = series(model_a), series(model_b)
    shared_dates = sorted(set(sa) & set(sb))
    e_a, e_b = [], []
    for dt in shared_dates:
        act_a, pred_a = sa[dt]
        act_b, pred_b = sb[dt]
        assert abs(act_a - act_b) < 1e-6, f"actual mismatch at {dt}: {act_a} vs {act_b}"
        e_a.append(act_a - pred_a)
        e_b.append(act_b - pred_b)
    return np.array(e_a), np.array(e_b), shared_dates


def dm_test(e1, e2, h=1, loss="MAE"):
    """Diebold-Mariano test. e1/e2: forecast errors (actual - predicted),
    same length, date-aligned. loss: 'MAE' or 'MSE'.
    Returns dict with DM stat (HLN-corrected), p-value, mean loss
    differential (d1 - d2; negative => model 1 more accurate), n, and the
    correlation between the two raw error series (relevant when one
    model is a weighted blend containing the other, e.g. Ensemble vs XGBoost)."""
    n = len(e1)
    if loss == "MSE":
        d = e1 ** 2 - e2 ** 2
    elif loss == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'MAE' or 'MSE'")

    d_bar = np.mean(d)

    # Newey-West / Bartlett-weighted long-run variance, h-1 lags (h=1 -> just var(d)/n)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * (1 - lag / h) * cov
    var_d /= n

    dm_raw = d_bar / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat = dm_raw * hln

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

    return {
        "n": n,
        "loss": loss,
        "mean_loss_diff": float(d_bar),
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "error_correlation": float(np.corrcoef(e1, e2)[0, 1]),
    }


def run_comparison(label, model_a, model_b):
    e_a, e_b, dates = joined_errors(model_a, model_b)
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  n = {len(dates)} paired observations, {dates[0]} -> {dates[-1]}")
    print(f"  MAE({model_a}) = {np.mean(np.abs(e_a)):.2f}   MAE({model_b}) = {np.mean(np.abs(e_b)):.2f}")
    print(f"  RMSE({model_a}) = {np.sqrt(np.mean(e_a**2)):.2f}   RMSE({model_b}) = {np.sqrt(np.mean(e_b**2)):.2f}")

    for loss in ("MAE", "MSE"):
        r = dm_test(e_a, e_b, h=1, loss=loss)
        winner = model_a if r["mean_loss_diff"] < 0 else model_b
        sig = "SIGNIFICANT (p<0.05)" if r["p_value"] < 0.05 else "not significant (p>=0.05)"
        print(f"\n  --- Loss = {loss} ---")
        print(f"  mean loss differential ({model_a} - {model_b}): {r['mean_loss_diff']:+.3f}")
        print(f"  DM statistic (HLN-corrected): {r['dm_stat']:+.3f}")
        print(f"  p-value: {r['p_value']:.4f}  -> {sig}")
        print(f"  lower-loss model this test favors: {winner}")
        print(f"  correlation between the two models' raw errors: {r['error_correlation']:.3f}")


run_comparison("XGBoost vs SARIMAX (ML vs classical baseline)", "xgboost", "sarimax")
run_comparison("XGBoost vs LSTM (best tree model vs deep learning)", "xgboost", "lstm")
run_comparison("Ensemble vs XGBoost (is the 10%-floor diversification cost real or noise)", "ensemble", "xgboost")
