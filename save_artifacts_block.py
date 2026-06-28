# ═══════════════════════════════════════════════════════════════════════════
# ─── SAVE MODEL ARTIFACTS ─────────────────────────────────────────────────
# Add this block to scfi_model_v3.py right before "# ─── 10. JSON output"
# (i.e. after lstm_final/xgb_final/rf_final/sar_res_full all exist, but
# before the script ends). Needs: pip install joblib
# ═══════════════════════════════════════════════════════════════════════════

import joblib

ARTIFACTS_DIR = SCRIPT_DIR / "model_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 60)
print("  SAVING MODEL ARTIFACTS")
print("=" * 60)

# 1. XGBoost — joblib handles sklearn-API xgboost models fine
joblib.dump(xgb_final, ARTIFACTS_DIR / "xgboost.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'xgboost.pkl'}")

# 2. Random Forest / ExtraTrees (best_rf_type tells you which one it actually is)
joblib.dump(rf_final, ARTIFACTS_DIR / "random_forest.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'random_forest.pkl'}  (actual type: {best_rf_type})")

# 3. SARIMAX — statsmodels results objects have their own .save(), use that
#    instead of joblib (joblib *can* pickle it, but .save() is the
#    statsmodels-blessed path and is more robust across versions)
sar_res_full.save(str(ARTIFACTS_DIR / "sarimax.pkl"))
print(f"  Saved → {ARTIFACTS_DIR / 'sarimax.pkl'}")

# 4. LSTM — Keras model, use its native .save() (creates a single .keras file)
lstm_final.save(str(ARTIFACTS_DIR / "lstm.keras"))
print(f"  Saved → {ARTIFACTS_DIR / 'lstm.keras'}")

# 5. The scalers — CRITICAL, easy to forget. Without these, you cannot turn
#    LSTM's scaled predictions back into real SCFI numbers, and you cannot
#    scale new input data into what the LSTM expects.
joblib.dump(feat_sc, ARTIFACTS_DIR / "lstm_feature_scaler.pkl")
joblib.dump(tgt_sc,  ARTIFACTS_DIR / "lstm_target_scaler.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'lstm_feature_scaler.pkl'}")
print(f"  Saved → {ARTIFACTS_DIR / 'lstm_target_scaler.pkl'}")

# 6. Ensemble weight + which RF variant won — small enough to just be JSON
with open(ARTIFACTS_DIR / "ensemble_config.json", "w") as f:
    json.dump({
        "xgb_weight": round(w_star, 4),
        "rf_weight": round(1 - w_star, 4),
        "rf_variant": best_rf_type,   # "RF" or "ET" - tells you which class to re-instantiate
    }, f, indent=2)
print(f"  Saved → {ARTIFACTS_DIR / 'ensemble_config.json'}")

# 7. Feature lists + last known row — needed to build the next prediction's
#    input vector without re-running the whole training script.
with open(ARTIFACTS_DIR / "feature_config.json", "w") as f:
    json.dump({
        "ml_features": ML_FEATURES,
        "exog_features": EXOG_FEATURES,
        "seq_len": SEQ_LEN,
        "last_scfi": last_scfi,
        "last_date": str(last_date.date()),
    }, f, indent=2)
print(f"  Saved → {ARTIFACTS_DIR / 'feature_config.json'}")

# 8. last_row is a pandas Series with NaNs/Timestamps possible - save as
#    pickle (joblib) rather than JSON to avoid serialization headaches
joblib.dump(last_row, ARTIFACTS_DIR / "last_row.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'last_row.pkl'}")

# 9. SHAP background data - REQUIRED for the API to compute SHAP at serve
#    time. Without these, global feature importance can only be computed
#    from a single row instead of a real background sample.
joblib.dump(X, ARTIFACTS_DIR / "X_train_full.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'X_train_full.pkl'}")

joblib.dump(X_seq[-30:], ARTIFACTS_DIR / "lstm_X_seq_background.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'lstm_X_seq_background.pkl'}")

# 10. SARIMAX SHAP background - the last 50 rows of exog history (mirrors
#     X_s[-50:] in the training script's own SARIMAX SHAP section).
#     Without real variation across rows, KernelExplainer's background
#     equals the explained point and every SHAP value comes out as zero.
joblib.dump(X_s[-50:], ARTIFACTS_DIR / "sarimax_exog_background.pkl")
print(f"  Saved → {ARTIFACTS_DIR / 'sarimax_exog_background.pkl'}")

print(f"\n  All artifacts saved to: {ARTIFACTS_DIR}")
