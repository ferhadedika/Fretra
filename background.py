import joblib, numpy as np, pathlib

ARTIFACTS_DIR = pathlib.Path(r"C:\Users\User\Desktop\freightai-app new\freightai-app\backend\model_artifacts")

# Load what we already have
feat_sc = joblib.load(ARTIFACTS_DIR / "lstm_feature_scaler.pkl")
last_row = joblib.load(ARTIFACTS_DIR / "last_row.pkl")
feature_config = __import__("json").loads((ARTIFACTS_DIR / "feature_config.json").read_text())

ml_features = feature_config["ml_features"]
seq_len = feature_config["seq_len"]  # 12

# Build a dummy background — 30 rows of the last known row, scaled
last_x = feat_sc.transform([last_row[ml_features].values])  # (1, n_feats)
X_bg = np.tile(last_x, (30, 1))  # (30, n_feats)
X_seq_bg = np.array([X_bg[i:i+seq_len] for i in range(len(X_bg) - seq_len + 1)])  # (19, 12, n_feats)

joblib.dump(X_seq_bg, ARTIFACTS_DIR / "lstm_X_seq_background.pkl")
print(f"Saved lstm_X_seq_background.pkl — shape: {X_seq_bg.shape}")