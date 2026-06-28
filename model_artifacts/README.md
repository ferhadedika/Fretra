# Model Artifacts

Drop your real trained model files here once you have them:

| File (example)      | Produced by                          |
|----------------------|---------------------------------------|
| `xgboost.pkl`        | `joblib.dump(model, "xgboost.pkl")`   |
| `random_forest.pkl`  | `joblib.dump(model, "random_forest.pkl")` |
| `sarimax.pkl`        | `model.save("sarimax.pkl")` (statsmodels) |
| `lstm.h5`            | `model.save("lstm.h5")` (Keras/TF)    |
| `ensemble_weights.json` | however your training script saves the blend weights |

None of these exist yet — that's expected. The API currently runs on
`MockForecastModel` (see `app/models/mock_predictor.py`), which generates
plausible-looking forecasts without needing any of these files.

To go from "script that trains a model in memory" to "artifact saved
here": at the end of your training script, add a line like
`joblib.dump(model, "backend/model_artifacts/xgboost.pkl")` for tree
models, or the equivalent `.save()` call for SARIMAX/LSTM. Then follow
the steps in `app/models/mock_predictor.py`'s docstring to wire a real
model class into `app/services/model_registry.py`.
