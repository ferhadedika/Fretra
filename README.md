# Fretra — SCFI Forecasting Web App

A production ensemble (XGBoost, Random Forest, SARIMAX, LSTM, blended
Ensemble) that forecasts the Shanghai Containerized Freight Index (SCFI)
12 weeks ahead, served behind a FastAPI backend with SHAP explanations,
Supabase-backed persistence, and a React dashboard.

```
freightai-app/
├── backend/     FastAPI: feature engineering, inference, SHAP, retraining
└── frontend/    React SPA: landing site, auth, dashboard, SHAP panel
```

**Current state:** all 5 models are real and trained — not mocks. Every
model's `predict()`/`explain()` runs against actual saved artifacts in
`model_artifacts/` (XGBoost R² 0.996, blended ensemble weights XGBoost
70% / RF 10% / SARIMAX 10% / LSTM 10%, MAE 49.15). A `MockForecastModel`
random-walk fallback still exists and kicks in automatically, per-model,
if any single artifact fails to load — so the app degrades gracefully
instead of crashing, and `GET /api/health` reports exactly which models
are running real vs. mock at any moment. Both `backend/` and `frontend/`
are deployed and live (Render + Vercel respectively, auto-deploying on
every push to `main`).

---

## Quickstart (local)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # fill in Supabase credentials - see below
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs — FastAPI's interactive API explorer.
**Every endpoint requires a valid Supabase session bearer token** (see
"Authentication" below) - there's no anonymous access anymore, so
`/docs`'s "Try it out" needs a real token pasted into the Authorize
dialog to get anything but a 401.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
cp .env.example .env              # needs VITE_API_BASE_URL + VITE_SUPABASE_* - see below
npm run dev
```

Open http://localhost:5173 — sign in (email/password or Google, via
Supabase Auth) to reach the dashboard at `/app`. The marketing pages
(`/`, `/models`, `/pipeline`, `/features`, `/about`) are public.

### Required environment variables

Both `.env.example` files (`backend/` and `frontend/`) are fully
commented and safe to copy as-is for local dev. In short, you need a
Supabase project for:
- **Auth** — email/password + Google sign-in gate the dashboard and
  every API endpoint (`app/dependencies/auth.py`).
- **`scfi_weekly` table** — the real historical SCFI series the whole
  pipeline is anchored on (`backend/supabase_upload.py` / `scraper_pipeline.py`
  populate it; without it, the API falls back to a synthetic series).
- **Rolling-forecast tables** (`forecast_history`, `forecast_shap`,
  `forecast_accuracy`, `pipeline_locks`) — written by the rolling
  forecast pipeline (see below). DDL lives in `backend/docs/sql/`.

Firebase/Firestore is optional and legacy — forecast history from the
older on-demand `/api/forecast` endpoint falls back to in-memory storage
if unconfigured; the live rolling-forecast pipeline uses Supabase only.

---

## How predictions actually get made

Every model implements one interface (`BaseForecastModel` in
`backend/app/models/base.py`):

```python
class BaseForecastModel(ABC):
    def load(self, artifacts_dir): ...                 # read the trained file once at startup
    def predict(self, horizon_weeks, anchor): ...       # return forecasted SCFI values
    def explain(self, horizon_weeks, anchor): ...       # return SHAP global + local importances
```

`app/services/model_registry.py` loads all 5 real classes
(`XGBoostModel`, `RandomForestModel`, `SarimaxModel`, `LstmModel`,
`EnsembleModel`) once at startup from `model_artifacts/`, falling back to
`MockForecastModel` per-model on a load failure (see "Current state"
above). Nothing downstream — routers, schemas, the frontend — ever talks
to a specific model class, only the shared interface.

**Two forecasting paths exist side by side:**

1. **Rolling pipeline** (the live one) — `app/forecasting/orchestrator.py`
   runs on a scheduler (`app/scheduler/poller.py`, every 30 min by
   default) whenever new SCFI data lands in Supabase, generates a fresh
   12-week forecast for all 5 models, and writes an immutable log to
   `forecast_history`/`forecast_shap`. Predictions are tagged `is_real`
   so a mock-fallback run can never quietly corrupt accuracy stats.
   `app/forecasting/accuracy.py` backfills actuals and recomputes
   MAE/RMSE/MAPE/bias once real data catches up to a past prediction.
2. **On-demand legacy path** (`POST /api/forecast`) — the original
   single-shot endpoint, Firestore-backed, still functional, superseded
   by the rolling pipeline for the dashboard's actual data but kept for
   direct API use.

**Retraining** (`POST /api/models/retrain`) runs `scfi_model_v3.py` as a
subprocess against live Supabase data, compares its CV metrics against
what's currently deployed, and only swaps in the new artifacts if
they're equal-or-better — backing up the old ones first and rolling back
automatically if the swap or reload fails. See
`backend/docs/PROJECT_DOCUMENTATION.md` for the full design writeup.

---

## Deployment

Already live - this section is for standing up your own copy.

### Backend (Render)

Free web service tier, Python natively supported.

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set all the env vars from `backend/.env.example` in Render's dashboard
  (`FIREBASE_CREDENTIALS_JSON` is supported as an alternative to
  `FIREBASE_CREDENTIALS_PATH` for platforms that can't mount a file).
- Sleeps after inactivity on the free tier (cold start on the next
  request) - fine for a demo, not for 24/7 traffic.

### Frontend (Vercel)

```bash
cd frontend
npm run build
```

`vercel.json` already rewrites all routes to `index.html` for React
Router. Set `VITE_API_BASE_URL` (your Render backend URL) and
`VITE_SUPABASE_URL`/`VITE_SUPABASE_ANON_KEY` as Vercel project env vars,
then connect the repo for auto-deploy on push.

Finally, add the deployed frontend's URL to the backend's
`ALLOWED_ORIGINS` env var so CORS allows it.

*(`firebase.json`/`.firebaserc` still exist from an earlier Firebase
Hosting setup and are unused in the current Render+Vercel deployment.)*

---

## API reference

Full interactive docs at `http://localhost:8000/docs`. All routes below
require `Authorization: Bearer <supabase-jwt>`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness + real/mock status per model + storage mode |
| GET | `/api/models` | List all 5 models + their CV accuracy metrics |
| GET | `/api/metrics` | Full training-time metrics + baseline comparisons |
| GET | `/api/rolling-forecast/latest` | Latest rolling-pipeline run, all models, filterable by horizon |
| GET | `/api/rolling-forecast/history` | Past rolling-pipeline runs |
| GET | `/api/rolling-forecast/accuracy` | Live MAE/RMSE/MAPE/bias snapshots |
| POST | `/api/rolling-forecast/run` | Force a rolling-pipeline run now |
| POST | `/api/models/retrain` | Kick off a background retrain (see above) |
| GET | `/api/models/retrain/status` | Status of the last/current retrain |
| GET | `/api/forecast/historical` | Historical SCFI series for the chart |
| POST | `/api/forecast` | Legacy on-demand forecast (body: `model_id`, `horizon_weeks`) |
| GET | `/api/forecast/history` | Past legacy forecast runs (Firestore) |
| GET | `/api/shap/{model_id}` | Legacy global + local SHAP feature importances |

---

## Folder reference

```
backend/
  app/
    main.py                  FastAPI app, CORS, startup model loading, scheduler
    config.py                 env-driven settings
    dependencies/auth.py       Supabase JWT verification (every route needs it)
    models/
      base.py                   the interface every model implements
      xgboost_model.py, random_forest_model.py,
      sarimax_model.py, lstm_model.py, ensemble_model.py   the 5 real models
      mock_predictor.py          per-model fallback if a real artifact fails to load
      anchor.py, feature_cascade.py, metrics_loader.py      shared prediction plumbing
    forecasting/
      orchestrator.py            the rolling-forecast pipeline
      accuracy.py                 actuals backfill + accuracy snapshots
      feature_engineering.py       raw data -> the 68 ml_features / 19 exog_features
      retrain.py                   background retrain + compare + safe swap
    scheduler/poller.py         runs the rolling pipeline on a timer
    services/
      model_registry.py          loads all 5 models once at startup
      supabase_service.py         all Supabase reads/writes
      forecast_service.py         legacy on-demand forecast + SHAP logic
      firestore_service.py        legacy Firestore wrapper w/ in-memory fallback
    routers/
      models_router.py, forecast_router.py, shap_router.py,
      metrics_router.py, rolling_forecast_router.py, retrain_router.py
    schemas/
      forecast.py, rolling_forecast.py   all request/response Pydantic models
  docs/
    PROJECT_DOCUMENTATION.md    full architecture + design-decision writeup
    sql/                         every Supabase table/RLS-policy DDL
  model_artifacts/             trained .pkl/.keras files + config JSON
  requirements.txt

frontend/
  src/
    App.jsx                     routing, auth-gated /app dashboard shell
    LandingPage.jsx, AboutPage.jsx, FeaturesPage.jsx,
    ModelsPage.jsx, PipelinePage.jsx, LoginPage.jsx   public marketing + auth pages
    supabaseClient.js            the one shared Supabase client
    hooks/
      useForecast.js               all async state for the dashboard
      useScfi.js, useBaselines.js   live Supabase reads (current SCFI, baseline forecasts)
    services/api.js               every backend call, in one place
    components/
      Header.jsx, SiteHeader.jsx, ForecastControls.jsx, KpiRow.jsx,
      ForecastChart.jsx, ModelComparison.jsx, BaselinesComparison.jsx,
      FoldMetrics.jsx, ProjectionsLedger.jsx, ShapPanel.jsx,
      FeatureDictionary.jsx, ui/fx.jsx    shared design tokens + FX primitives
    utils/featureMeta.js, modelMeta.js   shared feature/model metadata
    styles/                       dark theme CSS
  vercel.json                   SPA rewrite config for the live Vercel deploy
```
