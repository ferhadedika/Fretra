# FreightAI — SCFI Forecasting Web App

A lightweight web app that deploys your SCFI forecasting models (XGBoost,
Random Forest, SARIMAX, LSTM, Ensemble) behind a real API with a React
frontend.

```
freightai-app/
├── backend/     FastAPI: preprocessing, inference, SHAP explanations
└── frontend/    React SPA: model selector, forecast chart, SHAP panel
```

**Current state:** runs end-to-end right now using mock predictors (no
trained model files needed to try it). Swap in your real models and a
real Firebase project when you're ready — both are isolated behind clean
interfaces so neither swap touches the rest of the code. See
"Going to production" below.

---

## Quickstart (local, no setup required)

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs — FastAPI's interactive API explorer.
Try `POST /api/forecast` right there to see it work.

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173 — you should see live charts, KPIs, and SHAP
panels, all powered by the mock predictors.

---

## How the model swap works

Every model is wrapped behind one interface (`BaseForecastModel` in
`backend/app/models/base.py`):

```python
class BaseForecastModel(ABC):
    def load(self, artifacts_dir): ...      # read the trained file once at startup
    def predict(self, horizon_weeks): ...   # return forecasted SCFI values
    def explain(self, horizon_weeks): ...   # return SHAP global + local importances
```

`MockForecastModel` (`backend/app/models/mock_predictor.py`) implements
this with no real model — just a random walk + synthetic SHAP values, so
you can see the whole app working before you have trained artifacts.

**To plug in a real model:**

1. Save your trained model to a file (see `backend/model_artifacts/README.md`
   for the "what's an artifact" explainer + exact save calls per model type).
2. Create `backend/app/models/xgboost_model.py` (or similar), subclassing
   `BaseForecastModel`, loading your real `.pkl`/`.h5` file in `load()`.
3. In `backend/app/services/model_registry.py`, swap the
   `MockForecastModel(...)` line for your new class.
4. Nothing else changes — routers, schemas, and the entire frontend stay
   exactly as they are, because they only ever talk to the
   `BaseForecastModel` interface, never to a specific model class.

Repeat per model. You can mix real and mock models at any point (e.g.
real XGBoost, mock LSTM) while you finish training the rest.

---

## Going to production

### Firestore (free)

1. Create a Firebase project at https://console.firebase.google.com
   (free, no credit card for the Spark plan used here).
2. Enable Firestore in Native mode.
3. Project Settings → Service Accounts → Generate New Private Key →
   save the JSON file somewhere **outside** version control.
4. In `backend/.env` (copy from `.env.example`):
   ```
   FIREBASE_CREDENTIALS_PATH=/path/to/serviceAccountKey.json
   FIREBASE_PROJECT_ID=your-project-id
   ```
5. Restart the backend. Check `GET /api/health` — `firestore_mode`
   should now say `"firestore"` instead of `"in-memory"`.

Until you do this, forecast history is kept in memory and resets when
the backend restarts. Everything else works identically either way.

### Backend hosting (free)

Firebase Hosting only serves static files — it can't run a Python
server. Use a free always-on-ish host for FastAPI instead:

- **Render** (render.com) — free web service tier, Python natively
  supported. Connect your repo, set start command:
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Railway** (railway.app) — similar free tier, same start command.

Both sleep after inactivity on the free tier (cold start delay on the
next request) — fine for a student project / demo, not for 24/7 traffic.

Once deployed, note your backend's public URL (e.g.
`https://freightai-backend.onrender.com`).

### Frontend deploy (Firebase Hosting, free)

```bash
npm install -g firebase-tools
firebase login
cd frontend
```

Edit `.firebaserc` and replace `your-firebase-project-id` with your real
project ID. Then:

```bash
echo "VITE_API_BASE_URL=https://your-backend-url.onrender.com" > .env.production
npm run build
firebase deploy --only hosting
```

Finally, add your deployed frontend URL to the backend's
`ALLOWED_ORIGINS` env var (e.g. on Render's dashboard) so CORS allows it:

```
ALLOWED_ORIGINS=https://your-project.web.app
```

---

## Cost

Firebase Hosting + Firestore are free on the Spark plan for usage at
this scale (50K Firestore reads/day, 10GB hosting bandwidth/month — a
student project won't get close). The FastAPI backend needs a host that
runs Python; Render/Railway's free tiers cover that without a card.

The only thing that costs money is deliberately upgrading to Firebase's
Blaze (pay-as-you-go) plan, which this project doesn't require.

---

## API reference

Once the backend is running, full interactive docs are at
`http://localhost:8000/docs`. Summary:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness + which storage backend is active |
| GET | `/api/models` | List all 5 models + their accuracy metrics |
| GET | `/api/forecast/historical` | Historical SCFI series for the chart |
| POST | `/api/forecast` | Generate a forecast (body: `model_id`, `horizon_weeks`) |
| GET | `/api/forecast/history` | Past forecast runs (from Firestore) |
| GET | `/api/shap/{model_id}` | Global + local SHAP feature importances |

---

## Folder reference

```
backend/
  app/
    main.py              FastAPI app, CORS, startup model loading
    config.py             env-driven settings
    models/
      base.py              the interface every model implements
      mock_predictor.py     the mock standing in for real models today
    services/
      model_registry.py     loads all 5 models once at startup
      forecast_service.py    forecast + SHAP business logic
      firestore_service.py   Firestore wrapper w/ in-memory fallback
    routers/
      models_router.py, forecast_router.py, shap_router.py
    schemas/
      forecast.py            all request/response Pydantic models
  model_artifacts/         drop trained .pkl/.h5 files here
  requirements.txt

frontend/
  src/
    App.jsx                 page layout, wires hook -> components
    hooks/useForecast.js      all async state for the dashboard
    services/api.js           every backend call, in one place
    components/               Header, ForecastControls, KpiRow,
                               ForecastChart, ProjectionsLedger, ShapPanel
    styles/                   dark-navy theme matching the FreightAI dashboard
  firebase.json, .firebaserc  Firebase Hosting config
```
