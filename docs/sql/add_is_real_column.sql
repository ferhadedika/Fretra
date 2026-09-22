-- Adds provenance tracking so a model that silently fell back to
-- MockForecastModel's random-walk (e.g. because a real artifact failed to
-- load - see app/services/model_registry.py's real/mock switch) never
-- gets persisted indistinguishably from a real prediction. Without this,
-- fabricated mock numbers can end up in forecast_history/forecast_shap
-- and get folded into forecast_accuracy's MAE/RMSE/MAPE/bias snapshots
-- as if they were real model error.
--
-- Run this once in the Supabase SQL editor. Existing rows default to
-- true (they predate this column and were, in practice, written while
-- real_status was whatever it was at the time - this is a best-effort
-- backfill default, not a guarantee). Every row written going forward is
-- explicitly tagged by app/forecasting/orchestrator.py from
-- registry.real_status(), and app/forecasting/accuracy.py filters to
-- is_real = true when computing accuracy snapshots.

alter table forecast_history
  add column if not exists is_real boolean not null default true;

alter table forecast_shap
  add column if not exists is_real boolean not null default true;

comment on column forecast_history.is_real is
  'false if this row was produced by MockForecastModel''s random-walk fallback (a real model artifact failed to load) rather than the actual trained model. Excluded from forecast_accuracy snapshot computation.';
comment on column forecast_shap.is_real is
  'Same meaning as forecast_history.is_real - false for SHAP values computed against a mock-fallback prediction.';
