-- Adds approximate confidence-interval columns to forecast_history.
-- Not statistically rigorous (no quantile regression / retraining involved,
-- per "don't retrain the models") - see orchestrator.py's comment on
-- _confidence_interval() for the exact (documented, labeled-as-approximate)
-- method: each model's own training-time CV RMSE, widened by sqrt(week_ahead)
-- to approximate how error compounds across the recursive multi-step cascade.

alter table forecast_history
  add column if not exists lower_bound numeric,
  add column if not exists upper_bound numeric,
  add column if not exists confidence_level numeric;
