-- Enables RLS on all 4 data tables with an explicit, auditable "allow
-- everything for anon + authenticated" policy - a deliberate interim
-- state (see docs/PROJECT_DOCUMENTATION.md Phase 15.4 / audit item #1)
-- chosen because there is currently no service-role key configured
-- anywhere in this project, so the backend and scraper both write using
-- the SAME anon key the browser uses. This does NOT yet reduce the
-- actual risk (anon can still read/write everything) - it only makes
-- the permissive state explicit and creates the scaffolding to tighten
-- later (swap `using (true) with check (true)` for real conditions, or
-- add a service-role key and drop the anon write grant) without any
-- further schema changes.

alter table forecast_history  enable row level security;
alter table forecast_shap     enable row level security;
alter table forecast_accuracy enable row level security;
alter table scfi_weekly       enable row level security;

create policy "anon+authenticated full access" on forecast_history
  for all to anon, authenticated using (true) with check (true);

create policy "anon+authenticated full access" on forecast_shap
  for all to anon, authenticated using (true) with check (true);

create policy "anon+authenticated full access" on forecast_accuracy
  for all to anon, authenticated using (true) with check (true);

create policy "anon+authenticated full access" on scfi_weekly
  for all to anon, authenticated using (true) with check (true);
